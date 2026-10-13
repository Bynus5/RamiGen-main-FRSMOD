import ast
import random
import time
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from diffusers import schedulers
from diffusers.models import AutoencoderKL
from loguru import logger
from transformers import BertModel, BertTokenizer
from transformers.modeling_utils import logger as tf_logger

from .constants import SAMPLER_FACTORY, NEGATIVE_PROMPT, TRT_MAX_WIDTH, TRT_MAX_HEIGHT, TRT_MAX_BATCH_SIZE
from .diffusion.pipeline_controlnet import StableDiffusionControlNetPipeline
from .modules.models import HunYuanDiT, HUNYUAN_DIT_CONFIG
from .modules.controlnet import HunYuanControlNet
from .modules.posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop
from .modules.text_encoder import MT5Embedder
from .utils.tools import set_seeds


class Resolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f"{self.height}x{self.width}"


class ResolutionGroup:
    def __init__(self):
        self.data = [
            Resolution(1024, 1024), Resolution(1280, 1280), Resolution(1024, 768), Resolution(1152, 864),
            Resolution(1280, 960), Resolution(768, 1024), Resolution(864, 1152), Resolution(960, 1280),
            Resolution(1280, 768), Resolution(768, 1280),
        ]
        self.supported_sizes = set([(r.width, r.height) for r in self.data])

    def is_valid(self, width, height):
        return (width, height) in self.supported_sizes


STANDARD_RATIO = np.array([1.0, 4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0])
STANDARD_SHAPE = [[(1024, 1024), (1280, 1280)], [(1280, 960)], [(960, 1280)], [(1280, 768)], [(768, 1280)]]
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]


def get_standard_shape(target_width, target_height):
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height


def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return (val[0], val[0])
        if len(val) == 2:
            return tuple(val)
        raise ValueError(f"Invalid value: {val}")
    if isinstance(val, (int, float)):
        return (val, val)
    raise ValueError(f"Invalid value: {val}")


def get_pipeline(args, vae, text_encoder, tokenizer, model, device, rank, embedder_t5, infer_mode, controlnet, sampler=None):
    sampler = sampler or args.sampler
    kwargs = SAMPLER_FACTORY[sampler]["kwargs"]
    scheduler = SAMPLER_FACTORY[sampler]["scheduler"]
    kwargs["beta_schedule"] = args.noise_schedule
    kwargs["beta_start"] = args.beta_start
    kwargs["beta_end"] = args.beta_end
    kwargs["prediction_type"] = args.predict_type
    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)
    logger.debug(f"Using sampler: {sampler} with scheduler: {scheduler}")
    scheduler.set_timesteps(args.infer_steps, device)
    progress_bar_config = {} if rank == 0 else {"disable": True}

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=model,
        scheduler=scheduler,
        feature_extractor=None,
        safety_checker=None,
        requires_safety_checker=False,
        progress_bar_config=progress_bar_config,
        embedder_t5=embedder_t5,
        infer_mode=infer_mode,
        controlnet=controlnet,
    )
    return pipeline.to(device), sampler


class End2End(object):
    def __init__(self, args, models_root_path):
        self.args = args
        self.root = Path(models_root_path) / "t2i"
        logger.info(f"Got text-to-image model root path: {self.root}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        tf_logger.setLevel("ERROR")

        logger.info("Loading CLIP Text Encoder...")
        self.clip_text_encoder = BertModel.from_pretrained(str(self.root / "clip_text_encoder"), False, revision=None).to(self.device)
        logger.info("Loading CLIP Text Encoder finished")

        logger.info("Loading CLIP Tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(str(self.root / "tokenizer"))
        logger.info("Loading CLIP Tokenizer finished")

        logger.info("Loading T5 Text Encoder and T5 Tokenizer...")
        self.embedder_t5 = MT5Embedder(self.root / "mt5", torch_dtype=torch.float16, max_length=256)
        self.embedder_t5.model.to(self.device)
        logger.info("Loading t5_text_encoder and t5_tokenizer finished")

        logger.info("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(str(self.root / "sdxl-vae-fp16-fix")).to(self.device)
        logger.info("Loading VAE finished")

        logger.info("Building RamiGen model...")
        model_config = HUNYUAN_DIT_CONFIG[self.args.model]
        self.patch_size = model_config["patch_size"]
        self.head_size = model_config["hidden_size"] // model_config["num_heads"]
        self.resolutions, self.freqs_cis_img = self.standard_shapes()
        self.image_size = _to_tuple(self.args.image_size)
        latent_size = (self.image_size[0] // 8, self.image_size[1] // 8)
        self.infer_mode = self.args.infer_mode
        self.multi_control = self._is_multi_control_enabled()

        if self.infer_mode in ["fa", "torch"]:
            self.model = HunYuanDiT(self.args, input_size=latent_size, **model_config, log_fn=logger.info).half().to(self.device)
            if self.multi_control:
                self.control_types = self._get_multi_control_types()
                self.controlnet = nn.ModuleDict({
                    ctrl_type: HunYuanControlNet(self.args, input_size=latent_size, **model_config, log_fn=logger.info).half().to(self.device)
                    for ctrl_type in self.control_types
                })
            else:
                self.controlnet = HunYuanControlNet(self.args, input_size=latent_size, **model_config, log_fn=logger.info).half().to(self.device)
            self.load_torch_weights()
            if args.lora_ckpt:
                self.model.load_adapter(args.lora_ckpt)
                self.model.merge_and_unload()
            self.model.eval()
            if self.multi_control:
                for module in self.controlnet.values():
                    module.eval()
            else:
                self.controlnet.eval()
            logger.info("Loading torch model finished")
            logger.info("Loading controlnet finished")
        elif self.infer_mode == "trt":
            from .modules.trt.hcf_model import TRTModel
            trt_dir = self.root / "model_trt"
            plugin_path = trt_dir / "fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so"
            self.model = TRTModel(
                model_name="model_onnx",
                engine_dir=str(trt_dir / "engine"),
                image_height=TRT_MAX_HEIGHT,
                image_width=TRT_MAX_WIDTH,
                text_maxlen=args.text_len,
                embedding_dim=args.text_states_dim,
                plugin_path=str(plugin_path),
                max_batch_size=TRT_MAX_BATCH_SIZE,
            )
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        logger.info("Loading inference pipeline...")
        self.pipeline, self.sampler = self.load_sampler()
        logger.info("Loading pipeline finished")
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info("                Model is ready.                  ")
        logger.info("==================================================")

    def _is_multi_control_enabled(self):
        """
        Determine whether inference uses multiple independent ControlNets.

        use_condition_fusion=True means:
            multiple condition images
            -> one condition_fusion
            -> one ControlNet

        Therefore it must not enter the multi-ControlNet branch.
        """
        use_fusion = bool(getattr(self.args, "use_condition_fusion", False))
        use_multi_controlnet = bool(
            getattr(self.args, "multi_control", False)
            or getattr(self.args, "multi_controlnet_weights", None)
            or getattr(self.args, "multi_control_types", None)
        )

        if use_fusion and use_multi_controlnet:
            raise ValueError(
                "--use-condition-fusion cannot be used together with "
                "--multi-control, --multi-control-types, or "
                "--multi-controlnet-weights. Condition fusion uses one ControlNet."
            )

        return use_multi_controlnet

    def _get_multi_control_types(self):
        if self.args.multi_control_types:
            return list(self.args.multi_control_types)
        return ["edge", "mask", "depth"]

    def _parse_multi_control_weights(self):
        if not self.multi_control:
            return None
        default = {k: 1.0 for k in self.control_types}
        raw = self.args.multi_control_weights
        if not raw:
            return default
        try:
            parsed = ast.literal_eval(raw)
        except Exception as e:
            raise ValueError(f"Invalid --multi-control-weights: {raw}") from e
        if isinstance(parsed, dict):
            out = default.copy()
            out.update(parsed)
            return out
        if isinstance(parsed, (list, tuple)):
            if len(parsed) != len(self.control_types):
                raise ValueError(f"Length mismatch in --multi-control-weights: expected {len(self.control_types)}, got {len(parsed)}")
            return {k: v for k, v in zip(self.control_types, parsed)}
        raise TypeError("--multi-control-weights must parse to dict/list/tuple")

    def _resolve_single_weight_path(self, weight_arg: Path, ctrl_type: str):
        load_key = self.args.load_key
        if weight_arg.is_dir():
            ds_files = sorted(weight_arg.glob("*_model_states.pt"))
            if ds_files:
                return ds_files[0]
            return weight_arg / f"pytorch_model_{ctrl_type}_{load_key}.pt"
        return weight_arg

    def _resolve_multi_weight_map(self):
        if self.args.multi_controlnet_weights is None:
            controlnet_dir = self.root / "controlnet"
            return {ctrl_type: controlnet_dir / f"pytorch_model_{ctrl_type}_{self.args.load_key}.pt" for ctrl_type in self.control_types}
        if len(self.args.multi_controlnet_weights) != len(self.control_types):
            raise ValueError("--multi-controlnet-weights length must match --multi-control-types")
        mapping = OrderedDict()
        for ctrl_type, weight_str in zip(self.control_types, self.args.multi_controlnet_weights):
            mapping[ctrl_type] = self._resolve_single_weight_path(Path(weight_str), ctrl_type)
        return mapping

    def load_torch_weights(self):
        load_key = self.args.load_key
        if self.args.dit_weight is not None:
            dit_weight = Path(self.args.dit_weight)
            if dit_weight.is_dir():
                files = list(dit_weight.glob("*.pt"))
                if len(files) == 0:
                    raise ValueError(f"No model weights found in {dit_weight}")
                if str(files[0]).startswith("pytorch_model_"):
                    model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                    bare_model = True
                elif any(str(f).endswith("_model_states.pt") for f in files):
                    files = [f for f in files if str(f).endswith("_model_states.pt")]
                    model_path = files[0]
                    bare_model = False
                else:
                    raise ValueError(f"Invalid model path: {dit_weight}")
            elif dit_weight.is_file():
                model_path = dit_weight
                bare_model = "unknown"
            else:
                raise ValueError(f"Invalid model path: {dit_weight}")
        else:
            model_path = self.root / "model" / f"pytorch_model_{load_key}.pt"
            bare_model = True
        if not model_path.exists():
            raise ValueError(f"model_path not exists: {model_path}")
        logger.info(f"Loading torch model {model_path}...")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key not in state_dict:
                raise KeyError(f"Missing key `{load_key}` in checkpoint: {model_path}")
            state_dict = state_dict[load_key]
        self.model.load_state_dict(state_dict, strict=True)

        if self.multi_control:
            weight_map = self._resolve_multi_weight_map()
            for ctrl_type, controlnet_path in weight_map.items():
                if not controlnet_path.exists():
                    raise ValueError(f"controlnet_path not exists: {controlnet_path}")
                logger.info(f"Loading controlnet[{ctrl_type}] model {controlnet_path}...")
                controlnet_state_dict = torch.load(controlnet_path, map_location=lambda storage, loc: storage)
                if "module" in controlnet_state_dict:
                    controlnet_state_dict = controlnet_state_dict["module"]
                self.controlnet[ctrl_type].load_state_dict(controlnet_state_dict, strict=True)
        else:
            if self.args.controlnet_weight is not None:
                controlnet_weight = Path(self.args.controlnet_weight)
                if controlnet_weight.is_dir():
                    controlnet_path = self._resolve_single_weight_path(controlnet_weight, self.args.control_type)
                elif controlnet_weight.is_file():
                    controlnet_path = controlnet_weight
                else:
                    raise ValueError(f"Invalid controlnet path: {controlnet_weight}")
            else:
                controlnet_path = self.root / "controlnet" / f"pytorch_model_{self.args.control_type}_{load_key}.pt"
            if not controlnet_path.exists():
                raise ValueError(f"controlnet_path not exists: {controlnet_path}")
            logger.info(f"Loading controlnet model {controlnet_path}...")
            controlnet_state_dict = torch.load(controlnet_path, map_location=lambda storage, loc: storage)
            if "module" in controlnet_state_dict:
                controlnet_state_dict = controlnet_state_dict["module"]
            self.controlnet.load_state_dict(controlnet_state_dict, strict=True)

    def load_sampler(self, sampler=None):
        return get_pipeline(self.args, self.vae, self.clip_text_encoder, self.tokenizer, self.model, self.device, 0, self.embedder_t5, self.infer_mode, self.controlnet, sampler=sampler)

    def calc_rope(self, height, width):
        th = height // 8 // self.patch_size
        tw = width // 8 // self.patch_size

        rope_img = getattr(self.args, "rope_img", None)

        if rope_img == "base1024":
            base_image_size = 1024
        elif rope_img == "base512":
            base_image_size = 512
        else:
            # Keep original behavior for old checkpoints.
            base_image_size = 512

        base_size = base_image_size // 8 // self.patch_size

        logger.info(
            f"[RoPE] rope_img={rope_img}, "
            f"base_image_size={base_image_size}, "
            f"target=({height},{width}), "
            f"token_grid=({th},{tw}), "
            f"base_size={base_size}"
        )

        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        return get_2d_rotary_pos_embed(self.head_size, start, stop, (th, tw))

    def standard_shapes(self):
        resolutions = ResolutionGroup()
        freqs_cis_img = {str(reso): self.calc_rope(reso.height, reso.width) for reso in resolutions.data}
        return resolutions, freqs_cis_img


    def predict(
        self, 
        user_prompt, 
        image, 
        height=1024, 
        width=1024, 
        seed=None, 
        enhanced_prompt=None, 
        negative_prompt=None, 
        infer_steps=100, 
        guidance_scale=6, 
        batch_size=1, 
        src_size_cond=(1024, 1024), 
        sampler=None, 
        use_style_cond=False,
    ):
        if seed is None:
            seed = random.randint(0, 1_000_000)
        generator = set_seeds(seed, device=self.device)
        logger.info(f"Input (height, width) = ({height}, {width})")
        if self.infer_mode in ["fa", "torch"]:
            target_height = int((height // 16) * 16)
            target_width = int((width // 16) * 16)
            logger.info(f"Align to 16: (height, width) = ({target_height}, {target_width})")
        else:
            target_width, target_height = get_standard_shape(width, height)
            logger.info(f"Align to standard shape: (height, width) = ({target_height}, {target_width})")

        prompt = enhanced_prompt.strip() if enhanced_prompt else user_prompt.strip()
        negative_prompt = negative_prompt if negative_prompt else self.default_negative_prompt
        style = torch.as_tensor([0, 0] * batch_size, device=self.device) if use_style_cond else None
        if src_size_cond is None:
            size_cond = None
            image_meta_size = None
        else:
            if isinstance(src_size_cond, int):
                src_size_cond = [src_size_cond, src_size_cond]
            size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
            image_meta_size = torch.as_tensor([size_cond] * 2 * batch_size, device=self.device)

        start_time = time.time()
        logger.debug(f"\n                       prompt: {user_prompt}\n              enhanced prompt: {enhanced_prompt}\n                         seed: {seed}\n              (height, width): {(target_height, target_width)}\n              negative_prompt: {negative_prompt}\n                   batch_size: {batch_size}\n               guidance_scale: {guidance_scale}\n                  infer_steps: {infer_steps}\n              image_meta_size: {size_cond}\n        ")
        reso = f"{target_height}x{target_width}"
        freqs_cis_img = self.freqs_cis_img[reso] if reso in self.freqs_cis_img else self.calc_rope(target_height, target_width)
        if sampler is not None and sampler != self.sampler:
            self.pipeline, self.sampler = self.load_sampler(sampler)
        control_weight = self._parse_multi_control_weights() if self.multi_control else ast.literal_eval(self.args.control_weight)
#-------查看时间消耗
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # pipe_t0 = time.time()
        # logger.info("[TIME-INNER] pipeline start")
        samples = self.pipeline(
            height=target_height,
            width=target_width,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            image_meta_size=image_meta_size,
            style=style,
            return_dict=False,
            generator=generator,
            freqs_cis_img=freqs_cis_img,
            use_fp16=self.args.use_fp16,
            learn_sigma=self.args.learn_sigma,
            image=image,
            control_weight=control_weight,
        )[0]


        # =========================================================================
        # 查看时间消耗
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # logger.info(f"[TIME-INNER] pipeline end = {time.time() - pipe_t0:.2f}s")
                
        # torch.cuda.empty_cache()
        logger.debug(f"Success, time: {time.time() - start_time}")
        return {
            "images": samples,
            "seed": seed,
        }
