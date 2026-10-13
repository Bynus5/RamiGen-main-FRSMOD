# Licensed under the Apache License, Version 2.0

import inspect
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Union
import time
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ..modules.models import HunYuanDiT

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
Examples:
    ```py
    >>> import torch
    >>> from diffusers import StableDiffusionPipeline
    >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    >>> pipe = pipe.to("cuda")
    >>> prompt = "a photo of an astronaut riding a horse on mars"
    >>> image = pipe(prompt).images[0]
    ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


class StableDiffusionControlNetPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Union[BertModel, CLIPTextModel],
        tokenizer: Union[BertTokenizer, CLIPTokenizer],
        unet: Union[HunYuanDiT, UNet2DConditionModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        progress_bar_config: Dict[str, Any] = None,
        embedder_t5=None,
        infer_mode="torch",
        controlnet=None,
    ):
        super().__init__()
        self.embedder_t5 = embedder_t5
        self.infer_mode = infer_mode
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset` should be set to 1 instead of {scheduler.config.steps_offset}."
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning("You have disabled the safety checker.")
        if safety_checker is not None and feature_extractor is None:
            raise ValueError("Make sure to define a feature extractor if you want to use the safety checker.")

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, lora_scale=None):
        deprecate("_encode_prompt()", "1.0.0", "Use encode_prompt() instead.", standard_warn=False)
        prompt_embeds_tuple = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale)
        return torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None, lora_scale: Optional[float] = None, embedder=None):
        if embedder is None:
            text_encoder = self.text_encoder
            tokenizer = self.tokenizer
            max_length = self.tokenizer.model_max_length
        else:
            text_encoder = embedder.model
            tokenizer = embedder.tokenizer
            max_length = embedder.max_length

        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)
            text_inputs = tokenizer(prompt, padding="max_length", max_length=max_length, truncation=True, return_attention_mask=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(f"The following part of your input was truncated: {removed_text}")
            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]
            attention_mask = attention_mask.repeat(num_images_per_prompt, 1)
        else:
            attention_mask = None

        prompt_embeds_dtype = text_encoder.dtype if text_encoder is not None else (self.unet.dtype if self.unet is not None else prompt_embeds.dtype)
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError("`negative_prompt` should be the same type to `prompt`.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("`negative_prompt` batch size must match `prompt`.")
            else:
                uncond_tokens = negative_prompt
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
            uncond_attention_mask = uncond_input.attention_mask.to(device)
            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), attention_mask=uncond_attention_mask)[0]
            uncond_attention_mask = uncond_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            uncond_attention_mask = None

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask


    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            return image, None
        if torch.is_tensor(image):
            feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)
        safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
        return self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))

    def decode_latents(self, latents):
        deprecate("decode_latents", "1.0.0", "Use VaeImageProcessor.postprocess(...) instead", standard_warn=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu().permute(0, 2, 3, 1).float().numpy()

    def prepare_extra_step_kwargs(self, generator, eta):
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if callback_steps is None or (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and negative_prompt_embeds is not None and prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError("`prompt_embeds` and `negative_prompt_embeds` must have the same shape.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Length of generators must match effective batch size.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) if latents is None else latents.to(device)
        return latents * self.scheduler.init_noise_sigma

    def _is_multi_control(self):
        return isinstance(self.controlnet, (dict, Mapping, torch.nn.ModuleDict))

    def _normalize_control_weight_map(self, control_weight):
        if not self._is_multi_control():
            return control_weight
        control_keys = list(self.controlnet.keys())
        if control_weight is None:
            return {k: 1.0 for k in control_keys}
        if isinstance(control_weight, Mapping):
            return {k: control_weight.get(k, 1.0) for k in control_keys}
        if isinstance(control_weight, (list, tuple)):
            if len(control_weight) != len(control_keys):
                raise ValueError(f"control_weight length mismatch: expected {len(control_keys)}, got {len(control_weight)}")
            return {k: v for k, v in zip(control_keys, control_weight)}
        return {k: control_weight for k in control_keys}

    def _unwrap_controlnet(self):
        controlnet = self.controlnet

        while hasattr(controlnet, "module"):
            controlnet = controlnet.module

        return controlnet

    def _prepare_condition(
        self,
        image,
        generator,
        do_classifier_free_guidance,
        target_height,
        target_width,
    ):
        """
        Prepare condition input for three modes:

        1. Multi-ControlNet:
        each condition image -> shared VAE -> separate condition latent

        2. Latent-space condition fusion:
        edge/depth/mask -> concat -> condition_fusion -> one condition latent
        
        """

        # ============================================================
        # Mode 1: multiple independent ControlNets
        # ============================================================
        if self._is_multi_control():
            if not isinstance(image, Mapping):
                raise TypeError(
                    "For multi-ControlNet inference, `image` must be a dict "
                    "keyed by control type."
                )

            conditions = {}

            for name, img in image.items():
                if not torch.is_tensor(img):
                    raise TypeError(
                        f"Condition '{name}' must be a tensor, got {type(img)}."
                    )

                vae_dtype = next(self.vae.parameters()).dtype
                img = img.to(
                    device=self._execution_device,
                    dtype=vae_dtype,
                )

                cond = (
                    self.vae.encode(img)
                    .latent_dist.sample(generator)
                    .mul_(self.vae.config.scaling_factor)
                )

                if do_classifier_free_guidance:
                    cond = torch.cat([cond, cond], dim=0)

                conditions[name] = cond

            return conditions

        # ============================================================
        # Mode 2: multiple conditions fused by one ControlNet
        # ============================================================
        controlnet = self._unwrap_controlnet()
        fusion_module = getattr(controlnet, "condition_fusion", None)

        if fusion_module is not None:
            if not isinstance(image, Mapping):
                raise TypeError(
                    "Condition-fusion inference expects `image` to be a dict "
                    "containing edge, depth, and mask tensors."
                )

            fusion_types = list(
                getattr(
                    controlnet.args,
                    "fusion_control_types",
                    ["edge", "depth", "mask"],
                )
            )

            missing = [name for name in fusion_types if name not in image]
            if missing:
                raise KeyError(
                    f"Missing condition-fusion inputs: {missing}. "
                    f"Required order: {fusion_types}; received: {list(image.keys())}."
                )

            condition_parts = []

            for name in fusion_types:
                tensor = image[name]

                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"Condition '{name}' must be a tensor, got {type(tensor)}."
                    )

                condition_parts.append(tensor)

            # Fixed training order, normally:
            # edge(1ch) + depth(3ch) + mask(3ch) = 7ch
            condition = torch.cat(condition_parts, dim=1)

            expected_channels = int(
                getattr(controlnet.args, "condition_fusion_in_channels", 7)
            )

            if condition.shape[1] != expected_channels:
                channel_info = {
                    name: tuple(image[name].shape)
                    for name in fusion_types
                }
                raise ValueError(
                    f"Condition fusion expects {expected_channels} input channels, "
                    f"but concatenation produced {condition.shape[1]} channels. "
                    f"Inputs: {channel_info}"
                )

            fusion_dtype = next(
                fusion_module.parameters()
            ).dtype

            condition = condition.to(
                device=self._execution_device,
                dtype=fusion_dtype,
            )

            condition = fusion_module(
                condition
            )

            expected_height = (
                target_height // self.vae_scale_factor
            )

            expected_width = (
                target_width // self.vae_scale_factor
            )

            if condition.shape[1] != 4:
                raise ValueError(
                    "condition_fusion must output 4 channels, "
                    f"but got shape {tuple(condition.shape)}."
                )

            if condition.shape[-2:] != (
                expected_height,
                expected_width,
            ):
                raise ValueError(
                    "condition_fusion output spatial size does not "
                    "match the diffusion latent size: "
                    f"condition={tuple(condition.shape)}, "
                    f"expected spatial size="
                    f"({expected_height}, {expected_width})."
                )

            if do_classifier_free_guidance:
                condition = torch.cat([condition, condition], dim=0)

            return condition

    def _apply_layer_weights(self, controls, weight):
        if isinstance(weight, list):
            if len(weight) != len(controls):
                raise ValueError(f"Layer-weight length mismatch: expected {len(controls)}, got {len(weight)}")
            return [control * w for control, w in zip(controls, weight)]
        return [control * weight for control in controls]

    def _sum_controls(self, controls_accum, new_controls):
        if controls_accum is None:
            return [c for c in new_controls]
        return [a + b for a, b in zip(controls_accum, new_controls)]

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        height: int,
        width: int,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_embeds_t5: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds_t5: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        image_meta_size: Optional[torch.LongTensor] = None,
        style: Optional[torch.LongTensor] = None,
        progress: bool = True,
        use_fp16: bool = False,
        freqs_cis_img: Optional[tuple] = None,
        learn_sigma: bool = True,
        image=None,
        control_weight=1.0,
    ):
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None

        prompt_embeds, negative_prompt_embeds, attention_mask, uncond_attention_mask = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale)
        prompt_embeds_t5, negative_prompt_embeds_t5, attention_mask_t5, uncond_attention_mask_t5 = self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds_t5, negative_prompt_embeds=negative_prompt_embeds_t5, lora_scale=text_encoder_lora_scale, embedder=self.embedder_t5)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([uncond_attention_mask, attention_mask])
            prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5])
            attention_mask_t5 = torch.cat([uncond_attention_mask_t5, attention_mask_t5])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        condition = self._prepare_condition(
            image=image,
            generator=generator,
            do_classifier_free_guidance=do_classifier_free_guidance,
            target_height=height,
            target_width=width,
        )
        control_weight_map = self._normalize_control_weight_map(control_weight)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
        
            for i, t in enumerate(timesteps):
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # step_t0 = time.time()
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=latent_model_input.device)
                if use_fp16:
                    latent_model_input = latent_model_input.half()
                    t_expand = t_expand.half()
                    prompt_embeds = prompt_embeds.half()
                    ims = image_meta_size.half() if image_meta_size is not None else None
                else:
                    ims = image_meta_size if image_meta_size is not None else None

                if self.infer_mode in ["fa", "torch"]:
                    if self._is_multi_control():
                        fused_controls = None
                        for name, control_module in self.controlnet.items():
                            single_controls = control_module(
                                latent_model_input,
                                t_expand,
                                condition[name],
                                encoder_hidden_states=prompt_embeds,
                                text_embedding_mask=attention_mask,
                                encoder_hidden_states_t5=prompt_embeds_t5,
                                text_embedding_mask_t5=attention_mask_t5,
                                image_meta_size=ims,
                                style=style,
                                cos_cis_img=freqs_cis_img[0],
                                sin_cis_img=freqs_cis_img[1],
                                return_dict=False,
                            )
                            single_controls = self._apply_layer_weights(single_controls, control_weight_map[name])
                            fused_controls = self._sum_controls(fused_controls, single_controls)
                        controls = fused_controls

                    else:
                        controls = self.controlnet(
                            latent_model_input,
                            t_expand,
                            condition,
                            encoder_hidden_states=prompt_embeds,
                            text_embedding_mask=attention_mask,
                            encoder_hidden_states_t5=prompt_embeds_t5,
                            text_embedding_mask_t5=attention_mask_t5,
                            image_meta_size=ims,
                            style=style,
                            cos_cis_img=freqs_cis_img[0],
                            sin_cis_img=freqs_cis_img[1],
                            return_dict=False,
                        )
                        controls = self._apply_layer_weights(controls, control_weight_map)
                    noise_pred = self.unet(
                        latent_model_input,
                        t_expand,
                        encoder_hidden_states=prompt_embeds,
                        text_embedding_mask=attention_mask,
                        encoder_hidden_states_t5=prompt_embeds_t5,
                        text_embedding_mask_t5=attention_mask_t5,
                        image_meta_size=ims,
                        style=style,
                        cos_cis_img=freqs_cis_img[0],
                        sin_cis_img=freqs_cis_img[1],
                        return_dict=False,
                        controls=controls,
                    )
                elif self.infer_mode == "trt":
                    noise_pred = self.unet(
                        x=latent_model_input.contiguous(),
                        t_emb=t_expand.contiguous(),
                        context=prompt_embeds.contiguous(),
                        image_meta_size=ims.contiguous(),
                        style=style.contiguous(),
                        freqs_cis_img0=freqs_cis_img[0].to(device).contiguous(),
                        freqs_cis_img1=freqs_cis_img[1].to(device).contiguous(),
                        text_embedding_mask=attention_mask.contiguous(),
                        encoder_hidden_states_t5=prompt_embeds_t5.contiguous(),
                        text_embedding_mask_t5=attention_mask_t5.contiguous(),
                    )
                else:
                    raise ValueError(f"Unknown infer_mode: {self.infer_mode}")
                if learn_sigma:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents = results.prev_sample
                pred_x0 = results.pred_original_sample if hasattr(results, "pred_original_sample") else None
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents, pred_x0)
                # =============================================================
                # 时间消耗查看
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # if i == 0 or (i + 1) % 5 == 0 or i == len(timesteps) - 1:
                #     logger.info(
                #         f"[TIME-STEP] step {i + 1}/{len(timesteps)} = "
                #         f"{time.time() - step_t0:.2f}s"
                #     )

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
