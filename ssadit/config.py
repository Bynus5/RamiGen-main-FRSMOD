import argparse
import deepspeed

from .constants import *
from .diffusion.gaussian_diffusion import ModelVarType
from .modules.models import HUNYUAN_DIT_CONFIG


def model_var_type(value):
    try:
        return ModelVarType[value]
    except KeyError:
        valid_choices = [v.name for v in ModelVarType]
        raise ValueError(f"Invalid choice '{value}', valid choices are {valid_choices}")


def get_args(default_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-flag", type=str)
    parser.add_argument(
    "--overwrite",
    type=str,
    default="False",
    help="Whether to overwrite existing generated images. True=overwrite, False=skip.",
)

    parser.add_argument("--max-generate-image", type=int, default=None, help="Maximum number of images to generate before stopping.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--extra-fp16", action="store_true", help="Use extra fp16 for vae and text_encoder.")

    parser.add_argument("--model", type=str, choices=list(HUNYUAN_DIT_CONFIG.keys()), default="DiT-g/2")
    parser.add_argument("--image-size", type=int, nargs="+", default=[1024, 1024], help="Image size (h, w). If a single value is provided, the image will be treated to (value, value).")
    parser.add_argument("--qk-norm", action="store_true", help="Query Key normalization.")
    parser.set_defaults(qk_norm=True)
    parser.add_argument("--norm", type=str, choices=["rms", "layer"], default="layer", help="Normalization layer type")
    parser.add_argument("--text-states-dim", type=int, default=1024, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len", type=int, default=77, help="Token length of CLIP text encoder output.")
    parser.add_argument("--text-states-dim-t5", type=int, default=2048, help="Hidden size of T5 text encoder.")
    parser.add_argument("--text-len-t5", type=int, default=256, help="Token length of T5 text encoder output.")

    parser.add_argument("--training-parts", type=str, default="all", choices=["all", "lora", "ipadapter"], help="Training parts")
    parser.add_argument("--rank", type=int, default=64, help="Rank of LoRA")
    parser.add_argument("--lora-ckpt", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument("--target-modules", type=str, nargs="+", default=["Wqkv", "q_proj", "kv_proj", "out_proj"], help="Target modules for LoRA fine tune")
    
    parser.add_argument("--output-merge-path", type=str, default=None, help="Output path for merged model")

    parser.add_argument("--use-control", action="store_true", help="Enable loading control condition images for ControlNet training.")
    parser.add_argument("--control-type", type=str, default="edge", choices=["edge", "depth", "mask", "sketch"], help="Single-condition type; ignored when --use-condition-fusion is enabled")
    parser.add_argument("--control-weight", type=str, default="1.0", help="Single-ControlNet weight or per-layer list")
    parser.add_argument("--condition-image-path", type=str, default=None, help="Inference condition image path")

    parser.add_argument("--multi-control", action="store_true", help="Enable multi-ControlNet inference fusion.")
    parser.add_argument("--multi-control-types", type=str, nargs="+", default=None, help="Ordered control types for multi-ControlNet inference, e.g. edge mask depth")
    parser.add_argument("--multi-controlnet-weights", type=str, nargs="+", default=None, help="Ordered weight paths for multi-ControlNet inference")
    parser.add_argument("--multi-control-weights", type=str, default=None, help="Fusion weights for multi-ControlNet. Supports list like '[0.6,0.9,0.8]' or dict like '{\"edge\":0.6,\"mask\":0.9,\"depth\":0.8}'. Each item can also be a per-layer list.")
    parser.add_argument("--edge-image-path", type=str, default=None, help="Single-image mode edge condition path for multi-ControlNet inference")
    parser.add_argument("--mask-image-path", type=str, default=None, help="Single-image mode mask condition path for multi-ControlNet inference")
    parser.add_argument("--depth-image-path", type=str, default=None, help="Single-image mode depth condition path for multi-ControlNet inference")

    parser.add_argument("--is-ipa", type=bool, default=False, help="inference with IP-Adapter or not")
    parser.add_argument("--resume-ipa", type=bool, default=False, help="train with resume IP-Adapter model or not")
    parser.add_argument("--resume-ipa-root", type=str, default=None, help="ipa model path")
    parser.add_argument("--ref-image-path", type=str, default=None, help="Inference ref image path")
    parser.add_argument("--i-scale", type=float, default=1.0, help="IP-Adapter weight")

    parser.add_argument("--learn-sigma", action="store_true", help="Learn extra channels for sigma.")
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.set_defaults(learn_sigma=True)
    parser.add_argument("--predict-type", type=str, choices=list(PREDICT_TYPE), default="v_prediction", help="Diffusion predict type")
    parser.add_argument("--noise-schedule", type=str, choices=list(NOISE_SCHEDULES), default="scaled_linear", help="Noise schedule")
    parser.add_argument("--beta-start", type=float, default=0.00085, help="Beta start value")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta end value")
    parser.add_argument("--sigma-small", action="store_true")
    parser.add_argument("--mse-loss-weight-type", type=str, default="constant", help="Min-SNR-gamma.")
    parser.add_argument("--model-var-type", type=model_var_type, default=None, help="Specify the model variable type.")
    parser.add_argument("--noise-offset", type=float, default=0.0, help="Add extra noise to the input image.")

    parser.add_argument("--prompt", type=str, default="一只小猫", help="The prompt for generating images.")
    parser.add_argument("--model-root", type=str, default="ckpts", help="Root path of all the models")
    parser.add_argument("--dit-weight", type=str, default=None, help="Path to the RamiGen backbone weights.")
    parser.add_argument("--controlnet-weight", type=str, default=None, help="Path to the RamiGen ControlNet weights.")

    parser.add_argument("--load-key", type=str, choices=["ema", "module", "distill", "merge"], default="ema", help="Load model key for checkpoint.")
    parser.add_argument("--use-style-cond", action="store_true", help="Use style condition in RamiGen.")
    parser.add_argument("--size-cond", type=int, nargs="+", default=None, help="Size condition used in sampling.")
    parser.add_argument("--target-ratios", type=str, nargs="+", default=None, help="Target ratios for multi-resolution training.")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="Guidance scale for classifier-free.")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt.")

    parser.add_argument("--infer-mode", type=str, choices=["fa", "torch", "trt"], default="fa", help="Inference mode")
    parser.add_argument("--onnx-workdir", type=str, default="onnx_model", help="Path to save ONNX model")
    parser.add_argument("--sampler", type=str, choices=SAMPLER_FACTORY, default="ddpm", help="Diffusion sampler ddpm, ddim, or dpmms")
    parser.add_argument("--infer-steps", type=int, default=100, help="Inference steps")

    parser.add_argument("--enhance", action="store_true", help="Enhance prompt with mllm.")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false")
    parser.add_argument("--load-4bit", help="load DialogGen model with 4bit quantization.", action="store_true")
    parser.set_defaults(enhance=True)
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-training-steps", type=int, default=10_000_000)
    parser.add_argument("--gc-interval", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000)
    parser.add_argument("--ckpt-latest-every", type=int, default=10_000)
    parser.add_argument("--ckpt-every-n-epoch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=1234)
    parser.add_argument("--warmup-min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-num-steps", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)

# ================跳过已生成===========================
    parser.add_argument("--skip-existing",action="store_true",help="Skip samples whose expected output images already exist.")
# ---------------------------位置编码机制-------------------------------
    parser.add_argument("--rope-img", type=str, default=None, choices=["extend", "base512", "base1024"])
    parser.add_argument("--rope-real", action="store_true")
# ------------------------------------------------------------------
    parser.add_argument("--uncond-p", type=float, default=0.2)
    parser.add_argument("--uncond-p-t5", type=float, default=0.2)
    parser.add_argument("--uncond-p-img", type=float, default=0.05)

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-module-root", type=str, default=None)
    parser.add_argument("--resume-ema-root", type=str, default=None)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)

    parser.add_argument("--index-file", type=str, nargs="+", help="During training, provide a JSON file with data indices.")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--reset-loader", action="store_true")
    parser.add_argument("--multireso", action="store_true")
    parser.add_argument("--reso-step", type=int, default=None)
    parser.add_argument("--random-shrink-size-cond", action="store_true")
    parser.add_argument("--merge-src-cond", action="store_true")
    parser.add_argument("--base-ckpt", type=str, default=None)

    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-dtype", type=str, choices=["fp16", "fp32", "none"], default="none")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-warmup", action="store_true")
    parser.add_argument("--ema-warmup-power", type=float, default=None)
    parser.add_argument("--ema-reset-decay", action="store_true")

    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false")
    parser.add_argument("--use-zero-stage", type=int, default=1)
    parser.add_argument("--grad-accu-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gc-rate", default=1.0, type=float)
    parser.add_argument("--cpu-offloading", action="store_true")
    parser.add_argument("--save-optimizer-state", action="store_true")

    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--deepspeed-optimizer", action="store_true")
    parser.add_argument("--remote-device", type=str, default="none", choices=["none", "cpu", "nvme"])
    parser.add_argument("--zero-stage", type=int, default=1)

    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=443)
    parser.add_argument("--gradio_share", type=bool, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_img_path", type=str, default="app/output")


    
# 多条件特征融合模块
    parser.add_argument(
        "--use-condition-fusion",
        action="store_true",
        help="Fuse edge/depth/mask; ignore --control-type. Without this flag, use one RGB condition encoded by VAE."
    )

    parser.add_argument(
        "--fusion-control-types",
        type=str,
        nargs="+",
        default=["edge", "depth", "mask"],
        help="Condition types used for fusion, e.g. edge depth mask."
    )
    parser.add_argument(
        "--condition-fusion-in-channels",
        type=int,
        default=7,
        help="Input channel number for condition fusion. "
            "For edge/depth/mask with edge=L, depth=RGB, mask=RGB, use 7."
         )

    parser.add_argument(
        "--controlnet-train-mode",
        type=str,
        default="all",
        choices=["all", "attn_zero", "zero_only", "attn_norm_zero"],
        help=(
            "ControlNet trainable parameter mode. "
            "'all': train full ControlNet; "
            "'attn_zero': train attention layers plus before/after projection and condition_fusion; "
            "'zero_only': train only before/after projection and condition_fusion; "
            "'attn_norm_zero': train attention layers, norm layers, before/after projection and condition_fusion."),
    )

    parser.add_argument(
    "--resume-controlnet-root",
    type=str,
    default=None,
    help="Path to an existing ControlNet DeepSpeed checkpoint or model state for initialization.",
    )

    parser.add_argument("--racem-mode", choices=["preprocess", "batch_wise"], default="preprocess",
                        help="Read saved conditions, or extract RACEM conditions online per training batch")
    parser.add_argument("--racem-label-dir", default=None, help="Dataset-wide YOLO labels for online RACEM")
    parser.add_argument("--racem-region", choices=["full", "bbox", "clip"], default="clip")
    parser.add_argument("--racem-bbox-type", choices=["obb", "hbb"], default="obb")
    parser.add_argument("--racem-sam-weights", default=None)
    parser.add_argument("--racem-edge-weights", default=None)
    parser.add_argument("--racem-depth-weights", default=None)


    # ============================================================
    # Inference auto multi-resolution bucket
    # ============================================================
    parser.add_argument(
        "--auto-bucket",
        action="store_true",
        help=(
            "Enable per-record automatic resolution bucket selection during inference. "
            "It uses orig_width/orig_height from JSONL and resolution groups from "
            "--bucket-source-json."
        ),
    )

    parser.add_argument(
        "--bucket-source-json",
        type=str,
        default=None,
        help=(
            "Path to the multi-resolution bucket metadata JSON, "
            "for example porcelain_mt.json. It must contain a group_length dict."
        ),
    )

    parser.add_argument(
        "--bucket-min-count",
        type=int,
        default=1,
        help=(
            "Ignore buckets whose total group_length count is smaller than this value. "
            "Use 1 to keep all non-empty buckets."
        ),
    )

    parser.add_argument(
        "--bucket-area-weight",
        type=float,
        default=0.05,
        help=(
            "Area penalty weight for choosing bucket. "
            "Aspect ratio is the main criterion, area is secondary."
        ),
    )

    return parser.parse_args(default_args)
