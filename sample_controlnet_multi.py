from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms as T

from mllm.dialoggen_demo import DialogGen
from ssadit.config import get_args
from ssadit.inference_controlnet import End2End


norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

PROMPT_KEYS = (
    "prompt",
    "text",
    "text_zh",
    "text_en",
    "caption",
    "query",
)

MULTI_COND_KEYS = {
    "edge": (
        "edge_image",
        "edge_path",
        "conditioning_edge",
        "control_edge",
        "edge_condition_image",
    ),
    "mask": (
        "mask_image",
        "mask_path",
        "conditioning_mask",
        "control_mask",
        "mask_condition_image",
    ),
    "depth": (
        "depth_image",
        "depth_path",
        "conditioning_depth",
        "control_depth",
        "depth_condition_image",
    ),
}


SAVE_KEYS = (
    "save_name",
    "file_name",
    "basename",
    "name",
    "id",
)


def sanitize_name(name: str, fallback: str) -> str:
    name = str(name).strip()
    if not name:
        return fallback
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    return name[:120] if name else fallback


def resolve_weight_paths(args) -> None:
    if getattr(args, "controlnet_weight", None):
        controlnet_path = Path(args.controlnet_weight)
        if controlnet_path.is_dir():
            ds_files = sorted(controlnet_path.glob("*_model_states.pt"))
            if ds_files:
                args.controlnet_weight = str(ds_files[0])
                logger.info(
                    f"Resolved DeepSpeed controlnet checkpoint dir to file: {args.controlnet_weight}"
                )

    if getattr(args, "multi_controlnet_weights", None):
        resolved = []
        changed = False
        for weight in args.multi_controlnet_weights:
            p = Path(weight)
            if p.is_dir():
                ds_files = sorted(p.glob("*_model_states.pt"))
                if ds_files:
                    resolved.append(str(ds_files[0]))
                    changed = True
                    continue
            resolved.append(str(weight))
        if changed:
            args.multi_controlnet_weights = resolved
            logger.info(
                f"Resolved multi controlnet checkpoint dirs to files: {args.multi_controlnet_weights}"
            )


def inferencer():
    args = get_args()
    resolve_weight_paths(args)

    use_condition_fusion = bool(
        getattr(args, "use_condition_fusion", False)
        )

    use_multi_controlnet = bool(
        getattr(args, "multi_control", False)
        or getattr(args, "multi_controlnet_weights", None)
        or getattr(args, "multi_control_types", None)
        )

    if use_condition_fusion and use_multi_controlnet:
        raise ValueError(
            "Condition fusion uses one fused ControlNet and cannot be "
            "combined with --multi-control, --multi-control-types, "
            "or --multi-controlnet-weights."
            )

    if not use_condition_fusion and not use_multi_controlnet:
        raise ValueError(
            "Non-fusion inference must use the unified Multi-ControlNet "
            "interface. Specify --multi-control and at least one entry in "
            "--multi-control-types and --multi-controlnet-weights."
            )

    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    gen = End2End(args, models_root_path)

    if args.enhance:
        logger.info("Loading DialogGen model (for prompt enhancement)...")
        enhancer = DialogGen(str(models_root_path / "dialoggen"), args.load_4bit)
        logger.info("DialogGen model loaded.")
    else:
        enhancer = None

    return args, gen, enhancer


def iter_records(index_files: List[str]) -> Iterator[Dict[str, Any]]:
    for file_path in index_files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSONL in {path} at line {line_no}: {e}"
                        ) from e
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        raise TypeError(f"Each record must be a dict, got: {type(item)}")
                    yield item
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                for item in data["data"]:
                    if not isinstance(item, dict):
                        raise TypeError(f"Each record must be a dict, got: {type(item)}")
                    yield item
            else:
                raise ValueError(
                    f"Unsupported index file structure for {path}. Use JSONL or a JSON list."
                )


def resolve_prompt(record: Dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(
        f"Prompt not found. Expected one of keys: {PROMPT_KEYS}. Got keys: {list(record.keys())}"
    )


def get_active_condition_types(args) -> List[str]:
    """
    Return condition types according to the active inference mode.

    Fusion mode:
        order must exactly match training.

    Original multi-ControlNet mode:
        each condition is routed by name, so the configured order is retained.
    """
    if getattr(args, "use_condition_fusion", False):
        fusion_types = list(
            getattr(args, "fusion_control_types", None)
            or ["edge", "depth", "mask"]
        )

        if fusion_types != ["edge", "depth", "mask"]:
            logger.warning(
                f"Condition-fusion order is {fusion_types}. "
                "Ensure it exactly matches the training order."
            )

        return fusion_types

    active = list(getattr(args, "multi_control_types", []) or [])

    if active:
        return active

    return ["edge", "mask", "depth"]


def resolve_multi_condition_paths(record: Dict[str, Any], args) -> Dict[str, str]:
    cli_map = {
        "edge": getattr(args, "edge_image_path", None),
        "mask": getattr(args, "mask_image_path", None),
        "depth": getattr(args, "depth_image_path", None),
    }

    active_types = get_active_condition_types(args)
    out: Dict[str, str] = {}

    for name in active_types:
        cli_value = cli_map.get(name)
        if cli_value:
            out[name] = str(cli_value)
            continue

        if name not in MULTI_COND_KEYS:
            raise KeyError(
                f"Unsupported control type: {name}. Supported keys: {list(MULTI_COND_KEYS.keys())}"
            )

        for key in MULTI_COND_KEYS[name]:
            value = record.get(key)
            if value is not None and str(value).strip():
                out[name] = str(value).strip()
                break

        if name not in out:
            raise KeyError(
                f"Missing {name} condition path. Expected one of keys: {MULTI_COND_KEYS[name]}. "
                f"Got keys: {list(record.keys())}"
            )

    return out


def resolve_save_stem(record: Dict[str, Any], cond_path: str, record_idx: int) -> str:
    image_path = record.get("image")
    if image_path is not None and str(image_path).strip():
        return sanitize_name(
            Path(str(image_path)).stem, fallback=f"sample_{record_idx:06d}"
        )

    for key in SAVE_KEYS:
        value = record.get(key)
        if value is not None and str(value).strip():
            return sanitize_name(str(value), fallback=f"sample_{record_idx:06d}")

    return sanitize_name(Path(cond_path).stem, fallback=f"sample_{record_idx:06d}")


def resolve_hw(image_size):
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 1:
            return int(image_size[0]), int(image_size[0])
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
    raise ValueError(f"Invalid --image-size: {image_size}")

# ==========================================================
# Auto multi-resolution bucket selection from training metadata
# Bucket string convention: "HxW", e.g. "768x1280"
# ==========================================================

def parse_bucket_hw(bucket_name: str):
    """
    Parse bucket string in HxW format.

    Example:
        "768x1280" -> height=768, width=1280
    """
    bucket_name = str(bucket_name).strip().lower()

    if "x" not in bucket_name:
        raise ValueError(
            f"Invalid bucket name: {bucket_name}. Expected HxW, e.g. 768x1280."
        )

    h_str, w_str = bucket_name.split("x", 1)
    h = int(h_str)
    w = int(w_str)

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid bucket size: {bucket_name}")

    return h, w


def load_buckets_from_source_json(args):
    """
    Load resolution buckets from a multi-resolution bucket metadata JSON.

    Expected JSON structure:
        {
            "group_length": {
                "768x1280": [0, 0, 0, 2831, 3530, 1576],
                "1024x1024": [5000, 5000, 5000, 1153, 143, 63],
                ...
            }
        }

    Return:
        List[Dict], each item:
        {
            "bucket": "768x1280",
            "height": 768,
            "width": 1280,
            "count": 7937,
            "aspect_ratio": 1.666667
        }
    """
    source_json = getattr(args, "bucket_source_json", None)

    if not source_json:
        raise ValueError(
            "--auto-bucket requires --bucket-source-json. "
            "Please provide the training bucket metadata JSON, e.g. porcelain_mt.json."
        )

    source_path = Path(source_json)

    if not source_path.exists():
        raise FileNotFoundError(f"--bucket-source-json not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    group_length = meta.get("group_length", None)

    if not isinstance(group_length, dict):
        raise KeyError(
            f"{source_path} does not contain a valid 'group_length' dict."
        )

    min_count = int(getattr(args, "bucket_min_count", 1))

    buckets = []

    for bucket_name, counts in group_length.items():
        h, w = parse_bucket_hw(bucket_name)

        if isinstance(counts, list):
            total_count = int(sum(counts))
        else:
            total_count = int(counts)

        # skip empty or too-rare buckets
        if total_count < min_count:
            continue

        buckets.append(
            {
                "bucket": str(bucket_name),
                "height": int(h),
                "width": int(w),
                "count": int(total_count),
                "aspect_ratio": round(float(w) / float(h), 6),
                "area": int(w * h),
            }
        )

    if len(buckets) == 0:
        raise ValueError(
            f"No usable buckets found in {source_path}. "
            f"bucket_min_count={min_count}"
        )

    # Sort for stable logging and reproducibility.
    buckets = sorted(
        buckets,
        key=lambda x: (x["height"], x["width"]),
    )

    logger.info(
        f"[AutoBucket] Loaded {len(buckets)} buckets from {source_path}. "
        f"bucket_min_count={min_count}"
    )

    for b in buckets:
        logger.info(
            f"[AutoBucket] bucket={b['bucket']} HxW | "
            f"count={b['count']} | "
            f"ratio={b['aspect_ratio']} | "
            f"area={b['area']}"
        )

    return buckets


def choose_bucket_from_record(record: Dict[str, Any], buckets, args):
    """
    Choose the best inference bucket according to orig_width / orig_height.

    JSONL record should contain:
        orig_width
        orig_height

    Bucket convention:
        bucket["height"], bucket["width"]
    """
    if "orig_width" not in record or "orig_height" not in record:
        return None

    orig_w = int(record["orig_width"])
    orig_h = int(record["orig_height"])

    if orig_w <= 0 or orig_h <= 0:
        return None

    orig_ratio = float(orig_w) / float(orig_h)
    orig_area = float(orig_w * orig_h)

    area_weight = float(getattr(args, "bucket_area_weight", 0.05))

    def score(bucket):
        bw = float(bucket["width"])
        bh = float(bucket["height"])

        bucket_ratio = bw / bh
        bucket_area = bw * bh

        # log-space ratio distance is more symmetric than abs(r1-r2)
        ratio_score = abs(np.log(bucket_ratio / orig_ratio))

        # area only acts as a weak secondary constraint
        area_score = abs(bucket_area - orig_area) / max(orig_area, 1.0)

        return ratio_score + area_weight * area_score

    best = min(buckets, key=score)

    bucket_info = {
        "bucket_source": "group_length_json",
        "bucket": best["bucket"],
        "bucket_count": int(best["count"]),
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "orig_aspect_ratio": round(orig_ratio, 6),
        "target_height": int(best["height"]),
        "target_width": int(best["width"]),
        "bucket_aspect_ratio": round(float(best["width"]) / float(best["height"]), 6),
        "bucket_area": int(best["width"] * best["height"]),
    }

    return bucket_info


def resolve_record_hw(
    record: Dict[str, Any],
    args,
    fallback_height: int,
    fallback_width: int,
    buckets,
):
    """
    Resolve per-record inference height and width.

    Priority:
        1. If --auto-bucket is enabled:
           choose bucket from --bucket-source-json group_length
           using record["orig_width"] and record["orig_height"].
        2. Otherwise fallback to global --image-size.
    """
    if getattr(args, "auto_bucket", False):
        bucket_info = choose_bucket_from_record(
            record=record,
            buckets=buckets,
            args=args,
        )

        if bucket_info is not None:
            logger.info(
                f"[AutoBucket] id={record.get('id', 'N/A')} | "
                f"orig={bucket_info['orig_width']}x{bucket_info['orig_height']} WxH | "
                f"bucket={bucket_info['bucket']} HxW | "
                f"target=({bucket_info['target_height']}, {bucket_info['target_width']}) HxW | "
                f"orig_ratio={bucket_info['orig_aspect_ratio']} | "
                f"bucket_ratio={bucket_info['bucket_aspect_ratio']} | "
                f"bucket_count={bucket_info['bucket_count']}"
            )

            return (
                int(bucket_info["target_height"]),
                int(bucket_info["target_width"]),
                bucket_info,
            )

        logger.warning(
            f"[AutoBucket] Missing or invalid orig_width/orig_height for "
            f"id={record.get('id', 'N/A')}. "
            f"Fallback to global --image-size ({fallback_height}, {fallback_width}) HxW."
        )

    bucket_info = {
        "bucket_source": "global_image_size",
        "bucket": f"{fallback_height}x{fallback_width}",
        "target_height": int(fallback_height),
        "target_width": int(fallback_width),
    }

    return int(fallback_height), int(fallback_width), bucket_info


def load_condition_tensor(
    image_path: str,
    height: int,
    width: int,
    device: str,
):
    """
    Original RGB condition loader for unified Multi-ControlNet inference.

    A Multi-ControlNet configuration containing only one control type
    is treated as single-condition inference.

    Every condition image, including edge, is converted to RGB.
    Output: [1, 3, H, W]
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Condition image not found: {path}")

    with Image.open(path) as condition:
        condition = condition.convert("RGB")
        condition = condition.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )

        image = norm_transform(condition)

    image = image.unsqueeze(0).to(device)

    if image.shape[1] != 3:
        raise ValueError(
            f"Original ControlNet condition must have 3 channels, "
            f"but got shape={tuple(image.shape)}, path={path}"
        )

    return image

def load_fusion_condition_tensor(
    image_path: str,
    height: int,
    width: int,
    device: str,
    control_type: str,
):
    """
    Loader used only for latent-space multi-condition fusion.

    Training-compatible channel convention:
        edge  -> 1 channel
        depth -> 3 channels
        mask  -> 3 channels
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Condition image not found: {path}")

    control_type = str(control_type).lower()

    with Image.open(path) as condition:
        if control_type == "edge":
            condition = condition.convert("L")
            resample = Image.Resampling.BILINEAR
            expected_channels = 1

        elif control_type == "depth":
            condition = condition.convert("RGB")
            resample = Image.Resampling.BILINEAR
            expected_channels = 3

        elif control_type == "mask":
            condition = condition.convert("RGB")
            resample = Image.Resampling.NEAREST
            expected_channels = 3

        else:
            raise ValueError(
                f"Unsupported fusion condition type: {control_type}. "
                "Expected one of: edge, depth, mask."
            )

        condition = condition.resize(
            (width, height),
            resample=resample,
        )

        image = T.ToTensor()(condition)
        image = image.mul(2.0).sub(1.0)

    image = image.unsqueeze(0).to(device)

    if image.shape[1] != expected_channels:
        raise ValueError(
            f"Fusion condition '{control_type}' should contain "
            f"{expected_channels} channels, but got shape={tuple(image.shape)}, "
            f"path={path}"
        )

    return image


def maybe_enhance_prompt(enhancer, prompt: str) -> Optional[str]:
    if enhancer is None:
        return None
    success, enhanced_prompt = enhancer(prompt)
    if not success:
        logger.warning(f"Prompt enhancement refused, keep original prompt: {prompt}")
        return None
    logger.info(f"Enhanced prompt: {enhanced_prompt}")
    return enhanced_prompt
    

def expected_output_paths(results_dir: Path, stem: str, batch_size: int) -> List[Path]:
    """
    Return expected output image paths for one record.
    """
    batch_size = int(batch_size)

    if batch_size <= 1:
        return [results_dir / f"{stem}.png"]

    return [results_dir / f"{stem}_{sample_idx:02d}.png" for sample_idx in range(batch_size)]


def outputs_already_exist(results_dir: Path, stem: str, batch_size: int) -> bool:
    """
    Check whether all expected output images already exist.
    """
    paths = expected_output_paths(results_dir, stem, batch_size)
    return all(p.exists() for p in paths)


def save_images(
    images,
    results_dir: Path,
    meta_path: Path,
    record_idx: int,
    stem: str,
    prompt: str,
    condition_path: Any,
    used_seed: int,
    limit: Optional[int],
    generated_so_far: int,
    bucket_info: Optional[Dict[str, Any]] = None,
) -> int:
    saved = 0
    with meta_path.open("a", encoding="utf-8") as mf:
        for sample_idx, pil_img in enumerate(images):
            if limit is not None and generated_so_far + saved >= limit:
                break

            if len(images) == 1:
                save_name = f"{stem}.png"
            else:
                save_name = f"{stem}_{sample_idx:02d}.png"

            save_path = results_dir / save_name
            pil_img.save(save_path)
            logger.info(f"Saved image to {save_path}")

            meta_item = {
                "record_idx": record_idx,
                "file_name": save_name,
                "prompt": prompt,
                "condition_image_path": condition_path,
                "seed": used_seed,
                "generated_width": int(pil_img.size[0]),
                "generated_height": int(pil_img.size[1]),
            }

            if bucket_info is not None:
                meta_item["bucket_info"] = bucket_info


            mf.write(
                json.dumps(
                    meta_item,
                    ensure_ascii=False,
                )
                + "\n"
            )
            saved += 1
    return saved


def run_batch(args, gen, enhancer) -> None:
    if not args.index_file:
        raise ValueError("Batch mode requires --index-file")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta_path = results_dir / "meta.jsonl"

    fallback_height, fallback_width = resolve_hw(args.image_size)

    if getattr(args, "auto_bucket", False):
        buckets = load_buckets_from_source_json(args)
    else:
        buckets = None

    generated = 0
    processed = 0

    for record_idx, record in enumerate(iter_records(args.index_file)):
        # record_t0 = time.time()
        if args.max_generate_image is not None and generated >= args.max_generate_image:
            logger.info(
                f"Reached max_generate_image={args.max_generate_image}, stop inference."
            )
            break

        prompt = resolve_prompt(record)

        # Unified multi-control inference
        cond_paths = resolve_multi_condition_paths(record, args)
        condition_meta = cond_paths
        fallback_cond = next(iter(cond_paths.values()))
        stem = resolve_save_stem(record, fallback_cond, record_idx)

        # ==========================================================
        # Per-record inference resolution from training bucket group
        # ==========================================================
        height, width, bucket_info = resolve_record_hw(
            record=record,
            args=args,
            fallback_height=fallback_height,
            fallback_width=fallback_width,
            buckets=buckets,
        )

        # ================= Skip existing outputs =================
        if getattr(args, "skip_existing", False) and outputs_already_exist(
            results_dir,
            stem,
            args.batch_size,
        ):
            logger.info(f"[{record_idx}] Skip existing output: {stem}")
            processed += 1
            # logger.info(f"[TIME] total record = {time.time() - record_t0:.2f}s | record_idx={record_idx} | stem={stem}")
            continue

        use_condition_fusion = bool(
            getattr(args, "use_condition_fusion", False)
        )

        if use_condition_fusion:
            # New latent-space condition-fusion mode:
            # edge 1ch + depth 3ch + mask 3ch
            image = {
                name: load_fusion_condition_tensor(
                    image_path=path,
                    height=height,
                    width=width,
                    device=gen.device,
                    control_type=name,
                )
                for name, path in cond_paths.items()
            }

            fusion_types = list(
                getattr(args, "fusion_control_types", None)
                or ["edge", "depth", "mask"]
            )

            missing_types = [
                name for name in fusion_types
                if name not in image
            ]

            if missing_types:
                raise KeyError(
                    f"Missing fusion conditions: {missing_types}. "
                    f"Required order: {fusion_types}. "
                    f"Available inputs: {list(image.keys())}"
                )

            total_channels = sum(
                int(image[name].shape[1])
                for name in fusion_types
            )

            expected_channels = int(
                getattr(args, "condition_fusion_in_channels", 7)
            )

            if total_channels != expected_channels:
                shape_info = {
                    name: tuple(image[name].shape)
                    for name in fusion_types
                }

                raise ValueError(
                    f"Condition fusion expects {expected_channels} total channels, "
                    f"but got {total_channels}. Inputs: {shape_info}"
                )

            logger.info(
                f"[Condition Fusion] order={fusion_types}, "
                f"input_shapes="
                f"{ {name: tuple(image[name].shape) for name in fusion_types} }"
            )

        else:
            # Original single/multi-ControlNet mode.
            # Every condition, including edge, stays RGB.
            image = {
                name: load_condition_tensor(
                    image_path=path,
                    height=height,
                    width=width,
                    device=gen.device,
                )
                for name, path in cond_paths.items()
            }

        current_seed = args.seed + record_idx
        enhanced_prompt = maybe_enhance_prompt(enhancer, prompt)

        logger.info(
            f"[{record_idx}] prompt={prompt} | condition={condition_meta} | seed={current_seed}"
        )

        results = gen.predict(
            prompt,
            height=height,
            width=width,
            image=image,
            seed=current_seed,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=args.negative,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            batch_size=args.batch_size,
            
            src_size_cond=args.size_cond,
            use_style_cond=args.use_style_cond,
        )

        # torch.cuda.synchronize()
        # logger.info(
        #     f"[TIME] gen.predict = {time.time() - t0:.2f}s | "
        #     f"record_idx={record_idx} | stem={stem}"
        # )

        saved = save_images(
            results["images"],
            results_dir,
            meta_path,
            record_idx=record_idx,
            stem=stem,
            prompt=prompt,
            condition_path=condition_meta,
            used_seed=results["seed"],
            limit=args.max_generate_image,
            generated_so_far=generated,
            bucket_info=bucket_info,
        )


        generated += saved
        processed += 1

        # 查看耗时情况
        # logger.info(
        #     f"[TIME] total record = {time.time() - record_t0:.2f}s | "
        #     f"record_idx={record_idx} | stem={stem}"
        # )

        del image
        del results

        # if processed % 50 == 0:
        #     gc.collect()
        #     torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.empty_cache()

    # logger.info(
    #     f"Finished multi-control batch inference. "
    #     f"Processed records: {processed}, generated images: {generated}"
    # )


if __name__ == "__main__":
    args, gen, enhancer = inferencer()
    run_batch(args, gen, enhancer)
