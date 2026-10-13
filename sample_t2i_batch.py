from pathlib import Path
import json
from loguru import logger

from ssadit.config import get_args
from ssadit.inference import End2End


def load_jsonl(jsonl_path):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or "file_name" not in obj:
                raise ValueError(f"Line {idx} missing `text` or `file_name`")
            samples.append(obj)
    return samples


def inferencer():
    args = get_args()
    model_root = args.model_root[0] if isinstance(args.model_root, (list, tuple)) else args.model_root
    models_root_path = Path(model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    gen = End2End(args, models_root_path)
    return args, gen


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


if __name__ == "__main__":
    args, gen = inferencer()

    OVERWRITE = getattr(args, "overwrite", False)
    MAX_GENERATE_IMAGE = getattr(args, "max_generate_image", None)

    assert args.index_file is not None, "Please specify --index-file jsonl_path"
    index_file = args.index_file[0] if isinstance(args.index_file, (list, tuple)) else args.index_file
    jsonl_path = Path(index_file)
    samples = load_jsonl(jsonl_path)

    logger.info(f"Loaded {len(samples)} prompts from {jsonl_path}")

    results_dir = args.results_dir[0] if isinstance(args.results_dir, (list, tuple)) else args.results_dir
    save_dir = Path(results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(args.image_size, (list, tuple)) or len(args.image_size) != 2:
        raise ValueError(f"`image_size` should be [H, W], got: {args.image_size}")
    height, width = map(int, args.image_size)

    real_batch_size = int(args.batch_size)
    if real_batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got: {real_batch_size}")

    if MAX_GENERATE_IMAGE is not None:
        MAX_GENERATE_IMAGE = int(MAX_GENERATE_IMAGE)
        if MAX_GENERATE_IMAGE <= 0:
            raise ValueError(f"`max_generate_image` must be > 0, got: {MAX_GENERATE_IMAGE}")

    # 先过滤掉已存在样本，避免无效推理
    pending_samples = []
    for idx, sample in enumerate(samples, 1):
        out_name = Path(sample["file_name"]).with_suffix(".png").name
        save_path = save_dir / out_name

        if save_path.exists() and not OVERWRITE:
            logger.info(f"[{idx}/{len(samples)}] Skip existing file: {save_path.name}")
            continue

        pending_samples.append(sample)

    logger.info(f"Pending samples before limit: {len(pending_samples)} / {len(samples)}")

    if len(pending_samples) == 0:
        logger.info("Nothing to do. All files already exist.")
        raise SystemExit(0)

    # 限制最大生成数量
    if MAX_GENERATE_IMAGE is not None:
        original_pending = len(pending_samples)
        pending_samples = pending_samples[:MAX_GENERATE_IMAGE]
        logger.info(
            f"Apply max_generate_image={MAX_GENERATE_IMAGE}, "
            f"will generate {len(pending_samples)} / {original_pending} pending samples"
        )

    if len(pending_samples) == 0:
        logger.info("Nothing to do after applying max_generate_image.")
        raise SystemExit(0)

    # 真正的多 prompt batch 推理
    processed = 0
    total = len(pending_samples)

    for batch_id, batch_samples in enumerate(chunk_list(pending_samples, real_batch_size), 1):
        prompts = [sample["text"] for sample in batch_samples]
        out_names = [Path(sample["file_name"]).with_suffix(".png").name for sample in batch_samples]
        save_paths = [save_dir / out_name for out_name in out_names]

        logger.info("=" * 80)
        logger.info(f"Batch {batch_id}: {len(batch_samples)} samples")
        for j, (out_name, prompt) in enumerate(zip(out_names, prompts), 1):
            logger.info(f"  [{processed + j}/{total}] {out_name}")
            logger.info(f"  Prompt: {prompt}")

        
        results = gen.predict(
            prompts,
            height=height,
            width=width,
            seed=args.seed + processed,
            enhanced_prompt=None,
            negative_prompt=args.negative,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            batch_size=len(batch_samples),
            src_size_cond=args.size_cond,
            use_style_cond=args.use_style_cond,
        )

        if not isinstance(results, dict) or "images" not in results:
            raise RuntimeError(f"Invalid results returned from gen.predict: {type(results)}")

        images = results["images"]

        if len(images) != len(batch_samples):
            raise RuntimeError(
                f"Returned image count mismatch: got {len(images)}, expected {len(batch_samples)}"
            )

        for save_path, pil_img in zip(save_paths, images):
            pil_img.save(save_path)
            logger.info(f"Saved to {save_path}")

        processed += len(batch_samples)

    logger.info(f"🎉 Generation finished. Generated {processed} images.")
