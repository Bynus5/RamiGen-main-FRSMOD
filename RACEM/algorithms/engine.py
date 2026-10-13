"""One callable extraction pipeline shared by preprocessing and training."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .backends import WEIGHTS, SAMBackend, HEDBackend, DepthBackend, depth_to_bgr
from .labels import class_color_map, class_ids_in_directory, bbox_mask, read_labels
from .levelset import refine_mask


FEATURES = ("edge", "depth", "mask")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_features(features):
    if isinstance(features, str):
        features = features.split()
    features = tuple(features)
    if features == ("all",):
        return FEATURES
    if not features or len(set(features)) != len(features) or set(features) - set(FEATURES):
        raise ValueError("features must be all, or a non-repeating selection of edge/depth/mask")
    return tuple(name for name in FEATURES if name in features)


class RACEMExtractor:
    """Frozen, lazily loaded extractors. Input/output color arrays are RGB uint8.

    region controls edge/depth: full image, YOLO bbox, or refined-mask clip.
    The semantic mask always uses the original SAM2 + level-set refinement.
    Construct one instance per training rank, never one per DataLoader worker.
    """

    def __init__(self, features="all", *, label_dir=None, class_ids=None,
                 region="clip", bbox_type="obb", device="cpu",
                 sam_weights=None, edge_weights=None, depth_weights=None):
        self.features = normalize_features(features)
        if region not in {"full", "bbox", "clip"} or bbox_type not in {"obb", "hbb"}:
            raise ValueError("region must be full/bbox/clip and bbox_type must be obb/hbb")
        self.region, self.bbox_type = region, bbox_type
        self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        self.label_dir = Path(label_dir).expanduser().resolve() if label_dir else None
        self.needs_mask = "mask" in self.features or region == "clip"
        if self.needs_mask:
            if class_ids is None:
                if self.label_dir is None:
                    raise ValueError("mask/clip extraction requires label_dir or dataset-wide class_ids")
                class_ids = class_ids_in_directory(self.label_dir)
            self.colors = class_color_map(class_ids)
        else:
            self.colors = {}
        self.weights = {
            "mask": Path(sam_weights or WEIGHTS / "sam2.1_hiera_large.pt").expanduser().resolve(),
            "edge": Path(edge_weights or WEIGHTS / "model_hed.pdparams").expanduser().resolve(),
            "depth": Path(depth_weights or WEIGHTS / "dpt_beit_large_512.pt").expanduser().resolve(),
        }
        self._models = {}

    @property
    def required_backends(self):
        selected = set(self.features)
        if self.needs_mask:
            selected.add("mask")
        return tuple(name for name in ("mask", "edge", "depth") if name in selected)

    def check_available(self):
        """Validate resources without allocating models or altering the environment."""
        missing = []
        for name in self.required_backends:
            if name in self._models:
                continue
            if not self.weights[name].is_file():
                missing.append(f"{name} weights: {self.weights[name]}")
            dependencies = {"mask": ("hydra", "iopath"), "edge": ("paddle", "tifffile"),
                            "depth": ("timm", "torchvision")}[name]
            for module in dependencies:
                if importlib.util.find_spec(module) is None:
                    missing.append(f"{name} Python dependency: {module}")
        if missing:
            raise RuntimeError("RACEM resources missing:\n  " + "\n  ".join(missing) + "\nSee README.md.")

    def _backend(self, name):
        if name not in self._models:
            self.check_available()
            if name == "edge":
                self._models[name] = HEDBackend(self.weights[name], self.device)
            else:
                factory = {"mask": SAMBackend, "depth": DepthBackend}[name]
                self._models[name] = factory(self.weights[name], self.device)
        return self._models[name]

    def extract(self, image_rgb, labels=None, *, bbox_type=None):
        image_rgb = np.asarray(image_rgb)
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("RACEM expects HxWx3 RGB uint8; convert radiometry before extraction")
        if labels is None:
            if self.needs_mask or self.region == "bbox":
                raise ValueError("This extraction needs YOLO labels; [] explicitly represents a negative sample")
            labels = []
        if isinstance(labels, str):
            labels = labels.splitlines()
        kind = bbox_type or self.bbox_type
        height, width = image_rgb.shape[:2]
        refined = semantic = None
        if self.needs_mask:
            initial = self._backend("mask")(image_rgb, labels, self.colors, kind)
            refined, semantic = refine_mask(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), initial)
        maps = {}
        if "edge" in self.features:
            edge = self._backend("edge")(image_rgb)
            if self.region == "clip":
                edge = np.where(refined > 0, edge, 0).astype(np.uint8)
            elif self.region == "bbox":
                keep = bbox_mask(labels, width, height, kind, clip_vertices=True)
                edge = np.where(keep > 0, edge, 0).astype(np.uint8)
            maps["edge"] = edge
        if "depth" in self.features:
            depth = self._backend("depth")(image_rgb)
            if self.region == "bbox":
                keep = bbox_mask(labels, width, height, kind)
                depth = np.where(keep > 0, depth, 0)
            color = depth_to_bgr(depth)
            if self.region == "clip":
                color = np.where((refined > 0)[..., None], color, 0).astype(np.uint8)
            maps["depth"] = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        if "mask" in self.features:
            maps["mask"] = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        for name, feature in maps.items():
            expected = (height, width) if name == "edge" else (height, width, 3)
            if feature.shape != expected or feature.dtype != np.uint8:
                raise ValueError(f"Invalid {name} output: shape={feature.shape}, dtype={feature.dtype}")
        return maps

    def extract_batch(self, images, labels=None, *, bbox_type=None, as_tensor=True,
                      output_device=None):
        """One call per batch; native extractors retain their original per-image rules.

        Tensor input: [B,3,H,W], RGB in [-1,1]. NumPy input: list of RGB uint8.
        Tensor output: canonical edge(1)+depth(3)+mask(3) selection, in [-1,1].
        No gradient/inference tensor is returned to the trainable fusion CNN.
        """
        if torch.is_tensor(images):
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError("Expected training image tensor [B,3,H,W]")
            if images.shape[0] == 0:
                raise ValueError("Cannot extract an empty batch")
            if not torch.isfinite(images).all() or images.min() < -1.001 or images.max() > 1.001:
                raise ValueError("Training images must be finite and normalized to [-1,1]")
            output_device = images.device if output_device is None else output_device
            arrays = ((images.detach().float().cpu()+1)*127.5).round().clamp(0,255)
            arrays = arrays.to(torch.uint8).permute(0,2,3,1).numpy()
        else:
            arrays = list(images)
        if not len(arrays):
            raise ValueError("Cannot extract an empty batch")
        labels = [None] * len(arrays) if labels is None else labels
        if len(labels) != len(arrays):
            raise ValueError("Image/label batch lengths do not match")
        results = [self.extract(image, lines, bbox_type=bbox_type) for image, lines in zip(arrays, labels)]
        if not as_tensor:
            return results
        tensors = []
        for result in results:
            channels = [result[name][..., None] if name == "edge" else result[name] for name in self.features]
            fused = np.concatenate(channels, axis=-1).copy()
            tensors.append(torch.from_numpy(fused).permute(2,0,1).float().div(127.5).sub(1))
        if len({tuple(value.shape) for value in tensors}) != 1:
            raise ValueError("A tensor batch needs equal image sizes; use as_tensor=False for varying sizes")
        return torch.stack(tensors).to(output_device or "cpu").detach()

    def configuration(self):
        config = {"features": list(self.features), "region": self.region, "bbox_type": self.bbox_type,
                "weights": {k: str(self.weights[k]) for k in self.required_backends},
                "class_colors_bgr": {str(cid): list(color) for cid, (_, color) in self.colors.items()},
                "mask": "SAM2.1 large + original safe level-set defaults",
                "edge": "Paddle HED; original scales 0.5/1.0/1.5, fused sigmoid, per-image max normalization",
                "depth": "MiDaS dpt_beit_large_512 + INFERNO"}
        return config

    def preprocess(self, images_dir, output_dir, *, batch_size=1, overwrite=False):
        images_dir, output_dir = Path(images_dir), Path(output_dir)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f"No supported images in {images_dir}")
        stems = [p.stem for p in paths]
        if len(set(stems)) != len(stems):
            raise ValueError("Duplicate image stems would overwrite condition maps; rename those images")
        if self.label_dir is None and (self.needs_mask or self.region == "bbox"):
            raise ValueError("preprocess needs label_dir for mask/bbox/clip extraction")
        self.check_available()
        configuration = self.configuration()
        manifest = output_dir / "racem_config.json"
        if manifest.exists() and json.loads(manifest.read_text()) != configuration:
            raise ValueError("Output directory contains a different RACEM configuration; choose a new directory")
        if not manifest.exists() and not overwrite:
            for name in self.features:
                folder = output_dir / name / f"{name}_{self.region}"
                if folder.is_dir() and any(folder.glob("*.png")):
                    raise ValueError("Existing maps have no RACEM manifest; use a new directory or --overwrite")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
        completed = skipped = 0
        for offset in range(0, len(paths), batch_size):
            pending, images, labels = [], [], []
            for path in paths[offset:offset+batch_size]:
                targets = {name: output_dir / name / f"{name}_{self.region}" / f"{path.stem}.png"
                           for name in self.features}
                if not overwrite and all(target.is_file() for target in targets.values()):
                    skipped += 1
                    continue
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Cannot read image: {path}")
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                labels.append(read_labels(self.label_dir / f"{path.stem}.txt") if self.label_dir else None)
                pending.append(targets)
            if not images:
                continue
            results = self.extract_batch(images, labels, as_tensor=False)
            for result, targets in zip(results, pending):
                for name, target in targets.items():
                    if target.exists() and not overwrite:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    array = result[name]
                    if array.ndim == 3:
                        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(target), array):
                        raise OSError(f"Cannot save condition map: {target}")
                completed += 1
        return {"processed": completed, "skipped": skipped, "total": len(paths)}
