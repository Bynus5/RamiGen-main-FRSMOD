"""Lazy adapters around the provided extractor implementations, not replacements."""

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
WEIGHTS = ROOT.parent / "ckpts/racem"


def _vendor_import(root, module, package):
    """Use this checkout, and refuse an already-imported, unrelated vendor copy."""
    loaded = sys.modules.get(package)
    if loaded is not None and getattr(loaded, "__file__", None):
        if not Path(loaded.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"{package} was already imported from {loaded.__file__}, not {root}")
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))


class SAMBackend:
    def __init__(self, checkpoint, device):
        try:
            builder = _vendor_import(VENDOR / "sam2", "sam2.build_sam", "sam2")
            predictor = importlib.import_module("sam2.sam2_image_predictor")
        except ModuleNotFoundError as exc:
            raise ImportError(f"SAM2 requires '{exc.name}'; see RACEM/README.md") from exc
        self.device = torch.device(device)
        self.model = builder.build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml", str(checkpoint), device=str(self.device)
        ).eval().requires_grad_(False)
        self.predictor = predictor.SAM2ImagePredictor(self.model)
        from .mask_cleanup import process_mask_in_rbox
        self.clean_mask = process_mask_in_rbox

    def __call__(self, image_rgb, lines, colors, bbox_type):
        from .labels import rotated_boxes

        height, width = image_rgb.shape[:2]
        groups = {cid: np.zeros((height, width), dtype=np.uint8) for cid in colors}
        boxes = rotated_boxes(lines, width, height, bbox_type)
        if boxes:
            with torch.inference_mode():
                self.predictor.set_image(image_rgb)
        for cid, box in boxes:
            if cid not in colors:
                raise ValueError(f"Class {cid} is absent from the dataset-wide RACEM palette")
            points = cv2.boxPoints(box)
            prompt = np.r_[points.min(axis=0), points.max(axis=0)]
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
                masks, _, _ = self.predictor.predict(
                    box=torch.as_tensor(prompt[None], device=self.device), multimask_output=False
                )
            if torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            raw = np.asarray(masks).reshape(height, width).astype(np.uint8)
            clean = self.clean_mask(raw, box, height, width, min_area=10, kernel_size=3)
            groups[cid][clean] = 1
        color = np.zeros((height, width, 3), dtype=np.uint8)
        for cid, mask in groups.items():
            color[mask > 0] = colors[cid][1]
        return color  # Original Mask_extractor uses BGR semantic colors.


class HEDBackend:
    """Use the supplied Paddle HED and its original multi-scale inference."""

    def __init__(self, checkpoint, device):
        import os
        # Avoid reserving the entire GPU alongside PyTorch/SAM2/MiDaS.
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        try:
            import paddle
            from RACEM.edge_detector.hed_model import HED
            from RACEM.edge_detector.predict import hed_infer
        except ModuleNotFoundError as exc:
            raise ImportError(f"HED requires '{exc.name}'; see the project README.md") from exc
        self.paddle = paddle
        self.infer = hed_infer
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not paddle.is_compiled_with_cuda():
                raise RuntimeError("CUDA HED requires paddlepaddle-gpu, not the CPU-only Paddle package")
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.paddle_device = f"gpu:{index}"
        elif self.device.type == "cpu":
            self.paddle_device = "cpu"
        else:
            raise ValueError(f"HED supports CPU/CUDA, not {self.device}")
        previous = paddle.get_device()
        try:
            paddle.set_device(self.paddle_device)
            self.model = HED()
            state = paddle.load(str(checkpoint))
            expected = self.model.state_dict()
            missing = [name for name in expected if name not in state]
            mismatched = [name for name in expected if name in state
                          and list(state[name].shape) != list(expected[name].shape)]
            if missing or mismatched:
                raise ValueError(f"Incomplete HED checkpoint: missing={missing}, shape_mismatches={mismatched}")
            self.model.set_state_dict({name: state[name] for name in expected})
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.stop_gradient = True
        finally:
            paddle.set_device(previous)

    def __call__(self, image_rgb):
        previous = self.paddle.get_device()
        try:
            self.paddle.set_device(self.paddle_device)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            edge = self.infer(self.model, image_bgr)
        finally:
            self.paddle.set_device(previous)
        if edge.shape != image_rgb.shape[:2] or edge.dtype != np.uint8:
            raise ValueError(f"HED returned invalid edges: shape={edge.shape}, dtype={edge.dtype}")
        return edge


class DepthBackend:
    def __init__(self, checkpoint, device):
        try:
            loader = _vendor_import(VENDOR / "midas", "midas.model_loader", "midas")
        except ModuleNotFoundError as exc:
            raise ImportError(f"MiDaS requires '{exc.name}'; see RACEM/README.md") from exc
        self.device = torch.device(device)
        self.model, self.transform, _, _ = loader.load_model(
            self.device, str(checkpoint), "dpt_beit_large_512", optimize=False,
            height=None, square=False,
        )
        self.model.eval().requires_grad_(False)

    def __call__(self, image_rgb):
        height, width = image_rgb.shape[:2]
        transformed = self.transform({"image": image_rgb.astype(np.float64) / 255.0})["image"]
        sample = torch.from_numpy(transformed).to(self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(sample)
            return torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False
            )[0, 0].cpu().numpy()


def depth_to_bgr(depth):
    """The original MiDaS write_depth(..., grayscale=False, bits=2), without disk I/O."""
    depth = np.asarray(depth)
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = depth.min(), depth.max()
    value = 255 * (depth - low) / (high - low) if high-low > np.finfo("float").eps else np.zeros_like(depth)
    return cv2.applyColorMap(value.astype(np.uint8), cv2.COLORMAP_INFERNO)
