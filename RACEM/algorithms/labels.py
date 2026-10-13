"""YOLO label I/O, the original SAM class palette, and training geometry."""

from pathlib import Path

import cv2
import numpy as np


def read_labels(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing YOLO label: {path}; an empty .txt is a valid negative sample.")
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def class_ids_in_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {directory}")
    ids = set()
    for path in sorted(directory.glob("*.txt")):
        for line in read_labels(path):
            try:
                ids.add(int(line.split()[0]))
            except ValueError as exc:
                raise ValueError(f"Invalid class id in {path}: {line}") from exc
    return sorted(ids)


def class_color_map(class_ids):
    """Exactly the HSV palette used by Mask_extractor.generate_class_color_map."""
    class_ids = sorted(set(class_ids))
    colors = {}
    for idx, cid in enumerate(class_ids):
        hue = int(180 * idx / max(1, len(class_ids)))
        bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors[cid] = (f"class_{cid}", tuple(int(c) for c in bgr))
    return colors


def parse_labels(lines, bbox_type="obb"):
    """Return (class_id, normalized 4x2 corners); never change the class order."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    if bbox_type not in {"obb", "hbb"}:
        raise ValueError("bbox_type must be obb or hbb")
    parsed = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        count = 9 if bbox_type == "obb" else 5
        if len(parts) < count:
            raise ValueError(f"Expected YOLO {bbox_type} with {count} fields: {line}")
        cid = int(parts[0])
        values = np.asarray(parts[1:count], dtype=np.float64)
        if cid < 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid YOLO label: {line}")
        if bbox_type == "obb":
            points = values.reshape(4, 2)
        else:
            cx, cy, w, h = values
            if w < 0 or h < 0:
                raise ValueError(f"Negative HBB size: {line}")
            points = np.array([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2],
                               [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
        parsed.append((cid, points))
    return parsed


def rotated_boxes(lines, width, height, bbox_type="obb"):
    return [(cid, cv2.minAreaRect((points * [width, height]).astype(np.float32)))
            for cid, points in parse_labels(lines, bbox_type)]


def bbox_mask(lines, width, height, bbox_type="obb", clip_vertices=False):
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, rbox in rotated_boxes(lines, width, height, bbox_type):
        points = cv2.boxPoints(rbox).astype(np.int32)
        if clip_vertices:  # Preserve the original HED rbox_to_mask convention.
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask


def transform_labels(lines, source_size, target_size, crop_xy, horizontal_flip,
                     bbox_type="obb", multireso=False):
    """Follow the existing Arrow loader's resize/crop/flip, including rounding.

    Output is OBB even for HBB inputs. Keep unclipped corners for SAM prompts,
    but discard boxes with no visible area after the crop. There is no object cap.
    """
    sw, sh = source_size
    tw, th = target_size
    if multireso:
        if sh / sw < th / tw:
            rw, rh = int(round(th / sh * sw)), th
        else:
            rw, rh = tw, int(round(tw / sw * sh))
    elif sw < sh:
        rw, rh = tw, int(tw / (sw / sh))
    else:
        rw, rh = int(th * (sw / sh)), th
    rectangle = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    result = []
    for cid, points in parse_labels(lines, bbox_type):
        pixels = points * [rw, rh] - np.asarray(crop_xy)
        if horizontal_flip:
            pixels[:, 0] = tw - pixels[:, 0]
        hull = cv2.convexHull(pixels.astype(np.float32))
        area, _ = cv2.intersectConvexConvex(hull, rectangle)
        if area <= 0:
            continue
        normalized = pixels / [tw, th]
        result.append(str(cid) + " " + " ".join(f"{v:.12g}" for v in normalized.ravel()))
    return "\n".join(result)
