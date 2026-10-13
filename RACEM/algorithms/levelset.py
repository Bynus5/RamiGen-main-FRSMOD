"""Original safe level-set mathematics and fixed reproduction defaults.

Lifted unchanged from RACEM_Level_Set_optimized.py; no model or path I/O.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt


LS_SAVE_OVERLAY = False

LS_ACCEPT_MAX_HD95_TO_RAW = 8.0

LS_DELTA_T = 0.35

LS_ITER_NUM = 16

LS_EPS_STOP = 1e-4

LS_USE_NARROW_BAND = True

LS_BAND_WIDTH = 4.5

LS_SMOOTH_INTERVAL = 2

LS_SMOOTH_SIGMA = 0.45

LS_REINIT_INTERVAL = 4

LS_IMAGE_SMOOTH_SIGMA = 0.55

LS_USE_EVOLUTION_ROI = True

LS_ROI_DILATE_KERNEL_SIZE = 15

LS_ROI_DILATE_ITERATIONS = 1

LS_ENABLE_FALLBACK_TO_RAW = True

LS_ACCEPT_MIN_IOU = 0.55

LS_ACCEPT_MIN_AREA_RATIO = 0.70

LS_ACCEPT_MAX_AREA_RATIO = 1.55

LS_ACCEPT_MAX_COMPONENT_RATIO = 1.50

LS_COMPONENT_MIN_AREA = 10

LS_REQUIRE_ENERGY_NONINCREASE = True

LS_ACCEPT_ENERGY_TOL = 0.03

LS_ADAPTIVE_CORE_ERODE_AREA = 300

LS_ENABLE_MORPHOLOGY = True

LS_ENABLE_FILL_HOLES = True

LS_ENABLE_CLOSING = True

LS_CLOSING_KERNEL_SIZE = 3

LS_CLOSING_ITERATIONS = 1

LS_ENABLE_OPENING = True

LS_OPENING_KERNEL_SIZE = 3

LS_OPENING_ITERATIONS = 1

LS_ENABLE_DILATION = False

LS_DILATION_KERNEL_SIZE = 3

LS_DILATION_ITERATIONS = 1

LS_ENABLE_EROSION = False

LS_EROSION_KERNEL_SIZE = 3

LS_EROSION_ITERATIONS = 1

LS_REMOVE_SMALL_REGIONS = True

LS_MIN_AREA = 20

LS_FINAL_CONTOUR_COLOR_BGR = (0, 0, 255)

LS_FINAL_CONTOUR_WIDTH = 2

FILL_REFINED_EXPANSION_WITH_DOMINANT_COLOR = True

DEFAULT_UNKNOWN_COLOR_BGR = (255, 255, 255)

def ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    if img.max() <= 1.0:
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    return np.clip(img, 0, 255).astype(np.uint8)

def dominant_nonzero_color_bgr(color_pixels: np.ndarray):
    """
    Return dominant non-black BGR color from Nx3 pixels.
    """
    if color_pixels.size == 0:
        return DEFAULT_UNKNOWN_COLOR_BGR

    nonzero = color_pixels[np.any(color_pixels > 0, axis=1)]

    if nonzero.size == 0:
        return DEFAULT_UNKNOWN_COLOR_BGR

    colors, counts = np.unique(nonzero.reshape(-1, 3), axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]

    return tuple(int(v) for v in dominant)

def colorize_refined_mask_from_initial_semantic(
    final_mask: np.ndarray,
    initial_color_mask: np.ndarray
) -> np.ndarray:
    """
    Construct final semantic color mask by inheriting colors from the initial SAM semantic mask.

    Rules:
    1. Background remains black.
    2. Where final_mask overlaps initial semantic color mask, keep the initial semantic color.
    3. If final_mask expands outside initial semantic color mask:
       - fill that connected component by its dominant overlapping semantic color;
       - if no overlap exists, use DEFAULT_UNKNOWN_COLOR_BGR.
    """
    final_fg = final_mask > 0
    h, w = final_mask.shape[:2]

    if initial_color_mask.shape[:2] != (h, w):
        initial_color_mask = cv2.resize(
            initial_color_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    final_color = np.zeros((h, w, 3), dtype=np.uint8)

    init_fg = np.any(initial_color_mask > 0, axis=2)

    overlap = final_fg & init_fg
    final_color[overlap] = initial_color_mask[overlap]

    if not FILL_REFINED_EXPANSION_WITH_DOMINANT_COLOR:
        return final_color

    missing = final_fg & (~init_fg)

    if not np.any(missing):
        return final_color

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        final_fg.astype(np.uint8),
        connectivity=8
    )

    for comp_id in range(1, num_labels):
        comp = labels == comp_id
        comp_missing = comp & missing

        if not np.any(comp_missing):
            continue

        comp_overlap = comp & overlap
        overlap_pixels = initial_color_mask[comp_overlap]

        fill_color = dominant_nonzero_color_bgr(overlap_pixels.reshape(-1, 3))
        final_color[comp_missing] = fill_color

    return final_color

def build_evolution_roi(raw_mask_u8: np.ndarray) -> np.ndarray:
    """
    Build a local ROI around the raw SAM2 mask.
    Level-set updates are only allowed inside this ROI.
    """
    raw = (raw_mask_u8 > 0).astype(np.uint8)

    if not LS_USE_EVOLUTION_ROI:
        return np.ones_like(raw, dtype=bool)

    if np.sum(raw) == 0:
        return np.zeros_like(raw, dtype=bool)

    k = max(3, int(LS_ROI_DILATE_KERNEL_SIZE))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    roi = cv2.dilate(
        raw,
        kernel,
        iterations=max(1, int(LS_ROI_DILATE_ITERATIONS))
    )

    return roi.astype(bool)

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(inter) / float(union)

def extract_boundary_u8(mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)

    if np.sum(m) == 0:
        return np.zeros_like(m, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    boundary = (m > 0) ^ (eroded > 0)

    return boundary

def boundary_hd95_to_raw(mask_u8: np.ndarray, raw_u8: np.ndarray) -> float:
    b1 = extract_boundary_u8(mask_u8)
    b2 = extract_boundary_u8(raw_u8)

    if not np.any(b1) and not np.any(b2):
        return 0.0
    if not np.any(b1) or not np.any(b2):
        return float("inf")

    dist_to_b2 = distance_transform_edt(~b2)
    dist_to_b1 = distance_transform_edt(~b1)

    d1 = dist_to_b2[b1]
    d2 = dist_to_b1[b2]

    return float(np.percentile(np.concatenate([d1, d2]), 95.0))

def count_components(mask_u8: np.ndarray, min_area: int = 20) -> int:
    m = (mask_u8 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        m,
        connectivity=8
    )

    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1

    return count

def signed_distance_from_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Signed-distance level-set function: positive inside, negative outside."""
    inside = mask_u8 > 0
    if not np.any(inside):
        return -np.ones_like(mask_u8, dtype=np.float32)
    if np.all(inside):
        return np.ones_like(mask_u8, dtype=np.float32)

    dist_in = distance_transform_edt(inside)
    dist_out = distance_transform_edt(~inside)
    return (dist_in - dist_out).astype(np.float32)

def image_features(image_bgr: np.ndarray) -> np.ndarray:
    """Use smoothed normalized image channels for robust local region statistics."""
    img = ensure_uint8_image(image_bgr)
    if img.ndim == 2:
        feat = img.astype(np.float32)[..., None] / 255.0
    else:
        feat = img.astype(np.float32) / 255.0

    if LS_IMAGE_SMOOTH_SIGMA > 0:
        k = max(3, int(round(LS_IMAGE_SMOOTH_SIGMA * 4)) * 2 + 1)
        if k % 2 == 0:
            k += 1
        if feat.shape[2] == 1:
            feat[..., 0] = cv2.GaussianBlur(feat[..., 0], (k, k), LS_IMAGE_SMOOTH_SIGMA)
        else:
            feat = cv2.GaussianBlur(feat, (k, k), LS_IMAGE_SMOOTH_SIGMA)
    return feat.astype(np.float32)

def local_region_energy(image_bgr: np.ndarray, mask_u8: np.ndarray, roi: np.ndarray) -> float:
    """Chan-Vese-style local data energy inside the evolution ROI."""
    feat = image_features(image_bgr)
    inside = (mask_u8 > 0) & roi
    outside = (~(mask_u8 > 0)) & roi
    if not np.any(inside) or not np.any(outside):
        return float("inf")

    c_in = feat[inside].mean(axis=0)
    c_out = feat[outside].mean(axis=0)

    # Use the Chan-Vese data term as total squared residual normalized by ROI size.
    # This keeps the energy comparable when the contour expands or shrinks.
    sse_in = np.sum((feat[inside] - c_in) ** 2)
    sse_out = np.sum((feat[outside] - c_out) ** 2)
    return float((sse_in + sse_out) / max(1, int(roi.sum())))

def evaluate_refined_mask(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    refined_mask: np.ndarray,
    roi: np.ndarray,
):
    """Accept useful local refinements while rejecting only degenerate results."""
    raw = raw_mask > 0
    ref = refined_mask > 0

    raw_area = int(raw.sum())
    ref_area = int(ref.sum())

    if raw_area == 0:
        return False, "raw mask empty"
    if ref_area == 0:
        return False, "refined mask empty"

    area_ratio = ref_area / (raw_area + 1e-6)
    iou = mask_iou(raw_mask, refined_mask)
    raw_comp = count_components(raw_mask, min_area=LS_COMPONENT_MIN_AREA)
    ref_comp = count_components(refined_mask, min_area=LS_COMPONENT_MIN_AREA)

    if area_ratio < LS_ACCEPT_MIN_AREA_RATIO:
        return False, f"area too small: ratio={area_ratio:.3f}"
    if area_ratio > LS_ACCEPT_MAX_AREA_RATIO:
        return False, f"area too large: ratio={area_ratio:.3f}"
    if iou < LS_ACCEPT_MIN_IOU:
        return False, f"IoU too low: iou={iou:.3f}"
    if raw_comp > 0:
        allowed_comp = max(
            raw_comp + 1,
            int(np.ceil(LS_ACCEPT_MAX_COMPONENT_RATIO * max(raw_comp, 1)))
        )
        if ref_comp > allowed_comp:
            return False, f"too many components: raw={raw_comp}, refined={ref_comp}, allowed={allowed_comp}"

    hd95_to_raw = boundary_hd95_to_raw(refined_mask, raw_mask)
    if hd95_to_raw > LS_ACCEPT_MAX_HD95_TO_RAW:
        return False, f"HD95 to raw too large: hd95={hd95_to_raw:.3f}"


    # Mild component changes are handled by the allowed_comp gate above.
    # Do not reject every component increase, otherwise useful boundary recovery
    # under incomplete coarse masks will be suppressed.

    raw_energy = local_region_energy(image_bgr, raw_mask, roi)
    ref_energy = local_region_energy(image_bgr, refined_mask, roi)
    if LS_REQUIRE_ENERGY_NONINCREASE and np.isfinite(raw_energy) and np.isfinite(ref_energy):
        if ref_energy > raw_energy * (1.0 + LS_ACCEPT_ENERGY_TOL):
            return False, f"energy increased: raw={raw_energy:.6f}, refined={ref_energy:.6f}"

    return True, (
        f"accepted: IoU={iou:.3f}, area_ratio={area_ratio:.3f}, "
        f"components={ref_comp}, Eraw={raw_energy:.6f}, Eref={ref_energy:.6f}"
    )

def postprocess_mask(binary_mask: np.ndarray) -> np.ndarray:
    mask = (binary_mask > 0).astype(np.uint8)

    if LS_ENABLE_FILL_HOLES:
        mask = binary_fill_holes(mask).astype(np.uint8)

    if LS_ENABLE_CLOSING:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (LS_CLOSING_KERNEL_SIZE, LS_CLOSING_KERNEL_SIZE)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=LS_CLOSING_ITERATIONS
        )

    if LS_ENABLE_OPENING:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (LS_OPENING_KERNEL_SIZE, LS_OPENING_KERNEL_SIZE)
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=LS_OPENING_ITERATIONS
        )

    if LS_ENABLE_DILATION:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (LS_DILATION_KERNEL_SIZE, LS_DILATION_KERNEL_SIZE)
        )
        mask = cv2.dilate(mask, kernel, iterations=LS_DILATION_ITERATIONS)

    if LS_ENABLE_EROSION:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (LS_EROSION_KERNEL_SIZE, LS_EROSION_KERNEL_SIZE)
        )
        mask = cv2.erode(mask, kernel, iterations=LS_EROSION_ITERATIONS)

    if LS_REMOVE_SMALL_REGIONS:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
        filtered = np.zeros_like(mask)

        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= LS_MIN_AREA:
                filtered[labels == idx] = 1

        mask = filtered

    if LS_ENABLE_FILL_HOLES:
        mask = binary_fill_holes(mask).astype(np.uint8)

    return (mask * 255).astype(np.uint8)

def canonicalize_raw_mask(raw_mask_u8: np.ndarray) -> np.ndarray:
    """
    Build a stable canonical SAM mask before fallback or conservative fusion.
    This removes tiny fragments and fills small holes consistently for all alpha values.
    """
    mask = (raw_mask_u8 > 0).astype(np.uint8)

    # Fill small holes inside objects.
    mask = binary_fill_holes(mask).astype(np.uint8)

    # Mild closing to suppress small boundary gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove tiny isolated fragments.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    filtered = np.zeros_like(mask)
    min_area = max(LS_MIN_AREA, 15)

    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == idx] = 1

    filtered = binary_fill_holes(filtered).astype(np.uint8)

    return (filtered * 255).astype(np.uint8)

def keep_components_overlapping_seed(mask_u8: np.ndarray, seed_u8: np.ndarray) -> np.ndarray:
    """
    Keep only connected components in mask_u8 that overlap with seed_u8.
    This prevents new isolated candidate components from being introduced.
    """
    mask = (mask_u8 > 0).astype(np.uint8)
    seed = seed_u8 > 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    kept = np.zeros_like(mask)

    for idx in range(1, num_labels):
        comp = labels == idx
        if np.any(comp & seed):
            kept[comp] = 1

    return (kept * 255).astype(np.uint8)

def constrain_candidate_near_raw(
    raw_mask_u8: np.ndarray,
    candidate_u8: np.ndarray,
    inner_keep_kernel: int = 3,
    outer_allow_kernel: int = 5,
) -> np.ndarray:
    """
    Keep candidate changes within a narrow band around the canonical raw mask.

    - Pixels safely inside raw are kept.
    - Candidate is allowed only inside a small dilation of raw.
    - This prevents distant background absorption and topology drift.
    """
    raw = (raw_mask_u8 > 0).astype(np.uint8)
    cand = (candidate_u8 > 0).astype(np.uint8)

    if np.sum(raw) == 0:
        return np.zeros_like(candidate_u8, dtype=np.uint8)

    k_in = max(3, int(inner_keep_kernel))
    if k_in % 2 == 0:
        k_in += 1

    k_out = max(3, int(outer_allow_kernel))
    if k_out % 2 == 0:
        k_out += 1

    kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_in, k_in))
    kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_out, k_out))

    # Stable object core should not be removed by level-set.
    # For small/thin objects, keep the full raw support; for larger objects,
    # use a mildly eroded core to allow correction of coarse false-positive regions.
    if int(raw.sum()) >= LS_ADAPTIVE_CORE_ERODE_AREA:
        raw_core = cv2.erode(raw, kernel_in, iterations=1)
    else:
        raw_core = raw.copy()

    # Candidate cannot expand far away from raw SAM.
    allowed_region = cv2.dilate(raw, kernel_out, iterations=1)

    constrained = ((cand & allowed_region) | raw_core).astype(np.uint8)

    # Remove candidate components that do not overlap with the raw SAM support.
    constrained = keep_components_overlapping_seed(
        mask_u8=constrained * 255,
        seed_u8=raw * 255,
    )
    constrained = (constrained > 0).astype(np.uint8)

    constrained = binary_fill_holes(constrained).astype(np.uint8)

    # Remove tiny fragments after fusion.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        constrained,
        connectivity=8
    )

    filtered = np.zeros_like(constrained)
    min_area = max(LS_MIN_AREA, 15)

    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == idx] = 1

    return (filtered * 255).astype(np.uint8)

def evolve_level_set_from_mask(
    image_bgr: np.ndarray,
    init_mask_u8: np.ndarray
) -> np.ndarray:
    """Local region-aware level-set refinement initialized by the SAM mask."""
    target = init_mask_u8 > 0
    if not np.any(target):
        return np.zeros_like(init_mask_u8, dtype=np.uint8)
    if np.all(target):
        return postprocess_mask(init_mask_u8)

    feat = image_features(image_bgr)
    roi = build_evolution_roi(init_mask_u8)
    phi = signed_distance_from_mask(init_mask_u8)

    for it in range(LS_ITER_NUM):
        phi_old = phi.copy()
        mask_in = phi >= 0
        mask_out = ~mask_in

        # Region statistics are deliberately local: only the neighborhood around
        # the SAM initialization participates in the adaptive evolution.
        local_in = mask_in & roi
        local_out = mask_out & roi
        if not np.any(local_in) or not np.any(local_out):
            break

        c_in = feat[local_in].mean(axis=0)
        c_out = feat[local_out].mean(axis=0)

        # Positive force means the pixel is more similar to the object interior.
        d_in = np.sum((feat - c_in) ** 2, axis=-1)
        d_out = np.sum((feat - c_out) ** 2, axis=-1)
        F = d_out - d_in

        if LS_USE_NARROW_BAND:
            update_mask = (np.abs(phi) <= float(LS_BAND_WIDTH)) & roi
        else:
            update_mask = roi.copy()

        if not np.any(update_mask):
            break

        fscale = np.max(np.abs(F[update_mask])) + 1e-9
        F = F / fscale

        phi_y, phi_x = np.gradient(phi)
        grad_norm = np.sqrt(phi_y ** 2 + phi_x ** 2) + 1e-8

        phi_new = phi.copy()
        phi_new[update_mask] = (
            phi[update_mask]
            + LS_DELTA_T * F[update_mask] * grad_norm[update_mask]
        )

        # Smooth only within the permitted ROI; never let smoothing move the
        # contour into unrelated distant background.
        if LS_SMOOTH_INTERVAL > 0 and (it + 1) % LS_SMOOTH_INTERVAL == 0:
            smoothed = gaussian_filter(phi_new, sigma=LS_SMOOTH_SIGMA)
            phi_new[roi] = smoothed[roi]
            phi_new[~roi] = phi[~roi]

        # Reinitialize to a real signed-distance function instead of +/-1.
        if LS_REINIT_INTERVAL > 0 and (it + 1) % LS_REINIT_INTERVAL == 0:
            tmp_mask = (phi_new >= 0).astype(np.uint8) * 255
            phi_new = signed_distance_from_mask(tmp_mask)

        # Convergence is measured only where evolution is allowed.
        max_d = float(np.max(np.abs(phi_new[roi] - phi_old[roi])))
        phi = phi_new
        if max_d < LS_EPS_STOP:
            break

    final_mask = (phi >= 0).astype(np.uint8) * 255
    if LS_ENABLE_MORPHOLOGY:
        final_mask = postprocess_mask(final_mask)
    return final_mask

def refine_mask(image_bgr, initial_color_mask):
    """The original canonicalize/evolve/constrain/quality-gate/fallback pipeline."""
    raw = np.any(initial_color_mask > 0, axis=2).astype(np.uint8) * 255
    if not np.any(raw):
        return np.zeros_like(raw), np.zeros_like(initial_color_mask)
    safe_raw = canonicalize_raw_mask(raw)
    candidate = evolve_level_set_from_mask(image_bgr, safe_raw)
    candidate = constrain_candidate_near_raw(safe_raw, candidate, inner_keep_kernel=3, outer_allow_kernel=7)
    if LS_ENABLE_MORPHOLOGY:
        candidate = postprocess_mask(candidate)
    roi = build_evolution_roi(safe_raw)
    accepted, _ = evaluate_refined_mask(image_bgr, safe_raw, candidate, roi)
    final = candidate if accepted or not LS_ENABLE_FALLBACK_TO_RAW else safe_raw
    return final, colorize_refined_mask_from_initial_semantic(final, initial_color_mask)

