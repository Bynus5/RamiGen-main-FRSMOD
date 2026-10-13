"""Original bbox mask morphology, relocated unchanged from Mask_extractor.py."""

import cv2
import numpy as np


def process_mask_in_rbox(mask, rbox, H, W, min_area=10, kernel_size=3):
    """在旋转框区域内执行原有形态学修复。"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    box_mask = np.zeros((H, W), dtype=np.uint8)
    box_pts = cv2.boxPoints(rbox).astype(np.int32)
    cv2.fillPoly(box_mask, [box_pts], 1)

    local_mask = cv2.bitwise_and(mask, box_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local_mask, connectivity=8)
    cleaned = np.zeros_like(local_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)

    final = cv2.bitwise_and(filled, box_mask)
    final = cv2.erode(final, kernel, iterations=1)

    return final.astype(bool)
