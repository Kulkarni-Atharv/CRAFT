"""
Module 3 — Post-processing
Converts CRAFT score maps → bounding boxes (rotated rects or polygons).

Algorithm:
  1. Threshold the region map to find text pixels.
  2. Run connected-component labelling on the thresholded map.
  3. For each component, optionally expand using the affinity map.
  4. Fit a minimum-area rotated rectangle (or convex hull polygon).
  5. Scale boxes back to original image coordinates.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label

from utils.logger import get_logger

log = get_logger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _score_to_binary(score: np.ndarray, threshold: float) -> np.ndarray:
    return (score > threshold).astype(np.uint8)


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    labelled, n = scipy_label(binary)
    return labelled.astype(np.int32), n


def _component_to_box(
    mask: np.ndarray, poly: bool
) -> np.ndarray | None:
    pts = np.column_stack(np.where(mask > 0))[:, ::-1].astype(np.float32)  # → (x,y)
    if len(pts) < 4:
        return None
    if poly:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        return hull.reshape(-1, 2)
    rect   = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    box    = cv2.boxPoints(rect)
    return box.astype(np.float32)


# ── main class ────────────────────────────────────────────────────────────────

class PostProcessor:

    def __init__(self, cfg: dict[str, Any]):
        pcfg = cfg["postprocessing"]
        self.text_threshold: float = cfg["craft"]["text_threshold"]
        self.link_threshold: float = cfg["craft"]["link_threshold"]
        self.low_text:       float = cfg["craft"]["low_text"]
        self.poly:          bool  = pcfg["poly"]
        self.min_area:       int   = pcfg["min_box_area"]

    def extract_boxes(
        self,
        region_map:   np.ndarray,   # H×W float [0,1]
        affinity_map: np.ndarray,   # H×W float [0,1]
        scale:        float,         # preprocessing resize scale
        orig_size:    tuple[int, int],  # (orig_h, orig_w)
    ) -> list[np.ndarray]:
        """
        Returns a list of boxes.
        Each box is an Nx2 float32 array of (x, y) corners in *original* image space.
        For rotated-rect mode N=4; for polygon mode N≥3.
        """
        # threshold both maps independently, then OR them together
        # (matches original CRAFT post-processing algorithm)
        binary_region = (region_map   > self.low_text).astype(np.uint8)
        binary_link   = (affinity_map > self.link_threshold).astype(np.uint8)
        binary        = np.clip(binary_region + binary_link, 0, 1)

        labelled, n_comps = _connected_components(binary)
        log.info(
            "PostProcess — low_text=%.2f link_threshold=%.2f text_threshold=%.2f | "
            "components=%d | binary coverage=%.1f%%",
            self.low_text, self.link_threshold, self.text_threshold,
            n_comps, binary.mean() * 100,
        )

        boxes: list[np.ndarray] = []
        orig_h, orig_w = orig_size

        for comp_id in range(1, n_comps + 1):
            mask = (labelled == comp_id).astype(np.uint8)

            # apply region threshold inside each component
            seg = (region_map * mask) > self.text_threshold
            if not seg.any():
                continue

            box = _component_to_box(seg.astype(np.uint8), self.poly)
            if box is None:
                continue

            # area filter
            area = cv2.contourArea(box.reshape(-1, 1, 2).astype(np.float32))
            if area < self.min_area:
                continue

            # scale back to original image coordinates
            # score maps are at half the tensor resolution (CRAFT outputs /2)
            # and tensor was further downscaled by `scale`
            coord_scale = (1.0 / scale) * 2.0
            box = box * coord_scale

            # clamp to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)

            boxes.append(box.astype(np.float32))

        log.debug("Valid boxes after filtering: %d", len(boxes))
        return boxes
