"""
Module 4 — Cropping
Performs a perspective-corrected crop for each detected bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.image_utils import order_points
from utils.logger import get_logger

log = get_logger(__name__)


class Cropper:

    def __init__(self, cfg: dict[str, Any]):
        self.padding:    int  = cfg["postprocessing"]["padding"]
        self.save_crops: bool = cfg["pipeline"]["save_crops"]
        self.crops_dir: str  = cfg["paths"]["crops_dir"]

    def crop(
        self,
        img:    np.ndarray,       # H×W×3 RGB original image
        boxes:  list[np.ndarray], # list of Nx2 corner arrays
        prefix: str = "frame",    # used for saved filenames
    ) -> list[np.ndarray]:
        """
        Returns a list of cropped RGB patches, one per box.
        If save_crops is True, each patch is also written to disk.
        """
        crops = []
        for i, box in enumerate(boxes):
            crop = self._perspective_crop(img, box)
            if crop is None:
                continue
            crops.append(crop)
            if self.save_crops:
                out_path = Path(self.crops_dir) / f"{prefix}_{i:04d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        log.debug("Cropped %d regions", len(crops))
        return crops

    # ── internal ──────────────────────────────────────────────────────────────

    def _perspective_crop(
        self, img: np.ndarray, box: np.ndarray
    ) -> np.ndarray | None:
        """Four-point perspective transform → axis-aligned rectangle."""
        if len(box) == 4:
            src = order_points(box)
        else:
            # polygon — take its bounding rotated rect
            rect = cv2.minAreaRect(box.reshape(-1, 1, 2))
            src = order_points(cv2.boxPoints(rect).astype(np.float32))

        tl, tr, br, bl = src

        w = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl),
        )) + self.padding * 2

        h = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl),
        )) + self.padding * 2

        if w <= 0 or h <= 0:
            return None

        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )

        # add padding to source points
        src_padded = src + np.array(
            [[-self.padding, -self.padding],
             [ self.padding, -self.padding],
             [ self.padding,  self.padding],
             [-self.padding,  self.padding]], dtype=np.float32
        )
        src_padded[:, 0] = np.clip(src_padded[:, 0], 0, img.shape[1] - 1)
        src_padded[:, 1] = np.clip(src_padded[:, 1], 0, img.shape[0] - 1)

        M = cv2.getPerspectiveTransform(src_padded, dst)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped
