"""
Module 1 — Preprocessing
Resizes, normalises, and converts a raw image to a CRAFT-ready NCHW array.
No torch dependency — works on both dev machine and CM5 (ONNX-only).
"""

from __future__ import annotations

import numpy as np
from typing import Any

from utils.image_utils import resize_aspect, pad_to_multiple
from utils.logger import get_logger

log = get_logger(__name__)


class Preprocessor:
    """Stateless image preprocessor — returns plain numpy, not a torch Tensor."""

    def __init__(self, cfg: dict[str, Any]):
        pcfg = cfg["preprocessing"]
        self.target_size: int = pcfg["target_size"]
        self.mean = np.array(pcfg["mean"], dtype=np.float32)
        self.std  = np.array(pcfg["std"],  dtype=np.float32)

    def process(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Parameters
        ----------
        img : H×W×3 RGB uint8 ndarray

        Returns
        -------
        tensor    : 1×3×H'×W' float32 NCHW numpy array (padded to ×32)
        scale     : resize scale applied
        orig_size : (orig_h, orig_w)
        """
        orig_h, orig_w = img.shape[:2]
        resized, scale = resize_aspect(img, self.target_size)
        padded = pad_to_multiple(resized, multiple=32)

        x = padded.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        tensor = x.transpose(2, 0, 1)[np.newaxis]   # HWC → 1×C×H×W

        log.debug(
            "Preprocessed: orig=%s → padded=%s  scale=%.4f",
            (orig_h, orig_w), padded.shape[:2], scale,
        )
        return tensor, scale, (orig_h, orig_w)
