"""
Module 2 — CRAFT Detection
Runs the CRAFT model and returns raw region / affinity score maps.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from models.craft_model import CRAFT, load_craft
from utils.logger import get_logger

log = get_logger(__name__)


class CRAFTDetector:

    def __init__(self, cfg: dict[str, Any]):
        dcfg = cfg["craft"]
        self.text_threshold: float = dcfg["text_threshold"]
        self.link_threshold: float = dcfg["link_threshold"]
        self.low_text:       float = dcfg["low_text"]

        use_cuda = dcfg["cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        log.info("CRAFT running on: %s", self.device)

        weights = cfg["paths"]["craft_weights"]
        self.model: CRAFT = load_craft(weights, self.device)
        log.info("CRAFT weights loaded from: %s", weights)

    @torch.inference_mode()
    def detect(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        tensor : 1×3×H×W float tensor

        Returns
        -------
        region_map   : H×W float32 array — per-pixel character confidence
        affinity_map : H×W float32 array — per-pixel link confidence
        """
        t0 = time.perf_counter()
        tensor = tensor.to(self.device)
        region, affinity = self.model(tensor)

        r = region[0].cpu().numpy()
        a = affinity[0].cpu().numpy()
        log.debug("Inference done in %.1f ms", (time.perf_counter() - t0) * 1000)
        return r, a
