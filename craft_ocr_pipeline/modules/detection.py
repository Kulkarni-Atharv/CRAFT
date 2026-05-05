"""
Module 2 — CRAFT Detection
Supports two back-ends selected by config:

  craft.use_onnx: false  →  PyTorch  (dev machine / GPU server)
  craft.use_onnx: true   →  ONNX Runtime  (CM5 / edge device, no torch needed)

Both paths accept the numpy NCHW array from Preprocessor and return the same
(region_map, affinity_map) numpy arrays.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


class CRAFTDetector:

    def __init__(self, cfg: dict[str, Any]):
        dcfg = cfg["craft"]
        self.text_threshold: float = dcfg["text_threshold"]
        self.link_threshold: float = dcfg["link_threshold"]
        self.low_text:       float = dcfg["low_text"]

        use_onnx = dcfg.get("use_onnx", False)
        if not use_onnx:
            # auto-fallback: if torch is not installed, switch to ONNX silently
            try:
                import torch  # noqa: F401
            except ImportError:
                log.warning("torch not found — switching CRAFT to ONNX backend automatically")
                use_onnx = True

        self._use_onnx = use_onnx

        if self._use_onnx:
            self._init_onnx(cfg)
        else:
            self._init_torch(cfg)

    # ── back-end init ──────────────────────────────────────────────────────────

    def _init_torch(self, cfg: dict) -> None:
        import torch
        from models.craft_model import load_craft

        use_cuda = cfg["craft"]["cuda"] and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model  = load_craft(cfg["paths"]["craft_weights"], self._device)
        log.info("CRAFT backend: PyTorch  device=%s", self._device)

    def _init_onnx(self, cfg: dict) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise ImportError("Install onnxruntime: pip install onnxruntime") from e

        onnx_path = cfg["paths"]["craft_onnx"]
        if not __import__("pathlib").Path(onnx_path).exists():
            raise FileNotFoundError(
                f"CRAFT ONNX model not found: {onnx_path}\n"
                "Export it from your dev machine first:\n"
                "  python scripts/export_onnx.py --craft-only\n"
                "Then copy models/craft.onnx to the CM5."
            )
        self._session    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        log.info("CRAFT backend: ONNX Runtime  model=%s", onnx_path)

    # ── public API ─────────────────────────────────────────────────────────────

    def detect(
        self, tensor: np.ndarray          # 1×3×H×W float32 numpy
    ) -> tuple[np.ndarray, np.ndarray]:   # (region_map H×W, affinity_map H×W)
        t0 = time.perf_counter()

        if self._use_onnx:
            r, a = self._run_onnx(tensor)
        else:
            r, a = self._run_torch(tensor)

        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("CRAFT inference %.1f ms", elapsed)
        log.info(
            "CRAFT score maps — region: min=%.3f max=%.3f mean=%.3f | "
            "affinity: min=%.3f max=%.3f mean=%.3f",
            r.min(), r.max(), r.mean(),
            a.min(), a.max(), a.mean(),
        )
        return r, a

    # ── internal runners ───────────────────────────────────────────────────────

    def _run_torch(self, tensor: np.ndarray):
        import torch
        t = torch.from_numpy(tensor).to(self._device)
        with torch.inference_mode():
            region, affinity = self._model(t)
        return region[0].cpu().numpy(), affinity[0].cpu().numpy()

    def _run_onnx(self, tensor: np.ndarray):
        outputs = self._session.run(None, {self._input_name: tensor})
        # outputs[0] = region  [1, H, W]
        # outputs[1] = affinity [1, H, W]
        return outputs[0].squeeze(0), outputs[1].squeeze(0)
