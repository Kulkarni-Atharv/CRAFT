"""
Module 5 — Recognition
RapidOCR on character-level crops produced by CRAFT.

RapidOCR runs the same PP-OCR models as PaddleOCR but via ONNX Runtime —
no PaddlePaddle binary required. Works on ARM64/CM5 Python 3.11.

Install:
  pip install rapidocr-onnxruntime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    word_text:  str
    word_conf:  float
    char_preds: list[tuple[str, float]] = field(default_factory=list)


# ── Confidence filter ─────────────────────────────────────────────────────────

def filter_by_char_conf(
    result: RecognitionResult,
    threshold: float,
) -> tuple[str, float]:
    """
    Return (text, conf) if confidence >= threshold, else ("", 0.0).
    Rejected characters return 0.0 so they are excluded from word conf mean.
    "boy" with occluded b (conf 0.3 < 0.7) -> ("", 0.0) -> assembled word "oy"
    """
    if result.char_preds:
        kept = [(c, p) for c, p in result.char_preds if p >= threshold]
        if not kept:
            return ("", 0.0)
        return ("".join(c for c, _ in kept),
                sum(p for _, p in kept) / len(kept))

    if result.word_conf >= threshold:
        return (result.word_text, result.word_conf)
    return ("", 0.0)


# ── RapidOCR back-end ─────────────────────────────────────────────────────────

class RapidOCRRecognizer:
    """
    PP-OCR recognition via ONNX Runtime — same models as PaddleOCR, no
    PaddlePaddle binary needed. Works on ARM64 / CM5.

    Install:  pip install rapidocr-onnxruntime
    """

    def __init__(self, cfg: dict[str, Any]):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install RapidOCR: pip install rapidocr-onnxruntime"
            ) from e

        self._ocr = RapidOCR()
        log.info("RapidOCR recogniser ready (PP-OCR models via ONNX Runtime)")

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(RecognitionResult("", 0.0))
                continue
            text, conf = self._run_one(crop)
            results.append(RecognitionResult(str(text), float(conf)))
        return results

    def _run_one(self, crop: np.ndarray) -> tuple[str, float]:
        import cv2

        # Upscale tiny character crops — accuracy drops below 32px
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            scale = max(32.0 / h, 32.0 / w)
            crop = cv2.resize(
                crop,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            # use_det=False: skip detection, send whole crop to recogniser
            result, _ = self._ocr(crop, use_det=False, use_cls=False, use_rec=True)
            if result and result[0]:
                # result[0] = (box_or_None, text, confidence)
                row = result[0]
                text = str(row[1]).strip() if len(row) > 1 else ""
                conf = float(row[2])        if len(row) > 2 else 0.0
                return text, conf
        except Exception as e:
            log.debug("RapidOCR inference error: %s", e)

        return "", 0.0


# ── Factory ───────────────────────────────────────────────────────────────────

def build_recognizer(cfg: dict[str, Any]) -> RapidOCRRecognizer:
    engine = cfg["recognition"]["engine"].lower()
    if engine not in ("paddleocr", "rapidocr"):
        raise ValueError(
            f"Unknown recognition engine: {engine!r}. Use 'paddleocr' (runs via RapidOCR ONNX)."
        )
    return RapidOCRRecognizer(cfg)
