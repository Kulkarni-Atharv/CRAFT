"""
Module 5 — Recognition
PaddleOCR recognition on character-level crops produced by CRAFT.

Flow
----
  CRAFT region_map → individual character boxes → crop each box
  → PaddleOCR recognition per crop (det=False, recognition only)
  → filter_by_char_conf drops any character below confidence threshold
  → pipeline assembles surviving characters into words

"No prediction" guarantee
--------------------------
PaddleOCR returns a word-level confidence score for each crop.
If that score is below char_conf_threshold (default 0.7) the character
is returned as "" — it is silently skipped when words are assembled.
Example: "boy" with 'b' occluded → ["", "o", "y"] → word text "oy".
No guessing. No hallucination.

Platform note
-------------
PaddleOCR requires paddlepaddle which only supports Python 3.8-3.11.
On CM5 (Python 3.13) you must create a Python 3.11 venv via pyenv:
  pyenv install 3.11.9
  python -m venv venv311 && source venv311/bin/activate
  pip install paddlepaddle paddleocr onnxruntime numpy opencv-python scipy Pillow PyYAML
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
    # char_preds is empty for PaddleOCR (word-level API only)


# ── Confidence filter ─────────────────────────────────────────────────────────

def filter_by_char_conf(
    result: RecognitionResult,
    threshold: float,
) -> tuple[str, float]:
    """
    Return (text, conf) if confidence >= threshold, otherwise ("", 0.0).
    Returning 0.0 (not the original low conf) keeps word-confidence
    calculations clean — rejected characters don't dilute word scores.
    """
    if result.char_preds:
        kept = [(c, p) for c, p in result.char_preds if p >= threshold]
        if not kept:
            return ("", 0.0)
        return ("".join(c for c, _ in kept),
                sum(p for _, p in kept) / len(kept))

    # PaddleOCR — word-level confidence only
    if result.word_conf >= threshold:
        return (result.word_text, result.word_conf)
    return ("", 0.0)   # rejected: return 0.0 so it is excluded from word conf mean


# ── PaddleOCR back-end ────────────────────────────────────────────────────────

class PaddleOCRRecognizer:
    """
    Recognition-only PaddleOCR — CRAFT handles detection.
    Each crop is a single character extracted by CRAFT.
    Small crops are upscaled before recognition.
    """

    def __init__(self, cfg: dict[str, Any]):
        rcfg = cfg["recognition"]
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PaddleOCR requires paddlepaddle (Python 3.11 only).\n"
                "On CM5 install via pyenv:\n"
                "  pyenv install 3.11.9\n"
                "  pip install paddlepaddle paddleocr"
            ) from e

        # Try API variants — PaddleOCR v2/v3 have different constructor params
        for kwargs in [
            dict(use_angle_cls=True, lang=rcfg["lang"],
                 use_gpu=rcfg.get("use_gpu", False),
                 det=False, cls=True, show_log=False),
            dict(use_angle_cls=True, lang=rcfg["lang"],
                 device="gpu" if rcfg.get("use_gpu", False) else "cpu"),
            dict(lang=rcfg["lang"]),
        ]:
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError("Could not initialise PaddleOCR with any known API variant")

        log.info("PaddleOCR recogniser ready (lang=%s)", rcfg["lang"])

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

        # Upscale tiny character crops — PaddleOCR accuracy drops sharply below 32px
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            scale = max(32.0 / h, 32.0 / w)
            crop = cv2.resize(
                crop,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        for call_kwargs in [dict(det=False, cls=True), dict(det=False), {}]:
            try:
                out = self._ocr.ocr(crop, **call_kwargs)
                text, conf = self._parse(out)
                if text:
                    return text, conf
            except Exception:
                continue
        return "", 0.0

    @staticmethod
    def _parse(out) -> tuple[str, float]:
        """Handle PaddleOCR v2 and v3 output formats."""
        if not out:
            return "", 0.0
        first = out[0]
        if not first:
            return "", 0.0

        item = first[0] if isinstance(first, (list, tuple)) else first

        if isinstance(item, (list, tuple)) and len(item) == 2:
            t, c = item
            if isinstance(t, str):
                return t, float(c)
            # v2 format with bounding box: item = [box, ('text', conf)]
            if isinstance(c, (list, tuple)) and len(c) == 2:
                return str(c[0]), float(c[1])

        # v3 object format
        if hasattr(item, "text"):
            return str(item.text), float(getattr(item, "score", 0.0))

        return "", 0.0


# ── Factory ───────────────────────────────────────────────────────────────────

def build_recognizer(cfg: dict[str, Any]) -> PaddleOCRRecognizer:
    engine = cfg["recognition"]["engine"].lower()
    if engine != "paddleocr":
        raise ValueError(
            f"Unknown recognition engine: {engine!r}. "
            "Only 'paddleocr' is supported."
        )
    return PaddleOCRRecognizer(cfg)
