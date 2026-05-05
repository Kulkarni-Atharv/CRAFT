"""
Module 5 — Recognition
PaddleOCR 3.x recognition on character-level crops produced by CRAFT.

PaddleOCR 3.x uses PaddlePaddle 3.0 which has proper ARM64 support.
Install on CM5:
  pip uninstall paddlepaddle paddleocr -y
  pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu-aarch64/
  pip install paddleocr==3.4.1
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


# ── PaddleOCR 3.x back-end ────────────────────────────────────────────────────

class PaddleOCRRecognizer:
    """
    Recognition-only PaddleOCR 3.x — CRAFT handles detection.
    Requires PaddlePaddle 3.0 (ARM64-native) + PaddleOCR 3.4.1.

    Install on CM5:
      pip uninstall paddlepaddle paddleocr -y
      pip install paddlepaddle==3.0.0 \\
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu-aarch64/
      pip install paddleocr==3.4.1
    """

    def __init__(self, cfg: dict[str, Any]):
        rcfg = cfg["recognition"]
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install PaddleOCR 3.x:\n"
                "  pip install paddlepaddle==3.0.0 "
                "-i https://www.paddlepaddle.org.cn/packages/stable/cpu-aarch64/\n"
                "  pip install paddleocr==3.4.1"
            ) from e

        # PaddleOCR 3.x: disable heavy sub-models not needed for recognition
        for kwargs in [
            # 3.x preferred — disable doc orientation + unwarping classifiers
            dict(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=rcfg["lang"],
            ),
            # 3.x minimal
            dict(lang=rcfg["lang"]),
            # 2.x fallback (if older version somehow installed)
            dict(
                use_angle_cls=False,
                lang=rcfg["lang"],
                det=False,
                show_log=False,
            ),
        ]:
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError("Could not initialise PaddleOCR")

        log.info("PaddleOCR 3.x recogniser ready (lang=%s)", rcfg["lang"])

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

        # Upscale tiny character crops — accuracy drops sharply below 32px
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            scale = max(32.0 / h, 32.0 / w)
            crop = cv2.resize(
                crop,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        # Try recognition-only (det=False) — fall back to full pipeline
        for call_kwargs in [
            dict(det=False, cls=False),
            dict(det=False),
            {},
        ]:
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
        """Handle PaddleOCR 2.x and 3.x output formats."""
        if not out:
            return "", 0.0

        # 3.x returns a list of Result objects or list-of-lists
        first = out[0]
        if not first:
            return "", 0.0

        item = first[0] if isinstance(first, (list, tuple)) else first

        # Standard tuple format: ('text', conf)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            t, c = item
            if isinstance(t, str):
                return t.strip(), float(c)
            # 2.x with bounding box: [box, ('text', conf)]
            if isinstance(c, (list, tuple)) and len(c) == 2:
                return str(c[0]).strip(), float(c[1])

        # 3.x object format
        if hasattr(item, "text"):
            return str(item.text).strip(), float(getattr(item, "score", 0.0))

        # 3.x dict format
        if isinstance(item, dict):
            text = item.get("text", item.get("rec_text", ""))
            conf = item.get("score", item.get("rec_score", 0.0))
            return str(text).strip(), float(conf)

        return "", 0.0


# ── Factory ───────────────────────────────────────────────────────────────────

def build_recognizer(cfg: dict[str, Any]) -> PaddleOCRRecognizer:
    engine = cfg["recognition"]["engine"].lower()
    if engine != "paddleocr":
        raise ValueError(
            f"Unknown recognition engine: {engine!r}. Only 'paddleocr' is supported."
        )
    return PaddleOCRRecognizer(cfg)
