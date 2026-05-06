"""
Module — Isolated Character Classifier
Classifies one character crop at a time using a lightweight CNN.

Key design:
  - NO sequence model (no LSTM, no CTC, no language model)
  - NO dictionary correction or beam-search guessing
  - One crop → one softmax distribution → one character (or UNKNOWN)
  - Confidence gate: max(softmax) < threshold → UNKNOWN, not a guess

Model:
  ONNX export of a MobileNetV3-Small fine-tuned on EMNIST Extended.
  Input  : [1, 3, 32, 32] float32, ImageNet-normalised RGB
  Output : [1, N_CLASSES]  float32, raw logits (softmax applied here)

Fallback (no ONNX model present):
  RapidOCR in single-char mode with strict confidence gate.
  Disable use_det and use_cls to prevent sequence-level context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.char_verifier import VerificationResult
from modules.char_segmenter import CharSegment
from utils.logger import get_logger

log = get_logger(__name__)


# ── printable ASCII character set (95 chars, space … tilde) ──────────────────

PRINTABLE_ASCII = [chr(i) for i in range(32, 127)]   # 95 characters
N_CLASSES       = len(PRINTABLE_ASCII)                # 95

# ImageNet normalisation constants (same as preprocessor)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ClassifierResult:
    character:  str    # predicted character, "" (blank), or "?" (unknown)
    confidence: float  # max softmax probability  (0.0 if blank/unknown)
    status:     str    # "predicted" | "unknown" | "skipped"
    reason:     str    # "" | "low_confidence" | "verification_failed"


# ── Classifier ────────────────────────────────────────────────────────────────

class CharClassifier:
    """
    Isolated single-character classifier — no language model context.

    Usage:
        results = classifier.classify_batch(segments, verifier_results)
    """

    def __init__(self, cfg: dict[str, Any]):
        ccfg = cfg["char_classifier"]
        self.confidence_threshold: float = ccfg["confidence_threshold"]   # 0.70
        self.input_size:           int   = ccfg.get("input_size", 32)
        self._model_path:          str   = ccfg["model_path"]
        self._session = None
        self._fallback_ocr = None
        self._init_model()

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        from pathlib import Path
        if Path(self._model_path).exists():
            self._init_onnx()
        else:
            log.warning(
                "char_classifier.onnx not found at %s — using RapidOCR fallback. "
                "Train and export the model for full 'no-prediction' guarantee.",
                self._model_path,
            )
            self._init_rapidocr_fallback()

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("pip install onnxruntime") from e

        self._session    = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info("CharClassifier: ONNX model loaded from %s", self._model_path)

    def _init_rapidocr_fallback(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._fallback_ocr = RapidOCR()
            log.info("CharClassifier: RapidOCR fallback ready (sequence model — limited guarantee)")
        except ImportError as e:
            raise ImportError("pip install rapidocr-onnxruntime") from e

    # ── public API ────────────────────────────────────────────────────────────

    def classify_batch(
        self,
        segments:      list[CharSegment],
        ver_results:   list[VerificationResult],
    ) -> list[ClassifierResult]:
        """
        Classify a list of segments.  Segments whose verification status is not
        VERIFIED are passed through as blank/unknown without touching the model.
        """
        results: list[ClassifierResult] = []

        for seg, ver in zip(segments, ver_results):
            if ver.status == "BLANK":
                results.append(ClassifierResult("", 0.0, "skipped", "verification_failed"))

            elif ver.status == "UNKNOWN":
                results.append(ClassifierResult("?", 0.0, "unknown", "verification_failed"))

            else:  # VERIFIED
                results.append(self._classify_one(seg.crop))

        n_pred    = sum(1 for r in results if r.status == "predicted")
        n_unknown = sum(1 for r in results if r.status == "unknown")
        n_skipped = sum(1 for r in results if r.status == "skipped")
        log.info(
            "Classifier: %d predicted  %d unknown  %d skipped  (of %d)",
            n_pred, n_unknown, n_skipped, len(segments),
        )
        return results

    # ── single crop classification ─────────────────────────────────────────────

    def _classify_one(self, crop: np.ndarray) -> ClassifierResult:
        if self._session is not None:
            return self._classify_onnx(crop)
        return self._classify_rapidocr_fallback(crop)

    def _classify_onnx(self, crop: np.ndarray) -> ClassifierResult:
        """Run the isolated CNN classifier — no language model."""
        tensor = self._preprocess(crop)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]  # [N_CLASSES]
        probs  = _softmax(logits)
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])

        if conf < self.confidence_threshold:
            return ClassifierResult("?", conf, "unknown", "low_confidence")

        char = PRINTABLE_ASCII[idx] if idx < len(PRINTABLE_ASCII) else "?"
        return ClassifierResult(char, conf, "predicted", "")

    def _classify_rapidocr_fallback(self, crop: np.ndarray) -> ClassifierResult:
        """
        RapidOCR fallback — still a sequence model but single-char crops
        limit context.  Strict confidence gate reduces (but cannot eliminate)
        language-model guessing.
        """
        import cv2

        # upscale tiny crops
        h, w = crop.shape[:2]
        if h < 32 or w < 32:
            scale = max(32.0 / h, 32.0 / w)
            crop = cv2.resize(
                crop,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            result, _ = self._fallback_ocr(
                crop,
                use_det=False,   # skip detection — we already have the crop
                use_cls=False,   # skip angle classifier — crop is deskewed
                use_rec=True,
            )
            if result and result[0]:
                row  = result[0]
                text = str(row[1]).strip() if len(row) > 1 else ""
                conf = float(row[2])       if len(row) > 2 else 0.0

                # take only the first character — ignore sequence context
                char = text[:1] if text else ""

                if not char or conf < self.confidence_threshold:
                    return ClassifierResult("?", conf, "unknown", "low_confidence")

                return ClassifierResult(char, conf, "predicted", "")

        except Exception as exc:
            log.debug("RapidOCR fallback error: %s", exc)

        return ClassifierResult("?", 0.0, "unknown", "inference_error")

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize → float32 → ImageNet normalise → NCHW.
        Matches the training preprocessing of the CNN classifier.
        """
        import cv2

        h, w = crop.shape[:2]
        if h != self.input_size or w != self.input_size:
            crop = cv2.resize(crop, (self.input_size, self.input_size),
                              interpolation=cv2.INTER_CUBIC)

        x = crop.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD            # ImageNet normalise
        x = x.transpose(2, 0, 1)          # HWC → CHW
        return x[np.newaxis].astype(np.float32)   # → [1, 3, H, W]


# ── helpers ───────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()
