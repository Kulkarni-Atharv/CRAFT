"""
Module 5 — Recognition
Wraps PaddleOCR (default) or a custom CRNN for text recognition.

Key change: recognise() now returns list[RecognitionResult] instead of
list[tuple[str, float]].  Each result carries per-character (char, prob)
pairs so that low-confidence characters can be dropped before final output.

Character probability computation (CRNN)
-----------------------------------------
CRNN outputs logits of shape T × 1 × n_classes (T = time steps along the
width dimension after CNN).  Softmax over n_classes gives a probability
distribution at each time step.  CTC greedy decoding collapses repeated
tokens and removes blanks; a character may span several time steps.
We record the *peak* softmax probability across the span for that character —
this is the most informative single number because the model is most
"committed" at the peak, and low-blur characters typically have clean,
sharp peaks whereas blurred/noisy characters produce flat, low-confidence
distributions.

PaddleOCR limitation
---------------------
PaddleOCR's public API collapses per-character probs into a single word
confidence score and does not expose the raw character-level tensor.
We return an empty char_preds list and fall back to word-level filtering
(the whole word is kept or dropped based on its word confidence).
For true character-level filtering, switch engine to "crnn".
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
    word_text:  str                              # full unfiltered text
    word_conf:  float                            # model's word-level confidence
    char_preds: list[tuple[str, float]] = field(default_factory=list)
    # (char, peak_prob) pairs in order — empty when PaddleOCR is used


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_by_char_conf(
    result: RecognitionResult,
    threshold: float,
) -> tuple[str, float]:
    """
    Drop characters whose confidence is below `threshold`.

    Returns (filtered_text, mean_conf_of_kept_chars).

    CRNN path  — uses real per-character probabilities.
    Paddle path — no char_preds available; applies threshold to the whole
                  word (keep everything or drop everything).
    """
    if result.char_preds:
        kept = [(c, p) for c, p in result.char_preds if p >= threshold]
        if not kept:
            return ("", 0.0)
        text = "".join(c for c, _ in kept)
        conf = sum(p for _, p in kept) / len(kept)
        return (text, conf)

    # PaddleOCR word-level fallback
    if result.word_conf >= threshold:
        return (result.word_text, result.word_conf)
    return ("", result.word_conf)


# ── Optional: CRAFT region_map score fusion ───────────────────────────────────

def fuse_craft_scores(
    char_preds: list[tuple[str, float]],
    char_boxes: list[np.ndarray],   # one 4×2 box per character
    region_map: np.ndarray,         # H×W float [0,1] at score-map resolution
    scale:      float,              # preprocessing resize scale
    alpha:      float = 0.5,        # weight for OCR confidence (1-alpha for CRAFT)
) -> list[tuple[str, float]]:
    """
    Linearly blend OCR character confidence with the CRAFT region score
    sampled at each character's centre.

    A character that the OCR is unsure about (low prob) AND that sits in a
    low-activation region of the CRAFT map (debris / noise) gets a doubly
    penalised score, making the threshold cut more decisive.

    Safe to skip when char_boxes is not aligned with char_preds (e.g. when
    character splitting was not run); in that case call the function only
    when len(char_boxes) == len(char_preds).
    """
    if len(char_boxes) != len(char_preds):
        return char_preds   # alignment mismatch — return unchanged

    fused: list[tuple[str, float]] = []
    for (char, ocr_conf), box in zip(char_preds, char_boxes):
        cx = int(np.clip(box[:, 0].mean() * scale / 2.0, 0, region_map.shape[1] - 1))
        cy = int(np.clip(box[:, 1].mean() * scale / 2.0, 0, region_map.shape[0] - 1))
        craft_score = float(region_map[cy, cx])
        fused_conf  = alpha * ocr_conf + (1.0 - alpha) * craft_score
        fused.append((char, fused_conf))
    return fused


# ── PaddleOCR back-end ────────────────────────────────────────────────────────

class PaddleOCRRecognizer:
    """
    Recognition-only PaddleOCR (det=False — CRAFT handles detection).
    Returns word-level confidence; char_preds is always empty.
    For character-level filtering switch to engine='crnn'.
    """

    def __init__(self, cfg: dict[str, Any]):
        rcfg = cfg["recognition"]
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError("Install paddleocr: pip install paddleocr") from e

        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=rcfg["lang"],
            use_gpu=rcfg["use_gpu"],
            det=False,
            cls=True,
            show_log=False,
        )
        log.info("PaddleOCR recogniser initialised (lang=%s, gpu=%s)",
                 rcfg["lang"], rcfg["use_gpu"])
        log.warning(
            "PaddleOCR does not expose per-character probabilities. "
            "Filtering will use word-level confidence. "
            "Switch to engine='crnn' for true character-level filtering."
        )

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(RecognitionResult("", 0.0))
                continue
            out = self._ocr.ocr(crop, det=False, cls=True)
            if out and out[0]:
                text, conf = out[0][0]
                results.append(RecognitionResult(str(text), float(conf)))
            else:
                results.append(RecognitionResult("", 0.0))
        return results


# ── CRNN back-end ─────────────────────────────────────────────────────────────

class CRNNRecognizer:
    """
    CRNN with full per-character confidence via CTC greedy decoding.

    How per-character confidence is computed
    -----------------------------------------
    1. Run forward pass → logits  (T × 1 × n_classes)
    2. Softmax along class axis   → probs  (T × n_classes)
    3. Greedy argmax per time step → sequence of class indices
    4. Group consecutive identical non-blank indices into "character spans"
    5. Confidence of each character = max(probs) over its span
       (peak rather than mean: the model is most decisive at the peak;
        blurred characters produce flat distributions with a low peak)
    """

    DEFAULT_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, cfg: dict[str, Any]):
        import torch
        from models.crnn_model import CRNN  # type: ignore

        ccfg = cfg["recognition"]["crnn"]
        self.img_h  = ccfg["img_height"]
        self.img_w  = ccfg["img_width"]
        charset     = ccfg.get("charset", self.DEFAULT_CHARSET)
        self.chars  = ["-"] + list(charset)   # index 0 = CTC blank

        self.device = torch.device(
            "cuda" if cfg["recognition"]["use_gpu"] and torch.cuda.is_available()
            else "cpu"
        )
        from models.crnn_model import CRNN
        self.model = CRNN(img_height=self.img_h, n_classes=len(self.chars)).to(self.device)
        state = torch.load(ccfg["model_path"], map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        log.info("CRNN recogniser loaded (%d classes, device=%s)",
                 len(self.chars), self.device)

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        import torch
        import cv2

        results: list[RecognitionResult] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(RecognitionResult("", 0.0))
                continue
            results.append(self._decode_crop(crop))
        return results

    def _decode_crop(self, crop: np.ndarray) -> RecognitionResult:
        import torch
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.img_w, self.img_h))
        x = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        x = (x / 255.0 - 0.5) / 0.5
        x = x.to(self.device)

        with torch.inference_mode():
            logits = self.model(x)                       # T×1×n_classes

        probs   = torch.softmax(logits[:, 0, :], dim=-1) # T×n_classes
        indices = probs.argmax(dim=-1).cpu().tolist()     # T ints
        probs_np = probs.cpu().numpy()                    # T×n_classes

        char_preds = self._ctc_decode(indices, probs_np)
        word_text  = "".join(c for c, _ in char_preds)
        word_conf  = (sum(p for _, p in char_preds) / len(char_preds)
                      if char_preds else 0.0)
        return RecognitionResult(word_text, word_conf, char_preds)

    def _ctc_decode(
        self,
        indices:  list[int],
        probs_np: np.ndarray,   # T×n_classes
    ) -> list[tuple[str, float]]:
        """
        CTC greedy decode → list of (char, peak_confidence).

        Walk the time axis:
        - While the same non-blank index repeats, accumulate its probs.
        - When the index changes (or we reach the end), emit the character
          with its peak probability and start a new span.
        - Blank tokens (index 0) act as separators and are discarded.
        """
        char_preds: list[tuple[str, float]] = []
        prev_idx   = -1
        span_probs: list[float] = []

        for t, idx in enumerate(indices):
            if idx == prev_idx:
                if idx != 0:
                    span_probs.append(float(probs_np[t, idx]))
            else:
                # flush previous span
                if prev_idx > 0 and span_probs:
                    char_preds.append((self.chars[prev_idx], max(span_probs)))
                span_probs = [float(probs_np[t, idx])] if idx != 0 else []
                prev_idx = idx

        # flush last span
        if prev_idx > 0 and span_probs:
            char_preds.append((self.chars[prev_idx], max(span_probs)))

        return char_preds


# ── Factory ───────────────────────────────────────────────────────────────────

def build_recognizer(cfg: dict[str, Any]):
    engine = cfg["recognition"]["engine"].lower()
    if engine == "paddleocr":
        return PaddleOCRRecognizer(cfg)
    if engine == "crnn":
        return CRNNRecognizer(cfg)
    raise ValueError(f"Unknown recognition engine: {engine!r}. Use 'paddleocr' or 'crnn'.")
