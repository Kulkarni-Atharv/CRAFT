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


# ── Tesseract back-end ───────────────────────────────────────────────────────

class TesseractRecognizer:
    """
    Tesseract OCR via pytesseract — no deep-learning framework required.
    Works on any platform including ARM64 / Python 3.13.

    Install once on CM5:
        sudo apt install -y tesseract-ocr
        pip install pytesseract
    """

    def __init__(self, cfg: dict[str, Any]):
        try:
            import pytesseract  # type: ignore
            pytesseract.get_tesseract_version()   # raises if binary not found
        except ImportError as e:
            raise ImportError(
                "Install pytesseract: pip install pytesseract\n"
                "Also install the tesseract binary: sudo apt install -y tesseract-ocr"
            ) from e
        except Exception as e:
            raise RuntimeError(
                "tesseract binary not found.\n"
                "Install it: sudo apt install -y tesseract-ocr"
            ) from e

        self._tess      = pytesseract
        self._char_mode = False   # flipped to True by pipeline in char-level mode
        log.info("Tesseract recogniser initialised")

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        results: list[RecognitionResult] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(RecognitionResult("", 0.0))
                continue
            results.append(self._decode_crop(crop))
        return results

    def _decode_crop(self, crop: np.ndarray) -> RecognitionResult:
        import cv2
        from PIL import Image

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

        # upscale to at least 64 px tall — tesseract accuracy drops below 32 px
        h, w = gray.shape[:2]
        if h < 64:
            scale = 64.0 / h
            gray  = cv2.resize(gray, (max(1, int(w * scale)), 64),
                               interpolation=cv2.INTER_CUBIC)

        # add white border — tesseract needs context around glyphs
        gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, value=255)

        # try multiple strategies and return the best (longest) result
        best_text, best_conf = "", 0.0
        psm_list = (10, 8) if self._char_mode else (7, 8, 6)
        for variant in self._make_variants(gray):
            for psm in psm_list:
                cfg = f"--psm {psm} --oem 3"
                try:
                    data = self._tess.image_to_data(
                        Image.fromarray(variant), config=cfg,
                        output_type=self._tess.Output.DICT,
                    )
                    words = [
                        (t.strip(), int(c))
                        for t, c in zip(data["text"], data["conf"])
                        if t.strip() and int(c) > 0
                    ]
                    if words:
                        text = " ".join(t for t, _ in words).lower()
                        conf = sum(c for _, c in words) / len(words) / 100.0
                        if len(text) > len(best_text):
                            best_text, best_conf = text, conf
                except Exception:
                    continue
            if best_text:
                break

        return RecognitionResult(best_text, best_conf)

    @staticmethod
    def _make_variants(gray: np.ndarray) -> list:
        """Return image variants to try: Otsu-binarized and inverted."""
        import cv2
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(otsu)
        return [otsu, inverted]


# ── PaddleOCR back-end ────────────────────────────────────────────────────────

class PaddleOCRRecognizer:
    """
    Recognition-only PaddleOCR (CRAFT handles detection).
    Compatible with both PaddleOCR v2 and v3 APIs.
    """

    def __init__(self, cfg: dict[str, Any]):
        rcfg = cfg["recognition"]
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError("Install paddleocr: pip install paddleocr") from e

        # v3 removed show_log/use_gpu/det/cls from the constructor.
        # Try progressively simpler param sets until one works.
        for kwargs in [
            # v2-style
            dict(use_angle_cls=True, lang=rcfg["lang"],
                 use_gpu=rcfg.get("use_gpu", False), det=False, cls=True, show_log=False),
            # v3-style (no show_log, use_gpu → device)
            dict(use_angle_cls=True, lang=rcfg["lang"],
                 device="gpu" if rcfg.get("use_gpu", False) else "cpu"),
            # minimal fallback — works in any version
            dict(lang=rcfg["lang"]),
        ]:
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError("Could not initialise PaddleOCR with any known API variant")

        log.info("PaddleOCR recogniser initialised (lang=%s)", rcfg["lang"])

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
        # Try det=False (recognition-only); fall back to full pipeline
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
        # v2: [ [ ('text', conf), ... ] ]  — list of pages → list of lines
        # v3: [ [ ('text', conf), ... ] ]  or list of dicts
        first = out[0]
        if not first:
            return "", 0.0

        item = first[0] if isinstance(first, (list, tuple)) else first

        # v2 / v3 tuple format: ('text', conf)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            t, c = item
            if isinstance(t, str):
                return t, float(c)
            # v2 with bounding box: item = [box, ('text', conf)]
            if isinstance(c, (list, tuple)) and len(c) == 2:
                return str(c[0]), float(c[1])

        # v3 object format
        if hasattr(item, "text"):
            return str(item.text), float(getattr(item, "score", 0.0))

        return "", 0.0


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

        char_preds = _ctc_decode_numpy(indices, probs_np, self.chars)
        word_text  = "".join(c for c, _ in char_preds)
        word_conf  = (sum(p for _, p in char_preds) / len(char_preds)
                      if char_preds else 0.0)
        return RecognitionResult(word_text, word_conf, char_preds)


# ── CRNN ONNX back-end (lightweight — no torch on CM5) ───────────────────────

class CRNNONNXRecognizer:
    """
    CRNN recognition via ONNX Runtime.
    Identical accuracy to CRNNRecognizer — same model, different runtime.
    No torch required: ~750 MB saved on CM5.

    Export the model first on the dev machine:
        python scripts/export_onnx.py --crnn-only
    """

    DEFAULT_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, cfg: dict[str, Any]):
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise ImportError("Install onnxruntime: pip install onnxruntime") from e

        ccfg           = cfg["recognition"]["crnn"]
        self.img_h     = ccfg["img_height"]
        self.img_w     = ccfg["img_width"]
        charset        = ccfg.get("charset", self.DEFAULT_CHARSET)
        self.chars     = ["-"] + list(charset)   # index 0 = CTC blank

        onnx_path = cfg["paths"]["crnn_onnx"]
        if not __import__("pathlib").Path(onnx_path).exists():
            raise FileNotFoundError(
                f"CRNN ONNX model not found: {onnx_path}\n"
                "Export it from your dev machine first:\n"
                "  python scripts/export_onnx.py --crnn-only\n"
                "Then copy models/crnn.onnx to the CM5."
            )
        self._session    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        log.info("CRNN ONNX recogniser loaded  model=%s  classes=%d", onnx_path, len(self.chars))

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        import cv2
        results: list[RecognitionResult] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                results.append(RecognitionResult("", 0.0))
                continue
            results.append(self._decode_crop(crop))
        return results

    def _decode_crop(self, crop: np.ndarray) -> RecognitionResult:
        import cv2
        gray    = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray    = cv2.resize(gray, (self.img_w, self.img_h))
        x       = gray.astype(np.float32)
        x       = (x / 255.0 - 0.5) / 0.5
        x       = x[np.newaxis, np.newaxis]            # 1×1×H×W

        logits  = self._session.run(None, {self._input_name: x})[0]  # T×1×n_classes
        # softmax along class axis
        e       = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs   = e / e.sum(axis=-1, keepdims=True)    # T×1×n_classes
        probs2d = probs[:, 0, :]                       # T×n_classes
        indices = probs2d.argmax(axis=-1).tolist()     # T ints

        # reuse the same CTC decoder logic
        char_preds = _ctc_decode_numpy(indices, probs2d, self.chars)
        word_text  = "".join(c for c, _ in char_preds)
        word_conf  = (sum(p for _, p in char_preds) / len(char_preds)
                      if char_preds else 0.0)
        return RecognitionResult(word_text, word_conf, char_preds)


def _ctc_decode_numpy(
    indices:  list[int],
    probs_np: np.ndarray,   # T×n_classes
    chars:    list[str],
) -> list[tuple[str, float]]:
    """Pure-numpy CTC greedy decode — shared by CRNNRecognizer and CRNNONNXRecognizer."""
    char_preds: list[tuple[str, float]] = []
    prev_idx   = -1
    span_probs: list[float] = []

    for t, idx in enumerate(indices):
        if idx == prev_idx:
            if idx != 0:
                span_probs.append(float(probs_np[t, idx]))
        else:
            if prev_idx > 0 and span_probs:
                char_preds.append((chars[prev_idx], max(span_probs)))
            span_probs = [float(probs_np[t, idx])] if idx != 0 else []
            prev_idx   = idx

    if prev_idx > 0 and span_probs:
        char_preds.append((chars[prev_idx], max(span_probs)))

    return char_preds


# ── Null recognizer (detect-only mode) ───────────────────────────────────────

class NullRecognizer:
    """Returns empty results — used when detect_only=true or no engine available."""

    def recognise(self, crops: list[np.ndarray]) -> list[RecognitionResult]:
        return [RecognitionResult("", 0.0) for _ in crops]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_recognizer(cfg: dict[str, Any]):
    if cfg["recognition"].get("detect_only", False):
        log.info("Recognition disabled (detect_only=true) — returning bounding boxes only")
        return NullRecognizer()

    engine = cfg["recognition"]["engine"].lower()

    if engine == "tesseract":
        return TesseractRecognizer(cfg)

    if engine == "paddleocr":
        try:
            import paddle  # noqa: F401
        except ImportError:
            raise ImportError(
                "PaddleOCR requires paddlepaddle. Install it on the CM5:\n"
                "  pip install paddlepaddle paddleocr"
            )
        return PaddleOCRRecognizer(cfg)

    if engine == "crnn":
        return CRNNRecognizer(cfg)

    if engine == "crnn_onnx":
        onnx_path = cfg["paths"].get("crnn_onnx", "")
        if not __import__("pathlib").Path(onnx_path).exists():
            log.warning(
                "crnn.onnx not found at %s — running in detect-only mode. "
                "Export it: python scripts/export_onnx.py --crnn-only",
                onnx_path,
            )
            return NullRecognizer()
        return CRNNONNXRecognizer(cfg)

    raise ValueError(
        f"Unknown recognition engine: {engine!r}. "
        "Use 'paddleocr', 'crnn', or 'crnn_onnx'."
    )
