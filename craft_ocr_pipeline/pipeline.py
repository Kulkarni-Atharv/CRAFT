"""
OCR Pipeline — Char_Segmentation branch

Simple character-level flow:
  1. CRAFT detects individual character regions
  2. Each character is cropped separately (deskewed, CLAHE normalised)
  3. RapidOCR runs on each crop individually
  4. If confidence >= threshold  → keep character
     If confidence <  threshold  → blank  (no guessing)
  5. Characters are grouped into words by horizontal gap

mode: "char"  — above flow (this branch)
mode: "word"  — legacy word-level RapidOCR (main branch compatible)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.preprocessing import Preprocessor
from modules.detection     import CRAFTDetector
from modules.visualization import save_visualization
from utils.logger          import get_logger

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    source:     str
    frame_idx:  int
    boxes:      list[np.ndarray]
    texts:      list[str]
    confs:      list[float]
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source":     self.source,
            "frame_idx":  self.frame_idx,
            "detections": [
                {"text": t, "confidence": round(c, 4)}
                for t, c in zip(self.texts, self.confs)
            ],
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OCRPipeline:

    def __init__(self, cfg: dict[str, Any]):
        self.cfg      = cfg
        self.mode     = cfg["pipeline"].get("mode", "char")
        self.save_viz = cfg["pipeline"]["save_viz"]
        self.viz_dir  = cfg["paths"]["viz_dir"]

        self.preprocessor = Preprocessor(cfg)
        self.detector     = CRAFTDetector(cfg)

        if self.mode == "char":
            self._init_char_pipeline(cfg)
        else:
            self._init_word_pipeline(cfg)

        log.info("OCRPipeline ready  mode=%s", self.mode)

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_char_pipeline(self, cfg: dict) -> None:
        from modules.char_segmenter import CharSegmenter
        from modules.recognition    import build_recognizer

        self.segmenter            = CharSegmenter(cfg)
        self.recognizer           = build_recognizer(cfg)
        self.conf_threshold: float = cfg["recognition"].get("char_conf_threshold", 0.7)
        self.char_word_gap: float  = cfg["segmentation"].get("char_word_gap", 1.5)

        log.info(
            "Char pipeline ready  conf_threshold=%.2f  char_word_gap=%.1f",
            self.conf_threshold, self.char_word_gap,
        )

    def _init_word_pipeline(self, cfg: dict) -> None:
        from modules.postprocessing import PostProcessor
        from modules.cropping       import Cropper
        from modules.recognition    import build_recognizer, filter_by_char_conf

        self.postprocessor        = PostProcessor(cfg)
        self.cropper              = Cropper(cfg)
        self.recognizer           = build_recognizer(cfg)
        self.conf_threshold: float = cfg["recognition"].get("char_conf_threshold", 0.5)
        self.batch_size: int       = cfg["pipeline"]["batch_size"]
        self._filter              = filter_by_char_conf

        log.info("Word pipeline ready")

    # ── public API ────────────────────────────────────────────────────────────

    def process_image(self, img: np.ndarray, source: str = "image") -> FrameResult:
        return self._run_frame(img, source=source, frame_idx=0)

    def process_image_file(self, path: str) -> FrameResult:
        from utils.image_utils import load_image
        return self.process_image(load_image(path), source=path)

    def process_directory(self, dir_path: str) -> list[FrameResult]:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(dir_path).iterdir() if p.suffix.lower() in exts
        )
        log.info("Processing %d images from %s", len(paths), dir_path)
        return [self.process_image_file(str(p)) for p in paths]

    def process_video(self, video_path: str) -> list[FrameResult]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_skip  = self.cfg["video"]["frame_skip"]
        show_window = self.cfg["video"]["display"]
        results     = []
        frame_idx   = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self._run_frame(img, source=video_path,
                                            frame_idx=frame_idx)
                    results.append(result)

                    if show_window:
                        vis = _draw_result_on_frame(frame, result)
                        cv2.imshow("CRAFT OCR", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                frame_idx += 1
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

        log.info("Video: %d frames analysed", len(results))
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_frame(
        self, img: np.ndarray, source: str, frame_idx: int
    ) -> FrameResult:
        t0 = time.perf_counter()

        tensor, scale, orig_size = self.preprocessor.process(img)
        region_map, affinity_map = self.detector.detect(tensor)
        src_stem = Path(source).stem

        # save region heatmap for threshold diagnostics
        if self.save_viz:
            _save_region_heatmap(region_map, self.viz_dir, src_stem, frame_idx)

        if self.mode == "char":
            return self._run_char(
                img, region_map, affinity_map, scale, orig_size,
                source, src_stem, frame_idx, t0,
            )
        return self._run_word(
            img, region_map, affinity_map, scale, orig_size,
            source, src_stem, frame_idx, t0,
        )

    # ── char mode ─────────────────────────────────────────────────────────────

    def _run_char(
        self,
        img:          np.ndarray,
        region_map:   np.ndarray,
        affinity_map: np.ndarray,
        scale:        float,
        orig_size:    tuple,
        source:       str,
        src_stem:     str,
        frame_idx:    int,
        t0:           float,
    ) -> FrameResult:

        # ── Step 1: CRAFT → individual character crops ────────────────────────
        segs = self.segmenter.segment(
            img, region_map, affinity_map, scale, orig_size,
            prefix=f"{src_stem}_{frame_idx}",
        )

        if not segs:
            latency = (time.perf_counter() - t0) * 1000
            log.info("[%s | frame %d] 0 character regions | %.1f ms",
                     Path(source).name, frame_idx, latency)
            return FrameResult(source=source, frame_idx=frame_idx,
                               boxes=[], texts=[], confs=[], latency_ms=latency)

        # ── Step 2: RapidOCR on each character crop individually ──────────────
        char_texts: list[str]   = []
        char_confs: list[float] = []

        for seg in segs:
            char, conf = self._recognise_char(seg.crop)
            char_texts.append(char)
            char_confs.append(conf)

        # ── Step 3: group characters into words by horizontal gap ─────────────
        word_boxes, word_texts, word_confs = _assemble_words(
            [s.box for s in segs], char_texts, char_confs, self.char_word_gap,
        )

        latency = (time.perf_counter() - t0) * 1000

        if self.save_viz and word_boxes:
            out_path = str(Path(self.viz_dir) / f"{src_stem}_{frame_idx:06d}.jpg")
            save_visualization(img, word_boxes, word_texts, out_path,
                               region_map=region_map)

        full_text = " ".join(t for t in word_texts if t.strip())
        log.info(
            "[%s | frame %d] %d chars → %d words → %r | %.1f ms",
            Path(source).name, frame_idx,
            len(segs), len(word_texts), full_text, latency,
        )
        return FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=word_boxes, texts=word_texts, confs=word_confs,
            latency_ms=latency,
        )

    def _recognise_char(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Run RapidOCR on one character crop.

        Returns (character, confidence) if conf >= threshold,
        ("", 0.0) otherwise — never guesses.
        """
        rec_results = self.recognizer.recognise([crop])
        if not rec_results:
            return "", 0.0

        res  = rec_results[0]
        text = res.word_text.strip()
        conf = res.word_conf

        # take only the first character — ignore any sequence-model padding
        char = text[:1] if text else ""

        if char and conf >= self.conf_threshold:
            return char, conf

        return "", 0.0   # below threshold → blank, no guessing

    # ── word mode (legacy) ────────────────────────────────────────────────────

    def _run_word(
        self,
        img:          np.ndarray,
        region_map:   np.ndarray,
        affinity_map: np.ndarray,
        scale:        float,
        orig_size:    tuple,
        source:       str,
        src_stem:     str,
        frame_idx:    int,
        t0:           float,
    ) -> FrameResult:
        boxes = self.postprocessor.extract_boxes(
            region_map, affinity_map, scale, orig_size
        )
        crops = self.cropper.crop(img, boxes, prefix=f"{src_stem}_{frame_idx}")

        rec_results = []
        for i in range(0, len(crops), self.batch_size):
            rec_results.extend(
                self.recognizer.recognise(crops[i: i + self.batch_size])
            )

        texts, confs = [], []
        for res in rec_results:
            text, conf = self._filter(res, self.conf_threshold)
            texts.append(text)
            confs.append(conf)

        latency = (time.perf_counter() - t0) * 1000

        if self.save_viz and boxes:
            out_path = str(Path(self.viz_dir) / f"{src_stem}_{frame_idx:06d}.jpg")
            save_visualization(img, boxes, texts, out_path, region_map=region_map)

        log.info("[%s | frame %d] %d word regions | %.1f ms",
                 Path(source).name, frame_idx, len(texts), latency)
        return FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=boxes, texts=texts, confs=confs,
            latency_ms=latency,
        )


# ── word assembly ─────────────────────────────────────────────────────────────

def _assemble_words(
    boxes:      list[np.ndarray],
    char_texts: list[str],
    char_confs: list[float],
    gap_factor: float,
) -> tuple[list[np.ndarray], list[str], list[float]]:
    """
    Groups character boxes into words by horizontal gap.
    Gap > gap_factor × average character width → new word.
    Blank characters stay blank in the assembled word — no filling.
    """
    if not boxes:
        return [], [], []

    widths = [float(b[:, 0].max() - b[:, 0].min()) for b in boxes]
    avg_w  = float(np.mean([w for w in widths if w > 0])) if widths else 10.0
    threshold = avg_w * gap_factor

    word_boxes: list[np.ndarray] = []
    word_texts: list[str]        = []
    word_confs: list[float]      = []

    g_boxes, g_chars, g_confs = [boxes[0]], [char_texts[0]], [char_confs[0]]

    for i in range(1, len(boxes)):
        prev_right = boxes[i - 1][:, 0].max()
        curr_left  = boxes[i][:, 0].min()

        if curr_left - prev_right > threshold:
            _flush(g_boxes, g_chars, g_confs, word_boxes, word_texts, word_confs)
            g_boxes, g_chars, g_confs = [], [], []

        g_boxes.append(boxes[i])
        g_chars.append(char_texts[i])
        g_confs.append(char_confs[i])

    _flush(g_boxes, g_chars, g_confs, word_boxes, word_texts, word_confs)
    return word_boxes, word_texts, word_confs


def _flush(g_boxes, g_chars, g_confs, out_boxes, out_texts, out_confs) -> None:
    if not g_boxes:
        return
    all_pts = np.concatenate(g_boxes, axis=0)
    x1, y1  = all_pts[:, 0].min(), all_pts[:, 1].min()
    x2, y2  = all_pts[:, 0].max(), all_pts[:, 1].max()
    merged  = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    text    = "".join(g_chars)                                  # blanks stay as gaps
    conf    = float(np.mean([c for c in g_confs if c > 0] or [0.0]))
    out_boxes.append(merged)
    out_texts.append(text)
    out_confs.append(conf)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_region_heatmap(
    region_map: np.ndarray, viz_dir: str, src_stem: str, frame_idx: int
) -> None:
    heat = cv2.applyColorMap(
        (region_map * 255).clip(0, 255).astype("uint8"),
        cv2.COLORMAP_JET,
    )
    cv2.imwrite(
        str(Path(viz_dir) / f"{src_stem}_{frame_idx:06d}_region.jpg"), heat
    )


def _draw_result_on_frame(
    bgr_frame: np.ndarray, result: FrameResult
) -> np.ndarray:
    from modules.visualization import draw_boxes
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    vis = draw_boxes(rgb, result.boxes, result.texts)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
