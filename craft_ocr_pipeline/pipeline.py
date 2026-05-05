"""
OCR Pipeline Orchestrator
Wires Preprocessor → CRAFTDetector → PostProcessor → Cropper → Recognizer.
Supports single images, image directories, and video files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.preprocessing   import Preprocessor
from modules.detection        import CRAFTDetector
from modules.postprocessing   import PostProcessor
from modules.cropping         import Cropper
from modules.recognition      import build_recognizer, filter_by_char_conf
from modules.visualization    import save_visualization
from utils.logger             import get_logger

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    source:    str
    frame_idx: int
    boxes:     list[np.ndarray]
    texts:     list[str]
    confs:     list[float]
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


# ── Character assembly ────────────────────────────────────────────────────────

def _assemble_words(
    char_boxes:  list[np.ndarray],
    char_texts:  list[str],
    char_confs:  list[float],
    gap_factor:  float,
) -> tuple[list[np.ndarray], list[str], list[float]]:
    """
    Groups character boxes into words by horizontal gap.

    Characters are already sorted top-to-bottom, left-to-right by
    extract_char_boxes().  A new word starts when the gap between adjacent
    characters exceeds gap_factor × average character width in that line.
    """
    if not char_boxes:
        return [], [], []

    # average character width across all boxes (used as gap reference)
    avg_w = float(np.mean([b[:, 0].max() - b[:, 0].min() for b in char_boxes]))
    threshold = avg_w * gap_factor

    word_boxes:  list[np.ndarray] = []
    word_texts:  list[str]        = []
    word_confs:  list[float]      = []

    group_boxes:  list[np.ndarray] = [char_boxes[0]]
    group_chars:  list[str]        = [char_texts[0]]
    group_confs:  list[float]      = [char_confs[0]]

    for i in range(1, len(char_boxes)):
        prev_right = char_boxes[i - 1][:, 0].max()
        curr_left  = char_boxes[i][:, 0].min()
        gap        = curr_left - prev_right

        if gap > threshold:
            # flush current group as one word
            _flush_group(group_boxes, group_chars, group_confs,
                         word_boxes, word_texts, word_confs)
            group_boxes, group_chars, group_confs = [], [], []

        group_boxes.append(char_boxes[i])
        group_chars.append(char_texts[i])
        group_confs.append(char_confs[i])

    _flush_group(group_boxes, group_chars, group_confs,
                 word_boxes, word_texts, word_confs)

    return word_boxes, word_texts, word_confs


def _flush_group(
    g_boxes: list[np.ndarray], g_chars: list[str], g_confs: list[float],
    out_boxes: list, out_texts: list, out_confs: list,
) -> None:
    if not g_boxes:
        return
    # merge boxes into one encompassing box
    all_pts  = np.concatenate(g_boxes, axis=0)
    x1, y1   = all_pts[:, 0].min(), all_pts[:, 1].min()
    x2, y2   = all_pts[:, 0].max(), all_pts[:, 1].max()
    merged   = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
    text     = "".join(g_chars)
    conf     = float(np.mean([c for c in g_confs if c > 0] or [0.0]))
    out_boxes.append(merged)
    out_texts.append(text)
    out_confs.append(conf)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OCRPipeline:

    def __init__(self, cfg: dict[str, Any]):
        self.cfg        = cfg
        self.save_viz   = cfg["pipeline"]["save_viz"]
        self.viz_dir    = cfg["paths"]["viz_dir"]
        self.batch_size = cfg["pipeline"]["batch_size"]

        self.preprocessor  = Preprocessor(cfg)
        self.detector      = CRAFTDetector(cfg)
        self.postprocessor = PostProcessor(cfg)
        self.cropper       = Cropper(cfg)
        self.recognizer    = build_recognizer(cfg)

        self.char_conf_threshold: float = cfg["recognition"].get("char_conf_threshold", 0.7)
        self.char_level:          bool  = cfg["postprocessing"].get("char_level", False)
        self.char_word_gap:       float = cfg["postprocessing"].get("char_word_gap", 1.2)

        log.info(
            "OCR pipeline initialised  char_level=%s  char_conf_threshold=%.2f",
            self.char_level, self.char_conf_threshold,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def process_image(self, img: np.ndarray, source: str = "image") -> FrameResult:
        return self._run_frame(img, source=source, frame_idx=0)

    def process_image_file(self, path: str) -> FrameResult:
        from utils.image_utils import load_image
        img = load_image(path)
        return self.process_image(img, source=path)

    def process_directory(self, dir_path: str) -> list[FrameResult]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            p for p in Path(dir_path).iterdir()
            if p.suffix.lower() in exts
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
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self._run_frame(img, source=video_path, frame_idx=frame_idx)
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

        log.info("Video processed: %d frames analysed", len(results))
        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_frame(
        self, img: np.ndarray, source: str, frame_idx: int
    ) -> FrameResult:
        t0 = time.perf_counter()

        # 1. preprocess
        tensor, scale, orig_size = self.preprocessor.process(img)

        # 2. detect
        region_map, affinity_map = self.detector.detect(tensor)

        src_stem = Path(source).stem

        if self.char_level:
            return self._run_char_level(
                img, region_map, scale, orig_size,
                source, src_stem, frame_idx, t0,
            )

        # ── word-level path ───────────────────────────────────────────────────

        # 3. post-process → word boxes
        boxes = self.postprocessor.extract_boxes(
            region_map, affinity_map, scale, orig_size
        )

        # 4. crop
        crops = self.cropper.crop(img, boxes, prefix=f"{src_stem}_{frame_idx}")

        # 5. recognise (batched)
        rec_results = self._batched_recognise(crops)

        # 5a. filter low-confidence characters
        texts, confs = [], []
        for res in rec_results:
            text, conf = filter_by_char_conf(res, self.char_conf_threshold)
            texts.append(text)
            confs.append(conf)

        latency = (time.perf_counter() - t0) * 1000

        # 6. visualise
        if self.save_viz and boxes:
            out_path = str(Path(self.viz_dir) / f"{src_stem}_{frame_idx:06d}.jpg")
            save_visualization(img, boxes, texts, out_path, region_map=region_map)

        result = FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=boxes, texts=texts, confs=confs,
            latency_ms=latency,
        )
        log.info("[%s | frame %d] %d word regions | %.1f ms",
                 Path(source).name, frame_idx, len(texts), latency)
        return result

    def _run_char_level(
        self,
        img:        np.ndarray,
        region_map: np.ndarray,
        scale:      float,
        orig_size:  tuple[int, int],
        source:     str,
        src_stem:   str,
        frame_idx:  int,
        t0:         float,
    ) -> FrameResult:
        """
        Character-level detection path:
          region_map → individual char boxes → crop → recognise per char
          → sort & group by position → assemble words → FrameResult
        """
        # 3. individual character boxes (region map only, no affinity grouping)
        char_boxes = self.postprocessor.extract_char_boxes(
            region_map, scale, orig_size
        )

        if not char_boxes:
            latency = (time.perf_counter() - t0) * 1000
            return FrameResult(source=source, frame_idx=frame_idx,
                               boxes=[], texts=[], confs=[], latency_ms=latency)

        # 4. crop each character
        crops = self.cropper.crop(img, char_boxes,
                                  prefix=f"{src_stem}_{frame_idx}_char")

        # 5. recognise each character crop
        rec_results = self._batched_recognise(crops)

        char_texts = []
        char_confs = []
        for res in rec_results:
            text, conf = filter_by_char_conf(res, self.char_conf_threshold)
            # keep only the first non-space character from each crop
            ch = text.strip()[:1] if text.strip() else ""
            char_texts.append(ch)
            char_confs.append(conf)

        # 6. assemble characters → words by grouping on horizontal gap
        word_boxes, word_texts, word_confs = _assemble_words(
            char_boxes, char_texts, char_confs, self.char_word_gap
        )

        latency = (time.perf_counter() - t0) * 1000

        # 7. visualise
        if self.save_viz and word_boxes:
            out_path = str(Path(self.viz_dir) / f"{src_stem}_{frame_idx:06d}.jpg")
            save_visualization(img, word_boxes, word_texts, out_path,
                               region_map=region_map)

        full_text = " ".join(t for t in word_texts if t)
        log.info("[%s | frame %d] %d chars → %d words → %r | %.1f ms",
                 Path(source).name, frame_idx,
                 len(char_boxes), len(word_texts), full_text, latency)

        return FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=word_boxes, texts=word_texts, confs=word_confs,
            latency_ms=latency,
        )

    def _batched_recognise(self, crops: list[np.ndarray]):
        # returns list[RecognitionResult] — one per crop
        results = []
        for i in range(0, len(crops), self.batch_size):
            results.extend(self.recognizer.recognise(crops[i : i + self.batch_size]))
        return results


# ── video helper ──────────────────────────────────────────────────────────────

def _draw_result_on_frame(
    bgr_frame: np.ndarray, result: FrameResult
) -> np.ndarray:
    from modules.visualization import draw_boxes
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    vis = draw_boxes(rgb, result.boxes, result.texts)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
