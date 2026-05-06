"""
OCR Pipeline Orchestrator — Char_Segmentation branch

Two execution paths selected by config:

  pipeline.mode: "char"  (new)
    CRAFT → CharSegmenter → CharVerifier → CharClassifier → Assembler
    Each character is verified and classified independently.
    No language-model guessing. Occluded chars → blank.

  pipeline.mode: "word"  (legacy, main-branch compatible)
    CRAFT → PostProcessor → Cropper → RapidOCR
    Word-level crops sent to RapidOCR sequence recogniser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.preprocessing  import Preprocessor
from modules.detection      import CRAFTDetector
from modules.visualization  import save_visualization
from utils.logger           import get_logger

log = get_logger(__name__)


# ── Result dataclass (shared by both modes) ───────────────────────────────────

@dataclass
class FrameResult:
    source:     str
    frame_idx:  int
    boxes:      list[np.ndarray]
    texts:      list[str]
    confs:      list[float]
    latency_ms: float = 0.0
    # char-mode extras
    char_statuses: list[str] = field(default_factory=list)  # VERIFIED/BLANK/UNKNOWN per char
    char_reasons:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        detections = []
        for i, (t, c) in enumerate(zip(self.texts, self.confs)):
            entry = {"text": t, "confidence": round(c, 4)}
            if i < len(self.char_statuses):
                entry["status"] = self.char_statuses[i]
                entry["reason"] = self.char_reasons[i]
            detections.append(entry)
        return {
            "source":     self.source,
            "frame_idx":  self.frame_idx,
            "detections": detections,
            "latency_ms": round(self.latency_ms, 1),
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class OCRPipeline:

    def __init__(self, cfg: dict[str, Any]):
        self.cfg      = cfg
        self.mode     = cfg["pipeline"].get("mode", "word")
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
        from modules.char_segmenter  import CharSegmenter
        from modules.char_verifier   import CharVerifier
        from modules.char_classifier import CharClassifier

        self.segmenter  = CharSegmenter(cfg)
        self.verifier   = CharVerifier(cfg)
        self.classifier = CharClassifier(cfg)
        self.char_size  = cfg["segmentation"]["char_size"]
        self.char_word_gap: float = cfg["segmentation"].get("char_word_gap", 1.5)

        self.voting_enabled: bool = cfg.get("voting", {}).get("enabled", False)
        if self.voting_enabled:
            from modules.aggregator import MultiFrameAggregator
            self.aggregator = MultiFrameAggregator(cfg)

        log.info("Char pipeline: segmenter + verifier + classifier loaded")

    def _init_word_pipeline(self, cfg: dict) -> None:
        from modules.postprocessing import PostProcessor
        from modules.cropping       import Cropper
        from modules.recognition    import build_recognizer, filter_by_char_conf

        self.postprocessor          = PostProcessor(cfg)
        self.cropper                = Cropper(cfg)
        self.recognizer             = build_recognizer(cfg)
        self.char_conf_threshold    = cfg["recognition"].get("char_conf_threshold", 0.5)
        self.batch_size             = cfg["pipeline"]["batch_size"]
        self._filter_by_char_conf   = filter_by_char_conf

        log.info("Word pipeline: postprocessor + cropper + RapidOCR loaded")

    # ── public API ────────────────────────────────────────────────────────────

    def process_image(self, img: np.ndarray, source: str = "image") -> FrameResult:
        return self._run_frame(img, source=source, frame_idx=0)

    def process_image_file(self, path: str) -> FrameResult:
        from utils.image_utils import load_image
        return self.process_image(load_image(path), source=path)

    def process_directory(self, dir_path: str) -> list[FrameResult]:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(p for p in Path(dir_path).iterdir()
                       if p.suffix.lower() in exts)
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

        log.info("Video: %d frames analysed", len(results))
        return results

    # ── multi-frame API (char mode only) ──────────────────────────────────────

    def process_frames(
        self,
        frames: list[np.ndarray],
        source: str = "multi_frame",
    ) -> FrameResult:
        """
        Run char-mode pipeline on multiple frames captured from different angles,
        then aggregate with confidence voting for the final text.
        """
        if self.mode != "char":
            log.warning("process_frames() called in word mode — using last frame only")
            return self._run_frame(frames[-1], source=source, frame_idx=0)

        if not self.voting_enabled:
            log.warning("voting.enabled=false — using last frame only")
            return self._run_frame(frames[-1], source=source, frame_idx=0)

        self.aggregator.reset()

        for frame_id, img in enumerate(frames):
            segs, _, clf_results = self._run_char_detection(
                img, source=source, frame_idx=frame_id
            )
            self.aggregator.add_frame(frame_id, segs, clf_results)

        final_text, agg_chars = self.aggregator.get_final_text()

        boxes  = []   # no individual boxes when aggregating
        texts  = list(final_text)
        confs  = [a.confidence for a in agg_chars]

        return FrameResult(
            source=source, frame_idx=0,
            boxes=boxes, texts=texts, confs=confs,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _run_frame(
        self, img: np.ndarray, source: str, frame_idx: int
    ) -> FrameResult:
        t0 = time.perf_counter()

        tensor, scale, orig_size = self.preprocessor.process(img)
        region_map, affinity_map = self.detector.detect(tensor)

        src_stem = Path(source).stem

        # save region-map heatmap for diagnostic / threshold tuning
        if self.save_viz:
            _save_region_heatmap(region_map, self.viz_dir, src_stem, frame_idx)

        if self.mode == "char":
            result = self._run_char_frame(
                img, region_map, affinity_map, scale, orig_size,
                source, src_stem, frame_idx, t0,
            )
        else:
            result = self._run_word_frame(
                img, region_map, affinity_map, scale, orig_size,
                source, src_stem, frame_idx, t0,
            )

        return result

    # ── char-level frame ──────────────────────────────────────────────────────

    def _run_char_detection(
        self,
        img:        np.ndarray,
        source:     str,
        frame_idx:  int,
    ):
        """Run detection + segmentation + verify + classify.  Returns raw lists."""
        tensor, scale, orig_size = self.preprocessor.process(img)
        region_map, affinity_map = self.detector.detect(tensor)
        src_stem = Path(source).stem

        segs = self.segmenter.segment(
            img, region_map, affinity_map, scale, orig_size,
            prefix=f"{src_stem}_{frame_idx}",
        )
        ver_results = self.verifier.verify_batch(segs)
        clf_results = self.classifier.classify_batch(segs, ver_results)
        return segs, ver_results, clf_results

    def _run_char_frame(
        self,
        img:         np.ndarray,
        region_map:  np.ndarray,
        affinity_map: np.ndarray,
        scale:       float,
        orig_size:   tuple,
        source:      str,
        src_stem:    str,
        frame_idx:   int,
        t0:          float,
    ) -> FrameResult:
        segs = self.segmenter.segment(
            img, region_map, affinity_map, scale, orig_size,
            prefix=f"{src_stem}_{frame_idx}",
        )

        if not segs:
            latency = (time.perf_counter() - t0) * 1000
            log.info("[%s | frame %d] 0 character segments detected | %.1f ms",
                     Path(source).name, frame_idx, latency)
            return FrameResult(source=source, frame_idx=frame_idx,
                               boxes=[], texts=[], confs=[], latency_ms=latency)

        ver_results = self.verifier.verify_batch(segs)
        clf_results = self.classifier.classify_batch(segs, ver_results)

        # assemble chars → words preserving blanks
        word_boxes, word_texts, word_confs, statuses, reasons = _assemble_words_char(
            segs, clf_results, ver_results, self.char_word_gap
        )

        latency = (time.perf_counter() - t0) * 1000

        if self.save_viz and word_boxes:
            out_path = str(Path(self.viz_dir) / f"{src_stem}_{frame_idx:06d}.jpg")
            save_visualization(img, word_boxes, word_texts, out_path,
                               region_map=region_map)

        full_text = " ".join(t for t in word_texts if t.strip())
        log.info(
            "[%s | frame %d] %d segs → %d words → %r | %.1f ms",
            Path(source).name, frame_idx, len(segs), len(word_texts),
            full_text, latency,
        )
        return FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=word_boxes, texts=word_texts, confs=word_confs,
            latency_ms=latency,
            char_statuses=statuses, char_reasons=reasons,
        )

    # ── word-level frame (legacy) ─────────────────────────────────────────────

    def _run_word_frame(
        self,
        img:         np.ndarray,
        region_map:  np.ndarray,
        affinity_map: np.ndarray,
        scale:       float,
        orig_size:   tuple,
        source:      str,
        src_stem:    str,
        frame_idx:   int,
        t0:          float,
    ) -> FrameResult:
        boxes = self.postprocessor.extract_boxes(
            region_map, affinity_map, scale, orig_size
        )
        crops = self.cropper.crop(img, boxes, prefix=f"{src_stem}_{frame_idx}")

        rec_results = []
        for i in range(0, len(crops), self.batch_size):
            rec_results.extend(self.recognizer.recognise(crops[i: i + self.batch_size]))

        texts, confs = [], []
        for res in rec_results:
            text, conf = self._filter_by_char_conf(res, self.char_conf_threshold)
            texts.append(text)
            confs.append(conf)

        latency = (time.perf_counter() - t0) * 1000

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


# ── character assembly ────────────────────────────────────────────────────────

def _assemble_words_char(
    segs:        list,
    clf_results: list,
    ver_results: list,
    gap_factor:  float,
) -> tuple[list, list, list, list, list]:
    """
    Group character segments into words by horizontal gap.
    Blank characters ("") are included in position but not in the word string.
    Returns (word_boxes, word_texts, word_confs, statuses, reasons).
    """
    if not segs:
        return [], [], [], [], []

    # average character width across all boxes (used as gap reference)
    widths = []
    for seg in segs:
        w = float(seg.box[:, 0].max() - seg.box[:, 0].min())
        if w > 0:
            widths.append(w)
    avg_w     = float(np.mean(widths)) if widths else 10.0
    threshold = avg_w * gap_factor

    word_boxes: list[np.ndarray] = []
    word_texts: list[str]        = []
    word_confs: list[float]      = []

    g_segs, g_chars, g_confs = [segs[0]], [clf_results[0].character], [clf_results[0].confidence]

    for i in range(1, len(segs)):
        prev_right = segs[i - 1].box[:, 0].max()
        curr_left  = segs[i].box[:, 0].min()
        gap        = curr_left - prev_right

        if gap > threshold:
            _flush_char_group(g_segs, g_chars, g_confs,
                              word_boxes, word_texts, word_confs)
            g_segs, g_chars, g_confs = [], [], []

        g_segs.append(segs[i])
        g_chars.append(clf_results[i].character)
        g_confs.append(clf_results[i].confidence)

    _flush_char_group(g_segs, g_chars, g_confs,
                      word_boxes, word_texts, word_confs)

    # statuses and reasons are per-character (not per-word)
    all_statuses = [v.status for v in ver_results]
    all_reasons  = [v.reason for v in ver_results]
    return word_boxes, word_texts, word_confs, all_statuses, all_reasons


def _flush_char_group(
    g_segs:    list,
    g_chars:   list[str],
    g_confs:   list[float],
    out_boxes: list,
    out_texts: list,
    out_confs: list,
) -> None:
    if not g_segs:
        return
    all_pts = np.concatenate([s.box for s in g_segs], axis=0)
    x1, y1  = all_pts[:, 0].min(), all_pts[:, 1].min()
    x2, y2  = all_pts[:, 0].max(), all_pts[:, 1].max()
    merged  = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)

    text = "".join(g_chars)   # blanks ("") stay as gaps, no guessing
    conf = float(np.mean([c for c in g_confs if c > 0] or [0.0]))
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
        str(Path(viz_dir) / f"{src_stem}_{frame_idx:06d}_region.jpg"),
        heat,
    )


def _draw_result_on_frame(
    bgr_frame: np.ndarray, result: FrameResult
) -> np.ndarray:
    from modules.visualization import draw_boxes
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    vis = draw_boxes(rgb, result.boxes, result.texts)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
