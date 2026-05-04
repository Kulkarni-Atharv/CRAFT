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
from modules.recognition      import build_recognizer, filter_by_char_conf, fuse_craft_scores
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
        self.use_craft_fusion:    bool  = cfg["recognition"].get("use_craft_fusion", False)
        log.info(
            "OCR pipeline initialised  char_conf_threshold=%.2f  craft_fusion=%s",
            self.char_conf_threshold, self.use_craft_fusion,
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

        # 3. post-process → boxes
        boxes = self.postprocessor.extract_boxes(
            region_map, affinity_map, scale, orig_size
        )

        # 4. crop
        src_stem = Path(source).stem
        crops = self.cropper.crop(img, boxes, prefix=f"{src_stem}_{frame_idx}")

        # 5. recognise (batched) → RecognitionResult objects
        rec_results = self._batched_recognise(crops)

        # 5a. optional: blend CRAFT region score into per-char confidences
        #     requires character-level boxes aligned with char_preds — only
        #     meaningful when CharacterSplitter has been run upstream.
        if self.use_craft_fusion:
            for res in rec_results:
                if res.char_preds:
                    res.char_preds = fuse_craft_scores(
                        res.char_preds,
                        char_boxes=[],          # wire in char_boxes here if available
                        region_map=region_map,
                        scale=scale,
                    )

        # 5b. filter low-confidence characters
        texts, confs = [], []
        for res in rec_results:
            text, conf = filter_by_char_conf(res, self.char_conf_threshold)
            texts.append(text)
            confs.append(conf)

        latency = (time.perf_counter() - t0) * 1000

        # 6. visualise
        if self.save_viz and boxes:
            stem = Path(source).stem
            out_path = str(Path(self.viz_dir) / f"{stem}_{frame_idx:06d}.jpg")
            save_visualization(img, boxes, texts, out_path, region_map=region_map)

        result = FrameResult(
            source=source, frame_idx=frame_idx,
            boxes=boxes, texts=texts, confs=confs,
            latency_ms=latency,
        )
        log.info(
            "[%s | frame %d] %d regions | %.1f ms",
            Path(source).name, frame_idx, len(texts), latency,
        )
        return result

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
