"""
CRAFT OCR Pipeline — single entry point.

Camera mode (CM5 / Picamera2)
------------------------------
  python run.py
  python run.py --roi 100 50 800 400

  No --input needed. Live preview opens → SPACE to capture → results printed
  to terminal → script exits. Rerun for the next capture.

Static file mode
-----------------
  python run.py --input inputs/sample.jpg
  python run.py --input inputs/
  python run.py --input inputs/video.mp4
  python run.py --input inputs/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from utils.config_loader import load_config, ensure_dirs
from utils.logger        import get_logger
from pipeline            import OCRPipeline


YELLOW = (0, 200, 255)
BLACK  = (0, 0, 0)


# ── Camera helpers ────────────────────────────────────────────────────────────

def _start_camera():
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "picamera2 not found.\n"
            "  sudo apt install -y python3-picamera2"
        ) from e

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1456, 1088)},
        lores={"size": (640, 480)},
        display="main",
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)   # let AEC/AWB settle
    return picam2


def _capture_frame(picam2) -> np.ndarray:
    frame = picam2.capture_array()
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _draw_hud(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        "LIVE PREVIEW  |  SPACE = Capture & Exit    Q = Quit",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA,
    )


# ── Result printer ────────────────────────────────────────────────────────────

def _print_results(result, elapsed_ms: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  OCR Results  ({elapsed_ms:.1f} ms)")
    print(sep)
    if not result.boxes:
        print("  No text regions detected.")
    elif not result.texts or all(t == "" for t in result.texts):
        print(f"  {len(result.boxes)} text region(s) detected (recognition disabled).")
        print("  Run with recognition enabled to extract text.")
    else:
        for i, (text, conf) in enumerate(zip(result.texts, result.confs), 1):
            if text:
                print(f"  [{i:02d}]  {text!r:<40}  conf={conf:.3f}")
    print(f"{sep}\n")


# ── Camera mode ───────────────────────────────────────────────────────────────

def run_camera(pipeline: OCRPipeline, cfg: dict, roi: list[int] | None) -> None:
    log = get_logger("run.camera", cfg["paths"]["log_dir"])

    log.info("Starting camera...")
    picam2 = _start_camera()
    log.info("Camera ready — SPACE to capture, Q to quit")

    try:
        while True:
            frame = _capture_frame(picam2)

            if roi:
                x, y, w, h = roi
                display = frame[y : y + h, x : x + w].copy()
            else:
                display = frame.copy()

            _draw_hud(display)
            cv2.imshow("CRAFT OCR", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                raw = _capture_frame(picam2)
                crop = (raw[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                        if roi else raw)

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                print("\nCapturing and running OCR...")
                t0 = time.perf_counter()
                result = pipeline.process_image(rgb, source="camera")
                elapsed_ms = (time.perf_counter() - t0) * 1000

                _print_results(result, elapsed_ms)
                break   # exit after one capture — rerun for next

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        log.info("Done.")


# ── Static file mode ──────────────────────────────────────────────────────────

def run_static(pipeline: OCRPipeline, cfg: dict, args: argparse.Namespace) -> None:
    log = get_logger("run.static", cfg["paths"]["log_dir"])

    inp = Path(args.input)
    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if inp.is_dir():
        all_results = pipeline.process_directory(str(inp))
    elif inp.suffix.lower() in video_exts:
        all_results = pipeline.process_video(str(inp))
    else:
        all_results = [pipeline.process_image_file(str(inp))]

    print("\n" + "=" * 60)
    for r in all_results:
        print(f"[{Path(r.source).name} | frame {r.frame_idx}]  "
              f"{len(r.texts)} detections  ({r.latency_ms:.1f} ms)")
        for text, conf in zip(r.texts, r.confs):
            if text:
                print(f"  → {text!r:<40}  conf={conf:.3f}")
    print("=" * 60)

    if args.output:
        payload = [r.to_dict() for r in all_results]
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        log.info("Results saved to %s", args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRAFT OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Camera mode (no --input):\n"
            "  python run.py\n"
            "  python run.py --roi 100 50 800 400\n\n"
            "Static mode:\n"
            "  python run.py --input inputs/image.jpg\n"
            "  python run.py --input inputs/  --output results.json\n"
        ),
    )
    p.add_argument("--input",       default=None, help="Image, directory, or video (omit for camera)")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--output",      default=None, help="Save JSON results (static mode only)")
    p.add_argument("--roi",         nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="Camera crop region: x y w h (camera mode only)")
    p.add_argument("--detect-only", action="store_true",
                   help="Skip recognition — output bounding boxes only (useful when crnn.onnx not yet available)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    ensure_dirs(cfg)

    if args.detect_only:
        cfg["recognition"]["detect_only"] = True

    log = get_logger("run", cfg["paths"]["log_dir"], cfg["pipeline"]["log_level"])
    log.info("Loading OCR pipeline...")
    pipeline = OCRPipeline(cfg)

    if args.input is None:
        # ── camera mode ───────────────────────────────────────────────────────
        run_camera(pipeline, cfg, args.roi)
    else:
        # ── static file mode ──────────────────────────────────────────────────
        run_static(pipeline, cfg, args)


if __name__ == "__main__":
    main()
