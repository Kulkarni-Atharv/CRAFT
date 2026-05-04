"""
Live capture + CRAFT OCR for Raspberry Pi CM5 with Picamera2.

Controls:
  SPACE  → capture frame, run OCR, print results to terminal, then exit
  Q      → quit without capturing

Rerun the script for each new capture.

Install on CM5:
  pip install -r requirements_cm5.txt

Usage (run from craft_ocr_pipeline/):
  python scripts/capture_cm5.py
  python scripts/capture_cm5.py --config configs/config.yaml
  python scripts/capture_cm5.py --roi 100 50 800 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── allow imports from project root regardless of cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline            import OCRPipeline
from utils.config_loader import load_config, ensure_dirs
from utils.logger        import get_logger

YELLOW = (0, 200, 255)
BLACK  = (0, 0, 0)


# ── camera ────────────────────────────────────────────────────────────────────

def start_camera() -> "Picamera2":
    try:
        from picamera2 import Picamera2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "picamera2 not found. Install it:\n"
            "  sudo apt install -y python3-picamera2\n"
            "  # or: pip install picamera2"
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


def capture_frame(picam2) -> np.ndarray:
    frame = picam2.capture_array()
    # picamera2 returns RGB or XRGB (4-channel) depending on sensor format
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    # semi-transparent bar at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        "LIVE PREVIEW  |  SPACE = Capture & Exit    Q = Quit",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA,
    )


# ── terminal result printer ───────────────────────────────────────────────────

def print_results(result, elapsed: float) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  OCR Results  ({elapsed * 1000:.1f} ms)")
    print(sep)

    if not result.texts or all(t == "" for t in result.texts):
        print("  No text detected.")
    else:
        for i, (text, conf) in enumerate(zip(result.texts, result.confs), 1):
            if text:
                print(f"  [{i:02d}]  {text!r:<40}  conf={conf:.3f}")

    print(f"{sep}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    log = get_logger("capture_cm5", cfg["paths"]["log_dir"], cfg["pipeline"]["log_level"])

    log.info("Loading OCR pipeline...")
    pipeline = OCRPipeline(cfg)

    log.info("Starting camera...")
    picam2 = start_camera()
    log.info("Camera ready — SPACE to capture, Q to quit")

    roi = args.roi   # (x, y, w, h) or None

    try:
        while True:
            frame = capture_frame(picam2)

            if roi:
                x, y, w, h = roi
                display_frame = frame[y : y + h, x : x + w].copy()
            else:
                display_frame = frame.copy()

            draw_hud(display_frame)
            cv2.imshow("CRAFT OCR — CM5", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord(" "):
                # grab a clean frame (no HUD drawn on it)
                raw_frame = capture_frame(picam2)
                crop = (raw_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                        if roi else raw_frame)

                # BGR (OpenCV) → RGB (pipeline expects RGB)
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                print("\n[CM5] Capturing and running OCR...")
                t0 = time.time()
                result = pipeline.process_image(rgb_crop, source="cm5_capture")
                elapsed = time.time() - t0

                print_results(result, elapsed)
                break   # close preview and exit after one capture

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAFT OCR live capture on CM5")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Camera crop region in pixels: x y w h",
    )
    main(parser.parse_args())
