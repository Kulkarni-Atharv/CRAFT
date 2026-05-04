"""
Entry point — run the OCR pipeline from the command line.

Examples
--------
# single image
python run.py --input inputs/sample.jpg

# directory of images
python run.py --input inputs/

# video file
python run.py --input inputs/video.mp4

# custom config + save output JSON
python run.py --input inputs/ --config configs/config.yaml --output results.json

# character-level heatmap only (no recognition)
python run.py --input inputs/sample.jpg --heatmap-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.config_loader import load_config, ensure_dirs
from utils.logger        import get_logger
from pipeline            import OCRPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRAFT OCR Pipeline")
    p.add_argument("--input",       required=True,                help="Image, directory, or video path")
    p.add_argument("--config",      default="configs/config.yaml", help="YAML config path")
    p.add_argument("--output",      default=None,                  help="Save JSON results to this file")
    p.add_argument("--heatmap-only",action="store_true",           help="Run detection only, skip recognition")
    p.add_argument("--show",        action="store_true",           help="Display result images (requires display)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    ensure_dirs(cfg)

    if args.show:
        cfg["video"]["display"] = True

    log = get_logger("run", cfg["paths"]["log_dir"], cfg["pipeline"]["log_level"])
    pipeline = OCRPipeline(cfg)

    inp = Path(args.input)
    if not inp.exists():
        log.error("Input not found: %s", inp)
        sys.exit(1)

    # ── dispatch ──────────────────────────────────────────────────────────────
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if inp.is_dir():
        all_results = pipeline.process_directory(str(inp))
    elif inp.suffix.lower() in video_exts:
        all_results = pipeline.process_video(str(inp))
    else:
        all_results = [pipeline.process_image_file(str(inp))]

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for r in all_results:
        print(f"[{Path(r.source).name} | frame {r.frame_idx}]  {len(r.texts)} detections  ({r.latency_ms:.1f} ms)")
        for text, conf in zip(r.texts, r.confs):
            print(f"  → {text!r:<40}  conf={conf:.3f}")
    print("=" * 60)

    # ── save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        payload = [r.to_dict() for r in all_results]
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
