"""
Export CRAFT and CRNN models to ONNX.
Run this ONCE on your dev machine (Windows / Linux with PyTorch installed).
Copy the .onnx files to the CM5 — PyTorch is never needed there.

Usage (run from craft_ocr_pipeline/):
  python scripts/export_onnx.py                   # export both
  python scripts/export_onnx.py --craft-only
  python scripts/export_onnx.py --crnn-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config


# ── CRAFT export ──────────────────────────────────────────────────────────────

def export_craft(cfg: dict) -> None:
    from models.craft_model import load_craft

    weights = cfg["paths"]["craft_weights"]
    out     = cfg["paths"]["craft_onnx"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading CRAFT weights from {weights} ...")
    device = torch.device("cpu")
    model  = load_craft(weights, device)

    # dummy input — dynamic H/W so any image size works at inference
    dummy = torch.randn(1, 3, 608, 608, dtype=torch.float32)

    print(f"[export] Exporting CRAFT → {out}")
    torch.onnx.export(
        model,
        dummy,
        out,
        opset_version=12,
        input_names=["image"],
        output_names=["region_map", "affinity_map"],
        dynamic_axes={
            "image":        {2: "height", 3: "width"},
            "region_map":   {1: "height", 2: "width"},
            "affinity_map": {1: "height", 2: "width"},
        },
    )
    _verify_onnx(out)
    print(f"[export] CRAFT ONNX saved → {out}  ({Path(out).stat().st_size // 1024 // 1024} MB)")


# ── CRNN export ───────────────────────────────────────────────────────────────

def export_crnn(cfg: dict) -> None:
    from models.crnn_model import CRNN

    ccfg    = cfg["recognition"]["crnn"]
    charset = ccfg.get("charset", "0123456789abcdefghijklmnopqrstuvwxyz")
    n_cls   = len(charset) + 1      # +1 for CTC blank
    img_h   = ccfg["img_height"]
    img_w   = ccfg["img_width"]

    weights = ccfg["model_path"]
    out     = cfg["paths"]["crnn_onnx"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading CRNN weights from {weights} ...")
    model = CRNN(img_height=img_h, n_classes=n_cls)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, img_h, img_w, dtype=torch.float32)

    print(f"[export] Exporting CRNN → {out}")
    torch.onnx.export(
        model,
        dummy,
        out,
        opset_version=12,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}},
    )
    _verify_onnx(out)
    print(f"[export] CRNN ONNX saved → {out}  ({Path(out).stat().st_size // 1024} KB)")


# ── helper ────────────────────────────────────────────────────────────────────

def _verify_onnx(path: str) -> None:
    try:
        import onnx  # type: ignore
        onnx.checker.check_model(path)
        print(f"[export] ONNX model verified OK: {path}")
    except ImportError:
        print("[export] onnx package not installed — skipping verification (optional)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--craft-only", action="store_true")
    group.add_argument("--crnn-only",  action="store_true")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.crnn_only:
        export_crnn(cfg)
    elif args.craft_only:
        export_craft(cfg)
    else:
        export_craft(cfg)
        export_crnn(cfg)

    print("\n[export] Done. Copy the .onnx files to the CM5.")


if __name__ == "__main__":
    main()
