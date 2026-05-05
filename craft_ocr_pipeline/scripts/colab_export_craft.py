"""
Standalone CRAFT → ONNX export script.
Run this in Google Colab (or any machine with PyTorch).

Colab setup cells:
    !pip install torch torchvision --quiet
    !wget -q "https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.pth"

Then run this script:
    !python colab_export_craft.py

Download craft.onnx and craft.onnx.data, push to GitHub.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn

WEIGHTS = "craft_mlt_25k.pth"
OUT     = "craft.onnx"


# ── Exact CRAFT architecture (must match craft_model.py) ──────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class VGGBasenet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg      = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())   # 44 layers (0-43)
        self.slice1 = nn.Sequential(*features[:12])
        self.slice2 = nn.Sequential(*features[12:19])
        self.slice3 = nn.Sequential(*features[19:29])
        self.slice4 = nn.Sequential(*features[29:])      # ALL remaining VGG (29-43)
        self.slice5 = nn.Sequential(                      # ONLY custom dilated convs
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )
    def forward(self, x):
        s1 = self.slice1(x)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)
        s5 = self.slice5(s4)
        return s5, s4, s3, s2, s1


class CRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.basenet  = VGGBasenet()
        self.upconv1  = DoubleConv(1024, 512, 256)
        self.upconv2  = DoubleConv(512,  256, 128)
        self.upconv3  = DoubleConv(256,  128, 64)
        self.upconv4  = DoubleConv(128,  64,  32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),            nn.ReLU(inplace=True),
            nn.Conv2d(16,  2, 1),
        )

    def forward(self, x):
        s5, s4, s3, s2, s1 = self.basenet(x)
        y = F.interpolate(s5, size=s4.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv1(torch.cat([y, s4], dim=1))
        y = F.interpolate(y,  size=s3.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv2(torch.cat([y, s3], dim=1))
        y = F.interpolate(y,  size=s2.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv3(torch.cat([y, s2], dim=1))
        y = F.interpolate(y,  size=s1.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv4(torch.cat([y, s1], dim=1))
        y = self.conv_cls(y)
        return torch.sigmoid(y[:, 0]), torch.sigmoid(y[:, 1])


# ── Load weights ──────────────────────────────────────────────────────────────

def load_craft(weights_path: str) -> CRAFT:
    net   = CRAFT()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)

    if list(state.keys())[0].startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())

    # The original CRAFT-pytorch names slice5 by absolute VGG feature index
    # (39=MaxPool, 40=Conv1024, 41=Conv1024). Our Sequential uses 0-indexed keys.
    # Remap so the dilated conv weights actually load into our model.
    remap = {
        "basenet.slice5.39.": "basenet.slice5.0.",
        "basenet.slice5.40.": "basenet.slice5.1.",
        "basenet.slice5.41.": "basenet.slice5.2.",
    }
    state = OrderedDict(
        (next((k.replace(o, n) for o, n in remap.items() if k.startswith(o)), k), v)
        for k, v in state.items()
    )

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[load] missing keys   : {len(missing)}  (slice2-4 — kept from VGG ImageNet, expected)")
    print(f"[load] unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"       first few: {unexpected[:3]}")
    net.eval()
    return net


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(net: CRAFT) -> None:
    print("\n[sanity] Running PyTorch forward pass...")
    # simulate a text-like input: white background with a dark stripe
    x = torch.ones(1, 3, 608, 608) * 0.5
    x[:, :, 280:320, 100:500] = -1.0   # dark horizontal band = "text row"
    with torch.no_grad():
        region, affinity = net(x)
    print(f"[sanity] region   min={region.min():.3f}  max={region.max():.3f}  mean={region.mean():.3f}")
    print(f"[sanity] affinity min={affinity.min():.3f}  max={affinity.max():.3f}  mean={affinity.mean():.3f}")
    spread = float(region.max() - region.min())
    if spread < 0.05:
        print(f"[sanity] FAIL — output spread={spread:.4f} (flat ~0.5 = slice5 weights not loaded)")
        print("         The key remapping for basenet.slice5.39/40/41 may have failed.")
        sys.exit(1)
    else:
        print(f"[sanity] PASS — output spread={spread:.4f} (weights loaded correctly)")


# ── ONNX export ───────────────────────────────────────────────────────────────

def export(net: CRAFT, out_path: str) -> None:
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    dummy = torch.randn(1, 3, 608, 608, dtype=torch.float32)
    tmp   = out_path + ".tmp.onnx"

    print(f"\n[export] Exporting (legacy exporter, opset 11) → {tmp}")
    # Force the legacy TorchScript-based exporter (not dynamo).
    # dynamo exporter struggles with opset conversion and external data on Colab.
    torch.onnx.export(
        net, dummy, tmp,
        dynamo=False,                 # legacy exporter — stable, single-file output
        opset_version=11,
        input_names=["image"],
        output_names=["region_map", "affinity_map"],
        dynamic_axes={
            "image":        {2: "height", 3: "width"},
            "region_map":   {1: "height", 2: "width"},
            "affinity_map": {1: "height", 2: "width"},
        },
    )

    # The legacy exporter may still write an external-data sidecar for large models.
    # Consolidate everything into one self-contained file so only craft.onnx is needed.
    print("[export] Consolidating weights into single file...")
    m = onnx.load(tmp, load_external_data=False)
    load_external_data_for_model(m, str(Path(tmp).parent))
    onnx.save(m, out_path)

    for f in Path(".").glob("*.tmp.onnx*"):
        f.unlink()

    size_kb = Path(out_path).stat().st_size // 1024
    print(f"[export] Single-file ONNX saved ({size_kb} KB / {size_kb//1024} MB) → {out_path}")


# ── ONNX runtime sanity check ─────────────────────────────────────────────────

def verify_onnx(out_path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skipping (pip install onnxruntime)")
        return

    sess   = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    name   = sess.get_inputs()[0].name
    x      = np.random.randn(1, 3, 608, 608).astype(np.float32)
    region = sess.run(None, {name: x})[0]
    spread = float(region.max() - region.min())
    print(f"\n[verify] ONNX region spread={spread:.4f}  max={region.max():.3f}")
    if spread < 0.01:
        print("[verify] FAIL — output is flat. Re-check weights and re-export.")
        sys.exit(1)
    else:
        print("[verify] PASS — ONNX model outputs vary with input")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not Path(WEIGHTS).exists():
        print(f"Weights not found: {WEIGHTS}")
        print("Download with:")
        print('  !wget -q "https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.pth"')
        sys.exit(1)

    print(f"[main] Loading CRAFT weights from {WEIGHTS} ...")
    net = load_craft(WEIGHTS)
    sanity_check(net)
    export(net, OUT)
    verify_onnx(OUT)
    print(f"\n[done] Download {OUT} (and {OUT}.data if it exists)")
    print("       Push both files to GitHub → git pull on CM5")
