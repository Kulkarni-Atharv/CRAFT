"""
CRAFT network definition — matches the EXACT weight layout of craft_mlt_25k.pth.
Source: https://github.com/clovaai/CRAFT-pytorch  (MIT License)

State-dict key structure
-------------------------
  basenet.slice1.*   — fine-tuned, stored in .pth
  basenet.slice2-4.* — standard VGG layers, NOT in .pth (loaded from ImageNet)
  basenet.slice5.*   — dilated convs, fine-tuned, stored in .pth
  upconv1.conv.*     — decoder block 1, stored in .pth
  upconv2.conv.*     — decoder block 2
  upconv3.conv.*     — decoder block 3
  upconv4.conv.*     — decoder block 4
  conv_cls.*         — score head, stored in .pth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights


# ── Decoder block ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Matches the `double_conv` class in CRAFT-pytorch.
    State-dict keys: upconv{n}.conv.0  (Conv2d)
                     upconv{n}.conv.1  (BN)
                     upconv{n}.conv.3  (Conv2d)
                     upconv{n}.conv.4  (BN)
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── VGG-16 BN backbone ─────────────────────────────────────────────────────────

class VGGBasenet(nn.Module):
    """
    Matches the `vgg16_bn` class in CRAFT-pytorch.

    Slice boundaries (VGG16-BN features, 0-indexed):
      slice1 [:12]  → 128-ch output
      slice2 [12:19]→ 256-ch output  (pretrained only — not in .pth)
      slice3 [19:29]→ 512-ch output  (pretrained only — not in .pth)
      slice4 [29:39]→ 512-ch output  (pretrained only — not in .pth)
      slice5 [39:]  + dilated convs → 1024-ch output
    """

    def __init__(self):
        super().__init__()
        vgg      = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())   # 44 layers (0-43)

        self.slice1 = nn.Sequential(*features[:12])
        self.slice2 = nn.Sequential(*features[12:19])
        self.slice3 = nn.Sequential(*features[19:29])
        self.slice4 = nn.Sequential(*features[29:39])
        self.slice5 = nn.Sequential(
            *features[39:],                                        # 39-43 (5 layers)
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        s1 = self.slice1(x)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)
        s5 = self.slice5(s4)
        # return in decoder order: deepest first, shallowest last
        return s5, s4, s3, s2, s1


# ── Full CRAFT model ───────────────────────────────────────────────────────────

class CRAFT(nn.Module):

    def __init__(self):
        super().__init__()
        self.basenet  = VGGBasenet()
        self.upconv1  = DoubleConv(1024, 512, 256)   # s5(1024) cat s4(512)
        self.upconv2  = DoubleConv(512,  256, 128)   # up1(256) cat s3(512)  -- wait
        self.upconv3  = DoubleConv(256,  128, 64)    # up2(128) cat s2(256)
        self.upconv4  = DoubleConv(128,  64,  32)    # up3(64)  cat s1(128)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1),            nn.ReLU(inplace=True),
            nn.Conv2d(16,  2, 1),
        )

    def forward(self, x: torch.Tensor):
        s5, s4, s3, s2, s1 = self.basenet(x)

        y = F.interpolate(s5, size=s4.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv1(torch.cat([y, s4], dim=1))   # 1024+512 → 256

        y = F.interpolate(y,  size=s3.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv2(torch.cat([y, s3], dim=1))   # 256+512  → 128

        y = F.interpolate(y,  size=s2.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv3(torch.cat([y, s2], dim=1))   # 128+256  → 64

        y = F.interpolate(y,  size=s1.shape[2:], mode="bilinear", align_corners=False)
        y = self.upconv4(torch.cat([y, s1], dim=1))   # 64+128   → 32

        y = self.conv_cls(y)                           # 32 → 2
        return torch.sigmoid(y[:, 0]), torch.sigmoid(y[:, 1])


# ── Weight loader ──────────────────────────────────────────────────────────────

def load_craft(weights_path: str, device: torch.device) -> CRAFT:
    """
    Load CRAFT.
    slice2-4 keep their ImageNet-pretrained VGG weights (not stored in .pth).
    All other layers are overridden by the .pth.
    """
    from collections import OrderedDict

    net   = CRAFT().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)

    if list(state.keys())[0].startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())

    # strict=False: silently skips slice2-4 (not in .pth — pretrained VGG used)
    _, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        import logging
        logging.getLogger(__name__).warning("Unexpected keys in checkpoint: %s", unexpected[:3])

    net.eval()
    return net
