"""
CRAFT network definition.
Source: https://github.com/clovaai/CRAFT-pytorch  (MIT License)
Minor style edits only — architecture is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


# ── VGG-16 BN backbone (encoder) ──────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16_bn(weights=None)  # we load CRAFT weights separately
        features = list(vgg.features.children())

        self.slice1 = nn.Sequential(*features[:12])   # out: 64ch, /2
        self.slice2 = nn.Sequential(*features[12:19]) # out: 128ch, /4
        self.slice3 = nn.Sequential(*features[19:29]) # out: 256ch, /8
        self.slice4 = nn.Sequential(*features[29:39]) # out: 512ch, /16

        # extra conv layers (from CRAFT's "basenet")
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        for param in self.parameters():
            param.requires_grad = False  # backbone frozen at inference

    def forward(self, x: torch.Tensor):
        s1 = self.slice1(x)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)
        s5 = self.slice5(s4)
        return s1, s2, s3, s4, s5


# ── U-Net style decoder head ───────────────────────────────────────────────────

def _double_conv(in_ch: int, mid_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 1),
        nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class CRAFTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _double_conv(1024 + 512, 512, 256)
        self.conv2 = _double_conv(256 + 512,  256, 128)
        self.conv3 = _double_conv(128 + 256,  128, 64)
        self.conv4 = _double_conv(64  + 128,  64,  32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16,  16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16,  2,  1),            # [region_score, affinity_score]
        )

    def forward(self, features):
        s1, s2, s3, s4, s5 = features
        y = F.interpolate(s5, size=s4.shape[2:], mode="bilinear", align_corners=False)
        y = self.conv1(torch.cat([y, s4], dim=1))
        y = F.interpolate(y,  size=s3.shape[2:], mode="bilinear", align_corners=False)
        y = self.conv2(torch.cat([y, s3], dim=1))
        y = F.interpolate(y,  size=s2.shape[2:], mode="bilinear", align_corners=False)
        y = self.conv3(torch.cat([y, s2], dim=1))
        y = F.interpolate(y,  size=s1.shape[2:], mode="bilinear", align_corners=False)
        y = self.conv4(torch.cat([y, s1], dim=1))
        return self.conv_cls(y)


# ── Full CRAFT model ───────────────────────────────────────────────────────────

class CRAFT(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.basenet  = VGGFeatureExtractor()
        self.decoder  = CRAFTDecoder()

    def forward(self, x: torch.Tensor):
        features = self.basenet(x)
        y = self.decoder(features)
        # sigmoid to get [0,1] score maps
        region = torch.sigmoid(y[:, 0, :, :])
        affinity = torch.sigmoid(y[:, 1, :, :])
        return region, affinity


def load_craft(weights_path: str, device: torch.device) -> CRAFT:
    """Load CRAFT with pretrained weights."""
    net = CRAFT().to(device)
    state = torch.load(weights_path, map_location=device)

    # handle DataParallel-wrapped checkpoints
    if "module." in list(state.keys())[0]:
        from collections import OrderedDict
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())

    net.load_state_dict(state)
    net.eval()
    return net
