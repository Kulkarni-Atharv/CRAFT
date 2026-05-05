"""
Lightweight CRNN (CNN + BiLSTM + CTC) for recognition.
Used only when config recognition.engine == 'crnn'.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _BidirLSTM(nn.Module):
    def __init__(self, in_size: int, hidden: int, out_size: int):
        super().__init__()
        self.rnn  = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.proj(out)


class CRNN(nn.Module):
    """
    Input  : B×1×H×W  (H=32 by convention)
    Output : T×B×n_classes  (CTC format)
    """

    def __init__(self, img_height: int = 32, n_classes: int = 37, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64,  3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        # compute rnn input size dynamically — the last Conv2d(k=2) reduces H by 1,
        # so for H=32 the actual feature height is 1 (not 2), giving rnn_in=512
        with torch.no_grad():
            _probe = self.cnn(torch.zeros(1, 1, img_height, 32))
            _, _c, _h, _ = _probe.shape
            rnn_in = _c * _h
        self.rnn = nn.Sequential(
            _BidirLSTM(rnn_in, hidden, hidden),
            _BidirLSTM(hidden, hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                      # B×C×H'×W'
        b, c, h, w = feat.shape
        feat = feat.permute(0, 3, 1, 2)         # B×W'×C×H'
        feat = feat.reshape(b, w, c * h)        # B×W'×(C*H')
        out  = self.rnn(feat)                   # B×W'×n_classes
        return out.permute(1, 0, 2)             # W'×B×n_classes  (CTC)
