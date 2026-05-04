"""
One-time helper to download pretrained CRAFT weights.
Run: python utils/download_weights.py
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

CRAFT_URL = (
    "https://github.com/clovaai/CRAFT-pytorch/releases/download/"
    "pretrained/craft_mlt_25k.pth"
)


def download(url: str, dest: str) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if Path(dest).exists():
        print(f"Already exists: {dest}")
        return

    print(f"Downloading {url} → {dest}")

    def _progress(block, block_size, total):
        downloaded = block * block_size
        pct = downloaded / total * 100 if total > 0 else 0
        print(f"\r  {pct:.1f}%  ({downloaded // 1024} KB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\nDone: {dest}")


if __name__ == "__main__":
    download(CRAFT_URL, "models/craft_mlt_25k.pth")
