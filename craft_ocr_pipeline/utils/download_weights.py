"""
One-time helper to download pretrained CRAFT weights.
Run: python utils/download_weights.py
"""

from __future__ import annotations

import sys
import urllib.request
import urllib.error
from pathlib import Path

# Primary — direct asset link from clovaai GitHub release
CRAFT_URL_PRIMARY = (
    "https://github.com/clovaai/CRAFT-pytorch/releases/download/"
    "pretrained/craft_mlt_25k.pth"
)

# Fallback — raw file from a known working mirror
CRAFT_URL_FALLBACK = (
    "https://github.com/fcakyon/craft-text-detector/releases/download/"
    "v0.4.2/craft_mlt_25k.pth"
)

DEST = "models/craft_mlt_25k.pth"


def download(url: str, dest: str) -> bool:
    """Try to download url → dest. Returns True on success, False on 404."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if Path(dest).exists():
        print(f"Already exists: {dest}")
        return True

    print(f"Trying: {url}")

    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    urllib.request.install_opener(opener)

    def _progress(block, block_size, total):
        downloaded = min(block * block_size, total if total > 0 else block * block_size)
        pct = downloaded / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:.1f}%  ({downloaded // 1024} KB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\nSaved → {dest}  ({Path(dest).stat().st_size // 1024 // 1024} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  HTTP {e.code} — skipping")
        return False
    except Exception as e:
        print(f"\n  Error: {e} — skipping")
        return False


def main() -> None:
    if Path(DEST).exists():
        print(f"Already downloaded: {DEST}")
        return

    for url in (CRAFT_URL_PRIMARY, CRAFT_URL_FALLBACK):
        if download(url, DEST):
            return

    # both URLs failed — print manual instructions
    print("\n" + "=" * 60)
    print("Automatic download failed.")
    print("Download the file manually using one of these options:")
    print()
    print("  Option 1 — PowerShell:")
    print(f"    Invoke-WebRequest -Uri \"{CRAFT_URL_FALLBACK}\" \\")
    print(f"      -OutFile models\\craft_mlt_25k.pth")
    print()
    print("  Option 2 — curl (if installed):")
    print(f"    curl -L \"{CRAFT_URL_FALLBACK}\" -o models/craft_mlt_25k.pth")
    print()
    print("  Option 3 — browser:")
    print(f"    {CRAFT_URL_FALLBACK}")
    print(f"    Save as: {Path(DEST).resolve()}")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
