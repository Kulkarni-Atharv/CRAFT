"""
Train CRNN on synthetic alphanumeric data — no real dataset needed.
Run from craft_ocr_pipeline/ :

  python scripts/train_crnn_quick.py                  # ~40 k steps, ~15 min GPU
  python scripts/train_crnn_quick.py --iterations 20000  # faster / less accurate

After training:
  python scripts/export_onnx.py --crnn-only
  # push models/crnn.onnx to GitHub, then git pull on CM5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.crnn_model import CRNN
from utils.config_loader import load_config


# ── fonts ─────────────────────────────────────────────────────────────────────

def _load_fonts() -> list:
    candidates = [
        # Windows
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        # Linux / CM5
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    ]
    fonts: list = []
    for p in candidates:
        if p.exists():
            for size in (18, 22, 26, 30):
                try:
                    fonts.append(ImageFont.truetype(str(p), size))
                except Exception:
                    pass
    if not fonts:
        fonts = [ImageFont.load_default()]
        print("[train] No TrueType fonts found — using PIL default (lower quality)")
    else:
        print(f"[train] Loaded {len(fonts)} font variants from system fonts")
    return fonts


# ── synthetic dataset ─────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):

    def __init__(
        self,
        charset: str,
        size:    int  = 50_000,
        img_h:   int  = 32,
        img_w:   int  = 100,
        max_len: int  = 10,
    ):
        self.charset  = charset
        self.chars    = ["-"] + list(charset)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.size     = size
        self.img_h    = img_h
        self.img_w    = img_w
        self.max_len  = max_len
        self.fonts    = _load_fonts()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _idx):
        length = random.randint(1, self.max_len)
        word   = "".join(random.choice(self.charset) for _ in range(length))
        tensor = self._render(word)
        label  = [self.char2idx[c] for c in word]
        return tensor, label, len(label)

    def _render(self, word: str) -> torch.Tensor:
        font = random.choice(self.fonts)
        bg   = random.randint(200, 255)
        fg   = random.randint(0,   50)

        # render on large canvas, crop tight, then resize to target
        canvas = Image.new("L", (self.img_w * 4, self.img_h * 4), bg)
        ImageDraw.Draw(canvas).text((6, 6), word, fill=fg, font=font)

        bbox = canvas.getbbox()
        if bbox:
            canvas = canvas.crop(bbox)

        canvas = canvas.resize((self.img_w, self.img_h), Image.BILINEAR)

        # light augmentation
        if random.random() < 0.3:
            canvas = canvas.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8))
            )

        arr = np.array(canvas, dtype=np.float32)
        arr = np.clip(arr + random.uniform(-20, 20), 0, 255)
        arr = (arr / 255.0 - 0.5) / 0.5
        return torch.from_numpy(arr).unsqueeze(0)   # 1×H×W


def _collate(batch):
    images, labels, lengths = zip(*batch)
    images  = torch.stack(images, 0)
    flat    = torch.tensor([i for lbl in labels for i in lbl], dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, flat, lengths


# ── greedy decode ─────────────────────────────────────────────────────────────

def _greedy(logits: torch.Tensor, chars: list[str]) -> str:
    indices = logits.argmax(dim=-1).tolist()
    prev, out = -1, []
    for idx in indices:
        if idx != prev and idx != 0:
            out.append(chars[idx])
        prev = idx
    return "".join(out)


# ── validation ────────────────────────────────────────────────────────────────

def _validate(model, loader, device, chars) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, flat, lengths in loader:
            out    = model(images.to(device))   # T×B×n_cls
            offset = 0
            for b in range(images.size(0)):
                n      = lengths[b].item()
                target = "".join(chars[i] for i in flat[offset:offset + n].tolist())
                pred   = _greedy(out[:, b, :], chars)
                correct += int(pred == target)
                total   += 1
                offset  += n
    model.train()
    return correct / max(total, 1)


# ── main training loop ────────────────────────────────────────────────────────

def train(cfg_path: str, iterations: int, batch_size: int) -> None:
    cfg     = load_config(cfg_path)
    ccfg    = cfg["recognition"]["crnn"]
    img_h   = ccfg["img_height"]
    img_w   = ccfg["img_width"]
    charset = ccfg.get("charset", "0123456789abcdefghijklmnopqrstuvwxyz")
    chars   = ["-"] + list(charset)
    n_cls   = len(chars)
    out_path = ccfg["model_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  classes={n_cls}  img={img_h}×{img_w}  iterations={iterations}")

    model = CRNN(img_height=img_h, n_classes=n_cls).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15_000, gamma=0.1)
    ctc   = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    ds_train = SyntheticDataset(charset, size=iterations * batch_size,
                                img_h=img_h, img_w=img_w)
    ds_val   = SyntheticDataset(charset, size=512, img_h=img_h, img_w=img_w)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=32, collate_fn=_collate, num_workers=0)

    model.train()
    step, running_loss = 0, 0.0

    for images, flat, label_lengths in dl_train:
        if step >= iterations:
            break

        images = images.to(device)
        out    = model(images)                          # T×B×n_cls
        T      = out.size(0)
        in_len = torch.full((images.size(0),), T, dtype=torch.long)

        loss = ctc(
            F.log_softmax(out, dim=2),
            flat.to(device),
            in_len,
            label_lengths.to(device),
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        running_loss += loss.item()
        step         += 1

        if step % 1000 == 0:
            avg  = running_loss / 1000
            running_loss = 0.0
            acc  = _validate(model, dl_val, device, chars)
            print(f"[train] step {step:6d}/{iterations}  loss={avg:.4f}  val_acc={acc:.3f}")
            torch.save(model.state_dict(), out_path)
            if acc >= 0.95:
                print("[train] val_acc ≥ 0.95 — stopping early")
                break

    torch.save(model.state_dict(), out_path)
    final_acc = _validate(model, dl_val, device, chars)
    print(f"\n[train] Done.  val_acc={final_acc:.3f}  saved → {out_path}")
    print("[train] Next step: python scripts/export_onnx.py --crnn-only")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CRNN on synthetic data")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--iterations", type=int, default=40_000,
                   help="Training steps (default 40000 ≈ 15 min GPU, 45 min CPU)")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    train(args.config, args.iterations, args.batch_size)
