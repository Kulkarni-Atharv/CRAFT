"""
Train an isolated character classifier CNN.
Exports to ONNX → models/char_classifier.onnx

Run on dev machine (Windows/Linux) or Google Colab:
  pip install torch torchvision onnx pillow numpy tqdm
  python scripts/train_char_classifier.py

Colab one-liner:
  !python scripts/train_char_classifier.py --epochs 20 --batch-size 128

What this trains:
  MobileNetV3-Small (pretrained ImageNet) fine-tuned on:
    1. EMNIST Extended  — real handwritten A-Z, a-z, 0-9
    2. Synthetic renders — all 95 printable ASCII in 6 fonts
       with blur, rotation, brightness, perspective augmentation

Output:
  models/char_classifier.onnx   (~5 MB, runs on CM5 with onnxruntime)
"""

from __future__ import annotations

import argparse
import io
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm


# ── Character set ──────────────────────────────────────────────────────────────

PRINTABLE_ASCII = [chr(i) for i in range(32, 127)]   # 95 chars: space … ~
N_CLASSES       = len(PRINTABLE_ASCII)                # 95
CHAR_TO_IDX     = {c: i for i, c in enumerate(PRINTABLE_ASCII)}

# EMNIST "balanced" label → actual character mapping
# EMNIST balanced: 0-9=digits, 10-35=A-Z, 36-46=some lowercase
EMNIST_BALANCED_LABELS = (
    list("0123456789") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")[:37]
)


# ── Augmentation transforms ────────────────────────────────────────────────────

def _train_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15, fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=5,
            fill=255,
        ),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _val_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Dataset 1: EMNIST Extended ─────────────────────────────────────────────────

class EMNISTWrapper(Dataset):
    """
    Wraps torchvision EMNIST, remaps labels to PRINTABLE_ASCII indices,
    converts grayscale to RGB, and applies the shared transform.
    """

    def __init__(self, root: str, train: bool, transform):
        self._ds = EMNIST(
            root=root,
            split="balanced",
            train=train,
            download=True,
            transform=None,
        )
        self._transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, label = self._ds[idx]

        # EMNIST images are transposed — correct orientation
        img = img.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)

        # grayscale → RGB
        img = img.convert("RGB")

        # remap EMNIST label to PRINTABLE_ASCII index
        if label < len(EMNIST_BALANCED_LABELS):
            char = EMNIST_BALANCED_LABELS[label]
        else:
            char = "?"   # unknown — will be filtered
        mapped = CHAR_TO_IDX.get(char, -1)
        if mapped < 0:
            # skip unknowns by mapping to space (index 0)
            mapped = CHAR_TO_IDX[" "]

        return self._transform(img), mapped


# ── Dataset 2: Synthetic Renders ───────────────────────────────────────────────

FONTS = [
    "arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf",
    "cour.ttf",  "courbd.ttf",
]

def _load_fonts(size: int = 22) -> list:
    """Try to load system fonts; fall back to PIL default."""
    loaded = []
    for name in FONTS:
        try:
            loaded.append(ImageFont.truetype(name, size))
        except (OSError, IOError):
            pass
    if not loaded:
        # PIL built-in bitmap font — always available
        loaded.append(ImageFont.load_default())
    return loaded


class SyntheticCharDataset(Dataset):
    """
    Renders every printable ASCII character in multiple fonts with
    augmentation. Simulates what CRAFT character crops look like.

    samples_per_char: how many augmented variants to generate per (char, font)
    """

    def __init__(self, transform, samples_per_char: int = 20):
        self._transform        = transform
        self._samples_per_char = samples_per_char
        self._fonts            = _load_fonts(size=22)
        self._index            = self._build_index()

    def _build_index(self) -> list[tuple[str, int, object]]:
        items = []
        for char in PRINTABLE_ASCII:
            if char == " ":
                continue   # skip space — not a visible character
            for font in self._fonts:
                for _ in range(self._samples_per_char):
                    items.append((char, CHAR_TO_IDX[char], font))
        return items

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        char, label, font = self._index[idx]
        img = self._render(char, font)
        return self._transform(img), label

    @staticmethod
    def _render(char: str, font) -> Image.Image:
        """
        Render one character on a white 48×48 canvas, apply random
        background colour, noise, and blur to mimic real camera crops.
        """
        size = 48

        # random light background (simulate label paper)
        bg   = tuple(np.random.randint(200, 256, 3).tolist())
        fg   = tuple(np.random.randint(0, 80, 3).tolist())

        img  = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # centre the character
        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = font.getsize(char)

        x = (size - w) // 2 + np.random.randint(-3, 4)
        y = (size - h) // 2 + np.random.randint(-3, 4)
        draw.text((x, y), char, fill=fg, font=font)

        # random blur
        sigma = np.random.uniform(0.3, 1.5)
        img   = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        # add salt-and-pepper noise
        arr  = np.array(img, dtype=np.uint8)
        mask = np.random.rand(*arr.shape[:2]) < 0.03
        arr[mask] = np.random.randint(0, 256, (mask.sum(), 3), dtype=np.uint8)
        img  = Image.fromarray(arr)

        return img.resize((32, 32), Image.BICUBIC)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(n_classes: int) -> nn.Module:
    """MobileNetV3-Small with a replaced classifier head."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # replace the final linear layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Loading EMNIST + synthetic data ...")
    train_emnist = EMNISTWrapper(args.data_dir, train=True,  transform=_train_transform())
    val_emnist   = EMNISTWrapper(args.data_dir, train=False, transform=_val_transform())
    train_synth  = SyntheticCharDataset(_train_transform(), samples_per_char=args.synth_per_char)

    train_ds = ConcatDataset([train_emnist, train_synth])
    val_ds   = val_emnist

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(N_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    ckpt_path = Path(args.output).parent / "char_classifier_best.pth"

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

        train_acc = correct / total

        # ── validate ───────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds       = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:3d}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"  -> saved best checkpoint (val_acc={best_acc:.4f})")

    # ── export to ONNX ─────────────────────────────────────────────────────────
    print(f"\nBest val_acc: {best_acc:.4f}")
    print("Exporting to ONNX ...")

    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval().cpu()

    dummy = torch.zeros(1, 3, 32, 32)
    output_path = str(args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names  = ["input"],
        output_names = ["logits"],
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version = 11,
    )

    # verify
    import onnxruntime as ort
    sess    = ort.InferenceSession(output_path)
    out     = sess.run(None, {"input": dummy.numpy()})[0]
    pred_ch = PRINTABLE_ASCII[int(out[0].argmax())]
    print(f"ONNX verification: input=zeros, predicted='{pred_ch}' (random, expected)")
    print(f"Model saved to: {output_path}  ({Path(output_path).stat().st_size / 1e6:.1f} MB)")
    print("Done. Copy models/char_classifier.onnx to CM5.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train isolated character classifier")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--workers",         type=int,   default=4)
    p.add_argument("--synth-per-char",  type=int,   default=20,
                   help="Synthetic samples per (char, font) pair")
    p.add_argument("--data-dir",        default="data/",
                   help="Root directory for EMNIST download")
    p.add_argument("--output",          default="models/char_classifier.onnx",
                   help="Output ONNX model path")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
