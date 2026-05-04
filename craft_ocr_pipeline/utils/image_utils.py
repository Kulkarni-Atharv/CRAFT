"""Shared image helpers used across multiple modules."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize so the longest side == max_side; return (resized, scale)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    # ensure dimensions are multiples of 32 (CRAFT requirement)
    new_w = max(new_w - (new_w % 32), 32)
    new_h = max(new_h - (new_h % 32), 32)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def pad_to_multiple(img: np.ndarray, multiple: int = 32) -> np.ndarray:
    """Zero-pad image so H and W are multiples of `multiple`."""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
