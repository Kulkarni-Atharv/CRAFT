"""
Bonus Module — Visualization
Draws bounding boxes and recognised text on the source image.
Also supports character-level heatmap overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib.cm as cm


# ── box drawing ───────────────────────────────────────────────────────────────

def draw_boxes(
    img: np.ndarray,
    boxes: list[np.ndarray],
    texts: list[str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Returns a copy of img with boxes (and optionally text labels) drawn.
    """
    vis = img.copy()
    for i, box in enumerate(boxes):
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        if texts and i < len(texts) and texts[i]:
            origin = tuple(box[0].astype(int))
            cv2.putText(
                vis, texts[i], origin,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 0, 0), 1, cv2.LINE_AA,
            )
    return vis


# ── heatmap overlay ───────────────────────────────────────────────────────────

def overlay_heatmap(
    img: np.ndarray,
    score_map: np.ndarray,  # H×W float [0,1]
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a score heatmap on top of the image (character-level view)."""
    h, w = img.shape[:2]
    heat = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = (heat * 255).astype(np.uint8)
    heat_colored = cv2.applyColorMap(heat, colormap)
    heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 1 - alpha, heat_rgb, alpha, 0)
    return blended


# ── save helpers ──────────────────────────────────────────────────────────────

def save_visualization(
    img: np.ndarray,
    boxes: list[np.ndarray],
    texts: list[str] | None,
    out_path: str,
    region_map: np.ndarray | None = None,
) -> None:
    vis = draw_boxes(img, boxes, texts)
    if region_map is not None:
        vis = overlay_heatmap(vis, region_map, alpha=0.3)
    out = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out)
