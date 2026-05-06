"""
Module — Character Segmenter
Replaces postprocessing.py + cropping.py for the character-level pipeline.

Three-step process:
  A. Line grouping   — OR(affinity, region) → connected components → text lines
  B. Char extraction — region map within each line → individual character blobs
  C. Deskew + normalise — perspective warp to upright, CLAHE, resize to char_size²

Returns list[CharSegment] in reading order (top-to-bottom line, left-to-right char).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label

from utils.image_utils import order_points
from utils.logger import get_logger

log = get_logger(__name__)


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class CharSegment:
    """One isolated character extracted from the image."""
    box:         np.ndarray           # 4×2 float32 — corners in original image coords
    crop:        np.ndarray           # char_size×char_size RGB, deskewed + CLAHE
    craft_score: float                # max CRAFT region score inside the blob
    position:    tuple[float, float]  # (cx, cy) centre in original image coords
    line_id:     int                  # text-line index (0 = topmost)


# ── Main class ────────────────────────────────────────────────────────────────

class CharSegmenter:
    """
    CRAFT region map + affinity map → list[CharSegment].

    Affinity map:  groups adjacent character blobs into the same text line.
    Region map:    isolates individual character blobs within each line.
    """

    def __init__(self, cfg: dict[str, Any]):
        scfg = cfg["segmentation"]
        self.char_threshold:      float = scfg["char_threshold"]        # 0.55
        self.line_link_threshold: float = scfg["line_link_threshold"]   # 0.50
        self.craft_score_gate:    float = scfg["craft_score_gate"]      # 0.62
        self.min_char_area:       int   = scfg["min_char_area"]         # 15
        self.char_size:           int   = scfg["char_size"]             # 32
        self.save_crops:          bool  = cfg["pipeline"]["save_crops"]
        self.crops_dir:           str   = cfg["paths"]["crops_dir"]

    # ── public API ────────────────────────────────────────────────────────────

    def segment(
        self,
        img:          np.ndarray,        # H×W×3 RGB original image
        region_map:   np.ndarray,        # H'×W' float32 [0,1] CRAFT region scores
        affinity_map: np.ndarray,        # H'×W' float32 [0,1] CRAFT affinity scores
        scale:        float,             # preprocessing resize scale
        orig_size:    tuple[int, int],   # (orig_h, orig_w)
        prefix:       str = "frame",     # filename prefix when save_crops is True
    ) -> list[CharSegment]:
        """
        Returns CharSegments sorted in reading order.
        Blobs below craft_score_gate are included — the Verifier applies that gate.
        """
        # CRAFT outputs at half the tensor spatial resolution
        coord_scale = 2.0 / scale
        orig_h, orig_w = orig_size

        # ── A: group blobs into text lines ───────────────────────────────────
        line_masks = self._group_lines(affinity_map, region_map)
        log.info("CharSegmenter: %d text line(s) from affinity grouping", len(line_masks))

        segments: list[CharSegment] = []

        for line_id, line_mask in enumerate(line_masks):

            # ── B: isolate individual character blobs ─────────────────────
            char_blobs = self._extract_char_blobs(region_map, line_mask)
            log.debug("  line %d → %d blob(s)", line_id, len(char_blobs))

            for blob_mask in char_blobs:
                craft_score = float((region_map * blob_mask).max())

                # pixel positions in map space → original image space
                ys, xs = np.where(blob_mask > 0)
                if len(xs) < 4:
                    continue

                pts_orig = np.stack(
                    [xs * coord_scale, ys * coord_scale], axis=1
                ).astype(np.float32)

                cx = float(pts_orig[:, 0].mean())
                cy = float(pts_orig[:, 1].mean())

                # ── C: deskew + CLAHE + resize ────────────────────────────
                box, crop = self._deskew_crop(img, pts_orig, orig_w, orig_h)
                if crop is None:
                    continue

                seg = CharSegment(
                    box=box,
                    crop=crop,
                    craft_score=craft_score,
                    position=(cx, cy),
                    line_id=line_id,
                )
                segments.append(seg)

                if self.save_crops:
                    self._save_crop(crop, prefix, len(segments) - 1, craft_score)

        # reading order: line first, then left-to-right within each line
        segments.sort(key=lambda s: (s.line_id, s.position[0]))

        log.info(
            "CharSegmenter: %d segments extracted  "
            "(craft_gate=%.2f applied by Verifier, not here)",
            len(segments), self.craft_score_gate,
        )
        return segments

    # ── A: text line grouping ─────────────────────────────────────────────────

    def _group_lines(
        self,
        affinity_map: np.ndarray,
        region_map:   np.ndarray,
    ) -> list[np.ndarray]:
        """
        OR affinity and region maps, threshold, connected-components.
        Each component is one text line (or one word group on the same line).
        Components with no region pixels are discarded (pure affinity noise).
        """
        binary_link   = (affinity_map > self.line_link_threshold).astype(np.uint8)
        binary_region = (region_map   > self.char_threshold).astype(np.uint8)
        combined      = np.clip(binary_link + binary_region, 0, 1)

        labelled, n = scipy_label(combined)

        masks: list[np.ndarray] = []
        for comp_id in range(1, n + 1):
            mask = (labelled == comp_id).astype(np.uint8)
            # only keep components that contain actual character pixels
            if (mask * binary_region).any():
                masks.append(mask)

        # sort top-to-bottom by centroid row
        masks.sort(key=lambda m: float(np.where(m > 0)[0].mean()))
        return masks

    # ── B: character blob extraction ──────────────────────────────────────────

    def _extract_char_blobs(
        self,
        region_map: np.ndarray,
        line_mask:  np.ndarray,
    ) -> list[np.ndarray]:
        """
        Threshold the region map restricted to one line mask.
        Each connected component is one character blob.
        """
        line_region = region_map * line_mask
        binary      = (line_region > self.char_threshold).astype(np.uint8)

        labelled, n = scipy_label(binary)

        blobs: list[np.ndarray] = []
        for comp_id in range(1, n + 1):
            mask = (labelled == comp_id).astype(np.uint8)
            if int(mask.sum()) >= self.min_char_area:
                blobs.append(mask)
        return blobs

    # ── C: deskew crop ────────────────────────────────────────────────────────

    def _deskew_crop(
        self,
        img:    np.ndarray,   # full original RGB image
        pts:    np.ndarray,   # N×2 float32 blob pixel positions in orig coords
        orig_w: int,
        orig_h: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        1. Fit minAreaRect to blob pixel positions.
        2. Order corners canonically (top-left → top-right → bottom-right → bottom-left).
        3. Ensure crop is landscape (wider than tall) — swap if needed.
        4. getPerspectiveTransform → warpPerspective.
        5. CLAHE on the LAB L-channel.
        6. Resize to char_size × char_size.

        Returns (box_4pts_in_orig_coords, normalised_crop).
        """
        rect  = cv2.minAreaRect(pts.reshape(-1, 1, 2))
        box   = cv2.boxPoints(rect).astype(np.float32)   # 4×2

        # clamp to image boundary
        box[:, 0] = np.clip(box[:, 0], 0, orig_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, orig_h - 1)

        src = order_points(box)        # tl, tr, br, bl
        tl, tr, br, bl = src

        # measure the two side lengths
        w = float(max(
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl),
        ))
        h = float(max(
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr),
        ))

        if w < 1 or h < 1:
            return box, None

        # canonical orientation: wider side = horizontal (text reads left-to-right)
        # if the box is taller than wide, rotate source corners 90° clockwise
        if h > w:
            src  = np.array([bl, tl, tr, br], dtype=np.float32)
            w, h = h, w

        out_w = max(int(round(w)), 1)
        out_h = max(int(round(h)), 1)

        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        M    = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(
            img, M, (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # CLAHE — normalise contrast so angle/lighting changes don't shift pixel values
        crop = _apply_clahe(crop)

        # fixed square output so the classifier always sees the same input shape
        crop = cv2.resize(
            crop,
            (self.char_size, self.char_size),
            interpolation=cv2.INTER_CUBIC,
        )

        return box, crop

    # ── save helper ───────────────────────────────────────────────────────────

    def _save_crop(
        self,
        crop:  np.ndarray,
        prefix: str,
        idx:   int,
        score: float,
    ) -> None:
        from pathlib import Path
        out = Path(self.crops_dir) / f"{prefix}_seg{idx:04d}_s{score:.2f}.png"
        cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


# ── CLAHE helper (module-level, shared) ──────────────────────────────────────

def _apply_clahe(crop: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on the L-channel (LAB space) to normalise brightness and
    local contrast without shifting colour.  Handles both RGB and grayscale.
    """
    if crop.size == 0:
        return crop

    if crop.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(crop)

    lab      = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l        = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
