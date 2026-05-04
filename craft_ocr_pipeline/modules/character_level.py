"""
Bonus — Character-level output
Splits a word-level CRAFT region into individual character bounding boxes
by analysing the region score map peaks within each detected text region.

Usage
-----
    from modules.character_level import CharacterSplitter

    splitter = CharacterSplitter(cfg)
    char_boxes = splitter.split(img, word_box, region_map, scale)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
from scipy.signal import find_peaks

from utils.logger import get_logger

log = get_logger(__name__)


class CharacterSplitter:
    """
    For each word-level box, projects the region score vertically and
    finds character peaks to segment individual characters.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.text_threshold: float = cfg["craft"]["text_threshold"]
        self.low_text:       float = cfg["craft"]["low_text"]
        self.min_char_width: int   = 4   # px — ignore sub-pixel blobs

    def split(
        self,
        word_box:   np.ndarray,  # 4×2 corners in original image space
        region_map: np.ndarray,  # H×W float [0,1] — at score-map resolution
        scale:      float,       # preprocessing scale
    ) -> list[np.ndarray]:
        """
        Returns a list of 4×2 corner arrays, one per detected character.
        Falls back to the original word_box if segmentation fails.
        """
        # map word_box corners to score-map coordinates
        sm_box = (word_box * scale / 2.0).astype(np.float32)
        x1 = max(0, int(sm_box[:, 0].min()))
        y1 = max(0, int(sm_box[:, 1].min()))
        x2 = min(region_map.shape[1] - 1, int(sm_box[:, 0].max()))
        y2 = min(region_map.shape[0] - 1, int(sm_box[:, 1].max()))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return [word_box]

        patch = region_map[y1:y2, x1:x2]
        col_profile = patch.max(axis=0)  # vertical projection

        # find valleys (troughs between characters)
        inverted = 1.0 - col_profile
        peaks, _ = find_peaks(inverted, height=1.0 - self.low_text, distance=self.min_char_width)

        if len(peaks) < 1:
            return [word_box]  # can't split

        # boundaries: start, peaks (valleys), end
        boundaries = [0] + list(peaks) + [len(col_profile)]

        char_boxes = []
        coord_scale = (1.0 / scale) * 2.0  # back to original coords
        for i in range(len(boundaries) - 1):
            lx = (x1 + boundaries[i])   * coord_scale
            rx = (x1 + boundaries[i+1]) * coord_scale
            ty = y1 * coord_scale
            by = y2 * coord_scale
            if rx - lx < self.min_char_width * coord_scale:
                continue
            char_box = np.array(
                [[lx, ty], [rx, ty], [rx, by], [lx, by]], dtype=np.float32
            )
            char_boxes.append(char_box)

        return char_boxes if char_boxes else [word_box]
