"""
Module — Character Integrity Verifier
Runs 6 quality checks on each CharSegment crop before OCR is attempted.

Status returned per character:
  VERIFIED  — all checks passed, send to classifier
  BLANK     — character clearly absent or invisible, output ""
  UNKNOWN   — character present but unreadable, output "?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from modules.char_segmenter import CharSegment
from utils.logger import get_logger

log = get_logger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    status:        str    # "VERIFIED" | "BLANK" | "UNKNOWN"
    reason:        str    # "" | "low_craft_score" | "blur" | "low_visibility" |
                          #     "no_ink" | "debris" | "broken_stroke" | "partial"
    quality_score: float  # 0.0–1.0  (1.0 = perfect, 0.0 = completely unusable)


# ── Verifier ──────────────────────────────────────────────────────────────────

class CharVerifier:
    """
    Six-check integrity gate.  All checks must pass for a crop to reach OCR.

    Check 1 — CRAFT score gate      : detection confidence (from region map)
    Check 2 — Blur detection        : Laplacian variance on grayscale crop
    Check 3 — Contrast / visibility : foreground–background pixel range
    Check 4 — Ink coverage          : fraction of binarised crop that is "ink"
                                      too low  → no character present
                                      too high → debris covering the character
    Check 5 — Broken stroke         : connected-component count in ink mask
                                      too many fragments → damaged / occluded
    Check 6 — Partial character     : ink touching a crop edge → character cut off
    """

    def __init__(self, cfg: dict[str, Any]):
        vcfg = cfg["verification"]
        self.craft_score_gate:      float = vcfg["craft_score_gate"]       # 0.62
        self.blur_threshold:        float = vcfg["blur_threshold"]         # 80.0
        self.min_contrast:          float = vcfg["min_contrast"]           # 30.0
        self.min_ink_coverage:      float = vcfg["min_ink_coverage"]       # 0.05
        self.max_ink_coverage:      float = vcfg["max_ink_coverage"]       # 0.90
        self.max_components:        int   = vcfg["max_components"]         # 6
        self.edge_ink_threshold:    float = vcfg["edge_ink_threshold"]     # 0.30

    # ── public API ────────────────────────────────────────────────────────────

    def verify(self, segment: CharSegment) -> VerificationResult:
        """Run all checks on one CharSegment. Returns on first failure."""

        # ── Check 1: CRAFT detection confidence ──────────────────────────────
        if segment.craft_score < self.craft_score_gate:
            return VerificationResult(
                "BLANK", "low_craft_score",
                segment.craft_score / self.craft_score_gate,
            )

        crop  = segment.crop
        gray  = self._to_gray(crop)

        # ── Check 2: blur ─────────────────────────────────────────────────────
        blur_score = self._blur_score(gray)
        if blur_score < self.blur_threshold:
            return VerificationResult(
                "BLANK", "blur",
                blur_score / self.blur_threshold,
            )

        # ── Check 3: contrast / visibility ───────────────────────────────────
        contrast = float(gray.max()) - float(gray.min())
        if contrast < self.min_contrast:
            return VerificationResult(
                "BLANK", "low_visibility",
                contrast / self.min_contrast,
            )

        # ── Binarise once — used by checks 4, 5, 6 ───────────────────────────
        ink_mask = self._binarise(gray)    # 0 = background, 255 = ink
        ink_ratio = float(ink_mask.mean()) / 255.0

        # ── Check 4a: too little ink → no character ───────────────────────────
        if ink_ratio < self.min_ink_coverage:
            return VerificationResult("BLANK", "no_ink", ink_ratio)

        # ── Check 4b: too much ink → debris covering character ────────────────
        if ink_ratio > self.max_ink_coverage:
            return VerificationResult(
                "UNKNOWN", "debris",
                1.0 - (ink_ratio - self.max_ink_coverage) / (1.0 - self.max_ink_coverage),
            )

        # ── Check 5: broken strokes ───────────────────────────────────────────
        n_components = self._count_components(ink_mask)
        if n_components > self.max_components:
            return VerificationResult(
                "UNKNOWN", "broken_stroke",
                self.max_components / max(n_components, 1),
            )

        # ── Check 6: partial character (ink touching crop edge) ───────────────
        partial_score = self._edge_ink_ratio(ink_mask)
        if partial_score > self.edge_ink_threshold:
            return VerificationResult(
                "UNKNOWN", "partial",
                1.0 - partial_score,
            )

        # all checks passed
        quality = self._composite_quality(blur_score, contrast, ink_ratio, n_components)
        return VerificationResult("VERIFIED", "", quality)

    def verify_batch(
        self, segments: list[CharSegment]
    ) -> list[VerificationResult]:
        results = [self.verify(s) for s in segments]

        n_verified = sum(1 for r in results if r.status == "VERIFIED")
        n_blank    = sum(1 for r in results if r.status == "BLANK")
        n_unknown  = sum(1 for r in results if r.status == "UNKNOWN")
        log.info(
            "Verifier: %d VERIFIED  %d BLANK  %d UNKNOWN  (of %d)",
            n_verified, n_blank, n_unknown, len(segments),
        )

        # log rejection reasons at debug level
        reasons: dict[str, int] = {}
        for r in results:
            if r.reason:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if reasons:
            log.debug("Rejection reasons: %s", reasons)

        return results

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            return crop
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _binarise(gray: np.ndarray) -> np.ndarray:
        """
        Otsu threshold — works on both dark-on-light and light-on-dark text.
        Always returns ink=255 (foreground), background=0.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # pick whichever polarity has less ink (text pixels < background pixels)
        if binary.mean() > 127:
            binary = cv2.bitwise_not(binary)
        return binary

    @staticmethod
    def _count_components(ink_mask: np.ndarray) -> int:
        """Number of connected ink components (background excluded)."""
        _, labels = cv2.connectedComponents(ink_mask, connectivity=8)
        return int(labels.max())   # label 0 = background

    @staticmethod
    def _edge_ink_ratio(ink_mask: np.ndarray) -> float:
        """
        Maximum ink fraction along any single edge of the crop.
        High value → character is being cut off at the boundary.
        """
        top    = float(ink_mask[0,  :].mean())  / 255.0
        bottom = float(ink_mask[-1, :].mean())  / 255.0
        left   = float(ink_mask[:,  0].mean())  / 255.0
        right  = float(ink_mask[:, -1].mean())  / 255.0
        return max(top, bottom, left, right)

    def _composite_quality(
        self,
        blur_score:   float,
        contrast:     float,
        ink_ratio:    float,
        n_components: int,
    ) -> float:
        """
        0–1 quality estimate used for multi-frame confidence voting.
        Higher = better quality crop.
        """
        blur_q     = min(blur_score   / (self.blur_threshold * 3), 1.0)
        contrast_q = min(contrast     / 255.0, 1.0)
        ink_q      = 1.0 - abs(ink_ratio - 0.35) / 0.35   # ideal ink ≈ 35%
        comp_q     = max(0.0, 1.0 - (n_components - 1) / self.max_components)
        return float(np.mean([blur_q, contrast_q, max(ink_q, 0.0), comp_q]))
