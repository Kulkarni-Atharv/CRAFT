"""
Module — Multi-Frame Confidence Aggregator
Merges character predictions from multiple camera angles into one final string.

Voting strategies:
  confidence_weighted — sum classifier confidence per character across frames,
                        winner = highest total (accounts for frame quality)
  majority            — winner = most common prediction across frames
  max_confidence      — winner = single highest-confidence prediction

"No prediction" guarantee:
  If frames disagree beyond winner_threshold, output "" (blank).
  If a character appears in only 1 frame, a stricter gate applies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.char_classifier import ClassifierResult
from modules.char_segmenter  import CharSegment
from utils.logger import get_logger

log = get_logger(__name__)


# ── per-frame record ──────────────────────────────────────────────────────────

@dataclass
class FrameCharRecord:
    character:   str                  # predicted char, "" or "?"
    confidence:  float                # classifier softmax max
    craft_score: float                # CRAFT detection score
    position:    tuple[float, float]  # (cx, cy) in original image coords
    frame_id:    int


# ── aggregated result ─────────────────────────────────────────────────────────

@dataclass
class AggregatedChar:
    character:     str
    confidence:    float   # aggregated confidence (0–1)
    n_frames_seen: int     # how many frames contributed a non-blank prediction
    position:      tuple[float, float]


# ── aggregator ────────────────────────────────────────────────────────────────

class MultiFrameAggregator:
    """
    Collects per-frame character results and merges them into a final sequence.

    Typical usage:
        agg = MultiFrameAggregator(cfg)
        for frame_id, (segments, clf_results) in enumerate(frame_data):
            agg.add_frame(frame_id, segments, clf_results)
        final_text = agg.get_final_text()
    """

    def __init__(self, cfg: dict[str, Any]):
        vcfg = cfg["voting"]
        self.strategy:            str   = vcfg["strategy"]              # "confidence_weighted"
        self.position_tolerance:  float = vcfg["position_tolerance"]    # 20.0 px
        self.winner_threshold:    float = vcfg["winner_threshold"]      # 0.60
        self.single_frame_gate:   float = vcfg["single_frame_gate"]     # 0.80
        self.min_frames:          int   = vcfg["min_frames"]            # 2

        # frame_id → list[FrameCharRecord]
        self._frames: dict[int, list[FrameCharRecord]] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def add_frame(
        self,
        frame_id:    int,
        segments:    list[CharSegment],
        clf_results: list[ClassifierResult],
    ) -> None:
        records = []
        for seg, clf in zip(segments, clf_results):
            records.append(FrameCharRecord(
                character   = clf.character,
                confidence  = clf.confidence,
                craft_score = seg.craft_score,
                position    = seg.position,
                frame_id    = frame_id,
            ))
        self._frames[frame_id] = records
        log.debug("Aggregator: frame %d added (%d chars)", frame_id, len(records))

    def get_final_text(self) -> tuple[str, list[AggregatedChar]]:
        """
        Returns (final_string, list[AggregatedChar]) in reading order.
        Called after all frames have been added via add_frame().
        """
        if not self._frames:
            return "", []

        all_records = [r for records in self._frames.values() for r in records]

        # cluster all detections by position across frames
        clusters = self._cluster_by_position(all_records)

        # sort clusters left-to-right by average x position
        clusters.sort(key=lambda c: np.mean([r.position[0] for r in c]))

        aggregated: list[AggregatedChar] = []
        for cluster in clusters:
            agg_char = self._vote(cluster)
            aggregated.append(agg_char)

        final_text = "".join(a.character for a in aggregated)

        log.info(
            "Aggregator: %d frames  %d char slots  → %r",
            len(self._frames), len(aggregated), final_text,
        )
        return final_text, aggregated

    def reset(self) -> None:
        """Clear all frames — call between separate multi-frame captures."""
        self._frames.clear()

    # ── clustering ────────────────────────────────────────────────────────────

    def _cluster_by_position(
        self, records: list[FrameCharRecord]
    ) -> list[list[FrameCharRecord]]:
        """
        Greedy proximity clustering — each record joins the nearest existing
        cluster whose centroid is within position_tolerance pixels.
        Records from the same frame are never merged into the same cluster.
        """
        clusters: list[list[FrameCharRecord]] = []
        centroids: list[tuple[float, float]] = []

        for rec in records:
            best_idx  = -1
            best_dist = float("inf")

            for i, centroid in enumerate(centroids):
                dist = np.hypot(
                    rec.position[0] - centroid[0],
                    rec.position[1] - centroid[1],
                )
                if dist < self.position_tolerance and dist < best_dist:
                    # ensure this frame is not already in the cluster
                    if not any(r.frame_id == rec.frame_id for r in clusters[i]):
                        best_dist = dist
                        best_idx  = i

            if best_idx >= 0:
                clusters[best_idx].append(rec)
                # update centroid (running mean)
                cx = float(np.mean([r.position[0] for r in clusters[best_idx]]))
                cy = float(np.mean([r.position[1] for r in clusters[best_idx]]))
                centroids[best_idx] = (cx, cy)
            else:
                clusters.append([rec])
                centroids.append(rec.position)

        return clusters

    # ── voting ────────────────────────────────────────────────────────────────

    def _vote(self, cluster: list[FrameCharRecord]) -> AggregatedChar:
        """
        Given all records for one character slot, return the winning character.
        Blank ("") and unknown ("?") records are excluded from voting but
        reduce the winner's normalised share.
        """
        cx = float(np.mean([r.position[0] for r in cluster]))
        cy = float(np.mean([r.position[1] for r in cluster]))
        pos = (cx, cy)

        # only non-blank, non-unknown predictions participate in voting
        candidates = [r for r in cluster if r.character not in ("", "?")]
        n_frames   = len(self._frames)

        if not candidates:
            return AggregatedChar("", 0.0, 0, pos)

        if self.strategy == "confidence_weighted":
            char = self._vote_confidence_weighted(candidates, cluster, n_frames, pos)
        elif self.strategy == "majority":
            char = self._vote_majority(candidates, cluster, n_frames, pos)
        else:  # max_confidence
            char = self._vote_max_confidence(candidates, pos)

        return char

    def _vote_confidence_weighted(
        self,
        candidates: list[FrameCharRecord],
        cluster:    list[FrameCharRecord],
        n_frames:   int,
        pos:        tuple[float, float],
    ) -> AggregatedChar:
        tally: dict[str, float] = defaultdict(float)
        for r in candidates:
            tally[r.character] += r.confidence

        total    = sum(tally.values())
        winner   = max(tally, key=tally.__getitem__)
        w_conf   = tally[winner]
        # normalise by total votes cast — penalise frames that said blank/unknown
        w_share  = w_conf / (total + 1e-9)

        n_seen = len(candidates)

        # single-frame detection → apply stricter gate
        gate = (self.single_frame_gate
                if n_seen < self.min_frames
                else self.winner_threshold)

        if w_share < gate:
            return AggregatedChar("", 0.0, n_seen, pos)

        agg_conf = w_conf / max(n_frames, 1)
        return AggregatedChar(winner, float(agg_conf), n_seen, pos)

    def _vote_majority(
        self,
        candidates: list[FrameCharRecord],
        cluster:    list[FrameCharRecord],
        n_frames:   int,
        pos:        tuple[float, float],
    ) -> AggregatedChar:
        from collections import Counter
        counts  = Counter(r.character for r in candidates)
        winner, count = counts.most_common(1)[0]
        share   = count / max(n_frames, 1)
        n_seen  = len(candidates)

        gate = (self.single_frame_gate
                if n_seen < self.min_frames
                else self.winner_threshold)

        if share < gate:
            return AggregatedChar("", 0.0, n_seen, pos)

        avg_conf = float(np.mean([r.confidence for r in candidates
                                  if r.character == winner]))
        return AggregatedChar(winner, avg_conf, n_seen, pos)

    def _vote_max_confidence(
        self,
        candidates: list[FrameCharRecord],
        pos:        tuple[float, float],
    ) -> AggregatedChar:
        best = max(candidates, key=lambda r: r.confidence)
        gate = (self.single_frame_gate
                if len(candidates) < self.min_frames
                else self.winner_threshold)

        if best.confidence < gate:
            return AggregatedChar("", 0.0, len(candidates), pos)

        return AggregatedChar(best.character, best.confidence, len(candidates), pos)
