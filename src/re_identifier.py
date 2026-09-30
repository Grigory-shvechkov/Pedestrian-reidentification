"""Re-identification against an in-memory, multi-prototype gallery.

Changes vs. single-embedding cosine matching:
  * Each visitor keeps up to MAX_PROTOTYPES embeddings (front / back / jacket
    on / off...). Score = max similarity over the visitor's prototypes.
    The first (anchor) prototype is never evicted, which limits identity drift.
  * New prototypes are only added from confident matches and only when they are
    novel (not near-duplicates of what is already stored).
  * Three-way decision instead of a single threshold:
        best ≥ thr and clear margin over 2nd best  → match
        best within AMBIGUOUS_BAND below thr, or two profiles tied → pending
                                                     (pipeline collects more crops)
        otherwise                                  → new
    On a *final* decision (timeout / track ended) pending collapses to match/new.
  * Mutual exclusion: a profile that is currently assigned to another visible
    track cannot be matched (two people in frame can't be the same visitor).
  * Visit counting uses the gap since the profile was last *seen* (updated every
    cycle while tracked), not since its last decision, so track fragments of the
    same pass-by never count as extra visits or trigger "RECOGNISED".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Match:
    profile_id: int | None
    similarity: float
    second: float


@dataclass
class Decision:
    kind: str                 # "match" | "new" | "pending"
    profile_id: int | None
    similarity: float


class ReIdentifier:
    def __init__(self, db, dim: int, threshold: float, cfg):
        self.db = db
        self.dim = dim
        self.thr = threshold
        self.band = cfg.AMBIGUOUS_BAND
        self.min_margin = cfg.MIN_MARGIN
        self.revisit_gap = cfg.REVISIT_GAP_S
        self.max_protos = cfg.MAX_PROTOTYPES
        self.novelty = cfg.PROTOTYPE_NOVELTY
        self.update_margin = cfg.PROTOTYPE_UPDATE_MIN_SIM_MARGIN

        # visitor_id → {"last_seen": float, "protos": [(proto_id, emb, is_anchor)]}
        self.profiles: dict[int, dict] = {}
        for vid, last_seen, protos in db.load_gallery():
            protos = [p for p in protos if p[1].shape[0] == dim]
            if protos:
                self.profiles[vid] = {"last_seen": last_seen, "protos": protos}
        self._rebuild()
        log.info("Gallery: %d visitors, %d prototypes", len(self.profiles), len(self._owner))

    def _rebuild(self):
        rows, owner = [], []
        for vid, p in self.profiles.items():
            for _, emb, _ in p["protos"]:
                rows.append(emb)
                owner.append(vid)
        self._matrix = np.stack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), np.float32)
        self._owner = np.array(owner, np.int64)
        self._ids = np.array(sorted(self.profiles), np.int64)
        self._owner_idx = np.searchsorted(self._ids, self._owner) if len(self._owner) else self._owner

    # ── querying ──
    def match(self, emb: np.ndarray, exclude=()) -> Match:
        if len(self._ids) == 0:
            return Match(None, -1.0, -1.0)
        sims = self._matrix @ emb
        per = np.full(len(self._ids), -1.0, np.float32)
        np.maximum.at(per, self._owner_idx, sims)
        if exclude:
            per[np.isin(self._ids, list(exclude))] = -1.0
        order = np.argsort(-per)
        best = order[0]
        second = float(per[order[1]]) if len(order) > 1 else -1.0
        if per[best] <= -1.0:
            return Match(None, -1.0, -1.0)
        return Match(int(self._ids[best]), float(per[best]), second)

    def decide(self, emb: np.ndarray, exclude=(), final: bool = False) -> Decision:
        m = self.match(emb, exclude)
        if m.profile_id is None:
            return Decision("new", None, m.similarity)
        clear = (m.similarity - m.second) >= self.min_margin or m.second < self.thr
        if m.similarity >= self.thr and (clear or final):
            return Decision("match", m.profile_id, m.similarity)
        if not final and (m.similarity >= self.thr - self.band or (m.similarity >= self.thr and not clear)):
            return Decision("pending", m.profile_id, m.similarity)
        return Decision("new", None, m.similarity)

    # ── committing ──
    def commit_new(self, emb: np.ndarray, quality: float, now: float, track_id: int,
                   thumb: bytes | None = None) -> int:
        vid, proto_id = self.db.create_visitor(emb, quality, now, track_id, thumb)
        self.profiles[vid] = {"last_seen": now, "protos": [(proto_id, emb.copy(), True)]}
        self._rebuild()
        return vid

    def commit_match(self, vid: int, emb: np.ndarray, sim: float, quality: float,
                     now: float, track_id: int) -> bool:
        """Record a match. Returns True if it counts as a new visit."""
        prof = self.profiles[vid]
        new_visit = (now - prof["last_seen"]) >= self.revisit_gap
        prof["last_seen"] = now
        self.db.record_visit(vid, now, track_id, sim, new_visit)
        if sim >= self.thr + self.update_margin:
            self._maybe_add_prototype(vid, emb, quality, now)
        return new_visit

    def _maybe_add_prototype(self, vid, emb, quality, now):
        protos = self.profiles[vid]["protos"]
        existing = np.stack([p[1] for p in protos])
        if float((existing @ emb).max()) >= self.novelty:
            return  # nothing new about this view
        if len(protos) >= self.max_protos:
            # Evict the most redundant non-anchor prototype (highest similarity to the rest).
            sims = existing @ existing.T
            np.fill_diagonal(sims, -1)
            redundancy = sims.max(1)
            candidates = [i for i, p in enumerate(protos) if not p[2]]
            if not candidates:
                return
            evict = max(candidates, key=lambda i: redundancy[i])
            self.db.delete_prototype(protos[evict][0])
            protos.pop(evict)
        proto_id = self.db.add_prototype(vid, emb, quality, now)
        protos.append((proto_id, emb.copy(), False))
        self._rebuild()

    def touch(self, vid: int, now: float):
        """Called every cycle while a decided track is visible."""
        if vid in self.profiles:
            self.profiles[vid]["last_seen"] = now

    def flush(self, vid: int):
        if vid in self.profiles:
            self.db.touch(vid, self.profiles[vid]["last_seen"])
