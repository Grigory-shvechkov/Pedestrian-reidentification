"""ByteTrack-style multi-person tracker.

Improvements over greedy IoU matching:
  * Kalman filter (constant velocity in cx, cy, aspect, height) predicts where
    each person will be, so fast walkers and skipped frames still match.
  * Globally optimal assignment (Hungarian) instead of greedy per-track matching.
  * Two-stage association: low-confidence detections (partially occluded people)
    can keep an existing track alive but never create a new one. This is the main
    reason ByteTrack fragments tracks far less than plain IoU trackers — and every
    fragment was previously a fresh re-ID decision and a possible duplicate visitor.
  * Tentative → confirmed lifecycle: one-off false positives never get an ID.
  * Third stage matches by centre distance when boxes no longer overlap enough –
    people close to the camera move a lot between detections and IoU alone
    would split them into two tracks (= two "different" visitors).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


# ─── Box helpers ─────────────────────────────────────────────────────────────
def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], np.float64)


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


# ─── Kalman filter (DeepSORT parameterisation) ───────────────────────────────
class KalmanBox:
    _W_POS = 1.0 / 20
    _W_VEL = 1.0 / 160
    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)

    def __init__(self, box):
        m = xyxy_to_xyah(box)
        self.mean = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self._W_POS * h, 2 * self._W_POS * h, 1e-2, 2 * self._W_POS * h,
               10 * self._W_VEL * h, 10 * self._W_VEL * h, 1e-5, 10 * self._W_VEL * h]
        self.cov = np.diag(np.square(std))

    def predict(self):
        h = self.mean[3]
        q = [self._W_POS * h, self._W_POS * h, 1e-2, self._W_POS * h,
             self._W_VEL * h, self._W_VEL * h, 1e-5, self._W_VEL * h]
        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + np.diag(np.square(q))
        self.mean[3] = max(self.mean[3], 1.0)

    def update(self, box):
        z = xyxy_to_xyah(box)
        h = self.mean[3]
        r = [self._W_POS * h, self._W_POS * h, 1e-1, self._W_POS * h]
        S = self._H @ self.cov @ self._H.T + np.diag(np.square(r))
        K = np.linalg.solve(S, (self.cov @ self._H.T).T).T
        self.mean = self.mean + K @ (z - self._H @ self.mean)
        self.cov = (np.eye(8) - K @ self._H) @ self.cov

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)


# ─── Track ───────────────────────────────────────────────────────────────────
class Track:
    TENTATIVE, CONFIRMED = "tentative", "confirmed"

    def __init__(self, track_id: int, det: Detection, now: float):
        self.id = track_id
        self.kf = KalmanBox(det.box)
        self.det = det                 # last matched detection
        self.state = self.TENTATIVE
        self.hits = 1
        self.time_since_update = 0     # detection cycles
        self.first_seen = now
        self.last_seen = now

    @property
    def box(self):
        """Last measured box while visible, predicted box while lost."""
        return self.det.box if self.time_since_update == 0 else self.kf.box

    @property
    def confirmed(self):
        return self.state == self.CONFIRMED

    def mark_matched(self, det: Detection, now: float, min_hits: int):
        self.kf.update(det.box)
        self.det = det
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = now
        if self.state == self.TENTATIVE and self.hits >= min_hits:
            self.state = self.CONFIRMED


class Tracker:
    def __init__(self, high_conf=0.55, match_iou=0.3, low_match_iou=0.5, min_hits=3, max_age=10,
                 max_center_dist=0.6):
        self.high_conf = high_conf
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_center_dist = max_center_dist
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    @staticmethod
    def _assign(tracks, dets, min_iou):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        ious = iou_matrix(np.stack([t.kf.box for t in tracks]), np.stack([d.box for d in dets]))
        rows, cols = linear_sum_assignment(-ious)
        matches, mt, md = [], set(), set()
        for r, c in zip(rows, cols):
            if ious[r, c] >= min_iou:
                matches.append((r, c))
                mt.add(r)
                md.add(c)
        return (matches,
                [i for i in range(len(tracks)) if i not in mt],
                [j for j in range(len(dets)) if j not in md])

    def _assign_center(self, tracks, dets):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb = np.stack([t.kf.box for t in tracks]).astype(np.float64)
        db = np.stack([d.box for d in dets]).astype(np.float64)
        tc = (tb[:, :2] + tb[:, 2:]) / 2
        dc = (db[:, :2] + db[:, 2:]) / 2
        th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
        dh = np.maximum(db[:, 3] - db[:, 1], 1.0)
        dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / th[:, None]
        ratio = dh[None, :] / th[:, None]
        cost = np.where((ratio > 0.6) & (ratio < 1.67), dist, 1e6)
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= self.max_center_dist]
        mt, md = {r for r, _ in matches}, {c for _, c in matches}
        return (matches, [i for i in range(len(tracks)) if i not in mt],
                [j for j in range(len(dets)) if j not in md])

    def update(self, dets: list[Detection], now: float) -> tuple[list[Track], list[Track]]:
        """Advance one detection cycle. Returns (live_tracks, removed_tracks)."""
        for t in self.tracks:
            t.kf.predict()

        high = [d for d in dets if d.score >= self.high_conf]
        low = [d for d in dets if d.score < self.high_conf]

        # Stage 1: all tracks ↔ high-confidence detections
        m1, um_t, um_h = self._assign(self.tracks, high, self.match_iou)
        for ti, di in m1:
            self.tracks[ti].mark_matched(high[di], now, self.min_hits)

        # Stage 2: remaining *confirmed* tracks ↔ low-confidence detections
        rest = [self.tracks[i] for i in um_t if self.tracks[i].confirmed]
        m2, um_rest, _ = self._assign(rest, low, self.low_match_iou)
        for ti, di in m2:
            rest[ti].mark_matched(low[di], now, self.min_hits)

        matched_ids = {self.tracks[ti].id for ti, _ in m1} | {rest[ti].id for ti, _ in m2}

        # Stage 3: still-unmatched recent tracks ↔ leftover high detections, by centre distance
        left_t = [t for t in self.tracks if t.id not in matched_ids and t.time_since_update <= 3]
        left_d = [high[j] for j in um_h]
        m3, _, um3 = self._assign_center(left_t, left_d)
        for ti, di in m3:
            left_t[ti].mark_matched(left_d[di], now, self.min_hits)
            matched_ids.add(left_t[ti].id)
        um_h = [um_h[j] for j in um3]
        removed, alive = [], []
        for t in self.tracks:
            if t.id not in matched_ids:
                t.time_since_update += 1
                # Tentative tracks die on their first miss; confirmed ones coast up to max_age.
                if t.state == Track.TENTATIVE or t.time_since_update > self.max_age:
                    removed.append(t)
                    continue
            alive.append(t)

        for di in um_h:
            alive.append(Track(next(self._ids), high[di], now))

        self.tracks = alive
        return alive, removed
