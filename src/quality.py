"""Crop quality gates.

Bad crops are the #1 cause of re-ID errors: a half-visible person, a blurred
frame or two overlapping people produce embeddings that match nobody (→ duplicate
visitor) or everybody (→ wrong merge). Every crop passes these checks before it
is embedded, and survivors get a quality score used to weight aggregation.
"""
from __future__ import annotations

import cv2
import numpy as np


def sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian on a fixed-height grey crop (roughly scale-invariant)."""
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return 0.0
    scale = 128.0 / h
    g = cv2.cvtColor(cv2.resize(crop, (max(4, int(w * scale)), 128)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def occlusion(box: np.ndarray, others: list[np.ndarray]) -> float:
    """Largest fraction of `box` covered by any other box (intersection / own area)."""
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    worst = 0.0
    for o in others:
        iw = max(0.0, min(box[2], o[2]) - max(box[0], o[0]))
        ih = max(0.0, min(box[3], o[3]) - max(box[1], o[1]))
        worst = max(worst, iw * ih / area)
    return worst


def assess(det, crop, others, bounds, cfg, relaxed: bool = False) -> tuple[bool, str, float]:
    """Return (ok, reason, quality in 0..1).

    bounds: (x1, y1, x2, y2) of the region YOLO actually saw (ROI or full frame);
    a box touching it is probably cut off.
    relaxed: skip the edge and blur checks (used when a person has produced no
    usable crop for a while, so nobody walks through uncounted).
    """
    x1, y1, x2, y2 = det.box
    w, h = x2 - x1, y2 - y1
    if w * h < cfg.MIN_BBOX_AREA:
        return False, "small", 0.0
    if h < cfg.MIN_PERSON_HEIGHT:
        return False, "short", 0.0
    if w / max(h, 1) > cfg.MAX_ASPECT_RATIO:
        return False, "shape", 0.0

    bx1, by1, bx2, by2 = bounds
    m = cfg.EDGE_MARGIN
    touching = {"left": x1 <= bx1 + m, "top": y1 <= by1 + m, "right": x2 >= bx2 - m, "bottom": y2 >= by2 - m}
    if not relaxed and any(touching[s] for s in cfg.EDGE_SIDES):
        return False, "edge", 0.0

    occ = occlusion(det.box, others)
    if occ > cfg.MAX_OCCLUSION:
        return False, "occluded", 0.0

    sharp = sharpness(crop)
    if not relaxed and sharp < cfg.MIN_SHARPNESS:
        return False, "blurry", 0.0

    # Quality score: detector confidence × size × sharpness × visibility, each saturating.
    q = (
        min(1.0, det.score / 0.9)
        * min(1.0, h / 256.0) ** 0.5
        * min(1.0, sharp / (4 * cfg.MIN_SHARPNESS)) ** 0.5
        * (1.0 - occ)
    )
    return True, "ok", float(max(q, 1e-3))
