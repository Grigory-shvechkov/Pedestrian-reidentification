"""Frame annotation: boxes, per-track status, zone outline and the RECOGNISED banner."""
from __future__ import annotations

import cv2
import numpy as np

# BGR colours per status
COLORS = {
    "tentative": (160, 160, 160),
    "detecting": (0, 200, 255),     # waiting for good crops
    "collecting": (0, 220, 220),    # gathering samples
    "verifying": (255, 160, 0),     # ambiguous, gathering more
    "new": (60, 200, 60),
    "recognised": (255, 80, 200),
    "known": (200, 200, 60),        # same visitor, same visit (track fragment)
    "lost": (110, 110, 110),
}


def _label(img, text, x, y, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - base - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw(frame: np.ndarray, items: list[dict], zone: np.ndarray | None, draw_zone: bool,
         banner: dict | None, fps: float, backend: str) -> np.ndarray:
    """items: [{"box", "status", "text"}]. Draws in place and returns frame."""
    if zone is not None and draw_zone:
        cv2.polylines(frame, [zone.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["box"])
        color = COLORS.get(it["status"], (255, 255, 255))
        thick = 1 if it["status"] in ("tentative", "lost") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        _label(frame, it["text"], x1, y1 - 2, color)

    if banner:
        h, w = frame.shape[:2]
        text = f"SEEN BEFORE  #{banner['profile_id']}  {banner['similarity'] * 100:.0f}%  (visit {banner['visits']})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        x = (w - tw) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 16, 12), (x + tw + 16, 12 + th + 24), (255, 80, 200), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, text, (x, 12 + th + 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:4.1f} fps | {backend}", (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
