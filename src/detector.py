"""YOLOv8 person detector.

Returns detections down to a *low* confidence so the tracker can use them to
keep existing tracks alive (ByteTrack); only high-confidence ones start tracks.
With a *-seg model, each detection also carries a binary person mask aligned
with its integer box, used to keep background out of the colour descriptor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: np.ndarray                 # float32 [x1, y1, x2, y2], full-frame coords
    score: float
    ibox: tuple                     # int (x1, y1, x2, y2), clipped to frame; crop = frame[y1:y2, x1:x2]
    mask: np.ndarray | None = None  # uint8 {0,1}, shape == crop shape, or None


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def int_box(box, w: int, h: int) -> tuple:
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


class PersonDetector:
    def __init__(self, weights: str, low_conf: float = 0.25, imgsz: int = 640, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.is_seg = getattr(self.model, "task", "detect") == "segment"
        self.low_conf = low_conf
        self.imgsz = imgsz
        self.device = resolve_device(device)
        log.info("Detector %s on %s (seg=%s)", weights, self.device, self.is_seg)
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), classes=[0], verbose=False,
                           device=self.device, imgsz=imgsz)  # warm-up

    def detect(self, frame: np.ndarray, roi: tuple | None = None) -> list[Detection]:
        H, W = frame.shape[:2]
        x0 = y0 = 0
        img = frame
        if roi is not None:
            x0, y0, x1, y1 = roi
            img = frame[y0:y1, x0:x1]
        if img.size == 0:
            return []

        res = self.model.predict(
            img, classes=[0], conf=self.low_conf, imgsz=self.imgsz, device=self.device,
            verbose=False, retina_masks=self.is_seg,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy()
        masks = None
        if self.is_seg and res.masks is not None:
            masks = res.masks.data.cpu().numpy()  # (N, h, w) at ROI resolution (retina_masks)

        out = []
        offset = np.array([x0, y0, x0, y0], np.float32)
        for i, (b, s) in enumerate(zip(boxes, scores)):
            fb = b + offset
            ib = int_box(fb, W, H)
            if ib[2] - ib[0] < 2 or ib[3] - ib[1] < 2:
                continue
            m = None
            if masks is not None and masks.shape[1:] == img.shape[:2]:
                lx1, ly1, lx2, ly2 = ib[0] - x0, ib[1] - y0, ib[2] - x0, ib[3] - y0
                m = (masks[i, ly1:ly2, lx1:lx2] > 0.5).astype(np.uint8)
                if m.shape != (ib[3] - ib[1], ib[2] - ib[0]):
                    m = None
            out.append(Detection(fb, float(s), ib, m))
        return out
