"""Build a test video for "seen before?" from a free sample clip.

Source: one-by-one-person-detection.mp4 from github.com/intel-iot-devkit/sample-videos
(CC BY 4.0, © Intel Corporation). Six different people walk up to a table, one
at a time.

The output plays the clip twice:
  pass 1 – original                      → expect 6 × NEW
  pass 2 – mirrored, warmer/darker light,
           slightly zoomed                → expect 6 × SEEN BEFORE (visitors #1–#6)
Pass 2 is altered so it isn't pixel-identical to pass 1, which would be too easy.

    python tools/make_test_video.py                 # → videos/test_reid.mp4
    python main.py --source videos/test_reid.mp4
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.videoio import finalize_mp4  # noqa: E402

URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4"
SRC = ROOT / "data" / "one-by-one-person-detection.mp4"
OUT = ROOT / "videos" / "test_reid.mp4"
SIZE = (960, 540)       # upscaled so people clear MIN_PERSON_HEIGHT / MIN_BBOX_AREA
GAP_S = 3.0


def relight(frame: np.ndarray) -> np.ndarray:
    """Mirror, darken a little, warm colour cast, 6 % zoom."""
    f = cv2.flip(frame, 1).astype(np.float32)
    f = 255.0 * (f / 255.0) ** 1.15                       # gamma → slightly darker
    f *= np.array([0.90, 1.00, 1.08], np.float32)         # BGR gains → warmer light
    f = np.clip(f, 0, 255).astype(np.uint8)
    h, w = f.shape[:2]
    dy, dx = int(h * 0.03), int(w * 0.03)
    return cv2.resize(f[dy:h - dy, dx:w - dx], (w, h), interpolation=cv2.INTER_LINEAR)


def main():
    SRC.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.is_file():
        print(f"Downloading {URL} …")
        urllib.request.urlretrieve(URL, SRC)

    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, SIZE, interpolation=cv2.INTER_CUBIC))
    cap.release()
    if not frames:
        sys.exit(f"Could not read {SRC}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    for f in frames:
        out.write(f)
    for _ in range(int(GAP_S * fps)):          # empty room between passes
        out.write(frames[0])
    for f in frames:
        out.write(relight(f))
    out.release()
    codec = finalize_mp4(tmp, OUT, crf=23)
    total = (2 * len(frames)) / fps + GAP_S
    print(f"Wrote {OUT}  ({total / 60:.1f} min @ {fps:.0f} fps, {codec})")
    print("Expected: 6 NEW visitors in pass 1, then the same 6 as SEEN BEFORE in pass 2.")


if __name__ == "__main__":
    main()
