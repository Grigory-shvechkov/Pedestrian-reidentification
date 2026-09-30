"""Check every component on this machine before running the app.

    python tools/selftest.py              # uses CAMERA_INDEX from config.py
    python tools/selftest.py --source 1   # or another camera / a video file

Downloads model weights on first run. Prints PASS/FAIL per step and timings.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import src  # noqa: E402,F401

results = []


def step(name):
    def wrap(fn):
        t0 = time.perf_counter()
        try:
            info = fn() or ""
            results.append((name, True))
            print(f"[PASS] {name}  ({(time.perf_counter() - t0) * 1000:.0f} ms)  {info}")
        except Exception as e:  # noqa: BLE001
            results.append((name, False))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        return fn
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    args = ap.parse_args()
    if args.source is not None:
        config.CAMERA_INDEX = int(args.source) if args.source.isdigit() else args.source

    state = {}

    @step("python packages")
    def _():
        import cv2, PIL, scipy, torch, ultralytics  # noqa: F401,E401
        cuda = torch.cuda.is_available()
        return f"torch {torch.__version__}, CUDA {'yes – ' + torch.cuda.get_device_name(0) if cuda else 'no (CPU)'}"

    @step("camera")
    def _():
        from src.camera import Camera
        cam = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, config.CAMERA_BACKEND)
        try:
            r = cam.read(timeout=5.0)
            if r is None:
                raise RuntimeError("camera opened but delivered no frames")
            state["frame"] = r[2].copy()
            return f"{r[2].shape[1]}x{r[2].shape[0]}"
        finally:
            cam.release()

    @step("person detector (YOLO)")
    def _():
        from src.detector import PersonDetector
        det = PersonDetector(config.DETECTOR_WEIGHTS, config.DETECT_LOW_CONFIDENCE, config.DETECT_IMGSZ, config.DEVICE)
        frame = state.get("frame", np.zeros((480, 640, 3), np.uint8))
        t0 = time.perf_counter()
        people = det.detect(frame)
        state["people"] = [(frame, d) for d in people]
        return (f"{det.device}, seg={det.is_seg}, {(time.perf_counter() - t0) * 1000:.0f} ms/frame, "
                f"{len(people)} people in the test frame")

    @step("re-ID embedder")
    def _():
        from src.detector import resolve_device
        from src.embedder import Embedder
        emb = Embedder(config, resolve_device(config.DEVICE))
        crops = [f[d.ibox[1]:d.ibox[3], d.ibox[0]:d.ibox[2]] for f, d in state.get("people", [])]
        crops = crops or [np.random.randint(0, 255, (256, 110, 3), np.uint8)]
        emb.embed(crops)
        t0 = time.perf_counter()
        e = emb.embed(crops)
        ms = (time.perf_counter() - t0) * 1000 / len(crops)
        if emb.warning:
            print("       WARNING:", emb.warning.replace("\n", "\n       "))
        return f"{emb.signature}, {e.shape[1]}-d, {ms:.0f} ms/person"

    @step("database")
    def _():
        from src.database import VisitorDB
        with tempfile.TemporaryDirectory() as d:
            db = VisitorDB(Path(d) / "t.db", reset=True)
            db.create_visitor(np.ones(8, np.float32), 1.0, time.time(), 1, b"x")
            n = db.count_visitors()
            db.close()
        return f"{n} test row written"

    ok = all(r[1] for r in results)
    print("\nAll good – run: python main.py" if ok else "\nFix the FAIL lines above first.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
