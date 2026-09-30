"""Render a showcase video: the real pipeline's output with an info panel.

Runs detection → tracking → re-ID on every frame of a video (no UI, no frame
dropping, so the result is the same on a slow or fast PC) and writes:

    [ annotated video | panel: last person NEW / SEEN BEFORE, stored vs. now photo, counts ]

    python tools/render_showcase.py                                   # videos/test_reid.mp4
    python tools/render_showcase.py --input my_clip.mp4 --speed 1
    python tools/render_showcase.py --gif videos/showcase.gif         # also a short README preview

Uses its own database (data/showcase.db), so your visitor database is untouched.
If `imageio-ffmpeg` is installed the output is H.264 (plays everywhere, smaller);
otherwise OpenCV's MPEG-4 (plays in VLC / VS Code / browsers with the codec).
"""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.pipeline import Pipeline, ZoneState  # noqa: E402
from src.videoio import finalize_mp4  # noqa: E402

PANEL_W = 420
H = 540
WHITE, GREY, DARK = (245, 245, 245), (170, 170, 170), (32, 32, 36)
GREEN, PINK = (80, 200, 80), (200, 80, 255)
FONT, BOLD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX


def _text(img, s, x, y, scale=0.5, color=WHITE, font=FONT, thick=1):
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _jpg(data, size):
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _paste(dst, img, x, y, border=None):
    h, w = img.shape[:2]
    dst[y:y + h, x:x + w] = img
    if border:
        cv2.rectangle(dst, (x - 2, y - 2), (x + w + 1, y + h + 1), border, 2)


def draw_panel(stats, event, now, speed, model, thr):
    p = np.full((H, PANEL_W, 3), DARK, np.uint8)
    _text(p, "PEDESTRIAN PROFILER", 20, 34, 0.75, WHITE, BOLD)
    _text(p, "Is this person new, or seen before?", 20, 58, 0.48, GREY)

    _text(p, "LAST PERSON", 20, 92, 0.5, GREY)
    fresh = event is not None and now - event["ts"] < 2.5
    if event is None:
        _text(p, "nobody yet", 20, 128, 0.7, GREY, BOLD)
    else:
        seen = event["kind"] == "recognised"
        col = PINK if seen else GREEN
        label = f"SEEN BEFORE  #{event['profile_id']}" if seen else f"NEW VISITOR  #{event['profile_id']}"
        cv2.rectangle(p, (16, 102), (PANEL_W - 16, 142), col, -1 if fresh else 2)
        _text(p, label, 28, 131, 0.85, (20, 20, 20) if fresh else col, BOLD, 2)
        if seen:
            detail = f"match {event['similarity'] * 100:.0f}%   visit {event['visits']}"
        else:
            detail = (f"closest stored person {event['similarity'] * 100:.0f}%"
                      if event["similarity"] > -1 else "first person of the day")
        _text(p, detail, 20, 166, 0.5, WHITE)

        pw, ph = 80, 160
        then = _jpg(event["then"], (pw, ph))
        nowi = _jpg(event["now"], (pw, ph))
        x1, x2, y = 60, 60 + pw + 70, 182
        if then is not None:
            _paste(p, then, x1, y, col)
        else:
            cv2.rectangle(p, (x1, y), (x1 + pw, y + ph), (70, 70, 70), 1)
            _text(p, "first", x1 + 24, y + 84, 0.5, GREY)
            _text(p, "time", x1 + 26, y + 104, 0.5, GREY)
        if nowi is not None:
            _paste(p, nowi, x2, y, col)
        _text(p, "stored", x1 + 16, y + ph + 20, 0.45, GREY)
        _text(p, "now", x2 + 28, y + ph + 20, 0.45, GREY)

    y = 388
    for label, val in (("Unique visitors", stats.get("unique_visitors", 0)),
                       ("Returning visitors", stats.get("returning_visitors", 0)),
                       ("In view now", stats.get("in_view", 0))):
        _text(p, label, 20, y, 0.55, WHITE)
        _text(p, str(val), PANEL_W - 50, y, 0.65, WHITE, BOLD)
        y += 22

    x = 20
    for r in sorted(stats.get("recent", []), key=lambda r: r["id"])[:9]:
        t = _jpg(r.get("thumb"), (32, 64))
        if t is not None and x + 32 < PANEL_W:
            _paste(p, t, x, 462)
            _text(p, f"#{r['id']}", x, 457, 0.35, GREY)
            if r["visits"] > 1:
                _text(p, f"x{r['visits']}", x + 18, 457, 0.35, PINK)
            x += 44

    foot = f"{model} | thr {thr:.2f}" + (f" | {speed:g}x speed" if speed != 1 else "")
    _text(p, foot, 20, H - 5, 0.35, GREY)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "videos" / "test_reid.mp4")
    ap.add_argument("--output", type=Path, default=ROOT / "videos" / "showcase.mp4")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed of the output (2 = twice as fast)")
    ap.add_argument("--gif", type=Path, help="also write a short animated preview of a clear 'seen before' moment")
    ap.add_argument("--gif-start", type=float, help="start the preview at this video time (s) instead")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("ultralytics", "src.database", "src.re_identifier"):
        logging.getLogger(name).setLevel(logging.WARNING)

    config.DB_PATH = ROOT / "data" / "showcase.db"
    config.DB_RESET_ON_START = True
    config.AUDIO_CUE_ENABLED = False
    config.PROFILE_RETENTION_DAYS = None

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.input} (run tools/make_test_video.py first?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, round(fps / config.DETECT_HZ))

    p = Pipeline(config, queue.Queue(maxsize=1), ZoneState(), threading.Event())
    p.init_models()
    model = p.embedder.deep.name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.mp4")
    out = None
    base = time.time() - n_frames / fps
    gif_frames, gif_start = [], None
    pre = deque(maxlen=int(1.5 * fps))   # frames just before the moment
    t0 = time.perf_counter()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != H:
            frame = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
        now = base + i / fps
        p.process(frame, now, detect=(i % every == 0))
        vis = p.annotate(frame, now, fps)
        panel = draw_panel(p.stats_snapshot(now, fps), p.last_event, now, args.speed, model,
                           p.embedder.threshold)
        canvas = np.hstack([vis, panel])
        if out is None:
            out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps * args.speed,
                                  (canvas.shape[1], canvas.shape[0]))
        out.write(canvas)

        if args.gif:  # ~6 s around the first *clear* "seen before" (well above the threshold)
            small = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))
            ev = p.last_event
            if gif_start is None:
                if args.gif_start is not None:
                    hit = i / fps >= args.gif_start
                else:
                    hit = bool(ev and ev["kind"] == "recognised"
                               and ev["similarity"] >= p.embedder.threshold + 0.10)
                if hit:
                    gif_start = i
                    gif_frames = [f for k, f in enumerate(pre) if k % 2 == 0]
                else:
                    pre.append(small)
            elif i - gif_start < 4.5 * fps and i % 2 == 0:
                gif_frames.append(small)
        i += 1
        if i % 100 == 0:
            el = time.perf_counter() - t0
            print(f"\r  {i}/{n_frames} frames  ({i / el:.1f} fps, ~{(n_frames - i) / max(i / el, 1e-3):.0f}s left)",
                  end="", flush=True)
    p.finish(base + i / fps)
    cap.release()
    if out is not None:
        out.release()
    print()

    codec = finalize_mp4(tmp, args.output)
    s = p.db.stats()
    print(f"Wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB, {codec})")
    print(f"Result: {s['unique_visitors']} unique visitors, {s['returning_visitors']} recognised as returning")

    if args.gif and gif_frames:
        from PIL import Image
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(128) for f in gif_frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0,
                     optimize=True)
        print(f"Wrote {args.gif}  ({args.gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
