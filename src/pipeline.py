"""Worker-thread pipeline: camera → detect → track → quality gate → embed → re-ID.

Kept independent of Tkinter so it can run headless or in tests. The UI only
receives ready-to-show PIL images and plain dicts through a size-1 queue
(latest wins) and sends commands (e.g. "reset_db") through `self.commands`.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import quality
from .audio_cue import play_cue
from .database import _midnight_ts
from .embedder import l2n, white_balance_gains

log = logging.getLogger(__name__)

DECIDED = ("new", "recognised", "known")
THUMB_HW = (128, 64)


def fit(fw: int, fh: int, cw: int, ch: int) -> tuple[float, float, float]:
    """Scale + offset that letterboxes a fw×fh frame into a cw×ch canvas."""
    s = min(cw / fw, ch / fh) if fw and fh else 1.0
    return s, (cw - fw * s) / 2, (ch - fh * s) / 2


def make_thumb(crop: np.ndarray) -> bytes:
    h, w = THUMB_HW
    ch, cw = crop.shape[:2]
    s = min(w / cw, h / ch)
    small = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 40, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


class ZoneState:
    """Thread-safe holder for the user-drawn zone polygon (frame coordinates)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._poly = None

    def set(self, poly):
        with self._lock:
            self._poly = None if poly is None else np.asarray(poly, np.float32).copy()

    def get(self):
        with self._lock:
            return None if self._poly is None else self._poly.copy()


@dataclass
class TrackState:
    track_id: int
    status: str = "detecting"
    wait_reason: str = ""
    first_confirmed_ts: float | None = None
    relaxed: bool = False                          # quality gate loosened after a long wait
    co_visible: set = field(default_factory=set)   # tracks seen in the same frame → different people
    samples: list = field(default_factory=list)   # [(quality, emb, crop|None)]
    best_crop: np.ndarray | None = None
    best_q: float = -1.0
    first_sample_ts: float | None = None
    profile_id: int | None = None
    similarity: float | None = None

    @property
    def decided(self):
        return self.status in DECIDED

    def add_sample(self, emb, q, crop, keep_crop, now, max_samples):
        if self.first_sample_ts is None:
            self.first_sample_ts = now
        if q > self.best_q:
            self.best_q, self.best_crop = q, crop.copy()
        self.samples.append((q, emb, crop.copy() if keep_crop else None))
        if len(self.samples) > max_samples:  # keep the best crops
            self.samples.sort(key=lambda s: s[0], reverse=True)
            del self.samples[max_samples:]

    def aggregate(self):
        """Quality-weighted mean embedding with one round of outlier rejection
        (e.g. a crop where another person walked through the box)."""
        E = np.stack([s[1] for s in self.samples])
        w = np.array([s[0] for s in self.samples], np.float32)
        mean = l2n((w[:, None] * E).sum(0))
        sims = E @ mean
        keep = sims >= np.median(sims) - 0.10
        if keep.sum() >= 2:
            mean = l2n((w[keep, None] * E[keep]).sum(0))
        return mean.astype(np.float32), float(w.mean())


class Pipeline:
    def __init__(self, cfg, out_q: queue.Queue, zone: ZoneState, stop: threading.Event,
                 display_size=lambda: (960, 540)):
        self.cfg = cfg
        self.q = out_q
        self.zone = zone
        self.stop = stop
        self.display_size = display_size
        self.commands: queue.Queue = queue.Queue()
        self.states: dict[int, TrackState] = {}
        self.banner = None
        self.last_event = None          # latest new/recognised decision, shown in the UI
        self.active_zone = None
        self._api = ThreadPoolExecutor(max_workers=2, thread_name_prefix="api")
        self._stats = {}
        self._stats_ts = 0.0
        self._visible_ids: set[int] = set()
        self._track_profile: dict[int, int] = {}   # track id → visitor id, once decided
        self.cam = None
        self._view_since = 0.0

    # ── setup ─────────────────────────────────────────────────────────────
    def _init(self):
        self.init_models()
        log.info("Pipeline ready: %s, threshold %.2f", self.embedder.signature, self.embedder.threshold)
        self._open_source(self.cfg.CAMERA_INDEX)

    def _open_source(self, source):
        """Switch to a webcam index / video path. On failure the pipeline idles so the UI can pick another."""
        from .camera import Camera

        if self.cam is not None:
            self.cam.release()
            self.cam = None
        c = self.cfg
        self._emit({"status": "Opening camera…"})
        try:
            self.cam = Camera(source, c.FRAME_WIDTH, c.FRAME_HEIGHT, c.CAMERA_BACKEND)
        except RuntimeError as e:
            log.error("%s", e)
            self._emit({"status": f"{e} – try 'Use test video'"})
            return
        self._every = max(1, round(self.cam.fps / max(c.DETECT_HZ, 0.1)))
        self._last_seq, self._frame_i, self._eof_reported = -1, 0, False
        self._last_frame_t, self._stall_reported = time.monotonic(), False
        log.info("Source %r at %.0f fps → detecting every %d frame(s)", source, self.cam.fps, self._every)

    def init_models(self):
        """Load detector, embedder, database, tracker (everything except the camera).
        Also used by tools/render_showcase.py to run the pipeline without a UI."""
        from .database import VisitorDB
        from .detector import PersonDetector, resolve_device
        from .embedder import Embedder
        from .tracker import Tracker

        c = self.cfg
        self._emit({"status": "Loading person detector (first run downloads YOLO weights)…"})
        self.detector = PersonDetector(c.DETECTOR_WEIGHTS, c.DETECT_LOW_CONFIDENCE, c.DETECT_IMGSZ, c.DEVICE)
        self._emit({"status": "Loading re-ID model (first run downloads OSNet weights)…"})
        self.embedder = Embedder(c, resolve_device(c.DEVICE))
        path, reset = self._db_for(c.CAMERA_INDEX)
        self.db = VisitorDB(path, reset=reset)
        self.db.ensure_signature(self.embedder.signature, self.embedder.dim)
        if c.PROFILE_RETENTION_DAYS:
            self.db.prune(time.time() - c.PROFILE_RETENTION_DAYS * 86400)
        self._new_reid()
        self.tracker = Tracker(c.PERSON_CONFIDENCE, c.TRACK_MATCH_IOU, c.TRACK_LOW_MATCH_IOU,
                               c.TRACK_MIN_HITS, c.TRACK_MAX_AGE, c.TRACK_MAX_CENTER_DIST)

    def _db_for(self, source):
        c = self.cfg
        if isinstance(source, str) and Path(source).resolve() == Path(c.TEST_VIDEO).resolve():
            return c.TEST_DB_PATH, True
        return c.DB_PATH, c.DB_RESET_ON_START

    def _switch_db(self, source):
        from .database import VisitorDB

        path, reset = self._db_for(source)
        if Path(self.db.path).resolve() == Path(path).resolve() and not reset:
            return
        self.db.close()
        self.db = VisitorDB(path, reset=reset)
        self.db.ensure_signature(self.embedder.signature, self.embedder.dim)
        self._new_reid()

    def _new_reid(self):
        from .re_identifier import ReIdentifier
        self.reid = ReIdentifier(self.db, self.embedder.dim, self.embedder.threshold, self.cfg)

    # ── main loop ─────────────────────────────────────────────────────────
    def run(self):
        try:
            self._init()
            self._loop()
        except Exception as e:  # noqa: BLE001 – surface any failure in the UI
            log.exception("Pipeline crashed")
            self._emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            self._shutdown()

    def _loop(self):
        fps, t_prev, errors = 0.0, time.monotonic(), 0
        while not self.stop.is_set():
            self._handle_commands()
            if self.cam is None:
                self.stop.wait(0.1)
                continue
            r = self.cam.read(self._last_seq, timeout=1.0)
            if r is None:
                if self.cam.eof and not self._eof_reported:
                    self._eof_reported = True
                    self.finish(time.time())
                    self._emit({"status": "End of video"})
                elif (not self.cam.eof and not self._stall_reported
                      and time.monotonic() - self._last_frame_t > 3.0):
                    self._stall_reported = True
                    self._emit({"status": "No frames from the camera – is it in use or blocked "
                                          "by Windows privacy settings? Try 'Use test video'."})
                continue
            self._last_frame_t, self._stall_reported = time.monotonic(), False
            self._last_seq, _, frame = r
            now = time.time()
            frame_i = self._frame_i
            self._frame_i += 1
            if frame_i % self._every == 0:
                try:
                    self._cycle(frame, now)
                    errors = 0
                except Exception:  # noqa: BLE001
                    errors += 1
                    log.exception("Detection cycle failed (%d in a row)", errors)
                    if errors >= 20:
                        raise

            t = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-3)
            t_prev = t
            self._emit(self._render(frame, now, fps))

    def _handle_commands(self):
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            if cmd == "clear_view":
                self._view_since = time.time()
                self.banner = self.last_event = None
                self._stats_ts = 0.0
            elif cmd == "reset_db":
                self._reset_visitors()
                log.info("Visitor database cleared from UI")
            elif isinstance(cmd, tuple) and cmd[0] == "source" and cmd[1] is None:
                self.finish(time.time())
                if self.cam is not None:
                    self.cam.release()
                    self.cam = None
                self._emit({"status": "Stopped"})
            elif isinstance(cmd, tuple) and cmd[0] == "source":
                self.finish(time.time())
                self.tracker.tracks = []    # old source's people must not carry over
                self._switch_db(cmd[1])
                self._clear_session()
                self._open_source(cmd[1])

    def _reset_visitors(self):
        self.db.reset()
        self._new_reid()
        self._clear_session()

    def _clear_session(self):
        self.states.clear()   # everyone in view gets re-decided as a fresh visitor
        self._track_profile.clear()
        self.banner = self.last_event = None
        self._stats_ts = 0.0
        self._view_since = 0.0

    # ── one detection cycle ───────────────────────────────────────────────
    def _cycle(self, frame, now):
        c = self.cfg
        H, W = frame.shape[:2]
        zone = self.zone.get()
        zone = zone if zone is not None and len(zone) >= 3 else None
        self.active_zone = zone
        roi, bounds = None, (0, 0, W, H)
        if zone is not None:
            x, y, w, h = cv2.boundingRect(zone.astype(np.int32))
            top = 0 if c.ZONE_ANCHOR == "bottom" else max(0, y - c.ROI_PAD)   # floor zone: bodies stand above it
            roi = (max(0, x - c.ROI_PAD), top, min(W, x + w + c.ROI_PAD), min(H, y + h + c.ROI_PAD))
            bounds = roi

        dets = self.detector.detect(frame, roi)
        if zone is not None:
            contour = zone.reshape(-1, 1, 2).astype(np.float32)
            dets = [d for d in dets if cv2.pointPolygonTest(contour, self._anchor(d.box), False) >= 0]

        tracks, removed = self.tracker.update(dets, now)
        visible = [t for t in tracks if t.time_since_update == 0]
        self._visible_ids = {t.id for t in visible if t.confirmed}
        for t in visible:
            if t.confirmed:
                st = self.states.setdefault(t.id, TrackState(t.id, first_confirmed_ts=now))
                st.co_visible |= self._visible_ids - {t.id}
        for t in removed:
            self._on_track_end(t, now)

        todo = []
        for t in visible:
            if not t.confirmed:
                continue
            st = self.states.setdefault(t.id, TrackState(t.id, first_confirmed_ts=now))
            if st.decided:
                self.reid.touch(st.profile_id, now)
                continue
            x1, y1, x2, y2 = t.det.ibox
            crop = frame[y1:y2, x1:x2]
            others = [o.det.box for o in visible if o is not t]
            if not st.samples and now - st.first_confirmed_ts > c.QUALITY_RELAX_AFTER_S:
                st.relaxed = True
            ok, reason, q = quality.assess(t.det, crop, others, bounds, c, st.relaxed)
            if not ok:
                st.wait_reason = reason
                continue
            st.wait_reason = ""
            todo.append((st, crop, t.det.mask, q))

        if todo:
            gains = white_balance_gains(frame) if c.WHITE_BALANCE else None
            embs = self.embedder.embed([x[1] for x in todo], [x[2] for x in todo], gains)
            for (st, crop, _, q), e in zip(todo, embs):
                st.add_sample(e, q, crop, c.SAVE_CROPS, now, c.MAX_SAMPLES)
                if st.status == "detecting":
                    st.status = "collecting"

        for st in list(self.states.values()):
            if not st.decided and st.samples:
                self._maybe_decide(st, now, final=False)

    def _anchor(self, box):
        x1, y1, x2, y2 = (float(v) for v in box)
        return ((x1 + x2) / 2, y2 - 2) if self.cfg.ZONE_ANCHOR == "bottom" else ((x1 + x2) / 2, (y1 + y2) / 2)

    # ── decisions ─────────────────────────────────────────────────────────
    def _maybe_decide(self, st: TrackState, now: float, final: bool):
        c = self.cfg
        n = len(st.samples)
        if n < c.MIN_SAMPLES:
            return
        timed_out = st.first_sample_ts is not None and now - st.first_sample_ts >= c.MAX_DECISION_WAIT_S
        final = final or timed_out or n >= c.MAX_SAMPLES
        if not final and n < c.SAMPLES_TO_DECIDE:
            return

        emb, q = st.aggregate()
        # A visitor already assigned to a track that was in the same frame as this one
        # can't be this person. (Tracks that merely follow each other – a broken track –
        # are *not* excluded, so the fragments can re-join the same visitor.)
        exclude = {self._track_profile[t] for t in st.co_visible if t in self._track_profile}
        d = self.reid.decide(emb, exclude, final)

        if d.kind == "pending":
            st.status, st.similarity = "verifying", d.similarity
            return
        now_thumb = make_thumb(st.best_crop) if st.best_crop is not None else None
        if d.kind == "match":
            first_today = self.reid.profiles[d.profile_id]["last_seen"] < _midnight_ts()
            new_visit = self.reid.commit_match(d.profile_id, emb, d.similarity, q, now, st.track_id)
            st.profile_id, st.similarity = d.profile_id, d.similarity
            st.status = "recognised" if new_visit else "known"
            visits = self.db.visit_count(d.profile_id)
            if new_visit:
                self.banner = {"profile_id": d.profile_id, "similarity": d.similarity,
                               "visits": visits, "until": now + c.BANNER_SECONDS}
                self._set_event("recognised", d.profile_id, d.similarity, visits, now,
                                self.db.get_thumb(d.profile_id), now_thumb, first_today)
            log.info("Track %d → visitor #%d (%.2f, %s)", st.track_id, d.profile_id, d.similarity, st.status)
            if first_today:
                self._cue(d.profile_id, False, now)
        else:
            thumb = now_thumb if c.STORE_THUMBNAILS else None
            pid = self.reid.commit_new(emb, q, now, st.track_id, thumb)
            st.profile_id, st.similarity, st.status = pid, None, "new"
            self._set_event("new", pid, d.similarity, 1, now, None, now_thumb, True)
            log.info("Track %d → NEW visitor #%d (closest existing %.2f)", st.track_id, pid, d.similarity)
            self._api_new_person(pid, now)
            self._cue(pid, True, now)

        self._track_profile[st.track_id] = st.profile_id
        if c.SAVE_CROPS:
            self._save_crops(st)
        st.samples.clear()
        st.best_crop = None
        self._stats_ts = 0.0  # refresh stats promptly

    def _set_event(self, kind, pid, sim, visits, ts, then_jpg, now_jpg, first_today):
        seq = (self.last_event or {}).get("seq", 0) + 1
        self.last_event = {"seq": seq, "kind": kind, "profile_id": pid, "similarity": sim,
                           "visits": visits, "ts": ts, "then": then_jpg, "now": now_jpg,
                           "first_today": first_today}

    def _cue(self, pid, first_ever, now):
        if self.cfg.AUDIO_CUE_ENABLED:
            self._api.submit(play_cue, self.cfg, pid, first_ever, now)

    def _on_track_end(self, track, now):
        st = self.states.pop(track.id, None)
        if st is None:
            return
        if not st.decided and len(st.samples) >= self.cfg.MIN_SAMPLES:
            self._maybe_decide(st, now, final=True)   # brief pass-by: still count it
        if st.profile_id is not None:
            self.reid.flush(st.profile_id)

    def _save_crops(self, st: TrackState):
        out = Path(self.cfg.CROPS_DIR) / f"{st.profile_id:05d}"
        out.mkdir(parents=True, exist_ok=True)
        for i, (_, _, crop) in enumerate(st.samples):
            if crop is not None:
                cv2.imwrite(str(out / f"t{st.track_id:06d}_{i:02d}.jpg"), crop)

    def _api_new_person(self, profile_id: int, ts: float):
        url = self.cfg.API_URL
        payload = {"profile_id": profile_id, "first_seen": ts}
        if not url:
            return

        def post():
            try:
                req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=self.cfg.API_TIMEOUT_S).close()
            except Exception as e:  # noqa: BLE001
                log.warning("API call failed: %s", e)

        self._api.submit(post)  # never block the detection loop on the network

    # ── output ────────────────────────────────────────────────────────────
    def process(self, frame, now, detect: bool):
        """Run one detection cycle if `detect`; used by headless tools."""
        if detect:
            self._cycle(frame, now)

    def finish(self, now):
        """End all live tracks (decides people still in view). Headless tools call this at EOF."""
        for t in list(self.tracker.tracks):
            self._on_track_end(t, now)
        self.tracker.tracks = []

    def annotate(self, frame, now, fps):
        """Full-resolution BGR frame with boxes, labels and banner."""
        from . import visualizer

        items = []
        for t in self.tracker.tracks:
            if t.time_since_update > 2:
                continue
            st = self.states.get(t.id)
            if not t.confirmed:
                items.append({"box": t.box, "status": "tentative", "text": "…"})
                continue
            if st is None:
                st = TrackState(t.id)
            status, text = st.status, f"T{t.id}"
            if status == "detecting":
                text += f" wait: {st.wait_reason or '…'}"
            elif status == "collecting":
                text += f" {len(st.samples)}/{self.cfg.SAMPLES_TO_DECIDE}"
            elif status == "verifying":
                text += f" verifying {st.similarity * 100:.0f}%"
            elif status == "new":
                text = f"#{st.profile_id} NEW"
            elif status == "recognised":
                text = f"#{st.profile_id} SEEN BEFORE {st.similarity * 100:.0f}%"
            elif status == "known":
                text = f"#{st.profile_id}"
            if t.time_since_update > 0:
                status = "lost"
            items.append({"box": t.box, "status": status, "text": text})

        if self.banner and now > self.banner["until"]:
            self.banner = None

        return visualizer.draw(frame.copy(), items, self.active_zone, False, self.banner, fps,
                               self.embedder.family)

    def stats_snapshot(self, now, fps):
        if now - self._stats_ts >= self.cfg.STATS_REFRESH_S or now < self._stats_ts:
            self._stats = self.db.stats(since=self._view_since)
            self._stats["since"] = self._view_since
            for r in self._stats["recent"]:
                r["thumb"] = self.db.get_thumb(r["id"])
            self._stats_ts = now
        return dict(self._stats, in_view=len(self._visible_ids), fps=fps,
                    zone_source="drawn zone" if self.active_zone is not None else "full frame",
                    backend=self.embedder.signature, threshold=self.embedder.threshold,
                    warning=self.embedder.warning)

    def _render(self, frame, now, fps):
        from PIL import Image

        img = self.annotate(frame, now, fps)
        fh, fw = img.shape[:2]
        cw, ch = self.display_size()
        s, _, _ = fit(fw, fh, max(cw, 1), max(ch, 1))
        dw, dh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
        img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

        return {"image": Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                "frame_size": (fw, fh), "stats": self.stats_snapshot(now, fps), "event": self.last_event}

    def _emit(self, payload):
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def _shutdown(self):
        for st in list(self.states.values()):
            if st.profile_id is not None and hasattr(self, "reid"):
                self.reid.flush(st.profile_id)
        if self.cam is not None:
            self.cam.release()
        if hasattr(self, "db"):
            self.db.close()
        self._api.shutdown(wait=False)
