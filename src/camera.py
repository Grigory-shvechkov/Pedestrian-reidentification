"""Camera reader that always hands out the newest frame.

cv2.VideoCapture buffers frames internally; if processing is slower than the
camera you end up processing frames from seconds ago. A dedicated grab thread
keeps only the latest frame so latency stays constant.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)

_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class Camera:
    def __init__(self, source, width=None, height=None, backend="any"):
        self.is_file = isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://"))
        api = _BACKENDS.get(backend, cv2.CAP_ANY) if isinstance(source, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened() and api != cv2.CAP_ANY:
            log.warning("Camera backend %s failed, retrying with CAP_ANY", backend)
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source {source!r}")
        if isinstance(source, int):
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        reported = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if 1 <= (reported or 0) <= 240 else 30.0
        self._file_fps = self.fps

        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._stop = threading.Event()
        self.eof = False
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.eof = True
                    with self._cond:
                        self._cond.notify_all()
                    return
                fails += 1
                if fails % 100 == 0:
                    log.warning("Camera read failing (%d consecutive)", fails)
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._ts = time.monotonic()
                self._cond.notify_all()
            if self.is_file:  # play files at their native rate
                time.sleep(1.0 / self._file_fps)

    def read(self, last_seq: int = -1, timeout: float = 1.0):
        """Return (seq, timestamp, frame) newer than `last_seq`, or None on timeout/EOF.

        The returned frame must be treated as read-only (copy before drawing).
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq or self._stop.is_set() or self.eof, timeout
            )
            if self._frame is None or self._seq == last_seq:
                return None
            return self._seq, self._ts, self._frame

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
