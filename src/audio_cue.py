"""Audio cue API stub – fired when a person in the zone hasn't been seen yet today.

`play_cue` is the integration point: today it logs the event, plays a local sound
(so the behaviour can be tested at the desk) and, if AUDIO_CUE_URL is set, POSTs
the event as JSON. Replace or extend it for the real speaker / PA system.
"""
from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger(__name__)


def play_cue(cfg, visitor_id: int, first_ever: bool, ts: float) -> None:
    """Called off the detection thread, so it may block (network, sound playback)."""
    event = {"event": "not_seen_today", "visitor_id": visitor_id, "first_ever": first_ever, "ts": ts}
    log.info("Audio cue: visitor #%d not seen today%s", visitor_id, " (first ever)" if first_ever else "")
    if cfg.AUDIO_CUE_URL:
        try:
            req = urllib.request.Request(cfg.AUDIO_CUE_URL, json.dumps(event).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=cfg.API_TIMEOUT_S).close()
        except Exception as e:  # noqa: BLE001
            log.warning("Audio cue API call failed: %s", e)
    if cfg.AUDIO_CUE_SOUND:
        _play_local(cfg.AUDIO_CUE_SOUND)


def _play_local(sound) -> None:
    try:
        import winsound
    except ImportError:  # not Windows
        print("\a", end="", flush=True)
        return
    try:
        if sound == "beep":
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            winsound.PlaySound(str(sound), winsound.SND_FILENAME)
    except RuntimeError as e:
        log.warning("Could not play %s: %s", sound, e)
