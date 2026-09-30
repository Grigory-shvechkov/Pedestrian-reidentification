"""SQLite persistence.

Schema changes vs. the original single-table design:
  visitors   – one row per unique person (no embedding column)
  prototypes – up to N appearance embeddings per visitor, stored as float32 BLOBs
               (≈5× smaller and far faster to load than JSON)
  visits     – one row per counted visit → hourly/daily analytics, dwell time
  meta       – embedder signature; vectors from a different backend are
               incomparable, so a mismatch wipes the gallery instead of
               silently producing garbage similarities.

Only the worker thread writes; a lock makes accidental cross-thread use safe.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS visitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    thumb BLOB
);
CREATE TABLE IF NOT EXISTS prototypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id INTEGER NOT NULL REFERENCES visitors(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    quality REAL NOT NULL,
    created REAL NOT NULL,
    is_anchor INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id INTEGER NOT NULL REFERENCES visitors(id) ON DELETE CASCADE,
    track_id INTEGER,
    ts REAL NOT NULL,
    similarity REAL,
    is_new INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proto_visitor ON prototypes(visitor_id);
CREATE INDEX IF NOT EXISTS idx_visits_ts ON visits(ts);
"""


def _midnight_ts() -> float:
    return dt.datetime.combine(dt.date.today(), dt.time()).timestamp()


class VisitorDB:
    def __init__(self, path: Path, reset: bool = False):
        path = self.path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(visitors)")}
        if "thumb" not in cols:  # DB created by an earlier version
            self.conn.execute("ALTER TABLE visitors ADD COLUMN thumb BLOB")
        if reset:
            self.reset()

    def _tx(self, statements):
        """Run [(sql, params), ...] atomically; returns lastrowid of each statement."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("BEGIN")
            try:
                ids = []
                for sql, params in statements:
                    cur.execute(sql, params)
                    ids.append(cur.lastrowid)
                cur.execute("COMMIT")
                return ids
            except Exception:
                cur.execute("ROLLBACK")
                raise

    # ── lifecycle ──
    def reset(self):
        with self._lock:
            self.conn.executescript("DELETE FROM visits; DELETE FROM prototypes; DELETE FROM visitors;"
                                    "DELETE FROM sqlite_sequence WHERE name IN ('visitors','prototypes','visits');")
        log.info("Database reset")

    def ensure_signature(self, signature: str, dim: int):
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        n = self.count_visitors()
        if row and row[0] != signature and n:
            log.warning("Embedder changed (%s → %s): clearing %d incompatible visitor profiles",
                        row[0], signature, n)
            self.reset()
        with self._lock:
            self.conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                                  [("signature", signature), ("dim", str(dim))])

    def prune(self, older_than_ts: float) -> int:
        with self._lock:
            n = self.conn.execute("DELETE FROM visitors WHERE last_seen < ?", (older_than_ts,)).rowcount
        if n:
            log.info("Pruned %d visitors past retention", n)
        return n

    def close(self):
        with self._lock:
            self.conn.close()

    # ── gallery ──
    def load_gallery(self):
        """→ list of (visitor_id, last_seen, [(proto_id, emb, is_anchor), ...])"""
        with self._lock:
            vis = self.conn.execute("SELECT id, last_seen FROM visitors").fetchall()
            protos = self.conn.execute(
                "SELECT id, visitor_id, embedding, is_anchor FROM prototypes ORDER BY id").fetchall()
        by_vis = {}
        for pid, vid, blob, anchor in protos:
            by_vis.setdefault(vid, []).append((pid, np.frombuffer(blob, np.float32).copy(), bool(anchor)))
        return [(vid, last, by_vis.get(vid, [])) for vid, last in vis]

    def create_visitor(self, emb: np.ndarray, quality: float, ts: float, track_id: int,
                       thumb: bytes | None = None) -> tuple[int, int]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute("INSERT INTO visitors (first_seen, last_seen, visit_count, thumb) VALUES (?, ?, 1, ?)",
                            (ts, ts, thumb))
                vid = cur.lastrowid
                cur.execute("INSERT INTO prototypes (visitor_id, embedding, quality, created, is_anchor) "
                            "VALUES (?, ?, ?, ?, 1)", (vid, emb.astype(np.float32).tobytes(), quality, ts))
                proto_id = cur.lastrowid
                cur.execute("INSERT INTO visits (visitor_id, track_id, ts, similarity, is_new) "
                            "VALUES (?, ?, ?, NULL, 1)", (vid, track_id, ts))
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        return vid, proto_id

    def add_prototype(self, vid: int, emb: np.ndarray, quality: float, ts: float) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO prototypes (visitor_id, embedding, quality, created, is_anchor) VALUES (?, ?, ?, ?, 0)",
                (vid, emb.astype(np.float32).tobytes(), quality, ts))
            return cur.lastrowid

    def delete_prototype(self, proto_id: int):
        with self._lock:
            self.conn.execute("DELETE FROM prototypes WHERE id=?", (proto_id,))

    def record_visit(self, vid: int, ts: float, track_id: int, similarity: float, new_visit: bool):
        if new_visit:
            self._tx([
                ("UPDATE visitors SET last_seen=?, visit_count=visit_count+1 WHERE id=?", (ts, vid)),
                ("INSERT INTO visits (visitor_id, track_id, ts, similarity, is_new) VALUES (?, ?, ?, ?, 0)",
                 (vid, track_id, ts, similarity)),
            ])
        else:
            self.touch(vid, ts)

    def touch(self, vid: int, ts: float):
        with self._lock:
            self.conn.execute("UPDATE visitors SET last_seen=MAX(last_seen, ?) WHERE id=?", (ts, vid))

    # ── stats ──
    def get_thumb(self, vid: int) -> bytes | None:
        with self._lock:
            row = self.conn.execute("SELECT thumb FROM visitors WHERE id=?", (vid,)).fetchone()
        return row[0] if row else None

    def visit_count(self, vid: int) -> int:
        with self._lock:
            row = self.conn.execute("SELECT visit_count FROM visitors WHERE id=?", (vid,)).fetchone()
        return row[0] if row else 0

    def count_visitors(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]

    def stats(self, recent: int = 15, since: float = 0.0) -> dict:
        """`since`: only count activity from this time on (the UI's "Clear panel")."""
        t0 = max(_midnight_ts(), since)
        hour_ago = max(time.time() - 3600, since)
        with self._lock:
            c = self.conn
            unique = c.execute("SELECT COUNT(*) FROM visitors WHERE last_seen>=?", (since,)).fetchone()[0]
            visits_today = c.execute("SELECT COUNT(*) FROM visits WHERE ts>=?", (t0,)).fetchone()[0]
            new_today = c.execute("SELECT COUNT(*) FROM visits WHERE ts>=? AND is_new=1", (t0,)).fetchone()[0]
            last_hour = c.execute("SELECT COUNT(*) FROM visits WHERE ts>=?", (hour_ago,)).fetchone()[0]
            returning = c.execute("SELECT COUNT(*) FROM visitors WHERE visit_count>1 AND last_seen>=?",
                                  (since,)).fetchone()[0]
            rows = c.execute("SELECT id, first_seen, last_seen, visit_count FROM visitors WHERE last_seen>=? "
                             "ORDER BY last_seen DESC LIMIT ?", (since, recent)).fetchall()
        return {
            "unique_visitors": unique,
            "visits_today": visits_today,
            "new_today": new_today,
            "returning_today": visits_today - new_today,
            "visits_last_hour": last_hour,
            "returning_visitors": returning,
            "recent": [{"id": r[0], "first_seen": r[1], "last_seen": r[2], "visits": r[3]} for r in rows],
        }
