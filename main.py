"""Pedestrian profiler – detects people and tells you whether they've been seen before.

    python main.py                      # webcam CAMERA_INDEX from config.py
    python main.py --source 1           # another webcam
    python main.py --source clip.mp4    # a recorded video (great for testing)
    python main.py --reset-db           # start with an empty visitor list

Left: live view. Optional floor zone: "Draw zone", drag the 4 corners into any
quadrilateral (or drag inside to move it); only people whose feet are inside are
counted. Zones are saved per source. Someone not seen yet today triggers the audio
cue (src/audio_cue.py). Right: who was just seen, whether
they were seen before (stored photo vs. now), and visitor stats.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

import config
from src.pipeline import Pipeline, ZoneState, fit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("main")

HANDLE_R = 7
EVENT_IMG = (80, 160)   # w, h of the then/now photos
LIST_IMG = (28, 56)


def _jpg_to_photo(data: bytes | None, size) -> ImageTk.PhotoImage | None:
    if not data:
        return None
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Pedestrian Profiler")
        root.geometry("1440x860")
        root.minsize(900, 560)

        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.zone = ZoneState()
        self._display = (960, 540)
        self.frame_size = None
        self.zone_pts = None
        self.editing = False
        self._drag = None
        self._photo = None
        self._error_shown = False
        self._recent_key = None
        self._event_seq = None
        self._thumb_cache: dict[int, ImageTk.PhotoImage] = {}
        self._event_photos = []
        self._stopped = False
        self._webcam = config.CAMERA_INDEX if isinstance(config.CAMERA_INDEX, int) else 0

        self._build_ui()
        self.pipeline = Pipeline(config, self.q, self.zone, self.stop, display_size=lambda: self._display)
        self.worker = threading.Thread(target=self.pipeline.run, name="worker", daemon=True)
        self.worker.start()

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(config.UI_POLL_MS, self.poll)

    # ── layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Visitors.Treeview", rowheight=LIST_IMG[1] + 6)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        bar = ttk.Frame(left)
        bar.pack(fill=tk.X, padx=6, pady=4)
        self.edit_btn = ttk.Button(bar, text="Draw zone", command=self.toggle_edit)
        self.edit_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Clear zone", command=self.clear_zone).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Reset visitors", command=self.reset_db).pack(side=tk.LEFT, padx=4)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="Use webcam", command=lambda: self.set_source(self._webcam)).pack(side=tk.LEFT)
        ttk.Button(bar, text="Use test video", command=lambda: self.set_source(config.TEST_VIDEO)).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Stop", command=self.stop_source).pack(side=tk.LEFT)
        self.status = ttk.Label(bar, text="Starting…")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(left, bg="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        paned.add(left, weight=4)

        # scrollable right panel: everything lives in one frame inside a canvas
        right = self.right = ttk.Frame(paned, width=380)
        sc = self.panel_canvas = tk.Canvas(right, highlightthickness=0, width=360)
        sb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=sc.yview)
        f = self.panel = ttk.Frame(sc)
        win = sc.create_window((0, 0), window=f, anchor=tk.NW)
        f.bind("<Configure>", lambda e: sc.configure(scrollregion=(0, 0, e.width, e.height)))
        sc.bind("<Configure>", lambda e: sc.itemconfigure(win, width=e.width))
        sc.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # bind_all + pointer check: <Enter>/<Leave> on the canvas fire when crossing child widgets
        self.root.bind_all("<MouseWheel>", self._on_panel_wheel, add="+")
        paned.add(right, weight=1)

        self.warn_var = tk.StringVar(value="")
        self.warn = tk.Label(f, textvariable=self.warn_var, fg="#b00", wraplength=340, justify=tk.LEFT)
        self.warn.pack(anchor=tk.W, padx=10)

        # latest decision: then / now
        head = ttk.Frame(f)
        head.pack(fill=tk.X, padx=10, pady=(10, 2))
        ttk.Label(head, text="Last person", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        ttk.Button(head, text="Clear panel", command=self.clear_panel).pack(side=tk.RIGHT)
        self.event_var = tk.StringVar(value="Nobody yet")
        self.event_lbl = tk.Label(f, textvariable=self.event_var, font=("Segoe UI", 12, "bold"),
                                  justify=tk.LEFT, anchor=tk.W)
        self.event_lbl.pack(anchor=tk.W, padx=10)
        pics = ttk.Frame(f)
        pics.pack(anchor=tk.W, padx=10, pady=4)
        self.then_img = tk.Label(pics, text="first seen", compound=tk.TOP)
        self.now_img = tk.Label(pics, text="now", compound=tk.TOP)
        self.then_img.pack(side=tk.LEFT, padx=(0, 8))
        self.now_img.pack(side=tk.LEFT)

        vhead = ttk.Frame(f)
        vhead.pack(fill=tk.X, padx=10, pady=(12, 4))
        ttk.Label(vhead, text="Visitors", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        self.since_var = tk.StringVar(value="")
        ttk.Label(vhead, textvariable=self.since_var, foreground="#666").pack(side=tk.LEFT, padx=8)
        self.vars = {}
        for key, label in [("in_view", "In view now"), ("unique_visitors", "Unique visitors"),
                           ("visits_today", "Visits today"), ("new_today", "New today"),
                           ("returning_today", "Returning today"), ("visits_last_hour", "Visits last hour")]:
            row = ttk.Frame(f)
            row.pack(fill=tk.X, padx=10, pady=1)
            ttk.Label(row, text=label).pack(side=tk.LEFT)
            v = tk.StringVar(value="–")
            ttk.Label(row, textvariable=v, font=("Segoe UI", 11, "bold")).pack(side=tk.RIGHT)
            self.vars[key] = v

        ttk.Label(f, text="Recent visitors", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(14, 4))
        self.tree = ttk.Treeview(f, columns=("id", "visits", "first", "last"), show="tree headings",
                                 height=8, style="Visitors.Treeview")
        self.tree.column("#0", width=LIST_IMG[0] + 20, stretch=False)
        for col, text, w in [("id", "#", 45), ("visits", "Visits", 50), ("first", "First seen", 95),
                             ("last", "Last seen", 95)]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, padx=10)
        self.tree.bind("<MouseWheel>", self._on_panel_wheel)   # the panel scrolls, not the list

        ttk.Label(f, text="System", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=10, pady=(14, 4))
        self.sys_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.sys_var, wraplength=340, justify=tk.LEFT).pack(anchor=tk.W, padx=10)

    # ── coordinate mapping ────────────────────────────────────────────────
    def _mapping(self):
        fw, fh = self.frame_size
        return fit(fw, fh, *self._display)

    def _to_canvas(self, x, y):
        s, ox, oy = self._mapping()
        return x * s + ox, y * s + oy

    def _to_frame(self, cx, cy):
        s, ox, oy = self._mapping()
        fw, fh = self.frame_size
        return min(max((cx - ox) / s, 0), fw - 1), min(max((cy - oy) / s, 0), fh - 1)

    # ── zone editing ──────────────────────────────────────────────────────
    def toggle_edit(self):
        if self.frame_size is None:
            return
        self.editing = not self.editing
        self.edit_btn.configure(text="Done" if self.editing else "Edit zone")
        if self.editing and self.zone_pts is None:
            fw, fh = self.frame_size   # starting shape: a floor-like trapezoid in the lower part
            self.zone_pts = [[fw * 0.3, fh * 0.55], [fw * 0.7, fh * 0.55], [fw * 0.8, fh * 0.9], [fw * 0.2, fh * 0.9]]
            self.zone.set(self.zone_pts)
        if not self.editing:
            self._save_zone()
        self._draw_zone()

    def clear_zone(self):
        self._reset_zone()
        self._save_zone()

    def _reset_zone(self):
        self.zone_pts = None
        self.editing = False
        self.edit_btn.configure(text="Draw zone")
        self.zone.set(None)
        self._draw_zone()

    # zones.json: {source: {"size": [w, h], "pts": [[x, y] x 4]}} in frame pixels
    def _zone_key(self):
        src = config.CAMERA_INDEX
        return str(Path(src).resolve()) if isinstance(src, str) and Path(src).exists() else str(src)

    def _read_zones(self):
        try:
            return json.loads(Path(config.ZONES_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_zone(self):
        zones = self._read_zones()
        if self.zone_pts is None or self.frame_size is None:
            zones.pop(self._zone_key(), None)
        else:
            zones[self._zone_key()] = {"size": list(self.frame_size), "pts": self.zone_pts}
        path = Path(config.ZONES_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(zones, indent=1), encoding="utf-8")

    def _load_zone(self):
        z = self._read_zones().get(self._zone_key())
        if not z or len(z.get("pts", [])) != 4:
            return
        sx, sy = self.frame_size[0] / z["size"][0], self.frame_size[1] / z["size"][1]
        self.zone_pts = [[x * sx, y * sy] for x, y in z["pts"]]
        self.zone.set(self.zone_pts)
        self.edit_btn.configure(text="Edit zone")

    def reset_db(self):
        if messagebox.askyesno("Reset visitors", "Forget every stored visitor?"):
            self.pipeline.commands.put("reset_db")
            self._thumb_cache.clear()

    def _on_panel_wheel(self, e):
        if not str(e.widget).startswith(str(self.right)):
            return
        sc = self.panel_canvas
        if self.panel.winfo_height() > sc.winfo_height():
            sc.yview_scroll(int(-e.delta / 120), "units")
        return "break"

    def set_source(self, new):
        if isinstance(new, str) and not Path(new).is_file():
            messagebox.showerror("Video missing", f"Not found:\n{new}")
            return
        config.CAMERA_INDEX = new
        self._clear_view()
        self._stopped = False
        self.pipeline.commands.put(("source", new))

    def stop_source(self):
        self._stopped = True         # frames already queued must not repaint the canvas
        self._clear_view()
        self.pipeline.commands.put(("source", None))

    def _clear_view(self):
        self._reset_zone()           # the next source's saved zone loads with its first frame
        self.frame_size = None
        self._thumb_cache.clear()
        self._photo = None
        self.canvas.itemconfigure(self.img_item, image="")

    def clear_panel(self):
        """Empty the panel and count from now on; stored visitors are kept for recognition."""
        self.pipeline.commands.put("clear_view")
        self._recent_key = None
        self.tree.delete(*self.tree.get_children())
        for v in self.vars.values():
            v.set("0")
        self._show_event(None)

    def _draw_zone(self):
        self.canvas.delete("zone")
        if self.zone_pts is None or self.frame_size is None:
            return
        pts = [self._to_canvas(*p) for p in self.zone_pts]
        self.canvas.create_polygon(*[c for p in pts for c in p], outline="#00ffff", fill="", width=2, tags="zone")
        if self.editing:
            for x, y in pts:
                self.canvas.create_oval(x - HANDLE_R, y - HANDLE_R, x + HANDLE_R, y + HANDLE_R,
                                        fill="#00ffff", outline="#003333", tags="zone")

    def _on_press(self, e):
        if not self.editing or self.zone_pts is None:
            return
        for i, p in enumerate(self.zone_pts):
            cx, cy = self._to_canvas(*p)
            if (cx - e.x) ** 2 + (cy - e.y) ** 2 <= (HANDLE_R * 2) ** 2:
                self._drag = ("vertex", i)
                return
        if self._inside(e.x, e.y):
            self._drag = ("move", self._to_frame(e.x, e.y))

    def _on_release(self, e):
        if self._drag:
            self._save_zone()
        self._drag = None

    def _inside(self, x, y):
        pts = [self._to_canvas(*p) for p in self.zone_pts]
        inside, j = False, len(pts) - 1
        for i in range(len(pts)):
            (xi, yi), (xj, yj) = pts[i], pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
                inside = not inside
            j = i
        return inside

    def _on_drag(self, e):
        if not self._drag:
            return
        fx, fy = self._to_frame(e.x, e.y)
        if self._drag[0] == "vertex":
            fw, fh = self.frame_size
            pts = [list(p) for p in self.zone_pts]
            pts[self._drag[1]] = [min(max(fx, 0), fw - 1), min(max(fy, 0), fh - 1)]
            if not _is_simple_quad(pts):
                return   # edges would cross (bow-tie): keep the last valid shape
            self.zone_pts = pts
        else:
            sx, sy = self._drag[1]
            fw, fh = self.frame_size
            xs, ys = [p[0] for p in self.zone_pts], [p[1] for p in self.zone_pts]
            dx = min(max(fx - sx, -min(xs)), fw - 1 - max(xs))
            dy = min(max(fy - sy, -min(ys)), fh - 1 - max(ys))
            self.zone_pts = [[x + dx, y + dy] for x, y in self.zone_pts]
            self._drag = ("move", (sx + dx, sy + dy))
        self.zone.set(self.zone_pts)
        self._draw_zone()

    def _on_resize(self, e):
        self._display = (max(e.width, 1), max(e.height, 1))
        self._draw_zone()

    # ── polling ───────────────────────────────────────────────────────────
    def poll(self):
        try:
            p = self.q.get_nowait()
        except queue.Empty:
            p = None
        if p is not None:
            if "error" in p:
                self.status.configure(text="Error – see message", foreground="#c00")
                if not self._error_shown:
                    self._error_shown = True
                    messagebox.showerror("Pipeline error", p["error"])
            elif "status" in p:
                self.status.configure(text=p["status"], foreground="")
            else:
                self._show(p)
        if not self.stop.is_set():
            self.root.after(config.UI_POLL_MS, self.poll)

    def _show(self, p):
        if self._stopped:
            return
        first = self.frame_size is None
        self.frame_size = p["frame_size"]
        _, ox, oy = self._mapping()
        self._photo = ImageTk.PhotoImage(p["image"])      # must be created on the Tk thread
        self.canvas.itemconfigure(self.img_item, image=self._photo)
        self.canvas.coords(self.img_item, int(ox), int(oy))
        if first:
            self._load_zone()
            self._draw_zone()
        self.canvas.tag_raise("zone")

        s = p["stats"]
        for k, v in self.vars.items():
            v.set(str(s.get(k, "–")))
        self.status.configure(text=f"Counting in: {s['zone_source']}", foreground="")
        self.warn_var.set(s.get("warning", ""))
        self.sys_var.set(f"FPS: {s['fps']:.1f}\nRe-ID model: {s['backend']}\n"
                         f"Match threshold: {s['threshold']:.2f}")
        self.since_var.set(f"since {_fmt(s['since'])}" if s.get("since") else "")
        self._show_recent(s.get("recent", []))
        self._show_event(p.get("event"))

    def _show_recent(self, recent):
        key = tuple((r["id"], r["visits"], r["last_seen"]) for r in recent)
        if key == self._recent_key:
            return
        self._recent_key = key
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(height=max(1, len(recent)))
        for r in recent:
            if r["id"] not in self._thumb_cache and r.get("thumb"):
                self._thumb_cache[r["id"]] = _jpg_to_photo(r["thumb"], LIST_IMG)
            img = self._thumb_cache.get(r["id"])
            self.tree.insert("", tk.END, image=img or "", values=(
                r["id"], r["visits"], _fmt(r["first_seen"]), _fmt(r["last_seen"])))

    def _show_event(self, ev):
        if ev is None:
            if self._event_seq is not None:   # cleared by reset
                self._event_seq = None
                self.event_var.set("Nobody yet")
                self.then_img.configure(image="")
                self.now_img.configure(image="")
            return
        if ev["seq"] == self._event_seq:
            return
        self._event_seq = ev["seq"]
        now_ph = _jpg_to_photo(ev["now"], EVENT_IMG)
        then_ph = _jpg_to_photo(ev["then"], EVENT_IMG)
        self._event_photos = [now_ph, then_ph]   # keep references alive
        when = _fmt(ev["ts"])
        if ev["kind"] == "recognised":
            today = "first time today – audio cue" if ev.get("first_today") else "already seen today"
            self.event_var.set(f"SEEN BEFORE – visitor #{ev['profile_id']}\n"
                               f"match {ev['similarity'] * 100:.0f}%, visit {ev['visits']}  ({when})\n{today}")
            self.event_lbl.configure(fg="#c0288c")
            self.then_img.configure(image=then_ph or "", text="stored photo")
        else:
            closest = f"closest existing {ev['similarity'] * 100:.0f}%" if ev["similarity"] > -1 else "gallery empty"
            self.event_var.set(f"NEW visitor #{ev['profile_id']}\n{closest}  ({when})\nnot seen today – audio cue")
            self.event_lbl.configure(fg="#1d8a1d")
            self.then_img.configure(image="", text="(first time)")
        self.now_img.configure(image=now_ph or "", text="now")

    def on_close(self):
        self.stop.set()
        self.worker.join(timeout=3.0)
        self.root.destroy()


def _is_simple_quad(pts):
    """False if the quadrilateral's opposite edges cross."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def crosses(p1, p2, p3, p4):
        return cross(p3, p4, p1) * cross(p3, p4, p2) < 0 and cross(p1, p2, p3) * cross(p1, p2, p4) < 0

    return not crosses(pts[0], pts[1], pts[2], pts[3]) and not crosses(pts[1], pts[2], pts[3], pts[0])


def _fmt(ts):
    d = dt.datetime.fromtimestamp(ts)
    return d.strftime("%H:%M:%S") if d.date() == dt.date.today() else d.strftime("%d %b %H:%M")


def parse_args():
    ap = argparse.ArgumentParser(description="Pedestrian profiler")
    ap.add_argument("--source", help="webcam index (e.g. 0) or video file / RTSP URL")
    ap.add_argument("--reset-db", action="store_true", help="start with an empty visitor list")
    args = ap.parse_args()
    if args.source is not None:
        config.CAMERA_INDEX = int(args.source) if args.source.isdigit() else args.source
    if args.reset_db:
        config.DB_RESET_ON_START = True


if __name__ == "__main__":
    parse_args()
    root = tk.Tk()
    App(root)
    root.mainloop()
