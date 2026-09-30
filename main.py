"""Pedestrian profiler – detects people and tells you whether they've been seen before.

    python main.py                      # webcam CAMERA_INDEX from config.py
    python main.py --source 1           # another webcam
    python main.py --source clip.mp4    # a recorded video (great for testing)
    python main.py --keep-db            # don't wipe visitors on start

Left: live view. Optional zone: "Draw zone", drag the corners (or drag inside
to move it); only people inside are counted. Right: who was just seen, whether
they were seen before (stored photo vs. now), and visitor stats.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import queue
import threading
import tkinter as tk
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
        self.status = ttk.Label(bar, text="Starting…")
        self.status.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(left, bg="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        paned.add(left, weight=4)

        # scrollable right panel
        right = ttk.Frame(paned, width=380)
        sc = tk.Canvas(right, highlightthickness=0, width=360)
        sb = ttk.Scrollbar(right, orient=tk.VERTICAL, command=sc.yview)
        f = self.panel = ttk.Frame(sc)
        f.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.create_window((0, 0), window=f, anchor=tk.NW)
        sc.configure(yscrollcommand=sb.set)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.bind("<Enter>", lambda e: sc.bind_all("<MouseWheel>",
                                                 lambda ev: sc.yview_scroll(int(-ev.delta / 120), "units")))
        sc.bind("<Leave>", lambda e: sc.unbind_all("<MouseWheel>"))
        paned.add(right, weight=1)

        self.warn_var = tk.StringVar(value="")
        self.warn = tk.Label(f, textvariable=self.warn_var, fg="#b00", wraplength=340, justify=tk.LEFT)
        self.warn.pack(anchor=tk.W, padx=10)

        # latest decision: then / now
        ttk.Label(f, text="Last person", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, padx=10, pady=(10, 2))
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

        ttk.Label(f, text="Visitors", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, padx=10, pady=(12, 4))
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
            fw, fh = self.frame_size
            mx, my = fw * 0.25, fh * 0.1
            self.zone_pts = [[mx, my], [fw - mx, my], [fw - mx, fh - my], [mx, fh - my]]
            self.zone.set(self.zone_pts)
        self._draw_zone()

    def clear_zone(self):
        self.zone_pts = None
        self.editing = False
        self.edit_btn.configure(text="Draw zone")
        self.zone.set(None)
        self._draw_zone()

    def reset_db(self):
        if messagebox.askyesno("Reset visitors", "Forget every stored visitor?"):
            self.pipeline.commands.put("reset_db")
            self._thumb_cache.clear()

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
            self.zone_pts[self._drag[1]] = [fx, fy]
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
        first = self.frame_size is None
        self.frame_size = p["frame_size"]
        _, ox, oy = self._mapping()
        self._photo = ImageTk.PhotoImage(p["image"])      # must be created on the Tk thread
        self.canvas.itemconfigure(self.img_item, image=self._photo)
        self.canvas.coords(self.img_item, int(ox), int(oy))
        if first:
            self._draw_zone()
        self.canvas.tag_raise("zone")

        s = p["stats"]
        for k, v in self.vars.items():
            v.set(str(s.get(k, "–")))
        self.status.configure(text=f"Counting in: {s['zone_source']}", foreground="")
        self.warn_var.set(s.get("warning", ""))
        self.sys_var.set(f"FPS: {s['fps']:.1f}\nRe-ID model: {s['backend']}\n"
                         f"Match threshold: {s['threshold']:.2f}")
        self._show_recent(s.get("recent", []))
        self._show_event(p.get("event"))

    def _show_recent(self, recent):
        key = tuple((r["id"], r["visits"], r["last_seen"]) for r in recent)
        if key == self._recent_key:
            return
        self._recent_key = key
        self.tree.delete(*self.tree.get_children())
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
            self.event_var.set(f"SEEN BEFORE – visitor #{ev['profile_id']}\n"
                               f"match {ev['similarity'] * 100:.0f}%, visit {ev['visits']}  ({when})")
            self.event_lbl.configure(fg="#c0288c")
            self.then_img.configure(image=then_ph or "", text="stored photo")
        else:
            closest = f"closest existing {ev['similarity'] * 100:.0f}%" if ev["similarity"] > -1 else "gallery empty"
            self.event_var.set(f"NEW visitor #{ev['profile_id']}\n{closest}  ({when})")
            self.event_lbl.configure(fg="#1d8a1d")
            self.then_img.configure(image="", text="(first time)")
        self.now_img.configure(image=now_ph or "", text="now")

    def on_close(self):
        self.stop.set()
        self.worker.join(timeout=3.0)
        self.root.destroy()


def _fmt(ts):
    d = dt.datetime.fromtimestamp(ts)
    return d.strftime("%H:%M:%S") if d.date() == dt.date.today() else d.strftime("%d %b %H:%M")


def parse_args():
    ap = argparse.ArgumentParser(description="Pedestrian profiler")
    ap.add_argument("--source", help="webcam index (e.g. 0) or video file / RTSP URL")
    ap.add_argument("--keep-db", action="store_true", help="keep visitors from previous runs")
    args = ap.parse_args()
    if args.source is not None:
        config.CAMERA_INDEX = int(args.source) if args.source.isdigit() else args.source
    if args.keep_db:
        config.DB_RESET_ON_START = False


if __name__ == "__main__":
    parse_args()
    root = tk.Tk()
    App(root)
    root.mainloop()
