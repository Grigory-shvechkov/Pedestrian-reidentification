"""Pick SIMILARITY_THRESHOLD from data instead of guessing.

Embeds labelled crops with exactly the embedder configured in config.py,
then measures how similar the same person is across *different* tracklets
(genuine pairs) versus different people (impostor pairs).

Layouts
  folders (default) – <root>/<identity>/<tracklet>_<n>.jpg
                      This is what SAVE_CROPS=True writes to data/crops/.
                      Those labels come from the system itself, so review them
                      first: merge folders that are the same person, move
                      mistakes out. Calibrating on uncorrected auto-labels just
                      re-confirms the current threshold.
  market            – Market-1501 / MSMT-style names: 0002_c1s1_000451_03.jpg
                      (identity 0000 and -1 are skipped). Useful as a sanity
                      check, but numbers from your own camera are what matter —
                      and OSNet weights trained on Market will look optimistic on it.

Examples
  python tools/calibrate_threshold.py data/crops --agg 5
  python tools/calibrate_threshold.py D:/Market-1501/bounding_box_test --layout market --max-ids 300
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.detector import resolve_device  # noqa: E402
from src.embedder import Embedder, l2n  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(root: Path, layout: str):
    items = []
    if layout == "market":
        for p in root.iterdir():
            if p.suffix.lower() not in EXTS:
                continue
            parts = p.stem.split("_")
            if len(parts) < 2 or parts[0] in ("0000", "-1"):
                continue
            items.append((parts[0], parts[1], p))          # identity, camera+sequence, path
    else:
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            for p in d.iterdir():
                if p.suffix.lower() in EXTS:
                    tracklet = p.stem.rsplit("_", 1)[0]
                    items.append((d.name, tracklet, p))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--layout", choices=["folders", "market"], default="folders")
    ap.add_argument("--agg", type=int, default=config.SAMPLES_TO_DECIDE,
                    help="average this many crops per tracklet chunk, like the pipeline does")
    ap.add_argument("--far", type=float, default=0.01, help="target false-match rate per comparison")
    ap.add_argument("--max-ids", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = collect(args.root, args.layout)
    ids = sorted({i for i, _, _ in items})
    random.Random(args.seed).shuffle(ids)
    keep = set(ids[:args.max_ids])
    items = [x for x in items if x[0] in keep]
    if len(keep) < 2:
        sys.exit("Need at least two identities.")
    print(f"{len(items)} crops, {len(keep)} identities")

    emb = Embedder(config, resolve_device(config.DEVICE))
    print(f"Embedder: {emb.signature}  (current threshold {emb.threshold:.2f})")

    feats, kept = [], []
    for i in range(0, len(items), args.batch):
        batch = [(it, cv2.imread(str(it[2]))) for it in items[i:i + args.batch]]
        batch = [(it, c) for it, c in batch if c is not None and c.size]
        if batch:
            feats.append(emb.embed([c for _, c in batch]))
            kept += [it for it, _ in batch]
        print(f"\r  embedded {min(i + args.batch, len(items))}/{len(items)}", end="")
    print()
    feats, items = np.concatenate(feats), kept

    groups = defaultdict(list)
    for (pid, trk, _), f in zip(items, feats):
        groups[(pid, trk)].append(f)
    units, labels, tracks = [], [], []
    for (pid, trk), fs in groups.items():
        for j in range(0, len(fs), max(1, args.agg)):
            chunk = fs[j:j + args.agg]
            units.append(l2n(np.mean(chunk, 0)))
            labels.append(pid)
            tracks.append((pid, trk))
    U = np.stack(units)
    S = U @ U.T
    lab = np.array(labels)
    trk = np.array([hash(t) for t in tracks])
    iu = np.triu_indices(len(U), 1)
    same_id = lab[iu[0]] == lab[iu[1]]
    same_trk = trk[iu[0]] == trk[iu[1]]
    sims = S[iu]
    genuine = sims[same_id & ~same_trk]
    impostor = sims[~same_id]
    if len(genuine) == 0:
        sys.exit("No genuine cross-tracklet pairs: each identity needs crops from ≥2 tracklets/cameras.")

    print(f"\n{len(genuine)} genuine pairs  mean {genuine.mean():.3f} ± {genuine.std():.3f}")
    print(f"{len(impostor)} impostor pairs mean {impostor.mean():.3f} ± {impostor.std():.3f}\n")
    print(" thr   false-match  false-non-match")
    ts = np.round(np.arange(0.20, 0.96, 0.01), 2)
    fmr = np.array([(impostor >= t).mean() for t in ts])
    fnmr = np.array([(genuine < t).mean() for t in ts])
    for t, a, b in zip(ts, fmr, fnmr):
        if round(t * 100) % 5 == 0:
            print(f" {t:.2f}   {a * 100:8.2f}%   {b * 100:8.2f}%")

    eer_i = int(np.argmin(np.abs(fmr - fnmr)))
    ok = np.where(fmr <= args.far)[0]
    rec = ts[ok[0]] if len(ok) else ts[-1]
    print(f"\nEqual-error rate ≈ {(fmr[eer_i] + fnmr[eer_i]) * 50:.1f}% at {ts[eer_i]:.2f}")
    print(f"Suggested threshold for ≤{args.far * 100:.1f}% false matches per comparison: {rec:.2f} "
          f"(misses {(genuine < rec).mean() * 100:.1f}% of returning visitors)")
    f = (impostor >= rec).mean()
    for g in (50, 500, 5000):
        print(f"  with {g:>5} stored visitors, P(a newcomer wrongly matches someone) ≈ "
              f"{(1 - (1 - f) ** g) * 100:.1f}%")
    print("\nThe bigger the gallery, the stricter the threshold must be — or prune old visitors.")


if __name__ == "__main__":
    main()
