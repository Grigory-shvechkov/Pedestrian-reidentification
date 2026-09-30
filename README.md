# Pedestrian Profiler

Detects people on a camera and tells you whether each one has been seen before.

For every person who walks through, the app shows either **NEW visitor #N** or
**SEEN BEFORE – visitor #N (match %, visit count)**, with the stored photo next
to the current one so you can check the match with your own eyes.

![showcase](videos/showcase_preview.gif)

## Videos included

| file | what it is |
|---|---|
| `videos/test_reid.mp4` | **input** for testing (4.7 min): six people walk up to a table one at a time, then the same six again – mirrored, re-lit and zoomed so the repeat isn't pixel-identical. Correct result: 6 × NEW, then 6 × SEEN BEFORE. |
| `videos/showcase.mp4` | **output**: what the system actually produced on that video (2× speed) – boxes and labels on the left; on the right the latest decision with the stored photo next to the current one, plus visitor counts. |
| `videos/showcase_preview.gif` | a few seconds of the showcase, shown at the top of this README. |

Measured on the test video with the default model: every person stays one track,
and all six returning people are recognised (similarity 0.95–0.97). One mistake
remains and is visible in the showcase: the two men dressed entirely in black,
both seen only from behind, score 0.88–0.93 against each other, so the second is
labelled SEEN BEFORE as the first. That is the honest limit of clothing-based
recognition when no face is visible.

The repeat pass is easier than real life. For a realistic check, record yourself
walking past your own camera twice, and once more in a different jacket.

Source clip: Intel IoT DevKit sample videos, CC BY 4.0 (`data/VIDEO_CREDITS.txt`).
`tools/make_test_video.py` rebuilds the test video; `tools/render_showcase.py`
re-renders the showcase – point it at your own clip with `--input`.

## Setup in VS Code (Windows)

1. Open the folder: **File → Open Folder… → `pedestrian-profiler`**. Accept the
   recommended Python extensions if VS Code asks.
2. Run `setup_venv.bat` once (double-click, or **Terminal → Run Task… → Setup**).
   It creates `.venv` and installs everything.
3. **Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`**.
4. **Run and Debug (Ctrl+Shift+D)** → pick a configuration → **F5**:

| configuration | does |
|---|---|
| ▶ Profiler – test video | runs the app on `videos/test_reid.mp4` – start here |
| ▶ Profiler – webcam | live camera (`CAMERA_INDEX` in `config.py`) |
| ▶ Profiler – webcam, keep visitors | same, but remembers visitors from earlier runs |
| ▶ Profiler – choose video… | asks for a file path |
| ✔ Self-test | checks camera, detector, re-ID model and database, prints PASS/FAIL |
| 🎬 Render showcase video | re-creates `videos/showcase.mp4` with your installed models |
| 📏 Calibrate threshold | picks the match threshold from your own labelled crops |

From a terminal the same things are `python main.py --source videos/test_reid.mp4`,
`python main.py`, `python tools/selftest.py`, and so on.

First run downloads two small model files: YOLOv8 (by ultralytics) and the OSNet
re-ID weights into `models/`. If the OSNet download is blocked, the app still
runs on a weaker fallback model, shows a red warning, and the log prints the
link to download the file by hand and where to save it.

NVIDIA GPU: install the CUDA build of PyTorch first (pytorch.org → "Get started"),
then `pip install -r requirements.txt`. CPU works too.

## How it decides "seen before"

1. **Detect** people with YOLOv8 (~10 times per second).
2. **Track** each person across frames (Kalman filter + ByteTrack-style matching),
   so one walk-through = one track.
3. **Collect** ~5 good crops per track. Crops that are too small, cut off at the
   side of the frame, blurry or covered by another person are skipped. The box label
   shows why (`wait: edge`, `wait: blurry`, …).
4. **Embed** each crop: OSNet (a network trained specifically to re-identify people)
   plus a clothing-colour descriptor, averaged over the crops.
5. **Compare** against every stored visitor (cosine similarity, several stored
   looks per visitor).
   * above the threshold → **SEEN BEFORE**, visit count +1 (if they were gone ≥ `REVISIT_GAP_S`)
   * just below it → keeps collecting crops before deciding
   * otherwise → **NEW**, stored with a thumbnail.

## Box colours / labels

| label | meaning |
|---|---|
| `T5 wait: edge` (orange) | tracked, waiting for a usable crop |
| `T5 3/5` (yellow) | collecting crops |
| `T5 verifying 60%` (blue) | borderline, collecting more |
| `#3 NEW` (green) | first time seen, stored |
| `#3 SEEN BEFORE 82%` (pink) | matched a stored visitor |
| `#3` | same visitor, same visit (tracking hiccup, not counted twice) |

## Tuning (config.py)

* `SIMILARITY_THRESHOLD` (0.80 for the default model) – the main knob. Same person shown
  as NEW → lower it. Different people merged → raise it. For a data-driven value, set
  `SAVE_CROPS = True`, collect a day, clean up `data/crops/`, then run
  `python tools\calibrate_threshold.py data\crops`.
* `REID_MODEL` – `osnet_ain_x1_0_msmt17` is stronger but ~4× slower; its threshold
  needs calibrating (start around 0.75).
* `EDGE_SIDES`, `MIN_SHARPNESS`, `MIN_PERSON_HEIGHT` – if people stay on `wait: …` too long.
* `REVISIT_GAP_S` – how long someone must be gone for a return to count as a new visit.
* "Draw zone" in the UI – only count people inside an area (e.g. the entrance).

## Known limits

Recognition is based on appearance, mostly clothing. Same-day returns work well;
someone returning on another day in different clothes will usually be NEW.

## Privacy

Stored appearance data and thumbnails of identifiable people are personal data in
many jurisdictions (e.g. GDPR). `PROFILE_RETENTION_DAYS` deletes visitors not seen
recently, `STORE_THUMBNAILS = False` stops saving photos, and the default
`DB_RESET_ON_START = True` keeps nothing between runs.

## Files

| File | Role |
|---|---|
| `main.py` | Tkinter UI (`--source`, `--keep-db`) |
| `src/pipeline.py` | worker loop: detect → track → collect → decide |
| `src/camera.py` | camera/video reader (always newest frame) |
| `src/detector.py` | YOLOv8 person detection |
| `src/tracker.py` | multi-person tracker |
| `src/quality.py` | crop quality checks |
| `src/embedder.py` | OSNet + colour appearance embedding |
| `src/reid_models/` | OSNet architecture (from torchreid, MIT licence) |
| `src/weights.py` | one-time weight download |
| `src/re_identifier.py` | "seen before?" decision + visitor gallery |
| `src/database.py` | SQLite storage |
| `src/visualizer.py` | drawing on the video |
| `tools/selftest.py` | component check |
| `tools/calibrate_threshold.py` | pick the threshold from your own data |
| `tools/make_test_video.py` | builds `videos/test_reid.mp4` |
| `tools/render_showcase.py` | runs the pipeline on a video, writes `videos/showcase.mp4` |
| `src/videoio.py` | H.264 encoding helper |
