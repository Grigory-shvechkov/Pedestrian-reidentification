"""All tunables for the pedestrian profiler.

Values marked (CALIBRATE) depend on your camera, lighting and embedding
backend. Run `python tools/calibrate_threshold.py` on crops from your own
camera before trusting them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0            # int for a webcam, or a str path / RTSP URL
CAMERA_BACKEND = "dshow"    # "dshow" (fast open on Windows) | "msmf" | "any"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ─── Detection (YOLOv8) ──────────────────────────────────────────────────────
# A *-seg model gives person masks, which makes the colour descriptor ignore
# background pixels. Plain "yolov8n.pt" also works (falls back to a centre mask).
DETECTOR_WEIGHTS = "yolov8n-seg.pt"
DETECT_IMGSZ = 640
DETECT_HZ = 10.0            # detection cycles per second; frames in between are skipped
PERSON_CONFIDENCE = 0.55    # detections >= this can start tracks
DETECT_LOW_CONFIDENCE = 0.25  # ByteTrack 2nd stage: keeps tracks alive through occlusion
DEVICE = "auto"             # "auto" | "cpu" | "cuda" | "0"

# ─── Zone (optional; drawn in the UI, otherwise the full frame is used) ───────
ZONE_ANCHOR = "center"      # which point of the bbox must be inside the zone: "center" | "bottom"
ROI_PAD = 40                # px padding around the zone's bounding rect for the YOLO crop

# ─── Tracker (Kalman + Hungarian, ByteTrack-style) ───────────────────────────
TRACK_MATCH_IOU = 0.30      # 1st stage (high-confidence detections)
TRACK_LOW_MATCH_IOU = 0.50  # 2nd stage (low-confidence detections)
TRACK_MIN_HITS = 3          # consecutive hits before a track is confirmed
TRACK_MAX_CENTER_DIST = 0.6 # 3rd stage: match by centre distance (× box height) when IoU fails
TRACK_MAX_AGE = 10          # detection cycles a lost track survives (10 at 10 Hz ≈ 1 s)

# ─── Crop quality gates (applied before any embedding) ───────────────────────
MIN_BBOX_AREA = 6000        # px²
MIN_PERSON_HEIGHT = 100     # px
MAX_ASPECT_RATIO = 1.15     # width / height
MIN_SHARPNESS = 15.0        # variance of Laplacian on a 128-px-high grey crop (CALIBRATE)
EDGE_MARGIN = 4             # px from the frame/zone edge counts as "cut off"
EDGE_SIDES = ("left", "right")  # which edges reject a crop; add "top"/"bottom" if the camera sees full bodies
MAX_OCCLUSION = 0.30        # max fraction of the box covered by another person's box
QUALITY_RELAX_AFTER_S = 3.0 # if a person yields no good crop for this long, accept blurry/edge crops

# ─── Sample collection per track ─────────────────────────────────────────────
SAMPLES_TO_DECIDE = 5       # good crops needed before the first re-ID decision (was STABLE_FRAMES)
MAX_SAMPLES = 10            # keep collecting up to this many while a decision is ambiguous
MIN_SAMPLES = 2             # minimum to decide at timeout / when the track dies
MAX_DECISION_WAIT_S = 6.0   # force a decision this long after the first good crop

# ─── Embedder ────────────────────────────────────────────────────────────────
# "auto": OSNet (weights auto-downloaded to MODELS_DIR on first run) → MobileNet fallback.
# "osnet" / "mobilenet": force one.
EMBEDDER_BACKEND = "auto"
# osnet_x0_25_msmt17    – fast; threshold validated on videos/test_reid.mp4 (default)
# osnet_ain_x1_0_msmt17 – stronger on new cameras, ~4× slower; recalibrate the threshold
# also: osnet_x1_0_msmt17, osnet_x1_0_market1501, osnet_x0_25_market1501
REID_MODEL = "osnet_x0_25_msmt17"
MODELS_DIR = ROOT / "models"
REID_INPUT_HW = (256, 128)
FLIP_TTA = True             # average the embedding with its horizontal flip

# Weight of the colour descriptor in the fused cosine similarity:
#   cos(fused) = (1 - w) * cos(deep) + w * cos(colour)
COLOR_WEIGHT = {"osnet": 0.25, "mobilenet": 0.50}
WHITE_BALANCE = True        # grey-world gains estimated on the whole frame, not the crop

# ─── Re-identification ───────────────────────────────────────────────────────
# Cosine threshold for "seen before", per model (falls back to the family key). (CALIBRATE)
# 0.80 for osnet_x0_25 comes from the test video: same person 0.94–0.97, different
# people ≤ 0.76, except two men both dressed in black seen from behind (0.88–0.93).
SIMILARITY_THRESHOLD = {"osnet_x0_25_msmt17": 0.80, "osnet": 0.75, "mobilenet": 0.85}
AMBIGUOUS_BAND = 0.05       # best sim in [thr - band, thr) → keep collecting samples
MIN_MARGIN = 0.03           # best must beat 2nd-best profile by this much (else keep collecting)
REVISIT_GAP_S = 10.0        # profile must be unseen this long for a match to count as a new visit
MAX_PROTOTYPES = 8          # stored appearance prototypes per visitor
PROTOTYPE_NOVELTY = 0.90    # add a prototype only if its max sim to existing ones is below this
PROTOTYPE_UPDATE_MIN_SIM_MARGIN = 0.05  # ...and the match was at least thr + this (avoids drift)
BANNER_SECONDS = 3.0

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "data" / "visitors_v2.db"   # new schema; old JSON-embedding DB is not reused
DB_RESET_ON_START = True
# Visitors unseen for this many days are deleted on start-up (None = keep forever).
# Keeps the gallery small (fewer false matches) and limits how long appearance data is held.
PROFILE_RETENTION_DAYS = 30

STORE_THUMBNAILS = True     # keep one small photo per visitor so you can verify matches in the UI

# ─── Dataset building / debugging ────────────────────────────────────────────
SAVE_CROPS = False          # save decided tracks' crops to CROPS_DIR/<profile_id>/ for calibration / fine-tuning
CROPS_DIR = ROOT / "data" / "crops"

# ─── Integrations ────────────────────────────────────────────────────────────
API_URL = None              # e.g. "http://localhost:8000/visitors"; None → log only
API_TIMEOUT_S = 3.0

# ─── UI ──────────────────────────────────────────────────────────────────────
UI_POLL_MS = 30
STATS_REFRESH_S = 0.5
