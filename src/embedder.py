"""Person appearance embedder.

embedding = [ sqrt(1-w) · deep_feature , sqrt(w) · colour_feature ]

Both halves are L2-normalised on their own, so the fused vector has unit norm and
    cos(a, b) = (1 - w) · cos_deep(a, b) + w · cos_colour(a, b)
i.e. `w` is an exact, interpretable weight (the old "×3 scaling" was not).

Deep backends (first that works in "auto" mode):
  osnet     – OSNet trained for person re-ID (MSMT17 / Market-1501), running in plain
              PyTorch from a vendored copy of the architecture. Weights (1–11 MB) are
              downloaded automatically on first run. No torchreid install needed.
  mobilenet – ImageNet MobileNetV3-Small. Not trained for re-ID; used only if OSNet
              weights are unavailable (e.g. offline first run). The UI shows a warning.

Colour descriptor (112-d), per body part (upper / lower):
  * pixels restricted to the person mask (YOLO-seg) or a central column band,
  * frame-level grey-world white balance to cancel lighting colour casts,
  * chromatic pixels → 12×4 hue-saturation histogram (hue circularly smoothed),
    achromatic pixels (black/grey/white clothes, where hue is noise) → 8-bin value histogram,
  * joint normalisation (so "how much is grey vs coloured" is encoded), then sqrt
    (Hellinger), so cosine == Bhattacharyya coefficient.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _prep_batch(crops_bgr, hw, flip) -> np.ndarray:
    H, W = hw
    batch = []
    for c in crops_bgr:
        img = cv2.cvtColor(cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        batch.append(((img.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1))
    if flip:
        batch += [b[:, :, ::-1] for b in batch]
    return np.ascontiguousarray(np.stack(batch), dtype=np.float32)


def _merge_flip(feats: np.ndarray, n: int, flip: bool) -> np.ndarray:
    return l2n(feats[:n] + feats[n:]) if flip else l2n(feats)


# ─── Deep backends ───────────────────────────────────────────────────────────
class _OSNetBackend:
    family = "osnet"

    def __init__(self, model_name: str, models_dir, hw, flip, device):
        import torch

        from .reid_models import build
        from .weights import ensure_weights

        arch, path = ensure_weights(model_name, models_dir)
        self.torch = torch
        self.dev = torch.device("cuda" if device in ("cuda", "0") and torch.cuda.is_available() else "cpu")
        model = build(arch)
        self._load_checkpoint(model, path)
        self.model = model.eval().to(self.dev)
        self.half = self.dev.type == "cuda"
        if self.half:
            self.model.half()
        self.hw, self.flip = hw, flip
        self.name = f"{model_name}"
        self.dim = int(self([np.zeros((hw[0], hw[1], 3), np.uint8)]).shape[1])
        log.info("Embedder: %s on %s (%d-d)", self.name, self.dev, self.dim)

    def _load_checkpoint(self, model, path):
        torch = self.torch
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 – older torchreid checkpoints pickle extra metadata
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        own = model.state_dict()
        loaded, skipped = {}, []
        for k, v in state.items():
            k = k[7:] if k.startswith("module.") else k
            if k in own and own[k].shape == v.shape:
                loaded[k] = v
            else:
                skipped.append(k)
        needed = [k for k in own if not k.startswith("classifier")]
        missing = [k for k in needed if k not in loaded]
        if len(missing) > 0.05 * len(needed):
            raise RuntimeError(f"{path.name} does not fit this architecture "
                               f"({len(missing)}/{len(needed)} layers missing) – wrong or corrupt file?")
        own.update(loaded)
        model.load_state_dict(own)
        log.info("Loaded %s: %d tensors (%d ignored, e.g. classifier)", path.name, len(loaded), len(skipped))

    def __call__(self, crops):
        x = self.torch.from_numpy(_prep_batch(crops, self.hw, self.flip)).to(self.dev)
        if self.half:
            x = x.half()
        with self.torch.inference_mode():
            f = self.model(x).float().cpu().numpy()
        return _merge_flip(f, len(crops), self.flip)


class _MobileNetBackend:
    family = "mobilenet"

    def __init__(self, hw, flip, device):
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.torch = torch
        self.dev = torch.device("cuda" if device in ("cuda", "0") and torch.cuda.is_available() else "cpu")
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1),
                                         torch.nn.Flatten()).eval().to(self.dev)
        self.hw, self.flip = hw, flip
        self.name = "mobilenet_v3_small_imagenet"
        self.dim = 576
        log.warning("Embedder: falling back to ImageNet MobileNetV3 — re-ID accuracy will be limited.")

    def __call__(self, crops):
        x = self.torch.from_numpy(_prep_batch(crops, self.hw, self.flip)).to(self.dev)
        with self.torch.inference_mode():
            f = self.model(x).cpu().numpy()
        return _merge_flip(f, len(crops), self.flip)


# ─── Colour descriptor ───────────────────────────────────────────────────────
_PARTS = ((0.12, 0.52), (0.52, 0.95))   # upper body, lower body (fractions of box height)
_H_BINS, _S_BINS, _V_BINS = 12, 4, 8
_S_MIN, _V_MIN = 50, 40                 # below → achromatic (hue unreliable)
COLOR_DIM = len(_PARTS) * (_H_BINS * _S_BINS + _V_BINS)


def white_balance_gains(frame: np.ndarray) -> np.ndarray:
    """Grey-world gains from the whole (downscaled) frame, not the crop —
    estimating on the crop would cancel the person's own clothing colour."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    means = small.mean(0) + 1e-6
    return np.clip(means.mean() / means, 0.5, 2.0).astype(np.float32)


def color_descriptor(crop: np.ndarray, mask: np.ndarray | None, gains: np.ndarray | None) -> np.ndarray:
    h, w = crop.shape[:2]
    img = crop
    if gains is not None:
        img = np.clip(crop.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    if mask is None:  # no segmentation: keep the central 60% of columns (drops most background)
        mask = np.zeros((h, w), np.uint8)
        mask[:, int(w * 0.2):max(int(w * 0.8), int(w * 0.2) + 1)] = 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    chrom = ((hsv[..., 1] >= _S_MIN) & (hsv[..., 2] >= _V_MIN) & (mask > 0)).astype(np.uint8)
    achrom = ((chrom == 0) & (mask > 0)).astype(np.uint8)

    feats = []
    for a, b in _PARTS:
        y1, y2 = int(h * a), max(int(h * b), int(h * a) + 1)
        p_hsv = hsv[y1:y2]
        hs = cv2.calcHist([p_hsv], [0, 1], chrom[y1:y2], [_H_BINS, _S_BINS], [0, 180, _S_MIN, 256])
        hs = 0.5 * hs + 0.25 * np.roll(hs, 1, axis=0) + 0.25 * np.roll(hs, -1, axis=0)  # hue wraps
        v = cv2.calcHist([p_hsv], [2], achrom[y1:y2], [_V_BINS], [0, 256])
        vec = np.concatenate([hs.ravel(), v.ravel()]).astype(np.float32)
        s = vec.sum()
        vec = np.sqrt(vec / s) if s > 0 else np.full_like(vec, 1.0 / np.sqrt(len(vec)))
        feats.append(vec)
    return l2n(np.concatenate(feats))


# ─── Public API ──────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, cfg, device: str = "cpu"):
        self.deep = self._build_backend(cfg, device)
        fam = self.deep.family
        self.family = fam
        self.w = float(cfg.COLOR_WEIGHT[fam] if isinstance(cfg.COLOR_WEIGHT, dict) else cfg.COLOR_WEIGHT)
        thr = cfg.SIMILARITY_THRESHOLD
        if isinstance(thr, dict):
            thr = thr.get(self.deep.name, thr.get(fam))
        self.threshold = float(thr)
        self.dim = self.deep.dim + COLOR_DIM
        self.warning = getattr(self.deep, "warning", "")
        # Stored in the DB: embeddings from a different backend/weighting are not comparable.
        self.signature = f"{self.deep.name}|color{COLOR_DIM}|w{self.w:.2f}"

    @staticmethod
    def _build_backend(cfg, device):
        choice = cfg.EMBEDDER_BACKEND
        hw, flip = tuple(cfg.REID_INPUT_HW), cfg.FLIP_TTA
        attempts = {
            "osnet": lambda: _OSNetBackend(cfg.REID_MODEL, cfg.MODELS_DIR, hw, flip, device),
            "mobilenet": lambda: _MobileNetBackend(hw, flip, device),
        }
        order = ["osnet", "mobilenet"] if choice == "auto" else [choice]
        errors = []
        for name in order:
            try:
                backend = attempts[name]()
                if name == "mobilenet" and errors:
                    backend.warning = ("Using fallback MobileNet – re-ID accuracy is limited.\n" + errors[0])
                return backend
            except Exception as e:  # noqa: BLE001 – report and try the next backend
                errors.append(f"{name}: {e}")
                log.warning("Embedder backend %s unavailable: %s", name, e)
        raise RuntimeError("No embedder backend available:\n  " + "\n  ".join(errors))

    def embed(self, crops: list[np.ndarray], masks: list[np.ndarray | None] | None = None,
              gains: np.ndarray | None = None) -> np.ndarray:
        """crops: BGR person crops. Returns (N, dim) float32, unit-norm rows."""
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        masks = masks or [None] * len(crops)
        deep = self.deep(crops).astype(np.float32)
        col = np.stack([color_descriptor(c, m, gains) for c, m in zip(crops, masks)])
        return np.concatenate([np.sqrt(1 - self.w) * deep, np.sqrt(self.w) * col], axis=1).astype(np.float32)
