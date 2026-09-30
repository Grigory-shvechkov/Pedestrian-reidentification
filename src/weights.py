"""Download OSNet re-ID checkpoints (torchreid model zoo, hosted on Google Drive).

Files are small (1–11 MB) and are fetched once into models/. If the automatic
download fails (offline, firewall, Drive quota), the error message tells you
which link to open in a browser and where to save the file.
"""
from __future__ import annotations

import html
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# name → (architecture, Google Drive file id). Ids are the torchreid model-zoo checkpoints.
CATALOG = {
    "osnet_ain_x1_0_msmt17": ("osnet_ain_x1_0", "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal"),
    "osnet_x1_0_msmt17": ("osnet_x1_0", "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"),
    "osnet_x0_25_msmt17": ("osnet_x0_25", "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"),
    "osnet_x1_0_market1501": ("osnet_x1_0", "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"),
    "osnet_x0_25_market1501": ("osnet_x0_25", "1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"),
}

_MIN_BYTES = 200_000  # anything smaller is an HTML error page, not a checkpoint


def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _download_drive(file_id: str, dest: Path):
    try:
        import gdown  # most robust against Drive's changing confirm pages
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)
        if dest.is_file() and dest.stat().st_size > _MIN_BYTES:
            return
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001 – fall back to urllib
        log.info("gdown failed (%s), trying direct download", e)

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    with opener.open(drive_url(file_id), timeout=60) as r:
        data = r.read()
    # Drive sometimes answers with a "can't scan for viruses" form instead of the file.
    if data[:200].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        page = data.decode("utf-8", "ignore")
        action = re.search(r'<form[^>]+action="([^"]+)"', page)
        fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
        if not action:
            raise RuntimeError("Google Drive returned a page instead of the file (quota or access issue)")
        url = html.unescape(action.group(1)) + "?" + urllib.parse.urlencode(fields)
        with opener.open(url, timeout=120) as r:
            data = r.read()
    if len(data) < _MIN_BYTES:
        raise RuntimeError(f"Downloaded file is only {len(data)} bytes – not a checkpoint")
    dest.write_bytes(data)


def ensure_weights(name: str, models_dir: Path) -> tuple[str, Path]:
    """Return (architecture, checkpoint path), downloading the checkpoint if needed."""
    if name not in CATALOG:
        raise ValueError(f"Unknown REID_MODEL {name!r}; choose from {sorted(CATALOG)}")
    arch, file_id = CATALOG[name]
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pth"
    if path.is_file() and path.stat().st_size > _MIN_BYTES:
        return arch, path

    log.info("Downloading %s re-ID weights (one-time)…", name)
    tmp = path.with_suffix(".part")
    try:
        _download_drive(file_id, tmp)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {name} weights: {e}\n"
            f"Download manually from {drive_url(file_id)}\n"
            f"and save it as {path}"
        ) from e
    log.info("Saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return arch, path
