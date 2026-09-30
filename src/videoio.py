"""Small video helpers shared by the tools."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def finalize_mp4(tmp: Path, dst: Path, crf: int = 26) -> str:
    """Re-encode an OpenCV-written MPEG-4 file to H.264 if `imageio-ffmpeg` is
    installed (plays in Windows' Films & TV, browsers, VS Code, phones), else keep it.
    Returns the codec used."""
    try:
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
               "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        Path(tmp).unlink()
        return "H.264"
    except Exception as e:  # noqa: BLE001
        print(f"(keeping MPEG-4 – install imageio-ffmpeg for H.264: {e})")
        shutil.move(str(tmp), str(dst))
        return "MPEG-4"
