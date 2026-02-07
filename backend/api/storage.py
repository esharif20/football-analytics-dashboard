import os
import shutil
import subprocess
from pathlib import Path

from .config import settings

LOCAL_STORAGE_DIR = settings.LOCAL_STORAGE_DIR


def _normalize_key(rel_key: str) -> str:
    return rel_key.lstrip("/")


def ensure_dir(sub_path: str) -> Path:
    full = Path(LOCAL_STORAGE_DIR) / os.path.dirname(sub_path)
    full.mkdir(parents=True, exist_ok=True)
    return Path(LOCAL_STORAGE_DIR) / sub_path


def reencode_to_h264(video_path: str) -> None:
    """Re-encode video to H.264 so browsers can play it.

    OpenCV writes mp4v (MPEG-4 Part 2) which HTML5 <video> can't play.
    Converts in-place to H.264 using ffmpeg if available.
    """
    if not shutil.which("ffmpeg"):
        print("[WARNING] ffmpeg not found — uploaded video may not play in browser")
        return

    tmp = video_path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-movflags", "+faststart",
                "-an", tmp,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp, video_path)
        print(f"[INFO] Re-encoded video to H.264: {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] ffmpeg re-encode failed: {e.stderr.decode()[-200:]}")
        if os.path.exists(tmp):
            os.remove(tmp)


async def storage_put(rel_key: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
    """Write file to local storage. Returns {key, url}."""
    key = _normalize_key(rel_key)
    file_path = ensure_dir(key)
    file_path.write_bytes(data)
    url = f"/uploads/{key}"
    return {"key": key, "url": url, "path": str(file_path)}


async def storage_get(rel_key: str) -> dict:
    """Return local URL for a stored file. Returns {key, url}."""
    key = _normalize_key(rel_key)
    url = f"/uploads/{key}"
    return {"key": key, "url": url}
