import os
import mimetypes
from typing import Dict, Any, List

from preprocess.video_loader import process_video
from preprocess.audio_loader import process_audio

SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

def detect_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_VIDEO:
        return "video"
    if ext in SUPPORTED_AUDIO:
        return "audio"
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        if "video" in mime:
            return "video"
        if "audio" in mime:
            return "audio"
    return "unknown"

def route_input(file_path: str) -> Dict[str, Any]:
    kind = detect_type(file_path)
    if kind == "video":
        meta = process_video(file_path)
        return {"type": "video", "source": os.path.basename(file_path), "metadata": meta}
    elif kind == "audio":
        meta = process_audio(file_path)
        return {"type": "audio", "source": os.path.basename(file_path), "metadata": meta}
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def bulk_route(files: List[str]) -> List[Dict[str, Any]]:
    results = []
    for f in files:
        try:
            results.append(route_input(f))
        except Exception as e:
            print(f"[Router] Skipping {f}: {e}")
    return results