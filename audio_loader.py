
import os
import json
from typing import List, Dict
import librosa

def process_audio(
    audio_path: str,
    output_json: str = "outputs/logs/audio_segments.json",
    segment_duration: float = 2.0,
) -> List[Dict]:
    """
    Split audio into fixed-duration segments and record metadata.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    segments = []
    t = 0.0
    while t < duration - 1e-9:
        start = t
        end = min(t + segment_duration, duration)
        segments.append(
            {
                "source": os.path.basename(audio_path),
                "start_time_sec": round(float(start), 3),
                "end_time_sec": round(float(end), 3),
                "duration_sec": round(float(end - start), 3),
                "sample_rate": int(sr),
            }
        )
        t += segment_duration

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    existing = []
    if os.path.isfile(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    existing.extend(segments)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    return segments
