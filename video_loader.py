import cv2
import os
import json

def _video_duration_seconds(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps <= 0.0:
        return 0.0
    return float(total_frames) / float(fps)

def process_video(
    video_path: str,
    output_json: str = "outputs/logs/video_frames.json",
    sample_every_sec: float = 0.5,
):
    """
    Sample video frames by timestamp and record metadata.
    Args:
        video_path: Path to video file.
        output_json: Where to append/save metadata (JSON array).
        sample_every_sec: Sampling interval in seconds.
    Returns:
        A list of frame metadata dicts.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    duration = _video_duration_seconds(cap)
    if duration <= 0.0:
        cap.release()
        return []

    frame_data = []
    t = 0.0
    while t <= duration + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            t += sample_every_sec
            continue
        h, w = frame.shape[:2]
        info = {
            "source": os.path.basename(video_path),
            "timestamp_sec": round(float(t), 3),
            "frame_shape": [int(h), int(w), int(frame.shape[2]) if len(frame.shape) == 3 else 1],
            "frame_index_guess": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        }
        frame_data.append(info)
        t += sample_every_sec
    cap.release()

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
    existing.extend(frame_data)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return frame_data