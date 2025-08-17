import os
import json
from datetime import datetime

from preprocess.video_loader import process_video
from analysis.object_detection import detect_objects_from_video_frames
from router import bulk_route

DATA_VIDEOS_DIR = "inputs/video"
DATA_AUDIOS_DIR = "inputs/audio"
OUTPUT_COMBINED = "outputs/logs/combined_log.json"

def reset_logs():
    open("outputs/combined_log.json", "w").write("[]")
    open("outputs/logs/object_detections.json", "w").write("[]")
    open("outputs/logs/video_frames.json", "w").write("[]")
    open("outputs/logs/audio_segments.json", "w").write("[]")

def _scan_media_files():
    files = []
    for root, _, names in os.walk(DATA_VIDEOS_DIR):
        for n in names:
            files.append(os.path.join(root, n))
    for root, _, names in os.walk(DATA_AUDIOS_DIR):
        for n in names:
            files.append(os.path.join(root, n))
    return sorted(files)

def main():
    reset_logs()  # Always reset before each run

    files = _scan_media_files()
    if not files:
        print("No media files found. Put videos in 'inputs/video' and audios in 'inputs/audio'.")
        return

    print(f"Discovered {len(files)} file(s). Routing...")
    results = bulk_route(files)

    combined = {
        "run_meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_inputs": len(files),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_COMBINED), exist_ok=True)
    with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    n_video_entries = sum(len(r["metadata"]) for r in results if r["type"] == "video")
    n_audio_entries = sum(len(r["metadata"]) for r in results if r["type"] == "audio")

    print("\n=== Phase 1 Summary ===")
    print(f"Processed inputs: {len(files)}")
    print(f"Video entries (sampled frames): {n_video_entries}")
    print(f"Audio entries (segments): {n_audio_entries}")
    print(f"Combined log written to: {OUTPUT_COMBINED}")
    print("Individual logs: outputs/logs/video_frames.json, outputs/logs/audio_segments.json")

if __name__ == "__main__":
    reset_logs()
    main()
    detect_objects_from_video_frames()