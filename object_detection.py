
import cv2
from ultralytics import YOLO
import os
import json
from datetime import datetime

model = YOLO("yolov8n.pt")
OUTPUT_JSON = "outputs/logs/object_detections.json"

def get_frame_by_index(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def detect_objects_from_video_frames(video_json="outputs/logs/video_frames.json", video_dir="inputs/video"):
    if not os.path.exists(video_json):
        print(f"[ERROR] {video_json} not found.")
        return

    with open(video_json, "r") as f:
        frames = json.load(f)

    if len(frames) == 0:
        print("No frame information found.")
        return []

    video_source = os.path.join(video_dir, frames[0]["source"])
    all_detections = []

    for frame in frames:
        frame_index = frame["frame_index_guess"]
        timestamp_sec = frame["timestamp_sec"]
        img = get_frame_by_index(video_source, frame_index)
        if img is None:
            print(f"Could not load frame {frame_index} from {video_source}")
            continue

        # Inference and detection extraction -- robust version!
        results = model(img)
        detections_list = results if isinstance(results, list) else [results]
        for res in detections_list:
            if hasattr(res, "boxes"):
                for box in res.boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf)
                    cls_id = int(box.cls)
                    all_detections.append({
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "class": model.names[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [round(x, 2) for x in bbox],
                        "detection_time": datetime.now().isoformat()
                    })
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_detections, f, indent=4)
    print(f"[INFO] Object detections saved to {OUTPUT_JSON}")
    return all_detections