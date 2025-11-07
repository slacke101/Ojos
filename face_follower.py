#!/usr/bin/env python
"""face_follower.py

Real-time face detection and tracking using OpenCV.

Usage:
    python face_follower.py            # run with defaults
    python face_follower.py --source 1 # use external camera index 1
    python face_follower.py --tracker kcf --display False

Dependencies: see requirements.txt
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_tracker(kind: str = "csrt") -> cv2.Tracker:
    """Return an OpenCV object tracker instance by name."""
    kind = kind.lower()
    if kind == "csrt":
        return cv2.TrackerCSRT_create()
    if kind == "kcf":
        return cv2.TrackerKCF_create()
    if kind == "mosse":
        return cv2.TrackerMOSSE_create()

    raise ValueError(f"Unsupported tracker type: {kind}")


def load_detector() -> cv2.CascadeClassifier:
    """Load Haar cascade face detector bundled with OpenCV data."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        sys.exit(f"Failed to load Haar cascade from {cascade_path}")
    return detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face detection and tracking")
    parser.add_argument("--source", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--tracker", type=str, default="csrt",
                        choices=["csrt", "kcf", "mosse"], help="Tracker type")
    parser.add_argument("--display", type=lambda x: x.lower() != "false", default=True,
                        help="Show GUI window (True/False)")
    parser.add_argument("--min-size", type=int, default=80,
                        help="Minimum face size in pixels (height)")
    parser.add_argument("--record", type=lambda x: x.lower() != "false", default=False,
                        help="Save frames and bbox to data/raw for training")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save every Nth processed frame when recording")
    parser.add_argument("--yolo", type=str, default="", help="Path to custom YOLOv8 weights (.pt) for face detection")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera index {args.source}")

    detector = None  # could be Haar or YOLO
    if args.yolo:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo)
        detector_type = "yolo"
    else:
        detector = load_detector()
        detector_type = "haar"

    tracker = create_tracker(args.tracker)

    tracking = False
    bbox: Tuple[int, int, int, int] | None = None
    fps = 0.0
    last_time = time.time()
    frame_count = 0

    print("Press ESC to quit, SPACE to re-detect face.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Update tracker or run face detection --------------------------------
        if tracking and bbox is not None:
            ok, bbox = tracker.update(frame)
            if not ok:
                tracking = False  # lost track → fall back to detection
        if not tracking:
            if detector_type == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(args.min_size, args.min_size))
                if len(faces):
                    bbox = tuple(int(v) for v in faces[0])
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, bbox)
                    tracking = True
            else:  # YOLO detection
                results = yolo_model(frame, verbose=False)[0]
                if results.boxes:
                    x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    tracker = create_tracker(args.tracker)
                    tracker.init(frame, bbox)
                    tracking = True

        # Draw bounding box ----------------------------------------------------
        if bbox is not None and tracking:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if args.record and frame_count % args.save_every == 0:
                save_frame_and_label(frame, (x, y, w, h))

        # FPS calculation ------------------------------------------------------
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time))
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        if args.display:
            cv2.imshow("Face Follower", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord(" "):
            tracking = False  # force re-detect next loop
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def save_frame_and_label(frame, bbox):
    """Save frame and bbox JSON into data/raw/YYYY-MM-DD directory."""
    import json, os, datetime
    date_dir = Path("data/raw") / datetime.date.today().isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H%M%S_%f")
    img_path = date_dir / f"{ts}.jpg"
    json_path = date_dir / f"{ts}.json"
    cv2.imwrite(str(img_path), frame)
    x, y, w, h = map(int, bbox)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"x": x, "y": y, "w": w, "h": h, "img": img_path.name}, f)


if __name__ == "__main__":
    main()
