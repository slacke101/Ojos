#!/usr/bin/env python
"""register_face.py

Capture 40 face images of a new person and store them under data/people/<name>/.

Usage:
    python register_face.py --name Rafael --source 0
"""
import argparse
import sys
from pathlib import Path
import cv2

from face_follower import load_detector  # reuse Haar cascade


def main():
    parser = argparse.ArgumentParser(description="Register new person's face")
    parser.add_argument("--name", required=True, help="Person's name (directory will be created)")
    parser.add_argument("--count", type=int, default=40, help="Number of samples to capture")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    out_dir = Path("data/people") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit("Cannot open camera")

    detector = load_detector()
    saved = 0
    print("Look at the camera. Press ESC to abort.")
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
            cv2.imshow("Face", roi)
            cv2.imwrite(str(out_dir / f"{args.name}_{saved:03d}.png"), roi)
            saved += 1
            print(f"Saved {saved}/{args.count}")
            if saved >= args.count:
                break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
