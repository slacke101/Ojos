#!/usr/bin/env python
"""prepare_dataset.py

Convert frames saved by face_follower.py --record into a YOLOv8 dataset.
It walks through data/raw/YYYY-MM-DD directories, reads JSON labels,
copies/resizes images, creates YOLO TXT label files, splits into train/val,
and writes faces.yaml.

Usage:
    python prepare_dataset.py --split 0.8  # 80% train, 20% val
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/faces")
CLASSES = ["face"]  # single-class dataset


def bbox_to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert (x,y,w,h) in pixels to normalized YOLO (xc, yc, w, h)."""
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


def gather_samples() -> list[tuple[Path, Path]]:
    samples = []
    for json_file in RAW_DIR.rglob("*.json"):
        img_file = json_file.with_suffix(".jpg")
        if img_file.exists():
            samples.append((img_file, json_file))
    return sorted(samples)


def split_samples(samples, train_ratio: float = 0.8):
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]


def prepare(split_ratio: float = 0.8):
    samples = gather_samples()
    if not samples:
        raise RuntimeError("No samples found under data/raw. Run face_follower.py --record first.")

    train_samples, val_samples = split_samples(samples, split_ratio)

    for subset, subset_samples in {"train": train_samples, "val": val_samples}.items():
        img_out = OUT_DIR / "images" / subset
        lbl_out = OUT_DIR / "labels" / subset
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, json_path in tqdm(subset_samples, desc=f"{subset} set"):
            # copy image
            target_img = img_out / img_path.name
            shutil.copy(img_path, target_img)

            # read bbox
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            x, y, w, h = meta["x"], meta["y"], meta["w"], meta["h"]
            img = cv2.imread(str(img_path))
            ih, iw = img.shape[:2]
            xc, yc, wn, hn = bbox_to_yolo(x, y, w, h, iw, ih)

            # write label txt
            label_path = lbl_out / (img_path.stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    # write faces.yaml --------------------------------------------------------
    yaml_path = OUT_DIR / "faces.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write(f"names: {CLASSES}\n")

    print(f"Dataset prepared in {OUT_DIR}. YAML saved to {yaml_path}")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from recorded frames")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default 0.8)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    try:
        prepare(args.split)
    except Exception as e:
        sys.exit(str(e))
