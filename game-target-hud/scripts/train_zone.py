"""
Train zone expert YOLO trên dataset/zone.

Không đọc tests/input/**.

Chạy từ game-target-hud:
  .venv/bin/python scripts/train_zone.py
"""

import os
import shutil

from ultralytics import YOLO
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_YAML = os.path.join(ROOT, "dataset", "zone", "data.yaml")
PROJECT = os.path.join(ROOT, "runs", "zone")
OUT_ONNX = os.path.join(ROOT, "models", "zone.onnx")

FREEZE_EPOCHS = 5
UNFREEZE_EPOCHS = 8
UNFREEZE_LR0 = 0.001


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main():
    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError("BLOCKER: thiếu dataset/zone/data.yaml — chạy prepare_zone_dataset.py")
    train_dir = os.path.join(ROOT, "dataset", "zone", "images", "train")
    if "tests/input" in train_dir.replace("\\", "/"):
        raise RuntimeError("cấm train zone từ tests/input")
    n = len([f for f in os.listdir(train_dir) if f.endswith(".jpg")]) if os.path.isdir(train_dir) else 0
    if n < 1:
        raise RuntimeError("BLOCKER: dataset/zone/images/train rỗng")
    device = _device()
    print(f"device={device} train_images={n}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=FREEZE_EPOCHS,
        imgsz=640,
        freeze=10,
        device=device,
        project=PROJECT,
        name="z_freeze",
        exist_ok=True,
        pretrained=True,
        hsv_v=0.9,
        hsv_s=0.7,
    )
    freeze_best = os.path.join(PROJECT, "z_freeze", "weights", "best.pt")
    model = YOLO(freeze_best)
    model.train(
        data=DATA_YAML,
        epochs=UNFREEZE_EPOCHS,
        imgsz=640,
        freeze=0,
        lr0=UNFREEZE_LR0,
        device=device,
        project=PROJECT,
        name="z_unfreeze",
        exist_ok=True,
        hsv_v=0.9,
        hsv_s=0.7,
    )
    unfreeze_best = os.path.join(PROJECT, "z_unfreeze", "weights", "best.pt")
    model = YOLO(unfreeze_best)
    export_path = model.export(format="onnx", imgsz=640, simplify=True)
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    shutil.copy2(export_path, OUT_ONNX)
    print(f"exported {OUT_ONNX}")


if __name__ == "__main__":
    main()
