"""
Level 2 train: yolov8n.pt transfer learning.

1) freeze backbone (freeze=10), train head vài epoch
2) unfreeze, train tiếp với lr0 thấp hơn
3) export ONNX: format='onnx', imgsz=640, simplify=True

Chạy từ game-target-hud:
  .venv/bin/python scripts/train_level2.py
"""

import os
import shutil

from ultralytics import YOLO
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_YAML = os.path.join(ROOT, "dataset", "data.yaml")
PROJECT = os.path.join(ROOT, "runs", "level2")
OUT_ONNX = os.path.join(ROOT, "models", "yolov8n_person.onnx")

FREEZE_EPOCHS = 10
UNFREEZE_EPOCHS = 20
UNFREEZE_LR0 = 0.001


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main():
    device = _device()
    print(f"device={device}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=FREEZE_EPOCHS,
        imgsz=640,
        freeze=10,
        device=device,
        project=PROJECT,
        name="l2_freeze",
        exist_ok=True,
        pretrained=True,
    )
    freeze_best = os.path.join(PROJECT, "l2_freeze", "weights", "best.pt")
    model = YOLO(freeze_best)
    model.train(
        data=DATA_YAML,
        epochs=UNFREEZE_EPOCHS,
        imgsz=640,
        freeze=0,
        lr0=UNFREEZE_LR0,
        device=device,
        project=PROJECT,
        name="l2_unfreeze",
        exist_ok=True,
    )
    unfreeze_best = os.path.join(PROJECT, "l2_unfreeze", "weights", "best.pt")
    model = YOLO(unfreeze_best)
    export_path = model.export(format="onnx", imgsz=640, simplify=True)
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    shutil.copy2(export_path, OUT_ONNX)
    print(f"exported {OUT_ONNX}")


if __name__ == "__main__":
    main()
