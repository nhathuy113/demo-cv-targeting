"""
Level 2 evaluation: mAP theo class (ultralytics val) + recall so với Level 1 baseline
trên tập test (IoU 0.5).

Chạy từ game-target-hud:
  .venv/bin/python scripts/eval_level2.py
"""

import os
import sys

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detector import Detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_YAML = os.path.join(ROOT, "dataset", "data.yaml")
L2_PT = os.path.join(ROOT, "runs", "level2", "l2_unfreeze", "weights", "best.pt")
L2_ONNX = os.path.join(ROOT, "models", "yolov8n_person.onnx")
L1_ONNX = os.path.join(ROOT, "models", "yolov8n.onnx")
TEST_IMG = os.path.join(ROOT, "dataset", "images", "test")
TEST_LBL = os.path.join(ROOT, "dataset", "labels", "test")
IOU_THR = 0.5


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _yolo_to_xyxy(line, w, h):
    _, xc, yc, nw, nh = [float(x) for x in line.split()]
    bw, bh = nw * w, nh * h
    x1 = (xc * w) - bw / 2
    y1 = (yc * h) - bh / 2
    return (x1, y1, x1 + bw, y1 + bh)


def _recall(detector, class_filter):
    det = Detector(detector, conf_threshold=0.35, class_filter=class_filter)
    tp = 0
    n_gt = 0
    for name in sorted(os.listdir(TEST_IMG)):
        if not name.endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(TEST_IMG, name))
        h, w = frame.shape[:2]
        stem = os.path.splitext(name)[0]
        with open(os.path.join(TEST_LBL, stem + ".txt")) as f:
            gts = [_yolo_to_xyxy(line, w, h) for line in f if line.strip()]
        n_gt += len(gts)
        preds = [(d.x1, d.y1, d.x2, d.y2) for d in det.detect(frame)]
        used = set()
        for g in gts:
            for i, p in enumerate(preds):
                if i in used:
                    continue
                if _iou(g, p) >= IOU_THR:
                    tp += 1
                    used.add(i)
                    break
    return tp / n_gt if n_gt else 0.0, tp, n_gt


def main():
    print("=== mAP per class (Level 2 best.pt, ultralytics val) ===")
    model = YOLO(L2_PT)
    metrics = model.val(data=DATA_YAML, split="test", imgsz=640)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    for i, name in model.names.items():
        ap50 = float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") else float("nan")
        print(f"  class {i} {name}: AP50={ap50:.4f}")

    print("=== recall@0.5 trên test: L2 ONNX vs L1 COCO ONNX ===")
    r2, tp2, n2 = _recall(L2_ONNX, [0])
    r1, tp1, n1 = _recall(L1_ONNX, [0])
    print(f"L2 person ONNX: recall={r2:.4f} ({tp2}/{n2})")
    print(f"L1 COCO ONNX:   recall={r1:.4f} ({tp1}/{n1})")


if __name__ == "__main__":
    main()
