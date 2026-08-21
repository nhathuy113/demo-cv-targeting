"""
Train CNN 3 class trên crop person: normal / threat / theft_like_posture.

Input: dataset/behavior/videos/{normal,threat,theft_like}/train/*.mp4
  (threat/ tùy chọn — chưa có video thì class đó 0 sample)
Không đọc tests/input/**.
Holdout evaluate: tests/input/behavior_holdout/ (không train).

Output: models/attributes/behavior.onnx

Chạy từ game-target-hud:
  .venv/bin/python scripts/train_behavior.py
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detector import Detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT, "dataset", "behavior", "videos")
HOLDOUT_DIR = os.path.join(ROOT, "tests", "input", "behavior_holdout")
OUT_ONNX = os.path.join(ROOT, "models", "attributes", "behavior.onnx")
YOLO = os.path.join(ROOT, "models", "yolov8n.onnx")
CLASSES = ["normal", "threat", "theft_like_posture"]
CLASS_DIRS = ["normal", "threat", "theft_like"]
IMGSZ = 64
EPOCHS = 40
BATCH = 16
FRAMES_PER_VIDEO = 16


class TinyBehavior(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _crop_persons(detector, img):
    crops = []
    for d in detector.detect(img):
        if d.class_name != "person":
            continue
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
    return crops


def _augment(crop):
    out = [crop, cv2.flip(crop, 1)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)
    out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
    out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return out


def _to_tensor(bgr):
    rgb = cv2.resize(bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def _iter_frames(path, max_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, n // max_frames) if n else 10
    got = 0
    i = 0
    while got < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        got += 1
        i += step
    cap.release()


def _video_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(folder, f))
    )


def _collect_folder(detector, folder, cls_id, augment):
    xs, ys = [], []
    files = _video_files(folder)
    for fname in files:
        path = os.path.join(folder, fname)
        for frame in _iter_frames(path):
            crops = _crop_persons(detector, frame)
            for crop in crops:
                variants = _augment(crop) if augment else [crop]
                for aug in variants:
                    xs.append(_to_tensor(aug))
                    ys.append(cls_id)
    return xs, ys, len(files)


def _collect(detector):
    xs, ys = [], []
    print(f"HOLDOUT (not in train): {HOLDOUT_DIR}")
    for cls_id, (name, dirname) in enumerate(zip(CLASSES, CLASS_DIRS)):
        folder = os.path.join(SRC_DIR, dirname, "train")
        if "tests/input" in folder.replace("\\", "/"):
            raise RuntimeError(f"cấm train từ tests/input: {folder}")
        if dirname == "threat" and not _video_files(folder):
            print(f"{name}: 0 video (skip)")
            continue
        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        cx, cy, n_vid = _collect_folder(detector, folder, cls_id, True)
        if not cx and dirname != "threat":
            raise RuntimeError(f"empty crops {folder}")
        xs.extend(cx)
        ys.extend(cy)
        print(f"{name} (dir={dirname}/train): videos={n_vid} tensors={len(cx)}")
    if not xs:
        raise RuntimeError("no train tensors")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class TensorSet(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _eval_holdout(detector, model, device):
    from collections import Counter
    model.eval()
    print("holdout counts (không phải accuracy trên clip demo train):")
    for tag, cls_expect in (("theft_like", 2), ("normal", 0)):
        folder = os.path.join(HOLDOUT_DIR, tag)
        xs, ys, n_vid = _collect_folder(detector, folder, cls_expect, False)
        if not xs:
            print(f"  {tag}: videos={n_vid} crops=0")
            continue
        x = torch.stack(xs).to(device)
        pred = []
        with torch.no_grad():
            for i in range(0, len(x), BATCH):
                logits = model(x[i:i + BATCH])
                pred.extend(logits.argmax(1).tolist())
        cnt = Counter(CLASSES[p] for p in pred)
        print(f"  {tag}: videos={n_vid} crops={len(pred)} pred={dict(cnt)}")


def main():
    detector = Detector(YOLO, conf_threshold=0.25, class_filter=[0])
    x, y = _collect(detector)
    counts = [(y == i).sum().item() for i in range(len(CLASSES))]
    print(f"samples={len(y)} per_class={dict(zip(CLASSES, counts))}")
    w = []
    for c in counts:
        w.append(0.0 if c == 0 else 1.0 / c)
    weight = torch.tensor(w, dtype=torch.float32)
    if float(weight.sum()) > 0:
        weight = weight / weight.sum() * sum(1 for c in counts if c > 0)
    device = _device()
    print(f"device={device}")
    ds = TensorSet(x, y)
    loader = DataLoader(ds, batch_size=min(BATCH, len(ds)), shuffle=True)
    model = TinyBehavior().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=weight.to(device))
    model.train()
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch} loss={total / n:.4f} acc={correct / n:.3f}")
    model.eval()
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ, device=device)
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    cpu_model = model.to("cpu")
    dummy = dummy.to("cpu")
    torch.onnx.export(
        cpu_model, dummy, OUT_ONNX,
        input_names=["images"], output_names=["logits"],
        opset_version=12,
        dynamo=False,
    )
    print(f"exported {OUT_ONNX}")
    _eval_holdout(detector, cpu_model.to(device), device)


if __name__ == "__main__":
    main()
