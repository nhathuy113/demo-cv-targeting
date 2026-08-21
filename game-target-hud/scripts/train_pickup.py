"""
Train ResNet18 ImageNet 2 class pickup trên crop dataset/pickup/crops/{item_pickup,none}.
Reset mỗi lần (không load pickup.onnx cũ).
Không đọc tests/input/**.

Chạy từ game-target-hud:
  .venv/bin/python scripts/train_pickup.py
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "dataset", "pickup", "crops")
OUT_ONNX = os.path.join(ROOT, "models", "attributes", "pickup.onnx")
CLASSES = ["none", "item_pickup"]
IMGSZ = 224
EPOCHS = 20
BATCH = 16
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(bgr):
    rgb = cv2.resize(bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def _collect():
    xs, ys = [], []
    for cls_id, name in enumerate(CLASSES):
        folder = os.path.join(SRC, name)
        if "tests/input" in folder.replace("\\", "/"):
            raise RuntimeError(f"cấm train từ tests/input: {folder}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"BLOCKER: thiếu {folder}")
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            xs.append(_to_tensor(img))
            ys.append(cls_id)
        print(f"{name}: {len(files)} crops")
        if not files:
            raise RuntimeError(f"BLOCKER: empty {folder}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


class TensorSet(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def main():
    x, y = _collect()
    device = _device()
    print(f"device={device} samples={len(y)}")
    idx = list(range(len(y)))
    random.Random(0).shuffle(idx)
    n_val = max(1, len(idx) // 10)
    val_i = idx[:n_val]
    train_i = idx[n_val:]
    train_ds = TensorSet(x[train_i], y[train_i])
    val_ds = TensorSet(x[val_i], y[val_i])
    loader = DataLoader(train_ds, batch_size=min(BATCH, len(train_ds)), shuffle=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total, n, correct = 0.0, 0, 0
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
        model.eval()
        with torch.no_grad():
            vx, vy = val_ds.x.to(device), val_ds.y.to(device)
            vpred = model(vx).argmax(1)
            vacc = (vpred == vy).float().mean().item()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch} loss={total / n:.4f} acc={correct / n:.3f} val={vacc:.3f}")
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)
    cpu_model = model.to("cpu").eval()
    torch.onnx.export(
        cpu_model, dummy, OUT_ONNX,
        input_names=["images"], output_names=["logits"],
        opset_version=12,
        dynamo=False,
    )
    print(f"exported {OUT_ONNX}")


if __name__ == "__main__":
    main()
