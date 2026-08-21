"""
Augment class exit (Door OI) bằng glare trên chính bbox đã label.

Không đọc tests/input/**.

Chạy từ game-target-hud:
  .venv/bin/python scripts/augment_doorway_glare.py
"""

import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ZONE = os.path.join(ROOT, "dataset", "zone")
EXIT_ID = 2


def _boxes(lbl_path):
    rows = []
    with open(lbl_path) as f:
        for line in f:
            p = line.split()
            if int(p[0]) == EXIT_ID:
                rows.append(tuple(float(x) for x in p[1:]))
    return rows


def _glare(img, boxes):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for xc, yc, bw, bh in boxes:
        x1 = max(0, int((xc - bw / 2) * w))
        x2 = min(w, int((xc + bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        y2 = min(h, int((yc + bh / 2) * h))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = np.clip(patch * 3.0 + 80.0, 0, 255)
    return out.astype(np.uint8)


def main():
    if "tests/input" in ZONE.replace("\\", "/"):
        raise RuntimeError("cấm augment từ tests/input")
    n = 0
    for split in ("train", "val"):
        img_dir = os.path.join(ZONE, "images", split)
        lbl_dir = os.path.join(ZONE, "labels", split)
        for fname in sorted(os.listdir(lbl_dir)):
            if not fname.endswith(".txt") or fname.startswith("glare_"):
                continue
            boxes = _boxes(os.path.join(lbl_dir, fname))
            if not boxes:
                continue
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, stem + ".jpg")
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            gname = "glare_" + stem
            cv2.imwrite(os.path.join(img_dir, gname + ".jpg"), _glare(img, boxes))
            with open(os.path.join(lbl_dir, fname)) as src, open(
                os.path.join(lbl_dir, gname + ".txt"), "w"
            ) as dst:
                dst.write(src.read())
            n += 1
    print(f"glare copies={n}")
    if n < 1:
        raise RuntimeError("BLOCKER: không có ảnh exit để glare")


if __name__ == "__main__":
    main()
