"""
Tải Open Images val+test → dataset/zone (cashier/cash/exit/shelf).

Không đọc tests/input/**.
OI không có class POS/cash register → không tạo class POS.

Chạy từ game-target-hud:
  .venv/bin/python scripts/prepare_zone_dataset.py
"""

import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "dataset", "zone")
DOWNLOAD = os.path.join(DATASET, "_download")
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
ANN_URL = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"

OI_TO_CLASS = {
    "/m/0b3fp9": 0,  # Countertop → cashier
    "/m/0242l": 1,   # Coin → cash
    "/m/02dgv": 2,   # Door → exit
    "/m/0gjbg72": 3, # Shelf → shelf
}
CLASS_NAMES = ["cashier", "cash", "exit", "shelf"]
MAX_PER_CLASS = 100


def _curl(url: str, dest: str, max_time: int = 120):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    subprocess.check_call([
        "curl", "-fsSL", "--retry", "3", "--connect-timeout", "30",
        "--max-time", str(max_time), "-o", dest, url,
    ])


def _split_block(items):
    n = len(items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def _load_ann(split: str):
    path = os.path.join(DOWNLOAD, f"{split}-annotations-bbox.csv")
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        _curl(ANN_URL.format(split=split), path, max_time=180)
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            mid = p[2]
            if mid not in OI_TO_CLASS:
                continue
            if p[10] == "1" or p[11] == "1":
                continue
            xmin, xmax, ymin, ymax = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            rows.append((p[0], split, OI_TO_CLASS[mid], xmin, xmax, ymin, ymax))
    return rows


def main():
    if "tests/input" in DATASET.replace("\\", "/"):
        raise RuntimeError("cấm dataset zone trong tests/input")
    os.makedirs(DOWNLOAD, exist_ok=True)
    rows = _load_ann("validation") + _load_ann("test")
    by_class = {i: [] for i in range(len(CLASS_NAMES))}
    for r in rows:
        by_class[r[2]].append(r)
    selected_ids = []
    keep = set()
    for cid, name in enumerate(CLASS_NAMES):
        ids = sorted(set(r[0] for r in by_class[cid]))[:MAX_PER_CLASS]
        print(f"{name}: images={len(ids)}")
        if not ids:
            raise RuntimeError(f"BLOCKER: 0 ảnh class {name}")
        selected_ids.extend(ids)
        keep.update(ids)
    id_split = {}
    for r in rows:
        if r[0] in keep:
            id_split[r[0]] = r[1]
    selected = sorted(set(selected_ids))
    boxes = {}
    for r in rows:
        if r[0] not in keep:
            continue
        xc = (r[3] + r[4]) / 2.0
        yc = (r[5] + r[6]) / 2.0
        w = r[4] - r[3]
        h = r[6] - r[5]
        boxes.setdefault(r[0], []).append(f"{r[2]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    tr, va, te = _split_block(selected)
    splits = {"train": tr, "val": va, "test": te}
    for split, ids in splits.items():
        img_dir = os.path.join(DATASET, "images", split)
        lbl_dir = os.path.join(DATASET, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for image_id in ids:
            img_path = os.path.join(img_dir, image_id + ".jpg")
            if not os.path.isfile(img_path):
                _curl(IMG_URL.format(split=id_split[image_id], image_id=image_id), img_path)
            with open(os.path.join(lbl_dir, image_id + ".txt"), "w") as f:
                f.write("\n".join(boxes[image_id]) + "\n")
        print(f"{split}: {len(ids)} images")
    yaml_path = os.path.join(DATASET, "data.yaml")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {DATASET}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"names:\n{names}\n"
        )
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
