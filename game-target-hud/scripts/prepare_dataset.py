"""
Level 2 data collection: COCO val2017, chỉ class person (Phương án B).

~400 ảnh, YOLO txt labels, split 70/20/10 theo image_id (block split,
không shuffle ngẫu nhiên). Sample theo diện tích box (gần/trung/xa).

Chạy từ game-target-hud:
  .venv/bin/python scripts/prepare_dataset.py
"""

import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "dataset")
DOWNLOAD = os.path.join(DATASET, "_download")
ANN_JSON = os.path.join(DOWNLOAD, "instances_val2017.json")
ANN_URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"

PERSON_CAT = 1
TARGET_TOTAL = 400
NEAR_RATIO = 0.30
FAR_RATIO = 0.05
PER_BUCKET = TARGET_TOTAL // 3


def _curl(url: str, dest: str, max_time: int = 120):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.check_call([
        "curl", "-fsSL", "--retry", "3", "--connect-timeout", "30",
        "--max-time", str(max_time), "-o", dest, url,
    ])


def _coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    return xc, yc, w / img_w, h / img_h


def _bucket(max_area_ratio: float) -> str:
    if max_area_ratio > NEAR_RATIO:
        return "near"
    if max_area_ratio < FAR_RATIO:
        return "far"
    return "mid"


def _split_block(items):
    n = len(items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main():
    os.makedirs(DOWNLOAD, exist_ok=True)
    if not os.path.isfile(ANN_JSON) or os.path.getsize(ANN_JSON) < 1000:
        print(f"download {ANN_URL}")
        _curl(ANN_URL, ANN_JSON, max_time=180)

    with open(ANN_JSON) as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != PERSON_CAT:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)

    buckets = {"near": [], "mid": [], "far": []}
    for image_id, anns in by_image.items():
        im = images[image_id]
        area = im["width"] * im["height"]
        max_ratio = max((a["bbox"][2] * a["bbox"][3]) / area for a in anns)
        buckets[_bucket(max_ratio)].append(image_id)

    selected = []
    for name in ("near", "mid", "far"):
        ids = sorted(buckets[name])[:PER_BUCKET]
        print(f"bucket {name}: available={len(buckets[name])} take={len(ids)}")
        selected.extend(ids)

    selected = sorted(set(selected))
    print(f"selected images: {len(selected)}")

    splits = {"train": [], "val": [], "test": []}
    for name in ("near", "mid", "far"):
        ids = [i for i in selected if i in set(sorted(buckets[name])[:PER_BUCKET])]
        ids = sorted(ids)
        tr, va, te = _split_block(ids)
        splits["train"].extend(tr)
        splits["val"].extend(va)
        splits["test"].extend(te)

    for split, ids in splits.items():
        img_dir = os.path.join(DATASET, "images", split)
        lbl_dir = os.path.join(DATASET, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for image_id in ids:
            im = images[image_id]
            file_name = im["file_name"]
            img_path = os.path.join(img_dir, file_name)
            if not os.path.isfile(img_path):
                _curl(IMG_URL.format(file_name=file_name), img_path)
            lines = []
            for ann in by_image[image_id]:
                xc, yc, nw, nh = _coco_to_yolo(ann["bbox"], im["width"], im["height"])
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            stem = os.path.splitext(file_name)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        print(f"{split}: {len(ids)} images")

    yaml_path = os.path.join(DATASET, "data.yaml")
    yaml_text = (
        f"path: {DATASET}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: person\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
