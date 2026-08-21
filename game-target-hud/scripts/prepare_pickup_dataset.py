"""
Tải V-COCO hold/carry (Gupta et al.) → crop người (+ đồ nếu có) dataset/pickup.

HICO-DET JSON (hold/carry) dùng nếu đã có ảnh train2015 local.
Không đọc tests/input/**.

Chạy từ game-target-hud:
  .venv/bin/python scripts/prepare_pickup_dataset.py
"""

import json
import os
import subprocess
import zipfile

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "dataset", "pickup")
DOWNLOAD = os.path.join(DATASET, "_download")
VCOCO_URL = "https://raw.githubusercontent.com/s-gupta/v-coco/master/data/vcoco/vcoco_train.json"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMG = "http://images.cocodataset.org/{split}/COCO_{split}_{image_id:012d}.jpg"
MAX_EACH = 600
MIN_CROP = 32
UA = "demo-cv-targeting/1.0 (academic research; local demo)"


def _curl(url: str, dest: str, max_time: int = 180) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    subprocess.check_call([
        "curl", "-fL", "-A", UA, "--retry", "2", "--connect-timeout", "30",
        "--max-time", str(max_time), "-o", dest, url,
    ])


def _crop_xyxy(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _union_bbox(a, b):
    return (
        min(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), max(a[3], b[3]),
    )


def _ann_map(instances_path):
    data = json.load(open(instances_path))
    out = {}
    for ann in data["annotations"]:
        x, y, bw, bh = ann["bbox"]
        out[ann["id"]] = (ann["image_id"], (x, y, x + bw, y + bh))
    return out


def _load_coco_img(image_id):
    for split in ("train2014", "val2014"):
        dest = os.path.join(DOWNLOAD, "coco_images", f"{split}_{image_id:012d}.jpg")
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            img = cv2.imread(dest)
            if img is not None:
                return img
        url = COCO_IMG.format(split=split, image_id=image_id)
        try:
            _curl(url, dest, max_time=60)
        except subprocess.CalledProcessError:
            if os.path.isfile(dest):
                os.remove(dest)
            continue
        img = cv2.imread(dest)
        if img is not None:
            return img
    return None


def _vcoco_hold_carry(vcoco):
    hold = [x for x in vcoco if x["action_name"] == "hold"][0]
    carry = [x for x in vcoco if x["action_name"] == "carry"][0]
    by_ann = {}
    for block, key in ((hold, "hold"), (carry, "carry")):
        n = len(block["label"])
        roles = block["role_object_id"]
        for i in range(n):
            aid = block["ann_id"][i]
            rec = by_ann.setdefault(aid, {"image_id": block["image_id"][i], "hold": 0, "carry": 0, "obj": None})
            rec[key] = int(block["label"][i])
            if rec[key] == 1 and len(roles) == 2 * n:
                rec["obj"] = roles[2 * i + 1]
    return by_ann


def _vcoco_rows(vcoco, ann_map, want_pos):
    out = []
    for aid, rec in _vcoco_hold_carry(vcoco).items():
        if aid not in ann_map:
            continue
        is_pos = rec["hold"] == 1 or rec["carry"] == 1
        if want_pos and not is_pos:
            continue
        if (not want_pos) and is_pos:
            continue
        out.append((aid, rec["obj"] if want_pos else None))
    return out


def main():
    if "tests/input" in DATASET.replace("\\", "/"):
        raise RuntimeError("cấm pickup dataset trong tests/input")
    os.makedirs(DOWNLOAD, exist_ok=True)
    vcoco_path = os.path.join(DOWNLOAD, "vcoco_train.json")
    if not os.path.isfile(vcoco_path):
        _curl(VCOCO_URL, vcoco_path, max_time=60)
    zip_path = os.path.join(DOWNLOAD, "annotations_trainval2014.zip")
    inst_train = os.path.join(DOWNLOAD, "instances_train2014.json")
    inst_val = os.path.join(DOWNLOAD, "instances_val2014.json")
    if not os.path.isfile(inst_train):
        if not os.path.isfile(zip_path):
            _curl(COCO_ANN_ZIP, zip_path, max_time=600)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if base in ("instances_train2014.json", "instances_val2014.json"):
                    with zf.open(name) as src, open(os.path.join(DOWNLOAD, base), "wb") as dst:
                        dst.write(src.read())
    if not os.path.isfile(inst_train):
        raise FileNotFoundError(f"BLOCKER: thiếu {inst_train}")
    ann_map = _ann_map(inst_train)
    if os.path.isfile(inst_val):
        ann_map.update(_ann_map(inst_val))
    vcoco = json.load(open(vcoco_path))
    pos_dir = os.path.join(DATASET, "crops", "item_pickup")
    neg_dir = os.path.join(DATASET, "crops", "none")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    n_pos = 0
    for aid, obj_id in _vcoco_rows(vcoco, ann_map, True):
        if n_pos >= MAX_EACH:
            break
        image_id, box = ann_map[aid]
        if obj_id in ann_map and obj_id != aid:
            _, obox = ann_map[obj_id]
            box = _union_bbox(box, obox)
        img = _load_coco_img(image_id)
        if img is None:
            continue
        crop = _crop_xyxy(img, *box)
        if crop is None:
            continue
        cv2.imwrite(os.path.join(pos_dir, f"p{n_pos:04d}.jpg"), crop)
        n_pos += 1
    n_neg = 0
    for aid, _ in _vcoco_rows(vcoco, ann_map, False):
        if n_neg >= min(MAX_EACH, n_pos):
            break
        image_id, box = ann_map[aid]
        img = _load_coco_img(image_id)
        if img is None:
            continue
        crop = _crop_xyxy(img, *box)
        if crop is None:
            continue
        cv2.imwrite(os.path.join(neg_dir, f"n{n_neg:04d}.jpg"), crop)
        n_neg += 1
    print(f"item_pickup={n_pos} none={n_neg}")
    if n_pos < 1 or n_neg < 1:
        raise RuntimeError("BLOCKER: pickup crops thiếu")


if __name__ == "__main__":
    main()
