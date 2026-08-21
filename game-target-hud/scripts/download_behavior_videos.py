"""
Tải video UCF-Crime cho behavior: theft_like vs normal.

Train → dataset/behavior/videos/  (không dùng tests/input để train)
Holdout 5+5 → tests/input/behavior_holdout/  (evaluate only)

UCF Shoplifting chỉ có 50 video unique → train 45 + holdout 5.
Không tách nhiều clip từ 1 video để đủ 95 (leakage).
Normal: 95 train + 5 holdout (Testing_Normal, không trùng train).

Chạy từ game-target-hud:
  .venv/bin/python scripts/download_behavior_videos.py
"""

import json
import os
import subprocess
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_THEFT = os.path.join(ROOT, "dataset", "behavior", "videos", "theft_like", "train")
TRAIN_NORM = os.path.join(ROOT, "dataset", "behavior", "videos", "normal", "train")
VAL_THEFT = os.path.join(ROOT, "tests", "input", "behavior_holdout", "theft_like")
VAL_NORM = os.path.join(ROOT, "tests", "input", "behavior_holdout", "normal")
UA = "demo-cv-targeting/1.0 (academic research; local demo)"
HF = "https://huggingface.co/datasets/mteb/ucf-crime/resolve/main"
API = "https://huggingface.co/api/datasets/mteb/ucf-crime/tree/main"

HOLD_THEFT = [
    "Shoplifting001_x264.mp4",
    "Shoplifting052_x264.mp4",
    "Shoplifting053_x264.mp4",
    "Shoplifting054_x264.mp4",
    "Shoplifting055_x264.mp4",
]


def _list(folder: str) -> list[dict]:
    url = f"{API}/{folder}?limit=500"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _curl(url: str, dest: str, timeout: int = 300) -> bool:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        subprocess.check_call([
            "curl", "-fL", "-A", UA, "--retry", "2", "--connect-timeout", "30",
            "--max-time", str(timeout), "-o", dest, url,
        ])
    except subprocess.CalledProcessError:
        if os.path.isfile(dest):
            os.remove(dest)
        return False
    return os.path.isfile(dest) and os.path.getsize(dest) > 5000


def _clip30(src: str, dest: str) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
        return True
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", src, "-t", "30", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", dest,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return os.path.isfile(dest) and os.path.getsize(dest) > 5000


def _fetch_clip(rel: str, dest_clip: str) -> bool:
    if os.path.isfile(dest_clip) and os.path.getsize(dest_clip) > 5000:
        print(f"skip {os.path.basename(dest_clip)}")
        return True
    tmp = dest_clip + ".full.mp4"
    url = f"{HF}/{rel}"
    if not _curl(url, tmp):
        print(f"fail {rel}")
        return False
    ok = _clip30(tmp, dest_clip)
    if os.path.isfile(tmp):
        os.remove(tmp)
    print(("ok " if ok else "clip-fail ") + os.path.basename(dest_clip))
    return ok


def main():
    shop = _list("videos/Shoplifting")
    shop_names = [x["path"].split("/")[-1] for x in shop]
    hold_theft = [n for n in HOLD_THEFT if n in shop_names]
    train_theft = [n for n in shop_names if n not in set(hold_theft)]
    train_n = _list("videos/Training_Normal")
    train_n.sort(key=lambda x: x["size"])
    train_norm = [x["path"].split("/")[-1] for x in train_n[:95]]
    test_n = _list("videos/Testing_Normal")
    test_n.sort(key=lambda x: x["size"])
    hold_norm = [x["path"].split("/")[-1] for x in test_n[:5]]

    print(f"ucf_shoplifting_unique={len(shop_names)} train_theft={len(train_theft)} hold_theft={len(hold_theft)}")
    print(f"train_normal={len(train_norm)} hold_normal={len(hold_norm)}")
    if len(train_theft) < 95:
        print(f"BLOCKER: UCF chỉ {len(shop_names)} shoplifting; không đủ 95 train. Không nhân clip từ 1 video.")

    n_ok = 0
    n_all = 0
    for name in train_theft:
        n_all += 1
        dest = os.path.join(TRAIN_THEFT, name.replace(".mp4", "_clip30s.mp4"))
        if _fetch_clip(f"videos/Shoplifting/{name}", dest):
            n_ok += 1
    for name in hold_theft:
        n_all += 1
        dest = os.path.join(VAL_THEFT, name.replace(".mp4", "_clip30s.mp4"))
        if _fetch_clip(f"videos/Shoplifting/{name}", dest):
            n_ok += 1
    for name in train_norm:
        n_all += 1
        dest = os.path.join(TRAIN_NORM, name.replace(".mp4", "_clip30s.mp4"))
        if _fetch_clip(f"videos/Training_Normal/{name}", dest):
            n_ok += 1
    for name in hold_norm:
        n_all += 1
        dest = os.path.join(VAL_NORM, name.replace(".mp4", "_clip30s.mp4"))
        if _fetch_clip(f"videos/Testing_Normal/{name}", dest):
            n_ok += 1
    print(f"clips={n_ok}/{n_all}")


if __name__ == "__main__":
    main()
