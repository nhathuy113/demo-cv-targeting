"""
Tải ảnh công khai cho behavior: normal / threat / fraud.

Nguồn: Wikimedia Commons (CC) + ảnh person đã có trong tests/input và dataset.

Chạy từ game-target-hud:
  .venv/bin/python scripts/download_attribute_data.py
"""

import os
import shutil
import subprocess
import urllib.parse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "tests", "input", "attributes")
UA = "demo-cv-targeting/1.0 (academic research; local demo)"

THREAT_FILES = [
    "Boxing080905.jpg",
    "Judo.jpg",
    "Muay_Thai.jpg",
    "Kickboxing.jpg",
    "Sumo_wrestling.jpg",
    "Boxing at the 2018 Summer Youth Olympics – Girls' flyweight Bronze Medal Bout 098.jpg",
    "Boxing at the 2018 Summer Youth Olympics – Girls' lightweight Bronze Medal Bout 451.jpg",
    "Boxing at the 2018 Summer Youth Olympics – Girls' lightweight Bronze Medal Bout 387.jpg",
    "Boxing at the 2018 Summer Youth Olympics – Girls' lightweight Bronze Medal Bout 007.jpg",
    "Boxing at the 2018 Summer Youth Olympics – Girls' lightweight Bronze Medal Bout 281.jpg",
]

FRAUD_FILES = [
    "Pickpocket.jpg",
    "Theft.jpg",
    "Pickpocket girl.jpg",
    "USMC shoplifting.jpg",
]

NORMAL_LOCAL = [
    os.path.join(ROOT, "tests", "input", "zidane.jpg"),
    os.path.join(ROOT, "tests", "input", "bus.jpg"),
    os.path.join(ROOT, "tests", "input", "coco_000000000139.jpg"),
    os.path.join(ROOT, "tests", "input", "coco_000000000785.jpg"),
    os.path.join(ROOT, "tests", "input", "coco_000000001000.jpg"),
]


def _curl_file(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        subprocess.check_call([
            "curl", "-fL", "-A", UA, "--retry", "2", "--connect-timeout", "20",
            "--max-time", "40", "-o", dest, url,
        ])
    except subprocess.CalledProcessError:
        if os.path.isfile(dest):
            os.remove(dest)
        return False
    if not os.path.isfile(dest) or os.path.getsize(dest) < 4000:
        if os.path.isfile(dest):
            os.remove(dest)
        return False
    return True


def _commons(filename: str, dest: str) -> bool:
    quoted = urllib.parse.quote(filename)
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width=640"
    return _curl_file(url, dest)


def _copy_normal():
    dest_dir = os.path.join(OUT, "normal")
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for src in NORMAL_LOCAL:
        if not os.path.isfile(src):
            continue
        dest = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dest)
        n += 1
    coco_train = os.path.join(ROOT, "dataset", "images", "train")
    if os.path.isdir(coco_train):
        names = sorted(os.listdir(coco_train))[:40]
        for name in names:
            src = os.path.join(coco_train, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, name))
                n += 1
    return n


def main():
    for cls, files in (("threat", THREAT_FILES), ("fraud", FRAUD_FILES)):
        dest_dir = os.path.join(OUT, cls)
        os.makedirs(dest_dir, exist_ok=True)
        ok = 0
        for i, name in enumerate(files):
            dest = os.path.join(dest_dir, f"{i:02d}_{os.path.basename(name)}")
            if os.path.isfile(dest) and os.path.getsize(dest) > 4000:
                print(f"skip {dest}")
                ok += 1
                continue
            if _commons(name, dest):
                print(f"ok {dest}")
                ok += 1
            else:
                print(f"fail {name}")
        print(f"{cls}={ok}/{len(files)}")
    n = _copy_normal()
    print(f"normal copied={n}")


if __name__ == "__main__":
    main()
