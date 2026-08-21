"""
Tải video test trộm / cướp giật (Mixkit + Wikimedia).

Chạy từ game-target-hud:
  .venv/bin/python scripts/download_threat_videos.py
"""

import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "tests", "input", "threat_test")
UA = "demo-cv-targeting/1.0 (academic research; local demo)"

VIDEOS = [
    ("snatch_purse.mp4", "https://assets.mixkit.co/videos/31359/31359-720.mp4"),
    ("steal_phone.mp4", "https://assets.mixkit.co/videos/31354/31354-720.mp4"),
    ("cctv_thieves.mp4", "https://assets.mixkit.co/videos/31372/31372-720.mp4"),
    ("steal_laptop_car.mp4", "https://assets.mixkit.co/videos/31363/31363-720.mp4"),
]


def _curl(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        subprocess.check_call([
            "curl", "-fL", "-A", UA, "--retry", "2", "--connect-timeout", "20",
            "--max-time", "90", "-o", dest, url,
        ])
    except subprocess.CalledProcessError:
        if os.path.isfile(dest):
            os.remove(dest)
        return False
    return os.path.isfile(dest) and os.path.getsize(dest) > 10000


def _porch_theft():
    ogv = os.path.join(OUT, "porch_theft.ogv")
    mp4 = os.path.join(OUT, "porch_theft.mp4")
    if os.path.isfile(mp4) and os.path.getsize(mp4) > 10000:
        print(f"skip {mp4}")
        return True
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/Front%20Porch%20Theft%202012%2011%2017%20101117.ogv"
    if not _curl(url, ogv):
        print("fail porch_theft.ogv")
        return False
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", ogv, "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", mp4,
        ])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("fail ffmpeg porch_theft.mp4 (cần ffmpeg)")
        return False
    finally:
        if os.path.isfile(ogv):
            os.remove(ogv)
    print(f"ok {mp4}")
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, url in VIDEOS:
        dest = os.path.join(OUT, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 10000:
            print(f"skip {dest}")
            continue
        if _curl(url, dest):
            print(f"ok {dest}")
        else:
            print(f"fail {name}")
    _porch_theft()


if __name__ == "__main__":
    main()
