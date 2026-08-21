"""
Tải video crime/surveillance THẬT (UCF-Crime + Wikimedia), không phải stock diễn.

Chạy từ game-target-hud:
  .venv/bin/python scripts/download_crime_videos.py
  .venv/bin/python scripts/download_crime_videos.py take-item
  .venv/bin/python scripts/download_crime_videos.py shop-easy
  .venv/bin/python scripts/download_crime_videos.py shop-variety
"""

import os
import subprocess
import sys
from urllib.parse import quote

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "tests", "input", "crime_real")
UA = "demo-cv-targeting/1.0 (academic research; local demo)"
HF = "https://huggingface.co/datasets/mteb/ucf-crime/resolve/main/videos"
HF_SHOP2 = "https://huggingface.co/datasets/Centrique/tcc-shoplifting/resolve/main"
WMC = "https://commons.wikimedia.org/wiki/Special:FilePath"

UCF = [
    "Shoplifting/Shoplifting001_x264.mp4",
    "Shoplifting/Shoplifting005_x264.mp4",
    "Shoplifting/Shoplifting032_x264.mp4",
    "Shoplifting/Shoplifting045_x264.mp4",
    "Shoplifting/Shoplifting052_x264.mp4",
    "Shoplifting/Shoplifting055_x264.mp4",
    "Robbery/Robbery001_x264.mp4",
    "Robbery/Robbery005_x264.mp4",
    "Burglary/Burglary001_x264.mp4",
    "Fighting/Fighting003_x264.mp4",
    "Fighting/Fighting005_x264.mp4",
    "Stealing/Stealing020_x264.mp4",
    "Stealing/Stealing030_x264.mp4",
    "Assault/Assault001_x264.mp4",
]

# Full clip train-only (không phải demo/holdout). Không đặt vào tests/input.
UCF_SHOP_TRAIN = [
    "Shoplifting/Shoplifting003_x264.mp4",
    "Shoplifting/Shoplifting007_x264.mp4",
    "Shoplifting/Shoplifting010_x264.mp4",
    "Shoplifting/Shoplifting021_x264.mp4",
    "Shoplifting/Shoplifting028_x264.mp4",
    "Shoplifting/Shoplifting037_x264.mp4",
    "Shoplifting/Shoplifting041_x264.mp4",
    "Shoplifting/Shoplifting050_x264.mp4",
    "Shoplifting/Shoplifting054_x264.mp4",
]
TRAIN_OUT = os.path.join(ROOT, "dataset", "cctv_shop", "videos")

WIKI = [
    ("Robbery Suspects Caught on Surveillance Camera.webm", "wikimedia_robbery_la.webm"),
    ("Mail Theft Caught on Camera - Las Vegas.ogv", "wikimedia_mail_theft_lv.ogv"),
    ("NYPD Wanted - Robbery (Manhattan).webm", "wikimedia_nypd_robbery.webm"),
    ("Robbery 52 Precinct - 7-25-12.webm", "wikimedia_robbery_nypd52.webm"),
    ("Deputy Catches Man Stealing Beer while Refueling.webm", "wikimedia_steal_beer.webm"),
    ("Bodycam Captures Fleeing Robbery Suspect Crashing Into Deputy's Vehicle.webm",
     "wikimedia_robbery_snatch.webm"),
    ("Detectives Seeking Help to Identify Robbery Suspect.webm", "wikimedia_robbery_detective.webm"),
    ("Front Porch Theft 2012 11 17 101117.ogv", "wikimedia_porch_theft.ogv"),
]

# Full video, không cắt 30s. Cảnh lấy đồ (không bodycam bắt người).
TAKE_ITEM_DIR = os.path.join(OUT, "take_item")
TAKE_ITEM = [
    ("Clique STJ - Furto Supermercado (22-05-2018).webm", "furto_supermercado_1080p.webm"),
    ("Theft@McKean.ogv", "mckean_package_1080p.ogv"),
    ("Package Theft from home.ogv", "package_home.ogv"),
    ("Don't Tempt a Thief with Unattended Parcels.ogv", "porch_vpd_720p.ogv"),
    ("Mail Theft Caught on Camera - Las Vegas.ogv", "mail_theft_lv_720p.ogv"),
]


def _curl(url: str, dest: str, timeout: int = 180) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
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


def _to_mp4(src: str) -> str | None:
    if src.endswith(".mp4"):
        return src
    mp4 = os.path.splitext(src)[0] + ".mp4"
    if os.path.isfile(mp4) and os.path.getsize(mp4) > 5000:
        return mp4
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", src, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return mp4 if os.path.isfile(mp4) and os.path.getsize(mp4) > 5000 else None


def _clip(src_mp4: str, seconds: int = 30) -> str | None:
    clip = src_mp4.replace(".mp4", "_clip30s.mp4")
    if os.path.isfile(clip) and os.path.getsize(clip) > 5000:
        return clip
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-i", src_mp4, "-t", str(seconds), "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", clip,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return clip if os.path.isfile(clip) else None


SHOP_EASY_DIR = os.path.join(OUT, "shop_easy")
SHOP_EASY = [
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_0.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_4.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_12.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_18.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_25.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_32.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_47.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_54.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_61.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_69.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_76.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_83.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_102.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_110.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_111.mp4",
    "Shoplifting Dataset 2.0/shoplifting/shop_lifter_116.mp4",
]


# UCF shop thật, cam khác nhau (không trùng train cctv_shop; không 001/045 glare).
# Đã loại: quầy/văn phòng, workshop, MNNIT mailroom, Dataset 2.0 pantry.
SHOP_VARIETY_DIR = os.path.join(OUT, "shop_variety")
SHOP_VARIETY = [
    "Shoplifting/Shoplifting013_x264.mp4",  # quần áo + mannequin
    "Shoplifting/Shoplifting015_x264.mp4",  # aisle tủ lạnh / đồ gia dụng
    "Shoplifting/Shoplifting018_x264.mp4",  # mini-mart kệ giữa
    "Shoplifting/Shoplifting022_x264.mp4",  # aisle đồ hộp
    "Shoplifting/Shoplifting025_x264.mp4",  # shop quần áo kệ chật
    "Shoplifting/Shoplifting033_x264.mp4",  # boutique bàn giữa
    "Shoplifting/Shoplifting036_x264.mp4",  # shop vải/quần áo
    "Shoplifting/Shoplifting038_x264.mp4",  # quầy kính trang sức
    "Shoplifting/Shoplifting042_x264.mp4",  # shop bạc/đồ kim loại
    "Shoplifting/Shoplifting047_x264.mp4",  # convenience + cửa + checkout
    "Shoplifting/Shoplifting049_x264.mp4",  # shop điện thoại Samsung
    "Shoplifting/Shoplifting016_x264.mp4",  # aisle hộp điện tử
    "Shoplifting/Shoplifting017_x264.mp4",  # convenience + POS + kệ
    "Shoplifting/Shoplifting019_x264.mp4",  # hobby/RC, chạy cầm hộp
    "Shoplifting/Shoplifting026_x264.mp4",  # shop vải, cửa mở
    "Shoplifting/Shoplifting029_x264.mp4",  # shop jeans
    "Shoplifting/Shoplifting031_x264.mp4",  # hardware + quầy
    "Shoplifting/Shoplifting039_x264.mp4",  # cửa kính + kệ
    "Shoplifting/Shoplifting048_x264.mp4",  # quầy + kệ + giày
    "Shoplifting/Shoplifting053_x264.mp4",  # shop kính mắt
]


def _download_shop_variety() -> int:
    tok = 0
    os.makedirs(SHOP_VARIETY_DIR, exist_ok=True)
    for rel in SHOP_VARIETY:
        name = os.path.basename(rel)
        dest = os.path.join(SHOP_VARIETY_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
            print(f"skip {name}")
            tok += 1
            continue
        url = f"{HF}/{rel}"
        if _curl(url, dest, timeout=180):
            print(f"ok {name}")
            tok += 1
        else:
            print(f"fail {name}")
    print(f"shop_variety={tok}/{len(SHOP_VARIETY)}")
    return tok


def _download_shop_easy() -> int:
    tok = 0
    os.makedirs(SHOP_EASY_DIR, exist_ok=True)
    for rel in SHOP_EASY:
        name = os.path.basename(rel)
        dest = os.path.join(SHOP_EASY_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
            print(f"skip {name}")
            tok += 1
            continue
        url = f"{HF_SHOP2}/{quote(rel)}"
        if _curl(url, dest, timeout=180):
            print(f"ok {name}")
            tok += 1
        else:
            print(f"fail {name}")
    print(f"shop_easy={tok}/{len(SHOP_EASY)}")
    return tok


def _download_take_item() -> int:
    tok = 0
    for wiki_name, out_name in TAKE_ITEM:
        dest = os.path.join(TAKE_ITEM_DIR, out_name)
        got = os.path.isfile(dest) and os.path.getsize(dest) > 5000
        if got:
            print(f"skip {out_name}")
        else:
            url = f"{WMC}/{quote(wiki_name)}"
            got = _curl(url, dest, timeout=600)
            print(("ok " if got else "fail ") + out_name)
        if not got:
            continue
        tok += 1
        mp4 = _to_mp4(dest)
        if mp4:
            print(f"mp4 {os.path.basename(mp4)}")
    print(f"take_item={tok}/{len(TAKE_ITEM)}")
    return tok


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "take-item":
        _download_take_item()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "shop-easy":
        _download_shop_easy()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "shop-variety":
        _download_shop_variety()
        return
    ok = 0
    for rel in UCF:
        name = rel.replace("/", "_")
        dest = os.path.join(OUT, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
            print(f"skip {name}")
            ok += 1
            continue
        url = f"{HF}/{rel}"
        if _curl(url, dest):
            print(f"ok {name}")
            ok += 1
        else:
            print(f"fail {name}")
    print(f"ucf={ok}/{len(UCF)}")

    tok = 0
    os.makedirs(TRAIN_OUT, exist_ok=True)
    for rel in UCF_SHOP_TRAIN:
        name = os.path.basename(rel)
        dest = os.path.join(TRAIN_OUT, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
            print(f"skip train {name}")
            tok += 1
            continue
        url = f"{HF}/{rel}"
        if _curl(url, dest):
            print(f"ok train {name}")
            tok += 1
        else:
            print(f"fail train {name}")
    print(f"ucf_shop_train={tok}/{len(UCF_SHOP_TRAIN)}")

    wok = 0
    for wiki_name, out_name in WIKI:
        dest = os.path.join(OUT, out_name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 5000:
            print(f"skip {out_name}")
            wok += 1
            continue
        url = f"{WMC}/{quote(wiki_name)}"
        if _curl(url, dest, timeout=240):
            print(f"ok {out_name}")
            wok += 1
        else:
            print(f"fail {out_name}")
    print(f"wiki={wok}/{len(WIKI)}")
    _download_take_item()

    for fname in sorted(os.listdir(OUT)):
        if not fname.endswith((".mp4", ".webm", ".ogv")):
            continue
        path = os.path.join(OUT, fname)
        mp4 = _to_mp4(path)
        if mp4 and os.path.getsize(mp4) > 5_000_000:
            clip = _clip(mp4, 30)
            if clip:
                print(f"clip {os.path.basename(clip)}")


if __name__ == "__main__":
    main()
