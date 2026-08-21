"""
MVP Real-ESRGAN ncnn-vulkan (x4plus). Thiếu binary/model → False (caller dùng cubic).

Không đọc tests/input/**. Không train.
"""

import os
import subprocess
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_DIR = os.path.join(ROOT, "tools", "realesrgan-ncnn-vulkan")
ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/"
    "v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-macos.zip"
)
MODELS_ZIP_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
)
MODEL = "realesrgan-x4plus"
UA = "demo-cv-targeting/1.0 (academic research; local demo)"


def _curl(url: str, dest: str) -> None:
    subprocess.check_call([
        "curl", "-fL", "-A", UA, "--retry", "2", "--connect-timeout", "30",
        "--max-time", "180", "-o", dest, url,
    ])


def ncnn_bin() -> str | None:
    for dirpath, _, files in os.walk(TOOL_DIR):
        for f in files:
            if f in ("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"):
                p = os.path.join(dirpath, f)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
    return None


def _ensure_models(bin_path: str) -> str | None:
    models = os.path.join(os.path.dirname(bin_path), "models")
    param = os.path.join(models, f"{MODEL}.param")
    blob = os.path.join(models, f"{MODEL}.bin")
    if os.path.isfile(param) and os.path.isfile(blob):
        return models
    os.makedirs(TOOL_DIR, exist_ok=True)
    zpath = os.path.join(TOOL_DIR, "models-bundle.zip")
    _curl(MODELS_ZIP_URL, zpath)
    os.makedirs(models, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if base in (f"{MODEL}.param", f"{MODEL}.bin"):
                with zf.open(name) as src, open(os.path.join(models, base), "wb") as dst:
                    dst.write(src.read())
    os.remove(zpath)
    if os.path.isfile(param) and os.path.isfile(blob):
        return models
    return None


def fetch_ncnn() -> str | None:
    got = ncnn_bin()
    if got:
        os.chmod(got, 0o755)
        subprocess.call(["xattr", "-dr", "com.apple.quarantine", got],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return got
    os.makedirs(TOOL_DIR, exist_ok=True)
    zpath = os.path.join(TOOL_DIR, "bundle.zip")
    _curl(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(TOOL_DIR)
    os.remove(zpath)
    for dirpath, _, files in os.walk(TOOL_DIR):
        for f in files:
            if f == "realesrgan-ncnn-vulkan":
                p = os.path.join(dirpath, f)
                os.chmod(p, 0o755)
                subprocess.call(["xattr", "-dr", "com.apple.quarantine", p],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return p
    return None


def upscale_dir(src_dir: str, dest_dir: str) -> bool:
    """Folder in → folder out, model x4plus. False nếu binary/model lỗi."""
    bin_path = fetch_ncnn()
    if not bin_path:
        return False
    models = _ensure_models(bin_path)
    if not models:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        bin_path, "-i", src_dir, "-o", dest_dir,
        "-n", MODEL, "-s", "4", "-f", "jpg", "-m", models,
    ]
    try:
        subprocess.check_call(cmd, cwd=os.path.dirname(bin_path))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return any(n.endswith(".jpg") for n in os.listdir(dest_dir))
