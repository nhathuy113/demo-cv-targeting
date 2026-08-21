"""
E2E CCTV: video → detect mọi class COCO → box + đếm theo loại → file mp4.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_cctv_e2e.py
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from cctv import _app, run_cctv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(os.path.dirname(__file__), "input", "mix.mp4")
OUT = os.path.join(os.path.dirname(__file__), "output", "mix.mp4")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")


def test_cctv_video_boxes_and_count():
    assert os.path.isfile(SRC), f"thiếu {SRC}"
    _app()
    stats = run_cctv(
        source=SRC,
        model_path=MODEL,
        conf=0.35,
        target_classes=None,
        out_path=OUT,
        show=False,
        max_frames=None,
    )
    print(f"stats={stats}")
    assert stats["frames"] >= 1
    assert stats["max_count"] >= 1, f"không detect được object: {stats}"
    assert len(stats["classes"]) >= 2, f"phải có nhiều class, got {stats['classes']}"
    assert os.path.isfile(OUT) and os.path.getsize(OUT) > 0

    cap = cv2.VideoCapture(OUT)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    print(f"OK: CCTV {stats['frames']} frames, classes={stats['classes']} -> {OUT}")


if __name__ == "__main__":
    test_cctv_video_boxes_and_count()
    print("\nAll cctv e2e tests passed.")
