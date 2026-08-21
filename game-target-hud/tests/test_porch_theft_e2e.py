"""
E2E: wikimedia_porch_theft.ogv → detect person trộm gói hàng → mp4.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_porch_theft_e2e.py
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from attributes import BEHAVIOR_LABELS, format_person_label, AttributeEstimator
from cctv import _app, _output_mp4_path, run_cctv
from detector import Detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(os.path.dirname(__file__), "input", "crime_real", "wikimedia_porch_theft.ogv")
OUT = os.path.join(os.path.dirname(__file__), "output", "wikimedia_porch_theft.mp4")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")


def test_ogv_output_forced_mp4():
    assert _output_mp4_path(SRC, None).endswith(".mp4")
    assert _output_mp4_path(SRC, "tests/output/wikimedia_porch_theft.ogv").endswith(".mp4")


def test_porch_theft_detects_person_and_writes_mp4():
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
        attributes=True,
    )
    print(f"stats={stats}")
    assert stats["frames"] >= 100, stats
    assert "person" in stats["classes"], f"không detect person: {stats}"
    assert stats["max_count"] >= 1
    assert os.path.isfile(OUT) and os.path.getsize(OUT) > 0
    cap = cv2.VideoCapture(OUT)
    ok, frame = cap.read()
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    assert ok and frame is not None
    assert nfr >= 1
    print(f"OK: porch theft frames={stats['frames']} classes={stats['classes']} -> {OUT}")


def test_porch_theft_held_out_behavior_no_accuracy():
    """Clip này HELD OUT khỏi train. Chỉ log nhãn — không assert class, không report acc."""
    detector = Detector(MODEL, conf_threshold=0.35)
    est = AttributeEstimator()
    cap = cv2.VideoCapture(SRC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    detections = detector.detect(frame)
    persons = [d for d in detections if d.class_name == "person"]
    assert persons, f"frame 80 không có person: {[d.class_name for d in detections]}"
    attrs = est.estimate(frame, detections)
    labels = [format_person_label(d.class_name, a) for d, a in zip(detections, attrs)]
    print(f"HELD_OUT porch_theft frame80 labels={labels}")
    print("chưa có test set độc lập đủ để report accuracy cho theft_like_posture")
    person_attrs = [a for d, a in zip(detections, attrs) if d.class_name == "person" and a]
    assert person_attrs
    assert all(a.behavior in BEHAVIOR_LABELS for a in person_attrs)


if __name__ == "__main__":
    _app()
    test_ogv_output_forced_mp4()
    test_porch_theft_detects_person_and_writes_mp4()
    test_porch_theft_held_out_behavior_no_accuracy()
    print("\nAll porch theft e2e tests passed.")
