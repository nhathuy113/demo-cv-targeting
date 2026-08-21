"""
E2E attributes: person → nam/nữ, tuổi, threat/theft_like_posture/normal.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_attributes_e2e.py
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from attributes import AGE_BUCKETS, BEHAVIOR_LABELS, GENDER_LABELS, AttributeEstimator, format_person_label
from cctv import _app, annotate_frame, run_cctv
from detector import Detector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(os.path.dirname(__file__), "input")
OUT = os.path.join(os.path.dirname(__file__), "output")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")
ZIDANE = os.path.join(DATA, "zidane.jpg")
THREAT = os.path.join(DATA, "attributes", "threat", "00_Boxing080905.jpg")
FRAUD = os.path.join(DATA, "attributes", "fraud", "00_Pickpocket.jpg")
MIX = os.path.join(DATA, "mix.mp4")


def test_zidane_gender_age_behavior():
    detector = Detector(MODEL, conf_threshold=0.35, class_filter=None)
    est = AttributeEstimator()
    frame = cv2.imread(ZIDANE)
    assert frame is not None, f"thiếu {ZIDANE}"
    detections = detector.detect(frame)
    persons = [d for d in detections if d.class_name == "person"]
    assert len(persons) >= 1, f"không có person: {[d.class_name for d in detections]}"
    attrs = est.estimate(frame, detections)
    labels = [format_person_label(d.class_name, a) for d, a in zip(detections, attrs)]
    person_attrs = [a for d, a in zip(detections, attrs) if d.class_name == "person"]
    genders = [a.gender for a in person_attrs if a and a.gender]
    behaviors = [a.behavior for a in person_attrs if a and a.behavior]
    ages = [a.age for a in person_attrs if a and a.age]
    print(f"zidane labels={labels}")
    assert genders, "không infer được giới tính"
    assert all(g in GENDER_LABELS for g in genders)
    assert "nam" in genders
    assert all(b in BEHAVIOR_LABELS for b in behaviors)
    assert all(a in AGE_BUCKETS for a in ages)
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "zidane_attr.jpg")
    _app()
    annotated = annotate_frame(frame, detections, labels=labels)
    assert cv2.imwrite(out_path, annotated)
    print(f"OK: zidane attrs -> {out_path}")


def test_threat_and_fraud_images_have_behavior():
    detector = Detector(MODEL, conf_threshold=0.25, class_filter=None)
    est = AttributeEstimator()
    for path, tag in ((THREAT, "threat"), (FRAUD, "theft_like_posture")):
        frame = cv2.imread(path)
        assert frame is not None, f"thiếu {path}"
        detections = detector.detect(frame)
        attrs = est.estimate(frame, detections)
        labels = [format_person_label(d.class_name, a) for d, a in zip(detections, attrs)]
        behaviors = [a.behavior for d, a in zip(detections, attrs) if d.class_name == "person" and a]
        print(f"{tag} labels={labels}")
        assert behaviors, f"{tag}: không có person/behavior"
        assert all(b in BEHAVIOR_LABELS for b in behaviors)
        os.makedirs(OUT, exist_ok=True)
        out_path = os.path.join(OUT, f"{tag}_attr.jpg")
        _app()
        cv2.imwrite(out_path, annotate_frame(frame, detections, labels=labels))


def test_cctv_attributes_few_frames():
    out_path = os.path.join(OUT, "mix_attr.mp4")
    _app()
    stats = run_cctv(
        source=MIX,
        model_path=MODEL,
        conf=0.35,
        target_classes=None,
        out_path=out_path,
        show=False,
        max_frames=4,
        attributes=True,
    )
    print(f"cctv attr stats={stats}")
    assert stats["frames"] == 4
    assert os.path.isfile(out_path) and os.path.getsize(out_path) > 0


if __name__ == "__main__":
    _app()
    test_zidane_gender_age_behavior()
    test_threat_and_fraud_images_have_behavior()
    test_cctv_attributes_few_frames()
    print("\nAll attributes e2e tests passed.")
