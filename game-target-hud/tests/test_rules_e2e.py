"""
E2E Business Rules Layer + event log.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_rules_e2e.py
"""

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import cv2

from attributes import PersonAttributes
from cctv import _app, run_cctv
from event_log import EventLog
from i18n import parse_lang
from rules import (
    BEHAVIOR_CONF_THRESHOLD,
    OVERLAY_GENDER,
    UNCLEAR,
    display_behavior,
    format_display_label,
    hold_track_attrs,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "output")
DATA = os.path.join(os.path.dirname(__file__), "input")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")
MIX = os.path.join(DATA, "mix.mp4")
SHOP = os.path.join(DATA, "crime_real", "Shoplifting_Shoplifting001_x264_clip30s.mp4")
KEYS = ("ts_start", "ts_end", "class", "avg_confidence", "frame_thumbnail_path")
DISPLAY_CLASSES = {"normal", "unusual_motion", "theft_like_posture", UNCLEAR}


def test_behavior_threshold_and_names():
    assert display_behavior("normal", 0.99) == "normal"
    assert display_behavior("threat", 0.99) == "unusual_motion"
    assert display_behavior("fraud", 0.99) == "theft_like_posture"
    assert display_behavior("theft_like_posture", 0.99) == "theft_like_posture"
    assert display_behavior("threat", BEHAVIOR_CONF_THRESHOLD) == UNCLEAR
    assert display_behavior("normal", 0.5) == UNCLEAR
    assert display_behavior(None, 0.99) == UNCLEAR


def test_overlay_label_categories_pipe():
    attrs = PersonAttributes("nam", "25-32", "normal", 0.9, 0.9, 0.99)
    label = format_display_label("person", attrs)
    print(f"display label={label}")
    assert label == "male age 25-32 | behavior: normal"
    assert "person" not in label
    assert "typical" not in label
    vie = format_display_label("person", attrs, lang=parse_lang("vie"))
    assert vie == "nam tuổi 25-32 | hành vi: bình thường"
    threat = PersonAttributes("nam", "25-32", "threat", 0.9, 0.9, 0.99)
    assert format_display_label("person", threat) == "male age 25-32 | behavior: unusual_motion"
    assert format_display_label("person", threat, lang=parse_lang("vie")) == "nam tuổi 25-32 | hành vi: unusual_motion"
    missing = PersonAttributes(None, None, "normal", 0.0, 0.0, 0.99)
    no_age = format_display_label("person", missing)
    print(f"missing age label={no_age}")
    assert no_age == "age unclear | behavior: normal"
    assert "?" not in no_age
    assert format_display_label("person", attrs, overlay_fields=OVERLAY_GENDER) == "male"
    missing_g = format_display_label("person", missing, overlay_fields=OVERLAY_GENDER)
    assert missing_g == "person"
    assert "theft_like" not in missing_g
    assert "age" not in missing_g


def test_hold_track_attrs_keeps_age_behavior():
    locked = PersonAttributes("nam", "25-32", "theft_like_posture", 0.9, 0.8, 0.95)
    miss_face = PersonAttributes(None, None, "normal", 0.0, 0.0, 0.4)
    held = hold_track_attrs(locked, miss_face)
    assert held.age == "25-32"
    assert held.gender is None
    assert held.behavior == "theft_like_posture"
    update = PersonAttributes("nữ", "38-43", "normal", 0.9, 0.8, 0.99)
    next_a = hold_track_attrs(held, update)
    assert next_a.age == "38-43"
    assert next_a.behavior == "normal"


def test_event_log_merges_same_class():
    path = os.path.join(OUT, "events_unit.json")
    os.makedirs(OUT, exist_ok=True)
    log = EventLog(path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    log.apply("start", 0.0, "normal", 0.9, frame)
    log.apply("continue", 0.1, "normal", 0.8, frame)
    log.apply("switch", 0.2, "unusual_motion", 0.7, frame)
    log.apply("continue", 0.3, "unusual_motion", 0.6, frame)
    log.finish()
    log.write()
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"unit events={records}")
    assert len(records) == 2
    assert records[0]["class"] == "normal"
    assert records[0]["ts_start"] == 0.0
    assert records[0]["ts_end"] == 0.1
    assert abs(records[0]["avg_confidence"] - 0.85) < 1e-9
    assert records[1]["class"] == "unusual_motion"
    assert records[1]["ts_start"] == 0.2
    assert records[1]["ts_end"] == 0.3
    assert os.path.isfile(records[0]["frame_thumbnail_path"])
    assert os.path.isfile(records[1]["frame_thumbnail_path"])


def test_cctv_writes_events_json():
    out_path = os.path.join(OUT, "mix_rules.mp4")
    events_path = os.path.join(OUT, "events.json")
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
    print(f"cctv rules stats={stats}")
    assert stats["frames"] == 4
    assert os.path.isfile(events_path)
    with open(events_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"e2e events={records}")
    assert records, "events.json rỗng"
    for rec in records:
        assert tuple(rec.keys()) == KEYS, rec.keys()
        assert rec["class"] in DISPLAY_CLASSES, rec
        assert rec["ts_end"] >= rec["ts_start"]
        assert os.path.isfile(rec["frame_thumbnail_path"])


def test_byte_low_conf_keeps_box():
    from cctv import _OverlayTracks
    from detector import Detection

    tr = _OverlayTracks(high=0.35)
    for i in range(6):
        x1 = 10.0 + i * 8.0
        d = Detection(x1, 20.0, x1 + 24.0, 90.0, 0.9, 0)
        out = tr.update([d], i / 30.0)
        assert len(out) == 1
    x1 = 10.0 + 6 * 8.0
    low = Detection(x1, 20.0, x1 + 24.0, 90.0, 0.15, 0)
    out = tr.update([low], 6 / 30.0)
    assert len(out) == 1
    last = out[0]
    far = Detection(last.x1 + 80.0, last.y1, last.x2 + 80.0, last.y2, 0.9, 0)
    jumped = tr.update([far], 7 / 30.0)
    assert len(jumped) == 1
    ncx = (jumped[0].x1 + jumped[0].x2) * 0.5
    fcx = (far.x1 + far.x2) * 0.5
    assert abs(ncx - fcx) < 0.01, (ncx, fcx)
    empty = tr.update([], 8 / 30.0)
    assert len(empty) == 1
    assert abs(((empty[0].x1 + empty[0].x2) * 0.5) - ncx) < 0.01
    only_low = _OverlayTracks(high=0.35)
    ghost = only_low.update([Detection(40.0, 20.0, 64.0, 90.0, 0.12, 0)], 0.0)
    assert ghost == []
    lock = PersonAttributes("nam", "25-32", "normal", 0.9, 0.8, 0.99)
    miss = PersonAttributes(None, None, "normal", 0.0, 0.0, 0.3)
    tr.hold_attrs([lock])
    kept = tr.hold_attrs([miss])
    assert kept[0].age == "25-32"
    assert kept[0].gender is None
    assert format_display_label("person", kept[0]) == "age 25-32 | behavior: normal"
    assert format_display_label("person", kept[0], overlay_fields=OVERLAY_GENDER) == "person"


def test_shoplifting_overlay_person_no_qmark():
    from attributes import AttributeEstimator
    from cctv import _OverlayTracks
    from detector import Detector

    assert os.path.isfile(SHOP), f"thiếu {SHOP}"
    cap = cv2.VideoCapture(SHOP)
    det = Detector(MODEL, conf_threshold=0.10)
    est = AttributeEstimator()
    tracks = _OverlayTracks(high=0.35)
    n_person_frames = 0
    saw_unlabeled_age = False
    started = False
    gaps_after = 0
    for i in range(200):
        ok, frame = cap.read()
        if not ok:
            break
        raw = det.detect(frame)
        persons = [d for d in raw if d.class_name == "person"]
        draw = tracks.update(persons, i / 30.0)
        assert all(d.class_name == "person" for d in draw)
        attrs = tracks.hold_attrs(est.estimate(frame, draw))
        labels = [format_display_label(d.class_name, a, overlay_fields=OVERLAY_GENDER)
                  for d, a in zip(draw, attrs)]
        for lab in labels:
            assert "?" not in lab, lab
            assert "theft_like" not in lab
            assert "age " not in lab
            assert "behavior" not in lab
            if lab == "person":
                saw_unlabeled_age = True
        if any(p.conf >= 0.35 for p in persons) or (started and persons):
            assert draw, "BYTE miss khi vẫn có person box"
        if draw:
            started = True
            n_person_frames += 1
        elif started:
            gaps_after += 1
    cap.release()
    print(f"shoplifting person_frames={n_person_frames} gaps_after={gaps_after} tuoi_chua_ro={saw_unlabeled_age}")
    assert n_person_frames >= 1


if __name__ == "__main__":
    test_behavior_threshold_and_names()
    test_overlay_label_categories_pipe()
    test_hold_track_attrs_keeps_age_behavior()
    test_event_log_merges_same_class()
    test_cctv_writes_events_json()
    test_byte_low_conf_keeps_box()
    test_shoplifting_overlay_person_no_qmark()
    print("\nAll rules e2e tests passed.")
