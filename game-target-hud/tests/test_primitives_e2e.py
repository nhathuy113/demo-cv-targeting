"""
Trial primitives module: possession_transfer + robbery-prep ≥2 hits.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_primitives_e2e.py
"""

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from cctv import _app, run_cctv, scale_detections, view_scale
from detector import DOORWAY_BOX, DOORWAY_LABEL, Detection, doorway_overlay_dets
from pickup import PICKUP_SIZE
from primitives import PrimitiveEngine, Track, assign_owner, closer_to_doors
from rules import confirm_robbery_prep, confirm_shoplifting_suspected

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "output")
DATA = os.path.join(os.path.dirname(__file__), "input")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")
MIX = os.path.join(DATA, "mix.mp4")
FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _det(x1, y1, x2, y2, cid):
    return Detection(float(x1), float(y1), float(x2), float(y2), 0.9, cid)


def test_confirm_robbery_prep_needs_two():
    assert confirm_robbery_prep(set()) is None
    assert confirm_robbery_prep({"concealment"}) is None
    assert confirm_robbery_prep({"dwell"}) is None
    assert confirm_robbery_prep({"concealment", "dwell"}) == "ROBBERY_PREP_SUSPECTED"


def test_concealment_only_no_flag():
    eng = PrimitiveEngine(dwell_seconds=10.0, max_move_frac=0.01, thumb_dir=None)
    person = _det(0, 0, 40, 80, 0)
    # 2 frame, người chạy → không dwell; không face → concealment
    eng.step(0.0, [person], faces=[], frame_bgr=FRAME)
    ev = eng.step(0.1, [_det(50, 0, 90, 80, 0)], faces=[], frame_bgr=FRAME)
    print(f"concealment-only events={eng.events}")
    assert all(e["class"] != "ROBBERY_PREP_SUSPECTED" for e in eng.events)
    assert ev == [] or all(e["class"] != "ROBBERY_PREP_SUSPECTED" for e in ev)


def test_concealment_and_dwell_flags():
    eng = PrimitiveEngine(dwell_seconds=0.2, max_move_frac=0.5, thumb_dir=None)
    person = _det(0, 0, 40, 80, 0)
    eng.step(0.0, [person], faces=[], frame_bgr=FRAME)
    eng.step(0.2, [person], faces=[], frame_bgr=FRAME)
    ev = eng.step(0.4, [person], faces=[], frame_bgr=FRAME)
    print(f"prep events={eng.events}")
    classes = [e["class"] for e in eng.events]
    assert "ROBBERY_PREP_SUSPECTED" in classes
    rec = [e for e in eng.events if e["class"] == "ROBBERY_PREP_SUSPECTED"][0]
    assert set(rec["primitives"]) >= {"concealment", "dwell"}


def test_possession_transfer():
    eng = PrimitiveEngine(thumb_dir=None)
    pa = _det(0, 0, 50, 100, 0)
    pb = _det(80, 0, 130, 100, 0)
    bag_a = _det(10, 40, 90, 90, 24)
    bag_b = _det(40, 40, 120, 90, 24)
    eng.step(0.0, [pa, pb, bag_a], faces=[], frame_bgr=FRAME)
    ev = eng.step(0.1, [pa, pb, bag_b], faces=[], frame_bgr=FRAME)
    print(f"transfer events={ev}")
    transfers = [e for e in ev if e["class"] == "possession_transfer"]
    assert transfers, ev
    t = transfers[0]
    assert t["from"] != t["to"]
    assert t["object_id"] >= 1


def test_assign_owner_no_overlap():
    person = Track(tid=1, det=_det(0, 0, 10, 10, 0))
    bag = _det(50, 50, 60, 60, 24)
    assert assign_owner(bag, [person]) is None


def test_confirm_shoplifting_needs_two():
    assert confirm_shoplifting_suspected(set(), False) is None
    assert confirm_shoplifting_suspected({"item_pickup"}, False) is None
    assert confirm_shoplifting_suspected({"exit_approach"}, False) is None
    assert confirm_shoplifting_suspected({"item_pickup", "exit_approach"}, True) is None
    assert confirm_shoplifting_suspected({"item_pickup", "exit_approach"}, False) == (
        "SHOPLIFTING_HIGHLY_POSSIBLE"
    )


def test_pickup_flag_binds_one_track():
    eng = PrimitiveEngine(thumb_dir=None)
    pa = _det(0, 0, 50, 100, 0)
    pb = _det(30, 0, 80, 100, 0)
    faces = [[5, 5, 8, 8], [35, 5, 8, 8]]
    eng.step(0.0, [pa, pb], faces=faces, frame_bgr=FRAME, pickup_flags=[True, False])
    assert eng._holding == {1}, eng._holding


def test_pickup_and_exit_flags():
    names = ["cashier", "cash", "exit", "shelf"]
    eng = PrimitiveEngine(thumb_dir=None)
    person = _det(10, 10, 40, 80, 0)
    door = Detection(80.0, 10.0, 110.0, 40.0, 0.9, 2, class_names=names)
    eng.step(0.0, [person], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
             zone_dets=[door], pickup_flags=[True])
    person2 = _det(20, 10, 50, 80, 0)
    ev = eng.step(0.1, [person2], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
                  zone_dets=[door], pickup_flags=[True])
    print(f"shop events={eng.events}")
    assert any(e["class"] == "SHOPLIFTING_HIGHLY_POSSIBLE" for e in eng.events)
    assert ev and ev[0]["primitives"] == ["exit_approach", "item_pickup"]


def test_cashier_visit_blocks_shoplifting():
    names = ["cashier", "cash", "exit", "shelf"]
    eng = PrimitiveEngine(thumb_dir=None)
    person = _det(10, 10, 50, 80, 0)
    cashier = Detection(20.0, 20.0, 40.0, 60.0, 0.9, 0, class_names=names)
    door = Detection(80.0, 10.0, 110.0, 40.0, 0.9, 2, class_names=names)
    eng.step(0.0, [person], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
             zone_dets=[cashier, door], pickup_flags=[True])
    person2 = _det(20, 10, 60, 80, 0)
    ev = eng.step(0.1, [person2], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
                  zone_dets=[cashier, door], pickup_flags=[True])
    print(f"blocked shop events={eng.events} ev={ev}")
    assert all(e["class"] != "SHOPLIFTING_HIGHLY_POSSIBLE" for e in eng.events)


def test_door_persists_when_zone_empty():
    names = ["cashier", "cash", "exit", "shelf"]
    eng = PrimitiveEngine(thumb_dir=None)
    person = _det(10, 10, 40, 80, 0)
    door = Detection(80.0, 10.0, 110.0, 40.0, 0.9, 2, class_names=names)
    eng.step(0.0, [person], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
             zone_dets=[door], pickup_flags=[True])
    person2 = _det(20, 10, 50, 80, 0)
    ev = eng.step(0.1, [person2], faces=[[0, 0, 10, 10]], frame_bgr=FRAME,
                  zone_dets=[], pickup_flags=[True])
    print(f"persist shop events={eng.events} ev={ev}")
    assert any(e["class"] == "SHOPLIFTING_HIGHLY_POSSIBLE" for e in eng.events)


def test_doorway_overlay_small_label():
    names = ["cashier", "cash", "exit", "shelf"]
    door = Detection(80.0, 10.0, 110.0, 40.0, 0.9, 2, class_names=names)
    cash = Detection(0.0, 0.0, 40.0, 40.0, 0.9, 0, class_names=names)
    out = doorway_overlay_dets([door, cash])
    assert len(out) == 1
    assert out[0].class_name == DOORWAY_LABEL
    assert (out[0].x2 - out[0].x1) == DOORWAY_BOX


def test_closer_to_doors():
    names = ["cashier", "cash", "exit", "shelf"]
    door = Detection(80.0, 10.0, 110.0, 40.0, 0.9, 2, class_names=names)
    tr = Track(tid=1, det=_det(10, 10, 40, 80, 0),
               centers=[(20.0, 40.0, 0.0), (40.0, 40.0, 0.1)])
    assert closer_to_doors(tr, [door])
    tr2 = Track(tid=2, det=_det(10, 10, 40, 80, 0),
                centers=[(40.0, 40.0, 0.0), (10.0, 40.0, 0.1)])
    assert not closer_to_doors(tr2, [door])


def test_cctv_writes_pattern_json():
    out_path = os.path.join(OUT, "mix_prim.mp4")
    pattern_path = os.path.join(OUT, "pattern_events.json")
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
    print(f"cctv prim stats={stats}")
    assert stats["frames"] == 4
    assert os.path.isfile(pattern_path)
    with open(pattern_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"pattern_events={records}")
    assert isinstance(records, list)
    cap = cv2.VideoCapture(out_path)
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert oh >= 720, oh


def test_view_scale_to_720():
    assert view_scale(240) == 3
    assert view_scale(480) == 2
    assert view_scale(720) == 1
    assert view_scale(1080) == 1


def test_scale_detections_xy():
    d = Detection(10, 20, 30, 40, 0.9, 0)
    out = scale_detections([d], 4.0, 4.0)
    assert (out[0].x1, out[0].y1, out[0].x2, out[0].y2) == (40, 80, 120, 160)


def test_pickup_input_224():
    assert PICKUP_SIZE == 224


if __name__ == "__main__":
    test_confirm_robbery_prep_needs_two()
    test_concealment_only_no_flag()
    test_concealment_and_dwell_flags()
    test_possession_transfer()
    test_assign_owner_no_overlap()
    test_confirm_shoplifting_needs_two()
    test_pickup_flag_binds_one_track()
    test_pickup_and_exit_flags()
    test_cashier_visit_blocks_shoplifting()
    test_door_persists_when_zone_empty()
    test_doorway_overlay_small_label()
    test_closer_to_doors()
    test_cctv_writes_pattern_json()
    test_view_scale_to_720()
    test_scale_detections_xy()
    test_pickup_input_224()
    print("\nAll primitives e2e tests passed.")
