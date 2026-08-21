"""
E2E Level 1: ảnh thật → detector → geometry → overlay (offscreen) → file.

Chạy từ thư mục game-target-hud:
  .venv/bin/python tests/test_e2e.py
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from detector import Detector
from overlay import DetectionOverlay

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "input")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL = os.path.join(ROOT, "models", "yolov8n.onnx")

BUS_JPG = os.path.join(DATA_DIR, "bus.jpg")
ALL_IMAGES = [
    "bus.jpg",
    "zidane.jpg",
    "coco_000000000139.jpg",
    "coco_000000000785.jpg",
    "coco_000000001000.jpg",
]


class FrameOverlay(DetectionOverlay):
    """Overlay spec + nền ảnh, chỉ dùng trong test để xuất demo."""

    def __init__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        super().__init__(w, h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._bg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self._bg)
        painter.end()
        super().paintEvent(event)


_APP = None


def _app():
    global _APP
    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication([])
    return _APP


def _pipeline(detector, frame_bgr):
    detections = detector.detect(frame_bgr)
    items = []
    for d in detections:
        items.append((d.x1, d.y1, d.x2, d.y2, d.class_name))
    return detections, items


def _save_overlay(frame_bgr, items, out_path):
    _app()
    overlay = FrameOverlay(frame_bgr)
    overlay.update_detections(items)
    h, w = frame_bgr.shape[:2]
    image = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
    image.fill(QtCore.Qt.black)
    overlay.render(image)
    ok = image.save(out_path)
    assert ok, f"không ghi được {out_path}"
    assert os.path.getsize(out_path) > 0


def test_bus_jpg_matches_design_ac():
    detector = Detector(MODEL, conf_threshold=0.35, class_filter=None)
    frame = cv2.imread(BUS_JPG)
    assert frame is not None, f"thiếu {BUS_JPG}"
    detections = detector.detect(frame)
    persons = [d for d in detections if d.class_name == "person"]
    buses = [d for d in detections if d.class_name == "bus"]
    confs = [d.conf for d in detections]
    print(
        f"bus.jpg: person={len(persons)} bus={len(buses)} "
        f"conf={min(confs):.2f}–{max(confs):.2f}"
    )
    assert len(persons) == 4, f"expect 4 person, got {len(persons)}"
    assert len(buses) == 1, f"expect 1 bus, got {len(buses)}"
    assert min(confs) >= 0.43 and max(confs) <= 0.90
    print("OK: bus.jpg khớp AC thiết kế (4 person + 1 bus)")


def test_e2e_pipeline_all_images():
    os.makedirs(OUT_DIR, exist_ok=True)
    detector = Detector(MODEL, conf_threshold=0.35, class_filter=None)
    for name in ALL_IMAGES:
        path = os.path.join(DATA_DIR, name)
        frame = cv2.imread(path)
        assert frame is not None, f"thiếu {path}"
        t0 = time.time()
        detections, items = _pipeline(detector, frame)
        elapsed_ms = (time.time() - t0) * 1000
        assert len(detections) >= 1, f"{name}: không detect được object"
        assert len(items) == len(detections)
        names = {item[4] for item in items}
        if name == "bus.jpg":
            assert "person" in names and "bus" in names, names
        out_path = os.path.join(OUT_DIR, name)
        _save_overlay(frame, items, out_path)
        print(
            f"OK: {name} detections={len(items)} latency={elapsed_ms:.1f}ms "
            f"-> {out_path}"
        )
        for item in items:
            print(f"  {item[4]}")


if __name__ == "__main__":
    _app()
    test_bus_jpg_matches_design_ac()
    test_e2e_pipeline_all_images()
    print("\nAll e2e tests passed.")
