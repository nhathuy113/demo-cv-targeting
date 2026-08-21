"""
E2E Level 2: model custom 1 class person → geometry → overlay.

Chạy từ game-target-hud:
  .venv/bin/python tests/test_level2_e2e.py
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from detector import Detector, LEVEL2_CLASSES
from overlay import DetectionOverlay

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(os.path.dirname(__file__), "input")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL = os.path.join(ROOT, "models", "yolov8n_person.onnx")
IMAGES = ["bus.jpg", "coco_000000001000.jpg"]


class FrameOverlay(DetectionOverlay):
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


def test_level2_model_is_one_class_person():
    assert os.path.isfile(MODEL), f"thiếu {MODEL} — chạy scripts/train_level2.py trước"
    detector = Detector(MODEL, conf_threshold=0.35, class_filter=None)
    assert detector.nc == 1
    assert detector.class_names == LEVEL2_CLASSES
    frame = cv2.imread(os.path.join(DATA_DIR, "bus.jpg"))
    detections = detector.detect(frame)
    names = {d.class_name for d in detections}
    print(f"bus.jpg L2: n={len(detections)} names={names}")
    assert "bus" not in names
    assert len(detections) >= 1
    for d in detections:
        assert d.class_name == "person"
    print("OK: L2 model 1 class person, không ra class COCO khác")


def test_level2_pipeline_overlay():
    os.makedirs(OUT_DIR, exist_ok=True)
    _app()
    detector = Detector(MODEL, conf_threshold=0.35, class_filter=[0])
    for name in IMAGES:
        frame = cv2.imread(os.path.join(DATA_DIR, name))
        h, w = frame.shape[:2]
        t0 = time.time()
        detections = detector.detect(frame)
        items = []
        for d in detections:
            items.append((d.x1, d.y1, d.x2, d.y2, d.class_name))
        elapsed_ms = (time.time() - t0) * 1000
        assert len(detections) >= 1, f"{name}: không detect person"
        overlay = FrameOverlay(frame)
        overlay.update_detections(items)
        image = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        image.fill(QtCore.Qt.black)
        overlay.render(image)
        out_path = os.path.join(OUT_DIR, name)
        image.save(out_path)
        print(f"OK: {name} L2 detections={len(items)} latency={elapsed_ms:.1f}ms -> {out_path}")
        for item in items:
            print(f"  {item[4]}")


if __name__ == "__main__":
    _app()
    test_level2_model_is_one_class_person()
    test_level2_pipeline_overlay()
    print("\nAll level2 e2e tests passed.")
