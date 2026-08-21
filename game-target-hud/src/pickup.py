"""
Pickup expert: ResNet18 2 class none / item_pickup trên crop person.
Input 224, ImageNet mean/std. Không đọc tests/input/**.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from detector import Detection

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PICKUP_ONNX = os.path.join(_ROOT, "models", "attributes", "pickup.onnx")
PICKUP_SIZE = 224
PICKUP_LABELS = ["none", "item_pickup"]
PICKUP_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PICKUP_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PickupEstimator:
    def __init__(self, onnx_path: str | None = None):
        path = onnx_path or PICKUP_ONNX
        if not os.path.isfile(path):
            raise FileNotFoundError(f"BLOCKER: thiếu pickup model {path}")
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._inp = self._sess.get_inputs()[0].name

    def estimate(self, frame_bgr: np.ndarray, persons: list[Detection]) -> list[bool]:
        out = []
        ih, iw = frame_bgr.shape[:2]
        for det in persons:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(iw, int(det.x2))
            y2 = min(ih, int(det.y2))
            if x2 - x1 < 16 or y2 - y1 < 16:
                out.append(False)
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                out.append(False)
                continue
            rgb = cv2.resize(crop, (PICKUP_SIZE, PICKUP_SIZE))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - PICKUP_MEAN) / PICKUP_STD
            blob = rgb.transpose(2, 0, 1)[None, ...]
            logits = self._sess.run(None, {self._inp: blob})[0][0]
            out.append(int(np.argmax(logits)) == 1)
        return out
