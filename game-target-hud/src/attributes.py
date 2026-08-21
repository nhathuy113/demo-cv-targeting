"""
Phân loại chi tiết trên box person: nam/nữ, tuổi, threat/theft_like_posture/normal.

- Face: OpenCV YuNet ONNX
- Age/gender: Levi-Hassner GoogLeNet (ONNX Model Zoo)
- Behavior: CNN 3 class trên crop person (models/attributes/behavior.onnx)
  nhãn mô tả tư thế quan sát được, không suy ý định gian lận.
"""

from dataclasses import dataclass
import os

import cv2
import numpy as np
import onnxruntime as ort

from detector import Detection

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_LABELS = ["nam", "nữ"]
BEHAVIOR_LABELS = ["normal", "threat", "theft_like_posture"]

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ATTR = os.path.join(_ROOT, "models", "attributes")
FACE_ONNX = os.path.join(_ATTR, "face_detection_yunet_2023mar.onnx")
AGE_ONNX = os.path.join(_ATTR, "age_googlenet.onnx")
GENDER_ONNX = os.path.join(_ATTR, "gender_googlenet.onnx")
BEHAVIOR_ONNX = os.path.join(_ATTR, "behavior.onnx")

AGE_MEAN = (104.0, 117.0, 123.0)
FACE_SIZE = 224
BEHAVIOR_SIZE = 64


@dataclass
class PersonAttributes:
    gender: str | None
    age: str | None
    behavior: str | None
    gender_conf: float = 0.0
    age_conf: float = 0.0
    behavior_conf: float = 0.0


def format_person_label(class_name: str, attrs: PersonAttributes | None) -> str:
    if class_name != "person" or attrs is None:
        return class_name
    gender = attrs.gender or "?"
    age = attrs.age or "?"
    behavior = attrs.behavior or "?"
    return f"{class_name} {gender} {age} {behavior}"


def _face_center_in_box(face, det: Detection) -> bool:
    x, y, w, h = face[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    return det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y2


class AttributeEstimator:
    def __init__(self, attr_dir: str | None = None):
        d = attr_dir or _ATTR
        face_path = os.path.join(d, os.path.basename(FACE_ONNX))
        age_path = os.path.join(d, os.path.basename(AGE_ONNX))
        gender_path = os.path.join(d, os.path.basename(GENDER_ONNX))
        behavior_path = os.path.join(d, os.path.basename(BEHAVIOR_ONNX))
        self._yunet = cv2.FaceDetectorYN.create(face_path, "", (320, 320), 0.7, 0.3, 5000)
        self._age = ort.InferenceSession(age_path, providers=["CPUExecutionProvider"])
        self._gender = ort.InferenceSession(gender_path, providers=["CPUExecutionProvider"])
        self._behavior_sess = ort.InferenceSession(behavior_path, providers=["CPUExecutionProvider"])
        self._age_in = self._age.get_inputs()[0].name
        self._gender_in = self._gender.get_inputs()[0].name
        self._behavior_in = self._behavior_sess.get_inputs()[0].name
        self.last_faces = []

    def _detect_faces(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame_bgr)
        if faces is None:
            return []
        return list(faces)

    def _match_face(self, faces, det: Detection):
        best = None
        best_area = -1.0
        for face in faces:
            if not _face_center_in_box(face, det):
                continue
            area = float(face[2] * face[3])
            if area > best_area:
                best, best_area = face, area
        return best

    def _crop(self, frame_bgr, x, y, w, h):
        ih, iw = frame_bgr.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(iw, int(x + w))
        y2 = min(ih, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def _age_gender(self, face_bgr: np.ndarray) -> tuple[str | None, float, str | None, float]:
        if face_bgr is None or face_bgr.size == 0:
            return None, 0.0, None, 0.0
        blob = cv2.dnn.blobFromImage(
            face_bgr, 1.0, (FACE_SIZE, FACE_SIZE), AGE_MEAN, swapRB=False, crop=False,
        )
        age_out = self._age.run(None, {self._age_in: blob})[0][0]
        gen_out = self._gender.run(None, {self._gender_in: blob})[0][0]
        ai = int(np.argmax(age_out))
        gi = int(np.argmax(gen_out))
        return GENDER_LABELS[gi], float(gen_out[gi]), AGE_BUCKETS[ai], float(age_out[ai])

    def _behavior(self, person_bgr: np.ndarray) -> tuple[str | None, float]:
        if person_bgr is None or person_bgr.size == 0:
            return None, 0.0
        rgb = cv2.resize(person_bgr, (BEHAVIOR_SIZE, BEHAVIOR_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None, ...]
        logits = self._behavior_sess.run(None, {self._behavior_in: blob})[0][0]
        ex = np.exp(logits - np.max(logits))
        prob = ex / np.sum(ex)
        bi = int(np.argmax(prob))
        return BEHAVIOR_LABELS[bi], float(prob[bi])

    def estimate(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[PersonAttributes | None]:
        faces = self._detect_faces(frame_bgr)
        out: list[PersonAttributes | None] = []
        for det in detections:
            if det.class_name != "person":
                out.append(None)
                continue
            face = self._match_face(faces, det)
            gender, gconf, age, aconf = None, 0.0, None, 0.0
            if face is not None and face[2] >= 20 and face[3] >= 20:
                crop = self._crop(frame_bgr, face[0], face[1], face[2], face[3])
                gender, gconf, age, aconf = self._age_gender(crop)
            person = self._crop(frame_bgr, det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1)
            behavior, bconf = self._behavior(person)
            out.append(PersonAttributes(gender, age, behavior, gconf, aconf, bconf))
        self.last_faces = faces
        return out
