"""
detector.py — YOLOv8n ONNX inference wrapper.

Level 1 dùng thẳng COCO pretrained classes (không cần train). Class ID
đáng chú ý cho use-case này: 0 = person. Level 2 swap model ONNX fine-tune
(1 class `person`, Phương án B) — interface Detector.detect() giữ nguyên,
chỉ đổi file model + class list.
"""

from dataclasses import dataclass
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Level 2 Phương án B: 1 class chung (person thay tank — user chốt).
LEVEL2_CLASSES = ["person"]
ZONE_CLASSES = ["cashier", "cash", "exit", "shelf"]
DOORWAY_LABEL = "possible doorway exit"
DOORWAY_BOX = 36


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_names: list[str] | None = None

    @property
    def class_name(self) -> str:
        names = self.class_names if self.class_names is not None else COCO_CLASSES
        return names[self.class_id] if self.class_id < len(names) else str(self.class_id)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def doorway_overlay_dets(dets: list[Detection]) -> list[Detection]:
    """Khung nhỏ overlay cửa; không vẽ cashier/cash/shelf."""
    out = []
    half = DOORWAY_BOX / 2.0
    for d in dets:
        if d.class_name != "exit":
            continue
        cx, cy = d.center
        out.append(Detection(
            cx - half, cy - half, cx + half, cy + half,
            d.conf, 0, class_names=[DOORWAY_LABEL],
        ))
    return out


class Detector:
    def __init__(self, model_path: str, conf_threshold: float = 0.35, iou_threshold: float = 0.45,
                 class_filter: list[int] | None = None, class_names: list[str] | None = None):
        """
        class_filter: chỉ giữ lại detections thuộc các class_id này (vd [0] để chỉ giữ 'person'
        làm proxy cho 'target' ở Level 1 test). None = giữ tất cả.
        """
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,H,W]
        self.imgsz = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = set(class_filter) if class_filter else None
        out_ch = self.session.get_outputs()[0].shape[1]
        self.nc = out_ch - 4 if isinstance(out_ch, int) else 80
        if class_names is not None:
            self.class_names = class_names
        elif self.nc == 80:
            self.class_names = COCO_CLASSES
        elif self.nc == 1:
            self.class_names = LEVEL2_CLASSES
        else:
            self.class_names = [str(i) for i in range(self.nc)]

    def _letterbox(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        import cv2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = self.imgsz - new_w, self.imgsz - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, r, left, top

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        import cv2
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        if len(idxs) == 0:
            return []
        return idxs.flatten().tolist()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """frame_bgr: ảnh OpenCV BGR (vd từ mss screen capture, sau cv2.cvtColor). Trả list Detection theo
        toạ độ pixel gốc của frame_bgr (đã map ngược lại từ letterbox)."""
        h0, w0 = frame_bgr.shape[:2]
        padded, r, pad_left, pad_top = self._letterbox(frame_bgr)

        blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
        blob = blob.transpose(2, 0, 1)[None, ...]  # NCHW

        outputs = self.session.run(None, {self.input_name: blob})[0]  # (1, 4+nc, 8400)
        preds = outputs[0].T  # (8400, 4+nc): 4 box + nc class scores

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confs > self.conf_threshold
        boxes_xywh, class_ids, confs = boxes_xywh[keep], class_ids[keep], confs[keep]

        if self.class_filter is not None and len(class_ids) > 0:
            mask = np.array([c in self.class_filter for c in class_ids])
            boxes_xywh, class_ids, confs = boxes_xywh[mask], class_ids[mask], confs[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh (center) -> xyxy in letterboxed space
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        nms_boxes = np.stack([x1, y1, w, h], axis=1)

        keep_idx = self._nms(nms_boxes, confs)

        results = []
        for i in keep_idx:
            bx1 = x1[i] - pad_left
            by1 = y1[i] - pad_top
            bx2 = bx1 + w[i]
            by2 = by1 + h[i]
            # map back to original frame scale
            bx1, by1, bx2, by2 = bx1 / r, by1 / r, bx2 / r, by2 / r
            bx1 = max(0.0, min(bx1, w0))
            by1 = max(0.0, min(by1, h0))
            bx2 = max(0.0, min(bx2, w0))
            by2 = max(0.0, min(by2, h0))
            results.append(Detection(bx1, by1, bx2, by2, float(confs[i]), int(class_ids[i]),
                                    class_names=self.class_names))
        return results
