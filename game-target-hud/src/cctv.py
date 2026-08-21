"""
cctv.py — video/webcam security-style overlay: boxes, counts, person labels.

Run:
  python src/cctv.py --source tests/input/mix.mp4
  python src/cctv.py --source 0 --show
  python src/cctv.py --lang vie --source tests/input/mix.mp4
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from attributes import AttributeEstimator
from detector import Detector, Detection, ZONE_CLASSES, doorway_overlay_dets
from event_log import EventLog
from i18n import DEFAULT_LANG, parse_lang, t
from overlay import class_label_rect
from pickup import PickupEstimator
from primitives import ByteTracker, PrimitiveEngine
from rules import BusinessRules, hold_track_attrs
from upscale import upscale_dir

_APP = None


def _app():
    global _APP
    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication([])
    return _APP


def _qimage_to_bgr(qimg: QtGui.QImage) -> np.ndarray:
    qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.bits()
    ptr.setsize(bpl * h)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, bpl))[:, : w * 3].reshape((h, w, 3))
    return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)


def _output_mp4_path(source, out_path):
    """Always write .mp4; .ogv/.webm cannot use the mp4v encoder."""
    if out_path:
        root, ext = os.path.splitext(out_path)
        if ext.lower() != ".mp4":
            return root + ".mp4"
        return out_path
    if str(source).isdigit():
        return None
    name = os.path.splitext(os.path.basename(str(source)))[0] + ".mp4"
    return os.path.join("tests", "output", name)


def _events_json_path(out_path):
    if out_path:
        parent = os.path.dirname(out_path) or os.path.join("tests", "output")
        return os.path.join(parent, "events.json")
    return os.path.join("tests", "output", "events.json")


def _pattern_json_path(out_path):
    if out_path:
        parent = os.path.dirname(out_path) or os.path.join("tests", "output")
        return os.path.join(parent, "pattern_events.json")
    return os.path.join("tests", "output", "pattern_events.json")


def _class_banner(detections, lang: str = DEFAULT_LANG) -> str:
    counts = Counter(d.class_name for d in detections)
    if not counts:
        return t("detected_empty", lang)
    parts = ", ".join(f"{n} {name}" for name, n in counts.most_common())
    return t("detected", lang, n=sum(counts.values()), parts=parts)


def annotate_frame(frame_bgr: np.ndarray, detections, labels=None,
                   lang: str = DEFAULT_LANG) -> np.ndarray:
    """Draw red boxes, in-box class text, and a per-class count banner."""
    _app()
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
    painter = QtGui.QPainter(qimg)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    box_pen = QtGui.QPen(QtGui.QColor(255, 60, 60, 230), 3 if h >= VIEW_MIN_H else 2)
    text_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 255))
    font_px = 22 if h >= VIEW_MIN_H else 9
    font = QtGui.QFont("Consolas", font_px, QtGui.QFont.Bold)
    painter.setFont(font)
    text_bg = QtGui.QColor(0, 0, 0, 160)

    banner = _class_banner(detections, lang)
    metrics = painter.fontMetrics()
    bw = metrics.horizontalAdvance(banner) + 16
    bh = metrics.height() + 10
    painter.fillRect(QtCore.QRectF(8, 8, bw, bh), QtGui.QColor(0, 0, 0, 180))
    painter.setPen(text_pen)
    painter.drawText(QtCore.QRectF(16, 8, bw, bh), QtCore.Qt.AlignVCenter, banner)

    for i, d in enumerate(detections):
        label = labels[i] if labels is not None else d.class_name
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
        painter.setPen(box_pen)
        painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
        tx, ty, tw, th = class_label_rect(x1, y1, x2, y2, label, metrics)
        painter.fillRect(QtCore.QRectF(tx, ty, tw, th), text_bg)
        painter.setPen(text_pen)
        painter.drawText(QtCore.QRectF(tx + 4, ty, tw, th),
                         QtCore.Qt.AlignVCenter, label)

    painter.end()
    return _qimage_to_bgr(qimg)


BYTE_LOW = 0.10
VIEW_MIN_H = 720


def view_scale(h: int, min_h: int = VIEW_MIN_H) -> int:
    if h >= min_h:
        return 1
    return (min_h + h - 1) // h


def _overlay_on_sr_view(view, draw, sx, sy, estimator, rules, pickup_est, zone_det,
                        lang: str = DEFAULT_LANG):
    """Re-infer behavior / holding-item / doorway on the SR view. Person boxes stay from original (scaled)."""
    scaled = scale_detections(draw, sx, sy)
    extra = doorway_overlay_dets(zone_det.detect(view)) if zone_det is not None else []
    labels = None
    n_pick = 0
    if estimator is not None:
        labels = rules.overlay_labels(scaled, estimator.estimate(view, scaled))
        if pickup_est is not None:
            n_pick = sum(1 for f in pickup_est.estimate(view, scaled) if f)
    if labels is None:
        all_labels = [d.class_name for d in extra] if extra else None
    else:
        all_labels = labels + [d.class_name for d in extra]
    return scaled + extra, all_labels, n_pick, len(extra)


def scale_detections(dets, sx: float, sy: float | None = None):
    if sy is None:
        sy = sx
    if sx == 1 and sy == 1:
        return list(dets)
    out = []
    for d in dets:
        out.append(Detection(
            d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy,
            d.conf, d.class_id, d.class_names,
        ))
    return out


class _OverlayTracks:
    """BYTE overlay: live box = detector box; lost tracks keep last box until max_lost."""

    def __init__(self, high: float = 0.35):
        self._tr = ByteTracker(high=high, iou_min=0.2, max_lost=30, dist_frac=1.2)
        self._last: dict[int, Detection] = {}
        self._attrs: dict = {}
        self._order: list[int] = []

    def update(self, persons: list[Detection], ts: float) -> list[Detection]:
        self._tr.update(persons, ts)
        out = []
        self._order = []
        live = set()
        for t in self._tr.tracks:
            live.add(t.tid)
            if t.lost != 0:
                d = self._last.get(t.tid)
                if d is not None:
                    out.append(d)
                    self._order.append(t.tid)
                continue
            d = t.det
            self._last[t.tid] = d
            out.append(d)
            self._order.append(t.tid)
        self._last = {k: v for k, v in self._last.items() if k in live}
        self._attrs = {k: v for k, v in self._attrs.items() if k in live}
        return out

    def hold_attrs(self, attrs: list) -> list:
        held = []
        for tid, cur in zip(self._order, attrs):
            merged = hold_track_attrs(self._attrs.get(tid), cur)
            if merged is not None:
                self._attrs[tid] = merged
            held.append(merged)
        return held


def run_cctv(source, model_path, conf, target_classes, out_path, show, max_frames,
             attributes=False, sr=False, lang: str = DEFAULT_LANG):
    det_conf = min(conf, BYTE_LOW) if attributes else conf
    detector = Detector(model_path, conf_threshold=det_conf, class_filter=target_classes)
    estimator = AttributeEstimator() if attributes else None
    rules = BusinessRules(lang=lang) if attributes else None
    event_log = EventLog(_events_json_path(out_path)) if attributes else None
    pattern_path = _pattern_json_path(out_path)
    pattern_thumbs = os.path.join(os.path.dirname(pattern_path) or ".", "pattern_thumbs")
    engine = PrimitiveEngine(thumb_dir=pattern_thumbs) if attributes else None
    overlay_tracks = _OverlayTracks(high=conf) if attributes else None
    zone_onnx = os.path.join(os.path.dirname(__file__), "..", "models", "zone.onnx")
    pickup_onnx = os.path.join(os.path.dirname(__file__), "..", "models", "attributes", "pickup.onnx")
    if attributes:
        if not os.path.isfile(zone_onnx):
            raise FileNotFoundError(t("need_zone", lang, path=zone_onnx))
        if not os.path.isfile(pickup_onnx):
            raise FileNotFoundError(t("need_pickup", lang, path=pickup_onnx))
        zone_det = Detector(zone_onnx, conf_threshold=conf, class_names=ZONE_CLASSES)
        pickup_est = PickupEstimator(pickup_onnx)
    else:
        zone_det = None
        pickup_est = None
    cap_src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(t("need_video", lang, source=source))

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = view_scale(h)
    sw, sh = w * s, h * s
    sr_in = None
    overlay_meta = []
    if sr and out_path and not str(source).isdigit():
        sr_in = tempfile.mkdtemp(prefix="cctv_sr_in_")
    writer = None
    if out_path and sr_in is None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sw, sh))
        if not writer.isOpened():
            raise RuntimeError(t("need_write", lang, path=out_path))

    n_frames = 0
    max_count = 0
    seen_classes = set()
    while True:
        if max_frames is not None and n_frames >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        detections = detector.detect(frame)
        zone_dets = zone_det.detect(frame) if zone_det is not None else []
        labels = None
        draw = detections
        if estimator is not None:
            persons = [d for d in detections if d.class_name == "person"]
            draw_p = overlay_tracks.update(persons, n_frames / fps)
            attrs = overlay_tracks.hold_attrs(estimator.estimate(frame, draw_p))
            labels = rules.overlay_labels(draw_p, attrs)
            draw = draw_p
            action, ev_cls, ev_conf = rules.decide_event(draw_p, attrs)
            event_log.apply(action, n_frames / fps, ev_cls, ev_conf, frame)
            engine_dets = [d for d in detections if d.conf >= conf]
            people_e = [d for d in engine_dets if d.class_name == "person"]
            pickup_flags = pickup_est.estimate(frame, people_e) if pickup_est is not None else None
            engine.step(n_frames / fps, engine_dets, estimator.last_faces, frame,
                        zone_dets=zone_dets, pickup_flags=pickup_flags)
        if sr_in is not None:
            cv2.imwrite(
                os.path.join(sr_in, f"{n_frames:06d}.jpg"),
                frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            overlay_meta.append((
                [Detection(d.x1, d.y1, d.x2, d.y2, d.conf, d.class_id, d.class_names)
                 for d in draw],
                None if labels is None else list(labels),
            ))
            view = None
            annotated = None
        else:
            view = frame if s == 1 else cv2.resize(
                frame, (sw, sh), interpolation=cv2.INTER_CUBIC)
            annotated = annotate_frame(view, scale_detections(draw, s), labels=labels, lang=lang)
        elapsed_ms = (time.time() - t0) * 1000
        count = len(draw)
        if count > max_count:
            max_count = count
        seen_classes.update(d.class_name for d in draw)
        n_frames += 1
        if writer is not None and annotated is not None:
            writer.write(annotated)
        if show and annotated is not None:
            cv2.imshow("CCTV", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        print(f"\rframe {n_frames} | {_class_banner(draw, lang)} | {elapsed_ms:.1f} ms   ", end="")

    cap.release()
    sr_mode = "cubic"
    if sr_in is not None:
        sr_out = tempfile.mkdtemp(prefix="cctv_sr_out_")
        ok_sr = upscale_dir(sr_in, sr_out)
        sr_mode = "ncnn-x4plus" if ok_sr else "cubic-fallback"
        print(f"\nsr={sr_mode}", flush=True)
        names = sorted(f for f in os.listdir(sr_in) if f.endswith(".jpg"))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        n_pick_sr = 0
        n_exit_sr = 0
        for i, (draw, _labels) in enumerate(overlay_meta):
            src_jpg = os.path.join(sr_in, names[i])
            orig = cv2.imread(src_jpg)
            if orig is None:
                raise RuntimeError(t("need_sr_frame", lang, path=src_jpg))
            view = None
            if ok_sr:
                stem = os.path.splitext(names[i])[0]
                for cand in (
                    os.path.join(sr_out, names[i]),
                    os.path.join(sr_out, stem + ".jpg"),
                    os.path.join(sr_out, stem + ".png"),
                ):
                    if os.path.isfile(cand):
                        view = cv2.imread(cand)
                        if view is not None:
                            break
            if view is None:
                view = orig if s == 1 else cv2.resize(
                    orig, (sw, sh), interpolation=cv2.INTER_CUBIC)
            vh, vw = view.shape[:2]
            sx, sy = vw / float(w), vh / float(h)
            draw_v, labels, n_pick, n_exit = _overlay_on_sr_view(
                view, draw, sx, sy, estimator, rules, pickup_est, zone_det, lang)
            n_pick_sr += n_pick
            n_exit_sr += n_exit
            annotated = annotate_frame(view, draw_v, labels=labels, lang=lang)
            print(f"\rsr-infer {i + 1}/{len(overlay_meta)} pick={n_pick_sr} exit={n_exit_sr}   ",
                  end="", flush=True)
            if writer is None:
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
                if not writer.isOpened():
                    raise RuntimeError(t("need_write", lang, path=out_path))
            writer.write(annotated)
        shutil.rmtree(sr_in, ignore_errors=True)
        shutil.rmtree(sr_out, ignore_errors=True)
    if event_log is not None:
        event_log.finish()
        event_log.write()
    if engine is not None:
        engine.write(_pattern_json_path(out_path))
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    extra = f" sr={sr_mode}" if sr else ""
    if sr_in is not None:
        extra += f" sr_pick={n_pick_sr} sr_exit={n_exit_sr}"
    print(f"\nframes={n_frames} max_count={max_count} classes={sorted(seen_classes)} "
          f"out={out_path}{extra}")
    return {"frames": n_frames, "max_count": max_count, "classes": sorted(seen_classes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tests/input/mix.mp4",
                        help="video path, or 0 = webcam")
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--target-classes", type=int, nargs="*", default=None,
                        help="COCO class ids; omit = all classes")
    parser.add_argument("--out", default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-attributes", action="store_true",
                        help="disable gender, age, behavior on person boxes")
    parser.add_argument("--sr", action="store_true",
                        help="Real-ESRGAN ncnn x4plus on the view frame; detect stays on original")
    parser.add_argument("--lang", type=parse_lang, default=DEFAULT_LANG,
                        help="overlay language: en|eng (default) or vie|vi")
    args = parser.parse_args()
    args.out = _output_mp4_path(args.source, args.out or None)
    _app()
    run_cctv(
        args.source, args.model, args.conf, args.target_classes,
        args.out or None, args.show, args.max_frames,
        attributes=not args.no_attributes,
        sr=args.sr,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
