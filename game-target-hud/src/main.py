"""
main.py — Level 1 loop: capture -> detect -> geometry -> overlay text.

Run: python src/main.py
Quit: Ctrl+C.

Default class_filter=[0] (person) as a COCO stand-in for a later custom target class.
"""

import sys
import time
import argparse

from PyQt5 import QtCore, QtWidgets

from capture import ScreenCapture
from detector import Detector
from geometry import box_to_relative_position, relative_position_to_text
from i18n import DEFAULT_LANG, parse_lang, t
from overlay import DetectionOverlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0],
                         help="COCO class id(s) treated as targets. Default 0=person.")
    parser.add_argument("--fps-limit", type=float, default=15.0,
                         help="cap processing FPS so the game keeps CPU/GPU headroom")
    parser.add_argument("--lang", type=parse_lang, default=DEFAULT_LANG,
                         help="overlay language: en|eng (default) or vie|vi")
    args = parser.parse_args()
    lang = args.lang

    app = QtWidgets.QApplication(sys.argv)

    cap = ScreenCapture()
    frame_w, frame_h = cap.size

    detector = Detector(args.model, conf_threshold=args.conf, class_filter=args.target_classes)
    overlay = DetectionOverlay(frame_w, frame_h)
    overlay.show()

    frame_interval = 1.0 / args.fps_limit

    def tick():
        t0 = time.time()
        frame = cap.grab_bgr()
        detections = detector.detect(frame)

        items = []
        for d in detections:
            pos = box_to_relative_position(d.x1, d.y1, d.x2, d.y2, frame_w, frame_h, lang=lang)
            label = relative_position_to_text(pos, target_label=d.class_name, lang=lang)
            items.append((d.x1, d.y1, d.x2, d.y2, label))

        overlay.update_detections(items)

        elapsed = time.time() - t0
        print(f"\rframe latency: {elapsed*1000:.1f} ms | detections: {len(items)}   ", end="")

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(int(frame_interval * 1000))

    print(t("hud_start", lang, classes=args.target_classes, w=frame_w, h=frame_h))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
