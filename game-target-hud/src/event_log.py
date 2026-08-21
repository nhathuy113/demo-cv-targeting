"""
Event log / highlight list.

Mỗi record: {ts_start, ts_end, class, avg_confidence, frame_thumbnail_path}
Nhiều frame liên tiếp cùng class → 1 event.
Output mặc định: tests/output/events.json
"""

import json
import os

import cv2


class EventLog:
    def __init__(self, json_path: str):
        self.json_path = json_path
        parent = os.path.dirname(json_path) or "."
        self.thumb_dir = os.path.join(parent, "event_thumbs")
        self.records = []
        self._cur = None
        self._thumb_i = 0

    def apply(self, action: str, ts: float, cls: str | None, conf: float, frame_bgr):
        if action == "none":
            return
        if action == "continue":
            self._cur["confs"].append(conf)
            self._cur["ts_end"] = ts
            return
        if action == "close":
            self._flush()
            return
        if action == "switch":
            self._flush()
            self._start(ts, cls, conf, frame_bgr)
            return
        if action == "start":
            self._start(ts, cls, conf, frame_bgr)
            return
        raise ValueError(f"unknown event action: {action}")

    def _start(self, ts: float, cls: str, conf: float, frame_bgr):
        os.makedirs(self.thumb_dir, exist_ok=True)
        thumb = os.path.join(self.thumb_dir, f"{self._thumb_i:04d}.jpg")
        cv2.imwrite(thumb, frame_bgr)
        self._thumb_i += 1
        self._cur = {
            "class": cls,
            "ts_start": ts,
            "ts_end": ts,
            "confs": [conf],
            "frame_thumbnail_path": thumb,
        }

    def _flush(self):
        if self._cur is None:
            return
        confs = self._cur["confs"]
        self.records.append({
            "ts_start": self._cur["ts_start"],
            "ts_end": self._cur["ts_end"],
            "class": self._cur["class"],
            "avg_confidence": sum(confs) / len(confs),
            "frame_thumbnail_path": self._cur["frame_thumbnail_path"],
        })
        self._cur = None

    def finish(self):
        self._flush()

    def write(self):
        os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
