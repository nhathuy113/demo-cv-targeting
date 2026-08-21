"""
Trial module: behavior primitives + pattern B/C.

Không MediaPipe (hand-to-region / head-scan = TODO).
Không StreamSync/POS → không flag UNCONFIRMED_TRANSFER.
Object: COCO backpack, handbag, suitcase.
Robbery-prep: không bao giờ flag từ 1 primitive (rules.confirm_robbery_prep).
"""

from dataclasses import dataclass, field
import json
import os

import cv2

from detector import Detection
from rules import confirm_robbery_prep, confirm_shoplifting_suspected

ASSET_CLASSES = frozenset({"backpack", "handbag", "suitcase"})


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center_dist_norm(a: Detection, b: Detection) -> float:
    ac, bc = a.center, b.center
    dist = ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
    h = max((a.y2 - a.y1 + b.y2 - b.y1) / 2.0, 1.0)
    return dist / h


def _pred_box(tr) -> Detection:
    pred = getattr(tr, "pred", None)
    return pred if pred is not None else tr.det


def _best_track(det: Detection, tracks: list["Track"]) -> tuple["Track | None", float]:
    best = None
    best_s = 0.0
    for tr in tracks:
        score = _iou(tr.det, det)
        if score > best_s:
            best, best_s = tr, score
    return best, best_s


@dataclass
class Track:
    tid: int
    det: Detection
    misses: int = 0
    centers: list[tuple[float, float, float]] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)


class IoUTracker:
    """IoU match. Không Re-ID (TODO)."""

    def __init__(self, iou_min: float = 0.3, max_miss: int = 8, dist_frac: float | None = None):
        self.iou_min = iou_min
        self.max_miss = max_miss
        self.dist_frac = dist_frac
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection], ts: float) -> list[Track]:
        matched = _associate(self.tracks, detections, self.iou_min, self.dist_frac)
        used_t = {ti for ti, _ in matched}
        used_d = {di for _, di in matched}
        for ti, di in matched:
            tr = self.tracks[ti]
            det = detections[di]
            tr.det = det
            tr.misses = 0
            cx, cy = det.center
            tr.centers.append((cx, cy, ts))
            tr.heights.append(det.y2 - det.y1)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_miss]
        for di, det in enumerate(detections):
            if di in used_d:
                continue
            cx, cy = det.center
            self.tracks.append(Track(
                tid=self._next_id, det=det, misses=0,
                centers=[(cx, cy, ts)], heights=[det.y2 - det.y1],
            ))
            self._next_id += 1
        return list(self.tracks)


def _associate(tracks: list, dets: list[Detection], iou_min: float,
               dist_frac: float | None) -> list[tuple[int, int]]:
    """Greedy IoU rồi dist. Box = pred (ByteTrack) hoặc det. Trả list (ti, di)."""
    used_t = set()
    used_d = set()
    matched = []
    pairs = []
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(dets):
            pairs.append((_iou(_pred_box(tr), det), ti, di))
    pairs.sort(reverse=True)
    for score, ti, di in pairs:
        if score < iou_min or ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((ti, di))
    if dist_frac is None:
        return matched
    dist_pairs = []
    for ti, tr in enumerate(tracks):
        if ti in used_t:
            continue
        for di, det in enumerate(dets):
            if di in used_d:
                continue
            dist_pairs.append((_center_dist_norm(_pred_box(tr), det), ti, di))
    dist_pairs.sort()
    for dist, ti, di in dist_pairs:
        if dist > dist_frac or ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((ti, di))
    return matched


@dataclass
class ByteTrack:
    tid: int
    det: Detection
    cx: float
    cy: float
    w: float
    h: float
    last_ts: float
    vx: float = 0.0
    vy: float = 0.0
    lost: int = 0
    pred: Detection | None = None


class ByteTracker:
    """BYTE: match conf cao, rồi conf thấp với track chưa match. Không tạo track từ box thấp."""

    def __init__(self, high: float = 0.35, iou_min: float = 0.2, max_lost: int = 30,
                 dist_frac: float | None = 2.5):
        self.high = high
        self.iou_min = iou_min
        self.max_lost = max_lost
        self.dist_frac = dist_frac
        self._next_id = 1
        self.tracks: list[ByteTrack] = []

    def _predict(self, ts: float) -> None:
        for tr in self.tracks:
            dt = ts - tr.last_ts
            if dt <= 0:
                dt = 1.0 / 30.0
            cx = tr.cx + tr.vx * dt
            cy = tr.cy + tr.vy * dt
            tr.pred = Detection(
                cx - tr.w * 0.5, cy - tr.h * 0.5,
                cx + tr.w * 0.5, cy + tr.h * 0.5,
                tr.det.conf, tr.det.class_id, tr.det.class_names,
            )

    def _assign(self, tr: ByteTrack, det: Detection, ts: float) -> None:
        dt = ts - tr.last_ts
        if dt <= 0:
            dt = 1.0 / 30.0
        ncx, ncy = det.center
        tr.vx = 0.8 * ((ncx - tr.cx) / dt) + 0.2 * tr.vx
        tr.vy = 0.8 * ((ncy - tr.cy) / dt) + 0.2 * tr.vy
        tr.cx, tr.cy = ncx, ncy
        tr.w = det.x2 - det.x1
        tr.h = det.y2 - det.y1
        tr.det = det
        tr.lost = 0
        tr.last_ts = ts

    def update(self, detections: list[Detection], ts: float) -> list[Detection]:
        high = [d for d in detections if d.conf >= self.high]
        low = [d for d in detections if d.conf < self.high]
        self._predict(ts)
        high_pairs = _associate(self.tracks, high, self.iou_min, self.dist_frac)
        hit = set()
        used_high = set()
        for ti, di in high_pairs:
            self._assign(self.tracks[ti], high[di], ts)
            hit.add(ti)
            used_high.add(di)
        remain = [self.tracks[i] for i in range(len(self.tracks)) if i not in hit]
        remain_idx = [i for i in range(len(self.tracks)) if i not in hit]
        for ri, di in _associate(remain, low, self.iou_min, self.dist_frac):
            self._assign(self.tracks[remain_idx[ri]], low[di], ts)
            hit.add(remain_idx[ri])
        for i, tr in enumerate(self.tracks):
            if i not in hit:
                tr.lost += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        for di, det in enumerate(high):
            if di in used_high:
                continue
            cx, cy = det.center
            self.tracks.append(ByteTrack(
                tid=self._next_id, det=det,
                cx=cx, cy=cy, w=det.x2 - det.x1, h=det.y2 - det.y1,
                last_ts=ts,
            ))
            self._next_id += 1
        return [t.det for t in self.tracks if t.lost == 0]


def _head_box(det: Detection) -> tuple[float, float, float, float]:
    return det.x1, det.y1, det.x2, det.y1 + 0.5 * (det.y2 - det.y1)


def concealment(person: Detection, faces) -> bool:
    """Person có, không có face trong nửa trên bbox. Không dùng độc lập."""
    hx1, hy1, hx2, hy2 = _head_box(person)
    for face in faces:
        x, y, w, h = face[:4]
        cx, cy = x + w / 2.0, y + h / 2.0
        if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
            return False
    return True


def dwell(track: Track, ts: float, dwell_seconds: float, max_move_frac: float) -> bool:
    window = [(x, y, t, h) for (x, y, t), h in zip(track.centers, track.heights)
              if ts - t <= dwell_seconds]
    if len(window) < 2:
        return False
    t0 = min(p[2] for p in window)
    if ts - t0 + 1e-9 < dwell_seconds:
        return False
    mx = sum(p[0] for p in window) / len(window)
    my = sum(p[1] for p in window) / len(window)
    dist = max(((p[0] - mx) ** 2 + (p[1] - my) ** 2) ** 0.5 for p in window)
    h = sum(p[3] for p in window) / len(window)
    if h <= 0:
        return False
    return dist <= max_move_frac * h


def assign_owner(asset: Detection, persons: list[Track]) -> int | None:
    """Chủ tạm = person overlap bbox. Không overlap → không gán."""
    best_id = None
    best = 0.0
    for tr in persons:
        if tr.det.class_name != "person":
            continue
        score = _iou(asset, tr.det)
        if score > best:
            best = score
            best_id = tr.tid
    if best <= 0:
        return None
    return best_id


def closer_to_doors(track: Track, doors: list[Detection]) -> bool:
    if len(track.centers) < 2 or not doors:
        return False
    x0, y0, _ = track.centers[-2]
    x1, y1, _ = track.centers[-1]

    def _min_d2(x, y):
        return min((x - d.center[0]) ** 2 + (y - d.center[1]) ** 2 for d in doors)

    return _min_d2(x1, y1) < _min_d2(x0, y0)


class PrimitiveEngine:
    def __init__(self, dwell_seconds: float = 2.0, max_move_frac: float = 0.3,
                 thumb_dir: str | None = None):
        self.dwell_seconds = dwell_seconds
        self.max_move_frac = max_move_frac
        self.thumb_dir = thumb_dir
        self.persons = IoUTracker(max_miss=90)
        self.assets = IoUTracker()
        self.owners: dict[int, int | None] = {}
        self.events: list[dict] = []
        self._thumb_i = 0
        self._prep_on = False
        self._shop_on = False
        self._holding: set[int] = set()
        self._visited_cashier: set[int] = set()
        self._door: Detection | None = None
        self._hold_anchor: Detection | None = None

    def _sync_holding(self, people: list[Detection], flags: list[bool],
                      ptr: list[Track]) -> None:
        for det, flag in zip(people, flags):
            if not flag:
                continue
            tr, score = _best_track(det, ptr)
            if tr is not None and score > 0:
                self._holding.add(tr.tid)
                self._hold_anchor = tr.det
        live = {t.tid for t in ptr}
        self._holding = {tid for tid in self._holding if tid in live}
        if any(tr.tid in self._holding for tr in ptr):
            for tr in ptr:
                if tr.tid in self._holding:
                    self._hold_anchor = tr.det
                    break
            return
        if self._hold_anchor is None:
            return
        tr, score = _best_track(self._hold_anchor, ptr)
        if tr is not None and score >= self.persons.iou_min:
            self._holding.add(tr.tid)
            self._hold_anchor = tr.det

    def remember_exit(self, zone_dets: list[Detection] | None) -> list[Detection]:
        doors = [d for d in (zone_dets or []) if d.class_name == "exit"]
        if doors:
            self._door = max(doors, key=lambda d: d.conf)
        return [self._door] if self._door is not None else []

    def _thumb(self, frame_bgr) -> str | None:
        if self.thumb_dir is None or frame_bgr is None:
            return None
        os.makedirs(self.thumb_dir, exist_ok=True)
        path = os.path.join(self.thumb_dir, f"p{self._thumb_i:04d}.jpg")
        cv2.imwrite(path, frame_bgr)
        self._thumb_i += 1
        return path

    def step(self, ts: float, detections: list[Detection], faces, frame_bgr=None,
             zone_dets: list[Detection] | None = None,
             pickup_flags: list[bool] | None = None) -> list[dict]:
        new_events: list[dict] = []
        people = [d for d in detections if d.class_name == "person"]
        bags = [d for d in detections if d.class_name in ASSET_CLASSES]
        ptr = self.persons.update(people, ts)
        atr = self.assets.update(bags, ts)
        flags = pickup_flags if pickup_flags is not None else [False] * len(people)
        self._sync_holding(people, flags, ptr)
        live = {t.tid for t in ptr}
        self._visited_cashier = {tid for tid in self._visited_cashier if tid in live}

        hits: set[str] = set()
        if any(concealment(tr.det, faces) for tr in ptr):
            hits.add("concealment")
        if any(dwell(tr, ts, self.dwell_seconds, self.max_move_frac) for tr in ptr):
            hits.add("dwell")

        flag = confirm_robbery_prep(hits)
        if flag and not self._prep_on:
            rec = {
                "class": flag,
                "primitives": sorted(hits),
                "timestamp": ts,
                "evidence_frame": self._thumb(frame_bgr),
            }
            self.events.append(rec)
            new_events.append(rec)
            self._prep_on = True
        if not flag:
            self._prep_on = False

        zones = zone_dets or []
        cashiers = [d for d in zones if d.class_name == "cashier"]
        doors = self.remember_exit(zones)
        for tr in ptr:
            if any(_iou(tr.det, c) > 0 for c in cashiers):
                self._visited_cashier.add(tr.tid)
        shop = False
        shop_hits: set[str] = set()
        for tr in ptr:
            th: set[str] = set()
            if tr.tid in self._holding:
                th.add("item_pickup")
                if closer_to_doors(tr, doors):
                    th.add("exit_approach")
            sflag = confirm_shoplifting_suspected(th, tr.tid in self._visited_cashier)
            if sflag:
                shop = True
                shop_hits = th
                break
        if shop and not self._shop_on:
            rec = {
                "class": "SHOPLIFTING_HIGHLY_POSSIBLE",
                "primitives": sorted(shop_hits),
                "timestamp": ts,
                "evidence_frame": self._thumb(frame_bgr),
            }
            self.events.append(rec)
            new_events.append(rec)
        self._shop_on = shop

        current_owners: dict[int, int | None] = {}
        for tr in atr:
            owner = assign_owner(tr.det, ptr)
            current_owners[tr.tid] = owner
            prev = self.owners.get(tr.tid)
            if owner is not None and prev is not None and owner != prev:
                rec = {
                    "class": "possession_transfer",
                    "object_id": tr.tid,
                    "from": prev,
                    "to": owner,
                    "timestamp": ts,
                    "evidence_frame": self._thumb(frame_bgr),
                }
                self.events.append(rec)
                new_events.append(rec)
        self.owners = current_owners
        return new_events

    def write(self, json_path: str):
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
