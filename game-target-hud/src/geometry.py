"""Map a pixel bounding box to a relative direction/distance label."""

from dataclasses import dataclass
import math

from i18n import DEFAULT_LANG, direction_label, distance_label, t


@dataclass
class RelativePosition:
    direction: str
    distance_label: str
    angle_deg: float
    distance_score: float


def box_to_relative_position(x1: float, y1: float, x2: float, y2: float,
                              frame_w: int, frame_h: int,
                              lang: str = DEFAULT_LANG) -> RelativePosition:
    """Screen center is the camera. 0 deg is up (12 o'clock). Distance is a box-area proxy."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    screen_cx, screen_cy = frame_w / 2, frame_h / 2

    dx = cx - screen_cx
    dy = screen_cy - cy
    angle_deg = math.degrees(math.atan2(dx, dy))

    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = frame_w * frame_h
    distance_score = min(1.0, box_area / frame_area * 12.0)

    return RelativePosition(
        direction=direction_label(angle_deg, lang),
        distance_label=distance_label(distance_score, lang),
        angle_deg=angle_deg,
        distance_score=distance_score,
    )


def relative_position_to_text(pos: RelativePosition, target_label: str | None = None,
                             lang: str = DEFAULT_LANG) -> str:
    target = target_label if target_label is not None else t("target", lang)
    return t("rel_text", lang, target=target, direction=pos.direction, distance=pos.distance_label)
