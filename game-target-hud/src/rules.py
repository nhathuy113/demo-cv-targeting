"""Business rules: model output → overlay text + event class.

Rule 1: show behavior only above threshold; else "unclear" (events / full overlay).
Rule 2: HUD default overlay is gender only; age/behavior stay off the boxes.
Rule 3: class normal stays "normal" in English (not "typical").
Rule 4: format only — no auto-alert.
Rule 5: events point at the classifying frame (event_log.py).
"""

from attributes import PersonAttributes
from detector import Detection
from i18n import DEFAULT_LANG, gender_overlay, t

BEHAVIOR_CONF_THRESHOLD = 0.6
UNCLEAR = "unclear"

BEHAVIOR_DISPLAY_NAMES = {
    "normal": "normal",
    "threat": "unusual_motion",
    "fraud": "theft_like_posture",
    "theft_like_posture": "theft_like_posture",
}


def _behavior_overlay_name(behavior: str, lang: str = DEFAULT_LANG) -> str:
    if behavior == "normal":
        return t("behavior_normal", lang)
    if behavior == UNCLEAR:
        return t("unclear", lang)
    return behavior


def hold_track_attrs(prev: PersonAttributes | None, cur: PersonAttributes | None,
                     threshold: float = BEHAVIOR_CONF_THRESHOLD) -> PersonAttributes | None:
    """Keep age/behavior on a track when the face crop drops. Gender is this frame only."""
    if cur is None:
        return prev
    if prev is None:
        return cur
    gender, gconf = cur.gender, cur.gender_conf
    age = cur.age if cur.age else prev.age
    aconf = cur.age_conf if cur.age else prev.age_conf
    if cur.behavior is not None and cur.behavior_conf > threshold:
        behavior, bconf = cur.behavior, cur.behavior_conf
    else:
        behavior, bconf = prev.behavior, prev.behavior_conf
    return PersonAttributes(gender, age, behavior, gconf, aconf, bconf)


def display_behavior(behavior: str | None, conf: float,
                     threshold: float = BEHAVIOR_CONF_THRESHOLD) -> str:
    if behavior is None or conf <= threshold:
        return UNCLEAR
    if behavior not in BEHAVIOR_DISPLAY_NAMES:
        raise ValueError(f"unknown behavior class: {behavior}")
    return BEHAVIOR_DISPLAY_NAMES[behavior]


OVERLAY_FULL = ("gender", "age", "behavior")
OVERLAY_GENDER = ("gender",)


def format_display_label(class_name: str, attrs: PersonAttributes | None,
                         threshold: float = BEHAVIOR_CONF_THRESHOLD,
                         lang: str = DEFAULT_LANG,
                         overlay_fields: tuple[str, ...] = OVERLAY_FULL) -> str:
    if class_name != "person":
        return class_name
    if attrs is None:
        return class_name if overlay_fields == OVERLAY_GENDER else ""
    parts = []
    if "gender" in overlay_fields:
        g = gender_overlay(attrs.gender, lang)
        if g:
            parts.append(g)
        elif overlay_fields == OVERLAY_GENDER:
            return class_name
    if "age" in overlay_fields:
        if attrs.age:
            parts.append(t("age", lang, age=attrs.age))
        else:
            parts.append(t("age_unclear", lang))
    if "behavior" in overlay_fields:
        behavior = display_behavior(attrs.behavior, attrs.behavior_conf, threshold)
        parts.append(t("behavior", lang, name=_behavior_overlay_name(behavior, lang)))
    if not parts:
        return class_name
    if len(parts) == 1:
        return parts[0]
    head = " ".join(parts[:-1])
    return f"{head} | {parts[-1]}"


def frame_event_class(detections: list[Detection], attrs_list: list[PersonAttributes | None],
                      threshold: float = BEHAVIOR_CONF_THRESHOLD) -> tuple[str | None, float]:
    best = None
    best_conf = -1.0
    for det, attrs in zip(detections, attrs_list):
        if det.class_name != "person" or attrs is None:
            continue
        if attrs.behavior_conf > best_conf:
            best_conf = attrs.behavior_conf
            best = attrs
    if best is None:
        return None, 0.0
    return display_behavior(best.behavior, best.behavior_conf, threshold), best.behavior_conf


class BusinessRules:
    def __init__(self, conf_threshold: float = BEHAVIOR_CONF_THRESHOLD,
                 lang: str = DEFAULT_LANG,
                 overlay_fields: tuple[str, ...] = OVERLAY_GENDER):
        self.conf_threshold = conf_threshold
        self.lang = lang
        self.overlay_fields = overlay_fields
        self._prev_class = None

    def overlay_labels(self, detections: list[Detection],
                       attrs_list: list[PersonAttributes | None]) -> list[str]:
        return [
            format_display_label(
                d.class_name, a, self.conf_threshold, self.lang, self.overlay_fields)
            for d, a in zip(detections, attrs_list)
        ]

    def decide_event(self, detections: list[Detection],
                     attrs_list: list[PersonAttributes | None]) -> tuple[str, str | None, float]:
        cls, conf = frame_event_class(detections, attrs_list, self.conf_threshold)
        prev = self._prev_class
        if cls is None:
            self._prev_class = None
            return ("close" if prev is not None else "none"), None, 0.0
        if prev is None:
            self._prev_class = cls
            return "start", cls, conf
        if prev == cls:
            return "continue", cls, conf
        self._prev_class = cls
        return "switch", cls, conf


def confirm_robbery_prep(primitive_hits: set[str]) -> str | None:
    """Need ≥2 independent primitives. Concealment alone never flags."""
    if len(primitive_hits) < 2:
        return None
    return "ROBBERY_PREP_SUSPECTED"


def confirm_shoplifting_suspected(primitive_hits: set[str], visited_cashier: bool) -> str | None:
    """item_pickup + exit; skip if cashier already visited. One primitive never flags."""
    if visited_cashier:
        return None
    if "item_pickup" in primitive_hits and "exit_approach" in primitive_hits:
        return "SHOPLIFTING_HIGHLY_POSSIBLE"
    return None
