"""Overlay and CLI copy. Default en (aliases: eng). Vietnamese: vie or vi."""

from argparse import ArgumentTypeError

LANG_EN = "en"
LANG_VI = "vi"
DEFAULT_LANG = LANG_EN

_ALIASES = {
    "en": LANG_EN,
    "eng": LANG_EN,
    "vi": LANG_VI,
    "vie": LANG_VI,
}

_GENDER = {
    LANG_EN: {"nam": "male", "nữ": "female", "male": "male", "female": "female"},
    LANG_VI: {"nam": "nam", "nữ": "nữ", "male": "nam", "female": "nữ"},
}

_STRINGS = {
    LANG_EN: {
        "detected_empty": "Detected: 0",
        "detected": "Detected: {n} ({parts})",
        "holding_item": "holding item",
        "age": "age {age}",
        "age_unclear": "age unclear",
        "behavior": "behavior: {name}",
        "behavior_normal": "normal",
        "unclear": "unclear",
        "target": "Target",
        "dir_ahead": "ahead",
        "dir_ahead_right": "ahead-right",
        "dir_right": "right",
        "dir_behind_right": "behind-right",
        "dir_behind": "behind",
        "dir_behind_left": "behind-left",
        "dir_left": "left",
        "dir_ahead_left": "ahead-left",
        "dist_near": "near",
        "dist_mid": "mid",
        "dist_far": "far",
        "rel_text": "{target} {direction}, {distance}",
        "need_video": "cannot open video: {source}",
        "need_write": "cannot write video: {path}",
        "need_sr_frame": "cannot read SR input frame: {path}",
        "need_zone": "BLOCKER: missing zone model {path}",
        "need_pickup": "BLOCKER: missing pickup model {path}",
        "help_source": "video path, or 0 = webcam",
        "help_classes": "COCO class ids; omit = all classes",
        "help_no_attr": "disable gender, age, behavior on person boxes",
        "help_sr": "Real-ESRGAN ncnn x4plus on the view frame; detect stays on original",
        "help_lang": "overlay language: en|eng (default) or vie|vi",
        "help_target": "COCO class id(s) treated as targets. Default 0=person.",
        "help_fps": "cap processing FPS so the game keeps CPU/GPU headroom",
        "hud_start": "Level 1 overlay — target classes: {classes}, screen: {w}x{h}. Ctrl+C to quit.",
    },
    LANG_VI: {
        "detected_empty": "Phát hiện: 0",
        "detected": "Phát hiện: {n} ({parts})",
        "holding_item": "cầm đồ",
        "age": "tuổi {age}",
        "age_unclear": "tuổi chưa rõ",
        "behavior": "hành vi: {name}",
        "behavior_normal": "bình thường",
        "unclear": "chưa rõ",
        "target": "Mục tiêu",
        "dir_ahead": "phía trước",
        "dir_ahead_right": "phía trước-phải",
        "dir_right": "bên phải",
        "dir_behind_right": "phía sau-phải",
        "dir_behind": "phía sau",
        "dir_behind_left": "phía sau-trái",
        "dir_left": "bên trái",
        "dir_ahead_left": "phía trước-trái",
        "dist_near": "gần",
        "dist_mid": "trung bình",
        "dist_far": "xa",
        "rel_text": "{target} {direction}, {distance}",
        "need_video": "không mở được video: {source}",
        "need_write": "không ghi được video: {path}",
        "need_sr_frame": "không đọc được frame SR input: {path}",
        "need_zone": "BLOCKER: thiếu zone model {path}",
        "need_pickup": "BLOCKER: thiếu pickup model {path}",
        "help_source": "đường dẫn video, hoặc 0 = webcam",
        "help_classes": "lọc class id COCO; bỏ trống = mọi class",
        "help_no_attr": "tắt nam/nữ, tuổi, hành vi trên box person",
        "help_sr": "Real-ESRGAN ncnn x4plus trên frame xem; detect vẫn gốc",
        "help_lang": "ngôn ngữ overlay: en|eng (mặc định) hoặc vie|vi",
        "help_target": "COCO class id coi là mục tiêu. Mặc định 0=person.",
        "help_fps": "Giới hạn FPS xử lý để không ăn hết CPU/GPU của game.",
        "hud_start": "Chạy Level 1 overlay — target classes: {classes}, screen: {w}x{h}. Ctrl+C để thoát.",
    },
}

_DIR_KEYS = [
    (0, "dir_ahead"),
    (45, "dir_ahead_right"),
    (90, "dir_right"),
    (135, "dir_behind_right"),
    (180, "dir_behind"),
    (-135, "dir_behind_left"),
    (-90, "dir_left"),
    (-45, "dir_ahead_left"),
]


def parse_lang(value: str) -> str:
    key = (value or DEFAULT_LANG).strip().lower()
    if key not in _ALIASES:
        raise ArgumentTypeError(
            f"unsupported lang={value!r}; use en|eng (default) or vie|vi"
        )
    return _ALIASES[key]


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    table = _STRINGS[lang if lang in _STRINGS else DEFAULT_LANG]
    return table[key].format(**kwargs)


def gender_overlay(raw: str | None, lang: str = DEFAULT_LANG) -> str | None:
    if not raw:
        return None
    return _GENDER.get(lang, _GENDER[DEFAULT_LANG]).get(raw, raw)


def direction_label(angle_deg: float, lang: str = DEFAULT_LANG) -> str:
    best_key, best_diff = "dir_ahead", 999.0
    for ref_angle, key in _DIR_KEYS:
        diff = abs(((angle_deg - ref_angle) + 180) % 360 - 180)
        if diff < best_diff:
            best_diff, best_key = diff, key
    return t(best_key, lang)


def distance_label(score: float, lang: str = DEFAULT_LANG) -> str:
    if score > 0.5:
        return t("dist_near", lang)
    if score > 0.15:
        return t("dist_mid", lang)
    return t("dist_far", lang)
