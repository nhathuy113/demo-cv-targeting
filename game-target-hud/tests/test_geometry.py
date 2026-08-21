import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import box_to_relative_position, relative_position_to_text
from i18n import parse_lang

FRAME_W, FRAME_H = 1920, 1080
CX, CY = FRAME_W / 2, FRAME_H / 2


def test_dead_center_is_front_and_close():
    pos = box_to_relative_position(CX - 400, CY - 400, CX + 400, CY + 400, FRAME_W, FRAME_H)
    assert pos.direction == "ahead"
    assert pos.distance_label == "near"
    print("OK: dead center ->", relative_position_to_text(pos))
    vi = box_to_relative_position(
        CX - 400, CY - 400, CX + 400, CY + 400, FRAME_W, FRAME_H, lang=parse_lang("vie"))
    assert vi.direction == "phía trước"
    assert vi.distance_label == "gần"


def test_far_right_small_box_is_right_and_far():
    pos = box_to_relative_position(FRAME_W - 50, CY - 10, FRAME_W - 30, CY + 10, FRAME_W, FRAME_H)
    assert pos.direction == "right"
    assert pos.distance_label == "far"
    print("OK: far right small box ->", relative_position_to_text(pos))


def test_bottom_left_is_behind_left():
    pos = box_to_relative_position(50, FRAME_H - 100, 150, FRAME_H - 20, FRAME_W, FRAME_H)
    assert pos.direction in ("behind-left", "left")
    print("OK: bottom left ->", relative_position_to_text(pos))


def test_top_right_is_front_right():
    pos = box_to_relative_position(FRAME_W - 200, 20, FRAME_W - 100, 100, FRAME_W, FRAME_H)
    assert pos.direction in ("ahead-right", "right")
    print("OK: top right ->", relative_position_to_text(pos))


if __name__ == "__main__":
    test_dead_center_is_front_and_close()
    test_far_right_small_box_is_right_and_far()
    test_bottom_left_is_behind_left()
    test_top_right_is_front_right()
    print("\nAll geometry tests passed.")
