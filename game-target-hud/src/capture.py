"""
capture.py — screen capture bằng mss (nhanh hơn nhiều so với pyautogui.screenshot
cho use-case real-time). Trả về frame dạng numpy BGR để đưa thẳng vào Detector.

LƯU Ý: module này cần 1 display thật để chạy (không chạy được trong sandbox
headless không có màn hình — mình chỉ verify được import + logic crop vùng chụp,
chưa verify được visual thật vì môi trường build không có screen).
"""

import numpy as np
import mss
import cv2


class ScreenCapture:
    def __init__(self, region: dict | None = None, monitor_index: int = 1):
        """
        region: {"top": int, "left": int, "width": int, "height": int} — nếu None,
        chụp toàn bộ monitor theo monitor_index (1 = monitor chính trong mss).
        Cho phép giới hạn vùng chụp quanh khu vực chơi để giảm chi phí capture
        khi có HUD/overlay khác của game không cần detect.
        """
        self.sct = mss.mss()
        self.monitor = region if region is not None else self.sct.monitors[monitor_index]

    def grab_bgr(self) -> np.ndarray:
        """Chụp 1 frame, trả về numpy array BGR (khớp format OpenCV/Detector)."""
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @property
    def size(self) -> tuple[int, int]:
        return self.monitor["width"], self.monitor["height"]

    def close(self):
        self.sct.close()


if __name__ == "__main__":
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(description="Lưu frame màn hình mỗi N giây (Level 2 data collection).")
    parser.add_argument("--out", default="dataset/raw_frames")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="giây giữa 2 frame (spec: 0.5–1 fps, tránh frame gần giống nhau)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cap = ScreenCapture()
    i = 0
    print(f"Dump frames -> {args.out} mỗi {args.interval}s. Ctrl+C để dừng.")
    while True:
        frame = cap.grab_bgr()
        path = os.path.join(args.out, f"{i:04d}.jpg")
        cv2.imwrite(path, frame)
        print(f"\rsaved {path}", end="")
        i += 1
        time.sleep(args.interval)
