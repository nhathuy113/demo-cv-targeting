"""
overlay.py — cửa sổ overlay trong suốt, luôn nổi trên cùng (always-on-top),
click-through (không chặn chuột/bàn phím tới game bên dưới). Vẽ box + text
label cho mỗi detection.

LƯU Ý QUAN TRỌNG: mình đã viết đúng theo API chuẩn của PyQt5 cho window flags
(FramelessWindowHint | WindowStaysOnTopHint | Tool, cộng
WA_TranslucentBackground + WA_TransparentForMouseEvents để click-through), và
import/khởi tạo QApplication có kiểm tra chạy ổn trong sandbox này (không có
display thật -> dùng QT_QPA_PLATFORM=offscreen để test). Nhưng mình CHƯA thể
verify bằng mắt là overlay hiển thị đúng trên 1 game thật, vì sandbox build
không có GPU/display để mở game. Cần bạn test lại trên máy thật ở bước review.

Trên Windows, muốn overlay hiện được TRÊN CẢ cửa sổ game fullscreen-exclusive
thường cần game chạy ở chế độ "Borderless Windowed" thay vì Fullscreen thật —
đây là giới hạn của mọi HUD overlay dùng window layering, không riêng gì code
này.
"""

from PyQt5 import QtWidgets, QtCore, QtGui


def class_label_rect(x1, y1, x2, y2, label, metrics, pad=3.0):
    """Trong khung nếu đủ chỗ; không thì full chữ phía trên (hoặc dưới nếu sát mép)."""
    text_w = metrics.horizontalAdvance(label) + 8
    text_h = metrics.height() + 4
    inner_w = max(0.0, x2 - x1 - pad * 2)
    inner_h = max(0.0, y2 - y1 - pad * 2)
    if inner_w >= text_w and inner_h >= text_h:
        return x1 + pad, y1 + pad, text_w, text_h
    ty = y1 - text_h
    if ty < 0:
        ty = y2
    return x1, ty, text_w, text_h


class DetectionOverlay(QtWidgets.QWidget):
    def __init__(self, width: int, height: int):
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # click-through
        self.setGeometry(0, 0, width, height)
        self._items: list[tuple[float, float, float, float, str]] = []  # x1,y1,x2,y2,label

    def update_detections(self, items: list[tuple[float, float, float, float, str]]):
        """items: list (x1, y1, x2, y2, label_text) theo toạ độ pixel màn hình gốc."""
        self._items = items
        self.update()  # trigger repaint

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        box_pen = QtGui.QPen(QtGui.QColor(255, 60, 60, 230), 2)
        text_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 255))
        font = QtGui.QFont("Consolas", 9, QtGui.QFont.Bold)
        painter.setFont(font)

        for (x1, y1, x2, y2, label) in self._items:
            painter.setPen(box_pen)
            painter.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))

            text_bg = QtGui.QColor(0, 0, 0, 160)
            metrics = painter.fontMetrics()
            tx, ty, text_w, text_h = class_label_rect(x1, y1, x2, y2, label, metrics)
            painter.fillRect(QtCore.QRectF(tx, ty, text_w, text_h), text_bg)
            painter.setPen(text_pen)
            painter.drawText(QtCore.QRectF(tx + 4, ty, text_w, text_h),
                              QtCore.Qt.AlignVCenter, label)


def run_smoke_test():
    """Khởi tạo overlay ở chế độ offscreen để verify code không crash — KHÔNG
    phải test hiển thị thật, chỉ verify logic paintEvent/geometry chạy được."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication([])
    overlay = DetectionOverlay(1920, 1080)
    overlay.update_detections([
        (100, 100, 300, 300, "Target ahead, near"),
        (1500, 200, 1600, 260, "Target right, far"),
    ])
    overlay.repaint()
    metrics = overlay.fontMetrics()
    ix, iy, iw, ih = class_label_rect(100, 100, 400, 400, "person", metrics)
    assert iy >= 100, "large box: label stays inside"
    sx, sy, sw, sh = class_label_rect(100, 100, 115, 140, "person", metrics)
    assert sy < 100, "small box: label sits above, not clipped"
    assert sw >= metrics.horizontalAdvance("person") + 8
    print("Overlay smoke test OK — no crash, 2 detections painted (offscreen).")


if __name__ == "__main__":
    run_smoke_test()
