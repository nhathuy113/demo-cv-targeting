# demo-cv-targeting

Shop CCTV **Vision Narrator**: detect người trên frame gốc, vẽ overlay (box, đếm, nhãn pickup / zone / behavior) lên video phóng to, ghi `events.json`.

Không phải hệ thống tự báo trộm. Clip shop thật (`tests/input/crime_real/**`) **không** nằm trong git — chỉ dùng local để demo/evaluate, không train.

## Setup (một lệnh)

Cần Python **3.10+**.

```bash
./setup.sh
```

Tạo `game-target-hud/.venv`, cài `requirements.txt` (onnxruntime, opencv, numpy, PyQt5, mss). Model ONNX đã có trong repo (`yolov8n`, `pickup`, `zone`, `behavior`).

## Chạy demo

Clip có sẵn trong repo: `game-target-hud/tests/input/mix.mp4`.

```bash
cd game-target-hud
.venv/bin/python src/cctv.py --source tests/input/mix.mp4 --out tests/output/mix.mp4
```

`--sr` (tuỳ chọn, chậm): phóng view bằng Real-ESRGAN ncnn; detect vẫn trên frame gốc. Tool SR không commit — binary tự tải khi chạy `--sr` nếu thiếu.

Webcam:

```bash
cd game-target-hud
.venv/bin/python src/cctv.py --source 0 --show
```

## Train (không bắt buộc để demo)

Expert zone / pickup / behavior train từ `dataset/**`, không từ `tests/input/**`. Script: `game-target-hud/scripts/train_*.py` + `SOURCES.txt` trong từng dataset.

## Giới hạn

- Overlay đọc được trên clip rõ; nhãn behavior/pickup trên CCTV 240p dễ sai.
- `SHOPLIFTING_HIGHLY_POSSIBLE` chỉ khi pickup **và** cửa, không cashier — thường không bật nếu thiếu một trong hai.
- HUD game (`src/main.py`) là prototype cũ; sản phẩm hiện tại là `src/cctv.py`.
