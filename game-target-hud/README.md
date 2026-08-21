# game-target-hud

Current HUD: **`src/cctv.py`** (CCTV overlay). Repo setup: `./setup.sh` at the repo root. See the [root README](../README.md).

# Level 1 (detection → live text overlay)

Demo CV: capture a game window, detect objects, overlay relative position text ("Target ahead-right, near") while playing.

## Sandbox checks (real inference, not stubs)

| Part | Test | Result |
|---|---|---|
| `detector.py` | onnxruntime + `yolov8n.onnx` on Ultralytics `bus.jpg` | 4 person + 1 bus, conf 0.44–0.89 |
| `geometry.py` | 4 unit tests (`tests/test_geometry.py`) | pass |
| `overlay.py` | `QT_QPA_PLATFORM=offscreen` smoke | no crash |
| `capture.py` | import + crop logic | **not live-tested** (no display in sandbox) |
| `main.py` | full loop | **run on a real machine** |

CV detect + geometry is verified on real pixels. OS I/O (screen capture, overlay on a running game) follows `mss` / PyQt5 APIs but was not visually confirmed here.

## Install

Prefer `./setup.sh` from the repo root. From this folder:

```bash
pip install -r requirements.txt
python src/main.py --target-classes 0
```

`models/yolov8n.onnx` is already in the repo.

## Limits

- **Fullscreen exclusive**: overlay uses window layering and will not show. Use borderless windowed.
- **Class "person" is a stand-in** until a custom target class is trained.
- **Latency**: `main.py` prints per-frame latency. If it is too high, lower `--fps-limit`, crop capture `region`, or use a smaller model.
