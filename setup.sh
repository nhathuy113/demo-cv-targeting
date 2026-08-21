#!/usr/bin/env bash
# One-command setup: Python venv + infer dependencies (no train, no test-clip download).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/game-target-hud"
REQ="$APP/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Need python3 (>= 3.10) on PATH." >&2
  exit 1
fi

PY_OK="$(python3 -c 'import sys; print(int(sys.version_info[:2] >= (3, 10)))')"
if [[ "$PY_OK" != "1" ]]; then
  echo "Need Python 3.10+. Found: $(python3 --version)" >&2
  exit 1
fi

python3 -m venv "$APP/.venv"
"$APP/.venv/bin/python" -m pip install -U pip
"$APP/.venv/bin/pip" install -r "$REQ"
mkdir -p "$APP/tests/output"

echo
echo "Setup done. Demo:"
echo "  cd game-target-hud"
echo "  .venv/bin/python src/cctv.py --source tests/input/mix.mp4 --out tests/output/mix.mp4"
echo
echo "Output: tests/output/mix.mp4 + events.json"
