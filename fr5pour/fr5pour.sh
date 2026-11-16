#!/usr/bin/env bash
# fr5pour 버튼 화면 실행 (numpy·Pillow 가 되는 파이썬 — fr5auto 와 같은 것)
PY="${FR5POUR_PYTHON:-$HOME/miniforge3/envs/openvla_py310/bin/python}"
exec "$PY" "$(dirname "$(readlink -f "$0")")/pour_ui.py" "$@"
