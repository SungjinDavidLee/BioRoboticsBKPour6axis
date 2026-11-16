#!/usr/bin/env bash
# fr5pour 설치·업데이트 (Thor)
#   이 폴더에서:  bash setup_fr5pour.sh
#   zip 으로 받았으면: bash <(unzip -p fr5pour_vX.zip fr5pour_vX/setup_fr5pour.sh) fr5pour_vX.zip
# 하는 일: ~/fr5cap/pour 에 코드 복사(이전 파일은 .bak_<시각>/ 로), 명령 fr5pour, 바탕화면·앱 목록 아이콘,
#          점검(--selftest, 로봇 안 움직임). poses.json · work/ 는 건드리지 않는다.
set -euo pipefail
ZIP="${1:-}"
DST="$HOME/fr5cap/pour"
PY="${FR5POUR_PYTHON:-$HOME/miniforge3/envs/openvla_py310/bin/python}"
if [ -n "$ZIP" ]; then
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  unzip -q "$ZIP" -d "$TMP"
  SRC=$(ls -d "$TMP"/fr5pour_v* | head -1)
else
  SRC="$(dirname "$(readlink -f "$0")")"
fi
mkdir -p "$DST"
BAK="$DST/.bak_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BAK"
for f in pose_go.py pour_run.py pour_ui.py detect_probe.py pour_report.py pour_export.py fr5pour.sh fr5pour_ui.sh README.md; do
  if [ -f "$DST/$f" ]; then cp -p "$DST/$f" "$BAK/"; fi
  cp "$SRC/$f" "$DST/$f"
done
chmod +x "$DST/fr5pour.sh" "$DST/fr5pour_ui.sh"
mkdir -p "$HOME/.local/bin"
ln -sf "$DST/fr5pour.sh" "$HOME/.local/bin/fr5pour"
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
cat > "$APPS/fr5pour-ui.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=fr5pour 붓기 화면
Comment=비커 붓기 — 배치 표시·한 판 실행·fr5rec 녹화
Exec=$DST/fr5pour_ui.sh
Terminal=true
Icon=applications-science
Categories=Utility;
DESK
chmod +x "$APPS/fr5pour-ui.desktop"
DESKDIR=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
if [ -d "$DESKDIR" ]; then
  cp "$APPS/fr5pour-ui.desktop" "$DESKDIR/"
  chmod +x "$DESKDIR/fr5pour-ui.desktop"
  gio set "$DESKDIR/fr5pour-ui.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "설치: $DST  (이전 파일: $BAK)"
echo "아이콘: $APPS/fr5pour-ui.desktop  ${DESKDIR:+· $DESKDIR}"
echo "--- 점검(로봇 안 움직임) ---"
cd "$DST" && "$PY" pour_ui.py --selftest
md5sum "$DST/pose_go.py" "$DST/pour_run.py" "$DST/pour_ui.py" "$DST/detect_probe.py" "$DST/pour_report.py" "$DST/pour_export.py"
