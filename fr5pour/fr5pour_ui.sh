#!/usr/bin/env bash
# 바탕화면 아이콘이 여는 창: 버튼 화면을 켜고 브라우저를 띄운다. 창을 닫으면 화면도 꺼진다(하던 송신은 끊김 → 그 자리 정지).
"$(dirname "$(readlink -f "$0")")/fr5pour.sh" --open
echo
read -r -p "fr5pour 가 끝났다 — Enter 로 창 닫기" _
