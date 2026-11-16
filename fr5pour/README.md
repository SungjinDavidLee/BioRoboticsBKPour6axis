# fr5pour — 비커 붓기 자동 수집 모듈 (v0.1.7)

Thor `~/fr5cap/pour` 에 설치되는 코드다. 설명 · 그림은 상위 폴더의 README.md 에 있다.

| 파일 | 역할 |
|---|---|
| `pour_ui.py` | 버튼 화면(127.0.0.1:8792) · 윤곽 · 자동 찾기 · 한 판 실행 · fr5rec 녹화 연동 |
| `pour_run.py` | 한 판 계획(FK · IK · 최소 저크 · 기울이기) · 검사 · 명령줄 실행 |
| `pose_go.py` | 저장 자세 사이 관절 이동 · 자세 저장(`save 이름`) · 제어판 송신기 |
| `detect_probe.py` | 3인칭 사진에서 두 비커 자리 찾기(화면의 [자동 찾기]가 쓰는 Finder) · 단독 시험 |
| `pour_report.py` | fr5rec 데이터셋의 fr5pour 판 집계(성공률 · Wilson 95%) · 사진 묶기 |
| `pour_export.py` | 최근 fr5pour 판의 meta.json · 3인칭 · 손목 사진 묶기(영상 만들기용) |
| `poses.json` | 저장 자세 grasp · big (home · pour 는 `pose_go.py` 기본값) |
| `fr5pour.sh` · `fr5pour_ui.sh` · `setup_fr5pour.sh` | 실행 · 바탕화면 아이콘 · 설치 |

```bash
bash setup_fr5pour.sh                 # ~/fr5cap/pour 로 복사, 명령 fr5pour, 아이콘, --selftest (로봇 안 움직임)
fr5pour                               # 화면 켜기 → http://127.0.0.1:8792/
python detect_probe.py                # 지금 3인칭 사진으로 비커 찾기만 (로봇 안 움직임)
python pour_report.py <데이터셋>       # 집계 (읽기만)
python pour_run.py --dry              # 기준 자리 한 판 계획만 (로봇 안 움직임)
```

필요한 것: numpy · Pillow 가 되는 파이썬(`fr5pour.sh` 의 `FR5POUR_PYTHON`), `~/fr5cap/fr5auto`(3인칭 보정 `work/rack_calib.json` 과 `camera` · `rackcal`), fr5rec 서버(:8780), 제어판(:8088 · UDP 8097).
`poses.json` 은 설치 스크립트가 덮어쓰지 않는다. 새 장비면 이 파일을 복사하거나 `pose_go.py save grasp|big` 으로 다시 저장한다.
