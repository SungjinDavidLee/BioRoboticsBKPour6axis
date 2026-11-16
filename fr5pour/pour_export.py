#!/usr/bin/env python3
"""최근 fr5pour 판을 GitHub 영상용으로 묶기 (읽기만 · 로봇 안 움직임).

    python pour_export.py              # 저장된 fr5pour 판 중 최근 2판 (모든 데이터셋)
    python pour_export.py 3 <데이터셋>   # 그 데이터셋의 최근 3판

판마다 meta.json 과 3인칭 · 손목 사진(JPEG 품질 85)을 work/pour_export.tar 에 넣는다.
결정 안 한 판(ep_*.part)도 meta.json 이 있으면 들어간다(이름에 .part). 결정 안 한 판 목록은 따로 보인다.
"""
import glob
import io
import json
import os
import sys
import tarfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/fr5cap/datasets")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pat = f"{ROOT}/{sys.argv[2]}/ep_*/meta.json" if len(sys.argv) > 2 else f"{ROOT}/*/ep_*/meta.json"
    eps = []
    for m in glob.glob(pat):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        fx = d.get("fr5rec") or {}
        if (fx.get("tag") or {}).get("by") != "fr5pour":
            continue
        eps.append((str(fx.get("started_at") or ""), os.path.getmtime(m), m, d))
    eps.sort()
    parts = sorted(glob.glob(pat.replace("ep_*/meta.json", "ep_*.part")))
    if parts:
        print("결정 안 한 판:", ", ".join(p.split("/")[-2] + "/" + p.split("/")[-1] for p in parts), "— fr5rec 에서 저장하면 넣을 수 있다")
    if not eps:
        sys.exit("저장된 fr5pour 판이 없다")
    pick = eps[-n:]
    out = os.path.join(HERE, "work", "pour_export.tar")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tarfile.open(out, "w") as tf:
        for _, _, m, d in pick:
            ep_dir = os.path.dirname(m)
            ds, ep = ep_dir.split("/")[-2], ep_dir.split("/")[-1]
            name = f"{ds}__{ep}"
            tf.add(m, arcname=f"{name}/meta.json")
            cnt = {}
            for cam in ("rgb", "wrist"):
                fs = sorted(glob.glob(f"{ep_dir}/{cam}/*.png"))
                cnt[cam] = len(fs)
                for f in fs:
                    b = io.BytesIO()
                    Image.open(f).convert("RGB").save(b, "JPEG", quality=85)
                    ti = tarfile.TarInfo(f"{name}/{cam}/{os.path.basename(f)[:-4]}.jpg")
                    ti.size = b.tell()
                    b.seek(0)
                    tf.addfile(ti, b)
            tag = (d.get("fr5rec") or {}).get("tag") or {}
            print(f"  {ds}/{ep} · 결과 {d.get('outcome')} · 자리 {tag.get('src', '사람')} · v{tag.get('v')} · "
                  f"{d.get('duration_s') or 0:.0f} s · {(d.get('fr5rec') or {}).get('hz_real')} Hz · 3인칭 {cnt['rgb']} · 손목 {cnt['wrist']}장")
    print(f"묶음: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
