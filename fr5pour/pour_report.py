#!/usr/bin/env python3
"""fr5pour 판 집계 + GitHub 자료용 사진 묶기 (읽기만 · 로봇 안 움직임).

    python pour_report.py <데이터셋>          # 예: fr5_pour_balls_20261116

- ~/fr5cap/datasets/<데이터셋>/ep_*/meta.json 중 fr5pour 가 만든 판(fr5rec.tag.by == fr5pour)을 센다.
  저장 판: outcome(success/fail/unsure/없음)별, 자리 출처(자동 찾기/사람)별. 버린 판(_discarded)은 따로 센다.
- 성공률 = 성공 / (성공 + 실패), Wilson 95% 구간. 애매·결과 없음·버린 판은 분모에 넣지 않고 수만 보인다.
- work/pour_report.tar 에 묶음: 집계 json, 판 목록 csv, pour_runs.jsonl, detect_probe 그림,
  마지막 성공 판의 meta.json 과 3인칭·손목 사진 24장씩.
"""
import csv
import glob
import io
import json
import math
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/fr5cap/datasets")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def episodes(ds):
    rows = []
    for kind, pat in (("saved", f"{ROOT}/{ds}/ep_*/meta.json"), ("discarded", f"{ROOT}/{ds}/_discarded/*/meta.json")):
        for m in sorted(glob.glob(pat)):
            try:
                d = json.load(open(m))
            except Exception:
                continue
            fx = d.get("fr5rec") or {}
            tag = fx.get("tag") or {}
            if tag.get("by") != "fr5pour":
                continue
            rows.append({"ep": os.path.basename(os.path.dirname(m)), "kind": kind, "outcome": d.get("outcome") or "",
                         "src": tag.get("src", "사람"), "v": tag.get("v", ""), "S": tag.get("S"), "B": tag.get("B"),
                         "shift": tag.get("shift"), "grasp_shift": tag.get("grasp_shift"), "grasp_dz": tag.get("grasp_dz"),
                         "close": tag.get("close"), "plan_s": tag.get("sec"), "n_frames": d.get("n_frames"),
                         "duration_s": d.get("duration_s"), "hz_real": fx.get("hz_real"), "stop": fx.get("stop_reason"),
                         "started_at": fx.get("started_at"), "note": d.get("note") or "", "_meta": m})
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit("python pour_report.py <데이터셋>   (데이터셋 목록: " + ", ".join(sorted(os.listdir(ROOT))) + ")")
    ds = sys.argv[1]
    if not os.path.isdir(f"{ROOT}/{ds}"):
        sys.exit(f"{ROOT}/{ds} 없음")
    R = episodes(ds)
    saved = [r for r in R if r["kind"] == "saved"]
    summ = {"dataset": ds, "saved": len(saved), "discarded": len(R) - len(saved), "by_src": {}}
    print(f"{ds} · fr5pour 저장 판 {len(saved)} · 버린 판 {len(R) - len(saved)}")
    for src in ["전체"] + sorted({r["src"] for r in saved}):
        S = saved if src == "전체" else [r for r in saved if r["src"] == src]
        k = sum(r["outcome"] == "success" for r in S)
        f = sum(r["outcome"] == "fail" for r in S)
        u = len(S) - k - f
        lo, hi = wilson(k, k + f)
        summ["by_src"][src] = {"success": k, "fail": f, "other": u, "rate": (k / (k + f)) if k + f else None, "wilson95": [lo, hi]}
        rate = f"{k}/{k + f} = {100 * k / (k + f):.0f}% (95% {100 * lo:.0f}–{100 * hi:.0f}%)" if k + f else "—"
        print(f"  {src}: 성공 {k} · 실패 {f} · 애매/결과 없음 {u} → {rate}")
    vs = sorted({r["v"] for r in saved})
    print("  모듈 버전:", ", ".join(vs) or "—")
    out = os.path.join(HERE, "work", "pour_report.tar")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tarfile.open(out, "w") as tf:
        def add_bytes(name, b):
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
        add_bytes("summary.json", json.dumps(summ, ensure_ascii=False, indent=1).encode())
        sio = io.StringIO()
        keys = [k for k in R[0].keys() if not k.startswith("_")] if R else ["ep"]
        w = csv.DictWriter(sio, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in R:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items() if k in keys})
        add_bytes("episodes.csv", sio.getvalue().encode("utf-8-sig"))
        runs = os.path.join(HERE, "work", "pour_runs.jsonl")
        if os.path.exists(runs):
            tf.add(runs, arcname="pour_runs.jsonl")
        for f in sorted(glob.glob(os.path.join(HERE, "work", "detect_probe_*.jpg"))):
            tf.add(f, arcname="probe/" + os.path.basename(f))
        ok = [r for r in saved if r["outcome"] == "success"]
        if ok:
            last = ok[-1]
            ep = os.path.dirname(last["_meta"])
            tf.add(last["_meta"], arcname="ep_meta.json")
            for cam in ("rgb", "wrist"):
                fs = sorted(glob.glob(f"{ep}/{cam}/*.png"))
                for f in fs[::max(1, len(fs) // 24)]:
                    tf.add(f, arcname=f"ep_{cam}/" + os.path.basename(f))
            print("  사진: 마지막 성공 판", last["ep"])
    print(f"묶음: {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
