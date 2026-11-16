#!/usr/bin/env python3
"""data/reach_map.csv · figs/fig_reach_map.png — 비커를 어디까지 놓을 수 있나.

    python3 scripts/make_reach_map.py

한 비커만 격자로 옮기고(다른 비커는 기준 자리), 확정값 v0.1.6으로 한 판 전체를 계획 · 검사한다(pour_run.build/check +
fr5pour 화면의 S–B 거리 검사). 처음 걸린 이유를 남긴다. 3인칭 보정(fr5auto)이 있으면 비커가 3인칭 화면 안에 다 들어오는지도 본다.
"""
import csv
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fr5pour"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as M  # noqa: E402

PR, PG = M.PR, M.PG
STEP = 40.0
SPAN = (240.0, 160.0)
REASON = [("IK", "닿지 않음 (IK)"), ("J6", "J6 창 밖 (손목 카메라 선)"), ("한계", "관절 한계"), ("바닥", "바닥 한도"),
          ("뜀", "관절 뜀 (IK 가지)"), ("특이점", "손목 특이점"), ("테두리", "테두리 여유"), ("S–B", "S–B 140 mm 미만")]


def check_one(job):
    which, x, y = job
    P = PG.load()
    S_ref, B_ref = PR.fk(P["grasp"])[:3, 3][:2], PR.fk(P["big"])[:3, 3][:2]
    S, B = (np.array([x, y]), B_ref) if which == "S" else (S_ref, np.array([x, y]))
    if np.linalg.norm(S - B) < 140.0:
        return which, x, y, "S–B"
    try:
        _, pl, info, out, a, eff = M.plan(S.tolist(), B.tolist(), strict=False)
    except SystemExit:
        return which, x, y, "IK"
    for key, word in (("한계", "한계 밖"), ("J6", "J6"), ("바닥", "바닥 한도"), ("뜀", "뜀"), ("IK", "IK 오차"), ("특이점", "특이점"),
                      ("테두리", "테두리")):
        if any(word in b for b in out["bad"]):
            return which, x, y, key
    return which, x, y, "ok"


def main():
    P = PG.load()
    ref = {"S": PR.fk(P["grasp"])[:3, 3][:2], "B": PR.fk(P["big"])[:3, 3][:2]}
    jobs = []
    for which in ("S", "B"):
        for dx in np.arange(-SPAN[0], SPAN[0] + 1, STEP):
            for dy in np.arange(-SPAN[1], SPAN[1] + 1, STEP):
                jobs.append((which, round(float(ref[which][0] + dx), 1), round(float(ref[which][1] + dy), 1)))
    with Pool() as pool:
        rows = pool.map(check_one, jobs, chunksize=4)
    view = None
    try:
        sys.path.insert(0, str(Path.home() / "fr5cap" / "fr5auto"))
        import pour_ui as U
        cal = U.Cal()
        view = (cal, U)
    except Exception as e:  # 보정이 없으면 화면 안 검사는 건너뜀
        print("3인칭 보정 없음 — 화면 안 검사 건너뜀:", type(e).__name__)
    out = []
    for which, x, y, why in rows:
        inview = ""
        if view:
            cal, U = view
            r, h = (U.BIG_R, PR.BIG_H) if which == "B" else (U.SMALL_R, PR.SMALL_H)
            pts = np.vstack([cal.proj(U.ring((x, y), r, PR.TABLE_Z + h, 24)), cal.proj(U.ring((x, y), r * 0.85, PR.TABLE_Z, 24))])
            inview = int(bool((pts[:, 0] >= 0).all() and (pts[:, 0] < cal.W).all() and (pts[:, 1] >= 0).all() and (pts[:, 1] < cal.H).all()))
        out.append({"beaker": which, "x_mm": x, "y_mm": y, "plan": why, "in_third_view": inview})
    with open(M.DATA / "reach_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for which in ("S", "B"):
        R = [r for r in out if r["beaker"] == which]
        print(which, {k: sum(r["plan"] == k for r in R) for k in ["ok"] + [k for k, _ in REASON]})


def figure():
    plt, OI = M.plt, M.OI
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    rows = list(csv.DictReader(open(M.DATA / "reach_map.csv", encoding="utf-8")))
    P = PG.load()
    ref = {"S": PR.fk(P["grasp"])[:3, 3][:2], "B": PR.fk(P["big"])[:3, 3][:2]}
    col = {"ok": OI["green"], "IK": OI["vermillion"], "J6": OI["orange"], "한계": OI["purple"], "바닥": OI["sky"],
           "뜀": OI["yellow"], "특이점": OI["blue"], "테두리": M.INK, "S–B": M.MUTE}
    name = dict([("ok", "한 판 계획 통과")] + REASON)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw={"wspace": 0.06})
    used = set()
    for ax, which, other in zip(axs, ("S", "B"), ("B", "S")):
        R = [r for r in rows if r["beaker"] == which]
        for r in R:
            x, y = float(r["x_mm"]), float(r["y_mm"])
            used.add(r["plan"])
            ax.add_patch(Rectangle((x - STEP / 2 + 2, y - STEP / 2 + 2), STEP - 4, STEP - 4, fc=col[r["plan"]],
                                   alpha=0.85 if r["plan"] == "ok" else 0.6, ec="none"))
            if r["in_third_view"] == "0":
                ax.plot(x, y, "x", color="white", ms=7, mew=1.6)
        c = ref[which]
        ax.add_patch(Rectangle((c[0] - 120, c[1] - 80), 240, 160, fc="none", ec=M.INK, lw=1.8))
        ax.add_patch(Rectangle((c[0] - 160, c[1] - 120), 320, 240, fc="none", ec=M.INK, lw=1.0, ls="--"))
        ax.plot(*c, "+", color="white", ms=14, mew=2.5)
        ax.plot(*ref[other], "o", ms=10, mfc="none", mec=M.INK, mew=1.5)
        ax.text(ref[other][0], ref[other][1] + 26, f"{'큰' if other == 'B' else '작은'} 비커(기준 자리 고정)", ha="center",
                va="top", fontsize=8.5, color=M.INK)
        ax.set_title(f"{'작은 비커 S' if which == 'S' else '큰 비커 B'}를 옮길 때 (다른 비커는 기준 자리)")
        ax.set_xlabel("x (mm)")
        ax.grid(False)
    axs[0].set_ylabel("y (mm)")
    allx = [float(r["x_mm"]) for r in rows]
    ally = [float(r["y_mm"]) for r in rows]
    for ax in axs:
        ax.set_xlim(max(allx) + STEP, min(allx) - STEP)
        ax.set_ylim(max(ally) + STEP, min(ally) - STEP)
        ax.set_aspect("equal")
    h = [Line2D([], [], ls="", marker="s", ms=10, color=col[k], label=name[k]) for k in col if k in used]
    h += [Line2D([], [], color=M.INK, lw=1.8, label="무작위 자리 범위 (기준 ± 120 × 80)"),
          Line2D([], [], color=M.INK, lw=1.0, ls="--", label="자동 찾기 범위 (기준 ± 160 × 120)"),
          Line2D([], [], ls="", marker="x", ms=7, mew=1.6, color=M.MUTE, label="3인칭 화면 밖으로 잘림")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.6, bbox_to_anchor=(0.5, -0.08))
    M.heading(fig, "비커를 어디까지 놓을 수 있나 — 40 mm 격자마다 확정값 v0.1.6으로 한 판 전체를 계획 · 검사",
              "칸 색 = 그 자리에 놓았을 때 계획이 통과했는지, 막혔으면 처음 걸린 검사 · 위에서 본 3인칭 카메라 방향 · 흰 + = 기준 자리")
    fig.savefig(M.FIG / "fig_reach_map.png")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figure":
        figure()
    else:
        main()
        figure()
