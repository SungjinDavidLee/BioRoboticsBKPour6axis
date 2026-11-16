#!/usr/bin/env python3
"""figs/fig_*.png 를 fr5pour 코드와 data/*.csv · json 에서 다시 만든다.

    python3 scripts/make_figures.py

- 궤적 · 붓기 기하 · 기울이기 비교 그림은 fr5pour/pour_run.py 의 계획기를 그대로 돌려 만든다(로봇 없이).
  자리 · 설정은 data/settings_v016.json 의 확정값과 한 판 예(S, B)를 쓴다.
- 비커 찾기 검증 그림은 data/detect_validation.csv.
- 비커 찾기 원리 그림(fig_detect_method.png)은 3인칭 보정 파일이 있어야 해서 scripts/make_detect_figure.py 로 따로 만든다.
색은 Okabe & Ito (2008). 위에서 본 그림은 3인칭 카메라에서 본 방향(로봇 x+ = 화면 왼쪽, y+ = 화면 아래)이다.
"""
import csv
import json
import math
import sys
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "fr5pour"))
import pose_go as PG  # noqa: E402
import pour_run as PR  # noqa: E402

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
BG, INK, INK2, MUTE, GRID = "#FAFAF8", "#222222", "#555555", "#8A8A8A", "#E4E4E0"
plt.rcParams.update({
    "font.family": ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": MUTE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": BG,
    "figure.facecolor": BG, "axes.facecolor": BG, "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK,
})

# 단계 묶음: (이름, 색, 계획 꼬리표)
GROUPS = [("잡기", OI["blue"], ("start", "open", "to_small", "descend", "close")),
          ("들기 · 옮기기", OI["sky"], ("lift", "transport", "pre_pour")),
          ("기울이기 · 직선", OI["orange"], ("tilt",)),
          ("기울이기 · 관절 + 붓기 1 s", OI["vermillion"], ("tilt_end", "pour_hold")),
          ("되돌리기", OI["purple"], ("untilt", "up", "return")),
          ("내려놓기", OI["green"], ("place", "release", "retreat")),
          ("초기 자세로", INK2, ("home",))]
TAG2G = {t: i for i, (_, _, ts) in enumerate(GROUPS) for t in ts}
EXAMPLE = {"S": [-9.9, 621.2], "B": [214.8, 533.6]}          # 11-16 17:25 판(자동 찾기) 자리


def heading(fig, title, sub):
    fig.text(0.01, 1.0, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.01, 0.985, sub, fontsize=9.5, color=INK2, ha="left", va="top")


def camera_view(ax):
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")


def ring(c, r, z, n=72):
    a = np.linspace(0, 2 * math.pi, n + 1)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n + 1, z)])


def settings():
    return json.load(open(DATA / "settings_v016.json", encoding="utf-8"))


def plan(S=None, B=None, strict=True):
    """확정값으로 한 판 계획(초기 자세에서 시작). 돌려줌: Plan, info, out, a, 실제 쓰는 붓는 자리 옮김."""
    st = settings()["screen"]
    P = PG.load()
    a = PR.make_parser().parse_args([])
    Tp = PR.fk(P["pour"])
    eff = np.array(st["pour_shift_mm"], float) - Tp[:3, 2][:2] * st["grasp_dz_mm"]   # pour_ui.pshift 와 같은 식
    a.S, a.B = list(S or EXAMPLE["S"]), list(B or EXAMPLE["B"])
    a.pour_shift, a.grasp_shift, a.grasp_dz = eff.tolist(), list(st["grasp_shift_mm"]), st["grasp_dz_mm"]
    a.close, a.open = st["gripper_close"], st["gripper_open"]
    a.phases = ["grasp", "pour", "place", "home"]
    pl, info = PR.build(a, P, P["home"], st["gripper_open"] / 1000.0)
    out, bad = PR.check(pl, info, np.array(P["home"], float))
    out["bad"] = bad
    if strict:
        assert not bad, bad
    return P, pl, info, out, a, eff


def beaker_axis(T, hang):
    """쥔 작은 비커: 바닥 중심 · 입구 중심 (베이스 mm). 공구 z 가 비커 바닥 쪽."""
    z = T[:3, 2]
    return T[:3, 3] + z * hang, T[:3, 3] - z * (PR.SMALL_H - hang)


def write_plan_csv(pl):
    with open(DATA / "plan_example.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "tag", "j1", "j2", "j3", "j4", "j5", "j6", "x_mm", "y_mm", "z_mm", "grip_0_1000"])
        for k, (q, g, tag) in enumerate(zip(pl.q, pl.g, pl.tag)):
            if k % 5 and k != len(pl.q) - 1:
                continue
            p = PR.fk(q)[:3, 3]
            w.writerow([f"{k / PR.HZ:.2f}", tag] + [f"{x:.3f}" for x in q] + [f"{x:.1f}" for x in p] + [round(g * 1000)])


# ── 1. 한 판 흐름 ────────────────────────────────────────────────
def fig_workflow():
    lanes = [("사람", OI["orange"]), ("fr5pour\n(Thor 화면)", OI["blue"]), ("fr5rec\n(카메라 · 녹화)", OI["green"]),
             ("제어판 · FR5", INK2)]
    steps = [  # (칸, 줄, 제목, 설명)
        (0, 0, "비커 놓기", "작은 비커(공 3개)와\n큰 비커를 표시\n범위 안에 대충"),
        (1, 1, "자동 찾기", "3인칭 사진 1장\n→ 두 비커 중심\nS · B (약 2 s)"),
        (2, 1, "계획 · 검사", "IK · 관절 한계\nJ6 창 · 바닥\n특이점 · 테두리 여유"),
        (3, 0, "추종 시작", "제어판 /leader\n[리더 추종 시작]"),
        (4, 2, "녹화 시작", "판 꼬리표:\nS · B · 설정\n자리 출처"),
        (5, 3, "실행 103 s", "관절 명령 50 Hz\n잡기 → 붓기 →\n내려놓기 → 초기 자세"),
        (6, 2, "녹화 멈춤", "3인칭 · 손목 사진\n+ 로봇 상태\n(세션 기록 주기)"),
        (7, 0, "판정 · 저장", "1 성공 / 2 실패\n→ Enter 저장"),
        (8, 0, "되돌리기", "공 3개를\n작은 비커에,\n비커를 범위 안으로"),
    ]
    W = 0.9
    fig, ax = plt.subplots(figsize=(18, 6.4))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01)
    ax.set_xlim(-1.35, 9.0)
    ax.set_ylim(-0.95, 3.55)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (name, col) in enumerate(lanes):
        ax.add_patch(FancyBboxPatch((-1.3, i - 0.47), 10.25, 0.94, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=col, alpha=0.06, ec="none"))
        ax.text(-1.22, i, name, va="center", ha="left", fontsize=10.5, fontweight="bold", color=col if col != INK2 else INK)
    pos = {}
    for x, lane, title, body in steps:
        col = lanes[lane][1]
        ax.add_patch(FancyBboxPatch((x - W / 2, lane - 0.42), W, 0.84, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="white", ec=col, lw=1.6))
        ax.text(x, lane - 0.27, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x, lane + 0.1, body, ha="center", va="center", fontsize=7.8, color=INK2, linespacing=1.3)
        ax.text(x - W / 2 + 0.05, lane - 0.35, str(x + 1), ha="left", va="center", fontsize=7.5, color=col, fontweight="bold")
        pos[x] = (x, lane)
    for k in range(len(steps) - 1):
        (x0, y0), (x1, y1) = pos[k], pos[k + 1]
        ax.add_patch(FancyArrowPatch((x0 + W / 2, y0), (x1 - W / 2, y1), arrowstyle="-|>", mutation_scale=11, color=MUTE, lw=1.1))
    ax.add_patch(FancyArrowPatch((8, -0.42), (0, -0.42), arrowstyle="-|>", mutation_scale=11, color=OI["orange"], lw=1.1,
                                 ls=(0, (4, 3)), connectionstyle="arc3,rad=0.06"))
    ax.text(4, -0.88, "다음 판", ha="center", va="bottom", fontsize=8.5, color=OI["orange"])
    ax.text(5.5, 3.12, "실행 중 0.2 s마다 관절을 읽어\n8° 넘게 0.6 s 뒤처지면 멈춤", fontsize=7.8, color=INK2, ha="left", va="center")
    heading(fig, "한 판의 흐름 — 사람이 하는 일과 모듈이 하는 일",
            "사람: 놓기 · 추종 시작 · 판정 · 되돌리기 | 모듈: 비커 찾기 · 계획 · 검사 · 실행 · 녹화. 번호는 순서")
    fig.savefig(FIG / "fig_workflow.png")
    plt.close(fig)


# ── 2. 구성 ────────────────────────────────────────────────────
def fig_system():
    fig, ax = plt.subplots(figsize=(13.5, 6.3))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=col, lw=1.6))
        ax.text(x + 0.15, y + h - 0.22, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x + 0.15, y + h - 0.6, body, ha="left", va="top", fontsize=8, color=INK2, linespacing=1.35)

    def arrow(a, b, txt, col=MUTE, rad=0.0, tx=None, ha="center"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=col, lw=1.3,
                                     connectionstyle=f"arc3,rad={rad}"))
        if txt:
            m = tx or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12)
            ax.text(*m, txt, ha=ha, va="bottom", fontsize=7.8, color=INK2, linespacing=1.25)

    ax.add_patch(FancyBboxPatch((3.05, 0.15), 7.45, 5.95, boxstyle="round,pad=0,rounding_size=0.1", fc=OI["sky"], alpha=0.06, ec="none"))
    ax.text(3.2, 5.88, "Thor (FR5 제어 PC)", fontsize=9.5, color=OI["blue"], fontweight="bold", va="top")
    box(7.3, 4.35, 3.0, 1.35, "고정 자료", "poses.json — home · grasp · big · pour\nrack_calib.json — 3인칭 보정\n(fr5auto, 재투영 0.99 px)", MUTE)
    box(0.2, 2.4, 2.4, 1.5, "카메라 2대", "Intel D405 · 848×480 · 30 fps\n3인칭(작업대 맞은편)\n손목(그리퍼 위)", INK2)
    box(3.3, 2.2, 2.8, 1.9, "fr5rec  :8780", "카메라를 혼자 잡고 녹화\n판 = meta.json + PNG\n최근 프레임 /api/frame\n판 시작 · 멈춤 /api/episode", OI["green"])
    box(7.3, 2.2, 3.0, 1.9, "fr5pour  :8792", "화면 · 윤곽 · 버튼\n자동 찾기 (detect_probe)\n계획 · 검사 (pour_run)\n관절 명령 송신 · 판 기록", OI["blue"])
    box(11.2, 2.4, 2.1, 1.5, "워크스테이션", "브라우저로 같은 화면\nSSH 터널\n:8792 · :8780", MUTE)
    box(7.3, 0.35, 3.0, 1.35, "제어판 fr5d  :8088", "로봇 연결 · 안전 필터 · ServoJ\n리더 입력 UDP 8097\n0.3 s 끊기면 그 자리 정지", INK2)
    box(11.2, 0.35, 2.1, 1.35, "FAIRINO FR5", "6축 협동 로봇\n그리퍼 DH AG-160-95\n0 닫힘 – 1000 열림", INK2)
    arrow((2.6, 3.15), (3.3, 3.15), "USB 3")
    arrow((6.1, 3.55), (7.3, 3.55), "3인칭 사진")
    arrow((7.3, 2.75), (6.1, 2.75), "")
    ax.text(6.7, 2.65, "녹화 시작 · 멈춤\n(판 꼬리표)", ha="center", va="top", fontsize=7.8, color=INK2)
    arrow((8.8, 4.35), (8.8, 4.1), "")
    arrow((9.2, 2.2), (9.2, 1.7), "", OI["blue"])
    ax.text(9.3, 1.95, "{q[6], grip} 50 Hz", ha="left", va="center", fontsize=7.8, color=OI["blue"])
    arrow((8.0, 1.7), (8.0, 2.2), "")
    ax.text(7.9, 1.95, "상태 · 추종 여부 (GET)", ha="right", va="center", fontsize=7.8, color=INK2)
    arrow((7.3, 1.1), (5.4, 2.2), "", MUTE, rad=-0.15)
    ax.text(5.35, 1.3, "로봇 상태 (GET,\n프레임마다 기록)", ha="left", va="center", fontsize=7.8, color=INK2)
    arrow((10.3, 1.0), (11.2, 1.0), "ServoJ")
    arrow((11.2, 3.15), (10.3, 3.15), "SSH 터널")
    heading(fig, "구성 — 모듈과 주고받는 값",
            "fr5pour 는 제어판에 리더와 같은 형식(관절 6개 + 그리퍼)으로 명령을 보낸다. 녹화는 fr5rec 가 하고, fr5pour 는 시작 · 멈춤과 꼬리표만 보낸다")
    fig.savefig(FIG / "fig_system.png")
    plt.close(fig)


# ── 3. 계획된 한 판 ─────────────────────────────────────────────
def fig_trajectory():
    P, pl, info, out, a, eff = plan()
    write_plan_csv(pl)
    Q = np.array(pl.q)
    t = np.arange(len(Q)) / PR.HZ
    tcp = np.array([PR.fk(q)[:3, 3] for q in Q])
    gi = np.array([TAG2G[x] for x in pl.tag])
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.55], height_ratios=[1.5, 0.6, 1.1], hspace=0.28, wspace=0.12)
    a0 = fig.add_subplot(gs[:, 0])
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    S, B = np.array(a.S), np.array(a.B)
    for c, r, col, lab in ((S, 27.5, OI["orange"], "작은 비커 S"), (B, 46.0, OI["blue"], "큰 비커 B")):
        a0.add_patch(Circle(c, r, fc=col, alpha=0.12, ec=col, lw=1.4))
        a0.text(c[0], c[1] + r + 14, lab, ha="center", va="top", fontsize=9, color=col, fontweight="bold")
    for i, (name, col, _) in enumerate(GROUPS):
        m = gi == i
        idx = np.where(m)[0]
        for _, run in groupby(enumerate(idx), lambda kv: kv[1] - kv[0]):
            r = [k for _, k in run]
            back = name in ("되돌리기", "초기 자세로")                  # 위에서 보면 가는 길과 겹친다 → 점선
            a0.plot(tcp[r, 0], tcp[r, 1], color=col, lw=1.8 if back else 3.0, ls=(0, (4, 2.5)) if back else "-",
                    solid_capstyle="round", zorder=3 if back else 2)
            a1.plot(t[r], tcp[r, 2], color=col, lw=2.2, solid_capstyle="round")
    home = PR.fk(P["home"])[:3, 3]
    a0.plot(*home[:2], "s", color=INK2, ms=7)
    a0.text(home[0], home[1] - 12, "초기 자세", ha="center", va="bottom", fontsize=8.5, color=INK2)
    gpt = S + np.array(a.grasp_shift)
    a0.plot(*gpt, "x", color=OI["purple"], ms=9, mew=2.2)
    a0.annotate("잡는 자리 = S + 잡는 자리 옮김\n(1.3, 16) mm", gpt, xytext=(-50, 560), fontsize=8, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    pp = info["B"] + info["pour_off"]
    a0.plot(*pp, "D", color=OI["vermillion"], ms=6)
    a0.annotate("붓기 손끝", pp, xytext=(pp[0] - 60, pp[1] - 70), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    a0.set_xlim(330, -110)
    a0.set_ylim(720, 400)
    a0.set_aspect("equal")
    a0.set_xlabel("x (mm, 로봇 베이스)")
    a0.set_ylabel("y (mm)")
    a0.set_title("위에서 본 손끝 경로 (3인칭 카메라 방향)")
    a1.axhline(PR.TABLE_Z + PR.BIG_H, color=OI["blue"], lw=0.9, ls="--")
    a1.text(26, PR.TABLE_Z + PR.BIG_H + 3, "큰 비커 테두리 141.8", fontsize=8, color=OI["blue"], va="bottom")
    a1.axhline(info["z_g"], color=OI["orange"], lw=0.9, ls="--")
    a1.text(26, info["z_g"] - 3, f"잡는 높이 {info['z_g']:.1f}", fontsize=8, color=OI["orange"], va="top")
    a1.set_ylabel("손끝 높이 z (mm)")
    a1.set_ylim(60, 310)
    a1.set_title("손끝 높이")
    a2.plot(t, np.array(pl.g) * 1000, color=INK2, lw=1.6, drawstyle="steps-post")
    a2.set_ylabel("그리퍼")
    a2.set_yticks([510, 900])
    a2.set_ylim(420, 990)
    a2.set_title("그리퍼 명령 (0 닫힘 – 1000 열림)")
    for j, (col, ls) in zip((3, 4, 5), ((OI["sky"], "-"), (OI["vermillion"], "-"), (INK, "-"))):
        a3.plot(t, Q[:, j], color=col, lw=1.8, ls=ls, label=f"J{j + 1}")
    a3.axhspan(-5, 5, color=OI["vermillion"], alpha=0.08, lw=0)
    a3.axhline(12, color=OI["vermillion"], lw=0.8, ls=":")
    a3.text(t[-1], 14, "J5 = 12° 에서 직선 → 관절 공간", ha="right", va="bottom", fontsize=8, color=OI["vermillion"])
    a3.text(t[-1], -7, "J5 ±5° (손목 특이점 근처)", ha="right", va="top", fontsize=8, color=OI["vermillion"])
    a3.set_ylabel("관절 (°)")
    a3.set_xlabel("시간 (s)")
    a3.legend(loc="lower left", ncol=3, fontsize=8.5)
    a3.set_title("손목 관절 J4 · J5 · J6")
    for ax in (a1, a2):
        plt.setp(ax.get_xticklabels(), visible=False)
    h = [Line2D([], [], color=c, lw=3, label=n) for n, c, _ in GROUPS]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8.8, bbox_to_anchor=(0.5, -0.04))
    heading(fig, f"한 판 계획 — 확정값 v0.1.6, 자동 찾기 자리 S ({a.S[0]}, {a.S[1]}) · B ({a.B[0]}, {a.B[1]})",
            f"{out['sec']:.0f} s · 손끝 최저 {out['z_min']:.0f} mm · J6 {out['j6'][0]:.0f}~{out['j6'][1]:.0f}° (초기 자세 ±70° 안) · "
            f"직선 구간 |J5| 최소 {out['j5_cart']:.1f}° · 기울이기 직전 작은 비커 바닥이 큰 비커 테두리보다 {out['clear']:.0f} mm 위")
    fig.savefig(FIG / "fig_trajectory.png")
    plt.close(fig)
    return pl, info, a


# ── 4. 붓기 자세의 기하 ─────────────────────────────────────────
def fig_pour_geometry():
    P, pl, info, out, a, eff = plan()
    B = np.array(a.B)
    T = PR.fk(info["q_pour"])
    hang = info["hang"]
    bot, mou = beaker_axis(T, hang)
    ax_dir = (mou - bot) / np.linalg.norm(mou - bot)
    u = ax_dir[:2] / np.linalg.norm(ax_dir[:2])
    drop = mou[:2] - B
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.18})
    a0.add_patch(Circle(B, 46, fc=OI["blue"], alpha=0.08, ec=OI["blue"], lw=1.6))
    a0.plot(*B, "+", color=OI["blue"], ms=11, mew=1.8)
    rings = np.vstack([ring(bot, 27.5, 0)[:, :2], ring(mou, 27.5, 0)[:, :2]])
    # 기울인 원의 위 투영: 비커 축에 수직인 두 방향으로 원을 그린다
    e1 = np.cross(ax_dir, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax_dir, e1)
    ang = np.linspace(0, 2 * math.pi, 73)
    circ = lambda c: c + 27.5 * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    pts = np.vstack([circ(bot)[:, :2], circ(mou)[:, :2]])
    from matplotlib.path import Path as MPath
    hull = _hull(pts)
    a0.add_patch(Polygon(hull, closed=True, fc=OI["orange"], alpha=0.18, ec=OI["orange"], lw=1.4))
    a0.plot(*circ(mou)[:, :2].T, color=OI["orange"], lw=1.6)
    cb = circ(bot)[:, :2].mean(axis=0)
    a0.text(cb[0] + 8, cb[1] + 22, "쥔 작은 비커\n(입구 쪽이 큰 비커 위)", fontsize=8.5, color=OI["orange"], ha="center", va="top")
    a0.plot(*mou[:2], "o", color=OI["vermillion"], ms=7)
    a0.plot(*T[:2, 3], "D", color=INK2, ms=5)
    a0.annotate(f"입구 중심 · 큰 비커 중심에서 {np.linalg.norm(drop):.1f} mm", mou[:2], xytext=(mou[0] + 70, mou[1] - 75),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    a0.annotate("손끝(TCP)", T[:2, 3], xytext=(T[0, 3] - 45, T[1, 3] + 70), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    a0.text(B[0] - 50, B[1] + 52, "큰 비커 테두리 Ø92", fontsize=8.5, color=OI["blue"], ha="left", va="top")
    a0.set_xlim(B[0] + 110, B[0] - 130)
    a0.set_ylim(B[1] + 120, B[1] - 110)
    a0.set_aspect("equal")
    a0.set_xlabel("x (mm)")
    a0.set_ylabel("y (mm)")
    a0.set_title("위에서 본 붓기 자세 (3인칭 카메라 방향)")
    # 옆에서: 비커 축의 수평 방향 u 를 가로축으로
    s = lambda p: (p[0] - B[0]) * u[0] + (p[1] - B[1]) * u[1]
    zt, zr = PR.TABLE_Z, PR.TABLE_Z + PR.BIG_H
    a1.add_patch(Polygon([[-39, zt], [39, zt], [46, zr], [-46, zr]], closed=True, fc=OI["blue"], alpha=0.08, ec=OI["blue"], lw=1.6))
    a1.axhline(zt, color=MUTE, lw=1.0)
    nrm = np.array([-ax_dir[2], 0.0]) if False else None
    side = lambda p: np.array([s(p), p[2]])
    b2, m2 = side(bot), side(mou)
    d2 = (m2 - b2) / np.linalg.norm(m2 - b2)
    n2 = np.array([-d2[1], d2[0]])
    poly = [b2 + n2 * 27.5, m2 + n2 * 27.5, m2 - n2 * 27.5, b2 - n2 * 27.5]
    a1.add_patch(Polygon(poly, closed=True, fc=OI["orange"], alpha=0.18, ec=OI["orange"], lw=1.6))
    tcp2 = side(T[:3, 3])
    a1.plot(*tcp2, "D", color=INK2, ms=5)
    a1.plot(*m2, "o", color=OI["vermillion"], ms=7)
    lip = m2 - n2 * 27.5 if (m2 - n2 * 27.5)[1] < (m2 + n2 * 27.5)[1] else m2 + n2 * 27.5
    a1.annotate("", (lip[0], zr), xytext=lip, arrowprops=dict(arrowstyle="-|>", color=OI["vermillion"], lw=1.2, ls="--"))
    a1.text(lip[0] - 4, (lip[1] + zr) / 2, f"공이 떨어지는 높이\n약 {lip[1] - zr:.0f} mm", fontsize=8.5, color=OI["vermillion"],
            ha="right", va="center")
    a1.annotate(f"비커 축 {math.degrees(math.acos(ax_dir[2])):.0f}° 기울임\n(수평보다 {math.degrees(math.acos(ax_dir[2])) - 90:.0f}° 아래)",
                (b2 + m2) / 2, xytext=((b2[0] + m2[0]) / 2 + 30, 285), fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    a1.annotate(f"손끝은 입구에서 {PR.SMALL_H - hang:.1f} mm 안쪽\n(잡는 높이 −5 mm)", tcp2, xytext=(tcp2[0] + 25, 175), fontsize=8.5,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    a1.text(0, zr - 6, "큰 비커 (높이 105)", ha="center", va="top", fontsize=8.5, color=OI["blue"])
    a1.set_xlim(-110, 130)
    a1.set_ylim(20, 300)
    a1.set_aspect("equal")
    a1.set_xlabel("큰 비커 중심에서 비커 축의 수평 방향 거리 (mm)")
    a1.set_ylabel("z (mm)")
    a1.set_title("옆에서 본 붓기 자세 (비커 축 방향 단면)")
    heading(fig, "붓기 자세 — 쥔 작은 비커의 입구가 큰 비커 위 어디에 오나",
            f"확정값(붓는 자리 옮김 24, −20 · 잡는 높이 −5 mm)에서 입구 중심은 큰 비커 중심에서 {np.linalg.norm(drop):.1f} mm, "
            f"테두리 반지름 46 mm 안. 잡는 높이를 바꾸면 손끝을 비커 축 방향으로 같이 옮겨 입구 자리를 유지한다")
    fig.savefig(FIG / "fig_pour_geometry.png")
    plt.close(fig)


def _hull(P):
    P = sorted(map(tuple, P))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


# ── 5. 기울이기: 세 가지 방법 ────────────────────────────────────
def fig_tilt_compare():
    P, pl, info, out, a, eff = plan()
    hang = info["hang"]
    tags = np.array(pl.tag)
    k0 = int(np.where(tags == "tilt")[0][0]) - 1
    k1 = int(np.where(tags == "pour_hold")[0][0])
    ours = np.array(pl.q[k0:k1])
    q_pre, q_pour = np.array(pl.q[k0]), np.array(info["q_pour"])
    n = len(ours)
    s = PR.mj(n)
    joint = q_pre + np.outer(s, q_pour - q_pre)                      # 관절 공간 한 번
    T0, T1 = PR.fk(q_pre), PR.fk(q_pour)
    w = PR.rotvec(T0[:3, :3].T @ T1[:3, :3])
    cart, q = [], q_pre.copy()                                        # 직선(자세 · 위치 곧게) 끝까지
    for si in s:
        q = PR.ik(q, T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * si, T0[:3, :3] @ PR.expm(w * si))[0]
        cart.append(q.copy())
    cart = np.array(cart)
    B = np.array(a.B)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.2})
    tz = PR.TABLE_Z
    a0.add_patch(Polygon([[-39, tz], [39, tz], [46, tz + PR.BIG_H], [-46, tz + PR.BIG_H]], closed=True, fc=OI["blue"], alpha=0.08,
                         ec=OI["blue"], lw=1.4))
    Tp = PR.fk(q_pour)
    _, mou = beaker_axis(Tp, hang)
    u = (mou[:2] - B) / np.linalg.norm(mou[:2] - B)
    for Q, col, lab, ls in ((joint, OI["sky"], "관절 공간 한 번", "-"), (ours, OI["vermillion"], "이 모듈: 직선 → J5 12° 부터 관절", "-")):
        bots, mous = [], []
        for qq in Q[::3]:
            b_, m_ = beaker_axis(PR.fk(qq), hang)
            bots.append([(b_[0] - B[0]) * u[0] + (b_[1] - B[1]) * u[1], b_[2]])
            mous.append([(m_[0] - B[0]) * u[0] + (m_[1] - B[1]) * u[1], m_[2]])
        bots, mous = np.array(bots), np.array(mous)
        a0.plot(mous[:, 0], mous[:, 1], color=col, lw=2.2, ls=ls, label=lab + " · 입구")
        a0.plot(bots[:, 0], bots[:, 1], color=col, lw=1.4, ls=(0, (3, 2)), label=lab + " · 바닥")
        lowb = bots[np.argmin(bots[:, 1])]
        a0.plot(*lowb, "o", color=col, ms=5)
    a0.axhline(tz + PR.BIG_H, color=OI["blue"], lw=0.8, ls=":")
    a0.text(60, tz + PR.BIG_H + 3, "큰 비커 테두리", fontsize=8, color=OI["blue"], va="bottom")
    a0.set_xlim(-175, 120)
    a0.set_ylim(60, 300)
    a0.set_aspect("equal")
    a0.set_xlabel("큰 비커 중심에서 수평 거리 (mm, 붓는 방향 +)")
    a0.set_ylabel("z (mm)")
    a0.legend(loc="lower right", fontsize=7.8)
    a0.set_title("세운 자세 → 붓기 자세: 쥔 비커의 입구 · 바닥 경로")
    pr = np.linspace(0, 100, n)
    for Q, col, lab in ((cart, OI["sky"], "직선 끝까지"), (ours, OI["vermillion"], "이 모듈")):
        a1.plot(pr, Q[:, 3], color=col, lw=1.8, label=f"J4 · {lab}")
        a1.plot(pr, Q[:, 5], color=col, lw=1.8, ls="--", label=f"J6 · {lab}")
    jump = float(np.max(np.abs(np.diff(cart[:, [3, 5]], axis=0))))
    a1.set_xlabel("기울이기 진행 (%)")
    a1.set_ylabel("관절 (°)")
    a1.legend(loc="lower left", fontsize=8, ncol=2)
    a1.set_title(f"자세를 직선으로 끝까지 바꾸면 J5 = 0 근처에서 J4 · J6 이 튐 (0.02 s에 최대 {jump:.0f}°)")
    heading(fig, "기울이기 방법 — 왜 직선과 관절 공간을 이어 붙였나",
            "관절 공간 한 번이면 비커가 큰 비커 밖으로 크게 휘두르고, 자세를 끝까지 직선으로 바꾸면 손목 특이점(J5 = 0)에서 J4 · J6 이 튄다. "
            "이 모듈은 J5 가 12° 될 때까지 직선, 그 뒤 붓기 자세까지 관절 공간으로 간다")
    fig.savefig(FIG / "fig_tilt_compare.png")
    plt.close(fig)


# ── 6. 비커 찾기 검증 ───────────────────────────────────────────
def fig_detect_validation():
    R = list(csv.DictReader(open(DATA / "detect_validation.csv", encoding="utf-8")))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})
    a0.add_patch(Circle((0, 0), 8, fc=OI["green"], alpha=0.07, ec=OI["green"], lw=1.2, ls="--"))
    a0.text(0, 8.3, "기준 8 mm", ha="center", va="top", fontsize=8, color=OI["green"])
    for r in R:
        dx, dy = float(r["found_x_mm"]) - float(r["overlay_x_mm"]), float(r["found_y_mm"]) - float(r["overlay_y_mm"])
        col, mk = (OI["blue"], "o") if r["beaker"] == "B" else (OI["orange"], "^")
        cnt = r["counted"] == "1"
        a0.plot(dx, dy, mk, ms=9, color=col if cnt else BG, mec=col, mew=1.5)
        a0.text(dx, dy - 0.9, r["placement"], ha="center", va="bottom", fontsize=7.5, color=INK2)
    camera_view(a0)
    a0.set_xlim(10, -10)
    a0.set_ylim(10, -10)
    a0.axhline(0, color=MUTE, lw=0.6)
    a0.axvline(0, color=MUTE, lw=0.6)
    a0.set_xlabel("찾은 자리 − 사람이 맞춘 윤곽 자리, x (mm)")
    a0.set_ylabel("y (mm)")
    a0.set_title("자리 차이 (3인칭 카메라 방향)")
    a0.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, color=OI["blue"], label="큰 비커 B"),
                       Line2D([], [], ls="", marker="^", ms=8, color=OI["orange"], label="작은 비커 S"),
                       Line2D([], [], ls="", marker="o", ms=8, color=BG, mec=MUTE, label="T = 조정용 사진 (세지 않음)")],
              loc="upper right", fontsize=8)
    C = [r for r in R if r["counted"] == "1"]
    xs = np.arange(len(C) // 2)
    for j, (bk, col) in enumerate((("B", OI["blue"]), ("S", OI["orange"]))):
        v = [float(r["iou"]) for r in C if r["beaker"] == bk]
        a1.bar(xs + (j - 0.5) * 0.38, v, width=0.35, color=col, label=f"{'큰' if bk == 'B' else '작은'} 비커")
        for x, vv in zip(xs + (j - 0.5) * 0.38, v):
            a1.text(x, vv + 0.01, f"{vv:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    a1.axhline(0.75, color=OI["vermillion"], lw=1.0, ls="--", label="찾음 기준 IoU 0.75")
    a1.set_xticks(xs, [f"자리 {i + 1}" for i in xs])
    a1.set_ylim(0.5, 1.09)
    a1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a1.set_ylabel("모양 겹침 IoU")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper center", fontsize=8.5, ncol=3)
    a1.set_title("모양 겹침")
    d = [math.hypot(float(r["found_x_mm"]) - float(r["overlay_x_mm"]), float(r["found_y_mm"]) - float(r["overlay_y_mm"])) for r in C]
    heading(fig, "비커 자동 찾기 — 무작위 자리 5곳, 사람이 윤곽에 맞춰 놓은 자리와 비교",
            f"10개 모두 기준 통과(IoU ≥ 0.75 · 차이 ≤ 8 mm). 차이 평균 {np.mean(d):.1f} mm · 최대 {max(d):.1f} mm. "
            "차이에는 사람이 놓을 때의 오차도 들어 있다")
    fig.savefig(FIG / "fig_detect_validation.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_workflow()
    fig_system()
    fig_trajectory()
    fig_pour_geometry()
    fig_tilt_compare()
    fig_detect_validation()
    print("figs:", ", ".join(sorted(p.name for p in FIG.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
