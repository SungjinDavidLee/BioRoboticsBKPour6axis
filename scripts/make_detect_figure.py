#!/usr/bin/env python3
"""figs/fig_detect_method.png — 비커 자동 찾기 단계를 3인칭 사진 1장으로 보인다.

    python3 scripts/make_detect_figure.py <3인칭 사진.png|jpg>

fr5pour/detect_probe.py 의 Finder 를 그대로 쓴다. 3인칭 보정(fr5auto rack_calib.json)과 fr5auto 패키지가 있는
Thor(또는 같은 파일을 둔 컴퓨터)에서 돌아간다. 사진은 fr5rec 판의 첫 3인칭 프레임(rgb/00000.png)을 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fr5pour"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_probe as DP  # noqa: E402
from make_figures import BG, INK, INK2, OI, heading, plt  # noqa: E402

U, PR = DP.U, DP.PR


def main():
    img = Image.open(sys.argv[1]).convert("RGB")
    cal = U.Cal()
    poses = U.PG.load()
    S_ref, B_ref = PR.fk(poses["grasp"])[:3, 3][:2], PR.fk(poses["big"])[:3, 3][:2]
    span = np.array(U.SPAN) + 40.0
    boxes = {"S": (S_ref - span, S_ref + span), "B": (B_ref - span, B_ref + span)}
    F = DP.Finder(cal, img, boxes)
    hsv = np.asarray(img.convert("HSV"), float)
    hue = hsv[..., 0] * 360.0 / 255.0
    reg = Image.new("1", img.size, 0)
    for (xa, ya), (xb, yb) in boxes.values():
        c = [[x, y, z] for x in (xa, xb) for y in (ya, yb) for z in (PR.TABLE_Z, PR.TABLE_Z + PR.BIG_H)]
        ImageDraw.Draw(reg).polygon([tuple(q) for q in DP.hull(cal.proj(c))], fill=1)
    reg = np.asarray(reg, bool)
    raw = (hue > DP.HUE[0]) & (hue < DP.HUE[1]) & (hsv[..., 2] > DP.V_MIN) & reg
    res = F.run(boxes)
    base = np.asarray(img).astype(float)

    def tint(mask, col):
        o = base.copy()
        o[mask] = 0.45 * o[mask] + 0.55 * np.array(col)
        o[~reg] *= 0.55
        return o.astype(np.uint8)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 6.4), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    axs = axs.ravel()
    axs[0].imshow(img)
    axs[0].set_title("① 3인칭 사진 (848×480)")
    axs[1].imshow(tint(raw, (0, 158, 115)))
    axs[1].set_title(f"② 색 기준: 색상 {DP.HUE[0]:.0f}–{DP.HUE[1]:.0f}° · 밝기 > {DP.V_MIN}")
    axs[2].imshow(tint(F.mask, (0, 158, 115)))
    axs[2].set_title("③ 테두리로 둘러싼 구멍 채우기")
    axs[3].imshow(img)
    for kind, col in (("B", OI["blue"]), ("S", OI["orange"])):
        iou, xy = res[kind]
        poly = np.array(F.sil(*xy, kind) + [F.sil(*xy, kind)[0]])
        axs[3].plot(poly[:, 0], poly[:, 1], color=col, lw=2.2)
        u, v = cal.proj([[xy[0], xy[1], PR.TABLE_Z]])[0]
        axs[3].plot(u, v, "+", color=col, ms=12, mew=2.2)
        dx, ha = (-55, "right") if kind == "B" else (40, "left")
        axs[3].text(u + dx, v - 45, f"{'큰 비커 B' if kind == 'B' else '작은 비커 S'} ({xy[0]:.0f}, {xy[1]:.0f}) mm\nIoU {iou:.2f}",
                    color="white", fontsize=8.5, fontweight="bold", va="center", ha=ha,
                    bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none", alpha=0.9))
    axs[3].set_title("④ 비커 모양을 책상 위에서 옮겨 가며 겹침 최대")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(60, 700)
        ax.set_ylim(470, 200)
    heading(fig, "비커 자동 찾기 — 반투명 비커를 색으로 가리고, 크기를 아는 모양을 보정 카메라로 맞춘다",
            "어둡게 한 곳은 찾는 범위 밖. ④의 선은 찾은 자리에 놓은 비커 모양(위 테두리 원 + 바닥 원의 볼록 껍질)을 3인칭 보정 카메라로 그린 것 · + 는 바닥 중심")
    out = ROOT / "figs" / "fig_detect_method.png"
    fig.savefig(out)
    print(out, {k: (round(v[0], 2), [round(x, 1) for x in v[1]]) for k, v in res.items()})


if __name__ == "__main__":
    main()
