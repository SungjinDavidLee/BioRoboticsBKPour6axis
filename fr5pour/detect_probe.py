#!/usr/bin/env python3
"""비커 자동 찾기 시험(로봇 안 움직임 · 읽기만).

fr5rec 3인칭 사진에서 반투명 비커를 색으로 가려내고, 크기를 아는 비커 모양(큰 Ø92·H105, 작은 Ø55·H66.5)을
3인칭 보정 카메라로 그려 겹침이 가장 큰 책상 자리를 찾는다. 찾은 자리와 지금 화면 윤곽 자리(ui_state S·B)를 비교한다.

    python detect_probe.py              # fr5rec 지금 사진
    python detect_probe.py 사진.jpg      # 저장한 사진
결과 그림: work/detect_probe_<시각>.jpg (왼쪽: 찾은 모양 빨강·윤곽 자리 하늘색, 오른쪽: 비커로 본 픽셀 초록)
"""
import io
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import pour_ui as U

PR = U.PR
HUE = (70.0, 240.0)      # 비커로 보는 색상(도). 나무 책상은 40–60° 근처
V_MIN = 150              # 밝기 하한(0–255)
IOU_OK = 0.75            # 이보다 낮으면 '못 찾음'
KIND = {"B": (U.BIG_R, U.BIG_R * 0.85, PR.BIG_H), "S": (U.SMALL_R, U.SMALL_R * 0.85, PR.SMALL_H)}


def hull(P):
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
    return lo[:-1] + hi[:-1]


class Finder:
    def __init__(self, cal, img, boxes):
        self.cal, self.img = cal, img
        self.W, self.H = img.size
        hsv = np.asarray(img.convert("HSV"), float)
        hue = hsv[..., 0] * 360.0 / 255.0
        reg = Image.new("1", img.size, 0)          # 찾는 범위(책상 위 상자 · 비커 높이까지)만 본다
        for (xa, ya), (xb, yb) in boxes.values():
            c = [[x, y, z] for x in (xa, xb) for y in (ya, yb) for z in (PR.TABLE_Z, PR.TABLE_Z + PR.BIG_H)]
            ImageDraw.Draw(reg).polygon([tuple(q) for q in hull(cal.proj(c))], fill=1)
        m = (hue > HUE[0]) & (hue < HUE[1]) & (hsv[..., 2] > V_MIN) & np.asarray(reg, bool)
        bg = ImageOps.expand(Image.fromarray((~m).astype(np.uint8) * 255), 1, fill=255)
        ImageDraw.floodfill(bg, (0, 0), 128)             # 테두리로 둘러싸인 구멍(투명 몸통 너머 책상)은 채운다
        self.mask = m | (np.asarray(bg)[1:-1, 1:-1] == 255)

    def sil(self, x, y, kind):
        rt, rb, h = KIND[kind]
        z0 = PR.TABLE_Z
        P = np.vstack([self.cal.proj(U.ring((x, y), rt, z0 + h, 24)), self.cal.proj(U.ring((x, y), rb, z0, 24))])
        return hull(P)

    def score(self, x, y, kind, mask):
        poly = self.sil(x, y, kind)
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        x0, y0 = max(int(min(xs)) - 12, 0), max(int(min(ys)) - 12, 0)
        x1, y1 = min(int(max(xs)) + 12, self.W), min(int(max(ys)) + 12, self.H)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return 0.0
        m = Image.new("1", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(px - x0, py - y0) for px, py in poly], fill=1)
        m = np.asarray(m, bool)
        k = mask[y0:y1, x0:x1]
        inter = np.count_nonzero(m & k)
        return inter / max(np.count_nonzero(m) + np.count_nonzero(k) - inter, 1)

    def fit(self, kind, box, mask):
        (xa, ya), (xb, yb) = box
        best = (-1.0, None)
        for step, span in ((8.0, None), (2.0, 10.0), (0.5, 2.5)):
            if span is None:
                gx, gy = np.arange(xa, xb + 1e-6, step), np.arange(ya, yb + 1e-6, step)
            else:
                cx, cy = best[1]
                gx, gy = np.arange(cx - span, cx + span + 1e-6, step), np.arange(cy - span, cy + span + 1e-6, step)
            for x in gx:
                for y in gy:
                    s = self.score(x, y, kind, mask)
                    if s > best[0]:
                        best = (s, (float(x), float(y)))
        return best

    def run(self, boxes):
        out, mask = {}, self.mask.copy()
        for kind in ("B", "S"):                  # 큰 것부터 찾고, 찾은 모양 자리는 지운 뒤 작은 것
            iou, xy = self.fit(kind, boxes[kind], mask)
            out[kind] = (iou, xy)
            if iou >= IOU_OK:
                m = Image.new("1", (self.W, self.H), 0)
                ImageDraw.Draw(m).polygon([tuple(p) for p in self.sil(*xy, kind)], fill=1)
                mask &= ~np.asarray(m, bool)
        return out


def main():
    cal = U.Cal()
    st = {}
    try:
        st = json.load(open(U.UI_STATE))
    except Exception:
        pass
    cal.off[:2] = st.get("draw_off", [0.0, 0.0])
    poses = U.PG.load()
    S_ref, B_ref = PR.fk(poses["grasp"])[:3, 3][:2], PR.fk(poses["big"])[:3, 3][:2]
    S_ui, B_ui = np.array(st.get("S", S_ref), float), np.array(st.get("B", B_ref), float)
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1]).convert("RGB")
        src = sys.argv[1]
    else:
        img = Image.open(io.BytesIO(U.http(U.REC + "/api/frame/third.jpg", timeout=3.0))).convert("RGB")
        src = "fr5rec 지금 사진"
    span = np.array(U.SPAN) + 40.0              # 무작위 자리 범위보다 40 mm 넓게 찾음
    boxes = {"S": (S_ref - span, S_ref + span), "B": (B_ref - span, B_ref + span)}
    t0 = time.time()
    F = Finder(cal, img, boxes)
    res = F.run(boxes)
    print(f"{src} · {img.size[0]}×{img.size[1]} · 윤곽 맞추기 {cal.off[:2].round(1).tolist()} mm · 찾는 데 {time.time() - t0:.1f} s")
    ui = {"S": S_ui, "B": B_ui}
    a = img.copy()
    d = ImageDraw.Draw(a)
    for kind in ("B", "S"):
        iou, xy = res[kind]
        e = np.array(xy) - ui[kind]
        ok = iou >= IOU_OK
        print(f"  {kind}: 찾은 자리 ({xy[0]:.1f}, {xy[1]:.1f}) mm · 겹침 IoU {iou:.2f} {'' if ok else '← 못 찾음(기준 ' + str(IOU_OK) + ' 미만)'}"
              f" · 화면 윤곽 자리 ({ui[kind][0]:.1f}, {ui[kind][1]:.1f}) 와 차이 ({e[0]:+.1f}, {e[1]:+.1f}) = {np.linalg.norm(e):.1f} mm")
        poly = [tuple(p) for p in F.sil(*xy, kind)]
        d.line(poly + [poly[0]], fill=(230, 30, 30) if ok else (120, 120, 120), width=2)
        rt, _, h = KIND[kind]
        rim = cal.proj(U.ring(ui[kind], rt, PR.TABLE_Z + h, 48))
        d.line([tuple(p) for p in rim.tolist()], fill=(80, 200, 255), width=1)
    b = np.asarray(img).copy()
    b[F.mask] = (0.5 * b[F.mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    both = Image.new("RGB", (img.size[0] * 2, img.size[1]))
    both.paste(a, (0, 0)); both.paste(Image.fromarray(b), (img.size[0], 0))
    os.makedirs(U.WORK, exist_ok=True)
    fn = os.path.join(U.WORK, time.strftime("detect_probe_%H%M%S.jpg"))
    both.save(fn, quality=90)
    print("  그림:", fn)


if __name__ == "__main__":
    main()
