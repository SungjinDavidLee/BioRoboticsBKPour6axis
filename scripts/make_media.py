#!/usr/bin/env python3
"""실행 영상 · 장면 사진을 fr5rec 판에서 만든다.

    python3 scripts/make_media.py <pour_export 폴더>

입력: Thor 의 fr5pour/pour_export.py 가 만든 묶음을 푼 폴더 — <데이터셋>__<판>/meta.json, rgb/*.jpg(3인칭), wrist/*.jpg(손목).
출력:
  media/<판 이름>.mp4      3인칭 | 손목, 실제 속도 (fr5rec 기록 주기)
  media/<판 이름>_x6.gif   6배속 미리보기 (가로 720)
  figs/<판 이름>_scenes.jpg 장면 6장 (시작 · 쥠 · 들기 · 붓기 · 내려놓기 · 끝)
단계는 기록(그리퍼 값 · 관절로 계산한 공구 기울기)에서 나눈다: 쥐기 전 / 쥐고 옮기기 / 붓기(기울기 > 10°) / 되돌려 내려놓기 / 초기 자세로.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fr5pour"))
import pour_run as PR  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
BG, INK, INK2 = (250, 250, 248), (34, 34, 34), (85, 85, 85)
PH = [("쥐러 감", (0, 114, 178)), ("쥐고 옮기기", (86, 180, 233)), ("붓기", (213, 94, 0)), ("되돌려 내려놓기", (204, 121, 167)),
      ("초기 자세로", (85, 85, 85))]
OUT_LABEL = {"success": "성공", "fail": "실패", "unsure": "애매"}
SPEED = 6                                   # 미리보기 gif 배속


def font(size, w="Regular"):
    return ImageFont.truetype(FONT.format(w), size, index=1)


def load(ep):
    m = json.load(open(ep / "meta.json", encoding="utf-8"))
    F = m["frames"]
    t = np.array([f["t"] for f in F], float)
    t -= t[0]
    grip = np.array([f.get("grip_raw") if f.get("grip_raw") is not None else np.nan for f in F], float)
    tilt, z, last = [], [], None
    for f in F:
        q = [float(x) for x in f["joints_deg"]]
        if not all(q) and last is not None:              # 제어판이 관절을 0 으로 읽은 프레임은 앞 프레임 값
            q = last
        last = q
        T = PR.fk(q)
        tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, -T[2, 2])))))
        z.append(T[2, 3])
    tilt, z = np.array(tilt), np.array(z)
    closed = grip < 700
    k_close = int(np.argmax(closed)) if closed.any() else len(F)
    k_open = k_close + int(np.argmax(~closed[k_close:])) if closed[k_close:].any() and (~closed[k_close:]).any() else len(F)
    tilted = tilt > 10
    k_t0 = int(np.argmax(tilted)) if tilted.any() else len(F)
    k_t1 = int(np.where(tilted)[0][-1]) + 1 if tilted.any() else len(F)
    ph = np.zeros(len(F), int)
    ph[k_close:] = 1
    ph[k_t0:] = 2
    ph[k_t1:] = 3
    ph[k_open:] = 4
    keys = {"시작": 0, "쥠": min(k_close + 2, len(F) - 1), "들기": int(k_close + np.argmax(z[k_close:k_t0] >= z[k_close:k_t0].max() - 3)) if k_t0 > k_close else k_close,
            "붓기": int(k_t0 + np.argmax(tilt[k_t0:k_t1])) if k_t1 > k_t0 else k_t0, "내려놓기": max(k_open - 1, 0), "끝": len(F) - 1}
    return m, t, grip, tilt, z, ph, keys


def frame_img(ep, cam, i):
    return Image.open(ep / cam / f"{i:05d}.jpg").convert("RGB")


def compose(ep, i, m, t, grip, tilt, z, ph, title, W=848, H=480):
    top, bot = 46, 64
    im = Image.new("RGB", (2 * W, H + top + bot), BG)
    im.paste(frame_img(ep, "rgb", i).resize((W, H)), (0, top))
    im.paste(frame_img(ep, "wrist", i).resize((W, H)), (W, top))
    d = ImageDraw.Draw(im)
    d.text((14, 9), title, font=font(22, "Bold"), fill=INK)
    d.text((2 * W - 14, 13), "3인칭  |  손목", font=font(18), fill=INK2, anchor="ra")
    y0, x0, x1 = top + H + 12, 14, 2 * W - 14
    T = t[-1] if t[-1] > 0 else 1.0
    for k in range(len(t) - 1):
        a, b = x0 + (x1 - x0) * t[k] / T, x0 + (x1 - x0) * t[k + 1] / T
        d.rectangle((a, y0, b + 1, y0 + 12), fill=PH[ph[k]][1])
    cx = x0 + (x1 - x0) * t[i] / T
    d.rectangle((cx - 2, y0 - 4, cx + 2, y0 + 16), fill=INK)
    g = "—" if np.isnan(grip[i]) else f"{grip[i]:.0f}"
    d.text((x0, y0 + 20), f"{t[i]:5.1f} s · {PH[ph[i]][0]} · 손끝 높이 {z[i]:.0f} mm · 공구 기울기 {tilt[i]:.0f}° · 그리퍼 {g}",
           font=font(17), fill=INK)
    lx = x1
    for name, col in reversed(PH):
        w = d.textlength(name, font=font(15))
        d.text((lx, y0 + 22), name, font=font(15), fill=INK2, anchor="ra")
        d.rectangle((lx - w - 18, y0 + 26, lx - w - 6, y0 + 38), fill=col)
        lx -= w + 30
    return im


def main():
    src = Path(sys.argv[1])
    (ROOT / "media").mkdir(exist_ok=True)
    for ep in sorted(p for p in src.iterdir() if (p / "meta.json").exists()):
        m, t, grip, tilt, z, ph, keys = load(ep)
        tag = (m.get("fr5rec") or {}).get("tag") or {}
        name = ep.name.split("__")[-1].replace(".part", "")
        stem = f"pour_{name}_{m.get('outcome') or 'na'}"
        title = (f"fr5pour v{tag.get('v', '?')} · 자리 {tag.get('src', '사람')} · S ({tag['S'][0]:.0f}, {tag['S'][1]:.0f}) · "
                 f"B ({tag['B'][0]:.0f}, {tag['B'][1]:.0f}) · {OUT_LABEL.get(m.get('outcome'), '결과 없음')}")
        n = len(t)
        fps = max(1, round((n - 1) / t[-1])) if t[-1] > 0 else 3
        first = compose(ep, 0, m, t, grip, tilt, z, ph, title)
        Wd, Hd = first.size
        Hd -= Hd % 2
        mp4 = ROOT / "media" / f"{stem}.mp4"
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Wd}x{Hd}",
                              "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
        gif_frames = []
        step = SPEED
        for i in range(n):
            im = first if i == 0 else compose(ep, i, m, t, grip, tilt, z, ph, title)
            p.stdin.write(im.crop((0, 0, Wd, Hd)).tobytes())
            if i % step == 0:
                gif_frames.append(im.resize((720, int(720 * im.size[1] / im.size[0]))).convert("P", palette=Image.ADAPTIVE, colors=96))
        p.stdin.close()
        p.wait()
        gif = ROOT / "media" / f"{stem}_x{SPEED}.gif"
        gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
        # 장면 6장
        tiles = []
        for lab, i in keys.items():
            a = frame_img(ep, "rgb", i).resize((424, 240))
            b = frame_img(ep, "wrist", i).resize((424, 240))
            tile = Image.new("RGB", (424, 240 * 2 + 34), BG)
            tile.paste(a, (0, 34))
            tile.paste(b, (0, 34 + 240))
            ImageDraw.Draw(tile).text((6, 4), f"{lab} · {t[i]:.0f} s · 기울기 {tilt[i]:.0f}°", font=font(18, "Bold"), fill=INK)
            tiles.append(tile)
        sheet = Image.new("RGB", (424 * 3 + 2 * 8, (240 * 2 + 34) * 2 + 8), BG)
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % 3) * (424 + 8), (k // 3) * (240 * 2 + 34 + 8)))
        sheet.save(ROOT / "figs" / f"{stem}_scenes.jpg", quality=88)
        print(f"{ep.name}: {n} 프레임 · {t[-1]:.0f} s · {fps} fps → {mp4.name} ({mp4.stat().st_size / 1e6:.1f} MB), "
              f"{gif.name} ({gif.stat().st_size / 1e6:.1f} MB), {stem}_scenes.jpg · 장면 {keys}")


if __name__ == "__main__":
    main()
