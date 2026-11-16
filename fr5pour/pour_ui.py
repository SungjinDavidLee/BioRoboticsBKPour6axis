#!/usr/bin/env python3
"""fr5pour 버튼 화면 — 붓기 과제(작은 비커 → 큰 비커): 배치 표시 · 한 판 실행 · fr5rec 녹화.

  fr5pour [--open] [--port 8792]          (= python pour_ui.py …)
  fr5pour --selftest                      (보정·계획만 확인, 로봇 안 움직임)

화면: 에이전트뷰(3인칭)에 작은 비커 S·큰 비커 B 자리를 그린다. 그 윤곽에 실물을 맞춰 놓고 [한 판 시작].
실행 순서: 제어판 [조작 시작] 상태 → [한 판 시작] → 지금 자세를 보내며 기다림 → 제어판 /leader [리더 추종 시작]
→ (녹화 켬이면) fr5rec 녹화 시작 → 잡기 → 붓기 → S 에 내려놓기 → 초기 자세 → 녹화 멈춤 → 송신 끊음.
저장/버림은 fr5rec 화면에서 사람이 한다(결과 1 성공 / 2 실패 → Enter 저장 · Backspace 두 번 버림).
멈춤: 화면 [멈춤] · 이 창 Ctrl+C(송신이 끊겨 제어판이 0.3 s 뒤 그 자리에 선다) · 제어판 [정지] · 비상정지.
카메라는 직접 열지 않는다 — 사진은 fr5rec /api/frame 으로 받는다. 3인칭 보정은 fr5auto rack_calib.json.
"""
import argparse
import io
import json
import math
import os
import random
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/fr5cap/fr5auto"))
import pose_go as PG            # noqa: E402
import pour_run as PR           # noqa: E402
from fr5auto import camera as CM, rackcal as RC   # noqa: E402

VERSION = "0.1.7"
REC = "http://127.0.0.1:8780"
WORK = os.path.join(HERE, "work")
UI_STATE = os.path.join(WORK, "ui_state.json")
RUNS = os.path.join(WORK, "pour_runs.jsonl")
SMALL_R, BIG_R = 27.5, 46.0                 # 위 반지름 mm
MIN_SB = 140.0                              # S–B 중심 최소 거리(그리퍼가 큰 비커 테두리에 안 닿게)
SPAN = (120.0, 80.0)                        # 무작위 자리 범위: 기준 자리 ± (x, y) mm
COL = {"S": (230, 159, 0), "B": (0, 114, 178), "tcp": (204, 121, 167), "bad": (213, 94, 0), "ink": (255, 255, 255)}
PROXY_OFF = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(url, body=None, timeout=2.0, headers=None):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=h,
                                 method="GET" if body is None else "POST")
    with PROXY_OFF.open(req, timeout=timeout) as r:
        data = r.read()
        return data if r.headers.get("Content-Type", "").startswith("image") else json.loads(data or b"{}")


def font(size):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── 3인칭 카메라 ────────────────────────────────────────────────────
class Cal:
    def __init__(self):
        rc = RC.load(os.path.expanduser("~/fr5cap/fr5auto/work"))
        if not rc or not (rc.get("third") or {}).get("T_cb"):
            raise RuntimeError("fr5auto 3인칭 보정(rack_calib.json)이 없다")
        intr = rc["third"]["intr"]
        self.cam, self.T_bc = RC.third_cam(rc, intr)
        self.W, self.H = int(intr["width"]), int(intr["height"])
        self.off = np.zeros(3)                       # 윤곽 맞추기(베이스 mm) — 그림·화면 누르기에만 쓴다

    def proj(self, P):
        P = np.asarray(P, float).reshape(-1, 3) + self.off
        return np.asarray(CM.project_world(self.cam, self.T_bc, P), float).reshape(-1, 2)

    def table_xy(self, u, v, z):
        """픽셀 → 높이 z 평면 위 베이스 xy (project_world 만 써서 가우스-뉴턴)."""
        x = np.array([150.0, 550.0])
        for _ in range(40):
            f0 = self.proj([[x[0], x[1], z]])[0]
            J = np.column_stack([self.proj([[x[0] + 1, x[1], z]])[0] - f0, self.proj([[x[0], x[1] + 1, z]])[0] - f0])
            step = np.linalg.solve(J, f0 - np.array([u, v], float))
            x = x - step
            if np.linalg.norm(step) < 0.01:
                break
        return x


def ring(c, r, z, n=48):
    a = np.linspace(0, 2 * math.pi, n + 1)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n + 1, z)])


# ── 상태 ───────────────────────────────────────────────────────────
class App:
    def __init__(self):
        os.makedirs(WORK, exist_ok=True)
        self.cal = Cal()
        self.P = PG.load()
        miss = [k for k in ("home", "grasp", "big", "pour") if k not in self.P]
        if miss:
            raise RuntimeError(f"poses.json 에 {miss} 가 없다 (pose_go.py save …)")
        self.args = PR.make_parser().parse_args([])
        self.S_ref = PR.fk(self.P["grasp"])[:3, 3][:2].copy()
        self.B_ref = PR.fk(self.P["big"])[:3, 3][:2].copy()
        st = {}
        try:
            st = json.load(open(UI_STATE))
        except Exception:
            pass
        self.S = np.array(st.get("S", self.S_ref), float)
        self.B = np.array(st.get("B", self.B_ref), float)
        self.shift = list(st.get("shift", self.args.pour_shift))
        self.cal.off[:2] = st.get("draw_off", [0.0, 0.0])
        self.gshift = np.array(st.get("grasp_shift", self.args.grasp_shift), float)
        self.grip_close = int(st.get("grip_close", self.args.close))
        self.grip_open = int(st.get("grip_open", self.args.open))
        self.gdz = float(st.get("grasp_dz", self.args.grasp_dz))   # 잡는 높이: grasp 자세 기준 mm(음수 = 더 내려서)
        self.click = "S"
        self.src = {"how": "사람"}                     # S·B 를 누가 정했나(사람 / 자동 찾기)
        self.region = self._region()
        self.lock = threading.Lock()
        self.busy, self.abort = False, False
        self.phase, self.msg, self.prog = "대기", "", ""
        self.plan_txt, self.plan_ok = "", None
        self.log = []
        self._panel = (0.0, {}, {})
        self.say(f"fr5pour {VERSION} 시작 · 기준 S {np.round(self.S_ref, 1).tolist()} · B {np.round(self.B_ref, 1).tolist()}")

    def say(self, s):
        line = time.strftime("%H:%M:%S ") + s
        self.log = (self.log + [line])[-60:]
        print(line, flush=True)

    def save_ui(self):
        json.dump({"S": self.S.tolist(), "B": self.B.tolist(), "shift": self.shift,
                   "draw_off": self.cal.off[:2].tolist(), "grasp_shift": self.gshift.tolist(),
                   "grip_close": self.grip_close, "grip_open": self.grip_open, "grasp_dz": self.gdz}, open(UI_STATE, "w"))

    def _region(self):
        """무작위 자리 범위: 기준 자리 ± SPAN 상자(S·B 따로)."""
        box = lambda c: [[c[0] - SPAN[0], c[1] - SPAN[1]], [c[0] + SPAN[0], c[1] - SPAN[1]],
                         [c[0] + SPAN[0], c[1] + SPAN[1]], [c[0] - SPAN[0], c[1] + SPAN[1]]]
        return {"S": box(self.S_ref), "B": box(self.B_ref)}

    def panel(self):
        t, st, ld = self._panel
        if time.time() - t > 0.5:
            try:
                st = PG.get("/api/state", 0.5)
            except Exception:
                st = {}
            try:
                ld = PG.get("/api/leader/state", 0.5)
            except Exception:
                ld = {}
            self._panel = (time.time(), st, ld)
        return st, ld

    # ── 계획 ──
    def plan(self, phases, S=None, B=None, q_now=None, grip_now=None):
        a = PR.make_parser().parse_args([])
        a.S = list(map(float, self.S if S is None else S))
        a.B = list(map(float, self.B if B is None else B))
        a.pour_shift = self.pshift().tolist()
        a.grasp_shift = self.gshift.tolist()
        a.grasp_dz = self.gdz
        a.close, a.open = self.grip_close, self.grip_open
        a.phases = phases
        if q_now is None:
            st = PG.get("/api/state")
            q_now = [float(x) for x in st["cmd"]]
            grip_now = float(st["gripper"]["pos"]) / 1000.0
        try:
            pl, info = PR.build(a, self.P, q_now, grip_now)
        except SystemExit as e:
            return None, None, None, [str(e)], a
        out, bad = PR.check(pl, info, np.array(self.P["home"], float))
        if np.linalg.norm(np.array(a.S) - np.array(a.B)) < MIN_SB and any(p in phases for p in ("grasp", "pour")):
            bad.append(f"S–B 거리 {np.linalg.norm(np.array(a.S) - np.array(a.B)):.0f} mm < {MIN_SB:.0f} mm (잡을 때 그리퍼가 큰 비커에 닿을 수 있음)")
        return pl, info, out, bad, a

    def plan_text(self, phases):
        pl, info, out, bad, a = self.plan(phases)
        if pl is None:
            self.plan_ok, self.plan_txt = False, "계획 안 됨: " + "; ".join(bad)
            return
        j5 = "—" if out["j5_cart"] is None else f"{out['j5_cart']:.1f}°"
        t = (f"{','.join(phases)} · {out['sec']:.0f} s · TCP 최저 {out['z_min']:.0f} mm · J6 {out['j6'][0]:.0f}~{out['j6'][1]:.0f}° · "
             f"직선 |J5| 최소 {j5} · 기울이기 직전 테두리 여유 {out['clear']:.0f} mm")
        self.plan_ok = not bad
        self.plan_txt = t + ("" if not bad else "\n보내지 않음: " + "; ".join(bad))

    def quick_ok(self, S, B):
        """무작위 자리 거르기용 빠른 검사(핵심 자세만 IK)."""
        a = self.args
        Tg, Tb, Tp = PR.fk(self.P["grasp"]), PR.fk(self.P["big"]), PR.fk(self.P["pour"])
        Rg, Rb = PR.vertical(Tg[:3, :3]), PR.vertical(Tb[:3, :3])
        z_g = Tg[2, 3] + self.gdz
        Sg = np.asarray(S, float) + self.gshift
        tests = [(np.r_[Sg, z_g + a.approach], Rg, self.P["grasp"]), (np.r_[Sg, z_g], Rg, self.P["grasp"]),
                 (np.r_[B, Tb[2, 3]], Rb, self.P["big"]),
                 (Tp[:3, 3] + np.r_[B - self.B_ref + self.pshift(), 0.0], Tp[:3, :3], self.P["pour"])]
        home6 = self.P["home"][5]
        for p, R, seed in tests:
            q, ep, er = PR.ik(seed, p, R, iters=150)
            if ep > 0.5 or er > 0.3 or abs(q[5] - home6) > PR.J6_WIN - 5:
                return False
            if any(q[i] < lo or q[i] > hi for i, (lo, hi) in enumerate(PR.LIMITS)):
                return False
        return True

    def random_pair(self, tries=400, full=10):
        """기준 둘레 상자에서 뽑고 → 핵심 자세 IK → 전체 계획 검사까지 통과한 자리."""
        rng = random.Random()
        for _ in range(tries):
            S = np.array([rng.uniform(*sorted([p[0] for p in self.region["S"]])[::3]),
                          rng.uniform(*sorted([p[1] for p in self.region["S"]])[::3])])
            B = np.array([rng.uniform(*sorted([p[0] for p in self.region["B"]])[::3]),
                          rng.uniform(*sorted([p[1] for p in self.region["B"]])[::3])])
            if np.linalg.norm(S - B) < MIN_SB or not self.quick_ok(S, B):
                continue
            pl, info, out, bad, a = self.plan(["grasp", "pour", "place", "home"], S=S, B=B,
                                              q_now=self.P["home"], grip_now=0.9)
            if pl is not None and not bad:
                return S, B
            full -= 1
            if full <= 0:
                break
        return None, None

    def find(self):
        """3인칭 사진에서 두 비커 자리 찾기(detect_probe.Finder). 둘 다 기준 넘으면 S·B 를 바꾼다."""
        sys.modules.setdefault("pour_ui", sys.modules[__name__])
        import detect_probe as DP
        img = Image.open(io.BytesIO(http(REC + "/api/frame/third.jpg", timeout=3.0))).convert("RGB")
        span = np.array(SPAN) + 40.0
        boxes = {"S": (self.S_ref - span, self.S_ref + span), "B": (self.B_ref - span, self.B_ref + span)}
        res = DP.Finder(self.cal, img, boxes).run(boxes)
        iou = {k: round(float(v[0]), 2) for k, v in res.items()}
        if min(iou.values()) < DP.IOU_OK:
            return False, f"자동 찾기 실패: IoU S {iou['S']} · B {iou['B']} (기준 {DP.IOU_OK}) — 자리 안 바꿈"
        self.S, self.B = np.array(res["S"][1], float), np.array(res["B"][1], float)
        self.src = {"how": "자동 찾기", "iou": iou}
        return True, f"자동 찾기 S {np.round(self.S, 1).tolist()} · B {np.round(self.B, 1).tolist()} · IoU S {iou['S']} · B {iou['B']}"

    def pshift(self):
        """실제로 쓰는 붓는 자리 옮김: 잡는 높이를 바꿔도 붓기 자세에서 작은 비커 입구 자리가 그대로 있게 손끝을 비커 축 방향으로 보정."""
        return np.array(self.shift, float) - PR.fk(self.P["pour"])[:3, 2][:2] * self.gdz

    # ── 그림 ──
    def view(self):
        try:
            img = Image.open(io.BytesIO(http(REC + "/api/frame/third.jpg", timeout=1.5))).convert("RGB")
        except Exception as e:
            img = Image.new("RGB", (self.cal.W, self.cal.H), (40, 40, 40))
            ImageDraw.Draw(img).text((20, 20), f"fr5rec 3인칭 사진 없음 ({type(e).__name__}) — fr5rec serve · 세션 열기", font=font(16), fill=COL["ink"])
        d = ImageDraw.Draw(img)
        f = font(15)
        z0 = PR.TABLE_Z
        for key in ("S", "B"):
            poly = [tuple(p) for p in self.cal.proj([[x, y, z0] for x, y in self.region[key]]).tolist()]
            d.line(poly + [poly[0]], fill=COL[key], width=1)
        for name, c, r, h in (("B 큰 비커", self.B, BIG_R, PR.BIG_H), ("S 작은 비커", self.S, SMALL_R, PR.SMALL_H)):
            key = name[0]
            col = COL[key]
            rim = self.cal.proj(ring(c, r, z0 + h))
            base = self.cal.proj(ring(c, r * 0.8, z0))
            d.line([tuple(p) for p in base.tolist()], fill=col, width=1)
            d.line([tuple(p) for p in rim.tolist()], fill=col, width=3)
            (u0, v0), (u1, v1) = self.cal.proj([[c[0], c[1], z0], [c[0], c[1], z0 + h]])
            d.line((u0, v0, u1, v1), fill=col, width=1)
            d.line((u0 - 9, v0, u0 + 9, v0), fill=col, width=2)
            d.line((u0, v0 - 9, u0, v0 + 9), fill=col, width=2)
            d.text((u1 + 8, v1 - 22), f"{name} ({c[0]:.0f}, {c[1]:.0f})", font=f, fill=COL["ink"], stroke_width=3, stroke_fill=col)
        g = self.S + self.gshift                      # 로봇 손끝이 실제로 잡는 자리(비커 윗부분 높이)
        (ug, vg), = self.cal.proj([[g[0], g[1], PR.fk(self.P["grasp"])[2, 3] + self.gdz]])
        d.line((ug - 6, vg - 6, ug + 6, vg + 6), fill=COL["tcp"], width=3)
        d.line((ug - 6, vg + 6, ug + 6, vg - 6), fill=COL["tcp"], width=3)
        try:                                          # 붓는 입구: 붓기 자세에서 쥔 작은 비커 입구 중심(붓는 자리 옮김 반영)
            Tp = PR.fk(self.P["pour"])
            pm = Tp[:3, 3] + np.r_[self.B - self.B_ref + self.pshift(), 0.0]
            pm = pm - Tp[:3, 2] * (PR.SMALL_H - (PR.fk(self.P["grasp"])[2, 3] + self.gdz - PR.TABLE_Z))
            (um, vm), (ur, vr) = self.cal.proj([pm, [pm[0], pm[1], z0 + PR.BIG_H]])
            d.line((um, vm, ur, vr), fill=COL["tcp"], width=1)
            d.ellipse((um - 5, vm - 5, um + 5, vm + 5), outline=COL["tcp"], width=2)
            d.line((ur - 6, vr - 6, ur + 6, vr + 6), fill=COL["tcp"], width=3)
            d.line((ur - 6, vr + 6, ur + 6, vr - 6), fill=COL["tcp"], width=3)
            d.text((ur + 8, vr - 4), "붓기", font=font(13), fill=COL["ink"], stroke_width=2, stroke_fill=COL["tcp"])
        except Exception:
            pass
        st, _ = self.panel()
        if st.get("joints"):
            try:
                u, v = self.cal.proj([PR.fk([float(x) for x in st["joints"]])[:3, 3]])[0]
                d.line((u - 7, v, u + 7, v), fill=COL["tcp"], width=2)
                d.line((u, v - 7, u, v + 7), fill=COL["tcp"], width=2)
            except Exception:
                pass
        head = f"화면을 누르면 {self.click} 자리 · 굵은 원 = 테두리 · + = 바닥 중심 · × = 잡는 자리 · 붓기 × = 붓는 입구 바로 아래"
        d.rectangle((0, 0, 12 + d.textlength(head, font=f), 24), fill=(0, 0, 0))
        d.text((6, 3), head, font=f, fill=COL["ink"])
        bio = io.BytesIO()
        img.save(bio, "JPEG", quality=85)
        return bio.getvalue()

    # ── 실행 ──
    def start(self, kind, record):
        phases = {"episode": ["grasp", "pour", "place", "home"], "pour_held": ["pour", "place", "home"],
                  "place": ["place", "home"], "home": ["up", "home"], "recover": ["untilt", "place", "home"]}[kind]
        with self.lock:
            if self.busy:
                return False, "이미 실행 중"
            self.busy, self.abort = True, False
        threading.Thread(target=self._run, args=(kind, phases, record and kind == "episode"), daemon=True).start()
        return True, "시작"

    def _rec(self, path, body):
        return http(REC + path, body, timeout=3.0, headers={"X-FR5REC": "1"})

    def _run(self, kind, phases, record):
        snd, recording, t_start = None, False, time.time()
        rec_row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "S": self.S.tolist(), "B": self.B.tolist(),
                   "shift": self.shift, "record": record, "src": self.src, "grasp_dz": self.gdz}
        try:
            self.phase, self.msg, self.prog = "계획", "", ""
            st = PG.get("/api/state")
            if st.get("mode") != "engaged":
                raise PG.Abort("제어판이 조작 상태가 아니다 — 제어판에서 [조작 시작]")
            q0 = [float(x) for x in st["cmd"]]
            g0 = float(st["gripper"]["pos"]) / 1000.0
            pl, info, out, bad, a = self.plan(phases, q_now=q0, grip_now=g0)
            if pl is None or bad:
                hint = " — 기울인 채 멈췄으면 [기울인 채 멈췄을 때 되돌리기]" if kind == "home" and PR.fk(q0)[2, 2] > -0.9 else ""
                raise PG.Abort("보내지 않음: " + "; ".join(bad) + hint)
            self.say(f"{kind} 계획 {out['sec']:.0f} s · S {np.round(self.S, 1).tolist()} · B {np.round(self.B, 1).tolist()} · shift {self.shift}")
            if record:
                cs = http(REC + "/api/camera/state", timeout=2.0)
                if not cs.get("open"):
                    raise PG.Abort("fr5rec 카메라가 닫혀 있다 — fr5rec 에서 세션을 연다")
                if cs.get("recording"):
                    raise PG.Abort("fr5rec 가 이미 녹화 중")
            snd = PG.Sender(a.udp_host, a.udp_port, pl.g[0])
            snd.q = list(pl.q[0])
            snd.start()
            self.phase = "추종 기다림"
            self.msg = "제어판 /leader 에서 [리더 추종 시작]을 누른다"
            self.say(self.msg)
            t0 = time.time()
            while True:
                if self.abort:
                    raise PG.Abort("멈춤 버튼")
                try:
                    if PG.get("/api/leader/state", 1.0).get("following"):
                        break
                except Exception:
                    pass
                if time.time() - t0 > 180:
                    raise PG.Abort("3분 안에 추종이 켜지지 않았다")
                time.sleep(0.3)
            self.say("추종 확인")
            if record:
                tag = {"by": "fr5pour", "v": VERSION, "kind": kind, "S": [round(x, 1) for x in self.S],
                       "B": [round(x, 1) for x in self.B], "shift": self.shift, "grasp_shift": [round(x, 1) for x in self.gshift], "grasp_dz": self.gdz,
                       "close": self.grip_close, "sec": round(out["sec"], 1), "src": self.src["how"]}
                r = self._rec("/api/episode/start", {"tag": tag})
                if not r.get("ok"):                   # fr5rec 는 거절도 200 + ok false 로 답한다
                    raise PG.Abort("fr5rec 가 녹화를 시작하지 않았다: " + str(r.get("error") or r)[:90] + " — fr5rec 화면 확인 뒤 다시")
                recording = True
                self.say("fr5rec 녹화 시작")
                time.sleep(0.5)
            self.phase, self.msg = "실행", ""
            n, t0, chk, worst, over, last_tag = len(pl.q), time.monotonic(), 0.0, 0.0, 0, None
            while True:
                if self.abort:
                    raise PG.Abort("멈춤 버튼")
                k = min(n - 1, int((time.monotonic() - t0) * PG.HZ))
                snd.q, snd.grip = list(pl.q[k]), pl.g[k]
                if pl.tag[k] != last_tag:
                    last_tag = pl.tag[k]
                    self.say(f"  {k / PG.HZ:5.1f} s  {last_tag}")
                self.prog = f"{last_tag} · {k / PG.HZ:.0f} / {n / PG.HZ:.0f} s"
                now = time.monotonic()
                if now - chk >= 0.2:
                    chk = now
                    s2 = PG.get("/api/state", 1.0)
                    if s2.get("mode") != "engaged":
                        raise PG.Abort("제어판 조작 상태가 꺼졌다")
                    js = [float(x) for x in s2["joints"]]
                    err = max(abs(x - y) for x, y in zip(js, snd.q)) if all(js) else 0.0
                    worst = max(worst, err)
                    over = over + 1 if err > a.max_lag else 0
                    if over >= 3:
                        raise PG.Abort(f"로봇이 명령보다 {err:.1f}° 뒤처진다")
                if k >= n - 1:
                    break
                time.sleep(1.0 / PG.HZ)
            time.sleep(1.0)
            if recording:
                self._rec("/api/episode/stop", {"reason": "fr5pour 끝"})
                recording = False
                self.say("fr5rec 녹화 멈춤 — fr5rec 화면에서 결과·저장/버림")
            rec_row.update(end="끝", worst_lag=round(worst, 1))
            self.phase = "끝"
            self.msg = "fr5rec 화면에서 결과를 정하고 저장/버림" if record else "끝"
            self.say(f"끝 · 최대 뒤처짐 {worst:.1f}°")
        except Exception as e:
            why = str(e) if isinstance(e, PG.Abort) else f"{type(e).__name__}: {e}"
            if not isinstance(e, PG.Abort):
                traceback.print_exc()
            self.phase, self.msg = "멈춤", why
            rec_row.update(end="멈춤", why=why)
            self.say("멈춤: " + why)
            if recording:
                try:
                    self._rec("/api/episode/stop", {"reason": ("fr5pour 중단: " + why)[:60]})
                except Exception:
                    pass
        finally:
            if snd:
                snd.on = False
            rec_row["sec"] = round(time.time() - t_start, 1)
            with open(RUNS, "a") as fh:
                fh.write(json.dumps(rec_row, ensure_ascii=False) + "\n")
            self.busy = False

    def status(self):
        st, ld = self.panel()
        rec = {}
        try:
            rec = http(REC + "/api/camera/state", timeout=0.5)
        except Exception:
            pass
        return {"v": VERSION, "phase": self.phase, "msg": self.msg, "prog": self.prog, "busy": self.busy,
                "S": [round(x, 1) for x in self.S], "B": [round(x, 1) for x in self.B], "shift": self.shift,
                "S_ref": [round(x, 1) for x in self.S_ref], "B_ref": [round(x, 1) for x in self.B_ref],
                "click": self.click, "draw_off": [round(x, 1) for x in self.cal.off[:2]], "gshift": [round(x, 1) for x in self.gshift],
                "grip_close": self.grip_close, "grip_open": self.grip_open, "src": self.src["how"], "gdz": self.gdz, "plan": self.plan_txt, "plan_ok": self.plan_ok, "log": self.log[-25:],
                "panel": {"mode": st.get("mode"), "following": ld.get("following"), "status": ld.get("status"),
                          "grip": (st.get("gripper") or {}).get("pos")},
                "rec": {"open": rec.get("open"), "recording": rec.get("recording")}}


# ── 웹 ──────────────────────────────────────────────────────────────
PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>fr5pour 붓기</title><style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#dddbd3;--s:#e69f00;--b:#0072b2;--warn:#d55e00;--ok:#009e73}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1b1b1a;--card:#262624;--ink:#ecebe6;--mute:#a3a29b;--line:#3a3a37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Noto Sans KR",sans-serif}
header{position:sticky;top:0;z-index:2;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}.chip{padding:2px 8px;border-radius:10px;border:1px solid var(--line);font-size:12px;color:var(--mute)}
.chip.on{color:#fff;background:var(--ok);border-color:var(--ok)}.chip.warn{color:#fff;background:var(--warn);border-color:var(--warn)}
#band{flex:1 1 100%;font-weight:600}main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:14px;padding:14px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
#view{width:100%;display:block;border-radius:6px;cursor:crosshair}h3{margin:0 0 8px;font-size:13px;color:var(--mute);font-weight:600}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:6px 0}input[type=text]{width:120px;padding:5px 6px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--ink)}
button{padding:7px 11px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);cursor:pointer;font:inherit}
button:hover{border-color:var(--mute)}button.go{background:var(--ok);color:#fff;border-color:var(--ok);font-weight:600}
button.stop{background:var(--warn);color:#fff;border-color:var(--warn);font-weight:700}button:disabled{opacity:.45;cursor:default}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
pre{white-space:pre-wrap;margin:0;font:12px/1.4 ui-monospace,Menlo,monospace;color:var(--mute)}#plan{color:var(--ink)}
.sel{outline:2px solid var(--ink)}
</style></head><body>
<header><b>fr5pour 붓기</b><span class="chip" id="cPanel">제어판</span><span class="chip" id="cFol">추종</span><span class="chip" id="cRec">fr5rec</span>
<span class="chip" id="cGrip">그리퍼</span><span style="color:var(--mute);font-size:12px" id="ver"></span><div id="band">—</div></header>
<main><section class="card"><img id="view" alt="에이전트뷰"><div class="row" style="color:var(--mute);font-size:12px">
<span><span class="sw" style="background:var(--s)"></span>S 작은 비커(잡는 쪽)</span><span><span class="sw" style="background:var(--b)"></span>B 큰 비커(받는 쪽)</span>
<span>가는 상자 = 무작위 자리 범위 · 분홍 + = 지금 손끝</span></div></section>
<section><div class="card"><h3>자리 (베이스 mm)</h3>
<div class="row">화면을 누르면 <button id="kS" onclick="setClick('S')">S</button><button id="kB" onclick="setClick('B')">B</button> 자리</div>
<div class="row"><span class="sw" style="background:var(--s)"></span>S <input type="text" id="S"><span class="sw" style="background:var(--b)"></span>B <input type="text" id="B"><button onclick="setXY()">적용</button></div>
<div class="row"><button onclick="post('/api/set',{ref:1})">기준 자리</button><button onclick="post('/api/set',{random:1})">무작위 자리</button><button onclick="post('/api/set',{find:1})">자동 찾기</button><span id="src" style="color:var(--mute);font-size:12px"></span>
붓는 자리 옮김 <input type="text" id="shift" style="width:80px"><button onclick="setShift()">적용</button></div>
<div class="row">붓는 자리 옮기기 <button onclick="pnud(0,-1)">▲ 뒤</button><button onclick="pnud(0,1)">▼ 앞</button><button onclick="pnud(-1,0)">◀ 왼쪽</button><button onclick="pnud(1,0)">▶ 오른쪽</button><span style="color:var(--mute);font-size:12px">화면 기준 · 한 번 2 mm</span></div>
<div class="row">윤곽 맞추기 <button onclick="nud(0,-1)">▲</button><button onclick="nud(0,1)">▼</button><button onclick="nud(-1,0)">◀</button><button onclick="nud(1,0)">▶</button><button onclick="post('/api/set',{nudge_reset:1})">0</button><span id="doff" style="color:var(--mute)"></span></div>
<div class="row" style="color:var(--mute);font-size:12px">기준 자리에 실물이 놓였을 때 두 윤곽이 실물과 겹치도록 누른다(한 번 2 mm). 로봇이 가는 자리는 안 바뀐다.</div>
<div class="row">잡는 자리 옮기기 <button onclick="gnud(0,-1)">▲ 뒤</button><button onclick="gnud(0,1)">▼ 앞</button><button onclick="gnud(-1,0)">◀</button><button onclick="gnud(1,0)">▶</button><button onclick="post('/api/set',{gshift_reset:1})">0</button><span id="gsh" style="color:var(--mute)"></span></div>
<div class="row" style="color:var(--mute);font-size:12px">분홍 × = 로봇이 쥐는 자리. 비커 뒤쪽을 쥐면 ▼ 앞(한 번 2 mm). 잡고 내려놓는 자리가 같이 옮겨진다.</div>
<div class="row">잡는 높이 <button onclick="gdz(-1)">▼ 더 내려서</button><button onclick="gdz(1)">▲ 더 올려서</button><button onclick="post('/api/set',{gdz_reset:1})">0</button><span id="gdzv" style="color:var(--mute)"></span><span style="color:var(--mute);font-size:12px">grasp 자세 기준 · 한 번 1 mm · −15 ~ +5</span></div>
<div class="row">그리퍼 닫기 <input type="text" id="gc" style="width:60px"> 열기 <input type="text" id="go" style="width:60px"><button onclick="setGrip()">적용</button><span style="color:var(--mute);font-size:12px">0 닫힘 – 1000 열림</span></div>
<div class="row"><button onclick="post('/api/plan',{})">계획 확인</button></div><pre id="plan"></pre></div>
<div class="card" style="margin-top:12px"><h3>실행</h3>
<div class="row"><label><input type="checkbox" id="rec" checked> fr5rec 녹화</label></div>
<div class="row"><button class="go" id="bRun" onclick="run('episode')">한 판 시작</button><button class="stop" onclick="post('/api/stop',{})">멈춤</button></div>
<div class="row"><button id="bPlace" onclick="run('place')">쥔 비커 S 에 내려놓기</button><button id="bPH" onclick="run('pour_held')">쥔 채 붓기부터</button><button id="bHome" onclick="run('home')">초기 자세로</button></div>
<div class="row"><button id="bRec" onclick="run('recover')">기울인 채 멈췄을 때 되돌리기</button><span style="color:var(--mute);font-size:12px">붓기 길을 거꾸로 → S 에 내려놓기 → 초기 자세</span></div>
<pre id="prog"></pre></div>
<div class="card" style="margin-top:12px"><h3>기록</h3><pre id="log"></pre></div></section></main>
<script>
let busy=false,loading=false;const $=id=>document.getElementById(id);
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});const j=await r.json();if(j.err)alert(j.err);tick();return j}
function setClick(w){post('/api/set',{clickmode:w})}
function setXY(){const p=s=>s.split(',').map(Number);post('/api/set',{S:p($('S').value),B:p($('B').value)})}
function pnud(x,y){post('/api/set',{pnudge:[x,y]})}
function gnud(x,y){post('/api/set',{gnudge:[x,y]})}
function gdz(d){post('/api/set',{gdz:d})}
function setGrip(){post('/api/set',{grip:{close:Number($('gc').value),open:Number($('go').value)}})}
function nud(x,y){post('/api/set',{nudge:[x,y]})}
function setShift(){post('/api/set',{shift:$('shift').value.split(',').map(Number)})}
function run(k){if(k==='episode'&&!confirm('한 판을 시작한다. 비커가 윤곽에 맞게 놓였나?'))return;post('/api/run',{kind:k,record:$('rec').checked})}
$('view').onclick=e=>{const im=$('view');const u=e.offsetX*im.naturalWidth/im.clientWidth,v=e.offsetY*im.naturalHeight/im.clientHeight;post('/api/set',{click:{u,v}})};
function img(){if(loading)return;loading=true;const i=new Image();i.onload=()=>{$('view').src=i.src;loading=false};i.onerror=()=>{loading=false};i.src='/api/view.jpg?'+Date.now()}
function chip(id,txt,cls){const c=$(id);c.textContent=txt;c.className='chip'+(cls?' '+cls:'')}
let lastV=null;
async function tick(){try{const s=await (await fetch('/api/status')).json();if(lastV&&lastV!==s.v)location.reload();lastV=s.v;$('ver').textContent='v'+s.v;
busy=s.busy;['bRun','bPlace','bPH','bHome','bRec'].forEach(i=>$(i).disabled=busy);
const p=s.panel||{};chip('cPanel','제어판 '+(p.mode||'?'),p.mode==='engaged'?'on':'warn');chip('cFol',p.following?'추종 중':'추종 꺼짐',p.following?'on':'');
chip('cRec',s.rec.open?(s.rec.recording?'fr5rec 녹화 중':'fr5rec 카메라 열림'):'fr5rec 닫힘',s.rec.recording?'warn':(s.rec.open?'on':''));chip('cGrip','그리퍼 '+(p.grip??'?'));
$('band').textContent=s.phase+(s.msg?' — '+s.msg:'');$('band').style.color=s.phase==='멈춤'?'var(--warn)':(s.phase==='추종 기다림'?'var(--b)':'');
if(document.activeElement!==$('S'))$('S').value=s.S.join(',');if(document.activeElement!==$('B'))$('B').value=s.B.join(',');if(document.activeElement!==$('shift'))$('shift').value=s.shift.join(',');
$('kS').className=s.click==='S'?'sel':'';$('src').textContent=' 자리: '+s.src;$('kB').className=s.click==='B'?'sel':'';
$('gsh').textContent=' ('+s.gshift.join(', ')+') mm';$('gdzv').textContent=' '+s.gdz+' mm';if(document.activeElement!==$('gc'))$('gc').value=s.grip_close;if(document.activeElement!==$('go'))$('go').value=s.grip_open;$('doff').textContent=' ('+s.draw_off.join(', ')+') mm';$('plan').textContent=s.plan||'';$('plan').style.color=s.plan_ok===false?'var(--warn)':'';$('prog').textContent=s.prog||'';$('log').textContent=s.log.slice().reverse().join('\n')}catch(e){$('band').textContent='서버 연결 끊김'}}
setInterval(tick,500);setInterval(img,500);tick();img();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        A = self.app
        p = self.path.split("?")[0]
        try:
            if p == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif p == "/api/status":
                self._send(200, A.status())
            elif p == "/api/view.jpg":
                self._send(200, A.view(), "image/jpeg")
            else:
                self._send(404, {"err": "없음"})
        except Exception as e:
            self._send(500, {"err": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        A = self.app
        p = self.path.split("?")[0]
        if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
            return self._send(403, {"err": "Host"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            b = json.loads(self.rfile.read(n) or b"{}")
            if p == "/api/set":
                if A.busy and any(k in b for k in ("S", "B", "ref", "random", "click", "shift", "find")):
                    return self._send(200, {"err": "실행 중에는 자리를 못 바꾼다"})
                if any(k in b for k in ("S", "B", "ref", "random", "click")):
                    A.src = {"how": "사람"}
                if b.get("find"):
                    ok, m = A.find()
                    A.say(m)
                    if not ok:
                        return self._send(200, {"err": m})
                if "clickmode" in b and b["clickmode"] in ("S", "B"):
                    A.click = b["clickmode"]
                if "click" in b:
                    xy = A.cal.table_xy(float(b["click"]["u"]), float(b["click"]["v"]), PR.TABLE_Z)
                    if A.click == "S":
                        A.S = xy
                    else:
                        A.B = xy
                    A.say(f"{A.click} ← 화면 ({xy[0]:.1f}, {xy[1]:.1f})")
                if "S" in b and len(b["S"]) == 2:
                    A.S = np.array(b["S"], float)
                if "B" in b and len(b["B"]) == 2:
                    A.B = np.array(b["B"], float)
                if b.get("ref"):
                    A.S, A.B = A.S_ref.copy(), A.B_ref.copy()
                    A.say("기준 자리")
                if b.get("random"):
                    S, B = A.random_pair()
                    if S is None:
                        return self._send(200, {"err": "조건에 맞는 무작위 자리를 못 찾았다"})
                    A.S, A.B = S, B
                    A.say(f"무작위 자리 S {np.round(S, 1).tolist()} · B {np.round(B, 1).tolist()}")
                if "nudge" in b:                       # 화면 방향으로 윤곽 옮기기(베이스 mm 로 바꿔 쌓음)
                    c = np.r_[A.S_ref, PR.TABLE_Z]
                    f0 = A.cal.proj([c])[0]
                    J = np.column_stack([A.cal.proj([c + [1, 0, 0]])[0] - f0, A.cal.proj([c + [0, 1, 0]])[0] - f0])
                    dxy = np.linalg.solve(J, [float(x) for x in b["nudge"]])
                    A.cal.off[:2] += dxy / np.linalg.norm(dxy) * float(b.get("mm", 2.0))
                    A.say(f"윤곽 맞추기 {np.round(A.cal.off[:2], 1).tolist()} mm")
                if "gnudge" in b:                      # 로봇이 잡는 자리 옮기기(화면 방향 → 베이스 mm)
                    c = np.r_[A.S_ref, PR.TABLE_Z]
                    f0 = A.cal.proj([c])[0]
                    J = np.column_stack([A.cal.proj([c + [1, 0, 0]])[0] - f0, A.cal.proj([c + [0, 1, 0]])[0] - f0])
                    dxy = np.linalg.solve(J, [float(x) for x in b["gnudge"]])
                    A.gshift = A.gshift + dxy / np.linalg.norm(dxy) * float(b.get("mm", 2.0))
                    A.say(f"잡는 자리 옮김 {np.round(A.gshift, 1).tolist()} mm")
                if b.get("gshift_reset"):
                    A.gshift = np.zeros(2)
                    A.say("잡는 자리 옮김 0")
                if "gdz" in b or b.get("gdz_reset"):       # 잡는 높이(grasp 자세 기준, 음수 = 더 내려서)
                    v = 0.0 if b.get("gdz_reset") else round(A.gdz + float(b["gdz"]), 1)
                    if not (-15.0 <= v <= 5.0):
                        return self._send(200, {"err": "잡는 높이는 −15 ~ +5 mm"})
                    A.gdz = v
                    A.say(f"잡는 높이 {v:+.1f} mm (grasp 자세 기준)")
                if "grip" in b:
                    gc, go = int(b["grip"]["close"]), int(b["grip"]["open"])
                    if not (0 <= gc < go <= 1000):
                        return self._send(200, {"err": "그리퍼 값: 0 ≤ 닫기 < 열기 ≤ 1000"})
                    A.grip_close, A.grip_open = gc, go
                    A.say(f"그리퍼 닫기 {gc} · 열기 {go}")
                if b.get("nudge_reset"):
                    A.cal.off[:] = 0.0
                    A.say("윤곽 맞추기 0")
                if "pnudge" in b:                      # 붓는 자리 옮기기(화면 방향 → 베이스 mm)
                    c = np.r_[A.B_ref, PR.TABLE_Z]
                    f0 = A.cal.proj([c])[0]
                    J = np.column_stack([A.cal.proj([c + [1, 0, 0]])[0] - f0, A.cal.proj([c + [0, 1, 0]])[0] - f0])
                    dxy = np.linalg.solve(J, [float(x) for x in b["pnudge"]])
                    A.shift = [round(float(v), 1) for v in np.array(A.shift, float) + dxy / np.linalg.norm(dxy) * float(b.get("mm", 2.0))]
                    A.say(f"붓는 자리 옮김 {A.shift} mm")
                if "shift" in b and len(b["shift"]) == 2:
                    A.shift = [float(x) for x in b["shift"]]
                A.plan_txt, A.plan_ok = "", None
                A.save_ui()
                return self._send(200, {"ok": 1})
            if p == "/api/plan":
                A.plan_text(["grasp", "pour", "place", "home"])
                return self._send(200, {"ok": 1})
            if p == "/api/run":
                ok, m = A.start(b.get("kind", "episode"), bool(b.get("record", True)))
                return self._send(200, {"ok": 1} if ok else {"err": m})
            if p == "/api/stop":
                A.abort = True
                A.say("멈춤 눌림")
                return self._send(200, {"ok": 1})
            self._send(404, {"err": "없음"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"err": f"{type(e).__name__}: {e}"})


def selftest():
    A = App()
    S0 = A.S_ref
    uv = A.cal.proj([[S0[0], S0[1], PR.TABLE_Z]])[0]
    back = A.cal.table_xy(uv[0], uv[1], PR.TABLE_Z)
    print(f"보정 왕복: 픽셀 ({uv[0]:.1f}, {uv[1]:.1f}) → ({back[0]:.2f}, {back[1]:.2f}) · 차 {np.linalg.norm(back - S0):.3f} mm")
    print("무작위 자리 범위:", {k: [[round(x), round(y)] for x, y in v] for k, v in A.region.items()})
    pl, info, out, bad, a = A.plan(["grasp", "pour", "place", "home"], S=A.S_ref, B=A.B_ref,
                                    q_now=A.P["home"], grip_now=0.9)
    print(f"기준 자리 계획: {out['sec']:.1f} s · 막힘 {bad or '없음'}")
    t = time.time()
    S, B = A.random_pair()
    print(f"무작위 자리: S {None if S is None else np.round(S, 1).tolist()} · B {None if B is None else np.round(B, 1).tolist()} · {time.time() - t:.1f} s")
    if S is not None:
        pl, info, out, bad, a = A.plan(["grasp", "pour", "place", "home"], S=S, B=B, q_now=A.P["home"], grip_now=0.9)
        print(f"  그 자리 전체 계획: {out['sec']:.1f} s · 막힘 {bad or '없음'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8792)
    ap.add_argument("--open", action="store_true", help="브라우저 띄우기(화면 있는 Thor 에서)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    H.app = App()
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    except OSError:
        sys.exit(f"포트 {a.port} 를 이미 쓰고 있다 — 켜 둔 fr5pour 창을 닫고 다시")
    url = f"http://127.0.0.1:{a.port}/"
    print(f"fr5pour {VERSION} — {url}  (워크스테이션: ssh -N -L {a.port}:127.0.0.1:{a.port} <Thor>)  끄기 Ctrl+C", flush=True)
    if a.open and os.environ.get("DISPLAY"):
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        H.app.abort = True
        time.sleep(0.3)
        print("끝 — 송신이 있었다면 끊겼다(제어판이 0.3 s 뒤 그 자리에 선다)")


if __name__ == "__main__":
    main()
