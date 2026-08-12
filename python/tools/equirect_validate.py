#!/usr/bin/env python3
"""wme_equirect_convert 를 **합성 등장방형 영상** 으로 수치 검증한다.

왜 합성인가
-----------
저장소에 360° 실데이터가 없다. 그런데 여기서 답해야 하는 질문은 "예쁘게
보이는가" 가 아니라 "잘라낸 뷰가 정말 핀홀이고, 함께 내보낸 (fx, fy, cx, cy)
가 정말 그 영상의 내부파라미터인가" 다. 그 질문은 **정답을 아는 장면** 에서만
답할 수 있다. 실데이터로는 애초에 답이 안 나온다 - 비교할 진리값이 없다.

그래서 3D 좌표를 아는 장면을 해석적으로 등장방형에 투영해 원본을 만든다.
원본이 리샘플링 산물이 아니라 **직접 계산된 값** 이라는 점이 중요하다. 원본을
어디선가 가져와 워프했다면 원본의 오차와 도구의 오차가 섞여서 분리되지 않는다.

무엇을 재는가
-------------
  A. 오라클 차등  : C++ (tools/equirect_convert.cpp) 대 numpy
                    (wme/reference/equirect.py) 의 화소 일치.
  B. 재투영 오차  : 3D 좌표를 아는 체커보드 코너가, 도구가 **스스로 내보낸**
                    calib.txt 의 K 로 예측한 화소에 실제로 오는가. 단위는 px.
  C. 내부파라미터 회수 : 잘라낸 영상들만 가지고 cv2.calibrateCamera 를 돌려
                    fx, fy, cx, cy 를 되찾고 내보낸 값과 비교한다. B 가 "K 가
                    자기 자신과 일관된다" 를 보는 것이라면 C 는 "그 K 가 유일한
                    답인가" 를 본다 - 규약이 반 화소 밀려 있으면 여기서 걸린다.
  D. 360 스테레오 : 두 파노라마 중심에서 렌더한 쌍으로 깊이를 만들고 해석
                    깊이와 비교한다. 그리고 정렬 가능한 yaw 한계를 오라클로
                    닫힌 형태로 풀어, 그 **경계 양옆에서** 도구의 채택/거부가
                    갈리는지 본다 (거부하지 않는 것이 실패다).
  E. 종단 (--e2e) : 변환 산출물을 wme_tum_odometry 에 그대로 먹인다. 기하가
                    맞아도 산출물 형식이 규약과 어긋나면 소용이 없고, 그건
                    돌려 봐야 아는 것이다.

정직하게 말해 둘 것
-------------------
D 의 깊이 오차는 **하한** 이다. 합성 쌍에는 실제 360 스테레오가 겪는 것이 빠져
있다: 렌즈/노출 차이가 0 이고, 두 어안을 이어 붙일 때 생기는 봉합선 왜곡이
없으며, 표면은 완전 램버시안이다. 셋 다 SGBM 을 유리하게 만든다
(stereo_validate.py 가 가상 우안에 대해 같은 단서를 달아 둔 것과 같은 이유).
반면 A~C 의 기하 오차는 하한이 아니다 - 워프는 영상 내용과 무관하므로 실데이터
에서도 같은 크기다.

사용:
  python python/tools/equirect_validate.py [--exe <경로>] [--out <결과 json>]
  python python/tools/equirect_validate.py --quick     # 해상도를 낮춰 빠르게
  python python/tools/equirect_validate.py --e2e       # 러너까지 돌린다 (느리다)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import cv2  # noqa: E402

from wme.reference.equirect import (  # noqa: E402
    PinholeView,
    build_maps,
    derive_intrinsics,
    direction_from_equirect,
    max_vertical_disparity_px,
    project,
    rectifiable_yaw_limit_deg,
    render,
    view_rotation,
)

# OpenCV DLL 은 실행 파일 옆에 없다 (CMakeLists 의 wme_stage_runtime_dlls 는
# onnxruntime 만 배치한다). PATH 에 넣지 않으면 exe 가 exit -1073741515 로
# 죽고, 그 코드는 "DLL 을 못 찾았다" 라는 뜻인데 화면에는 아무것도 안 나온다.
OPENCV_BIN = Path("C:/opencv-dl/opencv/build/x64/vc16/bin")

DEFAULT_EXE = ROOT / "build" / "win" / "tools" / "wme_equirect_convert.exe"


# ---------------------------------------------------------------------------
# 장면: 해석적으로 셰이딩되는 3D 물체들
# ---------------------------------------------------------------------------

class Checkerboard:
    """3D 공간의 평면 체커보드. 코너의 3D 좌표를 우리가 안다.

    코너를 쓰는 이유: 코너는 워프에 **공변** 이다. 사영변환이든 등장방형
    워프든 코너는 코너로 간다. 반면 점(블롭)의 무게중심은 워프가 비등방으로
    늘이면 치우친다 - 그 치우침이 기하 오차와 섞이면 분리가 안 된다.
    """

    def __init__(self, center, u_axis, v_axis, square_m, cols, rows,
                 dark=30.0, light=225.0, quiet=245.0):
        self.center = np.asarray(center, dtype=float)
        u = np.asarray(u_axis, dtype=float)
        u /= np.linalg.norm(u)
        v = np.asarray(v_axis, dtype=float)
        v -= u * (v @ u)          # 직교화. 사용자가 준 축이 어긋나 있으면
        v /= np.linalg.norm(v)    # 그램-슈미트로 바로잡는다.
        self.u, self.v = u, v
        self.n = np.cross(u, v)
        self.s = float(square_m)
        self.cols, self.rows = int(cols), int(rows)   # **내부 코너** 개수
        # 내부 코너가 cols 개면 사각형은 cols+1 개다.
        self.half_a = (self.cols + 1) * self.s / 2.0
        self.half_b = (self.rows + 1) * self.s / 2.0
        self.margin = self.s      # 검출기가 요구하는 흰 여백
        self.dark, self.light, self.quiet = dark, light, quiet

    def corners3d(self) -> np.ndarray:
        """내부 코너의 3D 좌표 (rows*cols, 3). 순서는 행 우선(v 느리게, u 빠르게)."""
        i = np.arange(1, self.cols + 1) * self.s - self.half_a
        j = np.arange(1, self.rows + 1) * self.s - self.half_b
        jj, ii = np.meshgrid(j, i, indexing="ij")
        return (self.center
                + ii.reshape(-1, 1) * self.u
                + jj.reshape(-1, 1) * self.v)

    def corners_local(self) -> np.ndarray:
        """같은 순서의 판 좌표계 (a, b, 0). cv2.calibrateCamera 의 objectPoints."""
        i = np.arange(1, self.cols + 1) * self.s - self.half_a
        j = np.arange(1, self.rows + 1) * self.s - self.half_b
        jj, ii = np.meshgrid(j, i, indexing="ij")
        return np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size)], axis=1)

    def trace(self, o, d):
        """광선-평면 교차. (t, 적중여부). t 는 미적중에서 inf."""
        den = d @ self.n
        num = (self.center - o) @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
            p = o + t[:, None] * d
            rel = p - self.center
            a, b = rel @ self.u, rel @ self.v
            hit = (np.abs(den) > 1e-12) & np.isfinite(t) & (t > 1e-9) & \
                  (np.abs(a) <= self.half_a + self.margin) & \
                  (np.abs(b) <= self.half_b + self.margin)
        t = np.where(hit, t, np.inf)
        return t, hit, np.nan_to_num(a), np.nan_to_num(b)

    def shade_at(self, a, b):
        ia = np.floor((a + self.half_a) / self.s)
        ib = np.floor((b + self.half_b) / self.s)
        pat = np.where((ia + ib) % 2 == 0, self.dark, self.light)
        inside = (np.abs(a) <= self.half_a) & (np.abs(b) <= self.half_b)
        return np.where(inside, pat, self.quiet)


class BoardScene:
    """체커보드 여러 장 + 균일한 배경."""

    def __init__(self, boards, background=110.0):
        self.boards = list(boards)
        self.background = float(background)

    def shade(self, o, d):
        n = d.shape[0]
        best_t = np.full(n, np.inf)
        val = np.full(n, self.background)
        for b in self.boards:
            t, hit, a, bb = b.trace(o, d)
            closer = hit & (t < best_t)
            if np.any(closer):
                best_t = np.where(closer, t, best_t)
                val = np.where(closer, b.shade_at(a, bb), val)
        return val


class NoiseRoom:
    """직육면체 방. 벽 무늬는 **3D 공간에 고정된** 잡음이다.

    3D 에 고정해야 하는 이유: 스테레오는 두 카메라가 같은 표면점을 같은 밝기로
    본다는 전제(램버시안) 위에 선다. 무늬를 영상 좌표에 붙이면 그 전제가 깨져
    SGBM 이 못 맞추고, 그러면 우리가 재는 것이 기하가 아니라 렌더러 버그다.
    """

    def __init__(self, half, seed=20260812, cells=96, octaves=2):
        self.half = np.asarray(half, dtype=float)
        rng = np.random.default_rng(seed)
        self.lo = -self.half * 1.05
        span = 2.0 * self.half * 1.05
        self.grids, self.cell = [], []
        for k in range(octaves):
            n = cells * (2 ** k)
            self.grids.append(rng.random((n + 1, n + 1, n + 1)).astype(np.float32))
            self.cell.append(span / n)

    def trace(self, o, d):
        """방 안에서 밖으로 나가는 교차점까지의 거리 (슬래브 exit)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (np.sign(d) * self.half - o) / d
        t = np.where(np.abs(d) < 1e-12, np.inf, t)
        return np.min(t, axis=1)

    def _noise(self, p, k):
        g = (p - self.lo) / self.cell[k]
        i0 = np.floor(g).astype(np.int64)
        f = (g - i0).astype(np.float32)
        shp = np.array(self.grids[k].shape) - 2
        i0 = np.clip(i0, 0, shp)
        x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]
        G = self.grids[k]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = G[x, y, z] * (1 - fx) + G[x + 1, y, z] * fx
        c01 = G[x, y, z + 1] * (1 - fx) + G[x + 1, y, z + 1] * fx
        c10 = G[x, y + 1, z] * (1 - fx) + G[x + 1, y + 1, z] * fx
        c11 = G[x, y + 1, z + 1] * (1 - fx) + G[x + 1, y + 1, z + 1] * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def shade(self, o, d):
        t = self.trace(o, d)
        p = o + t[:, None] * d
        v = np.zeros(p.shape[0], dtype=np.float32)
        amp, tot = 1.0, 0.0
        for k in range(len(self.grids)):
            v += amp * self._noise(p, k)
            tot += amp
            amp *= 0.5
        v /= tot
        # 20~235 로 펴 둔다. 0/255 에 붙으면 클리핑이 정합을 도와 버려서
        # 측정이 낙관적으로 기운다.
        return 20.0 + 215.0 * v


# ---------------------------------------------------------------------------
# 해석적 렌더러
# ---------------------------------------------------------------------------

def render_equirect(src_w, src_h, origin, scene, ss=2, block=96, R_world=None):
    """등장방형 영상을 해석적으로 그린다 (화소당 ss x ss 초과표본).

    R_world 는 리그(파노라마) 자세 R_world_rig 다. 리그가 돌면 파노라마 내용도
    같이 돌아야 한다 - 360 카메라는 리그에 고정되어 있으므로.
    """
    origin = np.asarray(origin, dtype=float)
    img = np.zeros((src_h, src_w), dtype=np.float64)
    off = (np.arange(ss) + 0.5) / ss - 0.5
    us = np.arange(src_w)
    for y0 in range(0, src_h, block):
        y1 = min(y0 + block, src_h)
        vv = (np.arange(y0, y1)[:, None, None, None] + off[None, None, :, None])
        uu = (us[None, :, None, None] + off[None, None, None, :])
        vv, uu = np.broadcast_arrays(vv, uu)
        d = direction_from_equirect(uu, vv, src_w, src_h).reshape(-1, 3)
        if R_world is not None:
            d = d @ np.asarray(R_world, dtype=float).T
        val = scene.shade(origin, d).reshape(y1 - y0, src_w, ss, ss)
        img[y0:y1] = val.mean(axis=(2, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def render_pinhole(view: PinholeView, origin, scene, ss=2):
    """같은 장면을 **등장방형을 거치지 않고** 바로 핀홀로 그린다.

    이것이 있어야 재투영 오차를 두 조각으로 나눌 수 있다: 검출기 자체의 편향과
    등장방형 워프가 더한 오차. 전자는 두 경로에 공통이므로 차이가 곧 후자다.
    """
    origin = np.asarray(origin, dtype=float)
    R = view_rotation(view.yaw_deg, view.pitch_deg)
    off = (np.arange(ss) + 0.5) / ss - 0.5
    uu = (np.arange(view.width)[None, :, None, None] + off[None, None, None, :])
    vv = (np.arange(view.height)[:, None, None, None] + off[None, None, :, None])
    vv, uu = np.broadcast_arrays(vv, uu)
    x = (uu - view.cx) / view.fx
    y = (vv - view.cy) / view.fy
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d = (d.reshape(-1, 3) @ R.T)
    val = scene.shade(origin, d).reshape(view.height, view.width, ss, ss)
    return np.clip(np.rint(val.mean(axis=(2, 3))), 0, 255).astype(np.uint8)


def analytic_depth(view: PinholeView, origin, scene):
    """잘라낸 뷰의 화소별 **정답 깊이** Z (광축 방향 성분, 미터)."""
    origin = np.asarray(origin, dtype=float)
    R = view_rotation(view.yaw_deg, view.pitch_deg)
    uu, vv = np.meshgrid(np.arange(view.width, dtype=float),
                         np.arange(view.height, dtype=float))
    x = (uu - view.cx) / view.fx
    y = (vv - view.cy) / view.fy
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d /= norm
    t = scene.trace(origin, d.reshape(-1, 3) @ R.T)
    # 거리 t 를 광축 성분으로 바꾼다: Z = t * cos(광축과의 각) = t * d_cam.z.
    # d 는 정규화되어 있고 정규화 전 z 가 1 이었으므로 d_cam.z = 1/norm 이다.
    return t.reshape(view.height, view.width) / norm[..., 0]


# ---------------------------------------------------------------------------
# 도구 호출 / 산출물 읽기
# ---------------------------------------------------------------------------

def run_tool(exe: Path, args, expect_fail=False):
    env = dict(os.environ)
    if OPENCV_BIN.is_dir():
        env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    cmd = [str(exe)] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    if (r.returncode != 0) != expect_fail:
        raise RuntimeError(
            f"도구 실행 결과가 예상과 다르다 (rc={r.returncode}, expect_fail={expect_fail})\n"
            f"  {' '.join(cmd)}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return r


def read_calib(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0]
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        try:
            out[k.strip()] = float(v) if " " not in v else [float(x) for x in v.split()]
        except ValueError:
            out[k.strip()] = v
    return out


def view_from_calib(c: dict) -> PinholeView:
    """도구가 내보낸 calib.txt 만으로 뷰를 복원한다.

    오라클의 derive_intrinsics 를 다시 부르지 않는다 - 그러면 도구가 실제로
    무엇을 썼는지가 아니라 오라클이 무엇을 믿는지를 재게 된다.
    """
    return PinholeView(
        width=int(c["width"]), height=int(c["height"]),
        fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
        yaw_deg=c["equirect_yaw_deg"], pitch_deg=c["equirect_pitch_deg"],
        hfov_deg=c["equirect_hfov_deg"], vfov_deg=c["equirect_vfov_deg"])


# ---------------------------------------------------------------------------
# A. 오라클 차등
# ---------------------------------------------------------------------------

def test_oracle(exe: Path, work: Path, src: np.ndarray, cases, interp="cubic"):
    src_path = work / "src_A.png"
    cv2.imwrite(str(src_path), src)
    rows = []
    for (yaw, pitch, hfov, W, H) in cases:
        out = work / f"A_{yaw}_{pitch}_{hfov}"
        run_tool(exe, ["--in", src_path, "--out", out, "--yaw", yaw, "--pitch", pitch,
                       "--hfov", hfov, "--width", W, "--height", H, "--interp", interp])
        cpp = cv2.imread(str(out / "rgb" / "000000.png"), cv2.IMREAD_GRAYSCALE)
        calib = read_calib(out / "calib.txt")
        view = derive_intrinsics(W, H, hfov, yaw, pitch)
        ref = render(src, view, interp)
        if ref.ndim == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

        diff = np.abs(cpp.astype(np.int32) - ref.astype(np.int32))
        # map 자체도 비교한다. 화소가 같아도 map 이 미묘하게 다를 수 있고
        # (평탄한 영역에서는 티가 안 난다) 그 차이는 실데이터에서 드러난다.
        mx, my = build_maps(view, src.shape[1], src.shape[0])
        rows.append({
            "yaw": yaw, "pitch": pitch, "hfov": hfov, "size": [W, H],
            "px_max_abs": int(diff.max()),
            "px_rms": float(np.sqrt((diff.astype(float) ** 2).mean())),
            "px_gt1_ratio": float((diff > 1).mean()),
            "K_max_abs_err": float(max(abs(calib["fx"] - view.fx),
                                       abs(calib["fy"] - view.fy),
                                       abs(calib["cx"] - view.cx),
                                       abs(calib["cy"] - view.cy))),
            "map_x_range": [float(mx.min()), float(mx.max())],
        })
    return rows


# ---------------------------------------------------------------------------
# B/C. 체커보드 재투영 오차 + 내부파라미터 회수
# ---------------------------------------------------------------------------

def detect_corners(img, cols, rows):
    """서브픽셀 체스보드 코너. 실패하면 None (조용히 넘어가지 않는다)."""
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, pts = cv2.findChessboardCornersSB(img, (cols, rows), flags=flags)
    if not ok:
        return None
    return pts.reshape(-1, 2).astype(np.float64)


def match_to_predicted(detected, predicted):
    """검출 코너를 예측 코너에 최근접으로 짝짓는다.

    findChessboardCornersSB 의 반환 순서는 판의 방향에 따라 뒤집힌다. 순서를
    가정하면 오차가 아니라 짝짓기 실패를 재게 되므로, 짝은 거리로 정하고
    일대일인지까지 확인한다. 일대일이 아니면 예측 자체가 틀린 것이다.
    """
    d2 = ((detected[:, None, :] - predicted[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    if len(set(idx.tolist())) != len(idx):
        return None
    return idx


def test_board(exe: Path, work: Path, src_w, src_h, view_wh, hfov, ss):
    W, H = view_wh
    # 판을 여러 장, 서로 다른 자세로 둔다. 판마다 등장방형을 따로 그리므로
    # 서로 가려도 상관없고, 같은 뷰 파라미터로 자르므로 K 는 전부 같다 -
    # 즉 "카메라 하나 + 자세 여러 개" 라는 캘리브레이션의 전제가 성립한다.
    #
    # 기울기가 커야 한다. 정면 판만 있으면 초점거리와 판까지의 거리가 축퇴해
    # (f 를 키우고 Z 를 같은 비율로 키우면 영상이 같다) fx 가 흔들린다.
    # u/v 의 z 성분 0.5 는 약 27 도, 0.9 는 약 42 도 기울기다.
    boards = [
        Checkerboard([0.0, 0.0, 3.0], [1, 0, 0.60], [0, 1, 0.0], 0.12, 7, 5),
        Checkerboard([-1.0, -0.6, 3.2], [1, 0, -0.75], [0, 1, 0.30], 0.12, 7, 5),
        Checkerboard([1.1, 0.55, 3.4], [1, 0, 0.85], [-0.15, 1, 0.25], 0.12, 7, 5),
        Checkerboard([-1.2, 0.70, 3.6], [1, 0.25, 0.40], [0, 1, -0.60], 0.12, 7, 5),
        Checkerboard([1.3, -0.70, 3.8], [1, -0.20, -0.55], [0.15, 1, 0.35], 0.12, 7, 5),
        Checkerboard([0.0, 0.95, 4.4], [1, 0, 0.15], [0, 1, 0.90], 0.14, 7, 5),
        Checkerboard([0.1, -1.05, 4.2], [1, 0, -0.20], [0, 1, -0.85], 0.14, 7, 5),
        Checkerboard([0.3, 0.10, 2.6], [1, 0.15, 0.50], [0, 1, 0.10], 0.10, 7, 5),
    ]

    rows, obj_pts, img_pts, img_pts_direct = [], [], [], []
    for bi, board in enumerate(boards):
        scene = BoardScene([board])
        eq = render_equirect(src_w, src_h, [0, 0, 0], scene, ss=ss)
        src_path = work / f"B_{bi}.png"
        cv2.imwrite(str(src_path), eq)

        out = work / f"B_{bi}_out"
        run_tool(exe, ["--in", src_path, "--out", out, "--yaw", 0, "--pitch", 0,
                       "--hfov", hfov, "--width", W, "--height", H])
        img = cv2.imread(str(out / "rgb" / "000000.png"), cv2.IMREAD_GRAYSCALE)
        view = view_from_calib(read_calib(out / "calib.txt"))

        c3d = board.corners3d()
        pu, pv, pz = project(view, c3d)
        pred = np.stack([pu, pv], axis=1)
        inside = (pz > 0) & (pu > 8) & (pu < W - 9) & (pv > 8) & (pv < H - 9)
        if not inside.all():
            rows.append({"board": bi, "error": "판이 뷰 밖으로 나간다 - 장면 설정 문제"})
            continue

        det = detect_corners(img, board.cols, board.rows)
        if det is None:
            rows.append({"board": bi, "error": "코너 검출 실패"})
            continue
        idx = match_to_predicted(det, pred)
        if idx is None:
            rows.append({"board": bi, "error": "코너 짝짓기가 일대일이 아니다"})
            continue

        err = det - pred[idx]
        # 같은 장면을 등장방형을 거치지 않고 바로 핀홀로 그려 같은 검출기를
        # 돌린다. 그 오차는 검출기 편향이고, 두 경로의 차이가 워프의 몫이다.
        direct = render_pinhole(view, [0, 0, 0], scene, ss=ss)
        det_d = detect_corners(direct, board.cols, board.rows)
        derr, det_d_ord = None, None
        if det_d is not None:
            idx_d = match_to_predicted(det_d, pred)
            if idx_d is not None:
                derr = det_d - pred[idx_d]
                # obj_pts 와 순서를 맞춰 둔다. idx / idx_d 는 각자의 검출 순서를
                # 예측 순서로 보내는 사상이므로, 예측 순서로 되돌려 저장한다.
                det_d_ord = np.empty_like(det_d)
                det_d_ord[idx_d] = det_d

        rows.append({
            "board": bi,
            "n_corners": int(len(err)),
            "rms_px": float(np.sqrt((err ** 2).sum(axis=1).mean())),
            "max_px": float(np.sqrt((err ** 2).sum(axis=1)).max()),
            "bias_px": [float(err[:, 0].mean()), float(err[:, 1].mean())],
            "direct_rms_px": None if derr is None
                             else float(np.sqrt((derr ** 2).sum(axis=1).mean())),
            "direct_bias_px": None if derr is None
                              else [float(derr[:, 0].mean()), float(derr[:, 1].mean())],
        })
        if det_d_ord is None:
            continue     # 대조군이 없으면 두 캘리브레이션의 뷰 집합이 달라진다
        obj_pts.append(board.corners_local()[idx].astype(np.float32))
        img_pts.append(det.astype(np.float32))
        img_pts_direct.append(det_d_ord[idx].astype(np.float32))

    # C. 잘라낸 영상만으로 K 를 되찾는다.
    #
    # **대조군이 반드시 있어야 한다.** 여기서 나오는 오차에는 두 가지가 섞인다:
    # (1) 잘라낸 영상의 기하가 정말 틀린 몫, (2) 이 판 배치로 K 를 푸는 문제
    # 자체의 조건수(초점거리와 거리의 축퇴, 코너 검출 잡음). 같은 장면을
    # 등장방형을 거치지 않고 **정확히 그 K 로** 직접 렌더한 영상에 똑같은
    # 캘리브레이션을 돌리면 (2)만 남는다. 두 값이 비슷하면 (1)은 (2)에 묻힐
    # 만큼 작다는 뜻이고, 잘라낸 쪽만 크면 그때가 진짜 기하 오차다.
    def calibrate(pts, tag):
        # 왜곡은 0 으로 고정한다. 재매핑으로 만든 이상적 핀홀이라 왜곡 자유도를
        # 열어 두면 그것이 기하 오차를 흡수해 버려서, 정작 재고 싶은 fx/cx 의
        # 편차가 왜곡 계수로 숨는다.
        flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
                 cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        rms, K, _, _, _ = cv2.calibrateCamera(obj_pts, pts, (W, H), None, None,
                                              flags=flags)
        truth = derive_intrinsics(W, H, hfov)
        return {
            "which": tag,
            "n_views": len(obj_pts),
            "reproj_rms_px": float(rms),
            "fx": [float(K[0, 0]), truth.fx],
            "fy": [float(K[1, 1]), truth.fy],
            "cx": [float(K[0, 2]), truth.cx],
            "cy": [float(K[1, 2]), truth.cy],
            "rel_err_fx_pct": float(100.0 * abs(K[0, 0] - truth.fx) / truth.fx),
            "rel_err_fy_pct": float(100.0 * abs(K[1, 1] - truth.fy) / truth.fy),
            "err_cx_px": float(abs(K[0, 2] - truth.cx)),
            "err_cy_px": float(abs(K[1, 2] - truth.cy)),
        }

    if len(obj_pts) < 3:
        return rows, {"error": f"판이 모자라 캘리브레이션을 못 한다 ({len(obj_pts)} 장)"}
    return rows, {"cropped": calibrate(img_pts, "등장방형 -> 잘라낸 뷰"),
                  "direct": calibrate(img_pts_direct, "대조군: 핀홀 직접 렌더")}


# ---------------------------------------------------------------------------
# D. 360 스테레오 리그
# ---------------------------------------------------------------------------

def test_stereo(exe: Path, work: Path, src_w, src_h, view_wh, hfov,
                baseline, min_depth, ss):
    W, H = view_wh
    room = NoiseRoom(half=[6.0, 2.5, 8.0])
    # 베이스라인은 파노라마 X 축. 도구의 checkRectified 유도가 그 전제 위에 있다.
    left_o = np.array([-baseline / 2.0, 0.0, 0.0])
    right_o = np.array([+baseline / 2.0, 0.0, 0.0])

    lp, rp = work / "D_left.png", work / "D_right.png"
    cv2.imwrite(str(lp), render_equirect(src_w, src_h, left_o, room, ss=ss))
    cv2.imwrite(str(rp), render_equirect(src_w, src_h, right_o, room, ss=ss))

    out = work / "D_out"
    r = run_tool(exe, ["--in", lp, "--out", out, "--right", rp,
                       "--baseline", baseline, "--min-depth", min_depth,
                       "--yaw", 0, "--pitch", 0, "--hfov", hfov,
                       "--width", W, "--height", H])
    view = view_from_calib(read_calib(out / "calib.txt"))
    calib = read_calib(out / "calib.txt")

    d16 = cv2.imread(str(out / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    est = d16.astype(np.float64) / calib["depth_scale"]
    gt = analytic_depth(view, left_o, room)

    valid = est > 0
    err = est[valid] - gt[valid]
    rel = np.abs(err) / gt[valid]

    # 이론 한계도 같이 낸다: dZ = Z^2/(f*B) * dd. SGBM 의 시차 양자화는 1/16 px
    # 이지만 실제 정합 잡음이 지배하므로 dd = 0.5 px 로 본다 (StereoDepthConfig
    # 의 disparity_noise_px 기본값과 같은 근거).
    zmed = float(np.median(gt[valid]))
    sigma_theory = zmed ** 2 / (view.fx * baseline) * 0.5

    # 정렬 판정을 **경계에서** 확인한다. yaw 90 도만 거부되는지 보는 것은
    # 너무 무른 시험이다 - 판정식이 통째로 틀려도 90 도는 어차피 걸린다.
    # 오라클로 한계각을 닫힌 형태로 풀고, 그 양옆에서 도구의 답이 갈리는지 본다.
    lim = rectifiable_yaw_limit_deg(view, baseline, min_depth)
    below = run_tool(exe, ["--in", lp, "--out", work / "D_lim_ok", "--right", rp,
                           "--baseline", baseline, "--min-depth", min_depth,
                           "--yaw", round(lim * 0.9, 4), "--hfov", hfov,
                           "--width", W, "--height", H])
    above = run_tool(exe, ["--in", lp, "--out", work / "D_lim_ng", "--right", rp,
                           "--baseline", baseline, "--min-depth", min_depth,
                           "--yaw", round(lim * 1.1, 4), "--hfov", hfov,
                           "--width", W, "--height", H], expect_fail=True)
    bad = run_tool(exe, ["--in", lp, "--out", work / "D_bad", "--right", rp,
                         "--baseline", baseline, "--min-depth", min_depth,
                         "--yaw", 90, "--hfov", hfov, "--width", W, "--height", H],
                   expect_fail=True)

    return {
        "rectifiable_yaw_limit_deg": lim,
        "limit_boundary_ok": (below.returncode == 0) and (above.returncode != 0),
        "dv_at_limit_px": max_vertical_disparity_px(
            derive_intrinsics(W, H, hfov, lim, 0.0), baseline, min_depth),
        "baseline_m": baseline,
        "fx_px": view.fx,
        "valid_ratio": float(valid.mean()),
        "gt_depth_median_m": zmed,
        "gt_depth_range_m": [float(gt.min()), float(gt.max())],
        "abs_err_median_m": float(np.median(np.abs(err))),
        "abs_err_p95_m": float(np.percentile(np.abs(err), 95)),
        "rms_err_m": float(np.sqrt((err ** 2).mean())),
        "rel_err_median_pct": float(100.0 * np.median(rel)),
        "bias_m": float(np.mean(err)),
        "sigma_theory_at_median_m": float(sigma_theory),
        "yaw90_rejected": bad.returncode != 0,
        "yaw90_message_head": bad.stderr.strip().splitlines()[:3],
        "tool_stdout_tail": r.stdout.strip().splitlines()[-6:],
    }


# ---------------------------------------------------------------------------
# E. 파이프라인 종단 확인
# ---------------------------------------------------------------------------

def test_e2e(exe: Path, odom_exe: Path, work: Path, src_w, src_h, view_wh, hfov,
             baseline, min_depth, ss, n_frames=16, step_m=0.10, yaw_rate_deg=0.7):
    """변환 산출물을 **실제 러너** 에 그대로 먹여 본다.

    앞의 A~D 는 잘라낸 뷰의 기하를 잰다. 그것이 맞아도 산출물의 **형식** 이
    파이프라인 규약과 어긋나면 아무 소용이 없다 - calib.txt 를 dataset_calib.hpp
    가 읽는가, 인덱스 파일 짝이 맞는가, 깊이 스케일이 맞는가. 그건 코드를 읽어
    확인할 게 아니라 돌려 봐야 아는 것이다.

    리그를 **직선이 아니라 완만한 호** 로 움직인다. 직선 등속이면 Kabsch 정렬이
    진행 방향 축 회전에 대해 축퇴해서 ATE 가 실제보다 좋게 나온다.
    """
    W, H = view_wh
    room = NoiseRoom(half=[6.0, 2.5, 8.0])
    ldir, rdir = work / "E_left", work / "E_right"
    ldir.mkdir(exist_ok=True)
    rdir.mkdir(exist_ok=True)

    fps = 10.0
    poses = []      # (stamp, R_wc, t_wc)  카메라(왼쪽 크롭) 의 세계 자세
    pos = np.array([0.0, 0.0, -3.0])
    for k in range(n_frames):
        yaw = np.deg2rad(k * yaw_rate_deg)
        # 리그 자세: Y(아래) 축 회전. 파노라마계 규약과 같은 손이다.
        R = np.array([[np.cos(yaw), 0.0, np.sin(yaw)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(yaw), 0.0, np.cos(yaw)]])
        o_l = pos + R @ np.array([-baseline / 2.0, 0.0, 0.0])
        o_r = pos + R @ np.array([+baseline / 2.0, 0.0, 0.0])
        cv2.imwrite(str(ldir / f"{k:06d}.png"),
                    render_equirect(src_w, src_h, o_l, room, ss=ss, R_world=R))
        cv2.imwrite(str(rdir / f"{k:06d}.png"),
                    render_equirect(src_w, src_h, o_r, room, ss=ss, R_world=R))
        # yaw=0, pitch=0 으로 자르므로 크롭 카메라의 축은 리그 축과 같다.
        poses.append((k / fps, R, o_l))
        pos = pos + R @ np.array([0.0, 0.0, step_m])

    out = work / "E_out"
    r = run_tool(exe, ["--in", ldir, "--out", out, "--right", rdir,
                       "--baseline", baseline, "--min-depth", min_depth,
                       "--yaw", 0, "--pitch", 0, "--hfov", hfov,
                       "--width", W, "--height", H, "--fps", fps])

    # 진리값은 우리가 안다. 도구는 일부러 안 만들므로 (만들 근거가 없다)
    # 여기서 써 넣는다 - 채점기가 요구하는 파일이다.
    from wme.reference.geometry import matrix_to_quat
    with open(out / "groundtruth.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for (t, R, p) in poses:
            q = matrix_to_quat(R)
            f.write(f"{t:.6f} {p[0]:.7f} {p[1]:.7f} {p[2]:.7f} "
                    f"{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n")

    est = out / "est.txt"
    env = dict(os.environ)
    if OPENCV_BIN.is_dir():
        env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    od = subprocess.run([str(odom_exe), str(out), str(est)], capture_output=True,
                        text=True, encoding="utf-8", errors="replace", env=env)
    ev = subprocess.run([sys.executable, str(ROOT / "python" / "tools" / "tum_eval.py"),
                         str(out), str(est)], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", cwd=str(ROOT / "python"))

    path_len = float(sum(np.linalg.norm(poses[i + 1][2] - poses[i][2])
                         for i in range(len(poses) - 1)))
    return {
        "n_frames": n_frames,
        "path_length_m": path_len,
        "convert_stdout_tail": r.stdout.strip().splitlines()[-4:],
        "odometry_rc": od.returncode,
        "odometry_stdout_tail": od.stdout.strip().splitlines()[-12:],
        "odometry_stderr_tail": od.stderr.strip().splitlines()[-6:],
        "eval_rc": ev.returncode,
        "eval_stdout": ev.stdout.strip().splitlines(),
        "eval_stderr_tail": ev.stderr.strip().splitlines()[-6:],
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(DEFAULT_EXE))
    ap.add_argument("--out", default=str(ROOT / "results" / "equirect" / "validate.json"))
    ap.add_argument("--quick", action="store_true", help="해상도를 낮춰 빠르게")
    ap.add_argument("--keep", default="", help="중간 산출물을 남길 폴더")
    ap.add_argument("--e2e", action="store_true",
                    help="변환 산출물을 wme_tum_odometry 에 실제로 먹여 본다 (느리다)")
    args = ap.parse_args()

    exe = Path(args.exe)
    if not exe.is_file():
        print(f"실행 파일이 없다: {exe}\n"
              f"  build_msvc.bat --build build/win --target wme_equirect_convert", file=sys.stderr)
        return 1

    src_w, src_h = (2048, 1024) if args.quick else (4096, 2048)
    view_wh = (480, 360) if args.quick else (640, 480)
    ss = 1 if args.quick else 2
    room_w, room_h = (2048, 1024) if args.quick else (3072, 1536)

    tmp = tempfile.TemporaryDirectory(prefix="equirect_val_")
    work = Path(args.keep) if args.keep else Path(tmp.name)
    work.mkdir(parents=True, exist_ok=True)

    report = {"src": [src_w, src_h], "view": list(view_wh), "quick": args.quick}
    t0 = time.time()

    print(f"[A] 오라클 차등 (C++ 대 numpy)  원본 {src_w}x{src_h}")
    # 잘라내기가 원본의 어느 부분을 보든 같아야 하므로, 방 장면의 잡음 텍스처를
    # 원본으로 쓴다. 균일한 그림에서는 어떤 map 오차도 화소로 안 나타난다.
    src = render_equirect(src_w, src_h, [0, 0, 0], NoiseRoom([6.0, 2.5, 8.0]), ss=1)
    cases = [(0, 0, 90, *view_wh), (37.5, -12.0, 70, *view_wh),
             (180, 0, 90, *view_wh), (-90, 35.0, 110, *view_wh),
             (179.5, 0, 60, *view_wh)]   # 이음매(경도 +-180) 를 일부러 걸친다
    report["A_oracle"] = test_oracle(exe, work, src, cases)
    for r in report["A_oracle"]:
        print(f"    yaw={r['yaw']:>7} pitch={r['pitch']:>6} hfov={r['hfov']:>4}  "
              f"화소 최대차 {r['px_max_abs']:>3}  RMS {r['px_rms']:.4f}  "
              f">1 비율 {100*r['px_gt1_ratio']:.4f}%  K오차 {r['K_max_abs_err']:.2e}")

    print(f"[B/C] 체커보드 재투영 / 내부파라미터 회수  뷰 {view_wh[0]}x{view_wh[1]}")
    board_rows, calib_row = test_board(exe, work, src_w, src_h, view_wh, 70.0, ss)
    report["B_reprojection"] = board_rows
    report["C_calibration"] = calib_row
    for r in board_rows:
        if "error" in r:
            print(f"    board {r['board']}: {r['error']}")
        else:
            print(f"    board {r['board']}: n={r['n_corners']}  RMS {r['rms_px']:.4f} px  "
                  f"max {r['max_px']:.4f} px  bias ({r['bias_px'][0]:+.4f},"
                  f"{r['bias_px'][1]:+.4f})  직접렌더 RMS "
                  f"{r['direct_rms_px'] if r['direct_rms_px'] is None else format(r['direct_rms_px'], '.4f')} px")
    ok_rows = [r for r in board_rows if "rms_px" in r]
    if ok_rows:
        allr = float(np.sqrt(np.mean([r["rms_px"] ** 2 for r in ok_rows])))
        report["B_overall_rms_px"] = allr
        drows = [r["direct_rms_px"] for r in ok_rows if r["direct_rms_px"] is not None]
        alld = float(np.sqrt(np.mean(np.square(drows)))) if drows else None
        report["B_overall_direct_rms_px"] = alld
        # 두 값을 나란히 찍는다. 대조군보다 크지 않으면 워프가 더한 기하 오차가
        # 검출기 잡음 아래라는 뜻이고, 그게 이 시험이 답하려던 질문이다.
        print(f"    전체 재투영 RMS: {allr:.4f} px  "
              f"(대조군 = 핀홀 직접 렌더: "
              f"{'n/a' if alld is None else format(alld, '.4f')} px)")
    if "error" in calib_row:
        print(f"    {calib_row['error']}")
    else:
        for key in ("cropped", "direct"):
            c = calib_row[key]
            print(f"    [{c['which']}] {c['n_views']} 뷰, 재투영 {c['reproj_rms_px']:.4f} px\n"
                  f"        fx {c['fx'][0]:9.4f} (참 {c['fx'][1]:.4f}, {c['rel_err_fx_pct']:.3f}%)  "
                  f"fy {c['fy'][0]:9.4f} ({c['rel_err_fy_pct']:.3f}%)  "
                  f"cx {c['cx'][0]:8.4f} ({c['err_cx_px']:.3f} px)  "
                  f"cy {c['cy'][0]:8.4f} ({c['err_cy_px']:.3f} px)")

    print(f"[D] 360 스테레오 리그  원본 {room_w}x{room_h}")
    report["D_stereo"] = test_stereo(exe, work, room_w, room_h, view_wh, 90.0,
                                     baseline=0.30, min_depth=3.0, ss=ss)
    d = report["D_stereo"]
    print(f"    유효깊이 {100*d['valid_ratio']:.1f}%  정답깊이 중앙 {d['gt_depth_median_m']:.2f} m\n"
          f"    절대오차 중앙 {d['abs_err_median_m']:.3f} m  p95 {d['abs_err_p95_m']:.3f} m  "
          f"RMS {d['rms_err_m']:.3f} m  편향 {d['bias_m']:+.3f} m\n"
          f"    상대오차 중앙 {d['rel_err_median_pct']:.2f}%  "
          f"(이론 sigma {d['sigma_theory_at_median_m']:.3f} m)\n"
          f"    정렬 가능한 yaw 한계 {d['rectifiable_yaw_limit_deg']:.3f}도 "
          f"(그 지점 세로시차 {d['dv_at_limit_px']:.3f} px)  "
          f"경계 판정 일치: {d['limit_boundary_ok']}  yaw 90도 거부: {d['yaw90_rejected']}")

    if args.e2e:
        odom = exe.parent / "wme_tum_odometry.exe"
        if not odom.is_file():
            print(f"    wme_tum_odometry.exe 가 없다: {odom} - E 단계를 건너뛴다")
            report["E_e2e"] = {"error": f"러너 없음: {odom}"}
        else:
            print(f"[E] 파이프라인 종단 (wme_tum_odometry)  원본 {room_w}x{room_h}")
            e = test_e2e(exe, odom, work, room_w, room_h, view_wh, 90.0,
                         baseline=0.30, min_depth=3.0, ss=ss,
                         n_frames=8 if args.quick else 16)
            report["E_e2e"] = e
            print(f"    경로 길이 {e['path_length_m']:.3f} m, 러너 종료코드 {e['odometry_rc']}")
            for ln in e["odometry_stdout_tail"]:
                print(f"      | {ln}")
            for ln in e["eval_stdout"]:
                print(f"      # {ln}")
            for ln in e["eval_stderr_tail"]:
                print(f"      ! {ln}")

    report["elapsed_s"] = time.time() - t0
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과: {outp}  ({report['elapsed_s']:.1f} s)")
    if args.keep:
        print(f"중간 산출물: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
