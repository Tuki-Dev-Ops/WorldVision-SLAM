#!/usr/bin/env python3
"""노면 격자의 도로/건물 판정을 **분할 라벨에 대고** 잰다.

왜 이게 필요한가
----------------
뷰어는 노면 격자 칸마다 클래스를 붙이고(`RoadCell::seg`), 그 클래스는 거의
전부 SegFormer 화소 라벨에서 온다 - `wme_bench_viewer` 가 스스로 "판정 근거:
화소 90~92 %" 라고 찍는다. 그런데 그 숫자는 **어디서 왔는가만 말하고 맞는가는
말하지 않는다.** 화면에서 도로 옆이 통째로 건물색으로 덮여도 이 지표는 그대로
92 % 다.

그래서 여기서는 판정을 두 갈래로 갈라 잰다:

  (1) **격자 vs 분할.** 화소가 road 라고 말한 그 화소가 역투영되어 들어간
      칸이 결국 무슨 클래스가 되었는가. 기준이 분할 자신이므로 여기서 나오는
      불일치는 분할의 잘못이 아니다 - 깊이·포즈·셀 집계가 만든 것이다.
      **이 값이 곧 분할→격자 투영 단계의 오차다.**

  (2) **분할 vs 기하.** 노면 평면 ±band 안에 들어온 점의 화소 라벨이 building
      인 비율. 건물은 노면 높이에 없다. 여기서 나오는 것은 (분할 오분류) +
      (깊이 오차로 다른 데서 굴러온 점) 이고, 둘은 **영상에서 도로-건물 경계
      까지의 거리** 로 갈린다. 경계에 붙어 있으면 1/4 해상도 분할이 흘린
      것이고, 건물 영역 한복판인데 노면 높이에 앉았으면 깊이가 틀린 것이다.

정직하게 말해 둘 것
-------------------
KITTI odometry 에는 의미 분할 **정답이 없다.** 그러므로 이 스크립트는
SegFormer 의 절대 정확도를 재지 못한다. 재는 것은 두 가지뿐이다 -
격자가 분할을 얼마나 충실히 옮기는가(위 1번, 기준이 확실하다), 그리고
분할이 기하와 얼마나 어긋나는가(위 2번, 어느 쪽이 틀렸는지는 경계 거리와
관측 거리로 나눠 추정한다). 정답이 필요한 값은 "미확인" 으로 남는다.

이 지표가 판별하는가 (--control 로 실측, kitti_00 150 프레임)
--------------------------------------------------------------
"모든 입력에서 같은 값이 나오는 지표는 통과해도 아무 것도 증명하지 않는다."
그래서 입력을 일부러 흔들어 두 지표가 각각 어디에 반응하는지 실측했다:

  대조군          3) 일치   road IoU  bldg IoU   6a) building 노면통과율
  없음             83.5%     82.8%     47.2%          3.4%
  shift 4px        83.3%     82.6%     47.5%          3.4%
  swap             83.3%     47.2%     82.6%            -
  randcell         45.4%     47.7%      1.3%          3.4%
  allbuilding     100.0%       nan     100.0%         35.6%

읽는 법. 3) 은 randcell 에 무너지고(83.5 -> 45.4) swap 에 클래스가 정확히
뒤집히므로 **투영·집계를 판별한다.** 대신 allbuilding 에서 100 % 가 된다 -
통째로 틀린 분할은 못 잡는다. 6a) 는 정확히 그 반대다: randcell 에 꿈쩍
않지만 allbuilding 에서 3.4 -> 35.6 % 로 열 배가 된다. **둘 중 하나만
보면 안 되고, 둘을 같이 봐야 덮인다.** shift 4 px 에는 둘 다 거의 반응하지
않으므로, 이 측정은 **작은 정합 어긋남을 잡아내지 못한다** - 그 한계는
남는다.

사용:
  python python/tools/road_seg_audit.py
  python python/tools/road_seg_audit.py --seq kitti_04 --max-frames 60
  python python/tools/road_seg_audit.py --poses est --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import cv2  # noqa: E402

# --- 뷰어(tools/bench_viewer.cpp)에서 그대로 가져온 상수 ---------------------
# 값이 어긋나면 여기서 재는 것이 화면에 나오는 것과 다른 물건이 된다.
KROAD_CELL = 0.10        # kRoadCell
ROAD_BAND = 0.40         # |rel - floor_rel| <= 0.40 이 노면
H_LO, H_HI = -3.0, 9.5   # kitti 의 hlo / hhi (자차 높이 기준)
PIXEL_STRIDE = 3         # has_memory 인 kitti 패널의 stride
SKY_CLASS = 10           # 하늘은 backProject 에서 통째로 버린다
FLOOR_REL0 = -1.65       # 바닥 초기 가정
WORLD_UP = np.array([0.0, -1.0, 0.0])   # kitti: 정답 포즈가 카메라 좌표계

# Cityscapes 19 클래스 이름 (분할 산출물의 화소값)
CS_NAME = ["road", "sidewalk", "building", "wall", "fence", "pole",
           "light", "sign", "vegetation", "terrain", "sky", "person",
           "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
ROAD, BUILDING = 0, 2

# 경계 거리 구간 (분할 화소 단위 = 원본의 4 배). 0~1 은 경계에 붙은 화소다.
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]
BAND_NAME = ["<1px", "1-2px", "2-4px", "4-8px", ">=8px"]
# 관측 거리 구간 (m). 스테레오 깊이 오차는 z^2 로 자라므로 로그에 가깝게 나눈다.
RBINS = [3.2, 8.0, 15.0, 25.0, 40.0, 80.0]
# 경계까지의 거리를 **미터로** 환산한 구간. 화소로 재면 같은 화소 수가 가까이
# 서는 몇 cm 고 멀리서는 몇 m 라, 거리별로 쪼갤 때 원인이 뒤섞인다.
HBINS = [0.0, 0.2, 0.5, 1.0, 2.0, 1e9]
HBIN_NAME = ["<0.2m", "0.2-0.5m", "0.5-1m", "1-2m", ">2m"]


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
def read_index(path: Path):
    """rgb.txt / depth.txt 를 (stamp, 경로) 목록으로 읽는다."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, p = line.split()
        out.append((float(t), path.parent / p))
    return out


def read_calib(path: Path) -> dict:
    cal = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            cal[k.strip()] = float(v.split()[0])
        except (ValueError, IndexError):
            pass
    return cal


def read_poses(path: Path):
    """TUM 포즈를 (stamp, 4x4 T_world_cam) 목록으로."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = [float(x) for x in line.split()]
        if len(f) < 8:
            continue
        t, tx, ty, tz, qx, qy, qz, qw = f[:8]
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if n < 1e-12:
            continue
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (tx, ty, tz)
        out.append((t, T))
    return out


def nearest_pose(poses, stamp, tol=0.25):
    """뷰어와 같은 규칙: 0.25 s 안에서 가장 가까운 포즈."""
    best, bd = None, tol
    for t, T in poses:
        d = abs(t - stamp)
        if d < bd:
            bd, best = d, T
    return best


# ---------------------------------------------------------------------------
# 노면 격자 (뷰어의 RoadCell 을 그대로 옮긴 것)
# ---------------------------------------------------------------------------
def road_axes(up: np.ndarray):
    """뷰어의 road_a / road_b. up 하나로 결정된다."""
    u = up / np.linalg.norm(up)
    a = np.cross(u, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(u, np.array([1.0, 0.0, 0.0]))
    a = a / np.linalg.norm(a)
    b = np.cross(u, a)
    return a, b / np.linalg.norm(b)


def vote_seg_stream(order, labels, n_cells):
    """뷰어의 `RoadCell::voteSeg` 를 그대로 재현한다.

    **이것은 다수결이 아니다.** Boyer-Moore 과반 후보 알고리즘이라
    과반이 없으면 아무 값이나 남을 수 있고, 관측 **순서** 에 따라 답이 바뀐다.
    그 성질 자체가 측정 대상이므로 진짜 최빈값과 함께 둘 다 낸다.

    order: 셀 색인(관측 순서대로), labels: 같은 길이의 클래스 배열
    """
    seg = np.full(n_cells, 255, dtype=np.uint8)
    cnt = np.zeros(n_cells, dtype=np.int32)
    for ci, lb in zip(order.tolist(), labels.tolist()):
        if lb > 18:
            continue
        if seg[ci] > 18 or cnt[ci] == 0:
            seg[ci] = lb
            cnt[ci] = 1
        elif seg[ci] == lb:
            if cnt[ci] < 255:
                cnt[ci] += 1
        else:
            cnt[ci] -= 1
    return seg


def boundary_distance(seg_img: np.ndarray, a: int, b: int) -> np.ndarray:
    """분할 영상에서 클래스 a 영역과 b 영역의 **경계까지의 거리** (화소).

    경계 화소는 "a 이면서 b 와 4-이웃" 이거나 그 반대인 화소다. 경계가 아예
    없는 프레임에서는 거리를 무한대로 둔다 - 그런 프레임을 0 으로 채우면
    경계 밴드가 통째로 오염된다.
    """
    ma = (seg_img == a).astype(np.uint8)
    mb = (seg_img == b).astype(np.uint8)
    if ma.sum() == 0 or mb.sum() == 0:
        return np.full(seg_img.shape, 1e9, dtype=np.float32)
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
    edge = ((ma > 0) & (cv2.filter2D(mb, -1, k) > 0)) | \
           ((mb > 0) & (cv2.filter2D(ma, -1, k) > 0))
    if not edge.any():
        return np.full(seg_img.shape, 1e9, dtype=np.float32)
    src = np.where(edge, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(src, cv2.DIST_L2, 3).astype(np.float32)


def run_base_rows(mask: np.ndarray) -> np.ndarray:
    """화소마다 **자기가 속한 세로 연속 구간의 맨 아래 행** 을 돌려준다.

    왜 경계 거리로는 안 되는가: 도시 도로에서 road 영역과 building 영역은
    대개 **맞닿아 있지 않다.** 사이에 인도가 있고 차가 서 있고 담장이 있다.
    그러면 road-building 경계는 저 멀리 길 끝에만 생기고, 벽 밑동 화소가
    "경계에서 30 화소" 로 찍힌다 - 벽면 높이를 재려던 값이 엉뚱한 것을 잰다.

    자기 열에서 자기 덩어리의 바닥까지가 곧 벽면 높이다. 앞을 차가 가리면
    그만큼 **과소** 평가되므로, 여기서 나오는 높이는 하한이다.
    """
    H = mask.shape[0]
    base = np.zeros(mask.shape, np.int32)
    base[H - 1] = H - 1
    for v in range(H - 2, -1, -1):
        base[v] = np.where(mask[v] & mask[v + 1], base[v + 1], v)
    return base


# ---------------------------------------------------------------------------
# 한 시퀀스 훑기
# ---------------------------------------------------------------------------
def scan_sequence(seq_dir: Path, seg_dir: Path, poses, args):
    """프레임을 돌며 노면 격자를 쌓고, 관측마다 필요한 값을 기록한다.

    돌려주는 것은 **관측 단위 배열** 이다. 셀의 최종 클래스는 전부 쌓은
    뒤에야 정해지므로, 관측을 먼저 모아 두고 나중에 대조한다.
    """
    cal = read_calib(seq_dir / "calib.txt")
    fx, fy = cal["fx"], cal["fy"]
    cx, cy = cal["cx"], cal["cy"]
    dscale, dmin, dmax = cal["depth_scale"], cal["depth_min"], cal["depth_max"]

    rgb = read_index(seq_dir / "rgb.txt")
    dep = read_index(seq_dir / "depth.txt")
    dep_t = np.array([t for t, _ in dep])

    up = WORLD_UP
    ax_a, ax_b = road_axes(up)
    inv = 1.0 / KROAD_CELL

    # 관측 기록 (노면 밴드를 통과한 점만)
    keys, labs, bands, rngs, azs = [], [], [], [], []
    # 밴드를 통과하지 못한 점은 개수만 - 개수 자체가 투영 오차의 크기다.
    n_offered = n_taken = 0
    # 화소 라벨 분포: 전체 / 깊이 유효 / 노면 밴드 통과
    hist_all = np.zeros(20, np.int64)
    hist_valid = np.zeros(20, np.int64)
    hist_band = np.zeros(20, np.int64)
    # road 라벨 화소의 노면 대비 높이 잔차 (거리 구간별). 투영 오차의 직접 측정.
    res_sum = np.zeros(len(RBINS) - 1)
    res_sq = np.zeros(len(RBINS) - 1)
    res_n = np.zeros(len(RBINS) - 1, np.int64)
    res_in = np.zeros(len(RBINS) - 1, np.int64)   # 그 중 밴드 안에 든 수
    # building 라벨 화소: 전체 / 노면 밴드에 든 것, 경계 밴드 x 거리 구간별.
    #
    # **거리를 함께 쪼개야 원인이 갈린다.** 경계에서 먼 건물 화소가 노면
    # 높이에 앉는 비율이 거리와 무관하면 그것은 벽 밑동(분할이 맞다)이고,
    # 거리에 따라 치솟으면 깊이 오차(투영이 틀렸다)다. 둘은 같은 표에서
    # 같은 모양으로 보이므로 나누지 않으면 영영 안 갈린다.
    nrb = len(RBINS) - 1
    b_all = np.zeros((len(BANDS), nrb), np.int64)
    b_inband = np.zeros((len(BANDS), nrb), np.int64)
    # 같은 것을 **미터 환산** 경계거리로 다시. 도로-건물 경계선은 곧 벽 밑동
    # 이므로, 거기서 h m 떨어진 화소는 벽면 h m 높이여야 한다. 그런 화소가
    # 노면 높이(±0.40 m)에 앉았다면 h 가 클수록 물리적으로 불가능해진다.
    h_all = np.zeros((len(HBINS) - 1, nrb), np.int64)
    h_inband = np.zeros((len(HBINS) - 1, nrb), np.int64)

    n_frames = n_seg_missing = 0
    rng = np.random.default_rng(20260812)

    for fi, (stamp, rgb_path) in enumerate(rgb):
        if args.max_frames and n_frames >= args.max_frames:
            break
        if fi % args.frame_stride:
            continue
        di = int(np.argmin(np.abs(dep_t - stamp)))
        if abs(dep_t[di] - stamp) > 0.05:
            continue
        T = nearest_pose(poses, stamp)
        if T is None:
            continue
        d16 = cv2.imread(str(dep[di][1]), cv2.IMREAD_UNCHANGED)
        if d16 is None or d16.dtype != np.uint16:
            continue
        seg_img = cv2.imread(str(seg_dir / (rgb_path.stem + ".png")),
                             cv2.IMREAD_GRAYSCALE)
        if seg_img is None:
            n_seg_missing += 1
            continue

        # --- 대조군. 지표가 판별력이 있는지 보려고 입력을 일부러 흔든다. ---
        if args.control == "shift":
            seg_img = np.roll(seg_img, args.control_shift, axis=1)
        elif args.control == "swap":
            s2 = seg_img.copy()
            s2[seg_img == ROAD] = BUILDING
            s2[seg_img == BUILDING] = ROAD
            seg_img = s2
        elif args.control == "allbuilding":
            seg_img = np.where(seg_img == SKY_CLASS, seg_img,
                               np.uint8(BUILDING)).astype(np.uint8)

        H, W = d16.shape
        sh, sw = seg_img.shape
        bdist = boundary_distance(seg_img, ROAD, BUILDING)
        bbase = run_base_rows(seg_img == BUILDING)

        vs = np.arange(0, H, PIXEL_STRIDE)
        us = np.arange(0, W, PIXEL_STRIDE)
        uu, vv = np.meshgrid(us, vs)
        z = d16[vv, uu].astype(np.float64) / dscale
        # 분할 좌표는 뷰어와 같은 정수 나눗셈으로 뽑는다 (보간하지 않는다).
        su = (uu * sw) // W
        sv = (vv * sh) // H
        sc = seg_img[sv, su]
        bd = bdist[sv, su]
        # 자기 건물 덩어리의 바닥까지 몇 행 위인가 (분할 화소 단위)
        bh = (bbase[sv, su] - sv).astype(np.float32)

        hist_all += np.bincount(sc.ravel(), minlength=256)[:20]

        ok = (z > dmin) & (z <= dmax) & (sc != SKY_CLASS)
        if not ok.any():
            n_frames += 1
            continue
        z = z[ok]; sc = sc[ok]; bd = bd[ok]; bh = bh[ok]
        uu2 = uu[ok].astype(np.float64); vv2 = vv[ok].astype(np.float64)

        x = (uu2 - cx) * z / fx
        y = (vv2 - cy) * z / fy
        pc = np.stack([x, y, z], 1)
        pw = pc @ T[:3, :3].T + T[:3, 3]

        ego_h = float(T[:3, 3] @ up)
        rel = pw @ up - ego_h
        keep = (rel >= H_LO) & (rel <= H_HI)
        z = z[keep]; sc = sc[keep]; bd = bd[keep]; bh = bh[keep]
        pw = pw[keep]; rel = rel[keep]
        uu2 = uu2[keep]
        if z.size == 0:
            n_frames += 1
            continue
        hist_valid += np.bincount(sc, minlength=256)[:20]

        # --- 바닥 높이 (뷰어와 같은 하위 20 % 규칙) ---
        floor_rel = FLOOR_REL0
        lows = rel[(rel > floor_rel - 4.0) & (rel < floor_rel + 6.0)]
        if lows.size > 200:
            q = lows.size // 5
            floor_rel = float(np.partition(lows, q)[q])

        dh = rel - floor_rel
        inband = np.abs(dh) <= ROAD_BAND
        n_offered += z.size
        n_taken += int(inband.sum())
        hist_band += np.bincount(sc[inband], minlength=256)[:20]

        # road 라벨 화소의 잔차 - 라벨이 road 면 그 점은 노면에 있어야 한다.
        rmask = sc == ROAD
        if rmask.any():
            rb = np.digitize(z[rmask], RBINS) - 1
            rb = np.clip(rb, 0, len(RBINS) - 2)
            d_ = dh[rmask]
            np.add.at(res_sum, rb, d_)
            np.add.at(res_sq, rb, d_ * d_)
            np.add.at(res_n, rb, 1)
            np.add.at(res_in, rb[np.abs(d_) <= ROAD_BAND],
                      np.ones(int((np.abs(d_) <= ROAD_BAND).sum()), np.int64))

        # building 라벨 화소의 경계 거리 분포 - 전체와 "노면 높이에 앉은 것"
        bmask = sc == BUILDING
        if bmask.any():
            bb = np.digitize(bd[bmask], [b[1] for b in BANDS[:-1]])
            rr = np.clip(np.digitize(z[bmask], RBINS) - 1, 0, nrb - 1)
            np.add.at(b_all, (bb, rr), 1)
            sel = inband[bmask]
            if sel.any():
                np.add.at(b_inband, (bb[sel], rr[sel]), 1)
            # 분할 화소 하나가 원본 4 화소이므로 4 를 곱하고, 그 화소각을
            # 관측 거리로 곱하면 미터가 된다. 경계선이 대체로 수평(벽 밑동)
            # 이라는 가정 위의 근사다 - 건물 옆모서리에서는 과대평가된다.
            hm = bh[bmask] * 4.0 * z[bmask] / fy
            hh = np.clip(np.digitize(hm, HBINS[1:-1]), 0, len(HBINS) - 2)
            np.add.at(h_all, (hh, rr), 1)
            if sel.any():
                np.add.at(h_inband, (hh[sel], rr[sel]), 1)

        if inband.any():
            p = pw[inband]
            qi = np.floor((p @ ax_a) * inv).astype(np.int64)
            qj = np.floor((p @ ax_b) * inv).astype(np.int64)
            keys.append((qi << 32) | (qj & 0xFFFFFFFF))
            labs.append(sc[inband].astype(np.uint8))
            bandi = np.digitize(bd[inband], [b[1] for b in BANDS[:-1]])
            bands.append(bandi.astype(np.uint8))
            rngs.append(z[inband].astype(np.float32))
            azs.append((uu2[inband] - cx).astype(np.float32))

        n_frames += 1
        if args.verbose and n_frames % 100 == 0:
            print(f"    {n_frames} 프레임  누적 관측 "
                  f"{sum(k.size for k in keys)}", flush=True)

    if not keys:
        return None

    out = dict(
        keys=np.concatenate(keys), labs=np.concatenate(labs),
        bands=np.concatenate(bands), rngs=np.concatenate(rngs),
        azs=np.concatenate(azs),
        n_offered=n_offered, n_taken=n_taken, n_frames=n_frames,
        n_seg_missing=n_seg_missing,
        hist_all=hist_all, hist_valid=hist_valid, hist_band=hist_band,
        res_sum=res_sum, res_sq=res_sq, res_n=res_n, res_in=res_in,
        b_all=b_all, b_inband=b_inband, h_all=h_all, h_inband=h_inband,
        rng=rng,
    )
    return out


# ---------------------------------------------------------------------------
# 대조와 보고
# ---------------------------------------------------------------------------
def cell_classes(rec, args):
    """관측을 셀로 접어 최종 클래스를 만든다. 뷰어 규칙과 진짜 최빈값 둘 다.

    **관측 하나를 뺀 클래스도 함께 낸다(leave-one-out).** 칸의 클래스는 그
    칸에 들어온 관측들의 투표이므로, 관측을 그대로 그 투표 결과와 견주면
    한 번만 본 칸은 무조건 일치한다 - 자기 자신과 비교하는 셈이다. 그 값은
    아무 것도 증명하지 못한다. 자기 표를 빼고 나머지가 뭐라고 했는지를 물어야
    "이 화소의 라벨이 이웃 관측들과 맞는가" 라는 답이 나온다.
    """
    uniq, idx = np.unique(rec["keys"], return_inverse=True)
    n = uniq.size
    lab = rec["labs"]
    # 진짜 최빈값
    cnt = np.zeros((n, 20), np.int32)
    np.add.at(cnt, (idx, np.clip(lab, 0, 19)), 1)
    mode = cnt.argmax(1).astype(np.uint8)
    # 뷰어의 voteSeg (순서 의존). 느리므로 필요할 때만.
    vote = vote_seg_stream(idx, lab, n) if not args.no_vote else mode
    if args.control == "randcell":
        # 칸의 표 전체를 칸끼리 뒤섞는다. 지표가 우연 수준으로 내려가야 한다.
        #
        # **vote 만 섞으면 안 된다.** leave-one-out 판정은 vote 가 아니라 표
        # 자체(cnt)에서 다시 나오므로, 섞어야 할 것을 안 섞으면 대조군이
        # 아무 것도 흔들지 못한 채 통과한다 - 실제로 처음에 그렇게 나왔다.
        perm = rec["rng"].permutation(n)
        cnt = cnt[perm]
        vote = vote[perm]
        mode = cnt.argmax(1).astype(np.uint8)
    # leave-one-out 최빈값. 자기 표를 뺀 뒤의 argmax 이고, 뺐더니 표가 하나도
    # 남지 않으면(그 칸을 한 번만 봤으면) 255 로 두고 통계에서 제외한다.
    #
    # 관측마다 20 칸짜리 표를 복사하면 500 만 관측에서 400 MB 다. 그럴 필요가
    # 없다 - 자기 표를 하나 빼서 답이 바뀔 수 있는 경우는 **자기가 1 등이었고
    # 2 등과 한 표 차 이내일 때** 뿐이므로, 칸마다 1·2 등만 알면 된다.
    o = np.argsort(-cnt, axis=1, kind="stable")
    t1, t2 = o[:, 0].astype(np.uint8), o[:, 1].astype(np.uint8)
    c1 = cnt[np.arange(n), t1]
    c2 = cnt[np.arange(n), t2]
    ci = idx
    my = np.clip(lab, 0, 19)
    lo = t1[ci].copy()
    was_top = my == lo
    # 1 등에서 한 표 빠졌다. 2 등에게 밀리면(동점이면 클래스 번호가 작은 쪽)
    # 답이 바뀐다.
    flip = was_top & ((c1[ci] - 1 < c2[ci]) |
                      ((c1[ci] - 1 == c2[ci]) & (t2[ci] < t1[ci])))
    lo[flip] = t2[ci][flip]
    lo[cnt.sum(1)[ci] < 2] = 255
    return idx, n, vote, mode, cnt, lo


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    i = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return p, r, i


def confusion(pix, cell):
    """화소 라벨(기준) x 셀 판정. road / building / 그 외 3 x 3."""
    def code(v):
        c = np.full(v.shape, 2, np.int8)
        c[v == ROAD] = 0
        c[v == BUILDING] = 1
        return c
    a, b = code(pix), code(cell)
    m = np.zeros((3, 3), np.int64)
    np.add.at(m, (a, b), 1)
    return m


def report_sequence(name, rec, args, out_json):
    idx, ncell, vote, mode, cnt, lo = cell_classes(rec, args)
    lab = rec["labs"]
    naive = vote[idx]             # 그 관측이 들어간 칸의 최종 클래스 (자기 표 포함)
    # 대조는 leave-one-out 으로 한다. 자기 표가 든 판정과 견주면 한 번만 본
    # 칸이 전부 자동 정답이 되어 지표가 부풀려진다.
    ok = lo != 255
    lab, cell_lab = lab[ok], lo[ok]
    for k in ("bands", "rngs", "azs"):
        rec[k] = rec[k][ok]

    print(f"\n{'=' * 78}")
    print(f"{name}   프레임 {rec['n_frames']}  노면 칸 {ncell}  "
          f"노면 관측 {int(ok.size)} (2 회 이상 본 칸의 관측 {lab.size} 만 대조)")
    if rec["n_seg_missing"]:
        print(f"  분할 PNG 없는 프레임 {rec['n_seg_missing']} 개 - 건너뛰었다")

    # --- 1) 영상 수준 분할 라벨 분포 ---
    hv = rec["hist_valid"]
    tot = hv.sum()
    print("\n  1) 분할 라벨 분포 (깊이 유효 + 높이 문턱 통과 화소 기준)")
    order = np.argsort(-hv)[:6]
    print("     " + "  ".join(
        f"{CS_NAME[c]} {100 * hv[c] / max(1, tot):.1f}%" for c in order))
    print(f"     road {100 * hv[ROAD] / max(1, tot):.1f}%   "
          f"building {100 * hv[BUILDING] / max(1, tot):.1f}%   "
          f"building/road 비 {hv[BUILDING] / max(1, hv[ROAD]):.2f}")
    # 문턱을 하나도 안 건 날것의 분포도 함께 낸다. 깊이 유효성·높이 문턱은
    # 클래스마다 다르게 걸리므로, 걸러진 분포만 보고 "건물이 많다" 고 하면
    # 그것이 분할 탓인지 문턱 탓인지 알 수 없다.
    ha = rec["hist_all"]
    ta = ha.sum()
    print(f"     문턱 전 날것:  road {100 * ha[ROAD] / max(1, ta):.1f}%   "
          f"building {100 * ha[BUILDING] / max(1, ta):.1f}%   "
          f"sky {100 * ha[SKY_CLASS] / max(1, ta):.1f}%   "
          f"building/road 비 {ha[BUILDING] / max(1, ha[ROAD]):.2f}")

    # --- 2) 노면 문턱 통과율과 road 라벨 화소의 높이 잔차 ---
    print("\n  2) 투영 오차의 크기 - road 라벨 화소가 노면에 앉는가")
    print(f"     노면 격자 제시 {rec['n_offered']} 통과 {rec['n_taken']} "
          f"({100 * rec['n_taken'] / max(1, rec['n_offered']):.0f} %)")
    print(f"     {'거리':>10s} {'n':>9s} {'평균잔차':>9s} {'표준편차':>9s} "
          f"{'±0.40m 안':>9s}")
    for i in range(len(RBINS) - 1):
        if rec["res_n"][i] == 0:
            continue
        m = rec["res_sum"][i] / rec["res_n"][i]
        v = max(0.0, rec["res_sq"][i] / rec["res_n"][i] - m * m)
        print(f"     {RBINS[i]:4.0f}-{RBINS[i+1]:3.0f}m {rec['res_n'][i]:9d} "
              f"{m:8.3f}m {math.sqrt(v):8.3f}m "
              f"{100 * rec['res_in'][i] / rec['res_n'][i]:8.1f}%")
    print("     road 라벨 화소가 노면 밴드 밖으로 나가는 만큼이 곧 깊이·포즈"
          " 오차다.")

    # --- 3) 혼동행렬: 분할 라벨(기준) vs 격자 셀 판정 ---
    m = confusion(lab, cell_lab)
    print("\n  3) 혼동행렬  행=화소 분할 라벨(기준), 열=노면 격자 셀 판정")
    print(f"     {'':>10s} {'셀=road':>12s} {'셀=building':>12s} {'셀=기타':>12s}")
    for i, nm in enumerate(("화소 road", "화소 bldg", "화소 기타")):
        print(f"     {nm:>10s} {m[i,0]:12d} {m[i,1]:12d} {m[i,2]:12d}")
    pr = prf(m[0, 0], m[1, 0] + m[2, 0], m[0, 1] + m[0, 2])
    pb = prf(m[1, 1], m[0, 1] + m[2, 1], m[1, 0] + m[1, 2])
    print(f"     road      정밀도 {pr[0]*100:5.1f}%  재현율 {pr[1]*100:5.1f}%  "
          f"IoU {pr[2]*100:5.1f}%")
    print(f"     building  정밀도 {pb[0]*100:5.1f}%  재현율 {pb[1]*100:5.1f}%  "
          f"IoU {pb[2]*100:5.1f}%")
    agree = float((lab == cell_lab).mean())
    naive_agree = float((rec["labs"][ok] == naive[ok]).mean())
    print(f"     전체 일치 {agree*100:.1f}%  "
          f"(자기 표를 넣고 재면 {naive_agree*100:.1f}% - 그 값은 부풀려진 것이다)")
    print("     기준이 분할 자신이므로 이 불일치는 전부 투영·집계가 만든 것이다.")

    # --- 4) 경계 밴드별 ---
    print("\n  4) 도로-건물 경계까지의 거리별 (분할 화소 단위, 원본의 4 배)")
    print(f"     {'밴드':>7s} {'관측 n':>10s} {'일치':>7s} {'road IoU':>9s} "
          f"{'bldg IoU':>9s}")
    band_rows = []
    for bi, bn in enumerate(BAND_NAME):
        s = rec["bands"] == bi
        if s.sum() < 100:
            continue
        mm = confusion(lab[s], cell_lab[s])
        r_ = prf(mm[0, 0], mm[1, 0] + mm[2, 0], mm[0, 1] + mm[0, 2])
        b_ = prf(mm[1, 1], mm[0, 1] + mm[2, 1], mm[1, 0] + mm[1, 2])
        ag = float((lab[s] == cell_lab[s]).mean())
        print(f"     {bn:>7s} {int(s.sum()):10d} {ag*100:6.1f}% "
              f"{r_[2]*100:8.1f}% {b_[2]*100:8.1f}%")
        band_rows.append(dict(band=bn, n=int(s.sum()), agree=ag,
                              road_iou=r_[2], bldg_iou=b_[2]))

    # --- 5) 거리 / 방위각 분해 ---
    print("\n  5) 관측 거리 · 방위각별 일치율")
    print(f"     {'거리':>10s} {'n':>10s} {'일치':>7s}   "
          f"{'|u-cx|':>10s} {'n':>10s} {'일치':>7s}")
    az = np.abs(rec["azs"])
    azb = [0, 150, 300, 450, 1e9]
    rows = []
    for i in range(len(RBINS) - 1):
        s = (rec["rngs"] >= RBINS[i]) & (rec["rngs"] < RBINS[i + 1])
        s2 = (az >= azb[min(i, 3)]) & (az < azb[min(i, 3) + 1])
        c1 = f"{RBINS[i]:4.0f}-{RBINS[i+1]:3.0f}m {int(s.sum()):10d} " \
             f"{100*float((lab[s]==cell_lab[s]).mean()) if s.any() else 0:6.1f}%"
        c2 = ""
        if i < 4:
            c2 = f"{azb[i]:5.0f}-{azb[i+1] if azb[i+1]<1e8 else 999:4.0f}px " \
                 f"{int(s2.sum()):10d} " \
                 f"{100*float((lab[s2]==cell_lab[s2]).mean()) if s2.any() else 0:6.1f}%"
        print(f"     {c1}   {c2}")
        if s.any():
            rows.append(dict(rmin=RBINS[i], rmax=RBINS[i + 1], n=int(s.sum()),
                             agree=float((lab[s] == cell_lab[s]).mean())))

    # --- 6a) 클래스별 노면 밴드 침입률 ---
    #
    # 노면 격자는 "노면 평면 ±0.40 m" 라는 기하 조건 하나로 점을 받는다.
    # 그러니 각 클래스가 그 조건을 얼마나 통과하는지가 곧 **분할 라벨과 기하가
    # 얼마나 어긋나는가** 다. road 는 거의 다 통과해야 맞고, building 은 벽
    # 밑동 말고는 통과할 이유가 없다. 이 표가 "건물 과다" 를 직접 판정한다.
    hb, hvv = rec["hist_band"], rec["hist_valid"]
    print("\n  6a) 클래스별 노면 평면 ±0.40 m 통과율 - 기하가 분할을 심판한다")
    print(f"      {'클래스':>12s} {'화소 n':>11s} {'노면밴드 n':>11s} "
          f"{'통과율':>8s} {'노면격자 점유':>12s}")
    band_tot = hb.sum()
    for c in np.argsort(-hb)[:9]:
        if hb[c] == 0:
            continue
        print(f"      {CS_NAME[c]:>12s} {hvv[c]:11d} {hb[c]:11d} "
              f"{100*hb[c]/max(1,hvv[c]):7.1f}% {100*hb[c]/max(1,band_tot):11.1f}%")

    # --- 6) 원인 분리: 노면 높이에 앉은 building 화소는 어디서 오는가 ---
    print("\n  6b) 원인 분리 - 노면 평면에 앉은 building 화소")
    ba, bi_ = rec["b_all"].sum(1), rec["b_inband"].sum(1)
    print(f"     {'밴드':>7s} {'building 화소':>13s} {'그중 노면밴드':>13s} "
          f"{'침입률':>8s} {'전체분포':>9s} {'침입분포':>9s} {'농축':>7s}")
    enrich = []
    for i, bn in enumerate(BAND_NAME):
        if ba[i] == 0:
            continue
        sa = ba[i] / max(1, ba.sum())
        sb = bi_[i] / max(1, bi_.sum())
        e = sb / sa if sa > 0 else float("nan")
        print(f"     {bn:>7s} {ba[i]:13d} {bi_[i]:13d} "
              f"{100*bi_[i]/ba[i]:7.1f}% {100*sa:8.1f}% {100*sb:8.1f}% "
              f"{e:6.2f}x")
        enrich.append(dict(band=bn, n=int(ba[i]), inband=int(bi_[i]),
                           rate=float(bi_[i] / ba[i]), enrich=float(e)))
    near = bi_[0] + bi_[1]
    print(f"     경계 2 px 안에서 온 것 {100*near/max(1,bi_.sum()):.1f}%,"
          f" 건물 영역 한복판(>=4 px)에서 온 것 "
          f"{100*(bi_[3]+bi_[4])/max(1,bi_.sum()):.1f}%")
    print("     농축 > 1 이면 그 밴드가 침입을 과대 생산한다는 뜻이다.")

    # --- 6c) 같은 침입을 거리로 다시 쪼갠다. 여기서 원인이 갈린다. ---
    #
    # 벽 밑동은 언제나 노면 높이에 있으므로 침입률이 거리와 무관해야 한다.
    # 깊이 오차는 z^2 로 자라므로 침입률이 거리를 따라 올라간다. 경계에서 먼
    # 화소(>=4 px, 벽면 한복판)만 보면 벽 밑동이 빠지므로 그 기울기가 곧
    # 투영 오차의 몫이다.
    print("\n  6c) 같은 침입을 **미터 환산** 벽면 높이로 - 벽 밑동인가 오류인가")
    HA, HI = rec["h_all"], rec["h_inband"]
    print(f"      {'벽면 높이':>10s} {'building 화소':>13s} {'노면밴드':>11s} "
          f"{'침입률':>8s} {'침입 분포':>10s}")
    tot_i = HI.sum()
    rows6c = []
    for i, nm in enumerate(HBIN_NAME):
        n_, k_ = int(HA[i].sum()), int(HI[i].sum())
        if n_ == 0:
            continue
        print(f"      {nm:>10s} {n_:13d} {k_:11d} {100*k_/n_:7.1f}% "
              f"{100*k_/max(1,tot_i):9.1f}%")
        rows6c.append(dict(h=nm, n=n_, inband=k_, rate=float(k_ / n_)))
    print("     밑동(<0.4 m)에 몰려 있으면 ±0.40 m 밴드가 벽 밑을 삼킨 것이고,"
          " 그것은 설계 문제다.")
    print("     1 m 위인데 노면 높이에 앉은 화소는 물리적으로 불가능하다 -"
          " 깊이나 라벨 하나가 틀린 것이다.")
    print(f"      {'거리':>10s} " + " ".join(f"{n:>9s}" for n in HBIN_NAME))
    for i in range(len(RBINS) - 1):
        cells_ = []
        for hi in range(len(HBIN_NAME)):
            n_ = HA[hi, i]
            cells_.append(f"{100*HI[hi,i]/n_:8.1f}%" if n_ > 200 else f"{'-':>9s}")
        print(f"      {RBINS[i]:4.0f}-{RBINS[i+1]:3.0f}m " + " ".join(cells_))
    print("     한 열(같은 벽면 높이) 안에서 거리를 따라 침입률이 오르면"
          " 깊이 오차, 평평하면 분할 오차다.")

    # --- 7) 집계 규칙 자체의 기여 ---
    diff = int((vote != mode).sum())
    print(f"\n  7) 집계 규칙 - voteSeg(순서 의존) 와 진짜 최빈값이 다른 칸 "
          f"{diff} / {ncell} ({100*diff/max(1,ncell):.1f}%)")
    pure = cnt.max(1) / np.maximum(1, cnt.sum(1))
    seen = cnt.sum(1)
    multi = seen >= 3
    print(f"     3 회 이상 본 칸 {int(multi.sum())} 개의 라벨 순도 "
          f"중앙값 {float(np.median(pure[multi]))*100:.1f}%, "
          f"순도 <60 % 인 칸 {100*float((pure[multi]<0.6).mean()):.1f}%")
    print("     순도가 낮은 칸은 같은 자리를 여러 번 보고도 분할이 매번 다른"
          " 답을 냈다는 뜻이다.")
    # **도로 표와 건물 표를 둘 다 받은 칸.** 격자에서 도로/건물 경계가 실제로
    # 어디인지 다투는 자리가 여기다. 이 칸이 많다면 경계가 선이 아니라 띠다.
    both = (cnt[:, ROAD] > 0) & (cnt[:, BUILDING] > 0)
    contested = int(both.sum())
    has_b = int((cnt[:, BUILDING] > 0).sum())
    print(f"     도로 표와 건물 표를 함께 받은 칸 {contested} 개 "
          f"(건물 표가 있는 칸 {has_b} 개의 "
          f"{100*contested/max(1,has_b):.1f}%, 전체의 "
          f"{100*contested/max(1,ncell):.1f}%)")

    out_json[name] = dict(
        frames=int(rec["n_frames"]), cells=int(ncell), obs=int(lab.size),
        offered=int(rec["n_offered"]), taken=int(rec["n_taken"]),
        seg_share=dict(road=float(hv[ROAD] / max(1, tot)),
                       building=float(hv[BUILDING] / max(1, tot))),
        band_share={CS_NAME[c]: float(hb[c] / max(1, band_tot))
                    for c in range(19) if hb[c]},
        band_pass={CS_NAME[c]: float(hb[c] / max(1, hvv[c]))
                   for c in range(19) if hvv[c]},
        naive_agree=naive_agree,
        confusion=m.tolist(),
        road=dict(precision=pr[0], recall=pr[1], iou=pr[2]),
        building=dict(precision=pb[0], recall=pb[1], iou=pb[2]),
        agree=agree, bands=band_rows, ranges=rows, enrich=enrich,
        intrusion_by_range=rows6c, contested=contested,
        vote_vs_mode=diff,
        purity_median=float(np.median(pure[multi])) if multi.any() else None,
    )
    return agree


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", nargs="*",
                    default=["kitti_00", "kitti_04", "kitti_05", "kitti_07"])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seg", default=str(ROOT / "results" / "seg"))
    ap.add_argument("--poses", choices=("gt", "est"), default="gt",
                    help="gt = data/*/groundtruth.txt, "
                         "est = results/bench/*_wme.txt (정렬 안 함)")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    # 대조군은 지표마다 겨냥하는 곳이 다르다.
    #   shift       분할 영상을 옆으로 민다. 3) 은 기준과 예측이 **같은 영상**
    #               에서 나오므로 원리상 꿈쩍도 하지 않는다 - 흔들려야 하는 것은
    #               6a) 다. 3) 이 shift 에 반응하지 않는 것은 결함이 아니라
    #               그 지표가 분할을 심판하지 않는다는 증거다.
    #   swap        road 와 building 라벨을 맞바꾼다. 3) 의 클래스별 수치가
    #               따라 바뀌어야 한다. 안 바뀌면 클래스를 안 보고 있는 것이다.
    #   randcell    칸의 표를 칸끼리 섞는다. 3) 이 우연 수준으로 떨어져야 한다.
    #   allbuilding 전부 building 으로 만든다. **3) 은 100 % 가 된다** - 일관성
    #               지표는 통째로 틀린 분할을 잡아내지 못한다. 그 한계를 숨기지
    #               않으려고 대조군에 넣어 둔다.
    ap.add_argument("--control", default="none",
                    choices=("none", "shift", "swap", "allbuilding", "randcell"),
                    help="지표의 판별력을 보려고 입력을 흔든다")
    ap.add_argument("--control-shift", type=int, default=4,
                    help="shift 대조군에서 분할 영상을 옆으로 미는 화소 수")
    ap.add_argument("--no-vote", action="store_true",
                    help="voteSeg 재현을 건너뛰고 최빈값만 쓴다 (빠르다)")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("노면 격자 도로/건물 판정 감사")
    print(f"  포즈 {args.poses}  프레임 stride {args.frame_stride}  "
          f"화소 stride {PIXEL_STRIDE}  노면 밴드 ±{ROAD_BAND} m  "
          f"대조군 {args.control}")

    out_json = {}
    for name in args.seq:
        seq_dir = Path(args.data) / name
        seg_dir = Path(args.seg) / name
        if not seq_dir.is_dir():
            print(f"\n{name}: 시퀀스 없음 - 건너뜀")
            continue
        if not seg_dir.is_dir():
            print(f"\n{name}: 분할 산출물 없음 ({seg_dir}) - 건너뜀")
            continue
        pf = (seq_dir / "groundtruth.txt" if args.poses == "gt"
              else ROOT / "results" / "bench" / f"{name}_wme.txt")
        if not pf.exists():
            print(f"\n{name}: 포즈 파일 없음 ({pf}) - 건너뜀")
            continue
        t0 = time.time()
        rec = scan_sequence(seq_dir, seg_dir, read_poses(pf), args)
        if rec is None:
            print(f"\n{name}: 노면 밴드를 통과한 관측이 하나도 없다")
            continue
        report_sequence(name, rec, args, out_json)
        print(f"  ({time.time() - t0:.0f} s)")

    if len(out_json) > 1:
        print(f"\n{'=' * 78}")
        print("요약")
        print(f"  {'시퀀스':10s} {'셀':>9s} {'road IoU':>9s} {'bldg IoU':>9s} "
              f"{'일치':>7s} {'통과율':>7s} {'bldg/road 화소비':>16s}")
        for k, v in out_json.items():
            print(f"  {k:10s} {v['cells']:9d} {v['road']['iou']*100:8.1f}% "
                  f"{v['building']['iou']*100:8.1f}% {v['agree']*100:6.1f}% "
                  f"{100*v['taken']/max(1,v['offered']):6.1f}% "
                  f"{v['seg_share']['building']/max(1e-9,v['seg_share']['road']):15.2f}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(out_json, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
