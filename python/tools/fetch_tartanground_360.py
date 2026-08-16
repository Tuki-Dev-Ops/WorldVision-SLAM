# TartanGround 의 지상 차량 주행 한 토막을 이 저장소의 TUM 배치로 만든다.
#
# **왜 또 하나 만드는가.** `fetch_tartanair_360.py` 가 가져오던 TartanAir V2 는
# 등장방형을 그대로 배포해서 편했지만, 그 궤적은 드론/보행이지 주행이 아니다.
# 실내 집 한 채를 도는 시퀀스를 "360 주행" 자리에 올릴 수는 없어서 내렸다.
# TartanGround 는 같은 환경들 위를 **바퀴 달린 지상 로봇** 이 달린 것이고
# (`Data_diff` = 차동 구동, 제자리 회전과 직진만 한다 - 자동차와 같은 구속),
# `Downtown` 은 도심 가로다. 그래서 궤적이 주행이다.
#
# **대신 등장방형이 없다.** TartanGround 가 배포하는 것은 큐브맵 여섯 면
# (front/back/left/right/top/bottom, 각 640x640 90도) 이다. 여섯 면은 구 전체를
# 덮으므로 360 리그 자체는 갖춰져 있고, 등장방형은 여기서 **합성** 한다.
# 합성 규약은 `python/wme/reference/equirect.py` 의 것을 그대로 쓴다 - 그 모듈이
# `tools/equirect_convert.cpp` 의 numpy 오라클이므로 규약이 갈라질 자리가 없다.
#
# **합성이 맞다는 것을 어떻게 아는가 (이 스크립트가 매번 재는 것).**
#   1. 합성한 등장방형에서 yaw 0/90/180/270 로 90도 뷰를 다시 잘라내면
#      데이터셋이 준 front/right/back/left 면과 같아야 한다. 면의 방향을
#      하나라도 잘못 놓으면 여기서 바로 갈린다.
#   2. 같은 방식으로 자른 깊이가 데이터셋의 면 깊이(DepthPlanar)와 맞아야 한다.
#      반경거리<->광축거리 변환이 빠지면 화면 가장자리에서 벌어진다.
#   두 값을 실행할 때마다 찍고, 어긋나면 **멈춘다**. 조용히 틀린 파노라마를
#   내보내는 것이 이 스크립트에서 가장 나쁜 실패다.
#
# **왜 전체를 안 받는가.** 면 하나의 zip 이 0.8~2.1 GB 이고 우리가 쓰는 것은
# 수십 프레임이다. HuggingFace 가 `Accept-Ranges: bytes` 를 주므로 zip 끝의
# 중앙 디렉터리만 읽어 목록을 얻고 필요한 멤버 구간만 다시 요청한다. 범위
# 요청이 거절되면 조용히 전체를 받는 대신 **실패한다** (HttpFile 이 그렇게 짜여
# 있다 - fetch_tartanair_360.py 참조).
#
# **정직하게 남길 한계.**
#   - 합성 데이터다. 도심은 UE 로 지은 것이고 로봇은 시뮬레이터가 몬다.
#   - 자동차가 아니라 지상 로봇이다. 다만 구속(지면 위, 차동 구동, 카메라 높이
#     약 1.15 m 고정)은 차량 주행과 같은 종류다.
#   - 뷰어에 넣는 것은 구면 전체가 아니라 정면 한 방향을 잘라낸 원근 뷰다.
#     파이프라인(StereoDepth -> DirectAligner)이 핀홀을 가정하기 때문이고,
#     그 이유는 wme/reference/equirect.py 머리말에 적혀 있다. 다만 화각을
#     110 도로 두어 정면 면 하나로는 못 채우게 했다 - 좌/우 면이 실제로
#     들어와야 이 뷰가 만들어진다.
#
# 쓰는 법:
#   python python/tools/fetch_tartanground_360.py --out data/tartanground_downtown_360
#   (--env / --data / --traj / --start / --frames / --stride 로 다른 구간)

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wme.reference.equirect import (WRAP_PAD, build_maps, derive_intrinsics,
                                    direction_from_equirect)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 범위 요청 파일 객체는 TartanAir 쪽 것을 그대로 쓴다. 같은 HuggingFace 이고,
# 두 벌을 두면 한쪽만 고쳐지는 날이 온다.
from fetch_tartanair_360 import HttpFile  # noqa: E402

BASE = "https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/main/"

# PNG 값 = m * DEPTH_SCALE. uint16 상한 65535 이므로 표현 한계가 655.35 m 다.
# 이 장면의 실측 최대가 462 m 라 도심 가로의 끝까지 들어간다. TartanAir 쪽에서
# 쓰던 1000.0 은 65.5 m 에서 잘려 도심에 맞지 않는다.
DEPTH_SCALE = 100.0

# 큐브 면 여섯 개의 회전. 열이 그 면의 축을 파노라마 좌표계
# (X 오른쪽, Y 아래, Z 앞) 로 적은 것이다. 면 좌표계도 같은 규약이라
# d_face = R^T d 로 옮긴다.
#
# 이 값들은 **가정이다.** 맞는지는 verify() 가 데이터셋이 준 면과 견주어
# 판정한다 - 그래서 여기에 적어 두는 것이 안전하다.
FACES = {
    "front":  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    "right":  np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
    "back":   np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float),
    "left":   np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
    "top":    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
    "bottom": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
}
FACE_N = 640          # 면 한 변의 화소 수
FACE_F = FACE_N / 2.0   # 90 도이므로 fx = (N/2)/tan(45) = N/2
FACE_C = (FACE_N - 1) / 2.0   # equirect.py 와 같은 경계 기준 규약


# NED -> 카메라 광축계. **월드와 몸체에 같은 회전을 걸어** 옮긴다.
#
# TartanGround(TartanAir 와 같은 규약)의 `pose_lcam_*.txt` 는 NED 다:
# 월드 축이 (북, 동, 아래)고 자세 사원수도 같은 관례의 몸체 축(앞, 오른쪽,
# 아래)으로 적혀 있다. 이 저장소의 나머지는 KITTI 처럼 **카메라 광축계**
# (오른쪽, 아래, 앞)를 전제한다.
#
# **왜 그냥 두면 안 되는가 (실측으로 드러난 두 자리).**
#   1. bench_viewer 는 실외 시퀀스의 "위" 를 -Y 로 놓는다. NED 에서 위는
#      -Z 이므로, 그대로 넣었을 때 점군의 높이가 자차 기준 5~60 m 로 나왔다
#      (108463 복셀 중 104292 개가 8 m 초과). 지면 판정이 무너지고 유니티
#      장면은 건물이 공중에 뜬다. 실패가 아니라 그럴듯한 오답이다.
#   2. RPE 는 몸체 축이 어긋나면 상쇄되지 않는다. 첫 실행에서 baseline 의
#      ATE 가 31.58 cm(=궤적 형상은 맞다)인데 RPE 가 2021 mm 로 나왔다.
#      프레임당 참 이동이 1.4 m 인데 상대오차가 2 m 라는 것은 오도메트리가
#      아니라 좌표계가 틀렸다는 신호다.
#
# 옮기는 방법은 자의적이지 않다. (북, 동, 아래) -> (동, 아래, 북) 은 축의
# 순환 치환이고 그것이 곧 광축계다: 북을 보는 카메라의 오른쪽은 동, 아래는
# 아래, 앞은 북이다. 순환 치환이므로 회전(det = +1)이지 반사가 아니다.
# 월드와 몸체 양쪽에 같은 회전을 걸므로 궤적의 **형상은 바뀌지 않는다** -
# ATE 는 그대로고, 바뀌는 것은 어느 축이 위인가와 상대 자세의 표현뿐이다.
NED_TO_CAM = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0]])


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R: np.ndarray) -> tuple:
    """회전행렬 -> (qx, qy, qz, qw). 대각합이 작을 때를 위해 최대 성분 분기."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = np.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def pose_ned_to_cam(line: str) -> str:
    """`tx ty tz qx qy qz qw` (NED) -> 같은 형식의 광축계."""
    v = [float(s) for s in line.split()]
    P = NED_TO_CAM
    t = P @ np.array(v[:3])
    R = P @ quat_to_R(v[3:7]) @ P.T
    q = R_to_quat(R)
    return (f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")


def decode_depth(buf: bytes) -> np.ndarray:
    """TartanGround 깊이 PNG -> float32 (m, 광축 성분 = DepthPlanar).

    float32 를 H x W x 4 의 8 비트로 무손실 포장한 것이라 바이트를 그대로
    다시 읽는다. TartanAir V2 와 같은 포장이다.
    """
    a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    r = np.ascontiguousarray(a)
    return np.frombuffer(r.tobytes(), dtype=np.float32).reshape(
        a.shape[0], a.shape[1])


class Cube:
    """큐브맵 여섯 면 -> 등장방형. 맵은 한 번만 만들고 프레임마다 재사용한다."""

    def __init__(self, eq_w: int, eq_h: int):
        self.w, self.h = eq_w, eq_h
        vv, uu = np.meshgrid(np.arange(eq_h, dtype=float),
                             np.arange(eq_w, dtype=float), indexing="ij")
        d = direction_from_equirect(uu, vv, eq_w, eq_h)      # (H, W, 3)

        # 면마다 광축 성분을 재고, 가장 큰 면이 그 방향을 맡는다. 큐브맵은
        # 면끼리 겹치지 않으므로 이것이 곧 "어느 면에 있는가" 다.
        names = list(FACES)
        zc = np.stack([d @ FACES[n][:, 2] for n in names], axis=0)
        owner = np.argmax(zc, axis=0)

        self.maps = {}
        for i, n in enumerate(names):
            R = FACES[n]
            x = d @ R[:, 0]
            y = d @ R[:, 1]
            z = d @ R[:, 2]
            m = owner == i
            zz = np.where(m, z, 1.0)                 # 0 나눗셈 회피
            mx = (FACE_F * (x / zz) + FACE_C).astype(np.float32)
            my = (FACE_F * (y / zz) + FACE_C).astype(np.float32)
            # 반경거리 = 광축깊이 / cos(면 축과 이루는 각). d 가 단위벡터라
            # cos 가 곧 z 다. 이 항이 빠지면 면 가장자리가 가까워진다.
            self.maps[n] = (mx, my, m, np.where(m, z, 1.0).astype(np.float32))
        self.owner = owner

        # **구멍이 없다는 것을 여기서 확인한다.** argmax 는 언제나 답을 주므로
        # 그 값이 있다는 사실은 아무 것도 보증하지 않는다. 물어야 하는 것은
        # 고른 면의 광축 성분이 실제로 양수인가(= 그 방향이 그 면 앞에 있는가)와
        # 여섯 면이 모두 쓰였는가다. 하나라도 어긋나면 파노라마에 빈 자리가
        # 생기고, 그것은 검은 화소로 조용히 남는다.
        zbest = np.take_along_axis(zc, owner[None], axis=0)[0]
        if not np.all(zbest > 0.0):
            raise SystemExit("등장방형에 어느 면에도 속하지 않는 방향이 있다")
        used = np.unique(owner)
        if used.size != len(names):
            miss = [names[i] for i in range(len(names)) if i not in used]
            raise SystemExit(f"쓰이지 않은 면이 있다: {miss}")

    def stitch(self, faces: dict, interp: int, dtype) -> np.ndarray:
        shp = ((self.h, self.w, 3) if faces["front"].ndim == 3
               else (self.h, self.w))
        out = np.zeros(shp, dtype)
        for n, (mx, my, m, _z) in self.maps.items():
            w = cv2.remap(faces[n], mx, my, interp)
            out[m] = w[m]
        return out

    def stitch_depth(self, faces: dict) -> np.ndarray:
        """면의 DepthPlanar 를 등장방형의 **반경 거리** 로 옮긴다."""
        out = np.zeros((self.h, self.w), np.float32)
        for n, (mx, my, m, z) in self.maps.items():
            w = cv2.remap(faces[n], mx, my, cv2.INTER_NEAREST)
            out[m] = (w / z)[m]
        return out


def open_zip(rel: str):
    f = HttpFile(BASE + rel)
    return zipfile.ZipFile(io.BufferedReader(f, 256 * 1024)), f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="Downtown")
    ap.add_argument("--data", default="Data_diff",
                    help="Data_diff = 차동 구동 바퀴 로봇 (자동차와 같은 구속)")
    ap.add_argument("--traj", default="P1000")
    ap.add_argument("--start", type=int, default=440,
                    help="이 구간이 멈춤 없이 계속 움직인다 (실측: 최소 0.37 m/프레임)")
    ap.add_argument("--frames", type=int, default=118)
    # **stride 1 이다 — 원본이 찍힌 그대로다.** 처음에는 3 이었고, 이유는
    # "프레임당 1.4 m 라 KITTI 의 1.5 m 와 같은 자리" 였다. 그 기준이 틀렸다.
    # 직접 광도 정합이 감당해야 하는 것은 미터가 아니라 **화소 변위** 이고,
    # 그것은 장면 깊이와 프레임당 회전이 정한다. 이 크롭은 KITTI 보다 훨씬
    # 가깝고(중앙 깊이 3~6 m vs kitti_00 10.4 m) 훨씬 빨리 돈다:
    #
    #   프레임당 회전   kitti_04 0.22 deg / kitti_00 1.49 deg / 여기 stride3 3.38 deg
    #   회전 유도 광류  kitti_04 1.7 px  / kitti_00 6.6 px  / 여기 stride3 20.1 px
    #
    # 그래서 미터를 맞췄더니 화소가 3 배로 벌어졌고, 앞 열한 프레임에서 정합이
    # 참 이동의 4~70 % 밖에 못 잡았다(추정 0.05 m vs 참 1.32 m). 실측 비교는
    # 같은 구간(원본 440~476)에서 **끼워 넣은 두 장 말고는 완전히 같은 자료**로
    # 했다 - 겹치는 12 자리의 화소·포즈·시각이 비트까지 같다:
    #
    #   stride 3 (1.42 m/프레임)  ATE 348.84 cm   inlier 0.12   측광 RMSE 44.7
    #   stride 1 (0.475 m/프레임) ATE  10.50 cm   inlier 0.50   측광 RMSE 15.9
    #
    # 벤치 인자(--kf-dist 1.0 --depth-max 60)로는 6.82 cm 이고, 같은 자료에서
    # ORB+PnP 가 4.96 cm 다. 즉 이 시퀀스는 어렵지 않았고, 세 배 성기게
    # 뽑았을 뿐이다. stride 를 3 으로 되돌리려면 위 숫자부터 다시 재라.
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.1,
                    help="metadata.json 의 time_step. 추측이 아니라 데이터셋 값이다")
    ap.add_argument("--hfov", type=float, default=110.0)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--eq-width", type=int, default=2048)
    ap.add_argument("--poses-only", action="store_true",
                    help="groundtruth.txt 만 다시 쓴다. 영상/깊이는 이미 받아 둔 "
                         "것을 그대로 두므로 좌표계 규약을 고칠 때 241 MB 를 "
                         "다시 받지 않아도 된다")
    ap.add_argument("--tol-rgb", type=float, default=3.0,
                    help="합성 검증: 데이터셋 면과의 중앙값 화소차 상한 (LSB)")
    ap.add_argument("--tol-depth", type=float, default=0.05,
                    help="합성 검증: 면 깊이와의 중앙값 상대차 상한")
    a = ap.parse_args()

    base = f"{a.env}/{a.data}/{a.traj}"
    os.makedirs(f"{a.out}/rgb", exist_ok=True)
    os.makedirs(f"{a.out}/depth", exist_ok=True)

    # 포즈부터 연다. `--poses-only` 면 여기서 끝나므로 (면 zip 열두 개의 중앙
    # 디렉터리를 읽는 데만 몇 분이 걸린다) 순서가 이래야 한다.
    zm, fm = open_zip(f"{base}/metadata.zip")
    handles = [fm]

    poses = zm.read("pose_lcam_front.txt").decode().strip().splitlines()
    idxs = [a.start + i * a.stride for i in range(a.frames)]
    if idxs[-1] >= len(poses):
        raise SystemExit(
            f"{idxs[-1]} 번 프레임을 요구했는데 궤적은 {len(poses)} 개뿐이다")

    def write_gt() -> None:
        # TartanGround 포즈는 `tx ty tz qx qy qz qw` 라 컬럼 순서가 TUM 과 같다.
        # 시각만 앞에 붙이고, 좌표계는 NED 에서 카메라 광축계로 옮긴다 (왜
        # 옮겨야 하는지는 NED_TO_CAM 머리말에 실측과 함께 적어 두었다).
        # 잘라낸 뷰가 lcam_front 와 같은 방향(yaw 0)이라 pose_lcam_front 가
        # 그대로 이 뷰의 포즈다.
        with open(f"{a.out}/groundtruth.txt", "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("# 좌표계: 카메라 광축계 (X 오른쪽, Y 아래, Z 앞).\n")
            f.write("# 원본은 NED 이고 NED_TO_CAM 으로 옮겼다 - 월드와 몸체에\n")
            f.write("# 같은 순환 치환을 걸었으므로 궤적 형상은 원본과 같다.\n")
            for i in idxs:
                f.write(f"{i * a.dt:.7f} {pose_ned_to_cam(poses[i])}\n")

    if a.poses_only:
        write_gt()
        print(f"groundtruth.txt 만 다시 썼다: {a.out}  ({len(idxs)} 프레임)")
        print(f"# 받은 바이트: {fm.nread / 1024 / 1024:.1f} MB (metadata.zip 일부)")
        return 0

    zips = {}
    if not a.poses_only:
        for n in FACES:
            zi, fi = open_zip(f"{base}/image_lcam_{n}.zip")
            zd, fd = open_zip(f"{base}/depth_lcam_{n}.zip")
            zips[n] = (zi, zd)
            handles += [fi, fd]

    eq_h = a.eq_width // 2
    cube = Cube(a.eq_width, eq_h)
    view = derive_intrinsics(a.width, a.height, a.hfov, 0.0, 0.0)
    mx, my = build_maps(view, a.eq_width, eq_h)
    uu, vv = np.meshgrid(np.arange(float(a.width)), np.arange(float(a.height)))
    xr = (uu - view.cx) / view.fx
    yr = (vv - view.cy) / view.fy
    # 등장방형 깊이는 광선을 따라 잰 반경 거리다. 핀홀 깊이는 광축 성분이므로
    # cos 를 곱해 옮긴다. 빠지면 화면 가장자리가 멀어진다.
    cos_t = (1.0 / np.sqrt(xr * xr + yr * yr + 1.0)).astype(np.float32)

    def warp(img, interp):
        pad = np.concatenate(
            [img[:, -WRAP_PAD:], img, img[:, :WRAP_PAD]], axis=1)
        return cv2.remap(pad, mx + WRAP_PAD, my, interp)

    def read_faces(i: int):
        rgb, dep = {}, {}
        for n, (zi, zd) in zips.items():
            rgb[n] = cv2.imdecode(np.frombuffer(zi.read(
                f"image_lcam_{n}/{i:06d}_lcam_{n}.png"), np.uint8),
                cv2.IMREAD_COLOR)
            dep[n] = decode_depth(zd.read(
                f"depth_lcam_{n}/{i:06d}_lcam_{n}_depth.png"))
        return rgb, dep

    # --------------------------------------------------------------- 검증
    def verify(rgb_f: dict, dep_f: dict, eq_rgb, eq_dep) -> dict:
        """합성한 파노라마를 데이터셋이 준 면과 견준다.

        면 하나의 방향을 잘못 놓으면 그 면이 통째로 엉뚱한 자리에 붙는데,
        완성된 파노라마만 보고는 잘 보이지 않는다. 그래서 되잘라 본다.
        """
        v90 = {}
        for n, yaw in (("front", 0.0), ("right", 90.0),
                       ("back", 180.0), ("left", -90.0)):
            vw = derive_intrinsics(FACE_N, FACE_N, 90.0, yaw, 0.0)
            bx, by = build_maps(vw, a.eq_width, eq_h)

            def cut(img, interp):
                pad = np.concatenate(
                    [img[:, -WRAP_PAD:], img, img[:, :WRAP_PAD]], axis=1)
                return cv2.remap(pad, bx + WRAP_PAD, by, interp)

            got = cut(eq_rgb, cv2.INTER_CUBIC).astype(np.int16)
            ref = rgb_f[n].astype(np.int16)
            # 가장자리 4 화소는 이웃 면과의 보간이 섞이는 자리라 뺀다.
            d = np.abs(got - ref)[4:-4, 4:-4]
            med, p99 = float(np.median(d)), float(np.percentile(d, 99))

            gz = cut(eq_dep, cv2.INTER_NEAREST)
            # 반경 -> 이 면의 광축 성분
            fx2 = (uu2 - vw.cx) / vw.fx
            fy2 = (vv2 - vw.cy) / vw.fy
            gz = gz / np.sqrt(fx2 * fx2 + fy2 * fy2 + 1.0)
            rz = dep_f[n]
            ok = np.isfinite(rz) & (rz > 0) & (rz < 200)
            ok[:4] = ok[-4:] = False
            ok[:, :4] = ok[:, -4:] = False
            rel = float(np.median(np.abs(gz[ok] - rz[ok]) / rz[ok]))
            v90[n] = (med, p99, rel)
            print(f"  검증 {n:6s}: 화소차 중앙 {med:.2f} p99 {p99:.1f} LSB, "
                  f"깊이 상대차 중앙 {rel * 100:.2f} %")
            if med > a.tol_rgb or rel > a.tol_depth:
                raise SystemExit(
                    f"{n} 면의 합성이 데이터셋과 맞지 않는다 "
                    f"(화소 {med:.2f} > {a.tol_rgb} 또는 "
                    f"깊이 {rel:.4f} > {a.tol_depth}). FACES 의 회전을 의심하라.")

        # 위/아래 면은 위 네 개의 90 도 뷰가 건드리지 않는다 (그 뷰들은
        # 적도 띠만 본다). 그래서 **이음매의 연속성** 으로 따로 본다: 소유가
        # 옆면에서 위/아래로 바뀌는 줄의 세로 기울기가, 그 둘레 20 줄의
        # 기울기와 같은 크기여야 한다. 면이 돌아가 있으면 여기서 줄이 선다.
        g = cv2.cvtColor(eq_rgb, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dv = np.abs(np.diff(g, axis=0))
        names = list(FACES)
        for pole in ("top", "bottom"):
            pi = names.index(pole)
            rows = np.where((cube.owner[:-1] == pi) != (cube.owner[1:] == pi))[0]
            if rows.size == 0:
                raise SystemExit(f"{pole} 면이 파노라마에 한 화소도 없다")
            seam = float(np.median(dv[rows, :]))
            near = np.unique(np.clip(
                np.concatenate([rows + k for k in range(-10, 11) if k]),
                0, dv.shape[0] - 1))
            near = np.setdiff1d(near, rows)
            base = float(np.median(dv[near, :]))
            print(f"  이음매 {pole:6s}: 세로기울기 {seam:.2f} vs 둘레 {base:.2f} "
                  f"(비 {seam / max(base, 1e-6):.2f})")
            if seam > 4.0 * max(base, 1.0):
                raise SystemExit(
                    f"{pole} 이음매에 줄이 선다 ({seam:.2f} vs {base:.2f}). "
                    "FACES 의 회전을 의심하라.")
        return v90

    uu2, vv2 = np.meshgrid(np.arange(float(FACE_N)), np.arange(float(FACE_N)))

    rgb_idx, dep_idx, lo_all, hi_all = [], [], [], []
    for k, i in enumerate(idxs):
        rgb_f, dep_f = read_faces(i)
        eq_rgb = cube.stitch(rgb_f, cv2.INTER_LINEAR, np.uint8)
        eq_dep = cube.stitch_depth(dep_f)
        if k == 0:
            print(f"# 합성 검증 (프레임 {i}, 등장방형 {a.eq_width}x{eq_h})")
            verify(rgb_f, dep_f, eq_rgb, eq_dep)
            cv2.imwrite(f"{a.out}/equirect_{i:06d}.png", eq_rgb)

        rgb = warp(eq_rgb, cv2.INTER_CUBIC)
        # 깊이는 절대 섞지 않는다. 앞뒤 경계에서 평균은 어느 물체도 아닌 값이다.
        dep = warp(eq_dep, cv2.INTER_NEAREST) * cos_t

        valid = np.isfinite(dep) & (dep > 0)
        if valid.any():
            lo_all.append(np.percentile(dep[valid], 1))
            hi_all.append(np.percentile(dep[valid], 99))
        q = np.zeros(dep.shape, np.uint16)
        ok = valid & (dep * DEPTH_SCALE < 65535)
        q[ok] = (dep[ok] * DEPTH_SCALE).astype(np.uint16)   # 나머지 0 = 결측

        t = i * a.dt
        cv2.imwrite(f"{a.out}/rgb/{k:06d}.png", rgb)
        cv2.imwrite(f"{a.out}/depth/{k:06d}.png", q)
        rgb_idx.append(f"{t:.6f} rgb/{k:06d}.png")
        dep_idx.append(f"{t:.6f} depth/{k:06d}.png")
        if k % 5 == 0:
            mb = sum(h.nread for h in handles) / 1024 / 1024
            print(f"  {k}/{len(idxs)}  (원본 {i})  누적 {mb:.0f} MB", flush=True)

    with open(f"{a.out}/rgb.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp filename\n" + "\n".join(rgb_idx) + "\n")
    with open(f"{a.out}/depth.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp filename\n" + "\n".join(dep_idx) + "\n")

    write_gt()

    lo, hi = float(np.min(lo_all)), float(np.max(hi_all))
    with open(f"{a.out}/calib.txt", "w", encoding="utf-8") as f:
        f.write(
            f"# wme: TartanGround {base} 큐브맵 6 면 -> 등장방형 "
            f"{a.eq_width}x{eq_h} -> 정면 {a.hfov:.0f} 도 원근 뷰.\n"
            "# 깊이는 데이터셋의 **렌더 GT**(DepthPlanar) 이지 추정값이 아니다.\n"
            "# 내부파라미터는 hfov 와 출력 크기에서 유도한 값이다.\n"
            f"fx: {view.fx}\nfy: {view.fy}\n"
            f"cx: {view.cx}\ncy: {view.cy}\n"
            f"width: {view.width}\nheight: {view.height}\n"
            f"depth_scale: {DEPTH_SCALE}\n"
            f"# 이 {len(idxs)} 프레임의 실측 범위는 p1 최소 {lo:.2f} m, "
            f"p99 최대 {hi:.2f} m 다. 아래 상한은 그것과 별개로, 도심 원거리\n"
            "# 화소가 광도 정합에 보태는 것이 없어 KITTI 와 같은 자리에 둔 값이다.\n"
            "depth_min: 0.5\ndepth_max: 80.0\n"
            "dist: 0 0 0 0 0\n")

    mb = sum(h.nread for h in handles) / 1024 / 1024
    gb = sum(h.size for h in handles) / 1e9
    print(f"\n# 범위 요청으로 받은 바이트: {mb:.1f} MB (zip 전체는 {gb:.2f} GB)")
    print(f"# 실측 깊이 범위: p1 최소 {lo:.2f} m, p99 최대 {hi:.2f} m")
    print(f"완료: {a.out}  ({len(idxs)} 프레임, 원본 {idxs[0]}~{idxs[-1]})\n")
    name = f"tartanground_{a.env.lower()}_360"
    print("# results/bench/viewer.tsv 에 넣을 줄 (탭 구분):")
    print(f"SEQ\t{name}\ttartanground\t{os.path.abspath(a.out)}\t"
          f"{os.path.abspath(a.out)}/groundtruth.txt\t0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
