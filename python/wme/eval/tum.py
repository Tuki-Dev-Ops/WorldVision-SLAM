"""TUM RGB-D 데이터셋 로더.

WME 의 첫 실측 마일스톤이 이 데이터셋이다. 특히 dynamic 시퀀스
(walking_xyz, walking_halfsphere)가 중요하다 - 토큰 기반 동적 마스킹이
고전 직접법 대비 이득을 내는지를 여기서 처음 확인할 수 있다.

파일 형식:
  groundtruth.txt : timestamp tx ty tz qx qy qz qw
  rgb.txt / depth.txt : timestamp filename
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..reference.geometry import SE3, quat_to_matrix
from .trajectory import Trajectory


def _read_rows(path: Path) -> list[list[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def load_trajectory(path: str | Path) -> Trajectory:
    """groundtruth.txt 또는 같은 형식의 추정 결과를 읽는다."""
    rows = _read_rows(Path(path))
    stamps, poses = [], []
    for r in rows:
        if len(r) < 8:
            continue
        t = float(r[0])
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in r[1:8])
        stamps.append(t)
        poses.append(SE3(quat_to_matrix(qx, qy, qz, qw), np.array([tx, ty, tz])))
    if not poses:
        raise ValueError(f"포즈를 읽지 못함: {path}")
    return Trajectory(np.array(stamps), poses).sorted()


def save_trajectory(path: str | Path, traj: Trajectory) -> None:
    """TUM 형식으로 저장. 기존 평가 도구와 호환되게 하려면 이 형식을 지켜야 한다."""
    from ..reference.geometry import matrix_to_quat

    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p in zip(traj.stamps, traj.poses):
            q = matrix_to_quat(p.R)
            f.write(f"{t:.6f} {p.t[0]:.6f} {p.t[1]:.6f} {p.t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


@dataclass
class FrameEntry:
    stamp: float
    rgb_path: Path
    depth_path: Path | None
    pose: SE3 | None


# freiburg 그룹별 내부 파라미터와 왜곡 계수.
# 하나로 고정하면 다른 그룹에서 조용히 틀린 결과가 나온다 - 실패가 아니라
# 그럴듯한 오차로. fr3 은 TUM 이 보정해 배포하므로 계수가 0 이다.
INTRINSICS: dict[str, tuple[float, float, float, float, tuple[float, ...]]] = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989,
                  (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764,
                  (0.2312, -0.7849, -0.0033, -0.0001, 0.9172)),
    "freiburg3": (535.4, 539.2, 320.1, 247.6,
                  (0.0, 0.0, 0.0, 0.0, 0.0)),
}


def intrinsics_for(name: str) -> tuple[float, float, float, float, tuple[float, ...]]:
    """시퀀스 이름에서 그룹을 골라 내부 파라미터를 준다."""
    for group, values in INTRINSICS.items():
        if group in name:
            return values
    return INTRINSICS["freiburg1"]


@dataclass
class TumSequence:
    """RGB / depth / groundtruth 를 시각으로 정렬한 시퀀스."""
    root: Path
    frames: list[FrameEntry]
    fx: float = 517.306408
    fy: float = 516.469215
    cx: float = 318.643040
    cy: float = 255.313989
    distortion: tuple[float, ...] = (0.2624, -0.9531, -0.0054, 0.0026, 1.1633)
    depth_scale: float = 5000.0     # PNG 값 -> 미터

    def __len__(self) -> int:
        return len(self.frames)

    def trajectory(self) -> Trajectory:
        entries = [f for f in self.frames if f.pose is not None]
        return Trajectory(np.array([f.stamp for f in entries]),
                          [f.pose for f in entries])


def load_sequence(root: str | Path, max_difference: float = 0.02,
                  require_files: bool = True) -> TumSequence:
    """rgb / depth / groundtruth 를 시각 기준으로 짝지어 로드한다.

    depth 나 groundtruth 가 없어도 동작한다 (추론 전용 실행).

    require_files=True 면 실제로 존재하는 파일만 남긴다. 인덱스 txt 는 전체
    시퀀스를 나열하므로, 일부만 받아 둔 경우 없는 파일을 가리키는 항목이
    그대로 섞여 들어온다. 그 상태로 돌리면 로드 실패가 프레임 건너뜀으로
    조용히 바뀌어, 실제로는 재생되지 않은 구간까지 평가에 포함된다.
    """
    root = Path(root)
    rgb_rows = _read_rows(root / "rgb.txt")
    if not rgb_rows:
        raise ValueError(f"rgb.txt 를 읽지 못함: {root}")

    depth_rows = _read_rows(root / "depth.txt") if (root / "depth.txt").exists() else []
    gt = load_trajectory(root / "groundtruth.txt") if (root / "groundtruth.txt").exists() else None

    depth_stamps = np.array([float(r[0]) for r in depth_rows]) if depth_rows else np.empty(0)

    frames: list[FrameEntry] = []
    for r in rgb_rows:
        stamp = float(r[0])

        rgb_path = root / r[1]
        if require_files and not rgb_path.exists():
            continue

        depth_path = None
        if depth_stamps.size:
            j = int(np.argmin(np.abs(depth_stamps - stamp)))
            if abs(depth_stamps[j] - stamp) <= max_difference:
                depth_path = root / depth_rows[j][1]
                if require_files and not depth_path.exists():
                    depth_path = None

        pose = None
        if gt is not None:
            k = int(np.argmin(np.abs(gt.stamps - stamp)))
            if abs(gt.stamps[k] - stamp) <= max_difference:
                pose = gt.poses[k]

        frames.append(FrameEntry(stamp, rgb_path, depth_path, pose))

    if not frames:
        raise ValueError(f"사용 가능한 프레임이 없음: {root}")

    fx, fy, cx, cy, dist = intrinsics_for(root.name)
    return TumSequence(root=root, frames=frames,
                       fx=fx, fy=fy, cx=cx, cy=cy, distortion=dist)
