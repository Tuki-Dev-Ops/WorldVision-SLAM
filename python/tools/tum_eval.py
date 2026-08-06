"""TUM 시퀀스에서 추정 궤적을 채점한다.

추정은 C++ (wme_tum_odometry), 채점은 여기. 같은 코드로 둘 다 하면 두 쪽의
버그가 서로를 가려 준다.

기준선도 함께 낸다:
  - identity  : 카메라가 안 움직였다고 가정. 어떤 추정기든 이보다 나빠선 안 된다.
  - constant  : 진리값의 평균 속도로 직선 이동. 궤적의 "쉬운 부분" 이 얼마인지 보여준다.

사용: python tools/tum_eval.py <시퀀스경로> <추정.txt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate, evaluate_rpe  # noqa: E402
from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402


def _baselines(gt: Trajectory, stamps: np.ndarray) -> dict[str, Trajectory]:
    """추정기가 이겨야 하는 최소선."""
    identity = Trajectory(stamps, [SE3.identity() for _ in stamps])

    # 진리값 시작/끝을 잇는 등속 직선. 회전은 시작 자세로 고정.
    t0, t1 = gt.poses[0].t, gt.poses[-1].t
    s0, s1 = gt.stamps[0], gt.stamps[-1]
    R0 = gt.poses[0].R
    frac = np.clip((stamps - s0) / max(1e-9, s1 - s0), 0.0, 1.0)
    const = Trajectory(stamps, [SE3(R0, t0 + f * (t1 - t0)) for f in frac])
    return {"identity": identity, "constant-velocity": const}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, est_path = sys.argv[1], sys.argv[2]

    est = load_trajectory(est_path)
    gt = load_sequence(root).trajectory()

    path_len = float(np.linalg.norm(np.diff([p.t for p in gt.poses], axis=0), axis=1).sum())
    print(f"추정 {len(est.poses)} 포즈 | 진리 {len(gt.poses)} 포즈 | 진리 경로길이 {path_len:.2f} m")
    print()

    rows: list[tuple[str, float, float, float]] = []
    for name, traj in {"ECDA": est, **_baselines(gt, est.stamps)}.items():
        a = evaluate_ate(traj, gt, align=True)
        r = evaluate_rpe(traj, gt, delta=1.0)
        rows.append((name, a.rmse, r.trans_rmse, r.rot_rmse_deg))

    print(f"{'':20s} {'ATE RMSE':>10s} {'RPE 병진':>12s} {'RPE 회전':>12s}")
    print(f"{'':20s} {'(cm)':>10s} {'(cm/s)':>12s} {'(deg/s)':>12s}")
    for name, ate, rt, rr in rows:
        print(f"{name:20s} {ate*100:10.2f} {rt*100:12.2f} {rr:12.2f}")

    print()
    ecda = rows[0]
    for name, ate, _, _ in rows[1:]:
        verdict = "이김" if ecda[1] < ate else "짐"
        print(f"  vs {name}: {ate/max(ecda[1], 1e-9):.2f}x  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
