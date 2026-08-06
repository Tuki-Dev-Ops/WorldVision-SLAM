"""M4: 포즈를 추정 대상으로 놓은 상태에서 불확실성이 보정되는가.

M3 는 포즈를 참값으로 고정한 조건에서 통과했다. 그 가정을 없애면 랜드마크
오차와 포즈 오차가 결합되므로 훨씬 어렵다.

    python tools/m4_slam.py [--frames 90] [--channel haze]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from wme.calib import CalibratedNoise
from wme.eval.metrics import nees
from wme.graph.slam import SlamConfig, solve_object_slam
from wme.sim import CameraTrajectory, constant_condition, run, static_room


# M3 에서 적합된 값. 참 파라미터(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)에
# 매우 가깝게 복원된 모델이다.
M3_NOISE = CalibratedNoise(c_px=1.996, g_px=3.82, c_d=0.00557, g_d=4.14)


def make_sequence(level: float, frames: int, seed: int, channel: str):
    world = static_room(14, seed=seed)
    traj = CameraTrajectory.loop(radius=3.2, laps=2.0, n=frames)
    return run(world, traj, constant_condition(**{channel: level}),
               seed=seed, name=f"{channel}={level:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--channel", default="haze")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.4, 0.8])
    ap.add_argument("--trials", type=int, default=6,
                    help="조건당 시드 수. 랜드마크 14개짜리 단일 시행으로는 "
                         "보정 불량과 표본 잡음을 구분할 수 없다.")
    args = ap.parse_args()

    print("=" * 78)
    print("M4 게이트: 포즈까지 추정하는 조건에서의 불확실성 보정")
    print("=" * 78)
    print(f"조건당 {args.trials} 시드를 통합해 평가한다. 단일 시행 ANEES 는 "
          f"0.45~2.0 로 흔들려\n결론을 낼 수 없다 - 게이트가 아니라 평가 방법의 문제다.\n")
    print(f"{'level':>6} {'ATE0':>8} {'ATE':>8} {'개선':>7} {'lm_rmse':>8} "
          f"{'n':>4} {'ANEES/d':>9} {'band':>14}  verdict")

    all_ok = True
    for level in args.levels:
        errors_all, covs_all = [], []
        ate0_all, ate_all = [], []

        for seed in range(args.trials):
            seq = make_sequence(level, args.frames, seed, args.channel)
            result = solve_object_slam(seq, M3_NOISE, SlamConfig(seed=seed))
            try:
                e, c = result.landmark_nees_inputs()
            except ValueError:
                continue
            errors_all.append(e)
            covs_all.append(c)
            ate0_all.append(float(np.sqrt((result.initial_pose_errors() ** 2).mean())))
            ate_all.append(float(np.sqrt((result.pose_errors() ** 2).mean())))

        if not errors_all:
            print(f"{level:6.2f}  랜드마크 부족")
            all_ok = False
            continue

        E, C = np.vstack(errors_all), np.vstack(covs_all)
        r = nees(E, C)
        ate0, ate = float(np.mean(ate0_all)), float(np.mean(ate_all))
        lm_rmse = float(np.sqrt((np.linalg.norm(E, axis=1) ** 2).mean()))

        verdict = ("ok" if r.consistent
                   else ("OVERCONF" if r.overconfident else "underconf"))
        all_ok &= r.consistent

        band = f"[{r.lower:.2f},{r.upper:.2f}]"
        print(f"{level:6.2f} {ate0:8.4f} {ate:8.4f} {1 - ate / max(ate0, 1e-9):7.1%} "
              f"{lm_rmse:8.4f} {len(E):4d} {r.anees / r.dof:9.3f} {band:>14}  {verdict}")

    print()
    print("ATE0 = 오도메트리 적분 초기값, ATE = 최적화 후 (시드 평균).")
    print("랜드마크 공분산은 포즈 불확실성까지 주변화한 값이어야 한다.")
    print(f"M4 통과: {'YES' if all_ok else 'NO'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
