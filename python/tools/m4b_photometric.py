"""M4b: 실제 ECDA 측광 항으로 닫는 M4.

이전 M4 는 상대 포즈 팩터를 대리물로 썼다. 이제 렌더러가 있으므로 Tier 0 를
진짜로 돌린다.

    python tools/m4b_photometric.py [--frames 40] [--levels 0.0 0.4]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from wme.calib import CalibratedNoise
from wme.eval.metrics import nees
from wme.graph.photometric_slam import (
    PhotometricSlamConfig, calibrate_information_scale, render_sequence, solve,
)
from wme.reference.environment import Evidence, derive_adaptation
from wme.sim import CameraModel, CameraTrajectory, SimObject, SimWorld
from wme.sim.render import RenderScene

CAM = CameraModel(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)

# M3 에서 적합된 관측 잡음. 참값(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)에 근접.
NOISE = CalibratedNoise(c_px=1.996, g_px=3.82, c_d=0.00557, g_d=4.14)

_CLASSES = [(56, "chair"), (60, "dining table"), (41, "cup"), (75, "vase"),
            (62, "tv"), (73, "book"), (58, "potted plant"), (57, "couch")]


def make_world(n: int = 8, seed: int = 0, arc: float = 1.6) -> SimWorld:
    """카메라가 훑을 부채꼴 안에 객체를 배치한다.

    거리는 2~3.5 m 로 둔다. 더 가까우면 객체가 화면을 채우고 경계에 잘려
    관측이 쓸모없어진다 (초기 설정이 그랬다).
    """
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n):
        cid, name = _CLASSES[i % len(_CLASSES)]
        a = -arc * 0.5 + arc * i / max(1, n - 1) + rng.normal(0.0, 0.06)
        r = rng.uniform(2.6, 3.4)
        objs.append(SimObject(
            object_id=i + 1, class_id=cid, class_name=name,
            position=np.array([r * np.cos(a), r * np.sin(a), rng.uniform(0.6, 1.6)]),
            extent=np.array([0.26, 0.26, 0.30]),
        ))
    return SimWorld(objs)


def make_trajectory(n: int, radius: float = 0.9, arc: float = 1.6,
                    revisit: bool = True) -> CameraTrajectory:
    """부채꼴을 천천히 훑으며 *바깥*을 보는 궤적.

    세 가지가 중요하다.
      - 시선이 바깥이어야 한다. 기본 loop 는 중심을 향하므로 궤도 바깥의
        객체가 전부 카메라 뒤로 간다.
      - 프레임 간 회전이 작아야 한다. 직접정렬의 수렴 반경은 좁고, 한 바퀴를
        30 프레임에 도는 궤적은 프레임당 12도라 전혀 수렴하지 않는다.
        실제 VO 는 프레임당 1~2도 수준이다.
      - revisit=True 면 왕복한다. 이것이 없으면 객체 팩터가 아무 역할도 못 한다:
        짧고 정확한 오도메트리 사슬은 그 자체로 객체 관측보다 정보가 많고,
        드리프트가 누적되어야 비로소 객체가 교정할 여지가 생긴다.
    """
    stamps, poses = [], []
    half = n // 2 if revisit else n
    for i in range(n):
        if revisit:
            # 갔다가 되돌아온다. 처음 본 객체를 드리프트가 쌓인 뒤 다시 만난다.
            k = i if i < half else (n - 1 - i)
            s = k / max(1, half - 1)
        else:
            s = i / max(1, n - 1)
        a = -arc * 0.5 + arc * s
        eye = np.array([radius * np.cos(a), radius * np.sin(a), 1.3])
        target = np.array([4.5 * np.cos(a), 4.5 * np.sin(a), 1.0])
        poses.append(CameraTrajectory._look_at(eye, target))
        stamps.append(1.0 + i * 0.05)
    return CameraTrajectory(np.array(stamps), poses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.4])
    ap.add_argument("--channel", default="haze")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    print("=" * 84)
    print("M4b: 실제 ECDA 측광 오도메트리 + 객체 팩터")
    print("=" * 84)
    print(f"{'level':>6} {'a0':>6} {'ECDA':>9} {'scale':>9} {'ATE0':>8} {'ATE':>8} "
          f"{'개선':>7} {'lm_rmse':>8} {'n':>4} {'ANEES/d':>9} {'band':>14}  verdict")

    scene = RenderScene.room(size=4.0, height=2.6)
    all_ok = True

    for level in args.levels:
        ev = Evidence(**{args.channel: level})
        alpha = derive_adaptation(ev).alpha_photometric

        errors_all, covs_all = [], []
        ate0_all, ate_all, acc_all, rej_all, scales = [], [], [], [], []

        for seed in range(args.trials):
            world = make_world(8, seed)
            traj = make_trajectory(args.frames)
            frames = render_sequence(scene, world, traj, CAM, ev, seed=seed * 1000)

            # 측광 정보의 유효표본 보정. 픽셀 잔차는 독립이 아니므로
            # J^T W J 를 그대로 쓰면 정보가 크게 과대평가된다.
            scale, _raw = calibrate_information_scale(frames, CAM,
                                                      PhotometricSlamConfig(), ev)
            scales.append(scale)

            r = solve(frames, world, CAM, NOISE,
                      PhotometricSlamConfig(seed=seed,
                                            photometric_information_scale=scale), ev)
            acc_all.append(r.ecda_accepted)
            rej_all.append(r.ecda_rejected)
            try:
                e, c = r.landmark_nees_inputs()
            except ValueError:
                continue
            errors_all.append(e)
            covs_all.append(c)
            ate0_all.append(float(np.sqrt((r.initial_pose_errors() ** 2).mean())))
            ate_all.append(float(np.sqrt((r.pose_errors() ** 2).mean())))

        if not errors_all:
            print(f"{level:6.2f}  랜드마크 부족")
            all_ok = False
            continue

        E, C = np.vstack(errors_all), np.vstack(covs_all)
        res = nees(E, C)
        lm_rmse = float(np.sqrt((np.linalg.norm(E, axis=1) ** 2).mean()))
        ate0, ate = float(np.mean(ate0_all)), float(np.mean(ate_all))
        verdict = ("ok" if res.consistent
                   else ("OVERCONF" if res.overconfident else "underconf"))
        all_ok &= res.consistent

        ecda = f"{int(np.mean(acc_all))}/{int(np.mean(acc_all)) + int(np.mean(rej_all))}"
        band = f"[{res.lower:.2f},{res.upper:.2f}]"
        print(f"{level:6.2f} {alpha:6.3f} {ecda:>9} {np.mean(scales):9.2e} "
              f"{ate0:8.4f} {ate:8.4f} {1 - ate / max(ate0, 1e-9):7.1%} "
              f"{lm_rmse:8.4f} {len(E):4d} {res.anees / res.dof:9.3f} "
              f"{band:>14}  {verdict}")

    print()
    print("ECDA = 잔차 게이트를 통과한 정렬 수 / 전체. 랭크로는 못 잡는 실패를")
    print("잔차로 거른다 (docs/03-roadmap.md Phase 2g).")
    print(f"M4b 통과: {'YES' if all_ok else 'NO'}")
    print("=" * 84)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
