"""M7: 세 tier 융합이 실제로 작동하는가.

    python tools/m7_fusion.py [--frames 40] [--trials 2]

docs/02-correspondence-problem.md 5장의 핵심 주장을 잰다.

    Lambda_total = a0(E) Lambda_ECDA + a1(E) Lambda_TCG + a2(E) Lambda_SPA

검증할 것은 하나다: **열화가 심해질수록 융합이 Tier 0 단독보다 우월해지는가.**
맑은 조건에서 융합이 조금 나은 것은 별 의미가 없다. 측광이 무너지는 조건에서
나머지 tier 가 실제로 받쳐주는지가 이 아키텍처의 존재 이유다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from wme.calib import CalibratedNoise
from wme.graph.fusion import FusionConfig, solve
from wme.graph.photometric_slam import render_sequence
from wme.reference.environment import Evidence, derive_adaptation
from wme.sim import CameraTrajectory, SimObject, SimWorld
from wme.sim.render import RenderScene
from wme.sim.world import CameraModel

CAM = CameraModel(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)
NOISE = CalibratedNoise(c_px=1.996, g_px=3.82, c_d=0.00557, g_d=4.14)

_CLASSES = [(56, "chair"), (60, "dining table"), (41, "cup"), (75, "vase"),
            (62, "tv"), (73, "book"), (58, "potted plant"), (57, "couch")]


def make_world(n: int = 16, seed: int = 0, arc: float = 1.5) -> SimWorld:
    """객체를 좁은 부채꼴에 촘촘히 둔다.

    TCG 는 프레임당 min_nodes(기본 4) 이상의 안정 객체를 동시에 봐야 성좌를
    만든다. 넓게 흩어 두면 72도 화각에 서너 개밖에 안 들어와 Tier 1 이 통째로
    굶는다 - 실측에서 30 프레임 중 2 프레임만 자격을 갖췄다.
    """
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n):
        cid, name = _CLASSES[i % len(_CLASSES)]
        a = -arc * 0.5 + arc * i / max(1, n - 1) + rng.normal(0.0, 0.05)
        r = rng.uniform(2.6, 3.4)
        objs.append(SimObject(
            object_id=i + 1, class_id=cid, class_name=name,
            position=np.array([r * np.cos(a), r * np.sin(a), rng.uniform(0.6, 1.6)]),
            extent=np.array([0.26, 0.26, 0.30]),
        ))
    return SimWorld(objs)


def make_trajectory(n: int, radius: float = 0.9, arc: float = 2.0) -> CameraTrajectory:
    """왕복. 드리프트가 쌓인 뒤 처음 본 곳을 다시 봐야 루프가 의미를 갖는다."""
    stamps, poses = [], []
    half = n // 2
    for i in range(n):
        k = i if i < half else (n - 1 - i)
        s = k / max(1, half - 1)
        a = -arc * 0.5 + arc * s
        eye = np.array([radius * np.cos(a), radius * np.sin(a), 1.3])
        target = np.array([4.5 * np.cos(a), 4.5 * np.sin(a), 1.0])
        poses.append(CameraTrajectory._look_at(eye, target))
        stamps.append(1.0 + i * 0.05)
    return CameraTrajectory(np.array(stamps), poses)


ABLATIONS = [
    ("T0 only",     dict(use_photometric=True,  use_constellation=False, use_structural=False)),
    ("T2 only",     dict(use_photometric=False, use_constellation=False, use_structural=True)),
    ("T0+T1",       dict(use_photometric=True,  use_constellation=True,  use_structural=False)),
    ("T0+T2",       dict(use_photometric=True,  use_constellation=False, use_structural=True)),
    ("T0+T1+T2",    dict(use_photometric=True,  use_constellation=True,  use_structural=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.4, 0.8])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--channel", default="haze")
    args = ap.parse_args()

    scene = RenderScene.room(size=4.0, height=2.6)
    traj = make_trajectory(args.frames)

    print("=" * 88)
    print("M7: 세 tier 융합")
    print("=" * 88)
    print("검증 대상: 열화가 심해질수록 융합이 Tier 0 단독보다 우월해지는가.\n")

    header = f"{'level':>6} {'a0':>5} {'a1':>5} {'a2':>5} " + \
             "".join(f"{name:>11}" for name, _ in ABLATIONS) + f"{'ATE0':>9}"
    print(header)

    baseline_gap = []
    for level in args.levels:
        ev = Evidence(**{args.channel: level})
        a = derive_adaptation(ev)

        per_ablation = {name: [] for name, _ in ABLATIONS}
        ate0 = []
        counts = None

        for seed in range(args.trials):
            world = make_world(seed=seed)
            frames = render_sequence(scene, world, traj, CAM, ev, seed=seed * 977)

            for name, switches in ABLATIONS:
                cfg = FusionConfig(**switches)
                r = solve(frames, world, CAM, NOISE, cfg, ev)
                per_ablation[name].append(r.ate())
                if name == "T0+T1+T2":
                    ate0.append(r.ate_initial())
                    counts = (r.photometric.factors, r.constellation.factors,
                              r.structural.factors)

        row = f"{level:6.2f} {a.alpha_photometric:5.2f} {a.alpha_constellation:5.2f} " \
              f"{a.alpha_structural:5.2f} "
        row += "".join(f"{np.mean(per_ablation[name]):11.4f}" for name, _ in ABLATIONS)
        row += f"{np.mean(ate0):9.4f}"
        print(row)

        t0 = float(np.mean(per_ablation["T0 only"]))
        full = float(np.mean(per_ablation["T0+T1+T2"]))
        baseline_gap.append((level, t0, full, counts))

    print("\n팩터 수 (T0+T1+T2, 마지막 시드):")
    for level, t0, full, counts in baseline_gap:
        c = counts or (0, 0, 0)
        gain = (t0 - full) / max(t0, 1e-9)
        print(f"  haze={level:.2f}  측광 {c[0]:3d}  성좌 {c[1]:3d}  구조 {c[2]:3d}   "
              f"T0 대비 개선 {gain:+6.1%}")

    print("\n열화가 심할수록 개선 폭이 커져야 융합 주장이 성립한다.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
