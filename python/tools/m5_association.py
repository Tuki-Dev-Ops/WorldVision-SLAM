"""M5: 데이터 연관 전략 비교.

    python tools/m5_association.py [--levels 0.0 0.5] [--crowded]

--crowded 는 동일 클래스 객체를 가까이 배치해 연관을 일부러 애매하게 만든다.
전략 차이는 애매한 상황에서만 드러나므로, 이 시나리오가 본 시험이다.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from wme.association import AssociationConfig, compare_strategies
from wme.sim import CameraTrajectory, SimObject, SimWorld, constant_condition, run


def crowded_room(n_pairs: int = 5, seed: int = 0, separation: float = 0.16) -> SimWorld:
    """동일 클래스 객체를 게이트 반경 안쪽으로 붙여 놓는다.

    separation 이 게이트 반경(대략 3.4 sigma ~ 0.35 m)보다 작아야 실제로
    애매해진다. 그보다 멀면 클래스와 거리만으로 유일하게 갈려서 어떤 전략을
    써도 결과가 같다 - 초기 실험이 그 함정에 빠졌다.
    """
    rng = np.random.default_rng(seed)
    objs = []
    oid = 1
    for k in range(n_pairs):
        angle = 2.0 * np.pi * k / n_pairs
        base = np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.7])
        for side in (-1.0, 1.0):
            offset = np.array([-np.sin(angle), np.cos(angle), 0.0]) * side * separation
            objs.append(SimObject(
                object_id=oid, class_id=56, class_name="chair",
                position=base + offset + rng.normal(0.0, 0.01, 3),
                extent=np.array([0.12, 0.12, 0.25]),
            ))
            oid += 1
    return SimWorld(objs)


def spread_room(n: int = 10, seed: int = 0) -> SimWorld:
    """서로 떨어진 서로 다른 클래스. 연관이 쉬운 대조군."""
    rng = np.random.default_rng(seed)
    classes = [(56, "chair"), (60, "dining table"), (41, "cup"), (73, "book"),
               (75, "vase"), (62, "tv"), (57, "couch"), (66, "keyboard"),
               (58, "potted plant"), (72, "refrigerator")]
    objs = []
    for i in range(n):
        cid, name = classes[i % len(classes)]
        a = 2.0 * np.pi * i / n
        objs.append(SimObject(
            object_id=i + 1, class_id=cid, class_name=name,
            position=np.array([3.2 * np.cos(a), 3.2 * np.sin(a), rng.uniform(0.4, 1.2)]),
            extent=np.array([0.25, 0.25, 0.3]),
        ))
    return SimWorld(objs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 82)
    print("M5: 데이터 연관 전략 비교")
    print("=" * 82)
    print("inflate = 참 객체 1개당 만들어진 트랙 수 (1.00 이 이상적)")
    print("defer   = 확정을 미룬 관측 수 (Deferred 전용)\n")

    loop = CameraTrajectory.loop(radius=3.4, laps=1.5, n=args.frames)
    # 멀리서 시작해 접근한다. 초반 관측은 부정확해 연관이 모호하고,
    # 후반 정밀 관측이 그 모호성을 해소한다.
    spiral = CameraTrajectory.spiral(r_start=9.0, r_end=2.0, laps=1.5, n=args.frames)
    cfg = AssociationConfig()

    cases = [
        ("떨어진 배치 (쉬움)", spread_room(10, args.seed), loop),
        ("밀집 동일클래스 (끝까지 애매)", crowded_room(5, args.seed), loop),
        ("밀집 + 접근 (나중에 해소)", crowded_room(5, args.seed), spiral),
    ]

    for label, world, traj in cases:
        for level in args.levels:
            seq = run(world, traj, constant_condition(haze=level),
                      seed=args.seed, name=label)
            t0 = time.perf_counter()
            report = compare_strategies(seq, cfg, f"{label}, haze={level:.2f}")
            elapsed = time.perf_counter() - t0
            print(report)
            print(f"  ({len(seq)} 프레임 처리에 {elapsed:.2f}s)\n")

    print("모호성이 끝까지 해소되지 않으면 어떤 다중가설 기법도 이득이 없다.")
    print("MHT 의 이점은 '나중에 해소될 모호성'에서만 나온다 - 세 번째 케이스가 그 시험.")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
