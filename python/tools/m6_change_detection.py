"""M6: "무엇이 바뀌었는가" 를 수치로 만든다.

    python tools/m6_change_detection.py [--trials 5] [--levels 0.0 0.5]

두 가지를 잰다.
  1. 재방문 시나리오에서의 변화 검출 P/R/지연 (종류별)
  2. **변화가 전혀 없는데 센서만 나빠진** 시퀀스에서의 오보율

두 번째가 더 중요하다. 안개가 꼈다고 세계가 변했다고 보고하는 시스템은
계획기에 붙일 수 없다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from wme.eval.metrics import change_metrics
from wme.sim import constant_condition, run, scenario_revisit_with_changes
from wme.sim.world import CameraTrajectory
from wme.world.pipeline import PipelineConfig, detect_revisit_changes


def revisit_report(trials: int, level: float) -> None:
    gt_all, rep_all = [], []
    reid_total = 0

    for seed in range(trials):
        seq = scenario_revisit_with_changes(seed=seed, haze=level)
        result = detect_revisit_changes(seq, PipelineConfig())
        gt_all.extend(seq.changes)
        # 검출은 트랙 ID 로 나오므로 참 객체 ID 로 옮겨야 채점이 성립한다
        rep_all.extend(result.as_object_events())
        reid_total += len(result.reid_links)

    r = change_metrics(gt_all, rep_all, time_tolerance=40.0)
    kinds = sorted(set(r.precision) | set(r.recall))

    print(f"\n  haze={level:.2f}  ({trials} 시드, 재식별 연결 {reid_total}건)")
    print(f"  {'kind':<9} {'P':>6} {'R':>6} {'지연(s)':>9}")
    for k in kinds:
        lat = r.latency.get(k, float('inf'))
        lat_txt = "-" if not np.isfinite(lat) else f"{lat:.1f}"
        print(f"  {k:<9} {r.precision.get(k, 0):6.2f} {r.recall.get(k, 0):6.2f} {lat_txt:>9}")
    print(f"  검출 {r.detected}, 미검출 {r.missed}, 오보 {r.spurious} "
          f"(오보율 {r.false_change_rate:.2f})")


def _static_revisit(seed: int, level: float):
    """변화가 전혀 없는 재방문. 같은 세계를 두 번 훑는다."""
    from wme.sim import static_room
    world = static_room(12, seed=seed)
    traj = CameraTrajectory.revisit(radius=3.0, n_per_visit=90, gap_seconds=30.0)
    return run(world, traj, constant_condition(haze=level), seed=seed,
               name=f"static_revisit_{level:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5])
    args = ap.parse_args()

    print("=" * 74)
    print("M6: 변화 검출")
    print("=" * 74)

    print("\n[1] 재방문 + 실제 변화 (moved / removed / added)")
    for level in args.levels:
        revisit_report(args.trials, level)

    print("\n[2] 변화 없음 + 센서 열화만 - 오보가 나면 안 된다")
    print(f"  {'haze':>6} {'오보':>6} {'객체':>6} {'오보율':>8}  verdict")

    ok = True
    for level in [0.0, 0.3, 0.6, 0.9]:
        spurious = objects = 0
        for seed in range(args.trials):
            seq = _static_revisit(seed, level)
            result = detect_revisit_changes(seq, PipelineConfig())
            spurious += len(result.detected)
            objects += len(seq.world.objects)
        rate = spurious / max(1, objects)
        verdict = "ok" if rate < 0.05 else "FALSE CHANGES"
        ok &= rate < 0.05
        print(f"  {level:6.2f} {spurious:6d} {objects:6d} {rate:8.3f}  {verdict}")

    print()
    print("변화는 두 관측 *사이* 에만 정의된다. 판정 문턱을 절대 거리가 아니라")
    print("믿음의 공분산으로 잡았으므로, 센서가 나빠지면 문턱도 함께 넓어진다.")
    print(f"M6 오보 기준 통과: {'YES' if ok else 'NO'}")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
