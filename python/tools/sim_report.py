"""시뮬레이션 하네스 + 적응 로직 점검 리포트.

조건 스윕을 돌려 관측 모델의 열화와 엔진의 tier 가중치 반응을 나란히 출력한다.
설계 의도대로 움직이는지 눈으로 확인하는 용도이며, 수치 회귀는 테스트가 잡는다.

    python tools/sim_report.py [--channel haze] [--levels 6]
"""

from __future__ import annotations

import argparse

import numpy as np

from wme.eval.metrics import degradation_curve, identity_metrics
from wme.reference.environment import Evidence, derive_adaptation
from wme.sim import scenario_condition_sweep, scenario_revisit_with_changes, severity_of


def sweep_report(channel: str, levels: int, seed: int) -> None:
    print(f"\n=== 조건 스윕: {channel} (동일 세계 / 동일 궤적) ===")
    print(f"{'level':>6} {'sev':>6} {'det%':>6} {'FA/f':>6} {'sig_z':>7} "
          f"{'a0':>6} {'a1':>6} {'a2':>6} {'prior':>6} {'mem':>5}")

    severities, det_rates = [], []
    for i, seq in enumerate(scenario_condition_sweep(channel, levels, seed)):
        level = i / max(1, levels - 1)
        sev = seq.severity

        visible = sum(len(t.visible_objects) for t in seq.truths)
        detected = sum(len(t.detected_objects) for t in seq.truths)
        false_alarms = sum(1 for t in seq.truths for o in t.detection_to_object if o == -1)
        sigmas = [s for t in seq.truths for s in t.detection_depth_sigma if s < 20.0]

        # 엔진이 이 조건을 어떻게 해석하는가 (관측 모델과는 독립적인 경로)
        a = derive_adaptation(Evidence(**{channel: level}))

        det_pct = 100.0 * detected / max(1, visible)
        severities.append(sev)
        det_rates.append(det_pct)

        print(f"{level:6.2f} {sev:6.3f} {det_pct:6.1f} "
              f"{false_alarms / len(seq):6.2f} {np.mean(sigmas):7.4f} "
              f"{a.alpha_photometric:6.3f} {a.alpha_constellation:6.3f} "
              f"{a.alpha_structural:6.3f} {a.motion_prior:6.3f} "
              f"{a.memory_retention_scale:5.2f}")

    curve = degradation_curve(severities, [100.0 - d for d in det_rates],
                              metric_name="miss_rate_pct", failure_value=100.0)
    print(f"\n  {curve.summary()}")
    print("  a0 가 a1 보다 빠르게 무너지는 것이 TCG 의 존재 근거다.")


def revisit_report(seed: int) -> None:
    seq = scenario_revisit_with_changes(seed=seed)
    print(f"\n=== 재방문 + 장면 변화 ({len(seq)} 프레임) ===")
    for c in sorted(seq.changes, key=lambda e: (e.kind, e.object_id)):
        print(f"  {c.kind:8s} object {c.object_id:3d} @ t={c.time:.2f}")

    # 참값 연관으로 정체성 지표의 상한을 확인한다.
    # 실제 엔진은 이보다 나쁠 수밖에 없고, 이 값이 비교 기준이 된다.
    tracks = [{oid: oid for oid in t.detected_objects} for t in seq.truths]
    print(f"\n  참값 연관 기준선: {identity_metrics(tracks).summary()}")

    visible = sum(len(t.visible_objects) for t in seq.truths)
    detected = sum(len(t.detected_objects) for t in seq.truths)
    print(f"  가시 {visible}, 검출 {detected} ({100.0 * detected / max(1, visible):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="haze",
                    choices=["haze", "darkness", "motion_blur", "rain_streak",
                             "snow_particle", "lens_dirt", "texture_poverty"])
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sweep_report(args.channel, args.levels, args.seed)
    revisit_report(args.seed)
    print()


if __name__ == "__main__":
    main()
