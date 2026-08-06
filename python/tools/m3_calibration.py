"""M3 게이트 실행기.

    python tools/m3_calibration.py [--channel haze] [--seed 0]

훈련 조건에서 잡음모델을 적합하고 보지 않은 조건에서 NEES 일관성을 본다.
calibrated 가 통과하지 못하면 "하나의 사후분포" 논지는 지지되지 않는다.
"""

from __future__ import annotations

import argparse
import sys

from wme.calib import run_m3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="haze",
                    choices=["haze", "darkness", "motion_blur", "rain_streak",
                             "snow_particle", "lens_dirt"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--laps", type=float, default=2.0)
    args = ap.parse_args()

    report = run_m3(channel=args.channel, seed=args.seed,
                    n_frames=args.frames, laps=args.laps)
    print(report)
    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
