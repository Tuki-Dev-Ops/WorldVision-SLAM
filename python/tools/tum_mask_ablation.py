"""동적 마스킹 절제 실험.

마스킹을 켠 결과만 보면 승리를 주장할 수 없다. 정적 장면에서 대가가 없는지
같이 봐야 한다 - 움직일 수 있는 클래스를 무조건 지우는 것은 사전분포이지
판정이 아니고, 앉아 있는 사람도 벽처럼 훌륭한 랜드마크다.

wme_tum_odometry 를 --yolo 유무로 각각 돌려 둔 결과를 대조한다.
사용: python tools/tum_mask_ablation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate, evaluate_rpe  # noqa: E402
from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

SEQS = [
    ("freiburg3_walking_xyz", "사람 보행 (동적)"),
    ("freiburg3_sitting_xyz", "사람 앉음 (정적)"),
    ("freiburg1_xyz", "사람 없음"),
    ("freiburg1_desk", "사람 없음"),
]

# 긴 창(40 s). 믿음이 수렴하는 데 걸리는 시간은 8 초 창으로 정할 수 없다 -
# 그 창 안에서는 "아직 안 올랐다" 와 "안 오른다" 가 같은 그림이다.
LONG_SEQS = [("freiburg3_sitting_xyz", "사람 앉음 40 s")]

# 보류집합(halfsphere). 위의 네 줄은 전부 xyz 병진 운동이고, 17.2 는 그 위에서
# 맞춘 결론이 반구 스윕에서 뒤집히는 것을 뒤늦게 발견했다. 절제 실험 도구가
# 그 사례를 낼 수 없으면 같은 일이 다시 일어난다.
HELD_OUT_SEQS = [
    ("freiburg3_sitting_halfsphere", "앉음 + 반구스윕"),
    ("freiburg3_walking_halfsphere", "보행 + 반구스윕"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="data/long 의 40 s 창을 대조한다")
    ap.add_argument("--held-out", action="store_true",
                    help="halfsphere 보류집합을 대조한다 (17.2 가 뒤집힌 자리)")
    ap.add_argument("--results", default="../results",
                    help="궤적 파일이 있는 디렉토리")
    args = ap.parse_args()

    seqs = LONG_SEQS if args.long else (HELD_OUT_SEQS if args.held_out else SEQS)
    data_dir = "../data/long" if args.long else "../data"
    tag = "_long" if args.long else ""

    print(f"{'시퀀스':<30} {'성격':<18} {'OFF':>7} {'클래스':>7} {'믿음':>9} {'기준선':>8}")
    print("-" * 88)
    for name, note in seqs:
        gt = load_sequence(f"{data_dir}/rgbd_dataset_{name}").trajectory()
        vals = {}
        base = float("nan")
        for m in ("off", "on", "token"):
            p = Path(f"{args.results}/{name}{tag}_mask{m}.txt")
            if not p.exists():
                vals[m] = float("nan")
                continue
            est = load_trajectory(p)
            vals[m] = evaluate_ate(est, gt, align=True).rmse * 100
            if m == "off":
                ident = Trajectory(est.stamps, [SE3.identity() for _ in est.stamps])
                base = evaluate_ate(ident, gt, align=True).rmse * 100

        print(f"{name:<30} {note:<18} {vals['off']:7.2f} {vals['on']:7.2f} "
              f"{vals['token']:9.2f} {base:8.2f}")
    print("-" * 88)
    print("단위 cm (ATE RMSE). 기준선 = 카메라가 안 움직였다는 가정.")
    print("클래스 = 움직일 수 있는 클래스를 무조건 마스킹")
    print("믿음   = TokenStore 의 static_belief 로 판정 (관측이 결론을 바꾼다)")
    print("믿음 열은 OFF 보다 나쁠 수 없어야 한다 - 근거 없는 마스킹을 금지하면")
    print("연관이 무너져도 마스킹이 0 으로 퇴화하기 때문이다 (17.2 의 13053 cm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
