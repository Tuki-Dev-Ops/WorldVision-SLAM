"""TUM 실데이터에서 ECDA 파라미터를 훑는다.

합성에서 맞춘 값이 실센서에서도 맞다는 보장은 없다. 추측 대신 잰다.
각 설정마다 오도메트리를 돌리고 ATE 를 매긴 뒤, 기준선(identity)과 함께 표로 낸다.

사용: python tools/tum_sweep.py <시퀀스경로> <실행파일>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate, evaluate_rpe  # noqa: E402
from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

# 한 번에 한 축만 바꾼다. 여러 축을 동시에 흔들면 무엇이 효과였는지 알 수 없다.
BASE = ["--kf-dist", "0.03"]

SWEEPS: list[tuple[str, list[str]]] = [
    ("기본 (kf 0.03)", []),
    ("kf 0.02", ["--kf-dist", "0.02"]),
    ("kf 0.05", ["--kf-dist", "0.05"]),
    ("edge-ratio 10 (해제)", ["--edge-ratio", "10"]),
    ("edge-ratio 0.05", ["--edge-ratio", "0.05"]),
    ("huber-k 0.5", ["--huber-k", "0.5"]),
    ("huber-k 4", ["--huber-k", "4"]),
    ("noise-ratio 20", ["--noise-ratio", "20"]),
    ("levels 4", ["--levels", "4"]),
    ("levels 6", ["--levels", "6"]),
    ("grid 4", ["--grid", "4"]),
    ("grid 6", ["--grid", "6"]),
    ("min-grad 3", ["--min-grad", "3"]),
    ("min-grad 12", ["--min-grad", "12"]),
]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, exe = sys.argv[1], sys.argv[2]
    out_dir = Path("../results/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_sequence(root).trajectory()

    rows = []
    first_stamps = None
    for idx, (name, extra) in enumerate(SWEEPS):
        out = out_dir / f"run{idx:02d}.txt"
        # BASE 를 먼저 두어 extra 가 같은 키를 덮어쓰게 한다 (파서가 뒤를 우선한다).
        cmd = [exe, root, str(out)] + BASE + extra
        # C++ 쪽은 UTF-8 로 찍는데 윈도우 기본 코덱은 cp949 다. 명시하지 않으면 깨진다.
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            rows.append((name, float("nan"), float("nan"), 0.0, 0.0))
            continue

        stats = {"rmse": float("nan"), "pts": float("nan")}
        for line in r.stdout.splitlines():
            if "평균 측광 RMSE" in line:
                tok = line.split()
                stats["rmse"] = float(tok[tok.index("RMSE") + 1])
                if "점수" in tok:
                    stats["pts"] = float(tok[tok.index("점수") + 1])

        est = load_trajectory(out)
        if first_stamps is None:
            first_stamps = est.stamps
        a = evaluate_ate(est, gt, align=True)
        p = evaluate_rpe(est, gt, delta=0.034)
        rows.append((name, a.rmse * 100, p.trans_rmse * 1000, stats["rmse"], stats["pts"]))

    # 기준선: 카메라가 안 움직였다는 가정. 어떤 설정도 이보다 나쁘면 의미가 없다.
    ident = Trajectory(first_stamps, [SE3.identity() for _ in first_stamps])
    ai = evaluate_ate(ident, gt, align=True)
    pi = evaluate_rpe(ident, gt, delta=0.034)

    print(f"{'설정':<28} {'ATE':>8} {'프레임RPE':>10} {'측광RMSE':>9} {'점수':>8}")
    print(f"{'':<28} {'(cm)':>8} {'(mm)':>10} {'':>9} {'':>8}")
    print("-" * 68)
    for name, ate, rpe, rmse, pts in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<28} {ate:8.2f} {rpe:10.1f} {rmse:9.1f} {pts:8.0f}")
    print("-" * 68)
    print(f"{'identity (기준선)':<28} {ai.rmse*100:8.2f} {pi.trans_rmse*1000:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
