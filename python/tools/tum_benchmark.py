"""여러 TUM 시퀀스에서 ECDA 를 돌리고 한 표로 낸다.

시퀀스 하나로는 아무것도 주장할 수 없다. 특히 fr1_xyz 는 TUM 에서 가장 쉬운
축에 속한다 - 느리고, 텍스처가 많고, 장면이 정적이다.

이 표의 요점은 평균이 아니라 짝 비교다:
  sitting_xyz vs walking_xyz : 같은 장면, 같은 카메라 운동, 사람만 앉아 있느냐
                               걸어다니느냐. 두 값의 격차가 곧 동적 객체가
                               측광 정렬에 입히는 손해이고, WME 의 토큰 마스킹이
                               갚아야 할 빚이다.
  fr1_xyz vs fr1_360         : 병진 위주 vs 순수 회전. RGB-D 직접법이 회전에서
                               무너지는지 본다.

기준선(identity)을 매 시퀀스마다 같이 낸다. 이기지 못하는 줄이 있으면 그게
결과다.

사용: python tools/tum_benchmark.py <데이터경로> <실행파일> [--frames N]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate, evaluate_rpe  # noqa: E402
from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

SEQUENCES = [
    ("rgbd_dataset_freiburg1_xyz", "정적 · 병진 위주 (가장 쉬움)"),
    ("rgbd_dataset_freiburg1_desk", "정적 · 회전 포함"),
    ("rgbd_dataset_freiburg1_360", "정적 · 순수 회전"),
    ("rgbd_dataset_freiburg3_sitting_xyz", "사람 있음 · 거의 정지"),
    ("rgbd_dataset_freiburg3_walking_xyz", "사람 있음 · 보행 (동적)"),
]


def run_one(root: Path, exe: str, out: Path) -> dict:
    r = subprocess.run([exe, str(root), str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    stats = {"rmse": float("nan"), "pts": float("nan"), "failed": 0}
    if r.returncode != 0:
        return stats
    for line in r.stdout.splitlines():
        tok = line.split()
        if "평균" in line and "RMSE" in line:
            stats["rmse"] = float(tok[tok.index("RMSE") + 1])
            if "점수" in tok:
                stats["pts"] = float(tok[tok.index("점수") + 1])
        if line.startswith("프레임 ") and "정렬실패" in line:
            stats["failed"] = int(tok[tok.index("정렬실패") + 1])
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("exe")
    args = ap.parse_args()

    data = Path(args.data)
    out_dir = Path("../results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, note in SEQUENCES:
        root = data / name
        if not (root / "rgb.txt").exists():
            rows.append((name, note, None))
            continue

        out = out_dir / f"{name}_ecda.txt"
        stats = run_one(root, args.exe, out)

        gt = load_sequence(root).trajectory()
        est = load_trajectory(out)
        ident = Trajectory(est.stamps, [SE3.identity() for _ in est.stamps])

        a = evaluate_ate(est, gt, align=True)
        ai = evaluate_ate(ident, gt, align=True)
        p = evaluate_rpe(est, gt, delta=0.034)

        gp = np.array([q.t for q in gt.poses])
        path = float(np.linalg.norm(np.diff(gp, axis=0), axis=1).sum())

        rows.append((name, note, {
            "ate": a.rmse * 100, "base": ai.rmse * 100,
            "rpe": p.trans_rmse * 1000, "rot": p.rot_rmse_deg,
            "n": len(est.poses), "path": path, **stats,
        }))

    print(f"{'시퀀스':<38} {'ATE':>7} {'기준선':>7} {'배수':>6} "
          f"{'RPE':>7} {'회전':>6} {'RMSE':>6}")
    print(f"{'':<38} {'(cm)':>7} {'(cm)':>7} {'':>6} {'(mm)':>7} {'(deg)':>6} {'':>6}")
    print("-" * 82)
    for name, note, s in rows:
        short = name.replace("rgbd_dataset_", "")
        if s is None:
            print(f"{short:<38} {'(없음)':>7}")
            continue
        ratio = s["base"] / max(s["ate"], 1e-9)
        mark = "" if ratio > 1.0 else "  <- 기준선에 짐"
        print(f"{short:<38} {s['ate']:7.2f} {s['base']:7.2f} {ratio:5.1f}x "
              f"{s['rpe']:7.1f} {s['rot']:6.2f} {s['rmse']:6.1f}{mark}")
    print("-" * 82)

    # 짝 비교: 동적 물체가 입히는 손해
    got = {n: s for n, _, s in rows if s is not None}
    sit = got.get("rgbd_dataset_freiburg3_sitting_xyz")
    walk = got.get("rgbd_dataset_freiburg3_walking_xyz")
    if sit and walk:
        print(f"\n동적 객체 대가 (sitting -> walking, 같은 장면):")
        print(f"  ATE  {sit['ate']:.2f} -> {walk['ate']:.2f} cm   "
              f"({walk['ate']/max(sit['ate'],1e-9):.1f}배 악화)")
        print(f"  RPE  {sit['rpe']:.1f} -> {walk['rpe']:.1f} mm")
        print("  이 격차가 토큰 기반 동적 마스킹이 갚아야 할 빚이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
