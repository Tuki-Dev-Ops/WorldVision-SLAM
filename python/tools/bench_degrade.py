"""Degradation sweep on real data - the experiment 18.5 said was missing.

docs/06-results.md 1 is the architecture's headline: fusion's advantage *grows*
as conditions degrade, 6.8 % -> 53.6 % as haze rises. Every number in it comes
from a renderer whose noise model this project wrote. 18.5 left it here:

    "Whether the curve's *slope* is real remains untested, and testing it needs
     either degraded real data or an honest admission that the 53.6 % is a
     property of the renderer's noise model."

`wme_tum_degrade` makes the degraded real data. The transmission map comes from
TUM's measured depth, so the haze is not an approximation of scattering - it is
the scattering equation evaluated on a real sensor's depth. Verified: measured
contrast reduction tracks exp(-beta*d) to within 2 %.

Two questions, and they are different:

  1. Does WME degrade more gracefully **than the classical pipeline**? A
     descriptor pipeline should fall apart faster than a dense photometric one -
     that is the architecture's claim about why descriptors were dropped.
  2. Does the **fusion gain grow with degradation**, as 1 says? That is a claim
     about the tiers, not about WME-vs-baseline, and it needs the ablations.

Usage:
  python tools/bench_degrade.py [--out results/degrade] [--seq freiburg1_xyz ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# 콘솔이 cp949 라 em-dash 하나에 스크립트가 죽는다. 측정이 끝난 뒤 출력에서
# 죽는 것이 제일 아깝다 - 여기서 못박는다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

from fusion_eval import load_groundtruth  # noqa: E402
from bench_run import OPENCV_BIN, ROOT, find_build  # noqa: E402

# beta [1/m] 소산계수. 0 은 원본이다.
BETAS = [0.0, 0.15, 0.3, 0.5, 0.8, 1.5, 2.5, 4.0]

SYSTEMS = {
    "baseline": ("wme_tum_baseline.exe", [], "ORB + PnP (classical)"),
    "wme": ("wme_tum_odometry.exe", [], "WME ECDA (Tier 0)"),
}


def seq_path(seq: str, beta: float) -> Path:
    if beta == 0.0:
        return ROOT / "data" / f"rgbd_dataset_{seq}"
    # 생성기가 쓴 폴더 이름 규칙(뒤따르는 0 을 떼는 float 표기)을 그대로 따른다.
    tag = ("%g" % beta)
    return ROOT / "data" / "degraded" / f"haze{tag}_{seq}"


def run(exe: Path, seq: Path, traj: Path, diag: Path, extra: list[str]) -> bool:
    env = dict(os.environ)
    if OPENCV_BIN.exists():
        env["PATH"] = str(OPENCV_BIN) + os.pathsep + env.get("PATH", "")
    p = subprocess.run([str(exe), str(seq), str(traj), "--diag", str(diag), *extra],
                       capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    return p.returncode == 0 and traj.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/degrade")
    ap.add_argument("--seq", nargs="*", default=["freiburg1_xyz", "freiburg1_desk"])
    ap.add_argument("--skip-run", action="store_true",
                    help="재추정 없이 기존 궤적만 다시 채점한다")
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    build = find_build()
    report = {"betas": BETAS, "sequences": []}

    for seq in a.seq:
        gt_stamps, gt_poses = load_groundtruth(ROOT / "data" / f"rgbd_dataset_{seq}")
        gt = Trajectory(gt_stamps, gt_poses)
        entry = {"name": seq, "rows": []}
        print(f"\n=== {seq} ===")
        print(f"{'beta':>6} " + "".join(f"{SYSTEMS[k][2][:18]:>20}" for k in SYSTEMS)
              + f"{'floor':>9}")

        for beta in BETAS:
            sp = seq_path(seq, beta)
            if not (sp / "rgb.txt").exists():
                print(f"{beta:>6}  (없음: {sp.name})")
                continue
            row = {"beta": beta, "runs": {}}
            for key, (exe, extra, _) in SYSTEMS.items():
                tp = out / f"{seq}_b{beta:g}_{key}.txt"
                dp = out / f"{seq}_b{beta:g}_{key}_diag.csv"
                if not (a.skip_run and tp.exists()):
                    if not run(build / exe, sp, tp, dp, extra):
                        row["runs"][key] = None
                        continue
                est = load_trajectory(tp)
                try:
                    ate = float(evaluate_ate(est, gt, align=True).rmse) * 100.0
                except ValueError:
                    row["runs"][key] = None
                    continue
                lost = 0
                if dp.exists():
                    import csv
                    with open(dp, encoding="utf-8", errors="replace") as f:
                        for r in csv.DictReader(f):
                            if r.get("track_ok") == "0":
                                lost += 1
                row["runs"][key] = {"ate": ate, "lost": lost, "frames": len(est.poses)}
            # do-nothing 바닥값은 열화와 무관하다(궤적이 같으므로) - 그래도
            # 매번 다시 계산해 표가 자기 완결이 되게 둔다.
            any_r = next((v for v in row["runs"].values() if v), None)
            if any_r:
                est = load_trajectory(out / f"{seq}_b{beta:g}_wme.txt")
                ident = Trajectory(est.stamps, [SE3.identity() for _ in est.stamps])
                row["identity"] = float(evaluate_ate(ident, gt, align=True).rmse) * 100.0
            entry["rows"].append(row)
            cells = "".join(
                (f"{row['runs'][k]['ate']:>14.2f}cm" +
                 (f"/{row['runs'][k]['lost']:<4}" if row['runs'][k]['lost'] else "     ")
                 ) if row["runs"].get(k) else f"{'FAIL':>20}"
                for k in SYSTEMS)
            print(f"{beta:>6} {cells}{row.get('identity', float('nan')):>8.1f}")
        report["sequences"].append(entry)

    # --- 상대 열화율 -------------------------------------------------------
    # 절대 ATE 는 시퀀스마다 다르므로, 각 시스템을 자기 자신의 무열화 성능으로
    # 정규화해야 "누가 더 우아하게 무너지는가" 를 비교할 수 있다.
    print("\n" + "=" * 78)
    print("Degradation ratio  ATE(beta) / ATE(beta=0)   lower = degrades more gracefully")
    print("=" * 78)
    for e in report["sequences"]:
        base = {k: (e["rows"][0]["runs"].get(k) or {}).get("ate") for k in SYSTEMS}
        print(f"\n{e['name']}   (beta=0: " +
              ", ".join(f"{k} {base[k]:.2f}cm" for k in SYSTEMS if base[k]) + ")")
        print(f"{'beta':>6} " + "".join(f"{k:>14}" for k in SYSTEMS))
        for row in e["rows"]:
            cells = ""
            for k in SYSTEMS:
                r = row["runs"].get(k)
                cells += (f"{r['ate']/base[k]:>13.2f}x" if r and base.get(k)
                          else f"{'-':>14}")
            print(f"{row['beta']:>6} {cells}")

    # --- 실패 양상 ---------------------------------------------------------
    # 이 표가 없으면 위 표는 ORB 에게 유리하게 **오독된다**. 추적을 놓친 프레임은
    # 직전 포즈를 유지하므로, 대부분을 놓친 실행은 사실상 "카메라가 안 움직였다"
    # 가 되고 그 바닥값 근처 점수를 받는다. 즉 높은 haze 에서 ORB 의 낮은 ATE 는
    # 추적 성공이 아니라 **곱게 포기한 것**일 수 있다. 그 둘을 가르는 유일한
    # 방법은 놓친 비율과 바닥값을 같이 보는 것이다 (10.4 의 규칙).
    print("\n" + "=" * 78)
    print("Failure mode: is a low ATE tracking success, or a graceful give-up?")
    print("  lost% = frames where the front-end reported no track (pose frozen)")
    print("  vs floor = ATE / do-nothing ATE.  ~1.0 means it scored like a frozen camera.")
    print("=" * 78)
    for e in report["sequences"]:
        print(f"\n{e['name']}")
        print(f"{'beta':>6} " + "".join(f"{k+' lost%':>13}{'vs floor':>10}"
                                        for k in SYSTEMS))
        for row in e["rows"]:
            floor = row.get("identity") or float("nan")
            cells = ""
            for k in SYSTEMS:
                r = row["runs"].get(k)
                if not r:
                    cells += f"{'-':>13}{'-':>10}"
                    continue
                lost = 100.0 * r["lost"] / max(1, r["frames"])
                cells += f"{lost:>12.0f}%{r['ate']/floor:>9.2f}x"
            print(f"{row['beta']:>6} {cells}")

    p = out / "degrade.json"
    p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n저장: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
