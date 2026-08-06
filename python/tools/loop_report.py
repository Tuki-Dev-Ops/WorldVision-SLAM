"""Collect the loop-closure comparison into one JSON for the viewer.

Runs the shared back-end for both verification modes on the same sequence and
records, per mode: odometry ATE, post-optimisation ATE, how many loops were
accepted, and — the number that decides the whole thing — how accurate those
accepted loops actually were against ground truth.

A loop count without a loop *accuracy* is not a result: accepting many wrong
loops and accepting few right ones look identical in a count.

Usage:
  python tools/loop_report.py <sequence-name>
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3, quat_to_matrix, so3_log  # noqa: E402

from fusion_eval import load_groundtruth, interpolate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODES = {"orb": ("ORB descriptors", "orb"), "tcg": ("Object constellations", "tcg")}


def edge_accuracy(edges: Path, gt_stamps, gt_poses) -> dict:
    rows = [r for r in csv.DictReader(open(edges, encoding="utf-8", errors="replace"))
            if r.get("accepted") == "1"]
    te, re_ = [], []
    for r in rows:
        ga = interpolate(gt_stamps, gt_poses, float(r["stamp_i"]))
        gb = interpolate(gt_stamps, gt_poses, float(r["stamp_j"]))
        if ga is None or gb is None:
            continue
        truth = gb.inverse() @ ga
        meas = SE3(quat_to_matrix(float(r["qx"]), float(r["qy"]),
                                  float(r["qz"]), float(r["qw"])),
                   np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]))
        d = meas.inverse() @ truth
        te.append(float(np.linalg.norm(d.t)))
        re_.append(float(np.degrees(np.linalg.norm(so3_log(d.R)))))
    if not te:
        return {"n": len(rows), "scored": 0}
    te_a = np.array(te)
    return {
        "n": len(rows), "scored": len(te),
        "trans_median_cm": float(np.median(te_a)) * 100.0,
        "rot_median_deg": float(np.median(re_)),
        # 50 cm 넘게 틀린 것은 정밀도 문제가 아니라 **다른 장소**다. 이 둘을
        # 섞으면 "인식이 틀렸다" 와 "기하가 거칠다" 를 구분할 수 없다.
        "gross_false": int((te_a > 0.5).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", nargs="?", default="freiburg1_room")
    a = ap.parse_args()

    seq = ROOT / "data" / f"rgbd_dataset_{a.sequence}"
    loop = ROOT / "results" / "loop"
    gt_stamps, gt_poses = load_groundtruth(seq)
    gt = Trajectory(gt_stamps, gt_poses)

    out = {"sequence": a.sequence, "modes": {}}
    print(f"=== loop closure: {a.sequence} ===")
    print(f"{'mode':>6} {'odometry':>11} {'+loops':>11} {'change':>9} "
          f"{'loops':>7} {'edge err':>11} {'gross false':>12}")

    for key, (label, _) in MODES.items():
        odom_p = loop / f"fr1room_{'orb' if key == 'orb' else 'wme'}_odom.txt"
        edges_p = loop / f"fr1room_edges_{key}.csv"
        pgo_p = loop / f"fr1room_{key}_pgo.txt"
        if not (odom_p.exists() and edges_p.exists() and pgo_p.exists()):
            print(f"{key:>6}  (없음)")
            continue
        before = float(evaluate_ate(load_trajectory(odom_p), gt, align=True).rmse) * 100
        after = float(evaluate_ate(load_trajectory(pgo_p), gt, align=True).rmse) * 100
        acc = edge_accuracy(edges_p, gt_stamps, gt_poses)
        out["modes"][key] = {"label": label, "before": before, "after": after, **acc}
        print(f"{key:>6} {before:>10.2f}c {after:>10.2f}c "
              f"{(before-after)/before*100:>+8.1f}% {acc['n']:>7} "
              f"{acc.get('trans_median_cm', float('nan')):>10.1f}c "
              f"{acc.get('gross_false', 0)}/{acc.get('scored', 0):>10}")

    p = loop / "loop.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n저장: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
