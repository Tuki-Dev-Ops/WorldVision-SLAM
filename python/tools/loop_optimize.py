"""Pose-graph optimisation over loop edges — the shared back-end.

docs/06-results.md 22 compares front-ends: neither system had loop closure, so
the comparison could only claim odometry-vs-odometry. This closes that.

**The back-end is identical for both systems.** Same keyframe rule, same
candidate proposal, same factor graph, same robust kernel, same solver. The only
thing that differs is how a proposed loop is *verified*:

  orb : ORB descriptor matching + RANSAC PnP   (the classical answer)
  tcg : object-constellation matching + Kabsch (WME's Tier 1)

That is the point. If the back-end differed too, the final ATE difference could
not be attributed to recognition, and the experiment would measure nothing
(10.4).

Odometry edges get a tight information matrix and loop edges a looser one, both
fixed constants shared by the two modes — a per-mode weighting would be another
uncontrolled difference. Loop edges carry a Huber kernel because a single wrong
loop closure destroys a pose graph, and the whole question is whether the
recognition front-end produces wrong loops.

Usage:
  python tools/loop_optimize.py <sequence> <odometry.txt> <edges.csv> <out.txt>
"""

from __future__ import annotations

import argparse
import csv
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
from wme.graph.factors import BetweenPoseFactor, Huber, isotropic  # noqa: E402
from wme.graph.graph import FactorGraph, SolverOptions  # noqa: E402
from wme.graph.variables import PoseVariable  # noqa: E402
from wme.reference.geometry import SE3, quat_to_matrix  # noqa: E402

from fusion_eval import load_groundtruth  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# 두 모드가 공유하는 상수. 여기서 갈리면 비교가 인식 방법의 비교가 아니게 된다.
ODOM_SIGMA = 0.02      # m, 연속 키프레임 사이 상대 포즈
LOOP_SIGMA = 0.10      # m, 루프 간선은 더 느슨하게
HUBER_DELTA = 1.0      # 잘못된 루프 하나가 그래프를 부수는 것을 막는다


def read_edges(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [r for r in csv.DictReader(f)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("odometry")
    ap.add_argument("edges")
    ap.add_argument("out")
    ap.add_argument("--max-loop-trans", type=float, default=2.0,
                    help="이보다 큰 병진을 주장하는 루프는 버린다")
    a = ap.parse_args()

    seq = Path(a.sequence)
    odom = load_trajectory(Path(a.odometry))
    edges = read_edges(Path(a.edges))
    gt_stamps, gt_poses = load_groundtruth(seq)
    gt = Trajectory(gt_stamps, gt_poses)

    before = float(evaluate_ate(odom, gt, align=True).rmse) * 100.0

    # 그래프는 **키프레임 위에서** 푼다. 전체 프레임(수천 개)을 조밀 행렬로
    # 푸는 것은 비현실적이고, 루프 클로저가 고치는 것은 어차피 저주파 드리프트다.
    # 키프레임 목록은 C++ 도구가 남긴 것을 읽는다 - 여기서 규칙을 다시
    # 구현하면 두 구현이 갈라지는 순간 간선 인덱스가 조용히 어긋난다.
    kf_path = Path(str(a.edges) + ".keyframes.csv")
    if not kf_path.exists():
        print(f"키프레임 목록이 없다: {kf_path}")
        return 1
    with open(kf_path, encoding="utf-8") as f:
        kfs = [(int(r["kf"]), int(r["traj_idx"])) for r in csv.DictReader(f)]
    n = len(kfs)
    traj_idx = [t for _, t in kfs]

    g = FactorGraph()
    for k, ti in kfs:
        g.add_variable(f"x{k}", PoseVariable(odom.poses[ti]))
    g.fix("x0")                       # 게이지 고정

    # 오도메트리 간선: 추정 궤적의 키프레임 간 상대 포즈를 측정값으로 쓴다.
    odom_info = isotropic(6, ODOM_SIGMA)
    for k in range(n - 1):
        rel = odom.poses[traj_idx[k]].inverse() @ odom.poses[traj_idx[k + 1]]
        g.add_factor(BetweenPoseFactor(f"x{k}", f"x{k+1}", rel, odom_info, tier="odom"))

    loop_info = isotropic(6, LOOP_SIGMA)
    kernel = Huber(HUBER_DELTA)
    used = 0
    for e in edges:
        if e.get("accepted") != "1":
            continue
        i, j = int(e["i"]), int(e["j"])      # 키프레임 번호
        if not (0 <= i < n and 0 <= j < n):
            continue
        t = np.array([float(e["tx"]), float(e["ty"]), float(e["tz"])])
        if np.linalg.norm(t) > a.max_loop_trans or not np.all(np.isfinite(t)):
            continue
        R = quat_to_matrix(float(e["qx"]), float(e["qy"]),
                           float(e["qz"]), float(e["qw"]))
        # 도구는 T_b_a (a 좌표계 -> b 좌표계) 를 낸다. 팩터는 T_a_b 를 원한다.
        meas = SE3(R, t).inverse()
        g.add_factor(BetweenPoseFactor(f"x{i}", f"x{j}", meas, loop_info,
                                       kernel=kernel, tier="loop"))
        used += 1

    if used == 0:
        print("채택된 루프 간선이 없다 - 최적화해도 오도메트리 그대로다")
        Path(a.out).write_text(Path(a.odometry).read_text(encoding="utf-8"),
                               encoding="utf-8")
        print(f"ATE  before {before:.2f} cm   after {before:.2f} cm  (loops 0)")
        return 0

    res = g.optimize(SolverOptions(max_iterations=50))
    kf_poses = [g.value(f"x{k}").pose for k in range(n)]

    # 키프레임 보정을 전체 궤적에 되돌린다. 각 프레임은 자기 직전 키프레임에
    # **강체로** 붙어 따라간다: T_new = T_kf_new * (T_kf_old^-1 * T_frame_old).
    # 키프레임 사이를 보간하면 원래 오도메트리에 없던 운동을 지어내게 된다.
    poses = list(odom.poses)
    for k in range(n):
        lo = traj_idx[k]
        hi = traj_idx[k + 1] if k + 1 < n else len(poses)
        corr = kf_poses[k] @ odom.poses[lo].inverse()
        for t in range(lo, hi):
            poses[t] = corr @ odom.poses[t]
    for t in range(0, traj_idx[0]):          # 첫 키프레임 앞 구간
        poses[t] = (kf_poses[0] @ odom.poses[traj_idx[0]].inverse()) @ odom.poses[t]

    opt = Trajectory(odom.stamps, poses)
    after = float(evaluate_ate(opt, gt, align=True).rmse) * 100.0

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for s, p in zip(opt.stamps, opt.poses):
            q = _to_quat(p.R)
            f.write(f"{s:.6f} {p.t[0]:.6f} {p.t[1]:.6f} {p.t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"루프 간선 {used} / 제안 {sum(1 for e in edges)}   "
          f"수렴 {res.converged} cost {res.initial_cost:.3e} -> {res.final_cost:.3e}")
    print(f"ATE  before {before:.2f} cm   after {after:.2f} cm   "
          f"({(before-after)/before*100:+.1f} %)")
    return 0


def _to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


if __name__ == "__main__":
    raise SystemExit(main())
