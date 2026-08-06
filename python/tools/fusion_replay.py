"""Offline replay of three-tier fusion from the recorded tier CSVs.

`tools/tum_fusion.cpp` records every tier's `(T, Lambda)` per frame before any
weighting is applied, so the fusion itself can be re-run in numpy for an
*arbitrary* weight rule without touching a camera or the C++ engine. That is
what makes fitting `alpha_k` possible at all: docs/06-results.md 18.4 shows the
hand-designed schedule losing to uniform on 6 of 7 runs, and nothing in the
project has ever fitted it.

This file is the port of `wme::fusion::fuse` and nothing else. It is only
trustworthy to the extent it reproduces the C++ trajectories bit-for-bit-ish,
so `--validate` is not optional decoration - it is the gate that 10.4 demands
before any number produced here means anything.

Usage:
  python tools/fusion_replay.py --validate <data-root> <results-dir>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3, skew  # noqa: E402

from fusion_eval import (  # noqa: E402
    ABLATIONS,
    TIERS,
    Record,
    load_groundtruth,
    load_records,
)

# ===========================================================================
# port of wme::fusion  (src/fusion/PoseFusion.cpp)
# ===========================================================================

MAX_ITERATIONS = 10
CONVERGENCE_DELTA = 1e-10
# 갱신 절단은 수치 안전장치일 뿐이다. 퇴화 보고 문턱(1e-3)과 같은 값을 쓰면
# tier 사이 10^6 배 스케일 차이 때문에 약한 tier 의 기여가 통째로 버려진다.
UPDATE_OBSERVABLE_RATIO = 1e-12


def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _adjoint_algebra(xi: np.ndarray) -> np.ndarray:
    """ad(xi) = [[skew(phi), skew(rho)], [0, skew(phi)]],  xi = [rho, phi]."""
    P, W = skew(xi[:3]), skew(xi[3:])
    A = np.zeros((6, 6))
    A[:3, :3] = W
    A[:3, 3:] = P
    A[3:, 3:] = W
    return A


def se3_left_jacobian(xi: np.ndarray) -> np.ndarray:
    """J_l = sum_{n>=0} ad(xi)^n / (n+1)!  - the same truncated series as C++."""
    A = _adjoint_algebra(xi)
    J = np.eye(6)
    term = np.eye(6)
    fact = 1.0
    for n in range(1, 31):
        term = term @ A
        fact *= float(n + 1)
        add = term / fact
        J = J + add
        if np.linalg.norm(add) < 1e-18 * max(1.0, float(np.linalg.norm(J))):
            break
    return J


def se3_left_jacobian_inverse(xi: np.ndarray) -> np.ndarray:
    return np.linalg.inv(se3_left_jacobian(xi))


def _analyse(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of a symmetric matrix, descending - C++ ordering."""
    w, v = np.linalg.eigh(_symmetrize(m))
    return w[::-1].copy(), v[:, ::-1].copy()


def _renormalize(T: SE3) -> SE3:
    """C++ keeps SO(3) as a quaternion and calls q.normalize(); the closest
    equivalent for a matrix is the nearest orthogonal matrix."""
    u, _, vt = np.linalg.svd(T.R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt
    return SE3(R, T.t)


@dataclass
class FusionOut:
    ok: bool
    T_cur_ref: SE3
    information: np.ndarray
    used: list[bool]
    iterations: int = 0


def fuse(estimates: list[tuple[bool, SE3, np.ndarray, float]]) -> FusionOut:
    """estimates: (available, T_cur_ref, Lambda, weight) per tier, in tier order.

    `weight` is alpha*kappa already multiplied out - the C++ does the same and
    never uses the two separately inside fuse().
    """
    active: list[tuple[int, np.ndarray]] = []
    used = [False, False, False]
    for i, (available, _T, info, w) in enumerate(estimates):
        if not available:
            continue
        if not (w > 0.0):
            continue
        if not np.all(np.isfinite(info)) or np.trace(info) <= 0.0:
            continue
        active.append((i, _symmetrize(w * info)))
        used[i] = True

    if not active:
        return FusionOut(False, SE3.identity(), np.zeros((6, 6)), used)

    # 초기값은 정보가 가장 큰 tier. 랭크 부족한 tier 에서 출발하면 그 널공간을 헤맨다.
    seed = max(range(len(active)), key=lambda k: float(np.trace(active[k][1])))
    T = estimates[active[seed][0]][1]

    iterations = 0
    for it in range(MAX_ITERATIONS):
        H = np.zeros((6, 6))
        g = np.zeros(6)
        for idx, lam in active:
            eps = (T @ estimates[idx][1].inverse()).log()
            J = se3_left_jacobian_inverse(eps)
            JtL = J.T @ lam
            H += JtL @ J
            g += JtL @ eps
        H = _symmetrize(H)

        vals, vecs = _analyse(H)
        if vals[0] <= 0.0:
            break
        thresh = vals[0] * UPDATE_OBSERVABLE_RATIO
        delta = np.zeros(6)
        for k in range(6):
            if vals[k] <= thresh:
                continue
            delta -= (float(vecs[:, k] @ g) / vals[k]) * vecs[:, k]
        if not np.all(np.isfinite(delta)):
            break

        T = _renormalize(SE3.exp(delta) @ T)
        iterations = it + 1
        if float(np.linalg.norm(delta)) < CONVERGENCE_DELTA:
            break

    total = np.zeros((6, 6))
    for idx, lam in active:
        eps = (T @ estimates[idx][1].inverse()).log()
        J = se3_left_jacobian_inverse(eps)
        total += J.T @ lam @ J

    return FusionOut(True, T, _symmetrize(total), used, iterations)


# ===========================================================================
# replay
# ===========================================================================

def fuse_frames(records: list[Record], weight_fn,
                kappa=(1.0, 1.0, 1.0)) -> list[FusionOut | None]:
    """Per-frame fusion under an arbitrary weight rule, without integrating.

    Split out from `replay` so a caller sweeping many rules pays the fusion
    cost once per (frame, rule) and can then integrate any *selection* over
    those results for free.

    `weight_fn(record) -> (w0, w1, w2)`; a zero disables that tier for that
    frame, which is exactly how the C++ ablations are expressed.
    """
    out: list[FusionOut | None] = []
    for r in records:
        if r.ref_idx < 0:
            out.append(None)
            continue
        w = weight_fn(r)
        est = [(r.ok[t] and w[k] > 0.0, r.T[t], r.info[t], w[k] * kappa[k])
               for k, t in enumerate(TIERS)]
        out.append(fuse(est))
    return out


def integrate(records: list[Record], frames: list[FusionOut | None]) -> Trajectory:
    """Chain per-frame relative poses into a trajectory.

    Fusion failure holds the previous pose rather than resetting to the origin
    - matching tum_fusion.cpp, because a reset spikes the trajectory and
    overstates the failure in ATE.
    """
    poses: list[SE3] = [SE3.identity() for _ in records]
    for i, r in enumerate(records):
        if r.ref_idx < 0:
            continue
        fr = frames[i]
        if fr is not None and fr.ok:
            poses[i] = poses[r.ref_idx] @ fr.T_cur_ref.inverse()
        else:
            poses[i] = poses[i - 1] if i > 0 else SE3.identity()
    return Trajectory(np.array([r.stamp for r in records]), poses)


def replay(records: list[Record], weight_fn, kappa=(1.0, 1.0, 1.0)) -> Trajectory:
    """Integrate a trajectory under an arbitrary per-frame weight rule."""
    return integrate(records, fuse_frames(records, weight_fn, kappa))


def ablation_weights(mask: tuple[bool, bool, bool], mode: str):
    def fn(r: Record):
        return tuple((1.0 if mode == "uniform" else float(r.alpha[k])) if mask[k] else 0.0
                     for k in range(3))
    return fn


ABLATION_MASKS = {
    "t0": ((True, False, False), "env"),
    "t1": ((False, True, False), "env"),
    "t2": ((False, False, True), "env"),
    "t0t1": ((True, True, False), "env"),
    "t0t2": ((True, False, True), "env"),
    "t1t2": ((False, True, True), "env"),
    "t0t1t2": ((True, True, True), "env"),
    "t0t1t2_uniform": ((True, True, True), "uniform"),
}


def ate_cm(traj: Trajectory, gt: Trajectory) -> float:
    return float(evaluate_ate(traj, gt, align=True).rmse) * 100.0


# ===========================================================================
# validation - the gate
# ===========================================================================

def validate(data_root: Path, results_dir: Path) -> int:
    """Reproduce every C++ ablation trajectory from the recorded tiers.

    Two things are checked, and the second is the one 10.4 asks for:
      1. agreement with the C++ ATE for all 8 ablations;
      2. that the replay *can* disagree - a deliberately corrupted weight must
         move the number, or agreement means the replay is ignoring its input.
    """
    seqs = sorted(p for p in data_root.glob("rgbd_dataset_*")
                  if (results_dir / f"{p.name}_tiers.csv").exists())
    if not seqs:
        print(f"tiers.csv 가 없다: {results_dir}")
        return 2

    worst = 0.0
    rows: list[tuple[str, str, float, float, float]] = []
    guard_fail = 0

    for root in seqs:
        prefix = results_dir / root.name
        gt_stamps, gt_poses = load_groundtruth(root)
        gt = Trajectory(gt_stamps, gt_poses)
        records = load_records(Path(f"{prefix}_tiers.csv"), gt_stamps, gt_poses)

        for name in ABLATIONS:
            cpp_path = Path(f"{prefix}_{name}.txt")
            if not cpp_path.exists():
                continue
            cpp_ate = ate_cm(load_trajectory(cpp_path), gt)
            mask, mode = ABLATION_MASKS[name]
            py_ate = ate_cm(replay(records, ablation_weights(mask, mode)), gt)
            rel = abs(py_ate - cpp_ate) / max(cpp_ate, 1e-12) * 100.0
            worst = max(worst, rel)
            rows.append((root.name.replace("rgbd_dataset_", ""), name,
                         cpp_ate, py_ate, rel))

        # 판별력 확인: 일부러 틀린 가중치는 반드시 다른 숫자를 내야 한다.
        base = ate_cm(replay(records, ablation_weights(*ABLATION_MASKS["t0t1t2"])), gt)
        bad = ate_cm(replay(records, lambda r: (1.0, 1e6, 1e6)), gt)
        if abs(bad - base) / max(base, 1e-12) < 0.01:
            guard_fail += 1
            print(f"  판별 실패: {root.name} 가중치를 바꿔도 ATE 가 그대로다")

    print("=" * 84)
    print("fusion replay validation   (C++ tum_fusion vs numpy replay)")
    print("=" * 84)
    print(f"{'sequence':>26} {'ablation':>16} {'C++ (cm)':>11} {'replay (cm)':>12} {'rel':>8}")
    for seq, name, c, p, rel in rows:
        flag = "" if rel < 1.0 else "   <-- MISMATCH"
        print(f"{seq:>26} {name:>16} {c:>11.2f} {p:>12.2f} {rel:>7.3f}%{flag}")
    print(f"\nworst relative disagreement: {worst:.3f} %   over {len(rows)} trajectories")
    print(f"discrimination guard: {len(seqs) - guard_fail}/{len(seqs)} sequences respond "
          f"to a changed weight")
    ok = worst < 1.0 and guard_fail == 0
    print("\nGATE:", "PASS - replay reproduces the engine" if ok else "FAIL - do not trust "
          "anything fitted on this replay")
    return 0 if ok else 1


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 3 and argv[0] == "--validate":
        return validate(Path(argv[1]), Path(argv[2]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
