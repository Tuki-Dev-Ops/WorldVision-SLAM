"""Score the three-tier fusion on real TUM data.

Estimation is C++ (`wme_tum_fusion`), scoring is here — the same split as
`tum_eval.py`. Four questions, in the order they have to be answered:

  1. Are the three tiers' information matrices on a comparable scale?
     Fusing by information magnitude is meaningless if one tier is
     overconfident by 10^4 (docs/06-results.md 15 and 10.4). Measured by
     ANEES against the 6-DOF chi-square band [5.5, 6.5].
  2. What is the ATE of every ablation, against Tier-0-only and against the
     "camera never moved" floor?
  3. Does a tier actually fill the direction another tier is missing?
     Measured as error along Tier 0's own reported weakest axis, not as a
     drop in total error.
  4. Does alpha_k(E) beat uniform weighting?

Usage:
  python tools/fusion_eval.py <sequence-root> <output-prefix>
  python tools/fusion_eval.py --all <data-root> <results-dir>
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3, quat_to_matrix, so3_exp, so3_log  # noqa: E402

ABLATIONS = ["t0", "t1", "t2", "t0t1", "t0t2", "t1t2", "t0t1t2", "t0t1t2_uniform"]
TIERS = ["t0", "t1", "t2"]
TIER_NAMES = {"t0": "ECDA", "t1": "TCG", "t2": "SPA"}


# ===========================================================================
# ground truth
# ===========================================================================

def load_groundtruth(root: Path) -> tuple[np.ndarray, list[SE3]]:
    stamps, poses = [], []
    with open(root / "groundtruth.txt", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            v = line.split()
            if len(v) < 8:
                continue
            stamps.append(float(v[0]))
            poses.append(SE3(quat_to_matrix(*(float(x) for x in v[4:8])),
                             np.array([float(x) for x in v[1:4]])))
    order = np.argsort(np.array(stamps))
    return np.array(stamps)[order], [poses[k] for k in order]


def gt_max_gap(stamps: np.ndarray) -> float:
    """How wide a ground-truth gap may be interpolated across, for THIS sequence.

    The old code had this as the constant 0.05 s, and the constant was not
    wrong — it was TUM-shaped. TUM's ground truth is motion capture at 100 Hz
    (measured median gap 0.010 s), so 0.05 s is "do not interpolate across more
    than about five sampling intervals". KITTI's ground truth is one pose per
    image, median gap 0.208 s, so *every* gap exceeds the constant, every
    interpolation returns None, every T_gt is None, and sections 1 and 4 print
    `n=0 ... nan` for all three tiers. Not "no signal" — no comparison
    attempted, reported as if measured. That is the failure this repo calls
    quiet, and it hid the tier calibration on all four KITTI sequences.

    So the rule is stated in the units it belongs to: the sequence's own
    sampling interval. 1.5 intervals is one step plus margin — beyond that the
    linear/geodesic interpolant is extrapolating through unobserved motion.
    The 0.05 s floor stays as a floor, which is why TUM is bit-identical:
    1.5 * 0.010 = 0.015 < 0.05.
    """
    if stamps.size < 3:
        return 0.05
    return max(0.05, 1.5 * float(np.median(np.diff(np.sort(stamps)))))


def interpolate(stamps: np.ndarray, poses: list[SE3], t: float,
                max_gap: float = 0.05) -> SE3 | None:
    """Linear in translation, geodesic in rotation. None if the gap is too wide."""
    if t <= stamps[0] or t >= stamps[-1]:
        return None
    j = int(np.searchsorted(stamps, t))
    i = j - 1
    t0, t1 = float(stamps[i]), float(stamps[j])
    if t1 - t0 > max_gap:
        return None
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    p0, p1 = poses[i], poses[j]
    R = p0.R @ so3_exp(a * so3_log(p0.R.T @ p1.R))
    return SE3(R, (1.0 - a) * p0.t + a * p1.t)


# ===========================================================================
# per-tier records
# ===========================================================================

@dataclass
class Record:
    idx: int
    stamp: float
    ref_idx: int
    ref_stamp: float
    alpha: np.ndarray                 # (3,)
    ok: dict[str, bool]
    reason: dict[str, str]
    T: dict[str, SE3]
    info: dict[str, np.ndarray]       # 6x6
    t0_weak: np.ndarray               # (6,)
    t2_weak: np.ndarray
    counts: dict[str, int]
    depth_incons: float = -1.0        # -1 = 판정 불가
    depth_rel: float = 1.0            # depthReliability() 0..1
    T_gt: SE3 | None = None           # ref camera -> cur camera


def _unpack_info(row: dict, prefix: str) -> np.ndarray:
    L = np.zeros((6, 6))
    for a in range(6):
        for b in range(a, 6):
            v = float(row[f"{prefix}_i{a}{b}"])
            L[a, b] = v
            L[b, a] = v
    return L


def _pose(row: dict, prefix: str) -> SE3:
    return SE3(quat_to_matrix(float(row[f"{prefix}_qx"]), float(row[f"{prefix}_qy"]),
                              float(row[f"{prefix}_qz"]), float(row[f"{prefix}_qw"])),
               np.array([float(row[f"{prefix}_tx"]), float(row[f"{prefix}_ty"]),
                         float(row[f"{prefix}_tz"])]))


def load_records(csv_path: Path, gt_stamps: np.ndarray, gt_poses: list[SE3]) -> list[Record]:
    out: list[Record] = []
    gap = gt_max_gap(gt_stamps)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = Record(
                idx=int(row["idx"]),
                stamp=float(row["stamp"]),
                ref_idx=int(row["ref_idx"]),
                ref_stamp=float(row["ref_stamp"]),
                alpha=np.array([float(row[f"alpha{k}"]) for k in range(3)]),
                ok={t: row[f"{t}_ok"] == "1" for t in TIERS},
                reason={t: row[f"{t}_reason"] for t in TIERS},
                T={t: _pose(row, t) for t in TIERS},
                info={t: _unpack_info(row, t) for t in TIERS},
                t0_weak=np.array([float(row[f"t0_weak{k}"]) for k in range(6)]),
                t2_weak=np.array([float(row[f"t2_weak{k}"]) for k in range(6)]),
                counts={k: int(row[k]) for k in ("tcg_ref_nodes", "tcg_cur_nodes",
                                                 "spa_ref_planes", "spa_cur_planes",
                                                 "t0_dof", "t2_dof")},
                # 25절의 기하 정합성. 옛 CSV 에는 없으므로 없으면 중립(1.0)이다 -
                # 0 으로 두면 옛 기록을 읽는 순간 Tier 0 가 통째로 꺼진다.
                depth_incons=float(row.get("t0_depth_incons", -1.0) or -1.0),
                depth_rel=float(row.get("t0_depth_rel", 1.0) or 1.0),
            )
            if r.ref_idx >= 0:
                g_cur = interpolate(gt_stamps, gt_poses, r.stamp, gap)
                g_ref = interpolate(gt_stamps, gt_poses, r.ref_stamp, gap)
                if g_cur is not None and g_ref is not None:
                    r.T_gt = g_cur.inverse() @ g_ref
            out.append(r)

    # 진리값과 짝지어지지 않은 프레임이 얼마인지 반드시 말한다. 이것이 없으면
    # 1절/4절의 n 이 왜 작은지 사후에 물을 방법이 없다.
    paired = sum(1 for r in out if r.T_gt is not None)
    usable = sum(1 for r in out if r.ref_idx >= 0)
    if paired < usable:
        print(f"  [gt] paired {paired}/{usable} frames at max_gap {gap:.3f} s"
              f"  ({usable - paired} unpaired, dropped from sections 1 and 4)")
    return out


# ===========================================================================
# 1. relative calibration — the prerequisite
# ===========================================================================

def tier_calibration(records: list[Record]) -> dict:
    """ANEES per tier, plus the raw information scale.

    A tier whose Lambda is k times too large is k times overconfident, and in
    a sum of information matrices it silently wins by that same factor. So the
    scale ratio between tiers is only meaningful next to the ANEES.
    """
    out = {}
    for t in TIERS:
        errs, nees, traces, terr, rerr = [], [], [], [], []
        for r in records:
            if not r.ok[t] or r.T_gt is None:
                continue
            L = r.info[t]
            if not np.all(np.isfinite(L)) or np.trace(L) <= 0:
                continue
            # left-perturbation convention, same as the information matrix
            e = (r.T[t] @ r.T_gt.inverse()).log()
            if not np.all(np.isfinite(e)):
                continue
            errs.append(e)
            nees.append(float(e @ L @ e))
            traces.append(float(np.trace(L)))
            d = r.T[t].inverse() @ r.T_gt
            terr.append(float(np.linalg.norm(d.t)))
            rerr.append(float(np.linalg.norm(so3_log(d.R))))
        n = len(nees)
        v = np.array(nees) if n else np.zeros(0)
        out[t] = {
            "n": n,
            "anees": float(np.mean(v)) if n else float("nan"),
            "median_nees": float(np.median(v)) if n else float("nan"),
            "p90_nees": float(np.percentile(v, 90)) if n else float("nan"),
            "p99_nees": float(np.percentile(v, 99)) if n else float("nan"),
            # 6 DOF chi-square 99.9 % gate. A tier can be well calibrated on
            # average and still inject a handful of confident-and-wrong frames,
            # and in a pose chain those accumulate. Mean ANEES hides that;
            # this fraction does not.
            "gate_fail": float(np.mean(v > 22.46)) if n else float("nan"),
            "trace_median": float(np.median(traces)) if n else float("nan"),
            "trans_err_median": float(np.median(terr)) if n else float("nan"),
            "rot_err_median": float(np.median(rerr)) if n else float("nan"),
        }
        # Bias fraction. ANEES cannot separate a zero-mean error from a
        # systematically signed one, but a pose chain can: only the signed part
        # accumulates into ATE. Under zero-mean noise this sits near
        # sqrt(6/n); well above that means the tier drifts.
        if n:
            E = np.array(errs)
            rms = float(np.sqrt((E ** 2).sum(axis=1).mean()))
            out[t]["bias_ratio"] = float(np.linalg.norm(E.mean(axis=0)) / max(rms, 1e-15))
            out[t]["bias_null"] = float(np.sqrt(6.0 / n))
        else:
            out[t]["bias_ratio"] = float("nan")
            out[t]["bias_null"] = float("nan")
        # kappa that would move ANEES onto 6.0. This is what to pass to
        # --kappaK if you want to ask "does fusion work once the scales agree".
        out[t]["kappa_to_6"] = (6.0 / out[t]["anees"]) if n and out[t]["anees"] > 0 \
            else float("nan")
    return out


# ===========================================================================
# 2. ATE
# ===========================================================================

def ate_table(root: Path, prefix: Path, gt: Trajectory) -> dict[str, float]:
    out: dict[str, float] = {}
    stamps = None
    for name in ABLATIONS:
        p = Path(f"{prefix}_{name}.txt")
        if not p.exists():
            continue
        est = load_trajectory(p)
        stamps = est.stamps
        out[name] = float(evaluate_ate(est, gt, align=True).rmse)
    if stamps is not None:
        identity = Trajectory(stamps, [SE3.identity() for _ in stamps])
        out["identity"] = float(evaluate_ate(identity, gt, align=True).rmse)
    return out


# ===========================================================================
# 3. degeneracy complementarity
# ===========================================================================

def _fill_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """(b_weak/a_weak) / (b_strong/a_strong) in A's eigenbasis. 1 = same shape."""
    w, v = np.linalg.eigh(a)
    if w[-1] <= 0:
        return float("nan")
    u_w, u_s = v[:, 0], v[:, -1]
    a_w, a_s = float(u_w @ a @ u_w), float(u_s @ a @ u_s)
    b_w, b_s = float(u_w @ b @ u_w), float(u_s @ b @ u_s)
    if a_w <= 0 or a_s <= 0 or b_s <= 0:
        return float("nan")
    return (b_w / a_w) / (b_s / a_s)


def relative_poses(prefix: Path, name: str, records: list[Record]) -> dict[int, SE3]:
    """Recover each ablation's fused relative pose from its trajectory.

    The keyframe schedule is shared by construction (Tier 0 picks it), so
    T_rel[i] = (P[ref]^-1 P[i])^-1 is exactly what fuse() returned.
    """
    p = Path(f"{prefix}_{name}.txt")
    if not p.exists():
        return {}
    traj = load_trajectory(p)
    poses = {i: traj.poses[i] for i in range(len(traj.poses))}
    out = {}
    for r in records:
        if r.ref_idx < 0 or r.idx >= len(poses):
            continue
        out[r.idx] = (poses[r.ref_idx].inverse() @ poses[r.idx]).inverse()
    return out


def complementarity(records: list[Record], prefix: Path) -> dict:
    """Is the direction that got filled the direction that was missing?

    Two independent readings:
      (a) shape:  fill_ratio of Lambda_B into Lambda_A's eigenbasis.
      (b) error:  |e . u_weak| for Tier-0-only vs the fused run, where u_weak
                  is Tier 0's *own* reported weakest axis. If fusion helps for
                  the claimed reason, the error must fall preferentially along
                  that axis rather than uniformly.
    """
    fills = {"t2_into_t0": [], "t0_into_t2": [], "t1_into_t0": []}
    for r in records:
        if r.ok["t0"] and r.ok["t2"]:
            fills["t2_into_t0"].append(_fill_ratio(r.info["t0"], r.info["t2"]))
            fills["t0_into_t2"].append(_fill_ratio(r.info["t2"], r.info["t0"]))
        if r.ok["t0"] and r.ok["t1"]:
            fills["t1_into_t0"].append(_fill_ratio(r.info["t0"], r.info["t1"]))

    rel = {n: relative_poses(prefix, n, records) for n in ("t0", "t0t1", "t0t2", "t0t1t2")}

    axes = {}
    # Restrict to frames where Tier 0 *reports* a rank deficiency. Everywhere
    # else "weakest direction" only names the least-constrained axis of a
    # full-rank system, and filling it is not the claim under test.
    degenerate_only = [r for r in records
                       if r.ref_idx >= 0 and r.counts["t0_dof"] < 6]
    for name in ("t0t1", "t0t2", "t0t1t2"):
        weak_base, weak_fused, orth_base, orth_fused = [], [], [], []
        trans_base, trans_fused = [], []
        for r in records:
            if r.T_gt is None or r.idx not in rel["t0"] or r.idx not in rel[name]:
                continue
            u = r.t0_weak
            nu = np.linalg.norm(u)
            if nu < 1e-9:
                continue
            u = u / nu
            e0 = (rel["t0"][r.idx] @ r.T_gt.inverse()).log()
            ef = (rel[name][r.idx] @ r.T_gt.inverse()).log()
            if not (np.all(np.isfinite(e0)) and np.all(np.isfinite(ef))):
                continue
            weak_base.append(abs(float(e0 @ u)))
            weak_fused.append(abs(float(ef @ u)))
            orth_base.append(float(np.linalg.norm(e0 - (e0 @ u) * u)))
            orth_fused.append(float(np.linalg.norm(ef - (ef @ u) * u)))
            trans_base.append(float(np.linalg.norm(
                (rel["t0"][r.idx].inverse() @ r.T_gt).t)))
            trans_fused.append(float(np.linalg.norm(
                (rel[name][r.idx].inverse() @ r.T_gt).t)))
        if weak_base:
            axes[name] = {
                "n": len(weak_base),
                "weak_base": float(np.median(weak_base)),
                "weak_fused": float(np.median(weak_fused)),
                "orth_base": float(np.median(orth_base)),
                "orth_fused": float(np.median(orth_fused)),
                # ATE integrates the chain, so a few catastrophic frames matter
                # more than the typical one. Median and p95 can move opposite
                # ways, and when they do the median is the misleading number.
                "trans_med_base": float(np.median(trans_base)),
                "trans_med_fused": float(np.median(trans_fused)),
                "trans_p95_base": float(np.percentile(trans_base, 95)),
                "trans_p95_fused": float(np.percentile(trans_fused, 95)),
            }
            a = axes[name]
            a["weak_gain"] = a["weak_base"] / max(a["weak_fused"], 1e-15)
            a["orth_gain"] = a["orth_base"] / max(a["orth_fused"], 1e-15)
            # > 1 means the improvement is concentrated on the axis Tier 0
            # said it could not see, which is the claim under test.
            a["selectivity"] = a["weak_gain"] / max(a["orth_gain"], 1e-15)

    deg = {}
    for name in ("t0t2", "t0t1t2"):
        wb, wf = [], []
        for r in degenerate_only:
            if r.T_gt is None or r.idx not in rel["t0"] or r.idx not in rel[name]:
                continue
            u = r.t0_weak
            nu = np.linalg.norm(u)
            if nu < 1e-9:
                continue
            u = u / nu
            e0 = (rel["t0"][r.idx] @ r.T_gt.inverse()).log()
            ef = (rel[name][r.idx] @ r.T_gt.inverse()).log()
            if not (np.all(np.isfinite(e0)) and np.all(np.isfinite(ef))):
                continue
            wb.append(abs(float(e0 @ u)))
            wf.append(abs(float(ef @ u)))
        if wb:
            deg[name] = {"n": len(wb), "base": float(np.median(wb)),
                         "fused": float(np.median(wf)),
                         "gain": float(np.median(wb)) / max(float(np.median(wf)), 1e-15)}

    return {
        "fill": {k: (float(np.nanmedian(v)) if v else float("nan"))
                 for k, v in fills.items()},
        "fill_n": {k: len(v) for k, v in fills.items()},
        "axis": axes,
        "degenerate": deg,
        "degenerate_frames": len(degenerate_only),
    }


# ===========================================================================
# report
# ===========================================================================

def contribution_rates(records: list[Record]) -> dict:
    out = {}
    total = sum(1 for r in records if r.ref_idx >= 0)
    for t in TIERS:
        reasons: dict[str, int] = {}
        used = 0
        for r in records:
            if r.ref_idx < 0:
                continue
            if r.ok[t]:
                used += 1
            else:
                reasons[r.reason[t]] = reasons.get(r.reason[t], 0) + 1
        out[t] = {"used": used, "total": total,
                  "rate": used / total if total else 0.0, "reasons": reasons}
    return out


def report(name: str, root: Path, prefix: Path) -> dict:
    gt_stamps, gt_poses = load_groundtruth(root)
    gt = Trajectory(gt_stamps, gt_poses)
    records = load_records(Path(f"{prefix}_tiers.csv"), gt_stamps, gt_poses)

    res = {
        "sequence": name,
        "frames": len(records),
        "calibration": tier_calibration(records),
        "ate": ate_table(root, prefix, gt),
        "contribution": contribution_rates(records),
        "complementarity": complementarity(records, prefix),
    }

    print("=" * 92)
    print(f"{name}   frames {len(records)}")
    print("=" * 92)

    print("\n[1] Tier information calibration  (ANEES target [5.5, 6.5] for 6 DOF)")
    print(f"{'tier':>14} {'n':>6} {'ANEES':>12} {'medNEES':>10} {'p99':>11} "
          f"{'gatefail':>9} {'tr(L) med':>11} {'trans':>10} {'rot':>8} {'kappa->6':>10}")
    for t in TIERS:
        c = res["calibration"][t]
        print(f"{TIER_NAMES[t]:>14} {c['n']:>6} {c['anees']:>12.3e} "
              f"{c['median_nees']:>10.2e} {c['p99_nees']:>11.2e} "
              f"{c['gate_fail']*100:>8.1f}% {c['trace_median']:>11.2e} "
              f"{c['trans_err_median']*100:>8.2f}cm {c['rot_err_median']:>8.4f} "
              f"{c['kappa_to_6']:>10.3e}")
    print(f"{'':>14} bias |mean e|/rms  " + "  ".join(
        f"{TIER_NAMES[t]} {res['calibration'][t]['bias_ratio']:.3f}"
        f" (null {res['calibration'][t]['bias_null']:.3f})" for t in TIERS))
    ks = [res["calibration"][t]["kappa_to_6"] for t in TIERS]
    print(f"{'':>14} replay with:  --kappa0 {ks[0]:.4g} --kappa1 {ks[1]:.4g} "
          f"--kappa2 {ks[2]:.4g}")

    print("\n[2] ATE RMSE (cm)")
    base = res["ate"].get("t0", float("nan"))
    for k in [*ABLATIONS, "identity"]:
        if k not in res["ate"]:
            continue
        v = res["ate"][k]
        gain = (base - v) / base * 100.0 if np.isfinite(base) and base > 0 else float("nan")
        tag = "  <- baseline" if k == "t0" else ("  <- do-nothing floor" if k == "identity" else "")
        print(f"{k:>18} {v*100:>9.2f}   vs T0 {gain:+7.1f} %{tag}")

    print("\n[3] Per-frame tier contribution")
    for t in TIERS:
        c = res["contribution"][t]
        rs = ", ".join(f"{k} {v}" for k, v in sorted(c["reasons"].items(),
                                                     key=lambda kv: -kv[1]))
        print(f"{TIER_NAMES[t]:>14} {c['used']:>5}/{c['total']:<5} = {c['rate']*100:5.1f} %"
              f"   abstain: {rs or '-'}")
    n = [r for r in records if r.ref_idx >= 0]
    if n:
        print(f"{'':>14} median objects/frame {np.median([r.counts['tcg_cur_nodes'] for r in n]):.1f}"
              f"   planes/frame {np.median([r.counts['spa_cur_planes'] for r in n]):.1f}"
              f"   T0 dof {np.mean([r.counts['t0_dof'] for r in n]):.2f}"
              f"   T2 dof {np.mean([r.counts['t2_dof'] for r in n]):.2f}")

    print("\n[4] Degeneracy complementarity")
    f = res["complementarity"]["fill"]
    fn = res["complementarity"]["fill_n"]
    for k, v in f.items():
        print(f"{k:>18}  median fill_ratio {v:>12.3e}  (n={fn[k]})   "
              f"{'complementary' if v > 1.0 else 'not complementary'}")
    for k, a in res["complementarity"]["axis"].items():
        print(f"{k:>18}  along T0 weak axis {a['weak_base']:.3e} -> {a['weak_fused']:.3e} "
              f"({a['weak_gain']:.3f}x)   orthogonal {a['orth_gain']:.3f}x   "
              f"selectivity {a['selectivity']:.3f}")
        print(f"{'':>18}  rel trans err  median {a['trans_med_base']*100:.2f} -> "
              f"{a['trans_med_fused']*100:.2f} cm   "
              f"p95 {a['trans_p95_base']*100:.2f} -> {a['trans_p95_fused']*100:.2f} cm")
    nd = res["complementarity"]["degenerate_frames"]
    print(f"{'':>18}  frames where T0 reports rank < 6: {nd}")
    for k, d in res["complementarity"]["degenerate"].items():
        print(f"{k:>18}  [rank-deficient frames only, n={d['n']}] weak-axis error "
              f"{d['base']:.3e} -> {d['fused']:.3e}  ({d['gain']:.3f}x)")

    print("\n[5] alpha_k(E) vs uniform")
    if "t0t1t2" in res["ate"] and "t0t1t2_uniform" in res["ate"]:
        e, u = res["ate"]["t0t1t2"], res["ate"]["t0t1t2_uniform"]
        print(f"      env-scheduled {e*100:.2f} cm   uniform {u*100:.2f} cm   "
              f"{'schedule wins' if e < u else 'uniform wins' if u < e else 'identical'}"
              f"  ({(u - e) / max(u, 1e-12) * 100:+.2f} %)")
        a = np.array([r.alpha for r in records if r.ref_idx >= 0])
        if len(a):
            print(f"      alpha ranges: a0 [{a[:, 0].min():.3f}, {a[:, 0].max():.3f}]  "
                  f"a1 [{a[:, 1].min():.3f}, {a[:, 1].max():.3f}]  "
                  f"a2 [{a[:, 2].min():.3f}, {a[:, 2].max():.3f}]")
    print()
    return res


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 3 and argv[0] == "--all":
        data, results = Path(argv[1]), Path(argv[2])
        for d in sorted(data.glob("rgbd_dataset_*")):
            prefix = results / d.name
            if not Path(f"{prefix}_tiers.csv").exists():
                continue
            report(d.name.replace("rgbd_dataset_", ""), d, prefix)
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    report(Path(argv[0]).name, Path(argv[0]), Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
