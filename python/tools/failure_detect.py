"""Does any observable signal detect Tier 0's silent failures?

This is the open item 25 calls the most valuable next step, and it comes from two
measurements that agree:

  13.3 — on `walking_xyz` the estimator is 20x worse and **every photometric
         self-assessment signal sits at lift <= 1.12**. "No amount of tuning the
         photometric confidence path can fix this, because the information is
         not in the photometric channel."
  23.3 — under haze WME drifts to 11.6x the do-nothing floor while reporting
         **zero** failed frames. It does not merely fail; it fails silently.

So the search is for a signal in a *different channel*. `depth_incons` is that
candidate: project ref's 3D points into cur with the estimated pose and compare
against the depth cur actually measured. ECDA never reads cur's depth — it uses
ref's depth for geometry and cur's *intensity* for the residual — so this is an
independent observation, not a restatement of the thing that already failed.
Haze degrades the optics and leaves the IR depth sensor alone (23.1), which is
exactly the regime where the photometric channel goes blind.

Reported as **lift** = P(bad frame | flagged) / P(bad frame), flagged = the
signal's worst decile, bad = the true error's worst decile. Lift 1.0 means the
signal carries no information. Lift is used rather than recall because 13.4
showed recall scoring 100 % for a signal that was constant.

Usage:
  python tools/failure_detect.py <sequence-root> <estimate.txt> <diag.csv>
  python tools/failure_detect.py --all
"""

from __future__ import annotations

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

from wme.eval.tum import load_trajectory  # noqa: E402
from wme.reference.geometry import SE3, so3_log  # noqa: E402

from fusion_eval import load_groundtruth, interpolate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# 신호 이름 -> (열, 클수록 나쁜가)
SIGNALS = {
    "photometric RMSE":  ("rmse", True),
    "condition number":  ("cond", True),
    "1 - inlier ratio":  ("inlier_ratio", False),
    "6 - observable DOF": ("observable_dof", False),
    "1 / point count":   ("points", False),
    "depth inconsistency": ("depth_incons", True),
    "depth outlier frac":  ("depth_outlier", True),
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def lift(sig: np.ndarray, err: np.ndarray, q: float = 0.9) -> float:
    """P(bad | flagged) / P(bad).

    상수 신호는 문턱이 모든 표본을 통과시키므로 lift 가 정확히 1.0 이 된다 -
    13.4 에서 recall 이 100 % 를 보고했던 그 경우가 여기서는 "정보 없음" 으로
    올바르게 나온다.
    """
    n = len(sig)
    if n < 20:
        return float("nan")
    bad = err >= np.quantile(err, q)
    flagged = sig >= np.quantile(sig, q)
    if flagged.sum() == 0 or bad.mean() == 0:
        return float("nan")
    return float((bad & flagged).sum() / flagged.sum() / bad.mean())


def analyse(root: Path, est_p: Path, diag_p: Path) -> dict | None:
    rows = list(csv.DictReader(open(diag_p, encoding="utf-8", errors="replace")))
    if not rows:
        return None
    gs, gp = load_groundtruth(root)

    err, cols = [], {k: [] for k in SIGNALS}
    for r in rows:
        try:
            cur = float(r["timestamp"]); ref = float(r["ref_timestamp"])
            rel = SE3(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]), np.zeros(3))
            from wme.reference.geometry import quat_to_matrix
            rel = SE3(quat_to_matrix(float(r["rel_qx"]), float(r["rel_qy"]),
                                     float(r["rel_qz"]), float(r["rel_qw"])),
                      np.array([float(r["rel_tx"]), float(r["rel_ty"]),
                                float(r["rel_tz"])]))
        except (KeyError, ValueError):
            continue
        gc = interpolate(gs, gp, cur); gr = interpolate(gs, gp, ref)
        if gc is None or gr is None:
            continue
        truth = gc.inverse() @ gr                      # T_cur_ref
        d = rel.inverse() @ truth
        if not np.all(np.isfinite(d.t)):
            continue
        # **병진과 회전을 하나로 섞지 않는다.** 처음에는 `t + 2*rot` 로 합쳤는데,
        # 그 2.0 이 결론을 정했다: 병진만 보면 depth_incons 가 3.42 로 측광
        # RMSE(2.66)를 이기고, 섞으면 3.80 대 2.66 으로 뒤집힌다. 임의로 고른
        # 가중치가 어느 신호가 이기는지를 결정하면 그것은 측정이 아니다(10.4).
        # 두 축을 따로 보고하고, 서로 다른 채널이 서로 다른 축을 잡는다는
        # 사실 자체를 결과로 남긴다.
        e = (float(np.linalg.norm(d.t)), float(np.linalg.norm(so3_log(d.R))))

        vals = {}
        okrow = True
        for name, (col, bigbad) in SIGNALS.items():
            try:
                v = float(r[col])
            except (KeyError, ValueError):
                okrow = False; break
            if col == "depth_incons" or col == "depth_outlier":
                if v < 0:            # 표본 부족 - 이 프레임은 신호가 없다
                    okrow = False; break
            if not bigbad:
                v = (6.0 - v) if col == "observable_dof" else \
                    ((1.0 - v) if col == "inlier_ratio" else 1.0 / max(v, 1e-9))
            vals[name] = v
        if not okrow:
            continue
        err.append(e)
        for k, v in vals.items():
            cols[k].append(v)

    if len(err) < 20:
        return None
    te = np.array([e[0] for e in err])
    re_ = np.array([e[1] for e in err])
    return {"n": len(err),
            "signals": {k: {"lift_t": lift(np.array(v), te),
                            "lift_r": lift(np.array(v), re_),
                            "rho_t": spearman(np.array(v), te)}
                        for k, v in cols.items()}}


def main() -> int:
    argv = sys.argv[1:]
    jobs = []
    if argv and argv[0] == "--all":
        d = ROOT / "results" / "selfassess"
        for diag in sorted(d.glob("*_diag.csv")):
            name = diag.name.replace("_diag.csv", "")
            root = ROOT / "data" / f"rgbd_dataset_{name}"
            if root.exists():
                jobs.append((name, root, d / f"{name}.txt", diag))
    elif len(argv) >= 3:
        jobs.append((Path(argv[0]).name, Path(argv[0]), Path(argv[1]), Path(argv[2])))
    else:
        print(__doc__)
        return 2

    names = list(SIGNALS)
    table = {}
    for name, root, est, diag in jobs:
        res = analyse(root, est, diag)
        table[name] = res["signals"] if res else None

    print("lift = P(bad frame | flagged) / P(bad frame).  1.0 = no information.")
    for key, title in (("lift_t", "TRANSLATION error"), ("lift_r", "ROTATION error")):
        print(f"\n=== predicting {title} ===")
        print(f"{'signal':>22} " + "".join(f"{n[:16]:>18}" for n, *_ in jobs))
        for sig in names:
            row = ""
            for name, *_ in jobs:
                t = table.get(name)
                row += f"{t[sig][key]:>17.2f}" if t else f"{'-':>18}"
            star = "  <-- independent channel" if sig.startswith("depth") else ""
            print(f"{sig:>22} {row}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
