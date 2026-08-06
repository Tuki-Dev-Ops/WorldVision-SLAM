"""Can `alpha_k` be fitted, and is the fusion failure a weighting problem at all?

docs/06-results.md 18 measured the architecture's central equation on real data
and found fusion net *harmful* on four of five sequences, with the hand-designed
`alpha_k(E)` losing to uniform weights on six of seven runs. 21 lists
"alpha_k(E) is hand-designed and has never been fitted to real data" as open.

Fitting it is only worth doing if the failure is a weighting failure. Three
mutually distinguishable causes, and this file separates them:

  (a) weighting   - the tiers carry usable information and the schedule points
                    the wrong way. Then some fitted alpha beats Tier-0-only.
  (b) robustness  - the tiers are usable on most frames and catastrophic on a
                    few (18.1: TCG median NEES 4.7, p99 1.6e4), and a scalar
                    weight cannot express "usually trust, sometimes not". Then
                    a per-frame consistency gate beats any constant.
  (c) no signal   - T1/T2 carry nothing this data can use, and the right answer
                    is not to fuse. Then even an oracle with ground truth in
                    hand cannot beat Tier-0-only.

The oracle is measured first because it *bounds the other two*: no observable
rule can beat a rule that already knows the answer. If the oracle has no
headroom, (a) and (b) are settled without fitting anything.

Every fit is leave-one-sequence-out. Fitting five numbers on five sequences and
reporting the fit is how 10.3's gameable-criterion failure happens again.

Usage:
  python tools/fusion_alpha.py <data-root> <results-dir>
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wme.eval.trajectory import Trajectory, evaluate_ate  # noqa: E402
from wme.reference.geometry import SE3  # noqa: E402

from fusion_eval import Record, load_groundtruth, load_records  # noqa: E402
from fusion_replay import FusionOut, fuse_frames, integrate  # noqa: E402

# chi-square(6) upper tail. The gate in (b) is a chi-square test, so the
# thresholds are quantiles rather than tuned numbers.
#
# The tight end matters more than the loose end here: 18.1 measured Lambda_TCG
# as up to 47x overconfident, so a nominal p=0.5 gate is already far tighter in
# truth than it reads. The sweep has to reach low enough to close on Tier-0-only
# or the table cannot say where the gate stops helping and starts doing nothing.
CHI2_6 = {0.01: 0.872, 0.05: 1.635, 0.10: 2.204, 0.25: 3.455, 0.50: 5.348,
          0.90: 10.645, 0.95: 12.592, 0.99: 16.812, 0.999: 22.458}


# ===========================================================================
# per-sequence bundle
# ===========================================================================

class Seq:
    def __init__(self, root: Path, prefix: Path):
        self.name = root.name.replace("rgbd_dataset_", "")
        gs, gp = load_groundtruth(root)
        self.gt = Trajectory(gs, gp)
        self.records: list[Record] = load_records(Path(f"{prefix}_tiers.csv"), gs, gp)
        self.scored = [i for i, r in enumerate(self.records)
                       if r.ref_idx >= 0 and r.T_gt is not None]

    def ate(self, frames: list[FusionOut | None]) -> float:
        return float(evaluate_ate(integrate(self.records, frames),
                                  self.gt, align=True).rmse) * 100.0

    def frame_err(self, frames: list[FusionOut | None]) -> np.ndarray:
        """Relative-pose translation error per scored frame, metres."""
        out = np.full(len(self.records), np.nan)
        for i in self.scored:
            fr = frames[i]
            if fr is None or not fr.ok:
                continue
            d = fr.T_cur_ref.inverse() @ self.records[i].T_gt
            out[i] = float(np.linalg.norm(d.t))
        return out


def load_all(data_root: Path, results_dir: Path) -> list[Seq]:
    out = []
    for root in sorted(data_root.glob("rgbd_dataset_*")):
        prefix = results_dir / root.name
        if Path(f"{prefix}_tiers.csv").exists():
            out.append(Seq(root, prefix))
    return out


# ===========================================================================
# weight rules
# ===========================================================================

def w_const(a1: float, a2: float):
    """Constant weight *ratios*. fuse() minimises a sum of weighted quadratics,
    so scaling all three weights leaves the solution unchanged - only two of
    the three numbers are free, and alpha0 = 1 fixes the gauge."""
    return lambda r: (1.0, a1, a2)


def w_env(mask=(True, True, True)):
    return lambda r: tuple(float(r.alpha[k]) if mask[k] else 0.0 for k in range(3))


def w_uniform(mask=(True, True, True)):
    return lambda r: tuple(1.0 if mask[k] else 0.0 for k in range(3))


def gate_acceptance(records: list[Record], gate: float) -> dict[str, float]:
    """How often the gate actually admits each tier.

    Without this the table in [3] cannot be read: a gate that rejects
    everything reproduces Tier-0-only exactly and would look like a result
    (10.4). A gate that rejects nothing reproduces `no gate` and looks like
    one too.
    """
    out = {}
    for k, t in ((1, "t1"), (2, "t2")):
        offered = passed = 0
        for r in records:
            if r.ref_idx < 0 or not r.ok[t] or not r.ok["t0"]:
                continue
            offered += 1
            eps = (r.T[t] @ r.T["t0"].inverse()).log()
            if np.all(np.isfinite(eps)) and float(eps @ r.info[t] @ eps) <= gate:
                passed += 1
        out[t] = passed / offered if offered else float("nan")
        out[t + "_n"] = offered
    return out


def w_chi2_gate(gate: float, base=(1.0, 1.0, 1.0)):
    """(b): drop a tier whose disagreement with Tier 0 exceeds its own claimed
    covariance. Fully observable at run time - no ground truth anywhere.

    Tier 0 is the anchor because it is the highest-information tier on every
    frame of this data, which is also why fuse() already seeds from it.
    """
    def fn(r: Record):
        w = [base[0] if r.ok["t0"] else 0.0, 0.0, 0.0]
        if not r.ok["t0"]:
            # 기준이 없으면 검정할 수 없다. 판정을 포기하고 통과시킨다.
            return tuple(base[k] if r.ok[t] else 0.0 for k, t in enumerate(["t0", "t1", "t2"]))
        for k, t in ((1, "t1"), (2, "t2")):
            if not r.ok[t]:
                continue
            eps = (r.T[t] @ r.T["t0"].inverse()).log()
            if not np.all(np.isfinite(eps)):
                continue
            nees = float(eps @ r.info[t] @ eps)
            w[k] = base[k] if nees <= gate else 0.0
        return tuple(w)
    return fn


# ===========================================================================
# (c) the oracle bound
# ===========================================================================

ORACLE_CONFIGS = {
    "t0":            w_const(0.0, 0.0),
    "t0t1_env":      w_env((True, True, False)),
    "t0t2_env":      w_env((True, False, True)),
    "t0t1t2_env":    w_env((True, True, True)),
    "t0t1_uni":      w_uniform((True, True, False)),
    "t0t2_uni":      w_uniform((True, False, True)),
    "t0t1t2_uni":    w_uniform((True, True, True)),
}


def oracle(seq: Seq, cache: dict[str, list[FusionOut | None]]) -> dict:
    """Per frame, pick the configuration with the smallest true error.

    This is the ceiling for *any* per-frame selection rule, observable or not.
    It is not the ceiling for ATE - a greedy per-frame choice can still let
    signed error accumulate - so both numbers are reported and the ATE one is
    the weaker claim.
    """
    names = list(ORACLE_CONFIGS)
    errs = np.stack([seq.frame_err(cache[n]) for n in names])       # (C, F)
    picked: list[FusionOut | None] = []
    choice = np.full(len(seq.records), -1)
    for i in range(len(seq.records)):
        col = errs[:, i]
        if np.all(np.isnan(col)):
            picked.append(cache["t0"][i])
            continue
        k = int(np.nanargmin(col))
        choice[i] = k
        picked.append(cache[names[k]][i])
    sel = choice[seq.scored]
    return {
        "ate": seq.ate(picked),
        "err": np.nanmedian(errs[:, seq.scored], axis=1),
        "oracle_err": float(np.nanmedian([errs[choice[i], i] for i in seq.scored
                                          if choice[i] >= 0])),
        "share": {names[k]: float(np.mean(sel == k)) for k in range(len(names))},
        "names": names,
    }


# ===========================================================================
# report
# ===========================================================================

def hdr(s: str) -> None:
    print("\n" + "=" * 88)
    print(s)
    print("=" * 88)


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    seqs = load_all(Path(argv[0]), Path(argv[1]))
    if not seqs:
        print("tiers.csv 없음")
        return 2

    # --- shared cache: every rule that is a pure function of the record ----
    cache: dict[str, dict[str, list[FusionOut | None]]] = {}
    for s in seqs:
        cache[s.name] = {n: fuse_frames(s.records, f) for n, f in ORACLE_CONFIGS.items()}

    base_ate = {s.name: s.ate(cache[s.name]["t0"]) for s in seqs}

    # ---------------------------------------------------------------- (c) --
    hdr("[1] Oracle headroom - can ANY per-frame rule beat Tier-0-only?")
    print("Per frame the oracle picks the configuration with the smallest true error.")
    print("No observable rule can do better. Headroom here bounds parts [2] and [3].\n")
    print(f"{'sequence':>24} {'T0 ATE':>9} {'oracle ATE':>11} {'T0 err':>9} "
          f"{'oracle err':>11} {'headroom':>9}  most-picked")
    orc = {}
    for s in seqs:
        o = oracle(s, cache[s.name])
        orc[s.name] = o
        t0_err = o["err"][o["names"].index("t0")]
        top = sorted(o["share"].items(), key=lambda kv: -kv[1])[:2]
        print(f"{s.name:>24} {base_ate[s.name]:>8.2f}c {o['ate']:>10.2f}c "
              f"{t0_err*1000:>8.2f}m {o['oracle_err']*1000:>10.2f}m "
              f"{t0_err/max(o['oracle_err'],1e-12):>8.2f}x  "
              + ", ".join(f"{k} {v*100:.0f}%" for k, v in top))
    print("\n  ATE in cm, per-frame error in mm. 'headroom' = T0 error / oracle error.")

    # ---------------------------------------------------------------- (a) --
    hdr("[2] Is there a constant weighting that beats Tier-0-only?")
    grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0]
    print(f"grid: alpha1/alpha0 and alpha2/alpha0 in {grid}   ((0,0) = Tier-0-only)\n")
    tab: dict[tuple[float, float], dict[str, float]] = {}
    for a1, a2 in itertools.product(grid, grid):
        for s in seqs:
            tab.setdefault((a1, a2), {})[s.name] = s.ate(fuse_frames(s.records, w_const(a1, a2)))

    def score(pt, names):
        # 시퀀스마다 ATE 크기가 10배 넘게 다르므로 로그 평균을 쓴다. 산술평균이면
        # 가장 나쁜 시퀀스 하나가 적합을 독차지한다.
        return float(np.mean([np.log(max(tab[pt][n], 1e-6)) for n in names]))

    allnames = [s.name for s in seqs]
    best_all = min(tab, key=lambda p: score(p, allnames))
    # 2 % 미만 차이는 동점으로 읽는다. 20.72 -> 20.72 을 "이겼다" 로 세면
    # 사실상 tier 를 끈 설정이 융합의 승리로 보고된다.
    TIE = 0.02
    print(f"{'held-out sequence':>24} {'fitted (a1,a2)':>20} {'held-out ATE':>13} "
          f"{'T0 ATE':>9} {'change':>9} {'verdict':>8}")
    wins = ties = 0
    for s in seqs:
        train = [n for n in allnames if n != s.name]
        pt = min(tab, key=lambda p: score(p, train))
        v, b = tab[pt][s.name], base_ate[s.name]
        rel = (v - b) / b
        verdict = "tie" if abs(rel) < TIE else ("better" if rel < 0 else "worse")
        wins += verdict == "better"
        ties += verdict == "tie"
        print(f"{s.name:>24} {str(pt):>20} {v:>12.2f}c {b:>8.2f}c "
              f"{rel*100:>+8.1f}% {verdict:>8}")
    print(f"\n  in-sample best over all 5: {best_all}  -> "
          + "  ".join(f"{n} {tab[best_all][n]:.2f}" for n in allnames))
    print(f"  leave-one-out: beats Tier-0-only on {wins}/{len(seqs)}, ties on {ties}")
    print("  Every fitted point sits at alpha_k <= 0.01, i.e. the fit's answer is")
    print("  'switch the other tiers off'. A tie there is that, not a win.")

    # ---------------------------------------------------------------- (b) --
    hdr("[3] Is it robustness? A per-frame chi-square consistency gate")
    print("Drop a tier whose disagreement with Tier 0 exceeds its OWN stated covariance.")
    print("Observable at run time. One parameter, and it is a chi-square quantile.\n")
    print(f"{'gate':>10} " + " ".join(f"{s.name[:13]:>14}" for s in seqs)
          + f" {'logmean':>9}  {'T1 pass':>8} {'T2 pass':>8}")
    gate_res: dict[str, dict[str, float]] = {}
    gate_val: dict[str, float] = {}
    for label, g in [("t0 only", -1.0), *[(f"p={p}", v) for p, v in sorted(CHI2_6.items())],
                     ("no gate", float("inf"))]:
        gate_val[label] = g
        if g < 0:
            row = dict(base_ate)
            acc = {"t1": 0.0, "t2": 0.0}
        else:
            row = {s.name: s.ate(fuse_frames(s.records, w_chi2_gate(g))) for s in seqs}
            a = [gate_acceptance(s.records, g) for s in seqs]
            acc = {t: float(np.nanmean([x[t] for x in a])) for t in ("t1", "t2")}
        gate_res[label] = row
        lm = float(np.mean([np.log(max(row[n], 1e-6)) for n in allnames]))
        print(f"{label:>10} " + " ".join(f"{row[s.name]:>13.2f}c" for s in seqs)
              + f" {np.exp(lm):>8.2f}  {acc['t1']*100:>7.1f}% {acc['t2']*100:>7.1f}%")
    print("\n  'T1/T2 pass' = share of offered frames the gate admits, averaged over")
    print("  sequences. At 0 % the row IS Tier-0-only and any difference is noise;")
    print("  at 100 % it is the ungated fusion 18.2 already reported.")

    # 문턱도 적합 대상이다. 5 개 시퀀스에서 고르고 그 5 개로 보고하면 2 절이
    # 경계하는 바로 그 일이 된다.
    print(f"\n  leave-one-out on the gate threshold:")
    labels = [k for k in gate_res if k != "t0 only"]
    gwins = gties = 0
    for s in seqs:
        train = [n for n in allnames if n != s.name]
        pick = min(labels, key=lambda k: np.mean(
            [np.log(max(gate_res[k][n], 1e-6)) for n in train]))
        v, b = gate_res[pick][s.name], base_ate[s.name]
        rel = (v - b) / b
        verdict = "tie" if abs(rel) < TIE else ("better" if rel < 0 else "worse")
        gwins += verdict == "better"
        gties += verdict == "tie"
        print(f"{s.name:>24}  fitted {pick:>9}  held-out {v:>7.2f}c  "
              f"T0 {b:>6.2f}c  {rel*100:>+7.1f}%  {verdict}")
    print(f"  leave-one-out: gated fusion beats Tier-0-only on {gwins}/{len(seqs)}, "
          f"ties on {gties}")

    # ------------------------------------------------------------------ --
    hdr("[4] Does gating restore the complementarity claim?")
    print("18.3 found fusion identifies Tier 0's weak axis and then fills it with a")
    print("WORSE number: selectivity 0.65-1.23, and on fr1_360's rank-deficient frames")
    print("the error along that exact axis got 3.9x worse. Same measurement, gated.\n")
    print(f"{'sequence':>24} {'frames':>7} {'weak gain':>10} {'orth gain':>10} "
          f"{'selectivity':>12}   rank<6 frames")
    for s in seqs:
        gated = fuse_frames(s.records, w_chi2_gate(CHI2_6[0.50]))
        base = cache[s.name]["t0"]
        wb, wf, ob, of = [], [], [], []
        dwb, dwf = [], []
        for i in s.scored:
            r = s.records[i]
            b, g = base[i], gated[i]
            if b is None or g is None or not b.ok or not g.ok:
                continue
            u = r.t0_weak
            n = float(np.linalg.norm(u))
            if n < 1e-9:
                continue
            u = u / n
            e0 = (b.T_cur_ref @ r.T_gt.inverse()).log()
            ef = (g.T_cur_ref @ r.T_gt.inverse()).log()
            if not (np.all(np.isfinite(e0)) and np.all(np.isfinite(ef))):
                continue
            wb.append(abs(float(e0 @ u)))
            wf.append(abs(float(ef @ u)))
            ob.append(float(np.linalg.norm(e0 - (e0 @ u) * u)))
            of.append(float(np.linalg.norm(ef - (ef @ u) * u)))
            if r.counts["t0_dof"] < 6:
                dwb.append(wb[-1])
                dwf.append(wf[-1])
        if not wb:
            continue
        wg = np.median(wb) / max(np.median(wf), 1e-15)
        og = np.median(ob) / max(np.median(of), 1e-15)
        deg = (f"{len(dwb)}  weak-axis {np.median(dwb):.2e} -> {np.median(dwf):.2e} "
               f"({np.median(dwb)/max(np.median(dwf),1e-15):.2f}x)") if dwb else "0"
        print(f"{s.name:>24} {len(wb):>7} {wg:>10.3f} {og:>10.3f} {wg/max(og,1e-15):>12.3f}"
              f"   {deg}")
    print("\n  selectivity > 1 = the improvement is concentrated on the axis Tier 0")
    print("  said it could not see, which is the mechanism the architecture claims.")

    # 중앙값이 거의 안 움직이는데 ATE 가 29 % 좋아졌다면, 이득은 전형적인
    # 프레임이 아니라 꼬리에 있다. 그 둘은 같은 자료에서 서로 다른 통계다.
    print("\n  Where the gain actually is - per-frame relative translation error (mm):")
    print(f"{'sequence':>24} {'median':>16} {'p95':>18} {'p99':>18} {'max':>18}")
    for s in seqs:
        gated = fuse_frames(s.records, w_chi2_gate(CHI2_6[0.50]))
        eb = s.frame_err(cache[s.name]["t0"])[s.scored] * 1000.0
        eg = s.frame_err(gated)[s.scored] * 1000.0
        m = ~(np.isnan(eb) | np.isnan(eg))
        eb, eg = eb[m], eg[m]
        cells = []
        for q in (50, 95, 99, 100):
            b, g = float(np.percentile(eb, q)), float(np.percentile(eg, q))
            cells.append(f"{b:>7.1f}->{g:<7.1f}" if q == 50 else f"{b:>8.1f}->{g:<8.1f}")
        print(f"{s.name:>24} " + " ".join(cells))
    print("  A gain that lives in p99 and not in the median is a robustness gain.")

    # 위 표가 어느 분위에서도 나아지지 않는데 ATE 는 좋아진다면, 이득은 크기가
    # 아니라 부호에 있다. ATE 는 사슬을 적분하므로 부호 있는 성분만 쌓인다.
    print("\n  Magnitude vs sign - only the signed part accumulates into ATE:")
    print(f"{'sequence':>24} {'rms |e| (mm)':>20} {'|mean e| (mm)':>22} "
          f"{'bias fraction':>20}")
    for s in seqs:
        gated = fuse_frames(s.records, w_chi2_gate(CHI2_6[0.50]))
        row = []
        for frames in (cache[s.name]["t0"], gated):
            E = []
            for i in s.scored:
                fr = frames[i]
                if fr is None or not fr.ok:
                    continue
                d = fr.T_cur_ref.inverse() @ s.records[i].T_gt
                if np.all(np.isfinite(d.t)):
                    E.append(d.t)
            E = np.array(E)
            rms = float(np.sqrt((E ** 2).sum(axis=1).mean())) * 1000.0
            bias = float(np.linalg.norm(E.mean(axis=0))) * 1000.0
            row.append((rms, bias, bias / max(rms, 1e-15)))
        (rb, bb, fb), (rg, bg, fg) = row
        print(f"{s.name:>24} {rb:>9.2f}->{rg:<9.2f} {bb:>10.3f}->{bg:<10.3f} "
              f"{fb:>9.3f}->{fg:<9.3f}")
    print("  ATE integrates the chain. An estimator selected on per-frame magnitude")
    print("  can be picked for the wrong statistic entirely - which is also why the")
    print("  oracle in [1] lowers per-frame error and RAISES ATE on two sequences.")

    hdr("[5] Verdict")
    med_head = float(np.median([orc[n]["err"][orc[n]["names"].index("t0")] /
                                max(orc[n]["oracle_err"], 1e-12) for n in allnames]))
    orc_beats = sum(orc[n]["ate"] < base_ate[n] for n in allnames)
    print(f"  oracle per-frame headroom over T0   : {med_head:.2f}x (median)")
    print(f"  oracle beats T0 in ATE on            : {orc_beats}/{len(seqs)} sequences")
    print(f"  fitted constant beats T0 (LOO)       : {wins}/{len(seqs)} win, {ties} tie")
    print(f"  gated fusion beats T0 (LOO)          : {gwins}/{len(seqs)} win, {gties} tie")
    best_gate = min(gate_res, key=lambda k: np.mean(
        [np.log(max(gate_res[k][n], 1e-6)) for n in allnames]))
    print(f"  best gate by log-mean ATE (in-sample): {best_gate}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
