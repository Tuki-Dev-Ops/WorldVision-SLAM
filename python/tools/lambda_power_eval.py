"""스펙트럼 보정 Lambda' = c * Lambda^p 를 7 개 완전 시퀀스에서 채점한다.

lambda_shape.py 가 낸 판정을 궤적으로 확인한다. 거기서 나온 것은 두 가지다.

  1. **축별 보정은 없다.** log(shape_j) 는 시퀀스 잡음과 대체로 구분되지 않고,
     leave-one-sequence-out 에서 축별 보정의 비용비 기하평균이 1.000 이다.
     즉 순이득 0. 억지로 적합하지 않는다.
  2. **모양 오차는 축이 아니라 스펙트럼에 있다.** Lambda 는 자기가 강하다고 말한
     방향에서는 대체로 맞고, 약하다고 말한 방향으로 갈수록 실제보다 훨씬
     비관적이다 (S_0 ~ 1, S_5 ~ 0.01-0.3). 고유기저가 프레임마다 돌기 때문에
     축별 대각행렬로는 절대 못 고치지만, 고유값을 눌러서 고칠 수 있다.

여기서는 (2) 를 궤적 오차로 검정한다. 적합은 **leave-one-sequence-out**,
설정은 일곱 시퀀스에 대해 동일하고, 시퀀스별 상수는 쓰지 않는다.

비교 대상은 27 절이 요구한 셋에 스칼라 보정을 더한 넷이다:
    uniform    균등 가중 (C++ t0t1t2_uniform)
    current    현재 상태 (C++ t0t1t2, 환경 스케줄)
    tier0      Tier 0 단독 (C++ t0)              <- 이것을 이기는 것이 진짜 성과
    scale      티어별 스칼라 kappa, LOSO 적합     <- 27 절의 방법. 정당한 대조군
    power      티어별 (p, c), LOSO 적합          <- 이 문서의 제안

단위에 대한 정직한 경고: Lambda 는 m^-2 와 rad^-2 가 섞인 양이라 Lambda^p 는
차원이 맞지 않는다. c 가 그 스케일을 흡수하므로 저장소가 쓰는 단위(m, rad)
안에서는 잘 정의되지만, 단위를 바꾸면 적합된 c 가 바뀐다. 유도된 모형이 아니라
**측정에 맞춘 모형**이고, 그렇게 읽어야 한다.

사용:
  python tools/lambda_power_eval.py [--validate]
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fusion_eval as fe  # noqa: E402
import fusion_replay as fr  # noqa: E402
from lambda_shape import SEQUENCES, P_GRID, _cost, _spec_profile  # noqa: E402
from wme.eval.trajectory import Trajectory  # noqa: E402
from wme.eval.tum import load_trajectory  # noqa: E402

# 티어별 적합에 요구하는 최소 프레임 수. 이보다 적은 시퀀스는 그 티어의
# 적합 표본에서 빼되, 그 사실을 반드시 출력한다.
MIN_FIT_FRAMES = 20


def load_all() -> dict:
    out = {}
    for name, root, prefix in SEQUENCES:
        p = Path(f"{prefix}_tiers.csv")
        if not p.exists():
            print(f"건너뜀 {name}: {p} 없음")
            continue
        gt_stamps, gt_poses = fe.load_groundtruth(Path(root))
        recs = fe.load_records(p, gt_stamps, gt_poses)
        out[name] = {
            "records": recs,
            "gt": Trajectory(gt_stamps, gt_poses),
            "prefix": Path(prefix),
        }
    return out


def tier_samples(records) -> dict[str, dict]:
    """티어별 (lambda 고유값, 투영^2). lambda_shape.py 와 같은 재료."""
    out = {}
    for t in fe.TIERS:
        ev, pj = [], []
        for r in records:
            if not r.ok[t] or r.T_gt is None:
                continue
            L = 0.5 * (r.info[t] + r.info[t].T)
            if not np.all(np.isfinite(L)) or np.trace(L) <= 0:
                continue
            e = (r.T[t] @ r.T_gt.inverse()).log()
            if not np.all(np.isfinite(e)):
                continue
            w, V = np.linalg.eigh(L)
            o = np.argsort(w)[::-1]
            ev.append(w[o])
            pj.append((V[:, o].T @ e) ** 2)
        out[t] = {"ev": np.array(ev) if ev else np.zeros((0, 6)),
                  "pj": np.array(pj) if pj else np.zeros((0, 6))}
    return out


def fit_tier(samples: list[dict], power: bool) -> tuple[float, float]:
    """훈련 시퀀스들에서 (p, c). power=False 면 p=1 로 고정한 스칼라 보정."""
    ok = [s for s in samples if len(s["ev"]) >= MIN_FIT_FRAMES]
    if not ok:
        return 1.0, 1.0
    grid = P_GRID if power else np.array([1.0])
    best, bp, bc = np.inf, 1.0, 1.0
    for p in grid:
        L = np.log(np.maximum(
            np.array([_spec_profile(s["ev"], s["pj"], p) for s in ok]), 1e-300)).mean(axis=0)
        c = -L.mean()
        r = float(np.sqrt(np.mean((L + c) ** 2)))
        if r < best:
            best, bp, bc = r, float(p), float(np.exp(c))
    return bp, bc


def apply_power(records, fits: dict[str, tuple[float, float]]):
    """Lambda -> c * V diag(lambda^p) V^T. 음/영 고유값은 0 으로 둔다."""
    out = copy.deepcopy(records)
    for r in out:
        for t in fe.TIERS:
            if not r.ok[t]:
                continue
            p, c = fits[t]
            if p == 1.0 and c == 1.0:
                continue
            L = 0.5 * (r.info[t] + r.info[t].T)
            w, V = np.linalg.eigh(L)
            w = np.where(w > w.max() * 1e-12, w, 0.0)
            r.info[t] = c * (V @ np.diag(np.power(w, p, where=w > 0,
                                                  out=np.zeros_like(w))) @ V.T)
    return out


def cpp_ate(prefix: Path, name: str, gt: Trajectory) -> float:
    p = Path(f"{prefix}_{name}.txt")
    if not p.exists():
        return float("nan")
    return fr.ate_cm(load_trajectory(p), gt)


def validate(data: dict) -> bool:
    """numpy 재생이 C++ 를 재현하는가. 이 관문을 통과 못 하면 아래 숫자는 무의미하다."""
    print("=" * 88)
    print("[관문] numpy 재생 vs C++ tum_fusion   (7 개 완전 시퀀스)")
    print("=" * 88)
    worst, guard_ok = 0.0, 0
    for name, d in data.items():
        for ab in ("t0", "t0t1t2", "t0t1t2_uniform"):
            c = cpp_ate(d["prefix"], ab, d["gt"])
            if not np.isfinite(c):
                continue
            mask, mode = fr.ABLATION_MASKS[ab]
            p = fr.ate_cm(fr.replay(d["records"], fr.ablation_weights(mask, mode)), d["gt"])
            rel = abs(p - c) / max(c, 1e-12) * 100.0
            worst = max(worst, rel)
            print(f"{name:>10} {ab:>16} C++ {c:>8.2f}  replay {p:>8.2f}  "
                  f"{rel:>6.3f}%{'   <-- MISMATCH' if rel > 1.0 else ''}")
        # 판별력: 가중치를 망가뜨리면 반드시 숫자가 움직여야 한다.
        # 단, Tier 1/2 가 한 프레임도 기여하지 않는 시퀀스는 움직일 수가 없다.
        # 그것을 실패로 세면 판별 검사가 "티어가 있었는가" 를 재게 된다 -
        # kitti_04 가 정확히 그 경우다 (t0 = t0t1t2 = uniform = 629.33 cm).
        fires = sum(1 for r in d["records"] if r.ref_idx >= 0 and (r.ok["t1"] or r.ok["t2"]))
        if fires == 0:
            print(f"{name:>10} {'판별 면제':>16} Tier 1/2 가 한 프레임도 기여하지 않는다")
            guard_ok += 1
            continue
        base = fr.ate_cm(fr.replay(d["records"],
                                   fr.ablation_weights(*fr.ABLATION_MASKS["t0t1t2"])), d["gt"])
        bad = fr.ate_cm(fr.replay(d["records"], lambda r: (1.0, 1e6, 1e6)), d["gt"])
        moved = abs(bad - base) / max(base, 1e-12) >= 0.01
        guard_ok += int(moved)
        if not moved:
            print(f"{name:>10} {'판별 실패':>16} 가중치를 바꿔도 ATE 가 그대로다 "
                  f"(기여 프레임 {fires})")
    ok = worst < 1.0 and guard_ok == len(data)
    print(f"\n최대 상대 불일치 {worst:.3f} %   판별력 {guard_ok}/{len(data)}")
    print("관문:", "통과" if ok else "실패 - 아래 숫자를 믿지 마라")
    return ok


def main() -> int:
    data = load_all()
    if not data:
        return 2
    if "--validate" in sys.argv:
        return 0 if validate(data) else 1

    names = list(data)
    samples = {n: tier_samples(data[n]["records"]) for n in names}
    for t in fe.TIERS:
        thin = [n for n in names if len(samples[n][t]["ev"]) < MIN_FIT_FRAMES]
        if thin:
            print(f"  [적합 표본] {fe.TIER_NAMES[t]}: 프레임 {MIN_FIT_FRAMES} 개 미만이라 "
                  f"적합에서 제외되는 시퀀스: {', '.join(thin)}")
    print()

    rows = []
    for h in names:
        d = data[h]
        tr = [n for n in names if n != h]
        fits_p = {t: fit_tier([samples[n][t] for n in tr], True) for t in fe.TIERS}
        fits_s = {t: fit_tier([samples[n][t] for n in tr], False) for t in fe.TIERS}

        # ECDA 만 스펙트럼 보정, TCG/SPA 는 스칼라. ECDA 는 여섯 시퀀스에서
        # p 를 적합하지만 TCG 는 세 개, SPA 는 두세 개뿐이라 p 까지 적합하면
        # 그거야말로 과적합이다. 가장 방어 가능한 조합이 이것이다.
        fits_p0 = {t: (fits_p[t] if t == "t0" else fits_s[t]) for t in fe.TIERS}

        wf = fr.ablation_weights(*fr.ABLATION_MASKS["t0t1t2"])
        ate_p = fr.ate_cm(fr.replay(apply_power(d["records"], fits_p), wf), d["gt"])
        ate_s = fr.ate_cm(fr.replay(apply_power(d["records"], fits_s), wf), d["gt"])
        ate_p0 = fr.ate_cm(fr.replay(apply_power(d["records"], fits_p0), wf), d["gt"])

        rows.append({
            "seq": h,
            "uniform": cpp_ate(d["prefix"], "t0t1t2_uniform", d["gt"]),
            "current": cpp_ate(d["prefix"], "t0t1t2", d["gt"]),
            "tier0": cpp_ate(d["prefix"], "t0", d["gt"]),
            "scale": ate_s,
            "power": ate_p,
            "power0": ate_p0,
            "fits": fits_p,
        })

    print("=" * 100)
    print("ATE RMSE (cm)   적합은 leave-one-sequence-out, 설정은 일곱 시퀀스 공통")
    print("=" * 100)
    print(f"{'sequence':>10} {'uniform':>9} {'current':>9} {'tier0':>9} {'scale':>9} "
          f"{'power':>9} {'power0':>9}   {'power0 vs tier0':>16}")
    for r in rows:
        d = (r["tier0"] - r["power0"]) / r["tier0"] * 100.0
        print(f"{r['seq']:>10} {r['uniform']:>9.2f} {r['current']:>9.2f} {r['tier0']:>9.2f} "
              f"{r['scale']:>9.2f} {r['power']:>9.2f} {r['power0']:>9.2f}   {d:>+15.1f}%")

    print()
    for cand in ("scale", "power", "power0"):
        for base in ("uniform", "current", "tier0"):
            w = sum(1 for r in rows if r[cand] < r[base])
            med = float(np.median([(r[base] - r[cand]) / r[base] * 100.0 for r in rows]))
            print(f"  {cand:>7} 가 {base:>8} 를 이긴 시퀀스: {w}/{len(rows)}   "
                  f"중앙 개선 {med:>+6.1f} %")
        print()

    print("\n  LOSO 적합값 (fold 마다):")
    for r in rows:
        s = "  ".join(f"{fe.TIER_NAMES[t]} p={r['fits'][t][0]:.2f} c={r['fits'][t][1]:.3g}"
                      for t in fe.TIERS)
        print(f"{r['seq']:>10}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
