"""엔진의 자기평가가 실제 오차를 예측하는가.

WME 는 "무엇이 신뢰할 만한가" 를 출력한다고 주장한다. 그 주장이 장식인지
실체인지는 한 가지로 갈린다: 추정과 같은 시각에 남긴 자기평가가, 나중에
진리값으로 계산한 그 프레임의 오차와 함께 움직이는가.

프레임마다 ECDA 가 남긴 신호(inlier 비, 측광 RMSE, 관측가능 DOF, 정보행렬
조건수)와, 진리값 대비 상대 포즈 오차의 순위상관을 본다. 순위상관을 쓰는
이유는 눈금이 맞는지가 아니라 순서가 맞는지를 먼저 묻기 때문이다 - 눈금은
보정할 수 있지만 순서가 틀리면 보정할 것이 없다.

사용: python tools/tum_selfassess.py <시퀀스경로> <추정.txt> <진단.csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.tum import load_sequence, load_trajectory  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위상관. scipy 없이 계산한다."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    root, est_path, diag_path = sys.argv[1], sys.argv[2], sys.argv[3]

    est = load_trajectory(est_path)
    gt = load_sequence(root).trajectory()

    rows = []
    with open(diag_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    if len(rows) < 10:
        print("진단 표본이 부족", file=sys.stderr)
        return 1

    # 프레임별 실제 오차: 진리값 상대 운동 대비 추정 상대 운동의 차이.
    # ATE 는 전역 정렬을 거치므로 프레임 단위 신뢰도와 대조하기에 맞지 않는다.
    est_at = {round(t, 4): p for t, p in zip(est.stamps, est.poses)}
    gt_stamps = gt.stamps

    stamps, errs = [], []
    prev = None
    for row in rows:
        t = round(row["timestamp"], 4)
        p = est_at.get(t)
        if p is None:
            prev = None
            continue
        k = int(np.argmin(np.abs(gt_stamps - row["timestamp"])))
        if abs(gt_stamps[k] - row["timestamp"]) > 0.02:
            prev = None
            continue
        g = gt.poses[k]
        if prev is not None:
            pe, pg = prev
            # 상대 운동의 병진 차이 (m)
            d_est = pe.inverse() @ p
            d_gt = pg.inverse() @ g
            errs.append(float(np.linalg.norm(d_est.t - d_gt.t)))
            stamps.append(row["timestamp"])
        prev = (p, g)

    if len(errs) < 10:
        print("대조 가능한 프레임이 부족", file=sys.stderr)
        return 1

    err = np.array(errs)
    # 오차 계산에 쓴 프레임만 남긴다 (첫 프레임과 끊긴 구간 제외)
    keep = {round(s, 4) for s in stamps}
    sel = [r for r in rows if round(r["timestamp"], 4) in keep][1:len(err) + 1]
    sel = sel[:len(err)]
    err = err[:len(sel)]

    print(f"프레임 {len(err)}   상대 병진오차 중앙 {np.median(err)*1000:.1f} mm  "
          f"90% {np.percentile(err, 90)*1000:.1f} mm")
    print()
    print(f"{'신호':<22} {'순위상관':>9}   해석")
    print("-" * 62)

    # 부호를 맞춘다: 값이 클수록 '나쁜 상태' 가 되도록 정의한 뒤 오차와 비교한다.
    signals = {
        "측광 RMSE":        np.array([r["rmse"] for r in sel]),
        "1 - inlier 비":    np.array([1.0 - r["inlier_ratio"] for r in sel]),
        "조건수 (log)":     np.log10(np.array([max(r["cond"], 1.0) for r in sel])),
        "6 - 관측가능 DOF": np.array([6.0 - r["observable_dof"] for r in sel]),
        "1 / 점 개수":      1.0 / np.maximum(np.array([r["points"] for r in sel]), 1.0),
    }
    for name, s in signals.items():
        rho = spearman(s, err)
        if np.isnan(rho):
            verdict = "표본 부족"
        elif rho > 0.4:
            verdict = "예측력 있음"
        elif rho > 0.15:
            verdict = "약한 예측력"
        elif rho > -0.15:
            verdict = "무관 (장식)"
        else:
            verdict = "반대로 움직임"
        print(f"{name:<22} {rho:>+9.3f}   {verdict}")

    # 상위 10% 오차 프레임을 각 신호가 얼마나 잡아내는가.
    #
    # 재현율만 보면 안 된다. 신호가 상수면 임계가 최솟값과 같아져 전 프레임이
    # flag 되고, 재현율은 100 % 가 나온다 - 좋은 프레임도 전부 잡으면서.
    # 판별력 없는 지표가 만점으로 보이는 전형적인 경우다.
    # lift = P(나쁨|flag) / P(나쁨) 은 그 함정에 빠지지 않는다. 전부 flag 하면 1.0.
    thr = np.percentile(err, 90)
    bad = err >= thr
    base = float(bad.mean())
    print()
    print(f"상위 10% 오차 프레임({int(bad.sum())}개) 탐지력")
    print(f"{'신호':<22} {'flag비율':>8} {'재현율':>7} {'lift':>7}   (lift 1.0 = 정보 없음)")
    for name, s in signals.items():
        flag = s >= np.percentile(s, 90)
        frac = float(flag.mean())
        rec = float((flag & bad).sum()) / max(1, int(bad.sum()))
        prec = float((flag & bad).sum()) / max(1, int(flag.sum()))
        lift = prec / base if base > 0 else float("nan")
        print(f"  {name:<20} {frac*100:7.1f}% {rec*100:6.1f}% {lift:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
