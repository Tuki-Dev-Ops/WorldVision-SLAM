"""`depth_consistency` 문턱을 실제로 맞춘다.

25.5 는 이 상수를 이렇게 남겨 두었다: *"`depthConsistent()` 의 0.02 는 한
카메라의 깨끗한 장면 값에서 가져온 자리표시자이지 적합된 상수가 아니다."*
그 상태로 소비를 붙이면 문턱 하나가 판정을 정하게 되고, 그건 이 문서가
반복해서 잡아 온 실패 양상이다(15.2 의 nu, 20.2 의 sigma_c, 17.4 의 순환성).

여기서 요구하는 성질은 정확도 예측이 아니라 **분리** 다: 정상 프레임과 실패
프레임의 `depth_consistency` 분포가 문턱 하나로 갈리는가, 그리고 그 문턱이
카메라를 바꿔도 같은 자리에 있는가. 후자가 없으면 25.2 의 결론(신호가 실패를
따라 자란다)은 살아도 게이트로는 못 쓴다 - 17.1 의 nu 가 정확히 그랬다.

"실패 프레임" 은 진리값 상대 병진 오차의 상위 10 % 로 정의한다. 판정 자체에
depth_consistency 를 쓰면 동어반복이 되므로 정의에는 절대 넣지 않는다.

사용:
  python tools/depth_gate_calib.py
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

from wme.reference.geometry import SE3, quat_to_matrix  # noqa: E402

from fusion_eval import load_groundtruth, interpolate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def camera_of(name: str) -> str:
    for g in ("freiburg1", "freiburg2", "freiburg3"):
        if g in name:
            return g
    return "?"


def load(name: str):
    diag = ROOT / "results" / "selfassess" / f"{name}_diag.csv"
    root = ROOT / "data" / f"rgbd_dataset_{name}"
    if not (diag.exists() and root.exists()):
        return None
    gs, gp = load_groundtruth(root)
    di, err = [], []
    for r in csv.DictReader(open(diag, encoding="utf-8", errors="replace")):
        # 쓰는 중인 CSV 는 마지막 행이 잘려 있을 수 있다(None). 조용히 넘기지
        # 않고 건너뛰되, 표본 수가 줄어드는 것은 아래 최소 표본 검사가 잡는다.
        try:
            v = float(r["depth_incons"])
        except (KeyError, ValueError, TypeError):
            continue
        if v < 0:                      # 판정 불가는 표본이 아니다
            continue
        gc = interpolate(gs, gp, float(r["timestamp"]))
        gr = interpolate(gs, gp, float(r["ref_timestamp"]))
        if gc is None or gr is None:
            continue
        rel = SE3(quat_to_matrix(float(r["rel_qx"]), float(r["rel_qy"]),
                                 float(r["rel_qz"]), float(r["rel_qw"])),
                  np.array([float(r["rel_tx"]), float(r["rel_ty"]),
                            float(r["rel_tz"])]))
        d = rel.inverse() @ (gc.inverse() @ gr)
        if not np.all(np.isfinite(d.t)):
            continue
        di.append(v)
        err.append(float(np.linalg.norm(d.t)))
    if len(di) < 30:
        return None
    return np.array(di), np.array(err)


def youden(sig: np.ndarray, bad: np.ndarray) -> tuple[float, float, float]:
    """TPR - FPR 을 최대로 하는 문턱. 정밀도/재현율의 가중치를 임의로 고르지
    않아도 되는 유일한 기준이라 쓴다."""
    best = (0.0, np.nan, np.nan)
    for t in np.unique(np.quantile(sig, np.linspace(0.5, 0.995, 120))):
        tpr = float((sig[bad] >= t).mean()) if bad.any() else 0.0
        fpr = float((sig[~bad] >= t).mean()) if (~bad).any() else 0.0
        if tpr - fpr > best[0]:
            best = (tpr - fpr, t, tpr)
    return best[1], best[0], best[2]


def main() -> int:
    d = ROOT / "results" / "selfassess"
    names = sorted({p.name.replace("_diag.csv", "") for p in d.glob("*_diag.csv")
                    if not p.name.startswith("haze")})
    rows = []
    print("depth_consistency 분포와 분리 문턱  (실패 = 진리값 상대 병진오차 상위 10 %)")
    print(f"\n{'sequence':<34}{'cam':>5}{'n':>6}{'p50 ok':>9}{'p50 bad':>9}"
          f"{'ratio':>7}{'Youden t':>10}{'J':>7}")
    for n in names:
        got = load(n)
        if got is None:
            continue
        sig, err = got
        bad = err >= np.quantile(err, 0.90)
        t, J, _ = youden(sig, bad)
        ok_med = float(np.median(sig[~bad]))
        bad_med = float(np.median(sig[bad]))
        rows.append((n, camera_of(n), t, J, ok_med, bad_med))
        print(f"{n:<34}{camera_of(n)[-1]:>5}{len(sig):>6}{ok_med:>9.4f}{bad_med:>9.4f}"
              f"{bad_med/max(ok_med,1e-9):>7.2f}{t:>10.4f}{J:>7.2f}")

    if not rows:
        print("\n진단 CSV 가 없다 - wme_tum_odometry --diag 를 먼저 돌려야 한다")
        return 1

    print("\n" + "=" * 78)
    ts = np.array([r[2] for r in rows])
    print(f"문턱 범위 {ts.min():.4f} ~ {ts.max():.4f}   중앙값 {np.median(ts):.4f}"
          f"   퍼짐 {ts.max()/max(ts.min(),1e-9):.1f}x")
    for cam in ("freiburg1", "freiburg2", "freiburg3"):
        c = [r[2] for r in rows if r[1] == cam]
        if c:
            print(f"  {cam}: n={len(c)}  중앙값 {np.median(c):.4f}  "
                  f"범위 {min(c):.4f}~{max(c):.4f}")
    print("\n카메라별 중앙값이 서로 크게 다르면 이 값은 센서 상수이지 알고리즘")
    print("상수가 아니다 - 17.1 의 nu 가 정확히 그 경우였다(카메라 바꾸니 23배).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
