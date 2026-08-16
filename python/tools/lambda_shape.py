"""티어 정보행렬 Lambda 의 **축별** 보정이 존재하는지 판정한다.

docs/06-results.md 27 절이 남긴 미확립 항목을 그대로 받는다:

  "축별 보정이 존재하는지는 미확립이며, 검정하려면 각 티어의 오차 공분산을
   방향별로 정답에 대고 재야 한다 - 그것을 하지 않았다."

여기서 그것을 한다. 27 절은 ANEES 하나 - 즉 **스칼라** - 만 재고 kappa 를 뽑았다.
스칼라는 스케일만 고칠 수 있고 모양은 못 고친다. 이 도구는 둘을 분리해서 잰다.

좌표계 (코드로 확정, 세 티어 모두 같다)
---------------------------------------
세 티어의 Lambda 는 전부 xi = [rho(3), phi(3)], **좌측 섭동** T <- exp(xi) T_cur_ref
규약이고, 그 접선은 **cur 카메라 좌표계**에 산다.

  Tier 0 ECDA : src/localization/DirectAligner.cpp 270-274
                dQ/drho = I, dQ/dphi = -skew(Q), Q 는 cur 카메라 좌표
  Tier 1 TCG  : src/fusion/TierInformation.cpp 23-27
                J = [I, -skew(q)], q = T p_ref 즉 cur 카메라 좌표
  Tier 2 SPA  : src/geometry/StructuralAligner.cpp 220-233
                J = [n; t x n], n = current[..].normal 즉 cur 카메라 좌표

오차도 같은 규약으로 잰다: eps = log(T_k T_gt^-1) 이면 T_k = exp(eps) T_gt 이므로
eps 는 T_gt 에서 T_k 로 가는 좌측 섭동이고, cur 카메라 접선에 산다.
fusion_eval.py 의 tier_calibration() 과 같은 식이다 - 거기와 다른 것은 스칼라로
접지 않는다는 것뿐이다.

KITTI 정답은 poses/<seq>.txt 를 그대로 옮긴 것이라 좌 카메라 광학 좌표
(x 우, y 하, z 전방)이고, TUM 정답도 같은 광학 규약이다. 즉 정답 상대포즈와
Lambda 가 같은 좌표계에 있다. 섞이지 않았다.

무엇을 재는가
-------------
Lambda 가 옳다면 eps ~ N(0, Lambda^-1) 이다. 프레임마다 Lambda 가 다르므로
프레임별로 표준화해서 모은다.

  (A) 축별 주변 비율  R_j = mean_i [ eps_ij^2 / (Lambda_i^-1)_jj ]
      옳으면 전부 1. 축 j 에서 몇 배 과신인지를 그 축의 단위로 직접 말한다.
      Lambda 역이 필요하므로 조건수 게이트를 통과한 프레임만 쓴다.

  (B) 랭크 안전 지표  W_j = mean_i [ (Lambda_i eps_i)_j^2 / (Lambda_i)_jj ]
      E[Lambda eps eps^T Lambda] = Lambda 이므로 옳으면 전부 1.
      Lambda 를 뒤집지 않으므로 SPA 처럼 랭크 부족한 티어도 전 프레임을 쓴다.
      다만 **축별 추정량은 아니다**. Lambda 가 비대각이면 D 를 정확히 되찾지
      못한다 (판별력 검사에서 확인했다). 조건수 게이트가 표본을 골라냈는지
      보는 교차확인용이고, 방향(어느 블록이 과신인가)만 읽는다.

      처음에는 W_j = sum_i(...) / sum_i(...) 로 썼는데 틀렸다. 프레임 간
      Lambda 크기가 몇 자릿수씩 다르므로 그 비는 가장 큰 몇 프레임이 통째로
      정하고, 유효 표본수가 3 쯤으로 줄어 값이 널뛰었다. 프레임마다 먼저
      정규화하고 평균낸다.

  (C) 백색화 고유스펙트럼  C = mean_i [ Lambda_i^1/2 eps_i eps_i^T Lambda_i^1/2 ]
      옳으면 C = I. 고유값이 전부 같으면 **스칼라 kappa 로 고칠 수 있다**는 뜻이고,
      퍼져 있으면 스칼라로는 못 고친다. 27 절이 스칼라만 썼으므로 이것이
      "축별 보정이 필요한가" 의 직접 답이다.

스케일과 모양의 분리
--------------------
  scale = geomean_j R_j            (스칼라 kappa 가 고칠 수 있는 부분)
  shape_j = R_j / scale            (스칼라가 못 고치는 부분. 곱이 1)
  anisotropy = max_j shape_j / min_j shape_j

판별력 검사 (--selftest)
------------------------
재는 방법이 모양 오차를 실제로 구분하는지 먼저 확인한다. 실제 Lambda 를 그대로
쓰고 eps 만 합성한다:

  null   : eps ~ N(0, Lambda^-1)             -> R_j 가 전부 1, anisotropy 의 영분포
  scale  : eps ~ N(0, (c Lambda)^-1)         -> R_j 전부 c, anisotropy 는 1 근처
  shape  : eps ~ N(0, (D Lambda D)^-1)       -> R_j 가 D 를 되찾는가
  swap   : x 축과 z 축을 바꾼 Lambda 에서 표집 -> 축 뒤바뀜을 잡아내는가
  frame  : eps 만 고정 90 도 회전            -> 좌표계를 섞으면 모양 오차가 생기는가

마지막 것이 중요하다. 좌표계를 섞으면 모양 오차가 **만들어진다**는 것을 보여야
"섞지 않았다" 는 주장에 내용이 생긴다.

사용:
  python tools/lambda_shape.py --selftest
  python tools/lambda_shape.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.reference.geometry import so3_exp  # noqa: E402

import fusion_eval as fe  # noqa: E402

AXES = ["rho_x", "rho_y", "rho_z", "phi_x", "phi_y", "phi_z"]

# 7 개 완전 시퀀스. 27 절과 같은 집합, 같은 산출물(tcgsig_ = TCG sigma 수정 후).
SEQUENCES = [
    ("kitti_00", "data/kitti_00", "results/tcgsig_kitti_00"),
    ("kitti_04", "data/kitti_04", "results/tcgsig_kitti_04"),
    ("kitti_05", "data/kitti_05", "results/tcgsig_kitti_05"),
    ("kitti_07", "data/kitti_07", "results/tcgsig_kitti_07"),
    ("fr1_360", "data/rgbd_dataset_freiburg1_360",
     "results/tcgsig_rgbd_dataset_freiburg1_360"),
    ("fr1_desk", "data/rgbd_dataset_freiburg1_desk",
     "results/tcgsig_rgbd_dataset_freiburg1_desk"),
    ("fr1_xyz", "data/rgbd_dataset_freiburg1_xyz",
     "results/tcgsig_rgbd_dataset_freiburg1_xyz"),
]

# Lambda^-1 을 쓰는 통계(A, C)의 조건수 게이트. 배정밀도에서 역이 의미를 잃는 지점
# 훨씬 앞에 둔다. 여기서 걸러진 프레임 수는 반드시 보고한다 - 조용히 빠지면
# SPA 처럼 랭크 부족한 티어의 통계가 "잘 구속된 프레임만" 의 통계가 된다.
COND_GATE = 1e10


# ===========================================================================
# 표본 수집
# ===========================================================================

def collect(root: Path, prefix: Path) -> dict[str, dict]:
    """시퀀스 하나에서 티어별 (eps, Lambda) 표본을 모은다."""
    gt_stamps, gt_poses = fe.load_groundtruth(root)
    records = fe.load_records(Path(f"{prefix}_tiers.csv"), gt_stamps, gt_poses)

    out: dict[str, dict] = {}
    for t in fe.TIERS:
        eps, lam = [], []
        for r in records:
            if not r.ok[t] or r.T_gt is None:
                continue
            L = r.info[t]
            if not np.all(np.isfinite(L)) or np.trace(L) <= 0:
                continue
            e = (r.T[t] @ r.T_gt.inverse()).log()
            if not np.all(np.isfinite(e)):
                continue
            eps.append(e)
            lam.append(0.5 * (L + L.T))
        out[t] = {
            "eps": np.array(eps) if eps else np.zeros((0, 6)),
            "lam": np.array(lam) if lam else np.zeros((0, 6, 6)),
        }
    return out


# ===========================================================================
# 통계
# ===========================================================================

def _sym_sqrt_and_inv(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """대칭 제곱근, 역행렬, 조건수. 역은 조건수가 나쁘면 nan 으로 둔다."""
    w, V = np.linalg.eigh(lam)
    w = np.clip(w, 0.0, None)
    wmax = w.max()
    cond = np.inf if wmax <= 0 or w.min() <= 0 else float(wmax / w.min())
    half = V @ np.diag(np.sqrt(w)) @ V.T
    if cond > COND_GATE:
        inv = np.full((6, 6), np.nan)
    else:
        inv = V @ np.diag(1.0 / w) @ V.T
    return half, inv, cond


def measure(eps: np.ndarray, lam: np.ndarray, center: bool = False) -> dict:
    """축별 보정 통계 (A)(B)(C) 와 스케일/모양 분리.

    center=True 면 시퀀스 평균을 뺀다. 이것은 **보정이 아니라 진단**이다 -
    시퀀스별 상수를 쓰는 셈이라 보정으로는 금지된 것이고, 여기서는 "시퀀스 간
    산포가 잡음이 아니라 시퀀스마다 다른 편향에서 오는가" 만 묻는다.
    Lambda 는 영평균 오차를 주장하므로 편향은 그 자체로 Lambda 의 오류지만,
    축별 **모양**의 문제인지 아닌지는 갈라서 봐야 한다.
    """
    n = len(eps)
    if n == 0:
        return {"n": 0, "n_inv": 0}
    if center and n > 1:
        eps = eps - eps.mean(axis=0)

    halves = np.zeros_like(lam)
    invs = np.zeros_like(lam)
    conds = np.zeros(n)
    for i in range(n):
        halves[i], invs[i], conds[i] = _sym_sqrt_and_inv(lam[i])

    good = np.isfinite(invs[:, 0, 0])
    n_inv = int(good.sum())

    # (A) 축별 주변 비율
    if n_inv:
        var = np.einsum("ijj->ij", invs[good])          # (m,6) 주변 분산
        R = np.mean(eps[good] ** 2 / var, axis=0)
    else:
        R = np.full(6, np.nan)

    # (B) 랭크 안전 지표. Lambda 를 뒤집지 않는다. 프레임마다 먼저 정규화한다 -
    # 안 하면 Lambda 가 가장 큰 몇 프레임이 값을 통째로 정한다.
    Le = np.einsum("ijk,ik->ij", lam, eps)              # (n,6)
    d = np.einsum("ijj->ij", lam)                       # (n,6) 대각
    W = np.mean(np.where(d > 0, Le ** 2 / np.where(d > 0, d, 1.0), np.nan), axis=0)

    # (C) 백색화 공분산과 그 고유값
    z = np.einsum("ijk,ik->ij", halves, eps)            # (n,6)
    C = (z[:, :, None] * z[:, None, :]).mean(axis=0)
    cev = np.sort(np.linalg.eigvalsh(0.5 * (C + C.T)))[::-1]

    # 스케일 / 모양 분리
    pos = R[np.isfinite(R) & (R > 0)]
    scale = float(np.exp(np.mean(np.log(pos)))) if len(pos) == 6 else float("nan")
    shape = R / scale if np.isfinite(scale) else np.full(6, np.nan)
    aniso = float(np.nanmax(shape) / np.nanmin(shape)) if np.isfinite(scale) else float("nan")

    # 병진 블록 대 회전 블록. 축 여섯 개보다 물리적으로 다루기 쉬운 2 매개변수 판.
    blk = float(np.exp(np.mean(np.log(R[:3]))) / np.exp(np.mean(np.log(R[3:])))) \
        if len(pos) == 6 else float("nan")

    # (D) **스펙트럼 프로파일**. 축별 보정이 없다는 답이 나와도 C_spread 가 크면
    # 모양 오차는 분명히 있다 - 다만 좌표축이 아니라 다른 데 있는 것이다.
    # 여기서 그 다른 데를 찾는다: Lambda 자신의 고유기저.
    #
    #   z_k = sqrt(lambda_k) * (v_k . eps),  lambda 내림차순
    #   S_k = mean_i z_ik^2                  옳으면 전부 1
    #
    # k 가 커질수록(약한 축) S_k 가 체계적으로 작아지면 "Lambda 는 자기가 약하다고
    # 말한 방향에서 실제보다 훨씬 더 비관적" 이라는 뜻이다. 이것은 축별 D 로는
    # 절대 못 고친다 - 고유기저가 프레임마다 돌기 때문이다. 대신 스펙트럼을
    # 눌러서 (lambda -> lambda^gamma) 고칠 수 있다. 즉 고칠 수 있는 모양 오차가
    # 축이 아니라 **조건수**에 있는지를 묻는 것이다.
    prof = np.zeros(6)
    gam = np.full(n, np.nan)
    ev = np.zeros((n, 6))       # lambda_k 내림차순
    pj = np.zeros((n, 6))       # (v_k . eps)^2
    for i in range(n):
        w, V = np.linalg.eigh(lam[i])
        o = np.argsort(w)[::-1]
        w, V = w[o], V[:, o]
        ev[i], pj[i] = w, (V.T @ eps[i]) ** 2
        prof += w * (V.T @ eps[i]) ** 2
        # 이 프레임 하나가 요구하는 스펙트럼 지수. log z^2 를 log lambda 에 회귀.
        keep = w > w.max() * 1e-12
        if keep.sum() >= 3:
            y = np.log(np.maximum((V.T @ eps[i]) ** 2, 1e-300))[keep]
            x = np.log(w[keep])
            gam[i] = float(np.polyfit(x, y, 1)[0])
    prof /= n

    return {
        "n": n, "n_inv": n_inv,
        "ev": ev, "pj": pj,
        "spec": prof,
        "spec_slope": float(np.polyfit(np.arange(6), np.log(np.maximum(prof, 1e-300)), 1)[0]),
        "gamma_med": float(np.nanmedian(gam)) if np.any(np.isfinite(gam)) else float("nan"),
        "cond_median": float(np.median(conds[np.isfinite(conds)]))
        if np.any(np.isfinite(conds)) else float("inf"),
        "R": R, "W": W, "C_eig": cev,
        "anees": float(np.mean(np.einsum("ij,ijk,ik->i", eps, lam, eps))),
        "scale": scale, "shape": shape, "aniso": aniso, "block_tr": blk,
        # 스칼라로 고칠 수 있는가: C 의 고유값이 다 같으면 그렇다.
        "C_spread": float(cev[0] / cev[-1]) if cev[-1] > 0 else float("inf"),
    }


# ===========================================================================
# 판별력 검사
# ===========================================================================

def _sample(lam: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """eps ~ N(0, lam^-1) 을 프레임마다 뽑는다. 랭크 부족이면 영공간은 0 으로 둔다.

    영공간을 0 으로 두는 것은 자비가 아니라 필요다 - 정보가 0 인 방향에서
    표집하면 분산이 무한이라 검사 자체가 성립하지 않는다. 그 방향은 (B) 통계가
    자동으로 무시하므로 검사의 결론에 영향을 주지 않는다.
    """
    out = np.zeros((len(lam), 6))
    for i, L in enumerate(lam):
        w, V = np.linalg.eigh(0.5 * (L + L.T))
        keep = w > w.max() * 1e-12
        s = np.zeros(6)
        s[keep] = 1.0 / np.sqrt(w[keep])
        out[i] = V @ (s * rng.standard_normal(6))
    return out


def selftest() -> int:
    """측정이 모양 오차를 실제로 구분하는지 확인한다. 실패는 시끄럽게."""
    rng = np.random.default_rng(20260816)
    print("=" * 92)
    print("판별력 검사 - 실제 Lambda 위에서 eps 만 합성한다")
    print("=" * 92)

    # 실제 Lambda 를 쓴다. 합성 행렬을 쓰면 검사가 실제 조건수 분포를 안 본다.
    root = Path("data/kitti_00")
    prefix = Path("results/tcgsig_kitti_00")
    if not Path(f"{prefix}_tiers.csv").exists():
        print(f"  건너뜀: {prefix}_tiers.csv 가 없다")
        return 1
    s = collect(root, prefix)
    lam0 = s["t0"]["lam"]
    print(f"  Tier 0 Lambda {len(lam0)} 개를 kitti_00 에서 가져왔다\n")

    fails = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:28s} {detail}")

    # --- null: 영분포와 그 폭 -------------------------------------------------
    anisos, scales = [], []
    for _ in range(30):
        m = measure(_sample(lam0, rng), lam0)
        anisos.append(m["aniso"])
        scales.append(m["scale"])
    a_null = float(np.percentile(anisos, 95))
    print(f"  영분포 (n={len(lam0)}, 30 회):  scale {np.mean(scales):.3f}"
          f" +- {np.std(scales):.3f}   anisotropy 중앙 {np.median(anisos):.3f}"
          f"  95% {a_null:.3f}\n")
    check("null: scale ~ 1", abs(np.mean(scales) - 1.0) < 0.05,
          f"평균 {np.mean(scales):.4f}")
    check("null: anisotropy 영분포 유한", a_null < 2.0,
          f"95 백분위 {a_null:.3f}  <- 이 위가 모양 오차의 문턱")

    # --- scale: 스칼라 오차는 모양 오차로 새지 않아야 한다 ---------------------
    # eps 를 (c Lambda)^-1 에서 뽑으면 참 오차가 sqrt(c) 배 **작다**. 그것을
    # Lambda 로 재면 R = 1/c 다. (처음에 R = c 를 기대했다가 검사에 걸렸다.)
    c = 7.0
    m = measure(_sample(lam0 * c, rng), lam0)
    check("scale c=7: scale 회수", abs(m["scale"] * c - 1.0) < 0.15,
          f"측정 {m['scale']:.4f} (기대 {1/c:.4f})")
    check("scale c=7: 모양 오차 없음", m["aniso"] < a_null * 1.5,
          f"anisotropy {m['aniso']:.3f} (영분포 95% {a_null:.3f})")
    check("scale c=7: C 고유값 평평", m["C_spread"] < a_null * 2.0,
          f"C_spread {m['C_spread']:.3f}")

    # --- shape: 병진/회전 블록을 일부러 어긋내면 잡아내는가 --------------------
    D = np.diag([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    lam_d = np.einsum("jk,ikl,lm->ijm", D, lam0, D)
    m = measure(_sample(lam_d, rng), lam0)
    # eps 는 (D Lambda D)^-1 에서 왔으므로 Lambda 로 재면 회전축이 1/9 배다.
    got = m["R"]
    ok = (abs(got[:3].mean() - 1.0) < 0.25) and (abs(got[3:].mean() * 9.0 - 1.0) < 0.25)
    check("shape D=diag(1,1,1,3,3,3)", ok,
          f"R = {np.array2string(got, precision=3)}  (기대 1,1,1,1/9,1/9,1/9)")
    check("shape: block_tr 이 9 를 가리킴", abs(m["block_tr"] / 9.0 - 1.0) < 0.3,
          f"block_tr {m['block_tr']:.3f} (기대 9)")
    check("shape: C 고유값이 퍼짐", m["C_spread"] > a_null * 2.0,
          f"C_spread {m['C_spread']:.2f}  <- 스칼라로 못 고친다는 신호")

    # --- swap: 축을 뒤바꾸면 잡아내는가 ---------------------------------------
    P = np.eye(6)[[2, 1, 0, 5, 4, 3]]          # x <-> z (병진, 회전 모두)
    lam_p = np.einsum("jk,ikl,ml->ijm", P, lam0, P)
    m = measure(_sample(lam_p, rng), lam0)
    check("swap x<->z 를 잡아냄", m["aniso"] > a_null * 1.5,
          f"anisotropy {m['aniso']:.3f}   R = {np.array2string(m['R'], precision=2)}")

    # --- frame: 좌표계를 섞으면 모양 오차가 만들어지는가 ----------------------
    # eps 만 고정 90 도 회전시킨다. Lambda 는 그대로 - 이것이 "티어의 Lambda 와
    # 정답 오차를 다른 좌표계에서 쟀다" 의 정확한 모형이다.
    Rz = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
    M = np.zeros((6, 6))
    M[:3, :3] = Rz
    M[3:, 3:] = Rz
    e_true = _sample(lam0, rng)
    m_ok = measure(e_true, lam0)
    m_mix = measure(e_true @ M.T, lam0)
    check("frame 90 도 섞으면 모양 오차 발생", m_mix["aniso"] > max(a_null, m_ok["aniso"] * 2),
          f"섞기 전 {m_ok['aniso']:.3f} -> 섞은 뒤 {m_mix['aniso']:.3f}")

    # --- (B) 가 방향을 맞게 말하는가 ------------------------------------------
    # (B) 는 축별 추정량이 아니다. Lambda 가 비대각이면 D 를 정확히 되찾지
    # 못하므로 크기는 요구하지 않고, "회전 블록이 병진 블록보다 작다" 는
    # 방향만 요구한다. 그것이 (B) 가 SPA 에서 해 줄 수 있는 전부다.
    m = measure(_sample(lam_d, rng), lam0)
    check("(B) 랭크안전 지표가 방향을 맞힘",
          float(np.nanmean(m["W"][3:])) < float(np.nanmean(m["W"][:3])),
          f"W = {np.array2string(m['W'], precision=3)}  (회전 블록이 작아야 한다)")
    m_null = measure(_sample(lam0, rng), lam0)
    check("(B) null 에서 1 근처", abs(float(np.nanmean(m_null["W"])) - 1.0) < 0.35,
          f"W 평균 {np.nanmean(m_null['W']):.3f}")

    print(f"\n  {'검사 전부 통과' if fails == 0 else f'{fails} 개 실패'}")
    return 0 if fails == 0 else 1


# ===========================================================================
# 보고
# ===========================================================================

def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:>9.3g}" for x in v)


def _cost(R: np.ndarray) -> float:
    """1 에서 얼마나 먼가. log 로 재야 2 배 과신과 2 배 과소가 같은 벌점이 된다."""
    r = R[np.isfinite(R) & (R > 0)]
    if len(r) < 6:
        return float("nan")
    return float(np.sqrt(np.mean(np.log(r) ** 2)))


P_GRID = np.linspace(0.2, 1.4, 61)


def _spec_profile(ev: np.ndarray, pj: np.ndarray, p: float) -> np.ndarray:
    """Lambda' = Lambda^p 로 고쳤을 때의 S_k. c 는 빼고 모양만."""
    keep = ev > ev.max(axis=1, keepdims=True) * 1e-12
    v = np.where(keep, np.power(np.where(keep, ev, 1.0), p) * pj, np.nan)
    return np.nanmean(v, axis=0)


def _fit_power(ms: list[dict]) -> tuple[float, float]:
    """훈련 시퀀스들에서 (p, c) 를 고른다. c 는 p 가 정해지면 닫힌 형태다."""
    best, bp, bc = np.inf, 1.0, 1.0
    for p in P_GRID:
        # 시퀀스마다 프로파일을 내고 시퀀스에 걸쳐 평균한다. 긴 시퀀스가
        # 통째로 정하지 않도록 시퀀스를 동등하게 센다.
        L = np.log(np.maximum(np.array([_spec_profile(m["ev"], m["pj"], p) for m in ms]),
                              1e-300)).mean(axis=0)
        c = -L.mean()                       # log c. S_k -> 1 로 올리는 값
        r = float(np.sqrt(np.mean((L + c) ** 2)))
        if r < best:
            best, bp, bc = r, p, float(np.exp(c))
    return bp, bc


def loso_power(per: dict, tier: str, min_n: int = 30) -> None:
    """모양 오차가 **축**이 아니라 **스펙트럼**에 있는가.

    (D) 가 보여준 것: S_k 는 최강축에서 1 근처이고 최약축으로 갈수록 떨어진다.
    즉 Lambda 는 자기가 약하다고 말한 방향에서 실제보다 훨씬 비관적이다.
    이것은 고유기저가 프레임마다 도는 오차라 축별 D 로는 못 고치지만,
    스펙트럼을 눌러서 고칠 수 있다:

        Lambda' = c * V diag(lambda_k^p) V^T,   p < 1 이 스펙트럼을 압축한다

    매개변수 둘 (p, c) 뿐이고 좌표계를 타지 않는다. 축별 여섯 개와 달리
    프레임마다 도는 기저를 따라간다. leave-one-sequence-out 으로 검증한다.
    """
    names = [n for n in per if per[n][tier]["n"] >= min_n]
    print(f"  [LOSO-스펙트럼] {fe.TIER_NAMES[tier]}  시퀀스 {len(names)} 개")
    if len(names) < 3:
        print("         3 개 미만이라 성립하지 않는다. 판정 보류.\n")
        return
    print(f"{'held-out':>12} {'none':>9} {'scale':>9} {'power':>9} {'적합 p':>8} {'c':>10}   판정")
    win = {"none": 0, "scale": 0, "power": 0}
    rat = []
    for h in names:
        tr = [per[n][tier] for n in names if n != h]
        m = per[h][tier]
        p, c = _fit_power(tr)
        # scale 전용: p=1 로 고정하고 c 만 훈련에서 뽑는다
        L1 = np.log(np.maximum(np.array([_spec_profile(x["ev"], x["pj"], 1.0) for x in tr]),
                               1e-300)).mean(axis=0)
        c_scale = float(np.exp(-L1.mean()))
        cs = {
            "none": _cost(_spec_profile(m["ev"], m["pj"], 1.0)),
            "scale": _cost(c_scale * _spec_profile(m["ev"], m["pj"], 1.0)),
            "power": _cost(c * _spec_profile(m["ev"], m["pj"], p)),
        }
        best = min(cs, key=lambda k: cs[k])
        win[best] += 1
        rat.append(cs["power"] / max(cs["scale"], 1e-12))
        print(f"{h:>12} {cs['none']:>9.3f} {cs['scale']:>9.3f} {cs['power']:>9.3f} "
              f"{p:>8.2f} {c:>10.3g}   {best}")
    g = float(np.exp(np.mean(np.log(np.array(rat)))))
    print(f"{'승수':>12} {win['none']:>9} {win['scale']:>9} {win['power']:>9}")
    print(f"{'':>12} power/scale 비용비 기하평균 {g:.3f}"
          f"   ({'스펙트럼 보정이 이득' if g < 0.95 else '이득 없음' if g < 1.05 else '손해'})\n")


def significance(per: dict, tier: str, min_n: int = 30) -> None:
    """축별 모양이 시퀀스 간 산포보다 큰가.

    shape_j 가 1 에서 떨어져 있어도, 그 떨어짐이 시퀀스마다 제각각이면 보정할
    대상이 아니라 잡음이다. log 로 재고 시퀀스를 표본으로 삼아 t 를 낸다.
    (표본이 7 개뿐이라 t 는 방향 지시자이지 엄밀한 p 값이 아니다 - 그렇게 읽어라.)
    """
    names = [n for n in per if per[n][tier]["n_inv"] >= min_n]
    if len(names) < 3:
        return
    sh = np.log(np.array([per[n][tier]["shape"] for n in names]))
    m, s = sh.mean(axis=0), sh.std(axis=0, ddof=1)
    t = m / np.maximum(s / np.sqrt(len(names)), 1e-12)
    print(f"  [유의성] 시퀀스 {len(names)} 개를 표본으로 log(shape_j) 의 t")
    print(f"{'':>12} " + " ".join(f"{a:>9}" for a in AXES))
    print(f"{'shape':>12} {_fmt(np.exp(m))}")
    print(f"{'t':>12} {_fmt(t)}   <- |t| < 2 면 그 축은 시퀀스 잡음과 구분되지 않는다")
    print()


def loso(per: dict, tier: str, min_n: int = 30, label: str = "") -> None:
    """축별 보정이 **일반화되는가**. 이것이 판정이다.

    Lambda' = D Lambda D 로 고치면 (Lambda'^-1)_jj = (Lambda^-1)_jj / d_j^2 이므로
    R'_j = R_j d_j^2 다. 즉 R_j = 1 로 만드는 보정은 d_j = R_j^-1/2 로 닫힌 형태다.
    적합할 것이 없다 - 남은 질문은 **다른 시퀀스에서도 그 d 가 맞는가** 뿐이다.

    시퀀스 하나를 빼고 나머지에서 d 를 정한 뒤 뺀 시퀀스에 적용한다. 세 가지를
    같은 자리에서 비교한다:
      none    보정 없음 (d = 1)
      scale   스칼라 하나 (27 절의 kappa). 모양은 못 건드린다
      block   둘 - 병진 블록과 회전 블록. 물리적으로 가장 그럴듯한 모양 오차이고
              매개변수가 둘뿐이라 6 개짜리보다 과적합에 강하다
      axis    축별 여섯 개
    axis 도 block 도 scale 을 이기지 못하면 축별 보정은 존재하지 않는 것이다.
    """
    names = [n for n in per if per[n][tier]["n_inv"] >= min_n]
    print(f"  [LOSO] {fe.TIER_NAMES[tier]}  n_inv >= {min_n} 인 시퀀스 {len(names)} 개: "
          f"{', '.join(names) if names else '없음'}"
          + (f"  [{label}]" if label else ""))
    if len(names) < 3:
        print("         시퀀스가 3 개 미만이라 leave-one-out 이 성립하지 않는다. 판정 보류.\n")
        return

    print(f"{'held-out':>12} {'none':>9} {'scale':>9} {'block':>9} {'axis':>9}   판정")
    win = {"none": 0, "scale": 0, "block": 0, "axis": 0}
    ratios: dict[str, list[float]] = {"block": [], "axis": []}
    for h in names:
        train = np.array([per[n][tier]["R"] for n in names if n != h])
        Rh = per[h][tier]["R"]
        # 훈련 시퀀스들의 기하평균이 d^-2 를 준다.
        Rt = np.exp(np.mean(np.log(train), axis=0))
        Rb = np.concatenate([np.repeat(np.exp(np.mean(np.log(Rt[:3]))), 3),
                             np.repeat(np.exp(np.mean(np.log(Rt[3:]))), 3)])
        cs = {
            "none": _cost(Rh),
            "scale": _cost(Rh / np.exp(np.mean(np.log(Rt)))),
            "block": _cost(Rh / Rb),
            "axis": _cost(Rh / Rt),
        }
        best = min(cs, key=lambda k: cs[k])
        win[best] += 1
        for k in ratios:
            ratios[k].append(cs[k] / max(cs["scale"], 1e-12))
        print(f"{h:>12} {cs['none']:>9.3f} {cs['scale']:>9.3f} {cs['block']:>9.3f} "
              f"{cs['axis']:>9.3f}   {best}")
    print(f"{'승수':>12} {win['none']:>9} {win['scale']:>9} {win['block']:>9} "
          f"{win['axis']:>9}   <- scale 을 이기지 못하면 축별 보정은 없다")
    # 승수 세기는 약한 요약이다 - 이겨도 얼마나 이겼는지를 말하지 않는다.
    # 비용비의 기하평균이 진짜 답이다. 1 이면 축별 보정이 순이득 0 이라는 뜻.
    for k in ("block", "axis"):
        g = float(np.exp(np.mean(np.log(np.array(ratios[k])))))
        print(f"{'':>12} {k}/scale 비용비 기하평균 {g:.3f}"
              f"   ({'축별이 이득' if g < 0.95 else '이득 없음' if g < 1.05 else '축별이 손해'})")
    print()


def run_all(seqs=SEQUENCES, center: bool = False) -> dict:
    if center:
        print("### 진단 모드: 시퀀스 평균을 뺐다 (보정 아님, 편향의 몫을 보려는 것) ###\n")
    per: dict[str, dict[str, dict]] = {}
    for name, root, prefix in seqs:
        p = Path(f"{prefix}_tiers.csv")
        if not p.exists():
            print(f"건너뜀 {name}: {p} 없음")
            continue
        s = collect(Path(root), Path(prefix))
        per[name] = {t: measure(s[t]["eps"], s[t]["lam"], center=center)
                     for t in fe.TIERS}

    for t in fe.TIERS:
        print("=" * 108)
        print(f"Tier {t}  {fe.TIER_NAMES[t]}    축별 주변 비율 R_j = E[eps_j^2 / (Lambda^-1)_jj]"
              f"   (옳으면 전부 1)")
        print("=" * 108)
        print(f"{'sequence':>10} {'n':>5} {'n_inv':>6} {'ANEES':>9} "
              + " ".join(f"{a:>9}" for a in AXES)
              + f" {'scale':>9} {'aniso':>7} {'blk t/r':>8} {'Cspread':>8}")
        for name in per:
            m = per[name][t]
            if m["n"] == 0:
                print(f"{name:>10} {'0':>5}  -")
                continue
            print(f"{name:>10} {m['n']:>5} {m['n_inv']:>6} {m['anees']:>9.3g} "
                  f"{_fmt(m['R'])} {m['scale']:>9.3g} {m['aniso']:>7.2f} "
                  f"{m['block_tr']:>8.3g} {m['C_spread']:>8.2f}")

        # (D) Lambda 자신의 고유기저에서 본 프로파일. 축별 보정이 없어도 여기에
        # 구조가 있으면 고칠 수 있는 모양 오차가 남아 있는 것이다.
        print(f"\n{'(D) 스펙트럼':>10}  S_k = E[lambda_k (v_k.eps)^2], k=0 최강축 .. k=5 최약축")
        print(f"{'':>10} {'':>5} {'':>6} {'':>9} "
              + " ".join(f"{'S'+str(k):>9}" for k in range(6)) + f" {'기울기':>9} {'gamma':>8}")
        for name in per:
            m = per[name][t]
            if m["n"] == 0:
                continue
            print(f"{name:>10} {m['n']:>5} {'':>6} {'':>9} {_fmt(m['spec'])} "
                  f"{m['spec_slope']:>9.3f} {m['gamma_med']:>8.3f}")

        # (B) 랭크 안전 지표. SPA 는 조건수 게이트에서 대부분 빠지므로 이쪽만 남는다.
        print(f"\n{'(B) W_j':>10} {'(전 프레임, Lambda 를 뒤집지 않는다)':>28}")
        for name in per:
            m = per[name][t]
            if m["n"] == 0:
                continue
            print(f"{name:>10} {m['n']:>5} {'':>6} {'':>9} {_fmt(m['W'])}")

        # 모양이 시퀀스에 걸쳐 일관적인가. 이것이 판정의 전부다.
        sh = np.array([per[n][t]["shape"] for n in per if per[n][t]["n_inv"] > 0])
        if len(sh) >= 2:
            print(f"\n{'shape 기하평균':>10} {'':>5} {'':>6} {'':>9} "
                  f"{_fmt(np.exp(np.mean(np.log(sh), axis=0)))}")
            # 축별로 시퀀스 간 퍼짐. 1 에 가까우면 일관, 크면 제각각.
            spread = np.exp(np.std(np.log(sh), axis=0))
            print(f"{'시퀀스간 퍼짐':>10} {'':>5} {'':>6} {'':>9} {_fmt(spread)}"
                  f"   <- 1 에 가까울수록 일관")
        print()
        significance(per, t)
        loso(per, t)
        loso_power(per, t)
        # KITTI 와 TUM 을 섞으면 "센서가 다르다" 가 "축이 다르다" 로 보일 수 있다.
        # 도메인 안에서 따로 물어야 그 둘이 갈린다.
        for dom, keys in (("KITTI 만", [n for n in per if n.startswith("kitti")]),
                          ("TUM 만", [n for n in per if n.startswith("fr1")])):
            sub = {k: per[k] for k in keys}
            if sum(1 for k in keys if per[k][t]["n_inv"] >= 30) >= 3:
                loso(sub, t, label=dom)
    return per


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        return selftest()
    run_all(center="--debias" in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
