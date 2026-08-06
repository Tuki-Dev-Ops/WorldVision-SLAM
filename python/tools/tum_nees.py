"""ECDA 가 보고하는 불확실성이 실제로 보정되어 있는가 - 실데이터 NEES.

지금까지 NEES 는 전부 시뮬레이션에서만 쟀다. 여기서 처음으로 실제 센서
데이터에 대고 잰다. 대상은 DirectAligner 가 프레임마다 내놓는 6x6 정보행렬
Lambda 다. 아무도 이 행렬이 보정되어 있는지 확인한 적이 없다.

무엇을 재는가
  프레임마다 추정 상대 포즈 T_cur_ref 와 진리값 상대 포즈를 비교해 리군에서
  6차원 오차를 만들고, 그 오차를 보고된 정보행렬로 정규화한다.

    e = log( T_est * T_gt^-1 )        좌측 섭동 규약 (T <- exp(dxi)*T)
    NEES = e^T Lambda e

  추정기가 일관되면 E[NEES] = 6 이다. 6 보다 크면 공분산이 실제보다 작다는
  뜻, 곧 과신이다.

무엇을 같이 재야 하는가 (docs/06-results.md 10.3)
  일관성만 보면 안 된다. 공분산을 부풀린 추정기는 NEES 를 통과하면서 쓸모가
  없다. 그래서 같은 표에 상대 포즈 오차(정확도)를 반드시 같이 낸다.

스칼라 하나로 고칠 수 있는가
  ANEES 를 6 으로 만드는 배율은 언제나 존재한다(= ANEES/6). 그 배율이 답이
  되려면 오보정이 방향에 무관해야 한다. 그래서 병진 블록과 회전 블록을 따로,
  또 6개 축을 각각 따로 잰다. 블록별 배율이 서로 다르면 스칼라 하나로는
  고칠 수 없고, 그 사실 자체가 결과다.

상관인가 모델 오차인가 - 점 개수 스케일링
  과신 배율을 한 프레임 안에서 쪼개는 데는 한계가 있다. 잔차 상관과 N 에
  따라 줄지 않는 항은 한 번의 실행에서 구분되지 않는다. 구분하는 방법은
  하나뿐이다: 점 집합을 솎아 여러 번 돌리고 무엇이 N 을 따라가는지 본다.

    Lambda ~ N^1     : 구조적으로 당연하다 (점마다 J J^T 를 더한다)
    오차^2 ~ N^-1    : 잔차가 독립이면 이래야 한다
    오차^2 ~ N^-a    : 0<a<1 이면 상관. a=0 이면 아예 줄지 않는 항이다
    ANEES ~ N^(1-a)  : 위 둘의 곱. 기울기 1 이면 전부 비수축 항이다

  --scaling 이 이 기울기를 잰다. 점 개수는 wme_tum_odometry --max-points 로
  줄인다 (균등 stride 라 공간 분포가 유지된다).

파이프라인이 판별력을 갖는지 (docs/06-results.md 10.4)
  --selftest 는 일부러 부풀린/줄인 공분산과, 보고된 Lambda 로부터 직접
  뽑은 합성 오차를 넣어 ANEES 가 반드시 움직여야 하는 방향으로 움직이는지
  확인한다. 이걸 통과하지 못하는 측정은 아무것도 말해주지 않는다.

사용:
  python tools/tum_nees.py <시퀀스경로> <진단.csv>
  python tools/tum_nees.py --all <데이터경로> <진단디렉토리>
  python tools/tum_nees.py --scaling <데이터경로> <진단디렉토리>...
  python tools/tum_nees.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wme.eval.stats import chi2_interval  # noqa: E402
from wme.reference.geometry import SE3, quat_to_matrix, so3_exp, so3_log  # noqa: E402

DOF = 6
SEQUENCES = [
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_walking_xyz",
]

# 정보행렬 상삼각 21개의 열 이름. C++ 쪽 기록 순서와 반드시 같아야 한다.
INFO_COLUMNS = [f"info_{i}{j}" for i in range(6) for j in range(i, 6)]


# ===========================================================================
# 진리값 보간
# ===========================================================================

def interpolate_pose(stamps: np.ndarray, poses: list[SE3], t: float,
                     max_gap: float = 0.05) -> SE3 | None:
    """진리값 궤적을 시각 t 로 보간한다.

    최근접 표본을 쓰면 안 된다. TUM 진리값은 100 Hz 라 최근접 오차가 최대
    5 ms 이고, 0.3 m/s 로 움직이면 그것만으로 1.5 mm 가 생긴다. 재려는 상대
    포즈 오차 자체가 수 mm 수준이므로 같은 자릿수의 잡음을 스스로 집어넣는
    셈이 된다. 회전은 측지선 보간, 병진은 선형 보간.
    """
    if t < stamps[0] or t > stamps[-1]:
        return None
    j = int(np.searchsorted(stamps, t))
    if j == 0:
        return poses[0]
    i = j - 1
    t0, t1 = float(stamps[i]), float(stamps[j])
    if t1 - t0 > max_gap:
        return None
    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    p0, p1 = poses[i], poses[j]
    R = p0.R @ so3_exp(a * so3_log(p0.R.T @ p1.R))
    return SE3(R, (1.0 - a) * p0.t + a * p1.t)


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


# ===========================================================================
# 진단 CSV 읽기
# ===========================================================================

def unpack_information(row: dict) -> np.ndarray:
    """상삼각 21개 -> 대칭 6x6."""
    L = np.zeros((6, 6))
    k = 0
    for i in range(6):
        for j in range(i, 6):
            L[i, j] = L[j, i] = float(row[INFO_COLUMNS[k]])
            k += 1
    return L


def read_diag(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"진단 행이 없음: {path}")
    missing = [c for c in ("ref_timestamp", "rel_tx", "rel_qw", *INFO_COLUMNS)
               if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path}: 정보행렬 열이 없다. tools/tum_odometry.cpp 를 다시 빌드하고 "
            f"--diag 로 다시 돌려야 한다 (누락: {missing[:3]}...)")
    return rows


# ===========================================================================
# 프레임별 오차와 정보행렬
# ===========================================================================

@dataclass
class FrameData:
    stamp: float
    error: np.ndarray            # (6,) 리군 오차 [rho, phi]
    information: np.ndarray      # (6, 6)
    trans_err: float             # 상대 포즈 병진 오차 (m)
    rot_err: float               # 상대 포즈 회전 오차 (rad)
    motion: float                # 진리값 상대 운동 크기 (m)
    dt: float                    # ref -> cur 시간 간격 (s)
    points: float = 0.0          # 정렬에 쓴 점 개수
    rmse: float = 0.0            # 실제 측광 잔차 RMS (intensity)
    photo_var: float = 0.0       # Lambda 를 나눈 분산 (intensity^2)


def collect(root: Path, diag_path: Path, gt_tol: float = 0.05,
            gt_shift: float = 0.0, nearest: bool = False
            ) -> tuple[list[FrameData], dict]:
    """진단 CSV 와 진리값을 짝지어 프레임별 오차/정보행렬을 만든다.

    gt_shift / nearest 는 통제용이다. 재고 있는 오차가 추정기의 것인지 진리값
    정렬의 것인지 구분하려면, 진리값 시계를 일부러 흔들었을 때 결과가 얼마나
    움직이는지를 봐야 한다. 크게 움직이면 이 실험은 추정기가 아니라 시계를
    재고 있는 것이다.
    """
    stamps, poses = load_groundtruth(root)
    stamps = stamps + gt_shift
    rows = read_diag(diag_path)

    out: list[FrameData] = []
    skipped_gt = 0
    for r in rows:
        t_cur = float(r["timestamp"])
        t_ref = float(r["ref_timestamp"])
        if t_ref >= t_cur:
            continue                       # 참조 프레임 자신 (상대 포즈가 항등)

        if nearest:
            g_cur = poses[int(np.argmin(np.abs(stamps - t_cur)))]
            g_ref = poses[int(np.argmin(np.abs(stamps - t_ref)))]
        else:
            g_cur = interpolate_pose(stamps, poses, t_cur, gt_tol)
            g_ref = interpolate_pose(stamps, poses, t_ref, gt_tol)
        if g_cur is None or g_ref is None:
            skipped_gt += 1
            continue

        # 진리값 상대 포즈. ref 카메라 좌표 -> cur 카메라 좌표.
        T_gt = g_cur.inverse() @ g_ref
        T_est = SE3(quat_to_matrix(float(r["rel_qx"]), float(r["rel_qy"]),
                                   float(r["rel_qz"]), float(r["rel_qw"])),
                    np.array([float(r["rel_tx"]), float(r["rel_ty"]),
                              float(r["rel_tz"])]))

        # 좌측 섭동 규약: T_est = exp(e) * T_gt.
        # 정보행렬이 정의된 섭동과 같은 규약이어야 한다 - 우측 섭동을 쓰면
        # 같은 오차가 다른 좌표에서 표현되어 Lambda 와 짝이 맞지 않는다.
        e = (T_est @ T_gt.inverse()).log()

        d = T_est.inverse() @ T_gt
        out.append(FrameData(
            stamp=t_cur,
            error=e,
            information=unpack_information(r),
            trans_err=float(np.linalg.norm(d.t)),
            rot_err=float(np.linalg.norm(so3_log(d.R))),
            motion=float(np.linalg.norm(T_gt.t)),
            dt=t_cur - t_ref,
            points=float(r["points"]),
            rmse=float(r["rmse"]),
            photo_var=float(r.get("photo_var", "nan")),
        ))

    return out, {"rows": len(rows), "skipped_gt": skipped_gt}


# ===========================================================================
# NEES
# ===========================================================================

def nees_full(errors: np.ndarray, infos: np.ndarray) -> np.ndarray:
    """NEES_i = e^T Lambda e. 정보행렬을 그대로 쓰므로 역행렬이 필요 없다."""
    return np.einsum("ni,nij,nj->n", errors, infos, errors)


def nees_block(errors: np.ndarray, infos: np.ndarray, sl: slice,
               rcond: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """부분 블록의 주변(marginal) NEES.

    "병진 불확실성이 맞는가" 는 주변 질문이므로 Lambda 의 부분블록이 아니라
    P = Lambda^-1 의 부분블록을 써야 한다. 조건화(Lambda 부분블록)를 쓰면
    회전을 안다고 가정한 셈이 되어 다른 질문에 답하게 된다.
    """
    n = len(errors)
    vals = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            P = np.linalg.inv(infos[i])
        except np.linalg.LinAlgError:
            continue
        Pb = P[sl, sl]
        eb = errors[i][sl]
        # 대칭·양정치가 아니면 그 프레임은 버린다. 조용히 통과시키면
        # 퇴화한 프레임이 숫자를 만들어 낸다.
        w, _ = np.linalg.eigh(0.5 * (Pb + Pb.T))
        if w.min() <= rcond * max(w.max(), 1e-300):
            continue
        vals[i] = float(eb @ np.linalg.solve(Pb, eb))
        ok[i] = True
    return vals, ok


def anees_report(per_sample: np.ndarray, dof: int,
                 confidence: float = 0.95) -> dict:
    """ANEES 와 그 카이제곱 수용구간.

    N*ANEES ~ chi2(N*dof) 이므로 구간은 표본 수에 따라 좁아진다. 표본 수를
    빼고 "ANEES 가 1.2 라 거의 맞다" 같은 말을 하면 안 된다 - n=200 이면
    1.2 는 명백한 기각이다.
    """
    n = len(per_sample)
    a = float(np.mean(per_sample))
    lo, hi = chi2_interval(n * dof, confidence)
    lo, hi = lo / n, hi / n          # ANEES 스케일로
    return {
        "n": n,
        "anees": a,
        "anees_per_dof": a / dof,
        "lo": lo,
        "hi": hi,
        "lo_per_dof": lo / dof,
        "hi_per_dof": hi / dof,
        "consistent": bool(lo <= a <= hi),
        "overconfident": bool(a > hi),
        # 공분산에 곱해야 할 배율. Lambda 는 그만큼 나눈다.
        "inflation": a / dof,
        "median_nees": float(np.median(per_sample)),
    }


def analyse(frames: list[FrameData], scale: float = 1.0) -> dict:
    """한 시퀀스 전체. scale 은 공분산 배율 (판별력 검증용)."""
    e = np.array([f.error for f in frames])
    L = np.array([f.information for f in frames]) / scale

    full = anees_report(nees_full(e, L), DOF)

    blocks = {}
    for name, sl in (("trans", slice(0, 3)), ("rot", slice(3, 6))):
        v, ok = nees_block(e, L, sl)
        blocks[name] = (anees_report(v[ok], 3) if ok.sum() >= 10
                        else {"n": int(ok.sum())})
        blocks[name]["dropped"] = int((~ok).sum())

    # 축별. z_k = e_k / sqrt(P_kk), E[z^2] = 1 이어야 한다.
    # 블록 두 개로는 "회전 안에서도 축마다 다른가" 를 못 본다.
    per_axis = np.full(6, np.nan)
    zs = []
    for i in range(len(frames)):
        try:
            P = np.linalg.inv(L[i])
        except np.linalg.LinAlgError:
            continue
        d = np.diag(P)
        if np.any(d <= 0):
            continue
        zs.append(e[i] / np.sqrt(d))
    if zs:
        per_axis = np.mean(np.array(zs) ** 2, axis=0)

    # 프레임별 95 % 게이트 통과율. 평균만 보면 소수의 폭주 프레임이 전부를
    # 설명하는 경우를 놓친다.
    gate = chi2_interval(DOF, 0.95)[1]
    inside = float(np.mean(nees_full(e, L) <= gate))

    return {
        "full": full,
        "blocks": blocks,
        "per_axis": per_axis,
        "frac_within_95": inside,
        # 정확도. 일관성과 반드시 같이 봐야 한다.
        "trans_err_med": float(np.median([f.trans_err for f in frames])),
        "trans_err_rms": float(np.sqrt(np.mean([f.trans_err ** 2 for f in frames]))),
        "rot_err_med": float(np.median([f.rot_err for f in frames])),
        "motion_med": float(np.median([f.motion for f in frames])),
        "dt_med": float(np.median([f.dt for f in frames])),
    }


# ===========================================================================
# 판별력 검증
# ===========================================================================

def discrimination_check(frames: list[FrameData], seed: int = 0) -> list[str]:
    """일부러 틀린 공분산을 넣어 ANEES 가 움직이는지 본다.

    셋을 본다.
      1. 공분산을 s 배 부풀리면 ANEES 는 정확히 1/s 배가 되어야 한다.
      2. 보고된 Lambda 자체가 참이라고 가정하고 e ~ N(0, Lambda^-1) 를 뽑아
         넣으면 ANEES 는 6 이 나와야 한다. 나오지 않으면 오차 계산이나
         정보행렬 해석 중 하나가 틀린 것이지 추정기 문제가 아니다.
      3. 오차를 무작위로 섞으면(프레임과 정보행렬의 짝을 깨면) 결과가 달라져야
         한다. 달라지지 않으면 정보행렬이 프레임마다 다른 정보를 담고 있지
         않다는 뜻이다.
    """
    lines = []
    base = analyse(frames)["full"]["anees"]
    lines.append(f"  기준                      ANEES {base:10.2f}")
    for s in (0.01, 100.0):
        a = analyse(frames, scale=s)["full"]["anees"]
        lines.append(f"  공분산 x{s:<8g}         ANEES {a:10.2f}   "
                     f"(예상 {base / s:.2f}, 비 {a / (base / s):.4f})")

    rng = np.random.default_rng(seed)
    e_syn = []
    keep = []
    for f in frames:
        w, V = np.linalg.eigh(0.5 * (f.information + f.information.T))
        if w.min() <= 0:
            continue
        # e = V diag(w^-1/2) n  =>  Cov(e) = Lambda^-1
        e_syn.append(V @ (rng.standard_normal(6) / np.sqrt(w)))
        keep.append(f)
    syn = [FrameData(f.stamp, e, f.information, 0.0, 0.0, f.motion, f.dt)
           for f, e in zip(keep, e_syn)]
    r = analyse(syn)["full"]
    lines.append(f"  합성오차 ~ N(0, Lambda^-1)  ANEES {r['anees']:10.2f}   "
                 f"(반드시 6 근처, 구간 [{r['lo']:.2f}, {r['hi']:.2f}]) "
                 f"-> {'OK' if r['consistent'] else '실패'}")

    idx = rng.permutation(len(frames))
    shuffled = [FrameData(f.stamp, frames[k].error, f.information,
                          0.0, 0.0, f.motion, f.dt)
                for f, k in zip(frames, idx)]
    a = analyse(shuffled)["full"]["anees"]
    lines.append(f"  오차-정보행렬 짝 섞음      ANEES {a:10.2f}   "
                 f"(기준 대비 {a / base:.2f}x)")
    return lines


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위상관. scipy 없이."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def decompose(frames: list[FrameData]) -> list[str]:
    """과신 배율을 원인별로 쪼갠다. 주장이 아니라 측정으로.

    전제: Lambda 가 InformationModel::SensorVariance 로 만들어졌을 때만 (a)/(b)
    분해가 뜻을 갖는다. ClusterRobust 로 만든 Lambda 는 photo_var 로 나눈 것이
    아니므로 아래 (a) 는 그 실행과 무관한 숫자가 된다. 그 경우 볼 것은
    --scaling 의 N 기울기다.

    Lambda = (sum_i w_i J_i J_i^T) / sigma_assumed^2 이다. 과신에는 서로 다른
    원인이 곱으로 들어간다.

      (a) 분산 불일치. sigma_assumed^2 은 ImageQuality 가 잰 센서 잡음(+블러)
          이지 실제 잔차 분산이 아니다. 실제 잔차 RMS 는 진단에 남아 있으므로
          비율을 그대로 잴 수 있다.
      (b) 남은 배율. (a) 로 나누고 남은 것.

    (b) 를 곧바로 "잔차 상관" 이라고 부르면 안 된다. 상관으로 설명하려면
    유효 표본수 N_eff = N / (b) 가 최소 1 이어야 한다. N_eff 가 1 아래로
    내려가면 상관만으로는 설명이 되지 않고, N 에 따라 줄지 않는 항
    (docs/04-unified-objective.md 5.2.1(c) 의 Sigma_sys)이 남아 있다는 뜻이다.
    그 두 경우를 가르는 것이 이 함수의 목적이다.
    """
    e = np.array([f.error for f in frames])
    L = np.array([f.information for f in frames])
    per = nees_full(e, L)
    total = float(per.mean()) / DOF

    pv = np.array([f.photo_var for f in frames])
    rm = np.array([f.rmse for f in frames])
    npt = np.array([f.points for f in frames])
    lines = []
    if not np.all(np.isfinite(pv)):
        lines.append("  photo_var 열이 없어 분해 불가 (진단 CSV 를 다시 생성해야 함)")
        return lines

    # (a) 프레임별로 실제 잔차 분산으로 다시 스케일한 Lambda 로 NEES 를 다시 잰다.
    ratio = (rm ** 2) / np.maximum(pv, 1e-9)
    per_b = nees_full(e, L / ratio[:, None, None])
    rem = float(per_b.mean()) / DOF

    # 전제를 매번 찍는다. 자동으로 알아낼 방법이 없기 때문이다 - 진단 CSV 에는
    # 어떤 InformationModel 로 만든 Lambda 인지가 남지 않고, "N_eff 가 N 을
    # 넘으면 아니다" 같은 검사는 실측에서 통과해 버린다(fr1_360: N_eff 280 <
    # N 341 인데도 EffectiveSample 실행이었다). 조용히 그럴듯한 표를 내는 것보다
    # 매번 전제를 되풀이하는 편이 낫다 (docs/06-results.md 10.4).
    lines.append("  전제: Lambda = H / photo_var (InformationModel::SensorVariance).")
    lines.append("        다른 모델로 만든 Lambda 에는 아래 (a)/(b) 가 적용되지"
                 " 않는다. 그 경우 --scaling 의 N 기울기를 봐야 한다.")
    lines.append(f"  전체 배율                         {total:12.1f} x")
    lines.append(f"  (a) 가정분산 vs 실제 잔차분산     "
                 f"{float(np.mean(ratio)):12.1f} x   "
                 f"(sigma_assumed {np.sqrt(np.median(pv)):.2f} -> "
                 f"실측 잔차 RMS {np.median(rm):.2f} intensity)")
    lines.append(f"  (b) 그러고도 남는 배율            {rem:12.1f} x")

    n_eff = float(np.median(npt)) / rem
    lines.append(f"      점 개수 중앙 {np.median(npt):.0f} 에 대한 "
                 f"유효 표본수 N_eff = {n_eff:.2f}")
    if n_eff >= 1.0:
        lines.append("      -> N_eff >= 1. 잔차 상관만으로 설명 가능한 범위 안이다.")
    else:
        lines.append("      -> N_eff < 1. 잔차가 완전히 상관되어 한 점과 같아도"
                     " 모자란다.")
        lines.append("         즉 N 에 따라 줄지 않는 항(모델 오차)이 남아 있다.")

    # N 에 따라 줄지 않는 항이 있으면, 점이 많을수록 Lambda 는 커지는데 오차는
    # 줄지 않으므로 NEES 가 점 개수와 같이 커져야 한다. 상관이 원인이라면
    # (N_eff 가 대체로 일정하다면) 그런 추세가 없어야 한다.
    rho = spearman(npt, per_b)
    lines.append(f"  NEES(잔차분산 보정 후) vs 점 개수 순위상관 {rho:+.3f}"
                 f"   ({'점이 많을수록 더 과신 - 비수축 항의 징후' if rho > 0.2 else '뚜렷한 추세 없음'})")
    return lines


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """log y = s log x + c 의 s. 표본이 둘 미만이거나 비양수면 nan."""
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    return float(np.polyfit(lx, ly, 1)[0])


def scaling_report(runs: list[tuple[str, list[FrameData]]]) -> list[str]:
    """점 개수를 바꿔 여러 번 돌린 결과에서 N 스케일링 기울기를 뽑는다.

    한 프레임 안의 통계만으로는 잔차 상관과 비수축 항(Sigma_sys)이 구분되지
    않는다. 둘 다 "정보가 실제보다 크다" 로만 보이기 때문이다. 구분되는 곳은
    N 축이다.

      상관   : 유효 표본수 N_eff = N/kappa 로 여전히 N 에 비례해 줄어든다
               -> 오차^2 ~ N^-1, ANEES 는 N 에 무관 (기울기 0)
      비수축 : 오차가 N 과 무관하다
               -> 오차^2 ~ N^0,  ANEES ~ N^1 (기울기 1)

    기울기가 1 에 가까우면 상관 모델로는 설명할 수 없다. 유효 표본수를 아무리
    줄여도 N 에 비례하는 항인 이상 언젠가는 맞아떨어져야 하는데, 그러지 않기
    때문이다.
    """
    lines = []
    n, tr, an, lam = [], [], [], []
    lines.append(f"    {'run':<8} {'N':>7} {'trErr med':>10} {'tr(L_tt) med':>13} "
                 f"{'ANEES':>12}")
    for label, frames in runs:
        if len(frames) < 20:
            continue
        e = np.array([f.error for f in frames])
        L = np.array([f.information for f in frames])
        npt = float(np.median([f.points for f in frames]))
        te = float(np.median([f.trans_err for f in frames]))
        tl = float(np.median([np.trace(x[:3, :3]) for x in L]))
        a = float(np.mean(nees_full(e, L)))
        n.append(npt); tr.append(te); an.append(a); lam.append(tl)
        lines.append(f"    {label:<8} {npt:7.0f} {te*1000:10.3f} {tl:13.4g} "
                     f"{a:12.1f}")

    if len(n) < 3:
        lines.append("    실행이 3 개 미만이라 기울기를 내지 않는다")
        return lines

    x = np.array(n)
    s_lam = loglog_slope(x, np.array(lam))
    s_err = loglog_slope(x, np.array(tr) ** 2)
    s_an = loglog_slope(x, np.array(an))
    lines.append(f"    기울기(log-log)  Lambda~N^{s_lam:+.2f}   "
                 f"오차^2~N^{s_err:+.2f}   ANEES~N^{s_an:+.2f}")
    if s_an > 0.7:
        lines.append("    -> ANEES 가 N 을 그대로 따라간다. 점을 더 넣어도 오차는")
        lines.append("       줄지 않는데 정보만 커진다. 상관 모델(N_eff = N/kappa)")
        lines.append("       로는 설명되지 않는다 - N 에 따라 줄지 않는 항이다.")
    elif s_an > 0.3:
        lines.append("    -> 절반쯤 따라간다. 상관과 비수축 항이 섞여 있다.")
    else:
        lines.append("    -> ANEES 가 N 에 무관하다. 남은 오보정은 비수축 항이 아니라"
                     " 프레임 단위 상수다.")
    return lines


def selftest() -> int:
    """데이터 없이 파이프라인만 검증한다. 실패하면 이 도구를 믿으면 안 된다."""
    rng = np.random.default_rng(7)
    n = 4000
    ok = True

    # 참 공분산을 알고 있는 합성 문제
    A = rng.standard_normal((6, 6))
    P = A @ A.T + 6.0 * np.eye(6)
    L = np.linalg.inv(P)
    C = np.linalg.cholesky(P)
    e = (C @ rng.standard_normal((6, n))).T
    infos = np.repeat(L[None], n, axis=0)

    r = anees_report(nees_full(e, infos), DOF)
    print(f"보정된 공분산            ANEES {r['anees']:.3f} "
          f"구간 [{r['lo']:.3f}, {r['hi']:.3f}]  "
          f"{'OK' if r['consistent'] else '실패'}")
    ok &= r["consistent"]

    for s, label in ((0.1, "10x 과신 (공분산 1/10)"), (10.0, "10x 과소신뢰")):
        rs = anees_report(nees_full(e, infos / s), DOF)
        want = 6.0 / s
        good = abs(rs["anees"] / want - 1.0) < 0.1 and not rs["consistent"]
        print(f"{label:<24} ANEES {rs['anees']:.3f} (예상 {want:.2f})  "
              f"{'OK' if good else '실패'}")
        ok &= good

    # 블록 경로도 같은 방식으로 검증한다. 전체가 맞아도 블록이 틀릴 수 있다.
    v, m = nees_block(e, infos, slice(0, 3))
    rb = anees_report(v[m], 3)
    good = rb["consistent"]
    print(f"{'병진 블록 (주변)':<24} ANEES {rb['anees']:.3f} "
          f"구간 [{rb['lo']:.3f}, {rb['hi']:.3f}]  {'OK' if good else '실패'}")
    ok &= good

    # 리군 규약: e = log(T_est * T_gt^-1) 가 곧 T_est = exp(e) * T_gt 인가
    T_gt = SE3(so3_exp(np.array([0.1, -0.2, 0.05])), np.array([0.3, -0.1, 0.2]))
    xi = np.array([0.01, -0.02, 0.03, 0.004, -0.001, 0.002])
    T_est = SE3.exp(xi) @ T_gt
    e1 = (T_est @ T_gt.inverse()).log()
    good = float(np.max(np.abs(e1 - xi))) < 1e-9
    print(f"{'좌측 섭동 규약 왕복':<24} 최대오차 {np.max(np.abs(e1 - xi)):.2e}  "
          f"{'OK' if good else '실패'}")
    ok &= good

    # 스케일링 기울기. 두 극단을 넣어 판별력을 본다 - 이 검사가 없으면
    # "기울기 1" 이라는 결론이 도구의 성질인지 데이터의 성질인지 알 수 없다.
    for kind, want, label in (("indep", 0.0, "잔차 독립 (오차^2 ~ 1/N)"),
                              ("sys", 1.0, "비수축 항 (오차 N 무관)")):
        runs = []
        for npt in (100.0, 200.0, 400.0, 800.0, 1600.0):
            L = np.repeat((npt * np.eye(6))[None], 600, axis=0)
            s = 1.0 if kind == "sys" else 1.0 / np.sqrt(npt)
            er = s * rng.standard_normal((600, 6)) / np.sqrt(100.0)
            runs.append((f"n{int(npt)}", [
                FrameData(0.0, er[i], L[i], float(np.linalg.norm(er[i, :3])),
                          0.0, 0.0, 0.0, points=npt)
                for i in range(600)]))
        txt = "\n".join(scaling_report(runs))
        s_an = float(txt.split("ANEES~N^")[1].split()[0])
        g = abs(s_an - want) < 0.15
        print(f"{label:<24} ANEES 기울기 {s_an:+.2f} (예상 {want:+.2f})  "
              f"{'OK' if g else '실패'}")
        ok &= g

    print("\n판별력 자체검증 " + ("통과" if ok else "실패"))
    return 0 if ok else 1


# ===========================================================================
# 출력
# ===========================================================================

def print_one(name: str, frames: list[FrameData], meta: dict,
              show_check: bool) -> dict:
    print(f"\n===== {name} =====")
    print(f"프레임 {len(frames)}  (진단행 {meta['rows']}, 진리값 없음 "
          f"{meta['skipped_gt']})")
    if len(frames) < 20:
        print("표본이 부족해 판정하지 않는다")
        return {}

    a = analyse(frames)
    f = a["full"]

    print(f"참조간격 중앙 {a['dt_med']*1000:.0f} ms   "
          f"진리값 상대운동 중앙 {a['motion_med']*1000:.1f} mm")
    print(f"정확도   상대병진오차 중앙 {a['trans_err_med']*1000:.2f} mm  "
          f"RMS {a['trans_err_rms']*1000:.2f} mm   "
          f"회전오차 중앙 {np.degrees(a['rot_err_med']):.3f} deg")
    print()
    verdict = ("일관" if f["consistent"]
               else ("과신" if f["overconfident"] else "과소신뢰"))
    print(f"ANEES (6 dof)  {f['anees']:12.1f}   "
          f"수용구간 [{f['lo']:.2f}, {f['hi']:.2f}]  n={f['n']}  -> {verdict}")
    print(f"  중앙 NEES    {f['median_nees']:12.1f}   "
          f"프레임별 95% 게이트 통과율 {a['frac_within_95']*100:.1f} %")
    print(f"  ANEES=6 으로 만드는 공분산 배율 : {f['inflation']:.1f}x")

    # 그 배율을 정보모델의 상수로 환산한다. 배율 자체는 "얼마나 틀렸나" 만
    # 말해 주고, 어느 상수를 어디로 옮겨야 하는지는 말해 주지 않는다.
    #
    #   EffectiveSample : Lambda ~ nu           -> 필요한 nu = nu_used / s
    #   CoherentFrame   : Lambda ~ 1/sigma_c^2  -> 필요한 sigma_c = sigma_used * sqrt(s)
    #
    # 두 줄을 같이 찍는 이유는 시퀀스 사이에서 어느 쪽이 덜 흔들리는지가
    # 곧 어느 쪽이 센서에 안 묶인 상수인지이기 때문이다. 12 시퀀스 3 카메라
    # 실측: 필요한 nu 0.10~2.35 (23배), 필요한 sigma_c 10.0~24.3 (2.4배).
    s = f["inflation"]
    print(f"  같은 배율을 상수로 환산 (기본값 기준)")
    print(f"    EffectiveSample 이면 필요한 nu        {1.0 / s:8.3f}   (기본 1.0)")
    print(f"    CoherentFrame 이면 필요한 sigma_c     {15.5 * np.sqrt(s):8.2f}   (기본 15.5)")
    print()
    print("  블록별 (주변 공분산 기준, 각 3 dof)")
    for key, label in (("trans", "병진"), ("rot", "회전")):
        b = a["blocks"][key]
        if "anees" not in b:
            print(f"    {label}  표본 부족 (버림 {b['dropped']})")
            continue
        print(f"    {label}  ANEES {b['anees']:11.1f}  "
              f"구간 [{b['lo']:.2f}, {b['hi']:.2f}]  n={b['n']}  "
              f"배율 {b['inflation']:9.1f}x  (버림 {b['dropped']})")
    axis = a["per_axis"]
    print("  축별 E[z^2] (1 이어야 함): " +
          "  ".join(f"{v:.3g}" for v in axis))
    if np.all(np.isfinite(axis)) and axis.min() > 0:
        print(f"    최대/최소 비 {axis.max()/axis.min():.1f}  "
              f"-> {'스칼라로 안 됨' if axis.max()/axis.min() > 3 else '대체로 균일'}")

    print("\n  [분해] 과신 배율의 출처")
    for line in decompose(frames):
        print(line)

    if show_check:
        print("\n  [판별력] 일부러 틀린 입력에 대한 반응")
        for line in discrimination_check(frames):
            print(line)

    return a


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("root", nargs="?", help="시퀀스 경로")
    ap.add_argument("diag", nargs="?", help="진단 CSV")
    ap.add_argument("--all", nargs=2, metavar=("DATA", "DIAGDIR"),
                    help="다섯 시퀀스를 한 번에")
    ap.add_argument("--scaling", nargs="+", metavar=("DATA", "DIAGDIR"),
                    help="점 개수를 줄여가며 돌린 여러 진단디렉토리의 N 스케일링")
    ap.add_argument("--selftest", action="store_true",
                    help="합성 입력으로 파이프라인 판별력만 검증")
    ap.add_argument("--no-check", action="store_true",
                    help="시퀀스별 판별력 검증 생략")
    ap.add_argument("--gt-shift", type=float, default=0.0,
                    help="진리값 시계를 이만큼(초) 민다. 통제용")
    ap.add_argument("--gt-nearest", action="store_true",
                    help="보간 대신 최근접 진리값 표본을 쓴다. 통제용")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.scaling:
        if len(args.scaling) < 4:
            print("--scaling 은 <데이터경로> 와 진단디렉토리 3 개 이상이 필요하다",
                  file=sys.stderr)
            return 2
        data = Path(args.scaling[0])
        dirs = [Path(d) for d in args.scaling[1:]]
        for s in SEQUENCES:
            root = data / s
            if not (root / "rgb.txt").exists():
                continue
            runs = []
            for d in dirs:
                p = d / f"{s}_diag.csv"
                if p.exists():
                    runs.append((d.name, collect(root, p)[0]))
            if len(runs) < 3:
                continue
            print(f"\n===== {s} =====")
            for line in scaling_report(runs):
                print(line)
        return 0

    jobs: list[tuple[str, Path, Path]] = []
    if args.all:
        data, ddir = Path(args.all[0]), Path(args.all[1])
        for s in SEQUENCES:
            d = ddir / f"{s}_diag.csv"
            if (data / s / "rgb.txt").exists() and d.exists():
                jobs.append((s, data / s, d))
    elif args.root and args.diag:
        jobs.append((Path(args.root).name, Path(args.root), Path(args.diag)))
    else:
        print(__doc__)
        return 2

    if not jobs:
        print("돌릴 시퀀스가 없다", file=sys.stderr)
        return 1

    summary = []
    for name, root, diag in jobs:
        frames, meta = collect(root, diag, gt_shift=args.gt_shift,
                               nearest=args.gt_nearest)
        a = print_one(name, frames, meta, not args.no_check)
        if a:
            summary.append((name, a))

    if len(summary) > 1:
        print("\n\n===== 요약 =====")
        print(f"{'시퀀스':<38} {'n':>4} {'ANEES':>10} {'배율':>9} "
              f"{'병진배율':>9} {'회전배율':>9} {'상대오차mm':>10}")
        for name, a in summary:
            bt = a["blocks"]["trans"].get("inflation", float("nan"))
            br = a["blocks"]["rot"].get("inflation", float("nan"))
            print(f"{name:<38} {a['full']['n']:>4} {a['full']['anees']:>10.1f} "
                  f"{a['full']['inflation']:>9.1f} {bt:>9.1f} {br:>9.1f} "
                  f"{a['trans_err_med']*1000:>10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
