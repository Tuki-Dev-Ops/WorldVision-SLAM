"""궤적 평가 지표.

ATE / RPE 는 SLAM 논문의 표준 지표이고, 여기 구현은 Sturm et al. (IROS 2012)
정의를 따른다. 자체 정의를 만들면 기존 논문 수치와 비교가 불가능해진다.

ATE  - 전역 일관성. Sim(3)/SE(3) 정렬 후 절대 위치 오차.
RPE  - 국소 드리프트. 고정 시간 간격의 상대 변환 오차.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reference.geometry import SE3, so3_log


@dataclass
class Trajectory:
    """시각 오름차순 포즈열."""
    stamps: np.ndarray            # (N,) seconds
    poses: list[SE3]              # world <- body

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def positions(self) -> np.ndarray:
        return np.array([p.t for p in self.poses])

    def sorted(self) -> "Trajectory":
        idx = np.argsort(self.stamps)
        return Trajectory(self.stamps[idx], [self.poses[i] for i in idx])


def associate(est: Trajectory, gt: Trajectory,
              max_difference: float = 0.02) -> tuple[Trajectory, Trajectory]:
    """시각 기준으로 두 궤적을 짝짓는다.

    탐욕적 최근접이 아니라 '한 번 쓴 gt 는 다시 쓰지 않는' 방식이다.
    중복 사용하면 저프레임레이트 추정치가 실제보다 좋게 나온다.
    """
    est, gt = est.sorted(), gt.sorted()
    used = np.zeros(len(gt), dtype=bool)

    e_idx, g_idx = [], []
    for i, t in enumerate(est.stamps):
        diffs = np.abs(gt.stamps - t)
        diffs[used] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= max_difference:
            used[j] = True
            e_idx.append(i)
            g_idx.append(j)

    return (Trajectory(est.stamps[e_idx], [est.poses[i] for i in e_idx]),
            Trajectory(gt.stamps[g_idx], [gt.poses[j] for j in g_idx]))


def umeyama(src: np.ndarray, dst: np.ndarray,
            with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """src -> dst 정렬 변환 (R, t, s). 단안 SLAM 평가는 scale 자유도가 필요하다."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 3:
        raise ValueError("정렬에는 3개 이상의 대응이 필요")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    Xs, Xd = src - mu_s, dst - mu_d

    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0            # 반사 배제

    R = U @ S @ Vt
    if with_scale:
        var_s = float(np.sum(Xs ** 2) / n)
        s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


@dataclass
class AteResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    count: int
    scale: float
    errors: np.ndarray

    def summary(self) -> str:
        return (f"ATE  rmse {self.rmse:.4f} m  mean {self.mean:.4f}  "
                f"median {self.median:.4f}  max {self.max:.4f}  n={self.count}"
                + (f"  scale={self.scale:.4f}" if abs(self.scale - 1.0) > 1e-9 else ""))


def evaluate_ate(est: Trajectory, gt: Trajectory, *, align: bool = True,
                 with_scale: bool = False, max_difference: float = 0.02) -> AteResult:
    """Absolute Trajectory Error. 전역 일관성 지표."""
    e, g = associate(est, gt, max_difference)
    if len(e) < 3:
        raise ValueError(f"연관된 포즈가 부족: {len(e)}")

    P, Q = e.positions, g.positions
    scale = 1.0
    if align:
        R, t, scale = umeyama(P, Q, with_scale)
        P = scale * (P @ R.T) + t

    errors = np.linalg.norm(P - Q, axis=1)
    return AteResult(
        rmse=float(np.sqrt((errors ** 2).mean())),
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        std=float(errors.std()),
        min=float(errors.min()),
        max=float(errors.max()),
        count=len(errors),
        scale=float(scale),
        errors=errors,
    )


@dataclass
class RpeResult:
    trans_rmse: float
    trans_mean: float
    rot_rmse_deg: float
    rot_mean_deg: float
    count: int
    delta: float

    def summary(self) -> str:
        return (f"RPE(dt={self.delta}s)  trans rmse {self.trans_rmse:.4f} m  "
                f"rot rmse {self.rot_rmse_deg:.4f} deg  n={self.count}")


def evaluate_rpe(est: Trajectory, gt: Trajectory, *, delta: float = 1.0,
                 max_difference: float = 0.02) -> RpeResult:
    """Relative Pose Error. 국소 드리프트 지표 - 정렬에 무관하다."""
    e, g = associate(est, gt, max_difference)
    if len(e) < 2:
        raise ValueError("연관된 포즈가 부족")

    trans_err, rot_err = [], []
    j = 0
    for i in range(len(e)):
        target = e.stamps[i] + delta
        while j < len(e) and e.stamps[j] < target:
            j += 1
        if j >= len(e):
            break
        # 같은 구간에 대한 추정/참값 상대변환을 비교
        d_est = e.poses[i].inverse() @ e.poses[j]
        d_gt = g.poses[i].inverse() @ g.poses[j]
        err = d_gt.inverse() @ d_est
        trans_err.append(float(np.linalg.norm(err.t)))
        rot_err.append(float(np.linalg.norm(so3_log(err.R))))

    if not trans_err:
        raise ValueError(f"delta={delta}s 구간을 만족하는 쌍이 없음")

    te = np.array(trans_err)
    re = np.degrees(np.array(rot_err))
    return RpeResult(
        trans_rmse=float(np.sqrt((te ** 2).mean())),
        trans_mean=float(te.mean()),
        rot_rmse_deg=float(np.sqrt((re ** 2).mean())),
        rot_mean_deg=float(re.mean()),
        count=len(te),
        delta=delta,
    )
