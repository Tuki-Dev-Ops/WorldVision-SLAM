"""SO(3)/SE(3) 리군 연산 참조 구현.

C++ include/wme/core/SE3.hpp 와 규약이 동일해야 한다.
  - 접선벡터 xi = [rho(3), phi(3)]
  - 좌측 섭동: T <- exp(dxi) * T

이 모듈의 존재 이유는 편의가 아니라 차등 테스트다. C++ 과 같은 입력에 대해
같은 출력을 내지 않으면 둘 중 하나가 틀린 것이다.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def skew(v: np.ndarray) -> np.ndarray:
    """반대칭 행렬."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """회전벡터 -> 3x3 회전행렬. 소각에서 테일러 전개로 안정화."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS:
        # sin(t)/t -> 1 - t^2/6,  (1-cos t)/t^2 -> 1/2 - t^2/24
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    """회전행렬 -> 회전벡터. |phi| <= pi 를 보장한다."""
    R = np.asarray(R, dtype=float)
    trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(trace))

    if theta < _EPS:
        # 소각에서는 반대칭 성분이 곧 회전벡터
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5

    if abs(np.pi - theta) < 1e-5:
        # theta ~ pi 에서는 반대칭 성분이 0 에 가까워 부호가 불안정하다.
        # (R + I)/2 = n n^T 를 이용해 축을 직접 뽑는다.
        A = (R + np.eye(3)) * 0.5
        n = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(n))
        if n[k] > _EPS:
            n = A[:, k] / n[k]
        n = n / max(np.linalg.norm(n), _EPS)
        # 부호는 반대칭 성분에서 회수
        s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(n @ s) < 0.0:
            n = -n
        return n * theta

    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]])


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * W
    t2 = theta * theta
    return (np.eye(3)
            + ((1.0 - np.cos(theta)) / t2) * W
            + ((theta - np.sin(theta)) / (t2 * theta)) * (W @ W))


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    W = skew(phi)
    if theta < _EPS:
        return np.eye(3) - 0.5 * W
    half = 0.5 * theta
    c = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * W + c * (W @ W)


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """(x, y, z, w) 순서 쿼터니언 -> 회전행렬. 데이터셋 규약이 이 순서다."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """회전행렬 -> (x, y, z, w). 수치적으로 가장 큰 성분을 기준으로 계산한다."""
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


class SE3:
    """4x4 동차변환. C++ wme::SE3 와 1:1 대응."""

    __slots__ = ("R", "t")

    def __init__(self, R: np.ndarray | None = None, t: np.ndarray | None = None):
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=float).reshape(3, 3)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=float).reshape(3)

    @staticmethod
    def identity() -> "SE3":
        return SE3()

    @staticmethod
    def exp(xi: np.ndarray) -> "SE3":
        xi = np.asarray(xi, dtype=float).reshape(6)
        rho, phi = xi[:3], xi[3:]
        return SE3(so3_exp(phi), so3_left_jacobian(phi) @ rho)

    def log(self) -> np.ndarray:
        phi = so3_log(self.R)
        rho = so3_left_jacobian_inv(phi) @ self.t
        return np.concatenate([rho, phi])

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @staticmethod
    def from_matrix(T: np.ndarray) -> "SE3":
        T = np.asarray(T, dtype=float)
        return SE3(T[:3, :3], T[:3, 3])

    def inverse(self) -> "SE3":
        Rt = self.R.T
        return SE3(Rt, -(Rt @ self.t))

    def __matmul__(self, other):
        if isinstance(other, SE3):
            return SE3(self.R @ other.R, self.R @ other.t + self.t)
        p = np.asarray(other, dtype=float)
        if p.ndim == 1:
            return self.R @ p + self.t
        # (N, 3) 점 배열 일괄 변환
        return p @ self.R.T + self.t

    def adjoint(self) -> np.ndarray:
        A = np.zeros((6, 6))
        A[:3, :3] = self.R
        A[:3, 3:] = skew(self.t) @ self.R
        A[3:, 3:] = self.R
        return A

    def distance_to(self, other: "SE3") -> tuple[float, float]:
        """(병진 m, 회전 rad). 수렴 판정과 평가지표에 공용."""
        d = self.inverse() @ other
        return float(np.linalg.norm(d.t)), float(np.linalg.norm(so3_log(d.R)))

    def __repr__(self) -> str:
        rpy = so3_log(self.R)
        return f"SE3(t={np.round(self.t, 4).tolist()}, phi={np.round(rpy, 4).tolist()})"


def kabsch(src: np.ndarray, dst: np.ndarray) -> SE3:
    """대응점 집합으로부터 SE(3) 를 닫힌 형태로 복원한다 (스케일 없음).

    반사행렬(det = -1)은 배제한다. 공선 배치면 ValueError.
    C++ wme::kabsch 와 동일한 규약.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 3)
    dst = np.asarray(dst, dtype=float).reshape(-1, 3)
    if src.shape != dst.shape or len(src) < 3:
        raise ValueError("kabsch 는 3개 이상의 대응이 필요")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        V = Vt.T.copy()
        V[:, 2] *= -1.0
        R = V @ U.T

    if S[1] < 1e-6 * max(1.0, S[0]):
        raise ValueError("대응점이 공선 배치")

    return SE3(R, mu_d - R @ mu_s)
