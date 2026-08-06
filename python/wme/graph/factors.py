"""팩터: 하나의 우도 항.

docs/04-unified-objective.md 의 주장이 코드로 존재하는 곳이다. 각 팩터는
잔차와 **정보행렬**을 낸다. 가중치가 아니라 정보행렬이다 - 모든 항이 보정된
로그우도이면 자유 가중치는 없고 단위가 상쇄된다.

    -log p(Z|X) = sum_k  0.5 * r_k^T Lambda_k r_k

환경 적응은 별도 항이 아니라 Lambda_k(E) 의 스케일로만 들어온다.
`if (night)` 분기는 어디에도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reference.geometry import SE3, so3_log
from ..sim.world import CameraModel
from .variables import Variable


# --- 로버스트 커널 ---------------------------------------------------------

class RobustKernel:
    """IRLS 가중치를 주는 커널. w(rho) 를 잔차 제곱에 곱한다."""

    def weight(self, squared_norm: float) -> float:
        return 1.0


@dataclass
class Huber(RobustKernel):
    """|r| > delta 에서 선형으로 전환. 이상치의 영향을 제한한다.

    엔진 전반의 원칙대로, 이것을 '손실 재구성'이 아니라 두꺼운 꼬리를 가진
    명시적 우도로 본다. 그래야 유효 잡음 모델이 해석 가능한 채로 남는다.
    """
    delta: float = 3.0

    def weight(self, squared_norm: float) -> float:
        n = np.sqrt(max(squared_norm, 0.0))
        return 1.0 if n <= self.delta else self.delta / n


# --- 팩터 기반 ------------------------------------------------------------

class Factor:
    """잔차 + 정보행렬. 자코비안은 기본적으로 수치미분한다.

    수치미분을 기본으로 두는 이유: 해석적 자코비안 오류는 조용히 틀린 답을
    만들고 SLAM 에서 가장 잡기 어려운 버그다. 성능이 필요한 팩터만
    jacobians() 를 재정의한다.
    """

    keys: list[str] = []
    kernel: RobustKernel | None = None

    def residual(self, values: dict[str, Variable]) -> np.ndarray:
        raise NotImplementedError

    def information(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return self.information().shape[0]

    def jacobians(self, values: dict[str, Variable],
                  eps: float = 1e-6) -> list[np.ndarray] | None:
        """각 키에 대한 (dim x var.dim) 자코비안. None 이면 수치미분."""
        return None

    def numeric_jacobians(self, values: dict[str, Variable],
                          eps: float = 1e-6) -> list[np.ndarray]:
        base = self.residual(values)
        out = []
        for key in self.keys:
            var = values[key]
            J = np.zeros((base.size, var.dim))
            for k in range(var.dim):
                step = np.zeros(var.dim)
                step[k] = eps
                shifted = dict(values)
                shifted[key] = var.retract(step)
                plus = self.residual(shifted)
                shifted[key] = var.retract(-step)
                minus = self.residual(shifted)
                J[:, k] = (plus - minus) / (2.0 * eps)
            out.append(J)
        return out


# --- 구체 팩터 ------------------------------------------------------------

class PosePriorFactor(Factor):
    """포즈에 대한 사전분포.

    포즈 그래프는 상대 측정만으로는 6-DoF 게이지 자유도가 남아 H 가 특이해진다.
    최소 하나의 포즈를 고정해야 풀린다. 평가 시에는 참값에 고정해야
    추정 좌표계와 참값 좌표계가 일치한다.
    """

    def __init__(self, key: str, prior: SE3, information: np.ndarray):
        self.keys = [key]
        self.prior = prior
        self._info = np.asarray(information, float)

    def residual(self, values: dict[str, Variable]) -> np.ndarray:
        return (self.prior.inverse() @ values[self.keys[0]].pose).log()

    def information(self) -> np.ndarray:
        return self._info


class PointPriorFactor(Factor):
    def __init__(self, key: str, prior: np.ndarray, information: np.ndarray):
        self.keys = [key]
        self.prior = np.asarray(prior, float).ravel()
        self._info = np.asarray(information, float)

    def residual(self, values: dict[str, Variable]) -> np.ndarray:
        return values[self.keys[0]].value - self.prior

    def information(self) -> np.ndarray:
        return self._info


class BetweenPoseFactor(Factor):
    """상대 포즈 측정. ECDA(Tier 0) 와 TCG(Tier 1) 가 모두 이 형태로 들어온다.

    두 tier 의 차이는 팩터 종류가 아니라 정보행렬의 크기다. 안개가 끼면
    ECDA 팩터의 Lambda 가 줄고 TCG 팩터의 Lambda 가 상대적으로 커진다.
    이것이 docs/02 5장 융합 규칙의 실제 구현이다.
    """

    def __init__(self, key_a: str, key_b: str, measured: SE3,
                 information: np.ndarray, kernel: RobustKernel | None = None,
                 tier: str = "unknown"):
        self.keys = [key_a, key_b]
        self.measured = measured
        self._info = np.asarray(information, float)
        self.kernel = kernel
        self.tier = tier

    def residual(self, values: dict[str, Variable]) -> np.ndarray:
        Ta = values[self.keys[0]].pose
        Tb = values[self.keys[1]].pose
        predicted = Ta.inverse() @ Tb
        return (self.measured.inverse() @ predicted).log()

    def information(self) -> np.ndarray:
        return self._info

    def scaled(self, alpha: float) -> "BetweenPoseFactor":
        """환경 가중 alpha_k(E) 적용. 항을 더하지 않고 정보량만 조절한다."""
        return BetweenPoseFactor(self.keys[0], self.keys[1], self.measured,
                                 self._info * max(alpha, 1e-9), self.kernel, self.tier)


# 8개 꼭짓점 부호 조합 (객체 박스)
_SIGNS = np.array([[sx, sy, sz]
                   for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


class ObjectObservationFactor(Factor):
    """검출 박스 + 깊이 관측. 포즈 / 위치 / 치수를 함께 구속한다.

    M3 에서 확인된 대로, 박스 중심을 무게중심의 투영으로 두면 안 된다.
    박스 전체를 3D 상자의 투영으로 모델링하고 치수를 상태에 포함해야
    편향이 사라지고 공분산이 보정된다. 관측은 (x, y, w, h, depth) 이며,
    시뮬레이터가 네 값에 독립 잡음을 주므로 정보행렬이 대각이 된다.
    """

    def __init__(self, pose_key: str, point_key: str, extent_key: str,
                 measurement: np.ndarray, camera: CameraModel,
                 information: np.ndarray, kernel: RobustKernel | None = None,
                 depth_offset_coeff: float = 0.0):
        self.keys = [pose_key, point_key, extent_key]
        self.z = np.asarray(measurement, float).ravel()
        self.cam = camera
        self._info = np.asarray(information, float)
        self.kernel = kernel
        # 관측된 깊이가 무게중심 깊이가 아닐 때의 보정 계수.
        # 깊이 센서(또는 렌더 마스크)는 *보이는 표면* 을 재므로 무게중심보다
        # 가깝다. 이 항이 없으면 랜드마크가 그 편향만큼 통째로 밀린다.
        #
        # 단위 주의: extent 는 반치수다(위 corners = p_cam +- extent).
        # 계수는 "어느 영역의 깊이를 어떤 통계로 재는가" 에 딸린 값이지 보편 상수가
        # 아니다. 여기 0.778 은 마스크 *전체* 의 깊이 중앙값에 대한 실측치로,
        # 경사진 측면이 섞여 근접면보다 깊게 나오므로 1.0 보다 작다.
        # 박스 중앙 일부만 쓰는 구현(C++ TokenStore, depth_sample_shrink=0.5)은
        # 근접면에 가까우므로 계수가 1.0 이다. 그대로 옮겨 쓰면 안 된다 -
        # 실제로 0.778 을 C++ 에 복사해 4 m 물체에서 0.23 m 편향이 났다.
        self.depth_offset_coeff = depth_offset_coeff

    def _predict(self, T_world_cam: SE3, point: np.ndarray,
                 extent: np.ndarray) -> np.ndarray | None:
        p_cam = T_world_cam.inverse() @ point
        corners = p_cam + _SIGNS * extent
        if np.any(corners[:, 2] <= 1e-3):
            return None

        u = self.cam.fx * corners[:, 0] / corners[:, 2] + self.cam.cx
        v = self.cam.fy * corners[:, 1] / corners[:, 2] + self.cam.cy
        u_lo, u_hi = float(u.min()), float(u.max())
        v_lo, v_hi = float(v.min()), float(v.max())

        depth = float(p_cam[2]) - self.depth_offset_coeff * float(extent[2])
        return np.array([u_lo, v_lo, u_hi - u_lo, v_hi - v_lo, depth])

    def residual(self, values: dict[str, Variable]) -> np.ndarray:
        pred = self._predict(values[self.keys[0]].pose,
                             values[self.keys[1]].value,
                             values[self.keys[2]].value)
        if pred is None:
            # 카메라 뒤로 넘어간 상태. 큰 잔차로 되돌린다.
            return np.full(5, 1e3)
        return pred - self.z

    def information(self) -> np.ndarray:
        return self._info


def isotropic(dim: int, sigma: float) -> np.ndarray:
    """등방 정보행렬. sigma 는 표준편차."""
    return np.eye(dim) / (sigma * sigma)


def diagonal(sigmas: np.ndarray) -> np.ndarray:
    s = np.asarray(sigmas, float).ravel()
    return np.diag(1.0 / (s * s))
