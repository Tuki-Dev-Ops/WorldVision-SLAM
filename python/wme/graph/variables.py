"""팩터그래프 변수.

변수는 다양체 위에 산다. SE(3) 포즈를 6-벡터처럼 더하면 회전이 SO(3)를 벗어난다.
그래서 모든 변수는 접선공간 갱신(retract)을 스스로 정의한다.

규약은 C++ 엔진과 동일하다: 좌측 섭동 T <- exp(dxi) * T, xi = [rho, phi].
"""

from __future__ import annotations

import numpy as np

from ..reference.geometry import SE3


class Variable:
    """다양체 위의 변수. dim 은 접선공간 차원이다."""

    dim: int = 0

    def retract(self, delta: np.ndarray) -> "Variable":
        raise NotImplementedError

    def copy(self) -> "Variable":
        raise NotImplementedError


class PoseVariable(Variable):
    """SE(3) 포즈. 월드 <- 바디."""

    dim = 6

    def __init__(self, pose: SE3 | None = None):
        self.pose = pose if pose is not None else SE3.identity()

    def retract(self, delta: np.ndarray) -> "PoseVariable":
        # 좌측 섭동. 우측으로 하면 자코비안 규약이 전부 달라진다.
        return PoseVariable(SE3.exp(np.asarray(delta, float)) @ self.pose)

    def copy(self) -> "PoseVariable":
        return PoseVariable(SE3(self.pose.R.copy(), self.pose.t.copy()))

    def __repr__(self) -> str:
        return f"PoseVariable({self.pose})"


class VectorVariable(Variable):
    """유클리드 변수 (랜드마크 위치, 객체 치수, 속도 등)."""

    def __init__(self, value: np.ndarray):
        self.value = np.asarray(value, dtype=float).ravel()
        self.dim = self.value.size

    def retract(self, delta: np.ndarray) -> "VectorVariable":
        return VectorVariable(self.value + np.asarray(delta, float).ravel())

    def copy(self) -> "VectorVariable":
        return VectorVariable(self.value.copy())

    def __repr__(self) -> str:
        return f"VectorVariable({np.round(self.value, 4).tolist()})"


class PositiveVectorVariable(VectorVariable):
    """양수로 유지되어야 하는 변수 (객체 반치수 등).

    로그 파라미터화 대신 갱신 후 자르기를 쓴다. 로그로 두면 자코비안이
    스케일에 따라 극단적으로 변해 LM 감쇠 조정이 어려워진다.
    """

    def __init__(self, value: np.ndarray, lower: float = 0.02, upper: float = 5.0):
        super().__init__(value)
        self.lower = lower
        self.upper = upper

    def retract(self, delta: np.ndarray) -> "PositiveVectorVariable":
        updated = np.clip(self.value + np.asarray(delta, float).ravel(),
                          self.lower, self.upper)
        return PositiveVectorVariable(updated, self.lower, self.upper)

    def copy(self) -> "PositiveVectorVariable":
        return PositiveVectorVariable(self.value.copy(), self.lower, self.upper)
