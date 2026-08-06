"""팩터그래프 - 세 tier 를 하나의 사후분포로 융합하는 층.

docs/04-unified-objective.md 5.1 의 형식화가 실제로 구현된 곳이다.
환경 적응은 별도 항이 아니라 Lambda_k(E) 의 스케일로만 들어온다.
"""

from .factors import (
    BetweenPoseFactor,
    Factor,
    Huber,
    ObjectObservationFactor,
    PointPriorFactor,
    PosePriorFactor,
    RobustKernel,
    diagonal,
    isotropic,
)
from .graph import FactorGraph, OptimizationResult, SolverOptions
from .variables import PoseVariable, PositiveVectorVariable, Variable, VectorVariable

__all__ = [
    "Variable", "PoseVariable", "VectorVariable", "PositiveVectorVariable",
    "Factor", "PosePriorFactor", "PointPriorFactor", "BetweenPoseFactor",
    "ObjectObservationFactor", "RobustKernel", "Huber", "isotropic", "diagonal",
    "FactorGraph", "SolverOptions", "OptimizationResult",
]
