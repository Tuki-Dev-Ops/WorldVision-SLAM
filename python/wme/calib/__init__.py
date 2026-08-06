"""관측 잡음 보정.

"센서 신뢰도는 목적 함수의 항이 아니라 관측 모델의 정밀도다" 를 검증 가능한
실험으로 만든 모듈. docs/04-unified-objective.md 5.2, M3 게이트.
"""

from .estimator import (
    EstimationResult,
    ObjectEstimate,
    collect_samples,
    estimate_objects,
)
from .experiment import ConditionResult, M3Report, ModelResult, run_m3
from .noise import (
    CalibratedNoise,
    FixedNoise,
    NoiseModel,
    NoiseSample,
    ScheduledNoise,
    fit_calibrated,
    fit_fixed,
    negative_log_likelihood,
)
from .optimize import minimize_nelder_mead

__all__ = [
    "NoiseModel", "FixedNoise", "ScheduledNoise", "CalibratedNoise", "NoiseSample",
    "fit_fixed", "fit_calibrated", "negative_log_likelihood",
    "collect_samples", "estimate_objects", "EstimationResult", "ObjectEstimate",
    "run_m3", "M3Report", "ModelResult", "ConditionResult",
    "minimize_nelder_mead",
]
