"""평가 하네스.

ATE/RPE 는 Sturm et al. (IROS 2012) 정의를 따른다. 나머지 지표(보정, 열화,
정체성, 변화, 예측)는 ATE 가 측정하지 못하는 월드 모델 주장에 대응한다.
"""

from .metrics import (
    CalibrationResult,
    ChangeEvent,
    ChangeResult,
    DegradationCurve,
    IdentityResult,
    NeesResult,
    PredictionResult,
    WorldModelReport,
    calibration,
    change_metrics,
    degradation_curve,
    identity_metrics,
    nees,
    prediction_metrics,
    recovery_latency,
)
from .stats import chi2_cdf, chi2_interval, chi2_ppf
from .trajectory import (
    AteResult,
    RpeResult,
    Trajectory,
    associate,
    evaluate_ate,
    evaluate_rpe,
    umeyama,
)
from .tum import FrameEntry, TumSequence, load_sequence, load_trajectory, save_trajectory

__all__ = [
    "Trajectory", "associate", "umeyama",
    "evaluate_ate", "evaluate_rpe", "AteResult", "RpeResult",
    "load_trajectory", "save_trajectory", "load_sequence", "TumSequence", "FrameEntry",
    "nees", "NeesResult", "calibration", "CalibrationResult",
    "degradation_curve", "DegradationCurve", "recovery_latency",
    "identity_metrics", "IdentityResult",
    "change_metrics", "ChangeResult", "ChangeEvent",
    "prediction_metrics", "PredictionResult",
    "WorldModelReport",
    "chi2_cdf", "chi2_ppf", "chi2_interval",
]
