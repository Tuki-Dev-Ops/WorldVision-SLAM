"""합성 시뮬레이션 하네스.

실제 데이터셋이 줄 수 없는 것을 만든다: 조건 스윕, 가려짐 관통 정체성 참값,
장면 변화 참값, 미래 궤적 참값. docs/05-research-program.md 3장.
"""

from .scenarios import (
    ConditionFn,
    Sequence,
    add_walkers,
    apply_changes,
    burst_condition,
    constant_condition,
    ramp_condition,
    run,
    scenario_condition_sweep,
    scenario_degradation_burst,
    scenario_dynamic,
    scenario_revisit_with_changes,
    scenario_static,
    static_room,
)
from .sensor import FrameTruth, SensorConfig, SimSensor, severity_of
from .world import CameraModel, CameraTrajectory, SimObject, SimWorld

# CameraTrajectory.spiral 은 초반 모호 / 후반 해소 시나리오에 쓴다 (M5 참조).

__all__ = [
    "SimObject", "SimWorld", "CameraTrajectory", "CameraModel",
    "SimSensor", "SensorConfig", "FrameTruth", "severity_of",
    "Sequence", "ConditionFn", "run",
    "constant_condition", "ramp_condition", "burst_condition",
    "static_room", "add_walkers", "apply_changes",
    "scenario_static", "scenario_dynamic", "scenario_revisit_with_changes",
    "scenario_condition_sweep", "scenario_degradation_burst",
]
