"""데이터 연관.

docs/04-unified-objective.md 5.4: 이산 연관 변수를 모델 안에 넣는 문제.
지금까지의 모든 실험이 참 연관을 써서 피해 온 지점이다.
"""

from .experiment import AssociationResult, M5Report, compare_strategies, run_tracker
from .murty import murty_k_best
from .trackers import (
    AssociationConfig,
    DeferredTracker,
    GreedyTracker,
    HungarianTracker,
    Measurement,
    MhtTracker,
    TrackState,
    Tracker,
)

__all__ = [
    "murty_k_best",
    "AssociationConfig", "Measurement", "TrackState", "Tracker",
    "GreedyTracker", "HungarianTracker", "DeferredTracker", "MhtTracker",
    "run_tracker", "compare_strategies", "AssociationResult", "M5Report",
]
