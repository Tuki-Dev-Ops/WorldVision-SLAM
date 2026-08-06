"""신념층 - WME 를 SLAM 이 아니라 월드 모델로 만드는 부분.

  state.py       버전이 붙은 불변 스냅샷. "10초 전 세계" 가 재생이 아니라 질의가 된다.
  change.py      "무엇이 바뀌었는가". 오추정 / 이동 / 제거 / 추가를 구분한다.
  prediction.py  "무엇이 바뀔 것인가". 관측과 물리적으로 분리된 타입으로 강제한다.
"""

from .change import ChangeConfig, ChangeDetector
from .graph import (
    Relation,
    RelationConfig,
    RelationInference,
    WorldGraph,
    can_contain,
    can_support,
    is_agent,
    is_placeable,
)
from .memory import Episode, MemoryConfig, MemoryEngine, ObjectMemory
from .prediction import Forecast, PredictionConfig, PredictionEngine
from .state import TokenBelief, WorldSnapshot, WorldState

__all__ = [
    "TokenBelief", "WorldSnapshot", "WorldState",
    "ChangeDetector", "ChangeConfig",
    "PredictionEngine", "PredictionConfig", "Forecast",
    "MemoryEngine", "MemoryConfig", "Episode", "ObjectMemory",
    "WorldGraph", "Relation", "RelationConfig", "RelationInference",
    "can_support", "can_contain", "is_placeable", "is_agent",
]
