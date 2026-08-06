"""변화 검출 - "무엇이 바뀌었는가".

WME 가 SLAM 과 갈리는 지점이다. 지도는 관측과 어긋나면 덮어쓴다. 월드 모델은
어긋남의 원인을 구분해야 한다.

    내가 잘못 추정했다      -> 불확실성 안의 차이. 변화가 아니다.
    물체가 움직였다          -> moved
    물체가 치워졌다          -> removed
    없던 물체가 생겼다       -> added

이 넷은 서로 다른 행동을 요구하므로 뭉뚱그리면 안 된다.

핵심 설계: 판정 문턱을 절대 거리가 아니라 **믿음의 공분산**으로 잡는다.
그래야 센서가 나빠졌을 때 문턱이 함께 넓어지고, 안개가 꼈다고 세계가 변했다고
보고하지 않는다. 그 성질이 이 모듈의 유일한 비자명한 요구사항이며,
eval.metrics.change_metrics 의 false_change_rate 로 측정된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.metrics import ChangeEvent
from .state import TokenBelief, WorldSnapshot


@dataclass
class ChangeConfig:
    # 이동 판정: 마할라노비스 문턱 (자유도 3). 절대 거리가 아니라는 점이 중요하다.
    move_chi2: float = 16.27              # 99.9%
    min_move_metres: float = 0.15         # 공분산이 아주 작을 때의 하한

    # 존재/부재 판정
    removed_existence: float = 0.25       # 이 아래로 떨어지면 사라진 것으로 본다
    added_existence: float = 0.65         # 이 위여야 새로 생겼다고 주장한다

    # 두 스냅샷 모두에서 안정 랜드마크여야 이동을 논한다.
    # 움직이는 물체가 움직인 것은 '장면 변화'가 아니다.
    require_stable: bool = True
    min_observations: int = 3

    # 관측이 부족한 토큰은 판정하지 않는다. 안 본 것과 없어진 것은 다르다.
    min_reference_observations: int = 3


def _mahalanobis(diff: np.ndarray, cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    S = np.asarray(cov_a, float) + np.asarray(cov_b, float) + np.eye(3) * 1e-9
    try:
        return float(diff @ np.linalg.solve(S, diff))
    except np.linalg.LinAlgError:
        return float("inf")


class ChangeDetector:
    """두 스냅샷을 비교해 종류가 붙은 변화 사건을 낸다."""

    def __init__(self, config: ChangeConfig | None = None):
        self.cfg = config or ChangeConfig()

    # --- 개별 판정 ---------------------------------------------------------

    def _moved(self, ref: TokenBelief, cur: TokenBelief) -> bool:
        cfg = self.cfg
        if cfg.require_stable and not (ref.is_stable and cur.is_stable):
            return False

        diff = np.asarray(cur.position, float) - np.asarray(ref.position, float)
        distance = float(np.linalg.norm(diff))

        # 불확실성이 아주 작을 때 잡음 수준의 차이를 변화로 보고하지 않도록
        # 절대 하한을 둔다. 상한이 아니라 하한이라는 점이 중요하다.
        if distance < cfg.min_move_metres:
            return False

        return _mahalanobis(diff, ref.covariance, cur.covariance) > cfg.move_chi2

    def _removed(self, ref: TokenBelief, cur: TokenBelief | None) -> bool:
        cfg = self.cfg
        if ref.observation_count < cfg.min_reference_observations:
            return False        # 애초에 잘 몰랐던 것은 사라졌다고 말할 수 없다
        if cfg.require_stable and not ref.is_stable:
            return False

        if cur is None:
            # 스냅샷에서 아예 빠졌다. 추적을 포기한 것과 부재를 구분할 수 없으므로
            # 보수적으로 변화로 보지 않는다 - lifecycle 로 명시된 경우만 센다.
            return False
        if cur.lifecycle == "displaced":
            return True
        return cur.existence < cfg.removed_existence

    def _added(self, cur: TokenBelief, ref: WorldSnapshot) -> bool:
        cfg = self.cfg
        if cur.token_id in ref.tokens:
            return False
        if cur.observation_count < cfg.min_observations:
            return False
        if cfg.require_stable and not cur.is_stable:
            return False
        return cur.existence >= cfg.added_existence

    # --- 비교 -------------------------------------------------------------

    def compare(self, reference: WorldSnapshot, current: WorldSnapshot
                ) -> list[ChangeEvent]:
        """reference -> current 사이의 변화. 시각은 current 의 것을 쓴다."""
        events: list[ChangeEvent] = []
        t = current.stamp

        for ref_token in reference:
            cur_token = current.get(ref_token.token_id)

            if self._removed(ref_token, cur_token):
                events.append(ChangeEvent(ref_token.token_id, "removed", t))
                continue
            if cur_token is not None and self._moved(ref_token, cur_token):
                events.append(ChangeEvent(ref_token.token_id, "moved", t))

        for cur_token in current:
            if self._added(cur_token, reference):
                events.append(ChangeEvent(cur_token.token_id, "added", t))

        events.sort(key=lambda e: (e.kind, e.object_id))
        return events

    def describe(self, reference: WorldSnapshot, current: WorldSnapshot) -> list[str]:
        """사람이 읽을 요약. 계획기에 넘기기 전 확인용."""
        out = []
        for e in self.compare(reference, current):
            token = current.get(e.object_id) or reference.get(e.object_id)
            name = token.class_name or str(token.class_id) if token else "?"
            out.append(f"[{e.kind}] {name}#{e.object_id} @ t={e.time:.2f}")
        return out
