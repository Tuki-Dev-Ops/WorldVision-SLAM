"""의미 기반 객체 탐색 - "컵을 찾아라".

일반 탐색과 다른 점은 **어디를 볼지 세계 모델이 정한다**는 것이다.
사람은 컵을 찾을 때 바닥을 훑지 않는다. 테이블 위를 본다.

후보 출처가 넷이고, 각각 다른 층에서 온다.
  1. 지금 믿고 있는 그 클래스 객체      <- WorldState
  2. 예전에 봤던 자리 (지금은 안 보임)   <- MemoryEngine
  3. 그 클래스를 받칠 수 있는 표면 위    <- WorldGraph 어포던스
  4. 아직 못 본 공간                     <- Coverage

이게 신념층 전체를 한 번에 쓰는 유일한 컴포넌트다. 어느 한 층이 비어 있으면
탐색 품질이 그만큼 떨어지고, 그것이 테스트로 드러난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..world.graph import WorldGraph, can_support, is_placeable
from ..world.memory import MemoryEngine
from ..world.state import WorldSnapshot
from .risk import Coverage, RiskEstimator


@dataclass
class SearchCandidate:
    """가 볼 만한 곳 하나."""

    position: np.ndarray
    probability: float                    # 목표가 거기 있을 확률
    cost: float                           # 가는 비용 (m)
    source: str                           # believed / remembered / support / unobserved
    reason: str = ""
    risk: float = 0.0
    token_id: int | None = None

    @property
    def utility(self) -> float:
        """확률 / 비용. 위험은 비용에 실린다.

        기대 정보이득을 비용으로 나눈 고전적 형태다. 위험을 별도 항으로 빼면
        '위험하지만 확실한 곳' 과 '안전하지만 가망 없는 곳' 의 비교가 임의로 된다.
        """
        return self.probability / max(self.cost * (1.0 + 2.0 * self.risk), 1e-3)


@dataclass
class SearchConfig:
    max_candidates: int = 12

    # 출처별 사전확률. 지금 보이는 것이 가장 확실하고, 못 본 곳이 가장 막연하다.
    believed_prior: float = 0.95
    remembered_prior: float = 0.55
    support_prior: float = 0.25
    unobserved_prior: float = 0.10

    # 기억한 위치는 시간이 지날수록 덜 믿는다
    memory_half_life: float = 300.0       # s

    # 받침면 위 탐색 지점을 표면에서 얼마나 띄울지
    support_clearance: float = 0.06

    # 미관측 후보를 어디까지 뿌릴지
    unobserved_radius: float = 5.0
    unobserved_samples: int = 24
    unobserved_threshold: float = 0.3     # 커버리지가 이 아래면 미관측으로 본다


class ObjectSearch:
    """목표 클래스를 찾기 위해 어디를 볼지 순위를 매긴다."""

    def __init__(self, config: SearchConfig | None = None,
                 risk: RiskEstimator | None = None):
        self.cfg = config or SearchConfig()
        self.risk = risk or RiskEstimator()

    def plan(self, target_class: str, robot_position: np.ndarray,
             snapshot: WorldSnapshot,
             memory: MemoryEngine | None = None,
             graph: WorldGraph | None = None,
             coverage: Coverage | None = None,
             now: float | None = None) -> list[SearchCandidate]:
        cfg = self.cfg
        origin = np.asarray(robot_position, float)
        now = snapshot.stamp if now is None else now

        candidates: list[SearchCandidate] = []
        seen_positions: list[np.ndarray] = []

        def add(position, probability, source, reason, token_id=None):
            p = np.asarray(position, float)
            # 거의 같은 자리를 여러 번 제안하지 않는다
            for q in seen_positions:
                if float(np.linalg.norm(p - q)) < 0.25:
                    return
            seen_positions.append(p)
            cost = float(np.linalg.norm(p - origin)) + 0.1
            risk = self.risk.at(p, snapshot, coverage).total
            candidates.append(SearchCandidate(p, probability, cost, source,
                                              reason, risk, token_id))

        # 1) 지금 믿고 있는 객체
        for token in snapshot:
            if token.class_name != target_class:
                continue
            add(token.position, cfg.believed_prior * token.existence,
                "believed", "현재 관측 중", token.token_id)

        # 2) 기억 속의 자리. 오래될수록 덜 믿는다.
        if memory is not None:
            for mem in memory.objects.values():
                if snapshot.get(mem.object_id) is not None:
                    continue          # 지금 보이면 위에서 이미 넣었다
                if mem.class_name != target_class:
                    continue
                age = max(0.0, now - mem.last_seen)
                decay = 0.5 ** (age / max(cfg.memory_half_life, 1e-6))
                add(mem.persistent_position,
                    cfg.remembered_prior * mem.persistent_existence * decay,
                    "remembered", f"{age:.0f}s 전 관측", mem.object_id)

        # 3) 그 클래스를 받칠 수 있는 표면 위.
        #    사람이 컵을 찾을 때 바닥을 훑지 않는 이유다.
        if is_placeable(target_class):
            for token in snapshot:
                if not can_support(token.class_name):
                    continue
                top = np.asarray(token.position, float).copy()
                top[2] += float(token.extent[2]) + cfg.support_clearance
                # 이미 그 위에 무언가 있다고 알면 확률을 낮춘다
                occupied = 0.0
                if graph is not None:
                    occupied = max((r.confidence for r in graph.objects_on(token.token_id)),
                                   default=0.0)
                add(top, cfg.support_prior * token.existence * (1.0 - 0.5 * occupied),
                    "support", f"{token.class_name} 위", token.token_id)

        # 4) 아직 못 본 공간
        if coverage is not None and len(coverage) > 0:
            for p in self._unobserved_samples(origin, coverage):
                add(p, cfg.unobserved_prior, "unobserved", "미관측 영역")

        candidates.sort(key=lambda c: -c.utility)
        return candidates[: cfg.max_candidates]

    # --- 내부 -------------------------------------------------------------

    def _unobserved_samples(self, origin: np.ndarray,
                            coverage: Coverage) -> list[np.ndarray]:
        cfg = self.cfg
        out = []
        for i in range(cfg.unobserved_samples):
            angle = 2.0 * np.pi * i / cfg.unobserved_samples
            for radius in (cfg.unobserved_radius * 0.5, cfg.unobserved_radius):
                p = origin + np.array([radius * np.cos(angle),
                                       radius * np.sin(angle), 0.0])
                if coverage.observed(p) < cfg.unobserved_threshold:
                    out.append(p)
        return out

    # --- 요약 -------------------------------------------------------------

    @staticmethod
    def describe(candidates: list[SearchCandidate], limit: int = 5) -> list[str]:
        return [f"{i + 1}. [{c.source}] {np.round(c.position, 2).tolist()} "
                f"p={c.probability:.2f} cost={c.cost:.1f} risk={c.risk:.2f} "
                f"u={c.utility:.3f}  {c.reason}"
                for i, c in enumerate(candidates[:limit])]
