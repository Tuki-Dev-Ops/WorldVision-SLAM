"""World State - 엔진의 유일한 진실 원천.

docs/00-manifesto.md 의 핵심 주장을 실제 자료구조로 만든 것이다. 지도가 아니라
*지속적으로 개정되는 믿음*이며, 버전이 붙은 불변 스냅샷의 열로 표현된다.

버전 스냅샷이 필요한 이유는 편의가 아니다.
  - "10초 전 세계는 어땠는가" 가 재생이 아니라 질의가 된다
  - 변화 검출이 두 스냅샷의 차이로 정의된다 (변화는 두 관측 *사이* 에만 존재한다)
  - 시각화의 되감기와 Memory Engine 의 이력이 같은 메커니즘이 된다

불변성은 규약이 아니라 강제다. 스냅샷을 넘겨받은 쪽이 그것을 고칠 수 있으면
"그때 무엇을 믿었는가" 를 나중에 물을 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TokenBelief:
    """한 객체에 대한 현재 믿음. 불변이다."""

    token_id: int
    class_id: int
    class_name: str = ""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extent: np.ndarray = field(default_factory=lambda: np.full(3, 0.2))

    # 세 믿음을 분리해 두는 이유는 이들이 따로 실패하기 때문이다.
    # "저기 의자가 있다"(existence) / "아까 그 의자다"(identity) /
    # "안 움직인다"(static) 는 서로 독립적으로 틀릴 수 있다.
    existence: float = 0.5
    identity: float = 0.5
    static_belief: float = 0.5

    lifecycle: str = "provisional"          # provisional/active/occluded/dormant/displaced
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(np.trace(self.covariance) / 3.0, 0.0)))

    @property
    def is_stable(self) -> bool:
        """장소를 정의할 자격. 움직이는 물체는 랜드마크가 될 수 없다."""
        return (self.static_belief > 0.7 and self.existence > 0.6
                and self.observation_count >= 3 and self.sigma < 0.5)

    @property
    def is_dynamic(self) -> bool:
        return self.static_belief < 0.4

    def with_(self, **changes) -> "TokenBelief":
        """부분 갱신본을 새로 만든다. 원본은 변하지 않는다."""
        return replace(self, **changes)


@dataclass(frozen=True)
class WorldSnapshot:
    """한 시각의 세계 전체. 불변."""

    version: int
    stamp: float
    tokens: dict[int, TokenBelief] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[TokenBelief]:
        # ID 순서로 순회한다. 해시 순서가 결과에 새면 재현이 불가능해진다.
        return iter(self.tokens[k] for k in sorted(self.tokens))

    def get(self, token_id: int) -> TokenBelief | None:
        return self.tokens.get(token_id)

    def stable(self) -> list[TokenBelief]:
        return [t for t in self if t.is_stable]

    def dynamic(self) -> list[TokenBelief]:
        return [t for t in self if t.is_dynamic]

    def by_class(self, class_id: int) -> list[TokenBelief]:
        return [t for t in self if t.class_id == class_id]


class WorldState:
    """개정 가능한 믿음과 그 이력.

    쓰기는 단일 스레드를 가정한다. 읽기는 스냅샷을 통해서만 하며, 스냅샷은
    불변이므로 쓰기와 경합하지 않는다 - C++ 엔진의 copy-on-write 발행과
    같은 계약이다.
    """

    def __init__(self, history_capacity: int = 512):
        self._tokens: dict[int, TokenBelief] = {}
        self._version = 0
        self._stamp = 0.0
        self._history: list[WorldSnapshot] = []
        self._capacity = max(1, history_capacity)

    # --- 쓰기 -------------------------------------------------------------

    def put(self, belief: TokenBelief) -> None:
        self._tokens[belief.token_id] = belief

    def update(self, token_id: int, **changes) -> TokenBelief:
        current = self._tokens.get(token_id)
        if current is None:
            raise KeyError(f"미등록 토큰: {token_id}")
        updated = current.with_(**changes)
        self._tokens[token_id] = updated
        return updated

    def remove(self, token_id: int) -> None:
        """토큰을 세계에서 제거한다.

        주의: 이것은 '없어졌다'가 아니라 '더는 추적하지 않는다'이다.
        부재를 주장하려면 lifecycle 을 displaced 로 두고 남겨야 한다 -
        그래야 변화 검출이 그것을 볼 수 있다.
        """
        self._tokens.pop(token_id, None)

    def commit(self, stamp: float) -> WorldSnapshot:
        """현재 믿음을 불변 스냅샷으로 발행한다."""
        if stamp < self._stamp:
            raise ValueError(f"시각 역행: {stamp} < {self._stamp}")
        self._version += 1
        self._stamp = stamp
        snap = WorldSnapshot(self._version, stamp, dict(self._tokens))
        self._history.append(snap)
        while len(self._history) > self._capacity:
            self._history.pop(0)
        return snap

    # --- 읽기 -------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    @property
    def stamp(self) -> float:
        return self._stamp

    def current(self) -> WorldSnapshot:
        return WorldSnapshot(self._version, self._stamp, dict(self._tokens))

    def at(self, stamp: float) -> WorldSnapshot | None:
        """그 시각 이하의 가장 최근 스냅샷.

        "10초 전 세계는 어땠는가" 가 재생이 아니라 질의가 되는 지점이다.
        """
        found = None
        for snap in self._history:
            if snap.stamp <= stamp:
                found = snap
            else:
                break
        return found

    def snapshot_at_version(self, version: int) -> WorldSnapshot | None:
        for snap in self._history:
            if snap.version == version:
                return snap
        return None

    @property
    def history(self) -> list[WorldSnapshot]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return (f"WorldState(v{self._version} @ {self._stamp:.2f}s, "
                f"{len(self._tokens)} tokens, {len(self._history)} snapshots)")
