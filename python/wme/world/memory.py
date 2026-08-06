"""Memory Engine - 세계의 이력과 그 통합.

로그가 아니라 기억이다. 셋이 다르다.
  통합(consolidation)  에피소드들을 하나의 지속적 사실로 굳힌다
  망각(forgetting)     정보 가치가 낮은 것부터 버린다. 나이순이 아니다.
  질의(query)          "어제 여기 있었나" 에 답한다

핵심 설계 결정: **증거 단위는 프레임이 아니라 에피소드다.**

한 번 지나가며 본 100 프레임은 독립 관측 100 개가 아니다. 같은 조명, 같은
시점, 같은 오차가 실린 하나의 사건이다. 프레임 단위로 세면 한 번 스쳐 본
물체가 여러 번 재방문한 물체보다 확신이 높아진다 - 명백히 틀린 결론이다.
이 프로젝트에서 같은 상관 오류가 이미 세 번 나왔다(객체 융합, 측광 잔차,
포즈 체인). 메모리에서는 처음부터 에피소드로 센다.

망각도 나이순이 아니다. 이미 믿고 있던 것을 확인해 준 관측은 정보 가치가
낮고, 예상을 벗어난 관측은 높다. 후자를 먼저 버리면 세계가 변한 기록이
사라진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..reference.confidence import to_logodds, to_probability
from .state import TokenBelief, WorldSnapshot


@dataclass(frozen=True)
class Episode:
    """한 객체를 연속으로 관측한 구간. 증거 하나의 단위."""

    object_id: int
    class_id: int
    class_name: str
    start: float
    end: float
    frames: int

    position: np.ndarray                      # 구간 평균
    spread: float                             # 구간 내 위치 산포 (m)
    existence: float                          # 구간 평균
    static_belief: float
    reliability: float                        # 그때의 센서 신뢰도

    surprise: float = 0.0                     # 통합 믿음 대비 벗어난 정도

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def information_value(self) -> float:
        """망각 우선순위. 낮을수록 먼저 버린다.

        놀라웠던 관측과 신뢰도 높았던 관측을 남긴다. 이미 아는 것을 확인해 준
        관측은 버려도 잃는 게 적다.
        """
        return self.reliability * (0.3 + self.surprise)


@dataclass
class ObjectMemory:
    """한 객체에 대한 기억."""

    object_id: int
    class_id: int
    # 클래스 이름을 여기 들고 있어야 한다. 없으면 "컵을 찾아라" 같은 질의에서
    # 이름을 알아내려고 현재 스냅샷을 뒤져야 하고, 그러면 정작 안 보이는
    # 물체를 기억에서 찾는 경우에 쓸 수 없다.
    class_name: str = ""
    episodes: list[Episode] = field(default_factory=list)

    persistent_existence: float = 0.5
    persistent_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    persistent_covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    persistent_static: float = 0.5

    @property
    def first_seen(self) -> float:
        return self.episodes[0].start if self.episodes else 0.0

    @property
    def last_seen(self) -> float:
        return self.episodes[-1].end if self.episodes else 0.0

    @property
    def visits(self) -> int:
        """에피소드 수. 프레임 수가 아니라 이것이 증거의 양이다."""
        return len(self.episodes)

    def present_at(self, t: float) -> bool:
        """그 시각에 관측 중이었는가. 관측되지 않았다는 것과 없었다는 것은 다르다."""
        return any(e.start <= t <= e.end for e in self.episodes)

    def position_at(self, t: float) -> np.ndarray | None:
        """그 시각의 관측 위치. 없으면 None - 지어내지 않는다."""
        for e in self.episodes:
            if e.start <= t <= e.end:
                return e.position
        return None


@dataclass
class MemoryConfig:
    # 에피소드 분할: 이보다 오래 끊기면 새 에피소드로 본다
    episode_gap: float = 5.0

    # 통합. 에피소드 하나가 주는 로그오즈 증거량.
    # 프레임 수와 무관하다는 것이 이 모듈의 핵심 주장이다.
    episode_evidence: float = 1.2
    logodds_limit: float = 4.0

    # 망각
    base_retention: float = 120.0             # s
    max_episodes_per_object: int = 12
    max_objects: int = 4096

    # 환경 비례 보존: 나쁜 조건에서 본 것은 더 오래 들고 있는다.
    # 관측이 희소했다는 뜻이므로 같은 기준으로 버리면 정보가 과하게 준다.
    adversity_retention_scale: float = 4.0

    # 놀라움 판정 (통합 믿음 대비)
    surprise_sigma: float = 3.0


class MemoryEngine:
    """스냅샷을 받아 에피소드로 쪼개고, 통합하고, 잊는다."""

    def __init__(self, config: MemoryConfig | None = None):
        self.cfg = config or MemoryConfig()
        self._memory: dict[int, ObjectMemory] = {}
        self._open: dict[int, dict] = {}          # 진행 중인 에피소드 누적
        self._environment: list[tuple[float, float]] = []   # (시각, 신뢰도)

    # --- 수집 -------------------------------------------------------------

    def observe(self, snapshot: WorldSnapshot, reliability: float = 1.0) -> None:
        """스냅샷 하나를 기억에 넣는다. 에피소드 경계는 시간 간격으로 정한다."""
        cfg = self.cfg
        t = snapshot.stamp
        self._environment.append((t, float(np.clip(reliability, 0.0, 1.0))))

        seen = set()
        for token in snapshot:
            seen.add(token.token_id)
            open_ep = self._open.get(token.token_id)

            if open_ep is not None and t - open_ep["last"] > cfg.episode_gap:
                self._close(token.token_id)
                open_ep = None

            if open_ep is None:
                self._open[token.token_id] = {
                    "class_id": token.class_id,
                    "class_name": token.class_name,
                    "start": t, "last": t, "frames": 1,
                    "positions": [np.asarray(token.position, float)],
                    "existence": [token.existence],
                    "static": [token.static_belief],
                    "reliability": [reliability],
                }
            else:
                open_ep["last"] = t
                open_ep["frames"] += 1
                open_ep["positions"].append(np.asarray(token.position, float))
                open_ep["existence"].append(token.existence)
                open_ep["static"].append(token.static_belief)
                open_ep["reliability"].append(reliability)

        # 이번 스냅샷에 없던 진행 중 에피소드는 간격이 벌어지면 닫힌다
        for tid in list(self._open):
            if tid not in seen and t - self._open[tid]["last"] > cfg.episode_gap:
                self._close(tid)

    def _close(self, token_id: int) -> None:
        acc = self._open.pop(token_id, None)
        if acc is None:
            return

        positions = np.array(acc["positions"])
        mean = positions.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(positions - mean, axis=1)))

        mem = self._memory.get(token_id)
        surprise = 0.0
        if mem is not None and mem.episodes:
            # 통합 믿음에서 얼마나 벗어났는가. 이것이 망각 우선순위를 정한다.
            sigma = max(0.05, float(np.sqrt(np.trace(mem.persistent_covariance) / 3.0)))
            surprise = float(np.linalg.norm(mean - mem.persistent_position)
                             / (sigma * self.cfg.surprise_sigma))

        episode = Episode(
            object_id=token_id,
            class_id=acc["class_id"],
            class_name=acc["class_name"],
            start=acc["start"], end=acc["last"], frames=acc["frames"],
            position=mean, spread=spread,
            existence=float(np.mean(acc["existence"])),
            static_belief=float(np.mean(acc["static"])),
            reliability=float(np.mean(acc["reliability"])),
            surprise=surprise,
        )

        if mem is None:
            mem = ObjectMemory(token_id, acc["class_id"], acc["class_name"])
            self._memory[token_id] = mem
        mem.episodes.append(episode)
        self._consolidate(mem)

    def flush(self) -> None:
        """진행 중인 에피소드를 모두 닫는다. 시퀀스 끝에서 호출한다."""
        for tid in list(self._open):
            self._close(tid)

    # --- 통합 -------------------------------------------------------------

    def _consolidate(self, mem: ObjectMemory) -> None:
        """에피소드들을 하나의 지속적 사실로 굳힌다.

        증거는 에피소드당 한 번만 센다. 프레임 수가 아니라 방문 횟수가
        확신을 만든다 - 같은 시점에서 오래 본 것은 여러 번 다시 가서 본 것과
        같은 증거가 아니다.
        """
        cfg = self.cfg
        logodds = 0.0
        weights, means, spreads = [], [], []

        for e in mem.episodes:
            # 신뢰도가 낮았던 에피소드는 증거도 약하다
            direction = 1.0 if e.existence > 0.5 else -1.0
            strength = abs(to_logodds(e.existence)) / max(abs(to_logodds(0.95)), 1e-6)
            logodds += direction * cfg.episode_evidence * e.reliability * min(1.0, strength)

            w = e.reliability / max(e.spread ** 2, 1e-4)
            weights.append(w)
            means.append(e.position)
            spreads.append(max(e.spread, 0.02))

        logodds = float(np.clip(logodds, -cfg.logodds_limit, cfg.logodds_limit))
        mem.persistent_existence = to_probability(logodds)

        w = np.array(weights)
        mem.persistent_position = np.average(np.array(means), axis=0, weights=w)

        # 위치 불확실성: 에피소드 간 산포가 에피소드 내 산포보다 크면 그쪽을 쓴다.
        # 한 시점에서 정밀하게 본 것이 여러 시점의 불일치를 지우면 안 된다.
        between = float(np.mean(np.linalg.norm(
            np.array(means) - mem.persistent_position, axis=1))) if len(means) > 1 else 0.0
        within = float(np.average(spreads, weights=w))
        sigma = max(between, within, 0.02)
        mem.persistent_covariance = np.eye(3) * sigma ** 2

        mem.persistent_static = float(np.average([e.static_belief for e in mem.episodes],
                                                 weights=w))

    # --- 망각 -------------------------------------------------------------

    def forget(self, now: float, environment_scale: float = 1.0) -> int:
        """정보 가치가 낮은 것부터 버린다. 나이순이 아니다.

        environment_scale 은 EnvironmentState.memory_retention_scale 을 받는다.
        조건이 나빴다면 관측 자체가 희소했다는 뜻이므로 더 오래 들고 있는다.
        """
        cfg = self.cfg
        retention = cfg.base_retention * max(1.0, environment_scale)
        dropped = 0

        for mem in self._memory.values():
            if len(mem.episodes) <= cfg.max_episodes_per_object:
                continue
            # 나이 페널티와 정보 가치를 함께 본다. 오래됐어도 놀라웠던 것은 남긴다.
            def keep_score(e: Episode) -> float:
                age = max(0.0, now - e.end)
                return e.information_value * math.exp(-age / retention)

            ranked = sorted(mem.episodes, key=keep_score, reverse=True)
            kept = ranked[: cfg.max_episodes_per_object]
            dropped += len(mem.episodes) - len(kept)
            mem.episodes = sorted(kept, key=lambda e: e.start)
            self._consolidate(mem)

        # 객체 수 상한: 마지막 관측이 오래됐고 존재 믿음이 낮은 것부터
        if len(self._memory) > cfg.max_objects:
            ranked = sorted(self._memory.values(),
                            key=lambda m: (m.persistent_existence,
                                           -(now - m.last_seen)))
            for mem in ranked[: len(self._memory) - cfg.max_objects]:
                del self._memory[mem.object_id]
                dropped += 1
        return dropped

    # --- 질의 -------------------------------------------------------------

    def get(self, object_id: int) -> ObjectMemory | None:
        return self._memory.get(object_id)

    def was_present(self, object_id: int, t: float) -> bool | None:
        """그 시각에 관측 중이었는가.

        None 은 '모른다'다. False('없었다')와 구분해야 한다 - 안 본 것과
        없는 것을 섞으면 세계 모델이 아니다.
        """
        mem = self._memory.get(object_id)
        if mem is None:
            return None
        if t < mem.first_seen or t > mem.last_seen:
            return None                       # 기억의 범위 밖
        return mem.present_at(t)

    def reliability_at(self, t: float) -> float | None:
        """그 시각의 센서 신뢰도. '왜 그때 못 봤나' 를 설명하는 데 쓴다."""
        best = None
        for stamp, rel in self._environment:
            if stamp <= t:
                best = rel
            else:
                break
        return best

    def stable_objects(self, min_visits: int = 2,
                       min_existence: float = 0.7) -> list[ObjectMemory]:
        """여러 번 재방문해 확인된 객체. 장소를 정의할 자격이 있는 것들."""
        return [m for m in self._memory.values()
                if m.visits >= min_visits and m.persistent_existence >= min_existence]

    @property
    def objects(self) -> dict[int, ObjectMemory]:
        return dict(self._memory)

    def __len__(self) -> int:
        return len(self._memory)

    def __repr__(self) -> str:
        episodes = sum(m.visits for m in self._memory.values())
        return f"MemoryEngine({len(self._memory)} objects, {episodes} episodes)"
