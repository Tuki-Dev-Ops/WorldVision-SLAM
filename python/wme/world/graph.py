"""World Graph - 객체 사이의 관계.

"컵이 테이블 위에 있다" 는 참/거짓이 아니다. 컵이 테이블 상판보다 0.12 m 위에
있고 두 위치의 표준편차가 0.1 m 라면, 그 명제는 사후확률로만 말할 수 있다.
기존 씬그래프 연구는 대체로 이걸 단정하고, 그래서 관측이 나빠져도 관계가
그대로 유지된다.

여기서는 세 가지를 지킨다.
  1. 관계는 믿음에서 유도된다. 원시 관측이 아니라 공분산을 가진 믿음이므로,
     센서가 나빠지면 관계 확신도 함께 떨어진다.
  2. 증거는 방문 단위로 센다. 한 번 지나가며 본 100 프레임은 증거 100 개가
     아니다 - 이 프로젝트에서 네 번 물린 상관 오류다.
  3. 관계는 철회 가능하다. 컵을 테이블에서 치우면 'on' 이 사라져야 한다.
     누적만 하는 그래프는 세계가 아니라 기록이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .state import TokenBelief, WorldSnapshot

# --- 어포던스 --------------------------------------------------------------
# 무엇이 무엇을 받칠 수 있고 담을 수 있는지. 학습 모델을 더 붙이는 것은
# YOLO-only 제약 위반이므로 어휘에 대한 명시적 표를 쓴다.

_SUPPORTIVE = {"dining table", "desk", "chair", "couch", "bed", "bench",
               "shelf", "counter", "tv"}
_CONTAINER = {"cup", "bowl", "bottle", "vase", "backpack", "suitcase", "handbag",
              "refrigerator", "oven", "microwave", "sink", "toilet"}
_PLACEABLE = {"cup", "bowl", "bottle", "vase", "book", "laptop", "keyboard",
              "mouse", "remote", "cell phone", "banana", "apple", "orange",
              "potted plant", "clock", "scissors", "teddy bear"}
_AGENT = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "dog", "cat"}


def can_support(name: str) -> bool:
    return name in _SUPPORTIVE


def can_contain(name: str) -> bool:
    return name in _CONTAINER


def is_placeable(name: str) -> bool:
    return name in _PLACEABLE


def is_agent(name: str) -> bool:
    return name in _AGENT


# --- 관계 ------------------------------------------------------------------

@dataclass(frozen=True)
class Relation:
    """하나의 관계 명제. 확신도를 반드시 동반한다."""

    subject: int
    predicate: str                # on / supports / inside / contains / near
    obj: int
    confidence: float
    visits: int = 1               # 이 관계를 뒷받침한 방문 수 (프레임 수 아님)

    def inverse(self) -> "Relation":
        flip = {"on": "supports", "supports": "on",
                "inside": "contains", "contains": "inside", "near": "near"}
        return Relation(self.obj, flip[self.predicate], self.subject,
                        self.confidence, self.visits)

    def __repr__(self) -> str:
        return f"{self.subject} -{self.predicate}({self.confidence:.2f})-> {self.obj}"


@dataclass
class RelationConfig:
    # 'on' 판정: 아래 물체 상판과 위 물체 밑면의 간격이 0 에 가까워야 한다.
    on_gap_tolerance: float = 0.08        # m. 이 안이면 접촉으로 본다
    on_min_overlap: float = 0.5           # 수평 겹침 비율

    # 'near' 판정: 치수를 뺀 표면 간 거리
    near_distance: float = 1.0            # m

    # 'inside': 부피 포함 비율
    inside_min_containment: float = 0.7

    # 확신도 하한. 이보다 낮으면 관계를 주장하지 않는다.
    min_confidence: float = 0.35

    # 불확실성을 무시하지 않도록 하는 하한. 공분산이 0 에 수렴해도
    # 관계 판정이 무한히 날카로워지면 안 된다.
    min_sigma: float = 0.02

    require_stable: bool = True           # 움직이는 물체는 관계의 주어가 되기 어렵다


def _sigma(belief: TokenBelief, axis: int, floor: float) -> float:
    cov = np.asarray(belief.covariance, float)
    return max(float(np.sqrt(max(cov[axis, axis], 0.0))), floor)


def _gaussian_within(value: float, tolerance: float, sigma: float) -> float:
    """|value| < tolerance 일 확률 (정규 근사).

    0.5 * (erf((tol - v)/(s*sqrt2)) + erf((tol + v)/(s*sqrt2)))
    """
    s = max(sigma, 1e-9) * math.sqrt(2.0)
    return float(np.clip(
        0.5 * (math.erf((tolerance - value) / s) + math.erf((tolerance + value) / s)),
        0.0, 1.0))


def _footprint_overlap(a: TokenBelief, b: TokenBelief) -> float:
    """a 의 수평 단면이 b 안에 들어가는 비율."""
    ea, eb = np.asarray(a.extent, float), np.asarray(b.extent, float)
    pa, pb = np.asarray(a.position, float), np.asarray(b.position, float)

    area_a = 1.0
    inter = 1.0
    for k in (0, 1):
        lo = max(pa[k] - ea[k], pb[k] - eb[k])
        hi = min(pa[k] + ea[k], pb[k] + eb[k])
        inter *= max(0.0, hi - lo)
        area_a *= max(2.0 * ea[k], 1e-6)
    return float(np.clip(inter / area_a, 0.0, 1.0))


def _volume_containment(a: TokenBelief, b: TokenBelief) -> float:
    ea, eb = np.asarray(a.extent, float), np.asarray(b.extent, float)
    pa, pb = np.asarray(a.position, float), np.asarray(b.position, float)

    vol_a = 1.0
    inter = 1.0
    for k in range(3):
        lo = max(pa[k] - ea[k], pb[k] - eb[k])
        hi = min(pa[k] + ea[k], pb[k] + eb[k])
        inter *= max(0.0, hi - lo)
        vol_a *= max(2.0 * ea[k], 1e-6)
    return float(np.clip(inter / vol_a, 0.0, 1.0))


def _surface_distance(a: TokenBelief, b: TokenBelief) -> float:
    """축정렬 상자 표면 사이 거리. 중심 거리를 쓰면 큰 물체가 항상 멀어진다."""
    ea, eb = np.asarray(a.extent, float), np.asarray(b.extent, float)
    d = np.abs(np.asarray(a.position, float) - np.asarray(b.position, float)) - (ea + eb)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


class RelationInference:
    """스냅샷에서 관계를 유도한다. 확신도는 위치 불확실성에서 나온다."""

    def __init__(self, config: RelationConfig | None = None):
        self.cfg = config or RelationConfig()

    # --- 개별 술어 ---------------------------------------------------------

    def on(self, upper: TokenBelief, lower: TokenBelief) -> float:
        """upper 가 lower 위에 놓여 있을 확률."""
        cfg = self.cfg
        if not can_support(lower.class_name):
            return 0.0
        if upper.token_id == lower.token_id:
            return 0.0

        eu = np.asarray(upper.extent, float)
        el = np.asarray(lower.extent, float)
        gap = (float(upper.position[2]) - eu[2]) - (float(lower.position[2]) + el[2])

        # 간격의 불확실성은 두 물체 z 불확실성의 합
        sigma = math.hypot(_sigma(upper, 2, cfg.min_sigma), _sigma(lower, 2, cfg.min_sigma))
        p_contact = _gaussian_within(gap, cfg.on_gap_tolerance, sigma)

        overlap = _footprint_overlap(upper, lower)
        if overlap < cfg.on_min_overlap:
            # 완전히 벗어난 것과 살짝 걸친 것을 구분한다
            p_overlap = overlap / max(cfg.on_min_overlap, 1e-6)
        else:
            p_overlap = 1.0

        # 존재를 확신 못 하면 관계도 확신할 수 없다
        return float(p_contact * p_overlap * upper.existence * lower.existence)

    def inside(self, inner: TokenBelief, outer: TokenBelief) -> float:
        cfg = self.cfg
        if not can_contain(outer.class_name):
            return 0.0
        if inner.token_id == outer.token_id:
            return 0.0

        containment = _volume_containment(inner, outer)
        if containment < cfg.inside_min_containment:
            return 0.0

        # 포함 비율이 문턱을 넘더라도 위치 불확실성이 크면 확신할 수 없다
        sigma = math.sqrt(sum(_sigma(inner, k, cfg.min_sigma) ** 2 for k in range(3)))
        margin = float(np.min(np.asarray(outer.extent, float)
                              - np.asarray(inner.extent, float)))
        p_geometry = _gaussian_within(0.0, max(margin, 1e-3), sigma)

        return float(containment * p_geometry * inner.existence * outer.existence)

    def near(self, a: TokenBelief, b: TokenBelief) -> float:
        cfg = self.cfg
        if a.token_id == b.token_id:
            return 0.0
        d = _surface_distance(a, b)
        sigma = math.sqrt(sum(_sigma(a, k, cfg.min_sigma) ** 2
                              + _sigma(b, k, cfg.min_sigma) ** 2 for k in range(3)))
        p = _gaussian_within(d, cfg.near_distance, sigma)
        return float(p * a.existence * b.existence)

    # --- 스냅샷 전체 -------------------------------------------------------

    def infer(self, snapshot: WorldSnapshot) -> list[Relation]:
        cfg = self.cfg
        tokens = [t for t in snapshot
                  if not cfg.require_stable or t.is_stable or is_agent(t.class_name)]

        out: list[Relation] = []
        for a in tokens:
            for b in tokens:
                if a.token_id == b.token_id:
                    continue

                p_on = self.on(a, b)
                if p_on >= cfg.min_confidence:
                    out.append(Relation(a.token_id, "on", b.token_id, p_on))

                p_in = self.inside(a, b)
                if p_in >= cfg.min_confidence:
                    out.append(Relation(a.token_id, "inside", b.token_id, p_in))

                # near 는 대칭이므로 한 방향만 만든다
                if a.token_id < b.token_id:
                    p_near = self.near(a, b)
                    if p_near >= cfg.min_confidence:
                        out.append(Relation(a.token_id, "near", b.token_id, p_near))

        out.sort(key=lambda r: (r.predicate, r.subject, r.obj))
        return out


class WorldGraph:
    """관계 그래프. 방문 단위로 증거를 누적하고, 사라진 관계는 철회한다."""

    def __init__(self, config: RelationConfig | None = None,
                 evidence_gain: float = 0.35, decay: float = 0.5):
        self.cfg = config or RelationConfig()
        self.inference = RelationInference(self.cfg)
        # 증거 이득이 1 보다 작은 이유는 메모리와 같다. 방문 하나가 관계를
        # 확정짓지 않으며, 반증으로 되돌릴 여지를 남겨야 한다.
        self.evidence_gain = evidence_gain
        self.decay = decay
        self._edges: dict[tuple[int, str, int], Relation] = {}
        self._last_visit: int = -1

    def update(self, snapshot: WorldSnapshot, visit: int | None = None) -> None:
        """스냅샷 하나를 반영한다.

        visit 를 주면 그 번호가 바뀔 때만 증거로 센다. 한 번 지나가며 본
        여러 프레임이 관계를 여러 번 확인한 것으로 세지 않기 위해서다.
        """
        new_visit = visit is None or visit != self._last_visit
        if visit is not None:
            self._last_visit = visit
        if not new_visit:
            return

        observed = {(r.subject, r.predicate, r.obj): r
                    for r in self.inference.infer(snapshot)}
        present = set(snapshot.tokens)

        for key, rel in observed.items():
            prior = self._edges.get(key)
            if prior is None:
                self._edges[key] = Relation(*key, rel.confidence * self.evidence_gain
                                            + 0.5 * (1.0 - self.evidence_gain), 1)
            else:
                blended = ((1.0 - self.evidence_gain) * prior.confidence
                           + self.evidence_gain * rel.confidence)
                self._edges[key] = Relation(*key, blended, prior.visits + 1)

        # 두 객체가 모두 보이는데 관계가 관측되지 않았다면 철회 증거다.
        # 안 보인 것은 반증이 아니다 - 그냥 두어야 한다.
        for key, prior in list(self._edges.items()):
            if key in observed:
                continue
            if key[0] not in present or key[2] not in present:
                continue
            faded = prior.confidence * self.decay
            if faded < self.cfg.min_confidence * 0.5:
                del self._edges[key]
            else:
                self._edges[key] = Relation(*key, faded, prior.visits)

    # --- 질의 -------------------------------------------------------------

    def relations(self, min_confidence: float | None = None) -> list[Relation]:
        floor = self.cfg.min_confidence if min_confidence is None else min_confidence
        out = [r for r in self._edges.values() if r.confidence >= floor]
        out.sort(key=lambda r: (r.predicate, r.subject, r.obj))
        return out

    def confidence(self, subject: int, predicate: str, obj: int) -> float:
        rel = self._edges.get((subject, predicate, obj))
        return rel.confidence if rel else 0.0

    def objects_on(self, support_id: int) -> list[Relation]:
        """이 물체가 받치고 있는 것들."""
        return [r for r in self.relations() if r.predicate == "on" and r.obj == support_id]

    def supported_by(self, object_id: int) -> list[Relation]:
        return [r for r in self.relations() if r.predicate == "on" and r.subject == object_id]

    def contents_of(self, container_id: int) -> list[Relation]:
        return [r for r in self.relations()
                if r.predicate == "inside" and r.obj == container_id]

    def neighbours(self, object_id: int) -> list[int]:
        out = set()
        for r in self.relations():
            if r.predicate != "near":
                continue
            if r.subject == object_id:
                out.add(r.obj)
            elif r.obj == object_id:
                out.add(r.subject)
        return sorted(out)

    def regions(self) -> list[list[int]]:
        """near 로 이어진 연결 성분. 방(room) 의 대용이다.

        진짜 방 분할은 벽/평면이 필요하므로 Geometry Engine 이 생긴 뒤에나
        가능하다. 그때까지의 근사이며, 그렇게 이름 붙이지 않는다.
        """
        adjacency: dict[int, set[int]] = {}
        for r in self.relations():
            if r.predicate != "near":
                continue
            adjacency.setdefault(r.subject, set()).add(r.obj)
            adjacency.setdefault(r.obj, set()).add(r.subject)

        seen: set[int] = set()
        components: list[list[int]] = []
        for node in sorted(adjacency):
            if node in seen:
                continue
            stack, comp = [node], []
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.append(n)
                stack.extend(sorted(adjacency.get(n, ()) - seen))
            components.append(sorted(comp))
        return components

    def describe(self, snapshot: WorldSnapshot | None = None) -> list[str]:
        """사람이 읽을 형태. 'cup#3 on dining table#7 (0.82)'"""
        def name(tid: int) -> str:
            if snapshot is not None:
                t = snapshot.get(tid)
                if t is not None and t.class_name:
                    return f"{t.class_name}#{tid}"
            return f"#{tid}"

        return [f"{name(r.subject)} {r.predicate} {name(r.obj)} ({r.confidence:.2f})"
                for r in self.relations()]

    def __len__(self) -> int:
        return len(self._edges)

    def __repr__(self) -> str:
        return f"WorldGraph({len(self._edges)} relations)"
