"""Token Constellation Geometry (TCG) 참조 구현.

C++ src/token/ConstellationIndex.cpp 와 같은 알고리즘:
  클래스 다중집합 + 로그 구간 쌍거리 스펙트럼으로 역색인 검색
  -> 클래스 일관 최대 클리크로 대응 탐색
  -> Kabsch 로 SE(3) 복원
  -> 카이제곱 게이트 + 모호성 기각

설계 근거: docs/02-correspondence-problem.md 3장 Tier 1.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import SE3, kabsch


@dataclass
class Node:
    """성좌를 구성하는 최소 정보."""
    token_id: int
    class_id: int
    position: np.ndarray          # 성좌 기준 좌표계
    sigma: float = 0.05


@dataclass
class Place:
    place_id: int
    keyframe: int
    stamp: float
    anchor: SE3
    nodes: list[Node]
    # 이 장소 좌표계에서의 중력 방향. 카이랄리티 판정에 쓰인다.
    # None 이면 미상이고, 그 경우 카이랄리티는 적용하지 않는다.
    gravity: np.ndarray | None = None


# 경쟁 후보가 없을 때의 pose_margin 표시값. 무한대를 CSV 로 내보내지 않는다.
NO_RIVAL_MARGIN = 999.0

# 순위용 점수 격자. C++ 의 kScoreQuantum 과 같은 값이어야 한다.
#
# 점수는 Kabsch 의 SVD 를 거친 rms 로 만들어지므로, 수학적으로 완전히 동률인
# 후보들도 마지막 1 ULP 가 갈린다. 실측: 같은 방을 평행이동만 다르게 세 번 넣고
# 질의하면 세 후보 모두 정확 정합인데 rms 가 5.567e-16 / 5.574e-16 / 6.863e-16
# 으로 나오고 점수가 0x1.ffffffffffffap-1 대 0x1.ffffffffffff9p-1 이 된다.
# 그 1 ULP 가 place_id 타이브레이크를 건너뛰면 순위가 반올림 잡음에 걸리고,
# LAPACK 구현이 다른 기계에서는 다른 장소가 뽑힌다 (실제로 리눅스 CI 와
# 윈도우가 갈렸다). 격자에 올린 뒤 비교하면 동률이 실제로 동률로 판정된다.
SCORE_QUANTUM = 1e-12

# 같은 이유로 _retrieve 의 투표값에도 격자가 필요하다. C++ 의 kVoteQuantum 과
# 같은 값이어야 한다. 투표는 [0,1] 이 아니라 idf 합이라 간격이 다르다:
# idf = log(1+n/df) 는 장소 1000 개에서도 6.9 이하고, 서명 항이 수십 개, 노드 수
# 제곱근으로 나누므로 실제 값은 O(1)~O(1000) 이다. 그 상단에서 1 ULP 는 약
# 2.3e-13 이니 1e-9 는 잡음을 덮고, 진짜 차이(항 하나가 다르면 최소 O(0.1))
# 보다는 여덟 자리 아래다.
VOTE_QUANTUM = 1e-9


def _quantise(x: float, q: float) -> float:
    """양자화한 비교 키.

    파이썬 내장 round() 는 짝수 반올림이라 정확히 절반인 값에서 C++ std::round
    와 갈린다. 두 구현이 같은 순서를 내야 하므로 양쪽 다 floor(x+0.5) 를 쓴다.
    점수는 [0,1] 로 clip 되고 투표는 idf 합이라 둘 다 음수가 아니다.
    """
    return math.floor(x / q + 0.5)


def _rank_key(score: float) -> float:
    return _quantise(score, SCORE_QUANTUM)


def _vote_key(vote: float) -> float:
    return _quantise(vote, VOTE_QUANTUM)


@dataclass
class Match:
    place_id: int
    keyframe: int
    transform: SE3                # query 좌표계 -> place 좌표계
    rms_error: float
    score: float
    correspondences: list[tuple[int, int]] = field(default_factory=list)

    # --- 신뢰도 원자료 ---------------------------------------------------
    # score/rms 는 "대응이 서로 일관적인가" 만 잰다. 동일한 모니터 두 대에
    # 어긋나게 붙인 대응도 완벽히 자기일관적이다.
    n_query_nodes: int = 0
    n_place_nodes: int = 0
    n_inliers: int = 0
    explained: float = 0.0        # n_inliers / n_query_nodes
    chi2_dof: float = 0.0         # 노드 공분산 정규화 잔차, 자유도당

    # 후보 집합 전체를 봐야 정해지는 값들. annotate() 가 채운다.
    agree_count: int = 1          # 같은 포즈 군집의 후보 수 (자기 포함)
    support: float = 0.0          # 자기 군집의 질량 = 점수합
    rival_mass: float = 0.0       # 가장 무거운 *다른* 군집의 질량
    pose_margin: float = 0.0      # 그 경쟁 군집까지의 포즈 거리 (m)
    confidence: float = 0.0       # 최종 채택 신뢰도 0..1


@dataclass
class Config:
    min_distance: float = 0.20
    max_distance: float = 25.0
    distance_bins: int = 24
    min_nodes: int = 4
    max_nodes: int = 40
    top_candidates: int = 8
    max_pairs: int = 512
    # 허용오차 = 고정 base + 거리 비례(스케일 오차) + 3-sigma(관측 잡음).
    # base 를 크게 잡으면 큰 방에서 오대응 클리크가 정대응과 같은 크기로 자라
    # 정확한 데이터에서도 틀린 변환이 나온다.
    distance_tolerance: float = 0.12
    relative_tolerance: float = 0.03
    sigma_gate: float = 3.0
    min_inliers: int = 4
    max_rms_error: float = 0.45
    chi2_gate: float = 9.21
    use_chirality: bool = True

    # --- 모호성 판정 -----------------------------------------------------
    # 점수비(score2 > 0.85*score1)만으로 자르면 조밀하게 샘플링한 지도에서
    # 정대응이 전멸한다 - fr1_xyz 에서 36개 중 36개 기각(재현율 0 %). 이웃
    # 키프레임이 비슷한 점수를 받는 것은 지각적 혼동이 아니라 정상이다.
    # 물어야 할 것은 "상위 후보들이 같은 *포즈* 를 가리키는가" 다.
    pose_agree_radius: float = 0.50      # m. 이 안이면 같은 포즈 군집
    pose_agree_deg: float = 25.0         # deg
    pose_dominance: float = 1.5          # 1위 군집 질량 / 2위 군집 질량 하한
    min_agree: int = 1                   # 군집 최소 크기 (자기 포함)

    # 신뢰도의 카이제곱 항 스케일. 1/(1 + chi2_dof/scale) 로 감쇠시킨다.
    chi2_confidence_scale: float = 10.0
    # 채택 신뢰도 하한. 0 으로 두면 모호성 규칙만으로 판정한다.
    min_confidence: float = 0.55


def _max_clique(adj: list[set[int]], n: int, budget: int = 200_000) -> list[int]:
    """Bron-Kerbosch(피벗) 로 최대 클리크 하나를 찾는다. 예산으로 상한을 건다.

    최대 클리크는 유일하지 않다. 허용오차 그래프가 같은 크기의 클리크를 여럿
    허용하면 "어느 것을 반환하는가" 는 순전히 순회 순서가 정한다. C++ 은
    비트셋을 하위 인덱스부터 훑으므로 여기서도 정점을 반드시 오름차순으로
    돌아야 두 구현이 같은 답을 낸다 - 집합 순회 순서에 맡기면 안 된다.
    """
    best: list[int] = []
    visited = 0

    def expand(R: list[int], P: set[int], X: set[int]) -> None:
        nonlocal best, visited
        visited += 1
        if visited > budget:
            return
        if not P and not X:
            if len(R) > len(best):
                best = list(R)
            return
        # 남은 후보를 다 넣어도 최선을 못 넘으면 중단
        if len(R) + len(P) <= len(best):
            return

        # 피벗: P∪X 중 P 와의 인접이 가장 많은 정점. 동률이면 작은 인덱스.
        pivot, best_deg = -1, -1
        for u in sorted(P | X):
            deg = len(P & adj[u])
            if deg > best_deg:
                pivot, best_deg = u, deg

        for v in sorted(P - adj[pivot]):
            expand(R + [v], P & adj[v], X & adj[v])
            P.discard(v)
            X.add(v)
            if visited > budget:
                return

    expand([], set(range(n)), set())
    return best


class ConstellationIndex:
    """장소 = 특정 배치로 놓인 객체들. 기술자 없이 재지역화한다."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self._places: dict[int, Place] = {}
        self._inverted: dict[int, list[int]] = {}
        self._next_id = 1

    # --- 서명 -------------------------------------------------------------

    def _distance_bin(self, d: float) -> int:
        c = self.cfg
        if d < c.min_distance or d > c.max_distance:
            return -1
        t = math.log(d / c.min_distance) / math.log(c.max_distance / c.min_distance)
        return int(min(max(t * c.distance_bins, 0), c.distance_bins - 1))

    def _pair_key(self, ca: int, cb: int, d: float) -> int | None:
        b = self._distance_bin(d)
        if b < 0:
            return None
        lo, hi = (ca, cb) if ca <= cb else (cb, ca)
        # 클래스 순서를 정규화해 순서 불변성을 확보
        return (lo & 0xFFFF) << 32 | (hi & 0xFFFF) << 16 | b

    def signature(self, nodes: list[Node]) -> list[int]:
        keys = []
        for i, j in itertools.combinations(range(len(nodes)), 2):
            d = float(np.linalg.norm(nodes[i].position - nodes[j].position))
            k = self._pair_key(nodes[i].class_id, nodes[j].class_id, d)
            if k is not None:
                keys.append(k)
        return keys

    # --- 등록 / 질의 -------------------------------------------------------

    def insert(self, keyframe: int, stamp: float, anchor: SE3, nodes: list[Node],
               gravity: np.ndarray | None = None) -> int:
        """장소 등록.

        gravity 는 *이 장소 좌표계에서의* 중력 방향이어야 한다. 질의 프레임의
        중력과 같은 벡터를 쓰면 안 된다 - 두 프레임은 서로 회전되어 있다.
        """
        if len(nodes) < self.cfg.min_nodes:
            return 0
        nodes = nodes[: self.cfg.max_nodes]

        g = None if gravity is None else np.asarray(gravity, float) / max(
            float(np.linalg.norm(gravity)), 1e-9)

        pid = self._next_id
        self._next_id += 1
        self._places[pid] = Place(pid, keyframe, stamp, anchor, nodes, g)

        # 같은 키가 여러 번 나와도 장소는 한 번만 등록 (투표 편향 방지)
        for k in set(self.signature(nodes)):
            self._inverted.setdefault(k, []).append(pid)
        return pid

    def _retrieve(self, sig: list[int]) -> list[tuple[int, float]]:
        """idf 가중 투표. 어디에나 있는 흔한 쌍은 변별력이 없다."""
        n_places = max(1, len(self._places))
        votes: dict[int, float] = {}
        # 키를 *정렬해서* 훑는다. C++ retrieve 가 unique_keys 를 sort 한 뒤
        # 누적하므로 여기서도 같은 순서여야 한다. set 순회 순서로 더하면 같은
        # idf 들을 다른 순서로 더하게 되고, 부동소수 덧셈은 결합법칙이 성립하지
        # 않으므로 두 구현의 투표값이 마지막 비트에서 갈린다. 아래 격자가 그
        # 대부분을 덮지만, 애초에 같은 순서로 더하는 편이 옳다.
        for k in sorted(set(sig)):
            hits = self._inverted.get(k)
            if not hits:
                continue
            idf = math.log(1.0 + n_places / len(hits))
            for pid in hits:
                votes[pid] = votes.get(pid, 0.0) + idf

        ranked = [
            # 장소 크기로 정규화하지 않으면 객체가 많은 장소가 항상 이긴다
            (pid, s / math.sqrt(len(self._places[pid].nodes)))
            for pid, s in votes.items()
        ]
        # 투표값도 격자에 올려서 비교한다. 원시 float 로 재면 query_all 에서와
        # 같은 이유로 타이브레이크가 발동하지 못하고, top_candidates 절단이
        # 무엇을 자를지가 잡음에 걸린다 (VOTE_QUANTUM 주석 참고).
        ranked.sort(key=lambda x: (-_vote_key(x[1]), x[0]))
        return ranked[: self.cfg.top_candidates]

    def _verify(self, query: list[Node], place: Place,
                q_gravity: np.ndarray | None = None) -> Match | None:
        cfg = self.cfg
        # 양쪽 프레임의 중력을 모두 알 때만 카이랄리티를 적용한다.
        # 한쪽만 알고 같은 벡터로 비교하면 정상 대응을 걸러낸다.
        use_chirality = cfg.use_chirality and q_gravity is not None and place.gravity is not None

        # 1) 클래스 일관 대응 후보
        pairs = [(qi, pi)
                 for qi, q in enumerate(query)
                 for pi, p in enumerate(place.nodes)
                 if q.class_id == p.class_id]
        pairs = pairs[: cfg.max_pairs]
        if len(pairs) < cfg.min_inliers:
            return None

        # 2) 쌍거리 일관성 그래프
        n = len(pairs)
        adj: list[set[int]] = [set() for _ in range(n)]
        for a, b in itertools.combinations(range(n), 2):
            qa, pa = pairs[a]
            qb, pb = pairs[b]
            if qa == qb or pa == pb:            # 같은 노드 재사용 불가
                continue

            dq = float(np.linalg.norm(query[qa].position - query[qb].position))
            dp = float(np.linalg.norm(place.nodes[pa].position - place.nodes[pb].position))

            # 거리차의 표준편차는 네 노드 분산의 제곱합근이다. 단순 합은
            # 지나치게 관대해 오대응을 통과시킨다.
            var = (query[qa].sigma ** 2 + query[qb].sigma ** 2
                   + place.nodes[pa].sigma ** 2 + place.nodes[pb].sigma ** 2)
            tol = (cfg.distance_tolerance + cfg.relative_tolerance * max(dq, dp)
                   + cfg.sigma_gate * math.sqrt(var))
            if abs(dq - dp) > tol:
                continue

            if use_chirality:
                # 중력축 기준 상하 배치가 뒤집힌 대응을 배제 (거울 모호성).
                # 각 프레임의 높이는 그 프레임 자신의 중력벡터로 재야 한다.
                hq = float((query[qb].position - query[qa].position) @ q_gravity)
                hp = float((place.nodes[pb].position - place.nodes[pa].position) @ place.gravity)
                if abs(hq) > 0.3 and abs(hp) > 0.3 and hq * hp < 0.0:
                    continue

            adj[a].add(b)
            adj[b].add(a)

        # 3) 최대 클리크 = 최대 상호일관 대응 집합
        clique = _max_clique(adj, n)
        if len(clique) < cfg.min_inliers:
            return None

        src = np.array([query[pairs[i][0]].position for i in clique])
        dst = np.array([place.nodes[pairs[i][1]].position for i in clique])
        try:
            T = kabsch(src, dst)
        except ValueError:
            return None

        # 4) 잔차 + 카이제곱 게이트
        #
        # 카이제곱은 노드 자신의 위치 분산으로 정규화한 잔차다. rms 는 미터 단위라
        # 정밀한 근거리 노드와 거친 원거리 노드를 같은 저울에 올린다.
        residual = (T @ src) - dst
        r2 = np.sum(residual ** 2, axis=1)
        rms = float(np.sqrt(r2.mean()))

        sq = np.array([query[pairs[i][0]].sigma for i in clique])
        sp = np.array([place.nodes[pairs[i][1]].sigma for i in clique])
        gated = int(np.sum(r2 / ((sq + sp + 0.05) ** 2) < cfg.chi2_gate))

        if rms > cfg.max_rms_error or gated < cfg.min_inliers:
            return None

        # 축당 분산은 두 노드 분산의 합. 바닥값을 두어 sigma=0 발산을 막는다.
        var = np.maximum(1e-4, sq ** 2 + sp ** 2)
        chi2 = float(np.sum(r2 / var))
        dof = max(1.0, 3.0 * len(clique) - 6.0)   # Kabsch 가 6 DoF 를 소모

        inlier_term = gated / max(1, min(len(query), len(place.nodes)))
        error_term = 1.0 - min(1.0, rms / cfg.max_rms_error)
        score = float(np.clip(math.sqrt(max(0.0, inlier_term) * error_term), 0.0, 1.0))

        return Match(
            place_id=place.place_id,
            keyframe=place.keyframe,
            transform=T,
            rms_error=rms,
            score=score,
            correspondences=[(query[pairs[i][0]].token_id, place.nodes[pairs[i][1]].token_id)
                             for i in clique],
            n_query_nodes=len(query),
            n_place_nodes=len(place.nodes),
            n_inliers=gated,
            # 설명률: 질의 성좌 중 이 장소가 설명한 비율. 8개 중 4개만 설명한
            # 정합은 4개 중 4개를 설명한 정합과 같은 rms 를 가질 수 있지만
            # 훨씬 약한 증거다.
            explained=(gated / len(query)) if query else 0.0,
            chi2_dof=chi2 / dof,
        )

    def world_pose(self, m: Match) -> SE3 | None:
        """후보의 월드 포즈. anchor 를 곱해야 서로 비교할 수 있다."""
        place = self._places.get(m.place_id)
        if place is None:
            return None
        return place.anchor @ m.transform

    def annotate(self, all_m: list[Match]) -> None:
        """후보 집합 전체를 보고 포즈 합의 통계를 채운다.

        후보 하나만 봐서는 알 수 없는 양이라 _verify 가 아니라 여기서 계산한다.
        all_m 은 점수 내림차순이어야 한다 - 각 군집의 대표가 그 군집에서 가장
        점수가 높은 후보가 되기 때문이다.
        """
        if not all_m:
            return
        cfg = self.cfg
        poses = [self.world_pose(m) for m in all_m]

        # 1) 후보를 월드 포즈로 탐욕 군집화
        cluster_of = [0] * len(all_m)
        reps: list[int] = []
        for i in range(len(all_m)):
            if poses[i] is None:
                cluster_of[i] = len(reps)
                reps.append(i)
                continue
            hit = len(reps)
            for c, ri in enumerate(reps):
                rp = poses[ri]
                if rp is None:
                    continue
                dt, dr = poses[i].distance_to(rp)      # (병진 m, 회전 rad)
                if dt <= cfg.pose_agree_radius and math.degrees(dr) <= cfg.pose_agree_deg:
                    hit = c
                    break
            if hit == len(reps):
                reps.append(i)
            cluster_of[i] = hit

        # 2) 군집 질량 = 점수합. 개수가 아니라 질량으로 재야 약한 후보 여럿이
        #    강한 후보 하나를 이기지 못한다.
        mass = [0.0] * len(reps)
        size = [0] * len(reps)
        for i, m in enumerate(all_m):
            mass[cluster_of[i]] += m.score
            size[cluster_of[i]] += 1

        # 3) 각 후보에 자기 군집과 최강 경쟁 군집을 붙인다.
        for i, m in enumerate(all_m):
            ci = cluster_of[i]
            m.agree_count = size[ci]
            m.support = mass[ci]
            m.rival_mass = 0.0
            m.pose_margin = NO_RIVAL_MARGIN

            for c, ri in enumerate(reps):
                if c == ci:
                    continue
                if mass[c] > m.rival_mass:
                    m.rival_mass = mass[c]
                if poses[i] is not None and poses[ri] is not None:
                    m.pose_margin = min(m.pose_margin, poses[i].distance_to(poses[ri])[0])

            # 신뢰도 = 자기일관성 x 포즈 공간 우세 x 정규화 잔차.
            # 세 항이 서로 다른 것을 잰다. score 는 한 후보 안에서만 계산되므로
            # "옳은 성좌인가"를 모른다 - 우세 항이 그 구멍을 메운다. 절대 질량을
            # 쓰면 지도 샘플링 밀도를 재게 되므로 반드시 비로 쓴다.
            denom = m.support + m.rival_mass
            dominance = (m.support / denom) if denom > 1e-10 else 0.0
            chi2_term = 1.0 / (1.0 + max(0.0, m.chi2_dof)
                               / max(1e-10, cfg.chi2_confidence_scale))
            m.confidence = float(np.clip(m.score * dominance * chi2_term, 0.0, 1.0))

    def query_all(self, nodes: list[Node],
                  gravity: np.ndarray | None = None) -> list[Match]:
        """gravity 는 *질의 프레임 좌표계에서의* 중력 방향 (IMU 또는 지배평면 법선)."""
        if len(nodes) < self.cfg.min_nodes or not self._places:
            return []
        g = None if gravity is None else np.asarray(gravity, float) / max(
            float(np.linalg.norm(gravity)), 1e-9)

        out = []
        for pid, _vote in self._retrieve(self.signature(nodes)):
            m = self._verify(nodes, self._places[pid], g)
            if m is not None:
                out.append(m)
        # place_id 로 타이브레이크한다. C++ queryAll 이 같은 전순서를 쓴다.
        #
        # 점수만으로 정렬하면 파이썬은 안정 정렬이라 *후보 생성 순서* 가 동률
        # 구간에 남는다. 그 순서는 retrieve 의 순서, 즉 idf *투표* 점수 순이다.
        # 정합 점수는 같은데 투표 점수가 다른 두 장소는 얼마든지 있을 수 있고,
        # 그러면 C++ (place_id 순)과 파이썬 (투표 순)이 갈린다. query() 는
        # out[0] 을 대표로 뽑고 annotate() 는 이 순서로 군집 대표를 정하므로
        # 그 차이가 그대로 결과가 된다.
        #
        # 점수는 격자에 올려서 비교한다. 원시 float 로 재면 SVD 반올림 잡음
        # 1 ULP 가 타이브레이크를 건너뛰게 만든다 (SCORE_QUANTUM 주석 참고).
        out.sort(key=lambda m: (-_rank_key(m.score), m.place_id))
        self.annotate(out)
        return out

    def query(self, nodes: list[Node], gravity: np.ndarray | None = None) -> Match | None:
        """단일 최선 후보. 모호하면 아무것도 반환하지 않는다.

        예전 규칙은 score2 > 0.85*score1 이면 기각이었다. 그 규칙은 "점수가
        붙으면 지각적 혼동" 이라고 가정하지만, 5프레임 간격으로 등록한 지도에서는
        이웃 키프레임이 같은 장면을 보므로 점수가 붙는 것이 정상이다. fr1_xyz
        에서 이 규칙은 정대응 36개를 36개 모두 기각했다 (재현율 0 %).

        실제로 물어야 할 것은 "붙은 후보들이 같은 포즈를 가리키는가" 다.
          - 이웃 키프레임: 다른 anchor x 다른 transform -> 한 군집. 채택.
          - 동일한 복도 두 개: 같은 transform x 50 m 떨어진 anchor -> 두 군집,
            질량 동률 -> 기각.
        점수 공간에서는 두 상황이 똑같이 보이고, 포즈 공간에서는 다르게 보인다.

        대표는 여전히 최고점 후보다. "가장 무거운 군집의 대표" 도 시험했고 어느
        시퀀스에서도 이기지 못했다 (fr3_walking 정밀도 42 -> 28 %). 동적 장면에서는
        이웃 장소들이 같은 오답에 합의하므로 질량이 증거가 되지 못한다. 그래서
        질량은 *기각* 에만 쓰고 *선택* 에는 쓰지 않는다.
        """
        cfg = self.cfg
        if len(nodes) < cfg.min_nodes:
            return None
        all_m = self.query_all(nodes, gravity)
        if not all_m:
            return None

        top = all_m[0]
        if top.agree_count < cfg.min_agree:
            return None
        if top.rival_mass > 1e-10 and top.support < cfg.pose_dominance * top.rival_mass:
            return None
        if top.confidence < cfg.min_confidence:
            return None
        return top

    @property
    def place_count(self) -> int:
        return len(self._places)
