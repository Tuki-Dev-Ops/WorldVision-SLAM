"""데이터 연관 전략 네 가지.

docs/04-unified-objective.md 5.4 의 주장을 검증 가능한 형태로 만든 것.

  Greedy    - 가장 가까운 것부터. 붙어 있는 동일 클래스 객체에서 정체성을 섞는다.
  Hungarian - 프레임별 전역 최적. 현재 TokenStore 의 방식. 결정을 즉시 확정한다.
  Deferred  - 1등/2등 격차가 충분할 때만 확정. 애매하면 판단을 미룬다.
  MHT       - 연관 이력 자체에 대한 사후분포를 유지하고 가지치기한다.

핵심 주장: 객체는 프레임당 ~20개지 키포인트처럼 ~2000개가 아니다. 그래서
고전 SLAM 에서 불가능했던 MHT 가 객체 수준에서는 실현 가능하다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..reference.assignment import INFEASIBLE, solve_assignment
from .murty import murty_k_best


@dataclass
class TrackState:
    """정보필터로 유지되는 하나의 트랙."""

    track_id: int
    class_id: int
    information: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    info_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    count: int = 0
    last_frame: int = -1

    # 관측 수로 줄지 않는 계통 오차 성분 (M3 참조).
    # 이게 없으면 트랙이 확신할수록 게이트가 좁아지는데 편향은 그대로라
    # 정상 관측이 게이트 밖으로 밀려나고 새 트랙이 계속 생긴다.
    # 과신은 보고 문제로 끝나지 않고 연관을 직접 망가뜨린다.
    systematic_sigma: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.linalg.solve(self.information + np.eye(3) * 1e-9, self.info_vector)

    @property
    def covariance(self) -> np.ndarray:
        P = np.linalg.inv(self.information + np.eye(3) * 1e-9)
        return P + np.eye(3) * (self.systematic_sigma ** 2)

    def update(self, z: np.ndarray, R: np.ndarray, frame: int) -> None:
        R_inv = np.linalg.inv(R + np.eye(3) * 1e-12)
        self.information = self.information + R_inv
        self.info_vector = self.info_vector + R_inv @ z
        self.count += 1
        self.last_frame = frame

    def copy(self) -> "TrackState":
        return TrackState(self.track_id, self.class_id, self.information.copy(),
                          self.info_vector.copy(), self.count, self.last_frame,
                          self.systematic_sigma)


@dataclass
class Measurement:
    z: np.ndarray                 # 월드 좌표 관측
    R: np.ndarray                 # 관측 공분산
    class_id: int


@dataclass
class AssociationConfig:
    gate_chi2: float = 11.34          # 자유도 3, 99%
    p_detect: float = 0.9
    log_new_track: float = -6.0       # 새 트랙 생성 비용 (로그우도)
    log_false_alarm: float = -8.0
    allow_cross_class: bool = False

    # 트랙 공분산의 계통 성분 하한 (m). M3 에서 측정된 모델링 오차 규모.
    # 0 으로 두면 트랙이 확신할수록 게이트가 좁아져 연관이 붕괴한다.
    systematic_sigma: float = 0.06

    # 트랙 생애주기. TokenStore 의 Provisional -> Active -> Retired 에 대응한다.
    # 이게 없으면 오검출 하나하나가 영구 트랙이 되어, 연관 전략의 차이가
    # 쓰레기 트랙 수에 파묻힌다.
    confirm_count: int = 3            # 이 이상 관측되어야 확정 트랙
    prune_age: int = 12               # 미확정 트랙을 이만큼 방치하면 폐기

    # Deferred
    commit_margin: float = 2.0        # 1등/2등 비용 격차가 이 이상이면 확정
    defer_window: int = 5             # 이만큼 지나면 강제 확정

    # MHT
    max_hypotheses: int = 20
    k_best: int = 4


def _neg_log_likelihood(track: TrackState, m: Measurement,
                        cfg: AssociationConfig) -> float:
    """관측이 이 트랙에서 나왔을 음의 로그우도. 게이트 밖이면 INFEASIBLE."""
    if not cfg.allow_cross_class and track.class_id != m.class_id:
        return INFEASIBLE
    if track.count == 0:
        return INFEASIBLE

    S = track.covariance + m.R
    diff = m.z - track.position
    try:
        maha = float(diff @ np.linalg.solve(S, diff))
        sign, logdet = np.linalg.slogdet(S)
    except np.linalg.LinAlgError:
        return INFEASIBLE
    if sign <= 0 or maha > cfg.gate_chi2:
        return INFEASIBLE

    # 상수항까지 포함해야 '새 트랙' 가설과 정직하게 비교된다
    return 0.5 * (maha + logdet + 3.0 * math.log(2.0 * math.pi)) - math.log(cfg.p_detect)


def _build_cost(tracks: list[TrackState], measurements: list[Measurement],
                cfg: AssociationConfig) -> np.ndarray:
    """(검출 x (트랙 + 신규)) 비용행렬.

    신규 트랙 열을 명시적으로 두는 것이 중요하다. 이게 없으면 게이트를 통과하는
    트랙이 하나라도 있으면 무조건 붙어버린다.
    """
    n_d, n_t = len(measurements), len(tracks)
    cost = np.full((n_d, n_t + n_d), INFEASIBLE)

    for d, m in enumerate(measurements):
        for t, tr in enumerate(tracks):
            cost[d, t] = _neg_log_likelihood(tr, m, cfg)
        cost[d, n_t + d] = -cfg.log_new_track      # 자기 전용 '신규' 열
    return cost


class Tracker:
    """공통 인터페이스. update() 는 검출 -> 트랙 ID 매핑을 반환한다."""

    def __init__(self, cfg: AssociationConfig | None = None):
        self.cfg = cfg or AssociationConfig()
        self.tracks: dict[int, TrackState] = {}
        self._next_id = 1

    def _new_track(self, m: Measurement, frame: int) -> TrackState:
        tr = TrackState(self._next_id, m.class_id,
                        systematic_sigma=self.cfg.systematic_sigma)
        self._next_id += 1
        tr.update(m.z, m.R, frame)
        self.tracks[tr.track_id] = tr
        return tr

    def update(self, measurements: list[Measurement], frame: int) -> list[int]:
        raise NotImplementedError

    def prune(self, frame: int) -> None:
        """지지를 얻지 못한 미확정 트랙을 폐기한다.

        오검출은 한두 번 나타났다 사라진다. 그것을 영구 트랙으로 남기면
        지도가 유령으로 오염되고, 이후 연관도 그만큼 어려워진다.
        """
        cfg = self.cfg
        self.tracks = {
            tid: tr for tid, tr in self.tracks.items()
            if tr.count >= cfg.confirm_count or frame - tr.last_frame <= cfg.prune_age
        }

    @property
    def confirmed(self) -> dict[int, TrackState]:
        return {tid: tr for tid, tr in self.tracks.items()
                if tr.count >= self.cfg.confirm_count}

    @property
    def name(self) -> str:
        return type(self).__name__


class GreedyTracker(Tracker):
    """가장 좋은 쌍부터 확정. 붙어 있는 동일 클래스 객체에서 무너진다."""

    def update(self, measurements: list[Measurement], frame: int) -> list[int]:
        tracks = list(self.tracks.values())
        assigned = [-1] * len(measurements)
        used: set[int] = set()

        pairs = []
        for d, m in enumerate(measurements):
            for t, tr in enumerate(tracks):
                c = _neg_log_likelihood(tr, m, self.cfg)
                if c < INFEASIBLE:
                    pairs.append((c, d, t))
        pairs.sort()

        taken_d: set[int] = set()
        for c, d, t in pairs:
            if d in taken_d or t in used or c > -self.cfg.log_new_track:
                continue
            taken_d.add(d)
            used.add(t)
            tracks[t].update(measurements[d].z, measurements[d].R, frame)
            assigned[d] = tracks[t].track_id

        for d, m in enumerate(measurements):
            if assigned[d] < 0:
                assigned[d] = self._new_track(m, frame).track_id
        return assigned


class HungarianTracker(Tracker):
    """프레임별 전역 최적 할당을 즉시 확정한다. 현재 TokenStore 의 방식."""

    def update(self, measurements: list[Measurement], frame: int) -> list[int]:
        tracks = list(self.tracks.values())
        cost = _build_cost(tracks, measurements, self.cfg)
        r2c, _, _ = solve_assignment(cost)

        assigned = []
        for d, m in enumerate(measurements):
            col = int(r2c[d])
            if 0 <= col < len(tracks):
                tracks[col].update(m.z, m.R, frame)
                assigned.append(tracks[col].track_id)
            else:
                assigned.append(self._new_track(m, frame).track_id)
        return assigned


class DeferredTracker(Tracker):
    """1등과 2등의 격차가 충분할 때만 확정한다.

    애매한 관측은 보류 큐에 넣고, 트랙이 더 확실해지면 그때 붙인다.
    확정 결정을 되돌릴 수는 없지만, 나쁜 결정을 *하지 않을* 수는 있다.
    """

    def __init__(self, cfg: AssociationConfig | None = None):
        super().__init__(cfg)
        self._pending: list[tuple[Measurement, int]] = []

    def update(self, measurements: list[Measurement], frame: int) -> list[int]:
        # 보류분 중 창을 넘긴 것은 이제 확정한다
        revived = [m for m, f in self._pending if frame - f >= self.cfg.defer_window]
        self._pending = [(m, f) for m, f in self._pending
                         if frame - f < self.cfg.defer_window]

        tracks = list(self.tracks.values())
        cost = _build_cost(tracks, measurements, self.cfg)
        r2c, _, _ = solve_assignment(cost)

        assigned = []
        for d, m in enumerate(measurements):
            col = int(r2c[d])
            if not (0 <= col < len(tracks)):
                assigned.append(self._new_track(m, frame).track_id)
                continue

            best = cost[d, col]
            others = [cost[d, t] for t in range(len(tracks)) if t != col]
            second = min(others) if others else INFEASIBLE
            margin = (second - best) if second < INFEASIBLE else INFEASIBLE

            if margin >= self.cfg.commit_margin:
                tracks[col].update(m.z, m.R, frame)
                assigned.append(tracks[col].track_id)
            else:
                # 애매하다. 지금 붙이지 않는다.
                self._pending.append((m, frame))
                assigned.append(-1)

        for m in revived:
            # 창이 끝났으므로 그때의 최선으로 확정한다
            tracks = list(self.tracks.values())
            if tracks:
                costs = [_neg_log_likelihood(tr, m, self.cfg) for tr in tracks]
                best_t = int(np.argmin(costs))
                if costs[best_t] < -self.cfg.log_new_track:
                    tracks[best_t].update(m.z, m.R, frame)
                    continue
            self._new_track(m, frame)
        return assigned


@dataclass
class _Hypothesis:
    """하나의 연관 이력 가설.

    tracks 의 키는 *전역* 트랙 ID 다. 가설마다 ID 를 따로 매기면, 선두 가설이
    바뀔 때 보고되는 ID 가 통째로 뒤바뀌어 정체성 지표가 무의미해진다.
    가설이 다른 것은 '어느 검출을 어느 트랙에 붙이느냐'이지 '트랙이 무엇이냐'가
    아니므로, 트랙 정체성은 가설 사이에서 공유되어야 한다.
    """
    tracks: dict[int, TrackState]
    log_score: float
    assignment: list[int] = field(default_factory=list)
    history: list[tuple[int, list[int]]] = field(default_factory=list)

    def clone(self) -> "_Hypothesis":
        return _Hypothesis({k: v.copy() for k, v in self.tracks.items()},
                           self.log_score, list(self.assignment), list(self.history))


class MhtTracker(Tracker):
    """연관 이력에 대한 사후분포를 유지한다 (트랙 지향).

    프레임마다 각 가설을 k-최적 할당으로 확장하고, 누적 점수 상위 N 개만 남긴다.
    확정을 미루는 게 아니라 *여러 확정을 동시에 들고 간다*. 증거가 쌓이면 틀린
    가지가 저절로 죽는다 - 되돌릴 수 없는 결정을 아예 하지 않는 것이다.

    MHT 는 본질적으로 필터가 아니라 평활기다. 온라인 출력(update 반환값)은
    그 시점 최선 가설의 판단이라 아직 흔들릴 수 있고, 진짜 이득은 끝까지 간 뒤
    최선 가설의 이력 전체를 읽는 retrospective() 에서 나온다.

    객체가 프레임당 수십 개이므로 가설 20개로 묶어도 충분히 넓다. 키포인트
    수천 개에서는 성립하지 않는 조건이고, 이것이 객체 중심 표현의 구체적 이득이다.
    """

    def __init__(self, cfg: AssociationConfig | None = None):
        super().__init__(cfg)
        self._hyps: list[_Hypothesis] = [_Hypothesis({}, 0.0)]
        # (프레임, 검출 인덱스) -> 전역 트랙 ID.
        # 서로 다른 가설이 같은 검출로 새 트랙을 만들면 같은 ID 를 쓴다.
        self._new_track_ids: dict[tuple[int, int], int] = {}

    def _global_new_id(self, frame: int, det_index: int) -> int:
        key = (frame, det_index)
        if key not in self._new_track_ids:
            self._new_track_ids[key] = self._next_id
            self._next_id += 1
        return self._new_track_ids[key]

    def update(self, measurements: list[Measurement], frame: int) -> list[int]:
        cfg = self.cfg
        expanded: list[_Hypothesis] = []

        for hyp in self._hyps:
            track_ids = list(hyp.tracks)
            tracks = [hyp.tracks[t] for t in track_ids]
            cost = _build_cost(tracks, measurements, cfg)

            for solution, total in murty_k_best(cost, cfg.k_best):
                child = hyp.clone()
                child.assignment = []

                for d, m in enumerate(measurements):
                    col = int(solution[d])
                    if 0 <= col < len(track_ids):
                        tid = track_ids[col]
                        child.tracks[tid].update(m.z, m.R, frame)
                    else:
                        tid = self._global_new_id(frame, d)
                        tr = TrackState(tid, m.class_id,
                                        systematic_sigma=cfg.systematic_sigma)
                        tr.update(m.z, m.R, frame)
                        child.tracks[tid] = tr
                    child.assignment.append(tid)

                child.log_score -= total          # 비용이 음의 로그우도
                child.history.append((frame, list(child.assignment)))
                expanded.append(child)

        if not expanded:
            expanded = [h.clone() for h in self._hyps]

        # 가지치기. 점수 내림차순 상위 N 개.
        expanded.sort(key=lambda h: -h.log_score)
        self._hyps = expanded[: cfg.max_hypotheses]

        best = self._hyps[0]
        self.tracks = best.tracks
        return list(best.assignment)

    def prune(self, frame: int) -> None:
        """모든 가설에 대해 미확정 트랙을 정리한다."""
        cfg = self.cfg
        for hyp in self._hyps:
            hyp.tracks = {
                tid: tr for tid, tr in hyp.tracks.items()
                if tr.count >= cfg.confirm_count or frame - tr.last_frame <= cfg.prune_age
            }
        self.tracks = self._hyps[0].tracks

    def retrospective(self) -> list[tuple[int, list[int]]]:
        """최선 가설의 전체 연관 이력.

        MHT 의 실제 이득은 여기서 드러난다. 온라인 출력은 그 시점의 잠정 판단이고,
        나중에 증거가 쌓여 뒤집힌 결정까지 반영된 것이 이 이력이다.
        """
        return list(self._hyps[0].history)

    @property
    def hypothesis_count(self) -> int:
        return len(self._hyps)
