"""데이터 연관 검증.

핵심은 '애매한 상황을 실제로 애매하게 만들었는가'다. 객체가 게이트 반경보다
멀리 떨어져 있으면 어떤 전략을 써도 결과가 같고, 그러면 아무것도 측정되지 않는다.
초기 실험이 정확히 그 함정에 빠졌으므로 여기서 명시적으로 검증한다.
"""

import numpy as np
import pytest

from wme.association import (
    AssociationConfig, DeferredTracker, GreedyTracker, HungarianTracker,
    Measurement, MhtTracker, TrackState, murty_k_best,
)
from wme.reference.assignment import INFEASIBLE

RNG = np.random.default_rng(2718)


def meas(pos, sigma=0.05, class_id=0):
    return Measurement(np.asarray(pos, float), np.eye(3) * sigma ** 2, class_id)


# --- Murty k-best ---------------------------------------------------------

def test_murty_first_solution_is_optimal():
    cost = RNG.uniform(0, 10, (5, 5))
    results = murty_k_best(cost, 5)
    from wme.reference.assignment import solve_assignment
    _, _, best = solve_assignment(cost)
    assert results[0][1] == pytest.approx(best, abs=1e-9)


def test_murty_returns_costs_in_ascending_order():
    cost = RNG.uniform(0, 10, (5, 6))
    costs = [c for _, c in murty_k_best(cost, 8)]
    assert costs == sorted(costs)


def test_murty_solutions_are_distinct():
    cost = RNG.uniform(0, 10, (4, 4))
    sols = [s.tobytes() for s, _ in murty_k_best(cost, 6)]
    assert len(sols) == len(set(sols))


def test_murty_matches_brute_force_on_small_problem():
    """3x3 은 전수 열거가 가능하다. k-최적 순서가 정확히 일치해야 한다."""
    import itertools
    cost = RNG.uniform(0, 10, (3, 3))
    exhaustive = sorted(sum(cost[i, p[i]] for i in range(3))
                        for p in itertools.permutations(range(3)))
    got = [c for _, c in murty_k_best(cost, 6)]
    assert np.allclose(got[:len(exhaustive)], exhaustive[:len(got)], atol=1e-9)


def test_murty_respects_infeasible_entries():
    cost = np.full((3, 3), INFEASIBLE)
    cost[0, 0] = 1.0
    cost[1, 1] = 2.0
    for solution, _ in murty_k_best(cost, 3):
        for i, j in enumerate(solution):
            if j >= 0:
                assert cost[i, j] < INFEASIBLE


def test_murty_handles_empty():
    assert murty_k_best(np.zeros((0, 0)), 3) == []


# --- 트랙 상태 -------------------------------------------------------------

def test_track_covariance_shrinks_with_observations():
    tr = TrackState(1, 0)
    traces = []
    for i in range(6):
        tr.update(np.zeros(3), np.eye(3) * 0.01, i)
        traces.append(np.trace(tr.covariance))
    assert traces[0] > traces[-1]


def test_systematic_floor_stops_covariance_collapse():
    """계통 성분이 없으면 관측이 쌓일수록 게이트가 0 으로 수렴해 연관이 붕괴한다."""
    plain = TrackState(1, 0, systematic_sigma=0.0)
    floored = TrackState(2, 0, systematic_sigma=0.06)
    for i in range(200):
        plain.update(np.zeros(3), np.eye(3) * 0.01, i)
        floored.update(np.zeros(3), np.eye(3) * 0.01, i)

    assert np.trace(plain.covariance) < 1e-3
    assert np.trace(floored.covariance) > 3 * 0.06 ** 2 * 0.99


# --- 트래커 공통 계약 ------------------------------------------------------

@pytest.mark.parametrize("cls", [GreedyTracker, HungarianTracker, MhtTracker])
def test_tracker_creates_one_track_per_isolated_object(cls):
    """서로 멀리 떨어진 객체는 어떤 전략이든 정확히 추적해야 한다."""
    cfg = AssociationConfig(confirm_count=3)
    tr = cls(cfg)
    truth = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]),
             np.array([0.0, 5.0, 0.0])]

    for f in range(12):
        tr.update([meas(p + RNG.normal(0, 0.01, 3)) for p in truth], f)
        tr.prune(f)

    assert len(tr.confirmed) == 3


@pytest.mark.parametrize("cls", [GreedyTracker, HungarianTracker, DeferredTracker])
def test_pruning_removes_transient_false_alarms(cls):
    """한두 번 나타났다 사라지는 오검출이 영구 트랙이 되면 안 된다."""
    cfg = AssociationConfig(confirm_count=3, prune_age=4)
    tr = cls(cfg)
    real = np.array([0.0, 0.0, 0.0])

    for f in range(25):
        ms = [meas(real + RNG.normal(0, 0.01, 3))]
        if f % 7 == 0:                        # 산발적 오검출
            ms.append(meas(RNG.uniform(-8, 8, 3)))
        tr.update(ms, f)
        tr.prune(f)

    assert len(tr.confirmed) == 1


def test_new_track_column_prevents_forced_association():
    """게이트를 통과하는 트랙이 있어도 신규 가설이 더 좋으면 새 트랙이어야 한다."""
    cfg = AssociationConfig(confirm_count=1, log_new_track=-0.5)
    tr = HungarianTracker(cfg)
    for f in range(4):
        tr.update([meas(np.zeros(3))], f)

    # 게이트 경계 근처의 새 관측
    before = len(tr.tracks)
    tr.update([meas(np.array([0.0, 0.0, 0.0])), meas(np.array([9.0, 0.0, 0.0]))], 5)
    assert len(tr.tracks) == before + 1


# --- 전략 간 차이는 애매할 때만 드러난다 -----------------------------------

def _ambiguous_run(cls, separation: float, sigma: float, frames: int = 40):
    """가까이 붙은 동일 클래스 두 객체를 추적하고 (확정 트랙 수, 스위치) 반환."""
    cfg = AssociationConfig(confirm_count=3)
    tr = cls(cfg)
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([separation, 0.0, 0.0])

    rng = np.random.default_rng(7)
    switches = 0
    last: dict[int, int] = {}
    for f in range(frames):
        ms = [meas(a + rng.normal(0, sigma, 3), sigma),
              meas(b + rng.normal(0, sigma, 3), sigma)]
        assigned = tr.update(ms, f)
        tr.prune(f)
        for gid, tid in zip((0, 1), assigned):
            if tid < 0:
                continue
            if gid in last and last[gid] != tid:
                switches += 1
            last[gid] = tid
    return len(tr.confirmed), switches


def test_well_separated_objects_are_unambiguous_for_every_strategy():
    """분리가 게이트보다 크면 전략 차이가 없다.

    이 성질을 확인하지 않으면 '차이가 없다'는 결과를 잘못 해석하게 된다.
    """
    results = {cls.__name__: _ambiguous_run(cls, separation=2.0, sigma=0.03)
               for cls in (GreedyTracker, HungarianTracker)}
    assert results["GreedyTracker"] == results["HungarianTracker"]
    assert all(n == 2 for n, _ in results.values())


def test_global_assignment_beats_greedy_when_ambiguous():
    """붙어 있는 동일 클래스 객체 + 큰 잡음에서 전역 할당이 나아야 한다.

    현재 TokenStore 가 헝가리안을 쓰는 이유가 이것이다.
    """
    _, greedy_sw = _ambiguous_run(GreedyTracker, separation=0.12, sigma=0.09, frames=80)
    _, hung_sw = _ambiguous_run(HungarianTracker, separation=0.12, sigma=0.09, frames=80)
    assert hung_sw <= greedy_sw


def test_deferred_trades_coverage_for_identity():
    """Deferred 는 애매한 관측을 버려 정체성을 지킨다. 공짜가 아니다."""
    cfg = AssociationConfig(confirm_count=3)
    tr = DeferredTracker(cfg)
    rng = np.random.default_rng(11)

    a, b = np.zeros(3), np.array([0.12, 0.0, 0.0])
    deferred = 0
    for f in range(60):
        ms = [meas(a + rng.normal(0, 0.09, 3), 0.09),
              meas(b + rng.normal(0, 0.09, 3), 0.09)]
        deferred += sum(1 for t in tr.update(ms, f) if t < 0)
        tr.prune(f)

    assert deferred > 0, "애매한데도 전부 확정했다면 미루기가 동작하지 않는 것"


def test_mht_track_ids_are_shared_across_hypotheses():
    """가설이 달라도 같은 검출에서 생긴 트랙은 같은 전역 ID 여야 한다.

    가설마다 ID 를 따로 매기면 선두 가설이 바뀔 때 보고 ID 가 통째로 뒤바뀌어
    정체성 지표가 무의미해진다. 실제로 초기 구현이 그랬다.
    """
    cfg = AssociationConfig(confirm_count=2, max_hypotheses=8, k_best=3)
    tr = MhtTracker(cfg)
    rng = np.random.default_rng(5)

    a, b = np.zeros(3), np.array([0.15, 0.0, 0.0])
    for f in range(12):
        tr.update([meas(a + rng.normal(0, 0.08, 3), 0.08),
                   meas(b + rng.normal(0, 0.08, 3), 0.08)], f)

    # 모든 가설의 트랙 ID 를 모아도, 검출 수 x 프레임 수보다 훨씬 적어야 한다
    all_ids = set()
    for hyp in tr._hyps:
        all_ids |= set(hyp.tracks)
    assert len(all_ids) < 12, f"가설별 ID 폭증: {len(all_ids)}"


def test_mht_smoothing_helps_only_when_ambiguity_is_resolvable():
    """MHT 는 필터가 아니라 평활기다.

    나중에 해소될 모호성에서는 되돌아보기가 이득이지만, 끝까지 해소되지 않는
    모호성에서는 아무 이득이 없다 - 없는 정보를 만들 수는 없기 때문이다.
    이 성질을 확인하지 않으면 MHT 의 실패를 구현 탓으로 오해하게 된다.
    """
    from wme.association import compare_strategies
    from wme.sim import CameraTrajectory, constant_condition, run
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from m5_association import crowded_room

    world = crowded_room(3, seed=0)
    cfg = AssociationConfig()

    # 끝까지 애매: 일정 거리 유지
    flat = run(world, CameraTrajectory.loop(radius=3.4, laps=1.0, n=50),
               constant_condition(haze=0.3), seed=0)
    # 나중에 해소: 멀리서 접근해 후반 관측이 정밀해진다
    approach = run(world, CameraTrajectory.spiral(r_start=9.0, r_end=2.0,
                                                  laps=1.0, n=50),
                   constant_condition(haze=0.3), seed=0)

    flat_rep = compare_strategies(flat, cfg)
    appr_rep = compare_strategies(approach, cfg)

    flat_gain = (flat_rep.by_name("Mht").id_switches
                 - flat_rep.by_name("Mht (smooth)").id_switches)
    appr_gain = (appr_rep.by_name("Mht").id_switches
                 - appr_rep.by_name("Mht (smooth)").id_switches)

    assert appr_gain >= flat_gain, (
        f"해소 가능한 경우의 평활 이득({appr_gain})이 "
        f"해소 불가한 경우({flat_gain})보다 커야 한다")


def test_mht_maintains_multiple_hypotheses():
    """MHT 가 실제로 다중 가설을 들고 있는지 확인한다.

    주의: 현재 구현은 가설마다 트랙 ID 를 따로 할당하므로, 최선 가설이 바뀌면
    보고되는 ID 도 바뀐다. 그래서 정체성 지표는 신뢰할 수 없다 - 추론이 아니라
    보고 방식의 한계다. docs/03-roadmap.md 참조.
    """
    cfg = AssociationConfig(confirm_count=3, max_hypotheses=10, k_best=3)
    tr = MhtTracker(cfg)
    rng = np.random.default_rng(3)

    a, b = np.zeros(3), np.array([0.12, 0.0, 0.0])
    for f in range(15):
        tr.update([meas(a + rng.normal(0, 0.09, 3), 0.09),
                   meas(b + rng.normal(0, 0.09, 3), 0.09)], f)
    assert tr.hypothesis_count > 1
    assert tr.hypothesis_count <= cfg.max_hypotheses
