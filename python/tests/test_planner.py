"""Planner 검증.

두 가지가 핵심이다.
  - 불확실하면 더 위험해야 한다 (평균 거리만 보면 안 된다)
  - 모르는 공간을 빈 공간으로 취급하면 안 된다

둘 다 계획기의 고전적 치명상이라 대조군과 함께 못박는다.
"""

import numpy as np
import pytest

from wme.planner import Coverage, ObjectSearch, RiskConfig, RiskEstimator, SearchConfig
from wme.reference.geometry import SE3
from wme.world import (
    MemoryEngine, TokenBelief, WorldGraph, WorldSnapshot,
)


def obj(tid, name, pos, extent=(0.2, 0.2, 0.2), sigma=0.05,
        existence=0.95, static=0.95, velocity=(0, 0, 0), obs=10):
    return TokenBelief(
        token_id=tid, class_id=hash(name) % 80, class_name=name,
        position=np.asarray(pos, float),
        covariance=np.eye(3) * sigma ** 2,
        extent=np.asarray(extent, float),
        velocity=np.asarray(velocity, float),
        existence=existence, static_belief=static,
        observation_count=obs, lifecycle="active",
    )


def snap(objects, version=1, stamp=1.0):
    return WorldSnapshot(version, stamp, {o.token_id: o for o in objects})


def looking_at(eye, target):
    """target 을 바라보는 카메라 포즈."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z = z / max(np.linalg.norm(z), 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0, 0]) if n < 1e-6 else x / n
    return SE3(np.column_stack([x, np.cross(z, x), z]), eye)


# --- 충돌 확률과 불확실성 --------------------------------------------------

def test_collision_probability_is_high_when_touching():
    r = RiskEstimator()
    p = r.collision_probability(np.zeros(3), np.array([0.2, 0, 0]),
                                np.eye(3) * 0.01, np.array([0.2, 0.2, 0.2]))
    assert p > 0.9


def test_collision_probability_is_low_when_far():
    r = RiskEstimator()
    p = r.collision_probability(np.zeros(3), np.array([5.0, 0, 0]),
                                np.eye(3) * 0.01, np.array([0.2, 0.2, 0.2]))
    assert p < 0.01


def test_uncertainty_increases_risk_at_the_same_distance():
    """이 모듈의 핵심 주장.

    평균 거리가 같아도 그 위치를 잘 모르면 더 위험하다. 점추정만 쓰는
    계획기는 이걸 구분하지 못한다.
    """
    r = RiskEstimator()
    point = np.zeros(3)
    position = np.array([1.5, 0, 0])
    extent = np.array([0.2, 0.2, 0.2])

    confident = r.collision_probability(point, position, np.eye(3) * 0.01 ** 2, extent)
    vague = r.collision_probability(point, position, np.eye(3) * 0.8 ** 2, extent)

    assert vague > confident * 5.0, (confident, vague)


def test_covariance_is_projected_onto_the_approach_direction():
    """접근 방향의 불확실성만 위험하다. 옆으로 퍼진 불확실성은 덜하다."""
    r = RiskEstimator()
    point = np.zeros(3)
    position = np.array([1.2, 0, 0])
    extent = np.array([0.2, 0.2, 0.2])

    along = np.diag([0.6 ** 2, 0.01 ** 2, 0.01 ** 2])   # x 로 퍼짐 = 접근 방향
    across = np.diag([0.01 ** 2, 0.6 ** 2, 0.01 ** 2])  # y 로 퍼짐

    assert (r.collision_probability(point, position, along, extent)
            > r.collision_probability(point, position, across, extent))


def test_low_existence_lowers_risk_proportionally():
    r = RiskEstimator()
    solid = r.at(np.zeros(3), snap([obj(1, "chair", (0.5, 0, 0))]))
    ghost = r.at(np.zeros(3), snap([obj(1, "chair", (0.5, 0, 0), existence=0.2)]))
    assert ghost.total < solid.total


# --- 예측 기반 위험 --------------------------------------------------------

def test_approaching_object_is_riskier_than_receding_one():
    r = RiskEstimator()
    approaching = snap([obj(1, "person", (2.5, 0, 0), velocity=(-1.5, 0, 0), static=0.05)])
    receding = snap([obj(1, "person", (2.5, 0, 0), velocity=(1.5, 0, 0), static=0.05)])

    assert r.at(np.zeros(3), approaching).total > r.at(np.zeros(3), receding).total


def test_time_to_collision():
    r = RiskEstimator(RiskConfig(robot_radius=0.0))
    token = obj(1, "person", (3.0, 0, 0), extent=(0.0, 0.0, 0.0), velocity=(-1.0, 0, 0))
    assert r.time_to_collision(np.zeros(3), token) == pytest.approx(3.0, rel=0.05)


def test_time_to_collision_is_infinite_when_receding():
    r = RiskEstimator()
    token = obj(1, "person", (3.0, 0, 0), velocity=(1.0, 0, 0))
    assert np.isinf(r.time_to_collision(np.zeros(3), token))


def test_hazard_class_adds_risk():
    r = RiskEstimator()
    plain = r.at(np.zeros(3), snap([obj(1, "chair", (0.6, 0, 0))]))
    hazard = r.at(np.zeros(3), snap([obj(1, "car", (0.6, 0, 0))]))
    assert hazard.total > plain.total
    assert hazard.hazard > 0.0


# --- 미관측 공간 -----------------------------------------------------------

def test_unobserved_space_is_not_free_space():
    """계획기의 고전적 치명상. 못 본 곳을 자유공간으로 취급하면 안 된다."""
    cov = Coverage()
    cov.add(looking_at([0.0, 0.0, 1.0], [5.0, 0.0, 1.0]))   # +x 방향만 봤다

    r = RiskEstimator()
    empty = snap([])

    seen = r.at(np.array([3.0, 0.0, 1.0]), empty, cov)
    unseen = r.at(np.array([-3.0, 0.0, 1.0]), empty, cov)

    assert seen.unobserved < 0.4
    assert unseen.unobserved > 0.9
    assert unseen.total > seen.total


def test_coverage_respects_range_and_fov():
    cov = Coverage(RiskConfig(sensor_range=4.0, sensor_hfov=0.5))
    cov.add(looking_at([0.0, 0.0, 1.0], [5.0, 0.0, 1.0]))

    assert cov.observed(np.array([2.0, 0.0, 1.0])) > 0.2      # 정면 가까이
    assert cov.observed(np.array([9.0, 0.0, 1.0])) == 0.0     # 거리 밖
    assert cov.observed(np.array([2.0, 3.0, 1.0])) == 0.0     # 시야각 밖


def test_no_coverage_means_no_unobserved_penalty():
    """커버리지 정보를 안 주면 그 항을 지어내지 않는다."""
    r = RiskEstimator()
    assert r.at(np.zeros(3), snap([])).unobserved == 0.0


# --- 경로와 선택 -----------------------------------------------------------

def test_path_risk_takes_the_worst_point_not_the_average():
    """짧은 치명 구간이 긴 안전 구간에 희석되면 안 된다."""
    r = RiskEstimator()
    s = snap([obj(1, "chair", (5.0, 0, 0), extent=(0.3, 0.3, 0.3), sigma=0.02)])

    path = [np.array([float(x), 0.0, 0.0]) for x in range(0, 11)]
    worst = r.along(path, s)
    mean = float(np.mean([r.at(p, s).total for p in path]))

    assert worst.total > mean


def test_safest_candidate_is_chosen():
    r = RiskEstimator()
    s = snap([obj(1, "chair", (0.4, 0, 0), sigma=0.02)])
    candidates = [np.array([0.5, 0, 0]), np.array([4.0, 0, 0]), np.array([0.0, 0, 0])]

    index, assessment = r.safest(candidates, s)
    assert index == 1
    assert assessment.total < 0.2


def test_assessment_reports_contributors():
    """단일 스칼라만 주면 왜 위험한지 알 수 없다."""
    r = RiskEstimator()
    s = snap([obj(7, "chair", (0.4, 0, 0), sigma=0.02),
              obj(8, "cup", (6.0, 0, 0))])
    a = r.at(np.zeros(3), s)
    assert a.contributors and a.contributors[0][0] == 7


# --- 의미 기반 탐색 --------------------------------------------------------

def test_believed_object_ranks_first():
    search = ObjectSearch()
    s = snap([obj(1, "cup", (2.0, 0, 0.9), extent=(0.04,) * 3),
              obj(2, "dining table", (2.0, 0, 0.4), extent=(0.6, 0.4, 0.4))])

    plan = search.plan("cup", np.zeros(3), s)
    assert plan and plan[0].source == "believed"


def test_search_looks_on_tables_when_the_cup_is_not_visible():
    """사람은 컵을 찾을 때 바닥을 훑지 않는다. 테이블 위를 본다."""
    search = ObjectSearch()
    s = snap([obj(2, "dining table", (2.0, 0, 0.4), extent=(0.6, 0.4, 0.4))])

    plan = search.plan("cup", np.zeros(3), s)
    assert plan, "받침면 후보가 하나도 안 나왔다"
    top = plan[0]
    assert top.source == "support"
    # 상판 위쪽을 가리켜야 한다
    assert top.position[2] > 0.8


def test_support_candidate_is_downweighted_when_already_occupied():
    search = ObjectSearch()
    table = obj(2, "dining table", (2.0, 0, 0.4), extent=(0.6, 0.4, 0.4))
    other = obj(3, "bowl", (2.0, 0, 0.86), extent=(0.08, 0.08, 0.06))
    s = snap([table, other])

    graph = WorldGraph()
    for visit in range(4):
        graph.update(s, visit=visit)

    without = search.plan("cup", np.zeros(3), s)
    with_graph = search.plan("cup", np.zeros(3), s, graph=graph)

    def support_prob(plan):
        return next((c.probability for c in plan if c.source == "support"), 0.0)

    assert support_prob(with_graph) < support_prob(without)


def test_memory_provides_candidates_for_unseen_objects():
    """안 보이는 물체를 기억에서 찾는다. 메모리가 클래스 이름을 들고 있어야
    가능하다 - 현재 스냅샷을 뒤져 이름을 알아내는 방식이면 정작 이 경우에 못 쓴다.
    """
    search = ObjectSearch()
    memory = MemoryEngine()
    for visit in range(3):
        memory.observe(snap([obj(1, "cup", (3.0, 0, 0.9), extent=(0.04,) * 3)],
                            visit + 1, visit * 40.0), 1.0)
    memory.flush()

    gone = WorldSnapshot(9, 200.0, {})       # 지금은 아무것도 안 보인다
    plan = search.plan("cup", np.zeros(3), gone, memory=memory, now=200.0)

    remembered = [c for c in plan if c.source == "remembered"]
    assert remembered, "기억에서 후보가 안 나왔다"
    assert np.allclose(remembered[0].position, [3.0, 0, 0.9], atol=0.1)


def test_memory_candidates_decay_with_age():
    search = ObjectSearch(SearchConfig(memory_half_life=100.0))

    memory = MemoryEngine()
    memory.observe(snap([obj(1, "cup", (3.0, 0, 0.9), extent=(0.04,) * 3)], 1, 0.0), 1.0)
    memory.flush()

    def remembered_prob(now):
        gone = WorldSnapshot(2, now, {})
        plan = search.plan("cup", np.zeros(3), gone, memory=memory, now=now)
        return next((c.probability for c in plan if c.source == "remembered"), 0.0)

    assert remembered_prob(10.0) > 0.0
    assert remembered_prob(500.0) < remembered_prob(10.0)


def test_utility_prefers_close_low_risk_candidates():
    search = ObjectSearch()
    s = snap([obj(1, "cup", (1.0, 0, 0.9), extent=(0.04,) * 3),
              obj(2, "cup", (9.0, 0, 0.9), extent=(0.04,) * 3)])

    plan = search.plan("cup", np.zeros(3), s)
    assert plan[0].token_id == 1, "가깝고 안전한 쪽이 먼저여야 한다"


def test_duplicate_positions_are_not_proposed_twice():
    search = ObjectSearch()
    s = snap([obj(1, "cup", (2.0, 0, 0.9), extent=(0.04,) * 3),
              obj(2, "cup", (2.02, 0, 0.9), extent=(0.04,) * 3)])
    plan = search.plan("cup", np.zeros(3), s)
    assert len(plan) == 1


def test_search_returns_empty_for_unknown_class_without_hints():
    search = ObjectSearch()
    assert search.plan("giraffe", np.zeros(3), snap([])) == []


def test_describe_is_readable():
    search = ObjectSearch()
    s = snap([obj(1, "cup", (2.0, 0, 0.9), extent=(0.04,) * 3)])
    lines = ObjectSearch.describe(search.plan("cup", np.zeros(3), s))
    assert lines and "believed" in lines[0]
