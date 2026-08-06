"""World Graph 검증.

핵심은 관계가 참/거짓이 아니라 사후확률이라는 점, 그리고 그 확률이
위치 불확실성에서 실제로 나온다는 점이다. 확신도가 항상 1 이면 그건
그래프가 아니라 주장 모음이다.
"""

import numpy as np
import pytest

from wme.world import (
    RelationConfig, RelationInference, TokenBelief, WorldGraph, WorldSnapshot,
    can_contain, can_support,
)


def obj(tid, name, pos, extent=(0.1, 0.1, 0.1), sigma=0.02,
        existence=0.95, static=0.95, obs=10):
    return TokenBelief(
        token_id=tid, class_id=hash(name) % 80, class_name=name,
        position=np.asarray(pos, float),
        covariance=np.eye(3) * sigma ** 2,
        extent=np.asarray(extent, float),
        existence=existence, static_belief=static,
        observation_count=obs, lifecycle="active",
    )


def snap(objects, version=1, stamp=1.0):
    return WorldSnapshot(version, stamp, {o.token_id: o for o in objects})


def table(tid=1, pos=(0.0, 0.0, 0.4)):
    return obj(tid, "dining table", pos, extent=(0.6, 0.4, 0.4))


def cup_on_table(tid=2, tabletop=0.8, offset=0.0):
    # 컵 밑면이 상판에 닿도록: pos_z - e_z = tabletop + offset
    return obj(tid, "cup", (0.0, 0.0, tabletop + offset + 0.05), extent=(0.04, 0.04, 0.05))


# --- 어포던스 --------------------------------------------------------------

def test_affordance_table_is_explicit():
    assert can_support("dining table") and can_support("desk")
    assert not can_support("cup")
    assert can_contain("cup") and can_contain("refrigerator")
    assert not can_contain("dining table")


def test_non_supporting_class_never_supports():
    """받칠 수 없는 것 위에 놓였다고 말하면 안 된다."""
    inf = RelationInference()
    assert inf.on(cup_on_table(), obj(1, "cup", (0, 0, 0.4), (0.04, 0.04, 0.05))) == 0.0


# --- 'on' 의 불확실성 ------------------------------------------------------

def test_cup_resting_on_table_is_confident():
    inf = RelationInference()
    p = inf.on(cup_on_table(), table())
    assert p > 0.8, p


def test_cup_floating_far_above_is_rejected():
    inf = RelationInference()
    p = inf.on(cup_on_table(offset=0.5), table())
    assert p < 0.05, p


def test_confidence_falls_with_position_uncertainty():
    """이 모듈의 핵심 주장. 불확실하면 관계도 불확실해야 한다."""
    inf = RelationInference()
    precise = inf.on(cup_on_table(), table())

    vague_cup = obj(2, "cup", (0.0, 0.0, 0.85), extent=(0.04, 0.04, 0.05), sigma=0.4)
    vague_table = obj(1, "dining table", (0, 0, 0.4), extent=(0.6, 0.4, 0.4), sigma=0.4)
    vague = inf.on(vague_cup, vague_table)

    assert vague < precise
    assert 0.0 < vague < 0.9, vague


def test_ambiguous_gap_gives_intermediate_confidence():
    """경계에서는 확신도가 0 도 1 도 아니어야 한다. 그게 확률의 요점이다."""
    inf = RelationInference()
    p = inf.on(cup_on_table(offset=0.10), table())
    assert 0.05 < p < 0.95, p


def test_horizontal_offset_reduces_confidence():
    inf = RelationInference()
    centred = inf.on(cup_on_table(), table())
    off_edge = inf.on(obj(2, "cup", (0.9, 0.0, 0.85), extent=(0.04, 0.04, 0.05)), table())
    assert off_edge < centred


def test_low_existence_lowers_relation_confidence():
    """있는지도 확신 못 하는 물체의 관계를 확신할 수는 없다."""
    inf = RelationInference()
    solid = inf.on(cup_on_table(), table())
    ghost = inf.on(cup_on_table().with_(existence=0.3), table())
    assert ghost < solid * 0.5


# --- inside / near ---------------------------------------------------------

def test_object_inside_container():
    inf = RelationInference()
    fridge = obj(1, "refrigerator", (0, 0, 0.9), extent=(0.35, 0.35, 0.9))
    bottle = obj(2, "bottle", (0, 0, 0.9), extent=(0.04, 0.04, 0.12))
    assert inf.inside(bottle, fridge) > 0.7


def test_object_outside_container_is_rejected():
    inf = RelationInference()
    fridge = obj(1, "refrigerator", (0, 0, 0.9), extent=(0.35, 0.35, 0.9))
    bottle = obj(2, "bottle", (2.0, 0, 0.9), extent=(0.04, 0.04, 0.12))
    assert inf.inside(bottle, fridge) == 0.0


def test_near_uses_surface_distance_not_centres():
    """중심 거리를 쓰면 큰 물체가 항상 멀어진다."""
    inf = RelationInference(RelationConfig(near_distance=0.5))
    big = obj(1, "couch", (0, 0, 0.4), extent=(1.0, 0.5, 0.4))
    small = obj(2, "cup", (1.2, 0, 0.4), extent=(0.04, 0.04, 0.05))
    # 중심 거리는 1.2 m 지만 표면 거리는 0.16 m
    assert inf.near(big, small) > 0.5


def test_far_objects_are_not_near():
    inf = RelationInference(RelationConfig(near_distance=0.5))
    a = obj(1, "cup", (0, 0, 0), extent=(0.04, 0.04, 0.05))
    b = obj(2, "cup", (5, 0, 0), extent=(0.04, 0.04, 0.05))
    assert inf.near(a, b) < 0.05


# --- 그래프 누적과 철회 ----------------------------------------------------

def test_relation_strengthens_across_visits():
    g = WorldGraph()
    s = snap([table(), cup_on_table()])
    first = None
    for visit in range(5):
        g.update(s, visit=visit)
        c = g.confidence(2, "on", 1)
        if first is None:
            first = c
        assert c >= first
    assert g.confidence(2, "on", 1) > first


def test_repeated_frames_in_one_visit_count_once():
    """한 번 지나가며 본 여러 프레임이 관계를 여러 번 확인한 것이 되면 안 된다.

    이 프로젝트에서 네 번 물린 상관 오류다.
    """
    s = snap([table(), cup_on_table()])

    single = WorldGraph()
    single.update(s, visit=0)

    repeated = WorldGraph()
    for _ in range(50):
        repeated.update(s, visit=0)          # 같은 방문

    assert repeated.confidence(2, "on", 1) == pytest.approx(
        single.confidence(2, "on", 1), abs=1e-12)


def test_relation_is_retracted_when_the_cup_is_removed_from_the_table():
    """누적만 하는 그래프는 세계가 아니라 기록이다."""
    g = WorldGraph()
    on_table = snap([table(), cup_on_table()])
    for visit in range(6):
        g.update(on_table, visit=visit)
    assert g.confidence(2, "on", 1) > 0.6

    # 컵을 바닥으로 옮긴다. 둘 다 여전히 보이므로 철회 증거가 된다.
    moved = snap([table(), obj(2, "cup", (1.5, 0, 0.05), extent=(0.04, 0.04, 0.05))])
    for visit in range(6, 12):
        g.update(moved, visit=visit)

    assert g.confidence(2, "on", 1) < 0.35, g.relations()


def test_absence_is_not_retraction_evidence():
    """안 보인 것은 반증이 아니다. 그냥 두어야 한다."""
    g = WorldGraph()
    on_table = snap([table(), cup_on_table()])
    for visit in range(6):
        g.update(on_table, visit=visit)
    before = g.confidence(2, "on", 1)

    # 컵이 스냅샷에서 사라진다 (시야 밖). 테이블만 보인다.
    only_table = snap([table()])
    for visit in range(6, 12):
        g.update(only_table, visit=visit)

    assert g.confidence(2, "on", 1) == pytest.approx(before, abs=1e-12)


# --- 질의 ------------------------------------------------------------------

def test_queries_are_consistent():
    g = WorldGraph()
    s = snap([table(), cup_on_table(2), cup_on_table(3, offset=0.0)])
    for visit in range(4):
        g.update(s, visit=visit)

    on_table = {r.subject for r in g.objects_on(1)}
    assert 2 in on_table and 3 in on_table
    assert {r.obj for r in g.supported_by(2)} == {1}


def test_regions_are_connected_components_of_near():
    cfg = RelationConfig(near_distance=0.5)
    g = WorldGraph(cfg)
    s = snap([
        obj(1, "cup", (0.0, 0, 0), extent=(0.05,) * 3),
        obj(2, "cup", (0.3, 0, 0), extent=(0.05,) * 3),
        obj(10, "cup", (9.0, 0, 0), extent=(0.05,) * 3),
        obj(11, "cup", (9.3, 0, 0), extent=(0.05,) * 3),
    ])
    for visit in range(4):
        g.update(s, visit=visit)

    regions = g.regions()
    assert [1, 2] in regions and [10, 11] in regions


def test_relations_are_deterministically_ordered():
    """해시 순서가 결과에 새면 재현이 불가능해진다."""
    g = WorldGraph()
    s = snap([table(), cup_on_table(2), cup_on_table(3)])
    for visit in range(3):
        g.update(s, visit=visit)
    assert [(r.predicate, r.subject, r.obj) for r in g.relations()] == \
           [(r.predicate, r.subject, r.obj) for r in g.relations()]


def test_describe_is_readable():
    g = WorldGraph()
    s = snap([table(), cup_on_table()])
    for visit in range(4):
        g.update(s, visit=visit)
    lines = g.describe(s)
    assert any("cup#2 on dining table#1" in line for line in lines), lines


def test_empty_snapshot_is_safe():
    g = WorldGraph()
    g.update(snap([]), visit=0)
    assert g.relations() == []
    assert g.regions() == []
