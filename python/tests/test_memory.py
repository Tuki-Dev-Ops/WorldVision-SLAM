"""Memory Engine 검증.

가장 중요한 것은 증거 단위가 프레임이 아니라 에피소드라는 성질이다.
이 프로젝트에서 같은 상관 오류가 이미 세 번 나왔으므로(객체 융합, 측광 잔차,
포즈 체인), 메모리에서는 그것이 성립하는지를 대조군과 함께 못박는다.
"""

import numpy as np
import pytest

from wme.world import MemoryConfig, MemoryEngine, TokenBelief, WorldSnapshot


def belief(tid, pos, existence=0.9, static=0.9, sigma=0.05):
    return TokenBelief(
        token_id=tid, class_id=56, class_name="chair",
        position=np.asarray(pos, float),
        covariance=np.eye(3) * sigma ** 2,
        existence=existence, static_belief=static,
        observation_count=10, lifecycle="active",
    )


def snap(beliefs, version, stamp):
    return WorldSnapshot(version, stamp, {b.token_id: b for b in beliefs})


def feed(engine, positions, stamps, tid=1, reliability=1.0, **kw):
    for v, (p, t) in enumerate(zip(positions, stamps), start=1):
        engine.observe(snap([belief(tid, p, **kw)], v, t), reliability)


# --- 에피소드 분할 ---------------------------------------------------------

def test_contiguous_frames_form_one_episode():
    e = MemoryEngine(MemoryConfig(episode_gap=5.0))
    feed(e, [[0, 0, 0]] * 20, np.arange(20) * 0.1)
    e.flush()

    mem = e.get(1)
    assert mem.visits == 1
    assert mem.episodes[0].frames == 20


def test_time_gap_starts_a_new_episode():
    e = MemoryEngine(MemoryConfig(episode_gap=5.0))
    stamps = list(np.arange(10) * 0.1) + list(30.0 + np.arange(10) * 0.1)
    feed(e, [[0, 0, 0]] * 20, stamps)
    e.flush()

    assert e.get(1).visits == 2


def test_episode_records_its_own_spread():
    e = MemoryEngine()
    rng = np.random.default_rng(3)
    positions = [np.array([0.0, 0, 0]) + rng.normal(0, 0.1, 3) for _ in range(30)]
    feed(e, positions, np.arange(30) * 0.1)
    e.flush()

    assert 0.05 < e.get(1).episodes[0].spread < 0.3


# --- 이 모듈의 핵심 주장 ---------------------------------------------------

def test_visits_not_frames_determine_confidence():
    """한 번 오래 본 것이 여러 번 다시 가서 본 것보다 확신이 높으면 안 된다.

    프레임 단위로 증거를 세면 정확히 그 잘못된 결론이 나온다. 이 프로젝트에서
    같은 상관 오류가 세 번 반복됐으므로 여기서 명시적으로 막는다.
    """
    long_stare = MemoryEngine()
    feed(long_stare, [[0, 0, 0]] * 200, np.arange(200) * 0.05)     # 1회 방문, 200 프레임
    long_stare.flush()

    revisits = MemoryEngine()
    stamps, positions = [], []
    for visit in range(5):                                          # 5회 방문, 총 25 프레임
        base = visit * 40.0
        stamps.extend(base + np.arange(5) * 0.1)
        positions.extend([[0, 0, 0]] * 5)
    feed(revisits, positions, stamps)
    revisits.flush()

    assert long_stare.get(1).episodes[0].frames == 200
    assert sum(ep.frames for ep in revisits.get(1).episodes) == 25

    assert revisits.get(1).persistent_existence > long_stare.get(1).persistent_existence, (
        f"재방문 {revisits.get(1).persistent_existence:.3f} vs "
        f"장시간응시 {long_stare.get(1).persistent_existence:.3f}")


def test_frame_count_alone_does_not_increase_confidence():
    """같은 1회 방문이면 프레임이 10배여도 확신이 거의 같아야 한다."""
    short = MemoryEngine()
    feed(short, [[0, 0, 0]] * 10, np.arange(10) * 0.1)
    short.flush()

    long_ = MemoryEngine()
    feed(long_, [[0, 0, 0]] * 100, np.arange(100) * 0.1)
    long_.flush()

    assert long_.get(1).persistent_existence == pytest.approx(
        short.get(1).persistent_existence, abs=1e-9)


def test_between_visit_disagreement_widens_uncertainty():
    """방문마다 위치가 다르면, 각 방문이 아무리 정밀했어도 불확실해야 한다."""
    consistent = MemoryEngine()
    agreeing = MemoryEngine()

    for visit in range(4):
        base = visit * 40.0
        consistent.observe(snap([belief(1, [0, 0, 0])], visit + 1, base), 1.0)
        agreeing.observe(snap([belief(1, [0.4 * visit, 0, 0])], visit + 1, base), 1.0)
    consistent.flush()
    agreeing.flush()

    tight = np.trace(consistent.get(1).persistent_covariance)
    loose = np.trace(agreeing.get(1).persistent_covariance)
    assert loose > tight * 4.0


def test_low_reliability_episodes_carry_less_evidence():
    clear = MemoryEngine()
    foggy = MemoryEngine()
    for visit in range(4):
        t = visit * 40.0
        clear.observe(snap([belief(1, [0, 0, 0])], visit + 1, t), 1.0)
        foggy.observe(snap([belief(1, [0, 0, 0])], visit + 1, t), 0.2)
    clear.flush()
    foggy.flush()

    assert clear.get(1).persistent_existence > foggy.get(1).persistent_existence


def test_consolidated_existence_saturates():
    """무한히 확신하면 반증을 못 받는다. manifesto 의 '수리한다' 원칙."""
    e = MemoryEngine()
    for visit in range(200):
        e.observe(snap([belief(1, [0, 0, 0])], visit + 1, visit * 40.0), 1.0)
    e.flush()
    assert e.get(1).persistent_existence < 1.0


# --- 망각 ------------------------------------------------------------------

def test_memory_is_bounded():
    cfg = MemoryConfig(max_episodes_per_object=5)
    e = MemoryEngine(cfg)
    for visit in range(30):
        e.observe(snap([belief(1, [0, 0, 0])], visit + 1, visit * 40.0), 1.0)
    e.flush()
    e.forget(now=30 * 40.0)

    assert e.get(1).visits <= 5


def test_forgetting_keeps_surprising_episodes_over_redundant_ones():
    """놀라웠던 관측을 먼저 버리면 세계가 변한 기록이 사라진다."""
    cfg = MemoryConfig(max_episodes_per_object=3)
    e = MemoryEngine(cfg)

    # 같은 자리에서 여러 번 (중복), 중간에 한 번 크게 다른 자리 (놀라움)
    for visit in range(8):
        pos = [5.0, 0, 0] if visit == 4 else [0.0, 0, 0]
        e.observe(snap([belief(1, pos)], visit + 1, visit * 40.0), 1.0)
    e.flush()
    e.forget(now=8 * 40.0)

    kept = [float(ep.position[0]) for ep in e.get(1).episodes]
    assert any(x > 1.0 for x in kept), f"놀라웠던 에피소드가 버려졌다: {kept}"


def test_adverse_conditions_extend_retention():
    """조건이 나빴다면 관측이 희소했다는 뜻이다. 같은 기준으로 버리면 정보가 과하게 준다."""
    def run(scale):
        cfg = MemoryConfig(max_episodes_per_object=4, base_retention=30.0)
        e = MemoryEngine(cfg)
        for visit in range(12):
            e.observe(snap([belief(1, [0, 0, 0])], visit + 1, visit * 40.0), 1.0)
        e.flush()
        e.forget(now=12 * 40.0, environment_scale=scale)
        return min(ep.end for ep in e.get(1).episodes)

    # 보존 배율이 크면 더 오래된 에피소드까지 남는다
    assert run(8.0) <= run(1.0)


# --- 질의 ------------------------------------------------------------------

def test_was_present_distinguishes_absent_from_unknown():
    """'없었다' 와 '모른다' 를 섞으면 세계 모델이 아니다."""
    e = MemoryEngine(MemoryConfig(episode_gap=5.0))
    stamps = list(np.arange(10) * 0.1) + list(60.0 + np.arange(10) * 0.1)
    feed(e, [[0, 0, 0]] * 20, stamps)
    e.flush()

    assert e.was_present(1, 0.5) is True          # 첫 방문 중
    assert e.was_present(1, 30.0) is False        # 방문 사이 - 관측되지 않았다
    assert e.was_present(1, 500.0) is None        # 기억의 범위 밖 - 모른다
    assert e.was_present(999, 1.0) is None        # 그런 객체를 모른다


def test_position_at_does_not_invent_values():
    e = MemoryEngine(MemoryConfig(episode_gap=5.0))
    feed(e, [[1.0, 0, 0]] * 10, np.arange(10) * 0.1)
    e.flush()

    assert np.allclose(e.get(1).position_at(0.5), [1.0, 0, 0])
    assert e.get(1).position_at(100.0) is None    # 지어내지 않는다


def test_environment_history_explains_why_it_was_missed():
    e = MemoryEngine()
    for i in range(10):
        e.observe(snap([belief(1, [0, 0, 0])], i + 1, float(i)), 1.0 if i < 5 else 0.2)
    e.flush()

    assert e.reliability_at(2.0) == pytest.approx(1.0)
    assert e.reliability_at(8.0) == pytest.approx(0.2)
    assert e.reliability_at(-1.0) is None


def test_stable_objects_require_multiple_visits():
    """한 번 스쳐 본 것은 장소를 정의할 자격이 없다."""
    e = MemoryEngine()
    feed(e, [[0, 0, 0]] * 50, np.arange(50) * 0.1, tid=1)          # 1회 방문
    for visit in range(4):                                          # 4회 방문
        e.observe(snap([belief(2, [3, 0, 0])], visit + 1, visit * 40.0), 1.0)
    e.flush()

    stable = {m.object_id for m in e.stable_objects(min_visits=2)}
    assert stable == {2}


def test_engine_handles_empty_snapshots():
    e = MemoryEngine()
    e.observe(snap([], 1, 0.0), 1.0)
    e.flush()
    assert len(e) == 0
    assert e.forget(now=10.0) == 0
