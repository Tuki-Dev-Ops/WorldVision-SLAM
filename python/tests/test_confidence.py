"""신뢰도 엔진 검증. C++ tests/test_confidence.cpp 와 같은 성질을 확인한다."""

import math

import pytest

from wme.reference.confidence import Beliefs, ConfidenceEngine, to_logodds, to_probability


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_logodds_round_trip(p):
    assert to_probability(to_logodds(p)) == pytest.approx(p, abs=1e-9)


def test_logodds_saturates_safely():
    """0/1 에서 발산하면 그 믿음은 영원히 갱신 불가가 된다."""
    assert math.isfinite(to_logodds(0.0))
    assert math.isfinite(to_logodds(1.0))


def test_repeated_observations_increase_existence():
    e = ConfidenceEngine()
    b = Beliefs()
    prev = b.existence
    for _ in range(8):
        e.on_observed(b, 0.9, 1.0)
        assert b.existence >= prev
        prev = b.existence
    assert b.existence > 0.9


def test_existence_never_reaches_certainty_and_can_recover():
    """포화하면 반증을 못 받는다. '지도를 수리한다'는 원칙의 전제."""
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(500):
        e.on_observed(b, 1.0, 1.0)
    assert b.existence < 1.0

    high = b.existence
    for _ in range(200):
        e.on_missed(b, 1.0, 1.0)
    assert b.existence < high
    assert b.existence < 0.3


def test_miss_while_occluded_is_not_evidence_of_absence():
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(5):
        e.on_observed(b, 0.9, 1.0)
    before = b.existence

    for _ in range(30):
        e.on_missed(b, 0.05, 1.0)       # 거의 완전히 가려짐
    assert b.existence == before
    assert b.miss_count == 30


def test_low_reliability_weakens_evidence():
    """안개 속 관측도 방향은 같지만 정보량이 작아야 한다."""
    e = ConfidenceEngine()
    clear, foggy = Beliefs(), Beliefs()
    for _ in range(5):
        e.on_observed(clear, 0.9, 1.0)
        e.on_observed(foggy, 0.9, 0.3)
    assert clear.existence > foggy.existence
    assert foggy.existence > 0.5


def test_low_detection_confidence_gives_weaker_evidence():
    e = ConfidenceEngine()
    strong, weak = Beliefs(), Beliefs()
    for _ in range(5):
        e.on_observed(strong, 0.95, 1.0)
        e.on_observed(weak, 0.30, 1.0)
    assert strong.existence > weak.existence


def test_ambiguous_association_weakens_identity():
    e = ConfidenceEngine()
    distinct, ambiguous = Beliefs(), Beliefs()
    for _ in range(6):
        e.on_observed(distinct, 0.9, 1.0, assoc_margin=8.0)
        e.on_observed(ambiguous, 0.9, 1.0, assoc_margin=0.02)
    assert distinct.identity > ambiguous.identity


def test_identity_decays_out_of_view_but_existence_does_not():
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(6):
        e.on_observed(b, 0.9, 1.0, assoc_margin=5.0)
    e0, i0 = b.existence, b.identity

    e.on_out_of_view(b, 60.0)
    assert b.existence == e0, "안 보인다는 것은 없다는 뜻이 아니다"
    assert b.identity < i0
    assert b.identity > 0.5, "0.5 로 수렴해야지 뒤집히면 안 된다"


W = 0.5     # 판정 창. static_min_dt 기본값


def test_stationary_object_becomes_static():
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(20):
        e.update_static(b, 0.002, W, 1.0)
    assert b.static > 0.8


def test_moving_object_becomes_dynamic():
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(20):
        e.update_static(b, 0.35, W, 1.0)        # 0.7 m/s = 보행 속도
    assert b.static < 0.2


def test_evidence_is_a_likelihood_ratio_not_a_bounded_heuristic():
    """세기가 상수에 묶이면 채널은 토큰 수명만 세게 된다 (16.1).

    이전 구현은 증거가 [-0.31, +0.38] 로 유계여서 결정적인 관측과 애매한
    관측을 구분하지 못했다. 우도비는 관측이 결정적일수록 커져야 한다.
    """
    e = ConfidenceEngine()
    ratios = []
    for d in (0.0, 0.2, 0.5, 1.5):
        b = Beliefs()
        e.update_static(b, d, W, 1.0)
        ratios.append(b.static_diag.ratio)

    assert ratios[0] > 0.0, "안 움직이면 정적의 증거"
    assert ratios[1] < 0.0 < ratios[0], "움직이면 부호가 뒤집혀야 한다"
    assert ratios[2] < ratios[1], "더 크게 움직이면 더 강한 동적 증거"
    # 이전 유계 형태에서는 불가능했던 크기
    assert min(ratios) < -0.31


def test_outlier_cannot_erase_accumulated_evidence():
    """이상치 하나가 결론을 뒤집을 수는 있어도 지울 수는 없어야 한다 (16.4)."""
    e = ConfidenceEngine()
    b = Beliefs()
    for _ in range(20):
        e.update_static(b, 0.002, W, 1.0)
    before = b.static

    e.update_static(b, 3.0, W, 1.0)             # YOLO 박스 중심이 튄 경우
    assert b.static < before
    assert b.static > 0.2, "10 초치 증거가 한 번에 지워지면 안 된다"

    # 꼬리를 끄면 같은 관측이 채널을 바닥에 박아 놓는다
    from wme.reference.confidence import ConfidenceConfig
    e2 = ConfidenceEngine(ConfidenceConfig(static_outlier_rate=0.0))
    b2 = Beliefs()
    for _ in range(20):
        e2.update_static(b2, 0.002, W, 1.0)
    e2.update_static(b2, 3.0, W, 1.0)
    # 실측 -453 nat 과 같은 종류의 값. 한 번에 포화 하한에 박힌다.
    assert b2.static_diag.ratio < -100.0
    assert b2.static == pytest.approx(to_probability(-4.0), abs=1e-12)


def test_stopped_agent_can_be_believed_static_while_observed():
    """앉아 있는 사람은 벽만큼 좋은 랜드마크다.

    상한을 씌우면 채널이 통째로 죽는다 - 실측 대가는 fr3_sitting ATE
    1.01 -> 15.84 cm. 요구사항은 상한이 아니라 유효기간으로 다룬다.
    """
    e = ConfidenceEngine()
    b = Beliefs(is_agent=True)
    for _ in range(30):
        e.update_static(b, 0.002, W, 1.0)
    assert b.static > 0.7

    # 다만 같은 증거로 비-자율 개체보다 느리게 올라야 한다 (회의적 이득)
    slow, fast = Beliefs(is_agent=True), Beliefs()
    e.update_static(slow, 0.002, W, 1.0)
    e.update_static(fast, 0.002, W, 1.0)
    assert slow.static < fast.static


def test_agent_static_claim_expires_without_observation():
    """신호등 앞의 차는 언젠가 출발한다. 관측이 끊기면 주장이 사전분포로 돌아간다."""
    e = ConfidenceEngine()
    agent = Beliefs(is_agent=True, static=0.9, static_prior=0.5)
    desk = Beliefs(is_agent=False, static=0.9, static_prior=0.5)

    e.decay_static_belief(agent, 4.0)           # 반감기 1회
    e.decay_static_belief(desk, 4.0)

    assert agent.static == pytest.approx(0.75, abs=1e-12)
    assert desk.static == 0.9, "책상은 안 보이는 동안 걸어다니지 않는다"


def test_static_judgement_scales_with_measurement_uncertainty():
    """불확실한 *관측* 은 같은 변위로 동적이라 단정하면 안 된다.

    척도는 융합 추정의 불확실성이 아니라 관측의 것이다 - 융합 쪽은 관측 수에
    따라 줄어드는 반면 창 변위는 그렇지 않다 (16.3, 10.2 의 여섯 번째 반복).
    """
    e = ConfidenceEngine()
    precise = Beliefs(meas_sigma=0.01)
    vague = Beliefs(meas_sigma=0.5)
    for _ in range(10):
        e.update_static(precise, 0.15, W, 1.0)
        e.update_static(vague, 0.15, W, 1.0)
    assert precise.static < vague.static


def test_short_interval_is_not_judged():
    """0.05 s 창의 실측 분리비는 1.20 - 사실상 구분이 없다 (16.2)."""
    e = ConfidenceEngine()
    b = Beliefs()
    e.update_static(b, 1.0, 0.05, 1.0)
    assert b.static == 0.5
    assert b.static_diag.updates == 0


def test_merge_combines_evidence():
    e = ConfidenceEngine()
    a, c = Beliefs(), Beliefs()
    for _ in range(4):
        e.on_observed(a, 0.85, 1.0)
        e.on_observed(c, 0.85, 1.0)
    before = a.existence

    e.merge(a, c)
    assert a.existence >= before
    assert a.existence < 1.0
