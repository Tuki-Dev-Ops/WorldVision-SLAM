"""차등 테스트: C++ 엔진 vs Python 참조 구현.

이 파일이 이 저장소에서 가장 중요한 테스트다.

SLAM 은 오차가 조용히 누적되는 시스템이라, 단위 테스트가 전부 통과해도
부호 하나가 뒤집힌 채 "그럴듯한" 결과를 내며 굴러갈 수 있다. 같은 알고리즘을
두 번 독립적으로 구현해 같은 입력에 같은 답을 요구하면 그런 오류가 드러난다.

두 구현이 다르면 둘 중 하나가 틀린 것이다. 어느 쪽인지는 세 번째 근거
(해석적 정답, 합성 ground truth)로 가린다.

`_core` 확장을 빌드하지 않았으면 전부 skip 된다:
    cmake -S . -B build -DWME_BUILD_PYTHON=ON && cmake --build build
"""

import numpy as np
import pytest

import os

from wme import CORE_IMPORT_ERROR, HAS_NATIVE, core
from wme.reference import assignment as ref_assign
from wme.reference import confidence as ref_conf
from wme.reference import constellation as ref_const
from wme.reference import environment as ref_env
from wme.reference import geometry as ref_geom
from wme.reference import tokens as ref_tok

# skip 사유에 *원인* 을 적는다. "미빌드" 만 적혀 있으면 41 개 skip 이 초록으로
# 지나간다 - 실제로 그랬다 (docs/06-results.md 19장). 이제 skip 한 줄에
# ImportError 원문이 남고, WME_REQUIRE_NATIVE 를 켜면 skip 자체가 사라진다
# (모듈이 없으면 임포트에서 터진다). CI 는 그 변수를 켜고 돌려야 한다.
if not HAS_NATIVE and os.environ.get("WME_REQUIRE_NATIVE"):
    raise RuntimeError(f"WME_REQUIRE_NATIVE 인데 _core 가 없다: {CORE_IMPORT_ERROR!r}")

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE,
    reason=("C++ 확장(_core) 미빌드 - cmake -DWME_BUILD_PYTHON=ON 후 재실행. "
            f"원인: {CORE_IMPORT_ERROR!r}"),
)

RNG = np.random.default_rng(777)


# --- 리군 -----------------------------------------------------------------

def test_se3_exp_matches():
    for _ in range(200):
        xi = np.concatenate([RNG.uniform(-3, 3, 3), RNG.uniform(-2, 2, 3)])
        assert np.allclose(core.SE3.exp(xi).matrix(), ref_geom.SE3.exp(xi).matrix(), rtol=0, atol=1e-12)


def test_se3_log_matches():
    for _ in range(200):
        xi = np.concatenate([RNG.uniform(-3, 3, 3), RNG.uniform(-2, 2, 3)])
        if np.linalg.norm(xi[3:]) > np.pi * 0.95:
            continue
        assert np.allclose(core.SE3.exp(xi).log(), ref_geom.SE3.exp(xi).log(), atol=1e-9)


def test_se3_adjoint_matches():
    for _ in range(100):
        xi = np.concatenate([RNG.uniform(-3, 3, 3), RNG.uniform(-1.5, 1.5, 3)])
        assert np.allclose(core.SE3.exp(xi).adjoint(),
                           ref_geom.SE3.exp(xi).adjoint(), atol=1e-10)


def test_so3_left_jacobian_matches():
    for _ in range(100):
        phi = RNG.uniform(-2.5, 2.5, 3)
        assert np.allclose(core.SO3.left_jacobian(phi),
                           ref_geom.so3_left_jacobian(phi), atol=1e-10)


def test_kabsch_matches():
    for _ in range(100):
        src = RNG.uniform(-5, 5, (12, 3))
        T = ref_geom.SE3.exp(np.concatenate([RNG.uniform(-3, 3, 3), RNG.uniform(-1.5, 1.5, 3)]))
        dst = (T @ src) + RNG.normal(0, 0.01, (12, 3))

        native = core.kabsch(list(src), list(dst))
        reference = ref_geom.kabsch(src, dst)
        assert np.allclose(native.matrix(), reference.matrix(), atol=1e-9)


# --- 할당 -----------------------------------------------------------------

@pytest.mark.parametrize("shape", [(4, 4), (3, 7), (9, 5), (16, 16)])
def test_assignment_total_cost_matches(shape):
    """최적해가 여럿일 수 있으므로 배정 자체가 아니라 총비용을 비교한다."""
    for _ in range(40):
        cost = RNG.uniform(0, 20, shape)
        _, _, native_total = core.solve_assignment(cost)
        _, _, ref_total = ref_assign.solve_assignment(cost)
        assert np.isclose(native_total, ref_total, atol=1e-9)


def test_assignment_infeasible_handling_matches():
    cost = RNG.uniform(0, 5, (6, 6))
    cost[cost > 3.0] = core.INFEASIBLE

    n_rows, _, n_total = core.solve_assignment(cost)
    r_rows, _, r_total = ref_assign.solve_assignment(cost)

    assert np.isclose(n_total, r_total, atol=1e-9)
    # 금지 쌍은 양쪽 모두 절대 배정하지 않아야 한다
    for rows in (n_rows, r_rows):
        for i, j in enumerate(rows):
            if j >= 0:
                assert cost[i, j] < core.INFEASIBLE


# --- 신뢰도 ---------------------------------------------------------------

def test_logodds_conversion_matches():
    for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        assert np.isclose(core.to_logodds(p), ref_conf.to_logodds(p), rtol=0, atol=1e-12)
        l = ref_conf.to_logodds(p)
        assert np.isclose(core.to_probability(l), ref_conf.to_probability(l), rtol=0, atol=1e-12)


# 차등 비교는 rtol 을 0 으로 못박는다.
#
# np.isclose 의 기본 rtol 은 1e-5 다. atol=1e-12 만 적어 두면 "1e-12 로 비교"
# 처럼 읽히지만 실제 허용치는 상대 1e-5 이고, 그 폭 안에는 진짜 불일치가
# 얼마든지 들어간다. 실제로 detection_conf 의 float 저장(2.3e-9)이 그 틈에
# 숨어 있었다.
def _same(a, b, atol=1e-12):
    return np.isclose(a, b, rtol=0.0, atol=atol)


@pytest.mark.parametrize("reliability", [1.0, 0.6, 0.3])
@pytest.mark.parametrize("det_conf", [0.95, 0.6, 0.3])
def test_existence_belief_trajectory_matches(reliability, det_conf):
    """관측을 반복하며 매 스텝의 믿음이 일치해야 한다. 누적 오차가 드러난다."""
    tok = core.WorldToken()
    tok.existence_belief = 0.5
    tok.identity_belief = 0.5
    tok.static_belief = 0.5

    engine_n = core.ConfidenceEngine()
    engine_r = ref_conf.ConfidenceEngine()
    beliefs = ref_conf.Beliefs()

    for step in range(20):
        engine_n.on_observed(tok, det_conf, reliability)
        engine_r.on_observed(beliefs, det_conf, reliability, obs_reliability=1.0)
        assert _same(tok.existence_belief, beliefs.existence), f"step {step}"


@pytest.mark.parametrize("env_rel,obs_rel", [(1.0, 0.4), (0.4, 1.0), (0.8, 0.5),
                                             (0.5, 0.8), (0.6, 0.6)])
def test_observed_reliability_is_two_factor(env_rel, obs_rel):
    """C++ 은 env 신뢰도와 관측 신뢰도를 *곱한다*. 두 인자가 따로 있어야 잰다.

    예전 바인딩은 파이썬 인자 하나를 두 자리에 다 넣어 신뢰도를 제곱했고,
    참조 구현도 같은 값을 두 번 받도록 맞춰져 있었다. 두 인자가 항상 같으면
    r*r 인지 r 인지 구분할 수 없다 - 측정이 아니었다 (docs 10.4).
    여기서는 env != obs 인 조합을 넣어 곱셈 구조 자체를 잰다. 비대칭 두 쌍
    (1.0, 0.4) 과 (0.4, 1.0) 이 같은 답을 내야 곱이 맞다.
    """
    tok = core.WorldToken()
    tok.existence_belief = 0.5
    tok.identity_belief = 0.5
    b = ref_conf.Beliefs()

    engine_n, engine_r = core.ConfidenceEngine(), ref_conf.ConfidenceEngine()
    for step in range(10):
        engine_n.on_observed(tok, 0.9, env_rel, obs_sensor_reliability=obs_rel)
        engine_r.on_observed(b, 0.9, env_rel, obs_reliability=obs_rel)
        assert _same(tok.existence_belief, b.existence), (
            f"step {step}: env={env_rel} obs={obs_rel} "
            f"native {tok.existence_belief!r} vs reference {b.existence!r}")
        assert _same(tok.identity_belief, b.identity), f"identity step {step}"

    # 제곱 구현이면 (1.0, 0.4) 와 (0.4, 0.4) 가 같아진다. 그 둘이 갈리는지 본다.
    sq = core.WorldToken()
    sq.existence_belief = 0.5
    for _ in range(10):
        engine_n.on_observed(sq, 0.9, obs_rel, obs_sensor_reliability=obs_rel)
    if env_rel != obs_rel:
        assert not _same(tok.existence_belief, sq.existence_belief, atol=1e-9), (
            "제곱 구현과 구분되지 않는 입력 - 이 파라미터는 아무것도 재지 못한다")


def test_detection_conf_is_stored_as_float():
    """C++ Observation::detection_conf 는 float 다. 참조도 같이 잘라야 한다.

    double 을 그대로 쓰면 두 구현이 최대 2.3e-9 갈린다. 그 차이는 기존
    np.isclose(rtol=0, atol=1e-12) 가 기본 rtol=1e-5 를 함께 쓰는 바람에 46 개 테스트
    전부를 통과했다. rtol=0 으로 재면 즉시 드러난다.
    """
    engine_n, engine_r = core.ConfidenceEngine(), ref_conf.ConfidenceEngine()
    # float 로 정확히 표현되지 않는 값만 고른다. 0.5 / 0.25 는 정확해서
    # 이 테스트가 아무것도 재지 못한다.
    for dc in (0.95, 0.6, 0.3, 0.1, 0.77):
        assert float(np.float32(dc)) != dc, f"{dc} 는 float 로 정확해 측정이 안 된다"
        tok = core.WorldToken()
        tok.existence_belief = 0.5
        b = ref_conf.Beliefs()
        engine_n.on_observed(tok, dc, 1.0)
        engine_r.on_observed(b, dc, 1.0, obs_reliability=1.0)
        assert _same(tok.existence_belief, b.existence), (
            f"det_conf={dc}: native {tok.existence_belief!r} vs "
            f"reference {b.existence!r}")


def test_merge_beliefs_matches():
    """루프 클로저 병합. existence/static 은 로그오즈 합, identity 는 곱이다.

    세 채널이 서로 다른 규칙을 쓰므로 하나만 비교하면 나머지 둘이 무방비다.
    포화 한계(logodds_max=4)에 실제로 닿는 조합을 반드시 포함한다 - 안 그러면
    clamp 가 한 번도 실행되지 않는다.
    """
    engine_n, engine_r = core.ConfidenceEngine(), ref_conf.ConfidenceEngine()
    cases = [(0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
             (0.99, 0.99, 0.99, 0.99, 0.99, 0.99),   # 포화 상한
             (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),   # 포화 하한
             (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
             (0.95, 0.10, 0.30, 0.80, 0.60, 0.20)]

    saturated = False
    for ke, ki, ks, ae, ai, asx in cases:
        keep = core.WorldToken()
        keep.existence_belief, keep.identity_belief, keep.static_belief = ke, ki, ks
        absorbed = core.WorldToken()
        absorbed.existence_belief, absorbed.identity_belief, absorbed.static_belief = ae, ai, asx

        rk = ref_conf.Beliefs(existence=ke, identity=ki, static=ks)
        ra = ref_conf.Beliefs(existence=ae, identity=ai, static=asx)

        engine_n.merge_beliefs(keep, absorbed)
        engine_r.merge(rk, ra)

        assert _same(keep.existence_belief, rk.existence), (ke, ae)
        assert _same(keep.identity_belief, rk.identity), (ki, ai)
        assert _same(keep.static_belief, rk.static), (ks, asx)
        if abs(ref_conf.to_logodds(rk.existence) - 4.0) < 1e-9:
            saturated = True
    assert saturated, "포화 한계에 닿는 조합이 없으면 clamp 경로가 안 돈다"


def test_miss_penalty_matches():
    tok = core.WorldToken()
    tok.existence_belief = 0.9
    beliefs = ref_conf.Beliefs(existence=0.9)

    engine_n = core.ConfidenceEngine()
    engine_r = ref_conf.ConfidenceEngine()

    for visibility in [1.0, 0.8, 0.5, 0.2, 0.05]:
        for _ in range(5):
            engine_n.on_missed(tok, visibility, 1.0)
            engine_r.on_missed(beliefs, visibility, 1.0)
        assert np.isclose(tok.existence_belief, beliefs.existence, rtol=0, atol=1e-12)


def _meas_sigma_supported() -> bool:
    """바인딩이 meas_sigma 를 노출하는가. 없으면 정적 판정을 비교할 수 없다."""
    return hasattr(core.WorldToken(), "meas_sigma")


# 판정 창. C++ static_min_dt 기본값(0.50 s)과 같아야 갱신이 일어난다.
STATIC_W = 0.5


@pytest.mark.skipif(not HAS_NATIVE or not _meas_sigma_supported(),
                    reason="바인딩이 WorldToken.meas_sigma 를 노출하지 않음")
@pytest.mark.parametrize("meas_sigma", [0.0, 0.02, 0.10])
def test_static_belief_matches(meas_sigma):
    """정적/동적 증거는 유계 휴리스틱이 아니라 3-DOF 가우시안 로그우도비다.

    이전 형태는 [-0.31, +0.38] 로 유계여서 sitting 과 walking 모두 갱신의 84 %
    가 양의 증거였다 - 채널이 토큰 수명만 셌다 (docs/06-results.md 16.1).
    여기서 비교하는 것은 그 교체가 양쪽에서 같은 수를 내는가다. 유계 형태는
    같은 입력에서 크기가 수십 배 다르므로 이 테스트로 즉시 갈린다.
    """
    tok = core.WorldToken()
    tok.static_belief = 0.5
    tok.meas_sigma = meas_sigma
    beliefs = ref_conf.Beliefs(meas_sigma=meas_sigma)

    engine_n = core.ConfidenceEngine()
    engine_r = ref_conf.ConfidenceEngine()

    # 0 변위(정적 증거의 상한)부터 3 m 이상치(꼬리 하한)까지 훑는다
    for d in [0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0, 3.0]:
        for _ in range(4):
            engine_n.update_static(tok, np.array([d, 0.0, 0.0]), STATIC_W, 1.0)
            engine_r.update_static(beliefs, d, STATIC_W, 1.0)
        assert np.isclose(tok.static_belief, beliefs.static, rtol=0, atol=1e-12), (
            f"meas_sigma={meas_sigma} d={d}: "
            f"native {tok.static_belief!r} vs reference {beliefs.static!r}")


@pytest.mark.skipif(not HAS_NATIVE, reason="native")
def test_static_belief_window_gate_matches():
    """0.05 s 창은 실측 분리비 1.20 - 판정하지 않는 것이 맞다 (16.2).

    양쪽이 같은 창 임계를 써야 한다. 임계가 다르면 한쪽만 갱신하고, 그 차이는
    믿음의 절대값이 아니라 *갱신 횟수* 로 나타나 눈에 잘 띄지 않는다.
    """
    engine_n, engine_r = core.ConfidenceEngine(), ref_conf.ConfidenceEngine()
    for dt in (0.01, 0.05, 0.2, 0.49, 0.5, 1.0):
        tok = core.WorldToken()
        tok.static_belief = 0.5
        b = ref_conf.Beliefs()
        engine_n.update_static(tok, np.array([0.3, 0.0, 0.0]), dt, 1.0)
        engine_r.update_static(b, 0.3, dt, 1.0)
        assert np.isclose(tok.static_belief, b.static, rtol=0, atol=1e-12), f"dt={dt}"


@pytest.mark.skipif(not HAS_NATIVE or not hasattr(core.ConfidenceEngine(),
                                                 "decay_static_belief"),
                    reason="바인딩이 decay_static_belief 를 노출하지 않음")
def test_agent_static_claim_expiry_matches():
    """자율 개체의 정적 주장은 상한이 아니라 유효기간으로 다룬다.

    이전 Python 은 is_agent 이면 static 을 0.3 으로 clamp 했다. 그러면 앉아 있는
    사람이 통째로 마스킹되어 fr3_sitting ATE 가 1.01 -> 15.84 cm 로 무너진다.
    """
    engine_n, engine_r = core.ConfidenceEngine(), ref_conf.ConfidenceEngine()

    # 관측 중: 상한 없이 올라가야 한다
    tok = core.WorldToken()
    tok.static_belief = 0.5
    tok.is_agent = True
    b = ref_conf.Beliefs(is_agent=True)
    for _ in range(30):
        engine_n.update_static(tok, np.array([0.002, 0.0, 0.0]), STATIC_W, 1.0)
        engine_r.update_static(b, 0.002, STATIC_W, 1.0)
    assert np.isclose(tok.static_belief, b.static, rtol=0, atol=1e-12)
    assert b.static > 0.7, "상한이 다시 생기면 이 줄이 잡는다"

    # 관측이 끊기면 사전분포로 돌아간다
    for dt in (0.5, 2.0, 4.0, 10.0):
        engine_n.decay_static_belief(tok, dt)
        engine_r.decay_static_belief(b, dt)
        assert np.isclose(tok.static_belief, b.static, rtol=0, atol=1e-12), f"dt={dt}"


# --- 환경 적응 ------------------------------------------------------------

@pytest.mark.parametrize("preset", list(ref_env.PRESETS))
def test_tier_weights_match(preset):
    ev = ref_env.PRESETS[preset]

    native_ev = core.EnvironmentEvidence()
    for field in ("darkness", "haze", "rain_streak", "snow_particle", "dust",
                  "motion_blur", "lens_dirt", "water_drop", "camera_shake",
                  "noise", "texture_poverty"):
        setattr(native_ev, field, getattr(ev, field))

    n = core.derive_adaptation(native_ev)
    r = ref_env.derive_adaptation(ev)

    assert np.isclose(n.visibility, r.visibility, rtol=0, atol=1e-12)
    assert np.isclose(n.camera_health, r.camera_health, rtol=0, atol=1e-12)
    assert np.isclose(n.sensor_reliability, r.sensor_reliability, rtol=0, atol=1e-12)
    assert np.isclose(n.tier.photometric, r.alpha_photometric, rtol=0, atol=1e-12)
    assert np.isclose(n.tier.constellation, r.alpha_constellation, rtol=0, atol=1e-12)
    assert np.isclose(n.tier.structural, r.alpha_structural, rtol=0, atol=1e-12)
    assert np.isclose(n.tier.motion_prior, r.motion_prior, rtol=0, atol=1e-12)
    assert np.isclose(n.memory_retention_scale, r.memory_retention_scale, rtol=0, atol=1e-12)
    assert np.isclose(n.track_persistence_scale, r.track_persistence_scale, rtol=0, atol=1e-12)
    assert np.isclose(n.detection_threshold_scale, r.detection_threshold_scale, rtol=0, atol=1e-12)


def test_tier_weights_match_on_random_evidence():
    for _ in range(200):
        vals = RNG.uniform(0, 1, 11)
        fields = ("darkness", "haze", "rain_streak", "snow_particle", "dust",
                  "motion_blur", "lens_dirt", "water_drop", "camera_shake",
                  "noise", "texture_poverty")

        ev = ref_env.Evidence(**dict(zip(fields, vals)))
        native_ev = core.EnvironmentEvidence()
        for f, v in zip(fields, vals):
            setattr(native_ev, f, float(v))

        n = core.derive_adaptation(native_ev)
        r = ref_env.derive_adaptation(ev)
        assert np.isclose(n.tier.photometric, r.alpha_photometric, rtol=0, atol=1e-12)
        assert np.isclose(n.tier.constellation, r.alpha_constellation, rtol=0, atol=1e-12)
        assert np.isclose(n.tier.structural, r.alpha_structural, rtol=0, atol=1e-12)


# --- 성좌 -----------------------------------------------------------------

def _make_room(n=12, classes=5, extent=4.0, sigma=0.03, seed=99):
    rng = np.random.default_rng(seed)
    return [(i + 1, i % classes,
             np.array([rng.uniform(-extent, extent), rng.uniform(-extent, extent),
                       rng.uniform(-extent, extent) * 0.3]), sigma)
            for i in range(n)]


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_constellation_recovers_same_transform(seed):
    """양쪽 색인이 같은 SE(3) 를 복원해야 한다."""
    room = _make_room(seed=seed)

    native = core.ConstellationIndex()
    reference = ref_const.ConstellationIndex()

    native.insert(1, 1.0, core.SE3.identity(),
                  [core.ConstellationNode(i, c, p, s) for i, c, p, s in room])
    reference.insert(1, 1.0, ref_geom.SE3.identity(),
                     [ref_const.Node(i, c, p, s) for i, c, p, s in room])

    T = ref_geom.SE3.exp(np.array([1.5, -2.0, 0.3, 0.1, 0.4, -0.2]))
    Tinv = T.inverse()
    observed = [(i, c, Tinv @ p, s) for i, c, p, s in room]

    n_match = native.query([core.ConstellationNode(i, c, p, s) for i, c, p, s in observed])
    r_match = reference.query([ref_const.Node(i, c, p, s) for i, c, p, s in observed])

    assert (n_match is None) == (r_match is None), "한쪽만 매칭에 성공했다"
    if n_match is None:
        return

    assert np.allclose(n_match.transform.matrix(), r_match.transform.matrix(), atol=1e-8)
    assert sorted(n_match.correspondences) == sorted(r_match.correspondences)


def test_constellation_rejects_unknown_place_in_both():
    room_a = _make_room(seed=11)
    room_b = _make_room(seed=22)

    native = core.ConstellationIndex()
    reference = ref_const.ConstellationIndex()
    native.insert(1, 1.0, core.SE3.identity(),
                  [core.ConstellationNode(i, c, p, s) for i, c, p, s in room_a])
    reference.insert(1, 1.0, ref_geom.SE3.identity(),
                     [ref_const.Node(i, c, p, s) for i, c, p, s in room_a])

    n = native.query([core.ConstellationNode(i, c, p, s) for i, c, p, s in room_b])
    r = reference.query([ref_const.Node(i, c, p, s) for i, c, p, s in room_b])
    assert n is None and r is None


def test_constellation_gravity_handling_matches():
    """카이랄리티는 각 프레임 자신의 중력으로 재야 한다 - 양쪽 동일하게."""
    room = _make_room(seed=33)
    down = np.array([0.0, 0.0, -1.0])

    native = core.ConstellationIndex()
    reference = ref_const.ConstellationIndex()
    native.insert(1, 1.0, core.SE3.identity(),
                  [core.ConstellationNode(i, c, p, s) for i, c, p, s in room], gravity=down)
    reference.insert(1, 1.0, ref_geom.SE3.identity(),
                     [ref_const.Node(i, c, p, s) for i, c, p, s in room], gravity=down)

    T = ref_geom.SE3.exp(np.array([1.0, -1.0, 0.2, 0.15, 0.5, -0.1]))
    Tinv = T.inverse()
    observed = [(i, c, Tinv @ p, s) for i, c, p, s in room]
    q_gravity = Tinv.R @ down

    n = native.query([core.ConstellationNode(i, c, p, s) for i, c, p, s in observed],
                     gravity=q_gravity)
    r = reference.query([ref_const.Node(i, c, p, s) for i, c, p, s in observed],
                        gravity=q_gravity)

    assert (n is None) == (r is None)
    if n is not None:
        assert np.allclose(n.transform.matrix(), r.transform.matrix(), atol=1e-8)


def _native_nodes(room):
    return [core.ConstellationNode(i, c, p, s) for i, c, p, s in room]


def _ref_nodes(room):
    return [ref_const.Node(i, c, p, s) for i, c, p, s in room]


def _both_indices():
    return core.ConstellationIndex(), ref_const.ConstellationIndex()


def test_constellation_neighbour_keyframes_are_not_ambiguity():
    """모호성 판정의 본체. 이 경로는 단일 장소 색인으로는 절대 도달하지 못한다.

    5프레임 간격으로 등록한 지도에서는 이웃 키프레임이 같은 장면을 보므로
    상위 후보들의 *점수* 가 붙는 것이 정상이다. score2 > 0.85*score1 규칙은
    그것을 지각적 혼동으로 오인해 fr1_xyz 에서 정대응 36개를 36개 모두
    기각했다 (재현율 0 %). 물어야 할 것은 후보들이 같은 *포즈* 를 가리키는가다.

    두 키프레임은 서로 다른 anchor 와 서로 다른 transform 을 갖지만 월드 포즈는
    같은 곳을 가리키므로 한 군집이 되어야 하고, 따라서 채택되어야 한다.
    """
    room = _make_room(n=14, seed=101)
    native, reference = _both_indices()

    # 같은 방을 두 키프레임에서 등록한다. 두 번째는 30 cm 옆에서 본다.
    anchors_n = {}
    for k, shift in enumerate((0.0, 0.30)):
        offset = np.array([shift, 0.0, 0.0])
        xi = np.array([shift, 0.0, 0.0, 0.0, 0.0, 0.0])
        anchor_n = core.SE3.exp(xi)
        local = [(i, c, p - offset, s) for i, c, p, s in room]
        pid = native.insert(k + 1, 1.0, anchor_n, _native_nodes(local))
        reference.insert(k + 1, 1.0, ref_geom.SE3.exp(xi), _ref_nodes(local))
        anchors_n[pid] = anchor_n

    assert native.place_count == 2 and reference.place_count == 2

    # 첫 키프레임 좌표계에서 그대로 다시 본다
    n_all = native.query_all(_native_nodes(room))
    r_all = reference.query_all(_ref_nodes(room))
    assert len(n_all) == len(r_all) == 2, "두 후보가 모두 검증을 통과해야 측정이 성립한다"
    # 점수가 실제로 붙어 있어야 옛 규칙이 발동한다 - 그렇지 않으면 측정이 아니다
    assert r_all[1].score > 0.85 * r_all[0].score, "점수가 붙지 않으면 이 테스트는 아무것도 재지 않는다"

    n = native.query(_native_nodes(room))
    r = reference.query(_ref_nodes(room))
    assert (n is None) == (r is None), (
        f"한쪽만 기각: native={n is None} reference={r is None}")
    assert n is not None, "이웃 키프레임은 지각적 혼동이 아니다"

    # 두 후보의 점수는 *거의* 동률이지 정확히 동률이 아니다. 잡음 없는 장면이라
    # rms 가 1e-16 규모이고, 그 크기에서는 두 구현의 SVD 가 다른 순서로 오차를
    # 쌓아 순위가 바뀐다 (실측: cpp rms 1.41e-16 vs ref 7.58e-16, 5개 후보 중
    # 3·4위가 서로 반대). place_id 를 비교하면 그 부동소수점 잡음을 비교하게
    # 된다. 비교해야 할 불변량은 그 후보가 주장하는 *월드 포즈* 다.
    n_world = np.asarray((anchors_n[n.place_id] @ n.transform).matrix())
    r_world = reference.world_pose(r).matrix()
    assert np.allclose(n_world, r_world, atol=1e-8), (
        f"native place={n.place_id} vs reference place={r.place_id}, 월드 포즈 불일치")


def test_constellation_query_all_exact_tie_order_matches():
    """정확한 동률에서 queryAll 의 순서가 일치해야 한다.

    C++ queryAll/retrieve 는 (score 내림, place_id 오름) 을 *명시적으로* 쓴다.
    place_id 가 유일하므로 이 비교자는 전순서이고, 그래서 std::sort 가 안정
    정렬이 아니라는 사실은 결과에 새지 않는다 - 이미 못박혀 있다.

    파이썬은 점수만으로 정렬했고 sort 가 안정이라 동률 구간에 후보 *생성*
    순서(= idf 투표 순서)가 남아 있었다. 정합 점수가 같은데 투표 점수가 다른
    배치에서 두 구현이 갈린다. 같은 전순서를 파이썬에도 못박았다.

    측정이 성립하려면 *정확한* 동률이 있어야 한다. 배치가 조금이라도 다르면
    점수가 1e-16 수준에서 갈리고, 그 크기에서 순위를 비교하는 것은 두 SVD 의
    반올림 순서를 비교하는 것이다. 그래서 완전히 동일한 방 세 개를 쓴다.
    """
    room = _make_room(n=14, seed=808)
    native, reference = _both_indices()
    # 완전히 같은 노드 집합, anchor 만 다르다 -> 정합 점수가 비트까지 같다
    for k, x in enumerate((0.0, 50.0, 100.0)):
        xi = np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0])
        native.insert(k + 1, 1.0, core.SE3.exp(xi), _native_nodes(room))
        reference.insert(k + 1, 1.0, ref_geom.SE3.exp(xi), _ref_nodes(room))

    n_all = native.query_all(_native_nodes(room))
    r_all = reference.query_all(_ref_nodes(room))
    assert len(n_all) == len(r_all) == 3

    n_scores = [m.score for m in n_all]
    r_scores = [m.score for m in r_all]
    # 정확한 동률인가. 아니면 이 테스트는 타이브레이크가 아니라 잡음을 잰다.
    assert len(set(n_scores)) == 1, f"C++ 점수가 동률이 아니다: {n_scores}"
    assert len(set(r_scores)) == 1, f"참조 점수가 동률이 아니다: {r_scores}"

    assert [m.place_id for m in n_all] == [m.place_id for m in r_all], (
        f"동률 후보 순서가 다르다: {[m.place_id for m in n_all]} vs "
        f"{[m.place_id for m in r_all]}")
    assert [m.place_id for m in n_all] == [1, 2, 3], (
        f"동률은 place_id 오름차순이어야 한다: {[m.place_id for m in n_all]}")


def test_constellation_pose_cluster_statistics_match():
    """군집 질량/경쟁 질량/신뢰도는 후보 집합 전체를 봐야 정해진다."""
    room = _make_room(n=14, seed=202)
    native, reference = _both_indices()
    for k, shift in enumerate((0.0, 0.25, 0.5)):
        offset = np.array([shift, 0.0, 0.0])
        xi = np.array([shift, 0.0, 0.0, 0.0, 0.0, 0.0])
        local = [(i, c, p - offset, s) for i, c, p, s in room]
        native.insert(k + 1, 1.0, core.SE3.exp(xi), _native_nodes(local))
        reference.insert(k + 1, 1.0, ref_geom.SE3.exp(xi), _ref_nodes(local))

    n_all = native.query_all(_native_nodes(room))
    r_all = reference.query_all(_ref_nodes(room))
    assert len(n_all) == len(r_all) >= 2

    for a, b in zip(n_all, r_all):
        assert a.place_id == b.place_id
        assert np.isclose(a.score, b.score, rtol=0, atol=1e-12)
        assert np.isclose(a.chi2_dof, b.chi2_dof, atol=1e-9)
        assert a.agree_count == b.agree_count
        assert np.isclose(a.support, b.support, rtol=0, atol=1e-12)
        assert np.isclose(a.rival_mass, b.rival_mass, rtol=0, atol=1e-12)
        assert np.isclose(a.pose_margin, b.pose_margin, atol=1e-9)
        assert np.isclose(a.confidence, b.confidence, rtol=0, atol=1e-12)


def test_constellation_duplicate_rooms_are_rejected_by_both():
    """진짜 지각적 혼동: 같은 배치의 방 두 개가 50 m 떨어져 있다.

    변환은 같고 anchor 만 다르므로 월드 포즈가 두 군집으로 갈라지고 질량이
    동률이 된다. 점수 공간에서는 위 이웃 키프레임 경우와 구분되지 않는다.
    """
    room = _make_room(n=14, seed=303)
    native, reference = _both_indices()
    for k, x in enumerate((0.0, 50.0)):
        xi = np.array([x, 0.0, 0.0, 0.0, 0.0, 0.0])
        native.insert(k + 1, 1.0, core.SE3.exp(xi), _native_nodes(room))
        reference.insert(k + 1, 1.0, ref_geom.SE3.exp(xi), _ref_nodes(room))

    n_all = native.query_all(_native_nodes(room))
    r_all = reference.query_all(_ref_nodes(room))
    assert len(n_all) == len(r_all) == 2
    assert r_all[0].rival_mass > 0.0, "두 군집으로 갈라져야 이 테스트가 의미를 갖는다"

    n = native.query(_native_nodes(room))
    r = reference.query(_ref_nodes(room))
    assert (n is None) == (r is None)
    assert n is None, "동일한 방 두 개는 기각되어야 한다"


def test_constellation_confidence_floor_matches():
    """min_confidence 하한이 양쪽에서 같은 지점에 걸려야 한다."""
    room = _make_room(n=14, seed=404)

    for floor in (0.0, 0.3, 0.55, 0.9, 1.0):
        ncfg = core.ConstellationConfig()
        ncfg.min_confidence = floor
        rcfg = ref_const.Config(min_confidence=floor)
        native = core.ConstellationIndex(ncfg)
        reference = ref_const.ConstellationIndex(rcfg)

        native.insert(1, 1.0, core.SE3.identity(), _native_nodes(room))
        reference.insert(1, 1.0, ref_geom.SE3.identity(), _ref_nodes(room))

        n = native.query(_native_nodes(room))
        r = reference.query(_ref_nodes(room))
        assert (n is None) == (r is None), f"min_confidence={floor}"


# --- 직접정렬 (Tier 0) -----------------------------------------------------
#
# ECDA 는 오랫동안 차등 테스트가 하나도 없었다. 15 장의 2e4 과신은 차등
# 테스트를 통과한 것이 아니라 아무도 재지 않았을 뿐이다. 아래 네 테스트가
# 그 공백의 첫 조각이다: 정보행렬 모델, 결정성, 파이라미드 레벨,
# 그리고 바인딩의 입력 검증.
#
# 예전에는 별도 프로세스로 돌렸다. 바인딩의 align 이 프로세스를 죽였기
# 때문이다(0xC0000409/0xC0000374). 지금은 죽지 않으므로 인프로세스로 돌린다 -
# 다시 죽으면 pytest 가 통째로 죽고, 그건 xfail 한 줄보다 훨씬 눈에 띈다.

_ECDA_CAM = dict(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)

_MODEL_PAIRS = [
    ("CoherentFrame", "coherent_frame"),
    ("EffectiveSample", "effective_sample"),
    ("ResidualVariance", "residual_variance"),
    ("SensorVariance", "sensor_variance"),
]


def _ecda_scene():
    """양쪽 구현에 넣을 동일한 합성 장면 한 쌍."""
    from wme.sim.render import RenderScene, render_frame
    from wme.sim.world import CameraModel

    cam = CameraModel(**_ECDA_CAM)
    look_y = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                              np.array([0, 1.0, 0])])
    sc = RenderScene.room(size=4.0, height=2.6)
    T_ref = ref_geom.SE3(look_y, np.array([0.0, -1.0, 1.3]))
    T_rel = ref_geom.SE3.exp(np.array([0.04, -0.02, 0.01, 0.01, -0.01, 0.005]))
    ref_f = render_frame(sc, T_ref, cam, None, seed=0)
    cur_f = render_frame(sc, T_ref @ T_rel.inverse(), cam, None, seed=1)
    return cam, ref_f, cur_f


def _native_align(ref_f, cur_f, cam, levels=4, model=None, mask=None):
    cfg = core.DirectAlignerConfig()
    cfg.pyramid_levels = levels
    if model is not None:
        cfg.information_model = getattr(core.InformationModel, model)
    kw = {}
    if mask is not None:
        kw["static_mask"] = np.ascontiguousarray(mask, np.uint8)
    return core.DirectAligner(cfg).align(
        np.ascontiguousarray(np.clip(ref_f.gray, 0, 255), np.uint8),
        np.ascontiguousarray(ref_f.depth, np.float32),
        np.ascontiguousarray(np.clip(cur_f.gray, 0, 255), np.uint8),
        cam.fx, cam.fy, cam.cx, cam.cy, **kw)


@pytest.mark.parametrize("native_model,ref_model", _MODEL_PAIRS)
def test_ecda_information_model_agrees_in_magnitude(native_model, ref_model):
    """정보행렬의 잡음 모델이 양쪽에서 같은가 - 네 모델 전부에 대해.

    두 구현은 점 선택과 로버스트 커널 세부가 달라 Lambda 가 성분별로 일치하지는
    않는다. 그러나 *분모* 가 다르면 자릿수가 통째로 어긋난다. 실제로 그랬다:
    C++ 기본이 EffectiveSample 에서 CoherentFrame 으로 바뀌었을 때 Python 은
    따라가지 않았고, 같은 장면에서 trace 가 1.44e6 대 2.71e4 로 갈렸다 (53 배).
    이 테스트가 그것을 잡았다.

    모델을 네 개 다 도는 이유는 하나만 재면 "기본값이 같은가" 만 재게 되기
    때문이다. 분모 식 자체가 같은지는 모델을 바꿔 가며 봐야 나온다.
    """
    from wme.localization import EcdaConfig, InformationModel, align

    cam, ref_f, cur_f = _ecda_scene()
    native = _native_align(ref_f, cur_f, cam, levels=4, model=native_model)
    reference = align(ref_f.gray, ref_f.depth, cur_f.gray, cam,
                      cfg=EcdaConfig(pyramid_levels=4,
                                     information_model=InformationModel(ref_model)))

    n_tr = float(np.trace(native.information))
    r_tr = float(np.trace(reference.information))
    assert n_tr > 0.0 and r_tr > 0.0, f"{native_model}: trace 가 0 이면 비교가 성립 안 한다"

    ratio = r_tr / n_tr
    assert 0.1 < ratio < 10.0, (
        f"{native_model}: 정보행렬 자릿수 불일치 python/cpp = {ratio:.3e} "
        f"(python trace {r_tr:.4e}, cpp trace {n_tr:.4e})")


def test_ecda_information_models_are_actually_distinct():
    """네 모델이 서로 다른 답을 내야 위 테스트가 측정이 된다.

    분모 식이 모델마다 같아져 버리면 위 파라미터 네 개가 전부 같은 것을 재고,
    '네 모델을 다 봤다' 는 문장이 거짓이 된다 (10.4 의 실패 모드).
    """
    cam, ref_f, cur_f = _ecda_scene()
    traces = {m: float(np.trace(_native_align(ref_f, cur_f, cam, 4, m).information))
              for m, _ in _MODEL_PAIRS}
    vals = sorted(traces.values())
    for a, b in zip(vals, vals[1:]):
        assert b > a * 1.5, f"모델들이 구분되지 않는다: {traces}"


def test_ecda_binding_is_deterministic_across_levels():
    """같은 입력 -> 같은 답. 그리고 레벨마다 실제로 다른 답.

    바인딩의 align 은 한동안 프로세스를 죽였고(0xC0000409/0xC0000374, 실행마다
    번갈아) 그동안 ECDA 는 파이썬에서 한 번도 실행되지 않았다. 그 크래시가
    돌아오면 이 테스트가 pytest 를 통째로 죽여서 알린다.

    비결정적 누산(워커 순서 의존)이 생기면 반복 호출이 갈린다. 같은 인스턴스를
    재사용하는 경로(내부 버퍼 재사용)도 함께 밟는다.
    """
    cam, ref_f, cur_f = _ecda_scene()

    per_level = {}
    for levels in (1, 2, 3, 4, 5):
        cfg = core.DirectAlignerConfig()
        cfg.pyramid_levels = levels
        aligner = core.DirectAligner(cfg)
        gray_r = np.ascontiguousarray(np.clip(ref_f.gray, 0, 255), np.uint8)
        depth_r = np.ascontiguousarray(ref_f.depth, np.float32)
        gray_c = np.ascontiguousarray(np.clip(cur_f.gray, 0, 255), np.uint8)

        runs = [aligner.align(gray_r, depth_r, gray_c,
                              cam.fx, cam.fy, cam.cx, cam.cy) for _ in range(3)]
        for k, r in enumerate(runs[1:], start=2):
            assert np.array_equal(r.information, runs[0].information), (
                f"levels={levels} 호출 {k} 에서 정보행렬이 달라졌다")
            assert r.point_count == runs[0].point_count
            assert np.array_equal(r.T_cur_ref.matrix(), runs[0].T_cur_ref.matrix())
        per_level[levels] = runs[0].T_cur_ref.matrix()

    # 레벨을 늘리면 해가 실제로 움직여야 한다. 안 그러면 이 테스트는 레벨
    # 하나만 재고 나머지 넷은 장식이다 (거친 레벨이 통째로 건너뛰어지던 결함).
    assert not np.allclose(per_level[1], per_level[4], atol=1e-9), (
        "레벨 1 과 4 의 해가 같다 - 거친 레벨이 아무 일도 하지 않고 있다")


def test_ecda_recovers_the_synthetic_transform_in_both():
    """세 번째 근거. 두 구현이 *합성 ground truth* 에 같은 정도로 가까운가.

    자릿수 비교만으로는 둘 다 같은 방향으로 틀렸을 때를 못 잡는다. 장면을
    렌더한 상대포즈는 알고 있으므로 그것을 기준으로 양쪽을 각각 잰다.
    """
    from wme.localization import EcdaConfig, align

    cam, ref_f, cur_f = _ecda_scene()
    T_rel = ref_geom.SE3.exp(np.array([0.04, -0.02, 0.01, 0.01, -0.01, 0.005]))

    native = _native_align(ref_f, cur_f, cam, levels=4)
    reference = align(ref_f.gray, ref_f.depth, cur_f.gray, cam,
                      cfg=EcdaConfig(pyramid_levels=4))

    n_err = np.linalg.norm(
        ref_geom.SE3.from_matrix(np.asarray(native.T_cur_ref.matrix())).log() - T_rel.log())
    r_err = np.linalg.norm(reference.T_cur_ref.log() - T_rel.log())

    # 항등 초기화 대비 양쪽 모두 실제로 수렴했어야 한다
    identity_err = float(np.linalg.norm(T_rel.log()))
    assert n_err < 0.5 * identity_err, f"C++ 이 수렴하지 않았다: {n_err:.4f}"
    assert r_err < 0.5 * identity_err, f"Python 이 수렴하지 않았다: {r_err:.4f}"
    # 두 구현이 서로 같은 답에 가까워야 한다. 커널/점선택 차이만큼만 벌어진다.
    assert abs(n_err - r_err) < 0.5 * identity_err, (
        f"두 구현이 서로 다른 해로 갔다: cpp {n_err:.4f} vs python {r_err:.4f} "
        f"(항등 오차 {identity_err:.4f})")


def test_align_rejects_noncontiguous_input():
    """바인딩의 toMat 은 cv::Mat 이 표현할 수 없는 배열을 거부해야 한다.

    cv::Mat 은 행 사이 패딩만 표현할 수 있고 원소 사이 간격은 표현할 수 없다.
    예전 toMat 은 strides[0] 만 보고 마지막 축을 검사하지 않아, img[:, ::2]
    같은 배열에서 예외 없이 *다른 픽셀* 을 읽었다 (실측: 같은 픽셀 집합의
    품질점수가 0.974 대신 0.898). 조용히 틀린 답보다 예외가 낫다.
    """
    cam, ref_f, cur_f = _ecda_scene()
    gray_r = np.ascontiguousarray(np.clip(ref_f.gray, 0, 255), np.uint8)
    depth_r = np.ascontiguousarray(ref_f.depth, np.float32)
    gray_c = np.ascontiguousarray(np.clip(cur_f.gray, 0, 255), np.uint8)

    # 정상 입력은 통과한다 - 검증이 지나치게 조이지 않았음을 먼저 보인다
    _native_align(ref_f, cur_f, cam, levels=2)

    for name, bad in (("열 슬라이스", depth_r[:, ::2]),
                      ("F-order", np.asfortranarray(depth_r)),
                      ("행 뒤집기", depth_r[::-1])):
        with pytest.raises((ValueError, RuntimeError)):
            core.DirectAligner().align(gray_r[:, :bad.shape[1]], bad,
                                       gray_c[:, :bad.shape[1]],
                                       cam.fx, cam.fy, cam.cx, cam.cy)


# --- kabsch 퇴화 경로 ------------------------------------------------------

def test_kabsch_reflection_fix_matches():
    """반사(det = -1) 보정 경로. 무작위 well-conditioned 입력은 여기 못 온다.

    SVD 의 V U^T 가 det = -1 을 내는 배치를 일부러 만든다. 보정이 없으면 두
    구현 모두 거울상을 반환하고, 한쪽만 보정하면 회전이 통째로 갈린다.
    """
    hit = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        src = rng.uniform(-1, 1, (6, 3))
        # 거울 대응: dst 를 src 의 반사로 둔다. 최적 정합이 반사가 된다.
        dst = src * np.array([1.0, 1.0, -1.0]) + rng.normal(0, 1e-3, (6, 3))

        H = (src - src.mean(0)).T @ (dst - dst.mean(0))
        U, S, Vt = np.linalg.svd(H)
        if np.linalg.det(Vt.T @ U.T) >= 0.0:
            continue          # 이 표본은 보정 경로를 밟지 않는다
        hit += 1

        native = core.kabsch(list(src), list(dst))
        reference = ref_geom.kabsch(src, dst)
        assert np.allclose(native.matrix(), reference.matrix(), atol=1e-9), f"seed {seed}"
        assert np.isclose(np.linalg.det(np.asarray(native.R)), 1.0, atol=1e-9), (
            "반사가 그대로 반환됐다")
    assert hit >= 20, f"보정 경로에 도달한 표본이 {hit} 개뿐 - 측정이 성립 안 한다"


def test_kabsch_collinear_is_degenerate_in_both():
    """공선 배치는 양쪽 모두 실패로 보고해야 한다.

    회전이 축 방향으로 결정되지 않는데 값을 돌려주면 그 축의 오차가 그대로
    포즈에 들어간다. C++ 은 ErrorCode::Degenerate, Python 은 ValueError.
    """
    for n in (3, 5, 12):
        t = np.linspace(-2.0, 2.0, n)[:, None]
        src = t * np.array([1.0, 0.3, -0.7])          # 정확히 한 직선 위
        dst = src + np.array([0.5, -0.2, 0.1])

        with pytest.raises(RuntimeError):
            core.kabsch(list(src), list(dst))
        with pytest.raises(ValueError):
            ref_geom.kabsch(src, dst)

    # 같은 점들을 살짝 흩뜨리면 양쪽 다 성공해야 한다 - 게이트가 항상
    # 실패하는 것이 아님을 보인다
    t = np.linspace(-2.0, 2.0, 12)[:, None]
    src = t * np.array([1.0, 0.3, -0.7])
    src = src + np.random.default_rng(5).normal(0, 0.2, src.shape)
    dst = src + np.array([0.5, -0.2, 0.1])
    assert np.allclose(core.kabsch(list(src), list(dst)).matrix(),
                       ref_geom.kabsch(src, dst).matrix(), atol=1e-9)


def test_kabsch_too_few_points_matches():
    """3 점 미만은 양쪽 모두 거부한다."""
    pts = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    with pytest.raises(RuntimeError):
        core.kabsch(pts, pts)
    with pytest.raises(ValueError):
        ref_geom.kabsch(np.array(pts), np.array(pts))


# --- 검출 후처리 ----------------------------------------------------------

def test_box_iou_matches():
    from wme.yolo import box_iou as ref_iou

    for _ in range(500):
        a = tuple(RNG.uniform(0, 100, 2)) + tuple(RNG.uniform(1, 50, 2))
        b = tuple(RNG.uniform(0, 100, 2)) + tuple(RNG.uniform(1, 50, 2))
        assert np.isclose(core.box_iou(a, b), ref_iou(a, b), atol=1e-6)


def test_nms_matches():
    from wme.yolo import Detection as RefDetection
    from wme.yolo import non_max_suppression as ref_nms

    for _ in range(50):
        n = int(RNG.integers(5, 30))
        raw = [(int(RNG.integers(0, 3)),
                float(RNG.uniform(0, 200)), float(RNG.uniform(0, 200)),
                float(RNG.uniform(10, 80)), float(RNG.uniform(10, 80)),
                float(RNG.uniform(0.3, 0.99)))
               for _ in range(n)]

        native_in = [core.Detection(c, str(c), x, y, w, h, conf) for c, x, y, w, h, conf in raw]
        ref_in = [RefDetection(c, str(c), (x, y, w, h), conf) for c, x, y, w, h, conf in raw]

        native_out = core.non_max_suppression(native_in, 0.45)
        ref_out = ref_nms(ref_in, 0.45)

        assert len(native_out) == len(ref_out)
        for a, b in zip(native_out, ref_out):
            assert a.class_id == b.class_id
            assert np.allclose(a.box, b.box, atol=1e-5)


# --- 토큰 저장소 (TokenStore) ---------------------------------------------
#
# 16 장에서만 결함 다섯 개가 나온 파일인데 차등 테스트가 하나도 없었다.
# 바인딩에 아예 노출되지 않아 비교할 방법 자체가 없었던 것이 이유다.

_K = (320.0, 320.0, 319.5, 239.5, 640, 480)
_K_ARR = np.array(_K, dtype=float)


def _ref_K():
    return ref_tok.Intrinsics(*_K[:4], int(_K[4]), int(_K[5]))


def _flat_depth(value=3.0):
    return np.full((int(_K[5]), int(_K[4])), value, np.float32)


def _both_stores(cfg_kw=None):
    ncfg = core.TokenStoreConfig()
    rcfg = ref_tok.TokenStoreConfig()
    for k, v in (cfg_kw or {}).items():
        setattr(ncfg, k, v)
        setattr(rcfg, k, v)
    return core.TokenStore(ncfg), ref_tok.TokenStore(rcfg)


def _det_pair(class_id, name, box, conf=0.9):
    return (core.Detection(class_id, name, *box, conf),
            ref_tok.Detection(class_id, name, box, conf))


def _compare_stores(ns, rs, where=""):
    nt, rt = ns.all_tokens(), rs.all_tokens()
    assert len(nt) == len(rt), f"{where}: 토큰 수 {len(nt)} vs {len(rt)}"
    for a, b in zip(nt, rt):
        tag = f"{where} token {a.token_id}/{b.token_id}"
        assert a.token_id == b.token_id, tag
        assert a.class_id == b.class_id, tag
        assert a.lifecycle.name.lower() == b.lifecycle.value, (
            f"{tag}: lifecycle {a.lifecycle} vs {b.lifecycle}")
        assert a.observation_count == b.observation_count, tag
        # 기하량은 상대 허용치로 잰다. 3x3 역행렬을 Eigen 은 여인수로 numpy 는
        # LAPACK LU 로 풀고, 정보필터가 P <-> P^-1 을 매 프레임 왕복하면서
        # velocity_cov = (P_new + P_old)/dt^2 로 dt^-2 배(여기서 44 배) 증폭한다.
        # 그래서 ULP 차이가 12 프레임에 상대 1e-8 까지 자란다 - 알고리즘 차이지
        # 식의 차이가 아니다. 식이 다르면 최소 1e-3 단위로 갈리므로 1e-6 이면
        # 전부 잡힌다. 반면 믿음은 증폭 경로가 없어 아래에서 1e-11 로 조인다.
        assert np.allclose(a.position, b.position, rtol=1e-6, atol=1e-9), (
            f"{tag}: position {a.position} vs {b.position}")
        assert np.allclose(a.position_cov, b.position_cov, rtol=1e-6, atol=1e-15), (
            f"{tag}: cov\n{a.position_cov}\nvs\n{b.position_cov}")
        assert np.allclose(a.velocity, b.velocity, rtol=1e-6, atol=1e-9), (
            f"{tag}: velocity {a.velocity} vs {b.velocity}")
        assert _same(a.existence_belief, b.beliefs.existence, atol=1e-11), (
            f"{tag}: existence {a.existence_belief!r} vs {b.beliefs.existence!r}")
        assert _same(a.static_belief, b.beliefs.static, atol=1e-11), (
            f"{tag}: static {a.static_belief!r} vs {b.beliefs.static!r}")
        assert _same(a.identity_belief, b.beliefs.identity, atol=1e-11), tag
        assert a.static_updates == b.beliefs.static_diag.updates, (
            f"{tag}: static 갱신 횟수 {a.static_updates} vs "
            f"{b.beliefs.static_diag.updates}")


def _compare_reports(n, r, where=""):
    for f in ("matched", "created", "missed", "retired", "det_no_candidate",
              "det_gated_out", "det_unassigned"):
        assert getattr(n, f) == getattr(r, f), (
            f"{where}: {f} = {getattr(n, f)} vs {getattr(r, f)}")
    assert _same(n.dynamic_area_ratio, r.dynamic_area_ratio, atol=1e-12), where
    assert _same(n.gated_maha_sum, r.gated_maha_sum, atol=1e-9), where
    assert _same(n.gated_dist_sum, r.gated_dist_sum, atol=1e-9), where


def test_tokenstore_single_track_matches():
    """정지한 물체를 30 프레임 따라간다. 매 프레임 상태가 일치해야 한다.

    측정/융합/연관/생애주기/정적판정이 한 줄에 엮여 있으므로, 어느 하나가
    갈리면 그 다음 프레임부터 위치와 믿음이 같이 벌어진다.
    """
    ns, rs = _both_stores()
    depth = _flat_depth(3.0)
    K = _ref_K()

    for i in range(30):
        stamp = 1.0 + i * 0.1
        box = (300.0, 200.0, 60.0, 120.0)
        nd, rd = _det_pair(56, "chair", box)
        nrep = ns.integrate([nd], stamp, _K_ARR, depth)
        rrep = rs.integrate([rd], stamp, K, depth)
        _compare_reports(nrep, rrep, f"frame {i}")
        _compare_stores(ns, rs, f"frame {i}")

    tok = rs.all_tokens()[0]
    # 이 테스트가 실제로 판정 경로를 밟았는지 확인한다. 0 이면 static_min_dt
    # 게이트에 전부 걸린 것이고, 그러면 정적 채널은 아무것도 안 재고 있다.
    assert tok.beliefs.static_diag.updates >= 5, (
        f"정적 판정이 {tok.beliefs.static_diag.updates} 번만 돌았다 - "
        "창 길이를 늘리지 않으면 이 테스트는 사전분포만 비교한다")
    assert tok.lifecycle == ref_tok.Lifecycle.ACTIVE


def test_tokenstore_association_gate_matches():
    """게이트 전용 기동 여유항. 게이트를 통과/탈락하는 경계를 양쪽에서 잰다.

    16 장: 융합 공분산만으로 게이트를 세우면 정보필터가 사람 박스 중심을
    sigma 8 mm 로 안다고 주장해 174 mm 떨어진 정대응이 마할라노비스 474 로
    거부됐다. assoc_maneuver_speed 가 그 폭을 넣는다. 이 값을 0 으로 두면
    같은 입력에서 연관이 끊기고 새 토큰이 생겨야 한다 - 두 구현이 같은
    지점에서 끊기는가를 본다.
    """
    depth = _flat_depth(3.0)
    K = _ref_K()

    outcomes = {}
    # dt 를 두 종류 돈다. 여유항은 speed * max(dt, 1e-2) 인데, 30 Hz 이상에서만
    # 그 하한이 실제로 걸린다. 0.1 s 만 쓰면 max() 가 항상 dt 를 고르므로
    # 하한을 지워도 결과가 그대로다 - 그 상태로는 하한을 재지 못한다.
    for speed in (0.0, 2.0, 20.0):
        for dt in (0.1, 0.004):
            ns, rs = _both_stores({"assoc_maneuver_speed": speed})
            # 사람이 프레임마다 30 px 씩 옆으로 간다
            for i in range(6):
                stamp = 1.0 + i * dt
                box = (200.0 + 30.0 * i, 150.0, 50.0, 130.0)
                nd, rd = _det_pair(0, "person", box)
                nrep = ns.integrate([nd], stamp, _K_ARR, depth)
                rrep = rs.integrate([rd], stamp, K, depth)
                _compare_reports(nrep, rrep, f"speed={speed} dt={dt} frame {i}")
                _compare_stores(ns, rs, f"speed={speed} dt={dt} frame {i}")
            outcomes[(speed, dt)] = len(ns.all_tokens())

    # 여유항이 실제로 결과를 바꿔야 이 테스트가 무언가를 잰다
    assert outcomes[(0.0, 0.1)] != outcomes[(2.0, 0.1)], (
        f"기동 여유가 결과를 바꾸지 않는다: {outcomes} - 게이트를 재지 못한다")
    # dt < 1e-2 영역(250 Hz)에서도 여유항이 결과를 갈라야 한다. 이 영역이
    # max(dt, 1e-2) 의 하한이 실제로 선택되는 유일한 구간이다.
    assert outcomes[(2.0, 0.004)] != outcomes[(20.0, 0.004)], (
        f"dt 하한 영역에서 여유항이 결과를 바꾸지 않는다: {outcomes}")


def test_tokenstore_lifecycle_transitions_match():
    """관측이 끊긴 뒤의 전이. Occluded -> Dormant -> Retired 를 실제로 밟는다.

    타임아웃은 env 배율이 곱해진다. 배율을 1 로만 두면 그 곱셈이 한 번도
    실행되지 않으므로 다른 값도 함께 돈다.
    """
    depth = _flat_depth(3.0)
    K = _ref_K()

    for persist, retain in ((1.0, 1.0), (2.0, 0.05)):
        ns, rs = _both_stores()
        nenv = core.derive_adaptation(core.EnvironmentEvidence())
        nenv.track_persistence_scale = persist
        nenv.memory_retention_scale = retain

        for i in range(5):
            stamp = 1.0 + i * 0.2
            nd, rd = _det_pair(62, "tv", (100.0, 100.0, 80.0, 60.0))
            ns.integrate([nd], stamp, _K_ARR, depth, core.SE3.identity(), nenv)
            rs.integrate([rd], stamp, K, depth, None, 1.0, persist, retain)
        _compare_stores(ns, rs, f"persist={persist} 관측중")
        assert rs.all_tokens()[0].lifecycle == ref_tok.Lifecycle.ACTIVE

        seen = set()
        for i in range(40):
            stamp = 2.0 + i * 1.0
            nrep = ns.integrate([], stamp, _K_ARR, depth, core.SE3.identity(), nenv)
            rrep = rs.integrate([], stamp, K, depth, None, 1.0, persist, retain)
            _compare_reports(nrep, rrep, f"persist={persist} idle {i}")
            _compare_stores(ns, rs, f"persist={persist} idle {i}")
            for t in rs.all_tokens():
                seen.add(t.lifecycle)
            if not rs.all_tokens():
                seen.add(ref_tok.Lifecycle.RETIRED)
                break
        # 전이를 실제로 여러 개 밟았는가. 하나만 밟았으면 전이표를 안 잰 것이다.
        assert len(seen) >= 2, f"persist={persist}: 밟은 상태가 {seen} 뿐이다"
        assert ref_tok.Lifecycle.RETIRED in seen, "은퇴까지 가지 못했다"


def test_tokenstore_multi_object_assignment_matches():
    """같은 클래스 객체 넷이 서로 가까이 있을 때 배정이 같은가.

    비용행렬의 IoU 항은 C++ 에서 float 로 계산된다. 참조가 double 로 계산하면
    동점 근처에서 배정이 통째로 뒤집힌다 - 참조도 float32 로 자른다.
    """
    ns, rs = _both_stores()
    depth = _flat_depth(4.0)
    K = _ref_K()

    boxes0 = [(100.0, 100.0, 40.0, 40.0), (150.0, 105.0, 42.0, 38.0),
              (205.0, 98.0, 38.0, 44.0), (260.0, 110.0, 40.0, 40.0)]
    for i in range(12):
        stamp = 1.0 + i * 0.15
        shift = 6.0 * i
        boxes = [(x + shift, y, w, h) for (x, y, w, h) in boxes0]
        nds = [core.Detection(41, "cup", *b, 0.85) for b in boxes]
        rds = [ref_tok.Detection(41, "cup", b, 0.85) for b in boxes]
        nrep = ns.integrate(nds, stamp, _K_ARR, depth)
        rrep = rs.integrate(rds, stamp, K, depth)
        _compare_reports(nrep, rrep, f"frame {i}")
        _compare_stores(ns, rs, f"frame {i}")

    assert len(rs.all_tokens()) == 4, (
        f"토큰이 {len(rs.all_tokens())} 개 - 4 개가 안정적으로 유지되어야 "
        "배정 자체를 재는 것이 된다")


def test_tokenstore_static_mask_provenance_matches():
    """buildStaticMask 의 출처 회계. 마스크 자체보다 이쪽이 갈리기 쉽다.

    16.5: 지운 면적의 70~86 % 가 이번 프레임에 관측되지 않은 토큰에서 나왔고,
    64 % 중 32.5 pp 가 판정이 한 번도 안 돈 토큰에서 나왔다. 그래서 지금은
    둘 다 보류한다. 보류된 몫(withheld_unjudged)이 0 이 아닌 상태를 반드시
    만들어야 그 분기를 재는 것이 된다.
    """
    ns, rs = _both_stores()
    depth = _flat_depth(2.5)
    K = _ref_K()

    nd, rd = _det_pair(0, "person", (50.0, 150.0, 90.0, 200.0))
    ns.integrate([nd], 1.0, _K_ARR, depth)
    rs.integrate([rd], 1.0, K, depth)

    nmask, nrep = ns.build_static_mask(_K_ARR)
    rmask, rrep = rs.build_static_mask(K)
    assert np.array_equal(nmask, rmask), "마스크 픽셀이 다르다"
    assert nrep.n_withheld == rrep.n_withheld == 1
    assert rrep.withheld_unjudged > 0.0, "보류 분기를 밟지 않았다 - 측정이 아니다"
    assert _same(nrep.withheld_unjudged, rrep.withheld_unjudged, atol=1e-12)
    assert _same(nrep.masked_ratio, rrep.masked_ratio, atol=1e-12)
    assert rrep.masked_ratio == 0.0, "보류 상태인데 지워졌다"

    # 판정이 돌 만큼 시간을 준다 (static_min_dt = 0.5 s). 그리고 실제로
    # 동적이라는 증거가 쌓일 만큼 빠르게 움직여야 한다 - 안 그러면 믿음이
    # 0.4 위로 올라가 is_dynamic 이 거짓이 되고 마스킹 경로를 못 밟는다.
    for i in range(1, 14):
        stamp = 1.0 + i * 0.3
        nd, rd = _det_pair(0, "person", (50.0 + 25.0 * i, 150.0, 90.0, 200.0))
        ns.integrate([nd], stamp, _K_ARR, depth)
        rs.integrate([rd], stamp, K, depth)
    _compare_stores(ns, rs, "마스크 전")
    assert rs.all_tokens()[0].is_dynamic(), (
        "토큰이 동적으로 판정되지 않았다 - 마스킹 경로가 안 돈다")

    nmask, nrep = ns.build_static_mask(_K_ARR)
    rmask, rrep = rs.build_static_mask(K)
    assert rs.all_tokens()[0].beliefs.static_diag.updates > 0
    assert np.array_equal(nmask, rmask), "판정 후 마스크 픽셀이 다르다"
    for f in ("masked_ratio", "from_observed", "from_stale", "withheld_unjudged"):
        assert _same(getattr(nrep, f), getattr(rrep, f), atol=1e-12), f
    assert nrep.n_masking == rrep.n_masking
    assert rrep.masked_ratio > 0.0, "아무것도 안 지웠으면 회계를 재지 못한다"


def test_tokenstore_merge_matches():
    """루프 클로저 병합. 위치는 정보 가중 평균, 믿음은 로그오즈 합.

    두 토큰의 공분산이 같으면 정보 가중 평균과 단순 평균이 같은 값이 되어
    가중 여부를 재지 못한다. 그래서 깊이를 다르게 준다 - 관측 공분산이
    z^2 에 비례하므로 8 m 쪽이 2 m 쪽보다 16 배 불확실해진다.
    """
    ns, rs = _both_stores()
    depth = _flat_depth(2.0)
    depth[:, 320:] = 8.0
    K = _ref_K()

    for i in range(4):
        stamp = 1.0 + i * 0.3
        nds = [core.Detection(56, "chair", 100.0, 100.0, 60.0, 60.0, 0.9),
               core.Detection(62, "tv", 400.0, 300.0, 80.0, 50.0, 0.8)]
        rds = [ref_tok.Detection(56, "chair", (100.0, 100.0, 60.0, 60.0), 0.9),
               ref_tok.Detection(62, "tv", (400.0, 300.0, 80.0, 50.0), 0.8)]
        ns.integrate(nds, stamp, _K_ARR, depth)
        rs.integrate(rds, stamp, K, depth)
    _compare_stores(ns, rs, "병합 전")
    assert len(rs.all_tokens()) == 2

    ids = [t.token_id for t in rs.all_tokens()]
    sig = [rs.tokens[i].position_sigma() for i in ids]
    assert max(sig) > 3.0 * min(sig), (
        f"두 토큰의 불확실성이 비슷하다 {sig} - 정보 가중을 재지 못한다")
    ns.merge(ids[0], ids[1])
    rs.merge(ids[0], ids[1])

    n = ns.find(ids[0])
    r = rs.tokens[ids[0]]
    assert np.allclose(n.position, r.position, rtol=1e-6, atol=1e-9)
    assert np.allclose(n.position_cov, r.position_cov, rtol=1e-6, atol=1e-15)
    assert _same(n.existence_belief, r.beliefs.existence, atol=1e-11)
    assert _same(n.static_belief, r.beliefs.static, atol=1e-11)
    assert _same(n.identity_belief, r.beliefs.identity, atol=1e-11)
    assert n.observation_count == r.observation_count
    assert len(ns.all_tokens()) == len(rs.all_tokens()) == 1


# --- 성좌 구성 (buildFrom / isStableLandmark) ------------------------------

def test_constellation_build_from_matches():
    """어떤 토큰이 장소를 정의하는가. 네 조건 전부를 경계에서 밟는다.

    isStableLandmark 는 static > 0.7, existence > 0.6, obs >= 3, sigma < 0.5 다.
    네 조건 중 하나만 걸려도 탈락하므로, 각 조건을 단독으로 위반하는 토큰을
    넣어야 어느 조건이 빠져도 이 테스트가 잡는다.
    """
    reference = ref_geom.SE3.exp(np.array([0.4, -0.2, 0.1, 0.05, -0.1, 0.2]))
    n_ref = core.SE3.exp(np.array([0.4, -0.2, 0.1, 0.05, -0.1, 0.2]))

    # (static, existence, obs, sigma, 통과해야 하는가)
    specs = [
        (0.90, 0.90, 5, 0.10, True),
        (0.70, 0.90, 5, 0.10, False),   # static 경계값 (> 0.7 이라 탈락)
        (0.71, 0.90, 5, 0.10, True),
        (0.90, 0.60, 5, 0.10, False),   # existence 경계값
        (0.90, 0.61, 5, 0.10, True),
        (0.90, 0.90, 2, 0.10, False),   # 관측 수 경계
        (0.90, 0.90, 3, 0.10, True),
        (0.90, 0.90, 5, 0.50, False),   # sigma 경계
        (0.90, 0.90, 5, 0.49, True),
        (0.20, 0.90, 5, 0.10, False),   # 움직이는 물체
    ]

    n_tokens, r_tokens = [], []
    for i, (st, ex, obs, sig, _) in enumerate(specs):
        pos = np.array([0.5 * i, 0.2 * i - 1.0, 0.1 * i])
        cov = np.eye(3) * (sig * sig)

        nt = core.WorldToken()
        nt.static_belief, nt.existence_belief = st, ex
        nt.observation_count, nt.position, nt.position_cov = obs, pos, cov
        nt.class_id = i % 4
        n_tokens.append(nt)

        rt = ref_tok.Token(token_id=i + 1, class_id=i % 4, position=pos,
                           position_cov=cov, observation_count=obs)
        rt.beliefs.static, rt.beliefs.existence = st, ex
        r_tokens.append(rt)

    expected = [i for i, s in enumerate(specs) if s[4]]
    assert [i for i, t in enumerate(r_tokens) if t.is_stable_landmark()] == expected, (
        "참조의 isStableLandmark 가 명세와 다르다 - 기준부터 틀렸다")
    assert [i for i, t in enumerate(n_tokens) if t.is_stable_landmark()] == expected

    n_nodes = core.ConstellationIndex.build_from(n_tokens, n_ref, 40)
    r_nodes = ref_tok.build_constellation_from(r_tokens, reference, 40)

    assert len(n_nodes) == len(r_nodes) == len(expected), (
        f"노드 수 {len(n_nodes)} vs {len(r_nodes)}, 기대 {len(expected)}")
    for a, b in zip(n_nodes, r_nodes):
        assert a.class_id == b[1], f"{a.class_id} vs {b[1]}"
        assert np.allclose(a.position, b[2], rtol=0, atol=1e-12)
        assert _same(a.sigma, b[3])


def test_constellation_build_from_truncates_by_sigma():
    """max_nodes 절단은 sigma 오름차순이다. 정밀한 것부터 남아야 한다."""
    reference = ref_geom.SE3.identity()
    n_tokens, r_tokens = [], []
    for i in range(12):
        sig = 0.05 + 0.03 * ((7 * i) % 12)      # 일부러 섞인 순서
        pos = np.array([float(i), 0.0, 0.0])
        cov = np.eye(3) * (sig * sig)
        nt = core.WorldToken()
        nt.static_belief, nt.existence_belief = 0.9, 0.9
        nt.observation_count, nt.position, nt.position_cov = 5, pos, cov
        nt.class_id = i
        n_tokens.append(nt)
        rt = ref_tok.Token(token_id=i + 1, class_id=i, position=pos,
                           position_cov=cov, observation_count=5)
        rt.beliefs.static, rt.beliefs.existence = 0.9, 0.9
        r_tokens.append(rt)

    for cap in (3, 5, 12):
        n_nodes = core.ConstellationIndex.build_from(n_tokens,
                                                     core.SE3.identity(), cap)
        r_nodes = ref_tok.build_constellation_from(r_tokens, reference, cap)
        assert len(n_nodes) == len(r_nodes) == min(cap, 12)
        assert [n.class_id for n in n_nodes] == [b[1] for b in r_nodes], (
            f"cap={cap}: 절단 순서가 다르다")
        sigmas = [n.sigma for n in n_nodes]
        assert sigmas == sorted(sigmas), "sigma 오름차순이 아니다"
