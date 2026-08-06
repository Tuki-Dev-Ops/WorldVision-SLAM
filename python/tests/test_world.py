"""신념층 검증.

manifesto 의 여섯 약속 중 자료구조로 강제할 수 있는 것들을 테스트로 고정한다.
특히 두 가지가 중요하다.
  - 예측은 관측과 절대 섞이지 않는다
  - 센서가 나빠졌다고 세계가 변했다고 보고하지 않는다
"""

import numpy as np
import pytest

from wme.eval.metrics import change_metrics
from wme.world import (
    ChangeConfig, ChangeDetector, Forecast, PredictionConfig, PredictionEngine,
    TokenBelief, WorldSnapshot, WorldState,
)


def belief(token_id, pos, sigma=0.05, existence=0.9, static=0.9, obs=10,
           lifecycle="active", velocity=None):
    return TokenBelief(
        token_id=token_id, class_id=56, class_name="chair",
        position=np.asarray(pos, float),
        covariance=np.eye(3) * sigma ** 2,
        velocity=np.zeros(3) if velocity is None else np.asarray(velocity, float),
        existence=existence, identity=0.9, static_belief=static,
        lifecycle=lifecycle, observation_count=obs, last_seen=1.0,
    )


def snapshot(beliefs, version=1, stamp=1.0):
    return WorldSnapshot(version, stamp, {b.token_id: b for b in beliefs})


# --- WorldState ------------------------------------------------------------

def test_snapshots_are_immutable():
    """스냅샷을 넘겨받은 쪽이 고칠 수 있으면 '그때 무엇을 믿었는가'를 못 묻는다."""
    b = belief(1, [1.0, 0, 0])
    with pytest.raises(Exception):
        b.position = np.zeros(3)          # frozen dataclass


def test_with_returns_a_new_belief():
    a = belief(1, [1.0, 0, 0])
    c = a.with_(existence=0.2)
    assert a.existence == 0.9 and c.existence == 0.2
    assert a is not c


def test_commit_advances_version_and_preserves_history():
    ws = WorldState()
    ws.put(belief(1, [0.0, 0, 0]))
    s1 = ws.commit(1.0)

    ws.update(1, position=np.array([1.0, 0, 0]))
    s2 = ws.commit(2.0)

    assert s1.version == 1 and s2.version == 2
    # 과거 스냅샷은 갱신에 영향받지 않아야 한다
    assert np.allclose(s1.get(1).position, [0.0, 0, 0])
    assert np.allclose(s2.get(1).position, [1.0, 0, 0])


def test_time_query_returns_the_state_as_it_was():
    """'10초 전 세계는 어땠는가' 가 재생이 아니라 질의가 되는 지점."""
    ws = WorldState()
    for i in range(5):
        ws.put(belief(1, [float(i), 0, 0]))
        ws.commit(float(i))

    assert np.allclose(ws.at(2.0).get(1).position, [2.0, 0, 0])
    assert np.allclose(ws.at(2.9).get(1).position, [2.0, 0, 0])
    assert ws.at(-1.0) is None


def test_commit_rejects_time_going_backwards():
    ws = WorldState()
    ws.commit(5.0)
    with pytest.raises(ValueError):
        ws.commit(4.0)


def test_history_is_bounded():
    ws = WorldState(history_capacity=10)
    for i in range(50):
        ws.commit(float(i))
    assert len(ws.history) == 10
    assert ws.history[-1].stamp == 49.0


def test_iteration_is_deterministic():
    """해시 순서가 결과에 새면 재현이 불가능해진다."""
    s = snapshot([belief(i, [float(i), 0, 0]) for i in (7, 3, 11, 1)])
    assert [t.token_id for t in s] == [1, 3, 7, 11]


# --- 변화 검출 -------------------------------------------------------------

def test_no_change_when_nothing_changed():
    ref = snapshot([belief(1, [0, 0, 0]), belief(2, [2, 0, 0])])
    cur = snapshot([belief(1, [0, 0, 0]), belief(2, [2, 0, 0])], version=2, stamp=30.0)
    assert ChangeDetector().compare(ref, cur) == []


def test_detects_moved_object():
    ref = snapshot([belief(1, [0, 0, 0])])
    cur = snapshot([belief(1, [1.5, 0, 0])], version=2, stamp=30.0)

    events = ChangeDetector().compare(ref, cur)
    assert len(events) == 1
    assert events[0].kind == "moved" and events[0].object_id == 1


def test_detects_removed_object():
    ref = snapshot([belief(1, [0, 0, 0])])
    cur = snapshot([belief(1, [0, 0, 0], existence=0.1)], version=2, stamp=30.0)

    events = ChangeDetector().compare(ref, cur)
    assert [e.kind for e in events] == ["removed"]


def test_detects_displaced_lifecycle_as_removed():
    ref = snapshot([belief(1, [0, 0, 0])])
    cur = snapshot([belief(1, [0, 0, 0], lifecycle="displaced")], version=2, stamp=30.0)
    assert [e.kind for e in ChangeDetector().compare(ref, cur)] == ["removed"]


def test_detects_added_object():
    ref = snapshot([belief(1, [0, 0, 0])])
    cur = snapshot([belief(1, [0, 0, 0]), belief(2, [3, 0, 0])], version=2, stamp=30.0)

    events = ChangeDetector().compare(ref, cur)
    assert [(e.kind, e.object_id) for e in events] == [("added", 2)]


def test_does_not_report_absence_as_removal():
    """스냅샷에서 빠진 것은 '추적을 포기했다'이지 '없어졌다'가 아니다.

    부재를 주장하려면 lifecycle=displaced 로 명시해야 한다.
    """
    ref = snapshot([belief(1, [0, 0, 0])])
    cur = snapshot([], version=2, stamp=30.0)
    assert ChangeDetector().compare(ref, cur) == []


def test_moving_objects_are_not_scene_changes():
    """사람이 걸어간 것은 '장면이 바뀐 것'이 아니다."""
    ref = snapshot([belief(1, [0, 0, 0], static=0.1)])
    cur = snapshot([belief(1, [2.0, 0, 0], static=0.1)], version=2, stamp=30.0)
    assert ChangeDetector().compare(ref, cur) == []


def test_poorly_observed_tokens_are_not_judged():
    """안 본 것과 없어진 것은 다르다."""
    ref = snapshot([belief(1, [0, 0, 0], obs=1)])
    cur = snapshot([belief(1, [0, 0, 0], obs=1, existence=0.05)], version=2, stamp=30.0)
    assert ChangeDetector().compare(ref, cur) == []


# --- 이것이 이 모듈의 유일한 비자명 요구사항 --------------------------------

def test_sensor_degradation_does_not_look_like_world_change():
    """안개가 꼈을 뿐인데 세계가 변했다고 보고하는 시스템은 쓸 수 없다.

    문턱을 절대 거리가 아니라 믿음의 공분산으로 잡았으므로, 센서가 나빠져
    공분산이 커지면 문턱도 함께 넓어져야 한다.
    """
    rng = np.random.default_rng(7)
    detector = ChangeDetector()

    truth = [np.array([float(i), 0.0, 0.0]) for i in range(8)]
    false_alarms = 0
    trials = 40

    for _ in range(trials):
        # 열화 전: 정밀한 관측
        ref = snapshot([belief(i, p + rng.normal(0, 0.02, 3), sigma=0.02)
                        for i, p in enumerate(truth)])
        # 열화 후: 같은 세계, 잡음만 10배. 공분산도 그만큼 커진다.
        cur = snapshot([belief(i, p + rng.normal(0, 0.20, 3), sigma=0.20)
                        for i, p in enumerate(truth)], version=2, stamp=30.0)
        false_alarms += len(detector.compare(ref, cur))

    rate = false_alarms / (trials * len(truth))
    assert rate < 0.02, f"오보율 {rate:.3f} - 열화를 변화로 오인한다"


def test_absolute_threshold_would_fail_the_same_test():
    """대조군: 공분산을 무시하고 절대 거리로 자르면 오보가 터진다.

    이 테스트가 위 테스트의 의미를 만든다 - 그냥 통과한 게 아니라
    설계 선택 때문에 통과했다는 것을 보인다.
    """
    rng = np.random.default_rng(7)
    truth = [np.array([float(i), 0.0, 0.0]) for i in range(8)]

    absolute_hits = 0
    for _ in range(40):
        for p in truth:
            a = p + rng.normal(0, 0.02, 3)
            b = p + rng.normal(0, 0.20, 3)
            if np.linalg.norm(b - a) > 0.15:      # min_move_metres 를 절대 기준으로
                absolute_hits += 1

    assert absolute_hits > 40, "대조군이 오보를 내지 않으면 비교가 성립하지 않는다"


def test_change_detector_output_feeds_the_metric():
    """검출 결과가 평가 지표에 그대로 들어가야 루프가 닫힌다."""
    ref = snapshot([belief(1, [0, 0, 0]), belief(2, [2, 0, 0]), belief(3, [4, 0, 0])])
    cur = snapshot([belief(1, [1.8, 0, 0]),                       # moved
                    belief(2, [2, 0, 0], lifecycle="displaced"),  # removed
                    belief(4, [6, 0, 0])],                        # added
                   version=2, stamp=30.0)

    reported = ChangeDetector().compare(ref, cur)
    truth = [
        # 참값은 변화가 일어난 시각 기준. 검출은 그 뒤에 온다.
        type(reported[0])(1, "moved", 29.0),
        type(reported[0])(2, "removed", 29.0),
        type(reported[0])(4, "added", 29.0),
    ]
    result = change_metrics(truth, reported, time_tolerance=5.0)
    assert result.detected == 3
    assert result.false_change_rate == 0.0


# --- 예측 -----------------------------------------------------------------

def test_forecast_is_a_separate_type_from_belief():
    """예측이 관측 필드에 절대 쓰이지 않도록 타입으로 강제한다."""
    b = belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1)
    f = PredictionEngine().forecast(b, 2.0)
    assert isinstance(f, Forecast)
    assert not isinstance(f, TokenBelief)
    # 원본은 건드리지 않는다
    assert np.allclose(b.position, [0, 0, 0])


def test_forecast_extrapolates_motion():
    b = belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1)
    f = PredictionEngine().forecast(b, 1.0)
    assert 0.5 < f.position[0] < 1.1        # 감쇠가 있으므로 1.0 보다 약간 작다


def test_uncertainty_grows_with_horizon():
    """예측 불확실성이 자라지 않으면 그 예측은 정직하지 않다."""
    b = belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1)
    engine = PredictionEngine()
    sigmas = [engine.forecast(b, h).sigma for h in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(a < c for a, c in zip(sigmas, sigmas[1:])), sigmas


def test_static_objects_get_tighter_forecasts():
    engine = PredictionEngine()
    moving = belief(1, [0, 0, 0], static=0.05)
    parked = belief(2, [0, 0, 0], static=0.95)
    assert engine.forecast(parked, 3.0).sigma < engine.forecast(moving, 3.0).sigma


def test_surprise_is_low_when_prediction_holds():
    b = belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1)
    engine = PredictionEngine()
    f = engine.forecast(b, 1.0)
    assert not engine.is_surprising(f, f.position + np.array([0.01, 0, 0]))


def test_surprise_fires_when_the_world_changes():
    """놀라움이 변화의 조기 신호다. 이걸 뭉개면 월드 모델이 아니다."""
    b = belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1)
    engine = PredictionEngine()
    f = engine.forecast(b, 1.0)
    assert engine.is_surprising(f, f.position + np.array([5.0, 0, 0]))


def test_surprise_accounts_for_forecast_uncertainty():
    """같은 오차라도 불확실한 예측에서는 덜 놀라야 한다."""
    engine = PredictionEngine()
    confident = engine.forecast(belief(1, [0, 0, 0], sigma=0.02, static=0.95), 0.5)
    vague = engine.forecast(belief(2, [0, 0, 0], sigma=0.5, static=0.05), 0.5)

    offset = np.array([0.4, 0, 0])
    assert (engine.surprise(confident, confident.position + offset)
            > engine.surprise(vague, vague.position + offset))


def test_low_observation_count_lowers_reliability():
    engine = PredictionEngine()
    fresh = engine.forecast(belief(1, [0, 0, 0], obs=1), 1.0)
    settled = engine.forecast(belief(2, [0, 0, 0], obs=20), 1.0)
    assert fresh.reliability < settled.reliability


def test_score_summarises_forecast_quality():
    engine = PredictionEngine()
    snap = snapshot([belief(1, [0, 0, 0], velocity=[1.0, 0, 0], static=0.1),
                     belief(2, [5, 0, 0], static=0.9)])
    forecasts = engine.forecast_all(snap, 1.0)

    perfect = {f.token_id: f.position for f in forecasts}
    assert engine.score(forecasts, perfect)["ade"] == pytest.approx(0.0, abs=1e-12)

    shifted = {f.token_id: f.position + np.array([0.3, 0, 0]) for f in forecasts}
    assert engine.score(forecasts, shifted)["ade"] == pytest.approx(0.3, rel=1e-9)


# --- 파이프라인의 정적 판정 채널 -------------------------------------------

def test_pipeline_static_channel_actually_fires_and_discriminates():
    """판정 창(0.5 s)이 프레임 간격(0.05 s)보다 길다.

    프레임 간격을 그대로 dt 로 넘기면 update_static 이 한 번도 실행되지 않고
    믿음이 전부 0.5 에 머문다 - 채널이 조용히 죽는다. 출력이 그럴듯해 보이므로
    (모든 값이 사전분포) 지표만 봐서는 드러나지 않는다. 10.4 의 규칙:
    알고리즘을 판단하기 전에 측정이 변별하는지부터 확인한다.
    """
    from wme.sim import scenarios
    from wme.world.pipeline import PipelineConfig, WorldPipeline

    beliefs = {}
    for name in ("scenario_static", "scenario_dynamic"):
        pipeline = WorldPipeline(PipelineConfig())
        pipeline.process(getattr(scenarios, name)())
        beliefs[name] = list(pipeline._beliefs.values())

    for name, bs in beliefs.items():
        assert sum(b.static_diag.updates for b in bs) > 0, f"{name}: 갱신이 0회다"

    def judged(bs):
        return [b.static for b in bs if b.static_diag.updates > 0]

    static_max = max(judged(beliefs["scenario_static"]))
    dynamic_min = min(judged(beliefs["scenario_dynamic"]))
    assert static_max > 0.7, "정지 장면에서 정적 증거가 쌓이지 않는다"
    assert dynamic_min < 0.45, "동적 장면에서 부호가 뒤집히지 않는다"
