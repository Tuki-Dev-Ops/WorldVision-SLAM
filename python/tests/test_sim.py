"""시뮬레이션 하네스 검증.

시뮬레이터의 참 잡음 모델이 우리가 의도한 그대로여야 한다. 그래야 추정기가
그것을 복원하는지(Sigma_k(E) 보정)를 판정할 수 있다. 시뮬레이터가 틀리면
보정 실험 전체가 무의미해진다.
"""

import numpy as np
import pytest

from wme.reference.environment import Evidence
from wme.sim import (
    CameraModel, CameraTrajectory, SimObject, SimWorld, severity_of,
    scenario_condition_sweep, scenario_degradation_burst, scenario_dynamic,
    scenario_revisit_with_changes, scenario_static, static_room,
)


# --- 세계 / 궤적 -----------------------------------------------------------

def test_static_object_position_is_constant():
    o = SimObject(1, 56, "chair", np.array([1.0, 2.0, 0.5]))
    assert np.allclose(o.position_at(0.0), o.position_at(100.0))
    assert np.allclose(o.velocity_at(50.0), 0.0, atol=1e-9)


def test_waypoint_interpolation_and_velocity():
    o = SimObject(1, 0, "person", np.zeros(3), is_agent=True,
                  waypoints=[(0.0, np.array([0.0, 0, 0])), (10.0, np.array([10.0, 0, 0]))])
    assert np.allclose(o.position_at(5.0), [5.0, 0, 0])
    assert np.allclose(o.velocity_at(5.0), [1.0, 0, 0], atol=1e-6)   # 1 m/s
    # 구간 밖에서는 끝점으로 고정
    assert np.allclose(o.position_at(-5.0), [0.0, 0, 0])
    assert np.allclose(o.position_at(50.0), [10.0, 0, 0])


def test_presence_window():
    o = SimObject(1, 56, "chair", np.zeros(3), present_from=5.0, present_until=10.0)
    assert o.position_at(4.9) is None
    assert o.position_at(7.0) is not None
    assert o.position_at(10.0) is None


def test_loop_trajectory_returns_to_start():
    traj = CameraTrajectory.loop(radius=3.0, laps=1.0, n=100)
    d_t, _ = traj.poses[0].distance_to(traj.poses[-1])
    assert d_t < 1e-9


def test_camera_looks_at_target():
    traj = CameraTrajectory.loop(radius=3.0, height=1.0, laps=1.0, n=8)
    for pose in traj.poses:
        # 카메라 z축이 중심을 향해야 한다
        to_center = -pose.t
        to_center[2] = 0.0
        z_axis = pose.R[:, 2].copy()
        z_axis[2] = 0.0
        assert float(z_axis @ to_center) > 0.0


def test_revisit_trajectory_has_a_time_gap():
    traj = CameraTrajectory.revisit(n_per_visit=50, gap_seconds=30.0)
    gaps = np.diff(traj.stamps)
    assert gaps.max() > 25.0
    assert len(traj) == 100


# --- 관측 모델 -------------------------------------------------------------

def test_sequence_is_deterministic():
    """같은 seed 면 완전히 동일해야 한다. 재현되지 않는 실험은 실험이 아니다."""
    a = scenario_static(seed=7)
    b = scenario_static(seed=7)

    assert len(a) == len(b)
    for da, db in zip(a.detections, b.detections):
        assert len(da) == len(db)
        for x, y in zip(da.items, db.items):
            assert x.class_id == y.class_id
            assert np.allclose(x.box, y.box)
            assert x.confidence == pytest.approx(y.confidence)


def test_different_seeds_differ():
    a = scenario_static(seed=1)
    b = scenario_static(seed=2)
    assert sum(len(d) for d in a.detections) != sum(len(d) for d in b.detections)


def test_clear_conditions_detect_most_visible_objects():
    seq = scenario_static(seed=0)
    detected, visible = 0, 0
    for truth in seq.truths:
        visible += len(truth.visible_objects)
        detected += len(truth.detected_objects)
    assert visible > 0
    assert detected / visible > 0.85, "맑은 조건에서 검출률이 너무 낮다"


def test_detection_rate_falls_with_severity():
    """열화가 심할수록 검출이 줄어야 한다. 관측 모델의 기본 계약."""
    rates = []
    for seq in scenario_condition_sweep("haze", levels=5, seed=0):
        det = sum(len(t.detected_objects) for t in seq.truths)
        vis = sum(len(t.visible_objects) for t in seq.truths)
        rates.append(det / max(1, vis))

    assert rates[0] > rates[-1]
    # 단조는 아니어도(샘플링 잡음) 전반적 하락은 뚜렷해야 한다
    assert rates[-1] < rates[0] * 0.75, rates


def test_depth_noise_grows_with_severity():
    """이것이 Sigma_k(E) 보정이 복원해야 할 참 모델이다."""
    sigmas = []
    for seq in scenario_condition_sweep("haze", levels=4, seed=0):
        s = [x for t in seq.truths for x in t.detection_depth_sigma
             if x < 20.0]     # 오검출의 무정보 sigma 제외
        sigmas.append(float(np.mean(s)))

    assert all(a < b for a, b in zip(sigmas, sigmas[1:])), sigmas


def test_false_alarms_grow_with_severity():
    counts = []
    for seq in scenario_condition_sweep("haze", levels=4, seed=0):
        counts.append(sum(1 for t in seq.truths for o in t.detection_to_object if o == -1))
    assert counts[-1] > counts[0]


def test_depth_noise_follows_the_quadratic_range_model():
    """sigma_z = coeff * z^2. 실제 스테레오/ToF 센서의 형태이고,
    보정 실험이 복원해야 할 참 모델이다. 분위수 비교보다 관계 자체를 검증한다.
    """
    from wme.sim import SensorConfig

    seq = scenario_static(seed=3)
    pairs = [(d, s) for t in seq.truths
             for d, s, o in zip(t.detection_depth, t.detection_depth_sigma,
                                t.detection_to_object) if o >= 0]
    assert len(pairs) > 50

    depths = np.array([p[0] for p in pairs])
    sigmas = np.array([p[1] for p in pairs])

    # sigma 는 참 z 로 계산되고 depths 는 잡음 섞인 측정값이라 완전 상관은 아니다
    assert np.corrcoef(sigmas, depths ** 2)[0, 1] > 0.95

    # 맑은 조건이므로 계수가 설정값에 가까워야 한다
    coeff = float(np.mean(sigmas / depths ** 2))
    assert coeff == pytest.approx(SensorConfig().depth_noise_coeff, rel=0.15)


def test_ground_truth_association_is_aligned_with_detections():
    seq = scenario_static(seed=0)
    for det, truth in zip(seq.detections, seq.truths):
        assert len(det.items) == len(truth.detection_to_object)
        assert len(det.items) == len(truth.detection_depth)


def test_severity_is_monotone_in_each_channel():
    for channel in ["haze", "darkness", "motion_blur", "rain_streak", "lens_dirt"]:
        prev = -1.0
        for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = severity_of(Evidence(**{channel: level}))
            assert s >= prev - 1e-12, channel
            prev = s


def test_occlusion_hides_the_farther_object():
    """가까운 큰 물체가 뒤의 물체를 가려야 한다."""
    from wme.reference.geometry import SE3
    from wme.sim import SimSensor

    camera = CameraModel()
    sensor = SimSensor(camera, seed=0)

    near = SimObject(1, 56, "chair", np.array([0.0, 0.0, 2.0]),
                     extent=np.array([1.2, 1.2, 1.2]))
    far = SimObject(2, 41, "cup", np.array([0.0, 0.0, 5.0]),
                    extent=np.array([0.2, 0.2, 0.2]))

    # 카메라는 원점에서 +z 를 본다
    world = SimWorld([near, far])
    _, truth = sensor.observe(world, SE3.identity(), 1.0, Evidence(), 0)

    assert 1 in truth.visible_objects
    assert 2 not in truth.visible_objects, "가려진 객체가 가시로 표시됐다"


def test_out_of_frustum_objects_are_not_visible():
    from wme.reference.geometry import SE3
    from wme.sim import SimSensor

    behind = SimObject(1, 56, "chair", np.array([0.0, 0.0, -3.0]))
    world = SimWorld([behind])
    _, truth = SimSensor(CameraModel(), seed=0).observe(world, SE3.identity(), 1.0,
                                                        Evidence(), 0)
    assert not truth.visible_objects


# --- 시나리오 --------------------------------------------------------------

def test_dynamic_scenario_contains_moving_agents():
    seq = scenario_dynamic(seed=0, walkers=3)
    agents = [o for o in seq.world.objects if o.is_agent]
    assert len(agents) == 3
    for a in agents:
        assert np.linalg.norm(a.position_at(9.0) - a.position_at(1.0)) > 1.0


def test_revisit_scenario_produces_typed_changes():
    seq = scenario_revisit_with_changes(seed=0)
    kinds = {c.kind for c in seq.changes}
    assert kinds == {"moved", "removed", "added"}

    for c in seq.changes:
        obj = seq.world.by_id(c.object_id)
        assert obj is not None
        if c.kind == "removed":
            assert obj.position_at(c.time + 1.0) is None
        elif c.kind == "added":
            assert obj.position_at(c.time - 1.0) is None
            assert obj.position_at(c.time + 1.0) is not None
        else:
            before = obj.position_at(c.time - 1.0)
            after = obj.position_at(c.time + 1.0)
            assert np.linalg.norm(after - before) > 0.5


def test_changed_objects_are_never_agents():
    """사람이 움직인 것은 '장면 변화'가 아니다."""
    seq = scenario_revisit_with_changes(seed=0)
    for c in seq.changes:
        obj = seq.world.by_id(c.object_id)
        assert not obj.is_agent


def test_burst_scenario_has_clean_degraded_clean_phases():
    seq = scenario_degradation_burst(seed=0, channel="haze", level=0.9)
    sev = np.array([severity_of(t.evidence) for t in seq.truths])
    assert sev[0] < 0.05
    assert sev.max() > 0.5
    assert sev[-1] < 0.05, "조건이 걷혀야 복구를 측정할 수 있다"


def test_condition_sweep_shares_world_and_trajectory():
    """조건만 다르고 나머지가 같아야 통제된 비교가 된다."""
    seqs = scenario_condition_sweep("haze", levels=3, seed=0)
    ids = [{o.object_id for o in s.world.objects} for s in seqs]
    assert ids[0] == ids[1] == ids[2]
    for s in seqs[1:]:
        assert np.allclose(s.trajectory.stamps, seqs[0].trajectory.stamps)
        assert np.allclose(s.trajectory.poses[0].t, seqs[0].trajectory.poses[0].t)


def test_sequence_severity_property():
    clear = scenario_static(seed=0)
    foggy = scenario_static(seed=0, haze=0.8)
    assert clear.severity < 0.05
    assert foggy.severity > 0.4
