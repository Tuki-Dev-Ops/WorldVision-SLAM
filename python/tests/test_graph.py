"""팩터그래프 검증.

해석적으로 답을 아는 문제로 고정한다. 최적화기가 조용히 틀린 답에 수렴하는 것이
SLAM 에서 가장 잡기 어려운 버그이므로, 참값이 있는 문제만 쓴다.
"""

import numpy as np
import pytest

from wme.graph import (
    BetweenPoseFactor, FactorGraph, Huber, ObjectObservationFactor,
    PointPriorFactor, PoseVariable, PosePriorFactor, PositiveVectorVariable,
    SolverOptions, VectorVariable, diagonal, isotropic,
)
from wme.reference.geometry import SE3, so3_exp
from wme.sim.world import CameraModel

RNG = np.random.default_rng(31337)
CAM = CameraModel()


def random_pose(rot=0.3, trans=1.0):
    return SE3(so3_exp(RNG.uniform(-rot, rot, 3)), RNG.uniform(-trans, trans, 3))


# --- 변수 -----------------------------------------------------------------

def test_pose_retract_uses_left_perturbation():
    T = random_pose()
    v = PoseVariable(T)
    d = np.array([0.01, -0.02, 0.03, 0.004, -0.005, 0.006])
    expected = SE3.exp(d) @ T
    assert np.allclose(v.retract(d).pose.matrix(), expected.matrix(), atol=1e-12)


def test_pose_retract_stays_on_manifold():
    v = PoseVariable(random_pose())
    for _ in range(200):
        v = v.retract(RNG.normal(0, 0.1, 6))
    R = v.pose.R
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_positive_variable_is_clipped():
    v = PositiveVectorVariable(np.array([0.3, 0.3, 0.3]), lower=0.05, upper=2.0)
    assert np.all(v.retract(np.array([-10.0, -10.0, -10.0])).value >= 0.05)
    assert np.all(v.retract(np.array([10.0, 10.0, 10.0])).value <= 2.0)


# --- 팩터 -----------------------------------------------------------------

def test_between_factor_residual_is_zero_at_truth():
    Ta, Tb = random_pose(), random_pose()
    measured = Ta.inverse() @ Tb

    f = BetweenPoseFactor("a", "b", measured, isotropic(6, 0.01))
    r = f.residual({"a": PoseVariable(Ta), "b": PoseVariable(Tb)})
    assert np.allclose(r, 0.0, atol=1e-12)


def test_pose_prior_residual_is_zero_at_truth():
    T = random_pose()
    f = PosePriorFactor("x", T, isotropic(6, 0.01))
    assert np.allclose(f.residual({"x": PoseVariable(T)}), 0.0, atol=1e-12)


def test_numeric_jacobian_matches_finite_difference_of_cost():
    """수치 자코비안이 잔차의 실제 변화율과 맞아야 한다."""
    Ta, Tb = random_pose(), random_pose()
    f = BetweenPoseFactor("a", "b", Ta.inverse() @ Tb, isotropic(6, 0.05))
    values = {"a": PoseVariable(Ta), "b": PoseVariable(random_pose())}

    J = f.numeric_jacobians(values)
    delta = RNG.normal(0, 1e-5, 6)

    shifted = dict(values)
    shifted["b"] = values["b"].retract(delta)
    predicted = f.residual(values) + J[1] @ delta
    assert np.allclose(predicted, f.residual(shifted), atol=1e-8)


def test_huber_downweights_large_residuals():
    k = Huber(delta=2.0)
    assert k.weight(1.0) == 1.0                     # |r| = 1 < 2
    assert k.weight(100.0) == pytest.approx(0.2)    # |r| = 10 -> 2/10


# --- 최적화 ---------------------------------------------------------------

def test_single_pose_converges_to_prior():
    g = FactorGraph()
    target = random_pose()
    g.add_variable("x", PoseVariable(SE3.identity()))
    g.add_factor(PosePriorFactor("x", target, isotropic(6, 0.01)))

    result = g.optimize()
    assert result.converged, result.summary()
    d_t, d_r = g.value("x").pose.distance_to(target)
    assert d_t < 1e-7 and d_r < 1e-7


def test_pose_chain_recovers_ground_truth():
    """상대 측정 사슬 + 첫 포즈 앵커 -> 전체 궤적 복원."""
    n = 12
    truth = [SE3.identity()]
    for _ in range(n - 1):
        truth.append(truth[-1] @ SE3(so3_exp([0.0, 0.0, 0.2]), np.array([0.5, 0.05, 0.0])))

    g = FactorGraph()
    for i, T in enumerate(truth):
        # 참값에서 벗어난 초기값을 준다
        init = T @ SE3.exp(RNG.normal(0, 0.05, 6)) if i > 0 else T
        g.add_variable(f"x{i}", PoseVariable(init))

    g.add_factor(PosePriorFactor("x0", truth[0], isotropic(6, 1e-4)))
    for i in range(n - 1):
        measured = truth[i].inverse() @ truth[i + 1]
        g.add_factor(BetweenPoseFactor(f"x{i}", f"x{i+1}", measured, isotropic(6, 0.01)))

    result = g.optimize()
    assert result.converged, result.summary()
    for i, T in enumerate(truth):
        d_t, d_r = g.value(f"x{i}").pose.distance_to(T)
        assert d_t < 1e-5, f"pose {i}: {d_t}"
        assert d_r < 1e-5


def test_loop_closure_distributes_drift():
    """루프 클로저가 누적 드리프트를 전체에 분배해야 한다."""
    n = 10
    truth = [SE3(so3_exp([0.0, 0.0, 2 * np.pi * i / n]),
                 np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n), 0.0]))
             for i in range(n)]

    g = FactorGraph()
    for i in range(n):
        g.add_variable(f"x{i}", PoseVariable(truth[i] @ SE3.exp(RNG.normal(0, 0.02, 6))))
    g.add_factor(PosePriorFactor("x0", truth[0], isotropic(6, 1e-4)))

    for i in range(n - 1):
        g.add_factor(BetweenPoseFactor(
            f"x{i}", f"x{i+1}", truth[i].inverse() @ truth[i + 1], isotropic(6, 0.02)))

    # 루프 클로저 없이 먼저
    g.optimize()
    err_open = max(g.value(f"x{i}").pose.distance_to(truth[i])[0] for i in range(n))

    g.add_factor(BetweenPoseFactor(
        "x0", f"x{n-1}", truth[0].inverse() @ truth[n - 1], isotropic(6, 0.02),
        tier="constellation"))
    g.optimize()
    err_closed = max(g.value(f"x{i}").pose.distance_to(truth[i])[0] for i in range(n))

    assert err_closed <= err_open + 1e-9


def test_unanchored_graph_is_reported_as_degenerate():
    """게이지가 고정되지 않으면 6-DoF 가 관측 불가능해야 하고,
    그것을 조용히 통과시키는 대신 진단으로 드러나야 한다."""
    g = FactorGraph()
    for i in range(4):
        g.add_variable(f"x{i}", PoseVariable(random_pose()))
    for i in range(3):
        g.add_factor(BetweenPoseFactor(f"x{i}", f"x{i+1}", random_pose(), isotropic(6, 0.1)))

    eig, observable = g.degeneracy()
    assert observable < eig.size, "게이지 자유도가 드러나야 한다"
    assert eig.size - observable >= 6


def test_anchoring_removes_degeneracy():
    g = FactorGraph()
    for i in range(4):
        g.add_variable(f"x{i}", PoseVariable(random_pose()))
    g.add_factor(PosePriorFactor("x0", random_pose(), isotropic(6, 1e-3)))
    for i in range(3):
        g.add_factor(BetweenPoseFactor(f"x{i}", f"x{i+1}", random_pose(), isotropic(6, 0.1)))

    eig, observable = g.degeneracy()
    assert observable == eig.size


def test_fixed_variables_are_excluded_from_the_system():
    g = FactorGraph()
    anchor = random_pose()
    g.add_variable("x0", PoseVariable(anchor))
    g.add_variable("x1", PoseVariable(SE3.identity()))
    g.fix("x0")

    measured = anchor.inverse() @ (anchor @ SE3(so3_exp([0, 0, 0.1]), np.array([1.0, 0, 0])))
    g.add_factor(BetweenPoseFactor("x0", "x1", measured, isotropic(6, 0.01)))

    result = g.optimize()
    assert result.converged
    # 고정 변수는 움직이지 않아야 한다
    assert np.allclose(g.value("x0").pose.matrix(), anchor.matrix(), atol=1e-15)
    assert "x0" not in g.covariance()


def test_robust_kernel_limits_outlier_influence():
    """단일 이상치 측정이 궤적 전체를 끌고 가면 안 된다."""
    n = 8
    truth = [SE3(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(n)]

    def build(kernel):
        g = FactorGraph()
        for i in range(n):
            g.add_variable(f"x{i}", PoseVariable(truth[i]))
        g.add_factor(PosePriorFactor("x0", truth[0], isotropic(6, 1e-4)))
        for i in range(n - 1):
            g.add_factor(BetweenPoseFactor(
                f"x{i}", f"x{i+1}", truth[i].inverse() @ truth[i + 1],
                isotropic(6, 0.05), kernel=kernel))
        # 완전히 틀린 루프 클로저
        bogus = SE3(np.eye(3), np.array([0.0, 8.0, 0.0]))
        g.add_factor(BetweenPoseFactor("x0", f"x{n-1}", bogus,
                                       isotropic(6, 0.05), kernel=kernel))
        g.optimize()
        return max(g.value(f"x{i}").pose.distance_to(truth[i])[0] for i in range(n))

    assert build(Huber(delta=1.0)) < build(None)


# --- 객체 관측 팩터 --------------------------------------------------------

def test_object_factor_residual_is_zero_at_truth():
    from wme.calib.shape import predict_observation

    pose = SE3(so3_exp([0.1, -0.05, 0.2]), np.array([0.5, -0.3, 0.2]))
    point = np.array([1.2, 0.4, 3.0])
    extent = np.array([0.3, 0.25, 0.2])

    z = predict_observation(point, extent, pose.inverse(), CAM)
    assert z is not None

    f = ObjectObservationFactor("p", "l", "e", z, CAM, diagonal([2, 2, 2, 2, 0.05]))
    r = f.residual({"p": PoseVariable(pose),
                    "l": VectorVariable(point),
                    "e": PositiveVectorVariable(extent)})
    assert np.allclose(r, 0.0, atol=1e-9)


def test_object_observations_recover_position_and_extent():
    """여러 시점에서 본 박스만으로 위치와 3D 치수가 결정되어야 한다.

    깊이 방향 치수는 단일 뷰에서 관측 불가능하다 - 시점이 모여야 풀린다.
    M3 에서 이 성질이 편향 제거의 핵심이었다.
    """
    from wme.calib.shape import predict_observation

    point = np.array([0.6, -0.2, 0.4])
    extent = np.array([0.35, 0.28, 0.22])

    poses = []
    for a in np.linspace(0.0, 1.6, 10):
        eye = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.9])
        z = point - eye
        z = z / np.linalg.norm(z)
        x = np.cross(np.array([0.0, 0.0, 1.0]), z)
        x = x / np.linalg.norm(x)
        poses.append(SE3(np.column_stack([x, np.cross(z, x), z]), eye))

    g = FactorGraph()
    for i, T in enumerate(poses):
        g.add_variable(f"p{i}", PoseVariable(T))
        g.fix(f"p{i}")                       # 포즈는 알려져 있다고 두고 객체만 푼다

    g.add_variable("l", VectorVariable(point + np.array([0.25, -0.2, 0.15])))
    g.add_variable("e", PositiveVectorVariable(np.array([0.2, 0.2, 0.2])))

    info = diagonal([1.5, 1.5, 1.5, 1.5, 0.02])
    for i, T in enumerate(poses):
        z = predict_observation(point, extent, T.inverse(), CAM)
        assert z is not None
        g.add_factor(ObjectObservationFactor(f"p{i}", "l", "e", z, CAM, info))

    result = g.optimize(SolverOptions(max_iterations=80))
    assert result.converged, result.summary()
    assert np.allclose(g.value("l").value, point, atol=5e-3)
    assert np.allclose(g.value("e").value, extent, atol=2e-2)


def test_object_slam_reduces_odometry_drift():
    """오도메트리 적분 초기값보다 최적화 결과가 나아야 한다.

    이게 안 되면 그래프가 아무 일도 하지 않는 것이다.
    """
    from wme.calib import CalibratedNoise
    from wme.graph.slam import SlamConfig, solve_object_slam
    from wme.sim import CameraTrajectory, constant_condition, run, static_room

    seq = run(static_room(12, seed=3),
              CameraTrajectory.loop(radius=3.2, laps=1.0, n=40),
              constant_condition(), seed=3)
    noise = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)

    r = solve_object_slam(seq, noise, SlamConfig(seed=3, observation_stride=2))

    before = float(np.sqrt((r.initial_pose_errors() ** 2).mean()))
    after = float(np.sqrt((r.pose_errors() ** 2).mean()))
    assert after < before, f"{after:.4f} !< {before:.4f}"
    assert len(r.landmark_keys) >= 5
    assert all(np.all(np.isfinite(c)) for c in r.covariances.values())


def test_object_slam_anchor_stays_at_truth():
    """앵커가 움직이면 추정 좌표계와 참값 좌표계가 어긋나 평가가 무의미해진다."""
    from wme.calib import CalibratedNoise
    from wme.graph.slam import SlamConfig, solve_object_slam
    from wme.sim import CameraTrajectory, constant_condition, run, static_room

    seq = run(static_room(10, seed=1),
              CameraTrajectory.loop(radius=3.0, laps=1.0, n=30),
              constant_condition(), seed=1)
    r = solve_object_slam(seq, CalibratedNoise(c_px=2.0, c_d=0.006),
                          SlamConfig(seed=1, observation_stride=2))

    d_t, d_r = r.graph.value(r.pose_keys[0]).pose.distance_to(r.truth_poses[0])
    assert d_t < 1e-3 and d_r < 1e-3


def test_landmark_covariance_grows_with_degradation():
    """조건이 나빠지면 보고되는 불확실성도 커져야 한다.

    적응 로직이 팩터그래프까지 실제로 전달되는지 확인하는 계약이다.
    """
    from wme.calib import CalibratedNoise
    from wme.graph.slam import SlamConfig, solve_object_slam
    from wme.sim import CameraTrajectory, constant_condition, run, static_room

    noise = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)
    world = static_room(12, seed=7)
    traj = CameraTrajectory.loop(radius=3.2, laps=1.0, n=40)

    def mean_trace(level):
        seq = run(world, traj, constant_condition(haze=level), seed=7)
        r = solve_object_slam(seq, noise, SlamConfig(seed=7, observation_stride=2))
        return float(np.mean([np.trace(c) for c in r.covariances.values()]))

    assert mean_trace(0.8) > mean_trace(0.0)


def test_covariance_is_marginal_not_conditional():
    """주변 공분산은 조건부보다 커야 한다. 다른 변수를 안다고 가정하면 과신이다."""
    from wme.calib.shape import predict_observation

    point = np.array([0.4, 0.0, 0.5])
    extent = np.array([0.3, 0.3, 0.25])
    poses = [SE3(np.eye(3), np.array([0.0, -3.0 + 0.4 * i, 0.5])) for i in range(6)]
    # 카메라가 +y 를 보도록 회전
    R = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])])
    poses = [SE3(R, p.t) for p in poses]

    info = diagonal([2.0, 2.0, 2.0, 2.0, 0.05])

    def solve(fix_extent: bool):
        g = FactorGraph()
        for i, T in enumerate(poses):
            g.add_variable(f"p{i}", PoseVariable(T))
            g.fix(f"p{i}")
        g.add_variable("l", VectorVariable(point))
        g.add_variable("e", PositiveVectorVariable(extent))
        if fix_extent:
            g.fix("e")
        for i, T in enumerate(poses):
            z = predict_observation(point, extent, T.inverse(), CAM)
            if z is not None:
                g.add_factor(ObjectObservationFactor(f"p{i}", "l", "e", z, CAM, info))
        return np.trace(g.covariance(["l"])["l"])

    assert solve(fix_extent=False) > solve(fix_extent=True)
