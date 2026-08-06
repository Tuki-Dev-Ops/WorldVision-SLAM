"""궤적 평가 지표 검증.

지표가 틀리면 모든 실험 결과가 틀린다. 해석적으로 답을 아는 케이스로 고정한다.
"""

import numpy as np
import pytest

from wme.eval.trajectory import (
    Trajectory, associate, evaluate_ate, evaluate_rpe, umeyama,
)
from wme.eval.tum import load_trajectory, save_trajectory
from wme.reference.geometry import SE3, so3_exp

RNG = np.random.default_rng(1234)


def make_traj(n=50, dt=0.1, noise=0.0, transform: SE3 | None = None, scale=1.0):
    """원호를 그리는 합성 궤적."""
    stamps, poses = [], []
    for i in range(n):
        t = i * dt
        angle = 0.05 * i
        p = np.array([np.cos(angle) * 2.0, np.sin(angle) * 2.0, 0.1 * i])
        R = so3_exp([0.0, 0.0, angle])
        pose = SE3(R, p * scale)
        if transform is not None:
            pose = transform @ pose
        if noise > 0.0:
            pose = SE3(pose.R, pose.t + RNG.normal(0.0, noise, 3))
        stamps.append(t)
        poses.append(pose)
    return Trajectory(np.array(stamps), poses)


def test_ate_of_identical_trajectories_is_zero():
    gt = make_traj()
    r = evaluate_ate(gt, gt)
    assert r.rmse < 1e-9
    assert r.count == len(gt)


def test_ate_recovers_after_rigid_misalignment():
    """정렬 후에는 강체변환만큼의 차이는 사라져야 한다."""
    gt = make_traj()
    T = SE3(so3_exp([0.3, -0.2, 1.1]), np.array([5.0, -3.0, 2.0]))
    est = Trajectory(gt.stamps, [T @ p for p in gt.poses])

    assert evaluate_ate(est, gt, align=True).rmse < 1e-9
    assert evaluate_ate(est, gt, align=False).rmse > 1.0


def test_ate_with_scale_handles_monocular():
    """단안 SLAM 은 scale 자유도를 가지므로 sim(3) 정렬이 필요하다."""
    gt = make_traj()
    est = Trajectory(gt.stamps, [SE3(p.R, p.t * 2.5) for p in gt.poses])

    assert evaluate_ate(est, gt, with_scale=True).rmse < 1e-9
    assert evaluate_ate(est, gt, with_scale=False).rmse > 0.5


def test_ate_reports_scale_factor():
    gt = make_traj()
    est = Trajectory(gt.stamps, [SE3(p.R, p.t * 0.4) for p in gt.poses])
    r = evaluate_ate(est, gt, with_scale=True)
    assert r.scale == pytest.approx(2.5, rel=1e-6)


def test_ate_scales_with_noise():
    gt = make_traj()
    prev = -1.0
    for sigma in [0.0, 0.01, 0.05, 0.1]:
        est = Trajectory(gt.stamps, [SE3(p.R, p.t + RNG.normal(0, sigma, 3))
                                     for p in gt.poses])
        rmse = evaluate_ate(est, gt).rmse
        assert rmse > prev
        prev = rmse


def test_rpe_of_identical_trajectories_is_zero():
    gt = make_traj()
    r = evaluate_rpe(gt, gt, delta=1.0)
    assert r.trans_rmse < 1e-9
    assert r.rot_rmse_deg < 1e-9


def test_rpe_is_invariant_to_global_transform():
    """RPE 는 국소 지표라 전역 강체변환에 무관해야 한다."""
    gt = make_traj()
    T = SE3(so3_exp([0.5, 0.5, 0.5]), np.array([10.0, 10.0, 10.0]))
    est = Trajectory(gt.stamps, [T @ p for p in gt.poses])

    r = evaluate_rpe(est, gt, delta=1.0)
    assert r.trans_rmse < 1e-9


def test_rpe_detects_local_drift_that_ate_alignment_hides():
    """일정 드리프트는 ATE 정렬로 일부 가려지지만 RPE 는 그대로 잡아낸다."""
    gt = make_traj(n=60)
    drifted = []
    for i, p in enumerate(gt.poses):
        drifted.append(SE3(p.R, p.t + np.array([0.01 * i, 0.0, 0.0])))
    est = Trajectory(gt.stamps, drifted)

    r = evaluate_rpe(est, gt, delta=1.0)
    assert r.trans_rmse > 0.05


def test_associate_does_not_reuse_groundtruth():
    """gt 를 중복 사용하면 저프레임레이트 추정치가 실제보다 좋게 나온다."""
    gt = make_traj(n=20, dt=0.1)
    # 추정치가 gt 보다 촘촘한 경우
    est = make_traj(n=40, dt=0.05)

    e, g = associate(est, gt, max_difference=0.02)
    assert len(e) == len(g)
    assert len(set(g.stamps.tolist())) == len(g), "같은 gt 포즈가 두 번 쓰였다"


def test_associate_respects_max_difference():
    gt = make_traj(n=10, dt=1.0)
    est = Trajectory(gt.stamps + 0.5, gt.poses)     # 0.5 s 어긋남
    assert len(associate(est, gt, max_difference=0.02)[0]) == 0
    assert len(associate(est, gt, max_difference=0.6)[0]) > 0


def test_umeyama_recovers_known_similarity():
    src = RNG.uniform(-5, 5, (40, 3))
    R_true = so3_exp([0.2, -0.7, 0.4])
    t_true = np.array([1.0, 2.0, -3.0])
    s_true = 1.7
    dst = s_true * (src @ R_true.T) + t_true

    R, t, s = umeyama(src, dst, with_scale=True)
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)
    assert s == pytest.approx(s_true, rel=1e-9)


def test_umeyama_excludes_reflection():
    src = RNG.uniform(-5, 5, (30, 3))
    dst = src * np.array([1.0, 1.0, -1.0])
    R, _, _ = umeyama(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_ate_requires_enough_associations():
    gt = make_traj(n=10, dt=1.0)
    est = Trajectory(gt.stamps + 100.0, gt.poses)
    with pytest.raises(ValueError):
        evaluate_ate(est, gt)


def test_tum_format_round_trip(tmp_path):
    """기존 평가 도구와 호환되려면 TUM 형식을 정확히 지켜야 한다."""
    traj = make_traj(n=25)
    path = tmp_path / "traj.txt"
    save_trajectory(path, traj)

    loaded = load_trajectory(path)
    assert len(loaded) == len(traj)
    for a, b in zip(traj.poses, loaded.poses):
        d_t, d_r = a.distance_to(b)
        assert d_t < 1e-5
        assert d_r < 1e-5
