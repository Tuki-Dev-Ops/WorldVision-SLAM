"""측광 SLAM 통합 검증.

ECDA 오도메트리 자체는 검증된다. 그래프 통합은 아직 열려 있고
(docs/03-roadmap.md Phase 2h), 여기서는 무엇이 성립하고 무엇이 아닌지를
테스트로 명확히 남긴다.
"""

import numpy as np
import pytest

from wme.calib import CalibratedNoise
from wme.graph.photometric_slam import (
    PhotometricSlamConfig, build, calibrate_information_scale,
    render_sequence, run_ecda_odometry,
)
from wme.reference.environment import Evidence
from wme.reference.geometry import SE3
from wme.sim import CameraTrajectory, SimObject, SimWorld
from wme.sim.render import RenderScene, detections_from_render, render
from wme.sim.world import CameraModel

CAM = CameraModel(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)
NOISE = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)


def small_world(n=5, seed=0):
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n):
        a = -0.8 + 1.6 * i / max(1, n - 1)
        r = rng.uniform(2.6, 3.2)
        objs.append(SimObject(
            object_id=i + 1, class_id=56, class_name="chair",
            position=np.array([r * np.cos(a), r * np.sin(a), rng.uniform(0.6, 1.5)]),
            extent=np.array([0.26, 0.26, 0.30]),
        ))
    return SimWorld(objs)


def outward_arc(n=14, radius=0.9, arc=1.2):
    stamps, poses = [], []
    for i in range(n):
        a = -arc * 0.5 + arc * i / max(1, n - 1)
        eye = np.array([radius * np.cos(a), radius * np.sin(a), 1.3])
        target = np.array([4.5 * np.cos(a), 4.5 * np.sin(a), 1.0])
        poses.append(CameraTrajectory._look_at(eye, target))
        stamps.append(1.0 + i * 0.05)
    return CameraTrajectory(np.array(stamps), poses)


def make_frames(n=14, ev=None, seed=0):
    scene = RenderScene.room(size=4.0, height=2.6)
    return render_sequence(scene, small_world(5, seed), outward_arc(n), CAM,
                           ev or Evidence(), seed=seed)


# --- 렌더 기반 검출 --------------------------------------------------------

def test_detections_come_from_actual_masks():
    """검출이 렌더링된 마스크에서 나오면 가려짐과 잘림이 물리적으로 맞다."""
    scene = RenderScene.room(size=4.0, height=2.6).with_objects(small_world(5), 1.0)
    pose = outward_arc(3).poses[1]
    out = render(scene, pose, CAM)

    dets = detections_from_render(out, min_pixels=60)
    assert len(dets) >= 1
    for d in dets:
        assert d.pixels >= 60
        assert d.depth > 0.0
        x, y, w, h = d.box
        assert w > 0 and h > 0
        assert 0 <= x < CAM.width and 0 <= y < CAM.height


def test_truncation_flag_is_set_at_borders():
    scene = RenderScene(quads=[], boxes=[])
    from wme.sim.render import Box
    # 카메라 바로 앞의 큰 상자는 화면을 넘긴다
    scene = RenderScene(quads=[], boxes=[Box(np.array([0.0, 0.0, 1.0]),
                                             np.array([2.0, 2.0, 0.2]), object_id=1)])
    out = render(scene, SE3.identity(), CAM)
    dets = detections_from_render(out, min_pixels=60)
    assert dets and dets[0].truncated


# --- ECDA 오도메트리 -------------------------------------------------------

def test_ecda_odometry_is_accurate_on_rendered_frames():
    """이것이 Tier 0 의 실제 검증이다. 프레임 간 상대포즈를 되찾아야 한다."""
    frames = make_frames(12)
    cfg = PhotometricSlamConfig()
    odom = run_ecda_odometry(frames, CAM, cfg, Evidence())

    errors = []
    for i, (T_rel, _info, _rmse, ok) in enumerate(odom):
        if not ok:
            continue
        truth = frames[i + 1].pose.inverse() @ frames[i].pose   # ref -> cur
        errors.append(T_rel.distance_to(truth)[0])

    assert len(errors) >= len(odom) * 0.8, "대부분의 정렬이 채택되어야 한다"
    true_step = np.mean([np.linalg.norm(
        (frames[i + 1].pose.inverse() @ frames[i].pose).t) for i in range(len(frames) - 1)])
    assert np.median(errors) < true_step * 0.4, (
        f"오차 중앙값 {np.median(errors):.4f} vs 참 이동 {true_step:.4f}")


def test_residual_gate_is_relative_not_absolute():
    """정상 정렬의 rmse 는 장면에 따라 크게 다르다.

    빈 방에서는 0.2~3, 객체가 있으면 실루엣 경계 때문에 6~10 이었다.
    절대 임계를 쓰면 장면이 바뀔 때 정상 정렬을 대량으로 버린다.

    주장은 "이 장면의 rmse 가 어떤 특정 수보다 크다" 가 아니라 **정상 정렬끼리도
    rmse 가 크게 흩어진다** 이므로, 그 흩어짐 자체를 잰다. 원래는 중앙값이 4.0
    을 넘는지로 대신 쟀는데, 그 4.0 은 고정 huber(12.0) 커널이 내던 잔차 수준에
    맞춰 둔 상수였다. 커널이 적응형으로 바뀌자(11.4 를 파이썬에 포팅) 가중 잔차가
    줄어 중앙값이 3.39 가 되면서 테스트가 깨졌는데, 깨진 것은 주장이 아니라
    커널에 묶인 눈금이었다. 이제 커널과 무관한 양으로 쓴다.
    """
    frames = make_frames(12)
    odom = run_ecda_odometry(frames, CAM, PhotometricSlamConfig(), Evidence())
    accepted = [ok for *_, ok in odom]
    rmses = np.array([r for _, _, r, _ in odom])

    # 정상 정렬끼리 rmse 가 최소 2 배 이상 흩어진다. 이 흩어짐이 곧 "단일 절대
    # 임계로는 한쪽을 반드시 잘못 판정한다" 는 뜻이다.
    assert rmses.max() / max(rmses.min(), 1e-9) > 2.0, (
        f"rmse 분포가 좁으면 절대 임계도 통한다 - 이 테스트가 무의미해진다: {rmses}")
    # 그럼에도 대부분 채택되어야 한다 (상대 기준이므로)
    assert np.mean(accepted) > 0.8


def test_degradation_reduces_accepted_alignments_or_accuracy():
    """열화가 심하면 정렬 품질이 떨어져야 한다."""
    clear = make_frames(10, Evidence())
    foggy = make_frames(10, Evidence(haze=0.85, darkness=0.5))

    def median_error(frames):
        odom = run_ecda_odometry(frames, CAM, PhotometricSlamConfig(), Evidence())
        errs = [T.distance_to(frames[i + 1].pose.inverse() @ frames[i].pose)[0]
                for i, (T, _, _, ok) in enumerate(odom) if ok]
        return np.median(errs) if errs else np.inf

    assert median_error(foggy) > median_error(clear)


# --- 정보 스케일 보정 ------------------------------------------------------

def test_information_scale_calibration_reduces_overconfidence():
    """J^T W J 를 그대로 쓰면 픽셀 잔차를 독립으로 세어 정보가 과대평가된다.

    M3 에서 객체 융합이 1/N 로 과신하던 것과 같은 오류가 측광 층에서 재현된다.
    """
    frames = make_frames(14)
    scale, raw_anees = calibrate_information_scale(frames, CAM,
                                                   PhotometricSlamConfig(), Evidence())
    assert np.isfinite(raw_anees)
    assert raw_anees > 1.0, "보정 전에는 과신이어야 한다"
    assert 0.0 < scale < 1.0
    assert scale == pytest.approx(1.0 / raw_anees, rel=1e-9)


# --- 그래프 구성 -----------------------------------------------------------

def test_graph_contains_photometric_and_object_factors():
    # 짧은 시퀀스라 관측 최소치를 낮춘다. 구조 검증이 목적이다.
    cfg = PhotometricSlamConfig(observation_stride=1, min_observations=3)
    frames = make_frames(14)
    r = build(frames, small_world(5), CAM, NOISE, cfg, Evidence())

    tiers = {getattr(f, "tier", None) for f in r.graph.factors}
    assert "photometric" in tiers, "Tier 0 팩터가 그래프에 없다"
    assert r.ecda_accepted > 0
    assert len(r.landmark_keys) >= 1


def test_anchor_is_fixed_to_truth():
    frames = make_frames(12)
    r = build(frames, small_world(5), CAM, NOISE, PhotometricSlamConfig(), Evidence())
    d_t, _ = r.graph.value(r.pose_keys[0]).pose.distance_to(r.truth_poses[0])
    assert d_t < 1e-9


def test_graph_optimisation_improves_pose_accuracy():
    """xfail 표시를 뗀 시점 기록: 25.21.

    표시는 `strict=False` 였고 실제로는 통과하고 있었다 - 즉 "실패해도 초록,
    통과해도 초록" 이라 아무것도 재지 않는 상태였다(19.3 이 적어 둔 그 실패
    양식). 실측하면 네 가지 설정 전부에서 개선한다.

    개선 폭은 크지 않다(2~3 %). 그러니 회귀가 나면 방향이 아니라 폭에서 먼저
    보일 것이므로 값을 메시지에 남긴다.
    """
    from wme.graph.photometric_slam import solve

    frames = make_frames(20)
    scale, _ = calibrate_information_scale(frames, CAM, PhotometricSlamConfig(), Evidence())
    r = solve(frames, small_world(5), CAM, NOISE,
              PhotometricSlamConfig(photometric_information_scale=scale), Evidence())

    before = float(np.sqrt((r.initial_pose_errors() ** 2).mean()))
    after = float(np.sqrt((r.pose_errors() ** 2).mean()))
    assert after < before, f"최적화 후 {after:.6f} >= 전 {before:.6f}"
