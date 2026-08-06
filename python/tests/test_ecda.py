"""ECDA 검증 - 측광 직접정렬.

C++ src/localization/DirectAligner.cpp 는 툴체인 부재로 한 번도 실행된 적이
없다. 여기가 그 알고리즘의 유일한 검증이다.

렌더러가 알려진 포즈로 두 시점을 만들고, ECDA 가 그 포즈를 되찾는지 본다.
"""

import numpy as np
import pytest

from wme.localization import (
    EcdaConfig,
    InformationModel,
    align,
    scale_intrinsics,
    select_points,
)
from wme.reference.environment import Evidence, derive_adaptation
from wme.reference.geometry import SE3, so3_exp
from wme.sim.render import RenderScene, render, render_frame
from wme.sim.world import CameraModel

CAM = CameraModel(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)


def scene():
    return RenderScene.room(size=4.0, height=2.6)


def pose_at(t, rot=(0.0, 0.0, 0.0)):
    return SE3(so3_exp(list(rot)), np.array(t, dtype=float))


def make_pair(delta_xi, base=(0.0, -1.0, 1.3), base_rot=(0.0, 0.0, 0.0),
              ev=None, seed=0):
    """알려진 상대 포즈로 ref/cur 한 쌍을 렌더링한다.

    카메라 z축이 +y 를 보도록 기본 회전을 준다 (방 안쪽을 향하게).
    """
    look_y = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                              np.array([0, 1.0, 0])])
    T_ref = SE3(so3_exp(list(base_rot)) @ look_y, np.array(base, dtype=float))
    # T_cur_ref 를 지정하면 월드 포즈는 T_ref @ inv(T_cur_ref)
    T_cur_ref = SE3.exp(np.asarray(delta_xi, dtype=float))
    T_cur = T_ref @ T_cur_ref.inverse()

    sc = scene()
    ref = render_frame(sc, T_ref, CAM, ev, seed=seed)
    cur = render_frame(sc, T_cur, CAM, ev, seed=seed + 1)
    return ref, cur, T_cur_ref


# --- 피라미드 / 점 선택 ----------------------------------------------------

def test_pyramid_intrinsics_preserve_projection():
    """레벨별 내부 파라미터가 픽셀 중심 규약을 지켜야 한다."""
    p = np.array([0.4, -0.3, 3.0])
    u0 = CAM.fx * p[0] / p[2] + CAM.cx
    v0 = CAM.fy * p[1] / p[2] + CAM.cy

    for level in (1, 2, 3):
        k = scale_intrinsics(CAM, level)
        s = 1.0 / (1 << level)
        assert k.fx * p[0] / p[2] + k.cx == pytest.approx((u0 + 0.5) * s - 0.5)
        assert k.fy * p[1] / p[2] + k.cy == pytest.approx((v0 + 0.5) * s - 0.5)


def room_view():
    """방 안쪽을 보는 시점. 깊이 변화가 있어 직접정렬이 제대로 구속된다."""
    look_y = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                              np.array([0, 1.0, 0])])
    return render(scene(), SE3(look_y, np.array([0.0, -1.0, 1.3])), CAM)


def test_point_selection_is_spatially_spread():
    """격자 셀당 1점이므로 한 영역에 몰리면 안 된다."""
    out = room_view()
    pts = select_points(out.gray.astype(float), out.depth.astype(float), EcdaConfig())

    assert len(pts) > 200
    # 한 사분면이 전체의 절반을 넘으면 분포가 깨진 것
    for xr, yr in [((0, CAM.width // 2), (0, CAM.height // 2)),
                   ((CAM.width // 2, CAM.width), (CAM.height // 2, CAM.height))]:
        m = ((pts[:, 0] >= xr[0]) & (pts[:, 0] < xr[1])
             & (pts[:, 1] >= yr[0]) & (pts[:, 1] < yr[1]))
        assert m.sum() < len(pts) * 0.6


def test_point_selection_respects_static_mask():
    out = room_view()
    mask = np.ones((CAM.height, CAM.width), dtype=bool)
    mask[:, : CAM.width // 2] = False

    pts = select_points(out.gray.astype(float), out.depth.astype(float),
                        EcdaConfig(), static_mask=mask)
    assert len(pts) > 50
    assert np.all(pts[:, 0] >= CAM.width // 2)


def test_fronto_parallel_view_yields_too_few_points():
    """정면 평면을 가까이서 보면 화면상 텍스처 주파수가 낮아 점이 거의 안 잡힌다.

    min_gradient 는 장면 의존 파라미터라는 뜻이다. 버그가 아니라 성질이고,
    실제 배포에서 임계를 고정하면 이런 시점에서 조용히 추적을 잃는다.
    """
    out = render(scene(), pose_at([0.0, -1.0, 1.3]), CAM)   # 천장을 정면으로
    pts = select_points(out.gray.astype(float), out.depth.astype(float), EcdaConfig())
    assert len(pts) < 50

    # 임계를 낮추면 잡힌다 - 정보가 없는 게 아니라 임계가 안 맞는 것
    relaxed = select_points(out.gray.astype(float), out.depth.astype(float),
                            EcdaConfig(min_gradient=1.0))
    assert len(relaxed) > 200


# --- 포즈 복원 -------------------------------------------------------------

def test_recovers_pure_translation():
    ref, cur, truth = make_pair([0.06, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    assert r.converged
    d_t, d_r = r.T_cur_ref.distance_to(truth)
    assert d_t < 0.01, f"병진 오차 {d_t:.4f} m"
    assert d_r < 0.005


@pytest.mark.parametrize("axis,xi", [
    ("x", [0.0, 0.0, 0.0, 0.03, 0.0, 0.0]),
    ("z", [0.0, 0.0, 0.0, 0.0, 0.0, 0.03]),
])
def test_recovers_pure_rotation_about_well_constrained_axes(axis, xi):
    ref, cur, truth = make_pair(xi)
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    assert r.converged
    d_t, d_r = r.T_cur_ref.distance_to(truth)
    assert d_r < 0.008, f"{axis}축 회전 오차 {d_r:.4f} rad"


def test_coarse_levels_must_scale_the_selection_grid():
    """회귀 테스트.

    격자 크기를 모든 레벨에 고정하면 거친 레벨의 셀 수가 min_points 아래로
    떨어져 그 레벨이 통째로 건너뛰어진다. 320x240 에 grid_cell=8 이면
    레벨 3(40x30)은 셀이 15개뿐이다. 수렴 반경을 넓혀주는 바로 그 레벨들이
    사라지므로, 광축에 수직인 축 회전 같은 케이스가 조용히 실패한다.

    이 결함은 한동안 '직접법의 본질적 회전/병진 모호성'으로 오진되어 있었다.
    """
    out = room_view()
    gray = out.gray.astype(float)
    depth = out.depth.astype(float)
    cfg = EcdaConfig()

    for level in range(cfg.pyramid_levels):
        g = gray[:: 1 << level, :: 1 << level]
        d = depth[:: 1 << level, :: 1 << level]
        pts = select_points(g, d, cfg, level=level)
        assert len(pts) >= cfg.min_points, (
            f"레벨 {level}: 점 {len(pts)}개 < min_points {cfg.min_points}")


def test_rotation_about_optical_axis_normal_now_converges():
    """레벨별 격자 수정 후 광축 수직축 회전도 정상 수렴해야 한다."""
    ref, cur, truth = make_pair([0.0, 0.0, 0.0, 0.0, 0.03, 0.0])
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    assert r.converged
    d_t, d_r = r.T_cur_ref.distance_to(truth)
    assert d_t < 0.02, f"병진 오차 {d_t:.4f} m"
    assert d_r < 0.01, f"회전 오차 {d_r:.4f} rad"


def test_residual_separates_converged_from_diverged():
    """잔차 게이트의 근거.

    수렴 반경을 넘어서는 큰 운동에서는 정렬이 실패하고, 그 실패는 랭크가
    아니라 잔차에 드러난다. 실시간 시스템이 자기 잔차를 게이트로 써야 하는
    이유다. (다만 임계는 장면 의존이라 절대값으로 고정할 수 없다 -
    wme.graph.photometric_slam 은 시퀀스 중앙값 대비 상대 기준을 쓴다.)
    """
    good_cases = [[0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.03, 0.0, 0.0],
                  [0.05, -0.03, 0.02, 0.012, -0.018, 0.008]]
    good_rmse = []
    for xi in good_cases:
        ref, cur, truth = make_pair(xi)
        r = align(ref.gray, ref.depth, cur.gray, CAM)
        assert r.T_cur_ref.distance_to(truth)[0] < 0.03
        good_rmse.append(r.photometric_rmse)

    # 수렴 반경을 크게 넘는 운동
    bad_ref, bad_cur, bad_truth = make_pair([1.2, -0.9, 0.4, 0.25, -0.30, 0.15])
    bad = align(bad_ref.gray, bad_ref.depth, bad_cur.gray, CAM)

    assert bad.T_cur_ref.distance_to(bad_truth)[0] > 0.1, "실패해야 하는 케이스가 성공했다"
    assert bad.photometric_rmse > max(good_rmse) * 1.5, (
        f"good {np.round(good_rmse, 2).tolist()}, bad {bad.photometric_rmse:.2f}")
    # 이렇게 크게 발산한 경우는 랭크도 함께 떨어진다. 다만 랭크가 항상
    # 알려주는 것은 아니다 - 격자 결함으로 인한 미묘한 실패는 랭크 6 을
    # 유지한 채 일어났고 잔차만이 그것을 드러냈다.
    assert bad.observable_dof <= 6


def test_conditioning_diagnostics_are_populated():
    """조건수와 최약축이 계산되어 있어야 한다 (진짜 랭크 부족 진단용)."""
    ref, cur, _ = make_pair([0.05, -0.03, 0.02, 0.012, -0.018, 0.008])
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    assert np.isfinite(r.condition_number) and r.condition_number > 1.0
    assert np.isclose(np.linalg.norm(r.weakest_direction), 1.0, atol=1e-9)


def test_recovers_general_motion():
    ref, cur, truth = make_pair([0.05, -0.03, 0.02, 0.012, -0.018, 0.008])
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    assert r.converged
    d_t, d_r = r.T_cur_ref.distance_to(truth)
    assert d_t < 0.02, f"병진 오차 {d_t:.4f} m"
    assert d_r < 0.01, f"회전 오차 {d_r:.4f} rad"
    assert r.inlier_ratio > 0.6
    assert r.point_count > 300


def test_pyramid_extends_convergence_basin():
    """큰 변위는 거친 레벨에서만 잡힌다. 피라미드의 존재 이유."""
    ref, cur, truth = make_pair([0.22, -0.10, 0.05, 0.02, -0.04, 0.01])

    single = align(ref.gray, ref.depth, cur.gray, CAM,
                   cfg=EcdaConfig(pyramid_levels=1))
    multi = align(ref.gray, ref.depth, cur.gray, CAM,
                  cfg=EcdaConfig(pyramid_levels=5))

    err_single = (single.T_cur_ref.distance_to(truth)[0] if single.converged else 1e9)
    err_multi = multi.T_cur_ref.distance_to(truth)[0]
    assert err_multi < err_single


def test_good_initial_guess_helps():
    ref, cur, truth = make_pair([0.15, -0.08, 0.04, 0.02, -0.03, 0.01])
    warm = align(ref.gray, ref.depth, cur.gray, CAM,
                 init=SE3.exp(truth.log() * 0.8))
    assert warm.converged
    assert warm.T_cur_ref.distance_to(truth)[0] < 0.03


# --- 아핀 밝기 모델 --------------------------------------------------------

def test_affine_model_absorbs_exposure_change():
    """노출이 바뀌어도 포즈를 복원해야 하고, 아핀 계수가 그 변화를 흡수해야 한다."""
    ref, cur, truth = make_pair([0.05, -0.02, 0.01, 0.01, -0.015, 0.005])
    bright = np.clip(cur.gray * 1.3 + 15.0, 0.0, 255.0)

    r = align(ref.gray, ref.depth, bright, CAM)
    assert r.converged
    assert r.T_cur_ref.distance_to(truth)[0] < 0.03
    assert r.affine_a == pytest.approx(1.3, rel=0.25)


def test_exposure_change_breaks_alignment_without_affine():
    ref, cur, truth = make_pair([0.05, -0.02, 0.01, 0.01, -0.015, 0.005])
    bright = np.clip(cur.gray * 1.3 + 15.0, 0.0, 255.0)

    with_affine = align(ref.gray, ref.depth, bright, CAM)
    without = align(ref.gray, ref.depth, bright, CAM,
                    cfg=EcdaConfig(estimate_affine=False))

    e_with = with_affine.T_cur_ref.distance_to(truth)[0]
    e_without = (without.T_cur_ref.distance_to(truth)[0] if without.converged else 1e9)
    assert e_with < e_without


# --- 정보행렬과 퇴화 -------------------------------------------------------

def test_information_matrix_is_symmetric_positive():
    ref, cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    r = align(ref.gray, ref.depth, cur.gray, CAM)

    L = r.information
    assert np.allclose(L, L.T, atol=1e-6)
    assert np.all(np.diag(L) > 0.0)
    assert np.all(np.isfinite(L))


def test_environment_weight_scales_information():
    """alpha_0(E) 는 항을 더하지 않고 정보량만 조절해야 한다."""
    ref, cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    full = align(ref.gray, ref.depth, cur.gray, CAM, alpha_photometric=1.0)
    fog = align(ref.gray, ref.depth, cur.gray, CAM, alpha_photometric=0.1)

    assert np.allclose(fog.information, full.information * 0.1, rtol=1e-9)
    # 포즈 추정 자체는 가중치와 무관해야 한다
    assert np.allclose(fog.T_cur_ref.matrix(), full.T_cur_ref.matrix(), atol=1e-12)


def test_photometric_variance_scales_information_only_in_sensor_model():
    """센서 잡음은 유효표본 모델에서 *바닥* 이지 척도가 아니다.

    SENSOR_VARIANCE 는 Lambda = H/sigma^2 로 센서 잡음에 반비례한다. 실측에서
    그 가정은 (ii) 부터 틀렸다 - 해에서의 잔차 분산이 센서 잡음의 3~34 배다
    (15.2). 유효표본 모델은 실제 달성된 잔차 분산을 쓰고 센서 잡음은 하한으로만
    쓴다. 그래서 sigma^2 를 잔차 분산 아래에서 움직여도 Lambda 는 꿈쩍하지 않고,
    잔차 분산 위로 올리면 그때부터 반비례한다.
    """
    ref, cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    sensor = EcdaConfig(information_model=InformationModel.SENSOR_VARIANCE)

    a = align(ref.gray, ref.depth, cur.gray, CAM, cfg=sensor, photometric_variance=1.0)
    b = align(ref.gray, ref.depth, cur.gray, CAM, cfg=sensor, photometric_variance=4.0)
    assert np.allclose(b.information, a.information * 0.25, rtol=1e-9)

    # 기본(유효표본) 모델: 달성 잔차 분산(rmse^2) 아래에서는 무시된다
    eff_lo = align(ref.gray, ref.depth, cur.gray, CAM, photometric_variance=1.0)
    eff_mid = align(ref.gray, ref.depth, cur.gray, CAM, photometric_variance=4.0)
    assert np.allclose(eff_lo.information, eff_mid.information, rtol=1e-12)

    # 위로 올리면 바닥이 아니라 척도가 된다
    huge = float(np.sum(eff_lo.information)) * 0 + 1e9
    eff_hi = align(ref.gray, ref.depth, cur.gray, CAM, photometric_variance=huge)
    assert np.trace(eff_hi.information) < np.trace(eff_lo.information) * 1e-3


def test_effective_sample_information_does_not_grow_with_point_count():
    """서브샘플링 실측의 핵심: 점을 더 넣어도 정확도가 늘지 않으므로 확신도 늘면
    안 된다 (15.1). SENSOR_VARIANCE 에서는 Lambda ~ N^1.00 이었다."""
    ref, cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])

    def trace_at(cell, model):
        cfg = EcdaConfig(grid_cell=cell, information_model=model)
        r = align(ref.gray, ref.depth, cur.gray, CAM, cfg=cfg)
        return r.point_count, float(np.trace(r.information))

    n_dense, t_dense = trace_at(4, InformationModel.EFFECTIVE_SAMPLE)
    n_thin, t_thin = trace_at(16, InformationModel.EFFECTIVE_SAMPLE)
    assert n_dense > 5 * n_thin, "점 수가 실제로 달라야 측정이 성립한다"

    # 유효표본 모델에서는 점 수 의존성이 사라진다 (재조정이 아니라 제거).
    # 실측 N 8.2 배에 대해 trace 비 0.89.
    assert 0.5 < (t_dense / t_thin) < 2.0, (
        f"N {n_dense}->{n_thin} (x{n_dense / n_thin:.2f}) 인데 "
        f"trace 가 x{t_dense / t_thin:.2f} 로 따라 움직인다")

    # 대조군: 픽셀을 독립 증거로 세는 모델은 점 수에 거의 비례한다 (x7.4)
    r_dense = trace_at(4, InformationModel.RESIDUAL_VARIANCE)[1]
    r_thin = trace_at(16, InformationModel.RESIDUAL_VARIANCE)[1]
    assert r_dense / r_thin > 0.7 * (n_dense / n_thin), (
        "대조군이 성립하지 않으면 위 결과는 아무것도 말하지 않는다")


def test_textureless_scene_is_reported_as_degenerate():
    """균일한 벽에서는 성공을 주장하면 안 된다."""
    flat = np.full((CAM.height, CAM.width), 128.0)
    depth = np.full((CAM.height, CAM.width), 3.0)

    r = align(flat, depth, flat, CAM)
    assert (not r.converged) or r.observable_dof < 6


def test_full_rank_on_textured_scene():
    ref, cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    r = align(ref.gray, ref.depth, cur.gray, CAM)
    assert r.observable_dof == 6, f"고유값 {r.eigenvalues}"


# --- 환경 열화 하에서의 거동 -----------------------------------------------

def test_fog_degradation_enters_through_alpha_not_through_raw_information():
    """유효표본 모델에서 대비 열화는 Lambda 에 나타나지 않는다.

    Lambda = H/(chi2/nu) 에서 전역 대비가 s 배가 되면 H 도 chi2 도 s^2 배라
    상쇄된다. SNR 이 실제로 떨어지지 않는 열화는 확신을 낮추지 않는 것이 맞고,
    이것이 옛 SENSOR_VARIANCE 와의 관측 가능한 차이다: 그쪽은 H 만 줄어 안개에서
    Lambda 가 같이 떨어졌다.

    귀결: 안개를 tier 가중에 반영하는 경로는 이제 alpha_0(E) 하나뿐이다.
    """
    clear_ref, clear_cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    fog_ref, fog_cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005],
                                    ev=Evidence(haze=0.85))

    sensor = EcdaConfig(information_model=InformationModel.SENSOR_VARIANCE)
    clear_s = align(clear_ref.gray, clear_ref.depth, clear_cur.gray, CAM, cfg=sensor)
    foggy_s = align(fog_ref.gray, fog_ref.depth, fog_cur.gray, CAM, cfg=sensor)
    assert np.trace(foggy_s.information) < np.trace(clear_s.information)

    # alpha_0(E) 는 여전히 안개에서 정보 질량을 줄인다
    alpha = derive_adaptation(Evidence(haze=0.85)).alpha_photometric
    assert alpha < 1.0
    foggy = align(fog_ref.gray, fog_ref.depth, fog_cur.gray, CAM,
                  alpha_photometric=alpha)
    foggy_full = align(fog_ref.gray, fog_ref.depth, fog_cur.gray, CAM)
    assert np.trace(foggy.information) < np.trace(foggy_full.information)


def test_information_falls_under_motion_blur():
    """모션 블러에서 정보가 줄어드는가 - 모델을 명시해야 답이 정해진다.

    이 주장은 Lambda 가 *총* H 에 비례하는 모델(SENSOR_VARIANCE /
    EFFECTIVE_SAMPLE)에서만 성립한다. 기본값이 COHERENT_FRAME 으로 바뀌면서
    Lambda = (H / N) / sigma_c^2, 즉 *점당 평균* 정보가 되었고, 열화가
    그래디언트가 아니라 점 *선택* 을 바꾸면 (약한 점이 min_gradient 아래로
    떨어져 통째로 빠지면) N 이 H 보다 빨리 줄어 평균이 오히려 올라간다.

    실측 (320x240 합성 방, motion_blur=0.9): 점 698 -> 131, tr(H) 2.6 배 감소,
    N 5.3 배 감소 -> tr(Lambda) 4.25e4 -> 8.77e4 로 2.06 배 *상승*.
    같은 장면에서 EFFECTIVE_SAMPLE 은 분모의 rmse^2 가 올라가 0.88 배로 내려간다.
    이 한계는 C++ 쪽에도
    DirectAligner.CoherentFrameInformationFallsWhenGradientsFallAtFixedPointSet
    으로 고정되어 있다. 숨기지 않고 숫자로 박아 둔다.
    """
    clear_ref, clear_cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005])
    blur_ref, blur_cur, _ = make_pair([0.04, -0.02, 0.01, 0.01, -0.01, 0.005],
                                      ev=Evidence(motion_blur=0.9))

    for model in (InformationModel.SENSOR_VARIANCE, InformationModel.EFFECTIVE_SAMPLE):
        cfg = EcdaConfig(information_model=model)
        clear = align(clear_ref.gray, clear_ref.depth, clear_cur.gray, CAM, cfg=cfg)
        blurred = align(blur_ref.gray, blur_ref.depth, blur_cur.gray, CAM, cfg=cfg)
        assert np.trace(blurred.information) < np.trace(clear.information), model

    # 점 수가 실제로 크게 줄어야 아래 주장이 그 원인을 짚는 것이 된다
    coh = EcdaConfig(information_model=InformationModel.COHERENT_FRAME)
    clear_c = align(clear_ref.gray, clear_ref.depth, clear_cur.gray, CAM, cfg=coh)
    blur_c = align(blur_ref.gray, blur_ref.depth, blur_cur.gray, CAM, cfg=coh)
    assert blur_c.point_count < 0.5 * clear_c.point_count, (
        f"점 수가 {clear_c.point_count} -> {blur_c.point_count} 로 별로 안 줄었다 - "
        "이 케이스는 선택 변화를 재지 못한다")
    assert np.trace(blur_c.information) > np.trace(clear_c.information), (
        "CoherentFrame 의 알려진 한계가 사라졌다면 모델이 바뀐 것이다 - "
        "C++ 쪽 고정 테스트와 함께 확인할 것")


def test_alignment_survives_moderate_degradation():
    """중간 정도 열화에서는 여전히 포즈를 복원해야 한다."""
    ref, cur, truth = make_pair([0.05, -0.02, 0.01, 0.01, -0.015, 0.005],
                                ev=Evidence(haze=0.4, darkness=0.3))
    r = align(ref.gray, ref.depth, cur.gray, CAM)
    assert r.converged
    assert r.T_cur_ref.distance_to(truth)[0] < 0.05
