"""잡음모델 보정 검증 (M3).

가장 중요한 것은 대조군이다: 관측 모델이 *정확할* 때 추정기가 일관된 공분산을
내는가. 이게 안 되면 M3 실험의 어떤 결과도 해석할 수 없다.
"""

import numpy as np
import pytest

from wme.calib import (
    CalibratedNoise, FixedNoise, NoiseSample, ScheduledNoise,
    fit_calibrated, minimize_nelder_mead, negative_log_likelihood,
)
from wme.eval.metrics import nees
from wme.sim.world import CameraModel

RNG = np.random.default_rng(4242)
CAM = CameraModel()


# --- 최적화 ---------------------------------------------------------------

def test_nelder_mead_finds_quadratic_minimum():
    target = np.array([1.5, -2.0, 0.3])
    best, val = minimize_nelder_mead(lambda x: float(np.sum((x - target) ** 2)),
                                     np.zeros(3))
    assert np.allclose(best, target, atol=1e-5)
    assert val < 1e-9


def test_nelder_mead_handles_rosenbrock():
    def rosen(x):
        return float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                         for i in range(len(x) - 1)))
    best, _ = minimize_nelder_mead(rosen, np.array([-1.2, 1.0]), max_iter=5000)
    assert np.allclose(best, [1.0, 1.0], atol=1e-3)


def test_nelder_mead_survives_infinite_values():
    """발산 영역을 inf 로 표시해도 탐색이 죽으면 안 된다."""
    def fn(x):
        if x[0] < 0:
            return float("inf")
        return float((x[0] - 2.0) ** 2)
    best, _ = minimize_nelder_mead(fn, np.array([1.0]))
    assert best[0] == pytest.approx(2.0, abs=1e-4)


# --- 잡음모델 -------------------------------------------------------------

def test_covariance_propagates_depth_noise_laterally():
    """깊이 잡음이 횡방향으로도 새어야 한다. 이 항을 빼면 원거리에서 과신한다."""
    model = FixedNoise(sigma_px=2.0, depth_coeff=0.01)

    # 화면 중심에서는 새어나감이 없다
    center = model.covariance(CAM.cx, CAM.cy, 5.0, 0.0, CAM)
    assert abs(center[0, 2]) < 1e-12

    # 가장자리에서는 x-z 상관이 생긴다
    edge = model.covariance(CAM.cx + 200.0, CAM.cy, 5.0, 0.0, CAM)
    assert abs(edge[0, 2]) > 1e-6


def test_covariance_grows_with_depth():
    model = FixedNoise()
    near = model.covariance(CAM.cx, CAM.cy, 2.0, 0.0, CAM)
    far = model.covariance(CAM.cx, CAM.cy, 8.0, 0.0, CAM)
    assert np.trace(far) > np.trace(near) * 10.0


def test_fixed_model_ignores_condition():
    m = FixedNoise()
    assert m.sigma_depth(5.0, 0.0) == m.sigma_depth(5.0, 0.9)


def test_calibrated_model_responds_to_condition():
    m = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)
    assert m.sigma_depth(5.0, 0.9) > m.sigma_depth(5.0, 0.0)
    assert m.sigma_pixel(0.9) > m.sigma_pixel(0.0)


def test_scheduled_model_inflates_with_severity():
    base = FixedNoise()
    m = ScheduledNoise(base=base)
    assert m.sigma_depth(5.0, 0.8) > base.sigma_depth(5.0, 0.8)


def test_systematic_covariance_depends_on_object_size():
    """계통 오차의 기원은 기하학적이므로 크기에 의존해야 한다."""
    m = CalibratedNoise(k_geom=0.1)
    small = m.systematic_covariance(0.0, box_px=30.0, depth=3.0, cam=CAM)
    large = m.systematic_covariance(0.0, box_px=150.0, depth=3.0, cam=CAM)
    assert np.trace(large) > np.trace(small) * 10.0

    # 정보가 없으면 0 을 돌려야 한다 (모르는 것을 지어내지 않는다)
    assert np.allclose(m.systematic_covariance(0.0), 0.0)


# --- 우도와 적합 -----------------------------------------------------------

def _synth_samples(model: CalibratedNoise, n: int = 1500) -> list[NoiseSample]:
    """주어진 모델의 참 분포에서 관측을 생성한다."""
    out = []
    for _ in range(n):
        sev = float(RNG.uniform(0.0, 1.0))
        z = float(RNG.uniform(1.5, 8.0))
        u = float(RNG.uniform(60.0, CAM.width - 60.0))
        v = float(RNG.uniform(60.0, CAM.height - 60.0))

        s_px = model.sigma_pixel(sev)
        s_d = model.sigma_depth(z, sev)

        # 참 위치는 잡음 없는 역투영, 관측은 잡음 섞인 픽셀/깊이
        u_n, v_n = u + RNG.normal(0, s_px), v + RNG.normal(0, s_px)
        z_n = z + RNG.normal(0, s_d)
        true_cam = np.array([(u - CAM.cx) * z / CAM.fx, (v - CAM.cy) * z / CAM.fy, z])
        out.append(NoiseSample(u_n, v_n, z_n, sev, true_cam))
    return out


def test_nll_is_minimised_near_true_parameters():
    truth = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)
    samples = _synth_samples(truth, 800)

    at_truth = negative_log_likelihood(truth, samples, CAM)
    too_small = negative_log_likelihood(
        CalibratedNoise(c_px=0.5, g_px=4.0, c_d=0.0015, g_d=3.0), samples, CAM)
    too_large = negative_log_likelihood(
        CalibratedNoise(c_px=8.0, g_px=4.0, c_d=0.024, g_d=3.0), samples, CAM)

    assert at_truth < too_small
    assert at_truth < too_large, "logdet 항이 없으면 공분산 과대평가가 벌받지 않는다"


def test_fit_recovers_true_parameters():
    """M3 의 전제: 적합이 참 잡음 파라미터를 복원할 수 있어야 한다."""
    truth = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)
    fitted = fit_calibrated(_synth_samples(truth, 2500), CAM)

    assert fitted.c_d == pytest.approx(truth.c_d, rel=0.20)
    assert fitted.g_d == pytest.approx(truth.g_d, rel=0.35)
    assert fitted.c_px == pytest.approx(truth.c_px, rel=0.30)


def test_fit_rejects_tiny_sample():
    with pytest.raises(ValueError):
        fit_calibrated([], CAM)


# --- 대조군: 관측 모델이 정확하면 일관되어야 한다 ---------------------------

def test_estimator_is_consistent_with_a_perfect_measurement_model():
    """이 테스트가 M3 실험의 대조군이다.

    관측 모델에 편향이 전혀 없고(참 위치를 직접 관측) 잡음모델이 정확하면,
    정보필터 융합 결과의 NEES 는 일관되어야 한다. 이게 안 되면 M3 에서 나온
    과신을 잡음모델 탓으로 돌릴 수 없다.
    """
    n_objects, n_obs = 60, 40
    errors, covs = [], []

    for _ in range(n_objects):
        truth = RNG.uniform(-4.0, 4.0, 3)
        Lam = np.zeros((3, 3))
        xi = np.zeros(3)

        for _ in range(n_obs):
            # 관측마다 다른 등방 잡음 (실제로도 관측마다 정밀도가 다르다)
            s = float(RNG.uniform(0.02, 0.15))
            S = np.eye(3) * (s * s)
            z = truth + RNG.normal(0.0, s, 3)
            S_inv = np.linalg.inv(S)
            Lam += S_inv
            xi += S_inv @ z

        P = np.linalg.inv(Lam)
        errors.append(truth - P @ xi)
        covs.append(P)

    r = nees(np.array(errors), np.array(covs))
    assert r.consistent, f"대조군이 실패하면 실험을 해석할 수 없다: {r.summary()}"


def test_joint_extent_forward_model_is_exact():
    """순방향 모델이 시뮬레이터의 박스 생성 규칙과 정확히 일치해야 한다.

    여기가 틀리면 M3 결과 전체가 무의미하다.
    """
    from wme.calib.shape import predict_observation
    from wme.reference.geometry import SE3, so3_exp
    from wme.sim import CameraModel, SimObject, SimSensor, SimWorld
    from wme.reference.environment import Evidence

    cam = CameraModel()
    obj = SimObject(1, 56, "chair", np.array([0.4, -0.2, 3.5]),
                    extent=np.array([0.3, 0.25, 0.2]))
    T_world_cam = SE3(so3_exp([0.05, -0.1, 0.02]), np.array([0.2, 0.1, -0.3]))

    # 잡음 없는 관측을 얻기 위해 검출 확률 1, 잡음 0 으로 설정
    from wme.sim import SensorConfig
    cfg = SensorConfig(base_detection_prob=1.0, base_box_noise_px=0.0,
                       depth_noise_coeff=0.0, base_false_alarm_rate=0.0,
                       class_confusion_prob=0.0)
    det, truth = SimSensor(cam, cfg, seed=0).observe(
        SimWorld([obj]), T_world_cam, 1.0, Evidence(), 0)

    assert len(det.items) == 1
    x, y, w, h = det.items[0].box
    pred = predict_observation(obj.position, obj.extent,
                               T_world_cam.inverse(), cam)
    assert pred is not None
    assert np.allclose(pred[:4], [x, y, w, h], atol=1e-6)
    assert pred[4] == pytest.approx(truth.detection_depth[0], abs=1e-6)


def test_joint_extent_reduces_bias_versus_box_centre():
    """관측 모델 수정의 핵심 주장: 박스 중심 역투영의 편향이 줄어야 한다."""
    from wme.calib import estimate_objects
    from wme.calib.shape import estimate_objects_jointly
    from wme.sim import scenario_static

    seq = scenario_static(seed=5)
    model = CalibratedNoise(c_px=2.0, g_px=4.0, c_d=0.006, g_d=3.0)

    centre = estimate_objects(seq, model)
    joint = estimate_objects_jointly(seq, model)

    assert len(joint.estimates) >= 5
    assert np.linalg.norm(joint.bias(5)) < np.linalg.norm(centre.bias(5))


def test_unmodelled_bias_makes_fusion_overconfident():
    """M3 에서 관찰된 현상의 최소 재현.

    관측마다 공통으로 실리는 편향이 있으면, 융합할수록 공분산은 1/N 로 줄지만
    편향은 그대로 남아 과신이 N 에 비례해 커진다. 관측 단위 잔차만 보면
    이 성분은 보이지 않는다.
    """
    def run(bias_sigma: float, n_obs: int) -> float:
        errors, covs = [], []
        for _ in range(50):
            truth = RNG.uniform(-4.0, 4.0, 3)
            bias = RNG.normal(0.0, bias_sigma, 3)      # 객체마다 고정
            s = 0.05
            S = np.eye(3) * s * s
            S_inv = np.linalg.inv(S)

            Lam = S_inv * n_obs
            xi = sum(S_inv @ (truth + bias + RNG.normal(0, s, 3)) for _ in range(n_obs))
            P = np.linalg.inv(Lam)
            errors.append(truth - P @ xi)
            covs.append(P)
        r = nees(np.array(errors), np.array(covs))
        return r.anees / r.dof

    assert run(0.0, 40) < 1.6, "편향이 없으면 일관되어야 한다"
    few = run(0.03, 5)
    many = run(0.03, 80)
    assert many > few * 3.0, "과신이 관측 수에 비례해 악화되어야 한다"
