"""관측 잡음 모델과 그 적합.

docs/04-unified-objective.md 5.2 의 핵심 주장을 검증 가능한 형태로 만든 것:
센서 신뢰도는 목적 함수의 *항*이 아니라 관측 모델의 *정밀도* 여야 한다.

세 가지 모델을 같은 인터페이스로 두고 비교한다.
  FixedNoise      - 맑은 조건에서 한 번 적합하고 그대로 (현행 SLAM 관행)
  ScheduledNoise  - 손수 만든 스케줄로 공분산을 부풀림 (현행 WME)
  CalibratedNoise - sigma(z, E) 를 데이터에서 적합 (제안)

셋 다 같은 파라미터 족을 쓰되 자유도만 다르다. 그래야 비교가 공정하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.world import CameraModel
from .optimize import minimize_nelder_mead


@dataclass
class NoiseSample:
    """적합용 관측 한 건. 참값은 시뮬레이션에서만 얻을 수 있다."""
    u: float                 # 박스 중심 픽셀
    v: float
    depth: float             # 측정 깊이
    severity: float          # 조건 강도
    true_point_cam: np.ndarray   # 카메라 좌표 참 위치


class NoiseModel:
    """sigma_px(E), sigma_depth(z, E) -> 3x3 카메라 좌표 공분산."""

    def sigma_pixel(self, severity: float) -> float:
        raise NotImplementedError

    def sigma_depth(self, depth: float, severity: float) -> float:
        raise NotImplementedError

    def covariance(self, u: float, v: float, depth: float, severity: float,
                   cam: CameraModel) -> np.ndarray:
        """픽셀+깊이 잡음을 3D 로 전파한다.

        p = ((u-cx)d/fx, (v-cy)d/fy, d) 의 야코비안:
            dx/du = d/fx,  dx/dd = (u-cx)/fx
            dy/dv = d/fy,  dy/dd = (v-cy)/fy
            dz/dd = 1
        깊이 잡음이 횡방향으로도 새어 들어가는 것이 중요하다 - 이 항을 빼면
        원거리 객체에서 체계적으로 과신하게 된다.
        """
        s_px = self.sigma_pixel(severity)
        s_d = self.sigma_depth(depth, severity)

        J = np.array([
            [depth / cam.fx, 0.0, (u - cam.cx) / cam.fx],
            [0.0, depth / cam.fy, (v - cam.cy) / cam.fy],
            [0.0, 0.0, 1.0],
        ])
        S = np.diag([s_px ** 2, s_px ** 2, s_d ** 2])
        return J @ S @ J.T

    def systematic_covariance(self, severity: float, box_px: float = 0.0,
                              depth: float = 0.0, cam: CameraModel | None = None
                              ) -> np.ndarray:
        """객체당 계통 오차 공분산. 관측 수를 늘려도 줄지 않는 성분.

        이것이 없으면 정보필터가 관측을 독립으로 취급해, 융합할수록 공분산이
        1/N 로 줄어드는 반면 계통 오차는 그대로 남아 과신이 N 에 비례해 커진다.
        실제 SLAM 시스템의 공분산이 한 자릿수 이상 과소평가되는 주된 이유다.
        """
        return np.zeros((3, 3))

    def name(self) -> str:
        return type(self).__name__


@dataclass
class FixedNoise(NoiseModel):
    """조건과 무관하게 고정. 현행 SLAM 의 표준 관행."""
    sigma_px: float = 2.0
    depth_coeff: float = 0.006      # sigma_d = coeff * z^2

    def sigma_pixel(self, severity: float) -> float:
        return self.sigma_px

    def sigma_depth(self, depth: float, severity: float) -> float:
        return self.depth_coeff * depth * depth


@dataclass
class ScheduledNoise(NoiseModel):
    """손수 만든 스케줄로 부풀림.

    현장에서 흔히 하는 대응이고, 현재 WME 의 alpha_k(E) 도 이 부류다.
    reliability 로 나누는 형태를 쓴다 - 정보를 신뢰도에 비례해 줄이는,
    직관적으로 그럴듯한 선택.
    """
    base: FixedNoise
    inflate_power: float = 1.0      # sigma <- sigma / reliability^power

    def _reliability(self, severity: float) -> float:
        return max(0.05, 1.0 - severity)

    def sigma_pixel(self, severity: float) -> float:
        return self.base.sigma_pixel(severity) / self._reliability(severity) ** self.inflate_power

    def sigma_depth(self, depth: float, severity: float) -> float:
        return (self.base.sigma_depth(depth, severity)
                / self._reliability(severity) ** self.inflate_power)


@dataclass
class CalibratedNoise(NoiseModel):
    """sigma(z, E) 를 데이터에서 적합한 모델.

    sigma_px  = c_px * (1 + g_px * s)
    sigma_d   = c_d * z^2 * (1 + g_d * s)
    sigma_sys = s_sys * (1 + g_sys * s)        <- 관측 수로 줄지 않는 성분

    형태는 시뮬레이터의 참 모델과 같지만 계수는 모른다. 적합이 그 계수를
    복원하는지가 M3 게이트의 핵심이다.

    sigma_sys 는 시뮬레이터의 명시적 파라미터가 아니라 모델링 오차에서 나온다
    (박스 중심 != 무게중심 투영 등). 참값이 없으므로 2단계로 적합한다.
    """
    c_px: float = 2.0
    g_px: float = 0.0
    c_d: float = 0.006
    g_d: float = 0.0
    s_sys: float = 0.0        # 크기 무관 바닥값
    k_geom: float = 0.0       # 기하 모델링 오차 계수 (아래 유도 참조)
    g_sys: float = 0.0

    def sigma_pixel(self, severity: float) -> float:
        return self.c_px * (1.0 + self.g_px * severity)

    def sigma_depth(self, depth: float, severity: float) -> float:
        return self.c_d * depth * depth * (1.0 + self.g_d * severity)

    def systematic_covariance(self, severity: float, box_px: float = 0.0,
                              depth: float = 0.0, cam: CameraModel | None = None
                              ) -> np.ndarray:
        """계통 오차는 등방·상수가 아니라 객체 크기에 의존한다.

        기원: 박스 중심은 무게중심의 투영과 일치하지 않는다. 반치수 e 인 객체의
        미터 단위 편향은 대략 e^2 / z 이고, e ~ w_px * z / (2f) 이므로

            sigma_geom ~ k * w_px^2 * z / f^2

        크기가 큰 객체와 가까운 객체가 더 큰 계통 오차를 갖는다. 이 의존성을
        무시하고 상수로 두면 객체마다 오차가 달라 적합이 절충값에 머문다.
        """
        geom = 0.0
        if cam is not None and box_px > 0.0 and depth > 0.0:
            geom = self.k_geom * box_px * box_px * depth / (cam.fx * cam.fy)
        s = (self.s_sys + geom) * (1.0 + self.g_sys * severity)
        return np.eye(3) * (s * s)


# --- 적합 -----------------------------------------------------------------

def negative_log_likelihood(model: NoiseModel, samples: list[NoiseSample],
                            cam: CameraModel) -> float:
    """관측 잔차에 대한 가우시안 NLL.

        NLL = sum_i [ 0.5 r_i^T Sigma_i^-1 r_i + 0.5 log|Sigma_i| ]

    log|Sigma| 항이 필수다. 이게 없으면 공분산을 무한대로 키우는 것이 항상
    이득이 되어 적합이 발산한다. 즉 이 항이 "과대평가도 벌준다"를 담당한다.
    """
    total = 0.0
    for s in samples:
        p_meas = np.array([(s.u - cam.cx) * s.depth / cam.fx,
                           (s.v - cam.cy) * s.depth / cam.fy,
                           s.depth])
        r = p_meas - s.true_point_cam
        S = model.covariance(s.u, s.v, s.depth, s.severity, cam)
        S = S + np.eye(3) * 1e-12

        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(logdet):
            return float("inf")
        total += 0.5 * float(r @ np.linalg.solve(S, r)) + 0.5 * logdet
    return total / max(1, len(samples))


def fit_calibrated(samples: list[NoiseSample], cam: CameraModel,
                   condition_dependent: bool = True) -> CalibratedNoise:
    """최대우도로 잡음 파라미터를 적합한다.

    파라미터는 로그공간에서 다룬다 - 전부 양수여야 하고 스케일이 크게 다르다.
    """
    if len(samples) < 20:
        raise ValueError(f"적합 표본 부족: {len(samples)}")

    def unpack(theta: np.ndarray) -> CalibratedNoise:
        if condition_dependent:
            return CalibratedNoise(c_px=float(np.exp(theta[0])),
                                   g_px=float(np.exp(theta[1])),
                                   c_d=float(np.exp(theta[2])),
                                   g_d=float(np.exp(theta[3])))
        return CalibratedNoise(c_px=float(np.exp(theta[0])), g_px=0.0,
                               c_d=float(np.exp(theta[1])), g_d=0.0)

    def objective(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 20.0):
            return float("inf")
        return negative_log_likelihood(unpack(theta), samples, cam)

    x0 = (np.log([2.0, 1.0, 0.006, 1.0]) if condition_dependent
          else np.log([2.0, 0.006]))
    best, _ = minimize_nelder_mead(objective, x0, step=0.4, max_iter=4000)
    return unpack(best)


def fit_fixed(samples: list[NoiseSample], cam: CameraModel) -> FixedNoise:
    """조건 의존성 없는 모델을 적합한다. 보통 맑은 조건 데이터로만 부른다."""
    c = fit_calibrated(samples, cam, condition_dependent=False)
    return FixedNoise(sigma_px=c.c_px, depth_coeff=c.c_d)


@dataclass
class ObjectResidual:
    """융합 후 객체 하나의 잔차. 계통 성분 적합의 입력."""
    error: np.ndarray
    covariance: np.ndarray        # 계통 성분을 뺀 융합 공분산
    severity: float
    mean_box_px: float
    mean_depth: float


def fit_systematic(model: CalibratedNoise, residuals: list[ObjectResidual],
                   cam: CameraModel, condition_dependent: bool = True) -> CalibratedNoise:
    """2단계 적합: 객체 단위 잔차에서 계통 오차 성분을 추정한다.

    1단계(관측 단위)로는 이 성분을 볼 수 없다. 계통 오차는 한 객체의 모든
    관측에 공통으로 실려 있어서, 관측 하나만 보면 독립 잡음과 구분되지 않는다.
    융합 후 잔차가 1/sqrt(N) 로 줄지 않는 것에서만 드러난다.
    """
    if len(residuals) < 5:
        raise ValueError(f"객체 단위 표본 부족: {len(residuals)}")

    def unpack(theta: np.ndarray) -> CalibratedNoise:
        return CalibratedNoise(
            c_px=model.c_px, g_px=model.g_px, c_d=model.c_d, g_d=model.g_d,
            s_sys=float(np.exp(theta[0])),
            k_geom=float(np.exp(theta[1])),
            g_sys=float(np.exp(theta[2])) if condition_dependent else 0.0,
        )

    def nll(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 20.0):
            return float("inf")
        m = unpack(theta)

        total = 0.0
        for r in residuals:
            S = (r.covariance
                 + m.systematic_covariance(r.severity, r.mean_box_px, r.mean_depth, cam)
                 + np.eye(3) * 1e-12)
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0 or not np.isfinite(logdet):
                return float("inf")
            total += 0.5 * float(r.error @ np.linalg.solve(S, r.error)) + 0.5 * logdet
        return total / len(residuals)

    best, _ = minimize_nelder_mead(nll, np.log([0.005, 0.1, 1.0]),
                                   step=0.6, max_iter=5000)
    fitted = unpack(best)

    # 규모 항이 0 으로 수렴하면 성장 계수는 식별 불가능하다(0 x 무엇이든 0).
    # 의미 없는 큰 값을 남기면 읽는 사람을 오도한다.
    if fitted.s_sys < 1e-5 and fitted.k_geom < 1e-5:
        fitted.s_sys, fitted.k_geom, fitted.g_sys = 0.0, 0.0, 0.0
    return fitted
