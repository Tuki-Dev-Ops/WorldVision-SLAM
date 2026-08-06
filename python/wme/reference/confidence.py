"""신뢰도 엔진 참조 구현.

C++ src/confidence/ConfidenceEngine.cpp 와 동일한 로그오즈 갱신.
existence / identity / static 을 분리하는 이유는 이들이 따로 실패하기 때문이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceConfig:
    p_detect_visible: float = 0.90
    p_false_alarm: float = 0.03
    logodds_min: float = -4.0
    logodds_max: float = 4.0
    # 연속 프레임의 검출은 독립 관측이 아니다. 같은 오검출을 100번 봐도
    # 증거 100개가 아니다. 이 이득으로 프레임 간 상관을 반영해 감쇠하지 않으면
    # 두세 프레임 만에 포화해 증거의 세기를 구분하지 못하게 된다.
    evidence_gain: float = 0.25
    # 자율 개체가 "정적" 이라고 주장할 때만 증거를 약화시킨다. 동적 증거는
    # 감쇠하지 않는다 - 움직이는 것을 놓치는 대가가 더 크다. 금지가 아니라 회의다.
    agent_static_evidence_gain: float = 0.5
    min_visibility_for_penalty: float = 0.35
    identity_margin_scale: float = 1.5
    identity_decay: float = 0.02

    # --- 동적/정적 판정 ---------------------------------------------------
    # 판정 창. 운동은 v*T 로 자라고 관측 잡음은 T 와 무관하므로 분리도는 이 값이
    # 정한다. 0.05 s 에서 실측 분리비 1.20(sitting)/1.44(walking) - 사실상 구분이
    # 없다. 0.5 s 에서 2.26/3.90. (docs/06-results.md 16.2)
    static_min_dt: float = 0.50
    # 정지 물체가 이 창에서 실제로 보이는 변위의 표준편차 하한 (m).
    # 카메라 포즈 오차가 공통모드로 섞이므로 순수 관측 지터보다 크다.
    # 실측(고정된 tv/keyboard/chair, T=0.5 s): 중앙값 26 mm / 51 mm -> sigma 17/33 mm.
    motion_noise_floor: float = 0.035
    # 동적 가설이 예측하는 속도 (m/s). 사람의 느린 보행.
    dynamic_speed_ref: float = 0.7
    # 정적 가설의 꼬리 무게. 이 항이 없으면 우도비가 아래로 무계라 이상치 한 번
    # (실측 -453 nat)이 10 초치 증거를 지운다. 고정된 물체조차 모델의 3 sigma 를
    # 16 %(sitting) / 37 %(walking) 넘는다 - 꼬리는 예외가 아니라 상시다.
    static_outlier_rate: float = 0.2
    # 자율 개체의 "정적" 주장 반감기 (s). 신호등 앞의 차는 금지가 아니라
    # 유효기간으로 다룬다 - 관측이 끊기면 주장이 사전분포로 돌아간다.
    agent_static_halflife: float = 4.0


def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def _safe_prob(p: float) -> float:
    """0/1 에서 떼어놓는다. 로그오즈가 발산하면 그 믿음은 영원히 갱신 불가가 된다."""
    return min(max(p, 1e-6), 1.0 - 1e-6)


def to_logodds(p: float) -> float:
    s = _safe_prob(p)
    return math.log(s / (1.0 - s))


def to_probability(l: float) -> float:
    return 1.0 / (1.0 + math.exp(-l))


@dataclass
class StaticDiag:
    """정적 판정의 원자료. 출력만 보고는 왜 그렇게 됐는지 알 수 없다."""
    updates: int = 0
    disp: float = 0.0
    sigma: float = 0.0
    z: float = 0.0
    ratio: float = 0.0


@dataclass
class Beliefs:
    """토큰이 들고 다니는 세 가지 믿음."""
    existence: float = 0.5
    identity: float = 0.5
    static: float = 0.5
    miss_count: int = 0
    is_agent: bool = False
    # 판정 척도는 *관측* 불확실성이다. 융합 추정의 불확실성(positionSigma)은
    # 관측 수에 따라 줄어드는 반면 창 변위는 그렇지 않다 - 예측하는 통계가
    # 다르다 (10.2 의 반복된 실수, 여섯 번째).
    meas_sigma: float = 0.0
    # 생성 시 사전분포. 관측이 끊긴 자율 개체는 여기로 돌아간다.
    static_prior: float = 0.5
    static_diag: StaticDiag = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.static_diag is None:
            self.static_diag = StaticDiag()


class ConfidenceEngine:
    def __init__(self, config: ConfidenceConfig | None = None):
        self.cfg = config or ConfidenceConfig()

    def _apply(self, belief: float, log_ratio: float, reliability: float) -> float:
        """증거의 방향은 유지하고 세기만 신뢰도로 줄인다."""
        scaled = log_ratio * self.cfg.evidence_gain * _clamp01(reliability)
        l = to_logodds(belief) + scaled
        l = min(max(l, self.cfg.logodds_min), self.cfg.logodds_max)
        return to_probability(l)

    def on_observed(self, b: Beliefs, detection_conf: float, sensor_reliability: float,
                    image_quality: float = 1.0, assoc_margin: float = -1.0,
                    obs_reliability: float = 1.0) -> None:
        """관측 연관 시 갱신.

        C++ 는 환경 신뢰도(env)와 이 관측 자체의 신뢰도(obs)를 따로 받아 곱한다.
        둘을 하나로 합치면 "센서는 멀쩡한데 이 검출만 나쁘다"를 표현할 수 없다.
        """
        cfg = self.cfg
        # C++ Observation::detection_conf 는 float 다 (YOLO 출력 그대로). 그래서
        # 검출 신뢰도는 단정밀도로 잘린 뒤 double 로 승격된다. 여기서 double 을
        # 그대로 쓰면 두 구현이 6e-10 ~ 2e-9 만큼 갈리는데, 그 차이는 기존
        # np.isclose(atol=1e-12) 가 *기본 rtol=1e-5* 를 함께 쓰기 때문에 통째로
        # 가려져 있었다. 저장 형식을 참조 구현에서도 그대로 모사한다.
        conf = _clamp01(float(np.float32(detection_conf)))
        p_det = _safe_prob(cfg.p_detect_visible * conf)
        p_fa = _safe_prob(cfg.p_false_alarm)
        ratio = math.log(p_det / p_fa)

        reliability = (_clamp01(sensor_reliability) * _clamp01(obs_reliability)
                       * _clamp01(0.3 + 0.7 * image_quality))

        b.existence = self._apply(b.existence, ratio, reliability)

        if assoc_margin >= 0.0:
            # 1등/2등 격차가 클수록 다른 개체와 헷갈렸을 확률이 낮다
            margin = assoc_margin / max(1e-6, cfg.identity_margin_scale)
            b.identity = self._apply(b.identity, math.log1p(margin) * 2.0, reliability)
        else:
            b.identity = self._apply(b.identity, 0.4, reliability)

        b.miss_count = 0

    def on_missed(self, b: Beliefs, expected_visibility: float,
                  sensor_reliability: float) -> None:
        cfg = self.cfg
        b.miss_count += 1
        # 가려졌거나 시야 밖이면 부재의 증거가 아니다
        if expected_visibility < cfg.min_visibility_for_penalty:
            return

        p_det = _safe_prob(cfg.p_detect_visible * _clamp01(expected_visibility))
        p_fa = _safe_prob(cfg.p_false_alarm)
        ratio = math.log((1.0 - p_det) / (1.0 - p_fa))
        b.existence = self._apply(b.existence, ratio, _clamp01(sensor_reliability))

    def on_out_of_view(self, b: Beliefs, dt: float) -> None:
        """존재 믿음은 유지. 안 보인다는 것은 없다는 뜻이 아니다."""
        decay = math.exp(-self.cfg.identity_decay * max(0.0, dt))
        b.identity = 0.5 + (b.identity - 0.5) * decay

    def update_static(self, b: Beliefs, displacement, dt: float,
                      sensor_reliability: float) -> None:
        """창 양 끝 *원시 관측* 변위로 정적/동적 믿음을 갱신한다.

        이전 구현은 [-0.31, +0.38] 로 유계인 임의 함수여서, 아무리 결정적인
        관측이어도 증거의 세기가 상수에 묶였다. 실측: sitting 과 walking 모두
        갱신의 84 % 가 양의 증거였고 - 부호가 전혀 구분되지 않았다 - 믿음은
        사실상 토큰이 몇 번 살아남았는지만 셌다 (16.1).

          H_s (정적) : 변위 ~ N(0, sigma_d^2 I),  sigma_d = 독립 관측 두 번의 차
          H_d (동적) : 변위 ~ N(0, sigma_m^2 I),  sigma_m^2 = sigma_d^2 + (v*dt)^2
        """
        cfg = self.cfg
        if dt < cfg.static_min_dt:
            return

        d = float(np.linalg.norm(np.asarray(displacement, float)))

        # 관측 불확실성. 독립 관측 두 번의 차이므로 sqrt(2) 배다.
        sigma_d = max(cfg.motion_noise_floor, math.sqrt(2.0) * b.meas_sigma)
        travel = cfg.dynamic_speed_ref * dt
        sigma_m = math.sqrt(sigma_d * sigma_d + travel * travel)

        d2 = d * d
        # 3 자유도 등방 가우시안의 로그우도비. 변위 0 에서 3*log(sigma_m/sigma_d)
        # 로 포화한다 - 안 움직였다는 것은 상한이 있는 증거다 (동적 물체도 멈춘다).
        llr = (3.0 * math.log(sigma_m / sigma_d)
               + 0.5 * d2 * (1.0 / (sigma_m * sigma_m) - 1.0 / (sigma_d * sigma_d)))
        z = d / sigma_d

        # 정적 가설에 꼬리를 붙인다: p(d|H_s) = (1-e)N(0,sd^2) + e N(0,sm^2).
        # 우도비는 log(e) 아래로 내려가지 않는다 - 이상치 한 번이 결론을 뒤집을
        # 수는 있어도 지워 버릴 수는 없어야 한다.
        eps = min(max(cfg.static_outlier_rate, 0.0), 0.5)
        m = max(llr, 0.0)
        if eps <= 0.0:
            ratio = llr
        else:
            ratio = m + math.log((1.0 - eps) * math.exp(llr - m) + eps * math.exp(-m))

        # 자율 개체가 "정적" 이라고 주장하려면 더 많은 증거가 필요하다.
        # 동적이라는 증거는 그대로 받는다 - 비대칭이 맞다.
        gain = cfg.agent_static_evidence_gain if (b.is_agent and ratio > 0.0) else 1.0

        b.static = self._apply(b.static, ratio * gain, _clamp01(sensor_reliability))

        b.static_diag.updates += 1
        b.static_diag.disp = d
        b.static_diag.sigma = sigma_d
        b.static_diag.z = z
        b.static_diag.ratio = ratio

        # 상한을 씌우면 안 된다. 실측 대가: fr3_sitting 에서 앉아 있는 사람이
        # 통째로 마스킹되어 ATE 가 1.01 -> 15.84 cm 로 무너졌다. 그 사람들은
        # 벽만큼 좋은 랜드마크다. 그 요구는 유효기간으로 다룬다 - decay_static 참조.

    def decay_static_belief(self, b: Beliefs, dt: float) -> None:
        """관측이 끊긴 동안 자율 개체의 정적 주장을 사전분포로 되돌린다."""
        cfg = self.cfg
        # 책상은 안 보이는 동안 걸어다니지 않는다
        if not b.is_agent:
            return
        if dt <= 0.0 or cfg.agent_static_halflife <= 0.0:
            return

        l0 = to_logodds(b.static_prior)
        k = math.pow(0.5, dt / cfg.agent_static_halflife)
        b.static = to_probability(l0 + (to_logodds(b.static) - l0) * k)

    def merge(self, keep: Beliefs, absorbed: Beliefs) -> None:
        cfg = self.cfg

        def combine(a: float, c: float) -> float:
            l = min(max(to_logodds(a) + to_logodds(c), cfg.logodds_min), cfg.logodds_max)
            return to_probability(l)

        keep.existence = combine(keep.existence, absorbed.existence)
        keep.static = combine(keep.static, absorbed.static)
        # 병합 자체가 "동일 개체" 주장이므로 정체성은 보수적으로 취급한다
        keep.identity = min(max(keep.identity * absorbed.identity
                                + 0.5 * abs(keep.identity - absorbed.identity), 0.0), 1.0)
