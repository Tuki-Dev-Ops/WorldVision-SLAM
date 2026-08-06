"""예측 엔진 - "무엇이 바뀔 것인가".

docs/04-unified-objective.md 5.3 의 제약을 자료구조로 강제한다: 예측은 관측과
같은 목적 함수에 들어가면 안 된다. 같은 목적에 넣으면 최적화기가 *현재를 예측에
맞추는* 방식으로 비용을 줄이고, 그러면 진짜 놀라움이 뭉개진다. 놀라움이야말로
무언가 바뀌었다는 신호이므로, 그걸 억제하는 월드 모델은 자기 존재 이유를 잃는다.

따라서 여기서 나오는 Forecast 는 절대 TokenBelief.position 에 쓰이지 않는다.
쓰임새는 둘뿐이다.
  1. 다음 시각의 사전분포 (팩터그래프의 운동 사전분포로 넘어감)
  2. 나중 관측과 대조해 **채점** 되는 반증 가능한 주장

두 번째가 놀라움 검출이고, 그것이 변화 검출의 조기 신호가 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import TokenBelief, WorldSnapshot


@dataclass(frozen=True)
class Forecast:
    """미래 시각에 대한 반증 가능한 주장. 관측과 물리적으로 분리된 타입이다."""

    token_id: int
    horizon: float                        # 예측 대상까지의 시간 (s)
    stamp: float                          # 예측 대상 시각
    position: np.ndarray
    covariance: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    visibility: float = 1.0               # 그때 보일 확률
    reliability: float = 1.0              # 예측기 자체의 신뢰도

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(np.trace(self.covariance) / 3.0, 0.0)))


@dataclass
class PredictionConfig:
    # 등속 모델의 공정 잡음. 시간에 따라 불확실성이 자라야 한다.
    accel_sigma: float = 0.6              # m/s^2, 미지 가속도
    velocity_decay: float = 0.85          # 초당. 사람은 계속 같은 속도로 걷지 않는다

    # 정적으로 믿는 객체는 예측 자체가 거의 필요 없지만, 그 믿음이 틀릴
    # 여지를 남겨야 한다. static_belief 로 공정 잡음을 축소한다.
    static_noise_scale: float = 0.05

    # 놀라움 판정 (자유도 3 카이제곱)
    surprise_chi2: float = 11.34          # 99%
    min_observations: int = 3


class PredictionEngine:
    """등속 + 감쇠 모델. 단순하지만 불확실성이 정직한 것이 요점이다."""

    def __init__(self, config: PredictionConfig | None = None):
        self.cfg = config or PredictionConfig()

    def forecast(self, belief: TokenBelief, horizon: float,
                 now: float | None = None) -> Forecast:
        """horizon 초 뒤를 예측한다. belief 는 변경하지 않는다."""
        cfg = self.cfg
        dt = max(0.0, float(horizon))
        now = belief.last_seen if now is None else now

        # 속도는 시간이 지날수록 신뢰할 수 없다
        decay = cfg.velocity_decay ** dt
        v = np.asarray(belief.velocity, float) * decay
        mean = np.asarray(belief.position, float) + v * dt

        # 미지 가속도가 만드는 위치 불확실성: (1/2 a t^2)^2
        # 정적이라고 믿을수록 이 항을 줄이되 0 으로 두지는 않는다.
        motion_scale = 1.0 - (1.0 - cfg.static_noise_scale) * belief.static_belief
        q = (0.5 * cfg.accel_sigma * motion_scale * dt * dt) ** 2

        cov = np.asarray(belief.covariance, float) + np.eye(3) * q

        # 관측이 적으면 속도 자체가 못 믿을 값이다
        reliability = min(1.0, belief.observation_count / max(1, cfg.min_observations))
        return Forecast(
            token_id=belief.token_id,
            horizon=dt,
            stamp=now + dt,
            position=mean,
            covariance=cov,
            velocity=v,
            visibility=float(np.clip(belief.existence, 0.0, 1.0)),
            reliability=float(reliability * belief.existence),
        )

    def forecast_all(self, snapshot: WorldSnapshot, horizon: float) -> list[Forecast]:
        return [self.forecast(t, horizon, snapshot.stamp) for t in snapshot]

    # --- 채점 -------------------------------------------------------------

    def surprise(self, forecast: Forecast, observed_position: np.ndarray,
                 observation_covariance: np.ndarray | None = None) -> float:
        """정규화된 혁신(NIS). 1 근처면 예측대로, 크면 놀라움.

        예측을 관측에 맞춰 고치는 것이 아니라 *어긋난 정도를 재는* 것이 핵심이다.
        이 값이 커지는 것이 세계가 바뀌었다는 조기 신호다.
        """
        diff = np.asarray(observed_position, float) - forecast.position
        S = np.asarray(forecast.covariance, float)
        if observation_covariance is not None:
            S = S + np.asarray(observation_covariance, float)
        S = S + np.eye(3) * 1e-9
        try:
            return float(diff @ np.linalg.solve(S, diff)) / 3.0
        except np.linalg.LinAlgError:
            return float("inf")

    def is_surprising(self, forecast: Forecast, observed_position: np.ndarray,
                      observation_covariance: np.ndarray | None = None) -> bool:
        nis = self.surprise(forecast, observed_position, observation_covariance) * 3.0
        return nis > self.cfg.surprise_chi2

    def score(self, forecasts: list[Forecast],
              observations: dict[int, np.ndarray]) -> dict[str, float]:
        """예측 품질 요약. eval.metrics.prediction_metrics 에 넘길 재료."""
        errors, nis = [], []
        for f in forecasts:
            z = observations.get(f.token_id)
            if z is None:
                continue
            errors.append(float(np.linalg.norm(np.asarray(z, float) - f.position)))
            nis.append(self.surprise(f, z))
        if not errors:
            return {"count": 0, "ade": float("nan"), "mean_nis": float("nan")}
        return {
            "count": float(len(errors)),
            "ade": float(np.mean(errors)),
            "mean_nis": float(np.mean(nis)),
        }
