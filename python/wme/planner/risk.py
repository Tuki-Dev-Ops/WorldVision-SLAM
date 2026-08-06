"""위험도 추정 - 신념층이 만든 것을 실제로 소비하는 곳.

위험도는 별도 센서가 아니라 파생량이다. 필요한 재료는 이미 전부 있다:
위치와 공분산, 속도, 예측, 어포던스, 그리고 무엇을 아직 못 봤는지.

설계상 반드시 지켜야 할 두 가지.

  1. **불확실하면 더 위험하다.** 평균 거리만 보면 "1.5 m 떨어져 있으니 안전"
     이라고 말하는데, 그 위치의 표준편차가 1 m 면 전혀 안전하지 않다.
     충돌 확률을 접근 방향으로 투영한 공분산으로 계산하는 이유다.

  2. **모르는 공간은 빈 공간이 아니다.** 관측되지 않은 곳을 자유공간으로
     취급하는 것이 계획기의 고전적 치명상이다. 여기서는 관측 이력에서
     커버리지를 만들고, 커버리지 밖을 명시적 위험으로 센다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..reference.geometry import SE3
from ..world.graph import is_agent
from ..world.prediction import Forecast, PredictionEngine
from ..world.state import TokenBelief, WorldSnapshot

# 위험한 어포던스를 가진 클래스. 어휘 표를 쓰는 이유는 다른 곳과 같다 -
# 학습 분류기를 더 붙이면 YOLO-only 제약 위반이다.
_HAZARD = {"car", "bus", "truck", "motorcycle", "bicycle", "train",
           "oven", "toaster", "scissors", "knife", "stairs"}


@dataclass
class RiskConfig:
    robot_radius: float = 0.35
    horizon: float = 2.0                  # 예측 시계 (s)
    horizon_steps: int = 4

    collision_weight: float = 1.0
    hazard_weight: float = 0.4
    unobserved_weight: float = 0.6

    # 미관측 공간 판정
    sensor_range: float = 6.0
    sensor_hfov: float = 1.25             # rad, 수평 반각의 2배가 아니라 반각
    coverage_decay: float = 2.0           # m, 커버리지 밖으로 나갈수록 감쇠

    # 충돌 확률 계산 시 공분산 하한. 0 으로 두면 확신이 무한대가 된다.
    min_sigma: float = 0.05

    # 정적이라고 믿는 물체는 예측 위험이 낮지만 0 은 아니다
    static_discount: float = 0.35


@dataclass
class RiskAssessment:
    """위험도 분해. 단일 스칼라만 주면 왜 위험한지 알 수 없다."""

    total: float
    collision: float = 0.0
    hazard: float = 0.0
    unobserved: float = 0.0
    contributors: list[tuple[int, float]] = field(default_factory=list)

    def summary(self) -> str:
        top = ", ".join(f"#{tid}:{r:.2f}" for tid, r in self.contributors[:3])
        return (f"risk {self.total:.3f} "
                f"(collision {self.collision:.3f}, hazard {self.hazard:.3f}, "
                f"unobserved {self.unobserved:.3f})" + (f" [{top}]" if top else ""))


class Coverage:
    """무엇을 봤는지. 미관측 공간을 자유공간으로 착각하지 않기 위해 필요하다."""

    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self._poses: list[SE3] = []

    def add(self, pose: SE3) -> None:
        self._poses.append(pose)

    def extend(self, poses: list[SE3]) -> None:
        self._poses.extend(poses)

    def observed(self, point: np.ndarray) -> float:
        """이 점이 관측된 적 있는가 (0..1).

        시야각과 거리로만 판정한다. 가려짐은 반영하지 못하므로 이 값은
        상한이다 - 즉 실제보다 낙관적이며, 그 사실을 여기 적어 둔다.
        """
        if not self._poses:
            return 0.0
        cfg = self.cfg
        p = np.asarray(point, float)
        best = 0.0

        for pose in self._poses:
            rel = pose.inverse() @ p
            z = float(rel[2])
            if z <= 0.05 or z > cfg.sensor_range:
                continue
            angle = math.atan2(math.hypot(float(rel[0]), float(rel[1])), z)
            if angle > cfg.sensor_hfov:
                continue

            # 편안한 영역 안에서는 1, 한계 근처에서만 떨어진다.
            # 거리에 선형으로 두면 6 m 센서로 3 m 정면을 봐도 0.5 가 나와
            # '얼마나 잘 봤나' 와 '봤나 못 봤나' 가 뒤섞인다.
            edge = 0.4
            range_score = (cfg.sensor_range - z) / (edge * cfg.sensor_range)
            angle_score = (cfg.sensor_hfov - angle) / (edge * cfg.sensor_hfov)
            score = min(np.clip(range_score, 0.0, 1.0), np.clip(angle_score, 0.0, 1.0))
            best = max(best, float(score))
            if best > 0.99:
                break
        return best

    def __len__(self) -> int:
        return len(self._poses)


class RiskEstimator:
    """믿음과 예측에서 위험도를 유도한다."""

    def __init__(self, config: RiskConfig | None = None,
                 predictor: PredictionEngine | None = None):
        self.cfg = config or RiskConfig()
        self.predictor = predictor or PredictionEngine()

    # --- 충돌 -------------------------------------------------------------

    def collision_probability(self, point: np.ndarray, position: np.ndarray,
                              covariance: np.ndarray, extent: np.ndarray) -> float:
        """불확실성을 반영한 충돌 확률.

        평균 거리만 보면 "1.5 m 떨어져 있으니 안전" 이라고 말하는데, 그 위치의
        표준편차가 1 m 면 전혀 안전하지 않다. 접근 방향으로 공분산을 투영해
        그 축의 표준편차로 판정한다.
        """
        cfg = self.cfg
        p = np.asarray(point, float)
        x = np.asarray(position, float)

        diff = p - x
        distance = float(np.linalg.norm(diff))
        if distance < 1e-9:
            return 1.0

        u = diff / distance
        var = float(u @ np.asarray(covariance, float) @ u)
        sigma = max(math.sqrt(max(var, 0.0)), cfg.min_sigma)

        # 물체 반치수를 접근 방향으로 투영
        e = np.asarray(extent, float)
        reach = float(np.abs(u) @ e) + cfg.robot_radius
        margin = distance - reach

        # P(margin < 0) 을 정규 근사로
        return float(np.clip(0.5 * (1.0 - math.erf(margin / (sigma * math.sqrt(2.0)))),
                             0.0, 1.0))

    def time_to_collision(self, point: np.ndarray, token: TokenBelief) -> float:
        """상대 접근 속도 기준 충돌까지 시간. 멀어지면 inf."""
        cfg = self.cfg
        p = np.asarray(point, float)
        x = np.asarray(token.position, float)
        v = np.asarray(token.velocity, float)

        diff = p - x
        distance = float(np.linalg.norm(diff))
        if distance < 1e-9:
            return 0.0

        u = diff / distance
        closing = float(v @ u)                # 로봇 쪽으로 다가오는 성분
        if closing <= 1e-6:
            return float("inf")

        e = np.asarray(token.extent, float)
        reach = float(np.abs(u) @ e) + cfg.robot_radius
        return max(0.0, (distance - reach) / closing)

    # --- 종합 -------------------------------------------------------------

    def at(self, point: np.ndarray, snapshot: WorldSnapshot,
           coverage: Coverage | None = None) -> RiskAssessment:
        """한 지점의 위험도."""
        cfg = self.cfg
        p = np.asarray(point, float)

        collision = 0.0
        hazard = 0.0
        contributors: list[tuple[int, float]] = []

        steps = max(1, cfg.horizon_steps)
        horizons = [cfg.horizon * (i + 1) / steps for i in range(steps)]

        for token in snapshot:
            # 현재 위치에서의 충돌
            worst = self.collision_probability(p, token.position, token.covariance,
                                               token.extent)

            # 예측 위치에서의 충돌. 정적이라고 믿을수록 할인하되 0 은 아니다.
            if not token.is_dynamic or is_agent(token.class_name):
                for h in horizons:
                    f: Forecast = self.predictor.forecast(token, h)
                    pf = self.collision_probability(p, f.position, f.covariance,
                                                    token.extent)
                    discount = (cfg.static_discount if token.static_belief > 0.5
                                else 1.0)
                    worst = max(worst, pf * discount * f.reliability)

            # 존재를 확신 못 하는 물체도 위험은 위험이다. 다만 비례해 줄인다.
            worst *= max(token.existence, 0.0)
            if worst <= 1e-4:
                continue

            collision = max(collision, worst)
            if token.class_name in _HAZARD:
                hazard = max(hazard, worst)
            contributors.append((token.token_id, worst))

        unobserved = 0.0
        if coverage is not None:
            # 모르는 공간은 빈 공간이 아니다
            unobserved = 1.0 - coverage.observed(p)

        total = float(np.clip(
            cfg.collision_weight * collision
            + cfg.hazard_weight * hazard
            + cfg.unobserved_weight * unobserved, 0.0, 1.0))

        contributors.sort(key=lambda c: -c[1])
        return RiskAssessment(total, collision, hazard, unobserved, contributors)

    def along(self, path: list[np.ndarray], snapshot: WorldSnapshot,
              coverage: Coverage | None = None) -> RiskAssessment:
        """경로 전체의 위험도. 최악 지점이 경로를 대표한다.

        평균을 쓰면 짧은 치명 구간이 긴 안전 구간에 희석된다.
        """
        if not path:
            return RiskAssessment(0.0)

        worst = RiskAssessment(-1.0)
        for point in path:
            r = self.at(point, snapshot, coverage)
            if r.total > worst.total:
                worst = r
        return worst

    def safest(self, candidates: list[np.ndarray], snapshot: WorldSnapshot,
               coverage: Coverage | None = None) -> tuple[int, RiskAssessment]:
        """후보 지점 중 가장 안전한 곳의 (인덱스, 평가)."""
        if not candidates:
            raise ValueError("후보가 없음")
        scored = [(i, self.at(p, snapshot, coverage)) for i, p in enumerate(candidates)]
        return min(scored, key=lambda s: s[1].total)
