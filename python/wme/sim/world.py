"""합성 세계와 카메라 궤적.

왜 자체 시뮬레이터가 필요한가: WME 가 개선한다고 주장하는 것들 -
동일 궤적에서의 안개 강도 스윕, 가려짐을 통과하는 객체 정체성 참값,
"의자 하나만 움직인" 장면, 미래 궤적 참값 - 은 어떤 실제 데이터셋에도 없다.
Sigma_k(E) 를 *적합* 하려면 조건이 알려진 동일 궤적의 반복 주행이 필요하고,
그건 시뮬레이션으로만 얻는다. docs/05-research-program.md 3장 참조.

범위: 객체 수준 관측(검출 + 깊이)까지만 생성한다. 픽셀 렌더링은 하지 않으므로
ECDA 측광 항은 이 하네스로 평가할 수 없다. 연관/보정/변화 연구용이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reference.geometry import SE3, so3_exp


@dataclass
class SimObject:
    """세계에 놓인 하나의 객체."""
    object_id: int
    class_id: int
    class_name: str
    position: np.ndarray                       # 기본 위치 (월드)
    extent: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.3]))
    is_agent: bool = False

    # 이동 객체: (시각, 위치) 웨이포인트. 사이는 선형 보간.
    waypoints: list[tuple[float, np.ndarray]] | None = None

    # 등장/퇴장 시각. 장면 변화(추가/제거) 시나리오에 쓴다.
    present_from: float = -np.inf
    present_until: float = np.inf

    def is_present(self, t: float) -> bool:
        return self.present_from <= t < self.present_until

    def position_at(self, t: float) -> np.ndarray | None:
        if not self.is_present(t):
            return None
        if not self.waypoints:
            return self.position

        wp = self.waypoints
        if t <= wp[0][0]:
            return wp[0][1]
        if t >= wp[-1][0]:
            return wp[-1][1]
        for (t0, p0), (t1, p1) in zip(wp, wp[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                return (1.0 - a) * np.asarray(p0) + a * np.asarray(p1)
        return wp[-1][1]

    def velocity_at(self, t: float, dt: float = 0.05) -> np.ndarray:
        a, b = self.position_at(t - dt * 0.5), self.position_at(t + dt * 0.5)
        if a is None or b is None:
            return np.zeros(3)
        return (b - a) / dt


@dataclass
class SimWorld:
    objects: list[SimObject]

    def present_at(self, t: float) -> list[SimObject]:
        return [o for o in self.objects if o.is_present(t)]

    def positions_at(self, t: float) -> dict[int, np.ndarray]:
        out = {}
        for o in self.objects:
            p = o.position_at(t)
            if p is not None:
                out[o.object_id] = p
        return out

    def by_id(self, object_id: int) -> SimObject | None:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        return None


@dataclass
class CameraTrajectory:
    """월드 <- 카메라 포즈열. 카메라 광축은 +z."""
    stamps: np.ndarray
    poses: list[SE3]

    def __len__(self) -> int:
        return len(self.poses)

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray,
                 up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> SE3:
        """카메라 z축이 target 을 향하도록 하는 월드<-카메라 변환."""
        z = target - eye
        n = np.linalg.norm(z)
        z = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else z / n

        x = np.cross(up, z)
        nx = np.linalg.norm(x)
        if nx < 1e-6:                       # up 과 시선이 평행한 퇴화 상황
            x = np.cross(np.array([0.0, 1.0, 0.0]), z)
            nx = np.linalg.norm(x)
        x = x / nx
        y = np.cross(z, x)
        return SE3(np.column_stack([x, y, z]), eye)

    @staticmethod
    def loop(radius: float = 3.0, height: float = 1.2, laps: float = 1.0,
             n: int = 200, dt: float = 0.05,
             center: np.ndarray | None = None) -> "CameraTrajectory":
        """원형 주회. 루프 클로저와 재방문 시나리오의 기본 궤적."""
        center = np.zeros(3) if center is None else np.asarray(center, float)
        stamps, poses = [], []
        for i in range(n):
            a = 2.0 * np.pi * laps * i / max(1, n - 1)
            eye = center + np.array([radius * np.cos(a), radius * np.sin(a), height])
            poses.append(CameraTrajectory._look_at(eye, center + np.array([0, 0, height * 0.6])))
            stamps.append(1.0 + i * dt)
        return CameraTrajectory(np.array(stamps), poses)

    @staticmethod
    def corridor(length: float = 12.0, height: float = 1.2, n: int = 200,
                 dt: float = 0.05, yaw_amplitude: float = 0.10) -> "CameraTrajectory":
        """직선 전진 + 약한 요동. 텍스처 빈곤/퇴화 시나리오용."""
        stamps, poses = [], []
        for i in range(n):
            s = i / max(1, n - 1)
            eye = np.array([0.0, s * length, height])
            yaw = yaw_amplitude * np.sin(6.0 * np.pi * s)
            # 기본 시선은 +y, 카메라 z축을 그쪽으로 돌린다
            R = so3_exp([0.0, 0.0, yaw]) @ np.column_stack(
                [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])])
            poses.append(SE3(R, eye))
            stamps.append(1.0 + i * dt)
        return CameraTrajectory(np.array(stamps), poses)

    @staticmethod
    def spiral(r_start: float = 8.0, r_end: float = 1.8, height: float = 1.2,
               laps: float = 2.0, n: int = 200, dt: float = 0.05,
               center: np.ndarray | None = None) -> "CameraTrajectory":
        """멀리서 시작해 가까이 접근한다.

        관측 정밀도가 시간에 따라 좋아지는 궤적이다. 초반에는 모호했던 연관이
        후반의 정밀 관측으로 해소되므로, 필터와 평활기의 차이가 드러난다.
        모호성이 끝까지 해소되지 않는 장면에서는 어떤 다중가설 기법도
        이득을 낼 수 없다 - 없는 정보를 만들어낼 수는 없기 때문이다.
        """
        center = np.zeros(3) if center is None else np.asarray(center, float)
        stamps, poses = [], []
        for i in range(n):
            s = i / max(1, n - 1)
            a = 2.0 * np.pi * laps * s
            r = r_start + (r_end - r_start) * s
            eye = center + np.array([r * np.cos(a), r * np.sin(a), height])
            poses.append(CameraTrajectory._look_at(eye, center + np.array([0, 0, height * 0.6])))
            stamps.append(1.0 + i * dt)
        return CameraTrajectory(np.array(stamps), poses)

    @staticmethod
    def revisit(radius: float = 3.0, height: float = 1.2, n_per_visit: int = 120,
                gap_seconds: float = 30.0, dt: float = 0.05) -> "CameraTrajectory":
        """같은 장소를 두 번 방문한다. 두 방문 사이에 시간 간격이 있다.

        장면 변화 검출 시나리오의 필수 궤적 - 변화는 두 관측 사이에만 정의된다.
        """
        first = CameraTrajectory.loop(radius, height, 1.0, n_per_visit, dt)
        second = CameraTrajectory.loop(radius, height, 1.0, n_per_visit, dt)
        t0 = first.stamps[-1] + gap_seconds
        return CameraTrajectory(
            np.concatenate([first.stamps, t0 + (second.stamps - second.stamps[0])]),
            first.poses + second.poses,
        )


@dataclass
class CameraModel:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        z = max(p_cam[2], 1e-9)
        return np.array([self.fx * p_cam[0] / z + self.cx,
                         self.fy * p_cam[1] / z + self.cy])

    def in_image(self, px: np.ndarray, margin: float = 0.0) -> bool:
        return (margin <= px[0] < self.width - margin
                and margin <= px[1] < self.height - margin)
