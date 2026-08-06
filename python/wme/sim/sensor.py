"""조건 의존 관측 모델.

핵심: 잡음이 등분산이 아니다. 검출 확률, 박스 잡음, 깊이 잡음, 클래스 혼동,
오검출률이 전부 EnvironmentEvidence 의 함수다. 이것이 이 하네스의 존재 이유다 -
참 잡음 모델을 알고 있으므로, 추정기가 그것을 복원하는지(= Sigma_k(E) 보정이
되는지) 검증할 수 있다. docs/04-unified-objective.md 5.2 참조.

모든 난수는 주입된 seed 에서 나온다. 재현되지 않는 실험은 실험이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reference.environment import Evidence
from ..reference.geometry import SE3
from ..yolo import Detection, DetectionSet
from .world import CameraModel, SimObject, SimWorld


@dataclass
class SensorConfig:
    # 맑은 조건 기준값
    base_detection_prob: float = 0.95
    base_box_noise_px: float = 2.0
    depth_noise_coeff: float = 0.006      # sigma_z = coeff * z^2
    base_false_alarm_rate: float = 0.02   # 프레임당 기대 오검출 수
    class_confusion_prob: float = 0.01

    min_range: float = 0.4
    max_range: float = 25.0
    min_box_px: float = 12.0
    occlusion_threshold: float = 0.7      # 이 비율 이상 가려지면 미검출

    # 열화 결합 계수 - 참 잡음 모델의 파라미터.
    # 추정기가 이 값들을 복원해야 보정이 성공한 것이다.
    detection_falloff: float = 2.5        # 가시도 저하가 검출확률을 깎는 세기
    box_noise_growth: float = 4.0
    depth_noise_growth: float = 3.0
    false_alarm_growth: float = 8.0
    confusion_growth: float = 6.0


@dataclass
class FrameTruth:
    """한 프레임의 완전한 참값. 실제 데이터셋은 이 중 일부만 제공한다."""
    stamp: float
    T_world_cam: SE3
    evidence: Evidence
    # DetectionSet.items 와 인덱스가 정렬됨. 오검출은 -1.
    detection_to_object: list[int] = field(default_factory=list)
    visible_objects: set[int] = field(default_factory=set)     # 시야 안 + 미가려짐
    detected_objects: set[int] = field(default_factory=set)    # 실제로 검출된 것
    object_positions: dict[int, np.ndarray] = field(default_factory=dict)
    object_velocities: dict[int, np.ndarray] = field(default_factory=dict)
    # 검출별 참 깊이와 주입된 깊이 잡음 표준편차 (보정 연구용)
    detection_depth: list[float] = field(default_factory=list)
    detection_depth_sigma: list[float] = field(default_factory=list)


def severity_of(ev: Evidence) -> float:
    """증거를 하나의 열화 강도로 요약한다 (0 = 맑음, 1 = 극심).

    관측 모델 내부에서만 쓰는 편의값이다. 엔진은 이런 스칼라를 쓰지 않는다 -
    tier 별로 다르게 반응해야 하기 때문이다.
    """
    return float(np.clip(
        1.0 - (1.0 - 0.9 * ev.haze) * (1.0 - 0.7 * ev.darkness)
        * (1.0 - 0.5 * ev.rain_streak) * (1.0 - 0.5 * ev.snow_particle)
        * (1.0 - 0.6 * ev.motion_blur) * (1.0 - 0.5 * ev.lens_dirt),
        0.0, 1.0))


class SimSensor:
    """객체 수준 관측기. YOLO + 깊이 센서를 대체한다."""

    def __init__(self, camera: CameraModel, config: SensorConfig | None = None,
                 seed: int = 0):
        self.camera = camera
        self.cfg = config or SensorConfig()
        self.rng = np.random.default_rng(seed)
        self._next_false_id = 10_000

    # --- 내부 -------------------------------------------------------------

    def _box_of(self, obj: SimObject, p_cam: np.ndarray) -> tuple[float, float, float, float]:
        """3D AABB 8 꼭짓점을 투영해 2D 박스를 만든다."""
        e = np.asarray(obj.extent, float)
        corners = np.array([[sx, sy, sz] for sx in (-e[0], e[0])
                            for sy in (-e[1], e[1]) for sz in (-e[2], e[2])])
        pts = corners + p_cam
        pts = pts[pts[:, 2] > 1e-3]
        if len(pts) == 0:
            return 0.0, 0.0, 0.0, 0.0
        px = np.array([self.camera.project(p) for p in pts])
        lo, hi = px.min(axis=0), px.max(axis=0)
        return float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])

    @staticmethod
    def _overlap_ratio(inner: tuple, outer: tuple) -> float:
        """inner 박스가 outer 에 가려지는 면적 비율."""
        ax, ay, aw, ah = inner
        bx, by, bw, bh = outer
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1 or aw * ah <= 0.0:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / (aw * ah)

    # --- 관측 -------------------------------------------------------------

    def observe(self, world: SimWorld, T_world_cam: SE3, t: float,
                evidence: Evidence, frame_id: int,
                class_names: list[str] | None = None) -> tuple[DetectionSet, FrameTruth]:
        cfg = self.cfg
        sev = severity_of(evidence)
        T_cam_world = T_world_cam.inverse()

        truth = FrameTruth(stamp=t, T_world_cam=T_world_cam, evidence=evidence)

        # 1) 시야 안에 들어오는 객체를 깊이 순으로 모은다
        candidates = []
        for obj in world.present_at(t):
            pos = obj.position_at(t)
            truth.object_positions[obj.object_id] = pos
            truth.object_velocities[obj.object_id] = obj.velocity_at(t)

            p_cam = T_cam_world @ pos
            z = float(p_cam[2])
            if not (cfg.min_range < z < cfg.max_range):
                continue

            box = self._box_of(obj, p_cam)
            if box[2] < cfg.min_box_px or box[3] < cfg.min_box_px:
                continue
            cx, cy = box[0] + box[2] * 0.5, box[1] + box[3] * 0.5
            if not self.camera.in_image(np.array([cx, cy])):
                continue
            candidates.append((z, obj, box))

        candidates.sort(key=lambda c: c[0])      # 가까운 것부터

        # 2) 앞의 객체가 뒤를 가리는지 판정
        visible = []
        for i, (z, obj, box) in enumerate(candidates):
            occluded = 0.0
            for zj, _, boxj in candidates[:i]:   # 더 가까운 것들만
                occluded += self._overlap_ratio(box, boxj)
            if occluded >= cfg.occlusion_threshold:
                continue
            visible.append((z, obj, box, occluded))
            truth.visible_objects.add(obj.object_id)

        # 3) 검출 샘플링 - 조건이 나쁠수록 확률이 떨어진다
        items: list[Detection] = []
        for z, obj, box, occluded in visible:
            # 거리와 열화가 함께 작용한다. 안개 속 먼 물체가 먼저 사라진다.
            range_factor = np.exp(-cfg.detection_falloff * sev * z / cfg.max_range)
            p_det = cfg.base_detection_prob * range_factor * (1.0 - 0.5 * occluded)
            if self.rng.random() > p_det:
                continue

            box_sigma = cfg.base_box_noise_px * (1.0 + cfg.box_noise_growth * sev)
            noise = self.rng.normal(0.0, box_sigma, 4)
            noisy = (box[0] + noise[0], box[1] + noise[1],
                     max(cfg.min_box_px, box[2] + noise[2]),
                     max(cfg.min_box_px, box[3] + noise[3]))

            depth_sigma = cfg.depth_noise_coeff * z * z * (1.0 + cfg.depth_noise_growth * sev)
            measured_depth = z + float(self.rng.normal(0.0, depth_sigma))

            class_id = obj.class_id
            p_confuse = cfg.class_confusion_prob * (1.0 + cfg.confusion_growth * sev)
            if self.rng.random() < p_confuse:
                class_id = int(self.rng.integers(0, 80))

            name = obj.class_name
            if class_names and 0 <= class_id < len(class_names):
                name = class_names[class_id]

            conf = float(np.clip(self.rng.normal(0.85 - 0.35 * sev, 0.08), 0.05, 0.99))

            items.append(Detection(class_id, name, noisy, conf))
            truth.detection_to_object.append(obj.object_id)
            truth.detected_objects.add(obj.object_id)
            truth.detection_depth.append(measured_depth)
            truth.detection_depth_sigma.append(depth_sigma)

        # 4) 오검출 - 열화가 심할수록 늘어난다
        fa_rate = cfg.base_false_alarm_rate * (1.0 + cfg.false_alarm_growth * sev)
        for _ in range(int(self.rng.poisson(fa_rate))):
            w = float(self.rng.uniform(20, 90))
            h = float(self.rng.uniform(20, 90))
            x = float(self.rng.uniform(0, max(1.0, self.camera.width - w)))
            y = float(self.rng.uniform(0, max(1.0, self.camera.height - h)))
            cid = int(self.rng.integers(0, 80))
            name = class_names[cid] if class_names and cid < len(class_names) else str(cid)

            items.append(Detection(cid, name, (x, y, w, h),
                                   float(np.clip(self.rng.normal(0.4, 0.12), 0.05, 0.9))))
            truth.detection_to_object.append(-1)
            truth.detection_depth.append(float(self.rng.uniform(1.0, cfg.max_range)))
            truth.detection_depth_sigma.append(cfg.max_range)   # 사실상 무정보

        return DetectionSet(t, frame_id, items), truth
