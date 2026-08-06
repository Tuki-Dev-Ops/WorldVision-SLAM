"""최소 객체 위치 추정기.

의도적으로 SLAM 이 아니다. 카메라 포즈를 참값으로 고정해 **관측 모델 보정**
문제만 분리한다. 포즈 오차가 섞이면 NEES 가 나빠도 원인이 잡음모델인지
포즈인지 알 수 없다. M3 게이트는 잡음모델 하나만 묻는다.

융합은 정보필터다: Lambda = sum Sigma_i^-1, xi = sum Sigma_i^-1 z_i.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.scenarios import Sequence
from ..sim.sensor import severity_of
from ..sim.world import CameraModel
from .noise import NoiseModel, NoiseSample, ObjectResidual


@dataclass
class ObjectEstimate:
    object_id: int
    position: np.ndarray
    covariance: np.ndarray
    observations: int
    mean_box_px: float = 0.0
    mean_depth: float = 0.0

    def error_against(self, truth: np.ndarray) -> np.ndarray:
        return np.asarray(truth, float) - self.position


@dataclass
class EstimationResult:
    estimates: dict[int, ObjectEstimate] = field(default_factory=dict)
    truth: dict[int, np.ndarray] = field(default_factory=dict)

    def nees_inputs(self, min_observations: int = 5
                    ) -> tuple[np.ndarray, np.ndarray]:
        """NEES 계산용 (오차, 공분산) 배열."""
        errors, covs = [], []
        for oid, est in sorted(self.estimates.items()):
            if est.observations < min_observations or oid not in self.truth:
                continue
            errors.append(est.error_against(self.truth[oid]))
            covs.append(est.covariance)
        if not errors:
            raise ValueError("관측이 충분한 객체가 없음")
        return np.array(errors), np.array(covs)

    def bias(self, min_observations: int = 5) -> np.ndarray:
        """평균 잔차. 0 이 아니면 계통 오차가 있다는 뜻이고,
        그건 어떤 잡음모델로도 흡수할 수 없다."""
        e, _ = self.nees_inputs(min_observations)
        return e.mean(axis=0)


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w * 0.5, y + h * 0.5


def collect_samples(seq: Sequence, cam: CameraModel | None = None,
                    max_per_frame: int | None = None) -> list[NoiseSample]:
    """참값 연관을 이용해 잡음모델 적합용 표본을 모은다.

    참 연관을 쓰는 것은 의도적이다. 연관 오류가 섞이면 이상치가 되어
    잡음모델 적합이 오염되고, 그러면 이 실험은 두 문제를 동시에 묻게 된다.
    """
    cam = cam or seq.camera
    out: list[NoiseSample] = []

    for det, truth in zip(seq.detections, seq.truths):
        sev = severity_of(truth.evidence)
        T_cam_world = truth.T_world_cam.inverse()
        taken = 0

        for i, item in enumerate(det.items):
            oid = truth.detection_to_object[i]
            if oid < 0:                       # 오검출 제외
                continue
            if max_per_frame is not None and taken >= max_per_frame:
                break
            u, v = box_center(item.box)
            out.append(NoiseSample(
                u=u, v=v,
                depth=truth.detection_depth[i],
                severity=sev,
                true_point_cam=T_cam_world @ truth.object_positions[oid],
            ))
            taken += 1
    return out


def estimate_objects(seq: Sequence, model: NoiseModel,
                     cam: CameraModel | None = None,
                     use_true_association: bool = True) -> EstimationResult:
    """정적 객체들의 월드 좌표 위치를 정보필터로 융합한다.

    카메라 포즈는 참값을 쓴다 (위 모듈 설명 참조).
    """
    cam = cam or seq.camera
    result = EstimationResult()

    info: dict[int, np.ndarray] = {}          # Lambda
    vec: dict[int, np.ndarray] = {}           # xi
    count: dict[int, int] = {}
    box_sum: dict[int, float] = {}            # 계통 성분이 객체 크기에 의존한다
    depth_sum: dict[int, float] = {}

    for det, truth in zip(seq.detections, seq.truths):
        sev = severity_of(truth.evidence)
        R = truth.T_world_cam.R

        for i, item in enumerate(det.items):
            oid = truth.detection_to_object[i] if use_true_association else -1
            if oid < 0:
                continue
            obj = seq.world.by_id(oid)
            if obj is None or obj.is_agent:   # 움직이는 물체는 정적 융합 대상이 아니다
                continue

            u, v = box_center(item.box)
            d = truth.detection_depth[i]
            if d <= 0.1:
                continue

            p_cam = np.array([(u - cam.cx) * d / cam.fx,
                              (v - cam.cy) * d / cam.fy, d])
            z_world = truth.T_world_cam @ p_cam

            S_cam = model.covariance(u, v, d, sev, cam)
            S_world = R @ S_cam @ R.T + np.eye(3) * 1e-9
            S_inv = np.linalg.inv(S_world)

            info[oid] = info.get(oid, np.zeros((3, 3))) + S_inv
            vec[oid] = vec.get(oid, np.zeros(3)) + S_inv @ z_world
            count[oid] = count.get(oid, 0) + 1
            box_sum[oid] = box_sum.get(oid, 0.0) + max(item.box[2], item.box[3])
            depth_sum[oid] = depth_sum.get(oid, 0.0) + d

            # 참 위치는 마지막 관측 시점 기준 (정적이므로 시간 무관)
            result.truth[oid] = truth.object_positions[oid]

    # 시퀀스 평균 조건. 계통 성분이 조건에 의존할 수 있어 필요하다.
    mean_sev = float(np.mean([severity_of(t.evidence) for t in seq.truths]))

    for oid, Lam in info.items():
        P = np.linalg.inv(Lam)
        n = count[oid]
        Sigma_sys = model.systematic_covariance(
            mean_sev, box_sum[oid] / n, depth_sum[oid] / n, cam)

        # 계통 성분은 융합으로 줄지 않으므로 더한다.
        # 이것을 빼먹으면 관측 수에 비례해 과신하게 된다.
        result.estimates[oid] = ObjectEstimate(
            oid, P @ vec[oid], P + Sigma_sys, n,
            mean_box_px=box_sum[oid] / n, mean_depth=depth_sum[oid] / n)

    return result


def object_level_residuals(seq: Sequence, model: NoiseModel,
                           min_observations: int = 8) -> list[ObjectResidual]:
    """계통 성분 적합용 객체 단위 잔차.

    model 은 계통 성분이 없는 1단계 모델이어야 한다 - 여기서 추정하려는 것이
    바로 그 성분이기 때문이다.
    """
    result = estimate_objects(seq, model)
    sev = seq.severity
    out = []
    for oid, est in sorted(result.estimates.items()):
        if est.observations < min_observations or oid not in result.truth:
            continue
        out.append(ObjectResidual(
            error=est.error_against(result.truth[oid]),
            covariance=est.covariance,
            severity=sev,
            mean_box_px=est.mean_box_px,
            mean_depth=est.mean_depth,
        ))
    return out
