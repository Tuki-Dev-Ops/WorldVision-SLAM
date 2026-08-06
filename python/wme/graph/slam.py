"""객체 수준 SLAM: 포즈 / 랜드마크 위치 / 랜드마크 치수를 함께 추정한다.

M3 는 포즈를 참값으로 고정한 조건에서 통과했다. 여기서는 그 가정을 없앤다 -
포즈가 추정 대상이 되면 랜드마크 오차와 포즈 오차가 결합되므로 훨씬 어렵다.
보정된 불확실성이 이 조건에서도 유지되는지가 M4 의 질문이다.

그래프 구성:
    포즈 사전분포 (앵커 1개)          <- 게이지 고정
    상대 포즈 팩터                     <- Tier 0 (ECDA) 대응
    객체 관측 팩터 (박스 + 깊이)       <- Tier 1 (TCG) 의 정보원

주의: 이 하네스에는 실제 ECDA 측광 항이 없다. 시뮬레이터가 픽셀을 렌더링하지
않기 때문이다. 상대 포즈 팩터는 그 자리에 들어갈 대리물이며, 정보행렬 규모를
환경에 따라 조절해 융합 거동만 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calib.noise import NoiseModel
from ..reference.geometry import SE3
from ..sim.scenarios import Sequence
from ..sim.sensor import severity_of
from ..sim.world import CameraModel
from .factors import (
    BetweenPoseFactor, Huber, ObjectObservationFactor, PosePriorFactor, diagonal, isotropic,
)
from .graph import FactorGraph, SolverOptions
from .variables import PoseVariable, PositiveVectorVariable, VectorVariable


@dataclass
class SlamConfig:
    odometry_sigma_trans: float = 0.02      # m, 프레임 간
    odometry_sigma_rot: float = 0.005       # rad, 프레임 간
    observation_stride: int = 3             # 객체 관측 서브샘플링
    min_observations: int = 6
    huber_delta: float = 4.0
    anchor_sigma: float = 1e-4
    max_iterations: int = 40
    seed: int = 0


@dataclass
class SlamResult:
    graph: FactorGraph
    pose_keys: list[str]
    landmark_keys: dict[int, str]
    truth_poses: list[SE3]
    truth_points: dict[int, np.ndarray]
    initial_poses: list[SE3]
    optimization: object = None
    covariances: dict[str, np.ndarray] = field(default_factory=dict)

    def pose_errors(self) -> np.ndarray:
        """앵커 정렬 상태이므로 별도 정렬 없이 직접 비교한다."""
        out = []
        for key, truth in zip(self.pose_keys, self.truth_poses):
            out.append(self.graph.value(key).pose.distance_to(truth)[0])
        return np.array(out)

    def initial_pose_errors(self) -> np.ndarray:
        return np.array([init.distance_to(truth)[0]
                         for init, truth in zip(self.initial_poses, self.truth_poses)])

    def landmark_nees_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        errors, covs = [], []
        for oid, key in sorted(self.landmark_keys.items()):
            if key not in self.covariances:
                continue
            est = self.graph.value(key).value
            errors.append(self.truth_points[oid] - est)
            covs.append(self.covariances[key])
        if not errors:
            raise ValueError("공분산이 있는 랜드마크가 없음")
        return np.array(errors), np.array(covs)


def _simulate_odometry(truth: list[SE3], cfg: SlamConfig
                       ) -> tuple[list[SE3], list[SE3]]:
    """드리프트가 누적되는 상대 측정과 그것을 적분한 초기값.

    실제 전단(front-end)이 내놓는 것과 같은 형태다. 그래프가 이 드리프트를
    실제로 줄이는지 보려면 초기값이 참값이면 안 된다.
    """
    rng = np.random.default_rng(cfg.seed + 991)
    measured, initial = [], [truth[0]]

    for i in range(len(truth) - 1):
        rel_true = truth[i].inverse() @ truth[i + 1]
        noise = np.concatenate([
            rng.normal(0.0, cfg.odometry_sigma_trans, 3),
            rng.normal(0.0, cfg.odometry_sigma_rot, 3),
        ])
        rel = rel_true @ SE3.exp(noise)
        measured.append(rel)
        initial.append(initial[-1] @ rel)      # 드리프트가 누적된다
    return measured, initial


def build_object_slam(seq: Sequence, noise: NoiseModel,
                      cfg: SlamConfig | None = None,
                      cam: CameraModel | None = None) -> SlamResult:
    cfg = cfg or SlamConfig()
    cam = cam or seq.camera

    truth_poses = list(seq.trajectory.poses)
    measured_rel, initial_poses = _simulate_odometry(truth_poses, cfg)

    g = FactorGraph()
    pose_keys = [f"p{i}" for i in range(len(truth_poses))]
    for key, init in zip(pose_keys, initial_poses):
        g.add_variable(key, PoseVariable(init))

    # 게이지 고정. 참값에 앵커해야 추정 좌표계와 참값 좌표계가 일치한다.
    g.add_factor(PosePriorFactor(pose_keys[0], truth_poses[0],
                                 isotropic(6, cfg.anchor_sigma)))

    # Tier 0 대응: 상대 포즈 제약
    odom_info = np.diag(np.concatenate([
        np.full(3, 1.0 / cfg.odometry_sigma_trans ** 2),
        np.full(3, 1.0 / cfg.odometry_sigma_rot ** 2),
    ]))
    kernel = Huber(cfg.huber_delta)
    for i, rel in enumerate(measured_rel):
        g.add_factor(BetweenPoseFactor(pose_keys[i], pose_keys[i + 1], rel,
                                       odom_info, kernel=kernel, tier="photometric"))

    # 객체 관측. 참 연관을 쓴다 - 연관 문제와 분리하기 위함이다.
    grouped: dict[int, list[tuple[int, np.ndarray, float, float]]] = {}
    truth_points: dict[int, np.ndarray] = {}

    for fi, (det, truth) in enumerate(zip(seq.detections, seq.truths)):
        if fi % cfg.observation_stride != 0:
            continue
        sev = severity_of(truth.evidence)
        for k, item in enumerate(det.items):
            oid = truth.detection_to_object[k]
            if oid < 0:
                continue
            obj = seq.world.by_id(oid)
            if obj is None or obj.is_agent:      # 움직이는 물체는 랜드마크가 아니다
                continue
            d = truth.detection_depth[k]
            if d <= 0.1:
                continue
            x, y, w, h = item.box
            grouped.setdefault(oid, []).append((fi, np.array([x, y, w, h, d]), sev, d))
            truth_points[oid] = truth.object_positions[oid]

    landmark_keys: dict[int, str] = {}
    for oid, obs in grouped.items():
        if len(obs) < cfg.min_observations:
            continue

        # 초기값: 중간 관측의 박스 중심 역투영 (편향이 있지만 초기값으로는 충분)
        fi, z, _, d0 = obs[len(obs) // 2]
        u, v = z[0] + z[2] * 0.5, z[1] + z[3] * 0.5
        p_cam = np.array([(u - cam.cx) * d0 / cam.fx, (v - cam.cy) * d0 / cam.fy, d0])

        lkey, ekey = f"l{oid}", f"e{oid}"
        g.add_variable(lkey, VectorVariable(initial_poses[fi] @ p_cam))
        g.add_variable(ekey, PositiveVectorVariable(np.array([
            max(0.05, 0.5 * z[2] * d0 / cam.fx),
            max(0.05, 0.5 * z[3] * d0 / cam.fy),
            0.25 * (z[2] * d0 / cam.fx + z[3] * d0 / cam.fy),
        ])))
        landmark_keys[oid] = lkey

        for fi, z, sev, depth in obs:
            s_px = noise.sigma_pixel(sev)
            s_d = noise.sigma_depth(max(depth, 0.1), sev)
            info = diagonal([s_px, s_px, s_px, s_px, s_d])
            g.add_factor(ObjectObservationFactor(
                pose_keys[fi], lkey, ekey, z, cam, info, kernel=kernel))

    return SlamResult(g, pose_keys, landmark_keys, truth_poses, truth_points,
                      initial_poses)


def solve_object_slam(seq: Sequence, noise: NoiseModel,
                      cfg: SlamConfig | None = None,
                      cam: CameraModel | None = None) -> SlamResult:
    cfg = cfg or SlamConfig()
    result = build_object_slam(seq, noise, cfg, cam)
    result.optimization = result.graph.optimize(
        SolverOptions(max_iterations=cfg.max_iterations))
    result.covariances = result.graph.covariance(list(result.landmark_keys.values()))
    return result
