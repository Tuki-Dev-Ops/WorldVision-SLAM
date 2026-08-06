"""세 tier 융합 - docs/02-correspondence-problem.md 5장의 실제 구현.

    Lambda_total = a0(E) Lambda_ECDA + a1(E) Lambda_TCG + a2(E) Lambda_SPA

세 tier 는 팩터 종류가 다르지 않다. 전부 BetweenPoseFactor 로 들어가고
정보행렬만 다르다. 환경 적응은 alpha_k(E) 스케일 하나로 끝나며 분기 코드는
어디에도 없다 - 그 주장이 실제로 성립하는지 재는 것이 이 모듈의 목적이다.

각 tier 가 서로 다른 실패 양상을 갖는다는 것이 융합의 근거다.
  ECDA  텍스처가 필요하다. 안개/저조도/블러에 무너진다.
  TCG   객체가 필요하다. 긴 기선을 건너뛴다. 객체가 없으면 침묵한다.
  SPA   평면이 필요하다. 외관과 무관하지만 랭크가 부족할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calib.noise import NoiseModel
from ..geometry.planes import PlaneConfig, extract_planes
from ..geometry.spa import SpaConfig, align as spa_align
from ..localization import EcdaConfig
from ..reference.constellation import Config as ConstellationConfig
from ..reference.constellation import ConstellationIndex, Node
from ..reference.environment import Evidence, derive_adaptation
from ..reference.geometry import SE3
from ..sim.render import RenderedFrame
from ..sim.world import CameraModel, SimWorld
from .factors import BetweenPoseFactor, Huber, PosePriorFactor, isotropic
from .graph import FactorGraph, SolverOptions
from .photometric_slam import PhotometricSlamConfig, run_ecda_odometry
from .variables import PoseVariable


@dataclass
class FusionConfig:
    ecda: EcdaConfig = field(default_factory=lambda: EcdaConfig(pyramid_levels=4))
    spa: SpaConfig = field(default_factory=SpaConfig)
    planes: PlaneConfig = field(default_factory=PlaneConfig)
    constellation: ConstellationConfig = field(default_factory=ConstellationConfig)
    photometric: PhotometricSlamConfig = field(default_factory=PhotometricSlamConfig)

    # ablation 스위치. 어느 tier 가 무엇을 기여하는지 재려면 끌 수 있어야 한다.
    use_photometric: bool = True
    use_constellation: bool = True
    use_structural: bool = True

    keyframe_interval: int = 6
    min_loop_gap: int = 12               # 이만큼 떨어진 키프레임하고만 루프를 닫는다

    # 각 tier 의 정보 스케일. M4b 에서 측광은 유효표본 보정이 필요했다.
    photometric_scale: float = 0.2
    constellation_sigma_t: float = 0.10  # m
    constellation_sigma_r: float = 0.05  # rad
    structural_scale: float = 1.0

    anchor_sigma: float = 1e-4
    huber_delta: float = 4.0
    max_iterations: int = 40


@dataclass
class TierStats:
    factors: int = 0
    information_trace: float = 0.0
    rejected: int = 0

    def __repr__(self) -> str:
        return f"{self.factors} factors (tr {self.information_trace:.3e}, {self.rejected} rejected)"


@dataclass
class FusionResult:
    graph: FactorGraph
    pose_keys: list[str]
    truth_poses: list[SE3]
    initial_poses: list[SE3]
    photometric: TierStats = field(default_factory=TierStats)
    constellation: TierStats = field(default_factory=TierStats)
    structural: TierStats = field(default_factory=TierStats)
    optimization: object = None

    def pose_errors(self) -> np.ndarray:
        return np.array([self.graph.value(k).pose.distance_to(t)[0]
                         for k, t in zip(self.pose_keys, self.truth_poses)])

    def initial_errors(self) -> np.ndarray:
        return np.array([a.distance_to(b)[0]
                         for a, b in zip(self.initial_poses, self.truth_poses)])

    def ate(self) -> float:
        return float(np.sqrt((self.pose_errors() ** 2).mean()))

    def ate_initial(self) -> float:
        return float(np.sqrt((self.initial_errors() ** 2).mean()))


def _constellation_nodes(frame: RenderedFrame, world: SimWorld,
                         cam: CameraModel) -> list[Node]:
    """검출에서 카메라 좌표계 성좌를 만든다. 정적 객체만 장소를 정의한다."""
    T_cam_world = frame.pose.inverse()
    nodes = []
    for det in frame.detections:
        obj = world.by_id(det.object_id)
        if obj is None or obj.is_agent or det.truncated:
            continue
        x, y, w, h = det.box
        u, v = x + w * 0.5, y + h * 0.5
        d = det.depth
        p_cam = np.array([(u - cam.cx) * d / cam.fx, (v - cam.cy) * d / cam.fy, d])
        nodes.append(Node(token_id=det.object_id, class_id=obj.class_id,
                          position=p_cam, sigma=0.06 + 0.01 * d))
    return nodes


def build(frames: list[RenderedFrame], world: SimWorld, cam: CameraModel,
          noise: NoiseModel, cfg: FusionConfig | None = None,
          ev: Evidence | None = None) -> FusionResult:
    cfg = cfg or FusionConfig()
    ev = ev or Evidence()
    adaptation = derive_adaptation(ev)

    truth = [f.pose for f in frames]
    n = len(frames)
    kernel = Huber(cfg.huber_delta)

    g = FactorGraph()
    pose_keys = [f"p{i}" for i in range(n)]

    # --- Tier 0: 측광 오도메트리 (초기값의 출처이기도 하다) ---
    odom = run_ecda_odometry(frames, cam, cfg.photometric, ev)
    initial = [truth[0]]
    last_good = SE3.identity()
    for T_rel, _info, _rmse, ok in odom:
        step = T_rel if ok else last_good
        if ok:
            last_good = T_rel
        initial.append(initial[-1] @ step.inverse())

    for key, init in zip(pose_keys, initial):
        g.add_variable(key, PoseVariable(init))
    g.add_factor(PosePriorFactor(pose_keys[0], truth[0], isotropic(6, cfg.anchor_sigma)))

    result = FusionResult(g, pose_keys, truth, initial)

    if cfg.use_photometric:
        for i, (T_rel, info, _rmse, ok) in enumerate(odom):
            if not ok:
                result.photometric.rejected += 1
                continue
            scaled = info * cfg.photometric_scale * adaptation.alpha_photometric
            g.add_factor(BetweenPoseFactor(pose_keys[i], pose_keys[i + 1],
                                           T_rel.inverse(), scaled,
                                           kernel=kernel, tier="photometric"))
            result.photometric.factors += 1
            result.photometric.information_trace += float(np.trace(scaled))

    # --- Tier 2: 평면 정합 ---
    planes: list[list] = []
    if cfg.use_structural:
        for f in frames:
            depth = np.where(np.isfinite(f.depth), f.depth, 0.0).astype(float)
            planes.append(extract_planes(depth, cam, cfg.planes))

        for i in range(n - 1):
            if not planes[i] or not planes[i + 1]:
                result.structural.rejected += 1
                continue
            r = spa_align(planes[i], planes[i + 1], config=cfg.spa,
                          alpha_structural=adaptation.alpha_structural
                          * cfg.structural_scale)
            if not r.converged:
                result.structural.rejected += 1
                continue
            # 랭크 부족은 결함이 아니다. 정보행렬이 그 축을 구속하지 않을 뿐이며
            # 팩터그래프가 나머지 tier 로 채운다.
            g.add_factor(BetweenPoseFactor(pose_keys[i], pose_keys[i + 1],
                                           r.T_cur_ref.inverse(), r.information,
                                           kernel=kernel, tier="structural"))
            result.structural.factors += 1
            result.structural.information_trace += float(np.trace(r.information))

    # --- Tier 1: 성좌 루프 클로저 ---
    if cfg.use_constellation:
        index = ConstellationIndex(cfg.constellation)
        keyframes: dict[int, int] = {}          # place_id -> frame index

        for i in range(n):
            nodes = _constellation_nodes(frames[i], world, cam)
            if len(nodes) < cfg.constellation.min_nodes:
                continue

            # 충분히 떨어진 키프레임과만 루프를 닫는다. 이웃 프레임끼리
            # 닫으면 오도메트리와 같은 정보를 두 번 세는 꼴이 된다.
            match = index.query(nodes)
            if match is not None:
                j = keyframes.get(match.place_id)
                if j is not None and abs(i - j) >= cfg.min_loop_gap:
                    info = np.diag(np.concatenate([
                        np.full(3, 1.0 / cfg.constellation_sigma_t ** 2),
                        np.full(3, 1.0 / cfg.constellation_sigma_r ** 2),
                    ])) * match.score * adaptation.alpha_constellation
                    # match.transform 은 query -> place, 즉 T_cam_j <- cam_i 다.
                    g.add_factor(BetweenPoseFactor(pose_keys[j], pose_keys[i],
                                                   match.transform.inverse(), info,
                                                   kernel=kernel, tier="constellation"))
                    result.constellation.factors += 1
                    result.constellation.information_trace += float(np.trace(info))
                else:
                    result.constellation.rejected += 1

            if i % cfg.keyframe_interval == 0:
                pid = index.insert(i, float(i), SE3.identity(), nodes)
                if pid:
                    keyframes[pid] = i

    return result


def solve(frames: list[RenderedFrame], world: SimWorld, cam: CameraModel,
          noise: NoiseModel, cfg: FusionConfig | None = None,
          ev: Evidence | None = None) -> FusionResult:
    cfg = cfg or FusionConfig()
    result = build(frames, world, cam, noise, cfg, ev)
    result.optimization = result.graph.optimize(
        SolverOptions(max_iterations=cfg.max_iterations))
    return result
