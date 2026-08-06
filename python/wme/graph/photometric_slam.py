"""M4 완결판: 실제 ECDA 측광 항으로 구성한 객체 수준 SLAM.

이전 M4 는 상대 포즈 팩터를 대리물로 썼다. 시뮬레이터가 픽셀을 만들지 않아
Tier 0 를 실제로 돌릴 수 없었기 때문이다. 이제 렌더러가 있으므로 대리물을
치우고 진짜 측광 오도메트리를 넣는다.

그래프 구성
    포즈 사전분포 (앵커 1개)                       <- 게이지 고정
    ECDA 상대 포즈 팩터                            <- Tier 0, 실제 측광 정렬
    객체 관측 팩터 (렌더링된 마스크에서 추출)       <- Tier 1 의 정보원

두 가지를 M3/ECDA 실험에서 배운 대로 적용한다.
  - 측광 분산을 잔차에서 자기추정한다. 잡음모델은 그것이 쓰일 관측 모델과
    함께 보정되어야 한다 (M3).
  - 잔차 게이트로 실패한 정렬을 버린다. 랭크도 조건수도 그 실패를 못 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..calib.noise import NoiseModel
from ..localization import EcdaConfig, InformationModel, align
from ..reference.environment import Evidence
from ..reference.geometry import SE3
from ..sim.render import RenderScene, RenderedFrame, render_frame
from ..sim.sensor import severity_of
from ..sim.world import CameraModel, CameraTrajectory, SimWorld
from .factors import (
    BetweenPoseFactor, Huber, ObjectObservationFactor, PosePriorFactor, diagonal, isotropic,
)
from .graph import FactorGraph, SolverOptions
from .variables import PoseVariable, PositiveVectorVariable, VectorVariable


@dataclass
class PhotometricSlamConfig:
    # RESIDUAL_VARIANCE 를 명시한다. 기본값(EFFECTIVE_SAMPLE, nu=1)은 실데이터의
    # 측광 잔차가 프레임 단위로 상관되어 있다는 TUM 실측에서 나온 값이다
    # (docs/06-results.md 15.1). 이 모듈이 쓰는 렌더러는 그 전제가 성립하지
    # 않는 데이터를 만든다 - 잡음이 픽셀별 독립 화이트 노이즈다. 15.3 의 합성
    # 대조군이 바로 그 경우이고, 거기서 nu=1 은 점 개수만큼 *과소* 신뢰가 된다
    # (실측 ANEES 1.3e-4). 잔차 분산 모델이 이 데이터의 올바른 모드다.
    ecda: EcdaConfig = field(default_factory=lambda: EcdaConfig(
        pyramid_levels=4, information_model=InformationModel.RESIDUAL_VARIANCE))

    # 잔차 게이트. 랭크도 조건수도 정렬 실패를 못 잡으므로 잔차가 유일한
    # 방어선이다 (docs/03-roadmap.md Phase 2g).
    #
    # 다만 절대 임계는 쓸 수 없다. 정상 정렬의 rmse 는 장면에 따라 크게 다르다 -
    # 빈 방에서는 0.2~3, 객체가 있으면 실루엣 경계 때문에 6.6~9.7 이었다.
    # 절대값 8 로 자르면 정상 정렬의 절반을 버린다. 시퀀스 내 중앙값 대비
    # 상대 기준으로 이상치만 걸러낸다.
    rmse_outlier_factor: float = 2.0
    rmse_absolute_ceiling: float = 40.0    # 전부 망가진 경우의 최후 방어선
    min_inlier_ratio: float = 0.5
    # 측광 정보의 유효표본 보정.
    #
    # J^T W J / sigma^2 는 픽셀 잔차가 서로 독립일 때만 옳다. 실제로 이웃
    # 픽셀은 같은 표면·같은 텍스처를 보므로 강하게 상관되어 있고, 700 개를
    # 독립 관측으로 세면 정보량이 그만큼 과대평가된다. M3 에서 객체 융합이
    # 1/N 로 과신하던 것과 같은 오류가 측광 층에서 재현된다.
    #
    # 1.0 은 보정 전 값이다. calibrate_information_scale() 로 상대포즈 NEES 가
    # 1 이 되도록 맞춰 쓴다.
    photometric_information_scale: float = 1.0

    observation_stride: int = 2
    min_observations: int = 5
    min_detection_pixels: int = 120
    reject_truncated: bool = True

    # 렌더 마스크의 깊이 중앙값은 *보이는 표면* 이라 무게중심보다 가깝다.
    # 실측: 편향 = 0.778 * e_z (잔차 표준편차 0.021 m). 이 항을 빼면
    # 랜드마크가 그만큼 통째로 밀린다 - M3 의 박스중심 편향과 같은 부류다.
    #
    # 주의: 이 장면은 모든 객체의 e_z 가 같아서 "0.778 * e_z" 와
    # "상수 0.233 m" 를 구분할 수 없다. 기하학적 근거(큰 상자일수록 근접면이
    # 더 앞에 있다)로 비례 형태를 택했고, 크기가 다양한 장면에서 재확인해야 한다.
    depth_offset_coeff: float = 0.778

    huber_delta: float = 4.0
    anchor_sigma: float = 1e-4
    max_iterations: int = 40
    seed: int = 0


@dataclass
class PhotometricSlamResult:
    graph: FactorGraph
    pose_keys: list[str]
    landmark_keys: dict[int, str]
    truth_poses: list[SE3]
    truth_points: dict[int, np.ndarray]
    initial_poses: list[SE3]
    optimization: object = None
    covariances: dict[str, np.ndarray] = field(default_factory=dict)
    ecda_accepted: int = 0
    ecda_rejected: int = 0
    ecda_rmse: list[float] = field(default_factory=list)

    def pose_errors(self) -> np.ndarray:
        return np.array([self.graph.value(k).pose.distance_to(t)[0]
                         for k, t in zip(self.pose_keys, self.truth_poses)])

    def initial_pose_errors(self) -> np.ndarray:
        return np.array([i.distance_to(t)[0]
                         for i, t in zip(self.initial_poses, self.truth_poses)])

    def landmark_nees_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        errors, covs = [], []
        for oid, key in sorted(self.landmark_keys.items()):
            if key not in self.covariances or oid not in self.truth_points:
                continue
            errors.append(self.truth_points[oid] - self.graph.value(key).value)
            covs.append(self.covariances[key])
        if not errors:
            raise ValueError("공분산이 있는 랜드마크가 없음")
        return np.array(errors), np.array(covs)


def render_sequence(scene: RenderScene, world: SimWorld, traj: CameraTrajectory,
                    cam: CameraModel, ev: Evidence | None = None,
                    seed: int = 0) -> list[RenderedFrame]:
    """궤적을 따라 프레임을 렌더링한다. 객체는 매 시각의 위치로 배치한다."""
    frames = []
    for i, (t, pose) in enumerate(zip(traj.stamps, traj.poses)):
        sc = scene.with_objects(world, float(t))
        frames.append(render_frame(sc, pose, cam, ev, seed=seed + i))
    return frames


def run_ecda_odometry(frames: list[RenderedFrame], cam: CameraModel,
                      cfg: PhotometricSlamConfig, ev: Evidence | None = None):
    """연속 프레임 사이 측광 정렬. (상대포즈, 정보행렬, rmse, 채택여부) 목록."""
    ev = ev or Evidence()
    sev = severity_of(ev)
    # 환경 가중은 항을 더하지 않고 정보량만 조절한다 (docs/04 5.2)
    from ..reference.environment import derive_adaptation
    alpha = derive_adaptation(ev).alpha_photometric

    # 1단계: 전부 정렬한다. 게이트는 시퀀스 통계를 봐야 하므로 나중에 건다.
    raw = []
    prev_rel = SE3.identity()
    for a, b in zip(frames, frames[1:]):
        r = align(a.gray, a.depth, b.gray, cam, init=prev_rel, cfg=cfg.ecda,
                  alpha_photometric=alpha)
        raw.append(r)
        if r.converged and r.inlier_ratio >= cfg.min_inlier_ratio:
            prev_rel = r.T_cur_ref          # 등속 가정 초기값

    # 2단계: 시퀀스 내 중앙값 대비 이상치만 거른다
    rmses = np.array([r.photometric_rmse for r in raw if r.converged])
    median = float(np.median(rmses)) if rmses.size else np.inf
    limit = min(cfg.rmse_absolute_ceiling, median * cfg.rmse_outlier_factor)

    out = []
    for r in raw:
        # 측광 분산은 이제 align() 안에서 잔차로부터 자기추정된다
        # (InformationModel). 여기서 또 나누면 같은 정규화를 두 번 하게 된다.
        info = r.information * cfg.photometric_information_scale

        accepted = (r.converged
                    and r.photometric_rmse <= limit
                    and r.inlier_ratio >= cfg.min_inlier_ratio
                    and r.observable_dof == 6
                    and np.all(np.isfinite(info)))
        out.append((r.T_cur_ref, info, r.photometric_rmse, accepted))
    return out


def calibrate_information_scale(frames: list[RenderedFrame], cam: CameraModel,
                                cfg: PhotometricSlamConfig | None = None,
                                ev: Evidence | None = None) -> tuple[float, float]:
    """상대포즈 NEES 가 1 이 되도록 측광 정보 스케일을 적합한다.

    ECDA 가 보고하는 공분산(Lambda^-1)이 실제 상대포즈 오차 분포와 맞는지
    직접 재고, 어긋난 배수를 그대로 스케일로 삼는다. M3 와 같은 방법론이다.

    시뮬레이션에서만 가능한 보정이라는 점이 중요하다 - 참 상대포즈가 필요하다.
    실기에서는 이 스케일을 시뮬레이션에서 적합해 전이시키고, 그 전이 가능성
    자체가 검증 대상이 된다 (docs/05-research-program.md).

    Returns:
        (스케일, 보정 전 ANEES/dof)
    """
    cfg = cfg or PhotometricSlamConfig()
    odom = run_ecda_odometry(frames, cam, cfg, ev)

    chi2 = []
    for i, (T_rel, info, _rmse, ok) in enumerate(odom):
        if not ok:
            continue
        truth = frames[i + 1].pose.inverse() @ frames[i].pose   # ref -> cur
        err = (T_rel.inverse() @ truth).log()
        try:
            chi2.append(float(err @ info @ err))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(chi2) < 5:
        return 1.0, float("nan")

    anees = float(np.mean(chi2)) / 6.0
    # NEES 가 k 배 크면 정보가 k 배 과대평가된 것이다
    return float(1.0 / max(anees, 1e-9)), anees


def build(frames: list[RenderedFrame], world: SimWorld, cam: CameraModel,
          noise: NoiseModel, cfg: PhotometricSlamConfig | None = None,
          ev: Evidence | None = None) -> PhotometricSlamResult:
    cfg = cfg or PhotometricSlamConfig()
    ev = ev or Evidence()
    sev = severity_of(ev)

    truth_poses = [f.pose for f in frames]
    odom = run_ecda_odometry(frames, cam, cfg, ev)

    # 초기값: 채택된 ECDA 결과를 적분한다. 거부된 구간은 직전 상대운동을 재사용.
    initial = [truth_poses[0]]
    last_good = SE3.identity()
    for T_rel, _info, _rmse, ok in odom:
        step = T_rel if ok else last_good
        if ok:
            last_good = T_rel
        # T_cur_ref 는 ref -> cur 이므로 월드 포즈는 그 역으로 누적된다
        initial.append(initial[-1] @ step.inverse())

    g = FactorGraph()
    pose_keys = [f"p{i}" for i in range(len(frames))]
    for key, init in zip(pose_keys, initial):
        g.add_variable(key, PoseVariable(init))

    g.add_factor(PosePriorFactor(pose_keys[0], truth_poses[0],
                                 isotropic(6, cfg.anchor_sigma)))

    kernel = Huber(cfg.huber_delta)
    accepted = rejected = 0
    rmses = []
    for i, (T_rel, info, rmse, ok) in enumerate(odom):
        rmses.append(rmse)
        if not ok:
            rejected += 1
            continue
        accepted += 1
        # BetweenPoseFactor 는 T_a^-1 T_b (a -> b) 를 기대한다.
        # ECDA 의 T_cur_ref 는 ref -> cur 이므로 그 역이 필요하다.
        g.add_factor(BetweenPoseFactor(pose_keys[i], pose_keys[i + 1],
                                       T_rel.inverse(), info,
                                       kernel=kernel, tier="photometric"))

    # --- 객체 관측 ---------------------------------------------------------
    grouped: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    truth_points: dict[int, np.ndarray] = {}

    for fi, frame in enumerate(frames):
        if fi % cfg.observation_stride != 0:
            continue
        for det in frame.detections:
            obj = world.by_id(det.object_id)
            if obj is None or obj.is_agent:
                continue
            if det.pixels < cfg.min_detection_pixels:
                continue
            if cfg.reject_truncated and det.truncated:
                continue
            x, y, w, h = det.box
            grouped.setdefault(det.object_id, []).append(
                (fi, np.array([x, y, w, h, det.depth]), det.depth))
            truth_points[det.object_id] = np.asarray(obj.position, float)

    landmark_keys: dict[int, str] = {}
    for oid, obs in grouped.items():
        if len(obs) < cfg.min_observations:
            continue
        fi, z, d0 = obs[len(obs) // 2]
        u, v = z[0] + z[2] * 0.5, z[1] + z[3] * 0.5
        p_cam = np.array([(u - cam.cx) * d0 / cam.fx, (v - cam.cy) * d0 / cam.fy, d0])

        lkey, ekey = f"l{oid}", f"e{oid}"
        g.add_variable(lkey, VectorVariable(initial[fi] @ p_cam))
        g.add_variable(ekey, PositiveVectorVariable(np.array([
            max(0.05, 0.5 * z[2] * d0 / cam.fx),
            max(0.05, 0.5 * z[3] * d0 / cam.fy),
            0.25 * (z[2] * d0 / cam.fx + z[3] * d0 / cam.fy),
        ])))
        landmark_keys[oid] = lkey

        for fi, z, depth in obs:
            s_px = noise.sigma_pixel(sev)
            s_d = noise.sigma_depth(max(depth, 0.1), sev)
            g.add_factor(ObjectObservationFactor(
                pose_keys[fi], lkey, ekey, z, cam,
                diagonal([s_px, s_px, s_px, s_px, s_d]), kernel=kernel,
                depth_offset_coeff=cfg.depth_offset_coeff))

    return PhotometricSlamResult(
        g, pose_keys, landmark_keys, truth_poses, truth_points, initial,
        ecda_accepted=accepted, ecda_rejected=rejected, ecda_rmse=rmses)


def solve(frames: list[RenderedFrame], world: SimWorld, cam: CameraModel,
          noise: NoiseModel, cfg: PhotometricSlamConfig | None = None,
          ev: Evidence | None = None) -> PhotometricSlamResult:
    cfg = cfg or PhotometricSlamConfig()
    result = build(frames, world, cam, noise, cfg, ev)
    result.optimization = result.graph.optimize(
        SolverOptions(max_iterations=cfg.max_iterations))
    result.covariances = result.graph.covariance(list(result.landmark_keys.values()))
    return result
