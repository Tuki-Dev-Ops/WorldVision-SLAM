"""World Token Engine 참조 구현 (C++ src/token/TokenStore.cpp).

이 파일이 존재하는 이유는 하나다. TokenStore 는 06-results.md 16 장에서만
결함 다섯 개가 나온 파일인데 차등 테스트가 하나도 없었다. 연관 게이트,
기동 여유항, 생애주기 전이, 정적 마스크의 출처 회계 - 전부 C++ 단위 테스트
자기 자신하고만 비교되고 있었다.

규약 주의:
  - C++ 의 cv::Rect2f 는 float 다. 박스 좌표와 IoU 는 단정밀도로 계산된다.
    double 로 계산하면 비용행렬이 1e-8 수준에서 갈리고, 동점 근처에서
    할당이 통째로 뒤집힌다. 그래서 여기서도 float32 로 자른다.
  - Observation::detection_conf 도 float 다 (confidence.py 참조).
  - 토큰 순회는 항상 ID 오름차순. 해시 순회 순서가 결과로 새면 안 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .assignment import INFEASIBLE, solve_assignment
from .confidence import Beliefs, ConfidenceConfig, ConfidenceEngine
from .geometry import SE3


class Lifecycle(str, Enum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    OCCLUDED = "occluded"
    DORMANT = "dormant"
    DISPLACED = "displaced"
    RETIRED = "retired"


# C++ affordanceFor 와 같은 표. 학습된 분류기를 붙이는 것은 제약 위반이므로
# COCO 어휘에 대한 명시적 표를 쓴다. 여기서 필요한 것은 Agent 여부뿐이다.
_AGENT_CLASSES = frozenset({
    "person", "car", "bus", "truck", "bicycle", "motorcycle", "dog", "cat",
})


def is_agent_class(name: str) -> bool:
    return name in _AGENT_CLASSES


@dataclass
class TokenStoreConfig:
    max_association_distance: float = 2.0
    max_association_mahalanobis: float = 9.0
    # 게이트 전용 기동 여유 (m/s). 융합 공분산만으로 게이트를 세우면 정보필터가
    # 상관된 관측을 독립으로 세어 sigma 8 mm 를 주장하고, 실제 운동이 전부
    # 이상치가 된다 (실측 마할라노비스 474 / 535).
    assoc_maneuver_speed: float = 2.0
    iou_weight: float = 0.8
    distance_weight: float = 1.0
    allow_cross_class: bool = False

    depth_noise_coeff: float = 0.006
    bearing_noise_px: float = 2.0
    no_depth_sigma: float = 5.0
    depth_sample_shrink: float = 0.5
    # 관측 깊이(보이는 표면) -> 무게중심. extent 가 *반치수* 이므로 1.0 이
    # 볼록 물체의 기하학적 기본값이다.
    depth_centroid_offset: float = 1.0

    observations_to_activate: int = 3
    occluded_timeout_s: float = 1.5
    dormant_timeout_s: float = 45.0
    existence_retire_threshold: float = 0.12
    existence_displace_threshold: float = 0.25

    history_capacity: int = 64
    trajectory_capacity: int = 256
    dynamic_mask_dilate: float = 1.15


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def project(self, p):
        inv_z = 1.0 / p[2]
        return np.array([self.fx * p[0] * inv_z + self.cx,
                         self.fy * p[1] * inv_z + self.cy])

    def backproject(self, u, v, depth):
        return np.array([(u - self.cx) * depth / self.fx,
                         (v - self.cy) * depth / self.fy,
                         depth])


@dataclass
class Detection:
    class_id: int
    class_name: str
    box: tuple                      # (x, y, w, h) - float32 로 잘려 보관된다
    confidence: float = 0.9

    def __post_init__(self):
        self.box = tuple(float(np.float32(v)) for v in self.box)


@dataclass
class Token:
    token_id: int
    class_id: int = -1
    class_name: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    extent: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    visible_ratio: float = 1.0
    occlusion: float = 0.0
    lifecycle: Lifecycle = Lifecycle.PROVISIONAL
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 0
    beliefs: Beliefs = None          # type: ignore[assignment]
    static_ref_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    static_ref_meas: np.ndarray = field(default_factory=lambda: np.zeros(3))
    static_ref_stamp: float = 0.0    # 0 = 무효 (C++ Timestamp::valid() 는 ns > 0)
    meas_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_obs_cam_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    has_obs_cam_rot: bool = False

    def __post_init__(self):
        if self.beliefs is None:
            self.beliefs = Beliefs()

    # --- WorldToken 의 판정 함수들 (헤더에 인라인으로 있는 것과 같아야 한다) ---
    def position_sigma(self) -> float:
        return math.sqrt(max(0.0, float(np.trace(self.position_cov)) / 3.0))

    def is_dynamic(self) -> bool:
        return self.beliefs.static < 0.4

    def is_alive(self) -> bool:
        return self.lifecycle != Lifecycle.RETIRED

    def is_stable_landmark(self) -> bool:
        return (self.beliefs.static > 0.7 and self.beliefs.existence > 0.6
                and self.observation_count >= 3 and self.position_sigma() < 0.5)

    def silence_seconds(self, now: float) -> float:
        return (now - self.last_seen) if self.last_seen > 0.0 else 0.0


def _clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def clamp_rect(box, w: int, h: int, shrink: float):
    """C++ clampRect. float 로 중심/반치수를 잡고 floor/ceil 로 정수화한다."""
    bx, by, bw, bh = (np.float32(v) for v in box)
    cx = bx + bw * np.float32(0.5)
    cy = by + bh * np.float32(0.5)
    hw = bw * np.float32(0.5) * np.float32(shrink)
    hh = bh * np.float32(0.5) * np.float32(shrink)

    x0 = max(0, int(math.floor(float(cx - hw))))
    y0 = max(0, int(math.floor(float(cy - hh))))
    x1 = min(w, int(math.ceil(float(cx + hw))))
    y1 = min(h, int(math.ceil(float(cy + hh))))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def box_iou(a, b) -> float:
    """C++ boxIoU 와 같이 float 로 계산한다."""
    ax, ay, aw, ah = (np.float32(v) for v in a)
    bx, by, bw, bh = (np.float32(v) for v in b)
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    w, h = np.float32(x2 - x1), np.float32(y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return 0.0
    inter = np.float32(w * h)
    uni = np.float32(np.float32(aw * ah) + np.float32(bw * bh) - inter)
    return float(np.float32(inter / uni)) if uni > 0.0 else 0.0


def _median_depth_in_box(depth, roi, min_d, max_d):
    """박스 중앙부 깊이 중앙값. 8 표본 미만이면 실패."""
    x, y, w, h = roi
    if depth is None or w <= 0 or h <= 0:
        return None, 0.0
    patch = depth[y:y + h, x:x + w]
    vals = patch[np.isfinite(patch) & (patch > min_d) & (patch < max_d)]
    valid_ratio = float(len(vals)) / max(1.0, float(w * h))
    if len(vals) < 8:
        return None, valid_ratio
    # C++ 은 nth_element(mid) 로 mid 번째 작은 값을 고른다 (짝수여도 평균 아님)
    mid = len(vals) // 2
    return float(np.partition(vals, mid)[mid]), valid_ratio


@dataclass
class Measurement:
    position_cam: np.ndarray
    cov_cam: np.ndarray
    extent: np.ndarray
    has_depth: bool
    visible_ratio: float


@dataclass
class IntegrationReport:
    matched: int = 0
    created: int = 0
    missed: int = 0
    retired: int = 0
    dynamic_area_ratio: float = 0.0
    det_no_candidate: int = 0
    det_gated_out: int = 0
    det_unassigned: int = 0
    gated_maha_sum: float = 0.0
    gated_dist_sum: float = 0.0


@dataclass
class MaskReport:
    masked_ratio: float = 0.0
    from_observed: float = 0.0
    from_stale: float = 0.0
    withheld_unjudged: float = 0.0
    n_masking: int = 0
    n_withheld: int = 0


class TokenStore:
    def __init__(self, config: TokenStoreConfig | None = None,
                 confidence: ConfidenceConfig | None = None):
        self.cfg = config or TokenStoreConfig()
        self.confidence = ConfidenceEngine(confidence or ConfidenceConfig())
        self.tokens: dict[int, Token] = {}
        self._next_id = 1
        self._last_stamp = 0.0

    # --- 조회 -------------------------------------------------------------
    def all_tokens(self) -> list[Token]:
        """항상 ID 오름차순. 해시 순회 순서가 결과로 새지 않게 하는 통로다."""
        return [self.tokens[k] for k in sorted(self.tokens)]

    def active_tokens(self) -> list[Token]:
        return [t for t in self.all_tokens()
                if t.lifecycle in (Lifecycle.ACTIVE, Lifecycle.OCCLUDED)]

    def stable_landmarks(self) -> list[Token]:
        return [t for t in self.all_tokens() if t.is_stable_landmark()]

    def clear(self) -> None:
        self.tokens.clear()
        self._next_id = 1
        self._last_stamp = 0.0

    # --- 3D 관측 ----------------------------------------------------------
    def measure(self, det: Detection, K: Intrinsics, depth) -> Measurement:
        cfg = self.cfg
        cx_px = det.box[0] + det.box[2] * 0.5
        cy_px = det.box[1] + det.box[3] * 0.5

        roi = clamp_rect(det.box, K.width, K.height, cfg.depth_sample_shrink)
        d, valid_ratio = (None, 0.0)
        if depth is not None and roi[2] * roi[3] > 0:
            d, valid_ratio = _median_depth_in_box(depth, roi, 0.1, 60.0)

        has_depth = d is not None
        depth_v = d if has_depth else 3.0

        if has_depth:
            sz = cfg.depth_noise_coeff * depth_v * depth_v
            sx = cfg.bearing_noise_px * depth_v / K.fx
            sy = cfg.bearing_noise_px * depth_v / K.fy
            cov = np.diag([sx * sx, sy * sy, sz * sz])
            extent = np.array([
                0.5 * det.box[2] * depth_v / K.fx,
                0.5 * det.box[3] * depth_v / K.fy,
                0.25 * (det.box[2] * depth_v / K.fx + det.box[3] * depth_v / K.fy)])
            # 깊이 센서는 보이는 표면을 잰다. 반치수만큼 밀어 무게중심으로.
            depth_v = depth_v + cfg.depth_centroid_offset * extent[2]
        else:
            s = cfg.no_depth_sigma
            cov = np.diag([s * s * 0.25, s * s * 0.25, s * s])
            extent = np.array([0.2, 0.2, 0.2])

        pos = K.backproject(cx_px, cy_px, depth_v)

        margin = 3.0
        touches = (det.box[0] <= margin or det.box[1] <= margin
                   or det.box[0] + det.box[2] >= K.width - margin
                   or det.box[1] + det.box[3] >= K.height - margin)
        vis = 0.5 if touches else _clamp01(0.5 + 0.5 * valid_ratio)
        return Measurement(pos, cov, extent, has_depth, vis)

    def project_token(self, tok: Token, K: Intrinsics, T_cam_world: SE3):
        p_cam = T_cam_world @ tok.position
        if p_cam[2] < 0.1:
            return None
        e = np.maximum(tok.extent, np.array([0.05, 0.05, 0.05]))
        min_x = min_y = 1e9
        max_x = max_y = -1e9
        any_pt = False
        for i in range(8):
            corner = np.array([e[0] if (i & 1) else -e[0],
                               e[1] if (i & 2) else -e[1],
                               e[2] if (i & 4) else -e[2]])
            q = T_cam_world @ (tok.position + corner)
            if q[2] < 0.05:
                continue
            px = K.project(q)
            min_x, max_x = min(min_x, px[0]), max(max_x, px[0])
            min_y, max_y = min(min_y, px[1]), max(max_y, px[1])
            any_pt = True
        if not any_pt:
            return None
        box = (float(np.float32(min_x)), float(np.float32(min_y)),
               float(np.float32(max_x - min_x)), float(np.float32(max_y - min_y)))
        screen = (0.0, 0.0, float(K.width), float(K.height))
        visible = (box_iou(box, screen) > 0.0
                   or (box[0] < screen[2] and box[1] < screen[3]
                       and box[0] + box[2] > 0.0 and box[1] + box[3] > 0.0))
        return box if visible else None

    # --- 연관 -------------------------------------------------------------
    def associate(self, dets, meas, candidates, K, T_cam_world, T_world_cam,
                  stamp, report):
        cfg = self.cfg
        nd, nt = len(dets), len(candidates)
        det_to_token = [-1] * nd
        margins = [-1.0] * nd
        if nd == 0:
            return det_to_token, margins
        if nt == 0:
            report.det_no_candidate += nd
            return det_to_token, margins

        cost = np.full((nd, nt), INFEASIBLE)
        near_dist = [INFEASIBLE] * nd
        near_maha = [INFEASIBLE] * nd
        feasible = [False] * nd

        R_wc = T_world_cam.R
        for d in range(nd):
            det = dets[d]
            z_world = T_world_cam @ meas[d].position_cam
            # 관측 공분산은 카메라 좌표계 값이다. 게이트가 재는 혁신은 월드
            # 좌표이므로 여기서 회전시켜야 한다 - 안 하면 큰 깊이 불확실성이
            # 엉뚱한 월드 축에 붙는다.
            R_world = R_wc @ meas[d].cov_cam @ R_wc.T

            for t in range(nt):
                tok = candidates[t]
                if not cfg.allow_cross_class and tok.class_id != det.class_id:
                    continue

                dt = max(0.0, stamp - tok.last_seen) if tok.last_seen > 0.0 else 0.0
                predicted = tok.position + tok.velocity * dt
                diff = z_world - predicted
                dist = float(np.linalg.norm(diff))
                if dist > cfg.max_association_distance:
                    continue

                # 등속 예측이 틀릴 수 있는 폭은 융합 공분산 안에 없다.
                # 포즈 회전 오차항(assoc_pose_rot_sigma)은 17.2 의 가설이
                # 기각되면서 C++ 에서 사라졌다 - 게이트를 깨뜨리는 것은
                # 회전이 아니라 포즈 발산이다.
                slack = cfg.assoc_maneuver_speed * max(dt, 1e-2)

                S = tok.position_cov + R_world
                S = S + np.eye(3) * (slack * slack + 1e-6)
                maha = float(diff @ np.linalg.solve(S, diff))
                if dist < near_dist[d]:
                    near_dist[d] = dist
                    near_maha[d] = maha
                if maha > cfg.max_association_mahalanobis:
                    continue
                feasible[d] = True

                c = cfg.distance_weight * maha
                proj = self.project_token(tok, K, T_cam_world)
                if proj is not None:
                    c += cfg.iou_weight * (1.0 - box_iou(proj, det.box))
                else:
                    c += cfg.iou_weight
                cost[d, t] = c

        rows, _cols, _total = solve_assignment(cost)
        det_to_token = list(rows)

        for d in range(nd):
            if det_to_token[d] < 0:
                if near_dist[d] >= INFEASIBLE:
                    report.det_no_candidate += 1
                elif not feasible[d]:
                    report.det_gated_out += 1
                    report.gated_maha_sum += near_maha[d]
                    report.gated_dist_sum += near_dist[d]
                else:
                    report.det_unassigned += 1
                continue
            best = cost[d, det_to_token[d]]
            second = INFEASIBLE
            for t in range(nt):
                if t == det_to_token[d]:
                    continue
                second = min(second, cost[d, t])
            margins[d] = 10.0 if second >= INFEASIBLE else max(0.0, second - best)
        return det_to_token, margins

    # --- 융합 -------------------------------------------------------------
    def fuse_position(self, tok: Token, z_world, R_world, dt: float) -> None:
        P = tok.position_cov.copy()
        if dt > 0.0:
            P = P + tok.velocity_cov * dt * dt
        P_inv = np.linalg.inv(P)
        R_inv = np.linalg.inv(R_world)
        new_cov = np.linalg.inv(P_inv + R_inv)
        predicted = tok.position + tok.velocity * dt
        new_pos = new_cov @ (P_inv @ predicted + R_inv @ z_world)

        if dt > 1e-3:
            v_obs = (new_pos - tok.position) / dt
            alpha = 0.35
            v_prev = tok.velocity.copy()
            tok.velocity = (1.0 - alpha) * v_prev + alpha * v_obs
            tok.acceleration = (tok.velocity - v_prev) / dt
            tok.velocity_cov = (new_cov + tok.position_cov) / (dt * dt)

        tok.position = new_pos
        tok.position_cov = new_cov

    # --- 생애주기 ---------------------------------------------------------
    def update_lifecycle(self, tok: Token, now: float, env_track_persist: float,
                         env_memory_retention: float) -> None:
        cfg = self.cfg
        silence = tok.silence_seconds(now)
        occl_timeout = cfg.occluded_timeout_s * env_track_persist
        dorm_timeout = cfg.dormant_timeout_s * env_memory_retention

        if tok.beliefs.existence < cfg.existence_retire_threshold:
            tok.lifecycle = Lifecycle.RETIRED
            return
        if (tok.beliefs.existence < cfg.existence_displace_threshold
                and tok.lifecycle != Lifecycle.PROVISIONAL):
            tok.lifecycle = Lifecycle.DISPLACED
            return
        if tok.beliefs.miss_count == 0:
            tok.lifecycle = (Lifecycle.ACTIVE
                             if tok.observation_count >= cfg.observations_to_activate
                             else Lifecycle.PROVISIONAL)
            return
        if tok.lifecycle == Lifecycle.PROVISIONAL and silence > occl_timeout:
            tok.lifecycle = Lifecycle.RETIRED
            return
        if silence <= occl_timeout:
            tok.lifecycle = Lifecycle.OCCLUDED
        elif silence <= dorm_timeout:
            tok.lifecycle = Lifecycle.DORMANT
        else:
            tok.lifecycle = Lifecycle.RETIRED

    # --- 통합 -------------------------------------------------------------
    def integrate(self, dets, stamp: float, K: Intrinsics, depth=None,
                  T_world_cam: SE3 | None = None, sensor_reliability: float = 1.0,
                  track_persistence_scale: float = 1.0,
                  memory_retention_scale: float = 1.0) -> IntegrationReport:
        cfg = self.cfg
        T_world_cam = T_world_cam or SE3.identity()
        T_cam_world = T_world_cam.inverse()
        report = IntegrationReport()

        meas = [self.measure(d, K, depth) for d in dets]
        candidates = [t for t in self.all_tokens() if t.is_alive()]

        det_to_token, margins = self.associate(
            dets, meas, candidates, K, T_cam_world, T_world_cam, stamp, report)

        token_matched = [False] * len(candidates)
        R_wc = T_world_cam.R

        for d, det in enumerate(dets):
            z_world = T_world_cam @ meas[d].position_cam
            R_world = R_wc @ meas[d].cov_cam @ R_wc.T

            if det_to_token[d] >= 0:
                ti = det_to_token[d]
                tok = candidates[ti]
                token_matched[ti] = True

                dt = max(0.0, stamp - tok.last_seen) if tok.last_seen > 0.0 else 0.0
                self.fuse_position(tok, z_world, R_world, dt)

                tok.meas_world = z_world
                tok.beliefs.meas_sigma = math.sqrt(
                    max(0.0, float(np.trace(R_world)) / 3.0))
                tok.box = det.box
                tok.visible_ratio = meas[d].visible_ratio
                tok.occlusion = 1.0 - meas[d].visible_ratio
                tok.last_seen = stamp
                tok.last_obs_cam_rot = R_wc.copy()
                tok.has_obs_cam_rot = True
                tok.observation_count += 1

                if meas[d].has_depth and meas[d].visible_ratio > 0.7:
                    tok.extent = 0.7 * tok.extent + 0.3 * meas[d].extent

                self.confidence.on_observed(
                    tok.beliefs, det.confidence, sensor_reliability,
                    image_quality=sensor_reliability, assoc_margin=margins[d],
                    obs_reliability=sensor_reliability)

                # 정적 판정은 누적 창으로. 인접 프레임 변위는 필터의 동역학이지
                # 물체의 운동이 아니고, 30 Hz 에서는 최소 판정 시간에도 못 미친다.
                if tok.static_ref_stamp <= 0.0:
                    tok.static_ref_position = tok.position.copy()
                    tok.static_ref_meas = z_world.copy()
                    tok.static_ref_stamp = stamp
                else:
                    window = stamp - tok.static_ref_stamp
                    if window >= self.confidence.cfg.static_min_dt:
                        self.confidence.update_static(
                            tok.beliefs, z_world - tok.static_ref_meas, window,
                            sensor_reliability)
                        tok.static_ref_position = tok.position.copy()
                        tok.static_ref_meas = z_world.copy()
                        tok.static_ref_stamp = stamp
                self.update_lifecycle(tok, stamp, track_persistence_scale,
                                      memory_retention_scale)
                report.matched += 1
            else:
                tok = Token(token_id=self._next_id)
                self._next_id += 1
                tok.class_id = det.class_id
                tok.class_name = det.class_name
                tok.position = z_world
                tok.position_cov = R_world
                tok.extent = meas[d].extent
                tok.box = det.box
                tok.visible_ratio = meas[d].visible_ratio
                tok.first_seen = stamp
                tok.last_seen = stamp
                tok.observation_count = 1
                tok.lifecycle = Lifecycle.PROVISIONAL
                agent = is_agent_class(det.class_name)
                tok.beliefs = Beliefs(existence=0.5, identity=0.5,
                                      static=0.15 if agent else 0.6,
                                      is_agent=agent)
                tok.beliefs.static_prior = tok.beliefs.static
                tok.static_ref_position = z_world.copy()
                tok.static_ref_meas = z_world.copy()
                tok.static_ref_stamp = stamp
                tok.meas_world = z_world
                tok.beliefs.meas_sigma = math.sqrt(
                    max(0.0, float(np.trace(R_world)) / 3.0))
                tok.last_obs_cam_rot = R_wc.copy()
                tok.has_obs_cam_rot = True

                self.confidence.on_observed(
                    tok.beliefs, det.confidence, sensor_reliability,
                    image_quality=sensor_reliability, assoc_margin=-1.0,
                    obs_reliability=sensor_reliability)
                self.tokens[tok.token_id] = tok
                report.created += 1

        for t, tok in enumerate(candidates):
            if token_matched[t]:
                continue
            idle = max(0.0, stamp - self._last_stamp) if self._last_stamp > 0.0 else 0.0
            self.confidence.decay_static_belief(tok.beliefs, idle)

            in_view = self.project_token(tok, K, T_cam_world) is not None
            if in_view:
                self.confidence.on_missed(tok.beliefs, _clamp01(1.0 - tok.occlusion),
                                          sensor_reliability)
                report.missed += 1
            else:
                dt = max(0.0, stamp - self._last_stamp) if self._last_stamp > 0.0 else 0.0
                self.confidence.on_out_of_view(tok.beliefs, dt)
                tok.beliefs.miss_count += 1
            self.update_lifecycle(tok, stamp, track_persistence_scale,
                                  memory_retention_scale)

        for tid in [k for k, t in self.tokens.items()
                    if t.lifecycle == Lifecycle.RETIRED]:
            del self.tokens[tid]
            report.retired += 1

        # 동적 객체 화면 점유율. 합산은 ID 순으로 한다.
        dynamic_area = 0.0
        screen_area = float(K.width) * float(K.height)
        for t in self.all_tokens():
            if t.is_dynamic() and t.beliefs.miss_count == 0:
                dynamic_area += float(np.float32(t.box[2]) * np.float32(t.box[3]))
        report.dynamic_area_ratio = _clamp01(dynamic_area / max(1.0, screen_area))

        self._last_stamp = stamp
        return report

    # --- 정적 마스크 ------------------------------------------------------
    def build_static_mask(self, K: Intrinsics):
        """ECDA 가 쓸 정적 픽셀 마스크 (0 = 동적) + 출처 회계."""
        cfg = self.cfg
        mask = np.full((K.height, K.width), 255, np.uint8)
        rep = MaskReport()
        total = max(1.0, float(K.width) * float(K.height))
        a_obs = a_stale = a_unjudged = 0.0

        for t in self.all_tokens():
            if not t.is_dynamic() or not t.is_alive():
                continue
            # 이번 프레임에 관측되지 않은 토큰은 지우지 않는다. 예측 박스 자리에
            # 실제로 있는 것은 대개 정적 배경이다 (실측 70 % / 86 %).
            if t.beliefs.miss_count > 0:
                continue
            # 판정이 한 번도 안 돈 토큰도 지우지 않는다. 그 static_belief 는
            # 클래스 표에서 온 사전분포일 뿐이고, 그것으로 화면을 지우는 것은
            # 14.1 이 기각한 클래스 마스킹 그 자체다.
            if t.beliefs.static_diag.updates == 0:
                r = clamp_rect(t.box, K.width, K.height, cfg.dynamic_mask_dilate)
                a_unjudged += float(max(0, r[2] * r[3]))
                rep.n_withheld += 1
                continue

            before = int(np.count_nonzero(mask))
            r = clamp_rect(t.box, K.width, K.height, cfg.dynamic_mask_dilate)
            if r[2] * r[3] <= 0:
                continue
            mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = 0
            rep.n_masking += 1
            added = float(before - int(np.count_nonzero(mask)))
            if t.beliefs.miss_count > 0:
                a_stale += added
            else:
                a_obs += added

        rep.masked_ratio = (total - float(np.count_nonzero(mask))) / total
        rep.from_observed = a_obs / total
        rep.from_stale = a_stale / total
        rep.withheld_unjudged = a_unjudged / total
        return mask, rep

    # --- 병합 -------------------------------------------------------------
    def merge(self, keep_id: int, absorb_id: int) -> None:
        if keep_id == absorb_id:
            raise ValueError("동일 토큰 병합")
        if keep_id not in self.tokens or absorb_id not in self.tokens:
            raise KeyError("병합 대상 토큰 없음")

        keep, absorbed = self.tokens[keep_id], self.tokens[absorb_id]
        Pk_inv = np.linalg.inv(keep.position_cov)
        Pa_inv = np.linalg.inv(absorbed.position_cov)
        new_cov = np.linalg.inv(Pk_inv + Pa_inv)
        keep.position = new_cov @ (Pk_inv @ keep.position + Pa_inv @ absorbed.position)
        keep.position_cov = new_cov

        self.confidence.merge(keep.beliefs, absorbed.beliefs)
        keep.observation_count += absorbed.observation_count
        if absorbed.first_seen > 0.0 and (keep.first_seen <= 0.0
                                          or absorbed.first_seen < keep.first_seen):
            keep.first_seen = absorbed.first_seen
        if absorbed.last_seen > keep.last_seen:
            keep.last_seen = absorbed.last_seen
        del self.tokens[absorb_id]


def build_constellation_from(tokens, reference: SE3, max_nodes: int = 40):
    """ConstellationIndex::buildFrom. 안정 랜드마크만 장소를 정의한다.

    반환: [(token_id, class_id, position, sigma)] - sigma 오름차순 상위 max_nodes.
    """
    world_to_ref = reference.inverse()
    nodes = [(t.token_id, t.class_id, world_to_ref @ t.position, t.position_sigma())
             for t in tokens if t is not None and t.is_stable_landmark()]
    # C++ 은 std::sort - 안정 정렬이 아니다. 동률 sigma 에서 순서가 갈릴 수
    # 있으므로 비교는 (sigma, id) 로 전순서를 만들어 한다.
    nodes.sort(key=lambda n: n[3])
    return nodes[:max_nodes]
