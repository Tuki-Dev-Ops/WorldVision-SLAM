"""깊이맵에서 평면 추출.

Tier 2(SPA)의 재료다. 평면은 외관과 무관하므로 텍스처가 없거나 조명이
무너진 곳에서도 살아남는다 - 측광(Tier 0)이 죽는 바로 그 조건이다.

법선은 깊이 이웃의 외적으로 구한다. 학습 모델이 아니라 미분이므로
YOLO-only 제약과 무관하다 - docs/02 1.1 의 그래디언트 논증과 같은 근거다.

한계를 미리 적어 둔다: 여기 구현은 축정렬 격자 클러스터링이라 곡면이나
아주 얇은 구조는 못 잡는다. 방/복도의 벽·바닥·천장을 잡는 것이 목적이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.world import CameraModel


@dataclass(frozen=True)
class Plane:
    """카메라 좌표계의 평면. n·p = d, |n| = 1, d > 0 (카메라 앞)."""

    normal: np.ndarray
    distance: float
    inliers: int
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extent: float = 0.0                   # 평면 위 점들의 산포 (m)
    rms: float = 0.0                      # 평면 적합 잔차

    @property
    def confidence(self) -> float:
        """넓고 잘 맞는 평면일수록 믿을 만하다."""
        fit = 1.0 / (1.0 + 20.0 * self.rms)
        size = min(1.0, self.inliers / 400.0)
        return float(fit * size)

    def signed_distance(self, point: np.ndarray) -> float:
        return float(np.asarray(point, float) @ self.normal - self.distance)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Plane":
        """다른 좌표계로 옮긴다. n' = R n, d' = d + n'·t."""
        n = np.asarray(R, float) @ self.normal
        return Plane(n, float(self.distance + n @ np.asarray(t, float)),
                     self.inliers, np.asarray(R, float) @ self.centroid + t,
                     self.extent, self.rms)

    def __repr__(self) -> str:
        return (f"Plane(n={np.round(self.normal, 3).tolist()}, "
                f"d={self.distance:.3f}, n_in={self.inliers})")


@dataclass
class PlaneConfig:
    stride: int = 4                       # 깊이맵 서브샘플링
    min_depth: float = 0.2
    max_depth: float = 15.0

    normal_bins: int = 12                 # 법선 방향 격자 해상도
    distance_bin: float = 0.15            # m
    min_inliers: int = 150

    refine_threshold: float = 0.04        # m, 재적합 시 인라이어 판정
    refine_iterations: int = 2
    max_planes: int = 8

    # 법선 추정 시 이웃 깊이 차이가 이보다 크면 경계로 보고 버린다.
    # 이걸 안 하면 물체 실루엣에서 가짜 평면이 생긴다.
    depth_discontinuity: float = 0.15


def _backproject(depth: np.ndarray, cam: CameraModel, stride: int):
    """깊이맵 -> (H', W', 3) 카메라 좌표 점 배열."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[0:h:stride, 0:w:stride]

    x = (u - cam.cx) * d / cam.fx
    y = (v - cam.cy) * d / cam.fy
    return np.stack([x, y, d], axis=-1), d


def estimate_normals(points: np.ndarray, depth: np.ndarray,
                     cfg: PlaneConfig) -> tuple[np.ndarray, np.ndarray]:
    """이웃 외적으로 법선을 구한다. (법선, 유효 마스크)

    깊이 불연속에서는 법선을 만들지 않는다. 물체 실루엣을 평면으로 착각하는
    것이 이 방식의 대표적 실패라 명시적으로 막는다.
    """
    right = points[:, 1:, :] - points[:, :-1, :]
    down = points[1:, :, :] - points[:-1, :, :]

    a = right[:-1, :, :]
    b = down[:, :-1, :]
    n = np.cross(a, b)

    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-9
    n = n / np.maximum(norm, 1e-12)

    d0 = depth[:-1, :-1]
    dr = depth[:-1, 1:]
    dd = depth[1:, :-1]
    continuous = ((np.abs(dr - d0) < cfg.depth_discontinuity)
                  & (np.abs(dd - d0) < cfg.depth_discontinuity)
                  & (d0 > cfg.min_depth) & (d0 < cfg.max_depth))

    # Hesse 표준형 n·p = d, d > 0 에 맞춘다. 즉 법선은 카메라 반대쪽을 향한다.
    # 표면 법선(카메라 쪽)과 반대라 헷갈리기 쉬운데, d > 0 을 유지해야
    # transformed() 와 signed_distance() 가 단순해진다.
    p = points[:-1, :-1, :]
    flip = np.sum(n * p, axis=-1) < 0.0
    n = np.where(flip[..., None], -n, n)

    return n, valid & continuous


def extract_planes(depth: np.ndarray, cam: CameraModel,
                   config: PlaneConfig | None = None) -> list[Plane]:
    """깊이맵에서 지배적 평면들을 뽑는다. 신뢰도 내림차순."""
    cfg = config or PlaneConfig()
    depth = np.asarray(depth, float)
    if depth.ndim != 2:
        raise ValueError("깊이맵은 2차원이어야 함")

    depth = np.where(np.isfinite(depth), depth, 0.0)
    points, d = _backproject(depth, cam, cfg.stride)
    normals, valid = estimate_normals(points, depth[::cfg.stride, ::cfg.stride], cfg)

    if not np.any(valid):
        return []

    p = points[:-1, :-1, :][valid]
    n = normals[valid]
    dist = np.sum(n * p, axis=-1)

    keep = (dist > cfg.min_depth * 0.5) & (dist < cfg.max_depth)
    if not np.any(keep):
        return []
    p, n, dist = p[keep], n[keep], dist[keep]

    # (법선 방향, 거리) 격자로 후보를 모은다. 정확한 값은 나중에 재적합한다.
    key_n = np.round(n * cfg.normal_bins).astype(np.int64)
    key_d = np.round(dist / cfg.distance_bin).astype(np.int64)
    keys = np.stack([key_n[:, 0], key_n[:, 1], key_n[:, 2], key_d], axis=1)

    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                      return_counts=True)
    order = np.argsort(-counts)

    planes: list[Plane] = []
    used = np.zeros(len(p), dtype=bool)

    for idx in order:
        if counts[idx] < cfg.min_inliers or len(planes) >= cfg.max_planes:
            continue
        mask = (inverse == idx) & ~used
        if int(mask.sum()) < cfg.min_inliers:
            continue

        plane = _fit(p[mask], cfg)
        if plane is None:
            continue

        # 재적합: 격자 경계에서 잘린 인라이어를 되찾는다
        for _ in range(cfg.refine_iterations):
            residual = np.abs(p @ plane.normal - plane.distance)
            member = (residual < cfg.refine_threshold) & ~used
            if int(member.sum()) < cfg.min_inliers:
                break
            refined = _fit(p[member], cfg)
            if refined is None:
                break
            plane, mask = refined, member

        if plane.inliers < cfg.min_inliers:
            continue
        used |= mask
        planes.append(plane)

    planes.sort(key=lambda pl: -pl.confidence)
    return planes


def _fit(points: np.ndarray, cfg: PlaneConfig) -> Plane | None:
    """SVD 최소제곱 평면 적합."""
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centred = points - centroid

    try:
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vt[-1]
    # 최소 특이값이 다른 둘과 비슷하면 평면이 아니다 (구·모서리)
    if s[1] < 1e-9 or s[2] / s[1] > 0.3:
        return None

    distance = float(centroid @ normal)
    if distance < 0.0:
        normal, distance = -normal, -distance

    residual = points @ normal - distance
    return Plane(
        normal=normal,
        distance=distance,
        inliers=len(points),
        centroid=centroid,
        extent=float(np.mean(np.linalg.norm(centred, axis=1))),
        rms=float(np.sqrt(np.mean(residual ** 2))),
    )


def dominant_gravity(planes: list[Plane], min_confidence: float = 0.2) -> np.ndarray | None:
    """가장 넓은 수평면의 법선을 중력 방향으로 삼는다.

    ConstellationIndex 의 카이랄리티가 이 값을 필요로 한다. IMU 가 없을 때의
    대안이며, 수평면이 없으면 None 이다 - 지어내지 않는다.

    반환은 카메라 좌표계에서 '아래' 를 가리키는 단위벡터다.
    """
    best, best_score = None, 0.0
    for pl in planes:
        if pl.confidence < min_confidence:
            continue
        # 바닥은 카메라 y축(아래)과 거의 나란한 법선을 갖는다.
        # 카메라 좌표계는 y 가 아래이므로 법선의 y 성분이 지배적이어야 한다.
        vertical = abs(float(pl.normal[1]))
        if vertical < 0.85:
            continue
        score = vertical * pl.confidence * pl.inliers
        if score > best_score:
            best, best_score = pl, score

    if best is None:
        return None
    n = best.normal
    # 중력은 아래(+y) 방향
    return n if n[1] > 0 else -n
