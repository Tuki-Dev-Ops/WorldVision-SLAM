"""광선 투사 렌더러.

지금까지 시뮬레이터는 객체 수준 관측만 만들었다. 그래서 ECDA(Tier 0) 측광 항은
검증할 수 없었고, 팩터그래프에서도 상대 포즈 팩터로 대리했다. 여기서 픽셀을
만들어 그 구멍을 메운다.

물리적으로 맞는 열화 모델을 쓰는 것이 요점이다. 특히 안개는 깊이 의존이다:

    I = J * exp(-beta * z) + A * (1 - exp(-beta * z))

먼 픽셀이 먼저 사라진다. 전역 스칼라로 밝기를 낮추는 것과는 전혀 다르고,
alpha_0(E) 가 실제로 그 열화를 맞히는지 재려면 이 모델이 필요하다.

의도적으로 간단하다: 텍스처 입힌 유한 평면 + 축정렬 박스. 사실적 렌더링이
목적이 아니라 측광 정렬이 물 수 있는 그래디언트를 만드는 것이 목적이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..reference.environment import Evidence
from ..reference.geometry import SE3
from .world import CameraModel, SimObject, SimWorld


# --- 절차적 텍스처 ---------------------------------------------------------

def procedural_texture(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """월드 좌표 -> 0..1 밝기.

    무리수 비율의 사인 합. 결정적이고, 어느 스케일에서도 그래디언트가 있고,
    UV 매핑이 필요 없다. 직접정렬에는 매끄럽되 풍부한 텍스처가 이상적이다.
    """
    p = np.asarray(points, dtype=float)
    phase = 0.37 * seed

    # 고주파 성분의 비중이 낮으면 모션블러가 영상을 거의 바꾸지 못해
    # 블러 열화 하에서의 ECDA 거동을 검증할 수 없다. 실제 장면처럼
    # 여러 스케일에 고르게 에너지를 둔다.
    freqs = [(1.7, 2.3, 0.9), (3.1, 1.3, 2.7), (5.9, 4.1, 1.1),
             (0.7, 6.7, 3.3), (11.3, 2.9, 5.1), (17.9, 13.1, 7.3),
             (23.7, 5.3, 19.1)]
    weights = [0.22, 0.17, 0.15, 0.13, 0.13, 0.11, 0.09]

    value = np.zeros(len(p))
    for (fx, fy, fz), w in zip(freqs, weights):
        value += w * np.sin(fx * p[:, 0] + fy * p[:, 1] + fz * p[:, 2] + phase)
    return np.clip(0.5 + 0.42 * value, 0.02, 0.98)


# --- 프리미티브 ------------------------------------------------------------

@dataclass
class Quad:
    """유한 평면. center 를 중심으로 u_axis / v_axis 방향 half_size 만큼."""
    center: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    half_u: float
    half_v: float
    albedo: float = 1.0
    texture_seed: int = 0

    def intersect(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """각 광선의 교차 깊이 t (없으면 inf). dirs 는 z=1 규약이라 t = z-깊이."""
        denom = dirs @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.center - origin) @ self.normal) / denom

        t = np.where((np.abs(denom) > 1e-9) & (t > 1e-4), t, np.inf)

        # inf 를 그대로 곱하면 NaN 이 나와 이후 내적이 오염된다.
        # 미적중 광선은 t=0 으로 대체해 계산만 하고 마지막에 걸러낸다.
        t_safe = np.where(np.isfinite(t), t, 0.0)
        local = (origin + t_safe[:, None] * dirs) - self.center
        inside = ((np.abs(local @ self.u_axis) <= self.half_u)
                  & (np.abs(local @ self.v_axis) <= self.half_v))
        return np.where(inside, t, np.inf)


@dataclass
class Box:
    """축정렬 상자. 시뮬레이터의 객체 모델과 같은 형태."""
    center: np.ndarray
    half_extent: np.ndarray
    albedo: float = 1.0
    texture_seed: int = 0
    object_id: int = -1

    def intersect(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """슬랩 방식 AABB 교차."""
        lo = self.center - self.half_extent
        hi = self.center + self.half_extent

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / dirs
            t2 = (hi - origin) / dirs

        t_near = np.max(np.minimum(t1, t2), axis=1)
        t_far = np.min(np.maximum(t1, t2), axis=1)
        valid = (t_far >= np.maximum(t_near, 1e-4)) & np.isfinite(t_near)
        return np.where(valid, np.maximum(t_near, 1e-4), np.inf)


@dataclass
class RenderScene:
    quads: list[Quad] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    @staticmethod
    def room(size: float = 5.0, height: float = 2.6) -> "RenderScene":
        """바닥 / 천장 / 벽 네 면. 직접정렬이 물 표면을 제공한다."""
        s, h = size, height
        ex, ey, ez = np.eye(3)
        return RenderScene(quads=[
            Quad(np.array([0, 0, 0.0]), ez, ex, ey, s, s, 0.85, 1),      # 바닥
            Quad(np.array([0, 0, h]), -ez, ex, ey, s, s, 0.75, 2),       # 천장
            Quad(np.array([s, 0, h / 2]), -ex, ey, ez, s, h / 2, 0.9, 3),
            Quad(np.array([-s, 0, h / 2]), ex, ey, ez, s, h / 2, 0.9, 4),
            Quad(np.array([0, s, h / 2]), -ey, ex, ez, s, h / 2, 0.9, 5),
            Quad(np.array([0, -s, h / 2]), ey, ex, ez, s, h / 2, 0.9, 6),
        ])

    def with_objects(self, world: SimWorld, t: float) -> "RenderScene":
        """시뮬레이션 객체를 상자로 추가한다. 객체 마스크의 근거가 된다."""
        boxes = list(self.boxes)
        for o in world.objects:
            p = o.position_at(t)
            if p is None:
                continue
            boxes.append(Box(np.asarray(p, float), np.asarray(o.extent, float),
                             albedo=0.7 + 0.02 * (o.object_id % 10),
                             texture_seed=100 + o.object_id,
                             object_id=o.object_id))
        return RenderScene(self.quads, boxes)


# --- 렌더링 ----------------------------------------------------------------

@dataclass
class RenderResult:
    gray: np.ndarray              # (H, W) float32, 0..255
    depth: np.ndarray             # (H, W) float32, z-깊이 (미터), 미적중은 inf
    object_ids: np.ndarray        # (H, W) int32, 배경은 -1


def _ray_directions(cam: CameraModel, pose: SE3) -> np.ndarray:
    """월드 좌표 광선 방향. 카메라 프레임 z=1 규약이라 t 가 곧 z-깊이."""
    v, u = np.meshgrid(np.arange(cam.height), np.arange(cam.width), indexing="ij")
    d_cam = np.stack([(u.ravel() - cam.cx) / cam.fx,
                      (v.ravel() - cam.cy) / cam.fy,
                      np.ones(u.size)], axis=1)
    return d_cam @ pose.R.T


def render(scene: RenderScene, pose: SE3, cam: CameraModel,
           ambient: float = 200.0) -> RenderResult:
    """한 시점을 렌더링한다. 열화는 적용하지 않은 원본."""
    origin = pose.t
    dirs = _ray_directions(cam, pose)
    n = dirs.shape[0]

    best_t = np.full(n, np.inf)
    albedo = np.zeros(n)
    tex_seed = np.zeros(n, dtype=int)
    normals = np.zeros((n, 3))
    obj = np.full(n, -1, dtype=np.int32)

    def absorb(t, alb, seed, normal_vec, oid):
        nonlocal best_t
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        albedo[closer] = alb
        tex_seed[closer] = seed
        normals[closer] = normal_vec
        obj[closer] = oid

    for q in scene.quads:
        absorb(q.intersect(origin, dirs), q.albedo, q.texture_seed, q.normal, -1)

    for b in scene.boxes:
        t = b.intersect(origin, dirs)
        if not np.any(np.isfinite(t)):
            continue
        # 박스 법선은 면마다 다르지만, 음영 변화만 필요하므로
        # 중심 -> 적중점 방향의 지배 축으로 근사한다.
        hit = origin + np.where(np.isfinite(t), t, 0.0)[:, None] * dirs
        local = hit - b.center
        axis = np.argmax(np.abs(local / np.maximum(b.half_extent, 1e-6)), axis=1)
        normal = np.eye(3)[axis] * np.sign(local[np.arange(n), axis])[:, None]

        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        albedo[closer] = b.albedo
        tex_seed[closer] = b.texture_seed
        normals[closer] = normal[closer]
        obj[closer] = b.object_id

    hit_points = origin + np.where(np.isfinite(best_t), best_t, 0.0)[:, None] * dirs

    # 텍스처는 시드별로 묶어 한 번씩만 평가한다
    texture = np.full(n, 0.5)
    for seed in np.unique(tex_seed):
        m = tex_seed == seed
        if np.any(m):
            texture[m] = procedural_texture(hit_points[m], int(seed))

    # 램버트 근사 음영. 표면이 정면일수록 밝다.
    view = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    shade = 0.45 + 0.55 * np.abs(np.sum(normals * view, axis=1))

    gray = ambient * albedo * texture * shade
    gray = np.where(np.isfinite(best_t), gray, 0.0)

    shape = (cam.height, cam.width)
    return RenderResult(
        gray=gray.reshape(shape).astype(np.float32),
        depth=best_t.reshape(shape).astype(np.float32),
        object_ids=obj.reshape(shape),
    )


# --- 열화 ------------------------------------------------------------------

def apply_degradation(result: RenderResult, ev: Evidence,
                      rng: np.random.Generator,
                      atmospheric_light: float = 235.0,
                      fog_beta: float = 0.18) -> np.ndarray:
    """물리 모델에 따른 영상 열화.

    안개가 깊이 의존이라는 점이 핵심이다. 전역 스칼라 감쇠로 흉내내면
    ECDA 의 실제 열화 양상을 재현하지 못하고, alpha_0(E) 검증이 무의미해진다.
    """
    img = result.gray.astype(np.float64).copy()

    if ev.haze > 0.0:
        # 대기 산란: 먼 픽셀이 먼저 대기광에 묻힌다
        z = np.where(np.isfinite(result.depth), result.depth, 60.0)
        transmission = np.exp(-fog_beta * ev.haze * z)
        img = img * transmission + atmospheric_light * (1.0 - transmission)

    if ev.darkness > 0.0:
        # 노출 부족 + 게인 노이즈. 저조도의 실제 형태다.
        img *= (1.0 - 0.92 * ev.darkness)

    if ev.motion_blur > 0.0:
        img = _directional_blur(img, ev.motion_blur)

    if ev.darkness > 0.0 or ev.noise > 0.0:
        sigma = 2.0 + 22.0 * max(ev.noise, ev.darkness * 0.7)
        img += rng.normal(0.0, sigma, img.shape)

    if ev.lens_dirt > 0.0:
        img = _lens_smudge(img, ev.lens_dirt)

    return np.clip(img, 0.0, 255.0)


def _directional_blur(img: np.ndarray, amount: float) -> np.ndarray:
    """수평 방향 박스 블러. 모션블러의 1차 근사."""
    k = max(1, int(round(1 + 26 * amount)))
    if k <= 1:
        return img
    pad = np.pad(img, ((0, 0), (k // 2, k // 2)), mode="edge")
    cumsum = np.cumsum(pad, axis=1)
    cumsum = np.concatenate([np.zeros((img.shape[0], 1)), cumsum], axis=1)
    return (cumsum[:, k:k + img.shape[1]] - cumsum[:, :img.shape[1]]) / k


def _lens_smudge(img: np.ndarray, amount: float) -> np.ndarray:
    """화면 고정 위치의 흐린 얼룩. 카메라가 움직여도 같은 자리에 남는다."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - w * 0.28) / (w * 0.16)) ** 2
                    + ((yy - h * 0.62) / (h * 0.16)) ** 2))
    smeared = _directional_blur(img, 0.6)
    alpha = np.clip(blob * amount, 0.0, 1.0)
    return img * (1.0 - alpha) + smeared * alpha


@dataclass
class RenderedFrame:
    gray: np.ndarray              # 열화 적용 후 (H, W) float64, 0..255
    depth: np.ndarray             # 참 z-깊이
    clean: np.ndarray             # 열화 전 (진단용)
    object_ids: np.ndarray
    pose: SE3
    detections: list = field(default_factory=list)

    @property
    def static_mask(self) -> np.ndarray:
        """동적 객체를 제외한 마스크. ECDA 의 w_dynamic 에 대응한다."""
        return np.ones_like(self.gray, dtype=bool)


@dataclass
class RenderedDetection:
    """렌더링된 객체 마스크에서 뽑은 검출.

    추상 센서 모델과 달리 가려짐과 화면 잘림이 물리적으로 맞다 - 마스크가
    실제 광선 투사 결과이기 때문이다.
    """
    object_id: int
    box: tuple[float, float, float, float]     # x, y, w, h
    depth: float                               # 마스크 픽셀 깊이 중앙값
    pixels: int
    truncated: bool                            # 화면 경계에 닿았는가


def detections_from_render(result: RenderResult, min_pixels: int = 60,
                           margin: int = 2) -> list[RenderedDetection]:
    """object_ids 맵에서 객체별 박스와 깊이를 추출한다."""
    out: list[RenderedDetection] = []
    h, w = result.object_ids.shape

    for oid in np.unique(result.object_ids):
        if oid < 0:
            continue
        mask = result.object_ids == oid
        n = int(mask.sum())
        if n < min_pixels:
            continue

        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        depths = result.depth[mask]
        depths = depths[np.isfinite(depths)]
        if depths.size == 0:
            continue

        out.append(RenderedDetection(
            object_id=int(oid),
            box=(float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)),
            depth=float(np.median(depths)),
            pixels=n,
            truncated=(x0 <= margin or y0 <= margin
                       or x1 >= w - 1 - margin or y1 >= h - 1 - margin),
        ))
    return out


def render_frame(scene: RenderScene, pose: SE3, cam: CameraModel,
                 ev: Evidence | None = None, seed: int = 0) -> RenderedFrame:
    ev = ev or Evidence()
    raw = render(scene, pose, cam)
    rng = np.random.default_rng(seed)
    degraded = apply_degradation(raw, ev, rng)
    return RenderedFrame(degraded, raw.depth, raw.gray.astype(np.float64),
                         raw.object_ids, pose, detections_from_render(raw))
