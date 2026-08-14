"""ECDA 참조 구현 - Environment-Conditioned Direct Alignment (Tier 0).

C++ src/localization/DirectAligner.cpp 와 같은 알고리즘이다. C++ 쪽은 한 번도
실행된 적이 없으므로(툴체인 부재), 이 구현이 그 알고리즘의 유일한 검증 수단이다.

규약도 동일하게 맞춘다:
  - 좌측 섭동 T <- exp(dxi) * T, xi = [rho, phi]
  - 상태 = 포즈 6 + 밝기 아핀 2 (a, b), 아핀은 정보행렬에서 주변화
  - 퇴화 판정은 대각 정규화 후 고유값 스펙트럼으로 (단위가 섞여 있으므로)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..reference.geometry import SE3, skew
from ..sim.world import CameraModel


class InformationModel(str, Enum):
    """보고할 포즈 정보행렬을 어떤 잡음 모델로 만들 것인가.

    앞의 셋은 같은 식이고 유효 표본수 N_eff 만 다르다.

        Lambda = H_pose / (chi2 / N_eff)

    chi2 는 해에서의 가중 잔차 제곱합이므로 chi2/N_eff 는 "독립 관측 하나가 갖는
    잔차 분산" 이다. 즉 이 모델들은 "이 프레임의 측광 잔차가 독립 관측 몇 개어치
    인가" 라는 한 가지 질문에만 서로 다르게 답한다. TUM 실측 ANEES (수용구간
    [5.5, 6.5], 정확도는 셋 다 동일 - 정보행렬은 추정에 관여하지 않는다):

        SENSOR_VARIANCE     5 425 ~ 146 087
        RESIDUAL_VARIANCE     845 ~   8 621
        EFFECTIVE_SAMPLE      3.7 ~     9.4
    """

    # N_eff 를 쓰지 않고 Lambda = H / photometric_variance.
    # 잔차가 (i) 서로 독립이고 (ii) 그 분산이 센서 잡음과 같다고 본다.
    # 실측에서 (ii) 부터 틀린다 - 해에서의 잔차 분산이 센서 잡음의 3~34 배다.
    SENSOR_VARIANCE = "sensor_variance"
    # N_eff = sum(w). 가중된 픽셀 하나하나가 독립 증거. (ii) 만 뺀다.
    RESIDUAL_VARIANCE = "residual_variance"
    # N_eff = effective_samples. 프레임 전체가 독립 관측 nu 개. (i) 도 뺀다.
    EFFECTIVE_SAMPLE = "effective_sample"
    # Lambda = (H / N) / coherent_sigma^2. 프레임 = 관측 하나이되 그 관측의
    # 잔차 분산을 chi2/N 이 아니라 상수로 둔다. 12 시퀀스 3 카메라 실측에서
    # 달성한 잔차 크기는 어느 시퀀스가 더 부정확한지를 예측하지 못했다
    # (p=0 이 최적, 최악 이탈 9.76x -> 2.43x). C++ 의 현재 기본값이다.
    COHERENT_FRAME = "coherent_frame"


@dataclass
class EcdaConfig:
    pyramid_levels: int = 4
    max_iterations: int = 30
    grid_cell: int = 8              # 셀당 최상위 그래디언트 1점
    min_gradient: float = 6.0
    max_points: int = 6000
    min_points: int = 60
    min_depth: float = 0.15
    max_depth: float = 40.0
    # sigma_Z = depth_sigma_rel * Z^2 (1/m). 0 이면 끈다.
    # C++ DirectAlignerConfig::depth_sigma_rel 과 같은 뜻. 근거는 그쪽 주석.
    #
    # 미구현 차이 하나를 명시한다. C++ 는 정보행렬을 쌓을 때 이 c 를 그대로
    # 쓰지 않고, cur 깊이맵과의 불일치가 센서 잡음 바닥을 넘으면 그 배율만큼
    # sigma_Z 를 키워 다시 쌓는다 (AlignmentResult::depth_sigma_scale). 이 오라클은
    # 기하 정합성 채널(cur 깊이)을 아예 갖고 있지 않아 그 배율이 항상 1 이다.
    # 차등 테스트는 cur 깊이가 없는 합성 장면에서 돌므로 C++ 쪽 배율도 1 이고
    # 비교가 성립한다 - cur 깊이를 주는 차등 케이스를 추가하려면 이쪽에도
    # 그 채널을 먼저 포팅해야 한다.
    depth_sigma_rel: float = 0.0
    # 4-이웃 깊이 상대차가 이보다 크면 경계로 보고 버린다.
    # C++ DirectAlignerConfig::depth_edge_ratio 와 같은 값이어야 한다.
    depth_edge_ratio: float = 0.05
    # --- 로버스트 커널 ---------------------------------------------------
    # C++ 은 06-results.md 11.4 에서 고정 huber_delta=12.0 을 버렸다. 12 는
    # 밝기 단위인데 정렬이 어긋난 상태의 잔차는 50~100 이라, **변위 정보를 가진
    # 점들이 정확히 가장 세게 눌리는** 커널이었다. 0.06 m 에서 아무것도 안 하는
    # 것보다 두 배 멀어졌다.
    #
    # 대체는 측정된 센서 잡음에 묶인 점진적 비볼록화다:
    #     delta = huber_k * sigma_resid * max(1, sigma_resid / (ratio * sigma_noise))
    # 파이썬이 이 값을 안 따라오면 두 구현은 **다른 점을 다른 가중으로** 쓰게 되고,
    # 그러면 크기 수준의 비교밖에 할 수 없다. 26절이 ECDA 에 원소 단위 오라클이
    # 없다고 적은 이유가 이 한 줄이었다.
    huber_k: float = 1.345           # C++ DirectAlignerConfig 와 같은 값이어야 한다
    huber_noise_ratio: float = 2.0
    huber_min_delta: float = 2.0

    # inlier 는 커널 임계가 아니라 **고정** 임계로 센다. 13.4 에서 적응형 임계로
    # 세다가 "잔차가 커지면 임계도 커져 비율이 그대로" 인 동어반복이 됐고,
    # 그 신호의 lift 가 0.00 (완전 역상관) 이었다.
    health_k: float = 4.0
    estimate_affine: bool = True
    affine_prior_weight: float = 1e2
    degeneracy_ratio: float = 1e-3
    lambda_init: float = 1e-4
    lambda_up: float = 10.0
    lambda_down: float = 0.3

    # --- 정보행렬 --------------------------------------------------------
    # 기본은 EFFECTIVE_SAMPLE. 근거는 실측이다: 기존 SENSOR_VARIANCE 로 낸
    # Lambda 는 TUM 다섯 시퀀스에서 ANEES 5 425 ~ 146 087 (수용구간 [5.5, 6.5])
    # 였고 1086 프레임 중 0 개가 프레임별 95 % 게이트를 통과했다. 정확도는
    # 좋았으므로(상대 병진오차 중앙 5~16 mm) 부정확한 게 아니라 과신이었다.
    # C++ DirectAlignerConfig::information_model 과 반드시 같아야 한다.
    # 다르면 차등 테스트가 재는 것이 알고리즘이 아니라 설정 차이가 된다.
    information_model: InformationModel = InformationModel.COHERENT_FRAME

    # 유효 표본수 nu. "이 프레임의 측광 잔차 전체가 독립 관측 몇 개어치인가".
    #
    # 1 로 두는 근거는 서브샘플링 실측이다. 점 집합을 1/2 ~ 1/16 로 솎아 다섯
    # 시퀀스를 다시 돌리면
    #     Lambda ~ N^+1.00    당연하다. 점마다 J J^T 를 더한다
    #     오차^2 ~ N^-0.00    전혀 줄지 않는다
    #     ANEES  ~ N^+0.97    그래서 과신이 N 에 그대로 비례한다
    # 점을 16 배 넣어도 fr1_xyz 상대 병진오차 중앙값은 5.29 -> 5.11 mm 로
    # 그대로다. 즉 한 프레임의 측광 잔차는 점 개수와 무관하게 사실상 한 번의
    # 관측이다. 상관 모델(N_eff = N/kappa)로는 이 모양이 나오지 않는다.
    #
    # 한계: 측광 잔차가 정말 독립인 데이터(합성 장면 + 화이트 노이즈)에서는
    # 점 개수만큼 보수적이 된다. 그런 데이터에서는 RESIDUAL_VARIANCE 가 맞다.
    effective_samples: float = 1.0

    # 한 프레임이라는 관측 하나가 갖는 잔차 분산 (intensity). COHERENT_FRAME 전용.
    # 12 시퀀스 3 카메라에서 맞춘 상수다. leave-one-out 범위 15.0~16.25.
    coherent_sigma: float = 15.5


@dataclass
class EcdaResult:
    T_cur_ref: SE3
    information: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(6))
    observable_dof: int = 0
    # 랭크만으로는 부족하다. 직접법의 고전적 실패는 랭크 6 을 유지한 채
    # 회전과 병진이 거의 같은 영상 운동을 만드는 *약한 관측성* 이다
    # (예: 원거리 정면 평면에서 y축 회전 vs x축 병진).
    # 랭크 판정만 보고하면 그 상황을 정상으로 오인한다.
    condition_number: float = np.inf
    # 문턱 없는 유효 자유도. exp(-sum p log p), p = lam/sum(lam).
    # C++ AlignmentResult::effective_dof 와 같은 뜻이고 근거는 그쪽 주석이다.
    # observable_dof 의 계단(1e-3)이 KITTI 에서 한 번도 밟히지 않는다는 실측이
    # 이 항의 이유다.
    effective_dof: float = 0.0
    weakest_direction: np.ndarray = field(default_factory=lambda: np.zeros(6))
    affine_a: float = 1.0
    affine_b: float = 0.0
    photometric_rmse: float = 0.0
    inlier_ratio: float = 0.0
    point_count: int = 0
    iterations: int = 0
    converged: bool = False

    @property
    def full_rank(self) -> bool:
        return self.observable_dof == 6

    def well_conditioned(self, limit: float = 1e3) -> bool:
        """랭크가 6 이어도 조건수가 나쁘면 그 추정은 신뢰할 수 없다."""
        return self.full_rank and self.condition_number < limit


# --- 이미지 유틸 -----------------------------------------------------------

def _downsample(img: np.ndarray) -> np.ndarray:
    """2x2 평균 축소. 에일리어싱을 줄여 직접정렬 수렴에 유리하다."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _downsample_nearest(img: np.ndarray) -> np.ndarray:
    """깊이 축소: 2x2 블록의 **유효값 중 중앙값** (C++ buildDepthPyramid 와 동일).

    평균은 안 된다 - 깊이 경계에서 전경과 배경 사이 허공에 점이 생긴다.
    그렇다고 `img[::2, ::2]` (최근접) 도 안 된다. 밝기는 2x2 평균이라 출력 x 가
    입력 2x+0.5 를 대표하는데 최근접은 2x 를 집으므로, 레벨마다 반 픽셀씩
    밀린다. 평면 장면에서는 무해하지만 깊이가 가파른 장면에서는 크게 어긋난다.

    실측(06-results.md 25.21): 이 한 줄 때문에 두 구현이 정면 평면(2.5 m)에서는
    1.8 mm 로 일치하면서 3~12 m 경사면에서는 **68 mm** 벌어졌다. ECDA 차분
    테스트가 그동안 평면 장면만 썼기 때문에 보이지 않았다.
    """
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    b = img[:h, :w].reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3)
    v = b.reshape(h // 2, w // 2, 4)

    valid = v > 0.0
    n = valid.sum(axis=2)
    # 무효를 +inf 로 밀어 올리면 정렬 후 앞쪽 n 개가 유효값이 된다.
    s = np.sort(np.where(valid, v, np.inf), axis=2)
    idx = np.clip(n // 2, 0, 3)
    out = np.take_along_axis(s, idx[..., None], axis=2)[..., 0]
    return np.where(n > 0, out, 0.0)


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """중앙차분. 경계는 복제로 채운다."""
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    return gx, gy


def scale_intrinsics(cam: CameraModel, level: int) -> CameraModel:
    """피라미드 레벨의 내부 파라미터. 픽셀 중심 규약(+0.5)을 지킨다."""
    s = 1.0 / (1 << level)
    return CameraModel(fx=cam.fx * s, fy=cam.fy * s,
                       cx=(cam.cx + 0.5) * s - 0.5,
                       cy=(cam.cy + 0.5) * s - 0.5,
                       width=cam.width >> level, height=cam.height >> level)


def build_pyramid(gray: np.ndarray, depth: np.ndarray | None,
                  cam: CameraModel, levels: int):
    """(그레이, 깊이, 내부파라미터) 레벨별 목록."""
    grays, depths, cams = [np.asarray(gray, float)], [], [cam]
    depths.append(None if depth is None else np.asarray(depth, float))

    for l in range(1, levels):
        grays.append(_downsample(grays[-1]))
        depths.append(None if depths[-1] is None else _downsample_nearest(depths[-1]))
        cams.append(scale_intrinsics(cam, l))
    return grays, depths, cams


def _sample_bilinear(img: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                     x: np.ndarray, y: np.ndarray):
    """세 맵을 같은 가중치로 한 번에 읽는다. 경계 여유 1픽셀."""
    h, w = img.shape
    valid = (x >= 1.0) & (y >= 1.0) & (x < w - 2) & (y < h - 2)
    xs = np.clip(x, 1.0, w - 2.001)
    ys = np.clip(y, 1.0, h - 2.001)

    ix, iy = xs.astype(int), ys.astype(int)
    fx, fy = xs - ix, ys - iy
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    def lerp(m):
        return (w00 * m[iy, ix] + w10 * m[iy, ix + 1]
                + w01 * m[iy + 1, ix] + w11 * m[iy + 1, ix + 1])

    return lerp(img), lerp(gx), lerp(gy), valid


def select_points(gray: np.ndarray, depth: np.ndarray, cfg: EcdaConfig,
                  weight_map: np.ndarray | None = None,
                  static_mask: np.ndarray | None = None,
                  level: int = 0) -> np.ndarray:
    """격자 셀당 최상위 그래디언트 1점.

    텍스처가 몰린 영역이 전체 시스템을 지배하는 것을 막는다. 공간적으로 고르게
    퍼진 점 집합이 조건수를 좋게 만든다.

    격자 크기는 반드시 레벨에 따라 줄여야 한다. 고정 크기를 쓰면 거친 레벨의
    셀 수가 min_points 아래로 떨어져 그 레벨이 통째로 건너뛰어지고, 결과적으로
    수렴 반경을 넓혀주는 바로 그 레벨들이 사라진다. 320x240 에 grid_cell=8 이면
    레벨 3(40x30)은 셀이 15개뿐이라 항상 버려진다.
    """
    cell_size = max(2, cfg.grid_cell >> level)
    gx, gy = _gradients(gray)
    mag = np.hypot(gx, gy)

    if weight_map is not None:
        mag = mag * weight_map
    if static_mask is not None:
        mag = np.where(static_mask, mag, 0.0)

    valid = (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)

    # 깊이 경계는 버린다 (C++ DirectAligner::depthIsLocallyFlat 의 포트).
    # 4-이웃이 하나라도 무효이거나 상대차 임계를 넘으면 그 점을 쓰지 않는다.
    # 이 항이 없으면 깊이가 완만한 장면에서는 두 구현이 같은 점을 고르지만,
    # 깊이가 크게 변하는 장면(실외 스테레오)에서는 점 집합 자체가 갈린다 -
    # 그러면 차분 테스트가 재는 것이 식이 아니라 점 선택이 된다.
    if cfg.depth_edge_ratio > 0.0:
        d = depth
        tol = cfg.depth_edge_ratio * d
        flat = np.zeros_like(valid)
        inner = (slice(1, -1), slice(1, -1))
        nb = (d[1:-1, :-2], d[1:-1, 2:], d[:-2, 1:-1], d[2:, 1:-1])
        ok = np.ones_like(d[inner], dtype=bool)
        for n in nb:
            ok &= (n > 0.0) & (np.abs(n - d[inner]) <= tol[inner])
        flat[inner] = ok
        valid = valid & flat

    mag = np.where(valid, mag, 0.0)

    h, w = gray.shape
    c = cell_size
    picks = []
    for y0 in range(1, h - 1, c):
        y1 = min(y0 + c, h - 1)
        for x0 in range(1, w - 1, c):
            x1 = min(x0 + c, w - 1)
            cell = mag[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            k = int(np.argmax(cell))
            cy, cx = divmod(k, cell.shape[1])
            if cell[cy, cx] >= cfg.min_gradient:
                picks.append((x0 + cx, y0 + cy))

    if not picks:
        return np.zeros((0, 2), dtype=int)
    pts = np.array(picks, dtype=int)
    if len(pts) > cfg.max_points:
        stride = int(np.ceil(len(pts) / cfg.max_points))
        pts = pts[::stride]
    return pts


# --- 정렬 ------------------------------------------------------------------

def _accumulate(points_cam: np.ndarray, intensity_ref: np.ndarray,
                cur_gray, cur_gx, cur_gy, cam: CameraModel,
                T: SE3, a: float, b: float, cfg: EcdaConfig,
                noise_sigma: float = 0.0):
    """정규방정식 (8x8) 을 벡터화해 누산한다.

    로버스트 임계는 고정값이 아니라 이번 잔차 분포와 측정 잡음에서 나온다
    (C++ DirectAligner::robustDelta 와 같은 식). 06-results.md 11.4 참조.
    """
    Q = points_cam @ T.R.T + T.t
    z = Q[:, 2]
    ok = z > cfg.min_depth
    if not np.any(ok):
        return None

    Q = Q[ok]
    I_ref = intensity_ref[ok]
    inv_z = 1.0 / Q[:, 2]

    u2 = cam.fx * Q[:, 0] * inv_z + cam.cx
    v2 = cam.fy * Q[:, 1] * inv_z + cam.cy

    I2, gx, gy, valid = _sample_bilinear(cur_gray, cur_gx, cur_gy, u2, v2)
    if not np.any(valid):
        return None

    Q, I_ref, inv_z = Q[valid], I_ref[valid], inv_z[valid]
    I2, gx, gy = I2[valid], gx[valid], gy[valid]

    r = I2 - (a * I_ref + b)
    abs_r = np.abs(r)

    # --- 적응형 huber 임계 (C++ robustDelta 의 포트) ---------------------
    # 표본이 적으면 산포 추정이 이상치에 휘둘리므로 L2 로 둔다.
    if abs_r.size < 32:
        delta = np.inf
    else:
        # 아핀 보정 후 잔차는 0 중심이라 median(|r|) 이 곧 MAD 다.
        sigma = 1.4826 * float(np.median(abs_r))
        noise = max(cfg.huber_min_delta, noise_sigma)
        relax = max(1.0, sigma / (cfg.huber_noise_ratio * noise))
        delta = max(cfg.huber_min_delta, cfg.huber_k * sigma * relax)

    w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-9))

    # d(pixel)/d(point) 에 이미지 그래디언트를 곱한 1x3 행
    gx_fx = gx * cam.fx * inv_z
    gy_fy = gy * cam.fy * inv_z
    Jp = np.stack([gx_fx, gy_fy, -(gx_fx * Q[:, 0] + gy_fy * Q[:, 1]) * inv_z], axis=1)

    # 좌측 섭동: dQ/d(rho) = I, dQ/d(phi) = -skew(Q) -> 회전 성분은 Q x Jp
    J = np.zeros((len(Q), 8))
    J[:, 0:3] = Jp
    J[:, 3:6] = np.cross(Q, Jp)
    J[:, 6] = -I_ref
    J[:, 7] = -1.0

    # 깊이 불확실성을 잔차 분산으로 옮긴다 (C++ DirectAligner::accumulate 의 포트).
    #   Q = R*P + t,  P = d * ray  ->  dQ/dd = (Q - t)/d
    #   dr/dd = Jp . (Q - t)/d,   sigma_d = c * d^2
    # 무게 곱 sigma_I^2/(sigma_I^2 + (dr/dd * sigma_d)^2).
    # 잔차 크기로 자르는 huber 와 다른 신호다 - 먼 벽은 잔차가 작아도 깊이가
    # 못 미덥다. 06-results.md 25.20/25.21 참조.
    photo_sigma = max(cfg.huber_min_delta, noise_sigma)
    if cfg.depth_sigma_rel > 0.0 and photo_sigma > 0.0:
        RP = Q - T.t
        # d 는 **ref 카메라에서의** 깊이다. |RP| 가 아니다 - RP = d * (R @ ray)
        # 이고 ray 는 단위벡터가 아니므로 원래 points_cam 의 z 를 그대로 쓴다.
        d_ref = points_cam[ok][valid][:, 2]
        dr_dd = np.einsum("ij,ij->i", Jp, RP) / d_ref
        sigma_d = cfg.depth_sigma_rel * d_ref * d_ref
        var_depth = (dr_dd * sigma_d) ** 2
        photo_var = photo_sigma * photo_sigma
        w = w * (photo_var / (photo_var + var_depth))

    Jw = J * w[:, None]
    H = Jw.T @ J
    g = Jw.T @ r
    chi2 = float(np.sum(w * r * r))
    # inlier 는 **고정** 임계로 센다. 커널 임계로 세면 잔차가 커질 때 임계도
    # 같이 커져 비율이 유지되고, 그 신호는 측정이 아니라 동어반복이 된다 (13.4).
    health = cfg.health_k * max(cfg.huber_min_delta, noise_sigma)
    inliers = int(np.sum(abs_r <= health))
    # wsum = sum(w). RESIDUAL_VARIANCE 의 유효 표본수.
    return H, g, chi2, len(Q), inliers, float(np.sum(w))


def align(ref_gray: np.ndarray, ref_depth: np.ndarray, cur_gray: np.ndarray,
          cam: CameraModel, init: SE3 | None = None,
          cfg: EcdaConfig | None = None,
          photometric_variance: float = 1.0,
          alpha_photometric: float = 1.0,
          static_mask: np.ndarray | None = None,
          noise_sigma: float = 0.0) -> EcdaResult:
    """ref -> cur 상대 포즈를 측광 잔차 최소화로 추정한다."""
    cfg = cfg or EcdaConfig()
    T = init or SE3.identity()
    a, b = 1.0, 0.0

    ref_pyr, depth_pyr, cams = build_pyramid(ref_gray, ref_depth, cam, cfg.pyramid_levels)
    cur_pyr, _, _ = build_pyramid(cur_gray, None, cam, cfg.pyramid_levels)

    mask_pyr = [None] * cfg.pyramid_levels
    if static_mask is not None:
        m = np.asarray(static_mask, bool)
        mask_pyr[0] = m
        for l in range(1, cfg.pyramid_levels):
            mask_pyr[l] = mask_pyr[l - 1][::2, ::2]

    result = EcdaResult(T_cur_ref=T)
    final_H = None
    final_stats = None

    for level in range(cfg.pyramid_levels - 1, -1, -1):
        kcam = cams[level]
        pts = select_points(ref_pyr[level], depth_pyr[level], cfg,
                            static_mask=mask_pyr[level], level=level)
        if len(pts) < cfg.min_points:
            continue

        u, v = pts[:, 0].astype(float), pts[:, 1].astype(float)
        d = depth_pyr[level][pts[:, 1], pts[:, 0]]
        points_cam = np.stack([(u - kcam.cx) * d / kcam.fx,
                               (v - kcam.cy) * d / kcam.fy, d], axis=1)
        intensity_ref = ref_pyr[level][pts[:, 1], pts[:, 0]]

        cgx, cgy = _gradients(cur_pyr[level])
        lam = cfg.lambda_init

        acc = _accumulate(points_cam, intensity_ref, cur_pyr[level], cgx, cgy,
                          kcam, T, a, b, cfg, noise_sigma)
        if acc is None:
            continue

        for _ in range(cfg.max_iterations):
            result.iterations += 1
            H, g, chi2, used, inliers, _wsum = acc

            Hd, gd = H.copy(), g.copy()
            if cfg.estimate_affine:
                # a=1, b=0 사전분포. 텍스처가 약할 때 아핀이 포즈를 먹는 것을 막는다.
                Hd[6, 6] += cfg.affine_prior_weight
                Hd[7, 7] += cfg.affine_prior_weight
                gd[6] += cfg.affine_prior_weight * (a - 1.0)
                gd[7] += cfg.affine_prior_weight * b
            else:
                for k in (6, 7):
                    Hd[k, :] = 0.0
                    Hd[:, k] = 0.0
                    Hd[k, k] = 1.0
                    gd[k] = 0.0

            damped = Hd + lam * np.diag(np.maximum(np.diag(Hd), 1e-9))
            try:
                dx = -np.linalg.solve(damped, gd)
            except np.linalg.LinAlgError:
                lam *= cfg.lambda_up
                continue
            if not np.all(np.isfinite(dx)):
                lam *= cfg.lambda_up
                continue

            T_new = SE3.exp(dx[:6]) @ T
            a_new = a + dx[6] if cfg.estimate_affine else a
            b_new = b + dx[7] if cfg.estimate_affine else b

            acc_new = _accumulate(points_cam, intensity_ref, cur_pyr[level], cgx, cgy,
                                  kcam, T_new, a_new, b_new, cfg, noise_sigma)
            if acc_new is None:
                lam *= cfg.lambda_up
                continue

            # 평균 잔차로 비교한다. 유효 점 수가 달라지므로 총합을 비교하면
            # 시야를 벗어나는 방향으로 최적화된다.
            cost_old = chi2 / max(1, used)
            cost_new = acc_new[2] / max(1, acc_new[3])

            if acc_new[3] >= cfg.min_points and cost_new < cost_old:
                T, a, b = T_new, a_new, b_new
                acc = acc_new
                lam = max(1e-9, lam * cfg.lambda_down)
                result.converged = True
                if np.linalg.norm(dx) < 1e-8:
                    break
            else:
                lam *= cfg.lambda_up
                if lam > 1e8:
                    break

        final_H = acc[0]
        final_stats = acc

    if final_H is None or not result.converged:
        return result

    _, _, chi2, used, inliers, wsum = final_stats
    result.T_cur_ref = T
    result.affine_a, result.affine_b = a, b
    result.point_count = used
    result.inlier_ratio = inliers / max(1, used)
    result.photometric_rmse = float(np.sqrt(chi2 / max(1, used)))

    # 아핀 자유도 주변화. 그대로 두면 노출 불확실성이 포즈 확신도로 샌다.
    Lam = final_H[:6, :6].copy()
    if cfg.estimate_affine:
        Haa = final_H[6:, 6:] + np.eye(2) * cfg.affine_prior_weight
        Hpa = final_H[:6, 6:]
        Lam -= Hpa @ np.linalg.solve(Haa, Hpa.T)

    # --- Lambda = H_pose / (chi2 / N_eff) -------------------------------
    # 분모는 "독립 관측 하나가 갖는 잔차 분산" 이다. 모델에 따라 N_eff 만 다르다.
    #   RESIDUAL_VARIANCE : N_eff = sum(w)          픽셀 하나하나가 독립 증거
    #   EFFECTIVE_SAMPLE  : N_eff = effective_samples  프레임 전체가 관측 nu 개
    #   COHERENT_FRAME    : 분산 자체를 상수로 둔다 (coherent_sigma)
    var = max(1e-6, photometric_variance)
    if cfg.information_model == InformationModel.COHERENT_FRAME:
        # 프레임 = 관측 하나, 그 관측의 잔차 분산은 chi2/N 이 아니라 상수다.
        # N 은 rmse 와 같은 분모(used)를 써야 EFFECTIVE_SAMPLE 대비 바뀌는 것이
        # "rmse^2 -> coherent_sigma^2" 하나뿐이 되어 비교가 깨끗하다.
        n = float(max(1, used))
        var = max(var, n * cfg.coherent_sigma * cfg.coherent_sigma)
    elif cfg.information_model != InformationModel.SENSOR_VARIANCE:
        n_eff = (wsum if cfg.information_model == InformationModel.RESIDUAL_VARIANCE
                 else max(1e-9, cfg.effective_samples))
        if n_eff > 0.0:
            # 센서 잡음을 바닥으로 깐다. 잔차 분산이 측정된 센서 잡음보다 작을
            # 수는 없고, 이 바닥이 없으면 합성 장면처럼 잔차가 0 에 가까울 때
            # Lambda 가 발산한다.
            var = max(var, chi2 / n_eff)

    # 환경 가중 alpha_0(E). 팩터그래프가 다른 tier 와 섞을 때 쓰는 정보 질량.
    Lam = Lam / var * max(1e-6, alpha_photometric)
    Lam = 0.5 * (Lam + Lam.T)
    result.information = Lam

    # 퇴화 진단: 병진(m)과 회전(rad)은 단위가 달라 대각 정규화 후 본다
    scale = 1.0 / np.sqrt(np.maximum(np.diag(Lam), 1e-12))
    Lam_n = Lam * scale[:, None] * scale[None, :]
    eig, vecs = np.linalg.eigh(Lam_n)
    order = np.argsort(eig)[::-1]
    eig, vecs = eig[order], vecs[:, order]

    result.eigenvalues = eig
    result.observable_dof = int(np.sum(eig > max(eig[0], 1e-12) * cfg.degeneracy_ratio))
    result.condition_number = float(eig[0] / max(eig[-1], 1e-12))
    # 같은 스펙트럼을 문턱 없이 읽은 유효 자유도 (effective_dof 주석).
    p = np.maximum(eig, 1e-300) / max(float(np.sum(eig)), 1e-300)
    result.effective_dof = float(np.exp(-np.sum(p * np.log(p))))
    # 가장 약하게 구속된 접선 방향. Tier 2(SPA) 가 채워야 할 축이다.
    result.weakest_direction = vecs[:, -1]
    return result
