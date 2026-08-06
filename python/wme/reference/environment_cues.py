"""영상 -> EnvironmentEvidence 참조 구현 (numpy).

C++ ``EnvironmentAnalyzer::computeEvidence()`` 가 부르는 추정기들의 오라클.
``environment.py`` 는 증거 -> tier 가중치 (deriveAdaptation) 만 덮고 있어서
그 앞단, 즉 **픽셀에서 증거를 뽑는 쪽** 은 오라클이 없었다. docs/06-results.md
11.1 이 "죽은 채널" 로 적어 둔 backlight / specular / shadow_strength /
scene_complexity 가 정확히 그쪽이다 - 단위 테스트는 있으나 독립 구현과
대조된 적이 없다.

정확도 방침
-----------
OpenCV 를 흉내 내는 게 아니라 **같은 답을 내는 다른 코드** 를 쓴다. 그래서
cv2 를 부르지 않고 uint8 정수 경로(BGR2GRAY, BGR2HSV)까지 직접 편다.
OpenCV 가 고정소수점으로 반올림하는 지점을 부동소수로 계산하면 경계값
(``V > 210 && S < 40``) 에서 한 픽셀씩 어긋나므로, 그 두 변환만 정수식을
그대로 재현한다. 나머지(Scharr, 최소필터, 평균/표준편차)는 부동소수다.

리사이즈는 재현하지 않는다. ``resizeToWidth`` 는 ``cols <= target_w`` 이면
그대로 복사하므로, 테스트 영상을 ``analysis_width`` 이하로 만들면 그 분기는
항등이 된다. INTER_AREA 를 numpy 로 다시 쓰는 것은 이 파일이 검증하려는
대상이 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# EnvironmentConfig 기본값
ANALYSIS_WIDTH = 192
DCP_PATCH = 9
DARK_BRIGHTNESS = 0.18
TEXTURE_MIN_GRADIENT = 6.0
HISTORY_SIZE = 9


def _c01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


# --- OpenCV 정수 색변환 -----------------------------------------------------

def bgr_to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    """cv::cvtColor(BGR2GRAY) 의 8U 고정소수점 경로.

    OpenCV 는 계수를 2^14 로 스케일한 정수로 누적하고 반올림한다. 실수
    0.114/0.587/0.299 로 계산하면 최대 1 gray 차이가 나고, 그 1 이
    estimateShadow 의 32 빈 히스토그램에서 빈을 바꾼다.
    """
    b = bgr[:, :, 0].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    r = bgr[:, :, 2].astype(np.int32)
    acc = b * 1868 + g * 9617 + r * 4899 + (1 << 13)
    return (acc >> 14).astype(np.uint8)


def bgr_to_sv_u8(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cv::cvtColor(BGR2HSV) 의 S, V 채널 (8U 정수 경로)."""
    x = bgr.astype(np.int32)
    v = x.max(axis=2)
    vmin = x.min(axis=2)
    diff = v - vmin

    # sdiv_table[i] = round((255 << 12) / i), i >= 1
    idx = np.maximum(v, 1)
    sdiv = ((255 << 12) * 2 + idx) // (2 * idx)   # 반올림 나눗셈
    s = (diff * sdiv + (1 << 11)) >> 12
    s = np.where(v == 0, 0, s)
    return s.astype(np.int32), v.astype(np.int32)


# --- 필터 -------------------------------------------------------------------

def erode_rect(img: np.ndarray, ksize: int) -> np.ndarray:
    """cv::erode(MORPH_RECT) = 국소 최소값.

    OpenCV 의 기본 테두리는 BORDER_CONSTANT + morphologyDefaultBorderValue()
    로, 침식에서는 +INF 이므로 영상 밖은 최소값에 기여하지 않는다. uint8 에서는
    255 로 채우는 것과 같다.
    """
    r = ksize // 2
    pad = np.pad(img, r, mode="constant", constant_values=255)
    out = np.full(img.shape, 255, dtype=img.dtype)
    h, w = img.shape
    for dy in range(ksize):
        for dx in range(ksize):
            out = np.minimum(out, pad[dy:dy + h, dx:dx + w])
    return out


def scharr(gray_f: np.ndarray, scale: float = 1.0 / 32.0) -> tuple[np.ndarray, np.ndarray]:
    """cv::Scharr(dx=1)/(dy=1), BORDER_REFLECT_101, 주어진 스케일.

    numpy 의 ``mode='reflect'`` 가 곧 BORDER_REFLECT_101 (gfedcb|abcdefgh)
    이다. ``'symmetric'`` 이 아니다 - 그쪽은 BORDER_REFLECT 로 테두리 한 줄이
    어긋난다.
    """
    p = np.pad(gray_f.astype(np.float64), 1, mode="reflect")
    # 분리형 상관: kx = [-1,0,1], ky = [3,10,3] (dx=1). dy=1 은 전치.
    def sep(kx, ky):
        rows = ky[0] * p[0:-2, :] + ky[1] * p[1:-1, :] + ky[2] * p[2:, :]
        return (kx[0] * rows[:, 0:-2] + kx[1] * rows[:, 1:-1] + kx[2] * rows[:, 2:]) * scale

    gx = sep((-1.0, 0.0, 1.0), (3.0, 10.0, 3.0))
    gy = sep((3.0, 10.0, 3.0), (-1.0, 0.0, 1.0))
    return gx, gy


# --- 개별 추정기 ------------------------------------------------------------

def estimate_haze(bgr_small: np.ndarray, dcp_patch: int = DCP_PATCH) -> float:
    """Dark Channel Prior + 저대비 결합."""
    min_ch = bgr_small.min(axis=2)
    dark = erode_rect(min_ch, dcp_patch)
    mean_dark = float(dark.mean()) / 255.0
    sigma = float(min_ch.astype(np.float64).std())
    low_contrast = _c01(1.0 - sigma / 40.0)
    return _c01(math.sqrt(_c01(mean_dark / 0.55) * low_contrast))


def estimate_backlight(gray_f: np.ndarray) -> float:
    """주변부 - 중앙 휘도차. 60 gray 를 극심으로 본다."""
    rows, cols = gray_f.shape
    mx, my = cols // 4, rows // 4
    cw, ch = cols - 2 * mx, rows - 2 * my
    if cw <= 0 or ch <= 0:
        return 0.0
    center = gray_f[my:my + ch, mx:mx + cw]

    total_sum = float(gray_f.astype(np.float64).sum())
    center_sum = float(center.astype(np.float64).sum())
    border_n = rows * cols - cw * ch
    if border_n < 1:
        return 0.0
    mean_border = (total_sum - center_sum) / border_n
    return _c01((mean_border - float(center.mean())) / 60.0)


def estimate_specular(bgr_small: np.ndarray) -> float:
    """하단 절반에서 저채도-고휘도 픽셀 비율. 15% 를 1.0 으로 본다."""
    if bgr_small.ndim != 3 or bgr_small.shape[2] != 3:
        return 0.0
    s, v = bgr_to_sv_u8(bgr_small)
    y0 = bgr_small.shape[0] // 2
    s, v = s[y0:], v[y0:]
    if s.size == 0:
        return 0.0
    hits = int(np.count_nonzero((v > 210) & (s < 40)))
    return _c01(hits / s.size / 0.15)


def estimate_shadow(gray_f: np.ndarray) -> float:
    """휘도 히스토그램의 이중봉성 (봉우리 깊이 x 분리도)."""
    bins = 32
    b = np.clip((gray_f.astype(np.float32) * bins / 256.0).astype(np.int32), 0, bins - 1)
    hist = np.bincount(b.ravel(), minlength=bins).astype(np.int64)
    total = float(gray_f.size)
    if total < 1.0:
        return 0.0

    p1 = 0
    for i in range(1, bins):
        if hist[i] > hist[p1]:
            p1 = i
    p2 = -1
    for i in range(bins):
        if abs(i - p1) < 6:
            continue
        if p2 < 0 or hist[i] > hist[p2]:
            p2 = i
    if p2 < 0:
        return 0.0

    lo, hi = min(p1, p2), max(p1, p2)
    valley = int(hist[lo:hi + 1].min())
    peak_min = float(min(hist[p1], hist[p2]))
    if peak_min < total * 0.02:
        return 0.0
    depth = 1.0 - valley / peak_min
    separation = (hi - lo) / bins
    return _c01(depth * separation * 2.0)


def estimate_texture_poverty(gray_f: np.ndarray, noise_sigma: float,
                             min_gradient: float = TEXTURE_MIN_GRADIENT) -> float:
    """임계 미만 그래디언트 픽셀 비율. 임계는 잡음 이득으로 바닥이 올라간다."""
    gx, gy = scharr(gray_f)
    mag = np.hypot(gx, gy)
    noise_grad = 0.68 * max(0.0, noise_sigma)
    thresh = np.float32(max(min_gradient, 2.0 * noise_grad))
    weak = int(np.count_nonzero(mag.astype(np.float32) < thresh))
    return _c01(weak / mag.size)


def estimate_scene_complexity(gray_f: np.ndarray,
                              min_gradient: float = TEXTURE_MIN_GRADIENT) -> float:
    """그래디언트 방향(180도 주기) 히스토그램의 정규화 엔트로피."""
    bins = 18
    gx, gy = scharr(gray_f)
    mag = np.hypot(gx, gy)
    keep = mag >= min_gradient
    if not keep.any():
        return 0.0

    # float64 로 계산한다. atan2 를 float 로 계산하면 정확히 수평인 엣지가
    # 그래디언트 부호에 따라 bin 8 / bin 9 로 갈라진다 - 같은 방향인데.
    ang = np.arctan2(gy[keep], gx[keep])
    ang = np.where(ang < 0.0, ang + math.pi, ang)
    # clamp 가 아니라 wrap: ang == pi 는 ang == 0 과 같은 방향이다.
    idx = (ang / math.pi * bins).astype(np.int32) % bins
    hist = np.bincount(idx, weights=mag[keep], minlength=bins)
    total = float(hist.sum())
    if total < 1e-9:
        return 0.0
    p = hist / total
    p = p[p > 1e-9]
    entropy = float(-(p * np.log(p)).sum())
    return _c01(entropy / math.log(bins))


def estimate_indoorness(bgr_small: np.ndarray) -> float:
    """하늘 단서 3개(상단 밝기 / 상단 청색 / 넓은 동적범위)의 여집합."""
    if bgr_small.ndim != 3 or bgr_small.shape[2] != 3:
        return 0.5
    rows, cols = bgr_small.shape[:2]
    top = bgr_small[0:max(1, rows // 4)]
    y0 = rows * 3 // 4
    bot = bgr_small[y0:y0 + max(1, rows - y0)]

    mt = top.reshape(-1, 3).mean(axis=0)
    mb = bot.reshape(-1, 3).mean(axis=0)
    lum_t = 0.114 * mt[0] + 0.587 * mt[1] + 0.299 * mt[2]
    lum_b = 0.114 * mb[0] + 0.587 * mb[1] + 0.299 * mb[2]

    sky_bright = _c01((lum_t - lum_b) / 70.0)
    sky_blue = _c01((mt[0] - 0.5 * (mt[1] + mt[2])) / 30.0)

    gray = bgr_to_gray_u8(bgr_small).astype(np.float64)
    wide_range = _c01(float(gray.std()) / 70.0)

    outdoorness = _c01(0.45 * sky_bright + 0.35 * sky_blue + 0.20 * wide_range)
    return 1.0 - outdoorness


def estimate_darkness(brightness: float, noise_sigma: float,
                      dark_brightness: float = DARK_BRIGHTNESS) -> float:
    """noisy-OR. 가중합이 아니다 - 잡음 0 에서 Dark 라벨이 도달 불가능해진다."""
    by_lum = _c01((dark_brightness - brightness) / dark_brightness)
    by_noise = _c01(noise_sigma / 12.0)
    return _c01(1.0 - (1.0 - by_lum) * (1.0 - 0.5 * by_noise))


# --- 조립 ------------------------------------------------------------------

@dataclass
class CueSet:
    """한 프레임의 영상 유래 증거 (EMA 이전의 원값)."""
    darkness: float = 0.0
    backlight: float = 0.0
    haze: float = 0.0
    specular: float = 0.0
    shadow_strength: float = 0.0
    texture_poverty: float = 0.0
    motion_blur: float = 0.0
    noise: float = 0.0
    lens_dirt: float = 0.0
    water_drop: float = 0.0
    scene_complexity: float = 0.0
    indoorness: float = 0.5


@dataclass
class Quality:
    """ImageQuality 중 EnvironmentAnalyzer 가 소비하는 부분만."""
    brightness: float = 0.5
    noise_sigma: float = 0.0
    blur_free: float = 1.0
    noise_free: float = 1.0
    occlusion_free: float = 1.0


def compute_cues(image: np.ndarray, quality: Quality | None = None,
                 dcp_patch: int = DCP_PATCH,
                 min_gradient: float = TEXTURE_MIN_GRADIENT) -> CueSet:
    """computeEvidence() 중 시간 이력이 필요 없는 항목 전부.

    camera_shake(위상상관, 직전 프레임 필요) 와 rain/snow(시간 중앙값 링버퍼)
    는 여기 없다. dust 는 snow 에 곱해지므로 함께 빠진다.
    """
    q = quality or Quality()
    if image.ndim == 3:
        bgr = image
        gray_f = bgr_to_gray_u8(bgr).astype(np.float64)
    else:
        bgr = None
        gray_f = image.astype(np.float64)

    c = CueSet()
    c.darkness = estimate_darkness(q.brightness, q.noise_sigma)
    c.backlight = estimate_backlight(gray_f)
    c.texture_poverty = estimate_texture_poverty(gray_f, q.noise_sigma, min_gradient)
    c.shadow_strength = estimate_shadow(gray_f)
    c.scene_complexity = estimate_scene_complexity(gray_f, min_gradient)
    if bgr is not None:
        c.haze = estimate_haze(bgr, dcp_patch)
        c.specular = estimate_specular(bgr)
        c.indoorness = estimate_indoorness(bgr)

    c.motion_blur = _c01(1.0 - q.blur_free)
    c.noise = _c01(1.0 - q.noise_free)
    c.lens_dirt = _c01(1.0 - q.occlusion_free)
    c.water_drop = _c01(c.lens_dirt * (1.0 - c.lens_dirt) * 4.0)
    return c
