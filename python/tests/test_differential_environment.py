"""영상 -> EnvironmentEvidence 차분 테스트 (C++ EnvironmentAnalyzer vs numpy).

docs/06-results.md 11.1 은 backlight / specular / shadow_strength /
scene_complexity 를 "죽은 채널" 로 적어 두었다. 단위 테스트는 있었지만
독립 구현과 대조된 적이 없어, 식이 통째로 틀려도 초록이 나오는 상태였다.
여기서 처음으로 오라클을 붙인다.

경로에 대하여
-------------
추정기들은 private 이다. 열지 않고 공개 API ``update(frame, quality)`` 로
비교한다. ``evidence_ema=1.0`` 이면 EMA 가 항등이 되어 그 프레임의 원
추정값이 그대로 evidence 에 실린다. private 을 노출하면 실제로 쓰이는
경로가 아닌 것을 재게 된다.

리사이즈 분기를 피하려고 모든 입력은 폭 <= analysis_width 로 만든다.

10.4 대비: 각 채널마다 "이 입력들이 서로 다른 값을 낸다" 를 먼저 확인한다.
모든 입력에서 채널이 0 이면 0 == 0 은 통과하지만 아무것도 검증하지 않는다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wme import HAS_NATIVE
from wme.reference import environment_cues as ref

# 확장이 없으면 모듈 수집 단계에서 죽지 않고 skip 한다. 다른 차분 테스트들과
# 같은 규약이다 - `python` 워크플로는 확장을 일부러 빌드하지 않고, 그 잡의
# 계약이 "C++ 없이도 파이썬 층은 혼자 초록" 이기 때문이다.
pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="_core 미빌드")

if HAS_NATIVE:
    from wme import _core

W, H = 160, 120


def analyzer(**cfg):
    c = _core.EnvironmentConfig()
    c.analysis_width = 192
    c.evidence_ema = 1.0      # EMA 를 항등으로: 원 추정값이 그대로 나온다
    c.update_hz = 1e6         # 주기 제한 해제
    for k, v in cfg.items():
        setattr(c, k, v)
    return _core.EnvironmentAnalyzer(c)


def quality(brightness=0.5, noise_sigma=0.0, blur_free=1.0,
            noise_free=1.0, occlusion_free=1.0):
    q = _core.ImageQuality()
    q.brightness = brightness
    q.noise_sigma = noise_sigma
    q.blur_free = blur_free
    q.noise_free = noise_free
    q.occlusion_free = occlusion_free
    return q, ref.Quality(brightness, noise_sigma, blur_free, noise_free, occlusion_free)


# --- 입력 영상 ---------------------------------------------------------------

def _rng(seed):
    return np.random.default_rng(seed)


def img_checker(period=8, lo=40, hi=200):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.where(((xx // period) + (yy // period)) % 2 == 0, hi, lo).astype(np.uint8)
    return np.dstack([g, g, g])


def img_flat(v=128):
    return np.full((H, W, 3), v, dtype=np.uint8)


def img_noise(seed=0, mu=120, sigma=30):
    g = np.clip(_rng(seed).normal(mu, sigma, (H, W)), 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def img_backlit(border=220, center=40):
    g = np.full((H, W), border, dtype=np.uint8)
    g[H // 4:H - H // 4, W // 4:W - W // 4] = center
    return np.dstack([g, g, g])


def img_bimodal(dark=45, bright=205, seed=3):
    """두 봉우리 + 사이가 빈 분포. estimateShadow 가 노리는 모양."""
    r = _rng(seed)
    m = r.random((H, W)) < 0.5
    g = np.where(m, dark, bright).astype(np.uint8)
    return np.dstack([g, g, g])


def img_specular():
    """하단에 저채도-고휘도(V>210, S<40) 픽셀, 상단은 채도 높은 색."""
    im = np.zeros((H, W, 3), dtype=np.uint8)
    im[: H // 2] = (30, 60, 200)               # 채도 높음
    im[H // 2:] = (100, 100, 100)
    im[H // 2:, : W // 3] = (245, 245, 248)    # V=248, S=(248-245)*255/248=3
    return im


def img_sky(top=(215, 175, 150), bottom=(60, 62, 58)):
    im = np.zeros((H, W, 3), dtype=np.uint8)
    im[: H // 2] = top
    im[H // 2:] = bottom
    return im


def img_hazy(base=None, t=0.35, air=225):
    b = base if base is not None else img_checker()
    return np.clip(b.astype(np.float64) * t + air * (1.0 - t), 0, 255).astype(np.uint8)


def img_gradient():
    yy, xx = np.mgrid[0:H, 0:W]
    g = ((xx * 255) // (W - 1)).astype(np.uint8)
    return np.dstack([g, g, g])


def img_stripes(angle_deg, period=6, lo=50, hi=210):
    yy, xx = np.mgrid[0:H, 0:W]
    a = math.radians(angle_deg)
    proj = xx * math.cos(a) + yy * math.sin(a)
    g = np.where((proj.astype(np.int64) // period) % 2 == 0, hi, lo).astype(np.uint8)
    return np.dstack([g, g, g])


IMAGES = {
    "checker": img_checker(),
    "flat": img_flat(),
    "noise": img_noise(),
    "backlit": img_backlit(),
    "bimodal": img_bimodal(),
    "specular": img_specular(),
    "sky": img_sky(),
    "hazy": img_hazy(),
    "gradient": img_gradient(),
    "stripes0": img_stripes(0),
    "stripes45": img_stripes(45),
    "checker_fine": img_checker(period=3, lo=90, hi=150),
}

CHANNELS = ["backlight", "haze", "specular", "shadow_strength",
            "texture_poverty", "darkness", "motion_blur", "noise",
            "lens_dirt", "water_drop"]


def _run_all():
    """모든 입력에 대해 (C++ evidence, numpy CueSet) 쌍을 만든다."""
    out = {}
    for name, im in IMAGES.items():
        qc, qr = quality(brightness=0.30, noise_sigma=0.0)
        a = analyzer()
        st = a.update(im, qc, 1.0)
        out[name] = (st, ref.compute_cues(im, qr))
    return out


@pytest.fixture(scope="module")
def runs():
    return _run_all()


# --- 판별력 확인 (10.4) ------------------------------------------------------

@pytest.mark.parametrize("channel", CHANNELS + ["scene_complexity"])
def test_inputs_discriminate_each_channel(runs, channel):
    """입력 묶음이 이 채널을 실제로 흔드는지 먼저 본다.

    이 검사가 없으면 '모든 입력에서 0' 인 채널은 C++/numpy 가 나란히 0 을
    내며 통과한다. 그때 통과한 것은 식이 아니라 0 == 0 이다.
    """
    if channel == "scene_complexity":
        vals = [st.scene_complexity for st, _ in runs.values()]
    else:
        vals = [getattr(st.evidence, channel) for st, _ in runs.values()]
    spread = max(vals) - min(vals)
    if channel in ("motion_blur", "noise", "lens_dirt", "water_drop", "darkness"):
        # 품질 유래 채널은 영상에 무반응이 정상. 대신 품질 스윕으로 따로 검증한다.
        pytest.skip("품질 유래 채널 - test_quality_derived_channels 가 덮는다")
    assert spread > 0.05, f"{channel}: 모든 입력에서 {vals[0]:.4f} - 판별하지 못함"


# --- 채널별 차분 -------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(IMAGES))
def test_backlight_matches(runs, name):
    st, r = runs[name]
    assert st.evidence.backlight == pytest.approx(r.backlight, abs=2e-4)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_haze_matches(runs, name):
    st, r = runs[name]
    assert st.evidence.haze == pytest.approx(r.haze, abs=2e-4)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_specular_matches(runs, name):
    st, r = runs[name]
    assert st.evidence.specular == pytest.approx(r.specular, abs=2e-4)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_shadow_matches(runs, name):
    st, r = runs[name]
    assert st.evidence.shadow_strength == pytest.approx(r.shadow_strength, abs=2e-4)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_texture_poverty_matches(runs, name):
    st, r = runs[name]
    assert st.evidence.texture_poverty == pytest.approx(r.texture_poverty, abs=2e-4)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_scene_complexity_matches(runs, name):
    st, r = runs[name]
    assert st.scene_complexity == pytest.approx(r.scene_complexity, abs=2e-4)


# --- 잡음 스윕: texture_poverty 의 잡음 보정 가지 -----------------------------

@pytest.mark.parametrize("sigma", [0.0, 2.0, 4.41, 6.0, 12.0, 25.0])
def test_texture_poverty_noise_branch(sigma):
    """임계 = max(6, 2*0.68*sigma). sigma=4.41 부근에서 분기가 바뀐다.

    한쪽 가지만 밟으면 잡음 보정을 통째로 지워도 통과한다.
    """
    im = img_checker(period=3, lo=100, hi=140)   # 그래디언트가 임계 근처
    qc, qr = quality(brightness=0.4, noise_sigma=sigma)
    st = analyzer().update(im, qc, 1.0)
    assert st.evidence.texture_poverty == pytest.approx(
        ref.estimate_texture_poverty(ref.bgr_to_gray_u8(im).astype(np.float64), sigma),
        abs=2e-4)


def test_texture_poverty_threshold_actually_moves():
    """위 스윕이 진짜로 두 가지를 다 밟는지 확인한다."""
    im = img_checker(period=3, lo=100, hi=140)
    vals = []
    for sigma in (0.0, 4.41, 25.0):
        qc, _ = quality(brightness=0.4, noise_sigma=sigma)
        vals.append(analyzer().update(im, qc, 1.0).evidence.texture_poverty)
    assert vals[0] == pytest.approx(vals[1], abs=1e-6), "sigma<4.41 은 바닥 6 에 걸려야 한다"
    assert vals[2] > vals[0] + 0.2, "큰 잡음에서 임계가 올라가지 않았다"


# --- 품질 유래 채널 ----------------------------------------------------------

@pytest.mark.parametrize("brightness,sigma", [(0.02, 0.0), (0.10, 0.0), (0.18, 0.0),
                                              (0.30, 0.0), (0.30, 6.0), (0.30, 20.0),
                                              (0.05, 15.0)])
def test_quality_derived_channels(brightness, sigma):
    im = IMAGES["checker"]
    qc, qr = quality(brightness=brightness, noise_sigma=sigma,
                     blur_free=0.7, noise_free=0.55, occlusion_free=0.6)
    st = analyzer().update(im, qc, 1.0)
    r = ref.compute_cues(im, qr)
    assert st.evidence.darkness == pytest.approx(r.darkness, abs=2e-4)
    assert st.evidence.motion_blur == pytest.approx(r.motion_blur, abs=2e-4)
    assert st.evidence.noise == pytest.approx(r.noise, abs=2e-4)
    assert st.evidence.lens_dirt == pytest.approx(r.lens_dirt, abs=2e-4)
    assert st.evidence.water_drop == pytest.approx(r.water_drop, abs=2e-4)


def test_darkness_is_noisy_or_not_weighted_sum():
    """잡음 0 에서도 Dark 라벨 임계(0.75)에 도달할 수 있어야 한다.

    가중합 0.7*lum + 0.3*noise*lum 이면 상한이 0.7 이라 도달 불가능이었다.
    """
    qc, _ = quality(brightness=0.0, noise_sigma=0.0)
    st = analyzer().update(IMAGES["checker"], qc, 1.0)
    assert st.evidence.darkness > 0.99


# --- 이미지 경로의 그레이 변환 -----------------------------------------------

def test_gray_conversion_matches_opencv_fixed_point():
    """BGR2GRAY 정수 경로. 실수 계수로 계산하면 1 gray 씩 어긋난다.

    그 1 이 estimateShadow 의 8 gray 폭 빈을 바꾼다.
    """
    r = _rng(11)
    bgr = r.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    ours = ref.bgr_to_gray_u8(bgr).astype(np.int32)
    naive = np.rint(0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1]
                    + 0.299 * bgr[:, :, 2]).astype(np.int32)
    # 두 방식이 다른 픽셀이 실제로 존재한다 - 그래서 정수식이 필요하다
    assert np.any(ours != naive)
    # 그리고 우리 쪽이 C++ 과 맞는다는 것은 shadow/backlight 차분이 증명한다
    assert np.abs(ours - naive).max() <= 1


# --- 그레이스케일 입력 경로 --------------------------------------------------

def test_gray_input_path():
    """rgb 가 비면 gray 경로. haze/specular/indoorness 는 계산되지 않는다."""
    g = ref.bgr_to_gray_u8(IMAGES["backlit"])
    qc, qr = quality(brightness=0.3, noise_sigma=0.0)
    st = analyzer().update(g, qc, 1.0)
    r = ref.compute_cues(g, qr)
    assert st.evidence.backlight == pytest.approx(r.backlight, abs=2e-4)
    assert st.evidence.texture_poverty == pytest.approx(r.texture_poverty, abs=2e-4)
    assert st.evidence.shadow_strength == pytest.approx(r.shadow_strength, abs=2e-4)


# --- dcp_patch 스윕 ----------------------------------------------------------

# 패치 최소값이 0.55*255=140 을 넘으면 mean_dark/0.55 가 1 로 포화해 패치
# 크기가 결과를 못 바꾼다. 어두운 쪽이 남도록 옅은 안개 + 어두운 바탕을 쓴다.
_HAZE_IM = img_hazy(img_checker(period=6, lo=5, hi=90), t=0.75, air=200)


@pytest.mark.parametrize("patch", [3, 5, 9, 15])
def test_haze_patch_sweep(patch):
    qc, _ = quality(brightness=0.6)
    st = analyzer(dcp_patch=patch).update(_HAZE_IM, qc, 1.0)
    assert st.evidence.haze == pytest.approx(ref.estimate_haze(_HAZE_IM, patch), abs=2e-4)


def test_haze_patch_sweep_discriminates():
    vals = [ref.estimate_haze(_HAZE_IM, p) for p in (3, 5, 9, 15)]
    assert max(vals) - min(vals) > 0.02, f"패치가 haze 를 바꾸지 못함: {vals}"


# --- 방향 히스토그램: 같은 방향은 같은 빈에 들어가야 한다 ---------------------

@pytest.mark.parametrize("angle", [0, 90, 45, 135])
def test_single_orientation_has_zero_complexity(angle):
    """한 방향뿐인 장면의 방향 엔트로피는 0 이다.

    두 가지가 이 값을 부풀리고 있었다.
      - atan2 를 float 로 계산: 정확히 수평인 엣지가 그래디언트 부호에 따라
        bin 8 / bin 9 로 갈라졌다 (같은 방향인데).
      - 빈 인덱스를 wrap 이 아니라 clamp: 방향 0 과 방향 pi 가 bin 0 / bin 17
        로 갈라졌다.
    합쳐서 세로 줄무늬가 0.24, 체커가 0.58 을 보고했다.
    """
    im = img_stripes(angle, period=16, lo=30, hi=220)
    qc, _ = quality(brightness=0.5)
    st = analyzer().update(im, qc, 1.0)
    g = ref.bgr_to_gray_u8(im).astype(np.float64)
    assert st.scene_complexity == pytest.approx(ref.estimate_scene_complexity(g), abs=2e-4)
    # 계단 픽셀이 섞이는 45/135 도는 완전한 0 이 될 수 없다. 축 정렬은 정확히 0.
    limit = 0.02 if angle in (0, 90) else 0.35
    assert st.scene_complexity < limit, f"{angle}도 단일 방향인데 {st.scene_complexity:.4f}"


def test_complexity_still_separates_one_direction_from_many():
    """위 검사가 '전부 0' 으로 통과하지 않는지 확인한다 (10.4)."""
    qc, _ = quality(brightness=0.5)
    one = analyzer().update(img_stripes(0, period=16), qc, 1.0).scene_complexity
    many = analyzer().update(img_noise(seed=7), qc, 1.0).scene_complexity
    assert many - one > 0.5, f"단일방향 {one:.3f} vs 잡음 {many:.3f}"


# --- 간접 확인: 라벨/가중치가 이 증거에 실제로 반응하는지 ---------------------

def test_evidence_reaches_tier_weights():
    """영상만 바꿔 tier 가중치가 달라져야 한다. 안 달라지면 이 경로는 죽은 채널이다."""
    qc, _ = quality(brightness=0.5)
    a1 = analyzer().update(IMAGES["checker"], qc, 1.0)
    w1 = (a1.tier.photometric, a1.tier.constellation, a1.tier.structural)
    a2 = analyzer().update(IMAGES["flat"], qc, 1.0)
    w2 = (a2.tier.photometric, a2.tier.constellation, a2.tier.structural)
    assert max(abs(x - y) for x, y in zip(w1, w2)) > 0.02, f"{w1} vs {w2}"
