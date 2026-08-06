"""적응 로직 검증.

docs/02-correspondence-problem.md 5장의 주장을 그대로 테스트로 옮긴다.
"환경 열화는 각 정보원이 기여하는 정보량의 감소로만 표현된다"가 성립하는지,
그리고 tier 별 감쇠 속도의 순서 관계가 설계 의도와 맞는지.
"""

import pytest

from wme.reference.environment import PRESETS, Evidence, derive_adaptation


def test_clear_conditions_favor_photometric():
    a = derive_adaptation(PRESETS["clear_indoor"])
    assert a.alpha_photometric == pytest.approx(1.0, abs=1e-9)
    assert a.sensor_reliability > 0.9
    assert a.motion_prior < 0.2


@pytest.mark.parametrize("name", ["night", "fog", "heavy_rain", "motion_blur_burst",
                                  "night_fog_dirty", "dirty_lens"])
def test_degradation_reduces_photometric_weight(name):
    clear = derive_adaptation(PRESETS["clear_indoor"])
    bad = derive_adaptation(PRESETS[name])
    assert bad.alpha_photometric < clear.alpha_photometric, name


@pytest.mark.parametrize("name", ["night", "fog", "heavy_rain", "night_fog_dirty"])
def test_constellation_degrades_more_slowly_than_photometric(name):
    """YOLO 는 텍스처가 사라져도 큰 물체를 잡는다. 이것이 TCG 의 존재 근거다."""
    clear = derive_adaptation(PRESETS["clear_indoor"])
    bad = derive_adaptation(PRESETS[name])

    photo_drop = clear.alpha_photometric - bad.alpha_photometric
    const_drop = clear.alpha_constellation - bad.alpha_constellation
    assert photo_drop > const_drop, f"{name}: photo {photo_drop:.3f} vs const {const_drop:.3f}"


def test_textureless_scene_favors_structural():
    """텍스처 없는 복도에서는 평면 제약이 주 정보원이 되어야 한다."""
    a = derive_adaptation(PRESETS["textureless"])
    assert a.alpha_structural > a.alpha_photometric
    assert a.alpha_structural > 0.9


def test_motion_blur_collapses_photometric_hardest():
    """블러는 측광 잔차를 가장 직접적으로 무너뜨린다."""
    blur = derive_adaptation(PRESETS["motion_blur_burst"])
    assert blur.alpha_photometric < 0.15
    assert blur.motion_prior > 0.5
    assert blur.track_persistence_scale > 2.0


def test_combined_degradation_is_worse_than_any_single():
    """야간 + 안개 + 렌즈오염이 각각보다 나빠야 한다. 분기 코드 없이도 성립한다."""
    combined = derive_adaptation(PRESETS["night_fog_dirty"])
    for single in ["night", "fog", "dirty_lens"]:
        assert combined.alpha_photometric <= derive_adaptation(PRESETS[single]).alpha_photometric


def test_adversity_extends_memory_and_tracking():
    clear = derive_adaptation(PRESETS["clear_indoor"])
    for name in ["night", "fog", "night_fog_dirty"]:
        bad = derive_adaptation(PRESETS[name])
        assert bad.memory_retention_scale > clear.memory_retention_scale, name
        assert bad.track_persistence_scale > clear.track_persistence_scale, name


def test_darkness_lowers_detection_threshold_with_a_floor():
    """어두우면 임계를 낮추되 바닥은 지켜야 한다. 무한정 낮추면 오검출이 폭증한다."""
    clear = derive_adaptation(PRESETS["clear_indoor"])
    night = derive_adaptation(PRESETS["night"])
    assert night.detection_threshold_scale < clear.detection_threshold_scale
    assert night.detection_threshold_scale >= 0.35

    extreme = derive_adaptation(Evidence(darkness=1.0, haze=1.0))
    assert extreme.detection_threshold_scale >= 0.35


def test_all_outputs_stay_in_valid_range():
    """어떤 증거 조합에서도 가중치가 [0, 1] 을 벗어나면 안 된다."""
    import itertools

    for combo in itertools.product([0.0, 0.5, 1.0], repeat=4):
        ev = Evidence(darkness=combo[0], haze=combo[1],
                      motion_blur=combo[2], noise=combo[3])
        a = derive_adaptation(ev)
        for name in ("visibility", "camera_health", "sensor_reliability",
                     "alpha_photometric", "alpha_constellation",
                     "alpha_structural", "motion_prior"):
            v = getattr(a, name)
            assert 0.0 <= v <= 1.0, f"{name}={v} for {combo}"


def test_monotonic_in_each_evidence_channel():
    """단일 채널을 악화시키면 측광 가중이 단조 감소해야 한다."""
    for channel in ["darkness", "haze", "motion_blur", "noise", "lens_dirt", "rain_streak"]:
        prev = 2.0
        for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
            a = derive_adaptation(Evidence(**{channel: level}))
            assert a.alpha_photometric <= prev + 1e-12, f"{channel} at {level}"
            prev = a.alpha_photometric


def test_information_never_fully_vanishes():
    """모든 tier 가 0 이 되면 추정기가 발산한다. 최소 정보는 남아야 한다."""
    worst = derive_adaptation(Evidence(darkness=1.0, haze=1.0, motion_blur=1.0,
                                       noise=1.0, lens_dirt=1.0, rain_streak=1.0,
                                       snow_particle=1.0, dust=1.0))
    assert worst.alpha_constellation > 0.3
    assert worst.alpha_structural > 0.3
    assert worst.motion_prior > 0.5, "관측이 없으면 운동 사전분포에 의존해야 한다"
