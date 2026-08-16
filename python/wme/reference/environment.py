"""환경 -> tier 가중치 매핑 참조 구현.

C++ EnvironmentAnalyzer::deriveAdaptation() 과 동일한 식.
이 함수가 WME 적응 로직의 전부다. 야간/우천/안개별 분기 코드는 존재하지 않으며,
열화는 각 정보원이 기여하는 정보량의 감소로만 표현된다.

    Lambda_total = a0(E) * Lambda_ECDA + a1(E) * Lambda_TCG + a2(E) * Lambda_SPA + prior

docs/02-correspondence-problem.md 5장 참조.
"""

from __future__ import annotations

from dataclasses import dataclass


def _c01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


@dataclass
class Evidence:
    """각 현상의 연속 강도 (0 = 없음, 1 = 극심)."""
    darkness: float = 0.0
    backlight: float = 0.0
    haze: float = 0.0
    rain_streak: float = 0.0
    snow_particle: float = 0.0
    dust: float = 0.0
    motion_blur: float = 0.0
    lens_dirt: float = 0.0
    water_drop: float = 0.0
    specular: float = 0.0
    shadow_strength: float = 0.0
    camera_shake: float = 0.0
    noise: float = 0.0
    texture_poverty: float = 0.0


@dataclass
class Adaptation:
    visibility: float
    camera_health: float
    sensor_reliability: float
    alpha_photometric: float        # a0 : ECDA
    alpha_constellation: float      # a1 : TCG
    alpha_structural: float         # a2 : SPA
    motion_prior: float
    memory_retention_scale: float
    track_persistence_scale: float
    detection_threshold_scale: float


def derive_adaptation(ev: Evidence) -> Adaptation:
    # --- 가시도: 동시 악화를 반영하도록 곱으로 결합 ---
    #
    # rain_streak / snow_particle 의 계수 0.40 / 0.45 는 여전히 손으로 쓴
    # 숫자이고, 실제 비나 눈에 대조된 적이 없다 (docs/03-roadmap.md 의 미해결
    # 항목). 다만 docs/06-results.md 27 절이 잰 바에 따르면 이 저장소가 가진
    # 어떤 데이터에서도 그 계수는 결과를 바꾸지 않는다 - 입력이 되는 두 채널이
    # 움직이는 카메라에서 판정 불가로 기권하기 때문이다(자기운동 게이트).
    # 즉 이 두 계수는 "틀렸다" 가 아니라 "아직 시험된 적이 없다" 이고,
    # 시험하려면 정지 카메라의 실제 강수 영상이 필요하다.
    #
    # 27 절 이전에는 게이트가 없어서 이 두 항이 실내 손들기 시퀀스에서
    # visibility 를 0.61 로 눌렀다 - 강수가 전혀 없는데도.
    visibility = _c01((1.0 - 0.85 * ev.haze)
                      * (1.0 - 0.40 * ev.rain_streak)
                      * (1.0 - 0.45 * ev.snow_particle)
                      * (1.0 - 0.30 * ev.dust)
                      * (1.0 - 0.50 * ev.darkness))

    camera_health = _c01((1.0 - 0.7 * ev.lens_dirt)
                         * (1.0 - 0.5 * ev.water_drop)
                         * (1.0 - 0.4 * ev.noise))

    sensor_reliability = _c01(0.45 * visibility
                              + 0.30 * camera_health
                              + 0.25 * (1.0 - ev.motion_blur))

    # 측광: 블러/안개/저조도/노이즈에 직접 무너진다.
    # texture_poverty 와 camera_shake 도 여기 들어가야 한다 -
    # 텍스처가 없으면 측광 잔차가 포즈를 구속하지 못하고(랭크 부족),
    # 흔들림이 크면 coarse-to-fine 수렴 반경을 넘긴다.
    a0 = _c01((1.0 - 0.90 * ev.motion_blur)
              * (1.0 - 0.85 * ev.haze)
              * (1.0 - 0.85 * ev.texture_poverty)
              * (1.0 - 0.70 * ev.darkness)
              * (1.0 - 0.45 * ev.rain_streak)
              * (1.0 - 0.60 * ev.noise)
              * (1.0 - 0.50 * ev.lens_dirt)
              * (1.0 - 0.35 * ev.camera_shake))

    # 성좌: YOLO 는 저조도/안개에서도 큰 물체를 잡는다. 감쇠가 훨씬 완만하다.
    a1 = _c01(0.35 + 0.65 * (visibility ** 0.45))

    # 구조: 외관과 무관. 텍스처가 빈곤할수록 상대적 중요도가 오히려 올라간다.
    a2 = _c01(0.40 + 0.60 * ev.texture_poverty + 0.30 * (1.0 - a0))

    observability = _c01(0.5 * a0 + 0.3 * a1 + 0.2 * a2)
    motion_prior = _c01(0.05 + 0.95 * (1.0 - observability))

    # 나쁜 환경일수록 오래 기억하고 늦게 버린다
    adversity = 1.0 - sensor_reliability
    memory_scale = 1.0 + 3.0 * adversity
    track_scale = 1.0 + 2.5 * adversity + 1.5 * ev.motion_blur

    det_scale = max(0.35, _c01(1.0 - 0.45 * ev.darkness - 0.25 * ev.haze))

    return Adaptation(visibility, camera_health, sensor_reliability,
                      a0, a1, a2, motion_prior,
                      memory_scale, track_scale, det_scale)


# 문서 5장 표에 대응하는 대표 조건들. 회귀 테스트의 기준점으로 쓴다.
PRESETS: dict[str, Evidence] = {
    "clear_indoor":      Evidence(),
    "night":             Evidence(darkness=0.85, noise=0.5),
    "fog":               Evidence(haze=0.8),
    "heavy_rain":        Evidence(rain_streak=0.7, darkness=0.3),
    "snow":              Evidence(snow_particle=0.7, darkness=0.2),
    "motion_blur_burst": Evidence(motion_blur=0.9, camera_shake=0.8),
    "textureless":       Evidence(texture_poverty=0.95),
    "dirty_lens":        Evidence(lens_dirt=0.6, water_drop=0.4),
    "night_fog_dirty":   Evidence(darkness=0.8, haze=0.7, lens_dirt=0.5, noise=0.5),
}
