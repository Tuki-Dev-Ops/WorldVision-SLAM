#pragma once

// 환경 상태 기술자.
// 설계 원칙: 라벨은 결과이지 표현이 아니다. 내부 표현은 항상 연속 확률이고,
// 이산 라벨은 히스테리시스를 거친 파생물이다. "밤/낮" 같은 이진 판정이
// 프레임마다 뒤집히면 하위 엔진의 가중치가 진동하기 때문이다.

#include "wme/core/Types.hpp"

#include <array>
#include <string>
#include <string_view>

namespace wme {

enum class Lighting : std::uint8_t {
    Bright, Normal, LowLight, Dark, Backlit, HighDynamic, Unknown
};

enum class Weather : std::uint8_t {
    Clear, Rain, Snow, Fog, Smoke, Dust, Unknown
};

enum class SceneKind : std::uint8_t { Indoor, Outdoor, Unknown };

[[nodiscard]] std::string_view toString(Lighting v) noexcept;
[[nodiscard]] std::string_view toString(Weather v) noexcept;
[[nodiscard]] std::string_view toString(SceneKind v) noexcept;

// 각 현상의 연속 강도 (0 = 없음, 1 = 극심)
struct EnvironmentEvidence {
    double darkness{0.0};        // 저조도 정도
    double backlight{0.0};       // 역광 정도
    double haze{0.0};            // 안개/연무 (dark channel prior)
    double rain_streak{0.0};     // 비 줄무늬 시간적 증거
    double snow_particle{0.0};   // 눈 입자 시간적 증거
    double dust{0.0};            // 먼지 (저채도 + 균일 산란)
    double motion_blur{0.0};     // 모션 블러
    double lens_dirt{0.0};       // 렌즈 오염 (지속적 저그래디언트 영역)
    double water_drop{0.0};      // 물방울 (지속적 원형 왜곡 영역)
    double specular{0.0};        // 반사 바닥/젖은 노면
    double shadow_strength{0.0}; // 강한 그림자
    double camera_shake{0.0};    // 프레임 간 전역 이동량
    double noise{0.0};           // 센서 노이즈
    double texture_poverty{0.0}; // 텍스처 빈곤 (직접정렬 퇴화 예측)
};

struct EnvironmentState {
    Timestamp stamp{};

    // --- 이산 라벨 (히스테리시스 적용 결과) --------------------------------
    Lighting  lighting{Lighting::Unknown};
    Weather   weather{Weather::Unknown};
    SceneKind scene{SceneKind::Unknown};

    // --- 연속 증거 ---------------------------------------------------------
    EnvironmentEvidence evidence{};

    // --- 파생 지표 (0..1) --------------------------------------------------
    double visibility{1.0};        // 유효 가시거리 정규화
    double camera_health{1.0};     // 렌즈/센서 물리 상태
    double scene_complexity{0.0};  // 구조 복잡도 (엣지 엔트로피)
    double dynamic_level{0.0};     // 동적 객체 점유 비율 (Tracking 이 주입)
    double sensor_reliability{1.0};// 종합 센서 신뢰도

    // --- 3-tier 정보 가중치 alpha_k(E) -------------------------------------
    // docs/02-correspondence-problem.md 5장의 융합 규칙. 이 배열이 적응 로직의
    // 유일한 출력 채널이며, 다른 엔진은 라벨이 아니라 이 값을 읽는다.
    struct TierWeights {
        double photometric{1.0};   // alpha_0 : ECDA
        double constellation{1.0}; // alpha_1 : TCG
        double structural{1.0};    // alpha_2 : SPA
        double motion_prior{0.1};  // 운동 사전분포 의존도
    } tier;

    // --- 시간 정책 ---------------------------------------------------------
    // 환경이 나쁠수록 관측을 오래 기억하고 늦게 버린다.
    double memory_retention_scale{1.0};  // 메모리 보존 기간 배율
    double track_persistence_scale{1.0}; // 미관측 트랙 유지 배율
    double detection_threshold_scale{1.0};// YOLO conf 임계 배율

    [[nodiscard]] bool isDegraded() const { return sensor_reliability < 0.6; }
    [[nodiscard]] std::string summary() const;
};

// 증거 -> tier 가중치 + 시간 정책.
// 영상과 무관한 순수 함수로 분리했다. WME 적응 로직의 전부가 여기 있고,
// 야간/우천/안개별 분기 코드는 존재하지 않는다. 열화는 각 정보원이 기여하는
// 정보량의 감소로만 표현된다. 순수 함수라서 Python 참조 구현과 차등 검증이 된다.
// 라벨/장면복잡도/동적비율은 채우지 않는다 (영상이 필요한 항목).
[[nodiscard]] EnvironmentState deriveAdaptationFrom(const EnvironmentEvidence& ev);

}  // namespace wme
