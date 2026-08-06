#pragma once

// 프레임별 영상 품질 기술자.
// 스칼라 점수뿐 아니라 픽셀별 가중 맵을 제공하는 것이 핵심 - ECDA 가 이 맵으로
// 측광 잔차를 가중한다. 전역 점수만으로는 부분 과노출/부분 블러를 표현할 수 없다.

#include "wme/core/Types.hpp"

#include <opencv2/core.hpp>

namespace wme {

struct ImageQuality {
    // --- 전역 스칼라 (모두 0..1 정규화, 높을수록 좋음) ---------------------
    double sharpness{0.0};       // 그래디언트 에너지 기반 선명도
    double exposure{0.0};        // 노출 적정도 (과/부족 모두 감점)
    double brightness{0.0};      // 평균 휘도 (0..1, 좋고 나쁨 아닌 원시값)
    double contrast{0.0};        // RMS 대비
    double dynamic_range{0.0};   // 히스토그램 유효 점유폭
    double noise_free{0.0};      // 1 - 정규화 노이즈. 높을수록 깨끗
    double blur_free{0.0};       // 1 - 블러 정도
    double occlusion_free{0.0};  // 1 - 렌즈 가림 비율
    double score{0.0};           // 종합 ImageQualityScore

    // --- 물리 단위 원시값 --------------------------------------------------
    double noise_sigma{0.0};        // Immerkaer 추정 노이즈 표준편차 (0..255)
    double blur_extent_px{0.0};     // 추정 블러 커널 길이 (픽셀)
    double blur_direction_rad{0.0}; // 모션블러 주방향
    double saturated_high_ratio{0.0};
    double saturated_low_ratio{0.0};

    // --- 픽셀별 가중 맵 ----------------------------------------------------
    // CV_32F, 0..1. ECDA 의 w_quality(u) 로 그대로 곱해진다.
    cv::Mat weight_map;

    // 측광 잔차 분산의 *하한* [intensity^2]. 센서 잡음 + 블러 불일치.
    //
    // 이 값을 정보행렬의 분모로 그대로 쓰면 안 된다. TUM 다섯 시퀀스에서
    // ECDA 가 해에서 실제로 남기는 잔차 분산은 이 값의 3~34 배다 (sigma
    // 3.4~8.2 로 추정하는데 실측 잔차 RMS 는 9.6~22.8). 차이는 잡음이 아니라
    // 모델 오차다 - 깊이 오차, 밝기 불변 위배, 리샘플 오차. 그래서
    // DirectAligner 는 해에서 달성된 잔차로 스케일하고 이 값은 바닥으로만 쓴다
    // (InformationModel 주석 참조). 이 항목을 분모로 쓰던 시절의 실측 ANEES 는
    // 5 425 ~ 146 087 이었다 (수용구간 [5.5, 6.5]).
    [[nodiscard]] double photometricVariance() const {
        // 센서 노이즈 + 블러로 인한 모델 불일치를 합산
        const double blur_term = 4.0 * blur_extent_px * blur_extent_px;
        return noise_sigma * noise_sigma + blur_term + 1.0;
    }

    [[nodiscard]] bool usableForPhotometric() const {
        return score > 0.25 && sharpness > 0.15;
    }
};

}  // namespace wme
