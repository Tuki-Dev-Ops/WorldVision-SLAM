#pragma once

// 환경 조건을 지속 추정한다.
// 단일 프레임으로는 비/눈/흔들림을 판정할 수 없으므로 짧은 시간 창을 유지한다.
// 갱신 주기는 5 Hz 로 충분하며(날씨는 16 ms 안에 바뀌지 않는다), 그 사이에는
// 마지막 상태를 그대로 반환한다.

#include "wme/core/Frame.hpp"
#include "wme/perception/EnvironmentState.hpp"
#include "wme/perception/ImageQuality.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace wme {

struct EnvironmentConfig {
    int    analysis_width{192};      // 환경 분석 해상도. 품질보다 더 낮아도 된다
    int    history_size{9};          // 시간적 잔차 계산용 프레임 수 (홀수)
    double update_hz{5.0};           // 재평가 주기

    // 히스테리시스: 라벨 전환에 필요한 상/하 임계. 진동 방지가 목적이다.
    double enter_threshold{0.55};
    double exit_threshold{0.35};
    double evidence_ema{0.25};       // 증거 지수이동평균 계수

    // dark channel prior 패치 크기
    int    dcp_patch{9};

    // 야간 판정 기준 휘도
    double dark_brightness{0.18};
    double bright_brightness{0.55};

    // 이 값 미만의 그래디언트(gray/px)는 포즈를 구속하지 못한다고 본다.
    // DirectAligner::min_gradient 와 같은 기준을 쓴다 - 두 엔진이 "정보가 있는
    // 픽셀" 을 서로 다르게 정의하면 tier 가중치가 실제 가용 정보와 어긋난다.
    double texture_min_gradient{6.0};
};

class EnvironmentAnalyzer {
public:
    explicit EnvironmentAnalyzer(EnvironmentConfig config = {});

    // 프레임과 그 품질을 받아 환경 상태를 갱신한다.
    // update_hz 주기가 아직 안 됐으면 계산을 건너뛰고 직전 상태를 돌려준다.
    const EnvironmentState& update(const Frame& frame, const ImageQuality& quality);

    // Tracking Engine 이 동적 객체 점유율을 주입한다 (0..1).
    void setDynamicLevel(double ratio);

    [[nodiscard]] const EnvironmentState& state() const { return state_; }

    void reset();

private:
    void computeEvidence(const Frame& frame, const ImageQuality& quality,
                         EnvironmentEvidence& ev);

    // 시간적 잔차로 비/눈 입자를 분리한다
    void analyzeTransientParticles(const cv::Mat& gray_f, EnvironmentEvidence& ev);

    // dark channel prior 로 안개/연무 정도를 추정
    [[nodiscard]] double estimateHaze(const cv::Mat& bgr_small) const;

    // 위상상관으로 전역 이동량을 구해 카메라 흔들림 판정
    [[nodiscard]] double estimateShake(const cv::Mat& gray_f);

    [[nodiscard]] double estimateBacklight(const cv::Mat& gray_f) const;
    [[nodiscard]] double estimateSpecular(const cv::Mat& bgr_small) const;
    [[nodiscard]] double estimateShadow(const cv::Mat& gray_f) const;
    // 잡음도 그래디언트를 만든다. 잡음 몫을 임계에 반영해야 잡음이 텍스처로
    // 계산되지 않는다.
    [[nodiscard]] double estimateTexturePoverty(const cv::Mat& gray_f, double noise_sigma) const;
    [[nodiscard]] double estimateSceneComplexity(const cv::Mat& gray_f) const;
    [[nodiscard]] double estimateIndoorness(const cv::Mat& bgr_small) const;

    // 증거 -> 이산 라벨. 히스테리시스 적용.
    void resolveLabels(const ImageQuality& quality);

    // 증거 -> 3-tier 정보 가중치 및 시간 정책
    void deriveAdaptation();

    EnvironmentConfig cfg_;
    EnvironmentState  state_{};

    // 시간 창 링 버퍼. deque + clone() 은 프레임마다 Mat 을 새로 잡았다.
    // 자리를 미리 잡아 두고 copyTo 로 덮어쓴다. 논리적 i 번째(오래된 순)는
    // history_[(history_head_ + i) % history_.size()].
    std::vector<cv::Mat> history_;    // CV_32F 저해상도 그레이
    std::size_t history_head_{0};
    std::size_t history_count_{0};

    cv::Mat  prev_gray_;              // 흔들림 추정용
    cv::Mat  temporal_median_;
    std::vector<float> median_scratch_;   // 픽셀별 시간 중앙값 계산용
    double   indoorness_ema_{0.5};
    Timestamp last_update_{};
    bool     initialized_{false};
};

}  // namespace wme
