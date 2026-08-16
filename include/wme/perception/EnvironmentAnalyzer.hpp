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
#include <limits>
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

    // 비/눈 판정에 시간 중앙값을 쓸 수 있는 한계 이동량 (분석 해상도 px).
    // 화소별 중앙값은 그 화소가 창 내내 같은 장면점을 보고 있어야 배경 모형이
    // 되므로, 표본 간격인 1 px 가 그 한계다. 튜닝값이 아니라 통계의 단위다.
    // 무한으로 두면 게이트가 항상 열려 06-results.md 27 절 이전 동작이 된다 -
    // 그 비교를 하기 위한 손잡이이지 운영값이 아니다.
    double particle_max_window_shift_px{1.0};

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

    // 위상상관이 준 프레임 간 전역 이동량 (분석 해상도의 px). estimateShake 가
    // 0..1 로 눌러 버리기 전의 원값이라, 시간 중앙값이 유효한지 판정할 수 있다.
    // 초기값은 0 이 아니라 무한이다 - 아직 재지 않은 것을 "카메라가 서 있다" 로
    // 읽으면 게이트가 열린 채로 시작한다.
    double last_shift_px_{std::numeric_limits<double>::infinity()};
    // 직전 프레임이 있었는가 = 위 이동량이 실재하는 전이인가. 첫 프레임의
    // "해당 없음" 을 이동량 0/무한 어느 쪽으로도 접지 않기 위한 구별이다.
    bool   last_shift_valid_{false};
    // 창 안의 **전이** 이동량 링 (프레임 N 개면 전이는 N-1 개).
    // 합이 창 안의 경로길이가 되고, 그것이 알짜 변위의 상한이다.
    std::vector<double> shift_hist_;
    std::size_t shift_head_{0};
    std::size_t shift_count_{0};
    double   indoorness_ema_{0.5};
    Timestamp last_update_{};
    bool     initialized_{false};
};

}  // namespace wme
