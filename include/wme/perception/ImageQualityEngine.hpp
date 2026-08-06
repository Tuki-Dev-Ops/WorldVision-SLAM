#pragma once

// 매 프레임 영상 품질을 평가한다. 하위 해상도에서 계산해 60 FPS 예산 안에 들어간다.
// 스레드 안전성: 인스턴스 단위 비공유(호출 스레드 전용). 내부 버퍼를 재사용한다.

#include "wme/core/Frame.hpp"
#include "wme/perception/ImageQuality.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace wme {

struct ImageQualityConfig {
    int  analysis_width{320};      // 분석 해상도 (긴 변). 품질 지표는 저해상도로 충분
    int  weight_map_width{160};    // 가중 맵 해상도. ECDA 에서 업샘플해 사용
    int  tile_grid{8};             // 국소 통계 타일 수 (tile_grid x tile_grid)

    double noise_sigma_max{25.0};  // 정규화 기준 (8bit)
    double blur_extent_max{12.0};  // 정규화 기준 (px)
    double sharpness_ref{900.0};   // 그래디언트 에너지 정규화 기준

    double sat_high_thresh{250.0};
    double sat_low_thresh{5.0};

    // 종합 점수 가중치 (합 1.0)
    double w_sharpness{0.30};
    double w_exposure{0.20};
    double w_contrast{0.15};
    double w_noise{0.20};
    double w_occlusion{0.15};
};

class ImageQualityEngine {
public:
    explicit ImageQualityEngine(ImageQualityConfig config = {});

    // 프레임 품질 평가. gray 가 비어 있으면 rgb 로부터 생성한다.
    [[nodiscard]] ImageQuality evaluate(const Frame& frame);

    // 렌즈 오염 누적 상태 초기화 (센서 교체/청소 시)
    void resetLensModel();

    // 지속적 저그래디언트 영역 = 렌즈 오염 후보. 0..1 맵.
    [[nodiscard]] const cv::Mat& lensOcclusionMap() const { return lens_occlusion_; }

    [[nodiscard]] const ImageQualityConfig& config() const { return cfg_; }

private:
    // Immerkaer 라플라시안 마스크 기반 노이즈 표준편차 추정.
    // 응답 버퍼와 중앙값 버퍼를 멤버로 재사용하므로 static 이 아니다
    // (프레임당 307 KB 짜리 reserve 가 여기 있었다).
    [[nodiscard]] double estimateNoiseSigma(const cv::Mat& gray32f);

    // 그래디언트 방향 이방성으로 모션블러 길이/방향 추정
    void estimateBlur(const cv::Mat& gx, const cv::Mat& gy, ImageQuality& out) const;

    // 노출/대비/동적범위를 히스토그램에서 산출
    void analyzeHistogram(const cv::Mat& gray8u, ImageQuality& out) const;

    // 타일별 정보량으로 픽셀 가중 맵 구성
    void buildWeightMap(const cv::Mat& gray32f, const cv::Mat& grad_mag, ImageQuality& out);

    // 프레임 간 그래디언트 최솟값을 누적해 렌즈 오염 영역을 분리
    void updateLensModel(const cv::Mat& grad_mag_small);

    ImageQualityConfig cfg_;

    // 재사용 버퍼 (핫패스 할당 제거)
    cv::Mat small_gray_, small_f_, gx_, gy_, mag_, lap_;
    cv::Mat lens_occlusion_;      // CV_32F, 0..1
    cv::Mat lens_accum_;          // CV_32F, 그래디언트 지수이동 최솟값
    std::uint64_t lens_frames_{0};

    // 중앙값/가중맵 계산용 재사용 버퍼. 매 프레임 새로 잡지 않는다.
    cv::Mat            noise_response_;   // 라플라시안 응답
    std::vector<float> noise_vals_;       // nth_element 대상
    std::vector<float> lens_vals_;        // 렌즈 누적맵의 중앙값 계산용
    cv::Mat            weight_scratch_;   // 가중맵 원본 해상도 작업 버퍼
    cv::Mat            occ_scratch_;      // 렌즈 오염맵을 가중맵 크기로 리샘플
};

}  // namespace wme
