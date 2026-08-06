#pragma once

// 스테레오 깊이 프런트엔드.
//
// TUM 은 RGB-D 라 깊이가 센서에서 온다. EuRoC/KITTI 는 스테레오라 깊이를
// 우리가 만들어야 하고, 그 순간 깊이는 "측정값" 이 아니라 "추정값" 이 된다.
// DirectAligner 는 깊이를 참으로 믿고 3D 점을 만들므로, 이 경계를 흐리면
// 스테레오에서의 실패가 정렬기의 실패로 잘못 읽힌다. 그래서 이 클래스는
// 깊이와 함께 **어느 픽셀이 유효한가** 를 반드시 같이 낸다.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace wme {

struct StereoDepthConfig {
    // SGBM. 값은 KITTI 해상도(1241x376)를 기준으로 잡았다.
    int  min_disparity{0};
    int  num_disparities{128};   // 16 의 배수여야 한다
    int  block_size{5};          // 홀수
    int  p1_factor{8};           // P1 = p1_factor * ch * block^2
    int  p2_factor{32};
    int  disp12_max_diff{1};     // 좌우 일관성. -1 이면 검사 안 함
    int  pre_filter_cap{31};
    int  uniqueness_ratio{10};
    int  speckle_window_size{100};
    int  speckle_range{2};
    bool mode_hh{false};         // MODE_HH 는 정확하지만 메모리를 크게 쓴다

    // 기하 유효성
    double baseline_m{0.537};    // KITTI gray 스테레오 기본
    double focal_px{718.856};
    double min_depth_m{0.5};
    double max_depth_m{80.0};

    // 시차가 작을수록 깊이 오차가 제곱으로 커진다. dZ = Z^2/(f*B) * dd 이므로
    // 허용 깊이오차를 정하면 최소 시차가 정해진다. 이 쪽으로 자르는 편이
    // max_depth 로 자르는 것보다 정직하다 - 같은 말인데 근거가 보인다.
    double disparity_noise_px{0.5};
    double max_depth_sigma_m{2.0};
};

struct StereoDepthResult {
    cv::Mat depth;        // CV_32F, 미터. 무효 픽셀은 0
    cv::Mat disparity;    // CV_32F, 픽셀. 무효는 음수
    double  valid_ratio{0.0};
    double  median_depth_m{0.0};
    double  min_valid_disparity{0.0};   // 위 오차 예산에서 유도된 값

    // 탐색 범위가 표현할 수 있는 가장 가까운 거리 = f*B/(min_disp+num_disp-1).
    // 이보다 가까운 표면은 SGBM 이 "못 찾았다" 고 하지 않는다 - 범위 안에서
    // 가장 그럴듯한 값을 돌려준다. 그래서 조용히 **틀린 깊이** 가 나온다.
    double  min_representable_depth_m{0.0};
    // 시차 상한에 붙어 잘려 나간 화소 비율. 이 값이 크면 num_disparities 가
    // 이 (베이스라인, 최근접거리) 조합에 대해 부족하다는 뜻이다.
    double  clipped_ratio{0.0};
};

class StereoDepth {
public:
    explicit StereoDepth(StereoDepthConfig config = {});

    // left/right 는 같은 크기의 CV_8U (그레이) 또는 CV_8UC3.
    // 정렬(rectified)되어 있다고 가정한다 - KITTI/EuRoC 는 이미 그렇다.
    [[nodiscard]] StereoDepthResult compute(const cv::Mat& left, const cv::Mat& right);

    [[nodiscard]] const StereoDepthConfig& config() const { return cfg_; }

    // Z = f*B/d 의 미분에서 나온 시차 하한. 설정만으로 결정된다.
    [[nodiscard]] static double minValidDisparity(const StereoDepthConfig& c);

    // 최근접 거리 z 까지 표현하려면 시차 범위가 얼마나 필요한가 (16 의 배수로 올림).
    // 이 값을 안 맞추면 실패가 조용하다 - 가까운 표면이 무효가 아니라 틀린
    // 깊이로 나온다.
    [[nodiscard]] static int requiredDisparities(double focal_px, double baseline_m,
                                                 double min_depth_m);

private:
    StereoDepthConfig cfg_;
    cv::Ptr<cv::StereoSGBM> sgbm_;
};

}  // namespace wme
