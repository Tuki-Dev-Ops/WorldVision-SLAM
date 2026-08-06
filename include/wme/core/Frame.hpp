#pragma once

// 한 시각의 센서 입력 묶음. 파이프라인 전체가 이 단위로 흐른다.
// 이미지 버퍼는 cv::Mat 의 참조 카운팅에 의존하므로 복사는 얕은 복사다.

#include "wme/core/Types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace wme {

// 직접정렬(ECDA)용 다중 해상도 피라미드. 그레이/그래디언트를 함께 보관한다.
struct ImagePyramid {
    std::vector<cv::Mat> gray;     // CV_32F, 0..255
    std::vector<cv::Mat> grad_x;   // CV_32F
    std::vector<cv::Mat> grad_y;   // CV_32F
    std::vector<cv::Mat> grad_mag; // CV_32F
    std::vector<CameraIntrinsics> intrinsics;

    [[nodiscard]] int levels() const { return static_cast<int>(gray.size()); }
    [[nodiscard]] bool empty() const { return gray.empty(); }

    // 원본 그레이 영상으로부터 피라미드 구축
    static ImagePyramid build(const cv::Mat& gray_u8, const CameraIntrinsics& K, int levels);
};

struct Frame {
    FrameId   id{};
    Timestamp stamp{};

    cv::Mat rgb;      // CV_8UC3, BGR
    cv::Mat gray;     // CV_8UC1
    cv::Mat depth;    // CV_32F, 미터 단위. 없으면 empty
    cv::Mat right;    // 스테레오 우영상. 없으면 empty

    CameraIntrinsics intrinsics{};
    SensorKind       sensor{SensorKind::Unknown};

    ImagePyramid pyramid;

    // 정적 픽셀 마스크. 0 = 동적 객체이므로 측광 항에서 제외.
    // Tracking Engine 이 토큰 마스크로부터 매 프레임 갱신한다.
    cv::Mat static_mask;  // CV_8UC1

    [[nodiscard]] bool hasDepth() const { return !depth.empty(); }
    [[nodiscard]] bool hasStereo() const { return !right.empty(); }
    [[nodiscard]] bool valid() const { return !gray.empty() && intrinsics.valid() && stamp.valid(); }
};

using FramePtr      = std::shared_ptr<Frame>;
using ConstFramePtr = std::shared_ptr<const Frame>;

}  // namespace wme
