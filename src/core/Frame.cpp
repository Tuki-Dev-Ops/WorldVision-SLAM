#include "wme/core/Frame.hpp"

#include <opencv2/imgproc.hpp>

namespace wme {

ImagePyramid ImagePyramid::build(const cv::Mat& gray_u8, const CameraIntrinsics& K, int levels) {
    ImagePyramid p;
    if (gray_u8.empty() || levels < 1) return p;

    p.gray.resize(static_cast<std::size_t>(levels));
    p.grad_x.resize(static_cast<std::size_t>(levels));
    p.grad_y.resize(static_cast<std::size_t>(levels));
    p.grad_mag.resize(static_cast<std::size_t>(levels));
    p.intrinsics.resize(static_cast<std::size_t>(levels));

    gray_u8.convertTo(p.gray[0], CV_32F);
    p.intrinsics[0] = K;

    for (int l = 1; l < levels; ++l) {
        const auto prev = static_cast<std::size_t>(l - 1);
        // 2배 축소. INTER_AREA 는 에일리어싱을 줄여 직접정렬 수렴에 유리하다.
        cv::resize(p.gray[prev], p.gray[static_cast<std::size_t>(l)], cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        p.intrinsics[static_cast<std::size_t>(l)] = K.atPyramidLevel(l);
    }

    for (int l = 0; l < levels; ++l) {
        const auto i = static_cast<std::size_t>(l);
        // Scharr 는 Sobel 보다 회전 대칭성이 좋아 그래디언트 방향 편향이 적다.
        cv::Scharr(p.gray[i], p.grad_x[i], CV_32F, 1, 0, 1.0 / 32.0);
        cv::Scharr(p.gray[i], p.grad_y[i], CV_32F, 0, 1, 1.0 / 32.0);
        cv::magnitude(p.grad_x[i], p.grad_y[i], p.grad_mag[i]);
    }
    return p;
}

}  // namespace wme
