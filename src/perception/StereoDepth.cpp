#include "wme/perception/StereoDepth.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace wme {

double StereoDepth::minValidDisparity(const StereoDepthConfig& c) {
    // Z = f*B/d  ->  |dZ/dd| = f*B/d^2 = Z^2/(f*B)
    // sigma_Z = Z^2/(f*B) * sigma_d 를 max_depth_sigma_m 이하로 유지하려면
    //   d >= sqrt(f*B*sigma_d / sigma_Z)
    const double fb = c.focal_px * c.baseline_m;
    if (fb <= 0.0 || c.max_depth_sigma_m <= 0.0) return 1.0;
    const double d_err = std::sqrt(fb * c.disparity_noise_px / c.max_depth_sigma_m);
    // max_depth_m 로도 잘린다. 둘 중 엄격한 쪽.
    const double d_far = (c.max_depth_m > 0.0) ? fb / c.max_depth_m : 0.0;
    return std::max({d_err, d_far, 1.0 / 16.0});
}

int StereoDepth::requiredDisparities(double focal_px, double baseline_m, double min_depth_m) {
    if (min_depth_m <= 0.0) return 16;
    const double d = focal_px * baseline_m / min_depth_m;
    const int need = static_cast<int>(std::ceil((d + 1.0) / 16.0)) * 16;
    return std::max(16, need);
}

StereoDepth::StereoDepth(StereoDepthConfig config) : cfg_(config) {
    if (cfg_.num_disparities % 16 != 0 || cfg_.num_disparities <= 0) {
        throw std::invalid_argument("num_disparities 는 양의 16 배수여야 한다");
    }
    if (cfg_.block_size % 2 == 0) ++cfg_.block_size;

    const int ch = 1;   // 항상 그레이로 넘긴다
    const int b2 = cfg_.block_size * cfg_.block_size;
    sgbm_ = cv::StereoSGBM::create(
        cfg_.min_disparity, cfg_.num_disparities, cfg_.block_size,
        cfg_.p1_factor * ch * b2, cfg_.p2_factor * ch * b2,
        cfg_.disp12_max_diff, cfg_.pre_filter_cap, cfg_.uniqueness_ratio,
        cfg_.speckle_window_size, cfg_.speckle_range,
        cfg_.mode_hh ? cv::StereoSGBM::MODE_HH : cv::StereoSGBM::MODE_SGBM_3WAY);
}

StereoDepthResult StereoDepth::compute(const cv::Mat& left, const cv::Mat& right) {
    StereoDepthResult out;
    if (left.empty() || right.empty() || left.size() != right.size()) return out;

    cv::Mat l = left, r = right;
    if (l.channels() == 3) cv::cvtColor(left, l, cv::COLOR_BGR2GRAY);
    if (r.channels() == 3) cv::cvtColor(right, r, cv::COLOR_BGR2GRAY);

    cv::Mat d16;
    sgbm_->compute(l, r, d16);          // CV_16S, 시차 * 16

    cv::Mat disp;
    d16.convertTo(disp, CV_32F, 1.0 / 16.0);

    const double d_min = minValidDisparity(cfg_);
    out.min_valid_disparity = d_min;

    const double fb = cfg_.focal_px * cfg_.baseline_m;
    out.depth = cv::Mat::zeros(disp.size(), CV_32F);
    out.disparity = cv::Mat(disp.size(), CV_32F, cv::Scalar(-1.0f));

    // 탐색 상한. 이 값에 붙은 시차는 "정답이 범위 밖" 일 수 있으므로 믿지 않는다.
    const double d_ceil = static_cast<double>(cfg_.min_disparity + cfg_.num_disparities) - 1.0;
    out.min_representable_depth_m = (d_ceil > 0.0) ? fb / d_ceil : 0.0;

    std::vector<float> valid;
    valid.reserve(static_cast<std::size_t>(disp.total()) / 4);
    std::size_t clipped = 0;

    for (int y = 0; y < disp.rows; ++y) {
        const float* dp = disp.ptr<float>(y);
        float* zp = out.depth.ptr<float>(y);
        float* op = out.disparity.ptr<float>(y);
        for (int x = 0; x < disp.cols; ++x) {
            const double dd = dp[x];
            // SGBM 은 매칭 실패를 (min_disparity-1)*16 으로 표시한다.
            // convertTo 후에는 음수이거나 아주 작은 값이 된다.
            if (!(dd > d_min)) continue;
            // 상한에 붙은 화소. SGBM 은 범위 밖을 "실패" 로 보고하지 않고
            // 범위 안 최선을 돌려준다 - 즉 조용히 틀린 깊이가 된다.
            // 실측: TUM(1~5 m)에 KITTI 베이스라인 0.54 m 를 쓰면 참 시차가
            // 140 px 인데 범위는 128 px 이라, 중앙 스케일이 2.34 배로 나왔다.
            if (dd >= d_ceil) { ++clipped; continue; }
            const double z = fb / dd;
            if (z < cfg_.min_depth_m || z > cfg_.max_depth_m) continue;
            zp[x] = static_cast<float>(z);
            op[x] = static_cast<float>(dd);
            valid.push_back(static_cast<float>(z));
        }
    }
    out.clipped_ratio = static_cast<double>(clipped) /
                        static_cast<double>(std::max<std::size_t>(1, disp.total()));

    out.valid_ratio = static_cast<double>(valid.size()) /
                      static_cast<double>(std::max<std::size_t>(1, disp.total()));
    if (!valid.empty()) {
        const std::size_t mid = valid.size() / 2;
        std::nth_element(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(mid),
                         valid.end());
        out.median_depth_m = valid[mid];
    }
    return out;
}

}  // namespace wme
