// StereoDepth 검증.
//
// 이 테스트가 지키려는 것은 하나다: **오른쪽 영상을 만든 깊이와, 그
// 영상에서 되찾은 깊이가 같은가**. 자명해 보이지만 여기서 틀릴 수 있는
// 지점이 셋이다 - SGBM 의 16 배 고정소수점, f*B 곱, 그리고 무효 픽셀 표기.
//
// 순환성을 피하려고 (f, B) 를 스윕한다. 오른쪽 영상은 참 깊이와 참 (f,B)
// 로 만들고, StereoDepth 에는 같은 (f,B) 를 주되 그 값을 테스트마다 바꾼다.
// f*B 가 코드 안에서 상수로 굳어 있으면 스윕의 한 점에서만 맞는다.

#include <gtest/gtest.h>

#include "wme/perception/StereoDepth.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

using namespace wme;

namespace {

constexpr int kW = 640;
constexpr int kH = 480;

// 재현 가능한 고주파 텍스처. SGBM 은 텍스처가 없으면 아무것도 못 찾는다.
cv::Mat textureImage(unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    cv::Mat noise(kH, kW, CV_8U);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) noise.at<uchar>(y, x) = static_cast<uchar>(d(rng));
    }
    // 살짝 흐리게 해 블록 매칭이 걸릴 만한 구조를 만든다.
    cv::Mat img;
    cv::GaussianBlur(noise, img, cv::Size(3, 3), 0.8);
    return img;
}

// 시차 맵을 주고 왼쪽 영상을 오른쪽으로 워프한다.
// 정렬 스테레오에서 x_right = x_left - disparity.
cv::Mat warpToRight(const cv::Mat& left, const cv::Mat& disp) {
    cv::Mat mx(left.size(), CV_32F), my(left.size(), CV_32F);
    for (int y = 0; y < left.rows; ++y) {
        for (int x = 0; x < left.cols; ++x) {
            mx.at<float>(y, x) = static_cast<float>(x) + disp.at<float>(y, x);
            my.at<float>(y, x) = static_cast<float>(y);
        }
    }
    cv::Mat right;
    cv::remap(left, right, mx, my, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return right;
}

struct Recovered {
    double median_abs_err{0.0};
    double median_rel_err{0.0};
    double valid_ratio{0.0};
    double median_depth{0.0};
};

Recovered runCase(double focal, double baseline, const cv::Mat& truth_depth) {
    StereoDepthConfig cfg;
    cfg.focal_px = focal;
    cfg.baseline_m = baseline;
    cfg.num_disparities = 128;
    cfg.block_size = 5;
    cfg.max_depth_m = 200.0;
    cfg.max_depth_sigma_m = 100.0;   // 이 테스트에서는 오차예산으로 자르지 않는다

    cv::Mat left = textureImage();
    cv::Mat disp(truth_depth.size(), CV_32F);
    for (int y = 0; y < disp.rows; ++y) {
        for (int x = 0; x < disp.cols; ++x) {
            disp.at<float>(y, x) =
                static_cast<float>(focal * baseline / truth_depth.at<float>(y, x));
        }
    }
    cv::Mat right = warpToRight(left, disp);

    StereoDepth stereo(cfg);
    const StereoDepthResult r = stereo.compute(left, right);

    // 좌우 경계는 시차만큼 잘려 대응이 없다. 안쪽만 본다.
    const int margin = cfg.num_disparities + 8;
    std::vector<double> abs_err, rel_err;
    int valid = 0, total = 0;
    for (int y = 8; y < r.depth.rows - 8; ++y) {
        for (int x = margin; x < r.depth.cols - 8; ++x) {
            ++total;
            const float z = r.depth.at<float>(y, x);
            if (z <= 0.0f) continue;
            ++valid;
            const double t = truth_depth.at<float>(y, x);
            abs_err.push_back(std::abs(z - t));
            rel_err.push_back(std::abs(z - t) / t);
        }
    }
    Recovered out;
    out.valid_ratio = total ? static_cast<double>(valid) / total : 0.0;
    out.median_depth = r.median_depth_m;
    auto med = [](std::vector<double>& v) {
        if (v.empty()) return 1e9;
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    out.median_abs_err = med(abs_err);
    out.median_rel_err = med(rel_err);
    return out;
}

cv::Mat constantDepth(double z) {
    return cv::Mat(kH, kW, CV_32F, cv::Scalar(static_cast<float>(z)));
}

// 아래로 갈수록 가까워지는 지면. KITTI 하단이 이렇게 생겼다.
cv::Mat groundPlaneDepth(double z_near, double z_far) {
    cv::Mat d(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        // 1/Z 가 y 에 선형이어야 실제 평면이다 (시차가 y 에 선형).
        const double t = static_cast<double>(y) / (kH - 1);
        const double inv = (1.0 - t) / z_far + t / z_near;
        for (int x = 0; x < kW; ++x) d.at<float>(y, x) = static_cast<float>(1.0 / inv);
    }
    return d;
}

}  // namespace

// --- 규약: 만든 깊이를 되찾는다 ---------------------------------------------

TEST(StereoDepth, RecoversTheDepthThatGeneratedTheImage) {
    const Recovered r = runCase(718.856, 0.537, constantDepth(12.0));
    EXPECT_GT(r.valid_ratio, 0.9) << "유효 픽셀이 너무 적다";
    EXPECT_LT(r.median_rel_err, 0.02) << "중앙 상대오차 " << r.median_rel_err;
}

// f*B 를 스윕한다. 한 점에서만 맞는 구현은 여기서 떨어진다.
class StereoScale : public ::testing::TestWithParam<std::pair<double, double>> {};

// 서브픽셀 시차 오차 한계 (px). 이 아래로는 SGBM 이 못 준다.
constexpr double kSubpixelBudget = 0.30;

TEST_P(StereoScale, DepthTracksFocalTimesBaseline) {
    const auto [f, b] = GetParam();
    constexpr double kZ = 15.0;
    const Recovered r = runCase(f, b, constantDepth(kZ));
    EXPECT_GT(r.valid_ratio, 0.85) << "f=" << f << " B=" << b;

    // 허용오차를 상수로 두면 안 된다. dZ/Z = dd/d 이고 d = f*B/Z 이므로
    // 같은 15 m 라도 EuRoC 급 (d=3.2 px) 은 KITTI (d=25.7 px) 보다 8 배
    // 부정확한 것이 정상이다. 상수 3 % 를 요구하면 기하가 줄 수 없는 것을
    // 요구하게 되고, 그 실패는 구현의 실패로 잘못 읽힌다.
    const double d = f * b / kZ;
    const double tol = kSubpixelBudget / d;
    EXPECT_LT(r.median_rel_err, tol)
        << "f=" << f << " B=" << b << " 시차 " << d << " px, 상대오차 "
        << r.median_rel_err << " (예산 " << tol << ")";
    EXPECT_NEAR(r.median_depth, kZ, kZ * tol);
}

INSTANTIATE_TEST_SUITE_P(FocalBaseline, StereoScale,
    ::testing::Values(std::make_pair(718.856, 0.537),    // KITTI gray
                      std::make_pair(435.2, 0.110),      // EuRoC 급
                      std::make_pair(1000.0, 0.250),
                      std::make_pair(500.0, 0.800)));

// 위 스윕이 "느슨해서" 통과하는 게 아니라는 것. 오차는 상수가 아니라
// 1/시차로 움직여야 한다 - 그래야 깊이 오차가 시차 양자화에 지배된다는
// 주장이 성립한다. 각 설정에서 rel_err * d 가 같은 값이어야 한다.
TEST(StereoDepth, RelativeErrorScalesWithInverseDisparity) {
    constexpr double kZ = 15.0;
    const std::vector<std::pair<double, double>> fb = {
        {718.856, 0.537}, {435.2, 0.110}, {1000.0, 0.250}, {500.0, 0.800}};

    std::vector<double> sigma_d;   // rel_err * d = 유효 시차 오차 (px)
    std::vector<double> disp;
    for (const auto& [f, b] : fb) {
        const Recovered r = runCase(f, b, constantDepth(kZ));
        const double d = f * b / kZ;
        sigma_d.push_back(r.median_rel_err * d);
        disp.push_back(d);
    }
    const double lo = *std::min_element(sigma_d.begin(), sigma_d.end());
    const double hi = *std::max_element(sigma_d.begin(), sigma_d.end());

    // 시차는 3.2 ~ 26.7 px 로 8 배 넘게 흔들었다. 그 사이에서 시차 오차가
    // 대략 일정해야 한다.
    EXPECT_GT(*std::max_element(disp.begin(), disp.end()) /
              *std::min_element(disp.begin(), disp.end()), 5.0)
        << "스윕이 시차를 충분히 흔들지 않았다";
    EXPECT_LT(hi / lo, 4.0) << "시차 오차가 설정마다 " << lo << " ~ " << hi
                            << " px 로 흩어진다 - 양자화 지배가 아니다";
    EXPECT_LT(hi, kSubpixelBudget) << "서브픽셀 오차 " << hi << " px";
}

// 이 스윕이 정말로 판별하는지 - f*B 를 틀리게 주면 반드시 떨어져야 한다.
TEST(StereoDepth, WrongFocalTimesBaselineIsDetected) {
    StereoDepthConfig cfg;
    cfg.focal_px = 718.856;
    cfg.baseline_m = 0.537;
    cfg.max_depth_m = 200.0;
    cfg.max_depth_sigma_m = 100.0;

    cv::Mat left = textureImage();
    const double z_true = 15.0;
    // 오른쪽 영상은 참 f*B 로 만들고, 추정기에는 20% 틀린 f 를 준다.
    cv::Mat disp(kH, kW, CV_32F,
                 cv::Scalar(static_cast<float>(cfg.focal_px * cfg.baseline_m / z_true)));
    cv::Mat right = warpToRight(left, disp);

    cfg.focal_px *= 1.2;
    StereoDepth stereo(cfg);
    const StereoDepthResult r = stereo.compute(left, right);
    EXPECT_NEAR(r.median_depth_m, z_true * 1.2, 0.6)
        << "f 를 1.2 배로 주면 깊이도 1.2 배여야 한다 - 아니면 f*B 가 안 쓰이고 있다";
}

// --- 깊이가 변하는 장면 -----------------------------------------------------

TEST(StereoDepth, RecoversASlantedGroundPlane) {
    const Recovered r = runCase(718.856, 0.537, groundPlaneDepth(6.0, 40.0));
    EXPECT_GT(r.valid_ratio, 0.8);
    EXPECT_LT(r.median_rel_err, 0.05) << "중앙 상대오차 " << r.median_rel_err;
}

// --- 무효 픽셀 --------------------------------------------------------------

TEST(StereoDepth, TexturelessRegionYieldsNoDepthRatherThanWrongDepth) {
    StereoDepthConfig cfg;
    cfg.focal_px = 718.856;
    cfg.baseline_m = 0.537;
    cv::Mat flat(kH, kW, CV_8U, cv::Scalar(128));
    StereoDepth stereo(cfg);
    const StereoDepthResult r = stereo.compute(flat, flat);
    // 균일 영상에는 대응이 없다. 깊이를 지어내면 안 된다.
    EXPECT_LT(r.valid_ratio, 0.05) << "텍스처 없는 벽에서 깊이를 " << r.valid_ratio
                                   << " 비율로 만들어 냈다";
}

TEST(StereoDepth, DepthAndDisparityAgreeWhereValid) {
    StereoDepthConfig cfg;
    cfg.focal_px = 718.856;
    cfg.baseline_m = 0.537;
    cfg.max_depth_m = 200.0;
    cfg.max_depth_sigma_m = 100.0;
    cv::Mat left = textureImage();
    cv::Mat disp(kH, kW, CV_32F, cv::Scalar(30.0f));
    cv::Mat right = warpToRight(left, disp);

    StereoDepth stereo(cfg);
    const StereoDepthResult r = stereo.compute(left, right);
    ASSERT_GT(r.valid_ratio, 0.5);
    const double fb = cfg.focal_px * cfg.baseline_m;
    int checked = 0;
    for (int y = 0; y < r.depth.rows; y += 7) {
        for (int x = 200; x < r.depth.cols; x += 7) {
            const float z = r.depth.at<float>(y, x);
            const float d = r.disparity.at<float>(y, x);
            if (z <= 0.0f) { EXPECT_LT(d, 0.0f); continue; }
            EXPECT_NEAR(z, fb / d, 1e-3);
            ++checked;
        }
    }
    EXPECT_GT(checked, 100);
}

// --- 탐색 범위 부족 ---------------------------------------------------------
//
// 여기가 실측에서 물린 곳이다. TUM(깊이 1~5 m)에 KITTI 베이스라인 0.54 m 를
// 쓰면 참 시차가 140 px 인데 num_disparities=128 이라 범위 밖이다. SGBM 은
// "못 찾았다" 고 하지 않는다 - 범위 안 최선을 돌려주고, 중앙 깊이 스케일이
// 2.34 배로 나왔다. **조용히 틀린 답**이라, 무효 표시로 바꾸는 것이 맞다.

namespace {

// 앞쪽 절반은 범위 밖(가깝고), 뒤쪽 절반은 범위 안. 실제 장면은 늘 섞여 있고,
// 섞였을 때가 위험하다 - 범위 안 화소가 결과를 "그럴듯하게" 만들어 준다.
cv::Mat mixedDepth(double z_near, double z_far) {
    cv::Mat d(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        const float v = static_cast<float>(y < kH / 2 ? z_far : z_near);
        for (int x = 0; x < kW; ++x) d.at<float>(y, x) = v;
    }
    return d;
}

struct NearHalf {
    double valid_ratio{0.0};      // 근거리 영역 중 깊이가 나온 비율
    double median_rel_err{0.0};   // 그 깊이들의 참값 대비 중앙 상대오차
};

// 근거리 절반(아래쪽)만 떼어 본다. 전체 중앙값은 원거리 절반이 가려 버린다.
NearHalf nearHalfStats(const StereoDepthResult& r, double z_near, int left_margin) {
    std::vector<double> rel;
    int valid = 0, total = 0;
    for (int y = kH / 2 + 8; y < kH - 8; ++y) {
        for (int x = left_margin; x < kW - 8; ++x) {
            ++total;
            const float z = r.depth.at<float>(y, x);
            if (z <= 0.0f) continue;
            ++valid;
            rel.push_back(std::abs(z - z_near) / z_near);
        }
    }
    NearHalf out;
    out.valid_ratio = total ? static_cast<double>(valid) / total : 0.0;
    if (!rel.empty()) {
        std::nth_element(rel.begin(), rel.begin() + static_cast<std::ptrdiff_t>(rel.size() / 2),
                         rel.end());
        out.median_rel_err = rel[rel.size() / 2];
    }
    return out;
}

StereoDepthResult runMixed(int num_disparities, double z_near, double z_far) {
    StereoDepthConfig cfg;
    cfg.focal_px = 517.3;
    cfg.baseline_m = 0.54;
    cfg.num_disparities = num_disparities;
    cfg.min_depth_m = 0.3;
    cfg.max_depth_m = 12.0;
    cfg.max_depth_sigma_m = 100.0;

    cv::Mat left = textureImage();
    cv::Mat truth = mixedDepth(z_near, z_far);
    cv::Mat disp(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
            disp.at<float>(y, x) =
                static_cast<float>(cfg.focal_px * cfg.baseline_m / truth.at<float>(y, x));
    cv::Mat right = warpToRight(left, disp);
    return StereoDepth(cfg).compute(left, right);
}

}  // namespace

TEST(StereoDepth, TheSearchRangeCeilingIsReportedNotInferred) {
    // 유일하게 믿을 수 있는 신호는 설정 단계에서 계산되는 이 값이다.
    // 화소 단위로는 "범위 밖이라 틀렸다" 를 알아낼 방법이 없다 - 아래 참조.
    const StereoDepthResult r = runMixed(128, 1.5, 4.0);
    EXPECT_NEAR(r.min_representable_depth_m, 517.3 * 0.54 / 127.0, 1e-6);
    EXPECT_EQ(StereoDepth::requiredDisparities(517.3, 0.54, 1.5), 192)
        << "필요 시차 = 517.3*0.54/1.5 = 186.2 -> 192";
}

TEST(StereoDepth, OutOfRangeSurfacesFailSilentlyAndOnlyMoreRangeFixesIt) {
    // 이 테스트가 기록하는 것은 **위험** 이다. 개선 사항이 아니다.
    // 시차 범위가 모자라면 SGBM 은 실패를 보고하지 않고 그럴듯한 오답을 준다.
    // TUM(1~5 m)에 KITTI 베이스라인을 그대로 쓴 실측에서 중앙 스케일이
    // 2.34 배였다. 화소별 탐지는 불가능하므로 - 상한에 붙는 화소는 0.4 %
    // 뿐이고 나머지는 그냥 엉뚱한 곳에서 매칭된다 - 설정으로 막는 수밖에 없다.
    constexpr double kNear = 1.5, kFar = 4.0;
    const int need = StereoDepth::requiredDisparities(517.3, 0.54, kNear);
    const StereoDepthResult narrow = runMixed(128, kNear, kFar);
    const StereoDepthResult wide = runMixed(need, kNear, kFar);

    // 워프가 만든 무대응 띠를 양쪽 똑같이 잘라 낸다. 안 그러면 범위가 넓은
    // 쪽이 더 많이 잘려 비교가 불공정해진다.
    const NearHalf n = nearHalfStats(narrow, kNear, need + 8);
    const NearHalf w = nearHalfStats(wide, kNear, need + 8);

    // 실패는 커버리지에서 시끄럽고, 살아남은 값에서 조용하다.
    EXPECT_LT(n.valid_ratio, 0.2)
        << "근거리 커버리지가 " << n.valid_ratio << " - 범위 밖인데 다 잡혔다";
    EXPECT_GT(n.median_rel_err, 0.3)
        << "살아남은 " << n.valid_ratio << " 의 중앙 상대오차가 "
        << n.median_rel_err << " - 소수의 생존자가 맞는다면 위험이 아니다";
    EXPECT_LT(narrow.clipped_ratio, 0.05)
        << "상한 클리핑으로 잡히는 양. 이게 크면 화소별 탐지가 가능하다는 뜻이고,"
           " 그러면 설정이 아니라 코드로 막을 수 있다";

    // 범위를 맞추면 같은 코드가 같은 장면을 맞힌다.
    EXPECT_GT(w.valid_ratio, 0.8);
    EXPECT_LT(w.median_rel_err, 0.05)
        << "범위를 넓혀도 안 맞으면 위 실패는 범위 탓이 아니다";
}

// --- 오차 예산에서 유도된 시차 하한 -----------------------------------------

TEST(StereoDepth, MinimumDisparityFollowsTheDepthErrorBudget) {
    StereoDepthConfig c;
    c.focal_px = 718.856;
    c.baseline_m = 0.537;
    c.disparity_noise_px = 0.5;
    c.max_depth_m = 1e6;             // 원거리 컷을 비활성화해 오차예산만 남긴다

    c.max_depth_sigma_m = 2.0;
    const double d_loose = StereoDepth::minValidDisparity(c);
    c.max_depth_sigma_m = 0.5;
    const double d_tight = StereoDepth::minValidDisparity(c);

    // sigma 를 1/4 로 조이면 하한은 2 배가 된다 (sqrt 관계).
    EXPECT_NEAR(d_tight / d_loose, 2.0, 0.02);
    // 그 하한이 실제로 뜻하는 깊이
    const double z_loose = c.focal_px * c.baseline_m / d_loose;
    EXPECT_NEAR(z_loose, std::sqrt(c.focal_px * c.baseline_m * 2.0 / 0.5), 0.5);
}

TEST(StereoDepth, ErrorBudgetActuallyRemovesFarPixels) {
    // 같은 장면을 느슨한 예산과 빡빡한 예산으로 돌려 유효 비율이 줄어야 한다.
    // 줄지 않으면 이 설정은 아무것도 안 하는 장식이다.
    cv::Mat left = textureImage();
    auto run = [&](double sigma) {
        StereoDepthConfig cfg;
        cfg.focal_px = 718.856;
        cfg.baseline_m = 0.537;
        cfg.max_depth_m = 1e6;
        cfg.max_depth_sigma_m = sigma;
        cv::Mat truth = groundPlaneDepth(6.0, 60.0);
        cv::Mat disp(kH, kW, CV_32F);
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x)
                disp.at<float>(y, x) =
                    static_cast<float>(cfg.focal_px * cfg.baseline_m / truth.at<float>(y, x));
        cv::Mat right = warpToRight(left, disp);
        return StereoDepth(cfg).compute(left, right).valid_ratio;
    };
    const double loose = run(50.0);
    const double tight = run(0.5);
    EXPECT_GT(loose, tight + 0.1) << "loose=" << loose << " tight=" << tight;
}
