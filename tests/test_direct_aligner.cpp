// ECDA 검증.
// 평면 장면은 두 시점 사이 변환이 정확히 호모그래피이므로, 알려진 SE(3)로
// 합성한 영상에서 그 SE(3)를 되찾는지 오차 없이 확인할 수 있다.
//
// 검증 대상:
//   - 알려진 포즈 복원 정확도
//   - 노출 변화(아핀 밝기) 하에서의 불변성
//   - 동적 객체 마스킹의 실효성  <- 고전 직접법이 못 풀던 지점
//   - 퇴화 보고 (텍스처 빈곤)
//   - 결정적 재현성

#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/ImageQualityEngine.hpp"

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <random>

using namespace wme;

namespace {

constexpr int kW = 640, kH = 480;

CameraIntrinsics makeK() {
    CameraIntrinsics K;
    K.fx = 500.0; K.fy = 500.0;
    K.cx = kW * 0.5 - 0.5; K.cy = kH * 0.5 - 0.5;
    K.width = kW; K.height = kH;
    return K;
}

// 워프로 화면 밖에서 들어오는 영역을 진짜 텍스처로 채우기 위한 여백.
// 0.18 m / 2.5 m 병진은 약 36 px, 0.05 rad 회전은 약 25 px 이므로 120 이면 충분하다.
constexpr int kMargin = 120;

// 모든 곳에 그래디언트가 있는 결정적 텍스처. 직접법은 텍스처가 없으면 못 푼다.
cv::Mat makeTexture(int w, int h, unsigned seed = 5150) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> px(0, w - 1), py(0, h - 1);
    std::uniform_int_distribution<int> rad(6, 30), col(30, 225);

    cv::Mat img(h, w, CV_8UC1, cv::Scalar(110));
    // 저주파 배경 기울기
    for (int y = 0; y < h; ++y) {
        auto* row = img.ptr<std::uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            row[x] = static_cast<std::uint8_t>(80 + 60.0 * x / w + 40.0 * y / h);
        }
    }
    // 면적에 비례해 개수를 맞춰 여백을 넓혀도 텍스처 밀도가 같게 유지한다
    const int n_circle = 900 * w * h / (kW * kH);
    const int n_line   = 300 * w * h / (kW * kH);
    for (int i = 0; i < n_circle; ++i) {
        cv::circle(img, {px(gen), py(gen)}, rad(gen), cv::Scalar(col(gen)), cv::FILLED);
    }
    for (int i = 0; i < n_line; ++i) {
        cv::line(img, {px(gen), py(gen)}, {px(gen), py(gen)}, cv::Scalar(col(gen)), 2);
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

// 평면 n^T P = dist 에 대한 깊이맵. 픽셀 광선과 평면의 교점 거리.
cv::Mat makePlaneDepth(const CameraIntrinsics& K, const Vec3& n, double dist) {
    cv::Mat depth(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < kW; ++x) {
            const Vec3 ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0);
            const double denom = n.dot(ray);
            row[x] = static_cast<float>(dist / denom);
        }
    }
    return depth;
}

// P_cur = R P_ref + t 이고 평면 위 점이면 H = K (R + t n^T / dist) K^-1
cv::Mat planeHomography(const CameraIntrinsics& K, const SE3& T_cur_ref,
                        const Vec3& n, double dist) {
    Mat3 Km;
    Km << K.fx, 0.0, K.cx,
          0.0, K.fy, K.cy,
          0.0, 0.0, 1.0;
    const Mat3 M = T_cur_ref.rotation().matrix() + T_cur_ref.translation() * n.transpose() / dist;
    const Mat3 H = Km * M * Km.inverse();

    cv::Mat h(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) h.at<double>(r, c) = H(r, c);
    }
    return h;
}

cv::Mat warpByHomography(const cv::Mat& src, const cv::Mat& H) {
    cv::Mat dst;
    cv::warpPerspective(src, dst, H, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return dst;
}

Frame makeFrame(const cv::Mat& gray, const cv::Mat& depth, double t) {
    Frame f;
    f.id    = FrameId(static_cast<std::uint64_t>(t * 1000.0) + 1);
    f.stamp = Timestamp::fromSeconds(t + 1.0);   // 0 은 무효 시각 규약
    f.gray  = gray;
    f.depth = depth;
    f.intrinsics = makeK();
    f.sensor = depth.empty() ? SensorKind::Monocular : SensorKind::RgbD;
    return f;
}

struct Scene {
    Frame ref;
    Frame cur;
    SE3   truth;
};

// 알려진 포즈로 ref/cur 한 쌍을 합성한다.
//
// 여백을 둔 큰 캔버스에서 워프한 뒤 가운데를 잘라낸다. 화면 크기 그대로 워프하면
// 병진이 큰 경우 화면 밖에서 들어와야 할 띠를 BORDER_REPLICATE 가 지어내는데,
// 그 픽셀에는 참대응이 아예 없다. 거친 피라미드 레벨에서는 그 가짜 띠가 화면의
// 큰 비율을 차지해 강한 오답 인력이 되고, 실제로 그 때문에 피라미드가 단일
// 레벨보다 나쁜 결과(0.375 m vs 0.229 m)를 냈다. 엔진이 아니라 장면의 결함이었다.
Scene makeScene(const Vec3& rot, const Vec3& trans, unsigned seed = 5150) {
    const CameraIntrinsics K = makeK();
    const Vec3   n(0.15, 0.10, 0.98);   // 약간 기울어진 평면 (정면 평면은 조건수가 나쁘다)
    const double dist = 2.5;

    const int PW = kW + 2 * kMargin, PH = kH + 2 * kMargin;
    CameraIntrinsics Kp = K;
    Kp.cx += kMargin; Kp.cy += kMargin;
    Kp.width = PW;    Kp.height = PH;

    const cv::Mat tex_pad = makeTexture(PW, PH, seed);

    const SE3 T(SO3::exp(rot), trans);
    const cv::Mat cur_pad = warpByHomography(tex_pad, planeHomography(Kp, T, n.normalized(), dist));

    const cv::Rect roi(kMargin, kMargin, kW, kH);
    const cv::Mat tex   = tex_pad(roi).clone();
    const cv::Mat cur   = cur_pad(roi).clone();
    const cv::Mat depth = makePlaneDepth(K, n.normalized(), dist);

    return {makeFrame(tex, depth, 0.0), makeFrame(cur, cv::Mat(), 0.033), T};
}

}  // namespace

TEST(DirectAligner, RecoversKnownPose) {
    const auto scene = makeScene(Vec3(0.010, -0.015, 0.008), Vec3(0.05, -0.03, 0.02));

    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Vec2 d = r.value().T_cur_ref.distanceTo(scene.truth);
    EXPECT_LT(d.x(), 0.02) << "병진 오차 " << d.x() << " m";
    EXPECT_LT(d.y(), 0.01) << "회전 오차 " << d.y() << " rad";
    EXPECT_GT(r.value().inlier_ratio, 0.7);
    EXPECT_GE(r.value().point_count, 500u);
}

// 실측 수렴 반경 안의 큰 변위. 노름 0.175 m = 레벨0 기준 35 px.
// 이보다 크면(0.22 m 등) 어떤 냉시작 직접법도 못 잡으며, 그것이 운동 사전분포가
// 있는 이유다. 반경 자체는 PyramidDepthExtendsConvergenceRadius 가 측정한다.
inline Vec3 largeTranslation() { return {0.14, 0.095, -0.048}; }
inline Vec3 largeRotation()    { return {0.023, 0.031, -0.016}; }

TEST(DirectAligner, ConvergesFromIdentityOnLargeMotion) {
    // 피라미드가 있어야 큰 변위를 잡는다. 단일 해상도면 지역최소에 빠진다.
    const auto scene = makeScene(largeRotation(), largeTranslation());

    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Vec2 d = r.value().T_cur_ref.distanceTo(scene.truth);
    EXPECT_LT(d.x(), 0.05);
    EXPECT_LT(d.y(), 0.02);
}

TEST(DirectAligner, SingleLevelIsWorseThanPyramid) {
    // 피라미드의 존재 이유를 명시적으로 남긴다
    const auto scene = makeScene(largeRotation(), largeTranslation());

    DirectAlignerConfig single;
    single.pyramid_levels = 1;
    DirectAligner flat(single);
    const auto r_flat = flat.align(scene.ref, scene.cur, SE3::identity());

    DirectAligner pyr;
    const auto r_pyr = pyr.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r_pyr.ok());

    const double err_pyr = r_pyr.value().T_cur_ref.distanceTo(scene.truth).x();
    const double err_flat = r_flat.ok()
                                ? r_flat.value().T_cur_ref.distanceTo(scene.truth).x()
                                : 1e9;
    EXPECT_LT(err_pyr, err_flat);
}

TEST(DirectAligner, TruthIsAFixedPointOfTheResidual) {
    // 항등에서 못 가는 것이 수렴 반경 문제인지, 최적점 자체가 틀린 건지 가른다.
    // 진리값에서 출발했는데 벗어나면 잔차/야코비안이 틀린 것이고,
    // 그대로 머무르면 장면과 잔차는 일관되며 문제는 반경뿐이다.
    const auto scene = makeScene(Vec3(0.03, 0.04, -0.02), Vec3(0.18, 0.12, -0.06));

    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, scene.truth);
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Vec2 d = r.value().T_cur_ref.distanceTo(scene.truth);
    std::cout << "  진리값 초기화: t_err=" << d.x() << " m  r_err=" << d.y()
              << " rad  inlier=" << r.value().inlier_ratio << "\n";

    EXPECT_LT(d.x(), 0.01) << "진리값에서 출발했는데 벗어났다 - 최적점이 틀렸다";
    EXPECT_LT(d.y(), 0.005);
    EXPECT_GT(r.value().inlier_ratio, 0.7) << "진리값에서도 이상치가 많다 - 장면 합성이 틀렸다";
}

TEST(DirectAligner, BasinRadiusAndKernelWidth) {
    // 수렴 반경을 실측한다. 남은 오차 / 초기 오차 이므로 1 이면 전혀 못 움직인 것,
    // 1 보다 크면 진리값에서 오히려 멀어진 것이다.
    // 적응형 임계(huber_k)가 고정 임계보다 반경이 넓어야 한다.
    const double ks[]   = {1.345, 4.0, 1e9};
    const double mags[] = {0.03, 0.06, 0.10, 0.14, 0.18};

    double worst_at_010 = 0.0;
    for (double k : ks) {
        std::cout << "  huber_k=" << k << "  ";
        for (double mag : mags) {
            const Vec3 t = Vec3(0.8, 0.53, -0.27).normalized() * mag;
            const auto scene = makeScene(Vec3(0.03, 0.04, -0.02) * (mag / 0.18), t);

            DirectAlignerConfig cfg;
            cfg.huber_k = k;
            DirectAligner a(cfg);
            const auto r = a.align(scene.ref, scene.cur, SE3::identity());
            const double err = r.ok() ? r.value().T_cur_ref.distanceTo(scene.truth).x() : 1e9;
            std::cout << mag << "m:" << (err / mag) << "  ";
            if (k == 1.345 && mag <= 0.10) worst_at_010 = std::max(worst_at_010, err / mag);
        }
        std::cout << "\n";
    }
    // 기본 설정에서 0.10 m 까지는 초기 오차의 10 % 이내로 줄여야 한다.
    EXPECT_LT(worst_at_010, 0.10) << "기본 커널로 0.10 m 를 못 잡는다";
}

TEST(DirectAligner, PyramidDepthExtendsConvergenceRadius) {
    // 레벨을 늘릴수록 수렴 반경이 넓어져야 한다. 어느 레벨에서 깨지는지
    // 숫자로 남긴다 - 이 프로젝트에서 잔차만이 원인을 구분해 준 적이 여러 번 있었다.
    const auto scene = makeScene(Vec3(0.03, 0.04, -0.02), Vec3(0.18, 0.12, -0.06));

    // 한 크기의 오차만 보면 아무 것도 안 나온다. 레벨 수마다 "어디까지 잡히는가"
    // 를 재야 피라미드가 실제로 무엇을 해주는지 알 수 있다.
    const double mags[] = {0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.18};

    double radius_single = 0.0, radius_deep = 0.0;
    for (int levels = 1; levels <= 6; ++levels) {
        DirectAlignerConfig cfg;
        cfg.pyramid_levels = levels;

        double radius = 0.0;
        for (double mag : mags) {
            const Vec3 t = Vec3(0.8, 0.53, -0.27).normalized() * mag;
            const auto s = makeScene(Vec3(0.03, 0.04, -0.02) * (mag / 0.18), t);
            DirectAligner a(cfg);
            const auto r = a.align(s.ref, s.cur, SE3::identity());
            const double err = r.ok() ? r.value().T_cur_ref.distanceTo(s.truth).x() : 1e9;
            if (err / mag > 0.10) break;    // 여기서 반경이 끝난다
            radius = mag;
        }
        std::cout << "  levels=" << levels << "  수렴반경=" << radius << " m ("
                  << (radius * 500.0 / 2.5) << " px @ level0)\n";

        if (levels == 1) radius_single = radius;
        radius_deep = std::max(radius_deep, radius);
    }

    // 피라미드의 존재 이유는 반경이다. 정확도가 아니라 반경으로 검증한다.
    EXPECT_GT(radius_deep, radius_single * 2.0)
        << "피라미드가 수렴 반경을 넓히지 못한다 - 단일 " << radius_single
        << " m, 최대 " << radius_deep << " m";
}

TEST(DirectAligner, InvariantToExposureChange) {
    auto scene = makeScene(Vec3(0.008, -0.010, 0.005), Vec3(0.04, -0.02, 0.015));

    // 노출/게인 변화 시뮬레이션: I' = 1.25 I + 18
    cv::Mat bright;
    scene.cur.gray.convertTo(bright, CV_8UC1, 1.25, 18.0);
    scene.cur.gray = bright;

    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Vec2 d = r.value().T_cur_ref.distanceTo(scene.truth);
    EXPECT_LT(d.x(), 0.03);
    EXPECT_LT(d.y(), 0.015);

    // 아핀 파라미터가 실제 노출 변화를 흡수했는지 확인
    EXPECT_NEAR(r.value().affine_a, 1.25, 0.12);
    EXPECT_NEAR(r.value().affine_b, 18.0, 12.0);
}

TEST(DirectAligner, ExposureChangeBreaksAlignmentWithoutAffineModel) {
    auto scene = makeScene(Vec3(0.008, -0.010, 0.005), Vec3(0.04, -0.02, 0.015));
    cv::Mat bright;
    scene.cur.gray.convertTo(bright, CV_8UC1, 1.25, 18.0);
    scene.cur.gray = bright;

    DirectAlignerConfig no_affine;
    no_affine.estimate_affine = false;
    DirectAligner aligner(no_affine);
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());

    const double err = r.ok() ? r.value().T_cur_ref.distanceTo(scene.truth).x() : 1e9;

    DirectAligner with_affine;
    const auto r2 = with_affine.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r2.ok());
    EXPECT_LT(r2.value().T_cur_ref.distanceTo(scene.truth).x(), err);
}

TEST(DirectAligner, DynamicObjectMaskImprovesAccuracy) {
    // WME 의 핵심 주장 중 하나: YOLO 토큰 마스크가 직접법의 고질적 약점을 없앤다.
    // 화면의 큰 영역이 장면과 다르게 움직이면 측광 잔차가 포즈를 끌어당긴다.
    auto scene = makeScene(Vec3(0.008, -0.010, 0.006), Vec3(0.05, -0.03, 0.02));

    // cur 에만 존재하는, 장면 워프를 따르지 않는 큰 물체
    const cv::Rect intruder(120, 100, 260, 240);
    cv::Mat cur_with_object = scene.cur.gray.clone();
    cv::rectangle(cur_with_object, intruder, cv::Scalar(20), cv::FILLED);
    for (int i = 0; i < 60; ++i) {   // 물체에도 텍스처를 줘야 실제로 끌어당긴다
        cv::circle(cur_with_object,
                   {intruder.x + 20 + (i * 37) % (intruder.width - 40),
                    intruder.y + 20 + (i * 53) % (intruder.height - 40)},
                   9, cv::Scalar(200), cv::FILLED);
    }
    scene.cur.gray = cur_with_object;

    DirectAligner aligner;
    const auto without = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(without.ok()) << without.error().message();

    // 토큰이 알려주는 동적 영역을 ref 마스크에서 제외한다 (0 = 동적)
    Frame masked_ref = scene.ref;
    masked_ref.static_mask = cv::Mat(kH, kW, CV_8UC1, cv::Scalar(255));
    cv::rectangle(masked_ref.static_mask, intruder, cv::Scalar(0), cv::FILLED);

    DirectAligner aligner2;
    const auto with = aligner2.align(masked_ref, scene.cur, SE3::identity());
    ASSERT_TRUE(with.ok()) << with.error().message();

    const double err_without = without.value().T_cur_ref.distanceTo(scene.truth).x();
    const double err_with    = with.value().T_cur_ref.distanceTo(scene.truth).x();

    EXPECT_LT(err_with, err_without)
        << "마스킹 후 " << err_with << " m, 마스킹 전 " << err_without << " m";
}

TEST(DirectAligner, ReportsDegeneracyOnTexturelessScene) {
    // 균일한 벽. 측광 잔차가 포즈를 구속하지 못하므로 성공을 주장하면 안 된다.
    const CameraIntrinsics K = makeK();
    const cv::Mat flat(kH, kW, CV_8UC1, cv::Scalar(128));
    const cv::Mat depth = makePlaneDepth(K, Vec3(0, 0, 1), 2.0);

    const Frame ref = makeFrame(flat, depth, 0.0);
    const Frame cur = makeFrame(flat, cv::Mat(), 0.033);

    DirectAligner aligner;
    const auto r = aligner.align(ref, cur, SE3::identity());

    if (r.ok()) {
        // 성공을 반환하더라도 관측 가능 자유도가 6 미만이고 degraded 여야 한다
        EXPECT_TRUE(r.degraded());
        EXPECT_LT(r.value().observable_dof, 6);
    } else {
        EXPECT_EQ(r.error().code, ErrorCode::DidNotConverge);
    }
}

TEST(DirectAligner, RequiresDepthOnReference) {
    const auto scene = makeScene(Vec3(0.01, 0.0, 0.0), Vec3(0.03, 0.0, 0.0));
    Frame no_depth = scene.ref;
    no_depth.depth = cv::Mat();

    DirectAligner aligner;
    const auto r = aligner.align(no_depth, scene.cur, SE3::identity());
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, ErrorCode::InsufficientData);
}

TEST(DirectAligner, InformationMatrixIsPositiveDefiniteAndScaled) {
    const auto scene = makeScene(Vec3(0.008, -0.008, 0.004), Vec3(0.04, -0.02, 0.01));

    ImageQualityEngine iq;
    const ImageQuality q = iq.evaluate(scene.ref);

    EnvironmentState env;
    env.tier.photometric = 1.0;

    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity(), &q, &env);
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Mat6& L = r.value().information;
    EXPECT_TRUE(L.allFinite());
    EXPECT_NEAR((L - L.transpose()).norm(), 0.0, 1e-6);
    for (int k = 0; k < 6; ++k) EXPECT_GT(L(k, k), 0.0);

    // 환경 가중이 정보 질량을 실제로 줄이는지 (적응 로직의 계약)
    EnvironmentState fog = env;
    fog.tier.photometric = 0.1;
    DirectAligner aligner2;
    const auto r_fog = aligner2.align(scene.ref, scene.cur, SE3::identity(), &q, &fog);
    ASSERT_TRUE(r_fog.ok());
    EXPECT_LT(r_fog.value().information.trace(), L.trace());
}

// ---------------------------------------------------------------------------
// 정보행렬 - 잡음 모델
// ---------------------------------------------------------------------------

// 잡음 모델은 *보고* 만 바꾼다. 추정은 건드리지 않아야 한다.
// 이 계약이 없으면 "일관성을 얻었는데 정확도가 나빠졌다" 가 조용히 일어날 수
// 있다 (docs/06-results.md 10.3). 여기서 구조적으로 막는다.
TEST(DirectAligner, InformationModelDoesNotChangeTheEstimate) {
    const auto scene = makeScene(Vec3(0.008, -0.008, 0.004), Vec3(0.04, -0.02, 0.01));
    ImageQualityEngine iq;
    const ImageQuality q = iq.evaluate(scene.ref);

    const InformationModel models[] = {
        InformationModel::SensorVariance, InformationModel::ResidualVariance,
        InformationModel::EffectiveSample, InformationModel::ClusterRobust,
        InformationModel::CoherentFrame};

    SE3 first;
    bool have_first = false;
    for (InformationModel m : models) {
        DirectAlignerConfig cfg;
        cfg.information_model = m;
        DirectAligner aligner(cfg);
        const auto r = aligner.align(scene.ref, scene.cur, SE3::identity(), &q);
        ASSERT_TRUE(r.ok()) << r.error().message();

        const Mat6& L = r.value().information;
        EXPECT_TRUE(L.allFinite());
        EXPECT_NEAR((L - L.transpose()).norm(), 0.0, 1e-6);
        Eigen::SelfAdjointEigenSolver<Mat6> es(L);
        ASSERT_EQ(es.info(), Eigen::Success);
        EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);

        if (!have_first) { first = r.value().T_cur_ref; have_first = true; }
        const Vec2 d = r.value().T_cur_ref.distanceTo(first);
        EXPECT_NEAR(d.x(), 0.0, 1e-12);
        EXPECT_NEAR(d.y(), 0.0, 1e-12);
    }
}

// CoherentFrame 이 EffectiveSample 과 다른 점은 딱 하나여야 한다: 분모의
// 잔차 분산이 이 프레임이 달성한 rmse^2 이 아니라 상수 coherent_sigma^2 다.
// 그 계약을 식으로 고정한다. 이게 없으면 나중에 N 을 wsum 으로 바꾸는 식의
// 조용한 변경이 상수의 보정값을 무효로 만들어도 아무도 모른다.
TEST(DirectAligner, CoherentFrameScalesByAConstantNotByTheAchievedResidual) {
    const auto scene = makeScene(Vec3(0.008, -0.008, 0.004), Vec3(0.04, -0.02, 0.01));
    ImageQualityEngine iq;
    const ImageQuality q = iq.evaluate(scene.ref);

    DirectAlignerConfig cfg_e;
    cfg_e.information_model = InformationModel::EffectiveSample;
    DirectAligner ae(cfg_e);
    const auto re = ae.align(scene.ref, scene.cur, SE3::identity(), &q);
    ASSERT_TRUE(re.ok()) << re.error().message();

    DirectAlignerConfig cfg_c;
    cfg_c.information_model = InformationModel::CoherentFrame;
    DirectAligner ac(cfg_c);
    const auto rc = ac.align(scene.ref, scene.cur, SE3::identity(), &q);
    ASSERT_TRUE(rc.ok()) << rc.error().message();

    const double n     = static_cast<double>(rc.value().point_count);
    const double rmse  = rc.value().photometric_rmse;
    const double var_e = std::max(q.photometricVariance(), n * rmse * rmse);
    const double var_c = std::max(q.photometricVariance(),
                                  n * cfg_c.coherent_sigma * cfg_c.coherent_sigma);

    // 두 Lambda 는 같은 H 를 서로 다른 분산으로 나눈 것이므로 비가 정확히
    // var_e / var_c 여야 한다.
    const Mat6 scaled = re.value().information * (var_e / var_c);
    EXPECT_LT((scaled - rc.value().information).norm() /
                  std::max(1e-12, rc.value().information.norm()),
              1e-9);

    // 그리고 그 상수는 프레임의 잔차와 무관하다: coherent_sigma 를 2 배로
    // 하면 Lambda 는 정확히 1/4 이 된다.
    DirectAlignerConfig cfg_2 = cfg_c;
    cfg_2.coherent_sigma *= 2.0;
    DirectAligner a2(cfg_2);
    const auto r2 = a2.align(scene.ref, scene.cur, SE3::identity(), &q);
    ASSERT_TRUE(r2.ok()) << r2.error().message();
    EXPECT_LT((r2.value().information * 4.0 - rc.value().information).norm() /
                  std::max(1e-12, rc.value().information.norm()),
              1e-9);
}

// CoherentFrame 의 알려진 한계를 숫자로 박아 둔다.
//
// Lambda = (H / N) / sigma_c^2 는 "점당 평균 정보" 다. 그래디언트가 줄면
// H 도 줄어 정보가 제대로 내려가지만, 열화가 그래디언트가 아니라 *점 선택*
// 을 바꾸면 (min_gradient 아래로 떨어진 약한 점이 통째로 빠지면) N 이 H 보다
// 빨리 줄어 평균이 오히려 올라간다.
//
// 실측 (python 오라클, 320x240 합성 방, motion_blur=0.9): 점 698 -> 131,
// tr(H) 7.1e9 -> 2.8e9 (2.6배 감소) 인데 N 이 5.3 배 줄어 tr(Lambda) 가
// 4.25e4 -> 8.77e4 로 2.06 배 *올라간다*. EffectiveSample 은 같은 장면에서
// 0.88 배로 내려간다 - 분모의 rmse^2 이 올라가 주기 때문이다.
//
// 즉 12 시퀀스에서 측정한 "잔차 크기는 정보 스케일에 넣으면 안 된다" 와
// "잔차 크기가 열화 단조성을 만들어 주고 있었다" 가 같은 항이다. 실데이터
// 열화 시퀀스가 없어 어느 쪽이 더 비싼지는 아직 못 정한다.
//
// 여기서는 선택이 바뀌지 않는 범위의 열화에 대해서는 정보가 제대로 내려간다는
// 것만 고정한다. 이게 깨지면 모델이 열화 방향을 통째로 뒤집은 것이다.
TEST(DirectAligner, CoherentFrameInformationFallsWhenGradientsFallAtFixedPointSet) {
    const auto scene = makeScene(Vec3(0.008, -0.008, 0.004), Vec3(0.04, -0.02, 0.01));

    // 대비만 절반으로 낮춘다. 그래디언트가 같은 비율로 줄되 점 선택 순서는
    // 그대로라 N 이 거의 바뀌지 않는다.
    const auto dim = [](const cv::Mat& src) {
        cv::Mat out;
        src.convertTo(out, CV_8U, 0.5, 64.0);
        return out;
    };
    Frame ref_d = scene.ref, cur_d = scene.cur;
    ref_d.gray = dim(scene.ref.gray);
    cur_d.gray = dim(scene.cur.gray);
    ref_d.pyramid = ImagePyramid{};
    cur_d.pyramid = ImagePyramid{};

    ImageQualityEngine iq;
    const ImageQuality q  = iq.evaluate(scene.ref);
    const ImageQuality qd = iq.evaluate(ref_d);

    DirectAlignerConfig cfg;
    cfg.information_model = InformationModel::CoherentFrame;
    DirectAligner a1(cfg), a2(cfg);
    const auto r0 = a1.align(scene.ref, scene.cur, SE3::identity(), &q);
    const auto r1 = a2.align(ref_d, cur_d, SE3::identity(), &qd);
    ASSERT_TRUE(r0.ok()) << r0.error().message();
    ASSERT_TRUE(r1.ok()) << r1.error().message();

    // 점 집합이 실질적으로 같아야 이 측정이 뜻을 갖는다 (10.4).
    const double n0 = static_cast<double>(r0.value().point_count);
    const double n1 = static_cast<double>(r1.value().point_count);
    ASSERT_GT(n1 / n0, 0.8) << "점 선택이 바뀌면 재려던 것과 다른 양을 잰다";

    EXPECT_LT(r1.value().information.trace(), r0.value().information.trace());
}

// 클러스터가 모자라면 샌드위치를 쓰지 않고 물러서야 한다.
// M 은 8x8 인데 클러스터 수가 그에 못 미치면 Cov 가 특이해지고, 부호만 보는
// 검사는 반올림 오차만큼의 양수를 통과시켜 그 방향에 무한한 확신을 싣는다.
// 실측(TUM, 클러스터 4 개): ANEES 가 1e16 까지 튀었다.
TEST(DirectAligner, ClusterRobustFallsBackWhenClustersAreTooFew) {
    const auto scene = makeScene(Vec3(0.008, -0.008, 0.004), Vec3(0.04, -0.02, 0.01));
    ImageQualityEngine iq;
    const ImageQuality q = iq.evaluate(scene.ref);

    DirectAlignerConfig cfg;
    cfg.information_model = InformationModel::ClusterRobust;
    cfg.info_cluster_grid = 2;          // 4 개. 8 개 파라미터에 턱없이 모자라다
    DirectAligner aligner(cfg);
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity(), &q);
    ASSERT_TRUE(r.ok()) << r.error().message();

    Eigen::SelfAdjointEigenSolver<Mat6> es(r.value().information);
    ASSERT_EQ(es.info(), Eigen::Success);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0);
    // 조건수가 터지지 않았는지. 특이한 Cov 를 뒤집었다면 여기서 걸린다.
    EXPECT_LT(es.eigenvalues()(5) / es.eigenvalues()(0), 1e12);
}

// 실데이터 과신의 원인이 이 식인가, 센서인가.
//
// TUM 다섯 시퀀스에서 SensorVariance 의 ANEES 는 5e3~1.5e5 다. 원인이 여기
// 정보행렬 식에 있다면 통제된 장면에서도 같은 크기로 나와야 한다. 이 장면은
// 깊이가 해석적으로 정확하고 워프도 정확해서 오차원이 넣어 준 픽셀 잡음뿐이다.
// 그 조건에서 ResidualVariance(= 잔차가 독립이라는 가정만 남긴 모델)가 거의
// 보정되어 나오면, 식은 맞고 실데이터의 과신은 잔차 상관과 모델 오차다.
TEST(DirectAligner, InformationIsNearlyCalibratedWhenResidualsReallyAreIndependent) {
    constexpr int    kTrials = 16;
    constexpr double kSigma  = 4.0;   // intensity

    std::mt19937 gen(20260804);
    std::normal_distribution<double>       nz(0.0, kSigma);
    std::uniform_real_distribution<double> um(-1.0, 1.0);

    DirectAlignerConfig cfg;
    cfg.information_model = InformationModel::ResidualVariance;
    DirectAligner      aligner(cfg);
    ImageQualityEngine iq;

    const auto add_noise = [&](const cv::Mat& src) {
        cv::Mat out = src.clone();
        for (int y = 0; y < out.rows; ++y) {
            auto* row = out.ptr<std::uint8_t>(y);
            for (int x = 0; x < out.cols; ++x) {
                row[x] = cv::saturate_cast<std::uint8_t>(
                    static_cast<double>(row[x]) + nz(gen));
            }
        }
        return out;
    };

    double sum = 0.0;
    int    n   = 0;
    for (int k = 0; k < kTrials; ++k) {
        const Vec3 rot(0.004 * um(gen), 0.004 * um(gen), 0.004 * um(gen));
        const Vec3 tr(0.02 * um(gen), 0.02 * um(gen), 0.02 * um(gen));
        Scene scene = makeScene(rot, tr, 5150u + static_cast<unsigned>(k));

        // 두 영상 모두에 독립 잡음을 넣는다. cur 에만 넣으면 ref 밝기가 정확해
        // 잔차 분산이 절반이 되고, 재려던 것과 다른 양을 재게 된다.
        scene.ref.gray = add_noise(scene.ref.gray);
        scene.cur.gray = add_noise(scene.cur.gray);

        const ImageQuality q = iq.evaluate(scene.ref);
        const auto r = aligner.align(scene.ref, scene.cur, SE3::identity(), &q);
        ASSERT_TRUE(r.ok()) << r.error().message();

        // e = log(T_est * T_gt^-1). 정보행렬이 정의된 좌측 섭동과 같은 규약.
        const Vec6 e = (r.value().T_cur_ref * scene.truth.inverse()).log();
        sum += e.dot(r.value().information * e);
        ++n;
    }
    const double anees = sum / n;
    std::cout << "  합성 ANEES(ResidualVariance) = " << anees
              << "   (보정되면 6, 실데이터에서는 845~8621)\n";

    // 정확한 값을 고정하지 않는다 - 쌍선형 보간이 이웃 픽셀 잔차에 약한 상관을
    // 남기므로 정확히 6 이 나올 이유는 없다. 고정하는 것은 자릿수뿐이다:
    // 잔차가 진짜 독립이면 이 식은 두 자릿수 안에 들어와야 하고, 그렇다면
    // 실데이터의 1e3~1e5 는 식이 아니라 데이터가 만든 것이다.
    EXPECT_LT(anees, 100.0);
    EXPECT_GT(anees, 0.06);
}

TEST(DirectAligner, IsDeterministic) {
    // 고정 블록 누산이므로 워커 수와 무관하게 결과가 같아야 한다.
    // 이 성질이 없으면 SLAM 버그는 절대 재현되지 않는다.
    const auto scene = makeScene(Vec3(0.01, -0.012, 0.006), Vec3(0.06, -0.03, 0.02));

    ThreadPool pool_a(1), pool_b(6);
    DirectAligner a({}, &pool_a);
    DirectAligner b({}, &pool_b);

    const auto ra = a.align(scene.ref, scene.cur, SE3::identity());
    const auto rb = b.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(ra.ok());
    ASSERT_TRUE(rb.ok());

    const Vec2 d = ra.value().T_cur_ref.distanceTo(rb.value().T_cur_ref);
    EXPECT_NEAR(d.x(), 0.0, 1e-12);
    EXPECT_NEAR(d.y(), 0.0, 1e-12);
    EXPECT_EQ(ra.value().point_count, rb.value().point_count);
}

TEST(DirectAligner, GoodInitialGuessConvergesFaster) {
    const auto scene = makeScene(Vec3(0.02, 0.025, -0.015), Vec3(0.12, 0.08, -0.04));

    DirectAligner cold;
    const auto r_cold = cold.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r_cold.ok());

    // 등속 모델이 주는 초기 추정을 흉내낸다 (참값의 80%)
    const SE3 warm_init = SE3::exp(scene.truth.log() * 0.8);
    DirectAligner warm;
    const auto r_warm = warm.align(scene.ref, scene.cur, warm_init);
    ASSERT_TRUE(r_warm.ok());

    EXPECT_LE(r_warm.value().iterations, r_cold.value().iterations);
    EXPECT_LT(r_warm.value().T_cur_ref.distanceTo(scene.truth).x(), 0.05);
}

// ===========================================================================
// 기하 정합성 - 측광과 독립인 실패 신호
//
// 06-results.md 26 절의 주장을 코드에 고정한다. 13.3 과 23.3 이 함께 말하는
// 문제는 "정확도가 낮다"가 아니라 **엔진이 자기가 틀린 줄 모른다**는 것이었고,
// 그 이유는 측광 신호가 상한에 갇혀 있다는 데 있었다(잔차는 밝기 범위, inlier
// 비는 [0,1]). depth_consistency 는 cur 의 깊이맵과 비교하는데, 그 깊이맵은
// 정렬에 한 번도 쓰이지 않으므로 독립 관측이다.
// ===========================================================================

namespace {

// ref 평면을 cur 좌표계에서 본 깊이맵. 평면 n^T X = d 를 T 로 옮기면
// (R n)^T X_cur = d + (R n)^T t 이므로 법선은 R n, 거리는 d + (R n).t 다.
cv::Mat planeDepthInCur(const CameraIntrinsics& K, const Vec3& n_ref, double dist,
                        const SE3& T_cur_ref) {
    const Vec3 n_cur = T_cur_ref.rotation().matrix() * n_ref;
    const double d_cur = dist + n_cur.dot(T_cur_ref.translation());
    return makePlaneDepth(K, n_cur, d_cur);
}

// cur 에도 깊이를 붙인 장면. makeScene 은 cur 깊이를 비워 두므로
// (그래서 이 신호가 기존 테스트에서 한 번도 켜지지 않았다) 여기서 채운다.
Scene sceneWithCurDepth(const Vec3& rot, const Vec3& trans) {
    Scene s = makeScene(rot, trans);
    const Vec3 n(0.15, 0.10, 0.98);
    s.cur.depth = planeDepthInCur(makeK(), n.normalized(), 2.5, s.truth);
    s.cur.sensor = SensorKind::RgbD;
    return s;
}

}  // namespace

TEST(DirectAligner, DepthConsistencyIsLowWhenThePoseIsRight) {
    const auto scene = sceneWithCurDepth(Vec3(0.010, -0.015, 0.008),
                                         Vec3(0.05, -0.03, 0.02));
    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok());
    ASSERT_GE(r.value().depth_consistency, 0.0) << "표본 부족으로 판정되지 않았다";
    EXPECT_LT(r.value().depth_consistency, 0.02);
    EXPECT_TRUE(r.value().depthConsistent());
}

// 이것이 이 신호의 존재 이유다. cur 의 깊이만 15 % 늘리면 밝기 영상은 **전혀**
// 바뀌지 않으므로 측광 잔차와 inlier 비는 그대로다. 오직 깊이 채널만 어긋난다.
// 측광 경로가 볼 수 없는 오류를 이 채널이 본다는 것을 그대로 검사한다.
TEST(DirectAligner, DepthConsistencySeesAnErrorThePhotometricChannelCannot) {
    const auto scene = sceneWithCurDepth(Vec3(0.010, -0.015, 0.008),
                                         Vec3(0.05, -0.03, 0.02));
    DirectAligner aligner;
    const auto clean = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(clean.ok());

    Scene bad = scene;
    bad.cur.depth = scene.cur.depth.clone();
    bad.cur.depth *= 1.15f;                      // 기하만 어긋나게 한다

    DirectAligner aligner2;
    const auto r = aligner2.align(bad.ref, bad.cur, SE3::identity());
    ASSERT_TRUE(r.ok());

    // 측광 채널은 눈치채지 못한다 - 밝기는 한 화소도 바뀌지 않았다.
    EXPECT_NEAR(r.value().photometric_rmse, clean.value().photometric_rmse, 1e-9);
    EXPECT_NEAR(r.value().inlier_ratio, clean.value().inlier_ratio, 1e-9);

    // 깊이 채널은 본다.
    ASSERT_GE(r.value().depth_consistency, 0.0);
    EXPECT_GT(r.value().depth_consistency, 0.10);
    EXPECT_FALSE(r.value().depthConsistent());
}

// 판정 불가를 "괜찮다"로 읽지 않는다. 이 구분이 없으면 깊이가 없는 장면이
// 조용히 최고 신뢰도를 받는다 - 22.5 의 no_output 이 바닥값 점수를 받은 것과
// 같은 종류의 사고다.
TEST(DirectAligner, DepthConsistencyIsNotJudgedWithoutCurDepth) {
    const auto scene = makeScene(Vec3(0.010, -0.015, 0.008), Vec3(0.05, -0.03, 0.02));
    ASSERT_TRUE(scene.cur.depth.empty());
    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok());
    EXPECT_LT(r.value().depth_consistency, 0.0);
    EXPECT_FALSE(r.value().depthConsistent()) << "판정 불가가 합격으로 읽혔다";
}

// 신호를 만들어 놓고 아무도 안 읽으면 없는 것과 같다. 25.5 가 "노출되었을 뿐
// 소비되지 않는다" 로 남겨 둔 항목이고, 여기서 소비 경로가 실제로 도는지 본다.
TEST(DirectAligner, GeometricInconsistencyDegradesTheResult) {
    const auto scene = sceneWithCurDepth(Vec3(0.010, -0.015, 0.008),
                                         Vec3(0.05, -0.03, 0.02));
    DirectAligner aligner;
    const auto clean = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(clean.ok());
    // 정상 장면은 강등되지 않아야 한다. 이게 깨지면 게이트가 상시 발동한다.
    EXPECT_FALSE(clean.degraded()) << "정상 정렬이 강등됐다 - 문턱이 너무 조이다";
    EXPECT_DOUBLE_EQ(clean.reliability(), 1.0);

    Scene bad = scene;
    bad.cur.depth = scene.cur.depth.clone();
    bad.cur.depth *= 1.15f;

    DirectAligner aligner2;
    const auto r = aligner2.align(bad.ref, bad.cur, SE3::identity());
    ASSERT_TRUE(r.ok()) << "강등이지 실패가 아니다 - 포즈는 여전히 쓸 수 있다";
    EXPECT_TRUE(r.degraded()) << "기하가 15 % 어긋났는데 강등되지 않았다";
    EXPECT_LT(r.reliability(), 1.0);
    EXPECT_GT(r.reliability(), 0.0);
}

// 판정 불가를 나쁨으로 접지 않는다. 깊이 없는 정상 장면이 전부 열등해지면
// 그 게이트는 곧 꺼진다.
TEST(DirectAligner, UnjudgedGeometryDoesNotDegrade) {
    const auto scene = makeScene(Vec3(0.010, -0.015, 0.008), Vec3(0.05, -0.03, 0.02));
    ASSERT_TRUE(scene.cur.depth.empty());
    DirectAligner aligner;
    const auto r = aligner.align(scene.ref, scene.cur, SE3::identity());
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(r.degraded()) << "판정하지 못한 것을 나쁘다고 보고했다";
}
