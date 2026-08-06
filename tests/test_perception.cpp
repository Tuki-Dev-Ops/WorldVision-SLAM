// 영상품질/환경 엔진 검증.
// 합성 영상으로 각 열화를 독립적으로 주입하고, 해당 지표만 반응하는지 본다.
// 절대값을 단정하지 않고 "열화 전후의 순서 관계"를 검증하는 방식이다.
// 조명/센서마다 절대 스케일은 달라지지만 순서 관계는 유지되어야 하기 때문이다.

#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <random>

using namespace wme;

namespace {

std::mt19937& rng() {
    static std::mt19937 gen(31337);
    return gen;
}

// 구조가 있는 기준 장면 (체커보드 + 원 + 사각형)
cv::Mat makeScene(int w = 640, int h = 480) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(90, 95, 100));
    for (int y = 0; y < h; y += 40) {
        for (int x = 0; x < w; x += 40) {
            if (((x / 40) + (y / 40)) % 2 == 0) {
                cv::rectangle(img, cv::Rect(x, y, 40, 40), cv::Scalar(180, 175, 170), cv::FILLED);
            }
        }
    }
    cv::circle(img, {w / 3, h / 2}, 60, cv::Scalar(40, 200, 60), cv::FILLED);
    cv::rectangle(img, cv::Rect(w / 2, h / 3, 120, 150), cv::Scalar(200, 60, 40), cv::FILLED);
    return img;
}

Frame makeFrame(const cv::Mat& bgr, double t_sec) {
    Frame f;
    f.id    = FrameId(static_cast<std::uint64_t>(t_sec * 1000.0));
    f.stamp = Timestamp::fromSeconds(t_sec);
    f.rgb   = bgr;
    cv::cvtColor(bgr, f.gray, cv::COLOR_BGR2GRAY);
    f.intrinsics.fx = 525.0; f.intrinsics.fy = 525.0;
    f.intrinsics.cx = bgr.cols * 0.5; f.intrinsics.cy = bgr.rows * 0.5;
    f.intrinsics.width = bgr.cols; f.intrinsics.height = bgr.rows;
    f.sensor = SensorKind::Monocular;
    return f;
}

cv::Mat darken(const cv::Mat& src, double gain) {
    cv::Mat out;
    src.convertTo(out, CV_8UC3, gain, 0.0);
    return out;
}

cv::Mat addNoise(const cv::Mat& src, double sigma) {
    cv::Mat noise(src.size(), CV_32FC3);
    std::normal_distribution<float> d(0.0f, static_cast<float>(sigma));
    for (int y = 0; y < noise.rows; ++y) {
        auto* row = noise.ptr<cv::Vec3f>(y);
        for (int x = 0; x < noise.cols; ++x) {
            row[x] = {d(rng()), d(rng()), d(rng())};
        }
    }
    cv::Mat f;
    src.convertTo(f, CV_32FC3);
    f += noise;
    cv::Mat out;
    f.convertTo(out, CV_8UC3);
    return out;
}

// 대기광 A 와 투과율 t 로 안개 모델 I = J*t + A*(1-t) 적용
cv::Mat addHaze(const cv::Mat& src, double transmission) {
    cv::Mat f;
    src.convertTo(f, CV_32FC3);
    const cv::Scalar A(225.0, 225.0, 225.0);
    cv::Mat out = f * transmission + cv::Mat(src.size(), CV_32FC3, A * (1.0 - transmission));
    cv::Mat u8;
    out.convertTo(u8, CV_8UC3);
    return u8;
}

// 수직에 가까운 밝은 줄무늬 = 비
cv::Mat addRain(const cv::Mat& src, int count) {
    cv::Mat out = src.clone();
    std::uniform_int_distribution<int> px(0, src.cols - 1), py(0, src.rows - 1);
    std::uniform_int_distribution<int> len(10, 22);
    for (int i = 0; i < count; ++i) {
        const int x = px(rng()), y = py(rng()), l = len(rng());
        cv::line(out, {x, y}, {x + 2, y + l}, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }
    return out;
}

// 등방성 밝은 점 = 눈
cv::Mat addSnow(const cv::Mat& src, int count) {
    cv::Mat out = src.clone();
    std::uniform_int_distribution<int> px(0, src.cols - 1), py(0, src.rows - 1);
    std::uniform_int_distribution<int> rad(2, 4);
    for (int i = 0; i < count; ++i) {
        cv::circle(out, {px(rng()), py(rng())}, rad(rng()), cv::Scalar(245, 245, 245), cv::FILLED);
    }
    return out;
}

}  // namespace

// --- ImageQualityEngine ----------------------------------------------------

TEST(ImageQuality, BlurReducesSharpness) {
    ImageQualityEngine engine;
    const cv::Mat scene = makeScene();

    const auto sharp = engine.evaluate(makeFrame(scene, 0.0));

    cv::Mat blurred;
    cv::GaussianBlur(scene, blurred, cv::Size(15, 15), 6.0);
    const auto soft = engine.evaluate(makeFrame(blurred, 0.1));

    EXPECT_GT(sharp.sharpness, soft.sharpness);
    EXPECT_GT(sharp.blur_free, soft.blur_free);
    EXPECT_GT(soft.blur_extent_px, sharp.blur_extent_px);
}

TEST(ImageQuality, NoiseIsDetected) {
    ImageQualityEngine engine;
    const cv::Mat scene = makeScene();

    const auto clean = engine.evaluate(makeFrame(scene, 0.0));
    const auto noisy = engine.evaluate(makeFrame(addNoise(scene, 14.0), 0.1));

    EXPECT_GT(noisy.noise_sigma, clean.noise_sigma);
    EXPECT_LT(noisy.noise_free, clean.noise_free);
}

TEST(ImageQuality, OverexposurePenalizesScore) {
    ImageQualityEngine engine;
    const cv::Mat scene = makeScene();

    cv::Mat blown;
    scene.convertTo(blown, CV_8UC3, 2.6, 60.0);

    const auto normal = engine.evaluate(makeFrame(scene, 0.0));
    const auto over   = engine.evaluate(makeFrame(blown, 0.1));

    EXPECT_GT(over.saturated_high_ratio, normal.saturated_high_ratio);
    EXPECT_LT(over.exposure, normal.exposure);
}

TEST(ImageQuality, SaturatedPixelsGetZeroWeight) {
    ImageQualityEngine engine;
    cv::Mat scene = makeScene();
    // 좌측 절반을 완전 포화시킨다
    scene(cv::Rect(0, 0, scene.cols / 2, scene.rows)).setTo(cv::Scalar(255, 255, 255));

    const auto q = engine.evaluate(makeFrame(scene, 0.0));
    ASSERT_FALSE(q.weight_map.empty());

    const int half = q.weight_map.cols / 2;
    const double left  = cv::mean(q.weight_map(cv::Rect(0, 0, half, q.weight_map.rows)))[0];
    const double right = cv::mean(q.weight_map(cv::Rect(half, 0, q.weight_map.cols - half,
                                                        q.weight_map.rows)))[0];
    EXPECT_LT(left, right * 0.2) << "포화 영역은 측광 정보가 없으므로 가중치가 거의 0 이어야 한다";
}

TEST(ImageQuality, PhotometricVarianceGrowsWithDegradation) {
    ImageQualityEngine engine;
    const cv::Mat scene = makeScene();

    const auto clean = engine.evaluate(makeFrame(scene, 0.0));

    cv::Mat bad;
    cv::GaussianBlur(addNoise(scene, 16.0), bad, cv::Size(11, 11), 4.0);
    const auto degraded = engine.evaluate(makeFrame(bad, 0.1));

    EXPECT_GT(degraded.photometricVariance(), clean.photometricVariance());
}

TEST(ImageQuality, LensDirtNeedsAccumulation) {
    // 오염 판정은 장기 관측이 필요하다. 첫 프레임에 단정하면 안 된다.
    ImageQualityEngine engine;
    const auto q = engine.evaluate(makeFrame(makeScene(), 0.0));
    EXPECT_DOUBLE_EQ(q.occlusion_free, 1.0);
}

// --- EnvironmentAnalyzer ---------------------------------------------------

TEST(Environment, DetectsDarkness) {
    ImageQualityEngine   iq;
    EnvironmentAnalyzer  env({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});

    const cv::Mat dark = darken(makeScene(), 0.10);
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(dark, i * 0.1);
        env.update(f, iq.evaluate(f));
    }
    EXPECT_GT(env.state().evidence.darkness, 0.5);
    EXPECT_TRUE(env.state().lighting == Lighting::Dark ||
                env.state().lighting == Lighting::LowLight);
}

TEST(Environment, DetectsHaze) {
    ImageQualityEngine   iq;
    EnvironmentAnalyzer  env({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});

    const cv::Mat clear = makeScene();
    EnvironmentAnalyzer clear_env({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(clear, i * 0.1);
        clear_env.update(f, iq.evaluate(f));
    }

    const cv::Mat hazy = addHaze(clear, 0.25);
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(hazy, i * 0.1);
        env.update(f, iq.evaluate(f));
    }

    EXPECT_GT(env.state().evidence.haze, clear_env.state().evidence.haze);
    EXPECT_LT(env.state().visibility, clear_env.state().visibility);
}

TEST(Environment, SeparatesRainFromSnow) {
    // 비는 방향성 줄무늬, 눈은 등방성 입자. 구조텐서 이방성으로 갈린다.
    ImageQualityEngine iq;
    const cv::Mat base = makeScene();

    EnvironmentAnalyzer rain_env({.analysis_width = 192, .history_size = 7, .update_hz = 1000.0});
    for (int i = 0; i < 16; ++i) {
        const Frame f = makeFrame(addRain(base, 400), i * 0.05);
        rain_env.update(f, iq.evaluate(f));
    }

    EnvironmentAnalyzer snow_env({.analysis_width = 192, .history_size = 7, .update_hz = 1000.0});
    for (int i = 0; i < 16; ++i) {
        const Frame f = makeFrame(addSnow(base, 300), i * 0.05);
        snow_env.update(f, iq.evaluate(f));
    }

    const auto& r = rain_env.state().evidence;
    const auto& s = snow_env.state().evidence;

    EXPECT_GT(r.rain_streak, 0.0);
    EXPECT_GT(s.snow_particle, 0.0);
    // 상대 비교: 비 쪽이 더 줄무늬답고, 눈 쪽이 더 입자답다
    EXPECT_GT(r.rain_streak / (r.snow_particle + 1e-6),
              s.rain_streak / (s.snow_particle + 1e-6));
}

TEST(Environment, StaticSceneShowsNoParticles) {
    // 정지 장면에서는 시간적 잔차가 0 이므로 비/눈 오검출이 없어야 한다
    ImageQualityEngine  iq;
    EnvironmentAnalyzer env({.analysis_width = 192, .history_size = 7, .update_hz = 1000.0});

    const cv::Mat scene = makeScene();
    for (int i = 0; i < 16; ++i) {
        const Frame f = makeFrame(scene, i * 0.05);
        env.update(f, iq.evaluate(f));
    }
    EXPECT_LT(env.state().evidence.rain_streak, 0.05);
    EXPECT_LT(env.state().evidence.snow_particle, 0.05);
    EXPECT_EQ(env.state().weather, Weather::Clear);
}

TEST(Environment, DegradationShiftsTierWeights) {
    // 적응 로직의 핵심 계약: 측광이 무너지면 정보 질량이 성좌/구조로 옮겨간다.
    ImageQualityEngine iq;
    const cv::Mat base = makeScene();

    EnvironmentAnalyzer good({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(base, i * 0.1);
        good.update(f, iq.evaluate(f));
    }

    EnvironmentAnalyzer bad({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});
    cv::Mat degraded;
    cv::GaussianBlur(addNoise(addHaze(darken(base, 0.2), 0.3), 12.0), degraded,
                     cv::Size(13, 13), 5.0);
    ImageQualityEngine iq_bad;
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(degraded, i * 0.1);
        bad.update(f, iq_bad.evaluate(f));
    }

    EXPECT_LT(bad.state().tier.photometric, good.state().tier.photometric);
    EXPECT_GT(bad.state().tier.motion_prior, good.state().tier.motion_prior);

    // 성좌 가중치는 측광보다 완만하게 떨어져야 한다 (YOLO 는 더 오래 버틴다).
    //
    // 반드시 상대 하락으로 비교한다. 두 tier 는 시작값이 다르므로(성좌는 1.0 에서,
    // 측광은 장면 텍스처에 따라 그보다 낮은 값에서 출발한다) 절대 하락폭을
    // 비교하면 헤드룸이 큰 쪽이 무조건 이긴다. "더 빨리 무너진다" 는 비율의 주장이다.
    const double photo_drop = 1.0 - bad.state().tier.photometric /
                                        std::max(1e-9, good.state().tier.photometric);
    const double const_drop = 1.0 - bad.state().tier.constellation /
                                        std::max(1e-9, good.state().tier.constellation);

    const auto& eg = good.state().evidence;
    const auto& eb = bad.state().evidence;
    std::cout << "  깨끗: photo=" << good.state().tier.photometric
              << " const=" << good.state().tier.constellation
              << " | blur=" << eg.motion_blur << " dark=" << eg.darkness
              << " haze=" << eg.haze << " noise=" << eg.noise
              << " tex=" << eg.texture_poverty << " shake=" << eg.camera_shake << "\n";
    std::cout << "  열화: photo=" << bad.state().tier.photometric
              << " const=" << bad.state().tier.constellation
              << " | blur=" << eb.motion_blur << " dark=" << eb.darkness
              << " haze=" << eb.haze << " noise=" << eb.noise
              << " tex=" << eb.texture_poverty << " shake=" << eb.camera_shake << "\n";

    EXPECT_GT(photo_drop, const_drop);
}

TEST(Environment, AdversityExtendsMemoryAndTracking) {
    ImageQualityEngine iq;
    const cv::Mat base = makeScene();

    EnvironmentAnalyzer good({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(base, i * 0.1);
        good.update(f, iq.evaluate(f));
    }

    EnvironmentAnalyzer bad({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});
    ImageQualityEngine iq_bad;
    const cv::Mat night = addNoise(darken(base, 0.08), 14.0);
    for (int i = 0; i < 12; ++i) {
        const Frame f = makeFrame(night, i * 0.1);
        bad.update(f, iq_bad.evaluate(f));
    }

    EXPECT_GT(bad.state().memory_retention_scale,  good.state().memory_retention_scale);
    EXPECT_GT(bad.state().track_persistence_scale, good.state().track_persistence_scale);
    EXPECT_LT(bad.state().detection_threshold_scale, good.state().detection_threshold_scale);
    EXPECT_GE(bad.state().detection_threshold_scale, 0.35);   // 하한은 지켜야 한다
}

TEST(Environment, RespectsUpdateRate) {
    // 5 Hz 설정이면 그 사이 프레임에서는 재계산하지 않고 직전 상태를 준다
    ImageQualityEngine  iq;
    EnvironmentAnalyzer env({.analysis_width = 192, .history_size = 5, .update_hz = 5.0});

    // Timestamp(0) 은 무효 시각 규약이므로 1초부터 시작한다
    const cv::Mat scene = makeScene();
    const Frame f0 = makeFrame(scene, 1.0);
    env.update(f0, iq.evaluate(f0));
    const Timestamp first = env.state().stamp;
    ASSERT_TRUE(first.valid());

    const Frame f1 = makeFrame(darken(scene, 0.05), 1.05);   // 50 ms 뒤
    env.update(f1, iq.evaluate(f1));
    EXPECT_EQ(env.state().stamp, first) << "주기 이내에는 재평가하지 않아야 한다";

    const Frame f2 = makeFrame(darken(scene, 0.05), 1.5);    // 500 ms 뒤
    env.update(f2, iq.evaluate(f2));
    EXPECT_NE(env.state().stamp, first);
}

TEST(Environment, ResetClearsState) {
    ImageQualityEngine  iq;
    EnvironmentAnalyzer env({.analysis_width = 192, .history_size = 5, .update_hz = 1000.0});

    const cv::Mat dark = darken(makeScene(), 0.1);
    for (int i = 0; i < 10; ++i) {
        const Frame f = makeFrame(dark, i * 0.1);
        env.update(f, iq.evaluate(f));
    }
    ASSERT_GT(env.state().evidence.darkness, 0.3);

    env.reset();
    EXPECT_DOUBLE_EQ(env.state().evidence.darkness, 0.0);
    EXPECT_EQ(env.state().lighting, Lighting::Unknown);
}
