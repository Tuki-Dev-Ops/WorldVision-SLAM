// YOLO 백엔드 검증.
//
// 백엔드가 둘(OpenCV DNN, ONNX Runtime)이라 검증도 두 층이다:
//   1) 계약  - 없는 파일, 형상/클래스 수 불일치, 빈 프레임을 조용히 넘기지 않는가
//   2) 기하  - letterbox 왕복이 박스를 화면 안에 보존하는가
//   3) 일치  - 둘 다 로드되면 같은 프레임에서 같은 답을 내는가
//
// 검출 "품질" 은 여기서 주장하지 않는다. 합성 도형에 대한 정확도는 아무 뜻이 없다.
// 실데이터 검증은 tools/wme_tum_yolo 가 맡는다.

#include "wme/perception/YoloRuntimeCv.hpp"
#ifdef WME_HAS_ORT
#include "wme/perception/YoloRuntimeOrt.hpp"
#endif

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

using namespace wme;

namespace {

const char* kModel = "models/yolo11n.onnx";

bool modelExists() {
    std::ifstream f(kModel, std::ios::binary);
    return f.good();
}

YoloConfig baseConfig() {
    YoloConfig c;
    c.model_path = kModel;
    return c;
}

// 사용 가능한 백엔드를 만든다. ORT 를 먼저 시도한다 - OpenCV 의 ONNX 임포터는
// 최신 YOLO 를 못 읽는 경우가 있고(YOLO11 의 C2PSA 에서 실제로 막힌다),
// 그 경우 조용히 건너뛰는 대신 어느 쪽이 살아 있는지 남겨야 한다.
std::unique_ptr<IYoloRuntime> makeAny(YoloConfig c, std::string& which) {
#ifdef WME_HAS_ORT
    auto ort = YoloRuntimeOrt::create(c);
    if (ort.ok()) { which = "ORT"; return std::move(ort.value()); }
    std::cout << "  ORT 실패: " << ort.error().message() << "\n";
#else
    std::cout << "  ORT 미빌드 (WME_HAS_ORT 미정의)\n";
#endif
    auto cv_rt = YoloRuntimeCv::create(std::move(c));
    if (cv_rt.ok()) { which = "OpenCV DNN"; return std::move(cv_rt.value()); }
    which.clear();
    return nullptr;
}

Frame makeFrame(const cv::Mat& bgr) {
    Frame f;
    f.id    = FrameId(1);
    f.stamp = Timestamp::fromSeconds(1.0);
    f.rgb   = bgr;
    f.intrinsics.fx = 500; f.intrinsics.fy = 500;
    f.intrinsics.cx = bgr.cols * 0.5; f.intrinsics.cy = bgr.rows * 0.5;
    f.intrinsics.width = bgr.cols; f.intrinsics.height = bgr.rows;
    f.sensor = SensorKind::Monocular;
    return f;
}

cv::Mat busyImage(int w, int h) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(90, 110, 130));
    for (int i = 0; i < 40; ++i) {
        cv::rectangle(img, cv::Rect((i * 71) % std::max(1, w - 70),
                                    (i * 43) % std::max(1, h - 60), 60, 50),
                      cv::Scalar(200, 60, 40), cv::FILLED);
        cv::circle(img, {(i * 53) % w, (i * 37) % h}, 22, cv::Scalar(30, 200, 90), cv::FILLED);
    }
    return img;
}

}  // namespace

// --- 계약 -----------------------------------------------------------------

TEST(YoloRuntime, RejectsMissingModel) {
    YoloConfig c;
    c.model_path = "존재하지_않는_모델.onnx";
    EXPECT_FALSE(YoloRuntimeCv::create(c).ok());
#ifdef WME_HAS_ORT
    EXPECT_FALSE(YoloRuntimeOrt::create(c).ok());
#endif
}

TEST(YoloRuntime, RejectsEmptyPath) {
    EXPECT_FALSE(YoloRuntimeCv::create(YoloConfig{}).ok());
#ifdef WME_HAS_ORT
    EXPECT_FALSE(YoloRuntimeOrt::create(YoloConfig{}).ok());
#endif
}

TEST(YoloRuntime, RejectsClassCountMismatch) {
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    YoloConfig c = baseConfig();
    c.class_names = {"사람", "의자"};      // 모델은 80 클래스다
    std::string which;
    EXPECT_EQ(makeAny(c, which), nullptr)
        << "클래스 수가 달라도 통과하면 디코딩이 조용히 어긋난다";
}

TEST(YoloRuntime, LoadsAndReportsShape) {
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    std::string which;
    auto rt = makeAny(baseConfig(), which);
    ASSERT_NE(rt, nullptr) << "어떤 백엔드로도 모델을 열지 못했다";
    std::cout << "  사용 백엔드: " << which << "\n";
}

TEST(YoloRuntime, RejectsEmptyFrame) {
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    std::string which;
    auto rt = makeAny(baseConfig(), which);
    ASSERT_NE(rt, nullptr);

    Frame f;
    f.id = FrameId(1);
    EXPECT_FALSE(rt->infer(f).ok());
}

TEST(YoloRuntime, AcceptsGrayscale) {
    // TUM 은 흑백으로 읽어 쓴다. 여기서 막히면 실데이터 경로가 통째로 죽는다.
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    std::string which;
    auto rt = makeAny(baseConfig(), which);
    ASSERT_NE(rt, nullptr);

    Frame f;
    f.id    = FrameId(1);
    f.stamp = Timestamp::fromSeconds(1.0);
    f.gray  = cv::Mat(480, 640, CV_8UC1, cv::Scalar(120));
    f.intrinsics.width = 640; f.intrinsics.height = 480;
    const auto d = rt->infer(f);
    EXPECT_TRUE(d.ok()) << d.error().message();
}

// --- 기하 -----------------------------------------------------------------

TEST(YoloRuntime, BoxesStayInsideImageOnNonSquareInput) {
    // letterbox 를 잘못 되돌리면 박스가 화면 밖으로 나가거나 종횡비가 뒤틀린다.
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    YoloConfig c = baseConfig();
    c.confidence_threshold = 0.05f;      // 표본을 늘린다
    std::string which;
    auto rt = makeAny(c, which);
    ASSERT_NE(rt, nullptr);

    for (const auto sz : {cv::Size(1280, 360), cv::Size(360, 1280), cv::Size(640, 480)}) {
        const cv::Mat img = busyImage(sz.width, sz.height);
        const auto d = rt->infer(makeFrame(img));
        ASSERT_TRUE(d.ok()) << d.error().message();
        for (const auto& det : d.value().items) {
            EXPECT_GE(det.box.x, -0.01f);
            EXPECT_GE(det.box.y, -0.01f);
            EXPECT_LE(det.box.x + det.box.width, img.cols + 0.01f);
            EXPECT_LE(det.box.y + det.box.height, img.rows + 0.01f);
            EXPECT_GT(det.box.width, 0.f);
            EXPECT_GT(det.box.height, 0.f);
            EXPECT_EQ(det.class_scores.size(), 80u) << "분포를 통째로 남겨야 한다";
        }
        std::cout << "  " << sz.width << "x" << sz.height << ": 검출 "
                  << d.value().items.size() << " 개, " << d.value().inference_ms << " ms\n";
    }
}

TEST(YoloRuntime, ConfidenceScaleChangesDetectionCount) {
    // 환경 적응의 핵심 경로. 임계 배율이 검출 수를 바꾸지 않으면
    // detection_threshold_scale 이 죽은 채널이 된다.
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    std::string which;
    auto rt = makeAny(baseConfig(), which);
    ASSERT_NE(rt, nullptr);

    // 검출이 하나도 안 나오는 영상을 쓰면 0 >= 0 으로 공허하게 통과한다.
    // 판별하지 못하는 측정은 통과해도 아무것도 말해 주지 않는다.
    const Frame f = makeFrame(busyImage(1280, 360));

    rt->setConfidenceScale(2.0f);            // 임계를 올린다
    const auto hi = rt->infer(f);
    rt->setConfidenceScale(0.2f);            // 임계를 낮춘다
    const auto lo = rt->infer(f);
    ASSERT_TRUE(hi.ok() && lo.ok());

    ASSERT_GT(lo.value().items.size(), 0u)
        << "이 영상에서 검출이 0 이면 이 시험은 아무것도 재지 못한다";
    EXPECT_GT(lo.value().items.size(), hi.value().items.size())
        << "임계를 낮췄는데 검출이 늘지 않는다 - 배율이 반영되지 않는 것";
    std::cout << "  임계배율 1.0 -> " << hi.value().items.size()
              << " 개, 0.2 -> " << lo.value().items.size() << " 개\n";
}

TEST(YoloRuntime, IsDeterministic) {
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
    std::string which;
    auto rt = makeAny(baseConfig(), which);
    ASSERT_NE(rt, nullptr);

    const Frame f = makeFrame(busyImage(640, 480));
    const auto a = rt->infer(f);
    const auto b = rt->infer(f);
    ASSERT_TRUE(a.ok() && b.ok());
    ASSERT_EQ(a.value().items.size(), b.value().items.size());
    for (std::size_t i = 0; i < a.value().items.size(); ++i) {
        EXPECT_FLOAT_EQ(a.value().items[i].box.x, b.value().items[i].box.x);
        EXPECT_FLOAT_EQ(a.value().items[i].confidence, b.value().items[i].confidence);
    }
}

// --- 백엔드 일치 -----------------------------------------------------------

TEST(YoloRuntime, BackendsAgreeWhenBothAvailable) {
    // 두 백엔드가 같은 프레임에서 다르게 답하면 하나는 틀린 것이다.
    // 전처리/디코딩을 YoloDecode 한 벌로 공유하므로, 차이가 나면 추론 엔진 차이다.
    if (!modelExists()) GTEST_SKIP() << "모델 없음: " << kModel;
#ifndef WME_HAS_ORT
    GTEST_SKIP() << "ORT 미빌드 - 비교할 두 번째 백엔드가 없다";
#else
    auto ort = YoloRuntimeOrt::create(baseConfig());
    auto cvr = YoloRuntimeCv::create(baseConfig());
    if (!ort.ok() || !cvr.ok()) {
        GTEST_SKIP() << "한쪽만 모델을 연다 - ORT " << ort.ok()
                     << ", OpenCV " << cvr.ok();
    }

    const Frame f = makeFrame(busyImage(640, 480));
    const auto a = ort.value()->infer(f);
    const auto b = cvr.value()->infer(f);
    ASSERT_TRUE(a.ok() && b.ok());
    ASSERT_EQ(a.value().items.size(), b.value().items.size());
    for (std::size_t i = 0; i < a.value().items.size(); ++i) {
        EXPECT_NEAR(a.value().items[i].box.x, b.value().items[i].box.x, 1.0f);
        EXPECT_NEAR(a.value().items[i].confidence, b.value().items[i].confidence, 0.02f);
        EXPECT_EQ(a.value().items[i].class_id, b.value().items[i].class_id);
    }
#endif
}
