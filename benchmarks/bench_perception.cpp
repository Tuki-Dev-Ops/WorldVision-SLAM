// 프레임당 지연 측정. 60 FPS 예산(16.6 ms) 안에서 인지 계층이 차지하는 몫을 본다.
// 평균이 아니라 p99 를 보고해야 한다. 실시간 시스템을 깨는 건 꼬리 지연이다.

#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace wme;

namespace {

cv::Mat makeScene(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    std::mt19937 gen(7);
    std::uniform_int_distribution<int> c(0, 255);
    for (int y = 0; y < h; y += 32) {
        for (int x = 0; x < w; x += 32) {
            cv::rectangle(img, cv::Rect(x, y, 32, 32),
                          cv::Scalar(c(gen), c(gen), c(gen)), cv::FILLED);
        }
    }
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.2);
    return img;
}

Frame makeFrame(const cv::Mat& bgr, double t) {
    Frame f;
    f.stamp = Timestamp::fromSeconds(t);
    f.rgb = bgr;
    cv::cvtColor(bgr, f.gray, cv::COLOR_BGR2GRAY);
    f.intrinsics.fx = 525.0; f.intrinsics.fy = 525.0;
    f.intrinsics.cx = bgr.cols * 0.5; f.intrinsics.cy = bgr.rows * 0.5;
    f.intrinsics.width = bgr.cols; f.intrinsics.height = bgr.rows;
    return f;
}

}  // namespace

static void BM_ImageQuality(benchmark::State& state) {
    const int w = static_cast<int>(state.range(0));
    const int h = w * 3 / 4;
    const cv::Mat scene = makeScene(w, h);
    ImageQualityEngine engine;

    double t = 0.0;
    for (auto _ : state) {
        const Frame f = makeFrame(scene, t);
        t += 0.016;
        benchmark::DoNotOptimize(engine.evaluate(f));
    }
    state.SetLabel(std::to_string(w) + "x" + std::to_string(h));
}
BENCHMARK(BM_ImageQuality)->Arg(640)->Arg(1280)->Arg(1920)
    ->Unit(benchmark::kMicrosecond)
    ->ComputeStatistics("p99", [](const std::vector<double>& v) -> double {
        auto s = v;
        std::sort(s.begin(), s.end());
        return s[static_cast<std::size_t>(static_cast<double>(s.size()) * 0.99)];
    });

static void BM_EnvironmentAnalyzer(benchmark::State& state) {
    const cv::Mat scene = makeScene(1280, 960);
    ImageQualityEngine  iq;
    // 주기 제한을 끄고 최악의 경우(매 프레임 재평가) 비용을 측정
    EnvironmentAnalyzer env({.analysis_width = static_cast<int>(state.range(0)),
                             .history_size = 9, .update_hz = 1e6});

    double t = 0.0;
    for (auto _ : state) {
        const Frame f = makeFrame(scene, t);
        t += 0.016;
        const auto q = iq.evaluate(f);
        benchmark::DoNotOptimize(env.update(f, q));
    }
    state.SetLabel("analysis_width=" + std::to_string(state.range(0)));
}
BENCHMARK(BM_EnvironmentAnalyzer)->Arg(128)->Arg(192)->Arg(320)
    ->Unit(benchmark::kMicrosecond);

// 시간적 중앙값이 히스토리 길이에 선형인지 확인
static void BM_TemporalMedian(benchmark::State& state) {
    const cv::Mat scene = makeScene(640, 480);
    ImageQualityEngine  iq;
    EnvironmentAnalyzer env({.analysis_width = 192,
                             .history_size = static_cast<int>(state.range(0)),
                             .update_hz = 1e6});

    double t = 0.0;
    for (auto _ : state) {
        const Frame f = makeFrame(scene, t);
        t += 0.016;
        benchmark::DoNotOptimize(env.update(f, iq.evaluate(f)));
    }
    state.SetLabel("history=" + std::to_string(state.range(0)));
}
BENCHMARK(BM_TemporalMedian)->Arg(5)->Arg(9)->Arg(15)->Arg(25)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
