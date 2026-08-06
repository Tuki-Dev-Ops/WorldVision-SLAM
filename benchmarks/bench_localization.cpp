// ECDA / 연관 지연 측정.
// 60 FPS 예산 16.6 ms 안에서 Tier 0 이 차지하는 몫과, 스레딩/피라미드/점 개수의
// 실제 효과를 확인한다. 결정성 보장을 위해 고정 블록 누산을 쓰므로 워커 수를
// 늘려도 결과는 같고 시간만 줄어야 한다.

#include "wme/core/Assignment.hpp"
#include "wme/localization/DirectAligner.hpp"

#include <benchmark/benchmark.h>
#include <opencv2/imgproc.hpp>

#include <random>
#include <vector>

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

cv::Mat makeTexture() {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<int> px(0, kW - 1), py(0, kH - 1);
    std::uniform_int_distribution<int> rad(6, 28), col(30, 225);

    cv::Mat img(kH, kW, CV_8UC1, cv::Scalar(110));
    for (int i = 0; i < 900; ++i) {
        cv::circle(img, {px(gen), py(gen)}, rad(gen), cv::Scalar(col(gen)), cv::FILLED);
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

struct Pair {
    Frame ref, cur;
};

Pair makePair() {
    const CameraIntrinsics K = makeK();
    const Vec3 n = Vec3(0.15, 0.10, 0.98).normalized();
    const double dist = 2.5;

    cv::Mat depth(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < kW; ++x) {
            const Vec3 ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0);
            row[x] = static_cast<float>(dist / n.dot(ray));
        }
    }

    const SE3 T(SO3::exp(Vec3(0.01, -0.012, 0.006)), Vec3(0.06, -0.03, 0.02));
    Mat3 Km;
    Km << K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1;
    const Mat3 H = Km * (T.rotation().matrix() + T.translation() * n.transpose() / dist) *
                   Km.inverse();
    cv::Mat h(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) h.at<double>(r, c) = H(r, c);
    }

    const cv::Mat tex = makeTexture();
    cv::Mat warped;
    cv::warpPerspective(tex, warped, h, tex.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    Pair p;
    p.ref.gray = tex;   p.ref.depth = depth;
    p.ref.intrinsics = K; p.ref.stamp = Timestamp::fromSeconds(1.0);
    p.cur.gray = warped;
    p.cur.intrinsics = K; p.cur.stamp = Timestamp::fromSeconds(1.033);
    return p;
}

}  // namespace

// 피라미드 레벨 수에 따른 비용
static void BM_EcdaLevels(benchmark::State& state) {
    const Pair p = makePair();
    DirectAlignerConfig cfg;
    cfg.pyramid_levels = static_cast<int>(state.range(0));
    DirectAligner aligner(cfg);

    for (auto _ : state) {
        benchmark::DoNotOptimize(aligner.align(p.ref, p.cur, SE3::identity()));
    }
    state.SetLabel("levels=" + std::to_string(state.range(0)));
}
BENCHMARK(BM_EcdaLevels)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)
    ->Unit(benchmark::kMillisecond);

// 격자 셀 크기 = 점 밀도. 정확도/비용 트레이드오프의 주 손잡이.
static void BM_EcdaPointDensity(benchmark::State& state) {
    const Pair p = makePair();
    DirectAlignerConfig cfg;
    cfg.grid_cell = static_cast<int>(state.range(0));
    DirectAligner aligner(cfg);

    for (auto _ : state) {
        benchmark::DoNotOptimize(aligner.align(p.ref, p.cur, SE3::identity()));
    }
    state.SetLabel("cell=" + std::to_string(state.range(0)) + "px");
}
BENCHMARK(BM_EcdaPointDensity)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Unit(benchmark::kMillisecond);

// 워커 수에 따른 확장성. 결과값은 워커 수와 무관해야 한다(결정성).
static void BM_EcdaThreads(benchmark::State& state) {
    const Pair p = makePair();
    ThreadPool pool(static_cast<unsigned>(state.range(0)));
    DirectAligner aligner({}, &pool);

    for (auto _ : state) {
        benchmark::DoNotOptimize(aligner.align(p.ref, p.cur, SE3::identity()));
    }
    state.SetLabel(std::to_string(state.range(0)) + " workers");
}
BENCHMARK(BM_EcdaThreads)->Arg(1)->Arg(2)->Arg(4)->Arg(8)
    ->Unit(benchmark::kMillisecond);

// 헝가리안 확장성. 검출 x 토큰 규모가 커질 때 연관이 병목이 되는 지점을 찾는다.
static void BM_Assignment(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::mt19937 gen(77);
    std::uniform_real_distribution<double> d(0.0, 20.0);

    std::vector<double> cost(n * n);
    for (auto& c : cost) c = d(gen);

    for (auto _ : state) {
        benchmark::DoNotOptimize(solveAssignment(cost, n, n));
    }
    state.SetLabel(std::to_string(n) + "x" + std::to_string(n));
}
BENCHMARK(BM_Assignment)->Arg(8)->Arg(32)->Arg(64)->Arg(128)->Arg(256)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
