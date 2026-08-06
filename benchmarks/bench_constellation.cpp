// TCG 확장성 측정.
// 검증할 주장: 역색인 검색은 지도 크기에 거의 무관하고, 비용은 후보 검증
// (클리크 탐색)에 지배된다. 이 주장이 깨지면 대규모 지도에서 쓸 수 없다.

#include "wme/token/ConstellationIndex.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>

using namespace wme;

namespace {

std::vector<ConstellationNode> makeRoom(std::mt19937& gen, int n, int classes, double extent) {
    std::uniform_real_distribution<double> pos(-extent, extent);
    std::vector<ConstellationNode> nodes;
    nodes.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ConstellationNode nd;
        nd.id       = TokenId(static_cast<std::uint64_t>(i + 1));
        nd.class_id = i % classes;
        nd.position = Vec3(pos(gen), pos(gen), pos(gen) * 0.3);
        nd.sigma    = 0.04;
        nodes.push_back(nd);
    }
    return nodes;
}

std::vector<ConstellationNode> viewFrom(const std::vector<ConstellationNode>& src, const SE3& T) {
    std::vector<ConstellationNode> out = src;
    for (auto& n : out) n.position = T * n.position;
    return out;
}

}  // namespace

// 장소 수에 따른 질의 지연
static void BM_QueryVsMapSize(benchmark::State& state) {
    const auto places = static_cast<int>(state.range(0));
    std::mt19937 gen(11);

    ConstellationIndex index;
    for (int i = 0; i < places; ++i) {
        index.insert(KeyframeId(static_cast<std::uint64_t>(i)), Timestamp::fromSeconds(i),
                     SE3::identity(), makeRoom(gen, 15, 8, 5.0));
    }
    const auto target = makeRoom(gen, 15, 8, 5.0);
    index.insert(KeyframeId(99999), Timestamp::fromSeconds(1.0), SE3::identity(), target);

    const SE3 T(SO3::exp(Vec3(0.1, 0.2, 0.05)), Vec3(0.5, -0.3, 0.1));
    const auto query = viewFrom(target, T.inverse());

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.query(query));
    }
    state.SetLabel(std::to_string(places) + " places");
    state.counters["places"] = places;
}
BENCHMARK(BM_QueryVsMapSize)->Arg(100)->Arg(1000)->Arg(5000)->Arg(20000)
    ->Unit(benchmark::kMicrosecond);

// 성좌 크기에 따른 검증 비용. 클리크 탐색이 지수적으로 터지지 않는지 본다.
static void BM_VerifyVsNodeCount(benchmark::State& state) {
    const auto n = static_cast<int>(state.range(0));
    std::mt19937 gen(23);

    ConstellationIndex index;
    const auto room = makeRoom(gen, n, std::max(3, n / 3), 6.0);
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.05, 0.1, 0.02)), Vec3(0.4, 0.2, 0.0));
    const auto query = viewFrom(room, T.inverse());

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.query(query));
    }
    state.SetLabel(std::to_string(n) + " nodes");
}
BENCHMARK(BM_VerifyVsNodeCount)->Arg(6)->Arg(10)->Arg(20)->Arg(30)->Arg(40)
    ->Unit(benchmark::kMicrosecond);

// 최악의 경우: 모든 객체가 같은 클래스 -> 대응 후보가 n^2 로 폭증
static void BM_WorstCaseSingleClass(benchmark::State& state) {
    const auto n = static_cast<int>(state.range(0));
    std::mt19937 gen(29);

    ConstellationIndex index;
    const auto room = makeRoom(gen, n, 1, 6.0);   // 클래스 1종뿐
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.05, 0.0, 0.0)), Vec3(0.3, 0.0, 0.0));
    const auto query = viewFrom(room, T.inverse());

    for (auto _ : state) {
        benchmark::DoNotOptimize(index.query(query));
    }
    state.SetLabel(std::to_string(n) + " same-class nodes");
}
BENCHMARK(BM_WorstCaseSingleClass)->Arg(8)->Arg(12)->Arg(16)->Arg(20)
    ->Unit(benchmark::kMicrosecond);

static void BM_Kabsch(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    std::mt19937 gen(31);
    std::uniform_real_distribution<double> d(-5.0, 5.0);

    std::vector<Vec3> src(n), dst(n);
    const SE3 T(SO3::exp(Vec3(0.3, -0.2, 0.5)), Vec3(1.0, 2.0, -0.5));
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = Vec3(d(gen), d(gen), d(gen));
        dst[i] = T * src[i];
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(kabsch(src, dst));
    }
}
BENCHMARK(BM_Kabsch)->Arg(4)->Arg(10)->Arg(50)->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
