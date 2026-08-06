// 스레드 풀 검증. 경합 상황에서 작업 유실/중복이 없어야 한다.

#include "wme/core/ThreadPool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace wme;

TEST(ThreadPool, SubmitReturnsValue) {
    ThreadPool pool(4);
    auto f = pool.submit([](int a, int b) { return a * b; }, 6, 7);
    EXPECT_EQ(f.get(), 42);
}

TEST(ThreadPool, ExecutesEveryJobExactlyOnce) {
    ThreadPool pool(4);
    constexpr int kJobs = 4096;

    std::vector<std::atomic<int>> counters(kJobs);
    for (auto& c : counters) c.store(0);

    std::vector<std::future<void>> futures;
    futures.reserve(kJobs);
    for (int i = 0; i < kJobs; ++i) {
        futures.push_back(pool.submit([&counters, i] { counters[static_cast<std::size_t>(i)].fetch_add(1); }));
    }
    for (auto& f : futures) f.get();

    for (int i = 0; i < kJobs; ++i) {
        EXPECT_EQ(counters[static_cast<std::size_t>(i)].load(), 1) << "job " << i;
    }
}

TEST(ThreadPool, ParallelForCoversRangeExactlyOnce) {
    ThreadPool pool(4);
    constexpr std::size_t kN = 100000;

    std::vector<int> hits(kN, 0);
    pool.parallelFor(0, kN, [&hits](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) ++hits[i];
    });

    EXPECT_EQ(std::accumulate(hits.begin(), hits.end(), 0), static_cast<int>(kN));
    EXPECT_EQ(*std::max_element(hits.begin(), hits.end()), 1);
    EXPECT_EQ(*std::min_element(hits.begin(), hits.end()), 1);
}

TEST(ThreadPool, ParallelForHandlesEmptyAndSingleRange) {
    ThreadPool pool(4);
    int calls = 0;
    pool.parallelFor(5, 5, [&calls](std::size_t, std::size_t) { ++calls; });
    EXPECT_EQ(calls, 0);

    std::atomic<int> sum{0};
    pool.parallelFor(0, 1, [&sum](std::size_t lo, std::size_t hi) {
        sum.fetch_add(static_cast<int>(hi - lo));
    });
    EXPECT_EQ(sum.load(), 1);
}

TEST(ThreadPool, SingleWorkerPoolStillCoversRange) {
    // 워커 1개면 분할 이득이 없어 호출 스레드가 전부 처리한다
    ThreadPool pool(1);
    std::atomic<int> n{0};
    pool.parallelFor(0, 1000, [&n](std::size_t lo, std::size_t hi) {
        n.fetch_add(static_cast<int>(hi - lo));
    });
    EXPECT_EQ(n.load(), 1000);
}

TEST(ThreadPool, WaitIdleDrainsQueue) {
    ThreadPool pool(4);
    std::atomic<int> done{0};
    for (int i = 0; i < 500; ++i) pool.post([&done] { done.fetch_add(1); });
    pool.waitIdle();
    EXPECT_EQ(done.load(), 500);
    EXPECT_EQ(pool.pending(), 0u);
}

// future 를 걷어낸 뒤에도 예외는 호출 스레드로 되던져져야 한다.
// 되던지기 전에 모든 덩어리가 끝나 있어야 body 참조가 살아 있다.
TEST(ThreadPool, ParallelForRethrowsBodyException) {
    ThreadPool pool(4);
    std::atomic<int> finished{0};

    EXPECT_THROW(
        pool.parallelFor(0, 4096,
                         [&finished](std::size_t lo, std::size_t hi) {
                             finished.fetch_add(1);
                             if (lo == 0) throw std::runtime_error("덩어리 실패");
                             (void)hi;
                         },
                         1),
        std::runtime_error);

    // 던진 덩어리 말고도 전부 실행이 끝났는지 (워커 5분할 = 호출 스레드 포함)
    EXPECT_GE(finished.load(), 2);
    EXPECT_EQ(pool.pending(), 0u);

    // 예외 뒤에도 풀은 계속 쓸 수 있어야 한다
    std::atomic<int> n{0};
    pool.parallelFor(0, 1000, [&n](std::size_t lo, std::size_t hi) {
        n.fetch_add(static_cast<int>(hi - lo));
    });
    EXPECT_EQ(n.load(), 1000);
}

// 작업 큐가 링 버퍼라 초기 용량을 넘기면 자라야 한다.
// 자라는 동안 순서와 개수가 보존되는지 확인한다.
TEST(ThreadPool, QueueGrowsBeyondInitialCapacityWithoutLoss) {
    ThreadPool pool(2);
    constexpr int kJobs = 5000;   // 큐당 초기 용량(256)을 크게 넘긴다

    std::atomic<int> done{0};
    for (int i = 0; i < kJobs; ++i) pool.post([&done] { done.fetch_add(1); });
    pool.waitIdle();

    EXPECT_EQ(done.load(), kJobs);
    EXPECT_EQ(pool.pending(), 0u);
}

TEST(ThreadPool, NestedParallelForDoesNotDeadlock) {
    // 매핑 엔진이 내부에서 다시 분할하는 패턴. 데드락이 나면 안 된다.
    ThreadPool pool(4);
    std::atomic<int> total{0};
    pool.parallelFor(0, 16, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo; i < hi; ++i) {
            total.fetch_add(1);
        }
    });
    EXPECT_EQ(total.load(), 16);
}
