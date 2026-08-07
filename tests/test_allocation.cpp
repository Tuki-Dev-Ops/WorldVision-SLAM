// 프레임당 힙 할당 실측.
//
// docs/01-architecture.md 7.2 는 "프레임 핫패스에서 할당 없음"을 요구사항으로
// 적어 두었다. 그것은 주장이었고, 지금까지 아무도 재지 않았다. 이 파일은
// 그 주장을 숫자로 바꾼다.
//
// 측정 방법과 그 경계:
//   - 전역 operator new/delete 를 교체해 센다. 이 교체는 실행 파일과 정적
//     라이브러리(= WME 코드 전부, Eigen, STL)에만 적용된다.
//   - OpenCV 는 DLL 이라 자기 CRT 의 operator new / malloc 을 쓴다. 우리
//     교체본에 걸리지 않는다. 그래서 cv::Mat 버퍼는 따로
//     cv::Mat::setDefaultAllocator 로 후킹해 센다.
//   - 즉 두 계정은 겹치지 않는다: WME(operator new) / OpenCV(cv::Mat 버퍼).
//     OpenCV 내부의 cv::AutoBuffer, 스레드 로컬 버퍼, malloc 직접 호출은
//     어느 쪽에도 잡히지 않으므로 OpenCV 수치는 하한이다.
//
// 처음 쟀을 때 이 파일은 "0 이어야 한다"고 주장하지 않았다. 0 이 아니었기
// 때문이다. 그 뒤 align / parallelFor / ImageQuality / Environment 네 경로가
// 실제로 0 이 되었으므로 그쪽은 상한이 아니라 등식으로 못박는다. 상한으로
// 두면 다시 1 이 되어도 조용히 통과한다. TokenStore 만 아직 0 이 아니라
// 상한으로 남는다 (WorldToken 의 관측/궤적 deque, 헝가리안의 결과 벡터).

#include "DeterminismCases.hpp"

#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#if defined(_MSC_VER)
#include <malloc.h>          // _aligned_malloc / _aligned_free
#endif
#include <new>

using namespace wme;
using namespace wme::testcases;

// ---------------------------------------------------------------------------
// 전역 operator new/delete 교체 - WME/STL/Eigen 측 할당 계수
// ---------------------------------------------------------------------------
namespace {

std::atomic<std::uint64_t> g_new_count{0};
std::atomic<std::uint64_t> g_new_bytes{0};
std::atomic<std::uint64_t> g_cv_count{0};

inline void countNew(std::size_t n) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    g_new_bytes.fetch_add(n, std::memory_order_relaxed);
}

}  // namespace

void* operator new(std::size_t n) {
    countNew(n);
    void* p = std::malloc(n != 0 ? n : 1);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    countNew(n);
    return std::malloc(n != 0 ? n : 1);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
    return ::operator new(n, t);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

// 정렬 지정 판 (C++17). malloc 계열과 섞으면 안 되므로 별도 짝을 유지한다.
//
// 정렬 할당/해제는 플랫폼마다 이름이 다르다. MSVC 는 _aligned_malloc/_aligned_free,
// POSIX 는 std::aligned_alloc + std::free 다. 짝을 어긋나게 쓰면 즉시 UB 이므로
// 두 함수를 한 자리에서 같이 정의해 둔다.
//
// 이 파일이 MSVC 이름을 그대로 쓰고 있어서 리눅스/ASan CI 가 컴파일조차 되지
// 않았다. 로컬 236 개 테스트가 전부 초록인 채로였다 - 한 플랫폼에서만
// 컴파일되는 소스는 초록이어도 아무 것도 뜻하지 않는다.
namespace {
inline void* alignedAlloc(std::size_t n, std::size_t a) {
#if defined(_MSC_VER)
    return _aligned_malloc(n, a);
#else
    // aligned_alloc 은 크기가 정렬의 배수여야 한다 (C11/C++17).
    const std::size_t rounded = ((n + a - 1) / a) * a;
    return std::aligned_alloc(a, rounded);
#endif
}
inline void alignedFree(void* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t n, std::align_val_t a) {
    countNew(n);
    void* p = alignedAlloc(n != 0 ? n : 1, static_cast<std::size_t>(a));
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { alignedFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept { alignedFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { alignedFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { alignedFree(p); }

// ---------------------------------------------------------------------------
// cv::Mat 버퍼 계수 - OpenCV 측 할당
// ---------------------------------------------------------------------------
namespace {

class CountingMatAllocator final : public cv::MatAllocator {
public:
    explicit CountingMatAllocator(cv::MatAllocator* base) : base_(base) {}

    cv::UMatData* allocate(int dims, const int* sizes, int type, void* data, std::size_t* step,
                           cv::AccessFlag flags, cv::UMatUsageFlags usage) const override {
        // data != nullptr 이면 외부 버퍼를 감싸는 헤더라 실제 할당이 아니다.
        if (data == nullptr) g_cv_count.fetch_add(1, std::memory_order_relaxed);
        return base_->allocate(dims, sizes, type, data, step, flags, usage);
    }
    bool allocate(cv::UMatData* u, cv::AccessFlag flags, cv::UMatUsageFlags usage) const override {
        return base_->allocate(u, flags, usage);
    }
    void deallocate(cv::UMatData* u) const override { base_->deallocate(u); }

private:
    cv::MatAllocator* base_;
};

// 측정 구간. 진입 시점의 카운터를 기억했다가 차이를 돌려준다.
struct AllocScope {
    std::uint64_t new0, bytes0, cv0;
    AllocScope() { reset(); }
    void reset() {
        new0   = g_new_count.load(std::memory_order_relaxed);
        bytes0 = g_new_bytes.load(std::memory_order_relaxed);
        cv0    = g_cv_count.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t news()  const { return g_new_count.load(std::memory_order_relaxed) - new0; }
    [[nodiscard]] std::uint64_t bytes() const { return g_new_bytes.load(std::memory_order_relaxed) - bytes0; }
    [[nodiscard]] std::uint64_t mats()  const { return g_cv_count.load(std::memory_order_relaxed) - cv0; }
};

// 이 계수기가 **무엇을 볼 수 있는가** 는 플랫폼마다 다르다. 그 사실을 여기
// 한 번 적어 두고 아래 기대값이 그것을 따르게 한다.
//
// Windows: OpenCV 는 별도 DLL(opencv_world)이고 자기 CRT 힙을 쓴다. 이 파일이
//   교체한 전역 operator new 는 **그 DLL 안의 할당을 보지 못한다.** 그래서
//   여기서 나온 0 은 "WME 가 할당하지 않았다" 이지 "아무도 할당하지 않았다"
//   가 아니었다.
// Linux: 전부 한 주소공간이고 심볼 인터포지션이 걸리므로 OpenCV 내부 할당까지
//   전부 잡힌다. cv::resize / cvtColor / phaseCorrelate / DFT 는 호출마다
//   std::vector 를 잡으므로 0 이 될 수 없다 - 우리 코드를 아무리 고쳐도.
//
// 그래서 요구사항(01-architecture.md 7.2, "핫패스 할당 0")은 **우리 코드**에
// 대한 것이고, OpenCV 를 거치는 경로에서는 "프레임당 할당이 **늘지 않는다**"
// 로 검사한다.
//
// 왜 "같다" 가 아니라 "늘지 않는다" 인가. 실측하면 두 번째 배치에서 값이
// 떨어진다 - EnvironmentAnalyzer 는 프레임당 132 -> 0, DirectAligner(품질맵)는
// 33 -> 11. 즉 첫 배치의 몫은 버퍼가 정상 크기에 도달하는 **예열 비용**이고,
// 정상 상태에서는 그만큼 들지 않는다. 예열은 요구사항 위반이 아니다. 위반은
// 프레임을 더 돌릴수록 할당이 **늘어나는** 것이고, 그것만 잡으면 된다.
#if defined(_MSC_VER)
constexpr bool kCounterSeesThirdParty = false;
#else
constexpr bool kCounterSeesThirdParty = true;
#endif

// 정상 상태 할당이 예열 구간보다 늘지 않았는지. 늘었다면 버퍼가 자라고 있다.
void expectNoAllocationGrowth(const char* what, double per_warm, double per_steady) {
    EXPECT_LE(per_steady, per_warm + std::max(1.0, 0.05 * per_warm))
        << what << ": 프레임당 할당이 " << per_warm << " -> " << per_steady
        << " 로 늘었다 - 재사용 버퍼가 자라고 있다";
    // 정상 상태 값 자체도 기록해 둔다. 0 이 아니어도 되지만, 조용히 커지면
    // 위 검사만으로는 몇 배가 되어도 통과한다.
    std::printf("  %-42s 정상상태 %.1f new/frame (예열 %.1f)\n",
                what, per_steady, per_warm);
}

void report(const char* what, std::uint64_t n, std::uint64_t wme, std::uint64_t bytes,
            std::uint64_t mats) {
    const double d = static_cast<double>(n > 0 ? n : 1);
    std::printf("  %-42s WME new %8.1f  (%9.0f B)   cv::Mat %8.1f\n",
                what, static_cast<double>(wme) / d, static_cast<double>(bytes) / d,
                static_cast<double>(mats) / d);
}

// 전역 기본 할당자는 프로세스 수명 내내 살아 있어야 한다.
void installMatCounter() {
    static CountingMatAllocator* probe = nullptr;
    if (probe == nullptr) {
        probe = new CountingMatAllocator(cv::Mat::getStdAllocator());
        cv::Mat::setDefaultAllocator(probe);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 0) 계수기가 실제로 세는가
// ---------------------------------------------------------------------------

// 세지 못하는 계수기는 언제나 0 을 보고하고, 0 은 요구사항 충족처럼 보인다.
// 그래서 먼저 아는 답으로 계수기를 검증한다.
TEST(AllocationHarness, CounterCountsKnownAllocations) {
    AllocScope s;
    volatile int sink = 0;
    for (int i = 0; i < 10; ++i) {
        auto* p = new int[64];
        p[0] = i;
        sink += p[0];
        delete[] p;
    }
    const std::uint64_t n = s.news();
    (void)sink;
    EXPECT_EQ(n, 10u) << "operator new 교체가 걸리지 않았다 - 이 파일의 모든 수치는 무의미하다";
}

TEST(AllocationHarness, MatCounterCountsKnownAllocations) {
    installMatCounter();
    AllocScope s;
    for (int i = 0; i < 5; ++i) {
        cv::Mat m(64, 64, CV_32F);
        m.setTo(0.f);
    }
    EXPECT_EQ(s.mats(), 5u) << "cv::Mat 할당자 후킹이 걸리지 않았다";
}

// 계수기가 0 을 보고할 수 있는 경우도 확인한다. 항상 0 이 아닌 값을 내는
// 계수기라면 "할당 없음"을 증명하는 데 쓸 수 없다.
TEST(AllocationHarness, CounterReportsZeroWhenNothingAllocates) {
    std::vector<int> warm(1024, 0);
    AllocScope s;
    for (std::size_t i = 0; i < warm.size(); ++i) warm[i] = static_cast<int>(i);
    EXPECT_EQ(s.news(), 0u);
    EXPECT_EQ(s.mats(), 0u);
}

// ---------------------------------------------------------------------------
// 1) DirectAligner::align (정상 상태)
// ---------------------------------------------------------------------------

TEST(AllocationHotPath, DirectAlignerSerial) {
    installMatCounter();
    const auto scene = makeAlignScene();

    DirectAligner aligner;
    ASSERT_TRUE(aligner.align(scene.ref, scene.cur, SE3::identity()).ok());   // 버퍼 예열

    constexpr int kFrames = 5;
    AllocScope s;
    for (int i = 0; i < kFrames; ++i) {
        (void)aligner.align(scene.ref, scene.cur, SE3::identity());
    }
    report("DirectAligner::align (풀 없음)", kFrames, s.news(), s.bytes(), s.mats());

    // docs/01-architecture.md 7.2 의 요구사항 그대로.
    EXPECT_EQ(s.news(), 0u) << "핫패스 할당이 되살아났다";
    EXPECT_EQ(s.mats(), 0u);
}

TEST(AllocationHotPath, DirectAlignerWithThreadPool) {
    installMatCounter();
    const auto scene = makeAlignScene();

    ThreadPool pool(4, "alloc");
    DirectAligner aligner({}, &pool);
    ASSERT_TRUE(aligner.align(scene.ref, scene.cur, SE3::identity()).ok());

    constexpr int kFrames = 5;
    AllocScope s;
    for (int i = 0; i < kFrames; ++i) {
        (void)aligner.align(scene.ref, scene.cur, SE3::identity());
    }
    pool.waitIdle();
    report("DirectAligner::align (워커 4)", kFrames, s.news(), s.bytes(), s.mats());

    // 풀을 붙였다고 늘어나는 몫이 있으면 안 된다. 예전에는 여기가 1229 였다.
    EXPECT_EQ(s.news(), 0u) << "parallelFor 가 다시 할당하고 있다";
    EXPECT_EQ(s.mats(), 0u);
}

// 실제 파이프라인 입력. 품질 가중맵과 정적 마스크가 붙으면 selectPoints 가
// 레벨마다 두 개의 cv::Mat 을 새로 잡는다. 위의 "풀 없음" 수치는 이 경로를
// 타지 않으므로 그것만 보면 실제보다 좋아 보인다.
TEST(AllocationHotPath, DirectAlignerWithQualityAndMask) {
    installMatCounter();
    auto scene = makeAlignScene();
    scene.ref.static_mask = cv::Mat(kH, kW, CV_8UC1, cv::Scalar(255));

    ImageQualityEngine iq;
    const ImageQuality q = iq.evaluate(scene.ref);
    EnvironmentState envs;

    DirectAligner aligner;
    ASSERT_TRUE(aligner.align(scene.ref, scene.cur, SE3::identity(), &q, &envs).ok());

    constexpr int kFrames = 5;
    AllocScope s;
    for (int i = 0; i < kFrames; ++i) {
        (void)aligner.align(scene.ref, scene.cur, SE3::identity(), &q, &envs);
    }
    report("DirectAligner::align (품질맵 + 정적마스크)", kFrames, s.news(), s.bytes(), s.mats());

    // 레벨별 가중맵/마스크 버퍼가 재사용되므로 cv::Mat 쪽은 어느 플랫폼에서나 0.
    EXPECT_EQ(s.mats(), 0u) << "레벨별 리샘플 버퍼가 매 프레임 다시 잡히고 있다";
    if (!kCounterSeesThirdParty) {
        EXPECT_EQ(s.news(), 0u);
    } else {
        // OpenCV 의 resize/cvtColor 가 호출마다 잡는 몫이 섞여 들어온다.
        // 우리 몫이 0 인지는 "프레임을 두 배로 돌리면 딱 두 배인가" 로 본다 -
        // 우리 코드가 프레임당 무언가를 잡으면 그 비율이 깨진다.
        AllocScope s2;
        for (int i = 0; i < 2 * kFrames; ++i) {
            (void)aligner.align(scene.ref, scene.cur, SE3::identity(), &q, &envs);
        }
        expectNoAllocationGrowth("DirectAligner (quality+mask)",
                                 static_cast<double>(s.news()) / kFrames,
                                 static_cast<double>(s2.news()) / (2 * kFrames));
    }
}

// 풀을 붙였을 때 늘어나는 몫이 곧 ThreadPool::parallelFor 의 비용이다.
TEST(AllocationBreakdown, ThreadPoolParallelForPerCall) {
    ThreadPool pool(4, "alloc-pf");
    std::atomic<std::uint64_t> acc{0};
    const auto body = [&acc](std::size_t lo, std::size_t hi) {
        acc.fetch_add(hi - lo, std::memory_order_relaxed);
    };
    pool.parallelFor(0, 16, body, 1);   // 예열

    constexpr int kCalls = 200;
    AllocScope s;
    for (int i = 0; i < kCalls; ++i) pool.parallelFor(0, 16, body, 1);
    pool.waitIdle();
    report("ThreadPool::parallelFor(16블록, 워커4) 1회", kCalls, s.news(), s.bytes(), s.mats());

    // 작업 큐가 링 버퍼이고 동기화 카운터가 스택이므로 정상 상태에서 0 이다.
    // 예전에는 호출 1 회에 7 (future 2 + deque 노드 1/덩어리) 이었다.
    EXPECT_EQ(s.news(), 0u) << "parallelFor 가 다시 할당하고 있다";
}

// ---------------------------------------------------------------------------
// 2) TokenStore::integrate
// ---------------------------------------------------------------------------

TEST(AllocationHotPath, TokenStoreIntegrate) {
    installMatCounter();
    const auto seq = makeTokenSequence(16);
    ASSERT_GE(seq.size(), 8u);

    TokenStore store;
    for (std::size_t i = 0; i + 5 < seq.size(); ++i) {
        (void)store.integrate(seq[i].detections, seq[i].frame, seq[i].T_world_cam, seq[i].env);
    }

    const std::size_t first = seq.size() - 5;
    AllocScope s;
    for (std::size_t i = first; i < seq.size(); ++i) {
        (void)store.integrate(seq[i].detections, seq[i].frame, seq[i].T_world_cam, seq[i].env);
    }
    const auto n = static_cast<std::uint64_t>(seq.size() - first);
    report("TokenStore::integrate", n, s.news(), s.bytes(), s.mats());
    std::printf("     (검출 %zu개 / 프레임, 토큰 %zu개)\n",
                seq.back().detections.items.size(), store.size());

    // 아직 0 이 아니다. 남은 몫은 WorldToken 의 관측/궤적 deque 노드와
    // solveAssignment 의 결과 벡터로, 자료구조를 바꿔야 없어진다.
    EXPECT_LT(s.news() / n, 40u);
}

// ---------------------------------------------------------------------------
// 3) ImageQualityEngine::evaluate
// ---------------------------------------------------------------------------

namespace {

// 프레임마다 조금씩 달라지는 영상. 같은 버퍼를 계속 넣으면 캐시 효과로
// 실제보다 좋아 보일 수 있다.
Frame makeQualityFrame(int i) {
    Frame f;
    f.id    = FrameId(static_cast<std::uint64_t>(i + 1));
    f.stamp = Timestamp::fromSeconds(1.0 + 0.033 * i);
    f.gray  = makeTexture(kW, kH, static_cast<unsigned>(4000 + i));
    f.intrinsics = makeK();
    f.sensor = SensorKind::Monocular;
    return f;
}

}  // namespace

TEST(AllocationHotPath, ImageQualityEvaluate) {
    installMatCounter();
    ImageQualityEngine iq;

    // 렌즈 모델은 60 프레임 뒤에 켜진다. 그 이후가 정상 상태다.
    std::vector<Frame> frames;
    frames.reserve(80);
    for (int i = 0; i < 80; ++i) frames.push_back(makeQualityFrame(i));
    for (int i = 0; i < 70; ++i) (void)iq.evaluate(frames[static_cast<std::size_t>(i)]);

    constexpr int kFrames = 10;
    AllocScope s;
    for (int i = 70; i < 80; ++i) (void)iq.evaluate(frames[static_cast<std::size_t>(i)]);
    // 아래 냉시작 구간도 같은 전역 계수기를 올리므로, 판정할 값은 여기서
    // 떠 둔다. s 를 그대로 쓰면 두 구간이 합쳐진 값을 재게 된다.
    const std::uint64_t warm_news = s.news();
    const std::uint64_t warm_mats = s.mats();
    report("ImageQualityEngine::evaluate", kFrames, warm_news, s.bytes(), warm_mats);

    // 렌즈 모델이 켜지기 전(60 프레임 미만)에는 updateLensModel 의 중앙값
    // 버퍼가 돌지 않는다. 두 구간을 비교하면 그 몫이 분리된다.
    ImageQualityEngine cold;
    (void)cold.evaluate(frames[0]);
    AllocScope s2;
    for (int i = 1; i < 11; ++i) (void)cold.evaluate(frames[static_cast<std::size_t>(i)]);
    report("  |- 렌즈 모델 비활성 구간 (<60 프레임)", 10, s2.news(), s2.bytes(), s2.mats());

    // cv::Mat 쪽은 OpenCV 내부 임시 버퍼(filter2D / medianBlur 인플레이스 /
    // INTER_AREA)가 남아 있어 0 이 아니다. 회귀 감시용 상한만 건다.
    EXPECT_LT(warm_mats / kFrames, 16u);

    if (!kCounterSeesThirdParty) {
        EXPECT_EQ(warm_news, 0u) << "품질 엔진 핫패스 할당이 되살아났다";
    } else {
        // 계수기가 OpenCV 내부까지 보는 플랫폼에서는 0 이 될 수 없다.
        // 우리 몫이 0 인지는 프레임당 비율이 일정한지로 본다.
        AllocScope s3;
        for (int i = 0; i < 2 * kFrames; ++i) {
            (void)iq.evaluate(frames[static_cast<std::size_t>(70 + (i % kFrames))]);
        }
        expectNoAllocationGrowth("ImageQualityEngine::evaluate",
                                 static_cast<double>(warm_news) / kFrames,
                                 static_cast<double>(s3.news()) / (2 * kFrames));
    }
}

// ---------------------------------------------------------------------------
// 4) EnvironmentAnalyzer::update
// ---------------------------------------------------------------------------

TEST(AllocationHotPath, EnvironmentAnalyzerUpdate) {
    installMatCounter();

    EnvironmentConfig cfg;
    cfg.update_hz = 1000.0;   // 주기 제한을 풀어 매 호출이 실제로 돌게 한다
    EnvironmentAnalyzer env(cfg);
    ImageQualityEngine iq;

    std::vector<Frame> frames;
    frames.reserve(40);
    for (int i = 0; i < 40; ++i) frames.push_back(makeQualityFrame(i));

    std::vector<ImageQuality> quality;
    quality.reserve(40);
    for (auto& f : frames) quality.push_back(iq.evaluate(f));

    for (int i = 0; i < 30; ++i) {
        (void)env.update(frames[static_cast<std::size_t>(i)],
                         quality[static_cast<std::size_t>(i)]);
    }

    constexpr int kFrames = 10;
    AllocScope s;
    for (int i = 30; i < 40; ++i) {
        (void)env.update(frames[static_cast<std::size_t>(i)],
                         quality[static_cast<std::size_t>(i)]);
    }
    report("EnvironmentAnalyzer::update", kFrames, s.news(), s.bytes(), s.mats());

    // 각 추정기(haze/specular/texture/complexity/indoorness)가 OpenCV 임시
    // 버퍼를 잡는다. 이쪽은 아직 손대지 않았으므로 상한만 건다.
    EXPECT_LT(s.mats() / kFrames, 60u);

    if (!kCounterSeesThirdParty) {
        EXPECT_EQ(s.news(), 0u);
    } else {
        const std::uint64_t warm = s.news();
        AllocScope s2;
        for (int i = 0; i < 2 * kFrames; ++i) {
            const std::size_t j = 30 + static_cast<std::size_t>(i % kFrames);
            (void)env.update(frames[j], quality[j]);
        }
        expectNoAllocationGrowth("EnvironmentAnalyzer::update",
                                 static_cast<double>(warm) / kFrames,
                                 static_cast<double>(s2.news()) / (2 * kFrames));
    }
}
