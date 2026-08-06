// 비트 단위 결정성 하네스.
//
// 검증 대상 (docs/01-architecture.md 7.5 "결정적 재현"):
//   - DirectAligner::align   : 워커 1개 vs N개, 반복 실행
//   - TokenStore::integrate  : 토큰 ID / 순서 / 믿음
//   - ConstellationIndex     : 등록과 질의
//   - nonMaxSuppression      : 신뢰도 동점
//
// 비교는 전부 memcmp 다. 허용오차 비교는 비결정적 시스템에서도 통과하므로
// 결정성 테스트로는 아무것도 재지 못한다. 기존
// test_direct_aligner.cpp:IsDeterministic 이 쓰는 EXPECT_NEAR(1e-12) 는
// 마지막 비트가 흔들리는 상태를 그대로 통과시킨다 - 그것이 이 파일이 따로
// 존재하는 이유다.
//
// 이 하네스가 실패할 수 있다는 증거는 test_determinism_mutant 가 만든다.
// 같은 케이스 코드를, 순서 의존성을 주입한 소스로 빌드해 같은 비교가
// 반드시 실패하는지 확인한다. 통과만 하는 테스트는 증거가 아니다.

#include "DeterminismCases.hpp"

#include <gtest/gtest.h>

#include <iostream>

using namespace wme;
using namespace wme::testcases;

namespace {

void expectBitIdentical(const test::Blob& a, const test::Blob& b, const char* what) {
    if (a == b) return;
    const std::size_t at = a.firstDifference(b);
    ADD_FAILURE() << what << " 가 비트 단위로 다르다. "
                  << "크기 " << a.size() << " vs " << b.size()
                  << ", 첫 불일치 바이트 " << at
                  << ", 요약 " << a.hex() << " vs " << b.hex();
}

}  // namespace

// ---------------------------------------------------------------------------
// 0) 비교기 자체가 판별하는가
// ---------------------------------------------------------------------------

// 1 ULP 차이를 통과시키는 비교기로는 결정성을 주장할 수 없다.
TEST(DeterminismHarness, ComparatorRejectsOneUlp) {
    const auto scene = makeAlignScene();
    const test::Blob base = alignBlob(scene, nullptr, 1);
    ASSERT_GT(base.size(), 8u);

    // 첫 double 은 ok/degraded/reliability 뒤에 온다. 어느 double 을 흔들어도
    // 반드시 걸려야 하므로 여러 자리를 시도한다.
    bool any_detected = false;
    for (std::size_t k = 1; k < 8; ++k) {
        const test::Blob perturbed = base.perturbedByOneUlp(k);
        if (perturbed != base) any_detected = true;
    }
    EXPECT_TRUE(any_detected) << "비교기가 1 ULP 차이를 못 잡는다 - 판별력 없음";
}

// ---------------------------------------------------------------------------
// 1) DirectAligner
// ---------------------------------------------------------------------------

TEST(DeterminismAligner, RepeatedRunsAreBitIdentical) {
    const auto scene = makeAlignScene();
    const test::Blob a = alignBlob(scene, nullptr, 3);
    const test::Blob b = alignBlob(scene, nullptr, 3);
    expectBitIdentical(a, b, "단일 스레드 반복 정렬");
    std::cout << "  align(단일) 요약 = " << a.hex() << "\n";
}

// 같은 인스턴스를 반복 호출해도 내부 버퍼 재사용이 결과를 바꾸면 안 된다.
// alignBlob 이 3 회를 모두 기록하므로 회차 간 차이는 여기서 드러난다.
TEST(DeterminismAligner, WarmBuffersDoNotChangeResult) {
    const auto scene = makeAlignScene();
    const test::Blob three = alignBlob(scene, nullptr, 3);

    // 1 회분 Blob 을 3 번 이어붙인 것과 같아야 한다.
    const test::Blob one = alignBlob(scene, nullptr, 1);
    ASSERT_EQ(three.size(), one.size() * 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(0, std::memcmp(three.data() + static_cast<std::size_t>(i) * one.size(),
                                 one.data(), one.size()))
            << i + 1 << "회차가 첫 회차와 다르다 - 내부 버퍼가 상태를 흘린다";
    }
}

// 고정 축소 블록(reduction_blocks = 16) 의 존재 이유가 바로 이것이다.
TEST(DeterminismAligner, WorkerCountDoesNotChangeResult) {
    const auto scene = makeAlignScene();

    const test::Blob serial = alignBlob(scene, nullptr, 2);

    for (unsigned workers : {1u, 2u, 3u, 5u, 8u}) {
        ThreadPool pool(workers, "det");
        const test::Blob parallel = alignBlob(scene, &pool, 2);
        expectBitIdentical(serial, parallel,
                           (std::string("워커 ") + std::to_string(workers) + "개 정렬").c_str());
    }
}

// 같은 풀을 여러 정렬기가 공유해도 결과가 흔들리면 안 된다.
TEST(DeterminismAligner, SharedPoolAcrossAlignersIsStable) {
    const auto scene = makeAlignScene();
    ThreadPool pool(4, "det-shared");

    const test::Blob a = alignBlob(scene, &pool, 2);
    const test::Blob b = alignBlob(scene, &pool, 2);
    expectBitIdentical(a, b, "공유 풀 정렬");
}

// ---------------------------------------------------------------------------
// 2) TokenStore
// ---------------------------------------------------------------------------

TEST(DeterminismTokenStore, InterleavedStoresProduceIdenticalWorlds) {
    test::Blob a, b;
    tokenStoreInterleaved(a, b);
    ASSERT_GT(a.size(), 0u);
    expectBitIdentical(a, b, "동시 실행된 두 TokenStore");
    std::cout << "  TokenStore 요약 = " << a.hex() << " (" << a.size() << " B)\n";
}

TEST(DeterminismTokenStore, RepeatedSequencesProduceIdenticalWorlds) {
    test::Blob a1, b1, a2, b2;
    tokenStoreInterleaved(a1, b1);
    tokenStoreInterleaved(a2, b2);
    expectBitIdentical(a1, a2, "반복 실행된 TokenStore");
}

// ---------------------------------------------------------------------------
// 3) ConstellationIndex
// ---------------------------------------------------------------------------

TEST(DeterminismConstellation, InterleavedIndicesProduceIdenticalResults) {
    test::Blob a, b;
    constellationInterleaved(a, b);
    ASSERT_GT(a.size(), 0u);
    expectBitIdentical(a, b, "동시 실행된 두 ConstellationIndex");
    std::cout << "  Constellation 요약 = " << a.hex() << " (" << a.size() << " B)\n";
}

TEST(DeterminismConstellation, RepeatedRunsProduceIdenticalResults) {
    test::Blob a1, b1, a2, b2;
    constellationInterleaved(a1, b1);
    constellationInterleaved(a2, b2);
    expectBitIdentical(a1, a2, "반복 실행된 ConstellationIndex");
}

// ---------------------------------------------------------------------------
// 4) nonMaxSuppression - 동점
// ---------------------------------------------------------------------------

TEST(DeterminismNms, TiedConfidencesRepeatBitIdentical) {
    const auto dets = tiedDetections();
    expectBitIdentical(nmsBlob(dets), nmsBlob(dets), "동점 NMS 반복");
}

// 동점 규약이 "입력 인덱스 순"이라면 답은 해석적으로 정해진다.
// 반복 일치만 보면 안정적으로 틀린 순서도 통과하므로 정답을 따로 못박는다.
TEST(DeterminismNms, TieBreakingMatchesTheAnalyticRule) {
    expectBitIdentical(nmsBlob(tiedDetections()), nmsAnalyticExpectation(),
                       "동점 NMS 결과");
}
