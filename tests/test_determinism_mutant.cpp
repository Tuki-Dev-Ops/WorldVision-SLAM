// 결정성 하네스의 대조군.
//
// 이 실행 파일은 test_determinism 과 완전히 같은 케이스 코드를 쓰지만,
// 엔진 소스 세 곳에 순서 의존성을 일부러 주입한 사본으로 빌드된다
// (변이 목록과 생성 규칙은 tests/CMakeLists.txt 참조):
//
//   M1 DirectAligner : 축소 블록 수를 워커 수에 연동. 고정 블록(=16) 설계를
//                      되돌린 것과 같다. 부동소수 합산 묶음이 달라진다.
//   M2 Detection     : NMS 정렬을 stable_sort -> sort. 동점 타이브레이크가
//                      구현 정의 순서가 된다.
//   M3 TokenStore    : allTokens() 를 ID 가 아니라 shared_ptr 주소로 정렬.
//                      해시 순회 순서가 결과로 새어 나가던 상태로 되돌린다.
//
// 여기서 기대하는 것은 "다르다" 이다. 이 파일이 통과해야만
// test_determinism 의 "같다" 가 의미를 갖는다. 실패할 수 없는 테스트는
// 아무것도 말해 주지 않는다 (docs/06-results.md 10.4).

#include "DeterminismCases.hpp"

#include <gtest/gtest.h>

#include <iostream>

using namespace wme;
using namespace wme::testcases;

// M1: 워커 수에 따라 결과가 갈라져야 한다.
TEST(MutantControl, WorkerCountDifferenceIsDetected) {
    const auto scene = makeAlignScene();
    const test::Blob serial = alignBlob(scene, nullptr, 1);

    bool detected = false;
    for (unsigned workers : {1u, 2u, 3u, 5u, 8u}) {
        ThreadPool pool(workers, "mutant");
        const test::Blob parallel = alignBlob(scene, &pool, 1);
        if (parallel != serial) {
            std::cout << "  워커 " << workers << "개에서 차이 검출 (바이트 "
                      << serial.firstDifference(parallel) << ")\n";
            detected = true;
        }
    }
    EXPECT_TRUE(detected)
        << "주입한 워커 의존 축소를 하네스가 못 잡았다 - 정렬 결정성 검사는 판별력이 없다";
}

// M2: 동점 타이브레이크가 해석적 정답에서 벗어나야 한다.
TEST(MutantControl, NmsTieBreakDifferenceIsDetected) {
    const test::Blob actual   = nmsBlob(tiedDetections());
    const test::Blob expected = nmsAnalyticExpectation();
    EXPECT_NE(actual, expected)
        << "stable_sort 를 sort 로 바꿔도 결과가 같다 - 이 입력은 동점 규약을 판별하지 못한다";
    if (actual != expected) {
        std::cout << "  NMS 차이 검출 (바이트 " << actual.firstDifference(expected) << ")\n";
    }
}

// M3: 주소 정렬로 바뀌면 동시에 살아 있는 두 스토어의 출력 순서가 갈라져야 한다.
TEST(MutantControl, TokenOrderDifferenceIsDetected) {
    // 힙 배치를 여러 개 시도한다. 하나라도 차이가 나면 "이 검사는 주소 정렬
    // 변이를 검출할 수 있다" 가 성립한다.
    //
    // 배치 하나만 보면 안 되는 이유: 주소 순서가 id 순서와 우연히 일치하는
    // 힙에서는 변이가 있어도 결과가 같다. 실제로 이 대조군은 평소 통과하다가
    // 다른 프로세스가 메모리를 쓰는 동안 한 번 실패했다 - 알고리즘이 아니라
    // 할당자의 그날 기분을 재고 있었다는 뜻이다.
    for (std::size_t pad : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                            std::size_t{7}, std::size_t{16}}) {
        test::Blob a, b;
        tokenStoreInterleaved(a, b, pad);
        if (a != b) {
            std::cout << "  TokenStore 차이 검출 (pad " << pad << ", 바이트 "
                      << a.firstDifference(b) << ")\n";
            SUCCEED();
            return;
        }
    }
    ADD_FAILURE() << "다섯 가지 힙 배치 전부에서 allTokens() 주소 정렬 변이가 "
                     "같은 결과를 냈다 - 토큰 순서 검사는 판별력이 없다";
}
