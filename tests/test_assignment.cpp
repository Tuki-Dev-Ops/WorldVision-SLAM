// 이분 할당 검증.
// 탐욕적 매칭이 틀리는 경우를 반드시 포함해야 한다. 그게 이 코드가 존재하는 이유다.

#include "wme/core/Assignment.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>

using namespace wme;

TEST(Assignment, EmptyInputs) {
    EXPECT_EQ(solveAssignment({}, 0, 0).matched, 0u);
    EXPECT_EQ(solveAssignment({1.0}, 1, 0).matched, 0u);
    EXPECT_EQ(solveAssignment({1.0}, 0, 1).matched, 0u);
}

TEST(Assignment, IdentityMatrixMatchesDiagonal) {
    // 대각이 0, 나머지가 1 -> 대각이 유일 최적
    constexpr std::size_t n = 5;
    std::vector<double> cost(n * n, 1.0);
    for (std::size_t i = 0; i < n; ++i) cost[i * n + i] = 0.0;

    const auto r = solveAssignment(cost, n, n);
    EXPECT_EQ(r.matched, n);
    EXPECT_DOUBLE_EQ(r.total_cost, 0.0);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(r.row_to_col[i], static_cast<int>(i));
}

TEST(Assignment, BeatsGreedyOnTrapCase) {
    // 탐욕이 실패하는 고전 구조:
    //   행0 은 열0(1) 을 탐내지만, 그러면 행1 은 열1(100) 밖에 못 쓴다.
    //   전역 최적은 행0->열1(2), 행1->열0(3) 로 총 5.
    const std::vector<double> cost = {
        1.0,   2.0,
        3.0, 100.0,
    };
    const auto r = solveAssignment(cost, 2, 2);
    EXPECT_EQ(r.matched, 2u);
    EXPECT_DOUBLE_EQ(r.total_cost, 5.0);
    EXPECT_EQ(r.row_to_col[0], 1);
    EXPECT_EQ(r.row_to_col[1], 0);
}

TEST(Assignment, RectangularMoreColumnsThanRows) {
    // 검출 2개, 토큰 4개 -> 검출은 모두 배정되고 토큰 2개가 남는다
    const std::vector<double> cost = {
        9.0, 1.0, 9.0, 9.0,
        9.0, 9.0, 9.0, 2.0,
    };
    const auto r = solveAssignment(cost, 2, 4);
    EXPECT_EQ(r.matched, 2u);
    EXPECT_EQ(r.row_to_col[0], 1);
    EXPECT_EQ(r.row_to_col[1], 3);
    EXPECT_DOUBLE_EQ(r.total_cost, 3.0);
}

TEST(Assignment, RectangularMoreRowsThanColumns) {
    // 검출 4개, 토큰 2개 -> 검출 2개는 미배정(신규 토큰 후보)
    const std::vector<double> cost = {
        5.0, 9.0,
        1.0, 9.0,
        9.0, 9.0,
        9.0, 2.0,
    };
    const auto r = solveAssignment(cost, 4, 2);
    EXPECT_EQ(r.matched, 2u);
    EXPECT_EQ(r.row_to_col[1], 0);
    EXPECT_EQ(r.row_to_col[3], 1);
    EXPECT_EQ(r.row_to_col[0], -1);
    EXPECT_EQ(r.row_to_col[2], -1);
}

TEST(Assignment, InfeasiblePairsAreNeverAssigned) {
    // 게이트를 통과 못한 쌍은 절대 배정되면 안 된다.
    // 다른 선택지가 없어도 미배정으로 남아야 새 토큰이 만들어진다.
    const std::vector<double> cost = {
        kInfeasible, kInfeasible,
        kInfeasible, 1.0,
    };
    const auto r = solveAssignment(cost, 2, 2);
    EXPECT_EQ(r.matched, 1u);
    EXPECT_EQ(r.row_to_col[0], -1);
    EXPECT_EQ(r.row_to_col[1], 1);
}

TEST(Assignment, AllInfeasible) {
    const std::vector<double> cost(9, kInfeasible);
    const auto r = solveAssignment(cost, 3, 3);
    EXPECT_EQ(r.matched, 0u);
    for (int c : r.row_to_col) EXPECT_EQ(c, -1);
}

TEST(Assignment, MatchesBruteForceOnSmallRandomProblems) {
    // 작은 문제에서 완전탐색과 결과 비용이 일치해야 한다
    std::mt19937 gen(9001);
    std::uniform_real_distribution<double> d(0.0, 10.0);

    for (int trial = 0; trial < 200; ++trial) {
        constexpr std::size_t n = 5;
        std::vector<double> cost(n * n);
        for (auto& c : cost) c = d(gen);

        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        double best = 1e18;
        do {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += cost[i * n + perm[i]];
            best = std::min(best, sum);
        } while (std::next_permutation(perm.begin(), perm.end()));

        const auto r = solveAssignment(cost, n, n);
        EXPECT_EQ(r.matched, n);
        EXPECT_NEAR(r.total_cost, best, 1e-9) << "trial " << trial;
    }
}

TEST(Assignment, ConsistentRowAndColumnMaps) {
    std::mt19937 gen(4711);
    std::uniform_real_distribution<double> d(0.0, 5.0);

    std::vector<double> cost(7 * 9);
    for (auto& c : cost) c = d(gen);

    const auto r = solveAssignment(cost, 7, 9);
    for (std::size_t i = 0; i < 7; ++i) {
        if (r.row_to_col[i] < 0) continue;
        EXPECT_EQ(r.col_to_row[static_cast<std::size_t>(r.row_to_col[i])], static_cast<int>(i));
    }
}
