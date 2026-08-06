// 3-tier 융합 검증.
//
// 이 파일이 지켜야 할 것은 두 가지다.
//   1. 융합이 수학적으로 정보 가중 MAP 해와 같은가 (선형화점이 달라도).
//   2. 기권과 랭크 부족이 조용히 사라지지 않는가.
//
// 정확도 주장은 여기서 하지 않는다 - 그건 실데이터가 할 일이다.

#include "wme/fusion/PoseFusion.hpp"
#include "wme/fusion/TierInformation.hpp"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include <cmath>
#include <vector>

using namespace wme;
using namespace wme::fusion;

namespace {

TierEstimate make(Tier t, const SE3& T, const Mat6& info, double alpha = 1.0) {
    TierEstimate e;
    e.tier        = t;
    e.T_cur_ref   = T;
    e.information = info;
    e.alpha       = alpha;
    e.available   = true;
    e.reason      = Abstain::None;
    return e;
}

Mat6 diagInfo(double t, double r) {
    Vec6 d;
    d << t, t, t, r, r, r;
    return Mat6(d.asDiagonal());
}

// 융합 목적함수. 검증용으로 여기서 다시 쓴다 - 구현이 최소화하는 것과
// 같은 식을 독립적으로 적어야 "정말 최소인가" 를 물을 수 있다.
double cost(const std::vector<TierEstimate>& es, const SE3& T) {
    double c = 0.0;
    for (const auto& e : es) {
        if (!e.available) continue;
        const Vec6 eps = (T * e.T_cur_ref.inverse()).log();
        c += 0.5 * eps.dot(e.alpha * e.information * eps);
    }
    return c;
}

}  // namespace

// --- SE(3) 좌측 자코비안 ----------------------------------------------------
// 급수 구현이 정의와 맞는지 유한차분으로 못박는다. docs/06-results.md 7.1 이
// 닫힌 형태를 손으로 적다가 부호를 뒤집은 기록이므로, 여기서는 수치로 건다.
TEST(Se3LeftJacobian, MatchesFiniteDifference) {
    Vec6 xi;
    xi << 0.3, -0.2, 0.15, 0.7, -0.4, 1.1;   // |phi| ~ 1.37 rad, 작지 않다

    const Mat6 J = se3LeftJacobian(xi);
    const SE3  T = SE3::exp(xi);

    // d/du log( exp(xi + h e_k) * exp(xi)^-1 ) = J_l(xi) e_k
    const double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec6 d = Vec6::Zero();
        d(k) = h;
        const Vec6 plus  = (SE3::exp(Vec6(xi + d)) * T.inverse()).log();
        const Vec6 minus = (SE3::exp(Vec6(xi - d)) * T.inverse()).log();
        const Vec6 num   = (plus - minus) / (2.0 * h);
        EXPECT_LT((num - J.col(k)).norm(), 1e-6) << "열 " << k;
    }
}

TEST(Se3LeftJacobian, RotationBlockMatchesSo3) {
    Vec6 xi;
    xi << 0.0, 0.0, 0.0, 0.2, -0.9, 0.4;
    const Mat6 J = se3LeftJacobian(xi);
    const Mat3 So = SO3::leftJacobian(Vec3(xi.tail<3>()));
    EXPECT_LT((J.block<3, 3>(3, 3) - So).norm(), 1e-12);
    EXPECT_LT((J.block<3, 3>(0, 0) - So).norm(), 1e-12);
    EXPECT_LT((J.block<3, 3>(3, 0)).norm(), 1e-14);   // 블록 상삼각
}

TEST(Se3LeftJacobian, IdentityAtZero) {
    const Mat6 J = se3LeftJacobian(Vec6::Zero());
    EXPECT_LT((J - Mat6::Identity()).norm(), 1e-14);
    const Mat6 Ji = se3LeftJacobianInverse(Vec6::Zero());
    EXPECT_LT((Ji - Mat6::Identity()).norm(), 1e-14);
}

TEST(Se3LeftJacobian, InverseIsInverse) {
    Vec6 xi;
    xi << -1.4, 0.8, 2.2, 0.5, 2.9, -0.3;    // |phi| ~ 2.98, pi 에 가깝다
    const Mat6 J  = se3LeftJacobian(xi);
    const Mat6 Ji = se3LeftJacobianInverse(xi);
    EXPECT_LT((J * Ji - Mat6::Identity()).norm(), 1e-10);
}

// --- 융합의 기본 성질 -------------------------------------------------------

TEST(PoseFusion, IdenticalEstimatesAddInformation) {
    const SE3 T(SO3::exp(Vec3(0.05, -0.02, 0.01)), Vec3(0.1, 0.2, -0.05));
    const Mat6 L = diagInfo(1e4, 1e5);

    std::vector<TierEstimate> es{make(Tier::Photometric, T, L),
                                 make(Tier::Structural, T, L)};
    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    EXPECT_LT((r.value().T_cur_ref.inverse() * T).log().norm(), 1e-12);
    // 같은 점이므로 선형화점 이동이 없다 - 단순합과 옮긴 값이 같아야 한다.
    EXPECT_LT((r.value().information - 2.0 * L).norm() / (2.0 * L).norm(), 1e-12);
    EXPECT_LT((r.value().information - r.value().information_naive).norm(), 1e-9);
    EXPECT_EQ(r.value().contributing_tiers, 2);
}

TEST(PoseFusion, SingleTierIsPassthrough) {
    const SE3 T(SO3::exp(Vec3(0.3, 0.1, -0.2)), Vec3(1.0, -0.5, 0.25));
    std::vector<TierEstimate> es{make(Tier::Photometric, T, diagInfo(1e3, 1e4))};
    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    EXPECT_LT((r.value().T_cur_ref.inverse() * T).log().norm(), 1e-12);
    EXPECT_EQ(r.value().contributing_tiers, 1);
    EXPECT_TRUE(r.value().contributed(Tier::Photometric));
    EXPECT_FALSE(r.value().contributed(Tier::Constellation));
}

TEST(PoseFusion, DominantInformationWins) {
    // 정보가 1000 배 큰 쪽이 해를 지배해야 한다. 이것이 "정보로 융합한다" 의 뜻이고,
    // 포즈 평균이었다면 절반쯤에서 멈춘다.
    const SE3 A(SO3::identity(), Vec3(0.0, 0.0, 0.0));
    const SE3 B(SO3::identity(), Vec3(1.0, 0.0, 0.0));

    std::vector<TierEstimate> es{make(Tier::Photometric, A, diagInfo(1e6, 1e6)),
                                 make(Tier::Structural, B, diagInfo(1e3, 1e3))};
    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    const double x = r.value().T_cur_ref.translation().x();
    EXPECT_NEAR(x, 1e3 / (1e6 + 1e3), 1e-6);
    EXPECT_LT(x, 0.01);
}

TEST(PoseFusion, MinimisesTheObjectiveAtDifferentLinearisationPoints) {
    // 두 추정이 회전으로 크게 벌어져 있으면 접선 평균과 진짜 MAP 해가 다르다.
    // 구현이 실제로 목적함수의 최소점에 있는지 무작위 섭동으로 확인한다.
    const SE3 A(SO3::exp(Vec3(0.0, 0.0, 0.0)), Vec3(0.0, 0.0, 0.0));
    const SE3 B(SO3::exp(Vec3(0.0, 0.9, 0.0)), Vec3(0.4, -0.2, 0.1));

    std::vector<TierEstimate> es{make(Tier::Photometric, A, diagInfo(4e2, 9e2)),
                                 make(Tier::Structural, B, diagInfo(1e3, 2e2))};
    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());

    const SE3    Tstar = r.value().T_cur_ref;
    const double c0    = cost(es, Tstar);

    // 12 방향 * 3 크기. 어느 쪽으로도 내려갈 곳이 없어야 최소점이다.
    for (int k = 0; k < 6; ++k) {
        for (double s : {1e-4, 1e-3, 1e-2}) {
            for (double sign : {1.0, -1.0}) {
                Vec6 d = Vec6::Zero();
                d(k) = sign * s;
                EXPECT_GE(cost(es, SE3::exp(d) * Tstar), c0 - 1e-12)
                    << "축 " << k << " 크기 " << (sign * s);
            }
        }
    }
    // 그리고 단순 접선 평균과는 실제로 다르다 - 다르지 않다면 이 테스트는
    // 아무것도 재지 않는다.
    EXPECT_GT(std::abs(r.value().information.trace()
                       - r.value().information_naive.trace())
                  / r.value().information_naive.trace(), 1e-3);
}

// --- 퇴화 보완 --------------------------------------------------------------

TEST(PoseFusion, RankDeficientTierDoesNotMoveItsNullDirection) {
    // SPA 를 흉내낸다: x 축 병진을 전혀 구속하지 않는다 (복도 축).
    // 그 축의 해는 오직 Tier 0 이 정해야 한다.
    Mat6 spa = Mat6::Zero();
    spa(1, 1) = spa(2, 2) = 1e6;
    spa(3, 3) = spa(4, 4) = spa(5, 5) = 1e6;

    const SE3 T_spa(SO3::identity(), Vec3(5.0, 0.0, 0.0));   // x 는 쓰레기값
    const SE3 T_ecda(SO3::identity(), Vec3(0.2, 0.0, 0.0));
    Mat6 ecda = Mat6::Identity() * 1e2;

    std::vector<TierEstimate> es{make(Tier::Photometric, T_ecda, ecda),
                                 make(Tier::Structural, T_spa, spa)};
    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    // x: SPA 가 정보 0 이므로 ECDA 값 그대로. 구멍이 실제로 메워졌다.
    EXPECT_NEAR(r.value().T_cur_ref.translation().x(), 0.2, 1e-6);
    // y/z: SPA 가 1e6 대 1e2 로 지배
    EXPECT_NEAR(r.value().T_cur_ref.translation().y(), 0.0, 1e-3);

    // 그런데 observable_dof 는 여전히 5 라고 말한다. 이것은 결함이 아니라
    // 상대 문턱의 한계다 - x 축 정보 1e2 는 최대 고유값 1e6 대비 1e-4 라서
    // degeneracy_ratio=1e-3 아래로 떨어진다. 즉 **스케일이 크게 다른 tier 를
    // 융합하면 랭크 보고는 "메워졌다" 를 표현하지 못한다.** 실데이터에서 tier
    // 간 스케일 차가 10^6 배이므로 이 상황이 예외가 아니라 기본값이다.
    // 상보성을 랭크로 판정하면 안 되는 이유이며, 그래서 complementarity() 와
    // 축별 오차를 따로 본다.
    EXPECT_EQ(r.value().observable_dof, 5);
    EXPECT_GT(translationFraction(r.value().weakest_direction), 0.99);
    // 그 축의 고유값은 0 이 아니라 정확히 ECDA 가 넣은 1e2 다. "관측 불가" 로
    // 보고되는 축이 실제로는 구속되어 있다는 것을 숫자로 남겨 둔다.
    EXPECT_NEAR(r.value().eigenvalues(5), 1e2, 1e-6);
}

TEST(PoseFusion, TotalStaysRankDeficientWhenNoTierFillsTheGap) {
    Mat6 spa = Mat6::Identity() * 1e6;
    spa(0, 0) = 0.0;                       // x 병진 관측 불가
    std::vector<TierEstimate> es{make(Tier::Structural, SE3::identity(), spa)};

    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(r.degraded());
    EXPECT_EQ(r.value().observable_dof, 5);
    EXPECT_GT(std::abs(r.value().weakest_direction(0)), 0.99);   // 약축이 x 병진
    EXPECT_GT(translationFraction(r.value().weakest_direction), 0.99);
}

TEST(Complementarity, DetectsFillingAndIsScaleInvariant) {
    Mat6 a = Mat6::Identity() * 1e6;
    a(0, 0) = 1.0;                          // A 의 구멍: x 병진

    Mat6 b = Mat6::Identity() * 1.0;
    b(0, 0) = 1e6;                          // B 가 정확히 그 축을 채운다

    const Complementarity c = complementarity(a, b);
    ASSERT_TRUE(c.valid);
    EXPECT_GT(c.fill_ratio, 1e6);
    EXPECT_GT(translationFraction(c.weak_direction), 0.99);

    // 두 행렬을 각각 상수배해도 값이 같아야 한다. 그래야 스케일이 10^N 배
    // 차이나는 tier 사이에서 의미가 있다.
    const Complementarity s = complementarity(Mat6(a * 1e-7), Mat6(b * 3e5));
    ASSERT_TRUE(s.valid);
    EXPECT_NEAR(std::log10(s.fill_ratio), std::log10(c.fill_ratio), 1e-9);
}

TEST(Complementarity, SameShapeGivesExactlyOne) {
    // 중립 기준의 정의: B 가 A 를 상수배한 것이면 융합은 조건수를 바꾸지 못한다.
    // 이 값이 1 이 아니면 fill_ratio 를 "1 보다 크면 상보" 로 읽을 수 없다.
    Mat6 a = Mat6::Identity();
    a(0, 0) = 1.0;
    a(5, 5) = 1e6;
    const Complementarity c = complementarity(a, Mat6(a * 42.0));
    ASSERT_TRUE(c.valid);
    EXPECT_NEAR(c.fill_ratio, 1.0, 1e-9);
}

TEST(Complementarity, IsotropicBFillsByTheAnisotropyOfA) {
    Mat6 a = Mat6::Identity();
    a(0, 0) = 1.0;
    a(5, 5) = 1e6;
    const Mat6 b = Mat6::Identity() * 7.0;   // 방향성 없음. 그래도 약축은 채워진다
    const Complementarity c = complementarity(a, b);
    ASSERT_TRUE(c.valid);
    EXPECT_NEAR(c.fill_ratio, 1e6, 1.0);     // = cond(A)
}

// --- 기권이 보이는가 --------------------------------------------------------

TEST(PoseFusion, AbstentionIsReportedWithAReason) {
    TierEstimate tcg;
    tcg.tier      = Tier::Constellation;
    tcg.available = false;
    tcg.reason    = Abstain::NoInput;

    std::vector<TierEstimate> es{
        make(Tier::Photometric, SE3::identity(), diagInfo(1e3, 1e3)), tcg};

    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().contributing_tiers, 1);
    const auto& c = r.value().tiers[static_cast<std::size_t>(Tier::Constellation)];
    EXPECT_FALSE(c.used);
    EXPECT_EQ(c.reason, Abstain::NoInput);
    EXPECT_EQ(c.info_share, 0.0);
}

TEST(PoseFusion, ZeroAlphaIsAbstentionNotSilence) {
    // alpha_k(E) 가 0 이 되는 것과 tier 가 입력이 없는 것은 다른 사실이다.
    // 둘을 같은 코드로 표현하면 "왜 안 들어왔는가" 를 밖에서 물을 수 없다.
    auto e = make(Tier::Structural, SE3::identity(), diagInfo(1e3, 1e3), 0.0);
    std::vector<TierEstimate> es{
        make(Tier::Photometric, SE3::identity(), diagInfo(1e3, 1e3)), e};

    const auto r = fuse(es);
    ASSERT_TRUE(r.ok());
    const auto& c = r.value().tiers[static_cast<std::size_t>(Tier::Structural)];
    EXPECT_FALSE(c.used);
    EXPECT_EQ(c.reason, Abstain::ZeroWeight);
}

TEST(PoseFusion, AllTiersAbstainIsAnError) {
    std::vector<TierEstimate> es;
    TierEstimate e;
    e.tier = Tier::Photometric;
    e.available = false;
    e.reason = Abstain::Failed;
    es.push_back(e);
    const auto r = fuse(es);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.code(), ErrorCode::NotAvailable);
}

// --- alpha_k(E) -------------------------------------------------------------

TEST(TierWeights, EnvironmentAndUniformDifferUnderDegradation) {
    EnvironmentEvidence ev;
    ev.haze = 0.9;
    const EnvironmentState s = deriveAdaptationFrom(ev);

    const auto a_env = tierWeights(s, WeightMode::Environment);
    const auto a_uni = tierWeights(s, WeightMode::Uniform);

    EXPECT_EQ(a_uni[0], 1.0);
    EXPECT_EQ(a_uni[1], 1.0);
    EXPECT_EQ(a_uni[2], 1.0);
    // 안개에서 측광은 죽고 성좌는 버틴다 - 그것이 스케줄의 주장이다.
    EXPECT_LT(a_env[0], 0.3);
    EXPECT_GT(a_env[1], 0.5);
    EXPECT_GT(a_env[1], a_env[0]);
}

TEST(TierWeights, ClearIndoorLeavesPhotometricDominant) {
    const EnvironmentState s = deriveAdaptationFrom(EnvironmentEvidence{});
    const auto a = tierWeights(s, WeightMode::Environment);
    EXPECT_NEAR(a[0], 1.0, 1e-9);
    EXPECT_GT(a[1], 0.0);
    EXPECT_GT(a[2], 0.0);
}

// --- Tier 1 정보행렬 --------------------------------------------------------

TEST(ConstellationInformation, TwoPointsLeaveTheirAxisUnconstrained) {
    // 객체가 둘이면 그 둘을 잇는 축 둘레 회전은 관측되지 않는다.
    // 손으로 정한 대각행렬은 이 사실을 지우고 6 랭크라고 말한다.
    std::vector<ConstellationPair> pairs;
    for (double x : {-1.0, 1.0}) {
        ConstellationPair c;
        c.p_ref = Vec3(x, 0.0, 3.0);
        c.p_cur = c.p_ref;
        c.sigma = 0.05;
        pairs.push_back(c);
    }
    const Mat6 L = constellationInformation(pairs, SE3::identity());

    Eigen::SelfAdjointEigenSolver<Mat6> es(L);
    const double max_ev = es.eigenvalues()(5);
    int rank = 0;
    for (int k = 0; k < 6; ++k) if (es.eigenvalues()(k) > max_ev * 1e-9) ++rank;
    EXPECT_EQ(rank, 5);
}

TEST(ConstellationInformation, FourNonCoplanarPointsAreFullRank) {
    const std::vector<Vec3> pts = {{-1.0, -0.5, 2.5}, {1.2, 0.3, 3.1},
                                   {0.1, 1.4, 2.2},  {-0.6, 0.2, 4.0}};
    std::vector<ConstellationPair> pairs;
    for (const auto& p : pts) {
        ConstellationPair c;
        c.p_ref = p;
        c.p_cur = p;
        c.sigma = 0.06;
        pairs.push_back(c);
    }
    const Mat6 L = constellationInformation(pairs, SE3::identity());
    Eigen::SelfAdjointEigenSolver<Mat6> es(L);
    EXPECT_GT(es.eigenvalues()(0), es.eigenvalues()(5) * 1e-6);
}

TEST(ConstellationInformation, TooFewPairsGivesZero) {
    std::vector<ConstellationPair> pairs(1);
    EXPECT_EQ(constellationInformation(pairs, SE3::identity()).norm(), 0.0);
}

// --- 정보 이동 --------------------------------------------------------------

TEST(TransportInformation, IsTheChainRuleAtTheNewPoint) {
    // Lambda 는 T_from 접선의 이차형식이다. T_to 접선으로 옮긴다는 것은
    // delta_from = M * delta_to 의 M 으로 좌표변환하는 것이고, 그 M 은
    // delta_from = log(exp(delta_to) * T_to * T_from^-1) 의 자코비안이다.
    // M 을 유한차분으로 독립적으로 구해 비교한다.
    const SE3 A(SO3::exp(Vec3(0.1, -0.3, 0.2)), Vec3(0.5, 0.1, -0.2));
    const SE3 B(SO3::exp(Vec3(0.6, 0.25, -0.4)), Vec3(0.1, -0.7, 0.3));
    const Mat6 L  = diagInfo(1e3, 5e3);
    const Mat6 Lb = transportInformation(L, A, B);

    Mat6 M = Mat6::Zero();
    const double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec6 d = Vec6::Zero();
        d(k) = h;
        const Vec6 plus  = (SE3::exp(d) * B * A.inverse()).log();
        const Vec6 minus = (SE3::exp(Vec6(-d)) * B * A.inverse()).log();
        M.col(k) = (plus - minus) / (2.0 * h);
    }
    const Mat6 expected = M.transpose() * L * M;
    EXPECT_LT((Lb - expected).norm() / expected.norm(), 1e-6);
}

TEST(TransportInformation, IsIdentityWhenPointsCoincide) {
    const SE3 A(SO3::exp(Vec3(0.4, -0.1, 0.2)), Vec3(0.3, 0.2, -0.1));
    const Mat6 L = diagInfo(1e3, 5e3);
    EXPECT_LT((transportInformation(L, A, A) - L).norm(), 1e-10);
}
