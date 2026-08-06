// TCG 검증.
// 합성 성좌에 알려진 SE(3)를 적용해 재지역화가 그 변환을 되찾는지 확인한다.
// 노이즈/부분관측/오검출/거울배치/지각적혼동 각각을 별도 케이스로 다룬다.

#include "wme/token/ConstellationIndex.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <random>

using namespace wme;

namespace {

std::mt19937& rng() {
    static std::mt19937 gen(4242);
    return gen;
}

// 무작위이지만 재현 가능한 방 배치
std::vector<ConstellationNode> makeRoom(int n, int num_classes, double extent, double sigma = 0.03) {
    std::uniform_real_distribution<double> pos(-extent, extent);
    std::vector<ConstellationNode> nodes;
    nodes.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        ConstellationNode nd;
        nd.id       = TokenId(static_cast<std::uint64_t>(i + 1));
        nd.class_id = i % num_classes;
        nd.position = Vec3(pos(rng()), pos(rng()), pos(rng()) * 0.3);
        nd.sigma    = sigma;
        nodes.push_back(nd);
    }
    return nodes;
}

std::vector<ConstellationNode> transformed(const std::vector<ConstellationNode>& src,
                                           const SE3& T, double noise_sigma = 0.0) {
    std::normal_distribution<double> noise(0.0, noise_sigma);
    std::vector<ConstellationNode> out = src;
    for (auto& n : out) {
        n.position = T * n.position;
        if (noise_sigma > 0.0) {
            n.position += Vec3(noise(rng()), noise(rng()), noise(rng()));
            n.sigma = std::max(n.sigma, noise_sigma);
        }
    }
    return out;
}

}  // namespace

TEST(Kabsch, RecoversKnownTransform) {
    std::vector<Vec3> src{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    const SE3 T(SO3::exp(Vec3(0.3, -0.5, 0.9)), Vec3(2.0, -1.0, 0.5));

    std::vector<Vec3> dst;
    for (const auto& p : src) dst.push_back(T * p);

    const auto est = kabsch(src, dst);
    ASSERT_TRUE(est.ok()) << est.error().message();

    const Vec2 d = est.value().distanceTo(T);
    EXPECT_NEAR(d.x(), 0.0, 1e-9);
    EXPECT_NEAR(d.y(), 0.0, 1e-9);
}

TEST(Kabsch, RejectsCollinearPoints) {
    // 한 직선 위 점들은 축 회전을 결정하지 못한다. 잘못된 해를 내면 안 된다.
    std::vector<Vec3> src{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}};
    std::vector<Vec3> dst{{0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}};

    const auto est = kabsch(src, dst);
    EXPECT_FALSE(est.ok());
    EXPECT_EQ(est.error().code, ErrorCode::Degenerate);
}

TEST(Kabsch, NeverReturnsReflection) {
    // 거울 대칭 데이터를 줘도 det=+1 인 회전만 나와야 한다
    std::vector<Vec3> src{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
    std::vector<Vec3> dst;
    for (const auto& p : src) dst.emplace_back(p.x(), p.y(), -p.z());

    const auto est = kabsch(src, dst);
    ASSERT_TRUE(est.ok());
    EXPECT_NEAR(est.value().rotation().matrix().determinant(), 1.0, 1e-9);
}

TEST(Constellation, RelocalizesExactMatch) {
    ConstellationIndex index;
    const auto room = makeRoom(12, 5, 4.0);

    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.1, 0.4, -0.2)), Vec3(1.5, -2.0, 0.3));
    const auto observed = transformed(room, T.inverse());   // 다른 시점에서 본 같은 방

    const auto match = index.query(observed);
    ASSERT_TRUE(match.ok()) << match.error().message();

    // query -> place 변환이 T 여야 한다
    const Vec2 d = match.value().transform.distanceTo(T);
    EXPECT_LT(d.x(), 0.05);
    EXPECT_LT(d.y(), 0.02);
    EXPECT_GT(match.value().score, 0.5);
}

TEST(Constellation, RobustToPositionNoise) {
    ConstellationIndex index;
    const auto room = makeRoom(16, 6, 5.0, 0.05);
    index.insert(KeyframeId(7), Timestamp::fromSeconds(2.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(-0.2, 0.15, 0.5)), Vec3(-0.8, 1.2, -0.4));
    const auto observed = transformed(room, T.inverse(), 0.08);   // 8 cm 관측 노이즈

    const auto match = index.query(observed);
    ASSERT_TRUE(match.ok()) << match.error().message();
    const Vec2 d = match.value().transform.distanceTo(T);
    EXPECT_LT(d.x(), 0.30);
    EXPECT_LT(d.y(), 0.10);
}

TEST(Constellation, HandlesPartialObservation) {
    // 절반만 보여도 재지역화되어야 한다. 실제 운용에서 전부 보이는 경우는 드물다.
    ConstellationIndex index;
    const auto room = makeRoom(20, 7, 6.0);
    index.insert(KeyframeId(3), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.0, 0.6, 0.0)), Vec3(3.0, 0.0, 0.0));
    auto observed = transformed(room, T.inverse(), 0.03);
    observed.resize(10);

    const auto match = index.query(observed);
    ASSERT_TRUE(match.ok()) << match.error().message();
    EXPECT_LT(match.value().transform.distanceTo(T).x(), 0.25);
}

TEST(Constellation, SurvivesSpuriousDetections) {
    // 오검출 객체가 섞여도 클리크 탐색이 걸러내야 한다
    ConstellationIndex index;
    const auto room = makeRoom(14, 5, 5.0);
    index.insert(KeyframeId(9), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.2, -0.3, 0.1)), Vec3(1.0, 1.0, 0.0));
    auto observed = transformed(room, T.inverse(), 0.04);

    std::uniform_real_distribution<double> d(-6.0, 6.0);
    for (int i = 0; i < 5; ++i) {   // 유령 객체 5개 주입
        ConstellationNode ghost;
        ghost.id       = TokenId(static_cast<std::uint64_t>(1000 + i));
        ghost.class_id = i % 5;
        ghost.position = Vec3(d(rng()), d(rng()), d(rng()));
        ghost.sigma    = 0.2;
        observed.push_back(ghost);
    }

    const auto match = index.query(observed);
    ASSERT_TRUE(match.ok()) << match.error().message();
    EXPECT_LT(match.value().transform.distanceTo(T).x(), 0.35);
}

TEST(Constellation, RejectsUnknownPlace) {
    ConstellationIndex index;
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), makeRoom(12, 5, 4.0));

    // 전혀 다른 배치의 방 -> 매칭이 나오면 안 된다
    const auto other = makeRoom(12, 5, 4.0);
    const auto match = index.query(other);
    EXPECT_FALSE(match.ok());
}

TEST(Constellation, RejectsAmbiguousCandidates) {
    // 대조군 A - 진짜 지각적 혼동.
    // 동일한 복도가 50 m 떨어져 두 개 등록되어 있다. 두 후보의 transform 은
    // 같지만 anchor 가 달라 월드 포즈가 50 m 어긋난다. 기각해야 한다.
    // 이 테스트가 통과하지 않는 모호성 규칙은 규칙이 아니다.
    ConstellationIndex index;
    const auto corridor = makeRoom(10, 3, 5.0);

    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), corridor);
    index.insert(KeyframeId(2), Timestamp::fromSeconds(2.0),
                 SE3(SO3::identity(), Vec3(50.0, 0.0, 0.0)), corridor);

    const auto match = index.query(corridor);
    EXPECT_FALSE(match.ok()) << "동일 배치 두 곳을 구분했다고 주장하면 안 된다";
    EXPECT_EQ(match.error().code, ErrorCode::NotAvailable);

    // 다만 queryAll 은 두 후보를 모두 보고해야 한다 (상위 계층이 판단)
    EXPECT_GE(index.queryAll(corridor).size(), 2u);
}

TEST(Constellation, RejectsAmbiguityUnderRotation) {
    // 대조군 B - 병진은 같은데 회전만 다른 별칭.
    // 같은 방을 180도 돌려 등록하면 월드 병진은 붙고 자세만 어긋난다.
    // 포즈 합의를 병진만으로 재면 이 경우를 놓친다.
    ConstellationIndex index;
    const auto room = makeRoom(10, 3, 5.0);

    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);
    index.insert(KeyframeId(2), Timestamp::fromSeconds(2.0),
                 SE3(SO3::exp(Vec3(0.0, 0.0, kPi)), Vec3(0.0, 0.0, 0.0)), room);

    const auto match = index.query(room);
    EXPECT_FALSE(match.ok()) << "자세가 180도 어긋난 두 후보를 구분했다고 주장하면 안 된다";
    EXPECT_EQ(match.error().code, ErrorCode::NotAvailable);
}

TEST(Constellation, AcceptsDenselySampledMap) {
    // 회귀 테스트. 조밀하게 샘플링한 지도에서 정대응이 살아남아야 한다.
    //
    // 옛 규칙(score2 > 0.85*score1)은 여기서 전부 기각한다. 이웃 키프레임이
    // 같은 장면을 보므로 점수가 붙는 것은 지각적 혼동이 아니라 정상이며,
    // 실데이터 fr1_xyz 에서 이 규칙은 정대응 36/36 을 전부 날렸다.
    ConstellationIndex index;
    const auto room = makeRoom(14, 6, 4.0);   // 월드 좌표계의 방

    // 같은 방을 조금씩 옮겨 가며 5회 등록 (map_stride 5 프레임 간격의 축소판)
    for (int k = 0; k < 5; ++k) {
        const SE3 view(SO3::exp(Vec3(0.0, 0.03 * k, 0.0)), Vec3(0.08 * k, 0.02 * k, 0.0));
        index.insert(KeyframeId(static_cast<std::uint64_t>(k + 1)),
                     Timestamp::fromSeconds(k), view, transformed(room, view.inverse()));
    }

    // 등록 시점 사이의 새로운 시점에서 질의
    const SE3 q(SO3::exp(Vec3(0.0, 0.075, 0.0)), Vec3(0.20, 0.05, 0.0));
    const auto match = index.query(transformed(room, q.inverse(), 0.02));

    ASSERT_TRUE(match.ok()) << "조밀 지도의 정대응이 모호성 규칙에 걸렸다: "
                            << match.error().message();

    // 월드 포즈(anchor x transform)가 실제 질의 포즈여야 한다
    const Place* p = index.place(match.value().place_id);
    ASSERT_NE(p, nullptr);
    const Vec2 d = (p->anchor * match.value().transform).distanceTo(q);
    EXPECT_LT(d.x(), 0.10);
    EXPECT_LT(d.y(), 0.05);

    // 여러 후보가 같은 포즈에 합의했어야 한다 - 이것이 교차증거다
    EXPECT_GE(match.value().agree_count, 2u);
    EXPECT_GT(match.value().support, match.value().score);
    EXPECT_GT(match.value().confidence, 0.3);
}

TEST(Constellation, DiagnosticsArePopulated) {
    // 신뢰도 원자료가 실제로 채워지는지. 0 으로 남으면 채점기는 그것을
    // "예측력 없음" 으로 읽고, 우리는 알고리즘을 탓하게 된다.
    ConstellationIndex index;
    const auto room = makeRoom(12, 5, 4.0);
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.1, 0.4, -0.2)), Vec3(1.5, -2.0, 0.3));
    const auto match = index.query(transformed(room, T.inverse(), 0.02));
    ASSERT_TRUE(match.ok()) << match.error().message();

    const auto& m = match.value();
    EXPECT_EQ(m.n_query_nodes, room.size());
    EXPECT_EQ(m.n_place_nodes, room.size());
    EXPECT_GE(m.n_inliers, 4u);
    EXPECT_GT(m.explained, 0.5);
    EXPECT_LE(m.explained, 1.0);
    EXPECT_GT(m.chi2_dof, 0.0);
    EXPECT_DOUBLE_EQ(m.pose_margin, kNoRivalMargin);   // 경쟁자 없음
    EXPECT_GT(m.confidence, 0.0);
}

TEST(Constellation, PartialConstellationHasLowerExplained) {
    // 설명률이 실제로 변별하는지 확인하는 대조군.
    // 질의 성좌의 절반만 지도에 있는 경우, rms 는 좋아도 설명률은 낮아야 한다.
    // 설명률이 두 경우를 구분하지 못하면 신뢰도 항으로 쓸 수 없다.
    ConstellationIndex index;
    const auto room = makeRoom(16, 6, 5.0);
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.1, 0.2, 0.0)), Vec3(1.0, 0.5, 0.0));
    const auto full = index.query(transformed(room, T.inverse(), 0.02));
    ASSERT_TRUE(full.ok()) << full.error().message();

    // 지도에 없는 유령 객체를 8개 섞는다 - 성좌는 커지지만 설명되는 몫은 준다
    auto noisy = transformed(room, T.inverse(), 0.02);
    std::uniform_real_distribution<double> d(-8.0, 8.0);
    for (int i = 0; i < 8; ++i) {
        ConstellationNode g;
        g.id       = TokenId(static_cast<std::uint64_t>(2000 + i));
        g.class_id = i % 6;
        g.position = Vec3(d(rng()), d(rng()), d(rng()));
        g.sigma    = 0.05;
        noisy.push_back(g);
    }
    const auto partial = index.query(noisy);
    ASSERT_TRUE(partial.ok()) << partial.error().message();
    EXPECT_LT(partial.value().explained, full.value().explained);
}

TEST(Constellation, RequiresMinimumNodes) {
    ConstellationIndex index;
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), makeRoom(12, 5, 4.0));

    const auto tiny = makeRoom(3, 3, 2.0);
    const auto match = index.query(tiny);
    EXPECT_FALSE(match.ok());
    EXPECT_EQ(match.error().code, ErrorCode::InsufficientData);
}

TEST(Constellation, ScalesToManyPlaces) {
    // 역색인 검색이 장소 수에 따라 폭발하지 않는지 확인 (기능 검증, 성능은 벤치마크)
    ConstellationIndex index;
    const auto target = makeRoom(15, 8, 5.0);

    for (int i = 0; i < 300; ++i) index.insert(KeyframeId(static_cast<std::uint64_t>(i)),
                                               Timestamp::fromSeconds(i), SE3::identity(),
                                               makeRoom(15, 8, 5.0));
    const std::uint64_t pid =
        index.insert(KeyframeId(999), Timestamp::fromSeconds(999.0), SE3::identity(), target);

    const SE3 T(SO3::exp(Vec3(0.1, 0.1, 0.1)), Vec3(0.5, 0.5, 0.0));
    const auto match = index.query(transformed(target, T.inverse(), 0.03));

    ASSERT_TRUE(match.ok()) << match.error().message();
    EXPECT_EQ(match.value().place_id, pid);
}

TEST(Constellation, ChiralityUsesEachFramesOwnGravity) {
    // 회귀 테스트. 질의 프레임은 등록 시점 대비 회전되어 있으므로 그 프레임에서의
    // 중력은 더 이상 (0,0,-1) 이 아니다. 두 프레임에 같은 벡터를 쓰면 정상 대응이
    // 통째로 걸러져 클리크가 쪼그라들고 틀린 변환이 나온다.
    // (Python 오라클 tests/test_constellation.py 의 동명 테스트와 짝을 이룬다.)
    const Vec3 kDown(0.0, 0.0, -1.0);

    ConstellationIndex index;
    const auto room = makeRoom(12, 5, 4.0);
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room, kDown);

    const SE3 T(SO3::exp(Vec3(0.1, 0.4, -0.2)), Vec3(1.5, -2.0, 0.3));
    const auto observed = transformed(room, T.inverse());
    const Vec3 query_gravity = T.inverse().rotation() * kDown;   // 질의 프레임에서 본 중력

    const auto good = index.query(observed, query_gravity);
    ASSERT_TRUE(good.ok()) << good.error().message();
    EXPECT_EQ(good.value().correspondences.size(), room.size())
        << "정상 대응이 카이랄리티에 걸렸다";
    EXPECT_LT(good.value().transform.distanceTo(T).x(), 1e-6);

    // 같은 벡터를 그대로 쓰면(버그가 있던 방식) 대응이 줄어든다
    const auto bad = index.query(observed, kDown);
    if (bad.ok()) {
        EXPECT_LT(bad.value().correspondences.size(), room.size());
    }
}

TEST(Constellation, GravityIsOptional) {
    // IMU 도 지배평면도 없는 구성에서는 카이랄리티 없이 동작해야 한다
    ConstellationIndex index;
    const auto room = makeRoom(12, 5, 4.0);
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room);

    const SE3 T(SO3::exp(Vec3(0.1, 0.4, -0.2)), Vec3(1.5, -2.0, 0.3));
    const auto match = index.query(transformed(room, T.inverse()));

    ASSERT_TRUE(match.ok()) << match.error().message();
    EXPECT_LT(match.value().transform.distanceTo(T).x(), 1e-6);
}

TEST(Constellation, ChiralityRejectsMirroredLayout) {
    // 거울 배치는 쌍거리 스펙트럼이 같다. 카이랄리티만이 이를 구분한다.
    const Vec3 kDown(0.0, 0.0, -1.0);

    std::vector<ConstellationNode> room;
    const std::array<Vec3, 5> pts{{
        {0.0, 0.0, 0.0}, {2.0, 0.0, 1.5}, {0.0, 2.0, -1.5}, {2.0, 2.0, 1.0}, {1.0, 1.0, -1.2},
    }};
    for (std::size_t i = 0; i < pts.size(); ++i) {
        ConstellationNode n;
        n.id = TokenId(i + 1);
        n.class_id = static_cast<int>(i);
        n.position = pts[i];
        n.sigma = 0.03;
        room.push_back(n);
    }

    ConstellationIndex index;
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), room, kDown);

    auto mirrored = room;
    for (auto& n : mirrored) n.position.z() *= -1.0;

    EXPECT_FALSE(index.query(mirrored, kDown).ok());
}

TEST(Constellation, BuildFromFiltersUnstableTokens) {
    std::vector<WorldTokenPtr> tokens;

    auto stable = std::make_shared<WorldToken>();
    stable->id = TokenId(1);
    stable->class_id = 2;
    stable->position = Vec3(1.0, 2.0, 0.0);
    stable->position_cov = Mat3::Identity() * 0.01;
    stable->static_belief = 0.9;
    stable->existence_belief = 0.9;
    stable->observation_count = 10;
    tokens.push_back(stable);

    auto moving = std::make_shared<WorldToken>();   // 사람 - 장소를 정의할 수 없다
    moving->id = TokenId(2);
    moving->class_id = 0;
    moving->position = Vec3(0.0, 1.0, 0.0);
    moving->position_cov = Mat3::Identity() * 0.01;
    moving->static_belief = 0.1;
    moving->existence_belief = 0.9;
    moving->observation_count = 10;
    tokens.push_back(moving);

    auto fresh = std::make_shared<WorldToken>();    // 관측 부족
    fresh->id = TokenId(3);
    fresh->class_id = 5;
    fresh->position = Vec3(2.0, 0.0, 0.0);
    fresh->position_cov = Mat3::Identity() * 0.01;
    fresh->static_belief = 0.9;
    fresh->existence_belief = 0.9;
    fresh->observation_count = 1;
    tokens.push_back(fresh);

    const auto nodes = ConstellationIndex::buildFrom(tokens, SE3::identity(), 40);
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes.front().id, TokenId(1));
}

// 동점 후보의 순서 규약.
//
// retrieve() 는 unordered_map<place index, 표> 를 훑어 후보 목록을 만든다.
// 점수만으로 정렬하면 동점 구간에 그 순회 순서가 그대로 남고, 뒤이은
// top_candidates 절단이 무엇을 남길지가 컨테이너 구현에 달리게 된다.
//
// 이 케이스는 그 상황을 만든다: 완전히 같은 기하의 장소를 40 개 등록하면
// 표도 검증 점수도 비트 단위로 같아지므로 순서를 정하는 것은 비교기뿐이다.
//
// 판별력에 대한 정직한 한계 - 이 테스트는 지금 쓰는 표준 라이브러리에서는
// 결함을 잡지 못한다. MSVC 의 unordered_map 은 정수 키에 항등 해시를 쓰고
// 버킷 수를 원소 수보다 훨씬 크게 잡아(원소 20/40/100 에 버킷 128/256/512)
// 조밀한 인덱스 키의 순회가 항상 오름차순이 되고, 그것이 마침 place_id 순과
// 같다. 타이브레이크를 빼고 다시 빌드해도 이 케이스는 통과한다 (실측).
// 즉 여기서 고친 것은 "관측된 오답" 이 아니라 "관측되지 않은 의존" 이다.
// 소수 버킷 + 모듈로를 쓰는 libstdc++/libc++ 나, 삽입 순서가 달라지는 어떤
// 변경에서도 답이 흔들리지 않도록 규약을 해석적으로 못박아 두는 것이 목적이다.
TEST(ConstellationDeterminism, TiedCandidatesAreOrderedByPlaceId) {
    ConstellationConfig cfg;
    ConstellationIndex index(cfg);

    const auto room = makeRoom(6, 3, 2.0, 0.02);

    // 40 개인 이유: 동점만 있는 구간에서 std::sort 는 안정 정렬이 아니다.
    // MSVC 는 32 개까지 삽입정렬(안정)이라 그 아래에서는 입력 순서가 그대로 남는다.
    constexpr int kPlaces = 40;
    std::vector<std::uint64_t> ids;
    for (int i = 0; i < kPlaces; ++i) {
        ids.push_back(index.insert(KeyframeId(static_cast<std::uint64_t>(i + 1)),
                                   Timestamp::fromSeconds(1.0 + i), SE3::identity(), room));
    }
    ASSERT_EQ(index.placeCount(), static_cast<std::size_t>(kPlaces));

    const auto results = index.queryAll(room);
    ASSERT_FALSE(results.empty());
    ASSERT_LE(results.size(), cfg.top_candidates);

    // 전부 같은 기하이므로 점수가 비트 단위로 같아야 한다
    for (const auto& m : results) {
        EXPECT_EQ(0, std::memcmp(&m.score, &results.front().score, sizeof(double)))
            << "같은 기하인데 점수가 다르다 - 이 케이스는 동점을 시험하지 못한다";
    }

    // 동점이면 답은 "가장 작은 place_id 부터 오름차순" 으로 해석적으로 정해진다
    std::vector<std::uint64_t> got;
    for (const auto& m : results) got.push_back(m.place_id);

    std::vector<std::uint64_t> want(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(results.size()));
    EXPECT_EQ(got, want) << "동점 후보 순서가 place_id 오름차순이 아니다 - 순회 순서가 새고 있다";

    // 반복 질의도 같아야 한다
    const auto again = index.queryAll(room);
    ASSERT_EQ(again.size(), results.size());
    for (std::size_t i = 0; i < again.size(); ++i) {
        EXPECT_EQ(again[i].place_id, results[i].place_id);
    }
}
