// 신뢰도 엔진 검증.
// 확률 값 자체를 단정하지 않고, 베이즈 갱신이 가져야 할 성질을 검증한다:
// 단조성, 포화 한계, 신뢰도에 따른 정보량 스케일, 회복 가능성.

#include "wme/confidence/ConfidenceEngine.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace wme;

namespace {

EnvironmentState makeEnv(double reliability) {
    EnvironmentState e;
    e.sensor_reliability = reliability;
    return e;
}

Observation makeObs(float conf, double t, double reliability = 1.0) {
    Observation o;
    o.stamp = Timestamp::fromSeconds(t);
    o.detection_conf = conf;
    o.sensor_reliability = reliability;
    o.image_quality = 1.0;
    o.visible_ratio = 1.0;
    return o;
}

WorldToken makeToken() {
    WorldToken t;
    t.id = TokenId(1);
    t.class_id = 0;
    t.existence_belief = 0.5;
    t.identity_belief = 0.5;
    t.static_belief = 0.5;
    t.position_cov = Mat3::Identity() * 0.01;
    return t;
}

}  // namespace

TEST(Confidence, LogOddsRoundTrip) {
    for (double p : {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        EXPECT_NEAR(ConfidenceEngine::toProbability(ConfidenceEngine::toLogOdds(p)), p, 1e-9);
    }
}

TEST(Confidence, LogOddsSaturatesSafelyAtExtremes) {
    // 0/1 에서 발산하면 그 믿음은 영원히 갱신 불가가 된다
    EXPECT_TRUE(std::isfinite(ConfidenceEngine::toLogOdds(0.0)));
    EXPECT_TRUE(std::isfinite(ConfidenceEngine::toLogOdds(1.0)));
}

TEST(Confidence, RepeatedObservationsIncreaseExistence) {
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    double prev = t.existence_belief;
    for (int i = 0; i < 8; ++i) {
        engine.onObserved(t, makeObs(0.9f, i * 0.1), env);
        EXPECT_GE(t.existence_belief, prev);
        prev = t.existence_belief;
    }
    EXPECT_GT(t.existence_belief, 0.9);
}

TEST(Confidence, ExistenceNeverSaturatesToCertainty) {
    // 포화하면 반증을 받아들이지 못한다. "지도를 수리한다"는 원칙의 전제.
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    for (int i = 0; i < 500; ++i) engine.onObserved(t, makeObs(1.0f, i * 0.05), env);
    EXPECT_LT(t.existence_belief, 1.0);

    // 그리고 반증으로 되돌릴 수 있어야 한다
    const double high = t.existence_belief;
    for (int i = 0; i < 200; ++i) engine.onMissed(t, 1.0, env);
    EXPECT_LT(t.existence_belief, high);
    EXPECT_LT(t.existence_belief, 0.3);
}

TEST(Confidence, MissWhileOccludedIsNotEvidenceOfAbsence) {
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    for (int i = 0; i < 5; ++i) engine.onObserved(t, makeObs(0.9f, i * 0.1), env);
    const double before = t.existence_belief;

    // 가시성 0.05 = 거의 완전히 가려짐 -> 존재 믿음이 줄면 안 된다
    for (int i = 0; i < 30; ++i) engine.onMissed(t, 0.05, env);
    EXPECT_DOUBLE_EQ(t.existence_belief, before);
    EXPECT_EQ(t.miss_count, 30u);
}

TEST(Confidence, LowReliabilityWeakensEvidence) {
    // 안개 속 관측도 같은 방향이지만 정보량이 작아야 한다
    ConfidenceEngine engine;

    WorldToken clear_t = makeToken();
    WorldToken foggy_t = makeToken();

    for (int i = 0; i < 5; ++i) {
        engine.onObserved(clear_t, makeObs(0.9f, i * 0.1, 1.0), makeEnv(1.0));
        engine.onObserved(foggy_t, makeObs(0.9f, i * 0.1, 0.3), makeEnv(0.3));
    }
    EXPECT_GT(clear_t.existence_belief, foggy_t.existence_belief);
    EXPECT_GT(foggy_t.existence_belief, 0.5) << "방향은 여전히 긍정이어야 한다";
}

TEST(Confidence, LowDetectionConfidenceGivesWeakerEvidence) {
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken strong = makeToken(), weak = makeToken();
    for (int i = 0; i < 5; ++i) {
        engine.onObserved(strong, makeObs(0.95f, i * 0.1), env);
        engine.onObserved(weak,   makeObs(0.30f, i * 0.1), env);
    }
    EXPECT_GT(strong.existence_belief, weak.existence_belief);
}

TEST(Confidence, AmbiguousAssociationWeakensIdentity) {
    // 1등/2등 격차가 작으면 "같은 의자 두 개 중 어느 쪽인지 모른다"
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken distinct = makeToken(), ambiguous = makeToken();
    for (int i = 0; i < 6; ++i) {
        engine.onObserved(distinct,  makeObs(0.9f, i * 0.1), env, 8.0);   // 큰 격차
        engine.onObserved(ambiguous, makeObs(0.9f, i * 0.1), env, 0.02);  // 거의 동점
    }
    EXPECT_GT(distinct.identity_belief, ambiguous.identity_belief);
}

TEST(Confidence, IdentityDecaysOutOfViewButExistenceDoesNot) {
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    for (int i = 0; i < 6; ++i) engine.onObserved(t, makeObs(0.9f, i * 0.1), env, 5.0);
    const double e0 = t.existence_belief;
    const double i0 = t.identity_belief;

    engine.onOutOfView(t, 60.0);
    EXPECT_DOUBLE_EQ(t.existence_belief, e0) << "안 보인다는 것은 없다는 뜻이 아니다";
    EXPECT_LT(t.identity_belief, i0);
    EXPECT_GT(t.identity_belief, 0.5) << "0.5 아래로 뒤집히면 안 되고 0.5 로 수렴해야 한다";
}

// 판정 창은 ConfidenceConfig::static_min_dt 이상이어야 한다. 이보다 짧은 창에서는
// 운동(v*T)이 관측 잡음에 묻힌다 - 실측 분리비 1.20 (docs/06-results.md 16 장).
constexpr double kWin = 0.5;

TEST(Confidence, ShortWindowIsNotJudged) {
    // 창이 짧으면 판정하지 않는다. 이 게이트가 프레임률에 절벽을 만들지 않도록
    // 호출자(TokenStore)가 기준점을 누적한다 - 버리는 것이 아니라 미루는 것이다.
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const double before = t.static_belief;
    engine.updateStaticBelief(t, Vec3(0.5, 0.0, 0.0), 0.033, makeEnv(1.0));
    EXPECT_DOUBLE_EQ(t.static_belief, before);
    EXPECT_EQ(t.static_diag.updates, 0u);
}

TEST(Confidence, StationaryObjectBecomesStatic) {
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    for (int i = 0; i < 20; ++i) {
        engine.updateStaticBelief(t, Vec3(0.001, -0.002, 0.0), kWin, env);
    }
    EXPECT_GT(t.static_belief, 0.8);
}

TEST(Confidence, MovingObjectBecomesDynamic) {
    ConfidenceEngine engine;
    WorldToken t = makeToken();
    const auto env = makeEnv(1.0);

    for (int i = 0; i < 20; ++i) {
        engine.updateStaticBelief(t, Vec3(0.4, 0.0, 0.0), kWin, env);   // 0.8 m/s
    }
    EXPECT_LT(t.static_belief, 0.2);
    EXPECT_TRUE(t.isDynamic());
}

TEST(Confidence, EvidenceStrengthGrowsWithWindow) {
    // 우도비를 쓰는 이유. 같은 "안 움직였다" 라도 긴 창에서 관측한 쪽이
    // 더 강한 증거다 - 동적 가설이 그만큼 큰 변위를 예측했기 때문이다.
    // 이전 구현은 증거를 [-0.31, +0.38] 로 묶어 이 구분을 없앴다.
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken shortw = makeToken();
    WorldToken longw  = makeToken();
    engine.updateStaticBelief(shortw, Vec3::Zero(), 0.5, env);
    engine.updateStaticBelief(longw,  Vec3::Zero(), 2.0, env);
    EXPECT_GT(longw.static_belief, shortw.static_belief);
}

TEST(Confidence, OneOutlierCannotEraseAccumulatedEvidence) {
    // 검출 박스 중심의 3D 위치는 이상치를 낸다. 순수 가우시안 우도비는
    // 아래로 무계라서 이상치 한 번이 -453 nat 을 실어 10 초치 증거를 지웠다
    // (docs/06-results.md 16 장). 정적 가설에 꼬리를 붙여 증거를 log(eps) 에서
    // 막는다 - 이상치가 결론을 뒤집을 수는 있어도 지울 수는 없어야 한다.
    ConfidenceConfig cfg;
    ConfidenceEngine engine(cfg);
    const auto env = makeEnv(1.0);

    WorldToken t = makeToken();
    for (int i = 0; i < 10; ++i) engine.updateStaticBelief(t, Vec3::Zero(), 0.5, env);
    const double earned = t.static_belief;
    ASSERT_GT(earned, 0.9);

    engine.updateStaticBelief(t, Vec3(3.0, 0, 0), 0.5, env);   // 6 m/s, 명백한 이상치
    EXPECT_LT(t.static_belief, earned) << "이상치가 아무 영향도 없으면 채널이 죽은 것이다";
    EXPECT_GT(t.static_belief, 0.5)
        << "이상치 한 번이 10 창치 증거를 지웠다 (static_belief=" << t.static_belief << ")";

    // 증거의 하한은 설정된 꼬리 무게가 정한다. 꼬리를 없애면 무계로 돌아간다.
    ConfidenceConfig sharp = cfg;
    sharp.static_outlier_rate = 0.0;
    ConfidenceEngine strict(sharp);
    WorldToken u = makeToken();
    strict.updateStaticBelief(u, Vec3(3.0, 0, 0), 0.5, env);
    EXPECT_LT(u.static_diag.ratio, std::log(cfg.static_outlier_rate) - 1.0);
    EXPECT_GE(t.static_diag.ratio, std::log(cfg.static_outlier_rate) - 1e-9);
}

TEST(Confidence, StaticEvidenceIsBoundedAbove) {
    // "안 움직였다" 는 상한이 있는 증거다 - 아무리 정지해 있어도 동적 물체가
    // 잠시 멈춘 것일 수 있다. 변위 0 과 거의 0 은 같은 증거를 준다.
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);
    WorldToken a = makeToken(), b = makeToken();
    engine.updateStaticBelief(a, Vec3::Zero(), 1.0, env);
    engine.updateStaticBelief(b, Vec3(1e-9, 0, 0), 1.0, env);
    EXPECT_NEAR(a.static_diag.ratio, b.static_diag.ratio, 1e-6);
    EXPECT_LT(a.static_diag.ratio, 10.0);
}

TEST(Confidence, StoppedAgentIsDoubtedButNotForbidden) {
    // 신호 대기 중인 차는 정지해 있어도 쉽게 정적 랜드마크가 되면 안 된다.
    // 그렇다고 상한을 씌워 영원히 동적으로 못박으면 채널이 통째로 죽는다 -
    // 앉아 있는 사람이 통째로 마스킹되어 TUM fr3_sitting 의 ATE 가
    // 1.01 -> 15.84 cm 로 무너졌다 (docs/06-results.md 14 장).
    // 요구사항은 "금지" 가 아니라 "더 많은 증거를 요구" 다.
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken agent = makeToken();
    agent.affordance = Affordance::Agent;
    agent.static_belief = 0.15;              // TokenStore 가 주는 사전분포

    WorldToken thing = makeToken();
    thing.static_belief = 0.6;               // 비자율 개체 사전분포

    agent.static_prior = 0.15;
    thing.static_prior = 0.6;

    engine.updateStaticBelief(agent, Vec3::Zero(), kWin, env);
    engine.updateStaticBelief(thing, Vec3::Zero(), kWin, env);
    EXPECT_LT(agent.static_belief, thing.static_belief) << "회의가 반영되지 않았다";
    EXPECT_TRUE(agent.isDynamic()) << "관측 한 창으로 자율 개체를 정적이라 단정하면 안 된다";

    // 그러나 충분히 오래 정지해 있으면 결론이 바뀔 수 있어야 한다.
    for (int i = 0; i < 10; ++i) engine.updateStaticBelief(agent, Vec3::Zero(), kWin, env);
    EXPECT_GT(agent.static_belief, 0.4)
        << "증거가 아무리 쌓여도 안 바뀌면 갱신이 무의미하다 (static_belief="
        << agent.static_belief << ")";
    EXPECT_FALSE(agent.isDynamic());

    // 그리고 관측이 끊기면 그 주장은 유효기간이 지난다. 상한(금지)이 아니라
    // 감쇠(유효기간)로 "신호등 앞의 차" 를 다룬다.
    engine.decayStaticBelief(agent, 60.0);
    EXPECT_LT(agent.static_belief, 0.4)
        << "관측 없이도 영구 랜드마크로 남으면 정지한 차가 지도에 박힌다";
    EXPECT_NEAR(agent.static_belief, agent.static_prior, 0.02);
}

TEST(Confidence, StaticClaimOfNonAgentDoesNotExpire) {
    // 책상은 안 보이는 동안 걸어다니지 않는다. 감쇠는 자율 개체에만 적용된다.
    ConfidenceEngine engine;
    WorldToken thing = makeToken();
    thing.static_belief = 0.95;
    thing.static_prior  = 0.6;
    engine.decayStaticBelief(thing, 600.0);
    EXPECT_DOUBLE_EQ(thing.static_belief, 0.95);
}

TEST(Confidence, StaticJudgementScalesWithMeasurementNoise) {
    // 관측 불확실성이 큰 토큰은 같은 변위로도 동적이라고 단정하면 안 된다.
    //
    // 척도는 *관측* 잡음이지 융합 추정의 공분산이 아니다. 후자는 관측 수에
    // 따라 줄어드는데 변위는 그렇지 않다 - 예측하는 통계가 다르다.
    // 깊이가 없는 토큰(meas_sigma 가 수 m)은 이 경로로 스스로 판정을 포기한다.
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken precise = makeToken();
    precise.meas_sigma = 0.01;                        // 1 cm

    WorldToken vague = makeToken();
    vague.meas_sigma = 0.50;                          // 50 cm (깊이 없음에 가까움)

    for (int i = 0; i < 5; ++i) {
        engine.updateStaticBelief(precise, Vec3(0.30, 0, 0), kWin, env);
        engine.updateStaticBelief(vague,   Vec3(0.30, 0, 0), kWin, env);
    }
    EXPECT_LT(precise.static_belief, vague.static_belief);

    // 융합 공분산은 이 판정에 영향을 주면 안 된다.
    WorldToken tight = makeToken(), loose = makeToken();
    tight.meas_sigma = loose.meas_sigma = 0.02;
    tight.position_cov = Mat3::Identity() * 1e-6;
    loose.position_cov = Mat3::Identity() * 0.25;
    engine.updateStaticBelief(tight, Vec3(0.30, 0, 0), kWin, env);
    engine.updateStaticBelief(loose, Vec3(0.30, 0, 0), kWin, env);
    EXPECT_DOUBLE_EQ(tight.static_belief, loose.static_belief);
}

TEST(Confidence, MergeCombinesEvidence) {
    ConfidenceEngine engine;
    const auto env = makeEnv(1.0);

    WorldToken a = makeToken(), b = makeToken();
    b.id = TokenId(2);
    for (int i = 0; i < 4; ++i) {
        engine.onObserved(a, makeObs(0.85f, i * 0.1), env);
        engine.onObserved(b, makeObs(0.85f, i * 0.1), env);
    }
    a.first_seen = Timestamp::fromSeconds(10.0);
    a.last_seen  = Timestamp::fromSeconds(20.0);
    b.first_seen = Timestamp::fromSeconds(2.0);
    b.last_seen  = Timestamp::fromSeconds(30.0);
    a.observation_count = 4; b.observation_count = 4;

    const double before = a.existence_belief;
    ConfidenceEngine::mergeBeliefs(a, b, engine.config());

    EXPECT_GE(a.existence_belief, before);
    EXPECT_LT(a.existence_belief, 1.0);
    EXPECT_EQ(a.observation_count, 8u);
    // 루프 클로저의 값어치: 관측 구간이 하나로 이어진다
    EXPECT_DOUBLE_EQ(a.first_seen.seconds(), 2.0);
    EXPECT_DOUBLE_EQ(a.last_seen.seconds(), 30.0);
}
