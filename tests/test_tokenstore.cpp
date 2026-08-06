// World Token Engine 검증.
// 합성 장면에서 검출을 만들어 넣고, 정체성 유지 / 생애주기 / 동적 판정이
// 규격대로 동작하는지 본다. 특히 "안 보인다 != 없다" 규칙을 집중 검증한다.

#include "wme/token/TokenStore.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>

using namespace wme;

namespace {

constexpr int kW = 640, kH = 480;

CameraIntrinsics makeK() {
    CameraIntrinsics K;
    K.fx = 500.0; K.fy = 500.0;
    K.cx = kW * 0.5; K.cy = kH * 0.5;
    K.width = kW; K.height = kH;
    return K;
}

struct SceneObject {
    int         class_id{0};
    std::string name{"chair"};
    Vec3        position{Vec3::Zero()};   // 월드 좌표
    double      size{0.5};                // 한 변 (m)
    float       confidence{0.9f};
};

struct Rendered {
    Frame        frame;
    DetectionSet detections;
};

// 객체들을 주어진 카메라 포즈에서 렌더링해 프레임 + 검출을 만든다
Rendered render(const std::vector<SceneObject>& objects, const SE3& T_world_cam, double t) {
    Rendered out;
    const CameraIntrinsics K = makeK();
    const SE3 T_cam_world = T_world_cam.inverse();

    out.frame.id    = FrameId(static_cast<std::uint64_t>(t * 100) + 1);
    out.frame.stamp = Timestamp::fromSeconds(t);
    out.frame.gray  = cv::Mat(kH, kW, CV_8UC1, cv::Scalar(120));
    out.frame.depth = cv::Mat(kH, kW, CV_32F, cv::Scalar(0.f));   // 0 = 무효
    out.frame.intrinsics = K;
    out.frame.sensor = SensorKind::RgbD;

    out.detections.stamp = out.frame.stamp;
    out.detections.frame = out.frame.id;

    for (const auto& o : objects) {
        const Vec3 p_cam = T_cam_world * o.position;
        if (p_cam.z() < 0.3) continue;                       // 카메라 뒤

        const Vec2 c = K.project(p_cam);
        const double w = K.fx * o.size / p_cam.z();
        const double h = K.fy * o.size / p_cam.z();

        cv::Rect2f box(static_cast<float>(c.x() - w * 0.5), static_cast<float>(c.y() - h * 0.5),
                       static_cast<float>(w), static_cast<float>(h));

        // 화면 밖이면 검출되지 않는다
        const cv::Rect roi = cv::Rect(box) & cv::Rect(0, 0, kW, kH);
        if (roi.width < 10 || roi.height < 10) continue;

        // 깊이 센서는 앞면을 잰다. 무게중심 깊이를 쓰면 존재할 수 없는 센서를
        // 흉내내는 셈이고, 무게중심 보정 항이 그대로 편향으로 남는다.
        out.frame.depth(roi).setTo(static_cast<float>(p_cam.z() - o.size * 0.5));

        Detection d;
        d.class_id   = o.class_id;
        d.class_name = o.name;
        d.box        = box;
        d.confidence = o.confidence;
        out.detections.items.push_back(d);
    }
    return out;
}

EnvironmentState goodEnv() {
    EnvironmentState e;
    e.sensor_reliability = 1.0;
    e.track_persistence_scale = 1.0;
    e.memory_retention_scale = 1.0;
    return e;
}

}  // namespace

// --- NMS -------------------------------------------------------------------

TEST(Nms, SuppressesOverlappingSameClass) {
    std::vector<Detection> dets(3);
    for (auto& d : dets) { d.class_id = 1; d.box = {100, 100, 50, 50}; }
    dets[0].confidence = 0.9f;
    dets[1].confidence = 0.8f;
    dets[2].confidence = 0.7f;

    const auto kept = nonMaxSuppression(dets, 0.5f);
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_FLOAT_EQ(kept[0].confidence, 0.9f);
}

TEST(Nms, KeepsOverlappingDifferentClasses) {
    // 사람이 든 컵이 사람에 의해 지워지면 안 된다
    std::vector<Detection> dets(2);
    dets[0].class_id = 0; dets[0].box = {100, 100, 80, 200}; dets[0].confidence = 0.9f;
    dets[1].class_id = 41; dets[1].box = {110, 150, 30, 30}; dets[1].confidence = 0.6f;

    EXPECT_EQ(nonMaxSuppression(dets, 0.3f).size(), 2u);
}

TEST(Nms, KeepsNonOverlapping) {
    std::vector<Detection> dets(2);
    dets[0].class_id = 1; dets[0].box = {0, 0, 40, 40}; dets[0].confidence = 0.9f;
    dets[1].class_id = 1; dets[1].box = {200, 200, 40, 40}; dets[1].confidence = 0.8f;
    EXPECT_EQ(nonMaxSuppression(dets, 0.5f).size(), 2u);
}

TEST(Nms, IsDeterministicOnTies) {
    std::vector<Detection> dets(4);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        dets[i].class_id = 1;
        dets[i].confidence = 0.5f;                       // 전부 동점
        dets[i].box = {static_cast<float>(i) * 500.f, 0, 40, 40};
    }
    const auto a = nonMaxSuppression(dets, 0.5f);
    const auto b = nonMaxSuppression(dets, 0.5f);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i].box.x, b[i].box.x);
}

TEST(BoxIoU, KnownValues) {
    EXPECT_FLOAT_EQ(boxIoU({0, 0, 10, 10}, {0, 0, 10, 10}), 1.0f);
    EXPECT_FLOAT_EQ(boxIoU({0, 0, 10, 10}, {20, 20, 10, 10}), 0.0f);
    // 절반 겹침: 교집합 50, 합집합 150
    EXPECT_NEAR(boxIoU({0, 0, 10, 10}, {5, 0, 10, 10}), 50.f / 150.f, 1e-6);
}

// --- TokenStore ------------------------------------------------------------

TEST(TokenStore, CreatesTokensFromDetections) {
    TokenStore store;
    const std::vector<SceneObject> objs = {
        {56, "chair", Vec3(-0.5, 0.0, 4.0), 0.6f},
        {60, "dining table", Vec3(0.8, 0.0, 5.0), 1.0f},
    };
    const auto r = render(objs, SE3::identity(), 1.0);
    const auto rep = store.integrate(r.detections, r.frame, SE3::identity(), goodEnv());

    ASSERT_TRUE(rep.ok()) << rep.error().message();
    EXPECT_EQ(rep.value().created, 2u);
    EXPECT_EQ(store.size(), 2u);
    for (const auto& t : store.allTokens()) {
        EXPECT_EQ(t->lifecycle, TokenLifecycle::Provisional);
    }
}

TEST(TokenStore, EstimatesWorldPositionFromDepth) {
    TokenStore store;
    const Vec3 truth(-0.5, 0.2, 4.0);
    const auto r = render({{56, "chair", truth, 0.6}}, SE3::identity(), 1.0);
    ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());

    const auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_LT((tokens[0]->position - truth).norm(), 0.15);
}

TEST(TokenStore, PositionIsInWorldFrameNotCameraFrame) {
    // 카메라가 원점이 아닌 곳에 있어도 월드 좌표로 저장되어야 한다
    TokenStore store;
    const SE3 T_world_cam(SO3::exp(Vec3(0.0, 0.4, 0.0)), Vec3(2.0, -1.0, 0.5));
    const Vec3 truth(2.5, -0.8, 4.0);

    const auto r = render({{56, "chair", truth, 0.6}}, T_world_cam, 1.0);
    ASSERT_FALSE(r.detections.items.empty());
    ASSERT_TRUE(store.integrate(r.detections, r.frame, T_world_cam, goodEnv()).ok());

    const auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_LT((tokens[0]->position - truth).norm(), 0.2);
}

TEST(TokenStore, MaintainsIdentityAcrossFrames) {
    TokenStore store;
    const std::vector<SceneObject> objs = {
        {56, "chair", Vec3(-0.5, 0.0, 4.0), 0.6},
        {56, "chair", Vec3(0.7, 0.0, 4.2), 0.6},
    };
    TokenId first_a, first_b;

    for (int i = 0; i < 10; ++i) {
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
        const auto tokens = store.allTokens();
        ASSERT_EQ(tokens.size(), 2u) << "프레임 " << i << " 에서 토큰이 늘거나 줄었다";
        if (i == 0) { first_a = tokens[0]->id; first_b = tokens[1]->id; }
        EXPECT_EQ(tokens[0]->id, first_a);
        EXPECT_EQ(tokens[1]->id, first_b);
    }
    for (const auto& t : store.allTokens()) {
        EXPECT_EQ(t->lifecycle, TokenLifecycle::Active);
        EXPECT_GE(t->observation_count, 10u);
    }
}

TEST(TokenStore, DoesNotSwapIdentityOfAdjacentSameClassObjects) {
    // 같은 클래스 객체가 붙어 있을 때 탐욕적 매칭이 정체성을 섞는 상황.
    // 전역 최적 할당이 이걸 막는 것이 핵심.
    TokenStore store;
    std::vector<SceneObject> objs = {
        {56, "chair", Vec3(-0.35, 0.0, 3.0), 0.5},
        {56, "chair", Vec3( 0.35, 0.0, 3.0), 0.5},
    };

    auto r0 = render(objs, SE3::identity(), 1.0);
    ASSERT_TRUE(store.integrate(r0.detections, r0.frame, SE3::identity(), goodEnv()).ok());
    auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 2u);

    const bool left_is_first = tokens[0]->position.x() < tokens[1]->position.x();
    const TokenId left_id  = left_is_first ? tokens[0]->id : tokens[1]->id;
    const TokenId right_id = left_is_first ? tokens[1]->id : tokens[0]->id;

    // 두 의자가 서로를 향해 천천히 다가온다 (겹치지는 않음)
    for (int i = 1; i <= 8; ++i) {
        objs[0].position.x() += 0.02;
        objs[1].position.x() -= 0.02;
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }

    const auto left  = store.find(left_id);
    const auto right = store.find(right_id);
    ASSERT_NE(left, nullptr);
    ASSERT_NE(right, nullptr);
    EXPECT_LT(left->position.x(), right->position.x()) << "좌우 정체성이 뒤바뀌었다";
    EXPECT_EQ(store.size(), 2u);
}

TEST(TokenStore, MissingWhileInViewErodesExistence) {
    TokenStore store;
    const std::vector<SceneObject> objs = {{56, "chair", Vec3(0.0, 0.0, 4.0), 0.6}};

    for (int i = 0; i < 8; ++i) {
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    const double before = tokens[0]->existence_belief;
    const TokenId id = tokens[0]->id;

    // 같은 시야인데 검출이 사라진다 -> 부재의 증거. 먼저 믿음이 깎여야 한다.
    const auto missFrames = [&](int n, double t0) {
        for (int i = 0; i < n; ++i) {
            auto r = render(objs, SE3::identity(), t0 + i * 0.1);
            r.detections.items.clear();
            EXPECT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
        }
    };

    missFrames(2, 1.8);
    const auto t = store.find(id);
    ASSERT_NE(t, nullptr) << "두 번 놓쳤다고 즉시 삭제하면 안 된다";
    EXPECT_LT(t->existence_belief, before);
    EXPECT_NE(t->lifecycle, TokenLifecycle::Active);

    // 계속 안 보이면 결국 부재로 확정된다.
    // 몇 프레임인지를 고정하지 않는 이유: 증거 이득을 바꾸면 그 숫자가 따라 변하는데
    // 정작 요구사항은 "유한 시간 안에 반드시 사라진다" 이다. 상한만 걸고 센다.
    int misses = 0;
    double prev = t->existence_belief;
    while (store.find(id) != nullptr && misses < 60) {
        missFrames(1, 2.0 + misses * 0.1);
        ++misses;
        const auto cur = store.find(id);
        if (cur) {
            EXPECT_LE(cur->existence_belief, prev + 1e-9) << "반증 중에 믿음이 올라갔다";
            prev = cur->existence_belief;
            if (misses % 10 == 0) {
                std::cout << "  미검출 " << misses << "회: 존재믿음=" << cur->existence_belief
                          << "  가림=" << cur->occlusion
                          << "  생애=" << static_cast<int>(cur->lifecycle) << "\n";
            }
        }
    }
    EXPECT_EQ(store.find(id), nullptr) << "지속적 반증에도 사라지지 않으면 지도를 수리할 수 없다";
    // 10 fps 기준 상한 3 초. 이보다 빠르면 YOLO 가 한두 번 놓친 것에 지도가 흔들린다.
    EXPECT_GT(misses, 5) << "몇 번 놓쳤다고 지우면 안 된다";
    EXPECT_LT(misses, 30) << "너무 오래 붙들면 사라진 물체를 계속 피해 다닌다";
}

TEST(TokenStore, OutOfViewDoesNotErodeExistence) {
    // "안 보인다 != 없다". WME 의 6대 원칙 중 하나가 여기서 지켜지는지 본다.
    TokenStore store;
    const std::vector<SceneObject> objs = {{56, "chair", Vec3(0.0, 0.0, 4.0), 0.6}};

    for (int i = 0; i < 8; ++i) {
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    const auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    const TokenId id = tokens[0]->id;
    const double before = tokens[0]->existence_belief;

    // 카메라를 180도 돌린다 -> 객체는 뒤로 간다
    const SE3 turned(SO3::exp(Vec3(0.0, kPi, 0.0)), Vec3::Zero());
    for (int i = 0; i < 10; ++i) {
        const auto r = render(objs, turned, 1.8 + i * 0.1);
        EXPECT_TRUE(r.detections.items.empty());
        ASSERT_TRUE(store.integrate(r.detections, r.frame, turned, goodEnv()).ok());
    }

    const auto t = store.find(id);
    ASSERT_NE(t, nullptr) << "시야를 벗어났다고 토큰이 사라지면 안 된다";
    EXPECT_DOUBLE_EQ(t->existence_belief, before);

    // 돌아오면 같은 ID 로 다시 붙어야 한다 (새 토큰이 생기면 안 된다)
    for (int i = 0; i < 4; ++i) {
        const auto r = render(objs, SE3::identity(), 3.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    EXPECT_EQ(store.size(), 1u);
    ASSERT_NE(store.find(id), nullptr);
    EXPECT_EQ(store.find(id)->lifecycle, TokenLifecycle::Active);
}

TEST(TokenStore, AdverseEnvironmentExtendsPersistence) {
    // 악조건에서는 놓친 관측을 더 오래 참아야 한다
    const std::vector<SceneObject> objs = {{56, "chair", Vec3(0.0, 0.0, 4.0), 0.6}};

    const auto run = [&objs](const EnvironmentState& env) {
        TokenStore store;
        for (int i = 0; i < 8; ++i) {
            const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
            store.integrate(r.detections, r.frame, SE3::identity(), goodEnv());
        }
        const SE3 turned(SO3::exp(Vec3(0.0, kPi, 0.0)), Vec3::Zero());
        // 기본 dormant 타임아웃(45 s)은 넘고, 악조건 배율(x4)은 못 넘는 시간
        for (int i = 0; i < 80; ++i) {
            const auto r = render(objs, turned, 1.8 + i * 1.0);
            store.integrate(r.detections, r.frame, turned, env);
        }
        return store.size();
    };

    EnvironmentState fog = goodEnv();
    fog.sensor_reliability = 0.2;
    fog.memory_retention_scale = 4.0;
    fog.track_persistence_scale = 3.5;

    EXPECT_EQ(run(goodEnv()), 0u) << "정상 조건에서는 제때 잊어야 한다";
    EXPECT_EQ(run(fog), 1u) << "악조건에서는 더 오래 기억해야 한다";
}

TEST(TokenStore, MovingObjectIsMarkedDynamic) {
    TokenStore store;
    std::vector<SceneObject> objs = {{0, "person", Vec3(-1.0, 0.0, 4.0), 0.5}};

    for (int i = 0; i < 12; ++i) {
        objs[0].position.x() += 0.12;   // 1.2 m/s
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    const auto tokens = store.allTokens();
    ASSERT_GE(tokens.size(), 1u);
    EXPECT_TRUE(tokens[0]->isDynamic());
    EXPECT_TRUE(store.stableLandmarks().empty()) << "움직이는 물체는 장소를 정의할 수 없다";
}

TEST(TokenStore, StationaryObjectBecomesStableLandmark) {
    TokenStore store;
    const std::vector<SceneObject> objs = {{56, "chair", Vec3(0.3, 0.0, 3.5), 0.6}};

    for (int i = 0; i < 15; ++i) {
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    const auto landmarks = store.stableLandmarks();
    ASSERT_EQ(landmarks.size(), 1u);
    EXPECT_GT(landmarks[0]->static_belief, 0.7);
    EXPECT_LT(landmarks[0]->positionSigma(), 0.5);
}

TEST(TokenStore, PersonIsDynamicFromTheFirstFrame) {
    // 자율 개체 클래스는 사전분포부터 동적이어야 한다.
    // 관측 몇 번을 기다리면 그동안 측광 정렬이 이미 오염된다.
    TokenStore store;
    const auto r = render({{0, "person", Vec3(0.0, 0.0, 3.0), 0.6}}, SE3::identity(), 1.0);
    ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());

    const auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_TRUE(tokens[0]->isDynamic());
    EXPECT_TRUE(has(tokens[0]->affordance, Affordance::Agent));
}

TEST(TokenStore, StaticMaskExcludesDynamicObjects) {
    // ECDA 가 소비하는 마스크. 고전 직접법의 약점을 없애는 지점.
    TokenStore store;
    std::vector<SceneObject> objs = {
        {0,  "person", Vec3(-0.8, 0.0, 3.0), 0.7},
        {56, "chair",  Vec3( 0.9, 0.0, 3.5), 0.6},
    };
    for (int i = 0; i < 10; ++i) {
        objs[0].position.x() += 0.10;
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }

    const auto last = render(objs, SE3::identity(), 2.2);
    const cv::Mat mask = store.buildStaticMask(last.frame, SE3::identity());
    ASSERT_FALSE(mask.empty());

    const int zeros = kW * kH - cv::countNonZero(mask);
    EXPECT_GT(zeros, 0) << "동적 객체 영역이 마스킹되지 않았다";
    EXPECT_LT(zeros, kW * kH / 2) << "정적 객체까지 지워졌다";
}

TEST(TokenStore, StaticMaskIgnoresTokensNotObservedThisFrame) {
    // 연관이 끊긴 토큰의 투영 박스는 등속 모델의 예측이지 관측이 아니다.
    // 그 자리에 실제로 있는 것은 대개 정적 배경이므로, 지우면 잃기만 한다.
    // 실측: 지운 면적의 70 %(sitting) / 86 %(walking) 가 이번 프레임에
    // 관측되지 않은 토큰에서 나왔다 (docs/06-results.md 16 장).
    TokenStore store;
    // 실제로 걷게 한다. 마스킹의 근거는 사전분포가 아니라 판정이므로,
    // 판정이 한 번 돌 만큼(static_min_dt) 관측이 이어져야 지우기 시작한다.
    std::vector<SceneObject> objs = {{0, "person", Vec3(-0.5, 0.0, 3.0), 0.7}};
    for (int i = 0; i < 9; ++i) {
        objs[0].position.x() += 0.10;                        // 1.0 m/s
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    const auto seen = render(objs, SE3::identity(), 1.9);
    EXPECT_GT(kW * kH - cv::countNonZero(store.buildStaticMask(seen.frame, SE3::identity())), 0);

    // 사람이 사라진다 (검출 없음). 토큰은 살아 있지만 마스킹 근거는 사라진다.
    Rendered empty = render({}, SE3::identity(), 2.0);
    ASSERT_TRUE(store.integrate(empty.detections, empty.frame, SE3::identity(), goodEnv()).ok());
    const cv::Mat mask = store.buildStaticMask(empty.frame, SE3::identity());
    EXPECT_EQ(kW * kH - cv::countNonZero(mask), 0)
        << "관측이 끊긴 토큰이 계속 화면을 지우고 있다";
}

TEST(TokenStore, AssociationGateIsInvariantToCameraOrientation) {
    // 게이트가 재는 혁신은 월드 좌표인데 관측 공분산은 카메라 좌표계 값이다.
    // 회전시키지 않고 더하면, 같은 장면을 세계축만 돌려 놓고 봐도 게이트가
    // 다른 답을 낸다 - 카메라가 거의 안 도는 xyz 에서는 드러나지 않고
    // halfsphere 처럼 회전이 큰 운동에서만 나타난다 (10.2 의 양 불일치 계열).
    //
    // 장면 전체를 강체 회전시켜도 마할라노비스는 같아야 한다. 판별력을 위해
    // 게이트를 확실히 넘는 도약을 넣어 gated_maha_sum 에 값이 남게 한다.
    const auto trial = [](const SO3& R) {
        TokenStore store;
        const SE3 cam(R, Vec3::Zero());

        std::vector<SceneObject> objs = {{0, "person", R * Vec3(0.0, 0.0, 3.0), 0.5}};
        auto r0 = render(objs, cam, 1.0);
        EXPECT_EQ(r0.detections.items.size(), 1u);
        EXPECT_TRUE(store.integrate(r0.detections, r0.frame, cam, goodEnv()).ok());

        objs[0].position = R * Vec3(0.0, 0.0, 3.6);          // 깊이축으로 0.6 m 도약
        auto r1 = render(objs, cam, 1.0 + 1.0 / 30.0);
        const auto rep = store.integrate(r1.detections, r1.frame, cam, goodEnv());
        EXPECT_TRUE(rep.ok());
        return rep.value();
    };

    const auto a = trial(SO3::identity());
    const auto b = trial(SO3::exp(Vec3(0.0, 1.05, 0.0)));   // 약 60도 요

    ASSERT_EQ(a.det_gated_out, 1u) << "게이트를 넘지 못했다 - 이 케이스는 아무것도 재지 못한다";
    ASSERT_EQ(b.det_gated_out, 1u);
    EXPECT_NEAR(a.gated_maha_sum, b.gated_maha_sum, 1e-3 * a.gated_maha_sum)
        << "세계축을 돌렸을 뿐인데 게이트가 다른 답을 냈다 (" << a.gated_maha_sum
        << " vs " << b.gated_maha_sum << ")";
}

TEST(TokenStore, UnjudgedTokenDoesNotMask) {
    // 사전분포는 사전분포이지 판결이 아니다. 판정이 한 번도 안 돈 토큰으로
    // 화면을 지우면, 그것은 14.1 이 기각한 클래스 마스킹을 한 단계 아래에서
    // 다시 하는 것이다. 실측 대가는 fr3_sitting_halfsphere 에서 지운 면적
    // 64 % 중 32.5 pp 이고, 그 결과 ECDA 가 프레임당 130 점으로 굶어
    // 259 프레임 중 141 번 정렬에 실패했다.
    TokenStore store;
    std::vector<SceneObject> objs = {{0, "person", Vec3(-0.5, 0.0, 3.0), 0.7}};

    Rendered r = render(objs, SE3::identity(), 1.0);
    ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    ASSERT_EQ(store.allTokens().size(), 1u);
    ASSERT_TRUE(store.allTokens()[0]->isDynamic()) << "사전분포는 회의여야 한다";
    EXPECT_EQ(kW * kH - cv::countNonZero(store.buildStaticMask(r.frame, SE3::identity())), 0)
        << "판정 전 토큰이 화면을 지우고 있다 - 사전분포를 판결로 쓰는 것이다";

    // 판정 창이 찰 때까지 걷는다. 그 다음부터는 지워야 한다.
    for (int i = 1; i <= 8; ++i) {
        objs[0].position.x() += 0.10;                        // 1.0 m/s
        r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    ASSERT_GT(store.allTokens()[0]->static_diag.updates, 0u) << "판정이 아예 안 돌았다";
    ASSERT_TRUE(store.allTokens()[0]->static_evidenced);
    EXPECT_GT(kW * kH - cv::countNonZero(store.buildStaticMask(r.frame, SE3::identity())), 0)
        << "판정이 동적이라고 결론냈는데도 지우지 않는다";

    // 보류된 면적은 보고에 남아야 한다 - 사전분포가 지우려던 몫이 얼마였는지는
    // 사후에 알 수 있어야 한다.
    TokenStore fresh;
    const auto f0 = render(objs, SE3::identity(), 5.0);
    ASSERT_TRUE(fresh.integrate(f0.detections, f0.frame, SE3::identity(), goodEnv()).ok());
    TokenStore::MaskReport mr;
    (void)fresh.buildStaticMask(f0.frame, SE3::identity(), &mr);
    EXPECT_EQ(mr.n_masking, 0u);
    EXPECT_EQ(mr.n_withheld, 1u);
    EXPECT_GT(mr.withheld_unjudged, 0.0);
}

TEST(TokenStore, MotionVerdictSurvivesAnAssociationBreak) {
    // 연관이 끊겨 새 토큰이 생겼다고 해서 걷던 사람이 미판정 상태로 돌아가면
    // 안 된다. 운동 상태는 트랙 ID 의 속성이 아니라 그 자리에 있는 것의
    // 속성이다. 실측 대가: 이 항이 없으면 수명 중앙값 5 관측인
    // fr3_walking_halfsphere 의 짧은 트랙들이 마스킹에서 통째로 빠져
    // ATE 10.55 -> 19.24 cm.
    TokenStore store;
    std::vector<SceneObject> objs = {{0, "person", Vec3(-0.5, 0.0, 3.0), 0.7}};
    for (int i = 0; i < 9; ++i) {
        objs[0].position.x() += 0.10;                        // 1.0 m/s
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    ASSERT_EQ(store.allTokens().size(), 1u);
    const double judged = store.allTokens()[0]->static_belief;
    ASSERT_TRUE(store.allTokens()[0]->static_evidenced);
    EXPECT_LT(judged, 0.4) << "걷는 사람이 동적으로 판정되지 않았다";

    // 연관을 강제로 끊는다: 같은 자리에서 게이트 밖으로 도약시킨다.
    objs[0].position.x() += 1.2;
    const auto r = render(objs, SE3::identity(), 1.9 + 1.0 / 30.0);
    const auto rep = store.integrate(r.detections, r.frame, SE3::identity(), goodEnv());
    ASSERT_TRUE(rep.ok());
    ASSERT_EQ(rep.value().created, 1u) << "연관이 안 끊겼다 - 이 케이스는 아무것도 재지 못한다";

    // 새 토큰은 사전분포로 돌아가지 않고 판정을 물려받아야 한다.
    const auto tokens = store.allTokens();
    const WorldToken& fresh = *tokens.back();
    EXPECT_TRUE(fresh.static_evidenced) << "판정이 연관 단절에서 살아남지 못했다";
    EXPECT_DOUBLE_EQ(fresh.static_belief, judged);
    EXPECT_GT(kW * kH - cv::countNonZero(store.buildStaticMask(r.frame, SE3::identity())), 0)
        << "연관이 끊겼다는 이유만으로 걷는 사람이 마스킹에서 빠졌다";
}

TEST(TokenStore, AssociationSurvivesFastMotion) {
    // 게이트를 융합 공분산만으로 세우면, 상관된 관측으로 공분산이 무너진 뒤
    // 실제 운동이 전부 이상치가 되어 토큰이 매 프레임 새로 생긴다.
    // 실측: 자율개체 토큰 수명 중앙값 1~2 관측, 게이트 탈락 11~20 %.
    TokenStore store;
    std::vector<SceneObject> objs = {{0, "person", Vec3(-1.0, 0.0, 3.0), 0.7}};
    for (int i = 0; i < 20; ++i) {
        objs[0].position.x() += 0.04;                        // 1.2 m/s @ 30 Hz
        const auto r = render(objs, SE3::identity(), 1.0 + i / 30.0);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    EXPECT_EQ(store.size(), 1u) << "걷는 사람이 매 프레임 새 토큰이 되고 있다";
    ASSERT_EQ(store.allTokens().size(), 1u);
    EXPECT_GE(store.allTokens()[0]->observation_count, 18u);
}

TEST(TokenStore, MotionlessPersonEventuallyStopsBeingMasked) {
    // 14.1 의 대가가 여기서 갚아진다. "person" 은 클래스이지 운동 상태가 아니고,
    // 앉아 있는 사람은 벽만큼 좋은 랜드마크다. 사전분포는 회의로 시작하되
    // 관측이 결론을 바꿀 수 있어야 한다.
    TokenStore store;
    const std::vector<SceneObject> objs = {{0, "person", Vec3(0.0, 0.0, 3.0), 0.7}};

    Rendered r = render(objs, SE3::identity(), 1.0);
    ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    ASSERT_EQ(store.allTokens().size(), 1u);
    EXPECT_TRUE(store.allTokens()[0]->isDynamic()) << "사전분포는 회의여야 한다";

    // 5 초간 완전히 정지한 사람을 관측한다 (30 Hz).
    for (int i = 1; i < 150; ++i) {
        r = render(objs, SE3::identity(), 1.0 + i / 30.0);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    const auto& tok = *store.allTokens()[0];
    EXPECT_FALSE(tok.isDynamic())
        << "5 초간 정지한 사람이 여전히 동적이다 (static_belief=" << tok.static_belief << ")";
    EXPECT_EQ(kW * kH - cv::countNonZero(store.buildStaticMask(r.frame, SE3::identity())), 0);
}

TEST(TokenStore, MergeCombinesTwoTokens) {
    TokenStore store;
    const std::vector<SceneObject> objs = {
        {56, "chair", Vec3(-1.2, 0.0, 4.0), 0.6},
        {56, "chair", Vec3( 1.2, 0.0, 4.0), 0.6},
    };
    for (int i = 0; i < 5; ++i) {
        const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
        ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    }
    auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 2u);

    const TokenId keep = tokens[0]->id, absorb = tokens[1]->id;
    const std::uint32_t total = tokens[0]->observation_count + tokens[1]->observation_count;

    ASSERT_TRUE(store.merge(keep, absorb).ok());
    EXPECT_EQ(store.size(), 1u);
    ASSERT_NE(store.find(keep), nullptr);
    EXPECT_EQ(store.find(keep)->observation_count, total);
    EXPECT_EQ(store.find(absorb), nullptr);
}

TEST(TokenStore, MergeRejectsUnknownTokens) {
    TokenStore store;
    EXPECT_FALSE(store.merge(TokenId(1), TokenId(2)).ok());
    EXPECT_FALSE(store.merge(TokenId(1), TokenId(1)).ok());
}

TEST(TokenStore, RejectsInvalidInput) {
    TokenStore store;
    Frame bad;
    DetectionSet ds;
    ds.stamp = Timestamp::fromSeconds(1.0);
    const auto r = store.integrate(ds, bad, SE3::identity(), goodEnv());
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, ErrorCode::InvalidArgument);
}

TEST(TokenStore, HandlesMissingDepthGracefully) {
    // 깊이가 없으면 위치를 지어내지 말고 불확실성을 크게 잡아야 한다
    TokenStore store;
    auto r = render({{56, "chair", Vec3(0.0, 0.0, 4.0), 0.6}}, SE3::identity(), 1.0);
    r.frame.depth = cv::Mat();
    r.frame.sensor = SensorKind::Monocular;

    ASSERT_TRUE(store.integrate(r.detections, r.frame, SE3::identity(), goodEnv()).ok());
    const auto tokens = store.allTokens();
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_GT(tokens[0]->positionSigma(), 1.0);
    EXPECT_FALSE(tokens[0]->isStableLandmark());
}

TEST(TokenStore, DeterministicTokenOrdering) {
    // 해시 순회 순서가 결과에 새어나오면 재현이 불가능해진다
    const std::vector<SceneObject> objs = {
        {56, "chair", Vec3(-1.0, 0.0, 4.0), 0.6},
        {60, "dining table", Vec3(0.0, 0.0, 5.0), 1.0},
        {0,  "person", Vec3(1.0, 0.0, 3.5), 0.5},
    };
    const auto run = [&objs] {
        TokenStore store;
        for (int i = 0; i < 6; ++i) {
            const auto r = render(objs, SE3::identity(), 1.0 + i * 0.1);
            store.integrate(r.detections, r.frame, SE3::identity(), goodEnv());
        }
        std::vector<std::uint64_t> ids;
        for (const auto& t : store.allTokens()) ids.push_back(t->id.value);
        return ids;
    };
    EXPECT_EQ(run(), run());
}

// dynamic_area_ratio 의 합산 순서.
//
// 이 합은 부동소수 덧셈이라 순서에 따라 마지막 비트가 달라진다. 예전에는
// tokens_(unordered_map)를 그대로 훑어 더했으므로 그 순서가 해시 구현에
// 딸려 있었다. 지금은 allTokens() 와 같은 ID 정렬 경로를 쓴다.
//
// 판별력의 한계를 정직하게 적는다. 두 가지 이유로 이 케이스는 고치기 전
// 코드에서도 통과한다.
//   1) MSVC 의 정수 키 해시가 항등이고 버킷 수가 원소 수보다 훨씬 커서
//      unordered_map 순회가 ID 오름차순과 우연히 일치한다.
//   2) 더 근본적으로, 더하는 값이 float 박스 면적(가수 24 비트)이고 항이
//      수십 개뿐이라 부분합이 double 에 정확히 들어간다. 합산 순서를 실제로
//      뒤집어 봐도 비트가 같았다.
// 그래서 이 케이스는 "결함을 재현" 하는 것이 아니라 "합은 ID 순서로 정의된다"
// 는 규약을 비트 단위로 못박는다. 값 범위가 넓어지면 그때 이 규약이 답을 정한다.
TEST(TokenStoreDeterminism, DynamicAreaIsSummedInIdOrder) {
    // 사람(Agent 사전분포 -> 동적) 을 여러 개 둔다. 박스 크기를 크게 벌려
    // 덧셈 순서가 바뀌면 마지막 비트가 실제로 달라지게 만든다.
    std::vector<SceneObject> objs;
    for (int i = 0; i < 12; ++i) {
        SceneObject o;
        o.class_id = 0;
        o.name     = "person";
        o.position = Vec3(-3.0 + 0.55 * i, 0.0, 4.0 + 0.05 * i);
        o.size     = 0.30 + 0.55 * static_cast<double>(i % 4);
        objs.push_back(o);
    }

    TokenStore store;
    const SE3 pose = SE3::identity();

    Result<IntegrationReport> last{IntegrationReport{}};
    for (int k = 0; k < 4; ++k) {
        const auto scene = render(objs, pose, 1.0 + 0.1 * k);
        last = store.integrate(scene.detections, scene.frame, pose, goodEnv());
        ASSERT_TRUE(last.ok());
    }
    ASSERT_GE(store.size(), 8u);

    // allTokens() 는 ID 오름차순을 보장한다. 같은 순서로 다시 더한 값과
    // 비트 단위로 같아야 한다.
    double expected = 0.0;
    for (const auto& t : store.allTokens()) {
        if (t->isDynamic() && t->miss_count == 0) expected += t->box.area();
    }
    const double screen = static_cast<double>(kW) * static_cast<double>(kH);
    const double want   = std::clamp(expected / std::max(1.0, screen), 0.0, 1.0);

    EXPECT_GT(expected, 0.0) << "동적 토큰이 하나도 없다 - 이 케이스는 아무것도 재지 못한다";
    EXPECT_EQ(0, std::memcmp(&last.value().dynamic_area_ratio, &want, sizeof(double)))
        << "동적 면적 합이 ID 순서로 계산되지 않았다";
}
