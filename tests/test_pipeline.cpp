// 엔드투엔드 파이프라인 - 06-results.md 26 절의 "C++ 에는 E2E 테스트가 없다".
//
// 그 항목은 이렇게 적혀 있었다: *"서브시스템은 개별로 통과하지만 C++ 에서
// camera -> tokens -> pose -> map 이 도는 것은 아무 것도 없다."*
//
// 개별 통과가 통합 동작을 뜻하지 않는 이유는 이 문서가 반복해서 보여 준다:
// 11.3 의 Displaced 영구 유령, 14.2 의 세 겹 override, 16.5 의 "믿음은 계산되고
// 곧바로 우회되었다" 는 전부 각 부분이 자기 테스트를 통과하는 상태에서 일어났다.
// 여기서 재는 것은 정확도가 아니라 **연결** 이다: 한 단계의 출력이 다음 단계의
// 입력으로 실제로 흘러가는가, 그리고 흐름이 끊기면 조용히가 아니라 시끄럽게
// 끊기는가.
//
// 합성 장면을 쓴다. 실데이터는 도구(wme_tum_odometry)가 담당하고, 여기서는
// 진리값을 아는 상태로 배선을 검사한다.

#include "wme/confidence/ConfidenceEngine.hpp"
#include "wme/fusion/PoseFusion.hpp"
#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/ImageQualityEngine.hpp"
#include "wme/token/ConstellationIndex.hpp"
#include "wme/token/TokenStore.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

using namespace wme;

namespace {

constexpr int kW = 320, kH = 240;

CameraIntrinsics makeK() {
    CameraIntrinsics K;
    K.fx = 250.0; K.fy = 250.0;
    K.cx = kW / 2.0 - 0.5; K.cy = kH / 2.0 - 0.5;
    K.width = kW; K.height = kH;
    return K;
}

cv::Mat makeTexture(unsigned seed) {
    cv::Mat small(kH / 8, kW / 8, CV_32F);
    cv::RNG rng(seed);
    rng.fill(small, cv::RNG::NORMAL, 0.0, 1.0);
    cv::Mat big;
    cv::resize(small, big, cv::Size(kW, kH), 0, 0, cv::INTER_CUBIC);
    cv::Mat fine(kH, kW, CV_32F);
    rng.fill(fine, cv::RNG::NORMAL, 0.0, 1.0);
    cv::Mat img = 128.0 + 45.0 * big + 10.0 * fine;
    // 엔진은 8비트 그레이를 받는다. CV_32F 를 넘기면 정렬이 조용히 실패하는 게
    // 아니라 아예 실패를 보고한다 - 그 점은 좋지만, 여기서 재려는 것은 그게
    // 아니므로 실제 센서와 같은 형식으로 준다.
    cv::Mat out;
    img.convertTo(out, CV_8UC1);
    return out;
}

// 카메라를 마주보는 기울어진 평면의 깊이맵.
cv::Mat planeDepth(const CameraIntrinsics& K, double dist) {
    cv::Mat d(kH, kW, CV_32F);
    const Vec3 n = Vec3(0.12, 0.08, 0.99).normalized();
    for (int y = 0; y < kH; ++y) {
        auto* row = d.ptr<float>(y);
        for (int x = 0; x < kW; ++x) {
            const Vec3 ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0);
            row[x] = static_cast<float>(dist / n.dot(ray));
        }
    }
    return d;
}

Frame makeFrame(const cv::Mat& gray, const cv::Mat& depth, double t, std::uint64_t id) {
    Frame f;
    f.id = FrameId(id);
    // 0 은 무효 시각 규약이다. 그냥 0 을 넣으면 Frame::valid() 가 false 가 되고
    // align 이 "프레임이 유효하지 않음" 으로 실패한다 - 실제로 그렇게 한 번 걸렸다.
    f.stamp = Timestamp::fromSeconds(t + 1.0);
    f.gray = gray;
    f.depth = depth;
    f.intrinsics = makeK();
    f.sensor = depth.empty() ? SensorKind::Monocular : SensorKind::RgbD;
    return f;
}

// 평면 단응사상으로 ref 를 워프한다.
cv::Mat warpPlane(const cv::Mat& src, const CameraIntrinsics& K,
                  const SE3& T_cur_ref, double dist) {
    const Vec3 n = Vec3(0.12, 0.08, 0.99).normalized();
    Mat3 Kx;
    Kx << K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1;
    const Mat3 Hm = Kx * (T_cur_ref.rotation().matrix()
                          + T_cur_ref.translation() * n.transpose() / dist) * Kx.inverse();
    cv::Matx33d Hcv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) Hcv(r, c) = Hm(r, c);
    cv::Mat out;
    cv::warpPerspective(src, out, cv::Mat(Hcv), src.size(),
                        cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

// 합성 검출 하나. YOLO 백엔드 없이 토큰 경로를 돌리기 위한 것이다.
Detection makeDetection(int cls, const char* name, float cx, float cy, float w, float h) {
    Detection d;
    d.class_id = cls;
    d.class_name = name;
    d.box = {cx - w / 2, cy - h / 2, w, h};
    d.confidence = 0.9F;
    return d;
}

}  // namespace

// ===========================================================================
// 1. camera -> quality -> pose. 한 단계라도 끊기면 여기서 멈춘다.
// ===========================================================================

TEST(Pipeline, ImageQualityFeedsAlignmentAndPoseAccumulates) {
    const CameraIntrinsics K = makeK();
    const cv::Mat tex = makeTexture(11);
    const cv::Mat depth = planeDepth(K, 2.5);

    ImageQualityEngine iq;
    DirectAligner aligner;

    // 프레임마다 같은 증분으로 움직인다. 누적 포즈가 그만큼 늘어야 한다.
    const SE3 step(SO3::exp(Vec3(0.002, -0.003, 0.001)), Vec3(0.02, -0.01, 0.005));

    SE3 T_world_cam = SE3::identity();
    SE3 T_ref_true = SE3::identity();
    cv::Mat ref_gray = tex;
    int aligned = 0;

    for (int i = 1; i <= 5; ++i) {
        T_ref_true = step * T_ref_true;
        const cv::Mat cur_gray = warpPlane(tex, K, T_ref_true, 2.5);

        const Frame ref = makeFrame(ref_gray, depth, 0.0, 1);
        const Frame cur = makeFrame(cur_gray, cv::Mat(), i * 0.033, static_cast<std::uint64_t>(i + 1));

        // 품질은 실제로 정렬에 전달된다 - 로버스트 커널의 잡음 기준이 여기서 온다.
        const ImageQuality q = iq.evaluate(ref);
        ASSERT_GT(q.noise_sigma, 0.0) << "품질 채널이 죽어 있으면 커널이 물리량에 묶이지 않는다";

        const auto r = aligner.align(ref, cur, SE3::identity(), &q);
        ASSERT_TRUE(r.ok()) << "프레임 " << i << " 정렬 실패";
        ++aligned;
        T_world_cam = r.value().T_cur_ref.inverse();
    }

    ASSERT_EQ(aligned, 5);
    // 누적 포즈가 진리값 방향으로 자란다. 정확도가 아니라 배선을 재는 것이므로
    // 느슨하게 본다 - 배선이 끊기면 항등에 머물러 이 검사에 걸린다.
    const double moved = T_world_cam.translation().norm();
    const double truth = T_ref_true.inverse().translation().norm();
    EXPECT_GT(moved, 0.5 * truth) << "누적 포즈가 자라지 않는다 - 배선이 끊겼다";
    EXPECT_LT(moved, 2.0 * truth);
}

// ===========================================================================
// 2. detections -> tokens -> constellation. 객체 경로 전체.
// ===========================================================================

TEST(Pipeline, DetectionsBecomeTokensAndTokensBecomeAConstellation) {
    const CameraIntrinsics K = makeK();
    const cv::Mat depth = planeDepth(K, 2.5);

    TokenStore store;
    EnvironmentState env;
    env.sensor_reliability = 1.0;
    env.track_persistence_scale = 1.0;
    env.memory_retention_scale = 1.0;

    // 서로 다른 클래스 다섯 - min_nodes(4) 를 넘겨야 성좌가 만들어진다.
    const std::vector<Detection> dets = {
        makeDetection(56, "chair", 60, 90, 40, 40),
        makeDetection(62, "tv", 150, 70, 50, 35),
        makeDetection(63, "laptop", 240, 120, 45, 30),
        makeDetection(41, "cup", 100, 170, 25, 25),
        makeDetection(64, "mouse", 200, 190, 20, 18),
    };

    Frame f = makeFrame(makeTexture(3), depth, 0.0, 1);
    DetectionSet ds;
    ds.items = dets;
    ds.stamp = f.stamp;

    // 같은 관측을 여러 프레임 넣어 토큰이 승격되게 한다.
    for (int i = 0; i < 6; ++i) {
        f.stamp = Timestamp::fromSeconds(i * 0.1);
        ds.stamp = f.stamp;
        store.integrate(ds, f, SE3::identity(), env);
    }

    const auto tokens = store.activeTokens();
    ASSERT_GE(tokens.size(), dets.size())
        << "검출이 토큰으로 승격되지 않았다 - detection->token 배선이 끊겼다";

    // 토큰이 성좌 노드가 되고, 그 성좌가 색인에 들어가 자기 자신을 찾아야 한다.
    std::vector<ConstellationNode> nodes;
    for (const auto& t : tokens) {
        ConstellationNode n;
        n.id = t->id;
        n.class_id = t->class_id;
        n.position = t->position;
        nodes.push_back(n);
    }
    ASSERT_GE(nodes.size(), 4u) << "성좌를 만들 만큼의 토큰이 없다";

    ConstellationIndex index;
    index.insert(KeyframeId(1), Timestamp::fromSeconds(1.0), SE3::identity(), nodes);

    const auto m = index.query(nodes);
    ASSERT_TRUE(m.ok()) << "자기 자신을 질의했는데 찾지 못했다 - token->constellation 단절";
    // 자기질의는 항등이어야 한다. 아니면 좌표계가 어딘가에서 뒤집혔다.
    EXPECT_LT(m.value().transform.translation().norm(), 1e-6);
}

// ===========================================================================
// 3. pose + tiers -> fusion. 세 tier 가 실제로 한 포즈로 합쳐지는가.
// ===========================================================================

TEST(Pipeline, ThreeTiersFuseIntoOnePose) {
    const SE3 truth(SO3::exp(Vec3(0.003, -0.002, 0.001)), Vec3(0.03, -0.02, 0.01));

    std::vector<fusion::TierEstimate> es;
    const fusion::Tier tiers[3] = {fusion::Tier::Photometric,
                                   fusion::Tier::Constellation,
                                   fusion::Tier::Structural};
    const double scales[3] = {1e4, 1e3, 1e2};
    for (int k = 0; k < 3; ++k) {
        fusion::TierEstimate e;
        e.tier = tiers[k];
        // 각 tier 가 조금씩 다른 답을 낸다 - 융합이 실제로 섞는지 보려면 달라야 한다.
        e.T_cur_ref = SE3::exp(truth.log() + Vec6::Constant(0.001 * (k + 1)));
        e.information = scales[k] * Mat6::Identity();
        e.alpha = 1.0;
        e.available = true;
        es.push_back(e);
    }

    const auto r = fusion::fuse(es);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().contributing_tiers, 3)
        << "세 tier 를 줬는데 전부 기여하지 않았다";

    // 융합 결과는 가장 정보가 큰 tier 쪽에 가까워야 한다.
    const double d_fused = (r.value().T_cur_ref.inverse() * es[0].T_cur_ref).log().norm();
    const double d_weak  = (r.value().T_cur_ref.inverse() * es[2].T_cur_ref).log().norm();
    EXPECT_LT(d_fused, d_weak) << "정보 가중이 반영되지 않았다";
}

// ===========================================================================
// 4. 끊김은 조용하지 않아야 한다 - 이 문서가 반복해서 잡은 실패 양상
// ===========================================================================

TEST(Pipeline, ABrokenStageFailsLoudlyRatherThanSilently) {
    const CameraIntrinsics K = makeK();
    const cv::Mat tex = makeTexture(5);

    // 깊이 없는 ref: ECDA 는 3D 점을 만들 수 없다. 조용히 항등을 돌려주면
    // 상위에서 "정렬 성공, 카메라 정지" 로 읽힌다 - 22.5 의 no_output 이
    // 바닥값 점수를 받은 것과 같은 사고다.
    const Frame ref = makeFrame(tex, cv::Mat(), 0.0, 1);
    const Frame cur = makeFrame(tex, cv::Mat(), 0.033, 2);

    DirectAligner aligner;
    const auto r = aligner.align(ref, cur, SE3::identity());
    EXPECT_FALSE(r.ok()) << "깊이 없이도 성공을 보고했다";
}
