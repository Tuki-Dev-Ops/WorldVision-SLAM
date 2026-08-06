#pragma once

// 결정성 하네스가 실제로 재는 대상들.
//
// 이 헤더는 두 실행 파일이 공유한다:
//   test_determinism        - 원본 소스로 빌드. 두 실행의 Blob 이 같아야 한다.
//   test_determinism_mutant - 순서 의존성을 일부러 주입한 소스로 빌드.
//                             같은 비교가 그 결함을 반드시 잡아내야 한다.
//
// 대조군을 같은 코드로 돌리는 이유는 하나다. 통과하는 테스트는 그 테스트가
// 실패할 수 있다는 증거가 없으면 아무것도 말해 주지 않는다.

#include "wme/core/TestSupport.hpp"
#include "wme/core/ThreadPool.hpp"
#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/Detection.hpp"
#include "wme/token/ConstellationIndex.hpp"
#include "wme/token/TokenStore.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace wme::testcases {

// ---------------------------------------------------------------------------
// 공통 장면 합성
// ---------------------------------------------------------------------------

inline constexpr int kW = 640, kH = 480;

inline CameraIntrinsics makeK() {
    CameraIntrinsics K;
    K.fx = 500.0; K.fy = 500.0;
    K.cx = kW * 0.5 - 0.5; K.cy = kH * 0.5 - 0.5;
    K.width = kW; K.height = kH;
    return K;
}

// 그래디언트가 고르게 깔린 결정적 텍스처. 시드를 고정하므로 실행마다 같다.
inline cv::Mat makeTexture(int w, int h, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> px(0, w - 1), py(0, h - 1);
    std::uniform_int_distribution<int> rad(6, 30), col(30, 225);

    cv::Mat img(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        auto* row = img.ptr<std::uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            row[x] = static_cast<std::uint8_t>(80 + 60.0 * x / w + 40.0 * y / h);
        }
    }
    const int n_circle = 900 * w * h / (kW * kH);
    const int n_line   = 300 * w * h / (kW * kH);
    for (int i = 0; i < n_circle; ++i) {
        cv::circle(img, {px(gen), py(gen)}, rad(gen), cv::Scalar(col(gen)), cv::FILLED);
    }
    for (int i = 0; i < n_line; ++i) {
        cv::line(img, {px(gen), py(gen)}, {px(gen), py(gen)}, cv::Scalar(col(gen)), 2);
    }
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    return img;
}

inline cv::Mat makePlaneDepth(const CameraIntrinsics& K, const Vec3& n, double dist) {
    cv::Mat depth(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < kW; ++x) {
            const Vec3 ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0);
            row[x] = static_cast<float>(dist / n.dot(ray));
        }
    }
    return depth;
}

inline cv::Mat planeHomography(const CameraIntrinsics& K, const SE3& T, const Vec3& n, double d) {
    Mat3 Km;
    Km << K.fx, 0.0, K.cx,
          0.0, K.fy, K.cy,
          0.0, 0.0, 1.0;
    const Mat3 H = Km * (T.rotation().matrix() + T.translation() * n.transpose() / d) * Km.inverse();
    cv::Mat h(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) h.at<double>(r, c) = H(r, c);
    }
    return h;
}

struct AlignScene {
    Frame ref, cur;
    SE3   truth;
};

// 여백을 둔 캔버스에서 워프한 뒤 가운데를 잘라 화면 밖 가짜 텍스처를 없앤다.
inline AlignScene makeAlignScene(int pyramid_levels = 5) {
    constexpr int kMargin = 120;
    const CameraIntrinsics K = makeK();
    const Vec3   n = Vec3(0.15, 0.10, 0.98).normalized();
    const double dist = 2.5;

    const int PW = kW + 2 * kMargin, PH = kH + 2 * kMargin;
    CameraIntrinsics Kp = K;
    Kp.cx += kMargin; Kp.cy += kMargin;
    Kp.width = PW;    Kp.height = PH;

    const cv::Mat tex_pad = makeTexture(PW, PH, 5150);
    const SE3 T(SO3::exp(Vec3(0.010, -0.015, 0.008)), Vec3(0.06, -0.03, 0.02));

    cv::Mat cur_pad;
    cv::warpPerspective(tex_pad, cur_pad, planeHomography(Kp, T, n, dist), tex_pad.size(),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    const cv::Rect roi(kMargin, kMargin, kW, kH);

    AlignScene s;
    s.truth = T;

    s.ref.id    = FrameId(1);
    s.ref.stamp = Timestamp::fromSeconds(1.0);
    s.ref.gray  = tex_pad(roi).clone();
    s.ref.depth = makePlaneDepth(K, n, dist);
    s.ref.intrinsics = K;
    s.ref.sensor = SensorKind::RgbD;
    s.ref.pyramid = ImagePyramid::build(s.ref.gray, K, pyramid_levels);

    s.cur.id    = FrameId(2);
    s.cur.stamp = Timestamp::fromSeconds(1.033);
    s.cur.gray  = cur_pad(roi).clone();
    s.cur.intrinsics = K;
    s.cur.sensor = SensorKind::Monocular;
    s.cur.pyramid = ImagePyramid::build(s.cur.gray, K, pyramid_levels);

    return s;
}

// ---------------------------------------------------------------------------
// 1) DirectAligner::align
// ---------------------------------------------------------------------------

// 정렬 결과에서 관측 가능한 모든 값을 원시 비트로 기록한다.
// 하나라도 빠뜨리면 그 축의 비결정성은 영원히 안 보인다.
inline void putAlignment(test::Blob& blob, const Result<AlignmentResult>& r) {
    blob.put(r.ok());
    blob.put(r.degraded());
    blob.put(r.reliability());
    if (!r.ok()) {
        blob.put(static_cast<int>(r.error().code));
        return;
    }
    const AlignmentResult& v = r.value();
    blob.put(v.T_cur_ref);
    blob.putEigen(v.information);
    blob.putEigen(v.eigenvalues);
    blob.putEigen(v.weakest_direction);
    blob.put(v.affine_a);
    blob.put(v.affine_b);
    blob.put(v.photometric_rmse);
    blob.put(v.inlier_ratio);
    blob.put(static_cast<std::uint64_t>(v.point_count));
    blob.put(static_cast<std::uint64_t>(v.inlier_count));
    blob.put(v.iterations);
    blob.put(v.observable_dof);
}

// 같은 인스턴스로 repeat 회 반복한다. 내부 버퍼 재사용이 결과를 오염시키는지도
// 함께 보기 위해 매 회의 결과를 모두 기록한다.
inline test::Blob alignBlob(const AlignScene& s, ThreadPool* pool, int repeat = 3) {
    DirectAligner aligner({}, pool);
    test::Blob blob;
    for (int i = 0; i < repeat; ++i) {
        const auto r = aligner.align(s.ref, s.cur, SE3::identity());
        putAlignment(blob, r);
        for (std::size_t n : aligner.pointsPerLevel()) blob.put(static_cast<std::uint64_t>(n));
    }
    return blob;
}

// ---------------------------------------------------------------------------
// 2) TokenStore::integrate
// ---------------------------------------------------------------------------

// 같은 클래스의 물체를 가깝게 두 개 놓는다. 연관이 유일하면 순서 의존성이
// 결과에 아예 나타나지 않아 측정이 아무것도 판별하지 못한다
// (docs/06-results.md 10.4 - 0.70 m 떨어진 물체를 0.35 m 게이트로 재던 실험).
struct WorldObject {
    Vec3        position;
    int         class_id;
    const char* class_name;
    double      size;
};

inline std::vector<WorldObject> makeWorldObjects() {
    return {
        {Vec3(-0.60, 0.00, 4.0), 56, "chair", 0.45},
        {Vec3(-0.15, 0.02, 4.1), 56, "chair", 0.45},   // 같은 클래스, 가까움 -> 모호
        {Vec3( 0.90, 0.05, 3.6), 56, "chair", 0.45},
        {Vec3( 0.20, -0.30, 5.2), 60, "dining table", 0.80},
        {Vec3(-1.10, 0.40, 5.8), 0,  "person", 0.35},
        {Vec3( 1.40, -0.10, 6.4), 62, "tv", 0.50},
        {Vec3(-0.30, 0.55, 7.0), 41, "cup", 0.12},
        {Vec3( 0.70, 0.50, 7.4), 41, "cup", 0.12},     // 같은 클래스, 가까움
    };
}

// 카메라가 옆으로 미끄러지는 시퀀스. 프레임마다 결정적 관측 잡음을 준다.
struct TokenFrameInput {
    DetectionSet     detections;
    Frame            frame;
    SE3              T_world_cam;
    EnvironmentState env;
};

inline std::vector<TokenFrameInput> makeTokenSequence(int frames = 12) {
    const CameraIntrinsics K = makeK();
    const auto objects = makeWorldObjects();

    // 결정적 잡음원. 실행마다 같은 수열이어야 한다.
    std::mt19937 gen(20240816u);
    std::uniform_real_distribution<double> jitter(-1.0, 1.0);

    std::vector<TokenFrameInput> seq;
    seq.reserve(static_cast<std::size_t>(frames));

    cv::Mat gray(kH, kW, CV_8UC1, cv::Scalar(120));

    for (int f = 0; f < frames; ++f) {
        TokenFrameInput in;
        const double t = 1.0 + 0.1 * f;

        in.T_world_cam = SE3(SO3::exp(Vec3(0.0, 0.004 * f, 0.0)), Vec3(0.05 * f, 0.0, 0.0));
        const SE3 T_cam_world = in.T_world_cam.inverse();

        in.frame.id    = FrameId(static_cast<std::uint64_t>(f + 1));
        in.frame.stamp = Timestamp::fromSeconds(t);
        in.frame.gray  = gray;
        in.frame.intrinsics = K;
        in.frame.sensor = SensorKind::RgbD;
        in.frame.depth = cv::Mat(kH, kW, CV_32F, cv::Scalar(50.f));

        in.env.sensor_reliability = 0.9;
        in.env.track_persistence_scale = 1.0;
        in.env.memory_retention_scale  = 1.0;

        in.detections.stamp = Timestamp::fromSeconds(t);
        in.detections.frame = in.frame.id;

        for (std::size_t oi = 0; oi < objects.size(); ++oi) {
            const WorldObject& o = objects[oi];
            const Vec3 p_cam = T_cam_world * o.position;
            if (p_cam.z() < 0.5) continue;

            const Vec2 c = K.project(p_cam);
            const double half_w = 0.5 * o.size * K.fx / p_cam.z();
            const double half_h = 0.5 * o.size * K.fy / p_cam.z();

            const double dx = 0.6 * jitter(gen), dy = 0.6 * jitter(gen);
            cv::Rect2f box(static_cast<float>(c.x() - half_w + dx),
                           static_cast<float>(c.y() - half_h + dy),
                           static_cast<float>(2.0 * half_w),
                           static_cast<float>(2.0 * half_h));
            if (box.x < 0.f || box.y < 0.f ||
                box.x + box.width >= kW || box.y + box.height >= kH) continue;

            // 깊이맵에도 이 물체를 찍어 넣는다 (박스 안쪽만)
            const cv::Rect r(std::max(0, static_cast<int>(box.x)),
                             std::max(0, static_cast<int>(box.y)),
                             std::max(1, static_cast<int>(box.width)),
                             std::max(1, static_cast<int>(box.height)));
            if (r.x + r.width <= kW && r.y + r.height <= kH) {
                in.frame.depth(r).setTo(static_cast<float>(p_cam.z()));
            }

            Detection d;
            d.class_id   = o.class_id;
            d.class_name = o.class_name;
            d.box        = box;
            d.confidence = static_cast<float>(0.70 + 0.02 * static_cast<double>(oi));
            in.detections.items.push_back(std::move(d));
        }
        seq.push_back(std::move(in));
    }
    return seq;
}

inline void putTokens(test::Blob& blob, const TokenStore& store, const IntegrationReport& rep) {
    blob.put(static_cast<std::uint64_t>(rep.matched));
    blob.put(static_cast<std::uint64_t>(rep.created));
    blob.put(static_cast<std::uint64_t>(rep.missed));
    blob.put(static_cast<std::uint64_t>(rep.retired));
    blob.put(rep.dynamic_area_ratio);

    // allTokens() 가 돌려주는 *순서 그대로* 기록한다. ID 정렬이 사라지면
    // 여기서 순서가 갈라져야 한다.
    for (const auto& t : store.allTokens()) {
        blob.put(t->id);
        blob.put(t->class_id);
        blob.put(t->class_name);
        blob.putEigen(t->position);
        blob.putEigen(t->position_cov);
        blob.putEigen(t->extent);
        blob.putEigen(t->velocity);
        blob.putEigen(t->velocity_cov);
        blob.put(t->existence_belief);
        blob.put(t->identity_belief);
        blob.put(t->static_belief);
        blob.put(t->detection_conf);
        blob.put(t->visible_ratio);
        blob.put(t->occlusion);
        blob.put(static_cast<int>(t->lifecycle));
        blob.put(t->observation_count);
        blob.put(t->miss_count);
        blob.put(t->first_seen);
        blob.put(t->last_seen);
        blob.put(t->box.x); blob.put(t->box.y);
        blob.put(t->box.width); blob.put(t->box.height);
        blob.put(static_cast<std::uint64_t>(t->history.size()));
        blob.put(static_cast<std::uint64_t>(t->trajectory.size()));
        for (const Vec3& p : t->trajectory) blob.putEigen(p);
    }
}

// 두 스토어를 *동시에 살려 두고* 번갈아 먹인다.
// 순차로 돌리면 첫 스토어가 해제된 자리를 두 번째가 그대로 재사용해
// 주소 의존 버그가 우연히 같은 답을 내고 만다 - 판별하지 못하는 측정이 된다.
inline void tokenStoreInterleaved(test::Blob& a, test::Blob& b) {
    const auto seq = makeTokenSequence();

    TokenStore store_a, store_b;
    for (const auto& in : seq) {
        const auto ra = store_a.integrate(in.detections, in.frame, in.T_world_cam, in.env);
        const auto rb = store_b.integrate(in.detections, in.frame, in.T_world_cam, in.env);
        a.put(ra.ok());
        b.put(rb.ok());
        if (ra.ok()) putTokens(a, store_a, ra.value());
        if (rb.ok()) putTokens(b, store_b, rb.value());
    }
}

// ---------------------------------------------------------------------------
// 3) ConstellationIndex 등록/질의
// ---------------------------------------------------------------------------

inline std::vector<ConstellationNode> makePlaceNodes(unsigned seed, std::size_t count) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> pos(-6.0, 6.0);
    std::vector<ConstellationNode> nodes;
    nodes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ConstellationNode n;
        n.id       = TokenId(static_cast<std::uint64_t>(i + 1));
        n.class_id = static_cast<int>(i % 5);
        n.position = Vec3(pos(gen), pos(gen), pos(gen));
        // 동점을 일부러 만든다. sigma 가 모두 같으면 buildFrom 의 정렬이
        // 순서를 결정하지 못하므로, 그 자리가 결정적인지도 여기서 재게 된다.
        n.sigma    = 0.06;
        nodes.push_back(n);
    }
    return nodes;
}

inline void putMatch(test::Blob& blob, const ConstellationMatch& m) {
    blob.put(m.place_id);
    blob.put(m.keyframe);
    blob.put(m.transform);
    blob.put(m.rms_error);
    blob.put(m.score);
    blob.put(static_cast<std::uint64_t>(m.correspondences.size()));
    for (const auto& [q, p] : m.correspondences) { blob.put(q); blob.put(p); }
}

// 등록 -> 질의 전체를 두 인덱스에서 번갈아 수행한다.
inline void constellationInterleaved(test::Blob& a, test::Blob& b) {
    ConstellationIndex idx_a, idx_b;

    constexpr int kPlaces = 24;
    for (int p = 0; p < kPlaces; ++p) {
        auto nodes = makePlaceNodes(static_cast<unsigned>(9000 + p), 9);
        const SE3 anchor(SO3::exp(Vec3(0.0, 0.01 * p, 0.0)), Vec3(0.5 * p, 0.0, 0.0));
        const auto kf = KeyframeId(static_cast<std::uint64_t>(p + 1));
        const auto stamp = Timestamp::fromSeconds(1.0 + p);
        const Vec3 gravity(0.0, 1.0, 0.0);

        a.put(idx_a.insert(kf, stamp, anchor, nodes, gravity));
        b.put(idx_b.insert(kf, stamp, anchor, nodes, gravity));
    }

    // 완전히 같은 배치를 가진 장소를 두 개 더 등록한다. 투표 점수가 비트까지
    // 같아지므로 retrieve() 의 정렬이 동점을 어떻게 깨는지가 드러난다.
    // 이 경로를 안 밟으면 "동점이 없어서" 통과하는 측정이 된다.
    {
        const auto twin = makePlaceNodes(9000u, 9);
        for (int t = 0; t < 2; ++t) {
            const auto kf = KeyframeId(static_cast<std::uint64_t>(900 + t));
            const auto stamp = Timestamp::fromSeconds(500.0 + t);
            a.put(idx_a.insert(kf, stamp, SE3::identity(), twin, Vec3(0.0, 1.0, 0.0)));
            b.put(idx_b.insert(kf, stamp, SE3::identity(), twin, Vec3(0.0, 1.0, 0.0)));
        }
    }

    // 등록된 장소를 회전/평행이동시켜 질의로 쓴다. 정답이 존재하는 질의와
    // 존재하지 않는 질의를 모두 넣는다 - 기각 경로도 결정적이어야 한다.
    for (int p = 0; p < kPlaces; p += 3) {
        auto nodes = makePlaceNodes(static_cast<unsigned>(9000 + p), 9);
        const SE3 shift(SO3::exp(Vec3(0.0, 0.30, 0.0)), Vec3(1.2, -0.4, 0.7));
        for (auto& n : nodes) n.position = shift * n.position;

        const Vec3 gravity = shift.rotation() * Vec3(0.0, 1.0, 0.0);

        for (test::Blob* blob : {&a, &b}) {
            ConstellationIndex& idx = (blob == &a) ? idx_a : idx_b;
            const auto all = idx.queryAll(nodes, gravity);
            blob->put(static_cast<std::uint64_t>(all.size()));
            for (const auto& m : all) putMatch(*blob, m);

            const auto one = idx.query(nodes, gravity);
            blob->put(one.ok());
            if (one.ok()) putMatch(*blob, one.value());
            else          blob->put(static_cast<int>(one.error().code));
        }
    }

    // 존재하지 않는 장소
    auto unknown = makePlaceNodes(777u, 9);
    for (test::Blob* blob : {&a, &b}) {
        ConstellationIndex& idx = (blob == &a) ? idx_a : idx_b;
        const auto r = idx.query(unknown, Vec3(0.0, 1.0, 0.0));
        blob->put(r.ok());
        blob->put(static_cast<int>(r.ok() ? ErrorCode::None : r.error().code));
    }
}

// ---------------------------------------------------------------------------
// 4) nonMaxSuppression - 동점 신뢰도
// ---------------------------------------------------------------------------

// 겹치는 박스 3개가 한 그룹이고 그룹끼리는 겹치지 않는다.
// 그룹 *안에서는* confidence 가 비트까지 같고(동점), 그룹 *사이에서는* 서로
// 다르며 입력 순서는 정렬되어 있지 않다.
//
// 전부 같은 값으로 채우면 안 된다. MSVC 의 std::sort 는 모든 원소가 같은
// 입력에서 교환을 한 번도 하지 않아 stable_sort 와 결과가 같아진다 - 즉
// 그 입력으로는 타이브레이크 규약을 판별할 수 없다. 실제로 처음에 그렇게
// 만들었다가 대조군이 아무것도 잡지 못해 여기 이 형태로 바꿨다.
inline constexpr int kNmsGroups = 32;
inline constexpr int kNmsPerGroup = 3;

// 그룹별 confidence. 서로 다르고, 입력 순서와 무관하게 섞여 있다.
inline std::vector<float> nmsGroupConfidences() {
    std::vector<float> c;
    c.reserve(static_cast<std::size_t>(kNmsGroups));
    for (int g = 0; g < kNmsGroups; ++g) c.push_back(0.30f + 0.02f * static_cast<float>(g));
    std::mt19937 gen(1337u);
    std::shuffle(c.begin(), c.end(), gen);
    return c;
}

inline std::vector<Detection> tiedDetections() {
    const auto conf = nmsGroupConfidences();
    std::vector<Detection> dets;
    dets.reserve(static_cast<std::size_t>(kNmsGroups * kNmsPerGroup));
    for (int g = 0; g < kNmsGroups; ++g) {
        for (int k = 0; k < kNmsPerGroup; ++k) {
            Detection d;
            d.class_id   = 0;
            d.class_name = "person";
            d.confidence = conf[static_cast<std::size_t>(g)];
            d.box = cv::Rect2f(static_cast<float>(g * 200 + k), static_cast<float>(k),
                               100.f, 100.f);
            dets.push_back(std::move(d));
        }
    }
    return dets;
}

inline test::Blob nmsBlob(const std::vector<Detection>& in) {
    const auto kept = nonMaxSuppression(in, 0.5f, 300);
    test::Blob blob;
    blob.put(static_cast<std::uint64_t>(kept.size()));
    for (const auto& d : kept) {
        blob.put(d.class_id);
        blob.put(d.confidence);
        blob.put(d.box.x); blob.put(d.box.y);
        blob.put(d.box.width); blob.put(d.box.height);
    }
    return blob;
}

// 동점 규약이 "입력 인덱스 순"이라면 답은 해석적으로 정해진다:
// 그룹은 confidence 내림차순으로 나오고, 각 그룹에서 살아남는 것은
// 입력 인덱스가 가장 작은 박스다.
inline test::Blob nmsAnalyticExpectation() {
    const auto all  = tiedDetections();
    const auto conf = nmsGroupConfidences();

    std::vector<int> order(static_cast<std::size_t>(kNmsGroups));
    for (int g = 0; g < kNmsGroups; ++g) order[static_cast<std::size_t>(g)] = g;
    std::stable_sort(order.begin(), order.end(), [&conf](int x, int y) {
        return conf[static_cast<std::size_t>(x)] > conf[static_cast<std::size_t>(y)];
    });

    std::vector<Detection> expected;
    expected.reserve(order.size());
    for (int g : order) expected.push_back(all[static_cast<std::size_t>(g * kNmsPerGroup)]);
    test::Blob blob;
    blob.put(static_cast<std::uint64_t>(expected.size()));
    for (const auto& d : expected) {
        blob.put(d.class_id);
        blob.put(d.confidence);
        blob.put(d.box.x); blob.put(d.box.y);
        blob.put(d.box.width); blob.put(d.box.height);
    }
    return blob;
}

}  // namespace wme::testcases
