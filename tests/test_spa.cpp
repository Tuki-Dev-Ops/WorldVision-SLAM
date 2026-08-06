// Tier 2 (SPA) 검증 - 평면 추출 + 구조 정합.
//
// 이 파일의 핵심 주장은 정확도가 아니라 **랭크 부족을 숨기지 않는 것**이다.
// 복도에서 복도 축 병진은 평면으로 관측되지 않는데, 그걸 6-DoF 로 보고하면
// 융합 규칙이 없는 정보를 믿게 된다.
//
// docs/06-results.md 10.4 를 따라 모든 진단은 "일부러 나쁜 입력을 넣고 그것이
// 걸리는지" 로 검증한다. 특히:
//   - 부호 규약: Hesse 법선이 카메라 쪽을 향하면 추출 결과가 0 개가 된다
//   - 병진 부호: 항등 정합은 양변이 0 이라 부호 오류를 통과시킨다.
//     반드시 **항등이 아닌** 병진으로 확인한다
//   - 정보행렬: 랭크가 맞아도 정보행렬이 틀릴 수 있다. 실제 잔차 증가와
//     대조하지 않으면 관측 가능/불가능 축이 통째로 뒤바뀌어도 모른다
//   - 랭크 부족 방향: 절단하지 않으면 잡음이 수십 m 로 증폭된다

#include "wme/geometry/PlaneExtractor.hpp"
#include "wme/geometry/StructuralAligner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace wme;

namespace {

// --- 합성 장면 -------------------------------------------------------------
// Python 참조 하네스(wme/sim/render.py)의 Quad 와 같은 파라미터화이고 카메라도
// python/tests/test_geometry.py 와 동일하다. 두 구현의 숫자를 직접 비교할 수 있다.

struct Quad {
    Vec3   center;
    Vec3   normal;
    Vec3   u_axis;
    Vec3   v_axis;
    double half_u;
    double half_v;
};

constexpr int kW = 320, kH = 240;

CameraIntrinsics makeK() {
    CameraIntrinsics K;
    K.fx = 220.0; K.fy = 220.0;
    K.cx = 159.5; K.cy = 119.5;
    K.width = kW; K.height = kH;
    return K;
}

// 광선 투사. 카메라 z=1 규약이라 교차 파라미터 t 가 곧 z-깊이다. 미적중은 0.
cv::Mat renderDepth(const std::vector<Quad>& quads, const SE3& pose, const CameraIntrinsics& K) {
    cv::Mat depth(K.height, K.width, CV_32F, cv::Scalar(0.f));
    const Mat3 R = pose.rotation().matrix();
    const Vec3 o = pose.translation();

    for (int y = 0; y < K.height; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < K.width; ++x) {
            const Vec3 dir = R * Vec3((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.0);
            double best = std::numeric_limits<double>::infinity();
            for (const Quad& q : quads) {
                const double denom = dir.dot(q.normal);
                if (std::abs(denom) < 1e-9) continue;
                const double t = (q.center - o).dot(q.normal) / denom;
                if (t <= 1e-4 || t >= best) continue;
                const Vec3 local = o + t * dir - q.center;
                if (std::abs(local.dot(q.u_axis)) <= q.half_u &&
                    std::abs(local.dot(q.v_axis)) <= q.half_v) {
                    best = t;
                }
            }
            row[x] = std::isfinite(best) ? static_cast<float>(best) : 0.f;
        }
    }
    return depth;
}

const Vec3 kEx(1, 0, 0), kEy(0, 1, 0), kEz(0, 0, 1);

// 바닥/천장/벽 네 면
std::vector<Quad> room(double size = 4.0, double height = 2.6) {
    const double s = size, h = height;
    return {
        {Vec3(0, 0, 0),      kEz,  kEx, kEy, s, s},
        {Vec3(0, 0, h),     -kEz,  kEx, kEy, s, s},
        {Vec3(s, 0, h / 2), -kEx,  kEy, kEz, s, h / 2},
        {Vec3(-s, 0, h / 2), kEx,  kEy, kEz, s, h / 2},
        {Vec3(0, s, h / 2), -kEy,  kEx, kEz, s, h / 2},
        {Vec3(0, -s, h / 2), kEy,  kEx, kEz, s, h / 2},
    };
}

// 평행한 벽 둘 + 바닥. 복도 축(월드 y) 병진이 관측되지 않는 구조.
std::vector<Quad> corridor() {
    return {
        {Vec3(0, 0, 0),        kEz,  kEx, kEy, 2.0, 12.0},
        {Vec3(1.6, 0, 1.3),   -kEx,  kEy, kEz, 12.0, 1.3},
        {Vec3(-1.6, 0, 1.3),   kEx,  kEy, kEz, 12.0, 1.3},
    };
}

// 정면 벽 하나. 평면 하나짜리 장면 - 3 자유도만 구속된다.
std::vector<Quad> singleWall(double y = 3.0) {
    return {{Vec3(0, y, 0), -kEy, kEx, kEz, 6.0, 6.0}};
}

// eye 에서 target 을 보는 카메라 포즈 (카메라 -> 월드).
SE3 looking(const Vec3& eye, const Vec3& target) {
    Vec3 z = (target - eye);
    z /= std::max(z.norm(), 1e-9);
    Vec3 x = kEz.cross(z);
    const double n = x.norm();
    x = (n < 1e-6) ? kEx : Vec3(x / n);
    Mat3 R;
    R.col(0) = x;
    R.col(1) = z.cross(x);
    R.col(2) = z;
    return SE3(R, eye);
}

// 월드 +y 를 정면으로 보는 카메라. 정면 벽 장면에서 쓴다.
// 직교행렬을 손으로 적으면 det = -1 (반사)을 만들기 쉽고, Quat(R) 은 그것을
// 조용히 받아 쓰레기 회전을 낸다. 항상 looking() 으로 만든다.
SE3 frontal(const Vec3& eye) { return looking(eye, eye + kEy); }

std::vector<Plane> planesAt(const std::vector<Quad>& scene, const SE3& pose,
                            PlaneExtractorConfig cfg = {}) {
    const CameraIntrinsics K = makeK();
    PlaneExtractor ex(cfg);
    auto r = ex.extract(renderDepth(scene, pose, K), K);
    return r.ok() ? r.value() : std::vector<Plane>{};
}

// 수동 평면. 대응이 아니라 수치 거동을 재는 테스트에서 쓴다.
Plane makePlane(const Vec3& n, double d, std::size_t inliers = 400) {
    Plane p;
    p.normal   = n.normalized();
    p.distance = d;
    p.inliers  = inliers;
    p.rms      = 0.0;
    p.centroid = p.normal * d;
    return p;
}

}  // namespace

// =========================================================================
// 평면 추출
// =========================================================================

TEST(PlaneExtractor, NormalsFollowHesseConventionAndAreUnit) {
    // 이 규약을 뒤집으면(표면 법선 = 카메라 쪽) d < 0 이 되어 이후 게이트가
    // 전부 걸러내고 평면이 0 개가 된다. Python 참조 구현에서 실제로 났던 실패다.
    const CameraIntrinsics K = makeK();
    PlaneExtractor ex;
    const auto field = ex.estimateNormals(renderDepth(room(), looking(Vec3(0, -1, 1.3),
                                                                     Vec3(0, 3, 1.0)), K), K);
    ASSERT_TRUE(field.ok()) << field.error().message();
    ASSERT_GT(field.value().validCount(), 1000u);

    std::size_t checked = 0;
    for (std::size_t k = 0; k < field.value().size(); ++k) {
        if (field.value().valid[k] == 0) continue;
        const Vec3& n = field.value().normal[k];
        const Vec3& p = field.value().point[k];
        EXPECT_NEAR(n.norm(), 1.0, 1e-12);
        EXPECT_GT(n.dot(p), 0.0) << "법선이 카메라 쪽을 향한다 - Hesse 규약 위반";
        ++checked;
    }
    EXPECT_GT(checked, 1000u);
}

TEST(PlaneExtractor, NormalsMatchAnalyticGroundTruth) {
    // 기울어진 평면 하나. 법선은 해석적으로 알려져 있다.
    const CameraIntrinsics K = makeK();
    const Vec3 world_n = Vec3(0.3, -1.0, 0.2).normalized();
    const std::vector<Quad> scene = {{Vec3(0, 3, 1.0), world_n, kEx, kEz, 8.0, 8.0}};

    const SE3 pose = frontal(Vec3(0, 0, 1.0));
    // 카메라 좌표 법선. Hesse 규약이라 카메라 반대쪽 = z 성분 양수.
    Vec3 expected = pose.rotation().matrix().transpose() * world_n;
    if (expected.z() < 0.0) expected = -expected;

    PlaneExtractor ex;
    const auto field = ex.estimateNormals(renderDepth(scene, pose, K), K);
    ASSERT_TRUE(field.ok());

    double worst = 0.0;
    std::size_t n_valid = 0;
    for (std::size_t k = 0; k < field.value().size(); ++k) {
        if (field.value().valid[k] == 0) continue;
        worst = std::max(worst, (field.value().normal[k] - expected).norm());
        ++n_valid;
    }
    ASSERT_GT(n_valid, 1000u);
    // 이웃 외적은 이산 미분이라 평면에서는 편향이 없다. 남는 것은 깊이맵이
    // float32 라서 생기는 양자화뿐이다: 3 m 깊이의 1 ulp(3.6e-7 m)를 이웃 간격
    // 0.055 m 로 나누면 1e-5 수준이고, 실측 7.8e-6 이 정확히 그 크기다.
    // 법선 자체가 틀리면 오차는 0.1 단위로 나오므로 이 문턱은 여전히 판별력이 있다.
    EXPECT_LT(worst, 1e-4) << "법선 오차 " << worst;
    std::cout << "[측정] 법선 최대 오차 = " << worst << " (유효 " << n_valid << " 점)\n";
}

TEST(PlaneExtractor, WallDistanceMatchesGeometry) {
    const auto planes = planesAt(singleWall(3.0), frontal(Vec3::Zero()));
    ASSERT_EQ(planes.size(), 1u);
    EXPECT_NEAR(planes[0].distance, 3.0, 0.02);
    EXPECT_GT(planes[0].normal.z(), 0.99);
    EXPECT_LT(planes[0].rms, 1e-6);
}

TEST(PlaneExtractor, RoomYieldsThreeDominantPlanes) {
    const auto planes = planesAt(room(), looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6)));
    ASSERT_GE(planes.size(), 3u);
    for (const Plane& p : planes) {
        EXPECT_NEAR(p.normal.norm(), 1.0, 1e-12);
        EXPECT_GT(p.distance, 0.0);
        // 재적합 반경(refine_threshold = 0.04 m) 안에서 인접 평면의 점을 일부
        // 끌어오므로 완벽한 0 은 아니다. 그래도 한 자릿수 mm 여야 한다.
        EXPECT_LT(p.rms, 0.01);
    }
    // 바닥은 카메라 높이 1.3 m 에 있다. 그 평면이 잡혀야 한다.
    bool found_floor = false;
    for (const Plane& p : planes) {
        if (std::abs(p.normal.y()) > 0.95) {
            found_floor = true;
            EXPECT_NEAR(p.distance, 1.3, 0.03);
        }
    }
    EXPECT_TRUE(found_floor);
}

TEST(PlaneExtractor, DepthDiscontinuityDoesNotBecomeAPlane) {
    // 물체 실루엣에서 비스듬한 가짜 평면이 생기면 안 된다.
    const CameraIntrinsics K = makeK();
    cv::Mat depth(kH, kW, CV_32F);
    for (int y = 0; y < kH; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < kW; ++x) row[x] = (x < kW / 2) ? 3.f : 8.f;
    }

    PlaneExtractorConfig cfg;
    cfg.min_inliers = 50;
    PlaneExtractor ex(cfg);
    const auto r = ex.extract(depth, K);
    ASSERT_TRUE(r.ok()) << r.error().message();
    ASSERT_FALSE(r.value().empty());
    for (const Plane& p : r.value()) {
        EXPECT_GT(std::abs(p.normal.z()), 0.9) << "경계에서 생긴 기운 평면";
    }
}

TEST(PlaneExtractor, RandomDepthYieldsNoPlane) {
    // 판별력 확인용 음성 대조. 구조가 없는 입력에서 평면이 나오면
    // 추출기가 아무것도 재고 있지 않다는 뜻이다.
    const CameraIntrinsics K = makeK();
    cv::Mat depth(kH, kW, CV_32F);
    std::mt19937 gen(20260804);
    std::uniform_real_distribution<float> d(1.0f, 4.0f);
    for (int y = 0; y < kH; ++y) {
        auto* row = depth.ptr<float>(y);
        for (int x = 0; x < kW; ++x) row[x] = d(gen);
    }
    PlaneExtractor ex;
    const auto r = ex.extract(depth, K);
    if (r.ok()) EXPECT_TRUE(r.value().empty()) << "잡음에서 평면 " << r.value().size() << " 개";
}

TEST(PlaneExtractor, EmptyDepthIsReportedNotSilentlyEmpty) {
    const CameraIntrinsics K = makeK();
    PlaneExtractor ex;
    const auto r = ex.extract(cv::Mat(kH, kW, CV_32F, cv::Scalar(0.f)), K);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, ErrorCode::InsufficientData);
}

TEST(PlaneExtractor, TransformIsConsistent) {
    const auto planes = planesAt(room(), looking(Vec3(0, -1, 1.3), Vec3(0, 3, 1.0)));
    ASSERT_FALSE(planes.empty());

    const Mat3 R = SO3::exp(Vec3(0.1, -0.2, 0.05)).matrix();
    const Vec3 t(0.3, -0.1, 0.2);
    const Plane moved = planes[0].transformed(R, t);

    // 평면 위의 점은 옮겨도 평면 위에 있어야 한다
    const Vec3 on = planes[0].normal * planes[0].distance;
    EXPECT_NEAR(moved.signedDistance(R * on + t), 0.0, 1e-9);
}

TEST(PlaneExtractor, GravityFromFloorAndAbsentWithoutIt) {
    const auto planes = planesAt(room(), looking(Vec3(0, -1, 1.6), Vec3(0, 2.0, 0.2)));
    const auto g = dominantGravity(planes);
    ASSERT_TRUE(g.ok()) << g.error().message();
    EXPECT_NEAR(g.value().norm(), 1.0, 1e-12);

    // 반환값은 지배 수평면의 법선과 평행해야 한다
    bool parallel = false;
    for (const Plane& p : planes) {
        if (std::abs(p.normal.dot(g.value())) > 0.999) parallel = true;
    }
    EXPECT_TRUE(parallel);
    // 부호는 "카메라 y 가 아래" 규약에서 나오지 기하에서 유도되지 않는다.
    // 평면 하나만으로는 바닥과 천장을 구별할 수 없으므로 이 이상은 주장하지 않는다.
    EXPECT_GT(g.value().y(), 0.8);

    // 수평면이 없으면 지어내지 않는다
    const auto none = dominantGravity(planesAt(singleWall(3.0), frontal(Vec3::Zero())));
    EXPECT_FALSE(none.ok());
    EXPECT_EQ(none.error().code, ErrorCode::NotAvailable);
}

// =========================================================================
// SPA - 포즈 복원
// =========================================================================

TEST(StructuralAligner, RecoversRotationInARoom) {
    const auto scene = room();
    const SE3 a = looking(Vec3(0, -1, 1.3), Vec3(0, 3, 1.0));
    const SE3 b(SO3::exp(Vec3(0.0, 0.06, 0.0)) * a.rotation(), a.translation());

    StructuralAligner spa;
    const auto r = spa.align(planesAt(scene, a), planesAt(scene, b));
    ASSERT_TRUE(r.ok()) << r.error().message();

    const SE3 truth = b.inverse() * a;
    const Vec2 err = r.value().T_cur_ref.distanceTo(truth);
    EXPECT_LT(err.y(), 0.01) << "회전 오차 " << err.y() << " rad";
    EXPECT_EQ(r.value().rotation_rank, 3);
    std::cout << "[측정] 방(3평면) 회전: 참 0.060 rad, 오차 " << err.y()
              << " rad / 병진 오차 " << err.x() << " m\n";
}

TEST(StructuralAligner, RecoversTranslationWithNonIdentityMotion) {
    // 항등 정합은 n·t = d_c - d_r 의 양변이 0 이라 부호 오류를 통과시킨다.
    // 반드시 실제로 움직인 케이스로 확인한다.
    const auto scene = room();
    const SE3 a = looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6));
    const SE3 b(a.rotation(), a.translation() + Vec3(0.15, 0.10, 0.0));

    StructuralAligner spa;
    const auto r = spa.align(planesAt(scene, a), planesAt(scene, b));
    ASSERT_TRUE(r.ok()) << r.error().message();
    ASSERT_EQ(r.value().translation_rank, 3) << "이 시점은 병진이 완전히 구속되어야 한다";

    const SE3 truth = b.inverse() * a;
    const Vec2 err = r.value().T_cur_ref.distanceTo(truth);
    EXPECT_LT(err.x(), 0.03) << "병진 오차 " << err.x() << " m";
    EXPECT_LT(err.y(), 0.01);
    EXPECT_LT(r.value().offset_rms, 0.02);
    std::cout << "[측정] 방(3평면) 병진: 참 " << truth.translation().norm()
              << " m, 오차 " << err.x() << " m / 회전 오차 " << err.y()
              << " rad / offset_rms " << r.value().offset_rms << " m\n";
}

TEST(StructuralAligner, TranslationSignIsNotInverted) {
    const auto scene = room();
    const SE3 a = looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6));
    const SE3 b(a.rotation(), a.translation() + Vec3(0.25, 0.0, 0.0));

    StructuralAligner spa;
    const auto r = spa.align(planesAt(scene, a), planesAt(scene, b));
    ASSERT_TRUE(r.ok()) << r.error().message();

    const SE3 truth = b.inverse() * a;
    const Vec3 t_true = truth.translation();
    const Vec3 t_est  = r.value().T_cur_ref.translation();
    const double motion = t_true.norm();
    ASSERT_GT(motion, 0.2);

    const double err  = (t_est - t_true).norm();
    const double flip = (-t_est - t_true).norm();
    EXPECT_LT(err, 0.1 * motion) << "오차 " << err << " vs 이동 " << motion;
    // 이 테스트가 판별력을 갖는지 확인: 부호가 뒤집혔다면 오차가 이동의 2배다
    EXPECT_GT(flip, 1.5 * motion) << "부호 판별력 없음";
}

TEST(StructuralAligner, IdentityAlignmentIsIdentity) {
    const auto planes = planesAt(room(), looking(Vec3(-0.5, -1.0, 1.3), Vec3(1.5, 2.5, 0.8)));
    StructuralAligner spa;
    const auto r = spa.align(planes, planes);
    ASSERT_TRUE(r.ok()) << r.error().message();
    const Vec2 d = r.value().T_cur_ref.distanceTo(SE3::identity());
    EXPECT_LT(d.x(), 1e-9);
    EXPECT_LT(d.y(), 1e-9);
}

// =========================================================================
// SPA - 랭크 보고
// =========================================================================

TEST(StructuralAligner, SinglePlaneReportsThreeDofNotSix) {
    // 평면 하나는 법선 방향 병진 1 자유도와 법선에 수직인 회전 2 자유도만
    // 구속한다. 나머지 3 축은 관측되지 않으며, 그 축으로 움직이면 안 된다.
    const auto ref = planesAt(singleWall(3.0), frontal(Vec3::Zero()));
    const auto cur = planesAt(singleWall(3.0), frontal(Vec3(0.0, 0.2, 0.0)));
    ASSERT_EQ(ref.size(), 1u);
    ASSERT_EQ(cur.size(), 1u);

    // 기본 설정은 대응 2 개를 요구한다 - 성공을 주장하지 않는다
    StructuralAligner strict;
    EXPECT_FALSE(strict.align(ref, cur).ok());

    StructuralAlignerConfig cfg;
    cfg.min_matches = 1;
    StructuralAligner spa(cfg);
    const auto r = spa.align(ref, cur);
    ASSERT_TRUE(r.ok()) << r.error().message();
    EXPECT_TRUE(r.degraded());

    EXPECT_EQ(r.value().rotation_rank, 2) << "법선 둘레 회전만 관측 불가여야 한다";
    EXPECT_EQ(r.value().translation_rank, 1);
    EXPECT_EQ(r.value().observable_dof, 3);
    EXPECT_FALSE(r.value().fullRank());
    std::cout << "[측정] 단일 평면: rot rank " << r.value().rotation_rank << "/3, trans rank "
              << r.value().translation_rank << "/3, observable "
              << r.value().observable_dof << "/6, reliability " << r.reliability() << "\n";

    // 카메라가 +z 로 0.2 m 전진했으므로 t = (0, 0, -0.2)
    const Vec3 t = r.value().T_cur_ref.translation();
    EXPECT_NEAR(t.z(), -0.2, 0.02);
    // 구속되지 않은 두 축으로는 정확히 0 이어야 한다 (최소노름 해)
    EXPECT_LT(std::abs(t.x()), 1e-9);
    EXPECT_LT(std::abs(t.y()), 1e-9);

    // 관측 불가 축이 3 개, 그중 둘은 병진, 하나는 회전이어야 한다
    const auto weak = unobservableDirections(r.value());
    ASSERT_EQ(weak.size(), 3u);
    int trans_weak = 0, rot_weak = 0;
    for (const Vec6& v : weak) {
        if (v.head<3>().norm() > v.tail<3>().norm()) ++trans_weak; else ++rot_weak;
    }
    EXPECT_EQ(trans_weak, 2);
    EXPECT_EQ(rot_weak, 1);
}

TEST(StructuralAligner, CorridorAlongAxisTranslationIsUnobservable) {
    // 이 파일의 핵심 주장. 평행한 벽 둘 + 바닥이면 복도 축 병진이 관측되지 않는다.
    const SE3 pose = looking(Vec3(0, -4.0, 1.2), Vec3(0, 4.0, 1.0));
    const auto planes = planesAt(corridor(), pose);
    ASSERT_GE(planes.size(), 3u) << "복도에서 평면을 못 찾았다";

    StructuralAligner spa;
    const auto r = spa.align(planes, planes);
    ASSERT_TRUE(r.ok()) << r.error().message();

    EXPECT_LT(r.value().translation_rank, 3)
        << "복도 축 병진이 관측 가능하다고 보고했다: rank " << r.value().translation_rank;
    EXPECT_EQ(r.value().translation_rank, 2);
    // 서로 다른 방향의 법선이 둘 이상이면 회전은 완전히 구속된다
    EXPECT_EQ(r.value().rotation_rank, 3);
    EXPECT_LT(r.value().observable_dof, 6);
    EXPECT_TRUE(r.degraded());

    // 관측 불가 축은 복도 축(카메라 z 에 가깝다) 방향의 병진이어야 한다
    const auto weak = unobservableDirections(r.value());
    ASSERT_EQ(weak.size(), 1u);
    const Vec3 rho = weak[0].head<3>();
    EXPECT_GT(rho.norm(), 0.9) << "약한 축이 병진이 아니다";
    std::cout << "[측정] 복도: rot rank " << r.value().rotation_rank << "/3, trans rank "
              << r.value().translation_rank << "/3, observable "
              << r.value().observable_dof << "/6, eig="
              << r.value().eigenvalues.transpose() << "\n";
    EXPECT_GT(std::abs(rho.normalized().z()), 0.9) << "약한 축이 복도 축이 아니다: "
                                                   << rho.transpose();
}

TEST(StructuralAligner, RoomCornerViewpointIsFullRank) {
    const auto planes = planesAt(room(), looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6)));
    ASSERT_GE(planes.size(), 3u);

    StructuralAligner spa;
    const auto r = spa.align(planes, planes);
    ASSERT_TRUE(r.ok()) << r.error().message();
    EXPECT_EQ(r.value().rotation_rank, 3);
    EXPECT_EQ(r.value().translation_rank, 3);
    EXPECT_EQ(r.value().observable_dof, 6);
    EXPECT_FALSE(r.degraded());
    std::cout << "[측정] 방 모서리 시점: observable " << r.value().observable_dof
              << "/6, eig=" << r.value().eigenvalues.transpose() << "\n";
    EXPECT_TRUE(unobservableDirections(r.value()).empty());
}

// =========================================================================
// SPA - 정보행렬
// =========================================================================

TEST(StructuralAligner, InformationPredictsActualResidualIncrease) {
    // 랭크만 보면 정보행렬이 통째로 틀려도 모른다. 실제 잔차 증가와 대조한다.
    //
    // 회전 잔차 r = n_cur - exp(phi) R n_ref 의 정보는 sum w (I - n n^T) 이지
    // sum w n n^T (그 직교여공간) 이 **아니다**. 둘을 바꾸면 관측 가능한 축과
    // 불가능한 축이 정확히 뒤바뀌므로, 이 테스트가 그 오류를 잡는다.
    const auto planes = planesAt(room(), looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6)));
    ASSERT_GE(planes.size(), 3u);

    StructuralAlignerConfig cfg;
    StructuralAligner spa(cfg);
    const auto r = spa.align(planes, planes);   // 항등 => t = 0, 잔차 0
    ASSERT_TRUE(r.ok()) << r.error().message();
    const Mat6& Lam = r.value().information;

    const double inv_sr2 = 1.0 / (cfg.rotation_sigma * cfg.rotation_sigma);
    const double inv_st2 = 1.0 / (cfg.translation_sigma * cfg.translation_sigma);

    std::mt19937 gen(4242);
    std::normal_distribution<double> nd(0.0, 1.0);
    constexpr double kStep = 1e-5;

    for (int trial = 0; trial < 6; ++trial) {
        Vec3 dir(nd(gen), nd(gen), nd(gen));
        dir.normalize();

        // --- 회전 섭동 ---
        const Vec3 phi = kStep * dir;
        const Mat3 Rp = SO3::exp(phi).matrix();
        double chi2 = 0.0;
        for (const PlaneMatch& m : r.value().matches) {
            const Vec3& n = planes[m.cur_index].normal;
            chi2 += m.weight * (n - Rp * n).squaredNorm() * inv_sr2;
        }
        Vec6 xi = Vec6::Zero();
        xi.tail<3>() = phi;
        const double pred = xi.transpose() * Lam * xi;
        EXPECT_NEAR(chi2, pred, 0.02 * std::max(chi2, pred))
            << "회전 정보가 실제 잔차를 예측하지 못한다 (trial " << trial << ")";
        EXPECT_GT(pred, 0.0);

        // --- 병진 섭동 ---
        const Vec3 rho = kStep * dir;
        double chi2_t = 0.0;
        for (const PlaneMatch& m : r.value().matches) {
            const Vec3& n = planes[m.cur_index].normal;
            const double e = n.dot(rho);
            chi2_t += m.weight * e * e * inv_st2;
        }
        Vec6 xt = Vec6::Zero();
        xt.head<3>() = rho;
        const double pred_t = xt.transpose() * Lam * xt;
        EXPECT_NEAR(chi2_t, pred_t, 1e-9 * std::max(1.0, chi2_t));
    }
}

TEST(StructuralAligner, RotationInformationIsNotTheNormalScatter) {
    // 위 테스트가 실제로 판별력이 있는지 못박는다. 평면 하나짜리 장면에서
    // sum w n n^T 는 법선 **둘레** 회전에 정보를 주장하는데, 그 축은 관측되지
    // 않는 축이다. 두 형태가 서로 다른 답을 낸다는 사실 자체를 기록한다.
    const std::vector<Plane> one{makePlane(Vec3(0, 0, 1), 3.0)};
    StructuralAlignerConfig cfg;
    cfg.min_matches = 1;
    StructuralAligner spa(cfg);
    const auto r = spa.align(one, one);
    ASSERT_TRUE(r.ok()) << r.error().message();

    const Mat3 rot_block = r.value().information.block<3, 3>(3, 3);
    // 법선 둘레(z) 회전은 관측되지 않는다
    EXPECT_NEAR(rot_block(2, 2), 0.0, 1e-9);
    // 법선에 수직인 두 축은 구속된다
    EXPECT_GT(rot_block(0, 0), 0.0);
    EXPECT_GT(rot_block(1, 1), 0.0);
    // 반대로 병진은 법선 방향만 구속된다
    const Mat3 trans_block = r.value().information.block<3, 3>(0, 0);
    EXPECT_GT(trans_block(2, 2), 0.0);
    EXPECT_NEAR(trans_block(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(trans_block(1, 1), 0.0, 1e-9);
}

TEST(StructuralAligner, AlphaScalesInformationButNotThePose) {
    const auto planes = planesAt(room(), looking(Vec3(0, -1, 1.3), Vec3(0, 3, 1.0)));
    StructuralAligner spa;
    const auto full = spa.align(planes, planes, SE3::identity(), 1.0);
    const auto weak = spa.align(planes, planes, SE3::identity(), 0.1);
    ASSERT_TRUE(full.ok());
    ASSERT_TRUE(weak.ok());

    EXPECT_TRUE(weak.value().information.isApprox(full.value().information * 0.1, 1e-9));
    // 포즈와 랭크 판정은 가중치와 무관해야 한다 (정규화가 alpha 를 상쇄한다)
    EXPECT_TRUE(weak.value().T_cur_ref.matrix().isApprox(full.value().T_cur_ref.matrix(), 1e-12));
    EXPECT_EQ(weak.value().observable_dof, full.value().observable_dof);
}

// =========================================================================
// SPA - 수치 안전성과 실패 보고
// =========================================================================

TEST(StructuralAligner, NearNullDirectionIsTruncatedNotAmplified) {
    // 거의 영에 가까운 특이값을 그대로 살리면 잡음이 수십 m 로 증폭된다
    // (Python 참조 구현 실측: 40 m 초과). 절단해야 한다.
    constexpr double kEpsAxis = 1e-4;    // z 축을 아주 약하게만 구속하는 성분
    std::vector<Plane> ref{
        makePlane(Vec3(1, 0, 0), 2.0),
        makePlane(Vec3(0, 1, kEpsAxis), 1.5),
        makePlane(Vec3(0, -1, kEpsAxis), 1.5),
    };
    std::vector<Plane> cur = ref;
    // 거리에만 작은 잡음. 두 평행벽에 **같은 부호**로 넣어야 약한 z 축이 여기된다.
    // 반대 부호면 두 식이 상쇄해 z 성분이 0 으로 나오고 대조군이 발산하지 않는다.
    cur[0].distance += 0.01;
    cur[1].distance -= 0.01;
    cur[2].distance -= 0.01;

    StructuralAligner spa;
    const auto r = spa.align(ref, cur);
    ASSERT_TRUE(r.ok()) << r.error().message();
    EXPECT_EQ(r.value().translation_rank, 2);
    EXPECT_LT(r.value().T_cur_ref.translation().norm(), 0.1)
        << "병진 " << r.value().T_cur_ref.translation().transpose();

    // 판별력 확인: 절단하지 않은 정규방정식 해는 실제로 발산한다
    Mat3 ATA = Mat3::Zero();
    Vec3 ATb = Vec3::Zero();
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const Vec3& n = cur[i].normal;
        ATA.noalias() += n * n.transpose();
        ATb.noalias() += n * (cur[i].distance - ref[i].distance);
    }
    const Vec3 naive = ATA.fullPivLu().solve(ATb);
    std::cout << "[측정] 절단 해 |t| = " << r.value().T_cur_ref.translation().norm()
              << " m vs 비절단 |t| = " << naive.norm() << " m\n";
    EXPECT_GT(naive.norm(), 10.0) << "대조군이 발산하지 않아 이 테스트는 판별력이 없다";
}

TEST(StructuralAligner, WrongCorrespondenceShowsUpInResidual) {
    // 일부러 모순된 입력. 랭크는 여전히 6 이지만 어떤 단일 병진도 세 식을
    // 만족시키지 못하므로 잔차가 커져야 한다. 랭크만 보고하면 조용한 오답이 된다.
    // 네 개여야 한다. 직교 평면 셋은 정방계라 어떤 우변도 정확히 풀려서
    // 잔차가 항상 0 이 되고, 그러면 이 테스트가 아무것도 재지 않는다.
    std::vector<Plane> ref{
        makePlane(Vec3(1, 0, 0), 2.0),
        makePlane(Vec3(0, 1, 0), 1.5),
        makePlane(Vec3(0, 0, 1), 3.0),
        makePlane(Vec3(1, 1, 1), 4.0),
    };
    std::vector<Plane> consistent = ref;
    for (Plane& p : consistent) p.distance += p.normal.dot(Vec3(0.1, -0.05, 0.2));

    std::vector<Plane> broken = consistent;
    broken[3].distance += 0.3;           // 네 식이 서로 모순되게 만든다 (대응 게이트 안)

    StructuralAligner spa;
    const auto good = spa.align(ref, consistent);
    const auto bad  = spa.align(ref, broken);
    ASSERT_TRUE(good.ok());
    ASSERT_TRUE(bad.ok());

    EXPECT_EQ(good.value().translation_rank, 3);
    EXPECT_EQ(bad.value().translation_rank, 3);   // 랭크는 구별하지 못한다
    EXPECT_LT(good.value().offset_rms, 1e-12);
    EXPECT_GT(bad.value().offset_rms, 0.05) << "모순된 대응이 잔차에 나타나지 않는다";
    std::cout << "[측정] 일관 offset_rms " << good.value().offset_rms
              << " m vs 모순 " << bad.value().offset_rms << " m\n";
}

TEST(StructuralAligner, TooFewMatchesDoesNotClaimSuccess) {
    const auto planes = planesAt(room(), looking(Vec3(0, -1, 1.3), Vec3(0, 3, 1.0)));
    ASSERT_GE(planes.size(), 2u);
    StructuralAligner spa;

    const std::vector<Plane> one(planes.begin(), planes.begin() + 1);
    const auto r = spa.align(one, one);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, ErrorCode::InsufficientData);

    // 대응이 아예 없는 경우도 같다
    const std::vector<Plane> other{makePlane(Vec3(0.6, 0.8, 0.0), 9.0)};
    const auto none = spa.align(one, other);
    EXPECT_FALSE(none.ok());
}

TEST(StructuralAligner, MatchingRespectsAngleAndDistanceGates) {
    const auto planes = planesAt(room(), looking(Vec3(0, -1, 1.3), Vec3(0, 3, 1.0)));
    ASSERT_GE(planes.size(), 2u);

    StructuralAlignerConfig tight_cfg;
    tight_cfg.max_normal_angle = 1e-6;
    const std::size_t tight = StructuralAligner(tight_cfg).match(planes, planes).size();
    const std::size_t loose = StructuralAligner({}).match(planes, planes).size();
    EXPECT_LE(tight, loose);
    EXPECT_EQ(loose, planes.size());

    // 거리 게이트도 실제로 작동하는지 확인 - 게이트가 죽어 있으면 둘이 같다
    std::vector<Plane> shifted = planes;
    for (Plane& p : shifted) p.distance += 5.0;
    EXPECT_EQ(StructuralAligner({}).match(planes, shifted).size(), 0u);
}

TEST(StructuralAligner, InitialGuessIsUsedForAssociationOnly) {
    // 큰 운동에서는 init 없이는 대응을 못 찾는다. SPA 가 정제기인 이유다.
    const auto scene = room();
    const SE3 a = looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6));
    const SE3 b(SO3::exp(Vec3(0.0, 0.0, 0.5)) * a.rotation(), a.translation());

    const auto pa = planesAt(scene, a);
    const auto pb = planesAt(scene, b);
    StructuralAligner spa;

    const SE3 truth = b.inverse() * a;
    const std::size_t blind = spa.match(pa, pb).size();
    const std::size_t primed = spa.match(pa, pb, truth).size();
    EXPECT_GT(primed, blind) << "init 이 대응에 쓰이지 않는다";

    const auto r = spa.align(pa, pb, truth);
    ASSERT_TRUE(r.ok()) << r.error().message();
    // 해 자체는 init 이 아니라 법선에서 나와야 한다
    EXPECT_LT(r.value().T_cur_ref.distanceTo(truth).y(), 0.01);
}

TEST(StructuralAligner, ResultIsDeterministic) {
    const auto planes = planesAt(room(), looking(Vec3(-0.8, -1.2, 1.3), Vec3(2.0, 2.5, 0.6)));
    StructuralAligner spa;
    const auto a = spa.align(planes, planes);
    const auto b = spa.align(planes, planes);
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_TRUE(a.value().information.isApprox(b.value().information, 0.0));
    EXPECT_EQ(a.value().observable_dof, b.value().observable_dof);
}
