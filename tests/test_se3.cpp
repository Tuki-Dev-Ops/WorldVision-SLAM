// SE(3)/SO(3) 리군 연산 검증.
// 해석적으로 답을 아는 케이스 + 무작위 성질검사(exp/log 왕복, 수반행렬 항등식).

#include "wme/core/SE3.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace wme;

namespace {

std::mt19937& rng() {
    static std::mt19937 gen(20260803);   // 결정적 재현을 위해 고정 시드
    return gen;
}

Vec3 randomVec3(double scale) {
    std::uniform_real_distribution<double> d(-scale, scale);
    return {d(rng()), d(rng()), d(rng())};
}

Vec6 randomTangent(double rot_scale, double trans_scale) {
    Vec6 xi;
    xi.head<3>() = randomVec3(trans_scale);
    xi.tail<3>() = randomVec3(rot_scale);
    return xi;
}

}  // namespace

TEST(SO3, ExpOfZeroIsIdentity) {
    const SO3 R = SO3::exp(Vec3::Zero());
    EXPECT_NEAR((R.matrix() - Mat3::Identity()).norm(), 0.0, 1e-12);
}

TEST(SO3, ExpKnownRotation) {
    // z축 90도
    const SO3 R = SO3::exp(Vec3(0.0, 0.0, kPi / 2.0));
    const Vec3 x_rotated = R * Vec3(1.0, 0.0, 0.0);
    EXPECT_NEAR(x_rotated.x(), 0.0, 1e-12);
    EXPECT_NEAR(x_rotated.y(), 1.0, 1e-12);
    EXPECT_NEAR(x_rotated.z(), 0.0, 1e-12);
}

TEST(SO3, LogExpRoundTrip) {
    for (int i = 0; i < 500; ++i) {
        const Vec3 phi = randomVec3(2.5);       // |phi| < pi 범위 유지
        if (phi.norm() > kPi * 0.98) continue;
        const Vec3 back = SO3::exp(phi).log();
        EXPECT_NEAR((back - phi).norm(), 0.0, 1e-9) << "phi=" << phi.transpose();
    }
}

TEST(SO3, SmallAngleStability) {
    // 테일러 분기 경계에서 수치적으로 튀지 않아야 한다
    for (double t : {1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6}) {
        const Vec3 phi(t, -t * 0.5, t * 0.25);
        const Vec3 back = SO3::exp(phi).log();
        EXPECT_LT((back - phi).norm(), 1e-14 + 1e-6 * phi.norm());
    }
}

TEST(SO3, LeftJacobianInverseIsInverse) {
    for (int i = 0; i < 200; ++i) {
        const Vec3 phi = randomVec3(2.0);
        if (phi.norm() < 1e-6 || phi.norm() > 3.0) continue;
        const Mat3 J  = SO3::leftJacobian(phi);
        const Mat3 Ji = SO3::leftJacobianInverse(phi);
        EXPECT_NEAR((J * Ji - Mat3::Identity()).norm(), 0.0, 1e-8);
    }
}

TEST(SE3, InverseComposesToIdentity) {
    for (int i = 0; i < 300; ++i) {
        const SE3 T = SE3::exp(randomTangent(2.0, 10.0));
        const SE3 I = T * T.inverse();
        EXPECT_NEAR(I.translation().norm(), 0.0, 1e-9);
        EXPECT_NEAR(I.rotation().log().norm(), 0.0, 1e-9);
    }
}

TEST(SE3, LogExpRoundTrip) {
    for (int i = 0; i < 500; ++i) {
        const Vec6 xi = randomTangent(2.0, 5.0);
        if (xi.tail<3>().norm() > kPi * 0.95) continue;
        const Vec6 back = SE3::exp(xi).log();
        EXPECT_NEAR((back - xi).norm(), 0.0, 1e-8);
    }
}

TEST(SE3, AdjointIdentity) {
    // Adj(T) * xi == log(T * exp(xi) * T^-1)
    for (int i = 0; i < 200; ++i) {
        const SE3  T  = SE3::exp(randomTangent(1.5, 3.0));
        const Vec6 xi = randomTangent(0.05, 0.05);   // 작은 접선벡터에서 검증

        const Vec6 lhs = T.adjoint() * xi;
        const Vec6 rhs = (T * SE3::exp(xi) * T.inverse()).log();
        EXPECT_NEAR((lhs - rhs).norm(), 0.0, 1e-7);
    }
}

TEST(SE3, ActionMatchesMatrixForm) {
    const SE3 T = SE3::exp(randomTangent(1.0, 2.0));
    for (int i = 0; i < 100; ++i) {
        const Vec3 p = randomVec3(5.0);
        const Vec4 ph(p.x(), p.y(), p.z(), 1.0);
        const Vec4 q = T.matrix() * ph;
        EXPECT_NEAR((T * p - q.head<3>()).norm(), 0.0, 1e-10);
    }
}

TEST(SE3, DistanceToIsZeroForSelf) {
    const SE3 T = SE3::exp(randomTangent(1.0, 4.0));
    const Vec2 d = T.distanceTo(T);
    EXPECT_NEAR(d.x(), 0.0, 1e-12);
    EXPECT_NEAR(d.y(), 0.0, 1e-12);
}

TEST(CameraIntrinsics, PyramidPreservesProjection) {
    CameraIntrinsics K;
    K.fx = 525.0; K.fy = 525.0; K.cx = 319.5; K.cy = 239.5;
    K.width = 640; K.height = 480;

    const Vec3 p(0.4, -0.3, 3.0);
    const Vec2 px0 = K.project(p);

    for (int level = 1; level <= 3; ++level) {
        const Vec2 px_l = K.atPyramidLevel(level).project(p);
        const double s  = 1.0 / static_cast<double>(1 << level);
        // 픽셀 중심 규약(+0.5 오프셋)까지 일관되어야 한다
        EXPECT_NEAR(px_l.x(), (px0.x() + 0.5) * s - 0.5, 1e-9);
        EXPECT_NEAR(px_l.y(), (px0.y() + 0.5) * s - 0.5, 1e-9);
    }
}

TEST(CameraIntrinsics, BackprojectInvertsProject) {
    CameraIntrinsics K;
    K.fx = 600.0; K.fy = 610.0; K.cx = 320.0; K.cy = 240.0;
    K.width = 640; K.height = 480;

    const Vec3 p(1.2, 0.7, 4.5);
    const Vec2 px = K.project(p);
    const Vec3 q  = K.backproject(px, p.z());
    EXPECT_NEAR((p - q).norm(), 0.0, 1e-10);
}
