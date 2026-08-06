#pragma once

// SO(3)/SE(3) 리군 연산. 최적화 루프에서 매 프레임 호출되므로 헤더 인라인.
// 규약: 접선벡터 xi = [rho(3), phi(3)], 좌측 섭동 T <- exp(dxi) * T.

#include "wme/core/Types.hpp"

#include <cmath>

namespace wme {

// 반대칭 행렬
[[nodiscard]] inline Mat3 skew(const Vec3& v) {
    Mat3 m;
    m <<     0.0, -v.z(),  v.y(),
          v.z(),     0.0, -v.x(),
         -v.y(),  v.x(),     0.0;
    return m;
}

// --- SO(3) ----------------------------------------------------------------
class SO3 {
public:
    SO3() : q_(Quat::Identity()) {}
    explicit SO3(const Quat& q) : q_(q.normalized()) {}
    explicit SO3(const Mat3& R) : q_(Quat(R).normalized()) {}

    [[nodiscard]] const Quat& unitQuaternion() const { return q_; }
    [[nodiscard]] Mat3 matrix() const { return q_.toRotationMatrix(); }
    [[nodiscard]] SO3 inverse() const { return SO3(q_.conjugate()); }

    SO3 operator*(const SO3& o) const { return SO3(q_ * o.q_); }
    Vec3 operator*(const Vec3& p) const { return q_ * p; }

    // 지수사상: 회전벡터 -> SO(3). theta 소각에서 테일러 전개로 안정화.
    [[nodiscard]] static SO3 exp(const Vec3& phi) {
        const double theta = phi.norm();
        if (theta < 1e-8) {
            // sin(t/2)/t ~ 1/2 - t^2/48
            const double half = 0.5 - theta * theta / 48.0;
            return SO3(Quat(1.0 - theta * theta / 8.0, half * phi.x(), half * phi.y(), half * phi.z()));
        }
        const double half_theta = 0.5 * theta;
        const double s = std::sin(half_theta) / theta;
        return SO3(Quat(std::cos(half_theta), s * phi.x(), s * phi.y(), s * phi.z()));
    }

    // 로그사상: SO(3) -> 회전벡터
    [[nodiscard]] Vec3 log() const {
        // w 부호를 양수로 정규화해 |theta| <= pi 보장
        Quat q = (q_.w() < 0.0) ? Quat(-q_.w(), -q_.x(), -q_.y(), -q_.z()) : q_;
        const Vec3 v = q.vec();
        const double n = v.norm();
        if (n < 1e-8) {
            // atan2(n,w)/n ~ (1/w)(1 - n^2/(3w^2))
            const double f = 2.0 / q.w() * (1.0 - n * n / (3.0 * q.w() * q.w()));
            return f * v;
        }
        return (2.0 * std::atan2(n, q.w()) / n) * v;
    }

    // 좌측 자코비안. 불확실성 전파에 사용.
    [[nodiscard]] static Mat3 leftJacobian(const Vec3& phi) {
        const double t = phi.norm();
        if (t < 1e-8) return Mat3::Identity() + 0.5 * skew(phi);
        const Mat3 W = skew(phi);
        const double t2 = t * t;
        return Mat3::Identity() + ((1.0 - std::cos(t)) / t2) * W
                                + ((t - std::sin(t)) / (t2 * t)) * W * W;
    }

    [[nodiscard]] static Mat3 leftJacobianInverse(const Vec3& phi) {
        const double t = phi.norm();
        if (t < 1e-8) return Mat3::Identity() - 0.5 * skew(phi);
        const Mat3 W = skew(phi);
        const double c = (1.0 / (t * t)) - (1.0 + std::cos(t)) / (2.0 * t * std::sin(t));
        return Mat3::Identity() - 0.5 * W + c * W * W;
    }

    void normalize() { q_.normalize(); }

    [[nodiscard]] static SO3 identity() { return SO3(); }

private:
    Quat q_;
};

// --- SE(3) ----------------------------------------------------------------
class SE3 {
public:
    SE3() : t_(Vec3::Zero()) {}
    SE3(const SO3& R, const Vec3& t) : R_(R), t_(t) {}
    SE3(const Mat3& R, const Vec3& t) : R_(R), t_(t) {}
    explicit SE3(const Mat4& T) : R_(Mat3(T.block<3, 3>(0, 0))), t_(T.block<3, 1>(0, 3)) {}

    [[nodiscard]] const SO3& rotation() const { return R_; }
    [[nodiscard]] const Vec3& translation() const { return t_; }
    [[nodiscard]] Vec3& translation() { return t_; }

    [[nodiscard]] Mat4 matrix() const {
        Mat4 T = Mat4::Identity();
        T.block<3, 3>(0, 0) = R_.matrix();
        T.block<3, 1>(0, 3) = t_;
        return T;
    }

    [[nodiscard]] SE3 inverse() const {
        const SO3 Rinv = R_.inverse();
        return SE3(Rinv, -(Rinv * t_));
    }

    SE3 operator*(const SE3& o) const { return SE3(R_ * o.R_, R_ * o.t_ + t_); }
    Vec3 operator*(const Vec3& p) const { return R_ * p + t_; }

    [[nodiscard]] static SE3 exp(const Vec6& xi) {
        const Vec3 rho = xi.head<3>();
        const Vec3 phi = xi.tail<3>();
        return SE3(SO3::exp(phi), SO3::leftJacobian(phi) * rho);
    }

    [[nodiscard]] Vec6 log() const {
        const Vec3 phi = R_.log();
        Vec6 xi;
        xi.head<3>() = SO3::leftJacobianInverse(phi) * t_;
        xi.tail<3>() = phi;
        return xi;
    }

    // 수반행렬. 좌/우 섭동 좌표 변환에 사용.
    [[nodiscard]] Mat6 adjoint() const {
        const Mat3 R = R_.matrix();
        Mat6 A = Mat6::Zero();
        A.block<3, 3>(0, 0) = R;
        A.block<3, 3>(0, 3) = skew(t_) * R;
        A.block<3, 3>(3, 3) = R;
        return A;
    }

    // 두 포즈 사이 거리 (병진 m, 회전 rad). 수렴 판정과 평가지표에 공용.
    [[nodiscard]] Vec2 distanceTo(const SE3& other) const {
        const SE3 d = inverse() * other;
        return {d.translation().norm(), d.rotation().log().norm()};
    }

    // 누적 수치오차로 회전이 SO(3)를 벗어나는 것을 방지
    void normalize() { R_.normalize(); }

    [[nodiscard]] static SE3 identity() { return SE3(); }

private:
    SO3  R_;
    Vec3 t_;
};

// 좌측 섭동 적용
inline SE3 perturb(const SE3& T, const Vec6& dxi) { return SE3::exp(dxi) * T; }

}  // namespace wme
