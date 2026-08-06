#pragma once

// WME 전역 기본 타입. 모든 모듈이 이 헤더의 타입으로 통신한다.

#include <Eigen/Core>
#include <Eigen/Geometry>

// <compare> 는 operator<=> 때문에 필요하다. 내장 <=> 를 auto 반환으로 쓰려면
// std::strong_ordering 이 완전한 타입이어야 하고, 그 헤더 없이는 gcc/clang 이
// 곧바로 거부한다. MSVC 는 <chrono> 가 끌어와 주는 바람에 드러나지 않았다.
#include <chrono>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>

namespace wme {

using Scalar = double;   // 추정기 내부 연산 정밀도
using Real   = float;    // 대용량 버퍼(포인트클라우드/이미지) 저장 정밀도

using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
using Mat4 = Eigen::Matrix<Scalar, 4, 4>;
using Mat6 = Eigen::Matrix<Scalar, 6, 6>;
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Vec3f = Eigen::Matrix<Real, 3, 1>;
using Quat  = Eigen::Quaternion<Scalar>;

// --- 시간 -----------------------------------------------------------------
// 엔진 내부 시간은 항상 단조 나노초. 벽시계는 로깅에만 사용한다.
using Nanoseconds = std::chrono::nanoseconds;
using Duration    = std::chrono::duration<double>;

struct Timestamp {
    std::int64_t ns{0};

    constexpr Timestamp() = default;
    constexpr explicit Timestamp(std::int64_t nanos) : ns(nanos) {}

    [[nodiscard]] static Timestamp fromSeconds(double s) {
        return Timestamp(static_cast<std::int64_t>(s * 1e9));
    }
    [[nodiscard]] constexpr double seconds() const { return static_cast<double>(ns) * 1e-9; }
    [[nodiscard]] constexpr bool valid() const { return ns > 0; }

    friend constexpr bool operator==(Timestamp a, Timestamp b) { return a.ns == b.ns; }
    friend constexpr auto operator<=>(Timestamp a, Timestamp b) { return a.ns <=> b.ns; }
    friend constexpr Timestamp operator+(Timestamp a, Nanoseconds d) {
        return Timestamp(a.ns + d.count());
    }
    // 두 시각의 차이를 초 단위로 반환
    friend constexpr double operator-(Timestamp a, Timestamp b) {
        return static_cast<double>(a.ns - b.ns) * 1e-9;
    }
};

// --- 식별자 ---------------------------------------------------------------
// 강타입 ID. 서로 다른 도메인의 인덱스가 뒤섞이는 사고를 컴파일 타임에 막는다.
template <typename Tag>
struct Id {
    std::uint64_t value{kInvalid};
    static constexpr std::uint64_t kInvalid = std::numeric_limits<std::uint64_t>::max();

    constexpr Id() = default;
    constexpr explicit Id(std::uint64_t v) : value(v) {}

    [[nodiscard]] constexpr bool valid() const { return value != kInvalid; }
    friend constexpr bool operator==(Id a, Id b) { return a.value == b.value; }
    friend constexpr auto operator<=>(Id a, Id b) { return a.value <=> b.value; }
};

struct TokenTag;
struct FrameTag;
struct KeyframeTag;
struct NodeTag;

using TokenId    = Id<TokenTag>;
using FrameId    = Id<FrameTag>;
using KeyframeId = Id<KeyframeTag>;
using NodeId     = Id<NodeTag>;

// --- 카메라 ---------------------------------------------------------------
// 핀홀 + Brown-Conrady. 왜곡 보정은 전처리에서 끝내고 엔진 내부는 핀홀만 가정한다.
struct CameraIntrinsics {
    double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
    double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0}, k3{0.0};
    int width{0}, height{0};

    [[nodiscard]] bool valid() const {
        return fx > 0.0 && fy > 0.0 && width > 0 && height > 0;
    }

    // 정규화 좌표 -> 픽셀
    [[nodiscard]] Vec2 project(const Vec3& p_cam) const {
        const double inv_z = 1.0 / p_cam.z();
        return {fx * p_cam.x() * inv_z + cx, fy * p_cam.y() * inv_z + cy};
    }

    // 픽셀 + 깊이 -> 카메라 좌표
    [[nodiscard]] Vec3 backproject(const Vec2& px, double depth) const {
        return {(px.x() - cx) * depth / fx, (px.y() - cy) * depth / fy, depth};
    }

    // 피라미드 레벨에 맞춰 축소된 내부 파라미터
    [[nodiscard]] CameraIntrinsics atPyramidLevel(int level) const {
        const double s = 1.0 / static_cast<double>(1 << level);
        CameraIntrinsics k = *this;
        k.fx *= s; k.fy *= s;
        // 픽셀 중심 규약 유지를 위해 0.5 오프셋 보정
        k.cx = (cx + 0.5) * s - 0.5;
        k.cy = (cy + 0.5) * s - 0.5;
        k.width  = width  >> level;
        k.height = height >> level;
        return k;
    }
};

enum class SensorKind : std::uint8_t { Monocular, Stereo, RgbD, Unknown };

// --- 공통 상수 ------------------------------------------------------------
inline constexpr double kPi      = 3.14159265358979323846;
inline constexpr double kEps     = 1e-10;
inline constexpr double kDeg2Rad = kPi / 180.0;
inline constexpr double kRad2Deg = 180.0 / kPi;

}  // namespace wme

// 강타입 ID 를 해시 컨테이너 키로 쓰기 위한 특수화
namespace std {
template <typename Tag>
struct hash<wme::Id<Tag>> {
    size_t operator()(const wme::Id<Tag>& id) const noexcept {
        return std::hash<std::uint64_t>{}(id.value);
    }
};
}  // namespace std
