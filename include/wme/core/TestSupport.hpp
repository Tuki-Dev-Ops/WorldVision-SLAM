#pragma once

// 테스트 전용 지원 도구. 엔진 코드는 이 헤더를 포함하지 않는다.
//
// 존재 이유: 결정성은 "값이 비슷하다"가 아니라 "비트가 같다"로만 검증된다.
// 허용오차 비교는 비결정적인 시스템에서도 통과하므로 아무것도 재지 않는다
// (docs/06-results.md 10.4 - 측정이 판별하는지 먼저 확인할 것).
//
// 그래서 여기서는 관측 가능한 출력의 원시 바이트를 그대로 이어붙인 Blob 을
// 만들고 memcmp 로만 비교한다. double 은 bit_cast 로 다루므로 1 ULP 차이도
// 반드시 드러난다.

#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace wme::test {

// 출력의 원시 바이트 열. 순서까지 포함해 기록되므로 값이 같고 순서만 달라도
// 다른 Blob 이 된다 - 순서 비결정성도 잡아야 하기 때문이다.
class Blob {
public:
    // 산술/열거 타입을 바이트 그대로 담는다. 패딩이 있는 구조체는 받지 않는다.
    template <typename T>
    void put(const T& v) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
                      "패딩 바이트가 섞이면 비교가 무의미해진다 - 스칼라만 넣는다");
        const auto* p = reinterpret_cast<const std::byte*>(&v);
        bytes_.insert(bytes_.end(), p, p + sizeof(T));
    }

    void putBytes(const void* p, std::size_t n) {
        const auto* b = static_cast<const std::byte*>(p);
        bytes_.insert(bytes_.end(), b, b + n);
    }

    void put(const std::string& s) {
        put(static_cast<std::uint64_t>(s.size()));
        putBytes(s.data(), s.size());
    }

    // Eigen 고정 크기 행렬/벡터. 저장 순서를 명시적으로 고정한다.
    template <typename Derived>
    void putEigen(const Eigen::MatrixBase<Derived>& m) {
        for (Eigen::Index c = 0; c < m.cols(); ++c) {
            for (Eigen::Index r = 0; r < m.rows(); ++r) put(m(r, c));
        }
    }

    void put(const SE3& T) {
        putEigen(T.rotation().unitQuaternion().coeffs());
        putEigen(T.translation());
    }

    void put(Timestamp t) { put(t.ns); }

    template <typename Tag>
    void put(const Id<Tag>& id) {
        put(id.value);
    }

    [[nodiscard]] std::size_t size() const { return bytes_.size(); }
    [[nodiscard]] const std::byte* data() const { return bytes_.data(); }
    [[nodiscard]] bool empty() const { return bytes_.empty(); }

    // 비교는 오직 memcmp. 허용오차는 존재하지 않는다.
    friend bool operator==(const Blob& a, const Blob& b) {
        return a.bytes_.size() == b.bytes_.size() &&
               (a.bytes_.empty() ||
                std::memcmp(a.bytes_.data(), b.bytes_.data(), a.bytes_.size()) == 0);
    }
    friend bool operator!=(const Blob& a, const Blob& b) { return !(a == b); }

    // 첫 불일치 바이트 위치. 실패 메시지에 어디가 갈라졌는지 남긴다.
    [[nodiscard]] std::size_t firstDifference(const Blob& o) const {
        const std::size_t n = std::min(bytes_.size(), o.bytes_.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (bytes_[i] != o.bytes_[i]) return i;
        }
        return (bytes_.size() == o.bytes_.size()) ? npos : n;
    }
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // 로그용 64비트 요약 (FNV-1a). 비교 자체에는 쓰지 않는다 - 해시 충돌이
    // 결정성 결함을 가려서는 안 되므로 판정은 언제나 memcmp 다.
    [[nodiscard]] std::string hex() const {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::byte b : bytes_) {
            h ^= static_cast<std::uint64_t>(b);
            h *= 1099511628211ULL;
        }
        static const char* kDigits = "0123456789abcdef";
        std::string out(16, '0');
        for (int i = 15; i >= 0; --i) {
            out[static_cast<std::size_t>(i)] = kDigits[h & 0xF];
            h >>= 4;
        }
        return out;
    }

    // 1 ULP 만 어긋난 사본. 비교기가 정말 비트 단위인지 확인하는 대조군용.
    [[nodiscard]] Blob perturbedByOneUlp(std::size_t double_index) const {
        Blob copy = *this;
        const std::size_t off = double_index * sizeof(double);
        if (off + sizeof(double) > copy.bytes_.size()) return copy;
        std::uint64_t bits = 0;
        std::memcpy(&bits, copy.bytes_.data() + off, sizeof(bits));
        bits ^= 1ULL;   // 최하위 가수 비트 하나
        std::memcpy(copy.bytes_.data() + off, &bits, sizeof(bits));
        return copy;
    }

private:
    std::vector<std::byte> bytes_;
};

}  // namespace wme::test
