#pragma once

// 엔진 경계는 예외 대신 Result 로 실패를 전달한다.
// Degraded 는 "결과는 있지만 신뢰도가 낮다"는 뜻이며, Confidence Engine 이 소비한다.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace wme {

enum class ErrorCode : std::uint16_t {
    None = 0,
    InvalidArgument,
    NotInitialized,
    NotAvailable,       // 아직 데이터가 없음 (실패 아님)
    InsufficientData,   // 관측이 부족해 추정 불가
    Degenerate,         // 수치적 랭크 부족
    DidNotConverge,
    SensorFailure,
    TrackingLost,
    LocalizationLost,
    ResourceExhausted,
    BackendError,
    Timeout,
};

[[nodiscard]] std::string_view toString(ErrorCode code) noexcept;

struct Error {
    ErrorCode   code{ErrorCode::None};
    std::string detail;

    Error() = default;
    Error(ErrorCode c, std::string d = {}) : code(c), detail(std::move(d)) {}

    [[nodiscard]] std::string message() const {
        std::string s(toString(code));
        if (!detail.empty()) { s += ": "; s += detail; }
        return s;
    }
};

// 값 + 품질등급. 실패는 error() 로, 저품질 성공은 degraded() 로 구분한다.
template <typename T>
class Result {
public:
    Result(T value) : storage_(std::move(value)) {}                     // NOLINT(google-explicit-constructor)
    Result(Error error) : storage_(std::move(error)) {}                 // NOLINT(google-explicit-constructor)
    Result(ErrorCode code, std::string detail = {}) : storage_(Error{code, std::move(detail)}) {}

    [[nodiscard]] bool ok() const { return std::holds_alternative<T>(storage_); }
    explicit operator bool() const { return ok(); }

    [[nodiscard]] const T& value() const& { return std::get<T>(storage_); }
    [[nodiscard]] T& value() & { return std::get<T>(storage_); }
    [[nodiscard]] T&& value() && { return std::get<T>(std::move(storage_)); }

    [[nodiscard]] const Error& error() const { return std::get<Error>(storage_); }
    [[nodiscard]] ErrorCode code() const { return ok() ? ErrorCode::None : error().code; }

    // 실패 시 대체값 반환
    [[nodiscard]] T valueOr(T fallback) const& { return ok() ? value() : std::move(fallback); }

    // 저품질 성공 표시. 값은 유효하나 downstream 이 가중치를 낮춰야 한다.
    Result& markDegraded(double reliability, std::string reason = {}) {
        degraded_    = true;
        reliability_ = reliability;
        reason_      = std::move(reason);
        return *this;
    }
    [[nodiscard]] bool   degraded() const { return degraded_; }
    [[nodiscard]] double reliability() const { return ok() ? (degraded_ ? reliability_ : 1.0) : 0.0; }
    [[nodiscard]] const std::string& degradeReason() const { return reason_; }

    static Result degradedValue(T v, double reliability, std::string reason = {}) {
        Result r(std::move(v));
        r.markDegraded(reliability, std::move(reason));
        return r;
    }

private:
    std::variant<T, Error> storage_;
    bool        degraded_{false};
    double      reliability_{1.0};
    std::string reason_;
};

// 반환값이 없는 연산용
class Status {
public:
    Status() = default;
    Status(Error e) : error_(std::move(e)) {}                           // NOLINT(google-explicit-constructor)
    Status(ErrorCode c, std::string d = {}) : error_(Error{c, std::move(d)}) {}

    [[nodiscard]] bool ok() const { return error_.code == ErrorCode::None; }
    explicit operator bool() const { return ok(); }
    [[nodiscard]] const Error& error() const { return error_; }

private:
    Error error_{};
};

}  // namespace wme
