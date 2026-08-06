#include "wme/core/Result.hpp"

namespace wme {

std::string_view toString(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None:              return "None";
        case ErrorCode::InvalidArgument:   return "InvalidArgument";
        case ErrorCode::NotInitialized:    return "NotInitialized";
        case ErrorCode::NotAvailable:      return "NotAvailable";
        case ErrorCode::InsufficientData:  return "InsufficientData";
        case ErrorCode::Degenerate:        return "Degenerate";
        case ErrorCode::DidNotConverge:    return "DidNotConverge";
        case ErrorCode::SensorFailure:     return "SensorFailure";
        case ErrorCode::TrackingLost:      return "TrackingLost";
        case ErrorCode::LocalizationLost:  return "LocalizationLost";
        case ErrorCode::ResourceExhausted: return "ResourceExhausted";
        case ErrorCode::BackendError:      return "BackendError";
        case ErrorCode::Timeout:           return "Timeout";
    }
    return "Unknown";
}

}  // namespace wme
