#pragma once

// ONNX Runtime 기반 YOLO 백엔드.
//
// OpenCV DNN 백엔드(YoloRuntimeCv)는 의존성이 가볍지만 ONNX 임포터가 최신
// YOLO 를 못 읽는 경우가 있다 - YOLO11 의 C2PSA 블록에서 실제로 막혔다.
// ORT 는 ONNX 사양을 그대로 구현하므로 그 제약이 없고, 나중에 CUDA/TensorRT
// 실행 공급자를 붙이는 경로도 여기로 이어진다.
//
// 전처리/디코딩 규약은 CV 백엔드와 완전히 동일해야 한다. 두 백엔드가 같은
// 프레임에서 다른 결과를 내면 둘 중 하나가 틀린 것이고, 그것이 이 구조의 요점이다.
//
// 스레드 안전성: 인스턴스 단위 비공유.

#include "wme/core/Frame.hpp"
#include "wme/perception/Detection.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace wme {

class YoloRuntimeOrt final : public IYoloRuntime {
public:
    [[nodiscard]] static Result<std::unique_ptr<YoloRuntimeOrt>> create(YoloConfig config);

    ~YoloRuntimeOrt() override;
    YoloRuntimeOrt(const YoloRuntimeOrt&) = delete;
    YoloRuntimeOrt& operator=(const YoloRuntimeOrt&) = delete;

    [[nodiscard]] Result<DetectionSet> infer(const Frame& frame) override;

    void setConfidenceScale(float scale) override;

    [[nodiscard]] const YoloConfig& config() const override;

    // 진단용. 모델이 실제로 무엇을 내놓는지.
    [[nodiscard]] const std::vector<std::int64_t>& outputShape() const;

private:
    YoloRuntimeOrt();

    // ORT 헤더를 공개 헤더 밖으로 밀어낸다. 이 타입을 쓰는 쪽이
    // onnxruntime include 경로를 알 필요가 없어야 한다.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wme
