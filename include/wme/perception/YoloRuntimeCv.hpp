#pragma once

// OpenCV DNN 기반 YOLO 백엔드.
//
// TensorRT/ONNX Runtime 백엔드가 붙기 전까지의 기준 구현이자, 그 백엔드들이
// 맞춰야 할 정답이다. 의존성이 OpenCV 하나뿐이라 어디서든 돌고, 다른 백엔드가
// 같은 프레임에서 다른 결과를 내면 그쪽이 틀린 것이다.
//
// 지원: YOLOv8 / YOLO11 계열의 ONNX export.
//   출력 [1, 4+C, A] - 4 는 cx,cy,w,h, C 는 클래스 수, A 는 앵커 수.
//   v5 계열의 objectness 열은 없다. 클래스 점수는 export 시점에 이미 sigmoid 다.
//
// 스레드 안전성: 인스턴스 단위 비공유. cv::dnn::Net 은 동시 forward 를 보장하지 않는다.

#include "wme/core/Frame.hpp"
#include "wme/perception/Detection.hpp"
#include "wme/perception/YoloDecode.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wme {

class YoloRuntimeCv final : public IYoloRuntime {
public:
    // 모델 로드 실패를 생성자에서 던지지 않는다. 호출자가 Result 로 받아
    // 폴백을 정할 수 있어야 한다.
    [[nodiscard]] static Result<std::unique_ptr<YoloRuntimeCv>> create(YoloConfig config);

    [[nodiscard]] Result<DetectionSet> infer(const Frame& frame) override;

    void setConfidenceScale(float scale) override { conf_scale_ = std::max(0.05f, scale); }

    [[nodiscard]] const YoloConfig& config() const override { return cfg_; }

    // 진단용. 모델이 실제로 무엇을 내놓는지 확인할 때 쓴다.
    [[nodiscard]] const std::vector<int>& outputShape() const { return out_shape_; }

private:
    explicit YoloRuntimeCv(YoloConfig config) : cfg_(std::move(config)) {}

    YoloConfig    cfg_;
    cv::dnn::Net  net_;
    float         conf_scale_{1.0f};
    std::vector<int> out_shape_;

    // 핫패스 할당 제거
    cv::Mat bgr_, padded_, blob_;
};

}  // namespace wme
