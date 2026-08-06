#pragma once

// YOLO 출력의 엔진 내부 표현.
// 이 타입은 World Token Engine 경계까지만 존재한다. 그 너머로 박스가 넘어가지
// 않는 것이 WME 의 규칙이다.

#include "wme/core/Result.hpp"
#include "wme/core/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace wme {

// Frame 은 core/Frame.hpp 에 struct 로 정의돼 있다. 여기서 헤더를 끌어오지 않는
// 대신 전방선언만 둔다. 태그(struct/class)가 정의와 어긋나면 MSVC 는 C4099 로
// 넘어가지만 clang 은 -Wmismatched-tags 를 낸다.
struct Frame;

struct Detection {
    int         class_id{-1};
    std::string class_name;
    cv::Rect2f  box;                  // 원본 이미지 좌표
    float       confidence{0.f};

    // 클래스 확률 분포. argmax 만 남기면 재식별 시 오분류를 되돌릴 수 없다.
    std::vector<float> class_scores;

    cv::Mat mask;                     // CV_8UC1, 세그멘테이션 모델일 때만
};

struct DetectionSet {
    Timestamp              stamp{};
    FrameId                frame{};
    std::vector<Detection> items;
    double                 inference_ms{0.0};

    [[nodiscard]] bool empty() const { return items.empty(); }
};

// --- 후처리 유틸 -----------------------------------------------------------

[[nodiscard]] float boxIoU(const cv::Rect2f& a, const cv::Rect2f& b);

// 클래스별 NMS. 서로 다른 클래스 박스는 억제하지 않는다
// (사람이 든 컵이 사람에 의해 지워지면 안 된다).
[[nodiscard]] std::vector<Detection> nonMaxSuppression(std::vector<Detection> detections,
                                                       float iou_threshold,
                                                       std::size_t max_output = 300);

// --- 런타임 ---------------------------------------------------------------

struct YoloConfig {
    std::string model_path;
    int         input_width{640};
    int         input_height{640};
    float       confidence_threshold{0.25f};
    float       nms_iou_threshold{0.45f};
    std::size_t max_detections{300};
    int         device_id{0};
    bool        fp16{true};

    std::vector<std::string> class_names;
};

// 백엔드(TensorRT / ONNX Runtime / 테스트용 목)를 갈아끼우기 위한 경계.
// EnvironmentState 의 detection_threshold_scale 이 여기 임계에 곱해진다.
class IYoloRuntime {
public:
    virtual ~IYoloRuntime() = default;

    [[nodiscard]] virtual Result<DetectionSet> infer(const Frame& frame) = 0;

    // 환경 적응: 어두우면 임계를 낮춰 검출을 살리고, 오검출은 Confidence Engine 이
    // 사후확률로 걸러낸다. 임계를 고정하면 야간에 토큰이 통째로 사라진다.
    virtual void setConfidenceScale(float scale) = 0;

    [[nodiscard]] virtual const YoloConfig& config() const = 0;
};

}  // namespace wme
