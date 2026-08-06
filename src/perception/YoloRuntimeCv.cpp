#include "wme/perception/YoloRuntimeCv.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

namespace wme {
namespace {

constexpr int kPadValue = 114;   // 형상 확인용 더미 영상 값

bool fileReadable(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

}  // namespace

Result<std::unique_ptr<YoloRuntimeCv>> YoloRuntimeCv::create(YoloConfig config) {
    if (config.model_path.empty()) {
        return {ErrorCode::InvalidArgument, "YOLO 모델 경로가 비었다"};
    }
    if (!fileReadable(config.model_path)) {
        return {ErrorCode::InvalidArgument, "YOLO 모델을 열 수 없다: " + config.model_path};
    }
    if (config.class_names.empty()) config.class_names = cocoClassNames();

    std::unique_ptr<YoloRuntimeCv> rt(new YoloRuntimeCv(std::move(config)));
    try {
        rt->net_ = cv::dnn::readNetFromONNX(rt->cfg_.model_path);
    } catch (const cv::Exception& e) {
        return {ErrorCode::InvalidArgument, std::string("ONNX 로드 실패: ") + e.what()};
    }
    if (rt->net_.empty()) {
        return {ErrorCode::InvalidArgument, "ONNX 를 읽었지만 네트워크가 비었다"};
    }

    // 이 배포본의 OpenCV 는 CUDA 없이 빌드되어 있어 CPU 로 돈다.
    // GPU 는 TensorRT 백엔드가 맡는다 - 여기서 조용히 느려지지 않도록 명시한다.
    rt->net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    rt->net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 실제 출력 형상을 한 번 확인해 둔다. 모델이 바뀌면 여기서 드러나야지
    // 디코딩 단계에서 그럴듯한 쓰레기가 나오면 안 된다.
    try {
        cv::Mat probe(rt->cfg_.input_height, rt->cfg_.input_width, CV_8UC3,
                      cv::Scalar::all(kPadValue));
        cv::Mat blob = cv::dnn::blobFromImage(probe, 1.0 / 255.0,
                                              cv::Size(rt->cfg_.input_width,
                                                       rt->cfg_.input_height),
                                              cv::Scalar(), true, false);
        rt->net_.setInput(blob);
        const cv::Mat out = rt->net_.forward();
        rt->out_shape_.assign(out.size.p, out.size.p + out.dims);
    } catch (const cv::Exception& e) {
        return {ErrorCode::BackendError, std::string("첫 추론 실패: ") + e.what()};
    }

    if (rt->out_shape_.size() != 3 || rt->out_shape_[0] != 1) {
        return {ErrorCode::InvalidArgument,
                "지원하지 않는 출력 형상. [1, 4+C, A] 를 기대한다"};
    }
    const int chan = rt->out_shape_[1];
    if (chan <= 4 || chan - 4 != static_cast<int>(rt->cfg_.class_names.size())) {
        return {ErrorCode::InvalidArgument,
                "클래스 수 불일치: 모델 " + std::to_string(chan - 4) +
                ", 설정 " + std::to_string(rt->cfg_.class_names.size())};
    }
    return rt;
}

// 전처리/디코딩은 YoloDecode 한 벌만 쓴다. 백엔드마다 따로 구현하면
// 결과가 갈렸을 때 어느 쪽이 틀렸는지 가릴 근거가 없어진다.

Result<DetectionSet> YoloRuntimeCv::infer(const Frame& frame) {
    const auto t0 = std::chrono::steady_clock::now();

    if (!frame.rgb.empty()) {
        bgr_ = frame.rgb;
    } else if (!frame.gray.empty()) {
        // 흑백 입력도 받는다. 3채널로 복제하면 색 정보가 없을 뿐 형상은 유지된다.
        cv::cvtColor(frame.gray, bgr_, cv::COLOR_GRAY2BGR);
    } else {
        return {ErrorCode::InvalidArgument, "프레임에 영상이 없다"};
    }

    const Letterbox lb = letterbox(bgr_, cfg_.input_width, cfg_.input_height, padded_);
    blob_ = cv::dnn::blobFromImage(padded_, 1.0 / 255.0,
                                   cv::Size(cfg_.input_width, cfg_.input_height),
                                   cv::Scalar(), true, false);

    cv::Mat raw;
    try {
        net_.setInput(blob_);
        raw = net_.forward();
    } catch (const cv::Exception& e) {
        return {ErrorCode::BackendError, std::string("추론 실패: ") + e.what()};
    }
    if (raw.dims != 3) {
        return {ErrorCode::BackendError, "출력 차원이 3 이 아니다"};
    }

    const float thresh = std::clamp(cfg_.confidence_threshold * conf_scale_, 0.01f, 0.99f);
    std::vector<Detection> cand;
    cand.reserve(256);
    decodeYolo(raw.ptr<float>(), raw.size[1], raw.size[2], lb, bgr_.size(),
               thresh, cfg_.class_names, cand);

    DetectionSet set;
    set.stamp = frame.stamp;
    set.frame = frame.id;
    set.items = nonMaxSuppression(std::move(cand), cfg_.nms_iou_threshold, cfg_.max_detections);
    set.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return set;
}

}  // namespace wme
