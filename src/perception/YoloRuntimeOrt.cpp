#include "wme/perception/YoloRuntimeOrt.hpp"

#include "wme/perception/YoloDecode.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>

namespace wme {
namespace {

bool fileReadable(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
#endif

}  // namespace

// ORT 타입은 전부 여기 갇힌다. 공개 헤더는 onnxruntime 을 알지 못한다.
struct YoloRuntimeOrt::Impl {
    YoloConfig cfg;
    float      conf_scale{1.0f};

    Ort::Env            env{ORT_LOGGING_LEVEL_WARNING, "wme_yolo"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::string in_name, out_name;
    std::vector<std::int64_t> out_shape;

    // 핫패스 할당 제거
    cv::Mat bgr, padded;
    std::vector<float> input;
};

YoloRuntimeOrt::YoloRuntimeOrt() : impl_(std::make_unique<Impl>()) {}
YoloRuntimeOrt::~YoloRuntimeOrt() = default;

const YoloConfig& YoloRuntimeOrt::config() const { return impl_->cfg; }

void YoloRuntimeOrt::setConfidenceScale(float scale) {
    impl_->conf_scale = std::max(0.05f, scale);
}

const std::vector<std::int64_t>& YoloRuntimeOrt::outputShape() const {
    return impl_->out_shape;
}

Result<std::unique_ptr<YoloRuntimeOrt>> YoloRuntimeOrt::create(YoloConfig config) {
    if (config.model_path.empty()) {
        return {ErrorCode::InvalidArgument, "YOLO 모델 경로가 비었다"};
    }
    if (!fileReadable(config.model_path)) {
        return {ErrorCode::InvalidArgument, "YOLO 모델을 열 수 없다: " + config.model_path};
    }
    if (config.class_names.empty()) config.class_names = cocoClassNames();

    std::unique_ptr<YoloRuntimeOrt> rt(new YoloRuntimeOrt());
    Impl& im = *rt->impl_;
    im.cfg = std::move(config);

    try {
        im.opts.SetIntraOpNumThreads(0);   // 0 = 코어 수에 맡긴다
        im.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        im.session = std::make_unique<Ort::Session>(im.env, widen(im.cfg.model_path).c_str(),
                                                    im.opts);
#else
        im.session = std::make_unique<Ort::Session>(im.env, im.cfg.model_path.c_str(), im.opts);
#endif
    } catch (const Ort::Exception& e) {
        return {ErrorCode::InvalidArgument, std::string("ONNX 로드 실패: ") + e.what()};
    }

    if (im.session->GetInputCount() != 1 || im.session->GetOutputCount() < 1) {
        return {ErrorCode::InvalidArgument, "입력 1 / 출력 1 인 모델만 지원한다"};
    }

    Ort::AllocatorWithDefaultOptions alloc;
    im.in_name  = im.session->GetInputNameAllocated(0, alloc).get();
    im.out_name = im.session->GetOutputNameAllocated(0, alloc).get();

    // 출력 형상은 동적 축(-1)일 수 있어 메타데이터만으로는 못 믿는다.
    // 실제로 한 번 돌려 확인한다. 모델이 바뀌면 여기서 드러나야지, 디코딩
    // 단계에서 그럴듯한 쓰레기가 나오면 안 된다.
    try {
        const int W = im.cfg.input_width, H = im.cfg.input_height;
        std::vector<float> probe(static_cast<std::size_t>(3) * W * H, 114.0f / 255.0f);
        const std::array<std::int64_t, 4> shape{1, 3, H, W};
        auto in = Ort::Value::CreateTensor<float>(im.mem, probe.data(), probe.size(),
                                                  shape.data(), shape.size());
        const char* in_names[]  = {im.in_name.c_str()};
        const char* out_names[] = {im.out_name.c_str()};
        auto outs = im.session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        im.out_shape = outs.front().GetTensorTypeAndShapeInfo().GetShape();
    } catch (const Ort::Exception& e) {
        return {ErrorCode::BackendError, std::string("첫 추론 실패: ") + e.what()};
    }

    if (im.out_shape.size() != 3 || im.out_shape[0] != 1) {
        return {ErrorCode::InvalidArgument, "지원하지 않는 출력 형상. [1, 4+C, A] 를 기대한다"};
    }
    const std::int64_t chan = im.out_shape[1];
    if (chan <= 4 || chan - 4 != static_cast<std::int64_t>(im.cfg.class_names.size())) {
        return {ErrorCode::InvalidArgument,
                "클래스 수 불일치: 모델 " + std::to_string(chan - 4) +
                ", 설정 " + std::to_string(im.cfg.class_names.size())};
    }
    return rt;
}

Result<DetectionSet> YoloRuntimeOrt::infer(const Frame& frame) {
    const auto t0 = std::chrono::steady_clock::now();
    Impl& im = *impl_;

    if (!frame.rgb.empty()) {
        im.bgr = frame.rgb;
    } else if (!frame.gray.empty()) {
        // 흑백도 받는다. 색 정보가 없을 뿐 형상은 유지된다.
        cv::cvtColor(frame.gray, im.bgr, cv::COLOR_GRAY2BGR);
    } else {
        return {ErrorCode::InvalidArgument, "프레임에 영상이 없다"};
    }

    const int W = im.cfg.input_width, H = im.cfg.input_height;
    const Letterbox lb = letterbox(im.bgr, W, H, im.padded);

    // NCHW, RGB, 0..1. blobFromImage 와 같은 규약을 손으로 맞춘다.
    im.input.resize(static_cast<std::size_t>(3) * W * H);
    const std::size_t plane = static_cast<std::size_t>(W) * H;
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row = im.padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * W + x;
            im.input[i]             = row[x][2] / 255.0f;   // R
            im.input[plane + i]     = row[x][1] / 255.0f;   // G
            im.input[2 * plane + i] = row[x][0] / 255.0f;   // B
        }
    }

    std::vector<Detection> cand;
    try {
        const std::array<std::int64_t, 4> shape{1, 3, H, W};
        auto in = Ort::Value::CreateTensor<float>(im.mem, im.input.data(), im.input.size(),
                                                  shape.data(), shape.size());
        const char* in_names[]  = {im.in_name.c_str()};
        const char* out_names[] = {im.out_name.c_str()};
        auto outs = im.session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

        const auto info = outs.front().GetTensorTypeAndShapeInfo();
        const auto shp  = info.GetShape();
        if (shp.size() != 3) {
            return {ErrorCode::BackendError, "출력 차원이 3 이 아니다"};
        }
        const float thresh =
            std::clamp(im.cfg.confidence_threshold * im.conf_scale, 0.01f, 0.99f);
        cand.reserve(256);
        decodeYolo(outs.front().GetTensorData<float>(),
                   static_cast<int>(shp[1]), static_cast<int>(shp[2]),
                   lb, im.bgr.size(), thresh, im.cfg.class_names, cand);
    } catch (const Ort::Exception& e) {
        return {ErrorCode::BackendError, std::string("추론 실패: ") + e.what()};
    }

    DetectionSet set;
    set.stamp = frame.stamp;
    set.frame = frame.id;
    set.items = nonMaxSuppression(std::move(cand), im.cfg.nms_iou_threshold,
                                  im.cfg.max_detections);
    set.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return set;
}

}  // namespace wme
