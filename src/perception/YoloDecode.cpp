#include "wme/perception/YoloDecode.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace wme {
namespace {

constexpr int kPadValue = 114;   // YOLO 학습 시 쓰는 패딩 값

}  // namespace

const std::vector<std::string>& cocoClassNames() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};
    return names;
}

Letterbox letterbox(const cv::Mat& bgr, int width, int height, cv::Mat& dst) {
    Letterbox lb;
    lb.scale = std::min(static_cast<double>(width) / bgr.cols,
                        static_cast<double>(height) / bgr.rows);

    const int nw = std::max(1, static_cast<int>(std::round(bgr.cols * lb.scale)));
    const int nh = std::max(1, static_cast<int>(std::round(bgr.rows * lb.scale)));
    lb.pad_x = (width - nw) / 2;
    lb.pad_y = (height - nh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0,
               lb.scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);

    dst.create(height, width, CV_8UC3);
    dst.setTo(cv::Scalar::all(kPadValue));
    resized.copyTo(dst(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));
    return lb;
}

void decodeYolo(const float* data, int channels, int anchors,
                const Letterbox& lb, const cv::Size& src_size,
                float threshold, const std::vector<std::string>& class_names,
                std::vector<Detection>& out) {
    const int ncls = channels - 4;
    if (ncls <= 0 || anchors <= 0) return;

    // 앵커가 마지막 축이라 한 앵커의 값들이 stride=anchors 로 흩어져 있다.
    // 전치본을 만들면 캐시 지역성은 좋아지지만 [A, 4+C] 버퍼를 매 프레임
    // 새로 잡아야 한다. 임계를 넘는 앵커는 보통 전체의 1 % 미만이므로,
    // 먼저 최대 점수만 stride 접근으로 훑고 통과한 것만 상세히 읽는다.
    const double inv_s = 1.0 / std::max(1e-9, lb.scale);

    for (int i = 0; i < anchors; ++i) {
        int   best = -1;
        float best_score = threshold;
        for (int c = 0; c < ncls; ++c) {
            const float s = data[static_cast<std::size_t>(4 + c) * anchors + i];
            if (s > best_score) { best_score = s; best = c; }
        }
        if (best < 0) continue;

        const double cx = (static_cast<double>(data[0 * static_cast<std::size_t>(anchors) + i])
                           - lb.pad_x) * inv_s;
        const double cy = (static_cast<double>(data[1 * static_cast<std::size_t>(anchors) + i])
                           - lb.pad_y) * inv_s;
        const double w  = static_cast<double>(data[2 * static_cast<std::size_t>(anchors) + i]) * inv_s;
        const double h  = static_cast<double>(data[3 * static_cast<std::size_t>(anchors) + i]) * inv_s;

        double x0 = std::clamp(cx - w * 0.5, 0.0, static_cast<double>(src_size.width));
        double y0 = std::clamp(cy - h * 0.5, 0.0, static_cast<double>(src_size.height));
        double x1 = std::clamp(cx + w * 0.5, 0.0, static_cast<double>(src_size.width));
        double y1 = std::clamp(cy + h * 0.5, 0.0, static_cast<double>(src_size.height));
        if (x1 - x0 < 1.0 || y1 - y0 < 1.0) continue;

        Detection d;
        d.class_id   = best;
        if (static_cast<std::size_t>(best) < class_names.size()) {
            d.class_name = class_names[static_cast<std::size_t>(best)];
        }
        d.confidence = best_score;
        d.box = cv::Rect2f(static_cast<float>(x0), static_cast<float>(y0),
                           static_cast<float>(x1 - x0), static_cast<float>(y1 - y0));

        // 분포를 통째로 남긴다. argmax 만 남기면 나중에 오분류를 되돌릴 수 없고,
        // Confidence Engine 이 클래스 사후확률을 갱신할 근거도 사라진다.
        d.class_scores.resize(static_cast<std::size_t>(ncls));
        for (int c = 0; c < ncls; ++c) {
            d.class_scores[static_cast<std::size_t>(c)] =
                data[static_cast<std::size_t>(4 + c) * anchors + i];
        }
        out.push_back(std::move(d));
    }
}

}  // namespace wme
