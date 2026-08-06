#include "wme/perception/Detection.hpp"

#include <algorithm>

namespace wme {

float boxIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);

    const float w = x2 - x1, h = y2 - y1;
    if (w <= 0.f || h <= 0.f) return 0.f;

    const float inter = w * h;
    const float uni   = a.area() + b.area() - inter;
    return (uni > 0.f) ? inter / uni : 0.f;
}

std::vector<Detection> nonMaxSuppression(std::vector<Detection> detections,
                                         float iou_threshold, std::size_t max_output) {
    if (detections.empty()) return detections;

    // 신뢰도 내림차순. 동점은 인덱스 순으로 깨서 결정적 결과를 보장한다.
    std::vector<std::size_t> order(detections.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&detections](std::size_t a, std::size_t b) {
        return detections[a].confidence > detections[b].confidence;
    });

    std::vector<char> suppressed(detections.size(), 0);
    std::vector<Detection> kept;
    kept.reserve(std::min(detections.size(), max_output));

    for (std::size_t oi = 0; oi < order.size() && kept.size() < max_output; ++oi) {
        const std::size_t i = order[oi];
        if (suppressed[i]) continue;
        kept.push_back(detections[i]);

        for (std::size_t oj = oi + 1; oj < order.size(); ++oj) {
            const std::size_t j = order[oj];
            if (suppressed[j]) continue;
            // 클래스가 다르면 억제하지 않는다
            if (detections[j].class_id != detections[i].class_id) continue;
            if (boxIoU(detections[i].box, detections[j].box) > iou_threshold) suppressed[j] = 1;
        }
    }
    return kept;
}

}  // namespace wme
