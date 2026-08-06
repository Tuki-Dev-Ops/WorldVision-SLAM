#pragma once

// YOLO 전처리/후처리. 백엔드와 무관한 부분을 모아 둔다.
//
// 백엔드가 여럿(OpenCV DNN, ONNX Runtime, 나중에 TensorRT)일 때 이 부분을
// 각자 구현하면 "두 백엔드가 다른 결과를 낸다" 는 상황이 생기고, 그때 어느 쪽이
// 틀렸는지 가릴 근거가 없다. 여기 한 벌만 두면 백엔드 차이는 추론 엔진 차이로
// 좁혀진다.

#include "wme/perception/Detection.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace wme {

// COCO 80 클래스. 모델 export 가 이름을 담지 않으므로 여기서 준다.
// 백엔드와 무관하므로 특정 백엔드 헤더에 두지 않는다.
[[nodiscard]] const std::vector<std::string>& cocoClassNames();

// 종횡비를 유지한 채 패딩한 결과와, 원본 좌표로 되돌리는 데 필요한 값.
// 늘려서 맞추면 박스 종횡비가 뒤틀리고 그 왜곡이 토큰의 물리 치수로 넘어간다.
struct Letterbox {
    double scale{1.0};
    int    pad_x{0};
    int    pad_y{0};
};

// bgr -> (dst) 패딩된 정사각 영상. 반환값으로 역변환 정보를 준다.
Letterbox letterbox(const cv::Mat& bgr, int width, int height, cv::Mat& dst);

// YOLOv8/YOLO11 계열 출력 [4+C, A] 를 Detection 목록으로.
//   - 행 0..3 = cx, cy, w, h (letterbox 좌표)
//   - 행 4..  = 클래스 점수. export 시점에 이미 확률이라 sigmoid 를 다시 걸지 않는다.
//   - v5 계열의 objectness 열은 없다.
// data 는 [4+C, A] 를 행 우선으로 담은 연속 버퍼여야 한다.
void decodeYolo(const float* data, int channels, int anchors,
                const Letterbox& lb, const cv::Size& src_size,
                float threshold, const std::vector<std::string>& class_names,
                std::vector<Detection>& out);

}  // namespace wme
