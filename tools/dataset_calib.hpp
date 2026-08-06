#pragma once

// 시퀀스 폴더의 선택적 calib.txt 로 내부파라미터/왜곡/깊이스케일을 덮어쓴다.
//
// 왜 필요한가: tum_odometry / tum_baseline 은 경로에 "freiburg2" 가 들어있는지로
// 내부파라미터를 골라 왔다. TUM 안에서는 통했지만, 데이터셋이 하나 늘어나는
// 순간 이 방식은 조용히 틀린다 - KITTI 시퀀스 경로에는 freiburg 가 없으므로
// fr1 값(fx=517, 640x480)이 1241x376 영상에 적용되고, 결과는 실패가 아니라
// 그럴듯한 오차로 나온다. 파일이 있으면 파일이 이긴다.
//
// 형식 (한 줄 하나, '#' 주석):
//   fx: 718.856
//   fy: 718.856
//   cx: 607.1928
//   cy: 185.2157
//   width: 1241
//   height: 376
//   depth_scale: 256.0      # PNG 값 -> m 나눗셈 계수
//   depth_min: 0.5          # 이 밖의 깊이는 무효로 본다
//   depth_max: 80.0
//   dist: k1 k2 p1 p2 k3    # 없으면 0
//
// depth_min/max 가 왜 여기 있는가: 두 도구 모두 `z > 0.1 && z < 8.0` 을 코드에
// 박아 두고 "TUM 깊이 유효 범위" 라고 적어 두었다. KITTI 는 유효 깊이 중앙이
// 13.9 m, 최대 47.9 m 라서 그 상한이 특징점의 거의 전부를 버린다 - 실측에서
// 기준선의 3D 대응이 프레임당 164 개에서 35 개로 떨어졌고, 400 프레임 중 371
// 프레임이 추적 실패했다. 그 상태로 낸 비교는 알고리즘 비교가 아니다.

#include "wme/core/Frame.hpp"

#include <opencv2/core.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace wme_tools {

struct DatasetCalib {
    wme::CameraIntrinsics K{};
    cv::Vec<double, 5>    dist{0, 0, 0, 0, 0};
    double                depth_scale{5000.0};
    // 기본값은 TUM RGB-D 센서의 유효 범위다. calib.txt 가 덮어쓴다.
    double                depth_min{0.1};
    double                depth_max{8.0};
    bool                  from_file{false};
};

// TUM freiburg 그룹별 기본 내부파라미터/왜곡.
//
// 한 곳에만 둔다. 이 값들이 tum_odometry.cpp 와 tum_baseline.cpp 에 각각
// 복사되어 있었고, 두 도구가 같은 값을 쓴다는 것이 비교의 전제인데 그 전제를
// 복사본 두 개가 지키고 있었다. 세 번째 소비자(bench_viewer)가 생기는 김에
// 합친다.
//
// 왜곡 계수도 그룹마다 다르다. TUM 이 배포하는 fr1/fr2 영상은 보정되어 있지
// 않고 계수가 작지 않아서(fr1 k1=0.2624, k2=-0.9531), 보정 없이 핀홀을 쓰면
// 가장자리가 수 픽셀 어긋나고 그 오차가 매 프레임 같은 방향으로 쌓인다.
// 실측: 보정 전 ATE 67 cm -> 보정 후 24 cm. fr3 은 TUM 이 이미 보정해 배포한다.
inline DatasetCalib defaultCalibForPath(const std::string& root, bool announce = true) {
    DatasetCalib c;
    c.K.width = 640;
    c.K.height = 480;
    const char* which = nullptr;

    if (root.find("freiburg2") != std::string::npos) {
        c.K.fx = 520.908620; c.K.fy = 521.007327;
        c.K.cx = 325.141442; c.K.cy = 249.701764;
        c.dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
        which = "freiburg2";
    } else if (root.find("freiburg3") != std::string::npos) {
        c.K.fx = 535.4; c.K.fy = 539.2;
        c.K.cx = 320.1; c.K.cy = 247.6;
        c.dist = {0.0, 0.0, 0.0, 0.0, 0.0};
        which = "freiburg3 (왜곡 보정본)";
    } else {
        c.K.fx = 517.306408; c.K.fy = 516.469215;
        c.K.cx = 318.643040; c.K.cy = 255.313989;
        c.dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
        which = "freiburg1";
    }
    if (announce) std::cout << "내부파라미터: " << which << "\n";
    return c;
}

// 기본값 + calib.txt 덮어쓰기를 한 번에. 정의는 loadDatasetCalib 뒤에 있다.
inline bool resolveCalib(const std::string& root, DatasetCalib& out, bool announce = true);

// root/calib.txt 가 있으면 읽어 덮어쓴다. 없으면 인자를 그대로 돌려준다.
// 파일이 있는데 파싱이 깨지면 조용히 넘어가지 않고 실패시킨다.
inline bool loadDatasetCalib(const std::string& root, DatasetCalib& io) {
    std::ifstream f(root + "/calib.txt");
    if (!f) return true;   // 없는 것은 정상

    std::string line;
    int seen = 0;
    while (std::getline(f, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::istringstream vs(line.substr(colon + 1));

        // 앞뒤 공백 제거
        const auto b = key.find_first_not_of(" \t");
        const auto e = key.find_last_not_of(" \t");
        if (b == std::string::npos) continue;
        key = key.substr(b, e - b + 1);

        if      (key == "fx")     { vs >> io.K.fx; ++seen; }
        else if (key == "fy")     { vs >> io.K.fy; ++seen; }
        else if (key == "cx")     { vs >> io.K.cx; ++seen; }
        else if (key == "cy")     { vs >> io.K.cy; ++seen; }
        else if (key == "width")  { vs >> io.K.width;  ++seen; }
        else if (key == "height") { vs >> io.K.height; ++seen; }
        else if (key == "depth_scale") { vs >> io.depth_scale; ++seen; }
        else if (key == "depth_min")   { vs >> io.depth_min; }
        else if (key == "depth_max")   { vs >> io.depth_max; }
        else if (key == "dist") {
            for (int i = 0; i < 5; ++i) vs >> io.dist[i];
            ++seen;
        }
    }
    if (seen < 6) {
        std::cerr << "calib.txt 가 있으나 항목이 부족하다 (" << seen
                  << "개). fx/fy/cx/cy/width/height 는 필수다: " << root << "\n";
        return false;
    }
    io.from_file = true;
    std::cout << "내부파라미터: " << root << "/calib.txt  "
              << io.K.width << "x" << io.K.height
              << " f=(" << io.K.fx << "," << io.K.fy << ")"
              << " c=(" << io.K.cx << "," << io.K.cy << ")"
              << " depth_scale=" << io.depth_scale
              << " depth=[" << io.depth_min << "," << io.depth_max << "] m\n";
    return true;
}

inline bool resolveCalib(const std::string& root, DatasetCalib& out, bool announce) {
    out = defaultCalibForPath(root, announce);
    return loadDatasetCalib(root, out);
}

}  // namespace wme_tools
