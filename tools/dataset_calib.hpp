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

}  // namespace wme_tools
