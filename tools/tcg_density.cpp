// TCG 노드 밀도 - 무엇이 실제로 개수를 움직이는가.
//
// 24.2 는 실내 재측위 실패의 원인을 밀도로 지목했다: 키프레임 92 개 중 **26 개**
// 만 성좌 질의에 필요한 객체 4 개를 채웠고, 그 이유는 COCO 80 클래스가 사무실을
// 거의 덮지 못한다는 것이었다. 25.9 는 그 실패가 왜 중요한지를 보탰다 - 안개
// 속에서 측광 밖에 남는 유일한 복구 경로가 성좌인데, 그것이 작동하지 않는다.
//
// 고치기 전에 **어느 손잡이가 실제로 개수를 움직이는지** 부터 잰다. 후보는 네 개고
// 각각 다른 것을 희생한다:
//
//   1. 신뢰도 문턱 낮추기  - 검출은 늘지만 오검출도 늘어 성좌가 오염된다
//   2. 깊이 유효성         - 검출은 있는데 박스 중앙에 깊이가 없어 노드가 못 되는 경우
//   3. 시간 창 누적        - 검출 깜빡임을 평균한다. 진리값 아닌 상대 포즈가 필요하다
//   4. 클래스 중복 허용    - 같은 클래스 여러 개를 다 쓰는가
//
// 판정은 하지 않는다. 정책별 노드 수와 "4 개 이상" 비율만 세어 CSV 와 요약으로
// 내보낸다. 어느 것을 채택할지는 그 표를 보고 정한다.
//
// 사용:
//   wme_tcg_density <시퀀스> --yolo <model.onnx> [--kf-dist M] [--window N]
//                   [--nodes out.csv] [--stride N] [--conf T]
//
// --nodes 는 밀도표와 별개의 두 번째 목적이다: 성좌 노드를 그대로 CSV 로 흘려
// 보내 **노드 위치 오차를 진리값 포즈로 실측** 할 수 있게 한다. sigma 모델
// (sz = c*z^2) 의 c 가 맞는지, Kabsch 잔차 게이트 max_rms_error 가 맞는지는
// 둘 중 하나를 다른 하나에 맞춰 올리는 것으로는 판정되지 않는다. 같은 물체를
// 두 프레임에서 본 위치의 차이가 유일한 외부 근거다.
// 계산 규약은 tools/tum_fusion.cpp 의 nodesFromFrame 과 같아야 한다 - 다르면
// 여기서 잰 sigma 가 Tier 1 이 실제로 쓰는 sigma 가 아니다.

#include "dataset_calib.hpp"

#include "wme/core/Frame.hpp"
#include "wme/perception/YoloRuntimeOrt.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct IndexRow { double stamp{0.0}; std::string file; };

std::vector<IndexRow> readIndex(const std::string& p) {
    std::vector<IndexRow> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        IndexRow r;
        if (ss >> r.stamp >> r.file) out.push_back(r);
    }
    return out;
}

bool fileExists(const std::string& p) { std::ifstream f(p, std::ios::binary); return f.good(); }

int nearest(const std::vector<IndexRow>& rows, double stamp, double tol) {
    int best = -1; double bd = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

// 박스 중앙 절반의 중앙값 깊이. 한 화소만 보면 구멍 하나로 노드가 사라진다.
//
// 유효 범위는 데이터셋에서 온다. 예전에는 `zz > 0.1 && zz < 8.0` 이 박혀 있었고
// 그것은 TUM RGB-D 센서의 범위다 - KITTI(유효 [3.2, 80] m)에서는 노드의 거의
// 전부가 상한에 잘려 이 도구가 "밀도가 낮다" 는 잘못된 답을 냈다.
// 같은 상수가 두 도구를 굶긴 사고가 dataset_calib.hpp 머리말에 적혀 있다.
bool boxDepth(const cv::Mat& depth, const wme::Detection& d, double zmin, double zmax,
              double& z) {
    const int u = static_cast<int>(d.box.x + d.box.width * 0.5);
    const int v = static_cast<int>(d.box.y + d.box.height * 0.5);
    const int hw = std::max(1, static_cast<int>(d.box.width * 0.25));
    const int hh = std::max(1, static_cast<int>(d.box.height * 0.25));
    std::vector<double> zs;
    for (int y = std::max(0, v - hh); y < std::min(depth.rows, v + hh); ++y) {
        for (int x = std::max(0, u - hw); x < std::min(depth.cols, u + hw); ++x) {
            const float zz = depth.at<float>(y, x);
            if (std::isfinite(zz) && zz > static_cast<float>(zmin) &&
                zz < static_cast<float>(zmax)) zs.push_back(zz);
        }
    }
    if (zs.size() < 8) return false;
    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
    z = zs[zs.size() / 2];
    return true;
}

// 성좌 노드 한 프레임분을 CSV 로 흘린다.
//
// 계산은 tools/tum_fusion.cpp 의 nodesFromFrame 을 그대로 옮긴 것이다 -
// 박스 중앙 절반의 중앙값 깊이, 반치수 무게중심 보정, 그리고 sigma 모델.
// sigma_model 은 "현재 코드가 믿는 값" 이고 이 열의 존재 이유는 그것을 실측과
// 나란히 놓기 위해서다. 값이 아니라 비교가 목적이므로 여기서 고치지 않는다.
void dumpNodes(std::ofstream& out, const wme::DetectionSet& dets, const cv::Mat& depth,
               const wme::CameraIntrinsics& K, double zmin, double zmax,
               int frame_index, double stamp) {
    constexpr double kShrink = 0.5;      // tum_fusion 의 --shrink 기본값
    constexpr double kSzCoeff = 0.006;   // 현행 sz = c*z^2 의 c
    constexpr double kBearingPx = 2.0;

    for (const auto& d : dets.items) {
        const float cxb = d.box.x + d.box.width * 0.5f;
        const float cyb = d.box.y + d.box.height * 0.5f;
        const float hw = d.box.width  * 0.5f * static_cast<float>(kShrink);
        const float hh = d.box.height * 0.5f * static_cast<float>(kShrink);
        const int x0 = std::max(0, static_cast<int>(std::floor(cxb - hw)));
        const int y0 = std::max(0, static_cast<int>(std::floor(cyb - hh)));
        const int x1 = std::min(K.width,  static_cast<int>(std::ceil(cxb + hw)));
        const int y1 = std::min(K.height, static_cast<int>(std::ceil(cyb + hh)));
        if (x1 <= x0 || y1 <= y0) continue;

        std::vector<float> vals;
        for (int y = y0; y < y1; ++y) {
            const float* row = depth.ptr<float>(y);
            for (int x = x0; x < x1; ++x) {
                const float z = row[x];
                if (std::isfinite(z) && z > static_cast<float>(zmin) &&
                    z < static_cast<float>(zmax)) vals.push_back(z);
            }
        }
        if (vals.size() < 8) continue;
        const std::size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid),
                         vals.end());
        const double z_surface = vals[mid];
        const double z = z_surface +
                         0.25 * (d.box.width * z_surface / K.fx +
                                 d.box.height * z_surface / K.fy);

        const wme::Vec3 p = K.backproject(wme::Vec2(cxb, cyb), z);
        const double sz = kSzCoeff * z * z;
        const double sx = kBearingPx * z / K.fx;
        const double sy = kBearingPx * z / K.fy;
        const double sigma = std::sqrt((sx * sx + sy * sy + sz * sz) / 3.0);

        out << frame_index << "," << std::fixed << std::setprecision(6) << stamp << ","
            << d.class_id << "," << d.confidence << ","
            << cxb << "," << cyb << "," << z_surface << "," << z << ","
            << p.x() << "," << p.y() << "," << p.z() << "," << sigma << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifndef WME_HAS_ORT
    std::cerr << "ORT 백엔드 없이 빌드되어 이 도구를 쓸 수 없다\n";
    return 1;
#else
    if (argc < 2) {
        std::cerr << "사용: wme_tcg_density <시퀀스> --yolo <model> [--kf-dist M] [--window N]\n";
        return 2;
    }
    const std::string root = argv[1];
    std::string model;
    double kf_dist = 0.15;
    int    window = 5;
    int    stride = 10;
    double conf_lo = 0.10;
    std::string csv_path, nodes_path;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--yolo") model = argv[i + 1];
        else if (k == "--kf-dist") kf_dist = std::atof(argv[i + 1]);
        else if (k == "--window") window = std::atoi(argv[i + 1]);
        else if (k == "--csv") csv_path = argv[i + 1];
        else if (k == "--nodes") nodes_path = argv[i + 1];
        else if (k == "--stride") stride = std::atoi(argv[i + 1]);
        else if (k == "--conf") conf_lo = std::atof(argv[i + 1]);
    }
    if (model.empty()) { std::cerr << "--yolo 가 필요하다\n"; return 2; }
    if (stride < 1) stride = 1;

    const auto rgb = readIndex(root + "/rgb.txt");
    const auto dep = readIndex(root + "/depth.txt");
    if (rgb.empty() || dep.empty()) { std::cerr << "인덱스를 읽지 못했다\n"; return 1; }

    // 내부파라미터/왜곡/깊이스케일/유효깊이는 시퀀스가 정한다. 경로에
    // "freiburg" 가 들어있는지로 고르던 예전 방식은 KITTI 에서 조용히 틀린다.
    wme_tools::DatasetCalib dc;
    if (!wme_tools::resolveCalib(root, dc)) return 1;
    const wme::CameraIntrinsics K = dc.K;
    const double fx = K.fx, fy = K.fy, cx = K.cx, cy = K.cy;
    const cv::Matx33d Kcv(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat m1, m2;
    cv::initUndistortRectifyMap(Kcv, dc.dist, cv::Mat(), Kcv,
                                cv::Size(K.width, K.height), CV_16SC2, m1, m2);

    // 임계를 여러 개 보려면 가장 낮은 값으로 한 번 추론하고 뒤에서 걸러야 한다.
    // 임계마다 다시 추론하면 같은 영상을 여러 번 돌리는 낭비이고, 무엇보다
    // NMS 가 임계에 따라 다르게 작동해 비교가 흐려진다.
    wme::YoloConfig yc;
    yc.model_path = model;
    yc.confidence_threshold = static_cast<float>(conf_lo);
    auto rt = wme::YoloRuntimeOrt::create(yc);
    if (!rt.ok()) { std::cerr << "YOLO 로드 실패: " << rt.error().message() << "\n"; return 1; }
    auto yolo = std::move(rt.value());

    const float THR[3] = {0.25F, 0.15F, 0.10F};
    const char* THRN[3] = {"conf>=0.25", "conf>=0.15", "conf>=0.10"};

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "stamp,raw25,raw15,raw10,depth25,depth15,depth10,cls25,win25\n";
    }

    // 노드 덤프. tum_fusion.cpp 의 nodesFromFrame 과 같은 규약으로 계산한 값을
    // 그대로 흘린다. 진리값 포즈와 맞춰 노드 위치 오차를 재는 것이 목적이므로
    // 여기서는 걸러내지 않는다 - max_nodes 절단도, sigma 정렬도 하지 않는다.
    std::ofstream ncsv;
    if (!nodes_path.empty()) {
        ncsv.open(nodes_path);
        ncsv << "frame,stamp,class_id,conf,u,v,z_surface,z,X,Y,Z,sigma_model\n";
    }

    // 창 누적: 최근 N 키프레임의 노드 수를 합친다. 상대 포즈가 필요하므로
    // 여기서는 **개수의 상한** 만 본다 - 실제 성좌는 좌표 정렬이 되어야 하고,
    // 그 비용은 이 표가 답을 준 뒤에 치를 일이다.
    std::deque<int> win_hist;

    int kf = 0;
    long long sum_raw[3] = {0, 0, 0}, sum_depth[3] = {0, 0, 0};
    long long ge4_raw[3] = {0, 0, 0}, ge4_depth[3] = {0, 0, 0};
    long long sum_cls = 0, ge4_cls = 0, ge4_win = 0, sum_win = 0;

    int last_kf_idx = -1000;
    for (std::size_t fi = 0; fi < rgb.size(); ++fi) {
        // 진리값 포즈가 없으므로 프레임 간격으로 키프레임을 흉내낸다. 이 도구는
        // 밀도만 재므로 간격 규칙의 정확성은 결론에 영향을 주지 않는다.
        // (--nodes 모드에서는 --stride 1 로 두어 인접 프레임 쌍을 만든다.)
        if (static_cast<int>(fi) - last_kf_idx < stride) continue;
        const std::string rf = root + "/" + rgb[fi].file;
        if (!fileExists(rf)) continue;
        const int di = nearest(dep, rgb[fi].stamp, 0.02);
        if (di < 0) continue;
        const std::string df = root + "/" + dep[static_cast<std::size_t>(di)].file;
        if (!fileExists(df)) continue;
        cv::Mat bgr = cv::imread(rf, cv::IMREAD_COLOR);
        cv::Mat draw = cv::imread(df, cv::IMREAD_UNCHANGED);
        if (bgr.empty() || draw.empty()) continue;
        last_kf_idx = static_cast<int>(fi);

        cv::Mat bu, d32, du;
        cv::remap(bgr, bu, m1, m2, cv::INTER_LINEAR);
        draw.convertTo(d32, CV_32F, 1.0 / dc.depth_scale);
        cv::remap(d32, du, m1, m2, cv::INTER_NEAREST);
        cv::Mat gray;
        cv::cvtColor(bu, gray, cv::COLOR_BGR2GRAY);

        wme::Frame f;
        f.id = wme::FrameId(static_cast<std::uint64_t>(fi) + 1);
        f.stamp = wme::Timestamp::fromSeconds(rgb[fi].stamp);
        f.gray = gray;
        f.depth = du;
        f.intrinsics = K;
        f.sensor = wme::SensorKind::RgbD;

        const auto det = yolo->infer(f);
        if (!det.ok()) continue;

        if (ncsv.is_open()) dumpNodes(ncsv, det.value(), du, K, dc.depth_min, dc.depth_max,
                                      static_cast<int>(fi), rgb[fi].stamp);

        int raw[3] = {0, 0, 0}, withd[3] = {0, 0, 0};
        std::set<int> classes;
        for (const auto& d : det.value().items) {
            double z;
            const bool has_depth = boxDepth(du, d, dc.depth_min, dc.depth_max, z);
            for (int t = 0; t < 3; ++t) {
                if (d.confidence < THR[t]) continue;
                ++raw[t];
                if (has_depth) ++withd[t];
            }
            if (d.confidence >= THR[0] && has_depth) classes.insert(d.class_id);
        }

        win_hist.push_back(withd[0]);
        if (static_cast<int>(win_hist.size()) > window) win_hist.pop_front();
        int win_sum = 0;
        for (int v : win_hist) win_sum += v;

        for (int t = 0; t < 3; ++t) {
            sum_raw[t] += raw[t];       ge4_raw[t]   += (raw[t] >= 4);
            sum_depth[t] += withd[t];   ge4_depth[t] += (withd[t] >= 4);
        }
        sum_cls += static_cast<long long>(classes.size());
        ge4_cls += (classes.size() >= 4);
        sum_win += win_sum;
        ge4_win += (win_sum >= 4);
        ++kf;

        if (csv.is_open()) {
            csv << std::fixed << std::setprecision(6) << rgb[fi].stamp << ","
                << raw[0] << "," << raw[1] << "," << raw[2] << ","
                << withd[0] << "," << withd[1] << "," << withd[2] << ","
                << classes.size() << "," << win_sum << "\n";
        }
        if (kf % 20 == 0) std::cout << "  키프레임 " << kf << "\n";
    }

    if (kf == 0) { std::cerr << "키프레임이 없다\n"; return 1; }
    const auto pct = [&](long long n) { return 100.0 * static_cast<double>(n) / kf; };

    std::cout << "\n" << root << "   키프레임 " << kf << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "정책별 노드 밀도 (성좌 성립 조건: min_nodes = 4)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  policy                       mean nodes   >=4 nodes\n";
    for (int t = 0; t < 3; ++t) {
        std::cout << "  " << THRN[t] << ", 깊이 무시        "
                  << static_cast<double>(sum_raw[t]) / kf << "        "
                  << pct(ge4_raw[t]) << " %\n";
    }
    for (int t = 0; t < 3; ++t) {
        std::cout << "  " << THRN[t] << ", 깊이 필요        "
                  << static_cast<double>(sum_depth[t]) / kf << "        "
                  << pct(ge4_depth[t]) << " %\n";
    }
    std::cout << "  conf>=0.25, 서로 다른 클래스만     "
              << static_cast<double>(sum_cls) / kf << "        " << pct(ge4_cls) << " %\n";
    std::cout << "  conf>=0.25 + 깊이 + " << window << "키프레임 창   "
              << static_cast<double>(sum_win) / kf << "        " << pct(ge4_win) << " %\n";
    return 0;
#endif
}
