// 시퀀스에서 **의미 구조** 를 뽑아 뷰어가 읽을 수 있는 파일로 내보낸다.
//
// 왜 뷰어 안에서 하지 않는가
// -------------------------
// 뷰어는 이미 "지표를 다시 계산하지 않는다" 는 규약을 지킨다. 같은 이유로
// 검출과 평면 적합도 여기서 한 번만 하고 결과를 넘긴다. 뷰어가 YOLO 를 돌리면
// 재생 속도가 모델 추론 속도에 묶이고(CPU 60~150 ms/frame), 무엇보다 화면에
// 보이는 상자가 벤치마크가 실제로 쓴 상자와 다를 수 있다.
//
// 무엇을 내보내는가
//   PLANE  Tier 2(SPA)가 쓰는 지배 평면. 건물 벽/지면/책상 상판이 여기 잡힌다.
//   BOX    Tier 1(TCG)이 노드로 쓰는 물체. KITTI 에서는 주로 차량이다.
//
// 좌표계는 **카메라 좌표계** 다. 뷰어가 각 시스템의 추정 포즈로 옮긴다 -
// 그래야 같은 관측이 두 시스템에서 어디로 가는지가 곧 드리프트로 보인다.
//
// 사용:
//   wme_scene_export <시퀀스경로> <출력.tsv> [--stride N] [--yolo <onnx>]
//                    [--max-frames N] [--conf 0.35]

#include "dataset_calib.hpp"

#include "wme/geometry/PlaneExtractor.hpp"
#include "wme/perception/Detection.hpp"
#ifdef WME_HAS_ORT
#include "wme/perception/YoloRuntimeOrt.hpp"
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct IndexRow {
    double      stamp{0.0};
    std::string file;
};

std::vector<IndexRow> readIndex(const std::string& path) {
    std::vector<IndexRow> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        IndexRow r;
        if (ss >> r.stamp >> r.file) out.push_back(r);
    }
    return out;
}

int nearest(const std::vector<IndexRow>& rows, double stamp, double tol) {
    int best = -1;
    double bd = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

// 검출 상자 안의 깊이 중앙값으로 3D 상자를 만든다.
//
// 왜 중앙값인가: 상자 안에는 물체와 배경이 섞여 있다. 평균을 쓰면 뒤쪽 배경이
// 물체를 뒤로 끌고 가고, 최소값을 쓰면 앞을 지나는 잡음 한 점이 물체를 앞으로
// 끌어온다. 중앙값은 상자 면적의 절반 이상을 물체가 차지할 때 옳고, 검출기가
// 제 일을 했다면 그 조건은 성립한다.
bool boxTo3D(const cv::Mat& depth_m, const wme::CameraIntrinsics& K,
             const cv::Rect2f& box, double dmin, double dmax,
             wme::Vec3& center, wme::Vec3& size) {
    cv::Rect r(static_cast<int>(box.x), static_cast<int>(box.y),
               static_cast<int>(box.width), static_cast<int>(box.height));
    r &= cv::Rect(0, 0, depth_m.cols, depth_m.rows);
    if (r.width < 4 || r.height < 4) return false;

    std::vector<float> z;
    z.reserve(static_cast<std::size_t>(r.area()) / 4);
    for (int v = r.y; v < r.y + r.height; v += 2) {
        const float* row = depth_m.ptr<float>(v);
        for (int u = r.x; u < r.x + r.width; u += 2) {
            const float d = row[u];
            if (d > dmin && d < dmax) z.push_back(d);
        }
    }
    if (z.size() < 24) return false;

    std::nth_element(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(z.size() / 2), z.end());
    const double zc = z[z.size() / 2];

    // 상자 네 모서리를 그 깊이로 역투영하면 물리 크기가 나온다.
    const double x0 = (r.x - K.cx) * zc / K.fx;
    const double x1 = (r.x + r.width - K.cx) * zc / K.fx;
    const double y0 = (r.y - K.cy) * zc / K.fy;
    const double y1 = (r.y + r.height - K.cy) * zc / K.fy;

    center = wme::Vec3(0.5 * (x0 + x1), 0.5 * (y0 + y1), zc);

    // 깊이 방향 두께는 관측되지 않는다. 상자 안 깊이의 사분위 범위를 쓰되
    // 가로 폭보다 작아지지 않게 한다 - 0 으로 두면 판이 되어 버린다.
    std::vector<float> zs = z;
    std::sort(zs.begin(), zs.end());
    const double q1 = zs[zs.size() / 4];
    const double q3 = zs[3 * zs.size() / 4];
    const double thick = std::max(0.4 * (x1 - x0), std::min(6.0, 2.0 * (q3 - q1)));
    size = wme::Vec3(x1 - x0, y1 - y0, thick);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_scene_export <시퀀스경로> <출력.tsv> "
                     "[--stride N] [--yolo <onnx>] [--max-frames N] [--conf C]\n";
        return 2;
    }
    const std::string root = argv[1];
    const std::string out_path = argv[2];

    int stride = 4;
    int max_frames = 0;
    double conf = 0.35;
    std::string yolo_model;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--stride")           stride = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--max-frames")  max_frames = std::atoi(argv[i + 1]);
        else if (k == "--yolo")        yolo_model = argv[i + 1];
        else if (k == "--conf")        conf = std::atof(argv[i + 1]);
    }

    const auto rgb_rows   = readIndex(root + "/rgb.txt");
    const auto depth_rows = readIndex(root + "/depth.txt");
    if (rgb_rows.empty() || depth_rows.empty()) {
        std::cerr << "rgb.txt / depth.txt 를 읽지 못했다: " << root << "\n";
        return 1;
    }

    wme_tools::DatasetCalib cal;
    if (!wme_tools::resolveCalib(root, cal)) return 1;

    // 평면 탐색 범위는 데이터셋 깊이 범위를 따른다. KITTI 는 15 m 로 잘리면
    // 건물 벽이 전부 빠진다.
    wme::PlaneExtractorConfig pcfg;
    pcfg.min_depth = cal.depth_min;
    pcfg.max_depth = std::min(cal.depth_max, 45.0);
    pcfg.stride = 4;
    pcfg.max_planes = 6;
    wme::PlaneExtractor planes(pcfg);

    std::unique_ptr<wme::IYoloRuntime> yolo;
    if (!yolo_model.empty()) {
#ifdef WME_HAS_ORT
        wme::YoloConfig yc;
        yc.model_path = yolo_model;
        yc.confidence_threshold = static_cast<float>(conf);
        auto r = wme::YoloRuntimeOrt::create(yc);
        if (!r.ok()) {
            // 조용히 끄지 않는다. "상자가 없다" 와 "검출기를 못 열었다" 는
            // 화면에서 구분되지 않으므로 여기서 죽는 편이 낫다.
            std::cerr << "YOLO 로드 실패: " << r.error().message() << "\n";
            return 1;
        }
        yolo = std::move(r.value());
        std::cout << "YOLO ON: " << yolo_model << " (conf " << conf << ")\n";
#else
        std::cerr << "이 빌드에는 ONNX Runtime 이 없다 - --yolo 를 쓸 수 없다\n";
        return 1;
#endif
    }

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "출력 파일을 열 수 없다: " << out_path << "\n";
        return 1;
    }
    out << "# wme_scene_export  " << root << "\n"
        << "# 좌표계는 카메라 좌표계. 뷰어가 각 시스템의 추정 포즈로 옮긴다.\n"
        << "# PLANE\tframe\tnx\tny\tnz\td\tcx\tcy\tcz\textent\trms\tinliers\tconf\n"
        << "# BOX\tframe\tclass\tconf\tcx\tcy\tcz\tsx\tsy\tsz\n";
    out << std::fixed << std::setprecision(5);

    int written_planes = 0, written_boxes = 0, used = 0;
    for (std::size_t i = 0; i < rgb_rows.size(); i += static_cast<std::size_t>(stride)) {
        const int di = nearest(depth_rows, rgb_rows[i].stamp, 0.05);
        if (di < 0) continue;

        cv::Mat draw = cv::imread(root + "/" + rgb_rows[i].file, cv::IMREAD_COLOR);
        cv::Mat d16 = cv::imread(root + "/" + depth_rows[static_cast<std::size_t>(di)].file,
                                 cv::IMREAD_UNCHANGED);
        if (d16.empty() || d16.type() != CV_16U) continue;

        cv::Mat depth_m;
        d16.convertTo(depth_m, CV_32F, 1.0 / cal.depth_scale);

        // --- 평면 (건물 벽 / 지면 / 상판) ---
        const auto pr = planes.extract(depth_m, cal.K);
        if (pr.ok()) {
            for (const auto& p : pr.value()) {
                out << "PLANE\t" << i << "\t"
                    << p.normal.x() << "\t" << p.normal.y() << "\t" << p.normal.z() << "\t"
                    << p.distance << "\t"
                    << p.centroid.x() << "\t" << p.centroid.y() << "\t" << p.centroid.z() << "\t"
                    << p.extent << "\t" << p.rms << "\t" << p.inliers << "\t"
                    << p.confidence() << "\n";
                ++written_planes;
            }
        }

        // --- 물체 상자 (차량 / 사람 …) ---
        if (yolo && !draw.empty()) {
            wme::Frame f;
            f.rgb = draw;
            cv::cvtColor(draw, f.gray, cv::COLOR_BGR2GRAY);
            f.intrinsics = cal.K;
            f.stamp = wme::Timestamp::fromSeconds(rgb_rows[i].stamp);
            const auto dr = yolo->infer(f);
            if (dr.ok()) {
                for (const auto& det : dr.value().items) {
                    wme::Vec3 c, s;
                    if (!boxTo3D(depth_m, cal.K, det.box, cal.depth_min, cal.depth_max, c, s)) {
                        continue;
                    }
                    out << "BOX\t" << i << "\t" << det.class_name << "\t" << det.confidence
                        << "\t" << c.x() << "\t" << c.y() << "\t" << c.z()
                        << "\t" << s.x() << "\t" << s.y() << "\t" << s.z() << "\n";
                    ++written_boxes;
                }
            }
        }

        ++used;
        if (used % 25 == 0) {
            std::cout << "  " << used << " 프레임  평면 " << written_planes
                      << "  상자 " << written_boxes << "\n" << std::flush;
        }
        if (max_frames > 0 && used >= max_frames) break;
    }

    out.flush();
    if (!out) {
        std::cerr << "출력 쓰기 실패: " << out_path << "\n";
        return 1;
    }
    std::cout << "저장: " << out_path << "  프레임 " << used
              << " (stride " << stride << ")  평면 " << written_planes
              << "  상자 " << written_boxes << "\n";
    if (written_planes == 0 && written_boxes == 0) {
        std::cerr << "아무 것도 뽑히지 않았다 - 깊이 범위나 모델 경로를 확인하라\n";
        return 1;
    }
    return 0;
}
