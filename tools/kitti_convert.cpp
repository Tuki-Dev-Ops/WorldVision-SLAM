// KITTI odometry 시퀀스를 TUM RGB-D 배치로 변환한다.
//
// 왜 변환인가, 왜 전용 러너가 아닌가
// ---------------------------------
// wme_tum_odometry / wme_tum_baseline / bench_run.py / bench_report.py 는 전부
// TUM 배치(rgb.txt, depth.txt, groundtruth.txt) 위에 서 있고, 그 위에서 §22 의
// 12 시퀀스 비교와 §24 의 루프 클로저가 검증되어 있다. KITTI 전용 러너를 새로
// 쓰면 그 검증된 경로를 통째로 복제하게 되고, 두 경로가 어긋나는 순간 그
// 차이가 데이터셋 차이로 오독된다. 대신 데이터를 맞춘다.
//
// 만드는 것:
//   <out>/rgb/<idx>.png           image_0 (왼쪽 그레이) 그대로
//   <out>/depth/<idx>.png         StereoSGBM 깊이. uint16, 값 = m * depth_scale
//   <out>/rgb.txt, depth.txt      times.txt 의 시각
//   <out>/groundtruth.txt         poses/<seq>.txt -> TUM 형식 (tx ty tz qx qy qz qw)
//   <out>/calib.txt               fx fy cx cy width height depth_scale (dataset_calib.hpp)
//
// 깊이는 **추정값** 이다. 무효 화소는 0 으로 둔다 - TUM 의 깊이 결측과 같은
// 표기라, 하류가 이미 그것을 다룰 줄 안다.
//
// 사용:
//   wme_kitti_convert <kitti_dataset_root> <시퀀스번호> <출력경로>
//                     [--max-frames N] [--stride N] [--min-depth M]

#include "wme/perception/StereoDepth.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Calib {
    double fx{0}, fy{0}, cx{0}, cy{0};
    double baseline_m{0};
    bool   ok{false};
};

// calib.txt 의 P0(왼쪽), P1(오른쪽) 에서 f, c, 베이스라인을 뽑는다.
// P1 = K [I | t] 이고 t_x = -f*B 이므로 B = -P1[0,3]/f.
Calib readKittiCalib(const fs::path& p) {
    Calib c;
    std::ifstream f(p);
    if (!f) return c;
    std::string line;
    double P0[12]{}, P1[12]{};
    bool has0 = false, has1 = false;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "P0:") { for (double& v : P0) ss >> v; has0 = true; }
        else if (tag == "P1:") { for (double& v : P1) ss >> v; has1 = true; }
    }
    if (!has0 || !has1) return c;
    c.fx = P0[0]; c.fy = P0[5]; c.cx = P0[2]; c.cy = P0[6];
    if (c.fx <= 0.0) return c;
    c.baseline_m = -P1[3] / c.fx;
    c.ok = c.baseline_m > 0.0;
    return c;
}

void toQuat(const double R[9], double& qx, double& qy, double& qz, double& qw) {
    const double tr = R[0] + R[4] + R[8];
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (R[7] - R[5]) / s;
        qy = (R[2] - R[6]) / s;
        qz = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
        qw = (R[7] - R[5]) / s; qx = 0.25 * s;
        qy = (R[1] + R[3]) / s; qz = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
        qw = (R[2] - R[6]) / s; qx = (R[1] + R[3]) / s;
        qy = 0.25 * s;          qz = (R[5] + R[7]) / s;
    } else {
        const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
        qw = (R[3] - R[1]) / s; qx = (R[2] + R[6]) / s;
        qy = (R[5] + R[7]) / s; qz = 0.25 * s;
    }
}

std::string pad6(int i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d", i);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "사용: wme_kitti_convert <kitti_dataset_root> <시퀀스(예 00)> "
                     "<출력경로> [--max-frames N] [--stride N] [--min-depth M]\n"
                     "  kitti_dataset_root 는 sequences/ 와 poses/ 를 담은 폴더\n";
        return 2;
    }
    const fs::path root = argv[1];
    const std::string seq = argv[2];
    const fs::path out = argv[3];

    int max_frames = 0;
    int stride = 1;
    // KITTI 도로 장면의 최근접 거리. 이 값이 시차 탐색 범위를 정한다 -
    // 모자라면 SGBM 은 "범위 밖" 이라고 하지 않고 그럴듯한 오답을 준다
    // (06-results.md 25.19). 실측 근거가 있는 값이라 기본을 4 m 로 둔다.
    double min_depth = 4.0;

    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--max-frames")     max_frames = std::atoi(argv[i + 1]);
        else if (k == "--stride")    stride = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--min-depth") min_depth = std::atof(argv[i + 1]);
    }

    const fs::path sdir = root / "sequences" / seq;
    const Calib cal = readKittiCalib(sdir / "calib.txt");
    if (!cal.ok) {
        std::cerr << "calib.txt 에서 P0/P1 을 읽지 못했다: " << (sdir / "calib.txt") << "\n";
        return 1;
    }

    std::vector<double> times;
    {
        std::ifstream tf(sdir / "times.txt");
        if (!tf) { std::cerr << "times.txt 없음: " << sdir << "\n"; return 1; }
        double t;
        while (tf >> t) times.push_back(t);
    }
    if (times.empty()) { std::cerr << "times.txt 가 비었다\n"; return 1; }

    // 정답 궤적. 00~10 만 공개되어 있다. 없으면 groundtruth.txt 를 만들지 않고
    // 그 사실을 찍는다 - 빈 파일을 남기면 하류가 "정답이 있다" 고 믿는다.
    std::vector<std::array<double, 12>> poses;
    {
        std::ifstream pf(root / "poses" / (seq + ".txt"));
        std::string line;
        while (pf && std::getline(pf, line)) {
            std::istringstream ss(line);
            std::array<double, 12> P{};
            bool ok = true;
            for (double& v : P) if (!(ss >> v)) { ok = false; break; }
            if (ok) poses.push_back(P);
        }
    }
    const bool has_gt = !poses.empty();

    // 시차 범위를 장면에서 정한다. 25.19 참조.
    wme::StereoDepthConfig scfg;
    scfg.focal_px = cal.fx;
    scfg.baseline_m = cal.baseline_m;
    scfg.num_disparities =
        wme::StereoDepth::requiredDisparities(cal.fx, cal.baseline_m, min_depth);
    scfg.block_size = 5;
    scfg.min_depth_m = min_depth * 0.8;
    scfg.max_depth_m = 80.0;
    scfg.max_depth_sigma_m = 3.0;
    wme::StereoDepth stereo(scfg);

    // uint16 PNG 에 80 m 까지 담아야 한다. 80 * 256 = 20480 < 65535.
    constexpr double kDepthScale = 256.0;

    fs::create_directories(out / "rgb");
    fs::create_directories(out / "depth");

    std::ofstream rgb_idx(out / "rgb.txt");
    std::ofstream dep_idx(out / "depth.txt");
    if (!rgb_idx || !dep_idx) {
        std::cerr << "인덱스 파일을 열 수 없다: " << out << "\n";
        return 1;
    }
    rgb_idx << "# timestamp filename\n" << std::fixed << std::setprecision(6);
    dep_idx << "# timestamp filename\n" << std::fixed << std::setprecision(6);

    std::ofstream gt;
    if (has_gt) {
        gt.open(out / "groundtruth.txt");
        if (!gt) { std::cerr << "groundtruth.txt 를 열 수 없다\n"; return 1; }
        gt << "# timestamp tx ty tz qx qy qz qw\n" << std::fixed << std::setprecision(7);
    } else {
        std::cout << "정답 궤적 없음 (시퀀스 " << seq << "). groundtruth.txt 를 만들지 않는다.\n";
    }

    {
        std::ofstream cf(out / "calib.txt");
        if (!cf) { std::cerr << "calib.txt 를 쓸 수 없다\n"; return 1; }
        // 폭/높이는 첫 영상에서 확정한다. 아래에서 다시 쓴다.
        cf << "# wme_kitti_convert 가 생성\n";
    }

    const int n_total = static_cast<int>(times.size());
    int written = 0, w = 0, h = 0;
    double sum_valid = 0.0, sum_clip = 0.0;

    for (int i = 0; i < n_total; i += stride) {
        const std::string name = pad6(i);
        const fs::path lp = sdir / "image_0" / (name + ".png");
        const fs::path rp = sdir / "image_1" / (name + ".png");
        cv::Mat left = cv::imread(lp.string(), cv::IMREAD_GRAYSCALE);
        cv::Mat right = cv::imread(rp.string(), cv::IMREAD_GRAYSCALE);
        if (left.empty() || right.empty()) {
            std::cerr << "영상을 읽지 못했다: " << lp << "\n";
            return 1;
        }
        if (w == 0) { w = left.cols; h = left.rows; }

        const wme::StereoDepthResult sr = stereo.compute(left, right);
        sum_valid += sr.valid_ratio;
        sum_clip += sr.clipped_ratio;

        cv::Mat d16(sr.depth.size(), CV_16U);
        for (int y = 0; y < sr.depth.rows; ++y) {
            const float* zp = sr.depth.ptr<float>(y);
            std::uint16_t* op = d16.ptr<std::uint16_t>(y);
            for (int x = 0; x < sr.depth.cols; ++x) {
                const double v = zp[x] * kDepthScale;
                op[x] = (zp[x] > 0.0f && v < 65535.0)
                            ? static_cast<std::uint16_t>(v + 0.5) : 0;
            }
        }

        const std::string rel_rgb = "rgb/" + name + ".png";
        const std::string rel_dep = "depth/" + name + ".png";
        if (!cv::imwrite((out / rel_rgb).string(), left) ||
            !cv::imwrite((out / rel_dep).string(), d16)) {
            std::cerr << "PNG 쓰기 실패: " << name << "\n";
            return 1;
        }

        const double t = times[static_cast<std::size_t>(i)];
        rgb_idx << t << " " << rel_rgb << "\n";
        dep_idx << t << " " << rel_dep << "\n";

        if (has_gt && i < static_cast<int>(poses.size())) {
            const auto& P = poses[static_cast<std::size_t>(i)];
            const double R[9] = {P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]};
            double qx, qy, qz, qw;
            toQuat(R, qx, qy, qz, qw);
            gt << t << " " << P[3] << " " << P[7] << " " << P[11] << " "
               << qx << " " << qy << " " << qz << " " << qw << "\n";
        }

        ++written;
        if (written % 100 == 0) {
            std::cout << "  " << written << " 프레임  유효깊이 "
                      << (100.0 * sum_valid / written) << "%\n" << std::flush;
        }
        if (max_frames > 0 && written >= max_frames) break;
    }

    {
        std::ofstream cf(out / "calib.txt");
        cf << "# wme_kitti_convert: KITTI 시퀀스 " << seq << "\n"
           << "# 깊이는 StereoSGBM 추정값이다. 측정값이 아니다.\n"
           << "fx: " << std::setprecision(9) << cal.fx << "\n"
           << "fy: " << cal.fy << "\n"
           << "cx: " << cal.cx << "\n"
           << "cy: " << cal.cy << "\n"
           << "width: " << w << "\n"
           << "height: " << h << "\n"
           << "depth_scale: " << kDepthScale << "\n"
           << "depth_min: " << scfg.min_depth_m << "\n"
           << "depth_max: " << scfg.max_depth_m << "\n"
           << "dist: 0 0 0 0 0\n";
        if (!cf) { std::cerr << "calib.txt 쓰기 실패\n"; return 1; }
    }

    std::cout << "\n변환 완료: " << out << "\n"
              << "  프레임 " << written << " / " << n_total
              << " (stride " << stride << ")\n"
              << "  f=" << cal.fx << "  B=" << cal.baseline_m << " m"
              << "  시차범위 " << scfg.num_disparities
              << " (최근접 " << (cal.fx * cal.baseline_m /
                                 (scfg.num_disparities - 1)) << " m)\n"
              << "  평균 유효깊이 " << (100.0 * sum_valid / std::max(1, written)) << "%"
              << "  상한클리핑 " << (100.0 * sum_clip / std::max(1, written)) << "%\n"
              << "  정답궤적 " << (has_gt ? "있음" : "없음") << "\n";
    if (sum_clip / std::max(1, written) > 0.02) {
        std::cout << "  경고: 상한 클리핑이 2% 를 넘는다. --min-depth 를 낮춰라.\n";
    }
    return 0;
}
