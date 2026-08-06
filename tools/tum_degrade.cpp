// TUM 시퀀스에 물리 기반 열화를 입혀 새 시퀀스로 쓴다.
//
// 왜 필요한가. 06-results.md 1절의 헤드라인 - "융합 이득이 조건이 나빠질수록
// 커진다(6.8 % -> 53.6 %)" - 은 전부 시뮬레이션이고, 18.5 는 그것을 이렇게
// 남겨 두었다: *"기울기가 진짜인지는 미검증이며, 확인하려면 열화된 실데이터가
// 있거나, 아니면 53.6 % 가 렌더러 노이즈 모델의 성질임을 인정해야 한다."*
// TUM 에는 열화가 없다. 그래서 만든다.
//
// **핵심은 근사가 아니라는 것이다.** 대기 산란 모델
//
//     I_hazy = I * t + A * (1 - t),   t = exp(-beta * d)
//
// 의 투과율 t 는 깊이 d 를 요구한다. 렌더러는 d 를 알지만 실데이터는 보통
// 모른다 - 그런데 TUM 은 RGB-D 라서 **실측 깊이맵이 있다**. 따라서 여기서
// 투과율은 추정값이 아니라 센서가 보고한 값이고, 이 점이 렌더러로 만든 안개와
// 다른 유일하고 결정적인 차이다.
//
// 세 채널 모두 "그럴듯한 그림" 이 아니라 물리량에 걸어 둔다:
//   haze  : beta [1/m] 소산계수. 깊이 유효하지 않은 화소는 건드리지 않는다.
//   dark  : 노출 배율 g 와 그에 따른 광자 산탄잡음. 어두울수록 SNR 이 떨어지는
//           것이 어둠의 본질이지, 단순히 값이 작아지는 것이 아니다.
//   blur  : 진리값 자세에서 얻은 **실제 프레임간 운동**으로 방향성 블러를 건다.
//           임의 방향으로 뭉개면 운동블러가 아니라 그냥 흐린 그림이다.
//
// 깊이맵과 groundtruth 는 그대로 복사한다. 열화는 광학계에서 일어나는 일이고
// 깊이 센서(IR)는 별개다 - 둘 다 망가뜨리면 어느 쪽이 원인인지 못 가른다.
//
// 사용:
//   wme_tum_degrade <입력시퀀스> <출력시퀀스> [--haze B] [--dark G] [--blur K]

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Row { double stamp{0.0}; std::string file; };

std::vector<Row> readIndex(const std::string& path) {
    std::vector<Row> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Row r;
        if (ss >> r.stamp >> r.file) out.push_back(r);
    }
    return out;
}

struct Pose { double t, x, y, z, qx, qy, qz, qw; };

std::vector<Pose> readGt(const std::string& path) {
    std::vector<Pose> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Pose p;
        if (ss >> p.t >> p.x >> p.y >> p.z >> p.qx >> p.qy >> p.qz >> p.qw) out.push_back(p);
    }
    return out;
}

// 시각 t 부근의 병진 속도 [m/s]. 운동블러의 길이는 노출시간 x 속도이므로
// 방향과 크기를 둘 다 진리값에서 가져와야 의미가 있다.
bool velocityAt(const std::vector<Pose>& gt, double t, double& vx, double& vy, double& vz) {
    if (gt.size() < 2) return false;
    std::size_t i = 0;
    while (i + 1 < gt.size() && gt[i + 1].t < t) ++i;
    std::size_t j = std::min(i + 1, gt.size() - 1);
    const double dt = gt[j].t - gt[i].t;
    if (!(dt > 1e-6)) return false;
    if (std::abs(gt[i].t - t) > 0.5) return false;
    vx = (gt[j].x - gt[i].x) / dt;
    vy = (gt[j].y - gt[i].y) / dt;
    vz = (gt[j].z - gt[i].z) / dt;
    return true;
}

void copyFile(const fs::path& a, const fs::path& b) {
    fs::create_directories(b.parent_path());
    fs::copy_file(a, b, fs::copy_options::overwrite_existing);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_tum_degrade <입력> <출력> "
                     "[--haze BETA] [--dark GAIN] [--blur EXPOSURE_S] [--seed N]\n";
        return 2;
    }
    const fs::path in_root  = argv[1];
    const fs::path out_root = argv[2];

    double haze_beta = 0.0;    // [1/m] 소산계수. 0.35 면 3 m 에서 투과율 0.35
    double dark_gain = 1.0;    // 노출 배율. 0.25 = 2 스톱 어둡게
    double exposure_s  = 0.0;    // [s] 노출시간. 운동블러 길이 = v * exposure_s
    unsigned seed    = 1;

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--haze")      haze_beta = std::atof(argv[i + 1]);
        else if (k == "--dark") dark_gain = std::atof(argv[i + 1]);
        else if (k == "--blur") exposure_s  = std::atof(argv[i + 1]);
        else if (k == "--seed") seed      = static_cast<unsigned>(std::atoi(argv[i + 1]));
    }

    const auto rgb   = readIndex((in_root / "rgb.txt").string());
    const auto depth = readIndex((in_root / "depth.txt").string());
    const auto gt    = readGt((in_root / "groundtruth.txt").string());
    if (rgb.empty() || depth.empty()) {
        std::cerr << "rgb.txt / depth.txt 를 읽지 못했다: " << in_root << "\n";
        return 1;
    }

    fs::create_directories(out_root);
    // 깊이·진리값·인덱스는 그대로. 광학 열화는 깊이 센서와 무관하다.
    for (const char* f : {"rgb.txt", "depth.txt", "groundtruth.txt", "accelerometer.txt"}) {
        const fs::path src = in_root / f;
        if (fs::exists(src)) copyFile(src, out_root / f);
    }

    constexpr double kDepthScale = 5000.0;
    // 대기광 A. 실내 안개/연무는 장면보다 밝은 회백색으로 수렴한다.
    const cv::Scalar A(210, 212, 214);

    std::mt19937 rng(seed);
    int n = 0, blurred = 0;

    for (const auto& r : rgb) {
        const fs::path src = in_root / r.file;
        const fs::path dst = out_root / r.file;
        cv::Mat img = cv::imread(src.string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;
        fs::create_directories(dst.parent_path());

        cv::Mat f;
        img.convertTo(f, CV_32FC3);

        // --- 1. haze: 투과율은 실측 깊이에서 온다 -------------------------
        if (haze_beta > 0.0) {
            // 같은 시각의 깊이 프레임을 찾는다. 없으면 이 프레임은 건너뛴다 -
            // 깊이 없는 화소에 임의 투과율을 넣으면 그 순간 이 실험은
            // "렌더러 안개" 로 되돌아간다.
            int best = -1; double bd = 0.02;
            for (std::size_t i = 0; i < depth.size(); ++i) {
                const double d = std::abs(depth[i].stamp - r.stamp);
                if (d <= bd) { bd = d; best = static_cast<int>(i); }
            }
            if (best >= 0) {
                cv::Mat dep = cv::imread((in_root / depth[static_cast<std::size_t>(best)].file).string(),
                                         cv::IMREAD_UNCHANGED);
                if (!dep.empty() && dep.size() == img.size()) {
                    for (int y = 0; y < f.rows; ++y) {
                        auto* pf = f.ptr<cv::Vec3f>(y);
                        const auto* pd = dep.ptr<std::uint16_t>(y);
                        for (int x = 0; x < f.cols; ++x) {
                            const double z = pd[x] / kDepthScale;
                            // 깊이 무효(0). 값을 지어내지 않고 원본을 남긴다.
                            if (!(z > 0.0)) continue;
                            const double t = std::exp(-haze_beta * z);
                            for (int c = 0; c < 3; ++c)
                                pf[x][c] = static_cast<float>(pf[x][c] * t + A[c] * (1.0 - t));
                        }
                    }
                }
            }
        }

        // --- 2. motion blur: 방향과 길이를 진리값 운동에서 --------------
        if (exposure_s > 0.0) {
            double vx, vy, vz;
            if (velocityAt(gt, r.stamp, vx, vy, vz)) {
                // 화면상 변위는 대략 초점거리 x 횡방향속도 / 깊이. 장면 대표
                // 깊이를 2.5 m 로 두고 커널 길이를 화소로 환산한다.
                constexpr double kF = 525.0, kZ = 2.5;
                const double px = kF * vx * exposure_s / kZ;
                const double py = kF * vy * exposure_s / kZ;
                const int len = static_cast<int>(std::lround(std::hypot(px, py)));
                if (len >= 2) {
                    const int L = std::min(len | 1, 41);       // 홀수, 상한
                    cv::Mat k = cv::Mat::zeros(L, L, CV_32F);
                    const double ang = std::atan2(py, px);
                    const double cx = (L - 1) / 2.0, cy = (L - 1) / 2.0;
                    for (int i = 0; i < L; ++i) {
                        const double s = i - cx;
                        const int xx = static_cast<int>(std::lround(cx + s * std::cos(ang)));
                        const int yy = static_cast<int>(std::lround(cy + s * std::sin(ang)));
                        if (xx >= 0 && xx < L && yy >= 0 && yy < L) k.at<float>(yy, xx) += 1.0F;
                    }
                    k /= cv::sum(k)[0];
                    cv::filter2D(f, f, -1, k, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
                    ++blurred;
                }
            }
        }

        // --- 3. darkness: 노출 배율 + 그에 맞는 산탄잡음 -----------------
        if (dark_gain != 1.0) {
            // 광자 수는 노출에 비례하고 산탄잡음은 그 제곱근이다. 신호를 g 배
            // 줄이면 SNR 은 sqrt(g) 배로 떨어진다 - 어둠이 정합을 깨는 진짜
            // 이유는 값이 작아서가 아니라 이 SNR 저하다.
            const double scale = 255.0;
            for (int y = 0; y < f.rows; ++y) {
                auto* pf = f.ptr<cv::Vec3f>(y);
                for (int x = 0; x < f.cols; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const double lin = std::max(0.0, pf[x][c] / scale) * dark_gain;
                        // 전자수 상한 10000 e- 을 가정한 산탄잡음
                        const double e = lin * 10000.0;
                        std::poisson_distribution<int> pd(std::max(1e-6, e));
                        const double noisy = pd(rng) / 10000.0;
                        // 게인 보정으로 되돌린다(카메라 AGC). 신호는 복원되지만
                        // 잡음은 증폭된 채 남는다.
                        pf[x][c] = static_cast<float>(
                            std::clamp(noisy / dark_gain * scale, 0.0, 255.0));
                    }
                }
            }
        }

        cv::Mat out8;
        f.convertTo(out8, CV_8UC3);
        cv::imwrite(dst.string(), out8);
        ++n;
        if (n % 100 == 0) std::cout << "  " << n << " 프레임\n";
    }

    // 깊이는 원본 그대로 복사
    int dn = 0;
    for (const auto& d : depth) {
        const fs::path src = in_root / d.file;
        if (fs::exists(src)) { copyFile(src, out_root / d.file); ++dn; }
    }

    std::cout << "열화 완료: rgb " << n << "  depth " << dn
              << "  (haze beta=" << haze_beta << ", dark gain=" << dark_gain
              << ", exposure_s=" << exposure_s << "s, blurred " << blurred << ")\n"
              << "출력: " << out_root << "\n";
    return 0;
}
