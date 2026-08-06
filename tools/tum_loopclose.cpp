// 루프 후보를 찾고 기하 검증해 포즈그래프 간선으로 내보낸다.
//
// 왜 필요한가. 06-results.md 22 의 비교는 프런트엔드 대 프런트엔드다. 진짜
// ORB-SLAM3 에는 루프 클로저와 번들 조정이 있고, 그것이 없는 비교는 "오도메트리
// 끼리" 로만 유효하다고 그 절에 적어 두었다. 이 도구가 그 공백을 메운다.
//
// **설계의 핵심은 대칭이다.** 두 방식이 공유하는 것:
//   - 후보 제안 규칙 (시간상 멀고 현재 추정상 가까운 키프레임 쌍)
//   - 키프레임 선택 규칙
//   - 뒤에 붙는 포즈그래프 (python/wme/graph 의 FactorGraph)
// 다른 것은 오직 **무엇으로 그 쌍을 검증하는가** 뿐이다:
//   --mode orb : ORB 기술자 정합 + RANSAC PnP   (고전 파이프라인)
//   --mode tcg : 객체 성좌 정합 + Kabsch        (WME 의 Tier 1)
// 그래야 최종 ATE 차이를 "인식 방법의 차이" 로 읽을 수 있다. 백엔드가 다르면
// 그 차이가 무엇 때문인지 말할 수 없다.
//
// 이 도구는 판정하지 않는다. 간선과 그 근거(인라이어 수, 점수)를 CSV 로 남기고
// 최적화와 채점은 python 이 한다.
//
// 사용:
//   wme_tum_loopclose <시퀀스> <오도메트리.txt> <출력edges.csv>
//       [--mode orb|tcg] [--kf-dist M] [--min-gap S] [--radius M] [--yolo M]

#include "wme/core/SE3.hpp"
#include "wme/core/Frame.hpp"
#include "wme/token/ConstellationIndex.hpp"
#include "wme/perception/Detection.hpp"
#ifdef WME_HAS_ORT
#include "wme/perception/YoloRuntimeOrt.hpp"
#endif

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row { double stamp{0.0}; std::string file; };

std::vector<Row> readIndex(const std::string& p) {
    std::vector<Row> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Row r;
        if (ss >> r.stamp >> r.file) out.push_back(r);
    }
    return out;
}

struct TrajRow { double stamp; wme::SE3 T; };

// 가속도계. 카이랄리티에 쓸 카메라 좌표계 중력축이 여기서 나온다.
//
// **이걸 빼먹으면 성좌 서명의 세 성분 중 하나가 통째로 꺼진다** - 02 절의
// 서명은 "클래스 다중집합 + 로그빈 쌍거리 스펙트럼 + **카이랄리티**" 다.
// 처음 이 도구를 쓸 때 gravity 를 넘기지 않았고, 그 상태로 잰 오탐률을
// "서명이 장소를 구별하지 못한다" 로 읽을 뻔했다. 19.3 이 기록한 함정을
// 같은 서브시스템에서 두 번째로 밟은 것이다.
struct AccelRow { double stamp{0.0}; wme::Vec3 a{wme::Vec3::Zero()}; };

std::vector<AccelRow> readAccel(const std::string& path) {
    std::vector<AccelRow> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t, ax, ay, az;
        if (!(ss >> t >> ax >> ay >> az)) continue;
        out.push_back({t, wme::Vec3(ax, ay, az)});
    }
    return out;
}

bool accelAt(const std::vector<AccelRow>& rows, double stamp, double half_window,
             wme::Vec3& out) {
    wme::Vec3 sum = wme::Vec3::Zero();
    int n = 0;
    for (const auto& r : rows) {
        if (std::abs(r.stamp - stamp) > half_window) continue;
        sum += r.a;
        ++n;
    }
    if (n == 0) return false;
    out = sum / static_cast<double>(n);
    return out.norm() > 1e-6;
}

wme::Mat3 quatToR(double x, double y, double z, double w) {
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    x /= n; y /= n; z /= n; w /= n;
    wme::Mat3 R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
         2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y);
    return R;
}

std::vector<TrajRow> readTraj(const std::string& p) {
    std::vector<TrajRow> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t, x, y, z, qx, qy, qz, qw;
        if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
        out.push_back({t, wme::SE3(wme::SO3(quatToR(qx, qy, qz, qw)), wme::Vec3(x, y, z))});
    }
    return out;
}

int nearest(const std::vector<Row>& rows, double stamp, double tol) {
    int best = -1; double bd = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

void toQuat(const wme::Mat3& R, double& qx, double& qy, double& qz, double& qw) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s; qx = (R(2, 1) - R(1, 2)) / s;
        qy = (R(0, 2) - R(2, 0)) / s; qz = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = std::sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        qw = (R(2, 1) - R(1, 2)) / s; qx = 0.25 * s;
        qy = (R(0, 1) + R(1, 0)) / s; qz = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        const double s = std::sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        qw = (R(0, 2) - R(2, 0)) / s; qx = (R(0, 1) + R(1, 0)) / s;
        qy = 0.25 * s; qz = (R(1, 2) + R(2, 1)) / s;
    } else {
        const double s = std::sqrt(1 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        qw = (R(1, 0) - R(0, 1)) / s; qx = (R(0, 2) + R(2, 0)) / s;
        qy = (R(1, 2) + R(2, 1)) / s; qz = 0.25 * s;
    }
}

// 키프레임 하나가 들고 있어야 하는 것
struct KeyFrame {
    int    idx{0};
    double stamp{0.0};
    wme::SE3 T_world;                  // 오도메트리 추정
    cv::Mat gray, depth;
    cv::Mat desc;
    std::vector<cv::KeyPoint> kp;
    std::vector<cv::Point3f>  xyz;
    std::vector<bool>         has;
    std::vector<wme::ConstellationNode> nodes;   // tcg 모드
    std::optional<wme::Vec3> gravity;            // 카메라 좌표계 중력축
};

// 최근 W 개 키프레임의 노드를 마지막 키프레임 좌표계로 모아 하나의 성좌로 만든다.
//
// 25.10 이 잰 것은 창 안 노드 수의 **합** 이었고, 그것은 상한이다. 같은 의자를
// 다섯 프레임에서 보면 노드 다섯 개가 아니라 하나다. 합치지 않으면 거리
// 스펙트럼이 0 에 가까운 간격으로 가득 차 성좌 서명이 오히려 망가진다.
// 여기서 실제로 합치고, 병합 뒤의 개수를 호출자가 다시 센다.
std::vector<wme::ConstellationNode> mergeWindow(
    const std::vector<const KeyFrame*>& win, double merge_dist) {
    std::vector<wme::ConstellationNode> out;
    if (win.empty()) return out;
    const wme::SE3 T_world_anchor = win.back()->T_world;

    for (const KeyFrame* kf : win) {
        // 오도메트리의 상대 포즈로 앵커 좌표계에 옮긴다. 창이 짧으므로
        // 그 구간의 드리프트는 작고, 진리값을 쓰지 않는다는 점이 중요하다.
        const wme::SE3 T_anchor_kf = T_world_anchor.inverse() * kf->T_world;
        for (const auto& n : kf->nodes) {
            wme::ConstellationNode m = n;
            m.position = T_anchor_kf * n.position;

            bool merged = false;
            for (auto& e : out) {
                if (e.class_id != m.class_id) continue;
                if ((e.position - m.position).norm() > merge_dist) continue;
                // 같은 물체로 본다. 위치는 평균으로 둔다 - 마지막 관측만 쓰면
                // 검출 깜빡임이 그대로 노드 위치의 잡음이 된다.
                e.position = 0.5 * (e.position + m.position);
                merged = true;
                break;
            }
            if (!merged) {
                m.id = wme::TokenId(out.size());
                out.push_back(m);
            }
        }
    }
    // id 는 성좌 안에서 유일해야 한다. 병합으로 순서가 흔들렸을 수 있으므로
    // 마지막에 다시 매긴다.
    for (std::size_t i = 0; i < out.size(); ++i) out[i].id = wme::TokenId(i);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "사용: wme_tum_loopclose <시퀀스> <오도메트리.txt> <edges.csv> "
                     "[--mode orb|tcg] [--kf-dist M] [--min-gap S] [--radius M] "
                     "[--yolo MODEL]\n";
        return 2;
    }
    const std::string root = argv[1], traj_path = argv[2], out_path = argv[3];

    std::string mode = "orb";
    double kf_dist = 0.15;      // 루프용 키프레임은 오도메트리보다 성기게
    double min_gap = 3.0;       // [s] 이보다 가까운 시각끼리는 루프가 아니다
    double radius  = 1.0;       // [m] 현재 추정에서 이 안이면 후보
    std::string yolo_model;
    int    min_inliers = 25;    // orb 채택 문턱
    // 성좌 질의를 몇 개의 키프레임으로 만들 것인가. 1 = 단일 프레임(24절이
    // 잰 구성). 25.10 은 5 로 두면 "노드 4 개 이상" 비율이 34 % -> 97.8 % 로
    // 오른다고 쟀지만, 그것은 **중복 제거 전** 합계였다. 여기서는 실제로 합치고
    // 그 뒤의 개수를 다시 센다 - 같은 의자를 다섯 번 센 것은 노드 다섯 개가
    // 아니라 하나이며, 거리 스펙트럼에 0 에 가까운 간격을 잔뜩 만들어 성좌를
    // 오히려 망가뜨린다.
    int    query_window = 1;
    double merge_dist = 0.25;   // [m] 같은 클래스가 이보다 가까우면 같은 물체로 본다
    // 검출 신뢰도 문턱. 25.10 은 0.25 -> 0.10 이 단일 프레임에서 "노드 4 개 이상"
    // 비율을 34 % -> 76 % 로 올린다고 쟀다. 창과 달리 이 손잡이는 노드 위치에
    // 오차를 누적시키지 않는다 - 대신 오검출을 들인다. 그 대가가 얼마인지는
    // 재 봐야 아는 것이고, 이 옵션이 그 실험을 가능하게 한다.
    double yolo_conf = 0.25;
    for (int i = 4; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--mode")          mode = argv[i + 1];
        else if (k == "--kf-dist")  kf_dist = std::atof(argv[i + 1]);
        else if (k == "--min-gap")  min_gap = std::atof(argv[i + 1]);
        else if (k == "--radius")   radius = std::atof(argv[i + 1]);
        else if (k == "--yolo")     yolo_model = argv[i + 1];
        else if (k == "--min-inliers") min_inliers = std::atoi(argv[i + 1]);
        else if (k == "--window")      query_window = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--merge-dist")  merge_dist = std::atof(argv[i + 1]);
        else if (k == "--conf")        yolo_conf = std::atof(argv[i + 1]);
    }

    const auto rgb_rows   = readIndex(root + "/rgb.txt");
    const auto depth_rows = readIndex(root + "/depth.txt");
    const auto traj       = readTraj(traj_path);
    const auto accel      = readAccel(root + "/accelerometer.txt");
    if (rgb_rows.empty() || traj.empty()) {
        std::cerr << "입력을 읽지 못했다\n";
        return 1;
    }

    double fx, fy, cx, cy;
    cv::Vec<double, 5> dist;
    if (root.find("freiburg2") != std::string::npos) {
        fx = 520.908620; fy = 521.007327; cx = 325.141442; cy = 249.701764;
        dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
    } else if (root.find("freiburg3") != std::string::npos) {
        fx = 535.4; fy = 539.2; cx = 320.1; cy = 247.6;
        dist = {0, 0, 0, 0, 0};
    } else {
        fx = 517.306408; fy = 516.469215; cx = 318.643040; cy = 255.313989;
        dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
    }
    constexpr double kDepthScale = 5000.0;
    const cv::Matx33d Kcv(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    wme::CameraIntrinsics Kw;
    Kw.width = 640; Kw.height = 480;
    Kw.fx = fx; Kw.fy = fy; Kw.cx = cx; Kw.cy = cy;
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv, cv::Size(640, 480),
                                CV_16SC2, map1, map2);
    const cv::Mat zero_dist = cv::Mat::zeros(5, 1, CV_64F);

    auto orb = cv::ORB::create(1000, 1.2F, 8);
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    std::unique_ptr<wme::IYoloRuntime> yolo;
    if (mode == "tcg") {
#ifdef WME_HAS_ORT
        if (yolo_model.empty()) {
            std::cerr << "tcg 모드에는 --yolo 가 필요하다\n";
            return 1;
        }
        wme::YoloConfig yc;
        yc.model_path = yolo_model;
        yc.confidence_threshold = static_cast<float>(yolo_conf);
        auto r = wme::YoloRuntimeOrt::create(yc);
        if (!r.ok()) { std::cerr << "YOLO 로드 실패: " << r.error().message() << "\n"; return 1; }
        yolo = std::move(r.value());
#else
        std::cerr << "ORT 없이 빌드되어 tcg 모드를 쓸 수 없다\n";
        return 1;
#endif
    }

    // --- 1 패스: 키프레임 수집 ------------------------------------------
    std::vector<KeyFrame> kfs;
    wme::SE3 last_kf_T = wme::SE3::identity();
    bool have = false;

    for (std::size_t ti = 0; ti < traj.size(); ++ti) {
        const auto& tr = traj[ti];
        if (have && (tr.T.translation() - last_kf_T.translation()).norm() < kf_dist) continue;

        const int ri = nearest(rgb_rows, tr.stamp, 0.02);
        if (ri < 0) continue;
        const int di = nearest(depth_rows, tr.stamp, 0.02);
        if (di < 0) continue;

        cv::Mat bgr = cv::imread(root + "/" + rgb_rows[static_cast<std::size_t>(ri)].file,
                                 cv::IMREAD_COLOR);
        cv::Mat dep = cv::imread(root + "/" + depth_rows[static_cast<std::size_t>(di)].file,
                                 cv::IMREAD_UNCHANGED);
        if (bgr.empty() || dep.empty()) continue;

        KeyFrame kf;
        kf.idx = static_cast<int>(ti);
        kf.stamp = tr.stamp;
        kf.T_world = tr.T;
        {
            wme::Vec3 a;
            if (accelAt(accel, tr.stamp, 0.05, a)) kf.gravity = a.normalized();
        }
        cv::Mat bu, du;
        cv::remap(bgr, bu, map1, map2, cv::INTER_LINEAR);
        cv::remap(dep, du, map1, map2, cv::INTER_NEAREST);
        cv::cvtColor(bu, kf.gray, cv::COLOR_BGR2GRAY);
        kf.depth = du;

        if (mode == "orb") {
            orb->detectAndCompute(kf.gray, cv::noArray(), kf.kp, kf.desc);
            kf.xyz.resize(kf.kp.size());
            kf.has.assign(kf.kp.size(), false);
            for (std::size_t i = 0; i < kf.kp.size(); ++i) {
                const int u = static_cast<int>(std::lround(kf.kp[i].pt.x));
                const int v = static_cast<int>(std::lround(kf.kp[i].pt.y));
                if (u < 0 || v < 0 || u >= du.cols || v >= du.rows) continue;
                const double z = du.at<std::uint16_t>(v, u) / kDepthScale;
                if (!(z > 0.1) || z > 8.0) continue;
                kf.xyz[i] = cv::Point3f(static_cast<float>((kf.kp[i].pt.x - cx) * z / fx),
                                        static_cast<float>((kf.kp[i].pt.y - cy) * z / fy),
                                        static_cast<float>(z));
                kf.has[i] = true;
            }
        } else {
#ifdef WME_HAS_ORT
            // 성좌 노드 = 검출 박스 중심의 3D 위치(카메라 좌표계).
            wme::Frame frm;
            frm.id    = wme::FrameId(static_cast<std::uint64_t>(kfs.size()) + 1);
            frm.stamp = wme::Timestamp::fromSeconds(kf.stamp);
            frm.gray  = kf.gray;
            frm.depth = du;
            frm.intrinsics = Kw;
            frm.sensor = wme::SensorKind::RgbD;
            const auto det = yolo->infer(frm);
            if (det.ok()) {
                for (const auto& d : det.value().items) {
                    const int u = static_cast<int>(d.box.x + d.box.width * 0.5);
                    const int v = static_cast<int>(d.box.y + d.box.height * 0.5);
                    if (u < 0 || v < 0 || u >= du.cols || v >= du.rows) continue;
                    // 박스 중앙 절반의 중앙값 깊이. 한 화소는 구멍이면 끝난다.
                    std::vector<double> zs;
                    const int hw = std::max(1, static_cast<int>(d.box.width * 0.25));
                    const int hh = std::max(1, static_cast<int>(d.box.height * 0.25));
                    for (int y = std::max(0, v - hh); y < std::min(du.rows, v + hh); ++y)
                        for (int x = std::max(0, u - hw); x < std::min(du.cols, u + hw); ++x) {
                            const double z = du.at<std::uint16_t>(y, x) / kDepthScale;
                            if (z > 0.1 && z < 8.0) zs.push_back(z);
                        }
                    if (zs.size() < 8) continue;
                    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
                    const double z = zs[zs.size() / 2];
                    wme::ConstellationNode n;
                    // id 는 대응 보고에 쓰이므로 성좌 안에서 유일해야 한다.
                    n.id = wme::TokenId(kf.nodes.size());
                    n.class_id = d.class_id;
                    n.position = wme::Vec3((u - cx) * z / fx, (v - cy) * z / fy, z);
                    kf.nodes.push_back(n);
                }
            }
#endif
        }
        kfs.push_back(std::move(kf));
        last_kf_T = tr.T;
        have = true;
        if (kfs.size() % 20 == 0) std::cout << "  키프레임 " << kfs.size() << "\n";
    }
    std::cout << "키프레임 " << kfs.size() << " (mode=" << mode << ")\n";

    // 키프레임 목록을 따로 남긴다. 포즈그래프는 전체 프레임이 아니라 키프레임
    // 위에서 풀어야 하고(1362 개를 조밀 행렬로 푸는 것은 비현실적), 파이썬이
    // 키프레임 규칙을 **다시 구현하면** 두 구현이 갈라지는 순간 간선 인덱스가
    // 조용히 어긋난다. 규칙은 한 군데에만 있어야 한다.
    {
        std::ofstream kout(out_path + ".keyframes.csv");
        if (!kout) { std::cerr << "키프레임 목록을 쓸 수 없다\n"; return 1; }
        kout << "kf,traj_idx,stamp\n";
        kout << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < kfs.size(); ++i)
            kout << i << "," << kfs[i].idx << "," << kfs[i].stamp << "\n";
    }

    // --- 2 패스: 후보 제안 + 검증 ---------------------------------------
    std::ofstream out(out_path);
    if (!out) { std::cerr << "출력을 열 수 없다: " << out_path << "\n"; return 1; }
    // i,j 는 **키프레임 번호**다(궤적 행 번호가 아니다). 포즈그래프가 키프레임
    // 위에서 풀리므로 그쪽이 간선의 자연스러운 좌표다. 궤적 행 번호도 같이
    // 남겨 두어야 최적화 결과를 원래 궤적에 되돌릴 수 있다.
    out << "i,j,traj_i,traj_j,stamp_i,stamp_j,accepted,support,tx,ty,tz,qx,qy,qz,qw\n";
    out << std::fixed << std::setprecision(6);

    int proposed = 0, accepted = 0;

    // TCG 는 **하나의 색인에 모든 과거 장소를 넣고** 질의해야 한다. 후보 쌍마다
    // 장소 한 개짜리 색인을 새로 만들면 경쟁 후보가 존재할 수 없고, query() 의
    // 모호성 기각 분기가 구조적으로 도달 불가능해진다 - 06-results.md 19.3 이
    // 차분 테스트에서 정확히 이 함정을 기록했고, 이 도구도 처음에 그대로 밟았다.
    // 그 상태로 잰 수치는 "TCG 의 기각 규칙" 이 아니라 그것을 꺼 놓은 무언가다.
    if (mode == "tcg") {
        wme::ConstellationIndex index;
        std::vector<std::size_t> place_to_kf;
        std::size_t next_insert = 0;

        // 창 성좌는 지도(insert)와 질의(query) **양쪽에** 같은 규칙으로 써야 한다.
        // 한쪽만 창을 쓰면 노드 수가 달라 정합이 구조적으로 깨진다.
        const auto windowNodes = [&](std::size_t at) {
            std::vector<const KeyFrame*> win;
            const std::size_t first =
                (at + 1 >= static_cast<std::size_t>(query_window))
                    ? at + 1 - static_cast<std::size_t>(query_window) : 0;
            for (std::size_t i = first; i <= at; ++i) win.push_back(&kfs[i]);
            return mergeWindow(win, merge_dist);
        };

        long long node_sum = 0, node_n = 0, ge4 = 0;

        for (std::size_t b = 0; b < kfs.size(); ++b) {
            // 질의 시점 기준 min_gap 이전의 키프레임만 지도에 넣는다.
            while (next_insert < kfs.size() &&
                   kfs[next_insert].stamp <= kfs[b].stamp - min_gap) {
                const auto place_nodes = windowNodes(next_insert);
                if (place_nodes.size() >= 4) {
                    index.insert(wme::KeyframeId(static_cast<std::uint64_t>(next_insert)),
                                 wme::Timestamp{static_cast<std::int64_t>(
                                     kfs[next_insert].stamp * 1e9)},
                                 wme::SE3::identity(), place_nodes,
                                 kfs[next_insert].gravity);
                    place_to_kf.push_back(next_insert);
                }
                ++next_insert;
            }
            const auto query_nodes = windowNodes(b);
            // 병합 뒤의 개수를 센다. 25.10 의 15.48 은 병합 전 합계였으므로
            // 이 값이 그 주장의 실제 검증이다.
            node_sum += static_cast<long long>(query_nodes.size());
            ++node_n;
            ge4 += (query_nodes.size() >= 4);
            if (place_to_kf.empty() || query_nodes.size() < 4) continue;
            ++proposed;

            const auto m = index.query(query_nodes, kfs[b].gravity);
            if (!m.ok()) continue;
            // place_id 는 삽입 순서대로 1 부터 매겨진다고 가정하지 않는다.
            // keyframe 필드로 되돌린다.
            const auto a = static_cast<std::size_t>(m.value().keyframe.value);
            if (a >= kfs.size()) continue;

            // ConstellationMatch::transform 은 **query -> place** 다
            // (헤더: "query 좌표계 -> place 좌표계"). 여기서 place 는 a, query 는
            // b 이므로 그것은 T_a_b 이고, 이 CSV 가 약속한 것은 T_b_a 다.
            //
            // 처음에 이 줄은 `T_b_a = transform` 이었다. 그 상태로 잰 간선 오차가
            // 280 cm 대였고 "성좌 서명이 장소를 구별하지 못한다" 로 읽힐 뻔했다.
            // 뒤집어 보니 55.7 cm - 5 배 차이가 규약 하나에 있었다. 24 절의 TCG
            // 포즈그래프도 이 뒤집힌 간선을 먹고 있었다.
            const wme::SE3 T_b_a = m.value().transform.inverse();
            double qx, qy, qz, qw;
            toQuat(T_b_a.rotation().matrix(), qx, qy, qz, qw);
            const wme::Vec3 t = T_b_a.translation();
            out << a << "," << b << "," << kfs[a].idx << "," << kfs[b].idx << ","
                << kfs[a].stamp << "," << kfs[b].stamp << ",1,"
                << m.value().n_inliers << "," << t.x() << "," << t.y() << "," << t.z()
                << "," << qx << "," << qy << "," << qz << "," << qw << "\n";
            ++accepted;
        }
        std::cout << "창 " << query_window << "  병합 후 평균 노드 "
                  << (node_n ? static_cast<double>(node_sum) / node_n : 0.0)
                  << "  노드>=4 " << (node_n ? 100.0 * ge4 / node_n : 0.0) << " %\n";
        std::cout << "질의 " << proposed << "  채택 " << accepted
                  << "  (" << (proposed ? 100.0 * accepted / proposed : 0.0) << " %)\n"
                  << "저장: " << out_path << "\n";
        return 0;
    }

    for (std::size_t a = 0; a < kfs.size(); ++a) {
        for (std::size_t b = a + 1; b < kfs.size(); ++b) {
            // 후보 규칙은 두 모드가 **동일하다**. 여기서 갈리면 뒤의 차이가
            // 인식 방법의 차이가 아니게 된다.
            if (kfs[b].stamp - kfs[a].stamp < min_gap) continue;
            const double d = (kfs[b].T_world.translation()
                              - kfs[a].T_world.translation()).norm();
            if (d > radius) continue;
            ++proposed;

            bool ok = false;
            double support = 0.0;
            wme::SE3 T_b_a = wme::SE3::identity();     // a 좌표계 -> b 좌표계

            if (mode == "orb") {
                if (kfs[a].desc.empty() || kfs[b].desc.empty()) continue;
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(kfs[a].desc, kfs[b].desc, knn, 2);
                std::vector<cv::Point3f> obj;
                std::vector<cv::Point2f> img;
                for (const auto& m : knn) {
                    if (m.size() < 2 || m[0].distance >= 0.75 * m[1].distance) continue;
                    const int qi = m[0].queryIdx, tix = m[0].trainIdx;
                    if (qi < 0 || tix < 0 || !kfs[a].has[static_cast<std::size_t>(qi)]) continue;
                    obj.push_back(kfs[a].xyz[static_cast<std::size_t>(qi)]);
                    img.push_back(kfs[b].kp[static_cast<std::size_t>(tix)].pt);
                }
                if (obj.size() >= static_cast<std::size_t>(min_inliers)) {
                    cv::Mat rvec, tvec, inl;
                    if (cv::solvePnPRansac(obj, img, cv::Mat(Kcv), zero_dist, rvec, tvec,
                                           false, 300, 3.0F, 0.99, inl,
                                           cv::SOLVEPNP_ITERATIVE)
                        && inl.rows >= min_inliers) {
                        std::vector<cv::Point3f> oi; std::vector<cv::Point2f> ii;
                        for (int k = 0; k < inl.rows; ++k) {
                            oi.push_back(obj[static_cast<std::size_t>(inl.at<int>(k))]);
                            ii.push_back(img[static_cast<std::size_t>(inl.at<int>(k))]);
                        }
                        cv::solvePnPRefineLM(oi, ii, cv::Mat(Kcv), zero_dist, rvec, tvec);
                        cv::Mat Rcv; cv::Rodrigues(rvec, Rcv);
                        wme::Mat3 R;
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c) R(r, c) = Rcv.at<double>(r, c);
                        T_b_a = wme::SE3(wme::SO3(R), wme::Vec3(tvec.at<double>(0),
                                                                tvec.at<double>(1),
                                                                tvec.at<double>(2)));
                        support = inl.rows;
                        ok = true;
                    }
                }
            } else {
                // TCG: a 를 지도로 한 인덱스에 b 를 질의한다. 채택/기각 결정은
                // ConstellationIndex 가 내린다 - 여기서 문턱을 다시 만들지 않는다.
                if (kfs[a].nodes.size() < 4 || kfs[b].nodes.size() < 4) continue;
                wme::ConstellationIndex index;
                index.insert(wme::KeyframeId(static_cast<std::uint64_t>(a)),
                             wme::Timestamp{static_cast<std::int64_t>(kfs[a].stamp * 1e9)},
                             wme::SE3::identity(), kfs[a].nodes, kfs[a].gravity);
                // 채택/기각은 query() 가 내린다. 여기서 문턱을 다시 만들면
                // 엔진이 결정을 내리지 않는 것이 된다.
                const auto m = index.query(kfs[b].nodes, kfs[b].gravity);
                if (m.ok()) {
                    T_b_a = m.value().transform;      // query(b) -> place(a)
                    support = static_cast<double>(m.value().n_inliers);
                    ok = true;
                }
            }

            double qx, qy, qz, qw;
            toQuat(T_b_a.rotation().matrix(), qx, qy, qz, qw);
            const wme::Vec3 t = T_b_a.translation();
            out << a << "," << b << "," << kfs[a].idx << "," << kfs[b].idx << ","
                << kfs[a].stamp << "," << kfs[b].stamp << "," << (ok ? 1 : 0) << ","
                << support << "," << t.x() << "," << t.y() << "," << t.z() << ","
                << qx << "," << qy << "," << qz << "," << qw << "\n";
            if (ok) ++accepted;
        }
    }

    std::cout << "후보 " << proposed << "  채택 " << accepted
              << "  (" << (proposed ? 100.0 * accepted / proposed : 0.0) << " %)\n"
              << "저장: " << out_path << "\n";
    return 0;
}
