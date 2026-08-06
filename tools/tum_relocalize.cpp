// TCG(Tier 1, Token Constellation Geometry) 를 실센서 데이터에 처음 돌린다.
//
// 목적은 "성능"이 아니라 두 가지다.
//   1) TUM 실내 장면에 YOLO 검출 객체가 성좌를 이룰 만큼 있는가.
//      (min_nodes 미달이면 질의가 통째로 침묵하고, 그 침묵은 그럴듯한 숫자로
//       위장된다 - docs/06-results.md 10.4 의 네 번째 줄이 정확히 그 사고다.)
//   2) 있다면, 재지역화가 실제로 되는가. 안 되면 침묵하는가, 아니면 틀린 포즈를
//      자신 있게 내놓는가.
//
// 이 도구는 판정하지 않는다. 프레임별 검출/노드 수, 질의별 후보와 추정 포즈,
// 같은 시각의 진리값을 CSV 로 그대로 남기고, 재현율/정밀도/오탐률 계산은
// python/tools/tcg_eval.py 가 맡는다. 추정기와 채점기를 같은 코드로 만들면
// 두 쪽의 버그가 서로를 가려 준다.
//
// 프로토콜
//   - 지도: 시퀀스 앞 map_frac 구간. map_stride 프레임마다 Place 등록.
//     포즈는 진리값을 쓴다 - 여기서 재는 것은 TCG 이지 ECDA 의 드리프트가 아니다.
//   - 질의: 나머지 구간. 질의 성좌는 *그 한 프레임의 검출만* 으로 카메라 좌표계에
//     세운다. 지도를 만든 TokenStore 의 융합 위치를 쓰면 포즈 정보가 새어
//     재지역화가 아니라 항등변환 복원이 된다.
//   - 음성 대조군: --neg 로 준 다른 시퀀스의 프레임을 같은 지도에 질의한다.
//     지도에 없는 장소이므로 반환된 모든 결과가 오탐이다.
//   - 자기질의: 등록된 Place 의 노드를 그대로 질의해 자기 자신을 찾는지 본다.
//     이게 실패하면 그 아래 모든 수치는 측정이 아니라 침묵이다.
//   - 다중 프레임 질의(--query-window N): 실제 재지역화기는 한 프레임이 아니라
//     작은 로컬 맵으로 질의한다. 창 안의 노드를 마지막 프레임 좌표계로 모으면
//     노드 수가 늘고 검출 깜빡임이 평균된다.
//     * 창 안의 *상대* 포즈는 진리값을 쓴다 - 오도메트리 대용이다. 절대 포즈는
//       새지 않는다(마지막 프레임 좌표계로만 옮기므로 지도와의 관계는 미지).
//       이 대용을 쓰지 않으려면 ECDA 를 물려야 하고 그건 다른 실험이다.
//
// 판정은 여전히 하지 않지만, C++ 의 채택/기각 결정(ConstellationIndex::query)은
// 그대로 기록한다. 그 결정이 파이썬에 있으면 엔진은 결정을 내리지 않는 것이다.
//
// 사용:
//   wme_tum_relocalize <시퀀스경로> <출력접두사> --yolo models/yolo11n.onnx
//     [--neg <다른시퀀스>] [--map-frac 0.5] [--map-stride 5] [--query-stride 3]
//     [--node-mode token|frame] [--min-nodes 4] [--gravity accel|none]
//     [--drop-agents 1]      자율 개체를 성좌에서 빼는 대조군
//     [--query-window 1]     질의에 쓸 연속 프레임 수 (1 = 단일 프레임)
//     [--window-merge 0.30]  창 안 같은 클래스 노드 병합 반경 (m)
//     [--window-min-count 1] 창 안에서 이 횟수 이상 본 노드만 채택

#include "wme/token/ConstellationIndex.hpp"
#include "wme/token/TokenStore.hpp"
#ifdef WME_HAS_ORT
#include "wme/perception/YoloRuntimeOrt.hpp"
#endif

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct IndexRow {
    double      stamp{0.0};
    std::string file;
};

struct PoseRow {
    double     stamp{0.0};
    wme::SE3   pose;
};

struct AccelRow {
    double stamp{0.0};
    wme::Vec3 a{wme::Vec3::Zero()};
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

// groundtruth.txt: timestamp tx ty tz qx qy qz qw
std::vector<PoseRow> readGroundTruth(const std::string& path) {
    std::vector<PoseRow> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t, tx, ty, tz, qx, qy, qz, qw;
        if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        PoseRow r;
        r.stamp = t;
        r.pose  = wme::SE3(wme::SO3(wme::Quat(qw, qx, qy, qz)), wme::Vec3(tx, ty, tz));
        out.push_back(r);
    }
    return out;
}

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

bool fileExists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

template <typename Row>
int nearest(const std::vector<Row>& rows, double stamp, double tol) {
    int best = -1;
    double best_d = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

// 창 안의 가속도 평균. 한 샘플만 쓰면 운동가속도가 그대로 섞인다.
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

void toQuat(const wme::Mat3& R, double& qx, double& qy, double& qz, double& qw) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (R(2, 1) - R(1, 2)) / s;
        qy = (R(0, 2) - R(2, 0)) / s;
        qz = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        qw = (R(2, 1) - R(1, 2)) / s;
        qx = 0.25 * s;
        qy = (R(0, 1) + R(1, 0)) / s;
        qz = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        qw = (R(0, 2) - R(2, 0)) / s;
        qx = (R(0, 1) + R(1, 0)) / s;
        qy = 0.25 * s;
        qz = (R(1, 2) + R(2, 1)) / s;
    } else {
        const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        qw = (R(1, 0) - R(0, 1)) / s;
        qx = (R(0, 2) + R(2, 0)) / s;
        qy = (R(1, 2) + R(2, 1)) / s;
        qz = 0.25 * s;
    }
}

std::string poseFields(const wme::SE3& T) {
    double qx, qy, qz, qw;
    toQuat(T.rotation().matrix(), qx, qy, qz, qw);
    const wme::Vec3 t = T.translation();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6)
       << t.x() << "," << t.y() << "," << t.z() << ","
       << qx << "," << qy << "," << qz << "," << qw;
    return ss.str();
}

// freiburg 그룹별 내부 파라미터 + 왜곡. fr1/fr2 는 보정 전 영상이다.
void intrinsicsFor(const std::string& root, wme::CameraIntrinsics& K,
                   cv::Vec<double, 5>& dist, std::string& group) {
    K.width = 640; K.height = 480;
    if (root.find("freiburg2") != std::string::npos) {
        K.fx = 520.908620; K.fy = 521.007327;
        K.cx = 325.141442; K.cy = 249.701764;
        dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
        group = "freiburg2";
    } else if (root.find("freiburg3") != std::string::npos) {
        K.fx = 535.4; K.fy = 539.2;
        K.cx = 320.1; K.cy = 247.6;
        dist = {0.0, 0.0, 0.0, 0.0, 0.0};
        group = "freiburg3";
    } else {
        K.fx = 517.306408; K.fy = 516.469215;
        K.cx = 318.643040; K.cy = 255.313989;
        dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
        group = "freiburg1";
    }
}

constexpr double kDepthScale = 5000.0;

// 한 시퀀스의 입출력 자원
struct Sequence {
    std::string           root;
    std::vector<IndexRow> rgb, depth;
    std::vector<PoseRow>  gt;
    std::vector<AccelRow> accel;
    wme::CameraIntrinsics K;
    cv::Mat               map1, map2;
    std::string           group;

    bool load(const std::string& path) {
        root  = path;
        rgb   = readIndex(path + "/rgb.txt");
        depth = readIndex(path + "/depth.txt");
        gt    = readGroundTruth(path + "/groundtruth.txt");
        accel = readAccel(path + "/accelerometer.txt");
        if (rgb.empty() || depth.empty() || gt.empty()) return false;

        cv::Vec<double, 5> dist;
        intrinsicsFor(path, K, dist, group);
        const cv::Matx33d Kcv(K.fx, 0.0, K.cx, 0.0, K.fy, K.cy, 0.0, 0.0, 1.0);
        cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv,
                                    cv::Size(K.width, K.height), CV_16SC2, map1, map2);
        return true;
    }
};

// 한 프레임의 로드 결과
struct LoadedFrame {
    wme::Frame frame;
    wme::SE3   gt_pose;
    bool       has_gt{false};
    std::optional<wme::Vec3> gravity;   // 카메라 좌표계 중력축 (부호 규약은 전 프레임 공통)
};

bool loadFrame(const Sequence& s, std::size_t i, std::uint64_t fid, bool use_accel,
               LoadedFrame& out) {
    const auto& rr = s.rgb[i];
    const std::string rgb_file = s.root + "/" + rr.file;
    if (!fileExists(rgb_file)) return false;

    const int di = nearest(s.depth, rr.stamp, 0.02);
    if (di < 0) return false;
    const std::string depth_file = s.root + "/" + s.depth[static_cast<std::size_t>(di)].file;
    if (!fileExists(depth_file)) return false;

    cv::Mat bgr_raw = cv::imread(rgb_file, cv::IMREAD_COLOR);
    cv::Mat draw    = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
    if (bgr_raw.empty() || draw.empty()) return false;

    cv::Mat bgr;
    cv::remap(bgr_raw, bgr, s.map1, s.map2, cv::INTER_LINEAR);

    cv::Mat depth_raw, depth;
    draw.convertTo(depth_raw, CV_32F, 1.0 / kDepthScale);
    cv::remap(depth_raw, depth, s.map1, s.map2, cv::INTER_NEAREST);

    out.frame = wme::Frame{};
    out.frame.id    = wme::FrameId(fid);
    out.frame.stamp = wme::Timestamp::fromSeconds(rr.stamp);
    out.frame.rgb   = bgr;
    cv::cvtColor(bgr, out.frame.gray, cv::COLOR_BGR2GRAY);
    out.frame.depth      = depth;
    out.frame.intrinsics = s.K;
    out.frame.sensor     = wme::SensorKind::RgbD;

    const int gi = nearest(s.gt, rr.stamp, 0.03);
    out.has_gt = gi >= 0;
    if (out.has_gt) out.gt_pose = s.gt[static_cast<std::size_t>(gi)].pose;

    out.gravity.reset();
    if (use_accel) {
        wme::Vec3 a;
        if (accelAt(s.accel, rr.stamp, 0.05, a)) out.gravity = a.normalized();
    }
    return true;
}

// 한 프레임의 검출만으로 성좌 노드를 만든다. 카메라 좌표계, 지도 지식 없음.
// 이것이 정직한 질의 조건이다 - 융합된 월드 위치를 쓰면 포즈가 새어 들어온다.
std::vector<wme::ConstellationNode> nodesFromFrame(const wme::DetectionSet& dets,
                                                   const wme::Frame& frame,
                                                   double depth_shrink,
                                                   std::size_t max_nodes,
                                                   bool drop_agents) {
    std::vector<wme::ConstellationNode> nodes;
    const auto& K = frame.intrinsics;
    if (!frame.hasDepth()) return nodes;

    // 자율 개체를 성좌에서 뺀다. 이것은 판정이 아니라 *대조군* 이다 -
    // 걸어다니는 사람이 성좌 노드가 되어 정밀도를 깎는지 분리해 보기 위한 것.
    // 실제 시스템에서 이 결정은 클래스가 아니라 static_belief 가 해야 한다
    // (docs/06-results.md 14.1).
    static const std::vector<std::string> kAgents = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "bird", "cat", "dog", "horse", "sheep", "cow"};

    std::uint64_t seq = 1;
    for (const auto& d : dets.items) {
        if (drop_agents &&
            std::find(kAgents.begin(), kAgents.end(), d.class_name) != kAgents.end()) {
            continue;
        }
        // 박스 중앙부의 깊이 중앙값. 가장자리는 배경을 문다.
        const float cx = d.box.x + d.box.width * 0.5f;
        const float cy = d.box.y + d.box.height * 0.5f;
        const float hw = d.box.width  * 0.5f * static_cast<float>(depth_shrink);
        const float hh = d.box.height * 0.5f * static_cast<float>(depth_shrink);
        const int x0 = std::max(0, static_cast<int>(std::floor(cx - hw)));
        const int y0 = std::max(0, static_cast<int>(std::floor(cy - hh)));
        const int x1 = std::min(K.width,  static_cast<int>(std::ceil(cx + hw)));
        const int y1 = std::min(K.height, static_cast<int>(std::ceil(cy + hh)));
        if (x1 <= x0 || y1 <= y0) continue;

        std::vector<float> vals;
        vals.reserve(static_cast<std::size_t>((x1 - x0) * (y1 - y0)));
        for (int y = y0; y < y1; ++y) {
            const float* row = frame.depth.ptr<float>(y);
            for (int x = x0; x < x1; ++x) {
                const float z = row[x];
                if (std::isfinite(z) && z > 0.1f && z < 60.0f) vals.push_back(z);
            }
        }
        if (vals.size() < 8) continue;   // 깊이 없는 검출은 3D 노드가 될 수 없다
        const std::size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
        double z = vals[mid];

        // 관측 깊이는 보이는 표면이다. 무게중심은 반치수만큼 뒤에 있다.
        const double ez = 0.25 * (d.box.width * z / K.fx + d.box.height * z / K.fy);
        z += ez;

        wme::ConstellationNode n;
        n.id       = wme::TokenId(seq++);
        n.class_id = d.class_id;
        n.position = K.backproject(wme::Vec2(cx, cy), z);
        // 깊이 잡음 + 베어링 잡음. TokenStore 와 같은 모델을 쓴다.
        const double sz = 0.006 * z * z;
        const double sx = 2.0 * z / K.fx;
        const double sy = 2.0 * z / K.fy;
        n.sigma = std::sqrt((sx * sx + sy * sy + sz * sz) / 3.0);
        nodes.push_back(n);
    }
    std::sort(nodes.begin(), nodes.end(),
              [](const auto& a, const auto& b) { return a.sigma < b.sigma; });
    if (nodes.size() > max_nodes) nodes.resize(max_nodes);
    return nodes;
}

// 다중 프레임 질의용 창. 한 프레임의 노드와 그 프레임의 진리 포즈를 같이 들고 있다.
struct WindowFrame {
    wme::SE3                            pose;    // 진리값 (상대 변환에만 쓴다)
    std::vector<wme::ConstellationNode> nodes;
};

// 창 안의 노드를 마지막 프레임 좌표계로 모아 하나의 질의 성좌를 만든다.
//
// 같은 객체가 여러 프레임에 잡히면 하나로 합친다. 위치는 정밀도 가중 평균이지만
// sigma 는 1/sqrt(N) 로 줄이지 않는다 - 연속 프레임의 깊이 편향은 공유되므로
// N 개의 관측을 N 개의 독립 증거로 세면 과신한다 (docs/06-results.md 10.1).
std::vector<wme::ConstellationNode> mergeWindow(const std::vector<WindowFrame>& window,
                                                const wme::SE3& current,
                                                double merge_radius, int min_count,
                                                std::size_t max_nodes) {
    struct Cluster {
        int       class_id{-1};
        wme::Vec3 wsum{wme::Vec3::Zero()};
        double    wtot{0.0};
        double    min_sigma{1e9};
        int       count{0};
    };
    std::vector<Cluster> clusters;

    const wme::SE3 world_to_cur = current.inverse();
    for (const auto& wf : window) {
        const wme::SE3 rel = world_to_cur * wf.pose;   // 그 프레임 좌표계 -> 현재 좌표계
        for (const auto& n : wf.nodes) {
            const wme::Vec3 p = rel * n.position;
            const double    w = 1.0 / std::max(1e-6, n.sigma * n.sigma);

            int hit = -1;
            for (std::size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i].class_id != n.class_id) continue;
                const wme::Vec3 c = clusters[i].wsum / clusters[i].wtot;
                if ((c - p).norm() <= merge_radius) { hit = static_cast<int>(i); break; }
            }
            if (hit < 0) {
                clusters.push_back({n.class_id, p * w, w, n.sigma, 1});
            } else {
                auto& c = clusters[static_cast<std::size_t>(hit)];
                c.wsum += p * w;
                c.wtot += w;
                c.min_sigma = std::min(c.min_sigma, n.sigma);
                ++c.count;
            }
        }
    }

    std::vector<wme::ConstellationNode> out;
    std::uint64_t seq = 1;
    for (const auto& c : clusters) {
        if (c.count < min_count) continue;   // 깜빡이는 오검출 억제
        wme::ConstellationNode n;
        n.id       = wme::TokenId(seq++);
        n.class_id = c.class_id;
        n.position = c.wsum / c.wtot;
        n.sigma    = c.min_sigma;            // 독립성을 가정하지 않는다
        out.push_back(n);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.sigma < b.sigma; });
    if (out.size() > max_nodes) out.resize(max_nodes);
    return out;
}

// 검출 클래스 목록을 ';' 로 잇는다. 분포를 python 에서 볼 수 있게 원자료를 남긴다.
std::string classList(const wme::DetectionSet& dets) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < dets.items.size(); ++i) {
        if (i) ss << ';';
        std::string n = dets.items[i].class_name;
        std::replace(n.begin(), n.end(), ',', '_');
        ss << n;
    }
    return ss.str();
}

// 질의 한 건의 결과를 CSV 한 줄로.
// 채점(재현율/정밀도)은 여전히 파이썬이 한다. 다만 *채택/기각 결정* 은 C++ 의
// ConstellationIndex::query() 가 내린 것을 그대로 옮긴다 - 결정이 채점기 안에
// 있으면 엔진은 아무 결정도 내리지 않는 것이고, 그건 구멍이지 해법이 아니다.
void writeQueryRow(std::ofstream& out, const std::string& tag, double stamp,
                   std::size_t n_det, std::size_t n_nodes, std::size_t n_win,
                   const std::vector<wme::ConstellationMatch>& all,
                   const wme::Result<wme::ConstellationMatch>& decision,
                   const wme::ConstellationIndex& index,
                   bool has_gt, const wme::SE3& gt_pose) {
    out << tag << "," << std::fixed << std::setprecision(6) << stamp << ","
        << n_det << "," << n_nodes << "," << n_win << "," << all.size() << ",";

    if (all.empty()) {
        // 후보 없음. 빈 칸이 아니라 명시적으로 남긴다.
        out << "0,0,0,0,0,0,";
        out << ",,,,,,,";                                  // est pose (7)
        out << "0,,,,,,,,";                                // 2위 place_id + est2 pose (7)
        out << "0,0,0,0,0,0,0,";                           // 신뢰도 원자료 (7)
    } else {
        const auto& m = all.front();
        const double s2 = (all.size() >= 2) ? all[1].score : 0.0;
        const wme::Place* p = index.place(m.place_id);
        const double place_stamp = p ? p->stamp.seconds() : 0.0;
        out << m.place_id << "," << place_stamp << "," << m.score << "," << s2 << ","
            << m.rms_error << "," << m.correspondences.size() << ",";
        const wme::SE3 est = p ? (p->anchor * m.transform) : wme::SE3();
        out << poseFields(est) << ",";

        // 2위 후보의 포즈까지 남긴다. 점수비만으로 모호성을 판정하면 같은 장소를
        // 겹쳐 등록한 지도에서 정상 일치가 전부 "모호"로 기각된다 - 두 후보가
        // 같은 포즈를 가리키는지 아닌지가 실제로 물어야 할 질문이다.
        if (all.size() >= 2) {
            const wme::Place* p2 = index.place(all[1].place_id);
            out << all[1].place_id << ","
                << poseFields(p2 ? (p2->anchor * all[1].transform) : wme::SE3()) << ",";
        } else {
            out << "0,,,,,,,,";
        }

        // 신뢰도 원자료. 조합을 미리 정하지 않고 원자료를 그대로 남겨야
        // 어느 신호가 오차를 예측하는지 사후에 비교할 수 있다.
        out << m.explained << "," << m.chi2_dof << "," << m.n_inliers << ","
            << m.agree_count << "," << m.support << "," << m.rival_mass << ","
            << m.pose_margin << ",";
    }

    // C++ 의 결정. accept=1 이면 query() 가 이 포즈를 내놓았다는 뜻이다.
    if (decision.ok()) {
        const auto& d = decision.value();
        const wme::Place* p = index.place(d.place_id);
        out << "1," << d.confidence << "," << d.place_id << ","
            << poseFields(p ? (p->anchor * d.transform) : wme::SE3()) << ",";
    } else {
        // accept,conf,acc_place + 포즈 7칸 = 10칸. 칸 수가 틀리면 뒤의 진리값이
        // 통째로 밀려 채점기가 조용히 다른 열을 읽는다.
        out << "0,0,0,,,,,,,,";
    }
    out << (has_gt ? 1 : 0) << "," << poseFields(gt_pose) << "\n";
}

// 한 질의의 모든 후보를 CSV 로. 규칙 탐색을 오프라인으로 옮기기 위한 원자료.
void writeCandidates(std::ofstream& out, const std::string& tag, double stamp,
                     const std::vector<wme::ConstellationMatch>& all,
                     const wme::ConstellationIndex& index) {
    for (std::size_t i = 0; i < all.size(); ++i) {
        const auto& m = all[i];
        const wme::Place* p = index.place(m.place_id);
        out << tag << "," << stamp << "," << i << "," << m.place_id << ","
            << m.score << "," << m.rms_error << "," << m.explained << ","
            << m.chi2_dof << "," << m.n_inliers << "," << m.agree_count << ","
            << m.support << "," << m.rival_mass << "," << m.pose_margin << ","
            << poseFields(p ? (p->anchor * m.transform) : wme::SE3()) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_tum_relocalize <시퀀스경로> <출력접두사> "
                     "--yolo <model.onnx> [--neg <시퀀스>] [--map-frac F] "
                     "[--map-stride N] [--query-stride N] [--node-mode token|frame] "
                     "[--min-nodes N] [--gravity accel|none] [--max-frames N]\n";
        return 2;
    }
    const std::string root   = argv[1];
    const std::string prefix = argv[2];

    std::string yolo_model, neg_root;
    double map_frac     = 0.5;
    int    map_stride   = 5;
    int    query_stride = 3;
    int    max_frames   = 0;
    std::string node_mode = "frame";   // 지도 노드 출처. 질의는 언제나 프레임 단독.
    std::string gravity_mode = "accel";
    float conf_thresh = 0.25f;
    bool drop_agents = false;   // 대조군: 자율 개체를 성좌에서 제외
    int    query_window = 1;    // 질의에 쓸 연속 프레임 수
    double window_merge = 0.30; // 창 안 같은 클래스 병합 반경 (m)
    int    window_min_count = 1;

    wme::ConstellationConfig ccfg;

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if      (k == "--yolo")          yolo_model  = argv[i + 1];
        else if (k == "--neg")           neg_root    = argv[i + 1];
        else if (k == "--map-frac")      map_frac    = std::atof(argv[i + 1]);
        else if (k == "--map-stride")    map_stride  = std::atoi(argv[i + 1]);
        else if (k == "--query-stride")  query_stride = std::atoi(argv[i + 1]);
        else if (k == "--max-frames")    max_frames  = std::atoi(argv[i + 1]);
        else if (k == "--node-mode")     node_mode   = argv[i + 1];
        else if (k == "--gravity")       gravity_mode = argv[i + 1];
        else if (k == "--conf")          conf_thresh = static_cast<float>(std::atof(argv[i + 1]));
        else if (k == "--drop-agents")   drop_agents = std::atoi(argv[i + 1]) != 0;
        else if (k == "--query-window")  query_window = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--window-merge")  window_merge = std::atof(argv[i + 1]);
        else if (k == "--window-min-count") window_min_count = std::max(1, std::atoi(argv[i + 1]));
        else if (k == "--agree-radius")  ccfg.pose_agree_radius = std::atof(argv[i + 1]);
        else if (k == "--dominance")     ccfg.pose_dominance = std::atof(argv[i + 1]);
        else if (k == "--min-conf")      ccfg.min_confidence = std::atof(argv[i + 1]);
        else if (k == "--min-nodes")     ccfg.min_nodes =
                                             static_cast<std::size_t>(std::atoi(argv[i + 1]));
        else if (k == "--min-inliers")   ccfg.min_inliers =
                                             static_cast<std::size_t>(std::atoi(argv[i + 1]));
        else if (k == "--dist-tol")      ccfg.distance_tolerance = std::atof(argv[i + 1]);
        else if (k == "--max-rms")       ccfg.max_rms_error = std::atof(argv[i + 1]);
        else if (k == "--top-candidates") ccfg.top_candidates =
                                             static_cast<std::size_t>(std::atoi(argv[i + 1]));
        else {
            std::cerr << "알 수 없는 인자: " << k << "\n";
            return 2;
        }
    }
    if (yolo_model.empty()) {
        std::cerr << "--yolo 는 필수다. YOLO 없이 이 실험은 아무것도 재지 않는다.\n";
        return 2;
    }

    Sequence seq;
    if (!seq.load(root)) {
        std::cerr << "시퀀스를 읽지 못했다: " << root << "\n";
        return 1;
    }
    std::cout << "시퀀스 " << root << "  내부파라미터 " << seq.group
              << "  rgb " << seq.rgb.size() << "  gt " << seq.gt.size()
              << "  accel " << seq.accel.size() << "\n";

    std::unique_ptr<wme::IYoloRuntime> yolo;
#ifdef WME_HAS_ORT
    {
        wme::YoloConfig yc;
        yc.model_path = yolo_model;
        yc.confidence_threshold = conf_thresh;
        auto r = wme::YoloRuntimeOrt::create(yc);
        if (!r.ok()) {
            std::cerr << "YOLO 로드 실패: " << r.error().message() << "\n";
            return 1;
        }
        yolo = std::move(r.value());
    }
#else
    std::cerr << "ORT 백엔드 없이 빌드되어 이 도구를 쓸 수 없다\n";
    return 1;
#endif

    const bool use_accel = (gravity_mode == "accel");

    wme::ConstellationIndex index(ccfg);
    wme::TokenStore   tokens;
    wme::EnvironmentState env;
    env.sensor_reliability = 1.0;
    env.track_persistence_scale = 1.0;
    env.memory_retention_scale = 1.0;

    // --- 출력 파일 ---------------------------------------------------------
    std::ofstream fcsv(prefix + "_frames.csv");
    fcsv << "seq,stamp,phase,n_det,n_det_depth,n_class,n_active,n_obs3,n_static07,"
            "n_exist06,n_sigma05,n_stable,n_covis_stable,n_frame_nodes,classes\n";
    fcsv << std::fixed << std::setprecision(6);

    std::ofstream qcsv(prefix + "_queries.csv");
    qcsv << "tag,stamp,n_det,n_nodes,n_win,n_cand,place_id,place_stamp,score,score2,rms,n_corr,"
            "est_tx,est_ty,est_tz,est_qx,est_qy,est_qz,est_qw,"
            "place2_id,est2_tx,est2_ty,est2_tz,est2_qx,est2_qy,est2_qz,est2_qw,"
            "explained,chi2,n_inl,agree,support,rival,margin,"
            "accept,conf,acc_place,acc_tx,acc_ty,acc_tz,acc_qx,acc_qy,acc_qz,acc_qw,"
            "has_gt,gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw\n";

    // 모든 후보를 그대로 남긴다. 채택 규칙을 바꿔 볼 때마다 YOLO 를 다시 돌리면
    // 규칙 탐색이 실행 시간에 갇힌다. 후보와 그 월드 포즈가 남아 있으면
    // 어떤 규칙이든 오프라인에서 채점할 수 있다.
    std::ofstream ccsv(prefix + "_cands.csv");
    ccsv << "tag,stamp,rank,place_id,score,rms,explained,chi2,n_inl,agree,support,rival,margin,"
            "tx,ty,tz,qx,qy,qz,qw\n";
    ccsv << std::fixed << std::setprecision(6);

    std::ofstream pcsv(prefix + "_places.csv");
    pcsv << "place_id,stamp,n_nodes,classes,tx,ty,tz,qx,qy,qz,qw\n";
    pcsv << std::fixed << std::setprecision(6);

    // 유효 프레임(영상+깊이+진리값)만 세어 map/query 를 가른다.
    // rgb.txt 행 수로 자르면 디스크에 없는 프레임이 섞여 경계가 흔들린다.
    std::vector<std::size_t> usable;
    for (std::size_t i = 0; i < seq.rgb.size(); ++i) {
        const std::string f = seq.root + "/" + seq.rgb[i].file;
        if (!fileExists(f)) continue;
        if (nearest(seq.depth, seq.rgb[i].stamp, 0.02) < 0) continue;
        if (nearest(seq.gt, seq.rgb[i].stamp, 0.03) < 0) continue;
        usable.push_back(i);
        if (max_frames > 0 && static_cast<int>(usable.size()) >= max_frames) break;
    }
    const std::size_t map_cut = static_cast<std::size_t>(
        std::llround(static_cast<double>(usable.size()) * map_frac));
    std::cout << "사용 가능 프레임 " << usable.size()
              << "  지도 구간 [0," << map_cut << ")  질의 구간 [" << map_cut << ","
              << usable.size() << ")\n";
    // 깊이 표본 축소율은 TokenStore 설정에서 온다 (다른 곳에서 바뀔 수 있으므로
    // 실행마다 찍어 둔다 - 두 실행을 비교하려면 이 값이 같아야 한다).
    std::cout << "depth_sample_shrink " << tokens.config().depth_sample_shrink << "\n";
    std::cout << "질의 창 " << query_window << " 프레임"
              << (query_window > 1 ? "  (창 내부 상대 포즈는 진리값 - 오도메트리 대용)" : "")
              << "  병합반경 " << window_merge << " m  최소관측 " << window_min_count
              << "\n모호성: 포즈군집 반경 " << ccfg.pose_agree_radius << " m / "
              << ccfg.pose_agree_deg << " deg, 질량비 하한 "
              << ccfg.pose_dominance << ", 신뢰도 하한 "
              << ccfg.min_confidence << "\n";

    // 중력축 규약 확인용. 가속도계 벡터를 월드로 돌려 평균내면 어느 축이 위인지
    // 드러난다. 규약을 짐작으로 두면 카이랄리티가 조용히 정대응을 걸러 낸다.
    wme::Vec3 g_world_sum = wme::Vec3::Zero();
    int g_n = 0;

    // 다중 프레임 질의용 창. 질의 구간 프레임만 담는다.
    std::vector<WindowFrame> qwindow;

    std::size_t total_det = 0, det_frames = 0;
    double yolo_ms = 0.0;

    for (std::size_t u = 0; u < usable.size(); ++u) {
        LoadedFrame lf;
        if (!loadFrame(seq, usable[u], static_cast<std::uint64_t>(u) + 1, use_accel, lf)) continue;

        auto dres = yolo->infer(lf.frame);
        if (!dres.ok()) continue;
        const wme::DetectionSet& dets = dres.value();
        total_det += dets.items.size();
        yolo_ms += dets.inference_ms;
        ++det_frames;

        if (lf.gravity && lf.has_gt) {
            g_world_sum += lf.gt_pose.rotation() * (*lf.gravity);
            ++g_n;
        }

        // 토큰 통합. 포즈는 진리값 - 이 실험이 재는 것은 TCG 이지 ECDA 가 아니다.
        tokens.integrate(dets, lf.frame, lf.gt_pose, env);

        // 프레임 단독 노드 (질의에 쓰는 것과 동일한 구성)
        const auto frame_nodes = nodesFromFrame(dets, lf.frame,
                                                tokens.config().depth_sample_shrink,
                                                ccfg.max_nodes, drop_agents);

        // 토큰 게이트별 카운트. 노드가 0 이 될 때 어느 게이트가 죽였는지
        // 사후에 알 수 없으면, 침묵을 결과로 착각하게 된다.
        std::size_t n_active = 0, n_obs3 = 0, n_static07 = 0, n_exist06 = 0,
                    n_sigma05 = 0, n_stable = 0;
        std::vector<wme::WorldTokenPtr> covis;
        for (const auto& t : tokens.allTokens()) {
            const bool seen_now = (t->last_seen == lf.frame.stamp);
            if (t->lifecycle == wme::TokenLifecycle::Active ||
                t->lifecycle == wme::TokenLifecycle::Occluded) ++n_active;
            if (t->observation_count >= 3)  ++n_obs3;
            if (t->static_belief > 0.7)     ++n_static07;
            if (t->existence_belief > 0.6)  ++n_exist06;
            if (t->positionSigma() < 0.5)   ++n_sigma05;
            if (t->isStableLandmark())      ++n_stable;
            if (seen_now) covis.push_back(t);
        }
        const auto covis_nodes =
            wme::ConstellationIndex::buildFrom(covis, lf.gt_pose, ccfg.max_nodes);

        std::set<int> classes;
        std::size_t n_depth = 0;
        for (const auto& d : dets.items) classes.insert(d.class_id);
        n_depth = frame_nodes.size();

        const bool is_map = (u < map_cut);
        fcsv << "map," << lf.frame.stamp.seconds() << "," << (is_map ? "map" : "query") << ","
             << dets.items.size() << "," << n_depth << "," << classes.size() << ","
             << n_active << "," << n_obs3 << "," << n_static07 << "," << n_exist06 << ","
             << n_sigma05 << "," << n_stable << "," << covis_nodes.size() << ","
             << frame_nodes.size() << "," << classList(dets) << "\n";

        if (is_map) {
            if (static_cast<int>(u) % map_stride == 0) {
                const auto& nodes = (node_mode == "token") ? covis_nodes : frame_nodes;
                const std::uint64_t pid = index.insert(
                    wme::KeyframeId(static_cast<std::uint64_t>(u) + 1), lf.frame.stamp,
                    lf.gt_pose, nodes, lf.gravity);
                if (pid != 0) {
                    std::ostringstream cs;
                    for (std::size_t i = 0; i < nodes.size(); ++i) {
                        if (i) cs << ';';
                        cs << nodes[i].class_id;
                    }
                    pcsv << pid << "," << lf.frame.stamp.seconds() << "," << nodes.size()
                         << "," << cs.str() << "," << poseFields(lf.gt_pose) << "\n";
                }
            }
        } else {
            // 질의 성좌는 지도의 융합 위치를 절대 쓰지 않는다 - 쓰면 재지역화가
            // 아니라 자기 자신을 되찾는 실험이 된다. 창을 써도 마찬가지로
            // 질의 구간 프레임의 검출만 모은다.
            qwindow.push_back({lf.gt_pose, frame_nodes});
            if (static_cast<int>(qwindow.size()) > query_window) qwindow.erase(qwindow.begin());

            if (static_cast<int>(u - map_cut) % query_stride == 0) {
                const auto qnodes =
                    (query_window <= 1)
                        ? frame_nodes
                        : mergeWindow(qwindow, lf.gt_pose, window_merge, window_min_count,
                                      ccfg.max_nodes);
                const auto all      = index.queryAll(qnodes, lf.gravity);
                const auto decision = index.query(qnodes, lf.gravity);
                writeQueryRow(qcsv, "pos", lf.frame.stamp.seconds(), dets.items.size(),
                              frame_nodes.size(), qnodes.size(), all, decision, index,
                              lf.has_gt, lf.gt_pose);
                writeCandidates(ccsv, "pos", lf.frame.stamp.seconds(), all, index);
            }
        }

        if ((u + 1) % 50 == 0) {
            std::cout << "  " << (u + 1) << "/" << usable.size()
                      << " 프레임  장소 " << index.placeCount() << "\n";
        }
    }

    if (det_frames > 0) {
        std::cout << "YOLO: 프레임당 검출 "
                  << (static_cast<double>(total_det) / static_cast<double>(det_frames))
                  << "  추론 " << (yolo_ms / static_cast<double>(det_frames)) << " ms\n";
    }
    if (g_n > 0) {
        const wme::Vec3 g = (g_world_sum / static_cast<double>(g_n)).normalized();
        std::cout << "가속도계 평균 방향(월드) = [" << g.x() << ", " << g.y() << ", "
                  << g.z() << "]  (부호 규약은 전 프레임 공통이므로 카이랄리티에 무해)\n";
    }
    std::cout << "등록된 장소 " << index.placeCount() << "\n";

    // --- 자기질의: 측정이 실제로 작동하는지 확인한다 -------------------------
    // 등록된 Place 의 노드를 그대로 질의해 자기 자신이 최상위로 나와야 한다.
    // 여기서 실패하면 아래 어떤 수치도 알고리즘에 대한 진술이 아니다.
    {
        std::ofstream scsv(prefix + "_selftest.csv");
        scsv << "place_id,n_nodes,n_cand,top_place,score,rms,trans_err,rot_err_deg,"
                "accept,conf,acc_place,acc_trans_err\n";
        scsv << std::fixed << std::setprecision(6);
        std::size_t hit = 0, tried = 0, accepted = 0, acc_self = 0;
        for (std::uint64_t pid = 1; pid <= index.placeCount(); ++pid) {
            const wme::Place* p = index.place(pid);
            if (p == nullptr) continue;
            ++tried;
            const auto all = index.queryAll(p->nodes, p->gravity);
            if (all.empty()) {
                scsv << pid << "," << p->nodes.size() << ",0,0,0,0,,,0,0,0,\n";
                continue;
            }
            const auto& m = all.front();
            // 자기 자신이면 변환은 항등이어야 한다.
            const double te = m.transform.translation().norm();
            const double re = m.transform.rotation().log().norm() * wme::kRad2Deg;
            scsv << pid << "," << p->nodes.size() << "," << all.size() << ","
                 << m.place_id << "," << m.score << "," << m.rms_error << ","
                 << te << "," << re << ",";
            if (m.place_id == pid) ++hit;

            // 자기질의를 채택 규칙에도 통과시킨다. 조밀 지도에서는 이웃 장소가
            // 같이 나오므로, 규칙이 이웃을 "모호"로 읽으면 자기질의부터 죽는다.
            const auto d = index.query(p->nodes, p->gravity);
            if (d.ok()) {
                ++accepted;
                const wme::Place* dp = index.place(d.value().place_id);
                const wme::SE3 est = dp ? (dp->anchor * d.value().transform) : wme::SE3();
                const double dte = (est.translation() - p->anchor.translation()).norm();
                if (d.value().place_id == pid) ++acc_self;
                scsv << "1," << d.value().confidence << "," << d.value().place_id << ","
                     << dte << "\n";
            } else {
                scsv << "0,0,0,\n";
            }
        }
        std::cout << "자기질의: " << hit << "/" << tried << " 가 자기 장소를 최상위로 반환"
                  << "   채택 " << accepted << "/" << tried
                  << " (자기 장소 " << acc_self << ")\n";
    }

    // --- 음성 대조군: 지도에 없는 시퀀스를 질의한다 --------------------------
    // 여기서 나오는 모든 반환은 오탐이다. 침묵이 정답인 구간.
    if (!neg_root.empty()) {
        Sequence ns;
        if (!ns.load(neg_root)) {
            std::cerr << "음성 대조 시퀀스를 읽지 못했다: " << neg_root << "\n";
        } else {
            std::cout << "음성 대조군 " << neg_root << " (" << ns.group << ")\n";
            std::size_t n_used = 0;
            std::vector<WindowFrame> nwindow;
            for (std::size_t i = 0; i < ns.rgb.size(); ++i) {
                LoadedFrame lf;
                if (!loadFrame(ns, i, static_cast<std::uint64_t>(i) + 1, use_accel, lf)) continue;

                // 창을 쓰지 않으면 stride 밖 프레임에 YOLO 를 돌릴 이유가 없다.
                const bool is_query = (static_cast<int>(n_used) % query_stride == 0);
                if (query_window <= 1 && !is_query) { ++n_used; continue; }

                auto dres = yolo->infer(lf.frame);
                if (!dres.ok()) { ++n_used; continue; }
                const auto nodes = nodesFromFrame(dres.value(), lf.frame,
                                                  tokens.config().depth_sample_shrink,
                                                  ccfg.max_nodes, drop_agents);
                // 창은 매 프레임 채우고 질의만 stride 로 솎는다. 양성 구간과
                // 같은 조건이어야 오탐률을 비교할 수 있다.
                // 진리 포즈가 없는 프레임은 상대 변환을 만들 수 없으므로 창을 끊는다.
                if (!lf.has_gt) nwindow.clear();
                else {
                    nwindow.push_back({lf.gt_pose, nodes});
                    if (static_cast<int>(nwindow.size()) > query_window) {
                        nwindow.erase(nwindow.begin());
                    }
                }
                if (!is_query) { ++n_used; continue; }
                ++n_used;

                const auto qnodes =
                    (query_window <= 1 || nwindow.empty())
                        ? nodes
                        : mergeWindow(nwindow, lf.gt_pose, window_merge, window_min_count,
                                      ccfg.max_nodes);
                const auto all      = index.queryAll(qnodes, lf.gravity);
                const auto decision = index.query(qnodes, lf.gravity);
                writeQueryRow(qcsv, "neg", lf.frame.stamp.seconds(),
                              dres.value().items.size(), nodes.size(), qnodes.size(), all,
                              decision, index, false, wme::SE3::identity());
                writeCandidates(ccsv, "neg", lf.frame.stamp.seconds(), all, index);

                // 음성 대조군의 프레임 통계도 남긴다 - 노드가 없어서 침묵한 것인지
                // 노드가 있는데 기각한 것인지 구분해야 오탐률에 의미가 생긴다.
                fcsv << "neg," << lf.frame.stamp.seconds() << ",neg,"
                     << dres.value().items.size() << "," << nodes.size() << ",0,"
                     << "0,0,0,0,0,0,0," << nodes.size() << ","
                     << classList(dres.value()) << "\n";
            }
        }
    }

    std::cout << "저장: " << prefix << "_{frames,queries,places,selftest}.csv\n";
    return 0;
}
