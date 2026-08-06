// TUM RGB-D 시퀀스에서 3-tier 융합을 돌린다.
//
//   Lambda_total = a0(E) Lambda_ECDA + a1(E) Lambda_TCG + a2(E) Lambda_SPA
//
// docs/06-results.md 1장의 표(합성, ATE 개선 6.8 -> 53.6 %)를 실데이터에서
// 처음으로 재현 시도하는 도구다. 채점은 python/tools/fusion_eval.py 가 한다 -
// 추정기와 채점기를 같은 코드로 만들면 둘의 버그가 서로를 가려 준다.
//
// 설계에서 중요한 것 하나: **키프레임 선택은 Tier 0 이 정하고, 모든 ablation 이
// 그것을 공유한다.** 그래야 ablation 사이 차이가 오직 융합에서만 나온다.
// ablation 마다 참조 프레임이 달라지면 (ref, cur) 쌍 자체가 달라져서, 무엇을
// 비교한 것인지 말할 수 없게 된다. 프레임 루프는 한 번만 돌고 tier 별 (T, Lambda)
// 를 기록한 뒤, 융합과 적분은 그 기록 위에서 ablation 마다 다시 한다.
//
// 사용:
//   wme_tum_fusion <시퀀스경로> <출력접두사> [--yolo models/yolo11n.onnx]
//                  [--max-frames N] [--kf-dist M] [--kappa1 K] [--kappa2 K]

#include "wme/fusion/PoseFusion.hpp"
#include "wme/fusion/TierInformation.hpp"
#include "wme/geometry/PlaneExtractor.hpp"
#include "wme/geometry/StructuralAligner.hpp"
#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"
#include "wme/token/ConstellationIndex.hpp"
#ifdef WME_HAS_ORT
#include "wme/perception/YoloRuntimeOrt.hpp"
#endif

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
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

using namespace wme;
using wme::fusion::Abstain;
using wme::fusion::Tier;
using wme::fusion::TierEstimate;

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

bool fileExists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

int nearest(const std::vector<IndexRow>& rows, double stamp, double tol) {
    int best = -1;
    double best_d = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

void toQuat(const Mat3& R, double& qx, double& qy, double& qz, double& qw) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        qw = 0.25 * s; qx = (R(2, 1) - R(1, 2)) / s;
        qy = (R(0, 2) - R(2, 0)) / s; qz = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
        qw = (R(2, 1) - R(1, 2)) / s; qx = 0.25 * s;
        qy = (R(0, 1) + R(1, 0)) / s; qz = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
        qw = (R(0, 2) - R(2, 0)) / s; qx = (R(0, 1) + R(1, 0)) / s;
        qy = 0.25 * s; qz = (R(1, 2) + R(2, 1)) / s;
    } else {
        const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
        qw = (R(1, 0) - R(0, 1)) / s; qx = (R(0, 2) + R(2, 0)) / s;
        qy = (R(1, 2) + R(2, 1)) / s; qz = 0.25 * s;
    }
}

// 한 프레임의 검출만으로 성좌 노드를 만든다. 지도 지식 없음, 카메라 좌표계.
// tools/tum_relocalize.cpp 의 nodesFromFrame 과 같은 규약이다 (그 파일은 다른
// 에이전트 소유라 재사용 대신 같은 계산을 여기서 한다).
std::vector<ConstellationNode> nodesFromFrame(const DetectionSet& dets, const Frame& frame,
                                              double depth_shrink, std::size_t max_nodes) {
    std::vector<ConstellationNode> nodes;
    const auto& K = frame.intrinsics;
    if (!frame.hasDepth()) return nodes;

    std::uint64_t seq = 1;
    for (const auto& d : dets.items) {
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
        for (int y = y0; y < y1; ++y) {
            const float* row = frame.depth.ptr<float>(y);
            for (int x = x0; x < x1; ++x) {
                const float z = row[x];
                if (std::isfinite(z) && z > 0.1f && z < 60.0f) vals.push_back(z);
            }
        }
        if (vals.size() < 8) continue;
        const std::size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid),
                         vals.end());
        double z = vals[mid];
        // 관측 깊이는 보이는 표면이다. 무게중심은 반치수만큼 뒤에 있다.
        z += 0.25 * (d.box.width * z / K.fx + d.box.height * z / K.fy);

        ConstellationNode n;
        n.id       = TokenId(seq++);
        n.class_id = d.class_id;
        n.position = K.backproject(Vec2(cx, cy), z);
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

// 한 프레임의 기록. 융합/적분은 전부 이 위에서 다시 돈다.
struct FrameRecord {
    double stamp{0.0};
    int    ref_index{-1};                       // 이 프레임의 참조 프레임 (records 인덱스)
    std::array<TierEstimate, 3> tiers{};
    std::array<double, 3>       alpha_env{{1.0, 1.0, 1.0}};
    int    tcg_ref_nodes{0}, tcg_cur_nodes{0};
    int    spa_ref_planes{0}, spa_cur_planes{0};
    int    t0_dof{0}, t2_dof{0};
    // 25절의 기하 정합성. 정렬에 쓰이지 않은 cur 깊이에서 오므로, "Tier 0 를
    // 얼마나 믿을 것인가" 를 측광 밖에서 물을 수 있는 유일한 재료다.
    double t0_depth_incons{-1.0};
    double t0_depth_rel{1.0};
    Vec6   t0_weak{Vec6::Zero()};
    Vec6   t2_weak{Vec6::Zero()};
};

struct Ablation {
    std::string           name;
    std::array<bool, 3>   on{{true, true, true}};
    fusion::WeightMode    mode{fusion::WeightMode::Environment};
};

Abstain parseAbstain(const std::string& s) {
    if (s == "none")             return Abstain::None;
    if (s == "disabled")         return Abstain::Disabled;
    if (s == "failed")           return Abstain::Failed;
    if (s == "zero_information") return Abstain::ZeroInformation;
    if (s == "zero_weight")      return Abstain::ZeroWeight;
    return Abstain::NoInput;
}

std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::istringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

// 기록 CSV 를 다시 읽는다. 보정 배율 kappa 는 ANEES 를 재고 나서야 정해지므로
// (측정이 먼저다) 영상 루프를 다시 돌지 않고 융합만 다시 할 수 있어야 한다.
// 그래야 "보정하면 결론이 바뀌는가" 를 몇 초 만에 물을 수 있다.
bool loadRecords(const std::string& path, std::vector<FrameRecord>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    if (!std::getline(in, line)) return false;   // 헤더

    const auto head = splitCsv(line);
    auto col = [&](const std::string& name) -> int {
        for (std::size_t i = 0; i < head.size(); ++i) {
            if (head[i] == name) return static_cast<int>(i);
        }
        return -1;
    };
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto v = splitCsv(line);
        auto at = [&](const std::string& n) { return v[static_cast<std::size_t>(col(n))]; };

        FrameRecord r;
        r.stamp     = std::atof(at("stamp").c_str());
        r.ref_index = std::atoi(at("ref_idx").c_str());
        for (int k = 0; k < 3; ++k) {
            r.alpha_env[static_cast<std::size_t>(k)] =
                std::atof(at("alpha" + std::to_string(k)).c_str());
        }
        r.tcg_ref_nodes  = std::atoi(at("tcg_ref_nodes").c_str());
        r.tcg_cur_nodes  = std::atoi(at("tcg_cur_nodes").c_str());
        r.spa_ref_planes = std::atoi(at("spa_ref_planes").c_str());
        r.spa_cur_planes = std::atoi(at("spa_cur_planes").c_str());
        r.t0_dof = std::atoi(at("t0_dof").c_str());
        r.t2_dof = std::atoi(at("t2_dof").c_str());

        for (int t = 0; t < 3; ++t) {
            const std::string p = "t" + std::to_string(t);
            TierEstimate& e = r.tiers[static_cast<std::size_t>(t)];
            e.tier      = static_cast<Tier>(t);
            e.available = at(p + "_ok") == "1";
            e.reason    = parseAbstain(at(p + "_reason"));
            const Quat q(std::atof(at(p + "_qw").c_str()), std::atof(at(p + "_qx").c_str()),
                         std::atof(at(p + "_qy").c_str()), std::atof(at(p + "_qz").c_str()));
            e.T_cur_ref = SE3(SO3(q), Vec3(std::atof(at(p + "_tx").c_str()),
                                           std::atof(at(p + "_ty").c_str()),
                                           std::atof(at(p + "_tz").c_str())));
            for (int a = 0; a < 6; ++a) {
                for (int b = a; b < 6; ++b) {
                    const double x = std::atof(
                        at(p + "_i" + std::to_string(a) + std::to_string(b)).c_str());
                    e.information(a, b) = x;
                    e.information(b, a) = x;
                }
            }
        }
        for (int a = 0; a < 6; ++a) r.t0_weak(a) = std::atof(at("t0_weak" + std::to_string(a)).c_str());
        for (int a = 0; a < 6; ++a) r.t2_weak(a) = std::atof(at("t2_weak" + std::to_string(a)).c_str());
        out.push_back(r);
    }
    return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_tum_fusion <시퀀스경로> <출력접두사> "
                     "[--yolo M] [--max-frames N] [--kf-dist M] [--kappa1 K] [--kappa2 K]\n";
        return 2;
    }
    const std::string root   = argv[1];
    const std::string prefix = argv[2];

    double kf_dist = 0.03;
    int    max_frames = 0;
    std::string yolo_model;
    double depth_shrink = 0.5;
    // tier 별 상대 보정 배율. 기본 1 = 출하 상태 그대로. fusion_eval.py 가
    // ANEES 로 잰 값을 여기에 넣어 "보정하면 달라지는가" 를 따로 재기 위한 것.
    std::array<double, 3> kappa{{1.0, 1.0, 1.0}};
    std::string replay;   // 기록 CSV 를 다시 읽어 융합만 다시 한다

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--replay")          replay = argv[i + 1];
        else if (k == "--kf-dist")    kf_dist = std::atof(argv[i + 1]);
        else if (k == "--max-frames") max_frames = std::atoi(argv[i + 1]);
        else if (k == "--yolo")       yolo_model = argv[i + 1];
        else if (k == "--shrink")     depth_shrink = std::atof(argv[i + 1]);
        else if (k == "--kappa0")     kappa[0] = std::atof(argv[i + 1]);
        else if (k == "--kappa1")     kappa[1] = std::atof(argv[i + 1]);
        else if (k == "--kappa2")     kappa[2] = std::atof(argv[i + 1]);
    }

    const auto rgb_rows   = readIndex(root + "/rgb.txt");
    const auto depth_rows = readIndex(root + "/depth.txt");
    if (rgb_rows.empty() || depth_rows.empty()) {
        std::cerr << "rgb.txt / depth.txt 를 읽지 못했다: " << root << "\n";
        return 1;
    }

    // 내부 파라미터/왜곡은 freiburg 그룹마다 다르다. tum_odometry.cpp 와 동일.
    CameraIntrinsics K;
    K.width = 640; K.height = 480;
    cv::Vec<double, 5> dist;
    if (root.find("freiburg2") != std::string::npos) {
        K.fx = 520.908620; K.fy = 521.007327; K.cx = 325.141442; K.cy = 249.701764;
        dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
    } else if (root.find("freiburg3") != std::string::npos) {
        K.fx = 535.4; K.fy = 539.2; K.cx = 320.1; K.cy = 247.6;
        dist = {0.0, 0.0, 0.0, 0.0, 0.0};
    } else {
        K.fx = 517.306408; K.fy = 516.469215; K.cx = 318.643040; K.cy = 255.313989;
        dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
    }
    constexpr double kDepthScale = 5000.0;
    const cv::Matx33d Kcv(K.fx, 0.0, K.cx, 0.0, K.fy, K.cy, 0.0, 0.0, 1.0);
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv,
                                cv::Size(K.width, K.height), CV_16SC2, map1, map2);

    DirectAligner       aligner{DirectAlignerConfig{}};
    ImageQualityEngine  iq;
    EnvironmentAnalyzer envan;
    PlaneExtractor      planes_ref_ex, planes_cur_ex;
    StructuralAligner   spa{StructuralAlignerConfig{}};
    ConstellationConfig ccfg;

    std::unique_ptr<IYoloRuntime> yolo;
    if (!replay.empty()) {
        // 재생 모드에서는 영상도 YOLO 도 필요 없다.
    } else if (!yolo_model.empty()) {
#ifdef WME_HAS_ORT
        YoloConfig yc;
        yc.model_path = yolo_model;
        auto r = YoloRuntimeOrt::create(yc);
        if (!r.ok()) {
            std::cerr << "YOLO 로드 실패: " << r.error().message() << "\n";
            return 1;
        }
        yolo = std::move(r.value());
#else
        std::cerr << "ORT 백엔드 없이 빌드되어 --yolo 를 쓸 수 없다\n";
        return 1;
#endif
    } else {
        std::cout << "경고: --yolo 없이 돌면 Tier 1 은 항상 기권한다\n";
    }

    // --- 1 패스: tier 별 (T, Lambda) 기록 ----------------------------------
    std::vector<FrameRecord> records;

    if (!replay.empty()) {
        if (!loadRecords(replay, records)) {
            std::cerr << "기록 CSV 를 읽지 못했다: " << replay << "\n";
            return 1;
        }
        std::cout << "재생 " << records.size() << " 프레임  (kappa "
                  << kappa[0] << " / " << kappa[1] << " / " << kappa[2] << ")\n";
    } else {
    // 이하 "프레임 N Tier0 실패 M" 출력까지가 영상 1 패스다.
    // (재생 모드가 통째로 건너뛰는 구간이라 들여쓰기를 유지한다)
    Frame                          ref;
    std::vector<ConstellationNode> ref_nodes;
    std::vector<Plane>             ref_planes;
    std::optional<Vec3>            ref_gravity;
    int   ref_index = -1;
    bool  have_ref  = false;

    SE3 T_cur_ref_prev = SE3::identity();
    SE3 velocity       = SE3::identity();

    int used = 0, t0_fail = 0;
    for (const auto& rr : rgb_rows) {
        if (max_frames > 0 && used >= max_frames) break;

        const std::string rgb_file = root + "/" + rr.file;
        if (!fileExists(rgb_file)) continue;
        const int di = nearest(depth_rows, rr.stamp, 0.02);
        if (di < 0) continue;
        const std::string depth_file = root + "/" + depth_rows[static_cast<std::size_t>(di)].file;
        if (!fileExists(depth_file)) continue;

        cv::Mat gray_raw = cv::imread(rgb_file, cv::IMREAD_GRAYSCALE);
        cv::Mat draw     = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
        if (gray_raw.empty() || draw.empty()) continue;

        cv::Mat gray;
        cv::remap(gray_raw, gray, map1, map2, cv::INTER_LINEAR);
        cv::Mat depth_raw, depth;
        draw.convertTo(depth_raw, CV_32F, 1.0 / kDepthScale);
        cv::remap(depth_raw, depth, map1, map2, cv::INTER_NEAREST);

        Frame f;
        f.id    = FrameId(static_cast<std::uint64_t>(used) + 1);
        f.stamp = Timestamp::fromSeconds(rr.stamp);
        f.gray  = gray;
        f.depth = depth;
        f.intrinsics = K;
        f.sensor = SensorKind::RgbD;

        const ImageQuality     q   = iq.evaluate(f);
        const EnvironmentState env = envan.update(f, q);

        // 현재 프레임의 성좌 노드와 평면. 두 tier 의 재료를 같은 자리에서 만든다.
        std::vector<ConstellationNode> cur_nodes;
        if (yolo) {
            const auto d = yolo->infer(f);
            if (d.ok()) cur_nodes = nodesFromFrame(d.value(), f, depth_shrink, ccfg.max_nodes);
        }
        std::vector<Plane> cur_planes;
        if (auto pr = planes_cur_ex.extract(depth, K); pr.ok()) cur_planes = pr.value();
        std::optional<Vec3> cur_gravity;
        if (auto g = dominantGravity(cur_planes); g.ok()) cur_gravity = g.value();

        FrameRecord rec;
        rec.stamp = rr.stamp;
        for (int t = 0; t < 3; ++t) {
            rec.tiers[static_cast<std::size_t>(t)].tier = static_cast<Tier>(t);
            rec.tiers[static_cast<std::size_t>(t)].reason = Abstain::NoInput;
        }
        const auto aw = fusion::tierWeights(env, fusion::WeightMode::Environment);
        rec.alpha_env = aw;
        rec.tcg_cur_nodes  = static_cast<int>(cur_nodes.size());
        rec.spa_cur_planes = static_cast<int>(cur_planes.size());

        if (!have_ref) {
            rec.ref_index = -1;
            records.push_back(rec);
            ref = f; ref_nodes = cur_nodes; ref_planes = cur_planes; ref_gravity = cur_gravity;
            ref_index = static_cast<int>(records.size()) - 1;
            have_ref = true;
            ++used;
            continue;
        }

        rec.ref_index      = ref_index;
        rec.tcg_ref_nodes  = static_cast<int>(ref_nodes.size());
        rec.spa_ref_planes = static_cast<int>(ref_planes.size());

        // 세 tier 가 같은 운동 사전분포를 초기값으로 받는다. SPA 에게만 ECDA 의
        // 해를 주면 Tier 2 가 Tier 0 의 함수가 되어 융합이 자기 자신과의 융합이 된다.
        const SE3 init = velocity * T_cur_ref_prev;

        // --- Tier 0 ECDA ---
        SE3  T0 = init;
        bool t0_ok = false;
        if (const auto r = aligner.align(ref, f, init, &q); r.ok()) {
            auto& e = rec.tiers[0];
            e.T_cur_ref   = r.value().T_cur_ref;
            e.information = r.value().information;
            e.available   = true;
            e.reason      = Abstain::None;
            T0 = r.value().T_cur_ref;
            t0_ok = true;
            rec.t0_dof  = r.value().observable_dof;
            rec.t0_depth_incons = r.value().depth_consistency;
            rec.t0_depth_rel    = r.value().depthReliability();
            rec.t0_weak = r.value().weakest_direction;
        } else {
            rec.tiers[0].reason = Abstain::Failed;
            ++t0_fail;
        }

        // --- Tier 2 SPA ---
        if (ref_planes.empty() || cur_planes.empty()) {
            rec.tiers[2].reason = Abstain::NoInput;
        } else if (const auto r = spa.align(ref_planes, cur_planes, init, 1.0); r.ok()) {
            auto& e = rec.tiers[2];
            e.T_cur_ref   = r.value().T_cur_ref;
            e.information = r.value().information;
            e.available   = true;
            e.reason      = Abstain::None;
            rec.t2_dof  = r.value().observable_dof;
            rec.t2_weak = r.value().weakest_direction;
        } else {
            rec.tiers[2].reason = Abstain::Failed;
        }

        // --- Tier 1 TCG ---
        // ref 프레임 하나를 장소로 등록하고 cur 프레임으로 질의한다. 지도도
        // 누적 토큰도 쓰지 않는다 - 그쪽은 포즈 추정이 이미 섞여 들어와서
        // "Tier 1 이 독립적으로 무엇을 아는가" 를 물을 수 없게 된다.
        if (ref_nodes.size() < ccfg.min_nodes || cur_nodes.size() < ccfg.min_nodes) {
            rec.tiers[1].reason = Abstain::NoInput;
        } else {
            ConstellationIndex index(ccfg);
            index.insert(KeyframeId(1), ref.stamp, SE3::identity(), ref_nodes, ref_gravity);
            const auto m = index.query(cur_nodes, cur_gravity);
            if (!m.ok()) {
                rec.tiers[1].reason = Abstain::Failed;
            } else {
                // transform 은 query(cur) -> place(ref) 다. 즉 T_ref_cur.
                const SE3 T1 = m.value().transform.inverse();
                std::vector<fusion::ConstellationPair> pairs;
                for (const auto& [qid, pid] : m.value().correspondences) {
                    const ConstellationNode* qn = nullptr;
                    const ConstellationNode* pn = nullptr;
                    for (const auto& n : cur_nodes) if (n.id == qid) { qn = &n; break; }
                    for (const auto& n : ref_nodes) if (n.id == pid) { pn = &n; break; }
                    if (!qn || !pn) continue;
                    fusion::ConstellationPair c;
                    c.p_ref = pn->position;
                    c.p_cur = qn->position;
                    c.sigma = std::sqrt(qn->sigma * qn->sigma + pn->sigma * pn->sigma);
                    pairs.push_back(c);
                }
                const Mat6 L = fusion::constellationInformation(pairs, T1);
                if (L.trace() > 0.0) {
                    auto& e = rec.tiers[1];
                    e.T_cur_ref   = T1;
                    e.information = L;
                    e.available   = true;
                    e.reason      = Abstain::None;
                } else {
                    rec.tiers[1].reason = Abstain::ZeroInformation;
                }
            }
        }

        records.push_back(rec);
        ++used;

        // --- 키프레임 정책: Tier 0 이 정한다 (모든 ablation 공유) ---
        if (!t0_ok) {
            ref = f; ref_nodes = cur_nodes; ref_planes = cur_planes; ref_gravity = cur_gravity;
            ref_index = static_cast<int>(records.size()) - 1;
            T_cur_ref_prev = SE3::identity();
        } else {
            velocity       = T0 * T_cur_ref_prev.inverse();
            T_cur_ref_prev = T0;
            if (T0.inverse().translation().norm() > kf_dist) {
                ref = f; ref_nodes = cur_nodes; ref_planes = cur_planes; ref_gravity = cur_gravity;
                ref_index = static_cast<int>(records.size()) - 1;
                T_cur_ref_prev = SE3::identity();
            }
        }
        if (used % 50 == 0) std::cout << "  " << used << " 프레임\n";
    }

    std::cout << "프레임 " << records.size() << "  Tier0 실패 " << t0_fail << "\n";
    }

    // --- 2 패스: ablation 별 융합 + 적분 ------------------------------------
    const std::vector<Ablation> ablations = {
        {"t0",            {{true,  false, false}}, fusion::WeightMode::Environment},
        {"t1",            {{false, true,  false}}, fusion::WeightMode::Environment},
        {"t2",            {{false, false, true }}, fusion::WeightMode::Environment},
        {"t0t1",          {{true,  true,  false}}, fusion::WeightMode::Environment},
        {"t0t2",          {{true,  false, true }}, fusion::WeightMode::Environment},
        {"t1t2",          {{false, true,  true }}, fusion::WeightMode::Environment},
        {"t0t1t2",        {{true,  true,  true }}, fusion::WeightMode::Environment},
        {"t0t1t2_uniform",{{true,  true,  true }}, fusion::WeightMode::Uniform},
    };

    // 기여율 집계는 전체 tier 를 켠 구성에서만 의미가 있다.
    std::array<int, 3> contrib{{0, 0, 0}};
    std::array<int, 3> attempts{{0, 0, 0}};

    for (const Ablation& ab : ablations) {
        std::vector<SE3> poses(records.size(), SE3::identity());
        int fused_ok = 0, fused_fail = 0;

        for (std::size_t i = 0; i < records.size(); ++i) {
            const FrameRecord& rec = records[i];
            if (rec.ref_index < 0) continue;             // 첫 프레임 = 원점

            std::vector<TierEstimate> es;
            es.reserve(3);
            for (int t = 0; t < 3; ++t) {
                TierEstimate e = rec.tiers[static_cast<std::size_t>(t)];
                e.calibration = kappa[static_cast<std::size_t>(t)];
                if (!ab.on[static_cast<std::size_t>(t)]) {
                    e.available = false;
                    e.reason    = Abstain::Disabled;
                } else {
                    e.alpha = (ab.mode == fusion::WeightMode::Uniform)
                                  ? 1.0
                                  : rec.alpha_env[static_cast<std::size_t>(t)];
                }
                es.push_back(e);
            }

            const auto fr = fusion::fuse(es);
            if (fr.ok()) {
                poses[i] = poses[static_cast<std::size_t>(rec.ref_index)]
                           * fr.value().T_cur_ref.inverse();
                ++fused_ok;
                if (ab.name == "t0t1t2") {
                    for (int t = 0; t < 3; ++t) {
                        ++attempts[static_cast<std::size_t>(t)];
                        if (fr.value().tiers[static_cast<std::size_t>(t)].used) {
                            ++contrib[static_cast<std::size_t>(t)];
                        }
                    }
                }
            } else {
                // 어떤 tier 도 못 냈다 - 직전 포즈를 유지한다. 조용히 원점으로
                // 되돌리면 궤적이 튀어 ATE 가 실패를 과장한다.
                poses[i] = (i > 0) ? poses[i - 1] : SE3::identity();
                ++fused_fail;
                if (ab.name == "t0t1t2") for (int t = 0; t < 3; ++t) ++attempts[static_cast<std::size_t>(t)];
            }
        }

        std::ofstream out(prefix + "_" + ab.name + ".txt");
        out << "# timestamp tx ty tz qx qy qz qw\n";
        out << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < records.size(); ++i) {
            double qx, qy, qz, qw;
            toQuat(poses[i].rotation().matrix(), qx, qy, qz, qw);
            const Vec3 t = poses[i].translation();
            out << records[i].stamp << " " << t.x() << " " << t.y() << " " << t.z()
                << " " << qx << " " << qy << " " << qz << " " << qw << "\n";
        }
        std::cout << "  " << ab.name << ": 융합 " << fused_ok << " 실패 " << fused_fail << "\n";
    }

    for (int t = 0; t < 3; ++t) {
        const auto s = static_cast<std::size_t>(t);
        std::cout << "기여율 " << fusion::toString(static_cast<Tier>(t)) << ": "
                  << contrib[s] << "/" << attempts[s] << " = "
                  << (attempts[s] ? 100.0 * contrib[s] / attempts[s] : 0.0) << " %\n";
    }

    // --- tier 별 원자료 CSV -------------------------------------------------
    // 보정(ANEES)과 상보성 판정은 진리값이 필요하므로 python 이 한다.
    // 여기서는 판단하지 않고 기록만 남긴다.
    {
        std::ofstream csv(prefix + "_tiers.csv");
        csv << "idx,stamp,ref_idx,ref_stamp,alpha0,alpha1,alpha2,"
               "tcg_ref_nodes,tcg_cur_nodes,spa_ref_planes,spa_cur_planes,t0_dof,t2_dof"
               ",t0_depth_incons,t0_depth_rel";
        for (int t = 0; t < 3; ++t) {
            const std::string p = "t" + std::to_string(t);
            csv << "," << p << "_ok," << p << "_reason"
                << "," << p << "_tx," << p << "_ty," << p << "_tz"
                << "," << p << "_qx," << p << "_qy," << p << "_qz," << p << "_qw";
            for (int a = 0; a < 6; ++a)
                for (int b = a; b < 6; ++b) csv << "," << p << "_i" << a << b;
        }
        for (int a = 0; a < 6; ++a) csv << ",t0_weak" << a;
        for (int a = 0; a < 6; ++a) csv << ",t2_weak" << a;
        csv << "\n";
        csv << std::scientific << std::setprecision(12);

        for (std::size_t i = 0; i < records.size(); ++i) {
            const FrameRecord& r = records[i];
            const double ref_stamp = (r.ref_index >= 0)
                                         ? records[static_cast<std::size_t>(r.ref_index)].stamp
                                         : 0.0;
            csv << i << "," << r.stamp << "," << r.ref_index << "," << ref_stamp
                << "," << r.alpha_env[0] << "," << r.alpha_env[1] << "," << r.alpha_env[2]
                << "," << r.tcg_ref_nodes << "," << r.tcg_cur_nodes
                << "," << r.spa_ref_planes << "," << r.spa_cur_planes
                << "," << r.t0_dof << "," << r.t2_dof
                << "," << r.t0_depth_incons << "," << r.t0_depth_rel;
            for (int t = 0; t < 3; ++t) {
                const TierEstimate& e = r.tiers[static_cast<std::size_t>(t)];
                double qx, qy, qz, qw;
                toQuat(e.T_cur_ref.rotation().matrix(), qx, qy, qz, qw);
                const Vec3 tr = e.T_cur_ref.translation();
                csv << "," << (e.available ? 1 : 0) << "," << fusion::toString(e.reason)
                    << "," << tr.x() << "," << tr.y() << "," << tr.z()
                    << "," << qx << "," << qy << "," << qz << "," << qw;
                for (int a = 0; a < 6; ++a)
                    for (int b = a; b < 6; ++b) csv << "," << e.information(a, b);
            }
            for (int a = 0; a < 6; ++a) csv << "," << r.t0_weak(a);
            for (int a = 0; a < 6; ++a) csv << "," << r.t2_weak(a);
            csv << "\n";
        }
    }

    std::cout << "저장: " << prefix << "_*.txt, " << prefix << "_tiers.csv\n";
    return 0;
}
