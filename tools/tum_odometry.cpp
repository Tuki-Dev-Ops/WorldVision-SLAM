// TUM RGB-D 시퀀스에 ECDA 를 돌려 궤적을 뽑는다.
//
// 이 도구의 목적은 성능 자랑이 아니라 "합성에서 맞춘 것이 실제 센서에서도
// 맞는가" 를 처음으로 확인하는 것이다. 결과는 TUM 형식으로 저장되고,
// ATE/RPE 계산은 python 쪽(wme.eval)이 맡는다 - 추정기와 채점기를 같은
// 코드로 만들면 둘의 버그가 서로를 가려 준다.
//
// 사용:
//   wme_tum_odometry <시퀀스경로> <출력.txt> [--kf-dist M] [--max-frames N]
//     --kf-dist 0 이면 프레임간(frame-to-frame) 오도메트리
//     --mask-mode class|token|observe  (observe = 판정만 돌리고 마스크는 미적용)
//     --assoc-diag <csv>  검출별 연관 표본. 혁신이 물체 운동인지 포즈 오차인지

#include "wme/localization/DirectAligner.hpp"
#include "wme/perception/ImageQualityEngine.hpp"
#include "wme/token/TokenStore.hpp"
#include "dataset_calib.hpp"
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
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 깊이 유효 범위. 기본은 TUM 센서, calib.txt 가 있으면 main 이 덮어쓴다.
// 상수로 박아 두면 KITTI(중앙 13.9 m)에서 거의 모든 점이 버려진다 -
// tools/dataset_calib.hpp 머리말 참조.
double g_depth_min = 0.1;
double g_depth_max = 8.0;

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

// 스스로 움직일 수 있는 COCO 클래스. 이 목록은 사전분포이지 판정이 아니다 -
// 주차된 차는 정적이고 들린 컵은 동적이다. 제대로 하려면 토큰의 static_belief 가
// 관측으로 갱신되어야 하고, 여기서는 그 상위 경계를 재는 것뿐이다.
bool isMobileClass(const std::string& name) {
    static const std::vector<std::string> mobile = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra"};
    return std::find(mobile.begin(), mobile.end(), name) != mobile.end();
}

// 검출 박스를 정적 마스크로. 0 = 동적(측광 잔차에서 제외).
// 박스는 물체보다 크므로 조금 더 넓혀 경계 픽셀까지 확실히 제거한다 -
// 남은 몇 픽셀이 잔차를 지배하면 마스킹의 의미가 없어진다.
void buildStaticMask(const wme::DetectionSet& dets, const cv::Size& size,
                     double dilate_ratio, cv::Mat& mask, int& masked_px) {
    mask.create(size, CV_8UC1);
    mask.setTo(255);
    masked_px = 0;
    for (const auto& d : dets.items) {
        if (!isMobileClass(d.class_name)) continue;
        const double mx = d.box.width * dilate_ratio;
        const double my = d.box.height * dilate_ratio;
        cv::Rect r(static_cast<int>(d.box.x - mx), static_cast<int>(d.box.y - my),
                   static_cast<int>(d.box.width + 2 * mx),
                   static_cast<int>(d.box.height + 2 * my));
        r &= cv::Rect(0, 0, size.width, size.height);
        if (r.area() > 0) mask(r).setTo(0);
    }
    masked_px = size.area() - cv::countNonZero(mask);
}

// 기하 정합성: ref 의 3D 점을 추정 포즈로 cur 에 투영하고, 그 화소에서 cur 이
// **실제로 보고한 깊이** 와 비교한다.
//
// 왜 이 채널인가. 13.3 은 측광 자기평가가 동적 장면에서 완전히 눈멀었다는 것을
// 재고 이렇게 끝난다: *"측광 신뢰도 경로를 아무리 손봐도 고칠 수 없다. 정보가
// 측광 채널에 없기 때문이다."* 그러면 다른 채널을 봐야 한다.
//
// ECDA 는 ref 의 깊이로 3D 점을 만들고 cur 의 **밝기만** 읽는다. cur 의 깊이맵은
// 정렬에 한 번도 쓰이지 않는다. 따라서 이 비교는 추정에 쓰이지 않은 독립 관측이고,
// 포즈가 틀리면 예측 깊이와 측정 깊이가 어긋난다. 안개(23절)는 광학계만 망가뜨리고
// 깊이 센서는 건드리지 않으므로, 측광이 무너진 구간에서도 이 채널은 살아 있다.
//
// 반환: 유효 표본 중 |z_pred - z_meas| / z_meas 의 중앙값. 표본이 없으면 -1.
double depthConsistency(const wme::Frame& ref, const wme::Frame& cur,
                        const wme::SE3& T_cur_ref, double& outlier_frac) {
    outlier_frac = -1.0;
    if (ref.depth.empty() || cur.depth.empty()) return -1.0;
    const auto& K = cur.intrinsics;

    std::vector<double> rel;
    rel.reserve(2048);
    int outliers = 0, total = 0;
    // 성긴 격자면 충분하다 - 이것은 추정이 아니라 검사다.
    for (int v = 4; v < ref.depth.rows - 4; v += 6) {
        for (int u = 4; u < ref.depth.cols - 4; u += 6) {
            const float zr = ref.depth.at<float>(v, u);
            if (!(zr > g_depth_min) || zr > g_depth_max) continue;
            const wme::Vec3 p_ref((u - K.cx) * zr / K.fx, (v - K.cy) * zr / K.fy, zr);
            const wme::Vec3 p_cur = T_cur_ref * p_ref;
            if (!(p_cur.z() > 0.1)) continue;
            const double uc = K.fx * p_cur.x() / p_cur.z() + K.cx;
            const double vc = K.fy * p_cur.y() / p_cur.z() + K.cy;
            const int ui = static_cast<int>(std::lround(uc));
            const int vi = static_cast<int>(std::lround(vc));
            if (ui < 0 || vi < 0 || ui >= cur.depth.cols || vi >= cur.depth.rows) continue;
            const float zm = cur.depth.at<float>(vi, ui);
            if (!(zm > g_depth_min) || zm > g_depth_max) continue;
            const double r = std::abs(p_cur.z() - zm) / zm;
            rel.push_back(r);
            ++total;
            // 10 % 는 TUM 깊이 잡음(수 mm)보다 한참 위다. 이걸 넘으면 잡음이 아니라
            // 다른 표면을 보고 있다는 뜻이다 - 포즈가 틀렸거나 물체가 움직였거나.
            if (r > 0.10) ++outliers;
        }
    }
    if (rel.size() < 50) return -1.0;
    std::nth_element(rel.begin(), rel.begin() + rel.size() / 2, rel.end());
    outlier_frac = static_cast<double>(outliers) / std::max(1, total);
    return rel[rel.size() / 2];
}

// 지도에 남겨 두는 키프레임. 재측위는 **최근 프레임이 아니라 지도** 에 다시
// 앵커링하는 것이므로, 과거 포즈와 그때의 기술자를 들고 있어야 한다.
//
// 25.8 이 남긴 결론이 이 코드의 이유다: 기하 게이트로 키프레임을 갈아 끼우는
// 것은 추적이 힘든 구간에서 도움이 되지만(4승 2패), **이미 발산한 추정기는
// 구조하지 못한다** - 안개 구간 네 개가 전부 바닥값을 넘긴 채였고 하나는 13 배
// 더 나빠졌다. 국소적으로 참조를 바꾸는 것으로는 누적된 드리프트를 되돌릴 수
// 없기 때문이다. 되돌리려면 드리프트가 쌓이기 전의 좌표계로 돌아가야 한다.
struct MapKeyFrame {
    wme::SE3                  T_world;      // 이 키프레임의 (당시) 월드 포즈
    cv::Mat                   desc;
    std::vector<cv::KeyPoint> kp;
    std::vector<cv::Point3f>  xyz;          // 키프레임 카메라 좌표계
    std::vector<bool>         has_xyz;
};

// 가장 가까운 시각의 항목. 허용 오차를 넘으면 -1.
int nearest(const std::vector<IndexRow>& rows, double stamp, double tol) {
    int best = -1;
    double best_d = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= best_d) { best_d = d; best = static_cast<int>(i); }
    }
    return best;
}

// 회전행렬 -> 쿼터니언 (x,y,z,w). TUM 형식이 요구한다.
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_tum_odometry <시퀀스경로> <출력.txt> "
                     "[--kf-dist M] [--max-frames N]\n";
        return 2;
    }
    const std::string root = argv[1];
    const std::string out_path = argv[2];

    // TUM fr1_xyz 실측 최적값. 0.02 / 0.03 / 0.05 / 0.10 / 0.30 에서
    // ATE 1.97 / 1.55 / 1.62 / 2.32 / 4.98 cm.
    double kf_dist = 0.03;      // 이만큼 움직이면 키프레임 교체
    int    max_frames = 0;      // 0 = 전부
    bool   use_prior = true;    // 등속 사전분포로 예측 초기화
    std::string diag_path;      // 프레임별 진단 CSV (비면 안 씀)
    std::string yolo_model;     // 지정하면 동적 마스킹을 켠다
    double dilate_ratio = 0.10; // 박스를 이 비율만큼 넓혀 마스킹
    std::string token_diag_path;// 토큰별 진단 CSV (비면 안 씀)
    std::string assoc_diag_path;// 검출별 연관 표본 CSV (비면 안 씀)
    wme::TokenStoreConfig tcfg; // 연관 상수를 밖에서 흔들어 볼 수 있게 둔다

    // class  : 움직일 수 있는 클래스를 무조건 지운다. 사전분포를 판정으로 쓰는 것.
    // token  : TokenStore 의 static_belief 로 판정한다. 관측이 결론을 바꾼다.
    // observe: 토큰/판정은 전부 돌리되 마스크를 적용하지 않는다. 포즈가 성한
    //          상태에서 연관이 어떻게 도는지를 재는 대조군 - "연관이 깨져서
    //          발산했다" 와 "발산해서 연관이 깨졌다" 를 가르는 유일한 측정이다.
    std::string mask_mode = "token";
    // 25절 신호를 키프레임 교체로 소비할 것인가. 기본은 꺼 둔다 - 기존 수치와
    // 비교 가능해야 하고, 켠 쪽이 나은지는 재 봐야 아는 것이지 가정할 것이 아니다.
    bool depth_gate_kf = false;
    // 기하 게이트가 연속으로 이만큼 발동하면 지도에 재질의한다. 1 로 두면
    // 한 프레임의 잡음에도 재측위가 돌아 궤적이 튄다.
    int  reloc_after = 0;          // 0 = 재측위 끄기
    int  reloc_min_inliers = 40;   // PnP 인라이어가 이보다 적으면 채택하지 않는다

    wme::DirectAlignerConfig cfg;
    double depth_min_cli = -1.0, depth_max_cli = -1.0;   // <0 = 지정 안 함

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--kf-dist")          kf_dist = std::atof(argv[i + 1]);
        else if (k == "--max-frames")  max_frames = std::atoi(argv[i + 1]);
        else if (k == "--huber-k")     cfg.huber_k = std::atof(argv[i + 1]);
        else if (k == "--noise-ratio") cfg.huber_noise_ratio = std::atof(argv[i + 1]);
        else if (k == "--levels")      cfg.pyramid_levels = std::atoi(argv[i + 1]);
        else if (k == "--grid")        cfg.grid_cell = std::atoi(argv[i + 1]);
        else if (k == "--edge-ratio")  cfg.depth_edge_ratio = std::atof(argv[i + 1]);
        else if (k == "--min-grad")    cfg.min_gradient = std::atof(argv[i + 1]);
        else if (k == "--max-points")  cfg.max_points =
                                           static_cast<std::size_t>(std::atoi(argv[i + 1]));
        else if (k == "--prior")       use_prior = std::atoi(argv[i + 1]) != 0;
        else if (k == "--diag")        diag_path = argv[i + 1];
        else if (k == "--yolo")        yolo_model = argv[i + 1];
        else if (k == "--dilate")      dilate_ratio = std::atof(argv[i + 1]);
        else if (k == "--mask-mode")   mask_mode = argv[i + 1];
        else if (k == "--token-diag")  token_diag_path = argv[i + 1];
        else if (k == "--assoc-diag")  assoc_diag_path = argv[i + 1];
        else if (k == "--maneuver")    tcfg.assoc_maneuver_speed = std::atof(argv[i + 1]);
        else if (k == "--depth-gate-kf") depth_gate_kf = std::atoi(argv[i + 1]) != 0;
        else if (k == "--reloc")          reloc_after = std::atoi(argv[i + 1]);
        else if (k == "--reloc-inliers")  reloc_min_inliers = std::atoi(argv[i + 1]);
        // calib.txt 보다 뒤에 적용된다. 깊이 상한이 결과를 얼마나 흔드는지
        // 재려면 파일을 고치지 않고 흔들 수 있어야 한다.
        else if (k == "--depth-min")   depth_min_cli = std::atof(argv[i + 1]);
        else if (k == "--depth-max")   depth_max_cli = std::atof(argv[i + 1]);
        // sigma_Z = c * Z^2. 스테레오면 c = sigma_d/(f*B).
        else if (k == "--depth-sigma-rel") cfg.depth_sigma_rel = std::atof(argv[i + 1]);
    }
    tcfg.collect_assoc_probes = !assoc_diag_path.empty();

    const auto rgb_rows   = readIndex(root + "/rgb.txt");
    const auto depth_rows = readIndex(root + "/depth.txt");
    if (rgb_rows.empty() || depth_rows.empty()) {
        std::cerr << "rgb.txt / depth.txt 를 읽지 못했다: " << root << "\n";
        return 1;
    }

    // 내부 파라미터/왜곡/깊이 범위. 값은 dataset_calib.hpp 한 곳에만 있고
    // wme_tum_baseline 도 같은 함수를 부른다. calib.txt 가 있으면 그것이 이긴다 -
    // 경로 문자열 분기는 TUM 안에서만 맞는다. KITTI 경로에는 freiburg 가 없으므로
    // 640x480/fx=517 이 1241x376 영상에 적용되고, 결과는 실패가 아니라 그럴듯한
    // 오차로 나온다.
    wme_tools::DatasetCalib dc;
    if (!wme_tools::resolveCalib(root, dc)) return 1;
    wme::CameraIntrinsics K = dc.K;
    cv::Vec<double, 5> dist = dc.dist;
    double kDepthScale = dc.depth_scale;
    {
        g_depth_min = dc.depth_min;
        g_depth_max = dc.depth_max;
        // 정렬기의 유효 깊이도 데이터셋을 따라야 한다. DirectAlignerConfig 의
        // 기본 40 m 는 TUM 실내를 넉넉히 덮지만 KITTI(최대 47.9 m)는 자른다.
        cfg.min_depth = dc.depth_min;
        cfg.max_depth = dc.depth_max;
    }
    if (depth_min_cli > 0.0) { g_depth_min = depth_min_cli; cfg.min_depth = depth_min_cli; }
    if (depth_max_cli > 0.0) { g_depth_max = depth_max_cli; cfg.max_depth = depth_max_cli; }
    std::cout << "깊이 유효 범위: [" << g_depth_min << ", " << g_depth_max << "] m\n";

    const cv::Matx33d Kcv(K.fx, 0.0, K.cx,
                          0.0, K.fy, K.cy,
                          0.0, 0.0, 1.0);

    // 보정 후에도 같은 내부 파라미터를 유지하도록 newCameraMatrix 를 K 로 고정한다.
    // 그래야 깊이맵과 좌표계가 그대로 맞는다.
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv,
                                cv::Size(K.width, K.height), CV_16SC2, map1, map2);

    wme::DirectAligner aligner(cfg);
    wme::ImageQualityEngine iq;

    // 동적 마스킹. 모델이 지정됐는데 열리지 않으면 조용히 끄지 않고 실패한다 -
    // "마스킹을 켰다고 생각했는데 안 켜진 채로 나온 숫자" 가 제일 위험하다.
    std::unique_ptr<wme::IYoloRuntime> yolo;
    if (!yolo_model.empty()) {
#ifdef WME_HAS_ORT
        wme::YoloConfig yc;
        yc.model_path = yolo_model;
        auto r = wme::YoloRuntimeOrt::create(yc);
        if (!r.ok()) {
            std::cerr << "YOLO 로드 실패: " << r.error().message() << "\n";
            return 1;
        }
        yolo = std::move(r.value());
        std::cout << "동적 마스킹 ON  방식=" << mask_mode
                  << " (팽창 " << dilate_ratio << ")\n";
#else
        std::cerr << "ORT 백엔드 없이 빌드되어 --yolo 를 쓸 수 없다\n";
        return 1;
#endif
    }

    // 참조(키프레임) 상태
    wme::Frame ref;
    wme::SE3   T_world_ref = wme::SE3::identity();
    wme::SE3   T_world_cam = wme::SE3::identity();
    bool       have_ref = false;
    // 참조 프레임 시각. 정보행렬은 T_cur_ref 에 대한 것이므로, 밖에서 진리값과
    // 대조하려면 cur 뿐 아니라 ref 가 언제였는지도 있어야 한다.
    double     ref_stamp = 0.0;

    // 운동 사전분포. 매번 항등에서 시작하면 키프레임이 멀어질수록 그 거리를
    // 통째로 냉시작으로 수렴해야 하고, 실데이터의 수렴 반경은 합성 평면보다
    // 훨씬 좁다. 직전 추정에 직전 증분을 한 번 더 적용해 예측으로 넣는다.
    wme::SE3 T_cur_ref_prev = wme::SE3::identity();
    wme::SE3 velocity       = wme::SE3::identity();

    std::ofstream out(out_path);
    out << "# timestamp tx ty tz qx qy qz qw\n";
    out << std::fixed << std::setprecision(6);

    // 프레임별 진단. "시스템이 자신이 모른다는 것을 아는가" 를 사후에 검증하려면
    // 추정과 같은 시각의 자기평가가 남아 있어야 한다. 진리값과 대조하는 일은
    // python 이 한다 - 여기서는 판단하지 않고 기록만 한다.
    //
    // 기존 열은 다른 도구가 파싱하므로 순서를 바꾸지 않고 뒤에만 덧붙인다.
    // 덧붙이는 것: 참조 시각, 상대 포즈 T_cur_ref, 그리고 6x6 정보행렬의
    // 상삼각 21개. 이 셋이 있어야 NEES 를 밖에서 계산할 수 있다.
    //   - 정보행렬은 좌측 섭동 T <- exp(dxi)*T_cur_ref 에 대한 것이고
    //     dxi = [rho(3), phi(3)] 규약이다 (wme::SE3 와 동일).
    //   - 상삼각만 쓰는 이유는 대칭이 수학적으로 보장되어서다. 36개를 쓰면
    //     같은 값을 두 번 적는 것이고, 파일이 커질수록 파싱 실수만 늘어난다.
    std::ofstream diag;
    if (!diag_path.empty()) {
        diag.open(diag_path);
        diag << "timestamp,points,inliers,inlier_ratio,rmse,observable_dof,"
                "ev_min,ev_max,cond,trans_norm,rot_norm"
                ",ref_timestamp,rel_tx,rel_ty,rel_tz,rel_qx,rel_qy,rel_qz,rel_qw"
                ",photo_var,noise_sigma";
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) diag << ",info_" << i << j;
        }
        // 기존 열은 다른 도구가 파싱하므로 순서를 바꾸지 않고 뒤에만 덧붙인다.
        diag << ",depth_incons,depth_outlier";
        diag << "\n";
        diag << std::fixed << std::setprecision(6);
    }

    int used = 0, failed = 0, kf_switches = 0, kf_geom_switches = 0;
    std::vector<MapKeyFrame> map_kfs;
    int  bad_streak = 0, reloc_tried = 0, reloc_ok = 0;
    auto orb = cv::ORB::create(1000, 1.2F, 8);
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    double rmse_sum = 0.0, inlier_sum = 0.0, point_sum = 0.0, dof_sum = 0.0;
    int    stat_n = 0;
    double masked_sum = 0.0, det_sum = 0.0, yolo_ms = 0.0;
    double mask_obs_sum = 0.0, mask_stale_sum = 0.0, mask_unj_sum = 0.0;
    double assoc_match = 0.0, assoc_nocand = 0.0, assoc_gated = 0.0, assoc_unass = 0.0;
    double gated_maha = 0.0, gated_dist = 0.0;
    int    yolo_n = 0;
    double agent_static_sum = 0.0, agent_obs_sum = 0.0, agent_count_sum = 0.0;
    double agent_best_sb = 0.0, agent_best_obs = 0.0;
    int    agent_n = 0;
    double disp_sum = 0.0, sigma_sum = 0.0;
    int    zn = 0;
    std::map<wme::TokenId, wme::Vec3> last_pos;

    // 토큰 생존 통계. 사라진 시점의 관측수를 세야 "증거가 느린가" 와
    // "토큰이 죽는가" 를 나눌 수 있다. 살아 있는 토큰만 보면 둘이 섞인다.
    std::map<wme::TokenId, std::pair<std::uint32_t, bool>> seen_tokens;  // id -> (obs, agent)
    std::vector<std::uint32_t> agent_lifetimes;
    long long stable_frame_sum = 0, stable_agent_sum = 0;
    int stable_n = 0;

    std::ofstream tdiag;
    if (!token_diag_path.empty()) {
        tdiag.open(token_diag_path);
        tdiag << "stamp,id,cls,agent,lifecycle,miss,obs,static_belief,exist_belief,"
                 "pos_sigma,meas_sigma,su_updates,su_disp,su_fused_disp,su_sigma,su_z,su_ratio,"
                 "stable,dynamic,mx,my,mz\n";
        tdiag << std::fixed << std::setprecision(6);
    }

    std::ofstream adiag;
    if (!assoc_diag_path.empty()) {
        adiag.open(assoc_diag_path);
        adiag << "stamp,cam_dtheta,cam_dtrans,cls,agent,matched,gated,dt,dist,maha,"
                 "dwx,dwy,dwz,dcx,dcy,dcz,range,tok_sigma,meas_sigma\n";
        adiag << std::fixed << std::setprecision(6);
    }
    wme::SE3 T_world_cam_prev = wme::SE3::identity();
    bool     have_prev_cam = false;

    wme::TokenStore     tokens(tcfg);
    wme::EnvironmentState env;   // 실내 정상 조건. 이 실험은 환경 적응을 재지 않는다.
    env.sensor_reliability = 1.0;
    env.track_persistence_scale = 1.0;
    env.memory_retention_scale = 1.0;

    for (const auto& rr : rgb_rows) {
        if (max_frames > 0 && used >= max_frames) break;

        const std::string rgb_file = root + "/" + rr.file;
        if (!fileExists(rgb_file)) continue;      // 부분 데이터 허용

        const int di = nearest(depth_rows, rr.stamp, 0.02);
        if (di < 0) continue;
        const std::string depth_file = root + "/" + depth_rows[static_cast<std::size_t>(di)].file;
        if (!fileExists(depth_file)) continue;

        cv::Mat gray_raw = cv::imread(rgb_file, cv::IMREAD_GRAYSCALE);
        cv::Mat draw     = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
        if (gray_raw.empty() || draw.empty()) continue;

        cv::Mat gray;
        cv::remap(gray_raw, gray, map1, map2, cv::INTER_LINEAR);

        // 16bit PNG -> m. 0 은 무효값이며 min_depth 가 걸러 준다.
        // 깊이는 반드시 최근접으로 보정한다. 선형 보간을 쓰면 무효값 0 과
        // 실제 깊이가 섞여 허공에 점이 생긴다.
        cv::Mat depth_raw, depth;
        draw.convertTo(depth_raw, CV_32F, 1.0 / kDepthScale);
        cv::remap(depth_raw, depth, map1, map2, cv::INTER_NEAREST);

        wme::Frame f;
        f.id    = wme::FrameId(static_cast<std::uint64_t>(used) + 1);
        f.stamp = wme::Timestamp::fromSeconds(rr.stamp);
        f.gray  = gray;
        f.depth = depth;
        f.intrinsics = K;
        f.sensor = wme::SensorKind::RgbD;

        // 품질 지표는 로버스트 커널의 잡음 기준으로 쓰인다.
        // 이 경로가 있어야 임계가 실제 센서 잡음에 묶인다.
        const wme::ImageQuality q = iq.evaluate(f);

        // 동적 마스크는 ref 프레임에 붙어야 한다. ECDA 는 ref 에서 점을 고르므로,
        // cur 에 붙이면 아무 효과가 없다.
        if (yolo) {
            const auto d = yolo->infer(f);
            if (d.ok()) {
                if (mask_mode == "class") {
                    int px = 0;
                    buildStaticMask(d.value(), gray.size(), dilate_ratio, f.static_mask, px);
                    masked_sum += static_cast<double>(px) / gray.total();
                } else {
                    // 토큰 경로. 포즈는 직전 추정을 쓴다 - 실제 시스템도 그렇게 돈다.
                    // 마스크는 static_belief 가 낮은 토큰만 지운다.
                    if (auto ir = tokens.integrate(d.value(), f, T_world_cam, env); ir.ok()) {
                        assoc_match  += static_cast<double>(ir.value().matched);
                        assoc_nocand += static_cast<double>(ir.value().det_no_candidate);
                        assoc_gated  += static_cast<double>(ir.value().det_gated_out);
                        assoc_unass  += static_cast<double>(ir.value().det_unassigned);
                        gated_maha   += ir.value().gated_maha_sum;
                        gated_dist   += ir.value().gated_dist_sum;

                        if (adiag.is_open()) {
                            // 카메라가 이 프레임 사이에 얼마나 돌고 움직였는가.
                            // 혁신이 이 값을 따라 커지면 그것은 물체가 아니라 포즈다.
                            double dth = 0.0, dtr = 0.0;
                            if (have_prev_cam) {
                                const wme::SE3 dT = T_world_cam_prev.inverse() * T_world_cam;
                                dth = dT.rotation().log().norm();
                                dtr = dT.translation().norm();
                            }
                            for (const auto& p : ir.value().probes) {
                                adiag << rr.stamp << "," << dth << "," << dtr << ","
                                      << p.class_id << "," << (p.agent ? 1 : 0) << ","
                                      << (p.matched ? 1 : 0) << "," << (p.gated ? 1 : 0) << ","
                                      << p.dt << "," << p.dist << "," << p.maha << ","
                                      << p.diff_world.x() << "," << p.diff_world.y() << ","
                                      << p.diff_world.z() << ","
                                      << p.diff_cam.x() << "," << p.diff_cam.y() << ","
                                      << p.diff_cam.z() << "," << p.range << ","
                                      << p.tok_sigma << "," << p.meas_sigma << "\n";
                            }
                        }
                        T_world_cam_prev = T_world_cam;
                        have_prev_cam = true;
                    }
                    wme::TokenStore::MaskReport mr;
                    cv::Mat m = tokens.buildStaticMask(f, T_world_cam, &mr);
                    // observe 모드: 토큰과 판정은 전부 돌리되 마스크는 적용하지
                    // 않는다. 포즈가 성한 상태에서 연관이 어떻게 도는지를 보려면
                    // 마스크가 포즈를 망가뜨리지 않는 조건이 필요하다 - 그래야
                    // "연관이 깨져서 발산했다" 와 "발산해서 연관이 깨졌다" 가 갈린다.
                    if (mask_mode != "observe") f.static_mask = m;
                    masked_sum    += mr.masked_ratio;
                    mask_obs_sum  += mr.from_observed;
                    mask_stale_sum += mr.from_stale;
                    mask_unj_sum  += mr.withheld_unjudged;
                    // 토큰이 얼마나 오래 살아남는지와 믿음이 어디까지 가는지를
                    // 같이 본다. 믿음이 안 오르는 이유가 증거 속도인지 토큰 수명인지
                    // 이 둘을 나란히 놓지 않으면 구분할 수 없다.
                    double sb = 0.0, obs = 0.0, best_sb = 0.0, best_obs = 0.0;
                    std::size_t n = 0;
                    for (const auto& t : tokens.allTokens()) {
                        if (!has(t->affordance, wme::Affordance::Agent)) continue;
                        sb  += t->static_belief;
                        obs += static_cast<double>(t->observation_count);
                        best_sb  = std::max(best_sb, t->static_belief);
                        best_obs = std::max(best_obs, static_cast<double>(t->observation_count));
                        ++n;
                    }
                    if (n > 0) {
                        agent_static_sum += sb / static_cast<double>(n);
                        agent_obs_sum    += obs / static_cast<double>(n);
                        agent_best_sb     = std::max(agent_best_sb, best_sb);
                        agent_best_obs    = std::max(agent_best_obs, best_obs);
                        agent_count_sum  += static_cast<double>(n);
                        ++agent_n;
                    }

                    // z = 변위 / (sigma * scale) 을 직접 잰다.
                    // 이 값이 1 근처면 증거가 중립이라 믿음이 움직이지 않는다.
                    // 분자와 분모 중 어느 쪽이 문제인지는 둘을 따로 봐야 안다.
                    for (const auto& t : tokens.allTokens()) {
                        if (!has(t->affordance, wme::Affordance::Agent)) continue;
                        auto it = last_pos.find(t->id);
                        if (it != last_pos.end()) {
                            disp_sum  += (t->position - it->second).norm();
                            sigma_sum += t->positionSigma();
                            ++zn;
                        }
                        last_pos[t->id] = t->position;
                    }

                    // 안정 랜드마크 수 (TCG 입력). 프레임당 몇 개가 통과하는지가
                    // Tier 1 이 살아 있는지 아닌지를 그대로 말해 준다.
                    int n_stable = 0, n_stable_agent = 0;
                    for (const auto& t : tokens.allTokens()) {
                        if (!t->isStableLandmark()) continue;
                        ++n_stable;
                        if (has(t->affordance, wme::Affordance::Agent)) ++n_stable_agent;
                    }
                    stable_frame_sum += n_stable;
                    stable_agent_sum += n_stable_agent;
                    ++stable_n;

                    // 생존 추적: 이번 프레임에 없는 id 는 은퇴한 것이다.
                    std::map<wme::TokenId, std::pair<std::uint32_t, bool>> cur;
                    for (const auto& t : tokens.allTokens()) {
                        const bool ag = has(t->affordance, wme::Affordance::Agent);
                        cur[t->id] = {t->observation_count, ag};
                        if (tdiag.is_open()) {
                            tdiag << rr.stamp << "," << t->id.value << "," << t->class_name << ","
                                  << (ag ? 1 : 0) << "," << static_cast<int>(t->lifecycle) << ","
                                  << t->miss_count << ","
                                  << t->observation_count << "," << t->static_belief << ","
                                  << t->existence_belief << "," << t->positionSigma() << ","
                                  << t->meas_sigma << "," << t->static_diag.updates << ","
                                  << t->static_diag.disp << "," << t->static_diag.fused_disp << ","
                                  << t->static_diag.sigma << "," << t->static_diag.z << ","
                                  << t->static_diag.ratio << ","
                                  << (t->isStableLandmark() ? 1 : 0) << ","
                                  << (t->isDynamic() ? 1 : 0) << ","
                                  << t->meas_world.x() << "," << t->meas_world.y() << ","
                                  << t->meas_world.z() << "\n";
                        }
                    }
                    for (const auto& [id, v] : seen_tokens) {
                        if (cur.count(id) == 0 && v.second) agent_lifetimes.push_back(v.first);
                    }
                    seen_tokens = cur;
                }
                det_sum += static_cast<double>(d.value().items.size());
                yolo_ms += d.value().inference_ms;
                ++yolo_n;
            }
        }

        if (!have_ref) {
            ref = f;
            T_world_ref = T_world_cam;
            ref_stamp = rr.stamp;
            have_ref = true;
        } else {
            const wme::SE3 init = use_prior ? velocity * T_cur_ref_prev
                                            : wme::SE3::identity();
            const auto r = aligner.align(ref, f, init, &q);
            if (!r.ok()) {
                ++failed;
                // 실패를 조용히 넘기지 않는다. 참조를 갱신해 다음 프레임에 건다.
                ref = f;
                T_world_ref = T_world_cam;
                ref_stamp = rr.stamp;
                T_cur_ref_prev = wme::SE3::identity();
                ++kf_switches;
            } else {
                // align 은 T_cur_ref 를 준다. 월드 포즈는 그 역이 누적된다.
                const wme::SE3 T_cur_ref = r.value().T_cur_ref;
                const wme::SE3 T_ref_cur = T_cur_ref.inverse();
                T_world_cam = T_world_ref * T_ref_cur;

                velocity       = T_cur_ref * T_cur_ref_prev.inverse();
                T_cur_ref_prev = T_cur_ref;

                rmse_sum   += r.value().photometric_rmse;
                inlier_sum += r.value().inlier_ratio;
                point_sum  += static_cast<double>(r.value().point_count);
                dof_sum    += static_cast<double>(r.value().observable_dof);
                ++stat_n;

                if (diag.is_open()) {
                    const auto& v = r.value();
                    // 고유값은 내림차순이 아니라 Eigen 관례상 오름차순으로 담긴다.
                    const double ev_min = v.eigenvalues.minCoeff();
                    const double ev_max = v.eigenvalues.maxCoeff();
                    diag << rr.stamp << "," << v.point_count << "," << v.inlier_count << ","
                         << v.inlier_ratio << "," << v.photometric_rmse << ","
                         << v.observable_dof << "," << ev_min << "," << ev_max << ","
                         << (ev_max / std::max(1e-12, ev_min)) << ","
                         << T_ref_cur.translation().norm() << ","
                         << T_ref_cur.rotation().log().norm() << ","
                         << ref_stamp;

                    // 상대 포즈와 정보행렬은 자릿수가 크게 벌어진다(정보행렬은
                    // 1e3~1e14). 고정소수점으로 적으면 작은 성분의 유효자리가
                    // 통째로 날아가므로 이 열들만 지수표기로 쓴다.
                    std::ostringstream ext;
                    ext << std::scientific << std::setprecision(12);

                    double rqx, rqy, rqz, rqw;
                    toQuat(T_cur_ref.rotation().matrix(), rqx, rqy, rqz, rqw);
                    const wme::Vec3 rt = T_cur_ref.translation();
                    ext << "," << rt.x() << "," << rt.y() << "," << rt.z()
                        << "," << rqx << "," << rqy << "," << rqz << "," << rqw;

                    // 정보행렬을 나눈 잔차 분산. 이게 있어야 "과신이 잔차 분산을
                    // 잘못 잡아서인지, 잔차가 서로 상관되어서인지" 를 밖에서
                    // 나눠 볼 수 있다. 없으면 둘 중 하나를 추측하게 된다.
                    ext << "," << q.photometricVariance() << "," << q.noise_sigma;

                    for (int i = 0; i < 6; ++i) {
                        for (int j = i; j < 6; ++j) ext << "," << v.information(i, j);
                    }

                    // 기하 정합성. 추정에 쓰이지 않은 cur 의 깊이맵과 비교하므로
                    // 측광 채널이 눈먼 구간에서도 독립적으로 말할 수 있다.
                    double d_out = -1.0;
                    const double d_inc = depthConsistency(ref, f, T_cur_ref, d_out);
                    ext << "," << d_inc << "," << d_out;

                    diag << ext.str() << "\n";
                }

                // --- 재측위 ---------------------------------------------
                // 게이트가 연속으로 발동하면 지도에 다시 물어본다. 성공하면
                // 월드 포즈를 지도가 말하는 값으로 되돌린다 - 이것이 키프레임
                // 교체와 다른 점이다. 키프레임 교체는 앞으로의 추정을 고칠 뿐
                // 이미 쌓인 드리프트는 그대로 두지만, 재측위는 그 드리프트를
                // 지운다.
                if (reloc_after > 0) {
                    const bool bad = r.value().depth_consistency >= 0.0 &&
                                     !r.value().depthConsistent();
                    bad_streak = bad ? bad_streak + 1 : 0;

                    if (bad_streak >= reloc_after && map_kfs.size() >= 3) {
                        ++reloc_tried;
                        std::vector<cv::KeyPoint> qkp;
                        cv::Mat qdesc;
                        orb->detectAndCompute(gray, cv::noArray(), qkp, qdesc);

                        int    best_in = 0;
                        wme::SE3 best_T_world;
                        // 최근 키프레임은 건너뛴다. 바로 직전 것에 앵커링하면
                        // 드리프트가 같이 딸려와 재측위의 의미가 없다.
                        const std::size_t skip = 3;
                        if (!qdesc.empty() && map_kfs.size() > skip) {
                            for (std::size_t mi = 0; mi + skip < map_kfs.size(); ++mi) {
                                const auto& mk = map_kfs[mi];
                                if (mk.desc.empty()) continue;
                                std::vector<std::vector<cv::DMatch>> knn;
                                bf.knnMatch(mk.desc, qdesc, knn, 2);

                                std::vector<cv::Point3f> obj;
                                std::vector<cv::Point2f> img;
                                for (const auto& mm : knn) {
                                    if (mm.size() < 2 ||
                                        mm[0].distance >= 0.75 * mm[1].distance) continue;
                                    const int qi = mm[0].queryIdx, ti = mm[0].trainIdx;
                                    if (qi < 0 || ti < 0) continue;
                                    if (!mk.has_xyz[static_cast<std::size_t>(qi)]) continue;
                                    obj.push_back(mk.xyz[static_cast<std::size_t>(qi)]);
                                    img.push_back(qkp[static_cast<std::size_t>(ti)].pt);
                                }
                                if (obj.size() < static_cast<std::size_t>(reloc_min_inliers))
                                    continue;

                                cv::Mat rvec, tvec, inl;
                                if (!cv::solvePnPRansac(obj, img, Kcv, cv::Mat(), rvec, tvec,
                                                        false, 300, 3.0F, 0.99, inl,
                                                        cv::SOLVEPNP_ITERATIVE)) continue;
                                if (inl.rows <= best_in) continue;

                                cv::Mat Rcv; cv::Rodrigues(rvec, Rcv);
                                wme::Mat3 Rm;
                                for (int a2 = 0; a2 < 3; ++a2)
                                    for (int b2 = 0; b2 < 3; ++b2)
                                        Rm(a2, b2) = Rcv.at<double>(a2, b2);
                                // PnP 는 T_cur_kf 를 준다. 월드 포즈는 그 키프레임의
                                // 월드 포즈에 역을 붙인 것이다.
                                const wme::SE3 T_cur_kf(wme::SO3(Rm),
                                    wme::Vec3(tvec.at<double>(0), tvec.at<double>(1),
                                              tvec.at<double>(2)));
                                best_in = inl.rows;
                                best_T_world = mk.T_world * T_cur_kf.inverse();
                            }
                        }

                        if (best_in >= reloc_min_inliers) {
                            T_world_cam = best_T_world;
                            ++reloc_ok;
                            bad_streak = 0;
                            // 재측위 직후에는 참조를 반드시 새로 잡는다. 옛 참조에
                            // 상대적인 예측을 그대로 쓰면 방금 지운 드리프트가
                            // 다음 프레임에 되살아난다.
                            ref = f;
                            T_world_ref = T_world_cam;
                            ref_stamp = rr.stamp;
                            T_cur_ref_prev = wme::SE3::identity();
                            velocity = wme::SE3::identity();
                            ++kf_switches;
                        }
                    }
                }

                // 기하 정합성이 나쁘면 키프레임을 갈아 끼운다.
                //
                // 25.7 은 이 신호를 융합 가중으로 소비하면 15 개 구성 중 8 개가
                // 나빠진다는 것을 쟀다 - Tier 0 를 낮춰 봐야 3~15 배 부정확한
                // tier 로 무게가 갈 뿐이기 때문이다. 그래서 여기서는 "저걸 믿어라"
                // 가 아닌 행동을 준다: 참조가 더 이상 좋은 기준이 아니라고 보고
                // 바꾼다. 이 행동은 다른 tier 를 필요로 하지 않는다.
                const bool geom_bad =
                    depth_gate_kf && r.value().depth_consistency >= 0.0 &&
                    !r.value().depthConsistent();
                if (geom_bad) ++kf_geom_switches;
                if (T_ref_cur.translation().norm() > kf_dist || geom_bad) {
                    // 지도에 남긴다. 재측위가 돌아갈 좌표계는 여기서 만들어진다.
                    if (reloc_after > 0) {
                        MapKeyFrame mk;
                        mk.T_world = T_world_cam;
                        orb->detectAndCompute(gray, cv::noArray(), mk.kp, mk.desc);
                        mk.xyz.resize(mk.kp.size());
                        mk.has_xyz.assign(mk.kp.size(), false);
                        for (std::size_t pi = 0; pi < mk.kp.size(); ++pi) {
                            const int u = static_cast<int>(std::lround(mk.kp[pi].pt.x));
                            const int v2 = static_cast<int>(std::lround(mk.kp[pi].pt.y));
                            if (u < 0 || v2 < 0 || u >= depth.cols || v2 >= depth.rows) continue;
                            const float z = depth.at<float>(v2, u);
                            if (!(z > g_depth_min) || z > g_depth_max) continue;
                            mk.xyz[pi] = cv::Point3f(
                                static_cast<float>((mk.kp[pi].pt.x - K.cx) * z / K.fx),
                                static_cast<float>((mk.kp[pi].pt.y - K.cy) * z / K.fy), z);
                            mk.has_xyz[pi] = true;
                        }
                        map_kfs.push_back(std::move(mk));
                    }
                    ref = f;
                    T_world_ref = T_world_cam;
                    ref_stamp = rr.stamp;
                    // 새 참조 기준이므로 상대 추정은 항등으로 되돌린다.
                    // 속도는 프레임 간 증분이라 참조가 바뀌어도 그대로 쓴다.
                    T_cur_ref_prev = wme::SE3::identity();
                    ++kf_switches;
                }
            }
        }

        double qx, qy, qz, qw;
        toQuat(T_world_cam.rotation().matrix(), qx, qy, qz, qw);
        const wme::Vec3 t = T_world_cam.translation();
        out << rr.stamp << " " << t.x() << " " << t.y() << " " << t.z() << " "
            << qx << " " << qy << " " << qz << " " << qw << "\n";
        ++used;

        if (used % 50 == 0) {
            std::cout << "  " << used << " 프레임  실패 " << failed
                      << "  키프레임교체 " << kf_switches << "\n";
        }
    }

    out.close();
    if (reloc_after > 0) {
        std::cout << "재측위 시도 " << reloc_tried << "  성공 " << reloc_ok << "\n";
    }
    if (depth_gate_kf) {
        std::cout << "기하 게이트로 인한 키프레임 교체 " << kf_geom_switches << "\n";
    }
    std::cout << "프레임 " << used << "  정렬실패 " << failed
              << "  키프레임교체 " << kf_switches << "\n";
    if (stat_n > 0) {
        std::cout << "평균 측광 RMSE " << (rmse_sum / stat_n)
                  << "  inlier " << (inlier_sum / stat_n)
                  << "  점수 " << (point_sum / stat_n)
                  << "  관측가능 DOF " << (dof_sum / stat_n) << "\n";
    }
    if (yolo_n > 0) {
        std::cout << "YOLO: 프레임당 검출 " << (det_sum / yolo_n)
                  << "  마스킹 면적 " << (100.0 * masked_sum / yolo_n) << " %"
                  << "  추론 " << (yolo_ms / yolo_n) << " ms\n";
        if (mask_mode != "class") {
            // 지운 면적의 출처. 판정이 안 돈 토큰이 지운 몫은 "증거" 가 아니라
            // 사전분포를 판결로 쓴 것이다.
            std::cout << "  마스크 출처: 관측중 " << (100.0 * mask_obs_sum / yolo_n)
                      << " %  관측끊김 " << (100.0 * mask_stale_sum / yolo_n)
                      << " %  보류(판정전) " << (100.0 * mask_unj_sum / yolo_n) << " %\n";
            // 새 토큰이 생긴 이유. 게이트 탈락이 지배적이면 병목은 믿음이 아니라
            // 연관이고, 그때는 시퀀스를 늘려도 믿음이 수렴하지 않는다.
            const double tot = assoc_match + assoc_nocand + assoc_gated + assoc_unass;
            std::cout << "  연관: 매칭 " << (100.0 * assoc_match / std::max(1.0, tot))
                      << " %  후보없음 " << (100.0 * assoc_nocand / std::max(1.0, tot))
                      << " %  게이트탈락 " << (100.0 * assoc_gated / std::max(1.0, tot))
                      << " %  할당실패 " << (100.0 * assoc_unass / std::max(1.0, tot)) << " %";
            if (assoc_gated > 0) {
                std::cout << "  | 탈락 후보 평균 maha " << (gated_maha / assoc_gated)
                          << " 거리 " << (1000.0 * gated_dist / assoc_gated) << " mm";
            }
            std::cout << "\n";
        }
    }
    if (agent_n > 0) {
        // 믿음이 안 오르는 이유를 여기서 가른다.
        //   관측수가 적다  -> 토큰 수명(연관) 문제. 시퀀스를 늘려도 소용없다.
        //   관측수는 많은데 믿음이 낮다 -> 증거 속도 문제. 게인을 봐야 한다.
        std::cout << "자율개체: 프레임당 " << (agent_count_sum / agent_n) << " 개"
                  << "  평균 static_belief " << (agent_static_sum / agent_n)
                  << " (최대 " << agent_best_sb << ")"
                  << "  평균 관측수 " << (agent_obs_sum / agent_n)
                  << " (최대 " << agent_best_obs << ")\n";
    }
    if (zn > 0) {
        const double d = disp_sum / zn, s = sigma_sum / zn;
        std::cout << "정적 판정 입력: 평균 변위 " << (d * 1000.0) << " mm"
                  << "  평균 위치sigma " << (s * 1000.0) << " mm"
                  << "  비 z0=" << (d / std::max(1e-9, s)) << "\n";
    }
    if (stable_n > 0) {
        std::cout << "안정 랜드마크: 프레임당 "
                  << (static_cast<double>(stable_frame_sum) / stable_n)
                  << " 개 (자율개체 "
                  << (static_cast<double>(stable_agent_sum) / stable_n) << ")\n";
    }
    if (!agent_lifetimes.empty()) {
        // 사라진 자율개체 토큰이 죽을 때 몇 번 관측됐는지. 이 분포가 좌측에
        // 쏠려 있으면 병목은 증거 속도가 아니라 연관이다.
        std::sort(agent_lifetimes.begin(), agent_lifetimes.end());
        double sum = 0.0;
        for (auto v : agent_lifetimes) sum += v;
        const std::size_t n = agent_lifetimes.size();
        std::size_t le3 = 0;
        for (auto v : agent_lifetimes) if (v <= 3) ++le3;
        std::cout << "소멸 자율개체 토큰 " << n
                  << "  관측수 평균 " << (sum / n)
                  << "  중앙값 " << agent_lifetimes[n / 2]
                  << "  최대 " << agent_lifetimes.back()
                  << "  (<=3 관측: " << (100.0 * le3 / n) << " %)\n";
    }
    std::cout << "저장: " << out_path << "\n";
    return 0;
}
