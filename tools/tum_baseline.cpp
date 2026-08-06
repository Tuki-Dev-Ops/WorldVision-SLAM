// 고전 특징점 기반 RGB-D 오도메트리 - WME 의 대조군.
//
// 왜 이것을 기준으로 두는가. WME 의 중심 주장은 "기술자(descriptor)를 쓰지
// 않는다" 이다(README, 02-correspondence-problem.md). 그렇다면 정직한 대조군은
// 바로 그 기술자 파이프라인이어야 한다: ORB 검출 -> 해밍 기술자 정합 ->
// RANSAC PnP. 이 도구가 그것이다.
//
// **이것은 ORB-SLAM3 가 아니다.** 루프 클로저도, 번들 조정도, 지역 지도도 없다.
// ORB-SLAM3 의 *추적 프런트엔드* 에 해당하는 것만 있으므로, 비교는 "오도메트리
// 대 오도메트리" 로만 유효하다. 05-research-program.md 2절이 자체 구현 베이스
// 라인을 경고하는 이유가 이것이고, 그래서 이름에 ORB-SLAM3 를 쓰지 않는다.
//
// 공정성을 위해 wme_tum_odometry 와 다음을 **전부** 공유한다. 하나라도 다르면
// 그 차이가 알고리즘 차이로 오독된다:
//   - 시퀀스별 내부 파라미터와 왜곡 계수, 왜곡 보정 맵
//   - 깊이 스케일(5000) 과 rgb/depth 시각 연관(20 ms)
//   - 키프레임 교체 규칙과 기본 거리(0.03 m)
//   - 실패 시 직전 포즈 유지 규약
//
// 사용:
//   wme_tum_baseline <시퀀스경로> <출력.txt> [--kf-dist M] [--max-frames N]
//                    [--features N] [--ratio R] [--diag <csv>]

#include "wme/core/SE3.hpp"
#include "dataset_calib.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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

// 회전행렬 -> 쿼터니언 (x,y,z,w). tum_odometry.cpp 와 동일한 구현을 쓴다.
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

// 키프레임 상태. 3D 점은 키프레임 카메라 좌표계에서 만든다.
struct KeyFrame {
    cv::Mat                   gray;
    cv::Mat                   desc;
    std::vector<cv::KeyPoint> kp;
    std::vector<cv::Point3f>  xyz;     // kp[i] 의 3D 점. z<=0 이면 무효
    std::vector<bool>         has_xyz;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "사용: wme_tum_baseline <시퀀스경로> <출력.txt> "
                     "[--kf-dist M] [--max-frames N] [--features N] [--ratio R] "
                     "[--diag <csv>]\n";
        return 2;
    }
    const std::string root     = argv[1];
    const std::string out_path = argv[2];

    double kf_dist    = 0.03;   // wme_tum_odometry 기본값과 동일
    int    max_frames = 0;
    // ORB-SLAM3 의 TUM 설정과 같은 값. 임의로 고른 수가 아니라는 점이 중요하다 -
    // 약하게 맞춘 대조군은 비교 자체를 무효로 만든다.
    int    n_features = 1000;
    double scale_factor = 1.2;
    int    n_levels   = 8;
    double lowe_ratio = 0.75;   // Lowe 비율 검정. ORB 정합의 표준값
    std::string diag_path;
    double depth_min_cli = -1.0, depth_max_cli = -1.0;   // <0 = 지정 안 함

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--kf-dist")         kf_dist = std::atof(argv[i + 1]);
        else if (k == "--max-frames") max_frames = std::atoi(argv[i + 1]);
        else if (k == "--features")   n_features = std::atoi(argv[i + 1]);
        else if (k == "--ratio")      lowe_ratio = std::atof(argv[i + 1]);
        else if (k == "--levels")     n_levels = std::atoi(argv[i + 1]);
        else if (k == "--diag")       diag_path = argv[i + 1];
        else if (k == "--depth-min")  depth_min_cli = std::atof(argv[i + 1]);
        else if (k == "--depth-max")  depth_max_cli = std::atof(argv[i + 1]);
    }

    const auto rgb_rows   = readIndex(root + "/rgb.txt");
    const auto depth_rows = readIndex(root + "/depth.txt");
    if (rgb_rows.empty() || depth_rows.empty()) {
        std::cerr << "rgb.txt / depth.txt 를 읽지 못했다: " << root << "\n";
        return 1;
    }

    // 내부 파라미터/왜곡/깊이 범위. 값은 dataset_calib.hpp 한 곳에만 있고
    // wme_tum_odometry 도 같은 함수를 부른다 - 두 도구가 같은 값을 쓴다는 것이
    // 이 비교의 전제이므로, 그 전제를 복사본 두 개가 지키게 두지 않는다.
    // calib.txt 가 있으면 그것이 이긴다.
    wme_tools::DatasetCalib dc;
    if (!wme_tools::resolveCalib(root, dc)) return 1;
    double fx = dc.K.fx, fy = dc.K.fy, cx = dc.K.cx, cy = dc.K.cy;
    cv::Vec<double, 5> dist = dc.dist;
    double kDepthScale = dc.depth_scale;
    int    kW = dc.K.width, kH = dc.K.height;
    double depth_min = dc.depth_min, depth_max = dc.depth_max;
    if (depth_min_cli > 0.0) depth_min = depth_min_cli;
    if (depth_max_cli > 0.0) depth_max = depth_max_cli;
    std::cout << "깊이 유효 범위: [" << depth_min << ", " << depth_max << "] m\n";

    const cv::Matx33d Kcv(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv, cv::Size(kW, kH),
                                CV_16SC2, map1, map2);
    // 보정 후 좌표계이므로 PnP 에는 왜곡 0 을 넘긴다.
    const cv::Mat zero_dist = cv::Mat::zeros(5, 1, CV_64F);

    auto orb = cv::ORB::create(n_features, static_cast<float>(scale_factor), n_levels);
    cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/false);

    // 열리지 않았는데 계속 돌면 마지막에 "저장했다" 를 찍고 끝난다. 그 조합이
    // 제일 나쁘다 - 실패가 성공처럼 보고된다(06-results.md §19). 여기서 죽는다.
    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "출력 파일을 열 수 없다 (상위 폴더가 없는가?): " << out_path << "\n";
        return 1;
    }
    out << "# timestamp tx ty tz qx qy qz qw\n";
    out << std::fixed << std::setprecision(6);

    std::ofstream diag;
    if (!diag_path.empty()) {
        diag.open(diag_path);
        if (!diag) {
            std::cerr << "진단 CSV 를 열 수 없다: " << diag_path << "\n";
            return 1;
        }
        diag << "timestamp,ref_timestamp,keypoints,matches,pnp_inliers,inlier_ratio,"
                "reproj_rmse,track_ok,ms,rel_tx,rel_ty,rel_tz,rel_qx,rel_qy,rel_qz,rel_qw\n";
        diag << std::fixed << std::setprecision(6);
    }

    KeyFrame kf;
    bool     have_kf = false;
    double   kf_stamp = 0.0;
    wme::SE3 T_world_kf  = wme::SE3::identity();
    wme::SE3 T_world_cam = wme::SE3::identity();

    int used = 0, tracked = 0, lost = 0, kf_switches = 0;
    double kp_sum = 0.0, match_sum = 0.0, inlier_sum = 0.0, ms_sum = 0.0;
    int stat_n = 0;

    for (const auto& rr : rgb_rows) {
        if (max_frames > 0 && used >= max_frames) break;

        const std::string rgb_file = root + "/" + rr.file;
        if (!fileExists(rgb_file)) continue;
        const int di = nearest(depth_rows, rr.stamp, 0.02);
        if (di < 0) continue;
        const std::string depth_file =
            root + "/" + depth_rows[static_cast<std::size_t>(di)].file;
        if (!fileExists(depth_file)) continue;

        cv::Mat bgr = cv::imread(rgb_file, cv::IMREAD_COLOR);
        cv::Mat dep = cv::imread(depth_file, cv::IMREAD_UNCHANGED);
        if (bgr.empty() || dep.empty()) continue;

        const auto t_start = std::chrono::steady_clock::now();

        cv::Mat bgr_u, dep_u;
        cv::remap(bgr, bgr_u, map1, map2, cv::INTER_LINEAR);
        // 깊이는 최근접으로 보정한다. 선형 보간은 물체 경계에서 존재하지 않는
        // 중간 깊이를 만들어 내고, 그 점이 바로 PnP 의 3D 대응이 된다.
        cv::remap(dep, dep_u, map1, map2, cv::INTER_NEAREST);

        cv::Mat gray;
        cv::cvtColor(bgr_u, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb->detectAndCompute(gray, cv::noArray(), kp, desc);

        // 깊이 역투영. 깊이 0 은 TUM 의 무효값이다.
        std::vector<cv::Point3f> xyz(kp.size());
        std::vector<bool>        has(kp.size(), false);
        for (std::size_t i = 0; i < kp.size(); ++i) {
            const int u = static_cast<int>(std::lround(kp[i].pt.x));
            const int v = static_cast<int>(std::lround(kp[i].pt.y));
            if (u < 0 || v < 0 || u >= dep_u.cols || v >= dep_u.rows) continue;
            const double z = static_cast<double>(dep_u.at<std::uint16_t>(v, u)) / kDepthScale;
            // 유효 범위는 데이터셋이 정한다. 상수로 박아 두면 TUM 밖에서
            // 기준선이 조용히 굶는다 - dataset_calib.hpp 머리말 참조.
            if (!(z > depth_min) || z > depth_max) continue;
            xyz[i] = cv::Point3f(static_cast<float>((kp[i].pt.x - cx) * z / fx),
                                 static_cast<float>((kp[i].pt.y - cy) * z / fy),
                                 static_cast<float>(z));
            has[i] = true;
        }

        bool     track_ok = false;
        wme::SE3 T_cur_kf = wme::SE3::identity();
        int      n_match = 0, n_inlier = 0;
        double   reproj_rmse = 0.0;

        if (have_kf && !desc.empty() && !kf.desc.empty()) {
            // 키프레임 -> 현재 방향으로 정합한다. 3D 는 키프레임에서 오고
            // 2D 는 현재에서 오므로 PnP 가 바로 T_cur_kf 를 준다.
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(kf.desc, desc, knn, 2);

            std::vector<cv::Point3f> obj;
            std::vector<cv::Point2f> img;
            for (const auto& m : knn) {
                if (m.size() < 2) continue;
                if (m[0].distance >= lowe_ratio * m[1].distance) continue;  // Lowe 비율
                const int qi = m[0].queryIdx, ti = m[0].trainIdx;
                if (qi < 0 || ti < 0 || !kf.has_xyz[static_cast<std::size_t>(qi)]) continue;
                obj.push_back(kf.xyz[static_cast<std::size_t>(qi)]);
                img.push_back(kp[static_cast<std::size_t>(ti)].pt);
            }
            n_match = static_cast<int>(obj.size());

            if (n_match >= 6) {
                cv::Mat rvec, tvec, inliers;
                const bool ok = cv::solvePnPRansac(
                    obj, img, cv::Mat(Kcv), zero_dist, rvec, tvec,
                    /*useExtrinsicGuess=*/false, /*iterations=*/200,
                    /*reprojErr=*/3.0F, /*confidence=*/0.99, inliers,
                    cv::SOLVEPNP_ITERATIVE);
                n_inlier = ok ? inliers.rows : 0;
                if (ok && n_inlier >= 10) {
                    // 인라이어만으로 LM 정제. RANSAC 해는 최소집합 기반이라
                    // 그대로 쓰면 남은 인라이어의 정보를 버리게 된다.
                    std::vector<cv::Point3f> oin;
                    std::vector<cv::Point2f> iin;
                    oin.reserve(static_cast<std::size_t>(n_inlier));
                    iin.reserve(static_cast<std::size_t>(n_inlier));
                    for (int i = 0; i < n_inlier; ++i) {
                        const int k = inliers.at<int>(i);
                        oin.push_back(obj[static_cast<std::size_t>(k)]);
                        iin.push_back(img[static_cast<std::size_t>(k)]);
                    }
                    cv::solvePnPRefineLM(oin, iin, cv::Mat(Kcv), zero_dist, rvec, tvec);

                    std::vector<cv::Point2f> proj;
                    cv::projectPoints(oin, rvec, tvec, cv::Mat(Kcv), zero_dist, proj);
                    double s2 = 0.0;
                    for (std::size_t i = 0; i < proj.size(); ++i) {
                        const double dx = proj[i].x - iin[i].x;
                        const double dy = proj[i].y - iin[i].y;
                        s2 += dx * dx + dy * dy;
                    }
                    reproj_rmse = std::sqrt(s2 / std::max<std::size_t>(1, proj.size()));

                    cv::Mat Rcv;
                    cv::Rodrigues(rvec, Rcv);
                    wme::Mat3 R;
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) R(r, c) = Rcv.at<double>(r, c);
                    const wme::Vec3 t(tvec.at<double>(0), tvec.at<double>(1),
                                      tvec.at<double>(2));
                    T_cur_kf = wme::SE3(wme::SO3(R), t);
                    track_ok = true;
                }
            }
        }

        const auto t_end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (track_ok) {
            T_world_cam = T_world_kf * T_cur_kf.inverse();
            ++tracked;
        } else if (have_kf) {
            // 추적 실패 - 직전 포즈를 유지한다. 원점으로 되돌리면 궤적이 튀어
            // ATE 가 실패를 과장한다 (tum_fusion.cpp 와 같은 규약).
            ++lost;
        }

        double qx, qy, qz, qw;
        toQuat(T_world_cam.rotation().matrix(), qx, qy, qz, qw);
        const wme::Vec3 tw = T_world_cam.translation();
        out << rr.stamp << " " << tw.x() << " " << tw.y() << " " << tw.z()
            << " " << qx << " " << qy << " " << qz << " " << qw << "\n";

        if (diag.is_open()) {
            double rqx, rqy, rqz, rqw;
            toQuat(T_cur_kf.rotation().matrix(), rqx, rqy, rqz, rqw);
            const wme::Vec3 rt = T_cur_kf.translation();
            diag << rr.stamp << "," << (have_kf ? kf_stamp : 0.0) << ","
                 << kp.size() << "," << n_match << "," << n_inlier << ","
                 << (n_match ? static_cast<double>(n_inlier) / n_match : 0.0) << ","
                 << reproj_rmse << "," << (track_ok ? 1 : 0) << "," << ms << ","
                 << rt.x() << "," << rt.y() << "," << rt.z() << ","
                 << rqx << "," << rqy << "," << rqz << "," << rqw << "\n";
        }

        if (track_ok) {
            kp_sum += static_cast<double>(kp.size());
            match_sum += n_match;
            inlier_sum += n_inlier;
            ms_sum += ms;
            ++stat_n;
        }

        // 키프레임 교체 - wme_tum_odometry 와 동일한 규칙.
        const bool need_kf = !have_kf || !track_ok ||
                             T_cur_kf.inverse().translation().norm() > kf_dist;
        if (need_kf) {
            kf.gray = gray;
            kf.desc = desc;
            kf.kp = kp;
            kf.xyz = xyz;
            kf.has_xyz = has;
            kf_stamp = rr.stamp;
            T_world_kf = T_world_cam;
            if (have_kf) ++kf_switches;
            have_kf = true;
        }

        ++used;
        if (used % 50 == 0) std::cout << "  " << used << " 프레임\n";
    }

    std::cout << "프레임 " << used << "  추적성공 " << tracked << "  실패 " << lost
              << "  키프레임교체 " << kf_switches << "\n";
    if (stat_n) {
        std::cout << "평균  특징점 " << kp_sum / stat_n
                  << "  정합 " << match_sum / stat_n
                  << "  인라이어 " << inlier_sum / stat_n
                  << "  " << ms_sum / stat_n << " ms/frame\n";
    }
    std::cout << "저장: " << out_path << "\n";
    return 0;
}
