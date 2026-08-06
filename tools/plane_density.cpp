// 평면을 성좌 노드로 쓸 수 있는가 - 밀도와 **안정성** 을 함께 잰다.
//
// 25.12 는 TCG 의 실내 recall 한계를 노드 품질로 좁혔다: 성좌는 실재하고, 위치가
// 정확하고, 안정적으로 검출되는 물체 4 개를 요구하는데 사무실 시점당 잘 잡히는
// COCO 물체가 약 3 개다. 그래서 다른 프리미티브를 본다. 평면은 이미 SPA 가
// 뽑고 있고 사무실에 부족하지 않다.
//
// **다만 개수만 재면 안 된다.** 평면의 centroid 는 "보이는 부분" 의 무게중심이라
// 시점이 바뀌면 같은 벽도 다른 점을 준다. 성좌 서명은 쌍거리 스펙트럼이므로
// 노드가 시점에 따라 미끄러지면 서명 자체가 흔들린다. 25.11/25.12 에서 창과
// 문턱이 각각 위치와 실재성을 대가로 양을 늘렸다가 실패한 것과 같은 자리다.
//
// 그래서 두 가지를 잰다:
//   1) 키프레임당 평면 수 (밀도)
//   2) **같은 물리 평면이 두 시점에서 주는 centroid 가 얼마나 움직이는가** (안정성)
// 2 번은 진리값 포즈로 두 키프레임을 같은 좌표계에 놓고, 법선과 거리로 같은
// 평면을 짝지은 뒤 centroid 거리를 본다. 이 값이 크면 평면 centroid 는
// 성좌 노드로 못 쓴다 - 개수가 아무리 많아도.
//
// 사용:
//   wme_plane_density <시퀀스> [--kf-stride N] [--max-kf N]

#include "wme/core/SE3.hpp"
#include "wme/geometry/PlaneExtractor.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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

struct GtRow { double stamp; wme::SE3 T; };

wme::Mat3 quatToR(double x, double y, double z, double w) {
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    x /= n; y /= n; z /= n; w /= n;
    wme::Mat3 R;
    R << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
         2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y);
    return R;
}

std::vector<GtRow> readGt(const std::string& p) {
    std::vector<GtRow> out;
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

bool gtAt(const std::vector<GtRow>& g, double t, wme::SE3& out) {
    double bd = 0.02; int best = -1;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double d = std::abs(g[i].stamp - t);
        if (d <= bd) { bd = d; best = static_cast<int>(i); }
    }
    if (best < 0) return false;
    out = g[static_cast<std::size_t>(best)].T;
    return true;
}

int nearest(const std::vector<Row>& rows, double stamp, double tol) {
    int best = -1; double bd = tol;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const double d = std::abs(rows[i].stamp - stamp);
        if (d <= bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

struct KF {
    wme::SE3 T_world;
    std::vector<wme::Plane> planes;   // 월드 좌표계
    std::vector<wme::Vec3>  corners;  // 세 평면의 교점 (월드)
};

// 세 평면의 교점. n_i . x = d_i 를 푼다.
//
// centroid 와 결정적으로 다른 점: 이 점은 각 평면이 **얼마나 보이는지와 무관**
// 하다. 벽의 절반만 보여도 세 평면이 정의하는 모서리는 같은 자리에 있다.
// 25.13 이 centroid 를 기각한 이유가 바로 가시 영역 의존성이었다.
//
// 법선이 서로 가까우면 교점이 수치적으로 폭발하므로 조건수로 막는다. 그리고
// 장면 밖으로 튀어나간 해는 버린다 - 세 평면을 무한히 연장하면 교점은 언제나
// 존재하지만, 그것이 관측된 기하 안에 있다는 보장은 없다.
bool intersect3(const wme::Plane& a, const wme::Plane& b, const wme::Plane& c,
                double max_range, wme::Vec3& out) {
    wme::Mat3 N;
    N.row(0) = a.normal.transpose();
    N.row(1) = b.normal.transpose();
    N.row(2) = c.normal.transpose();
    const double det = N.determinant();
    // |det| 는 세 법선이 이루는 평행육면체 부피다. 0.3 이면 세 축이 충분히 벌어진
    // 경우만 남는다 (직교하면 1).
    if (std::abs(det) < 0.3) return false;
    const wme::Vec3 d(a.distance, b.distance, c.distance);
    out = N.colPivHouseholderQr().solve(d);
    if (!out.allFinite()) return false;
    // 세 평면 모두에서 실제로 가까워야 한다 - 수치해가 아니라 기하적 모서리인지.
    if (std::abs(a.signedDistance(out)) > 0.05) return false;
    if (std::abs(b.signedDistance(out)) > 0.05) return false;
    if (std::abs(c.signedDistance(out)) > 0.05) return false;
    return out.norm() < max_range;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "사용: wme_plane_density <시퀀스> [--kf-stride N] [--max-kf N]\n";
        return 2;
    }
    const std::string root = argv[1];
    int stride = 20, max_kf = 60;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--kf-stride") stride = std::atoi(argv[i + 1]);
        else if (k == "--max-kf") max_kf = std::atoi(argv[i + 1]);
    }

    const auto rgb = readIndex(root + "/rgb.txt");
    const auto dep = readIndex(root + "/depth.txt");
    const auto gt  = readGt(root + "/groundtruth.txt");
    if (rgb.empty() || dep.empty() || gt.empty()) {
        std::cerr << "입력을 읽지 못했다\n";
        return 1;
    }

    double fx, fy, cx, cy;
    cv::Vec<double, 5> dist;
    if (root.find("freiburg2") != std::string::npos) {
        fx = 520.908620; fy = 521.007327; cx = 325.141442; cy = 249.701764;
        dist = {0.2312, -0.7849, -0.0033, -0.0001, 0.9172};
    } else if (root.find("freiburg3") != std::string::npos) {
        fx = 535.4; fy = 539.2; cx = 320.1; cy = 247.6; dist = {0, 0, 0, 0, 0};
    } else {
        fx = 517.306408; fy = 516.469215; cx = 318.643040; cy = 255.313989;
        dist = {0.2624, -0.9531, -0.0054, 0.0026, 1.1633};
    }
    const cv::Matx33d Kcv(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat m1, m2;
    cv::initUndistortRectifyMap(Kcv, dist, cv::Mat(), Kcv, cv::Size(640, 480),
                                CV_16SC2, m1, m2);
    wme::CameraIntrinsics K;
    K.width = 640; K.height = 480; K.fx = fx; K.fy = fy; K.cx = cx; K.cy = cy;

    wme::PlaneExtractor extractor;
    std::vector<KF> kfs;

    for (std::size_t fi = 0; fi < rgb.size() && static_cast<int>(kfs.size()) < max_kf;
         fi += static_cast<std::size_t>(stride)) {
        const int di = nearest(dep, rgb[fi].stamp, 0.02);
        if (di < 0) continue;
        cv::Mat draw = cv::imread(root + "/" + dep[static_cast<std::size_t>(di)].file,
                                  cv::IMREAD_UNCHANGED);
        if (draw.empty()) continue;
        wme::SE3 T;
        if (!gtAt(gt, rgb[fi].stamp, T)) continue;

        cv::Mat d32, du;
        draw.convertTo(d32, CV_32F, 1.0 / 5000.0);
        cv::remap(d32, du, m1, m2, cv::INTER_NEAREST);

        const auto r = extractor.extract(du, K);
        if (!r.ok()) continue;

        KF kf;
        kf.T_world = T;
        // 평면을 월드로 옮긴다. n_w = R n_c, d_w = d_c + n_w . t
        for (const auto& pl : r.value()) {
            wme::Plane w = pl;
            w.normal = T.rotation() * pl.normal;
            w.distance = pl.distance + w.normal.dot(T.translation());
            w.centroid = T * pl.centroid;
            kf.planes.push_back(w);
        }
        // 모서리. 월드 평면으로 계산하고 관측 범위는 카메라 기준으로 본다.
        for (std::size_t i = 0; i < kf.planes.size(); ++i) {
            for (std::size_t j = i + 1; j < kf.planes.size(); ++j) {
                for (std::size_t k = j + 1; k < kf.planes.size(); ++k) {
                    wme::Vec3 x;
                    if (!intersect3(kf.planes[i], kf.planes[j], kf.planes[k], 1e9, x)) continue;
                    // 카메라에서 8 m 안이어야 관측된 기하로 본다.
                    if ((T.inverse() * x).norm() > 8.0) continue;
                    kf.corners.push_back(x);
                }
            }
        }
        kfs.push_back(std::move(kf));
    }

    if (kfs.size() < 2) { std::cerr << "키프레임이 부족하다\n"; return 1; }

    long long sum = 0, ge4 = 0, csum = 0, cge4 = 0;
    for (const auto& k : kfs) { sum += static_cast<long long>(k.planes.size());
                                ge4 += (k.planes.size() >= 4);
                                csum += static_cast<long long>(k.corners.size());
                                cge4 += (k.corners.size() >= 4); }

    // --- 안정성: 같은 물리 평면이 두 시점에서 주는 centroid 거리 --------------
    // 같은 평면인지는 법선(<15도)과 거리(<0.15 m)로 판정한다. 그 판정이 맞다면
    // centroid 가 얼마나 움직이는지가 곧 "성좌 노드로 쓸 수 있는가" 다.
    std::vector<double> cdrift;
    for (std::size_t a = 0; a + 1 < kfs.size(); ++a) {
        for (std::size_t b = a + 1; b < kfs.size(); ++b) {
            const double base = (kfs[a].T_world.translation()
                                 - kfs[b].T_world.translation()).norm();
            if (base < 0.3 || base > 2.0) continue;
            // 모서리끼리는 가장 가까운 상대를 짝으로 본다. 같은 모서리라면
            // 그 거리가 곧 미끄러짐이고, 다른 모서리라면 큰 값이 나온다.
            for (const auto& ca : kfs[a].corners) {
                double best = 1e18;
                for (const auto& cb : kfs[b].corners) best = std::min(best, (ca - cb).norm());
                if (best < 1e17) cdrift.push_back(best);
            }
        }
    }

    std::vector<double> drift;
    for (std::size_t a = 0; a + 1 < kfs.size(); ++a) {
        for (std::size_t b = a + 1; b < kfs.size(); ++b) {
            // 시점이 충분히 달라야 의미가 있다.
            const double base = (kfs[a].T_world.translation()
                                 - kfs[b].T_world.translation()).norm();
            if (base < 0.3 || base > 2.0) continue;
            for (const auto& pa : kfs[a].planes) {
                for (const auto& pb : kfs[b].planes) {
                    if (pa.normal.dot(pb.normal) < std::cos(15.0 * M_PI / 180.0)) continue;
                    if (std::abs(pa.distance - pb.distance) > 0.15) continue;
                    drift.push_back((pa.centroid - pb.centroid).norm());
                }
            }
        }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n" << root << "   키프레임 " << kfs.size() << "\n";
    std::cout << std::string(72, '=') << "\n";
    std::cout << "1) 밀도:  평면/키프레임 평균 "
              << static_cast<double>(sum) / static_cast<double>(kfs.size())
              << "   4개 이상 " << 100.0 * static_cast<double>(ge4) /
                                    static_cast<double>(kfs.size()) << " %\n";
    if (drift.empty()) {
        std::cout << "2) 안정성: 짝지어진 평면 쌍이 없다 - 판정 불가\n";
        return 0;
    }
    std::sort(drift.begin(), drift.end());
    const auto q = [&](double f) { return drift[static_cast<std::size_t>(f * (drift.size() - 1))]; };
    std::cout << "2) 안정성: 같은 평면의 centroid 이동  표본 " << drift.size()
              << "\n     중앙값 " << q(0.5) << " m   p90 " << q(0.9)
              << " m   최대 " << drift.back() << " m\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << "3) 모서리(세 평면 교점):  개수/키프레임 "
              << static_cast<double>(csum) / static_cast<double>(kfs.size())
              << "   4개 이상 " << 100.0 * static_cast<double>(cge4) /
                                    static_cast<double>(kfs.size()) << " %\n";
    if (!cdrift.empty()) {
        std::sort(cdrift.begin(), cdrift.end());
        const auto cq = [&](double f) {
            return cdrift[static_cast<std::size_t>(f * (cdrift.size() - 1))]; };
        std::cout << "   최근접 모서리 거리  중앙값 " << cq(0.5)
                  << " m   p90 " << cq(0.9) << " m   표본 " << cdrift.size() << "\n";
    } else {
        std::cout << "   모서리 쌍이 없다 - 판정 불가\n";
    }
    std::cout << "\n성좌 서명은 쌍거리 스펙트럼이다. 노드가 이만큼 미끄러지면\n"
                 "같은 장소에서도 서명이 달라진다 - COCO 물체의 위치 오차(수 cm)와\n"
                 "비교해서 읽어야 한다.\n";
    return 0;
}
