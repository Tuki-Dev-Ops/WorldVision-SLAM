// WorldVision-SLAM — 벤치마크 뷰어 (네이티브 실행 파일)
//
// 왼쪽에 고전 기술자 파이프라인, 오른쪽에 WME 를 놓고 같은 데이터셋을 같은
// 속도로 재생하면서, 각자가 추정한 포즈로 **깊이맵을 3D 로 재구성** 해 나란히
// 그린다. 드리프트는 숫자로만 보면 추상적이지만 점군에서는 곧바로 보인다 -
// 궤적이 어긋난 쪽은 같은 벽이 두 겹으로 쌓인다.
//
// 계측에 대하여
// -------------
// **ATE/RPE/속도를 여기서 계산하지 않는다.** results/bench/viewer.tsv 에서
// 읽는다. 그 파일은 python/tools/bench_run.py 가 계산한 값을 그대로 옮긴
// 것이다. 뷰어가 다시 계산하면 화면의 숫자와 문서의 숫자가 갈릴 수 있고,
// 그러면 어느 쪽이 맞는지 아무도 모른다. 매니페스트가 없으면 실행을 거부한다 -
// 0 을 그리고 마는 것이 제일 나쁘다 (06-results.md 19 장).
//
// 화면에서 **계산되는** 것은 표시용 강체 정렬(Kabsch)뿐이고, 그 사실을 화면에
// 적어 둔다. 추정 궤적은 자기 좌표계에 있으므로 정렬 없이 겹쳐 그리면 아무
// 것도 비교되지 않는다. 스케일은 맞추지 않는다 - 맞추면 스케일 드리프트가
// 화면에서 사라진다.
//
// 색에 대하여
// -----------
// 두 시스템 강조색은 눈으로 고르지 않고 검증했다. #FF6B35 / #00D9C0 은
// 색각 이상(deuteranopia) 에서도 OKLab ΔE 16.1 로 갈라지고 정상 시야에서는
// 33.0 이다. 배경 대비는 둘 다 3:1 을 넘는다. 깊이 색은 turbo 를 쓴다 - jet 은
// 밝기가 단조롭지 않아 없는 경계를 만들어 낸다.
//
// 의존성은 OpenCV(core/imgproc/imgcodecs/highgui)와 Eigen 뿐이다.
//
// 조작
//   SPACE  재생/정지        , .  한 프레임
//   N / P  시퀀스           1 / 2  왼쪽 / 오른쪽 모델 교체
//   A / D  궤도 회전        W / S  줌
//   R      처음으로         V  뷰 모드(점군 / 궤적 / 나란히)
//   F      스크린샷         Q / ESC  종료

#include "dataset_calib.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ===========================================================================
// 디자인 토큰
// ===========================================================================
// OpenCV 는 BGR 이다. 아래 주석의 #RRGGBB 가 진짜 값이고 Scalar 는 그 BGR 배열.
const cv::Scalar C_BG      ( 16,  12,  10);   // #0A0C10
const cv::Scalar C_PANEL   ( 29,  22,  18);   // #12161D
const cv::Scalar C_RAISED  ( 41,  32,  26);   // #1A2029
const cv::Scalar C_LINE    ( 54,  43,  35);   // #232B36
const cv::Scalar C_INK     (245, 237, 232);   // #E8EDF5
const cv::Scalar C_INK2    (184, 167, 154);   // #9AA7B8
const cv::Scalar C_INK3    (117, 102,  90);   // #5A6675
const cv::Scalar C_A       ( 53, 107, 255);   // #FF6B35  고전
const cv::Scalar C_B       (192, 217,   0);   // #00D9C0  WME
const cv::Scalar C_GOOD    (140, 214,  61);   // #3DD68C
const cv::Scalar C_WARN    ( 77, 184, 255);   // #FFB84D
const cv::Scalar C_BAD     ( 87,  71, 255);   // #FF4757
const cv::Scalar C_GT      (110, 110, 110);

constexpr int WIN_W = 1920, WIN_H = 1080;
constexpr int PAD = 20;                  // 기본 간격 단위
constexpr int TOPBAR_H = 64;
constexpr int BOTTOM_H = 176;

// 타입 스케일 (FONT_HERSHEY_SIMPLEX 배율, 1920x1080 기준)
constexpr double T_HERO   = 1.55;
constexpr double T_VALUE  = 0.78;
constexpr double T_TITLE  = 0.60;
constexpr double T_BODY   = 0.44;
constexpr double T_LABEL  = 0.36;
constexpr double T_MICRO  = 0.32;

// ===========================================================================
// Turbo 컬러맵
// ===========================================================================
// Google 이 공개한 다항 근사. jet 을 쓰지 않는 이유는 jet 의 밝기가 단조롭지
// 않아 데이터에 없는 경계선을 만들어 내기 때문이다 - 깊이맵에서는 그것이
// 존재하지 않는 벽처럼 보인다.
cv::Scalar turbo(double x) {
    x = std::clamp(x, 0.0, 1.0);
    const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
    const double r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2
                   + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5;
    const double g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2
                   - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5;
    const double b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2
                   + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5;
    return {std::clamp(b, 0.0, 1.0) * 255.0,
            std::clamp(g, 0.0, 1.0) * 255.0,
            std::clamp(r, 0.0, 1.0) * 255.0};
}

// ===========================================================================
// 자료구조
// ===========================================================================

struct Pose {
    double t{0.0};
    Eigen::Vector3d p{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
    Eigen::Isometry3d iso() const {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = q.toRotationMatrix();
        T.translation() = p;
        return T;
    }
};

struct Run {
    std::string system, label, kind, status, traj_file;
    double ate_cm{std::nan("")}, rpe_mm{std::nan("")}, ms{std::nan("")};
    int    frames{0};
    std::vector<Pose> traj;                 // 원본 (추정 좌표계)
    std::vector<Eigen::Isometry3d> aligned; // 표시용 정렬 후 (정답 좌표계)
    std::vector<double> err_cm;             // 프레임별 ATE 오차 (bench_run.py 계산)
};

// 의미 구조. wme_scene_export 가 만든 파일에서 읽는다. 좌표는 카메라계이므로
// 뷰어가 각 시스템의 추정 포즈로 옮긴다 - 같은 관측이 두 시스템에서 어디로
// 가는지가 곧 드리프트다.
struct ScenePlane {
    int frame{0};
    Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()}, centroid{Eigen::Vector3d::Zero()};
    double extent{0.0}, conf{0.0};
};

struct SceneBox {
    int frame{0};
    std::string cls;
    double conf{0.0};
    Eigen::Vector3d center{Eigen::Vector3d::Zero()}, size{Eigen::Vector3d::Ones()};
};

struct Seq {
    std::string name, dataset, dir, gt_file;
    double identity_ate_cm{std::nan("")};
    std::vector<Pose> gt;
    std::vector<std::pair<double, std::string>> rgb, depth;
    std::map<std::string, Run> runs;
    std::vector<std::string> systems;       // 표시 순서 고정
    wme_tools::DatasetCalib calib;
    std::vector<ScenePlane> planes;
    std::vector<SceneBox>   boxes;
    bool loaded{false};
};

// ===========================================================================
// 파싱
// ===========================================================================

std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream ss(s);
    while (std::getline(ss, cur, d)) out.push_back(cur);
    return out;
}

double toD(const std::string& s) {
    try { return std::stod(s); } catch (...) { return std::nan(""); }
}

std::vector<Pose> readTum(const std::string& path) {
    std::vector<Pose> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Pose p;
        double qx, qy, qz, qw;
        if (ss >> p.t >> p.p.x() >> p.p.y() >> p.p.z() >> qx >> qy >> qz >> qw) {
            p.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            out.push_back(p);
        }
    }
    return out;
}

std::vector<std::pair<double, std::string>> readIndex(const std::string& dir,
                                                      const std::string& name) {
    std::vector<std::pair<double, std::string>> out;
    std::ifstream f(dir + "/" + name);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t;
        std::string rel;
        if (ss >> t >> rel) out.emplace_back(t, dir + "/" + rel);
    }
    return out;
}

int nearestIdx(const std::vector<Pose>& v, double t) {
    if (v.empty()) return -1;
    int best = 0;
    double bd = std::abs(v[0].t - t);
    for (std::size_t i = 1; i < v.size(); ++i) {
        const double d = std::abs(v[i].t - t);
        if (d < bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

template <typename T>
int nearestStamp(const std::vector<std::pair<double, T>>& v, double t) {
    if (v.empty()) return -1;
    int best = 0;
    double bd = std::abs(v[0].first - t);
    for (std::size_t i = 1; i < v.size(); ++i) {
        const double d = std::abs(v[i].first - t);
        if (d < bd) { bd = d; best = static_cast<int>(i); }
    }
    return best;
}

// ===========================================================================
// 표시용 강체 정렬
// ===========================================================================
struct Rigid {
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
};

Rigid kabsch(const std::vector<Eigen::Vector3d>& a, const std::vector<Eigen::Vector3d>& b) {
    Rigid out;
    if (a.size() < 3 || a.size() != b.size()) return out;
    Eigen::Vector3d ca = Eigen::Vector3d::Zero(), cb = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < a.size(); ++i) { ca += a[i]; cb += b[i]; }
    ca /= static_cast<double>(a.size());
    cb /= static_cast<double>(b.size());
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < a.size(); ++i) H += (a[i] - ca) * (b[i] - cb).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0) D(2, 2) = -1.0;
    out.R = svd.matrixV() * D * svd.matrixU().transpose();
    out.t = cb - out.R * ca;
    return out;
}

// ===========================================================================
// 궤도 카메라 + z-버퍼 점군 렌더러
// ===========================================================================
// 점을 깊이순으로 정렬하면 30 만 점에서 프레임당 수십 ms 가 든다. z-버퍼는
// O(n) 이고 결과가 같다.
struct Orbit {
    // dist 는 0 으로 시작한다. 0 이 "아직 정해지지 않음" 의 표시이고, 아래
    // 초기화 분기가 데이터셋에 맞는 거리를 넣는다. 기본값을 1.0 으로 두면
    // 그 분기가 최초 프레임에 실행되지 않아 실내 장면에서 카메라가 점군
    // 안쪽에 놓이고, 화면이 통째로 비어 버린다 - 실제로 그렇게 나왔다.
    double yaw{0.6}, pitch{0.40}, dist{0.0};
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};

    // 세계의 "위". TUM/KITTI 의 정답 포즈는 **카메라 좌표계** 라서 +y 가
    // 아래다. 이걸 +y 위로 두면 점군이 통째로 뒤집혀 건물이 화면 위쪽에
    // 매달리고 지면이 사라진다 - 실제로 처음에 그렇게 나왔다.
    Eigen::Vector3d world_up{0, -1, 0};

    Eigen::Matrix3d basis() const {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        // 지면 평면의 두 축 (world_up 에 수직)
        Eigen::Vector3d a = world_up.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = world_up.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = world_up.cross(a).normalized();
        // yaw 로 지면 안에서 방향을 정하고, pitch 만큼 위에서 내려다본다.
        const Eigen::Vector3d ground = (a * sy + b * cy).normalized();
        const Eigen::Vector3d fwd = (ground * cp - world_up * sp).normalized();
        Eigen::Vector3d up = world_up;
        Eigen::Vector3d right = fwd.cross(up).normalized();
        up = right.cross(fwd).normalized();
        Eigen::Matrix3d M;
        M.row(0) = right; M.row(1) = up; M.row(2) = fwd;
        return M;
    }
};

struct Splat {
    Eigen::Vector3f p;
    float depth_norm;      // turbo 입력 (0..1)
    float age;             // 0 = 최신, 1 = 가장 오래됨
    int   seen{0};         // 마지막으로 관측된 프레임. 복셀 병합 시 최신을 남긴다
};

// ---------------------------------------------------------------------------
// 복셀 누적 지도
// ---------------------------------------------------------------------------
// 왜 링버퍼가 아니라 복셀인가.
//
// 프레임마다 5만 점이 들어오므로 링버퍼로는 최근 열 몇 프레임밖에 못 담는다.
// 그러면 화면에는 "지금 보이는 것" 만 남고 지나온 곳은 사라진다 - 그건 SLAM
// 이 아니라 깊이 카메라 뷰어다. WME 의 주장 자체가 "세계를 기억한다" 이므로,
// 지나온 곳이 남지 않는 그림은 그 주장을 보여 주지 못한다.
//
// 같은 표면을 다시 보면 같은 복셀에 떨어지므로, 메모리는 지나온 **거리** 가
// 아니라 실제로 본 **표면적** 에 비례해서 자란다. 상한을 넘으면 복셀을 키워
// 다시 병합한다 - 잘라내면 지나온 곳이 사라지지만, 키우면 해상도만 낮아진다.
inline std::int64_t voxKey(const Eigen::Vector3f& p, float inv) {
    const auto q = [inv](float v) {
        return static_cast<std::int64_t>(std::floor(static_cast<double>(v) * inv));
    };
    // 축당 21 비트 = +-1,048,576 복셀. 0.3 m 복셀이면 +-314 km 로 충분하다.
    return ((q(p.x()) & 0x1FFFFF) << 42) |
           ((q(p.y()) & 0x1FFFFF) << 21) |
            (q(p.z()) & 0x1FFFFF);
}

struct VoxelMap {
    std::unordered_map<std::int64_t, Splat> cells;
    float voxel{0.3f};     // m
    int   grown{0};        // 상한 때문에 복셀을 키운 횟수

    void clear() { cells.clear(); grown = 0; }

    void insert(const Splat& s, std::size_t cap) {
        cells[voxKey(s.p, 1.0f / voxel)] = s;
        if (cells.size() > cap) coarsen();
    }

    // 복셀 두 배로 키우고 전부 다시 담는다.
    void coarsen() {
        voxel *= 2.0f;
        ++grown;
        std::unordered_map<std::int64_t, Splat> next;
        next.reserve(cells.size() / 2 + 1);
        const float inv = 1.0f / voxel;
        for (const auto& [k, v] : cells) {
            auto& slot = next[voxKey(v.p, inv)];
            // 같은 복셀이면 더 최근에 본 쪽을 남긴다.
            if (slot.seen <= v.seen) slot = v;
        }
        cells.swap(next);
    }
};

// 월드에 고정되는 물체 기억. 지나온 곳에서 본 물체가 그 자리에 남는다.
//
// **관측마다 위치를 덮어쓰면 안 된다.** 그러면 카메라가 움직일 때마다 물체가
// 따라 움직이는 것처럼 보인다 - 월드에 고정된 기억이 아니라 그때그때의 관측을
// 다시 그리는 것이 된다. 여러 관측의 평균을 유지해야 자리에 가만히 있는다.
struct MemoryObject {
    std::string cls;
    Eigen::Vector3d sum_c{Eigen::Vector3d::Zero()};    // 관측 중심의 합
    Eigen::Vector3d sum_s{Eigen::Vector3d::Zero()};    // 관측 크기의 합
    int    seen{0};        // 마지막 관측 프레임
    int    count{0};       // 몇 번 봤는가. 여러 번 본 것일수록 진하게 그린다

    Eigen::Vector3d center() const { return sum_c / std::max(1, count); }
    Eigen::Vector3d size()   const { return sum_s / std::max(1, count); }

    void observe(const Eigen::Vector3d& c, const Eigen::Vector3d& sz, int frame) {
        sum_c += c;
        sum_s += sz;
        ++count;
        seen = frame;
    }
};

class CloudView {
public:
    void reset(int w, int h) {
        if (img_.cols != w || img_.rows != h) {
            img_.create(h, w, CV_8UC3);
            zbuf_.create(h, w, CV_32F);
        }
        img_.setTo(C_BG);
        zbuf_.setTo(std::numeric_limits<float>::max());
    }

    // 화면에 투영. f 는 초점거리(px), 카메라는 orbit 중심을 바라본다.
    void draw(const std::vector<Splat>& pts, const Orbit& orb, double f,
              double fade) {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const double cx = img_.cols * 0.5, cy = img_.rows * 0.5;

        for (const auto& s : pts) {
            const Eigen::Vector3d v = M * (s.p.cast<double>() - eye);
            if (v.z() < 1e-3) continue;
            const double u = cx + f * v.x() / v.z();
            const double w = cy - f * v.y() / v.z();
            const int iu = static_cast<int>(u), iw = static_cast<int>(w);
            // 거리에 따라 점 크기를 줄이되 0 으로 보내지 않는다. 0 이면 먼 구조가
            // 통째로 사라져 "저기엔 아무 것도 없다" 로 읽힌다.
            const int r = std::clamp(static_cast<int>(1.4 * orb.dist / v.z()), 1, 4);
            if (iu < -r || iw < -r || iu >= img_.cols + r || iw >= img_.rows + r) continue;

            cv::Scalar col = turbo(s.depth_norm);
            // 오래된 점은 배경으로 가라앉힌다 - 최신 관측이 앞에 오게.
            const double a = 1.0 - fade * static_cast<double>(s.age);
            col = {col[0] * a + C_BG[0] * (1 - a),
                   col[1] * a + C_BG[1] * (1 - a),
                   col[2] * a + C_BG[2] * (1 - a)};

            for (int dy = -r; dy <= r; ++dy) {
                const int y = iw + dy;
                if (y < 0 || y >= img_.rows) continue;
                float* z = zbuf_.ptr<float>(y);
                auto* px = img_.ptr<cv::Vec3b>(y);
                for (int dx = -r; dx <= r; ++dx) {
                    const int x = iu + dx;
                    if (x < 0 || x >= img_.cols) continue;
                    const float d = static_cast<float>(v.z());
                    if (d >= z[x]) continue;
                    z[x] = d;
                    px[x] = cv::Vec3b(static_cast<uchar>(col[0]),
                                      static_cast<uchar>(col[1]),
                                      static_cast<uchar>(col[2]));
                }
            }
        }
    }

    // 3D 선분 (궤적, 프러스텀). z-버퍼를 존중한다.
    void line3(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
               const Orbit& orb, double f, const cv::Scalar& col, int th = 1) {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const Eigen::Vector3d va = M * (a - eye), vb = M * (b - eye);
        if (va.z() < 1e-3 || vb.z() < 1e-3) return;
        const double cx = img_.cols * 0.5, cy = img_.rows * 0.5;
        const cv::Point pa(static_cast<int>(cx + f * va.x() / va.z()),
                           static_cast<int>(cy - f * va.y() / va.z()));
        const cv::Point pb(static_cast<int>(cx + f * vb.x() / vb.z()),
                           static_cast<int>(cy - f * vb.y() / vb.z()));
        cv::line(img_, pa, pb, col, th, cv::LINE_AA);
    }

    // 복셀 격자를 **선으로 이어** 표면처럼 보이게 한다.
    //
    // 점만 찍으면 밀도가 낮은 곳이 빈 공간처럼 보이고, 무엇이 벽이고 무엇이
    // 흩어진 잡음인지 구분되지 않는다. 이웃한 복셀이 있을 때만 선을 그으면
    // 연결된 표면에서만 격자가 생기므로, 구조가 있는 곳과 없는 곳이 갈린다.
    //
    // 이웃 판정은 +x/+y/+z 세 방향만 본다. 여섯 방향을 다 보면 같은 선을 두 번
    // 긋게 된다. basis 와 eye 는 한 번만 계산한다 - 선마다 다시 구하면 수만 번
    // 행렬 곱이 돈다.
    void lattice(const std::unordered_map<std::int64_t, Splat>& cells, float voxel,
                 const Orbit& orb, double f, double fade) {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const double cx = img_.cols * 0.5, cy = img_.rows * 0.5;
        const float inv = 1.0f / voxel;

        auto toScreen = [&](const Eigen::Vector3f& p, cv::Point& out) {
            const Eigen::Vector3d v = M * (p.cast<double>() - eye);
            if (v.z() < 1e-3) return false;
            out = {static_cast<int>(cx + f * v.x() / v.z()),
                   static_cast<int>(cy - f * v.y() / v.z())};
            return true;
        };

        static const Eigen::Vector3f kStep[3] = {
            {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};

        for (const auto& [key, v] : cells) {
            cv::Point pa;
            if (!toScreen(v.p, pa)) continue;
            cv::Scalar col = turbo(v.depth_norm);
            const double a = 1.0 - fade * static_cast<double>(v.age);
            col = {col[0] * a + C_BG[0] * (1 - a),
                   col[1] * a + C_BG[1] * (1 - a),
                   col[2] * a + C_BG[2] * (1 - a)};
            for (const auto& d : kStep) {
                const auto it = cells.find(voxKey(v.p + d * voxel, inv));
                if (it == cells.end()) continue;
                cv::Point pb;
                if (!toScreen(it->second.p, pb)) continue;
                // 화면 밖으로 크게 벗어난 선은 그리지 않는다. 투영이 튀면
                // 화면을 가로지르는 긴 선이 생겨 없는 구조를 만들어 낸다.
                if (std::abs(pa.x - pb.x) > img_.cols / 3 ||
                    std::abs(pa.y - pb.y) > img_.rows / 3) continue;
                cv::line(img_, pa, pb, col, 1, cv::LINE_AA);
            }
        }
    }

    // 3D 점을 화면 좌표로. 카메라 뒤면 false.
    // 라벨처럼 2D 로 그려야 하는 것들이 이걸 쓴다.
    bool project3(const Eigen::Vector3d& p, const Orbit& orb, double f,
                  cv::Point& out) const {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const Eigen::Vector3d v = M * (p - eye);
        if (v.z() < 1e-3) return false;
        out = {static_cast<int>(img_.cols * 0.5 + f * v.x() / v.z()),
               static_cast<int>(img_.rows * 0.5 - f * v.y() / v.z())};
        return true;
    }

    // 지면 투영 발자국(footprint). 항공뷰에서 물체의 점유 면적을 읽는 단위다.
    // 상자만 그리면 위에서 볼 때 선 네 개로 뭉개지고, 무엇이 어디를 차지하는지
    // 알 수 없다 - 자율주행 BEV 시각화가 늘 발자국을 그리는 이유다.
    void footprint(const Eigen::Isometry3d& T, const Eigen::Vector3d& c,
                   const Eigen::Vector3d& s, const Eigen::Vector3d& up,
                   const Orbit& orb, double f, const cv::Scalar& col, int th = 2) {
        // 발자국은 **지면 평면 위** 사각형이다. up 에 수직인 두 축으로 만든다 -
        // 카메라계 축을 그대로 쓰면 물체가 기울어진 판처럼 보인다.
        Eigen::Vector3d a = up.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = up.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = up.cross(a).normalized();
        const double hw = 0.5 * std::max(s.x(), 0.05);
        const double hl = 0.5 * std::max(s.z(), 0.05);
        // 물체 바닥으로 내린다.
        const Eigen::Vector3d base = c - up * (0.5 * s.y());
        const Eigen::Vector3d q[4] = {
            T * (base + a * hw + b * hl), T * (base + a * hw - b * hl),
            T * (base - a * hw - b * hl), T * (base - a * hw + b * hl)};
        for (int i = 0; i < 4; ++i) line3(q[i], q[(i + 1) % 4], orb, f, col, th);
    }

    // 속도 벡터. 화살촉까지 그려 방향이 보이게 한다.
    void arrow3(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                const Orbit& orb, double f, const cv::Scalar& col, int th = 2) {
        cv::Point pa, pb;
        if (!project3(a, orb, f, pa) || !project3(b, orb, f, pb)) return;
        cv::arrowedLine(img_, pa, pb, col, th, cv::LINE_AA, 0, 0.3);
    }

    // 3D 상자 (차량/사람). 12 개 모서리를 그린다.
    void box3(const Eigen::Isometry3d& T, const Eigen::Vector3d& c,
              const Eigen::Vector3d& s, const Orbit& orb, double f,
              const cv::Scalar& col, int th = 1) {
        const Eigen::Vector3d h = 0.5 * s;
        Eigen::Vector3d v[8];
        int k = 0;
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sy = -1; sy <= 1; sy += 2)
                for (int sz = -1; sz <= 1; sz += 2)
                    v[k++] = T * (c + Eigen::Vector3d(sx * h.x(), sy * h.y(), sz * h.z()));
        // 위 루프의 비트 순서: idx = (sx+1)/2*4 + (sy+1)/2*2 + (sz+1)/2
        static const int E[12][2] = {{0,1},{0,2},{0,4},{1,3},{1,5},{2,3},
                                     {2,6},{3,7},{4,5},{4,6},{5,7},{6,7}};
        for (const auto& e : E) line3(v[e[0]], v[e[1]], orb, f, col, th);
    }

    // 평면 사각형 (건물 벽 / 지면 / 상판). 법선에 수직한 두 축으로 폭을 만든다.
    void planeQuad(const Eigen::Isometry3d& T, const Eigen::Vector3d& c,
                   const Eigen::Vector3d& n, double half, const Orbit& orb,
                   double f, const cv::Scalar& col) {
        Eigen::Vector3d a = n.cross(Eigen::Vector3d::UnitY());
        if (a.norm() < 1e-6) a = n.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = n.cross(a).normalized();
        const Eigen::Vector3d q[4] = {
            T * (c + (a + b) * half), T * (c + (a - b) * half),
            T * (c - (a + b) * half), T * (c - (a - b) * half)};
        for (int i = 0; i < 4; ++i) line3(q[i], q[(i + 1) % 4], orb, f, col, 1);
        // 대각선은 긋지 않는다. 화면에서는 면의 표시가 아니라 장면을 가로지르는
        // 긴 선으로 읽혀서, 있지도 않은 구조를 보태는 것처럼 보인다.
        // 법선을 짧게 세워 방향만 표시한다 - 벽인지 바닥인지는 그것으로 갈린다.
        line3(T * c, T * (c + n * (half * 0.35)), orb, f, col, 1);
    }

    // 자차 주변 거리 링. 점군만 있으면 크기를 알 수 없다 - 라이다 시각화가
    // 늘 이 링을 그리는 이유이고, 여기서는 그것이 곧 축척이다.
    void rangeRings(const Eigen::Vector3d& at, const Eigen::Vector3d& up,
                    const Orbit& orb, double f, double step, int count,
                    const cv::Scalar& col) {
        // 지면 평면의 두 축을 up 에 수직으로 잡는다.
        Eigen::Vector3d e1 = up.cross(Eigen::Vector3d::UnitX());
        if (e1.norm() < 1e-6) e1 = up.cross(Eigen::Vector3d::UnitZ());
        e1.normalize();
        const Eigen::Vector3d e2 = up.cross(e1).normalized();
        constexpr int kSeg = 72;
        for (int r = 1; r <= count; ++r) {
            const double rad = step * r;
            Eigen::Vector3d prev = at + e1 * rad;
            for (int i = 1; i <= kSeg; ++i) {
                const double a = 2.0 * 3.14159265358979 * i / kSeg;
                const Eigen::Vector3d cur =
                    at + (e1 * std::cos(a) + e2 * std::sin(a)) * rad;
                line3(prev, cur, orb, f, col, 1);
                prev = cur;
            }
        }
    }

    const cv::Mat& image() const { return img_; }

private:
    cv::Mat img_, zbuf_;
};

// ===========================================================================
// 그리기 유틸
// ===========================================================================

void text(cv::Mat& c, const std::string& s, cv::Point at, double sc,
          const cv::Scalar& col, int th = 1) {
    cv::putText(c, s, at, cv::FONT_HERSHEY_SIMPLEX, sc, col, th, cv::LINE_AA);
}

int textW(const std::string& s, double sc, int th = 1) {
    int base = 0;
    return cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, sc, th, &base).width;
}

// 대문자 + 자간. 벤치마크 UI 의 라벨은 거의 전부 이 형태다 - 값과 라벨을
// 형태로 갈라 놓으면 색을 쓰지 않고도 위계가 생긴다.
void label(cv::Mat& c, const std::string& s, cv::Point at, const cv::Scalar& col,
           double sc = T_LABEL, int track = 2) {
    int x = at.x;
    for (char ch : s) {
        const std::string one(1, static_cast<char>(std::toupper(ch)));
        text(c, one, {x, at.y}, sc, col, 1);
        x += textW(one, sc, 1) + track;
    }
}

void fill(cv::Mat& c, cv::Rect r, const cv::Scalar& col) {
    r &= cv::Rect(0, 0, c.cols, c.rows);
    if (r.width > 0 && r.height > 0) c(r).setTo(col);
}

void card(cv::Mat& c, cv::Rect r, const cv::Scalar& accent) {
    fill(c, r, C_PANEL);
    cv::rectangle(c, r, C_LINE, 1, cv::LINE_AA);
    fill(c, {r.x, r.y, r.width, 3}, accent);   // 상단 강조선
}

std::string fmt(double v, int nd, const char* unit = "") {
    if (!std::isfinite(v)) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(nd) << v << unit;
    return ss.str();
}

// 스파크라인. 값 하나만 있으면 추세를 알 수 없다.
// mx 를 밖에서 받는다. 두 카드가 각자 자동맞춤하면 더 나쁜 쪽의 그래프가
// 더 좋아 보인다 - 세로축이 다르다는 것을 아무도 눈치채지 못하기 때문이다.
void spark(cv::Mat& c, cv::Rect r, const std::vector<double>& v, int upto,
           const cv::Scalar& col, double mx) {
    const int n = std::min<int>(upto, static_cast<int>(v.size()));
    if (n < 2 || mx <= 0.0) return;
    std::vector<cv::Point> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double x = r.x + r.width * static_cast<double>(i) / (v.size() - 1);
        const double y = r.y + r.height * (1.0 - v[static_cast<std::size_t>(i)] / mx);
        pts.emplace_back(static_cast<int>(x), static_cast<int>(y));
    }
    for (std::size_t i = 1; i < pts.size(); ++i) {
        cv::line(c, pts[i - 1], pts[i], col, 1, cv::LINE_AA);
    }
    cv::circle(c, pts.back(), 2, col, -1, cv::LINE_AA);
}

// ===========================================================================
// 매니페스트
// ===========================================================================

bool loadManifest(const fs::path& p, std::vector<Seq>& seqs) {
    std::ifstream f(p);
    if (!f) return false;
    std::map<std::string, std::size_t> idx;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line, '\t');
        if (c.empty()) continue;
        if (c[0] == "SEQ" && c.size() >= 6) {
            Seq s;
            s.name = c[1]; s.dataset = c[2]; s.dir = c[3]; s.gt_file = c[4];
            s.identity_ate_cm = toD(c[5]);
            idx[s.name] = seqs.size();
            seqs.push_back(std::move(s));
        } else if (c[0] == "RUN" && c.size() >= 11) {
            auto it = idx.find(c[1]);
            if (it == idx.end()) continue;
            Run r;
            r.system = c[2]; r.label = c[3]; r.kind = c[4];
            r.ate_cm = toD(c[5]); r.rpe_mm = toD(c[6]); r.ms = toD(c[7]);
            r.frames = std::atoi(c[8].c_str());
            r.status = c[9]; r.traj_file = c[10];
            seqs[it->second].systems.push_back(r.system);
            seqs[it->second].runs[r.system] = std::move(r);
        } else if (c[0] == "ERR" && c.size() > 3) {
            auto it = idx.find(c[1]);
            if (it == idx.end()) continue;
            auto rit = seqs[it->second].runs.find(c[2]);
            if (rit == seqs[it->second].runs.end()) continue;
            for (std::size_t i = 3; i < c.size(); ++i) {
                rit->second.err_cm.push_back(toD(c[i]));
            }
        }
    }
    return !seqs.empty();
}

// wme_scene_export 의 출력. 없으면 조용히 비운다 - 의미 구조는 선택 사항이고,
// 없다는 사실 자체는 화면에 "scene: none" 으로 적는다.
void loadScene(Seq& s, const fs::path& dir) {
    const fs::path p = dir / ("scene_" + s.name + ".tsv");
    std::ifstream f(p);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line, '\t');
        if (c[0] == "PLANE" && c.size() >= 13) {
            ScenePlane q;
            q.frame = std::atoi(c[1].c_str());
            q.normal = {toD(c[2]), toD(c[3]), toD(c[4])};
            q.centroid = {toD(c[6]), toD(c[7]), toD(c[8])};
            q.extent = toD(c[9]);
            q.conf = toD(c[12]);
            s.planes.push_back(q);
        } else if (c[0] == "BOX" && c.size() >= 10) {
            SceneBox b;
            b.frame = std::atoi(c[1].c_str());
            b.cls = c[2];
            b.conf = toD(c[3]);
            b.center = {toD(c[4]), toD(c[5]), toD(c[6])};
            b.size = {toD(c[7]), toD(c[8]), toD(c[9])};
            s.boxes.push_back(b);
        }
    }
    std::cout << p.filename().string() << ": 평면 " << s.planes.size()
              << ", 상자 " << s.boxes.size() << "\n";
}

void loadSeq(Seq& s) {
    if (s.loaded) return;
    s.gt = readTum(s.gt_file);
    s.rgb = readIndex(s.dir, "rgb.txt");
    s.depth = readIndex(s.dir, "depth.txt");
    wme_tools::resolveCalib(s.dir, s.calib, /*announce=*/false);

    for (auto& [key, r] : s.runs) {
        r.traj = readTum(r.traj_file);
        std::vector<Eigen::Vector3d> A, B;
        for (const auto& e : r.traj) {
            const int j = nearestIdx(s.gt, e.t);
            if (j < 0 || std::abs(s.gt[static_cast<std::size_t>(j)].t - e.t) > 0.25) continue;
            A.push_back(e.p);
            B.push_back(s.gt[static_cast<std::size_t>(j)].p);
        }
        const Rigid T = kabsch(A, B);
        Eigen::Isometry3d Ta = Eigen::Isometry3d::Identity();
        Ta.linear() = T.R;
        Ta.translation() = T.t;
        r.aligned.clear();
        r.aligned.reserve(r.traj.size());
        for (const auto& e : r.traj) r.aligned.push_back(Ta * e.iso());
    }
    s.loaded = true;
}

// 깊이맵을 세계 좌표 점군으로. stride 로 개수를 조절한다.
void backProject(const cv::Mat& depth16, const wme_tools::DatasetCalib& cal,
                 const Eigen::Isometry3d& T_world_cam, int stride,
                 double dmin, double dmax, double cmin, double cmax,
                 std::vector<Splat>& out) {
    if (depth16.empty()) return;
    const double fx = cal.K.fx, fy = cal.K.fy, cx = cal.K.cx, cy = cal.K.cy;
    const double sc = cal.depth_scale;
    for (int v = 0; v < depth16.rows; v += stride) {
        const auto* row = depth16.ptr<std::uint16_t>(v);
        for (int u = 0; u < depth16.cols; u += stride) {
            const double z = static_cast<double>(row[u]) / sc;
            if (!(z > dmin) || z > dmax) continue;
            const Eigen::Vector3d p_cam((u - cx) * z / fx, (v - cy) * z / fy, z);
            Splat s;
            s.p = (T_world_cam * p_cam).cast<float>();
            // 색 범위는 **유효 범위와 다르다.** KITTI 의 유효 범위는 3~80 m 지만
            // 실제 점의 대부분은 5~30 m 에 있어서, 유효 범위로 정규화하면 전부
            // turbo 의 파란 끝에 몰려 색이 정보를 잃는다.
            s.depth_norm = static_cast<float>(
                std::clamp((z - cmin) / std::max(1e-6, cmax - cmin), 0.0, 1.0));
            s.age = 0.0f;
            out.push_back(s);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    fs::path manifest = "results/bench/viewer.tsv";
    int start_seq = 0;
    bool autoplay = true;
    std::string shot_path;
    int shot_frame = 0;
    int cloud_cap = 260000;
    int start_cam = 0;      // 0 추격 / 1 항공 / 2 자유

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--manifest") manifest = argv[i + 1];
        else if (k == "--seq") start_seq = std::atoi(argv[i + 1]);
        else if (k == "--autoplay") autoplay = std::atoi(argv[i + 1]) != 0;
        else if (k == "--screenshot") shot_path = argv[i + 1];
        else if (k == "--frame") shot_frame = std::atoi(argv[i + 1]);
        else if (k == "--cloud-cap") cloud_cap = std::atoi(argv[i + 1]);
        else if (k == "--cam") start_cam = std::atoi(argv[i + 1]);
    }

    const fs::path scene_dir = fs::path(manifest).parent_path();
    std::vector<Seq> seqs;
    if (!loadManifest(manifest, seqs)) {
        std::cerr <<
            "매니페스트를 읽지 못했다: " << manifest << "\n"
            "  python tools/bench_run.py     (두 시스템 실행 + 채점)\n"
            "  python tools/bench_export.py  (-> results/bench/viewer.tsv)\n"
            "지표를 여기서 다시 계산하지 않으므로, 없으면 실행하지 않는다.\n";
        return 1;
    }

    const bool headless = !shot_path.empty();
    const std::string win = "WorldVision-SLAM  |  SLAM Benchmark";
    if (!headless) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    int si = std::clamp(start_seq, 0, static_cast<int>(seqs.size()) - 1);
    int frame = headless ? shot_frame : 0;
    int pick[2] = {0, 1};          // 각 패널이 보여 줄 시스템 인덱스
    bool playing = autoplay;
    int shot = 0;
    // 카메라 모드. 0 = 추격(진행 방향), 1 = 항공(바로 위에서), 2 = 자유.
    //   추격: 지금 무엇을 보고 있는가. 점군의 구조가 잘 읽힌다.
    //   항공: 무엇이 어디를 차지하는가. 물체 배치와 궤적 형상이 잘 읽힌다.
    // 둘은 서로 다른 질문에 답하므로 한쪽으로 고정하지 않는다.
    int cam_mode = start_cam;
    constexpr bool kDrawCloud = true;

    Orbit orb;
    double user_yaw = 0.0;         // A/D 가 더하는 시선 오프셋
    double base_dist = 0.0;        // 모드 전환 시 되돌아갈 기준 거리
    Eigen::Vector3d ego_pos = Eigen::Vector3d::Zero();   // 지금 프레임의 자차 위치
    bool   user_zoomed = false;    // W/S 를 눌렀으면 자동 거리 조정을 멈춘다
    bool   show_mesh = true;       // 표면 격자 (M 으로 토글)
    cv::Mat canvas(WIN_H, WIN_W, CV_8UC3);
    CloudView cloud[2];
    VoxelMap acc[2];
    // 표면 격자용 굵은 복셀. acc 를 그대로 이으면 선분이 300 만 개가
    // 되어 프레임을 못 채운다. 구조를 보여 주는 데는 성긴 격자로 충분하다.
    VoxelMap mesh[2];
    std::vector<MemoryObject> mem[2];
    int acc_frame[2] = {-1, -1};
    std::vector<double> err_series[2];

    // 카드 기하
    const int card_w = (WIN_W - 3 * PAD) / 2;
    const int card_y = TOPBAR_H + PAD;
    const int card_h = WIN_H - BOTTOM_H - card_y - PAD;
    const int vp_h = card_h - 250;

    while (true) {
        Seq& s = seqs[static_cast<std::size_t>(si)];
        const bool first_load = !s.loaded;
        loadSeq(s);
        if (first_load) loadScene(s, scene_dir);
        if (s.systems.empty()) { si = (si + 1) % static_cast<int>(seqs.size()); continue; }
        for (int k = 0; k < 2; ++k) {
            pick[k] = std::clamp(pick[k], 0, static_cast<int>(s.systems.size()) - 1);
        }

        const int nframes = std::max<int>(1, static_cast<int>(s.rgb.size()));
        if (frame >= nframes) frame = nframes - 1;

        // 궤도는 **현재 카메라를 따라간다.** 전체 궤적에 맞추면 점군이 화면
        // 한구석의 작은 덩어리가 되어 아무 것도 보이지 않는다. 자율주행
        // 시각화가 자차 중심인 것과 같은 이유다 - 관심은 지금 보고 있는
        // 주변이고, 전체 궤적은 선으로 이미 그려져 있다.
        {
            const double stamp = s.rgb.empty() ? 0.0
                               : s.rgb[static_cast<std::size_t>(frame)].first;
            const int j = nearestIdx(s.gt, stamp);
            const Eigen::Vector3d target = (j >= 0)
                ? s.gt[static_cast<std::size_t>(j)].p : Eigen::Vector3d::Zero();
            ego_pos = target;

            // 시점은 **인식 시점(자차)에 고정** 한다.
            //
            // 자차를 화면 중앙에 두고, 시선 방향은 흔들지 않는다. 두 가지를
            // 각각 틀려 봤다:
            //   - 궤적 전체 중심에 고정 -> 자차가 화면 구석으로 밀려나 지금
            //     무엇을 보고 있는지 알 수 없다.
            //   - 진행 방향을 따라 yaw 를 돌림 -> 코너마다 화면이 통째로 회전해
            //     쌓인 지도가 읽히지 않는다.
            // 자차 중앙 + 고정 방위가 자율주행 BEV 의 기본형인 이유가 이것이다.
            orb.center = (acc_frame[0] < 0) ? target
                                            : orb.center * 0.80 + target * 0.20;
            if (orb.dist <= 0.0) {
                // 장면이 화면을 채우는 거리. 유효 깊이 상한(KITTI 80 m)을 그대로
                // 쓰면 너무 멀어 점군이 먼지처럼 보인다.
                orb.dist = (s.dataset == "kitti") ? 70.0 : 5.0;
                base_dist = orb.dist;

                // 세계의 "위" 는 데이터셋마다 다르다. 하나로 고정하면 한쪽에서
                // 카메라가 옆으로 누워 점군이 화면 구석으로 밀려난다.
                //   KITTI: 정답 포즈가 **카메라 좌표계** 라 +y 가 아래 -> up = -y
                //   TUM  : 정답 포즈가 모션캡처 **월드 좌표계** 이고 z 가 위
                orb.world_up = (s.dataset == "kitti") ? Eigen::Vector3d(0, -1, 0)
                                                      : Eigen::Vector3d(0, 0, 1);
                // 표면 격자는 성기게. 촘촘하면 선이 뭉쳐 면처럼 뭉개진다.
                mesh[0].voxel = mesh[1].voxel =
                    (s.dataset == "kitti") ? 2.2f : 0.16f;
            }

            // 카메라 모드별 시선각. 항공뷰는 거의 수직으로 내려다보되 완전한
            // 90 도는 피한다 - 정확히 수직이면 지면 위 높이가 전부 한 점에
            // 겹쳐 물체의 높이 정보가 사라진다.
            //   0 MAP   비스듬히 내려다보는 고정 지도. 구조와 높이가 같이 보인다
            //   1 BIRD  거의 수직. 배치와 궤적 형상이 가장 잘 읽힌다
            //   2 CHASE 자차 추종. "지금 무엇을 보고 있는가"
            const double want_pitch = (cam_mode == 1) ? 1.30
                                    : (cam_mode == 0) ? 0.85 : 0.40;
            // headless 는 루프를 한 번만 돈다. 보간만 하면 목표각에
            // 도달하지 못한 그림이 저장된다.
            orb.pitch += (want_pitch - orb.pitch) * (headless ? 1.0 : 0.25);
            if (cam_mode == 1 && !user_zoomed) {
                // 위에서 보면 같은 거리라도 훨씬 좁은 영역만 담긴다.
                orb.dist = base_dist * 1.1;
            }
            // yaw 는 사용자 조작(A/D)에만 반응한다. 진행 방향을 따라
            // 돌리면 코너마다 지도가 통째로 회전해서, 무엇이 쌓였는지가 아니라
            // 차가 어느 쪽을 향하는지만 보이게 된다.
            orb.yaw = 0.6 + user_yaw;
        }

        // 두 카드의 오차 그래프가 같은 세로 스케일을 쓰도록 여기서 잡는다.
        double err_max = 0.0;
        for (int k = 0; k < 2; ++k) {
            for (double e : s.runs.at(s.systems[static_cast<std::size_t>(pick[k])]).err_cm) {
                err_max = std::max(err_max, e);
            }
        }

        canvas.setTo(C_BG);

        // ---------------------------------------------------------------
        // 상단 바
        // ---------------------------------------------------------------
        fill(canvas, {0, 0, WIN_W, TOPBAR_H}, C_RAISED);
        fill(canvas, {0, TOPBAR_H - 1, WIN_W, 1}, C_LINE);
        text(canvas, "WORLDVISION", {PAD + 2, 41}, 0.72, C_INK, 2);
        text(canvas, "-SLAM", {PAD + 2 + textW("WORLDVISION", 0.72, 2), 41}, 0.72, C_B, 2);
        label(canvas, "descriptor-free visual slam  /benchmark",
              {PAD + 260, 40}, C_INK3, T_LABEL, 1);
        {
            std::ostringstream ss;
            ss << s.name << "   " << (si + 1) << " / " << seqs.size();
            const std::string t = ss.str();
            const int w = textW(t, T_TITLE, 1);
            text(canvas, t, {WIN_W - PAD - w, 40}, T_TITLE, C_INK, 1);
            const std::string ds = s.dataset == "kitti" ? "KITTI  OUTDOOR" : "TUM  INDOOR";
            const int dw = textW(ds, T_MICRO, 1) + 20;
            fill(canvas, {WIN_W - PAD - w - dw - 16, 18, dw, 26}, C_PANEL);
            text(canvas, ds, {WIN_W - PAD - w - dw - 6, 36}, T_MICRO, C_INK2, 1);
        }

        // ---------------------------------------------------------------
        // 두 카드
        // ---------------------------------------------------------------
        for (int k = 0; k < 2; ++k) {
            const cv::Scalar accent = (k == 0) ? C_A : C_B;
            const cv::Rect box(PAD + k * (card_w + PAD), card_y, card_w, card_h);
            card(canvas, box, accent);

            const std::string sys = s.systems[static_cast<std::size_t>(pick[k])];
            const Run& run = s.runs.at(sys);

            // 카드 머리
            label(canvas, k == 0 ? "left  /  existing slam" : "right  /  worldvision (this project)",
                  {box.x + 18, box.y + 28}, C_INK3);
            text(canvas, run.label, {box.x + 18, box.y + 58}, T_TITLE, accent, 2);
            {
                const std::string chip = run.kind;
                const int cw = textW(chip, T_MICRO, 1) + 18;
                fill(canvas, {box.x + box.width - cw - 18, box.y + 38, cw, 24}, C_RAISED);
                text(canvas, chip, {box.x + box.width - cw - 9, box.y + 55}, T_MICRO, C_INK2);
            }

            // ---- 3D 뷰포트 ----
            const cv::Rect vp(box.x + 14, box.y + 74, box.width - 28, vp_h);
            cloud[k].reset(vp.width, vp.height);

            // **왼쪽에는 세계 모델이 없다.**
            //
            // ORB+PnP 는 포즈를 추정할 뿐 세계를 기억하지 않는다. 누적 지도도,
            // 물체 토큰도, 평면도 그 파이프라인의 산출물이 아니다. 그것들을
            // 양쪽에 똑같이 그리면 WME 의 기능을 대조군에 공짜로 붙여 주는
            // 것이고, 비교의 대상 자체가 지워진다.
            //
            // 그래서 descriptor 계열은 **지금 프레임의 성긴 점만** 보여 준다.
            // ORB 가 프레임당 다루는 1000 점 규모에 맞춘 것이고, 프레임이
            // 지나가면 사라진다 - 실제로 그 파이프라인이 남기는 것이 없다.
            const bool has_memory = (run.kind != "descriptor");

            // 점군 누적. 프레임이 뒤로 가면 다시 쌓는다.
            if (kDrawCloud) {
                if (!has_memory) {
                    // 기억이 없으므로 매 프레임 지운다. 지금 프레임만 남는다.
                    acc[k].clear();
                    mesh[k].clear();
                    mem[k].clear();
                    acc_frame[k] = frame - 1;
                } else if (acc_frame[k] > frame || acc_frame[k] < 0) {
                    acc[k].clear();
                    mesh[k].clear();
                    acc_frame[k] = -1;
                }
                const int from = acc_frame[k] + 1;
                for (int fi = from; fi <= frame; ++fi) {
                    const double stamp = s.rgb[static_cast<std::size_t>(fi)].first;
                    const int di = nearestStamp(s.depth, stamp);
                    if (di < 0) continue;
                    // 이 프레임에 대응하는 추정 포즈
                    int pi = -1;
                    double bd = 0.25;
                    for (std::size_t j = 0; j < run.traj.size(); ++j) {
                        const double d = std::abs(run.traj[j].t - stamp);
                        if (d < bd) { bd = d; pi = static_cast<int>(j); }
                    }
                    if (pi < 0) continue;
                    cv::Mat d16 = cv::imread(s.depth[static_cast<std::size_t>(di)].second,
                                             cv::IMREAD_UNCHANGED);
                    if (d16.empty() || d16.type() != CV_16U) continue;
                    std::vector<Splat> pts;
                    const double cmin = (s.dataset == "kitti") ? 4.0 : 0.6;
                    const double cmax = (s.dataset == "kitti") ? 24.0 : 3.6;
                    // 기억이 없는 쪽은 성기게. ORB 가 프레임당 쓰는 점 수에 맞춘다.
                    const int stride = has_memory ? (s.dataset == "kitti" ? 3 : 2)
                                                  : (s.dataset == "kitti" ? 22 : 16);
                    backProject(d16, s.calib, run.aligned[static_cast<std::size_t>(pi)],
                                stride,
                                s.calib.depth_min, s.calib.depth_max, cmin, cmax, pts);
                    // 색을 **관측 거리가 아니라 월드 높이** 로 다시 칠한다.
                    //
                    // 누적 지도에서 "카메라로부터 몇 m" 는 지도의 성질이 아니라
                    // 그때 어디서 봤는지의 성질이다. 같은 벽을 멀리서 한 번,
                    // 가까이서 한 번 보면 색이 달라지고, 전진하는 차량은 먼
                    // 관측을 훨씬 많이 만들어 내므로 지도 전체가 붉게 덮인다 -
                    // 실제로 그렇게 나왔다.
                    //
                    // 높이는 월드에 고정된 값이라 언제 어디서 봐도 같다. 그래서
                    // 노면은 낮고 건물은 높은, 구조가 읽히는 지도가 된다.
                    const double h0 = (s.dataset == "kitti") ? -1.8 : -0.6;
                    const double h1 = (s.dataset == "kitti") ?  6.0 :  1.6;
                    const Eigen::Vector3f up_f = orb.world_up.cast<float>();
                    const float ego_h = static_cast<float>(ego_pos.dot(orb.world_up));
                    for (auto& p : pts) {
                        const float h = p.p.dot(up_f) - ego_h;
                        p.depth_norm = static_cast<float>(
                            std::clamp((h - h0) / (h1 - h0), 0.0, 1.0));
                        p.seen = fi;
                        acc[k].insert(p, static_cast<std::size_t>(cloud_cap));
                        mesh[k].insert(p, 120000);
                    }

                    // 이 프레임에서 본 물체를 월드에 **남긴다.** 지나온 곳의
                    // 물체가 그 자리에 계속 있어야 "세계를 기억한다" 가 화면에
                    // 보인다. 같은 자리의 같은 클래스는 하나로 합치고, 몇 번
                    // 봤는지를 센다 - 여러 번 본 것일수록 확실한 관측이다.
                    if (has_memory && !s.boxes.empty()) {
                        const Eigen::Isometry3d& Tw =
                            run.aligned[static_cast<std::size_t>(pi)];
                        // 병합 반경. 복셀 키에 클래스 해시를 섞으면 인접 판정이
                        // 깨져 같은 물체가 여러 개로 늘어선다 - 실제로 그렇게
                        // 나왔다. 물체 수는 수백 개뿐이므로 최근접 탐색으로
                        // 같은 클래스 중 가장 가까운 것에 합친다.
                        const double merge_r = (s.dataset == "kitti") ? 2.5 : 0.35;
                        for (const auto& b : s.boxes) {
                            if (b.frame != fi) continue;
                            const double mx = b.size.maxCoeff();
                            if (!(mx > 0.2) || mx > 14.0) continue;
                            const Eigen::Vector3d wc = Tw * b.center;

                            MemoryObject* hit = nullptr;
                            double bestd = merge_r;
                            for (auto& m : mem[k]) {
                                if (m.cls != b.cls) continue;
                                const double d = (m.center() - wc).norm();
                                if (d < bestd) { bestd = d; hit = &m; }
                            }
                            if (hit == nullptr) {
                                MemoryObject m;
                                m.cls = b.cls;
                                m.observe(wc, b.size, fi);
                                mem[k].push_back(std::move(m));
                            } else {
                                hit->observe(wc, b.size, fi);
                            }
                        }
                    }
                }
                acc_frame[k] = frame;

                std::vector<Splat> flat;
                flat.reserve(acc[k].cells.size());
                for (const auto& [key, v] : acc[k].cells) flat.push_back(v);
                // age 는 최근성이다. 오래 전에 본 표면은 배경으로 가라앉혀
                // 지금 보고 있는 것과 구분한다 - 지우지는 않는다.
                const double span = std::max(1, frame);
                for (auto& v : flat) {
                    v.age = static_cast<float>(
                        std::clamp(1.0 - static_cast<double>(v.seen) / span, 0.0, 1.0));
                }
                cloud[k].draw(flat, orb, vp.width * 0.9, 0.62);
                // 점 위에 표면 격자를 덧그린다. 순서가 중요하다 - 먼저 그리면
                // 점군이 격자를 덮어 아무 것도 이어져 보이지 않는다.
                if (show_mesh && has_memory) {
                    for (auto& v : mesh[k].cells) {
                        v.second.age = static_cast<float>(std::clamp(
                            1.0 - static_cast<double>(v.second.seen) / span, 0.0, 1.0));
                    }
                    cloud[k].lattice(mesh[k].cells, mesh[k].voxel, orb,
                                     vp.width * 0.9, 0.72);
                }
            }

            // 궤적 (정답 + 추정)
            const double f3 = vp.width * 0.9;

            // 거리 링. 간격은 장면 규모에서 정한다 - 실내 0.5 m, 도로 10 m.
            {
                const double step = (s.dataset == "kitti") ? 10.0 : 0.5;
                // **인식하는 그 시점** 에 링을 건다. 화면 중심(고정 지도의
                // 중심)에 그리면 링이 자차와 따로 놀아서 "지금 어디서 무엇을
                // 보고 있는가" 를 전혀 말해 주지 못한다.
                cloud[k].rangeRings(ego_pos, orb.world_up, orb, f3, step, 5,
                                    cv::Scalar(78, 62, 50));
            }

            for (std::size_t i = 1; i < s.gt.size(); ++i) {
                cloud[k].line3(s.gt[i - 1].p, s.gt[i].p, orb, f3, C_GT, 1);
            }
            {
                const double frac = static_cast<double>(frame + 1) / nframes;
                const int upto = std::max(1, static_cast<int>(
                    frac * static_cast<double>(run.aligned.size())));
                for (int i = 1; i < upto; ++i) {
                    cloud[k].line3(run.aligned[static_cast<std::size_t>(i - 1)].translation(),
                                   run.aligned[static_cast<std::size_t>(i)].translation(),
                                   orb, f3, accent, 2);
                }
                // --- 의미 구조 ---
                // 평면(건물 벽/지면)과 물체 상자(차량/사람)를 **이 시스템의
                // 추정 포즈** 로 옮겨 그린다. 같은 관측이므로 두 패널의 차이는
                // 곧 포즈 차이다 - 점군보다 훨씬 읽기 쉽다.
                if (has_memory && upto > 0 && (!s.planes.empty() || !s.boxes.empty())) {
                    const Eigen::Isometry3d& T = run.aligned[static_cast<std::size_t>(upto - 1)];

                    // **지금 프레임 하나만** 그린다. 여러 프레임을 겹쳐 그리면
                    // 같은 차가 궤적을 따라 줄줄이 늘어서서 장면이 뭉개진다 -
                    // 누적 표현은 점군이 이미 하고 있고, 상자는 "지금 무엇이
                    // 보이는가" 를 말해야 한다.
                    //
                    // scene 파일은 stride 로 듬성듬성하므로 가장 가까운
                    // 내보내진 프레임으로 스냅한다.
                    int snap = -1, best = 1 << 30;
                    for (const auto& b : s.boxes) {
                        const int d = std::abs(b.frame - frame);
                        if (d < best) { best = d; snap = b.frame; }
                    }
                    for (const auto& q : s.planes) {
                        const int d = std::abs(q.frame - frame);
                        if (d < best) { best = d; snap = q.frame; }
                    }

                    // 상자/평면은 **snap 프레임의 카메라 좌표계** 에 있다.
                    // 현재 프레임의 포즈로 옮기면 그 사이 이동만큼 어긋난다
                    // (stride 4, 프레임당 1.4 m 면 최대 3 m). 관측이 놓인
                    // 자리를 틀리게 그리면 드리프트를 보여 주려던 그림이
                    // 스스로 드리프트를 만든다.
                    Eigen::Isometry3d Tsnap = T;
                    if (snap >= 0 && snap < static_cast<int>(s.rgb.size())) {
                        const double st = s.rgb[static_cast<std::size_t>(snap)].first;
                        double bd = 0.25;
                        for (std::size_t j = 0; j < run.traj.size(); ++j) {
                            const double d = std::abs(run.traj[j].t - st);
                            if (d < bd) { bd = d; Tsnap = run.aligned[j]; }
                        }
                    }

                    // --- 기억된 물체 (지나온 곳에 남는 것) ---
                    // 월드에 고정된 좌표이므로 자차가 지나가도 그 자리에 남는다.
                    // 지금 프레임에서 다시 보이는 것은 아래에서 진하게 덧그린다.
                    for (const auto& m : mem[k]) {
                        if (m.cls.empty() || m.count < 2) continue;   // 한 번뿐인 것은 잡음
                        const bool vehicle = (m.cls == "car" || m.cls == "truck" ||
                                              m.cls == "bus" || m.cls == "motorcycle" ||
                                              m.cls == "bicycle" || m.cls == "train");
                        const bool person = (m.cls == "person");
                        const cv::Scalar base = vehicle ? accent
                                              : (person ? C_WARN : C_INK3);
                        // 오래 전에 본 것일수록 어둡게. 지우지는 않는다 -
                        // 기억이 사라지는 것과 흐려지는 것은 다르다.
                        const double a = std::clamp(
                            0.30 + 0.70 * static_cast<double>(m.seen) /
                                   std::max(1, frame), 0.25, 1.0);
                        const cv::Scalar col{base[0] * a + C_BG[0] * (1 - a),
                                             base[1] * a + C_BG[1] * (1 - a),
                                             base[2] * a + C_BG[2] * (1 - a)};
                        // 이미 월드 좌표이므로 항등 변환으로 그린다.
                        cloud[k].footprint(Eigen::Isometry3d::Identity(), m.center(),
                                           m.size(), orb.world_up, orb, f3, col, 1);
                    }

                    if (snap >= 0) {
                        for (const auto& q : s.planes) {
                            if (q.frame != snap || q.conf < 0.25) continue;
                            // extent 는 평면 위 점들의 **평균 산포** 다. 그대로
                            // 반지름으로 쓰면 실제 패치보다 훨씬 커 보인다.
                            const double half = std::clamp(q.extent, 0.3,
                                           s.dataset == "kitti" ? 5.0 : 1.2);
                            cloud[k].planeQuad(Tsnap, q.centroid, q.normal, half, orb, f3,
                                               cv::Scalar(122, 100, 82));
                        }
                        // 이전 내보내기 프레임을 찾아 둔다. 같은 물체의
                        // 프레임 간 이동이 곧 속도 벡터가 된다.
                        int prev_snap = -1;
                        for (const auto& b : s.boxes) {
                            if (b.frame < snap && b.frame > prev_snap) prev_snap = b.frame;
                        }

                        for (const auto& b : s.boxes) {
                            if (b.frame != snap) continue;
                            // 물리적으로 말이 되는 크기만. 검출 상자 안이 배경이면
                            // 깊이 중앙값이 엉뚱해져 50 m 짜리 자동차가 나온다.
                            const double mx = b.size.maxCoeff();
                            if (!(mx > 0.2) || mx > 14.0) continue;
                            const bool vehicle = (b.cls == "car" || b.cls == "truck" ||
                                                  b.cls == "bus" || b.cls == "motorcycle" ||
                                                  b.cls == "bicycle" || b.cls == "train");
                            const bool person = (b.cls == "person");
                            const cv::Scalar col = vehicle ? accent
                                                 : (person ? C_WARN : C_INK3);

                            cloud[k].box3(Tsnap, b.center, b.size, orb, f3, col, 1);
                            // 지면 발자국. 항공뷰에서는 이것이 물체의 본체다.
                            cloud[k].footprint(Tsnap, b.center, b.size, orb.world_up,
                                               orb, f3, col, 2);

                            // 속도 벡터: 직전 내보내기 프레임에서 가장 가까운
                            // 같은 클래스 상자와 이어 붙인다. 추적기가 아니라
                            // 최근접 연관이므로, 밀집 구간에서는 틀릴 수 있다 -
                            // 그래서 화면에도 "nearest-match" 라고 적어 둔다.
                            if (prev_snap >= 0) {
                                const SceneBox* match = nullptr;
                                double bd = 4.0;
                                for (const auto& q : s.boxes) {
                                    if (q.frame != prev_snap || q.cls != b.cls) continue;
                                    const double d = (q.center - b.center).norm();
                                    if (d < bd) { bd = d; match = &q; }
                                }
                                if (match != nullptr && bd > 0.15) {
                                    const Eigen::Vector3d a = Tsnap * match->center;
                                    const Eigen::Vector3d e = Tsnap * b.center;
                                    // 3 배로 늘려 그린다. 실제 이동량은 짧아서
                                    // 화면에서 방향이 보이지 않는다.
                                    cloud[k].arrow3(e, e + (e - a) * 3.0, orb, f3, col, 2);
                                }
                            }

                            // 클래스 라벨. 무엇을 잡았는지 화면에서 읽혀야
                            // "물체를 인식한다" 는 주장이 확인 가능해진다.
                            cv::Point at;
                            const Eigen::Vector3d top =
                                Tsnap * (b.center - Eigen::Vector3d(0, 0.5 * b.size.y(), 0));
                            if (cloud[k].project3(top, orb, f3, at)) {
                                at.x += vp.x; at.y += vp.y - 6;
                                if (at.x > vp.x && at.x < vp.x + vp.width - 60 &&
                                    at.y > vp.y + 12 && at.y < vp.y + vp.height) {
                                    text(canvas, b.cls, at, T_LABEL, col, 1);
                                }
                            }
                        }
                    }
                }

                // 현재 카메라 프러스텀
                if (upto > 0) {
                    const Eigen::Isometry3d& T = run.aligned[static_cast<std::size_t>(upto - 1)];
                    const double sc3 = std::max(0.05, orb.dist * 0.03);
                    const Eigen::Vector3d o = T.translation();
                    const Eigen::Vector3d c[4] = {
                        T * Eigen::Vector3d(-sc3, -sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d( sc3, -sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d( sc3,  sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d(-sc3,  sc3 * 0.75, sc3 * 1.4)};
                    for (int e = 0; e < 4; ++e) {
                        cloud[k].line3(o, c[e], orb, f3, C_INK, 1);
                        cloud[k].line3(c[e], c[(e + 1) % 4], orb, f3, C_INK, 1);
                    }
                }
            }
            cloud[k].image().copyTo(canvas(vp));
            cv::rectangle(canvas, vp, C_LINE, 1, cv::LINE_AA);

            // ---- 카메라 화면 (PIP) ----
            const int pip_w = 268, pip_h = 200;
            const cv::Rect pip(vp.x + vp.width - pip_w - 12,
                               vp.y + vp.height - pip_h - 12, pip_w, pip_h);
            fill(canvas, pip, C_BG);
            if (!s.rgb.empty()) {
                cv::Mat img = cv::imread(s.rgb[static_cast<std::size_t>(frame)].second,
                                         cv::IMREAD_COLOR);
                if (!img.empty()) {
                    const double sc2 = std::min(static_cast<double>(pip.width) / img.cols,
                                                static_cast<double>(pip.height) / img.rows);
                    cv::Mat small;
                    cv::resize(img, small, {}, sc2, sc2, cv::INTER_AREA);
                    const int ox = pip.x + (pip.width - small.cols) / 2;
                    const int oy = pip.y + (pip.height - small.rows) / 2;
                    small.copyTo(canvas(cv::Rect(ox, oy, small.cols, small.rows)));
                }
            }
            cv::rectangle(canvas, pip, C_LINE, 1, cv::LINE_AA);
            label(canvas, "camera", {pip.x + 8, pip.y + 18}, C_INK2, T_MICRO, 1);

            // 범례. 점군 위에 그냥 얹으면 배경색이 제각각이라 읽히지 않는다.
            {
                const cv::Rect scrim(vp.x, vp.y, 500,
                                     (s.boxes.empty() && s.planes.empty()) ? 74 : 112);
                cv::Mat roi = canvas(scrim & cv::Rect(0, 0, canvas.cols, canvas.rows));
                roi *= 0.35;
                label(canvas, "ground truth", {vp.x + 14, vp.y + 24}, C_GT, T_MICRO, 1);
                label(canvas, "estimate  rigidly aligned, scale untouched",
                      {vp.x + 14, vp.y + 44}, accent, T_MICRO, 1);
                label(canvas, has_memory ? "map = depth back-projected by this system's pose, accumulated"
                         : "points = depth back-projected by this system's pose",
                      {vp.x + 14, vp.y + 64}, C_INK2, T_MICRO, 1);
                if (!has_memory) {
                    label(canvas,
                          "pose only - this pipeline keeps no world model",
                          {vp.x + 14, vp.y + 84}, C_WARN, T_MICRO, 1);
                    label(canvas,
                          "sparse points, current frame only. nothing persists",
                          {vp.x + 14, vp.y + 102}, C_INK3, T_MICRO, 1);
                } else if (!s.boxes.empty() || !s.planes.empty()) {
                    label(canvas,
                          "persistent map + objects (tier 1) + planes (tier 2)",
                          {vp.x + 14, vp.y + 84}, C_INK2, T_MICRO, 1);
                    label(canvas,
                          "arrows = frame-to-frame motion, nearest-match (not a tracker)",
                          {vp.x + 14, vp.y + 102}, C_INK3, T_MICRO, 1);
                }
            }

            // ---- 지표 ----
            const int my = vp.y + vp.height + 34;
            const std::string other_sys = s.systems[static_cast<std::size_t>(pick[1 - k])];
            const double other = s.runs.at(other_sys).ate_cm;
            const bool better = std::isfinite(run.ate_cm) && std::isfinite(other)
                                && run.ate_cm < other;

            label(canvas, "ate rmse", {box.x + 18, my}, C_INK3);
            text(canvas, fmt(run.ate_cm, 2), {box.x + 18, my + 62}, T_HERO,
                 better ? C_GOOD : C_INK, 3);
            text(canvas, "cm", {box.x + 20 + textW(fmt(run.ate_cm, 2), T_HERO, 3), my + 62},
                 T_BODY, C_INK3, 1);
            if (better) {
                const int bx = box.x + 18;
                fill(canvas, {bx, my + 74, 92, 22}, C_GOOD);
                text(canvas, "BETTER", {bx + 12, my + 90}, T_MICRO, C_BG, 1);
            }

            const int col2 = box.x + 300;
            label(canvas, "rpe trans median", {col2, my}, C_INK3);
            text(canvas, fmt(run.rpe_mm, 2, " mm"), {col2, my + 34}, T_VALUE, C_INK, 1);
            label(canvas, "ms / frame", {col2, my + 66}, C_INK3);
            text(canvas, fmt(run.ms, 1), {col2, my + 100}, T_VALUE, C_INK, 1);

            const int col3 = box.x + 520;
            label(canvas, "vs floor", {col3, my}, C_INK3);
            const double gain = (std::isfinite(run.ate_cm) && run.ate_cm > 0)
                                ? s.identity_ate_cm / run.ate_cm : std::nan("");
            text(canvas, fmt(gain, 1, "x"), {col3, my + 34}, T_VALUE,
                 (std::isfinite(gain) && gain > 1.0) ? C_GOOD : C_BAD, 1);
            label(canvas, "frames", {col3, my + 66}, C_INK3);
            text(canvas, std::to_string(run.frames), {col3, my + 100}, T_VALUE, C_INK, 1);

            // 프레임별 오차. 두 점군은 짧은 구간에서 거의 같아 보이므로
            // **언제** 갈라지는지는 여기서만 보인다. 두 카드가 같은 세로
            // 스케일을 쓰도록 바깥에서 최대값을 잡는다 - 각자 자동맞춤하면
            // 나쁜 쪽이 더 좋아 보인다.
            {
                const int sx = box.x + 660;
                const cv::Rect plot(sx, my + 6, box.x + box.width - sx - 18, 94);
                label(canvas, "ate error over time", {sx, my}, C_INK3);
                fill(canvas, plot, C_BG);
                const double frac = static_cast<double>(frame + 1) / nframes;
                const int upto = std::max(1, static_cast<int>(
                    frac * static_cast<double>(run.err_cm.size())));
                spark(canvas, plot, run.err_cm, upto, accent, err_max);
                if (err_max > 0.0) {
                    text(canvas, fmt(err_max, 0, " cm"), {plot.x + 6, plot.y + 16},
                         T_MICRO, C_INK3);
                    text(canvas, "0", {plot.x + 6, plot.y + plot.height - 6},
                         T_MICRO, C_INK3);
                }
            }

            if (run.status != "ok") {
                text(canvas, "STATUS " + run.status + " - ATE IS MEANINGLESS",
                     {box.x + 18, my + 130}, T_BODY, C_BAD, 1);
            }
        }

        // ---------------------------------------------------------------
        // 하단 바
        // ---------------------------------------------------------------
        const int fy = WIN_H - BOTTOM_H;
        constexpr int W_MODE_X = WIN_W - 320;
        fill(canvas, {0, fy, WIN_W, BOTTOM_H}, C_RAISED);
        fill(canvas, {0, fy, WIN_W, 1}, C_LINE);

        // 진행 막대
        {
            const cv::Rect bar(PAD, fy + 26, WIN_W - 2 * PAD, 6);
            fill(canvas, bar, C_BG);
            const int done = static_cast<int>(bar.width *
                             (static_cast<double>(frame + 1) / nframes));
            fill(canvas, {bar.x, bar.y, std::max(2, done), bar.height}, C_B);
            std::ostringstream ss;
            ss << "FRAME " << (frame + 1) << " / " << nframes;
            text(canvas, ss.str(), {PAD, fy + 58}, T_BODY, C_INK);
            const char* mode = (cam_mode == 1) ? "BIRD'S EYE"
                             : (cam_mode == 2) ? "CHASE" : "MAP";
            label(canvas, "camera", {W_MODE_X, fy + 40}, C_INK3);
            text(canvas, mode, {W_MODE_X, fy + 62}, T_BODY, C_INK2);
            const std::string st = playing ? "RUNNING" : "PAUSED";
            text(canvas, st, {WIN_W - PAD - textW(st, T_BODY, 1), fy + 58}, T_BODY,
                 playing ? C_GOOD : C_INK3);
        }

        // ATE 막대 비교 (왼쪽 절반)
        {
            const int bx = PAD, by = fy + 82, bw = 560, bh = 16;
            label(canvas, "ate rmse   shorter is better", {bx, by - 10}, C_INK3);
            double worst = 1e-9;
            for (int k = 0; k < 2; ++k) {
                worst = std::max(worst,
                    s.runs.at(s.systems[static_cast<std::size_t>(pick[k])]).ate_cm);
            }
            for (int k = 0; k < 2; ++k) {
                const Run& r = s.runs.at(s.systems[static_cast<std::size_t>(pick[k])]);
                const cv::Rect track(bx, by + 6 + k * (bh + 8), bw, bh);
                fill(canvas, track, C_BG);
                const int w = static_cast<int>(bw * std::clamp(r.ate_cm / worst, 0.0, 1.0));
                fill(canvas, {track.x, track.y, std::max(3, w), bh}, k == 0 ? C_A : C_B);
                text(canvas, fmt(r.ate_cm, 2, " cm"), {bx + bw + 12, track.y + 13},
                     T_BODY, C_INK);
            }
        }

        // 조작 안내 (오른쪽 절반, 두 줄로 나눠 겹치지 않게)
        {
            const int hx = 780;
            label(canvas, "controls", {hx, fy + 72}, C_INK3);
            text(canvas, "SPACE play    , . step    N/P sequence    1/2 swap model",
                 {hx, fy + 100}, T_BODY, C_INK2);
            text(canvas, "A/D orbit     W/S zoom    V camera (chase/bird)    R restart    F shot    Q quit",
                 {hx, fy + 126}, T_BODY, C_INK2);
        }
        {
            const std::string src =
                "ATE / RPE / ms-per-frame read from results/bench/viewer.tsv "
                "(computed by python/tools/bench_run.py) - not recomputed here";
            text(canvas, src, {WIN_W - PAD - textW(src, T_MICRO, 1), WIN_H - 16},
                 T_MICRO, C_INK3);
        }

        if (headless) {
            // 왜 비었는지 사후에 물어볼 수 있어야 한다. 화면이 검은 것과
            // 데이터가 없는 것은 그림만 봐서는 구분되지 않는다.
            for (int k = 0; k < 2; ++k) {
                std::cerr << "  패널 " << k << ": 복셀 " << acc[k].cells.size()
                          << " (" << acc[k].voxel << " m, coarsen " << acc[k].grown << ")"
                          << ", 기억물체 " << mem[k].size()
                          << ", 평면 " << s.planes.size()
                          << ", 상자 " << s.boxes.size()
                          << ", orbit dist " << orb.dist
                          << " pitch " << orb.pitch << "\n";
            }
            if (!cv::imwrite(shot_path, canvas)) {
                std::cerr << "스크린샷 저장 실패: " << shot_path << "\n";
                return 1;
            }
            std::cout << "저장: " << shot_path << "  (" << s.name << ", frame "
                      << (frame + 1) << "/" << nframes << ")\n";
            return 0;
        }

        cv::imshow(win, canvas);
        const int key = cv::waitKey(playing ? 1 : 0);

        if (key == 'q' || key == 'Q' || key == 27) break;
        else if (key == ' ') playing = !playing;
        else if (key == 'r' || key == 'R') { frame = 0; acc_frame[0] = acc_frame[1] = -1;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            mesh[0].clear(); mesh[1].clear(); }
        else if (key == 'v' || key == 'V') { cam_mode = (cam_mode + 1) % 3; }
        else if (key == 'm' || key == 'M') show_mesh = !show_mesh;
        else if (key == 'n' || key == 'N') {
            si = (si + 1) % static_cast<int>(seqs.size());
            frame = 0; acc_frame[0] = acc_frame[1] = -1; orb.dist = 0.0; user_zoomed = false;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            mesh[0].clear(); mesh[1].clear();
        } else if (key == 'p' || key == 'P') {
            si = (si + static_cast<int>(seqs.size()) - 1) % static_cast<int>(seqs.size());
            frame = 0; acc_frame[0] = acc_frame[1] = -1; orb.dist = 0.0; user_zoomed = false;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            mesh[0].clear(); mesh[1].clear();
        } else if (key == '1') {
            pick[0] = (pick[0] + 1) % static_cast<int>(s.systems.size());
            acc_frame[0] = -1; mem[0].clear(); acc[0].clear(); mesh[0].clear();
        } else if (key == '2') {
            pick[1] = (pick[1] + 1) % static_cast<int>(s.systems.size());
            acc_frame[1] = -1; mem[1].clear(); acc[1].clear(); mesh[1].clear();
        } else if (key == 'a' || key == 'A') user_yaw -= 0.12;
        else if (key == 'd' || key == 'D') user_yaw += 0.12;
        else if (key == 'w' || key == 'W') { orb.dist *= 0.88; user_zoomed = true; }
        else if (key == 's' || key == 'S') { orb.dist *= 1.14; user_zoomed = true; }
        else if (key == 'f' || key == 'F') {
            const std::string f2 = "viewer_" + s.name + "_" + std::to_string(shot++) + ".png";
            if (cv::imwrite(f2, canvas)) std::cout << "저장: " << f2 << "\n";
        } else if (key == ',') frame = std::max(0, frame - 1);
        else if (key == '.') frame = std::min(nframes - 1, frame + 1);
        else if (playing && frame + 1 < nframes) ++frame;
        else if (playing) playing = false;   // 끝나면 멈춘다. 되감기는 R.

        if (cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1.0) break;
    }

    cv::destroyAllWindows();
    return 0;
}
