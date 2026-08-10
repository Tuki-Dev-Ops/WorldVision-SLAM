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
#include <chrono>
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
// 3D 뷰포트는 **순수 검정** 이다. 라이다 지도가 늘 검은 바탕인 것은
// 취향이 아니라 대비 때문이다 - 어두운 남색 위에서는 낮은 밝기의 점이
// 배경에 묻혀, 성긴 곳과 아무 것도 없는 곳이 구분되지 않는다.
const cv::Scalar C_VOID    (  0,   0,   0);
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

    // 진행 방향. 1 인칭/3 인칭에서는 시선이 이 방향을 따라간다.
    // 영벡터면 쓰지 않는다 (고정 방위 지도 뷰).
    Eigen::Vector3d heading{Eigen::Vector3d::Zero()};

    Eigen::Matrix3d basis() const {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double cp = std::cos(pitch), sp = std::sin(pitch);
        // 지면 평면의 두 축 (world_up 에 수직)
        Eigen::Vector3d a = world_up.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = world_up.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = world_up.cross(a).normalized();
        // 진행 방향이 주어지면 그것을 지면 안에 눕혀 시선의 기준으로 쓴다.
        // 1 인칭은 "지금 무엇을 향해 가고 있는가" 가 화면이어야 하므로
        // 고정 방위로는 성립하지 않는다.
        if (heading.squaredNorm() > 1e-12) {
            Eigen::Vector3d hg = heading - world_up * heading.dot(world_up);
            if (hg.norm() > 1e-9) {
                hg.normalize();
                const Eigen::Vector3d fwd2 = (hg * cp - world_up * sp).normalized();
                Eigen::Vector3d up2 = world_up;
                Eigen::Vector3d right2 = fwd2.cross(up2).normalized();
                up2 = right2.cross(fwd2).normalized();
                Eigen::Matrix3d M2;
                M2.row(0) = right2; M2.row(1) = up2; M2.row(2) = fwd2;
                return M2;
            }
        }
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
    // **얼마나 멀리서 본 것인가.**
    //
    // 스테레오 깊이 오차는 sigma_Z = c*Z^2 이므로 거리의 **제곱** 으로 커진다.
    // 60 m 에서 본 점은 6 m 에서 본 점보다 분산이 100 배다. 그래서 먼 관측은
    // 하늘로도 땅속으로도 튀고, 그것이 지금까지 "하늘에 뭔가 생긴다" 로
    // 보이던 것의 정체다.
    //
    // 거리를 같이 들고 있으면 두 가지를 할 수 있다: 먼 것은 흐리게 그려
    // 잠정임을 표시하고, 나중에 가까이 가서 같은 자리를 다시 보면 **먼 관측을
    // 폐기** 할 수 있다. 지도가 시간이 지나며 좋아지는 것이 그 뜻이다.
    float range{0.0f};     // 관측 당시 카메라로부터의 거리 (m)

    // --- 잡음을 거르는 두 가지 근거: 공간과 시간 ---
    //
    // **공간**: 반경 이웃 수 (Radius Outlier Removal). 주변에 아무 것도 없는
    // 점은 표면이 아니라 SGBM 오매칭의 산탄이다. PCL 의 ROR 과 같은 판정이되,
    // 이미 해시 복셀이라 kd-tree 가 필요 없다 - 26 이웃 키를 조회하면 끝이다.
    //
    // **시간**: 몇 번, 그리고 **몇 개의 서로 다른 시점** 에서 봤는가. 이 둘은
    // 다른 것을 잡는다. 스테레오 유령은 같은 오매칭이 인접 프레임에서 반복
    // 재현되므로 관측 횟수만으로는 안 걸러지고, 시점이 바뀌면 사라진다.
    // 공간 판정이 절대 못 잡는 밀집 유령 덩어리가 여기서 걸린다.
    // **관측된 밝기.**
    //
    // 지금까지 지도는 깊이만 쓰고 영상의 밝기를 버렸다. 그런데 노면의 정보는
    // 거의 전부 밝기에 있다 - 차선, 주차선, 횡단보도는 높이 차이가 0 이라
    // 기하로는 존재하지 않고, 흰 페인트라는 사실로만 존재한다. 라이다 지도가
    // intensity 로 차선을 보여 주는 것과 같은 이야기이고, 여기서는 카메라
    // 밝기가 그 자리를 그대로 대신한다.
    std::uint8_t  intensity{0};
    std::uint8_t  nbr{0};        // 26 이웃 중 점유 수
    std::uint8_t  hits{0};       // 총 관측 수 (포화 255)
    std::uint16_t view_mask{0};  // 관측한 시점 섹터 비트

    // 그릴 만한가. hits 만 보면 유령이 통과하고, 시점만 보면 웜업이 길다.
    bool confirmed() const {
        int v = 0;
        for (std::uint16_t m = view_mask; m; m >>= 1) v += (m & 1u);
        return hits >= 3 && v >= 2;
    }
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

    void clear() { cells.clear(); grown = 0; revised = 0; }

    // 가까이서 본 관측이 멀리서 본 관측을 **이긴다.**
    long long revised{0};      // 폐기한 먼 관측의 수. 개선이 일어났다는 증거다.

    void insert(const Splat& s, std::size_t cap, int view_sector = -1) {
        const float inv = 1.0f / voxel;
        const auto key = voxKey(s.p, inv);
        auto it = cells.find(key);

        // **증거는 누적된다.** 관측 하나가 다른 관측을 대체하더라도, 이 칸을
        // 몇 번 어디서 봤는지는 그 칸의 이력이지 개별 관측의 속성이 아니다.
        // 덮어쓸 때 이것까지 갈아치우면 시간 판정이 통째로 무력해진다.
        std::uint8_t hits = 0;
        std::uint16_t vm = 0;
        std::uint8_t nbr = 0;
        if (it != cells.end()) {
            hits = it->second.hits;
            vm   = it->second.view_mask;
            nbr  = it->second.nbr;
        }
        if (hits < 255) ++hits;
        if (view_sector >= 0) vm |= static_cast<std::uint16_t>(1u << (view_sector & 15));

        if (it != cells.end() && it->second.range > 0.0f && s.range > 0.0f &&
            it->second.range < s.range * 0.8f) {
            // 같은 칸에 이미 더 가까이서 본 관측이 있다. 먼 것으로 덮으면
            // 좋은 값을 나쁜 값으로 바꾸는 것이다 - 최신이 항상 나은 것은
            // 아니고, 여기서 나은 것은 **가까운** 것이다. 다만 이력은 갱신한다.
            it->second.hits = hits;
            it->second.view_mask = vm;
            return;
        }

        const bool fresh = (it == cells.end());
        Splat& slot = cells[key];
        slot = s;
        slot.hits = hits;
        slot.view_mask = vm;
        slot.nbr = nbr;

        // 새 칸이 생기면 26 이웃의 카운터를 올리고 자기 것도 세어 온다.
        // 삽입할 때만 26 회 조회하면 렌더에서는 비교 한 번으로 끝난다 -
        // 전체 90 만 셀을 매 프레임 훑으면 1.6 초다.
        if (fresh) {
            int mine = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (!dx && !dy && !dz) continue;
                        const Eigen::Vector3f q = s.p + Eigen::Vector3f(
                            dx * voxel, dy * voxel, dz * voxel);
                        const auto n = cells.find(voxKey(q, inv));
                        if (n == cells.end()) continue;
                        ++mine;
                        if (n->second.nbr < 255) ++n->second.nbr;
                    }
                }
            }
            cells[key].nbr = static_cast<std::uint8_t>(std::min(mine, 255));
        }
        if (cells.size() > cap) coarsen();
    }

    // **가까이 와서 다시 본 자리의 먼 관측을 지운다.**
    //
    // 멀리서 본 점은 자기 칸에만 틀리게 앉는 것이 아니라 **엉뚱한 칸에**
    // 앉는다 - 깊이가 몇 m 씩 어긋나므로 아예 다른 자리에 찍힌다. 그래서
    // 같은 칸을 덮어쓰는 것만으로는 지워지지 않고, 유령이 그 자리에 남는다.
    //
    // 가까운 관측이 들어오면 그 주변에서 훨씬 멀리서 찍힌 것들을 걷어낸다.
    // "지금 바로 앞에서 보고 있는데 저기 있다던 것이 없다" 는 뜻이므로,
    // 지우는 것이 맞다.
    void supersede(const Eigen::Vector3f& p, float new_range, int reach) {
        if (!(new_range > 0.0f)) return;
        const float inv = 1.0f / voxel;
        for (int dx = -reach; dx <= reach; ++dx) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dz = -reach; dz <= reach; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const Eigen::Vector3f q = p + Eigen::Vector3f(
                        dx * voxel, dy * voxel, dz * voxel);
                    const auto it = cells.find(voxKey(q, inv));
                    if (it == cells.end()) continue;
                    // 지금 관측보다 2.5 배 넘게 멀리서 찍힌 것만 걷어낸다.
                    // 비슷한 거리에서 본 것은 둘 다 믿을 만하므로 남긴다.
                    if (it->second.range > new_range * 2.5f) {
                        cells.erase(it);
                        ++revised;
                    }
                }
            }
        }
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

// ---------------------------------------------------------------------------
// 장면 라벨 — 지도가 스스로 말하는 것
// ---------------------------------------------------------------------------
// **검출기는 이 라벨을 낼 수 없다.** yolo11n 은 COCO 80 클래스이고 거기에
// building / tree / fence 는 없다. kitti_00 에서 나온 검출은 car 465, person 5,
// truck 4 - 도로 양옆의 건물도 가로수도 담장도 한 번도 나오지 않는다. 그런데
// 그것들은 장면의 대부분이다.
//
// 그래서 클래스를 **누적 지도의 기하** 에서 만든다. 우회가 아니라 이쪽이 맞는
// 방법이다: 건물은 연속된 수직 벽면이고, 나무는 줄기 위에 흩어진 수관이며,
// 담장은 낮고 연속된 수직면이다. 그 차이는 이미 복셀 배치 안에 있다.
//
// 판정 단위는 **지면 격자 한 칸의 수직 기둥** 이다. 기둥 하나가 어느 높이에
// 얼마나 채워져 있는가가 곧 그것이 무엇인가다.
enum class Stuff : std::uint8_t { Unknown, Ground, Building, Fence, Vegetation, Pole };

inline const char* stuffName(Stuff s) {
    switch (s) {
        case Stuff::Ground:     return "ground";
        case Stuff::Building:   return "building";
        case Stuff::Fence:      return "fence / wall";
        case Stuff::Vegetation: return "tree";
        case Stuff::Pole:       return "pole";
        default:                return "";
    }
}

// 클래스 색. turbo 팔레트와 섞이면 안 되므로 채도를 낮춘 고유색을 쓴다.
inline cv::Scalar stuffColor(Stuff s) {
    switch (s) {                                    // BGR
        case Stuff::Ground:     return {104,  92,  78};
        case Stuff::Building:   return {196, 156, 108};
        case Stuff::Fence:      return {132, 176, 190};
        case Stuff::Vegetation: return {110, 190, 130};
        case Stuff::Pole:       return {168, 168, 214};
        default:                return {90, 90, 90};
    }
}

// 기둥 하나. 핵심은 **높이 점유 비트마스크** 다.
//
// 처음에는 최저/최고 높이와 복셀 개수만 셌는데, 그것으로는 나무와 건물이
// 갈리지 않았다. 복셀 개수는 "몇 층이 찼는가" 가 아니다 - 격자칸이 복셀보다
// 넓으면 같은 높이에 여러 개가 들어가므로 연속성과 아무 관계가 없어진다.
// KITTI 00 이 통째로 초록(나무)이 된 것이 그 결과다.
//
// 나무와 건물을 가르는 것은 **프로파일에 구멍이 있는가** 이고, 그것은 개수가
// 아니라 어느 높이가 찼는지를 알아야 나온다. 0.5 m 씩 32 칸, 지면 위 16 m.
constexpr double kPiV = 3.14159265358979323846;
constexpr float kBinH = 0.5f;
// 노면 격자 크기. 주차선 폭이 15 cm 이므로 그보다 촘촘해야 한다.
constexpr float kRoadCell = 0.10f;
constexpr int   kBins = 32;

struct Column {
    std::uint32_t bins{0};   // 지면 기준 높이 점유
    float low{1e9f};         // 절대 최저 높이 (이웃 지면 추정용)
    float high{-1e9f};       // 절대 최고 높이
    int   i{0}, j{0};        // 지면 격자 좌표 (이웃 조회용)
    Eigen::Vector3f rep{Eigen::Vector3f::Zero()};   // 그릴 대표점 (기둥 꼭대기)
    float ground{0.0f};      // 이 칸에 쓰인 지면 높이

    // 구조 텐서용 누적. 이웃 아홉 칸의 것을 더하면 그 자리의 공분산이 된다 -
    // 점 목록을 따로 들고 있지 않아도 된다.
    int    n{0};
    Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d sumsq{Eigen::Matrix3d::Zero()};

    // 분류 결과와 그 근거. 근거를 남기는 이유는 화면에서 틀렸을 때 어느
    // 판별자가 틀렸는지 알아야 하기 때문이다.
    float planarity{0.0f}, linearity{0.0f}, scatter{0.0f};
    float vert{0.0f};        // 법선이 수평에 가까운 정도 (1 = 수직면)
    // 구조 텐서가 준 평면 법선. 건물 벽면을 평면으로 그릴 때 쓴다 -
    // 잡음을 펴는 것과 원래 평면인 것을 평면으로 그리는 것은 다르다.
    Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
    // 5x5 이웃 중 점유된 칸 수. 기둥과 벽을 가르는 데 쓴다 -
    // 기둥은 홀로 서 있고 벽은 이웃이 빽빽하다.
    std::uint8_t crowd{0};
    Stuff cls{Stuff::Unknown};

    // 최고 점유 칸. 이상치 하나에 그대로 끌려간다.
    int topBinRaw() const {
        for (int b = kBins - 1; b >= 0; --b) if (bins & (1u << b)) return b;
        return -1;
    }
    // **지지받는 최고 칸.** 바로 아래 칸도 차 있어야 인정한다.
    //
    // 최고점을 그대로 쓰면 복셀 하나짜리 스파이크가 기둥을 하늘까지 늘리고,
    // 평평한 노면도 그 꼭대기를 따라 울퉁불퉁해진다 - 하늘이 덮이고 바닥이
    // 이상하게 보이던 것이 둘 다 여기서 나왔다. 두 칸이 이어져야 구조다.
    int topBin() const {
        for (int b = kBins - 1; b >= 1; --b) {
            if ((bins & (1u << b)) && (bins & (1u << (b - 1)))) return b;
        }
        return (bins & 1u) ? 0 : -1;
    }

    // **아래에서부터 이어져 올라간 꼭대기.**
    //
    // topBin 은 아래 칸이 찼는지만 묻지, 그 두 칸이 지면과 이어져 있는지는
    // 묻지 않는다. 나뭇가지나 전선처럼 공중에 뜬 두 칸이 있으면 그 칸의
    // 높이가 통째로 그리로 끌려가고, 화면에는 8 m 짜리 집 위에 16 m 짜리
    // 탑이 선다 - 지도가 막대그래프처럼 보이던 이유의 절반이 이것이다.
    //
    // 첫 점유 칸에서 출발해 위로 올라가되, 빈 칸이 셋(1.5 m) 넘게 이어지면
    // 거기서 멈춘다. 한두 칸의 구멍은 관측이 성겨서 생기므로 넘어간다.
    // 출발을 첫 점유 칸으로 잡는 것도 같은 이유다 - 주차된 차에 가려 벽의
    // 발치를 못 본 칸이 흔한데, 지면부터 요구하면 그 벽이 통째로 사라진다.
    // 지면부터 그 칸까지 몇 층이 찼는가. 0..1.
    //
    // 지붕과 수관은 둘 다 "위쪽에 있는 수평 평면 조각" 이라 구조 텐서로는
    // 갈리지 않는다 - 스테레오가 보는 것은 수관의 **겉면** 이고 겉면은
    // 평면이기 때문이다. 갈리는 곳은 그 **아래** 다: 지붕 밑에는 벽이 있어
    // 기둥이 위아래로 차 있고, 수관 밑에는 공기와 가는 줄기뿐이라 비어 있다.
    float fillTo(int top) const {
        if (top < 0) return 0.0f;
        int occ = 0;
        for (int b = 0; b <= top; ++b) if (bins & (1u << b)) ++occ;
        return static_cast<float>(occ) / static_cast<float>(top + 1);
    }

    int topRun() const {
        int best = -1, gap = 0;
        bool started = false;
        for (int b = 0; b < kBins; ++b) {
            if (bins & (1u << b)) { best = b; gap = 0; started = true; }
            else if (started && ++gap > 3) break;
        }
        return best;
    }
};

// 지면 격자. up 에 수직인 두 축으로 만든 2 차원 격자다. 데이터셋마다 "위" 가
// 다르므로 축을 고정하면 한쪽에서 격자가 벽을 따라 잘린다.
struct GroundGrid {
    Eigen::Vector3f up, a, b;
    float inv;
    GroundGrid(const Eigen::Vector3f& up_in, float cell) : up(up_in.normalized()) {
        a = up.cross(Eigen::Vector3f::UnitZ());
        if (a.norm() < 1e-6f) a = up.cross(Eigen::Vector3f::UnitX());
        a.normalize();
        b = up.cross(a).normalized();
        inv = 1.0f / cell;
    }
    std::pair<int, int> ij(const Eigen::Vector3f& p) const {
        return {static_cast<int>(std::floor(p.dot(a) * inv)),
                static_cast<int>(std::floor(p.dot(b) * inv))};
    }
    static std::int64_t key(int i, int j) {
        return (static_cast<std::int64_t>(i & 0x3FFFFFF) << 26) |
                static_cast<std::int64_t>(j & 0x3FFFFFF);
    }
    std::int64_t key(const Eigen::Vector3f& p) const {
        const auto [i, j] = ij(p);
        return key(i, j);
    }
};

// 기둥 하나를 읽는다.
//
// **점유 프로파일의 구멍으로는 안 된다.** 처음에 그렇게 했다가 KITTI 00 의
// 87 % 가 나무로 분류됐다. 수동 스테레오 지도에서 복셀이 없다는 것은 그 자리가
// 비었다는 뜻이 아니라 **거기서 표면을 못 봤다** 는 뜻이다. 점은 표면에만
// 생기고 관측은 한쪽에서만 오므로 멀쩡한 벽면에도 구멍이 가득하다. 구멍을
// 세는 판별자는 구조가 아니라 센서의 희소성을 재고 있었다.
//
// 국소 구조 텐서는 그 함정에 빠지지 않는다. 있는 점들이 **어떻게 퍼져 있는가**
// 만 보기 때문이다:
//
//   평면성 λ1-λ2  벽·바닥은 한 방향으로 납작하다
//   선형성 λ0-λ1  기둥·가로등은 한 방향으로 길다
//   산포   λ2/λ0  잎은 세 방향으로 고르게 퍼진다
//
// 그리고 평면의 **법선 방향** 이 바닥과 벽을 가른다. 높이만으로는 못 가른다.
inline Stuff classifyColumn(const Column& c) {
    const int t = c.topBin();
    if (t < 0 || c.n < 6) return Stuff::Unknown;
    const float top_m = static_cast<float>(t + 1) * kBinH;

    // 문턱은 실측 분포의 사분위에서 골랐다 (11718 개 기둥, kitti_00 302 프레임):
    //   planarity  p25 0.20  p50 0.34  p75 0.48  p90 0.60
    //   linearity  p25 0.19  p50 0.33  p75 0.49  p90 0.63
    //   scatter    p25 0.14  p50 0.25  p75 0.36  p90 0.46
    //   vert       p25 0.03  p50 0.10  p75 0.28  p90 0.62
    // 눈대중으로 정하면 분포의 어디에 걸리는지 알 수 없고, 한 클래스가 90 %
    // 를 먹는 일이 조용히 일어난다 - 실제로 두 번 그랬다.

    // **독립 문턱을 차례로 묻지 않는다. 무엇이 지배적인가를 묻는다.**
    //
    // 앞서는 scatter > 0.36 을 먼저 물었고, 그것은 p75 이라 전체의 4 분의 1 이
    // 무조건 통과했다. 벽면의 스테레오 잡음으로 그 선을 넘으면 **평면성과
    // 법선을 보기도 전에** 나무가 되었다 - 건물 41 칸에 나무 388 칸이 그
    // 순서의 결과다. 한 칸이 동시에 "퍼져 있고" "납작하다" 일 수 있는데
    // 먼저 물은 쪽이 이기는 구조였다.
    //
    // 세 값은 고유값에서 나오고 합이 1 이다. 즉 서로 배타적인 후보이므로
    // 가장 큰 것을 고르는 것이 원래 맞는 사용법이다 (Weinmann 등이 쓰는
    // 차원 특징의 표준 형태다).
    const float pl = c.planarity, li = c.linearity, sc = c.scatter;

    if (pl >= li && pl >= sc) {
        // 평면이다. 법선의 방향이 바닥과 벽을 가른다.
        if (c.vert > 0.55f) {
            // 수직면. 높으면 건물, 낮으면 담장.
            return (top_m >= 2.5f) ? Stuff::Building : Stuff::Fence;
        }
        if (top_m < 1.2f) return Stuff::Ground;

        // **서 있지도 누워 있지도 않은, 기울어진 면.**
        //
        // 여기가 나무가 사라지던 자리다. 스테레오가 보는 것은 수관의 **겉면**
        // 이고, 겉면은 국소적으로 평면이다. 그래서 가로수길인 kitti_04 에서
        // 이 가지에 502 칸이 몰렸고 전부 건물이 되었다 - 화면에는 나무가
        // 스물네 그루뿐인 파란 블록 숲이 남았다.
        //
        // 프로파일의 빈 칸으로는 못 가른다. 실측에서 나무길 쪽이 오히려 더
        // 꽉 차 있었다 (fill p50 = 1.00 대 주택가 0.94) - 길가 덤불부터
        // 수관까지 이어져 보이기 때문이다.
        //
        // 가르는 것은 **법선의 기울기** 다. 같은 가지 안에서:
        //
        //   kitti_04 (가로수길)  vert p25 0.24  p50 0.42  p75 0.47
        //   kitti_00 (주택가)    vert p25 0.01  p50 0.04  p75 0.13
        //
        // 도로면과 지붕은 진짜로 수평이라 vert 가 0 에 붙는다. 잎 덩어리의
        // 겉면은 어느 쪽으로든 기울어 있어 0.4 근처에 앉는다. 0.30 은 두
        // 분포 사이의 빈 구간이다.
        if (c.vert > 0.30f) return Stuff::Vegetation;
        return Stuff::Building;
    }
    if (li >= sc) {
        // 선형이다. 그런데 **선형이라고 다 기둥은 아니다.**
        //
        // 벽면을 부분적으로만 관측하면 3 m 폭의 점이 세로로 한 줄만 남아
        // 선형으로 나온다. 그것 전부를 기둥으로 부르면 주택가 도로에 기둥이
        // 1804 개 서게 된다 - 실제로 그렇게 나왔다.
        //
        // 진짜 기둥은 홀로 서 있다. 가로등 하나가 차지하는 지면칸은 서너
        // 칸이고 그 주위는 비어 있다. 벽은 반대로 이웃이 빽빽하다. 이웃
        // 밀집도가 그 둘을 가른다.
        //
        // 문턱 21 은 밀집도가 "칸이 있는가" 를 세던 시절의 값이라 사실상
        // 아무 것도 거르지 못했다. 솟은 칸만 세도록 고친 뒤 다시 재니
        // kitti_00 에서 기둥 판정 칸의 밀집도가 p25 14 / p50 16 이었고,
        // **확실한 벽면(vert > 0.55)이 p25 15 / p50 16 으로 같았다.**
        // 기둥이라고 부르던 312 칸이 전부 벽·덤불 조각이었다는 뜻이다.
        //
        // 가로등이라면 자기 칸 서넛뿐이라 밀집도가 한 자릿수여야 한다.
        // 8 로 자르면 주택가 한 구간에 기둥 스무남은 개가 남는다 - 가로등과
        // 표지판의 수가 원래 그 정도다.
        if (top_m < 1.5f) return Stuff::Ground;
        if (c.crowd >= 8) {
            // 이웃이 빽빽하다 - 홀로 선 것이 아니라 덩어리의 일부다.
            //
            // **여기서 덩어리가 벽인지 나무인지는 가르지 못한다.** 세 가지를
            // 재 봤고 셋 다 갈라지지 않았다 (kitti_00 주택가 vs kitti_04
            // 가로수길, 이 가지에 든 칸만):
            //
            //           kitti_00(벽)          kitti_04(나무)
            //   vert    p25 .03 p50 .13 p75 .44   p25 .07 p50 .29 p75 .44
            //   scatter p25 .07 p50 .118 p75 .21  p25 .04 p50 .121 p75 .20
            //   밝기    p25 55  p50 85   p75 121   p25 55  p50 100  p75 148
            //
            // vert 는 중앙값만 벌어지고 p75 가 겹친다. 산포는 소수점 셋째
            // 자리까지 같다. 밝기는 가설과 **반대로** 나무 쪽이 밝다 -
            // 하늘을 등진 잎이 회백색으로 날아가기 때문이다.
            //
            // 0.3 m 복셀에 1 m 격자에서, 옆에서 본 벽 조각과 나무줄기 줄은
            // 국소 구조가 실제로 같다. 여기서 문턱을 만들면 kitti_04 에서
            // 170 칸을 얻고 kitti_00 에서 160 칸을 잃는 맞바꾸기일 뿐이라,
            // 가르지 못한다는 사실을 그대로 두고 덩어리 기본값을 쓴다.
            return (top_m >= 2.5f) ? Stuff::Building : Stuff::Fence;
        }
        return Stuff::Pole;
    }
    // 산포가 지배적이다 - 면도 선도 아닌 부피. 잎이 그렇다.
    if (top_m >= 1.5f) return Stuff::Vegetation;
    return Stuff::Ground;
}

// 지도 전체를 기둥으로 접어 라벨을 붙인다.
//
// **지면은 이웃에서 온다.** 각 칸의 최저점을 그 칸의 지면으로 삼으면 top 은
// "그 칸에서 관측된 수직 범위" 가 되어 버린다. 길에서 본 건물 벽면은 3 m 짜리
// 슬라이스로만 보이므로 담장으로 분류되고, 실제로 첫 시도에서 KITTI 00 의
// 지도가 거의 전부 fence 가 되었다.
//
// 반경 안의 최저점을 지면으로 쓰면 벽면 칸은 옆 도로의 높이를 기준으로 삼게
// 되어 진짜 높이가 나온다. 전역 최저점을 쓰지 않는 이유는 반대다 - KITTI 00
// 은 평지가 아니고, 언덕 하나에 지도의 절반이 건물이 된다.
// **자차 주변만 새로 라벨하고, 나머지는 이미 붙은 것을 그대로 둔다.**
//
// 프레임마다 500 m 지도 전체를 다시 분류하면 310 ms 가 든다 (실측). 3 fps 다.
// 그런데 지도는 자차가 지금 보고 있는 곳에서만 자란다 - 뒤쪽 200 m 의 벽면은
// 이번 프레임에도, 다음 프레임에도 같은 벽면이다. 다시 계산할 이유가 없다.
//
// 자차가 결국 모든 곳을 지나가므로 지도는 빠짐없이 라벨된다. 잘라내는 것이
// 아니라 **한 번 계산한 것을 재사용** 하는 것이므로 화면에서 사라지는 것도 없다.
inline void labelScene(const std::unordered_map<std::int64_t, Splat>& cells,
                       const Eigen::Vector3d& up_d, float cell, float voxel,
                       const Eigen::Vector3d& ego, float work_radius,
                       std::unordered_map<std::int64_t, Column>& cols,
                       int ground_radius = 3) {
    const GroundGrid g(up_d.cast<float>(), cell);
    const Eigen::Vector3f e = ego.cast<float>();
    const float r2 = work_radius * work_radius;

    // 이번에 손댈 칸만 모은다. 이전 라벨은 지우지 않는다.
    std::unordered_map<std::int64_t, Column> fresh;
    fresh.reserve(cells.size() / 8 + 1);

    // 1 차: 칸마다 최저/최고 높이와 구조 텐서 누적.
    for (const auto& [k, v] : cells) {
        if ((v.p - e).squaredNorm() > r2) continue;
        const float h = v.p.dot(g.up);
        const auto [i, j] = g.ij(v.p);
        auto& c = fresh[GroundGrid::key(i, j)];
        c.i = i; c.j = j;
        if (h > c.high) { c.high = h; c.rep = v.p; }
        c.low = std::min(c.low, h);
        const Eigen::Vector3d p = v.p.cast<double>();
        ++c.n;
        c.sum += p;
        c.sumsq += p * p.transpose();
    }

    // 2 차: 이웃 반경 안에서 그 칸의 지면 높이를 잡는다.
    //
    // **최솟값을 쓰면 안 된다.** 스테레오 깊이의 이상치 하나가 반경 안의 모든
    // 칸의 지면을 함께 끌어내린다 - 실측에서 지면 위 높이의 중앙값이 11.5 m
    // 로 나왔다. 도로 장면에서 그럴 수는 없다.
    //
    // 이웃 칸들의 최저 높이 **중앙값** 은 이상치 하나에 흔들리지 않는다.
    // 벽면 칸의 최저점도 결국 도로면이므로 중앙값은 지면에 앉는다.
    std::vector<float> nb;
    for (auto& [k, c] : fresh) {
        nb.clear();
        for (int di = -ground_radius; di <= ground_radius; ++di) {
            for (int dj = -ground_radius; dj <= ground_radius; ++dj) {
                const auto it = fresh.find(GroundGrid::key(c.i + di, c.j + dj));
                if (it != fresh.end()) nb.push_back(it->second.low);
            }
        }
        if (nb.empty()) { c.ground = c.low; continue; }
        const std::size_t mid = nb.size() / 2;
        std::nth_element(nb.begin(), nb.begin() + static_cast<std::ptrdiff_t>(mid),
                         nb.end());
        c.ground = nb[mid];
    }

    // 3 차: 지면이 정해졌으니 점유 프로파일을 채운다. 셀을 한 번 더 훑는
    // 대가로 "어느 높이가 찼는가" 를 얻는다 - 그 정보 없이는 나무와 벽이
    // 구분되지 않는다.
    for (const auto& [k, v] : cells) {
        if ((v.p - e).squaredNorm() > r2) continue;
        const auto it = fresh.find(g.key(v.p));
        if (it == fresh.end()) continue;
        const float rel = v.p.dot(g.up) - it->second.ground;
        const int b = static_cast<int>(std::floor(rel / kBinH));
        if (b >= 0 && b < kBins) it->second.bins |= (1u << b);
    }

    // 대표점의 **높이를 지지받는 꼭대기로 내린다.** 수평 위치는 그대로 둔다.
    //
    // rep 은 1 차에서 최고 높이 복셀로 잡혔는데, 그것은 기둥에서 가장 잡음이
    // 심한 값이다. 하늘에 뜬 복셀 하나가 그 칸의 대표가 되면 건물 덩어리가
    // 하늘까지 솟고, 노면 칸 하나가 1 m 위 잡음을 대표로 삼으면 평평한
    // 도로가 울퉁불퉁해진다. 지지받는 꼭대기는 그런 스파이크를 무시한다.
    for (auto& [k, c] : fresh) {
        const int t = c.topBin();
        if (t < 0) continue;
        const float want = c.ground + static_cast<float>(t + 1) * kBinH;
        const float have = c.rep.dot(g.up);
        c.rep += g.up * (want - have);
    }

    // 4 차: 구조 텐서. **국소 3 차원 이웃에서** 계산한다.
    //
    // 처음에는 기둥 아홉 칸을 통째로 더했는데, 그것은 1 x 1 m 바닥에 높이
    // 11 m 인 부피다. 그런 상자 안의 점은 무엇이 들었든 세로로 길게 퍼지므로
    // 선형성이 항상 1 에 가깝게 나온다 - 실측 중앙값이 0.926 이었고 평면성은
    // 0.019 였다. 판별자가 장면이 아니라 자기 이웃의 모양을 재고 있었다.
    //
    // 이웃은 기둥이 아니라 **공** 이어야 한다. 복셀 ±2 칸이면 KITTI 에서 약
    // 3 m 로, 벽면 한 조각과 수관 한 덩이를 가르기에 맞는 크기다.
    for (auto& [k, c] : fresh) {
        int n = 0;
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d sq = Eigen::Matrix3d::Zero();
        const float inv = 1.0f / voxel;
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dz = -2; dz <= 2; ++dz) {
                    const Eigen::Vector3f q = c.rep + Eigen::Vector3f(
                        dx * voxel, dy * voxel, dz * voxel);
                    const auto it = cells.find(voxKey(q, inv));
                    if (it == cells.end()) continue;
                    const Eigen::Vector3d p = it->second.p.cast<double>();
                    ++n; sum += p; sq += p * p.transpose();
                }
            }
        }
        if (n < 8) { c.cls = Stuff::Unknown; continue; }
        const Eigen::Vector3d mean = sum / n;
        Eigen::Matrix3d cov = sq / n - mean * mean.transpose();
        // 수치 안정. 대칭이 깨지면 고유해가 복소수로 샌다.
        cov = 0.5 * (cov + cov.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Vector3d ev = es.eigenvalues();          // 오름차순
        for (int q = 0; q < 3; ++q) ev[q] = std::max(ev[q], 0.0);
        const double l0 = ev[2], l1 = ev[1], l2 = ev[0];   // l0 >= l1 >= l2
        if (l0 < 1e-9) { c.cls = Stuff::Unknown; continue; }
        c.linearity = static_cast<float>((l0 - l1) / l0);
        c.planarity = static_cast<float>((l1 - l2) / l0);
        c.scatter   = static_cast<float>(l2 / l0);
        // 최소 고유값의 고유벡터가 평면의 법선이다.
        const Eigen::Vector3d nrm = es.eigenvectors().col(0).normalized();
        // 법선이 "위" 와 이루는 각. 수직면이면 법선이 수평이라 내적이 0 이다.
        c.vert = static_cast<float>(1.0 - std::abs(nrm.dot(g.up.cast<double>())));
        c.normal = nrm.cast<float>();
        // 이웃 밀집도. 5x5 = 25 칸 중 **지면 위로 솟은** 칸이 몇인가.
        //
        // 선형으로 나온 칸이 홀로 선 기둥인지 옆에서 본 벽인지는 이것으로만
        // 갈린다 - 구조 텐서는 그 칸 안만 보므로 답을 줄 수 없다.
        //
        // **칸이 있는지를 세면 안 된다.** 처음에 그렇게 했더니 도로 위에서는
        // 노면 칸이 25 칸을 다 채워, 인도에 홀로 선 가로등이 벽면보다 높은
        // 밀집도를 받았다 - 재려던 것과 정확히 반대다. 실측에서 기둥으로
        // 판정된 칸의 중앙값이 18 이었고 문턱 21 은 그 바로 위였다.
        //
        // 세어야 하는 것은 **솟은 것이 몇 칸인가** 다. 가로등은 자기 한두
        // 칸뿐이고 벽은 한 줄이 통째로 솟는다.
        {
            int occ = 0;
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    const auto it = fresh.find(GroundGrid::key(c.i + di, c.j + dj));
                    if (it != fresh.end() && it->second.topBin() >= 2) ++occ;
                }
            }
            c.crowd = static_cast<std::uint8_t>(occ);
        }
        c.cls = classifyColumn(c);
    }

    // **공간 일관성.** 칸마다 따로 판정하면 건물 벽면 한 장이 법선 잡음
    // 때문에 조각조각 흩어진다 - kitti_00 에서 나무 2640 대 건물 382 가
    // 나온 것이 그것이다. 주택가 도로가 그렇게 생겼을 리 없다.
    //
    // 구조물은 덩어리로 존재한다. 벽면 한가운데의 칸이 혼자 나무일 수는
    // 없으므로, 5x5 이웃의 다수결로 고친다. 임계값을 더 만지는 것보다
    // 이쪽이 옳다 - 문제는 경계선이 아니라 판정에 이웃이 안 들어간 것이었다.
    {
        std::unordered_map<std::int64_t, Stuff> voted;
        voted.reserve(fresh.size());
        for (const auto& [k, c] : fresh) {
            // **Unknown 은 투표에서 남을 덮지 못한다. 다만 덮일 수는 있다.**
            //
            // Unknown 은 "점이 부족해 판정을 못 했다" 이지 하나의 클래스가
            // 아니다. 그것을 표로 세면 성긴 칸이 다수를 이루는 곳에서 멀쩡히
            // 분류된 벽면까지 Unknown 으로 덮인다 - 실제로 미상이 2897 에서
            // 10776 으로 늘고 건물이 382 에서 100 으로 줄었다. 모른다는 것이
            // 안다는 것을 이길 수는 없다. 그래서 Unknown 은 위 집계에서 빠진다.
            //
            // 그런데 **반대 방향은 막을 이유가 없다.** 벽면 한가운데의 칸이
            // 구조 텐서에 필요한 점 여덟 개를 못 모았다고 해서 그 자리가
            // 벽이 아닌 것은 아니다. 그 칸은 화면에서 통째로 빠지고, 이어져야
            // 할 벽면이 빗살처럼 뚫린다.
            //
            // 두 가지를 요구한다. 첫째, 그 칸이 **자기 관측을 가질 것**
            // (topBin >= 0) - 아무 것도 못 본 칸을 이웃으로 메우면 그것은
            // 없는 구조를 지어내는 것이다. 둘째, 여덟 이웃 중 **넷 이상이
            // 한 클래스일 것** - 덩어리 가장자리가 아니라 안쪽이어야 한다.
            //
            // 실측(kitti_00 한 패스): 미상 1354 칸 중 자기 관측이 있는 것이
            // 789, 그중 이웃 다수가 셋 이상인 것이 300 이다. 나머지 565 는
            // 지면 위에 아무 것도 없는 칸이라 애초에 그릴 것이 없다.
            if (c.cls == Stuff::Unknown) {
                voted[k] = c.cls;
                if (c.topBin() < 0) continue;
                int m[6] = {0, 0, 0, 0, 0, 0};
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const auto it = fresh.find(GroundGrid::key(c.i + di, c.j + dj));
                        if (it == fresh.end()) continue;
                        const Stuff s = it->second.cls;
                        if (s != Stuff::Unknown) m[static_cast<int>(s)]++;
                    }
                }
                int fill_best = 0, fill_n = 0;
                for (int ci = 1; ci < 6; ++ci) {
                    if (m[ci] > fill_n) { fill_n = m[ci]; fill_best = ci; }
                }
                if (fill_n >= 4) voted[k] = static_cast<Stuff>(fill_best);
                continue;
            }
            int n[6] = {0, 0, 0, 0, 0, 0};
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    const auto it = fresh.find(GroundGrid::key(c.i + di, c.j + dj));
                    if (it != fresh.end() && it->second.cls != Stuff::Unknown) {
                        n[static_cast<int>(it->second.cls)]++;
                    }
                }
            }
            // 자기 자신에 가중치를 준다. 다수결이 완전히 지배하면 가느다란
            // 기둥이나 홀로 선 나무가 주변 지면에 먹혀 사라진다.
            n[static_cast<int>(c.cls)] += 2;
            int best = static_cast<int>(c.cls), bn = -1;
            for (int ci = 1; ci < 6; ++ci) {
                if (n[ci] > bn) { bn = n[ci]; best = ci; }
            }
            voted[k] = static_cast<Stuff>(best);
        }
        for (auto& [k, c] : fresh) c.cls = voted[k];
    }

    // **누적 지도에 합친다.** 이번에 손대지 않은 칸의 라벨은 그대로 남는다 -
    // 지나온 곳의 건물이 시야에서 벗어났다고 사라지면 그것은 세계 모델이
    // 아니라 지금 보이는 것의 목록이다.
    for (const auto& [k, c] : fresh) cols[k] = c;
}

// ---------------------------------------------------------------------------
// 앞을 내다보기 — 관측된 구조를 진행 방향으로 외삽하고, 가서 채점한다
// ---------------------------------------------------------------------------
// **이것은 생성 모델이 아니다.** 학습된 것이 없고, 여기서 하는 일은 지금까지
// 본 회랑(지면 높이, 좌우 구조의 측방 거리, 그 클래스)을 진행 방향으로 늘리는
// 것뿐이다. 그렇게 부르지 않는 이유는 간단하다 - 그럴듯한 그림은 아무것도
// 증명하지 않고, 이 저장소가 반복해 기록한 실패가 정확히 그것이다.
//
// 대신 **채점한다.** 예측한 자리에 실제로 도달하면 거기 무엇이 있었는지 보고
// 맞았는지 틀렸는지를 센다. 적중률이 화면에 계속 떠 있으므로, 예측이 쓸모가
// 있는지 없는지가 그림이 아니라 숫자로 나온다. 나중에 생성 모델로 바꾸더라도
// 같은 자로 재면 비교가 된다.
struct Prediction {
    Eigen::Vector3f p{Eigen::Vector3f::Zero()};  // 예측한 자리 (기둥 꼭대기)
    float  height{0.0f};                          // 지면 위 높이
    Stuff  cls{Stuff::Unknown};
    float  conf{0.0f};      // 거리에 따라 감쇠. 먼 예측은 약하게 건다
    int    made{0};         // 몇 번 프레임에서 예측했나
    int    grade{0};        // 0 미채점 / +1 적중 / -1 빗나감
};

// 예측의 성적표. 성장형의 "성장" 은 이 숫자가 움직이는 것으로만 확인된다.
struct PredictScore {
    int hit{0}, miss{0}, missed{0};   // 적중 / 헛집음 / 놓침
    // 회랑 폭 추정. 채점 결과가 이 값을 밀고 당긴다 - 그것이 되먹임이다.
    float lateral{6.0f};
    float lateral_n{1.0f};

    double precision() const {
        const int d = hit + miss;
        return d > 0 ? static_cast<double>(hit) / d : std::nan("");
    }
    double recall() const {
        const int d = hit + missed;
        return d > 0 ? static_cast<double>(hit) / d : std::nan("");
    }
    void clear() { hit = miss = missed = 0; lateral = 6.0f; lateral_n = 1.0f; }
};

// ---------------------------------------------------------------------------
// 노면 텍스처 — 도로는 2 차원이므로 3 차원 해상도를 쓸 이유가 없다
// ---------------------------------------------------------------------------
// 주차선은 폭이 15 cm 다. 0.3 m 복셀로는 선이 칸보다 가늘어서 표현할 수 있는
// 크기가 아니고, 그래서 밝기를 들고 와도 선이 나오지 않았다.
//
// 그렇다고 부피 복셀을 0.1 m 로 줄이면 메모리가 27 배가 된다. 그럴 필요가
// 없다 - **노면은 면이지 부피가 아니다.** 높이 축을 뺀 2 차원 격자를 따로
// 두면 같은 메모리로 세 배 촘촘한 해상도를 얻는다. 55 m 반경 0.1 m 격자가
// 100 만 칸, 칸당 12 바이트로 12 MB 다.
struct RoadCell {
    float        h{0.0f};        // 그 칸의 노면 높이
    float        range{0.0f};    // 관측 거리. 가까이서 본 것이 이긴다
    std::uint8_t intensity{0};   // 관측된 밝기 - 흰 페인트가 여기 남는다
    std::uint8_t hits{0};
};

// 노면 격자 키. 높이 축을 뺀 2 차원이다.
inline std::int64_t roadKey(const Eigen::Vector3f& p, const Eigen::Vector3f& a,
                            const Eigen::Vector3f& b, float inv) {
    const auto q = [inv](float v) {
        return static_cast<std::int64_t>(std::floor(static_cast<double>(v) * inv));
    };
    return ((q(p.dot(a)) & 0x3FFFFFF) << 26) | (q(p.dot(b)) & 0x3FFFFFF);
}

// ---------------------------------------------------------------------------
// Naive Surface Nets — 지나온 곳 전부를 하나의 삼각형 표면으로
// ---------------------------------------------------------------------------
// 복셀을 큐브로 채우면 지도는 "블록 더미" 로 남는다. 지나온 곳을 진짜 3D
// 모형으로 만들려면 닫힌 표면이 있어야 하고, 그것이 이 단계다.
//
// **Marching Cubes 가 아니라 Surface Nets 인 이유.** 이진 점유 필드에서 MC 는
// 모든 엣지 교차가 t=0.5 에 고정되어 정점이 격자 중점에만 놓이고, 결과가
// 계단으로 나온다. 게다가 256x16 삼각형 룩업 테이블이 필요하다. Surface Nets
// 는 셀마다 정점 하나이고 (Gibson, MERL TR99-24), 같은 구에서 정점 272 개 대
// MC 1140 개로 넉 배 가볍다.
//
// 이진 필드를 그대로 쓰면 SN 도 중점 고정 문제를 겪는다. 그래서 3x3x3 이웃의
// 점유 **비율** 을 스칼라 필드로 쓴다 - 한 줄로 이진 필드가 밀도장이 되고,
// 엣지 교차가 0.5 에서 벗어나면서 표면이 매끄러워진다.
struct SurfMesh {
    std::vector<Eigen::Vector3f> vert;
    std::vector<Eigen::Vector3f> norm;
    std::vector<std::array<int, 3>> tri;
    void clear() { vert.clear(); norm.clear(); tri.clear(); }
};

// 밀도장. **한 번에 미리 만든다.**
//
// 코너마다 27 이웃을 세면 조회가 셀당 8x27 이 되고, 70 만 복셀이면 10 억 회를
// 넘어 한 프레임이 분 단위가 된다 (실측: 600 초에도 안 끝났다). 반대로 하면
// 된다 - 점유 복셀 하나가 자기 주변 27 칸의 카운터를 올리게 하면 전체가
// 70 만 x 27 = 1900 만 회 한 패스로 끝나고, 이후 조회는 O(1) 이다.
//
// 같은 답을 얻는 두 방향인데 한쪽은 50 배 싸다. 이웃을 "묻는" 대신 "알리는"
// 것이 그 차이다.
using DensField = std::unordered_map<std::int64_t, std::uint8_t>;

inline DensField buildDensity(const std::unordered_map<std::int64_t, Splat>& cells,
                              float voxel, const Eigen::Vector3f& e, float r2,
                              const Eigen::Vector3f& up, float hmin,
                              const std::unordered_map<std::int64_t, Column>& labels,
                              const GroundGrid& lg) {
    DensField d;
    d.reserve(cells.size() * 3);
    const float inv = 1.0f / voxel;
    for (const auto& [k, v] : cells) {
        if ((v.p - e).squaredNorm() > r2) continue;
        // **노면은 메시에 넣지 않는다.** 도로는 이미 밝기 텍스처로 그린다.
        // 같은 것을 메시로 또 만들면, 평평해야 할 노면이 스테레오 잡음을 따라
        // 울퉁불퉁한 바위 덩어리가 된다 - 실제로 그렇게 나왔다.
        if (v.p.dot(up) < hmin) continue;
        // **라벨이 구조물인 칸만 메시로 만든다.**
        //
        // 도로 한가운데 떠 있던 회색 암석 덩어리는 라벨이 미상인 잡음이었다.
        // 미상은 "점이 부족해 판정을 못 했다" 이므로 구조물이라는 근거가 없고,
        // 근거 없는 것을 표면으로 만들면 없는 바위가 생긴다. 분류가 이미
        // 끝나 있으니 그 답을 쓰면 된다.
        {
            const auto lit = labels.find(lg.key(v.p));
            if (lit == labels.end()) continue;
            const Stuff sc = lit->second.cls;
            if (sc == Stuff::Unknown || sc == Stuff::Ground) continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const Eigen::Vector3f q = v.p + Eigen::Vector3f(
                        dx * voxel, dy * voxel, dz * voxel);
                    auto& c = d[voxKey(q, inv)];
                    if (c < 255) ++c;
                }
            }
        }
    }
    return d;
}

// 스칼라 필드. 안쪽이 음수다.
inline float sdfAt(const DensField& dens, const Eigen::Vector3f& p,
                   float inv, float iso) {
    const auto it = dens.find(voxKey(p, inv));
    const float n = (it == dens.end()) ? 0.0f : static_cast<float>(it->second);
    return iso - n / 27.0f;
}

// 지도를 삼각형 표면으로 바꾼다. 자차 주변 반경만 - 전체를 매 프레임 다시
// 만들 이유가 없고, 결과는 캐시된다.
inline void surfaceNets(const std::unordered_map<std::int64_t, Splat>& cells,
                        float voxel, const Eigen::Vector3d& ego, double radius,
                        const Eigen::Vector3d& up_d, double ground_h,
                        double min_above,
                        const std::unordered_map<std::int64_t, Column>& labels,
                        const GroundGrid& lg, SurfMesh& out) {
    out.clear();
    if (cells.empty()) return;
    const float inv = 1.0f / voxel;
    const float iso = 0.35f;
    const Eigen::Vector3f e = ego.cast<float>();
    const float r2 = static_cast<float>(radius * radius);
    // 밀도장을 먼저 한 패스로 만든다. 이후 모든 조회가 O(1) 이 된다.
    const Eigen::Vector3f upf = up_d.cast<float>().normalized();
    const float hmin = static_cast<float>(ground_h + min_above);
    const DensField dens = buildDensity(cells, voxel, e, r2, upf, hmin,
                                        labels, lg);

    // 셀 -> 정점 인덱스. 셀마다 정점 하나이므로 중복 제거가 필요 없다.
    std::unordered_map<std::int64_t, int> vidx;
    vidx.reserve(cells.size());

    // 후보 셀: 점유 복셀과 그 이웃. 빈 공간 전체를 훑지 않는다.
    std::unordered_map<std::int64_t, Eigen::Vector3f> cand;
    cand.reserve(cells.size() * 2);
    for (const auto& [k, v] : cells) {
        if ((v.p - e).squaredNorm() > r2) continue;
        if (v.p.dot(upf) < hmin) continue;      // 노면 제외
        {
            const auto lit = labels.find(lg.key(v.p));
            if (lit == labels.end()) continue;
            const Stuff sc = lit->second.cls;
            if (sc == Stuff::Unknown || sc == Stuff::Ground) continue;
        }
        // **표면 복셀만 후보로 삼는다.**
        //
        // 복셀마다 여덟 칸을 후보로 넣으면 70 만 복셀에서 후보가 560 만 개가
        // 되고, 그 대부분은 덩어리 **안쪽** 이라 코너 여덟 개를 다 조회한 뒤
        // mask 가 0 이나 0xFF 여서 버려진다. 조회 1 억 회가 통째로 낭비다.
        //
        // 표면은 정의상 빈 이웃이 있는 자리에만 있다. 여섯 면 중 하나라도
        // 비어 있지 않으면 그 복셀은 안쪽이고, 표면이 지나갈 수 없다.
        bool boundary = false;
        for (int ax = 0; ax < 3 && !boundary; ++ax) {
            for (int sgn = -1; sgn <= 1 && !boundary; sgn += 2) {
                Eigen::Vector3f d = Eigen::Vector3f::Zero();
                d[ax] = sgn * voxel;
                if (!cells.count(voxKey(v.p + d, inv))) boundary = true;
            }
        }
        if (!boundary) continue;
        for (int dx = -1; dx <= 0; ++dx) {
            for (int dy = -1; dy <= 0; ++dy) {
                for (int dz = -1; dz <= 0; ++dz) {
                    const Eigen::Vector3f c = v.p + Eigen::Vector3f(
                        dx * voxel, dy * voxel, dz * voxel);
                    cand[voxKey(c, inv)] = c;
                }
            }
        }
    }

    static const int kCorner[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                                      {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    static const int kEdge[12][2] = {{0,1},{1,2},{2,3},{3,0},
                                     {4,5},{5,6},{6,7},{7,4},
                                     {0,4},{1,5},{2,6},{3,7}};

    for (const auto& [key, c] : cand) {
        float d[8];
        int mask = 0;
        for (int i = 0; i < 8; ++i) {
            const Eigen::Vector3f q = c + Eigen::Vector3f(
                kCorner[i][0] * voxel, kCorner[i][1] * voxel, kCorner[i][2] * voxel);
            d[i] = sdfAt(dens, q, inv, iso);
            if (d[i] < 0.0f) mask |= (1 << i);
        }
        if (mask == 0 || mask == 0xFF) continue;   // 표면이 지나가지 않는다

        // 부호가 바뀌는 엣지의 교차점들. 그 **무게중심** 이 이 셀의 정점이다.
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        int cnt = 0;
        for (const auto& ed : kEdge) {
            const bool a = (mask >> ed[0]) & 1, b = (mask >> ed[1]) & 1;
            if (a == b) continue;
            const float da = d[ed[0]], db = d[ed[1]];
            const float t = (std::abs(da - db) < 1e-9f) ? 0.5f : da / (da - db);
            const Eigen::Vector3f pa = c + Eigen::Vector3f(
                kCorner[ed[0]][0] * voxel, kCorner[ed[0]][1] * voxel,
                kCorner[ed[0]][2] * voxel);
            const Eigen::Vector3f pb = c + Eigen::Vector3f(
                kCorner[ed[1]][0] * voxel, kCorner[ed[1]][1] * voxel,
                kCorner[ed[1]][2] * voxel);
            sum += pa + (pb - pa) * t;
            ++cnt;
        }
        if (cnt == 0) continue;
        vidx[key] = static_cast<int>(out.vert.size());
        out.vert.push_back(sum / static_cast<float>(cnt));
        // 법선은 필드의 기울기다. 중앙차분 여섯 번.
        const float h = voxel * 0.5f;
        const Eigen::Vector3f g(
            sdfAt(dens, out.vert.back() + Eigen::Vector3f(h,0,0), inv, iso) -
            sdfAt(dens, out.vert.back() - Eigen::Vector3f(h,0,0), inv, iso),
            sdfAt(dens, out.vert.back() + Eigen::Vector3f(0,h,0), inv, iso) -
            sdfAt(dens, out.vert.back() - Eigen::Vector3f(0,h,0), inv, iso),
            sdfAt(dens, out.vert.back() + Eigen::Vector3f(0,0,h), inv, iso) -
            sdfAt(dens, out.vert.back() - Eigen::Vector3f(0,0,h), inv, iso));
        out.norm.push_back(g.norm() > 1e-9f ? g.normalized()
                                            : Eigen::Vector3f(0, 1, 0));
    }

    // **제약 이완.** Gibson 논문의 핵심이고, "Naive" 판이 생략한 부분이다.
    //
    // 교차점 무게중심만 쓰면 표면이 입력만큼 거칠다. 스테레오 지도는 원래
    // 거칠므로 결과가 각진 바위 덩어리로 나온다 - 실제로 그렇게 나왔다.
    //
    // 각 정점을 이웃 정점들의 평균 쪽으로 조금씩 당기되, **자기 셀 밖으로는
    // 못 나가게** 묶는다. 그 제약이 Gibson 이 말한 요점이다: 제약이 없으면
    // 표면이 수축해 얇은 것들이 사라지고, 제약이 있으면 세부는 남으면서
    // 잡음만 펴진다.
    {
        // 정점 이웃 관계는 셀 격자에서 온다. 여섯 면 이웃의 정점이 이웃이다.
        std::vector<Eigen::Vector3f> home = out.vert;
        const Eigen::Vector3f ax6[6] = {{voxel,0,0},{-voxel,0,0},{0,voxel,0},
                                        {0,-voxel,0},{0,0,voxel},{0,0,-voxel}};
        std::vector<std::int64_t> keys(out.vert.size());
        for (const auto& [key, idx] : vidx) keys[static_cast<std::size_t>(idx)] = key;

        std::vector<Eigen::Vector3f> next(out.vert.size());
        for (int iter = 0; iter < 4; ++iter) {
            for (std::size_t i = 0; i < out.vert.size(); ++i) {
                Eigen::Vector3f sum = Eigen::Vector3f::Zero();
                int n = 0;
                for (const auto& d : ax6) {
                    const auto it = vidx.find(voxKey(home[i] + d, inv));
                    if (it == vidx.end()) continue;
                    sum += out.vert[static_cast<std::size_t>(it->second)];
                    ++n;
                }
                if (n < 2) { next[i] = out.vert[i]; continue; }
                Eigen::Vector3f p = out.vert[i] * 0.5f + (sum / static_cast<float>(n)) * 0.5f;
                // 자기 셀 안으로 되돌린다. 이것이 수축을 막는 제약이다.
                const Eigen::Vector3f lo = home[i] - Eigen::Vector3f::Constant(voxel * 0.5f);
                const Eigen::Vector3f hi = home[i] + Eigen::Vector3f::Constant(voxel * 0.5f);
                next[i] = p.cwiseMax(lo).cwiseMin(hi);
            }
            out.vert.swap(next);
        }
        // 이완 후 법선을 다시 잡는다. 옛 법선은 옛 위치의 것이다.
        for (std::size_t i = 0; i < out.vert.size(); ++i) {
            const float h = voxel * 0.5f;
            const Eigen::Vector3f g(
                sdfAt(dens, out.vert[i] + Eigen::Vector3f(h,0,0), inv, iso) -
                sdfAt(dens, out.vert[i] - Eigen::Vector3f(h,0,0), inv, iso),
                sdfAt(dens, out.vert[i] + Eigen::Vector3f(0,h,0), inv, iso) -
                sdfAt(dens, out.vert[i] - Eigen::Vector3f(0,h,0), inv, iso),
                sdfAt(dens, out.vert[i] + Eigen::Vector3f(0,0,h), inv, iso) -
                sdfAt(dens, out.vert[i] - Eigen::Vector3f(0,0,h), inv, iso));
            if (g.norm() > 1e-9f) out.norm[i] = g.normalized();
        }
    }

    // 면: 부호가 바뀌는 격자 엣지마다, 그 엣지를 공유하는 네 셀의 정점을 잇는다.
    const Eigen::Vector3f ax[3] = {{voxel,0,0}, {0,voxel,0}, {0,0,voxel}};
    for (const auto& [key, c] : cand) {
        const auto self = vidx.find(key);
        if (self == vidx.end()) continue;
        for (int a = 0; a < 3; ++a) {
            const int b = (a + 1) % 3, d2 = (a + 2) % 3;
            const auto i1 = vidx.find(voxKey(c - ax[b], inv));
            const auto i2 = vidx.find(voxKey(c - ax[b] - ax[d2], inv));
            const auto i3 = vidx.find(voxKey(c - ax[d2], inv));
            if (i1 == vidx.end() || i2 == vidx.end() || i3 == vidx.end()) continue;
            // **빈 공간을 가로지르는 삼각형을 만들지 않는다.**
            //
            // 격자에서 이웃한 셀이라도 그 안의 정점은 셀 안 어디에나 놓일 수
            // 있다. 두 정점이 멀면 삼각형이 관측이 없는 구간을 건너뛰어,
            // 화면에는 있지도 않은 커다란 각진 덩어리가 생긴다.
            const float lim2 = 4.0f * voxel * voxel;
            const Eigen::Vector3f& v0 = out.vert[static_cast<std::size_t>(self->second)];
            if ((out.vert[static_cast<std::size_t>(i1->second)] - v0).squaredNorm() > lim2 ||
                (out.vert[static_cast<std::size_t>(i2->second)] - v0).squaredNorm() > lim2 ||
                (out.vert[static_cast<std::size_t>(i3->second)] - v0).squaredNorm() > lim2) {
                continue;
            }
            out.tri.push_back({self->second, i1->second, i2->second});
            out.tri.push_back({self->second, i2->second, i3->second});
        }
    }
}

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

    // --- 움직이는가 ---
    //
    // 판정은 **기억 모델 안에서** 한다. 클래스로 정하지 않는다: 주차된 차는
    // 지도의 일부이고 서 있는 사람도 마찬가지다. "person 이니까 동적" 은
    // 관측이 아니라 편견이고, 06-results.md 14.1 이 그 규칙으로 앉아 있는
    // 사람에게 15.7 배를 잃은 기록이다.
    //
    // 재는 것은 **처음 본 자리에서 얼마나 멀어졌는가의 최댓값** 이다.
    // 관측 간 이동량의 합이 아니다 - 그러면 제자리에서 떠는 검출 상자가
    // 전부 동적이 된다. 왕복해서 제자리로 돌아오는 보행자를 놓치지도 않는다.
    Eigen::Vector3d first_c{Eigen::Vector3d::Zero()};  // 처음 본 자리
    Eigen::Vector3d cur_c{Eigen::Vector3d::Zero()};    // 가장 최근에 본 자리
    double span{0.0};      // first_c 로부터의 최대 거리
    bool   dynamic{false};
    // 동적으로 판정되면 이 자리들에 남긴 복셀을 지운다. 지금 자리도 다음
    // 프레임이면 과거가 되므로 계속 채워진다.
    std::vector<Eigen::Vector3d> trail;

    Eigen::Vector3d center() const { return dynamic ? cur_c : sum_c / std::max(1, count); }
    Eigen::Vector3d size()   const { return sum_s / std::max(1, count); }

    void observe(const Eigen::Vector3d& c, const Eigen::Vector3d& sz, int frame) {
        if (count == 0) first_c = c;
        sum_c += c;
        sum_s += sz;
        ++count;
        seen = frame;
        cur_c = c;
        span = std::max(span, (c - first_c).norm());

        // 판정에는 관측이 몇 번 필요하다. 두 번으로 정하면 검출 하나가
        // 튀는 것만으로 지도의 일부가 지워진다.
        if (count >= 4) {
            // 문턱은 물체 자신의 크기에서 나온다 - 제 발자국을 벗어났으면
            // 움직인 것이고, 몇 cm 떠는 것은 검출 상자의 잡음이다.
            //
            // 최대 변보다 작게 잡는 이유: 사람의 최대 변은 **키** 라 1.7 m 다.
            // 그걸 그대로 쓰면 방을 가로질러 걷는 사람도 한참 동안 정적으로
            // 남는다. 움직임과 비교할 크기는 높이가 아니라 폭이므로 0.6 배를
            // 쓰고, 아주 작은 검출이 잡음으로 동적이 되지 않게 바닥을 둔다.
            const double own = std::max(0.30, 0.6 * size().maxCoeff());
            if (span > own) dynamic = true;
        }
        if (dynamic) trail.push_back(c);
    }
};

// 복셀 지도에서 상자 하나가 덮는 영역을 지운다.
//
// 움직이는 물체가 지나간 자리에는 그때의 표면이 그대로 남는다. 지우지 않으면
// 보행자 한 명이 지도에 자기 모습을 수십 개 남기고, 그것이 "세계를 기억한다"
// 가 아니라 "지나간 것을 못 잊는다" 가 된다.
inline void eraseBox(VoxelMap& vm, const Eigen::Vector3d& c,
                     const Eigen::Vector3d& size) {
    // 상자보다 조금 넉넉히. 검출 상자는 물체 경계에 딱 맞지 않는다.
    const Eigen::Vector3d h = size * 0.5 * 1.35 + Eigen::Vector3d::Constant(vm.voxel);
    const float inv = 1.0f / vm.voxel;
    // 지울 복셀 수의 상한. 상자가 터무니없이 크게 잡히면 지도를 통째로
    // 지울 수 있으므로 막는다 - 지우기는 되돌릴 수 없는 연산이다.
    const double cells = (2 * h.x() / vm.voxel + 2) * (2 * h.y() / vm.voxel + 2) *
                         (2 * h.z() / vm.voxel + 2);
    if (!(cells > 0.0) || cells > 20000.0) return;
    for (double x = c.x() - h.x(); x <= c.x() + h.x() + vm.voxel; x += vm.voxel) {
        for (double y = c.y() - h.y(); y <= c.y() + h.y() + vm.voxel; y += vm.voxel) {
            for (double z = c.z() - h.z(); z <= c.z() + h.z() + vm.voxel; z += vm.voxel) {
                vm.cells.erase(voxKey(Eigen::Vector3f(static_cast<float>(x),
                                                      static_cast<float>(y),
                                                      static_cast<float>(z)), inv));
            }
        }
    }
}

// 지금까지 본 회랑을 앞으로 100 m 늘린다.
//
// 재는 것은 세 가지뿐이다. 관측 범위 안에서
//   - 지면 높이가 진행 방향으로 어떻게 변하는가 (선형 추세)
//   - 구조(건물/담장/나무)가 진행축에서 좌우로 얼마나 떨어져 있는가
//   - 그 구조가 어느 클래스이고 얼마나 높은가
// 그리고 그것을 앞으로 늘린다. 곡선 도로는 직선으로 예측되고 교차로는 예측이
// 통째로 틀린다 - 그것을 숨기지 않는 것이 채점을 붙인 이유다.
inline void predictAhead(const std::unordered_map<std::int64_t, Column>& cols,
                         const Eigen::Vector3d& ego, const Eigen::Vector3d& dir_d,
                         const Eigen::Vector3d& up_d, float cell, int frame,
                         PredictScore& score,
                         std::unordered_map<std::int64_t, Prediction>& out) {
    if (cols.empty() || dir_d.norm() < 1e-6) return;
    const GroundGrid g(up_d.cast<float>(), cell);
    const Eigen::Vector3f up = g.up;
    const Eigen::Vector3f fwd = dir_d.cast<float>().normalized();
    const Eigen::Vector3f left = up.cross(fwd).normalized();
    const Eigen::Vector3f e = ego.cast<float>();

    // 1) 최근 관측에서 회랑을 읽는다. 뒤로 30 m, 좌우 40 m 안쪽만 본다 -
    //    더 넓히면 지나온 다른 길의 구조가 섞인다.
    struct Side { float sum{0}; int n{0}; float h{0}; int cls_n[6] = {0,0,0,0,0,0}; };
    Side sd[2];
    float gsum = 0.0f, gn = 0.0f, gslope_num = 0.0f, gslope_den = 0.0f;
    for (const auto& [k, c] : cols) {
        const Eigen::Vector3f d = c.rep - e;
        const float along = d.dot(fwd);
        if (along < -30.0f || along > 5.0f) continue;
        const float lat = d.dot(left);
        if (std::abs(lat) > 40.0f) continue;
        const float h = static_cast<float>(c.topBin() + 1) * kBinH;

        if (c.cls == Stuff::Ground) {
            const float gh = c.rep.dot(up) - h;
            gsum += gh; gn += 1.0f;
            gslope_num += along * gh; gslope_den += along * along;
            continue;
        }
        if (c.cls == Stuff::Unknown || h < 1.0f) continue;
        Side& s = sd[lat >= 0.0f ? 0 : 1];
        s.sum += std::abs(lat); s.n += 1; s.h += h;
        s.cls_n[static_cast<int>(c.cls)]++;
    }
    if (gn < 4.0f) return;
    const float g0 = gsum / gn;
    const float slope = (gslope_den > 1e-3f) ? (gslope_num / gslope_den) : 0.0f;

    // 2) 좌우 측방 거리. 관측이 없으면 지금까지의 추정을 그대로 쓴다 -
    //    그것이 되먹임으로 갱신되는 값이다.
    for (int s = 0; s < 2; ++s) {
        if (sd[s].n >= 3) {
            const float lat = sd[s].sum / static_cast<float>(sd[s].n);
            // 지수 이동. 한 프레임의 관측으로 통째로 갈아치우면 교차로에서
            // 회랑 추정이 튄다.
            score.lateral = 0.9f * score.lateral + 0.1f * lat;
        }
    }

    // 3) 앞으로 늘린다. 신뢰도는 거리에 따라 떨어뜨린다 - 100 m 앞의 예측과
    //    5 m 앞의 예측을 같은 무게로 채점하면 채점이 의미를 잃는다.
    for (float d = 4.0f; d <= 100.0f; d += cell) {
        const Eigen::Vector3f centre = e + fwd * d;
        const float gh = g0 + slope * d;
        const float conf = std::clamp(1.0f - d / 120.0f, 0.05f, 1.0f);

        for (int s = 0; s < 2; ++s) {
            if (sd[s].n < 3) continue;
            // 그 쪽에서 가장 많이 본 클래스를 그대로 세운다.
            int best = 0, bn = 0;
            for (int ci = 1; ci < 6; ++ci) {
                if (sd[s].cls_n[ci] > bn) { bn = sd[s].cls_n[ci]; best = ci; }
            }
            if (best == 0) continue;
            const float h = sd[s].h / static_cast<float>(sd[s].n);
            const float sign = (s == 0 ? 1.0f : -1.0f);

            // **선이 아니라 띠로 예측한다.**
            //
            // 회랑 가장자리에 한 줄만 세우면 재현율이 구조적으로 낮다 -
            // 실측 2 % 였다. 장면의 구조는 벽면 한 겹이 아니라 그 뒤로
            // 이어지는 부피이기 때문이다. 건물 정면이 보였으면 그 뒤에도
            // 건물이 있고, 가로수가 한 그루 있으면 그 줄로 이어진다.
            //
            // 넓히면 정밀도가 떨어질 수 있다. 어느 쪽이 얼마나 움직이는지는
            // 아래 채점이 말해 준다 - 그것이 이 숫자를 붙인 이유다.
            for (float w = 0.0f; w <= 6.0f; w += cell) {
                const float lat = (score.lateral + w) * sign;
                const Eigen::Vector3f p = centre + left * lat + up * (gh + h);
                const std::int64_t key = g.key(p);
                // 이미 실제로 관측한 자리에는 예측을 세우지 않는다. 예측이
                // 관측을 덮으면 채점이 자기 자신을 맞히게 된다.
                if (cols.count(key) || out.count(key)) continue;
                // 안쪽일수록 확신이 크다. 회랑에서 멀어질수록 근거가 약하다.
                const float cw = conf * std::clamp(1.0f - w / 8.0f, 0.2f, 1.0f);
                out[key] = Prediction{p, h, static_cast<Stuff>(best), cw, frame, 0};
            }
        }
    }
}

// 도달한 자리의 예측을 채점한다.
//
// 채점 시점이 중요하다. 자차가 그 자리를 **지나간 뒤** 에 봐야 관측이 다
// 들어와 있다. 지나가기 전에 채점하면 아직 안 본 것을 "없다" 로 세게 된다 -
// 그것은 예측이 아니라 관측 진행 상황을 재는 것이다.
inline void gradePredictions(std::unordered_map<std::int64_t, Prediction>& preds,
                             const std::unordered_map<std::int64_t, Column>& cols,
                             const Eigen::Vector3d& ego, const Eigen::Vector3d& dir_d,
                             const Eigen::Vector3d& up_d, float cell,
                             PredictScore& score) {
    if (preds.empty() || dir_d.norm() < 1e-6) return;
    const GroundGrid g(up_d.cast<float>(), cell);
    const Eigen::Vector3f fwd = dir_d.cast<float>().normalized();
    const Eigen::Vector3f e = ego.cast<float>();

    for (auto& [k, pr] : preds) {
        if (pr.grade != 0) continue;
        const float along = (pr.p - e).dot(fwd);
        if (along > -3.0f) continue;            // 아직 안 지나갔다
        const auto it = cols.find(k);
        if (it == cols.end()) {
            // 지나갔는데 그 칸에 아무 관측도 없다. 관측 자체가 없는 것은
            // "틀렸다" 가 아니라 "확인 불가" 다 - 채점하지 않고 남겨 둔다.
            continue;
        }
        const bool structure = (it->second.cls != Stuff::Ground &&
                                it->second.cls != Stuff::Unknown);
        if (structure) { pr.grade = 1;  ++score.hit; }
        else           { pr.grade = -1; ++score.miss; }
    }

    // 놓친 것: 지나온 자리에 구조가 실제로 있었는데 예측이 없었던 칸.
    // 이것을 세지 않으면 "적게 예측하고 다 맞히기" 가 만점을 받는다.
    for (const auto& [k, c] : cols) {
        if (c.cls == Stuff::Ground || c.cls == Stuff::Unknown) continue;
        const float along = (c.rep - e).dot(fwd);
        if (along > -3.0f || along < -20.0f) continue;   // 방금 지나온 구간만
        if (!preds.count(k)) ++score.missed;
    }
}

class CloudView {
public:
    void reset(int w, int h) {
        if (img_.cols != w || img_.rows != h) {
            img_.create(h, w, CV_8UC3);
            zbuf_.create(h, w, CV_32F);
        }
        img_.setTo(C_VOID);
        zbuf_.setTo(std::numeric_limits<float>::max());
    }

    // 화면에 투영. f 는 초점거리(px), 카메라는 orbit 중심을 바라본다.
    // 기준 바닥 격자.
    //
    // 노면을 칸마다 사각형으로 그리던 것을 대신한다. 관측된 지면 칸을 하나씩
    // 그리면 관측이 성긴 곳은 구멍이 나고 잡음이 있는 곳은 타일이 어긋나서,
    // 평평해야 할 바닥이 누더기로 보인다. 바닥은 **재현할 대상이 아니라 척도**
    // 다 - 3D 뷰어가 예외 없이 균일한 격자를 까는 이유가 그것이다.
    //
    // 높이는 관측된 지면에서 받아 오므로 지도와 따로 놀지는 않는다.
    void groundPlane(const Eigen::Vector3d& at, double height,
                     const Eigen::Vector3d& up_d, const Orbit& orb, double f,
                     double step, double reach) {
        const Eigen::Vector3d up = up_d.normalized();
        Eigen::Vector3d a = up.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = up.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = up.cross(a).normalized();

        // 격자를 자차와 함께 미끄러뜨리지 않는다. 월드에 고정된 간격에
        // 스냅해야 선이 흐르지 않고 바닥이 정지해 보인다.
        const double ca = std::floor(at.dot(a) / step) * step;
        const double cb = std::floor(at.dot(b) / step) * step;
        const Eigen::Vector3d o = a * ca + b * cb + up * height;

        const int n = std::max(1, static_cast<int>(reach / step));
        // 격자는 **배경**이다. 지도보다 밝으면 눈이 먼저 격자로 가고, 척도로
        // 쓰라고 깐 것이 정작 보여 줄 것을 가린다.
        const cv::Scalar thin(30, 26, 22), thick(52, 45, 38);
        for (int i = -n; i <= n; ++i) {
            // 다섯 칸마다 굵게. 균일한 선만 있으면 거리가 안 읽힌다.
            const cv::Scalar& col = (((static_cast<int>(std::llround(ca / step)) + i) % 5) == 0)
                                  ? thick : thin;
            const Eigen::Vector3d p0 = o + a * (i * step) - b * reach;
            const Eigen::Vector3d p1 = o + a * (i * step) + b * reach;
            line3(p0, p1, orb, f, col, 1);
            const cv::Scalar& col2 = (((static_cast<int>(std::llround(cb / step)) + i) % 5) == 0)
                                   ? thick : thin;
            const Eigen::Vector3d q0 = o + b * (i * step) - a * reach;
            const Eigen::Vector3d q1 = o + b * (i * step) + a * reach;
            line3(q0, q1, orb, f, col2, 1);
        }
    }

    // 노면 텍스처. 정밀 2 차원 격자를 밝기로 칠한다.
    //
    // 부피 복셀(0.3 m)이 아니라 노면 전용 격자(0.1 m)를 그리므로, 폭 15 cm 인
    // 주차선이 칸보다 굵어져 비로소 화면에 나온다. 도로는 면이니 높이 방향
    // 해상도를 쓸 이유가 없고, 그 예산을 전부 가로세로에 쓰는 것이다.
    void roadTexture(const std::unordered_map<std::int64_t, RoadCell>& cells,
                     const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                     const Eigen::Vector3d& up_d, const Orbit& orb, double f,
                     const Eigen::Vector3d& ego, double radius) {
        const Eigen::Vector3d up = up_d.normalized();
        const Eigen::Vector3d ad = a.cast<double>(), bd = b.cast<double>();
        // **칸을 칸보다 크게 그린다.**
        //
        // 정확히 칸 크기로 그리면 관측되지 않은 칸이 검은 구멍으로 남고,
        // 도로가 아스팔트가 아니라 자갈밭으로 보인다 - 화면에서 가장 넓은
        // 면적이 그 얼룩이었다.
        //
        // 구멍은 노면이 없어서가 아니라 그 0.1 m 칸을 못 봐서 생긴다. 도로는
        // 연속면이므로 이웃 칸의 밝기로 덮는 것이 관측에 더 가깝다. 1.7 배로
        // 겹쳐 그리면 한 칸짜리 구멍이 사라지고, 그보다 넓게 빈 곳은 그대로
        // 남는다 - 정말로 못 본 곳까지 메우지는 않는다.
        const double h = 0.85 * kRoadCell;
        const double r2 = radius * radius;
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;

        // 밝기 범위는 **보이는 범위에서** 잡는다. 지도 전체로 정규화하면
        // 그늘진 구간이 대비를 통째로 먹어 여기 선이 안 뜬다.
        int lo = 255, hi = 0;
        for (const auto& [k, c] : cells) {
            if (c.hits == 0) continue;
            const Eigen::Vector3d p = ad * 0.0 + up * c.h;   // 높이만 필요 없음
            (void)p;
            lo = std::min(lo, static_cast<int>(c.intensity));
            hi = std::max(hi, static_cast<int>(c.intensity));
        }
        const double span = std::max(20, hi - lo);

        for (const auto& [key, c] : cells) {
            if (c.hits == 0) continue;
            // 키에서 격자 좌표를 되찾는다. 부호 확장을 해야 음수 칸이 살아난다.
            std::int64_t qi = (key >> 26) & 0x3FFFFFF;
            std::int64_t qj = key & 0x3FFFFFF;
            if (qi & 0x2000000) qi -= 0x4000000;
            if (qj & 0x2000000) qj -= 0x4000000;
            const double ca = (static_cast<double>(qi) + 0.5) * kRoadCell;
            const double cb = (static_cast<double>(qj) + 0.5) * kRoadCell;
            const Eigen::Vector3d c3 = ad * ca + bd * cb + up * c.h;
            if ((c3 - ego).squaredNorm() > r2) continue;

            const Eigen::Vector3d q[4] = {c3 - ad * h - bd * h, c3 + ad * h - bd * h,
                                          c3 + ad * h + bd * h, c3 - ad * h + bd * h};
            cv::Point poly[4];
            bool ok = true;
            for (int i = 0; i < 4 && ok; ++i) ok = project3(q[i], orb, f, poly[i]);
            if (!ok) continue;
            const double w = std::max({std::abs(poly[0].x - poly[2].x),
                                       std::abs(poly[0].y - poly[2].y)});
            if (w > img_.cols * 0.4) continue;
            // **화면에서 1 px 도 안 되는 칸은 그리지 않는다.**
            //
            // 0.1 m 격자를 55 m 까지 전부 그리면 사각형이 100 만 개가 되고
            // 프레임당 361 ms 가 든다 (실측). 그 대부분은 화면에서 한 점보다
            // 작아 아무 것도 보태지 않는다 - 멀리 있는 노면은 어차피 아래
            // 격자면이 대신한다.
            if (w < 1.0) continue;

            // **밑바닥을 올린다.** 아스팔트는 실제로 어둡다. 밝기를 그대로
            // 옮기면 노면 전체가 검은 격자로 보이고, 도로가 있는 자리와 아무
            // 것도 없는 자리가 화면에서 구분되지 않는다 - 실제로 그렇게
            // 나왔다. 노면은 여기 도로가 있다는 사실부터 보여야 한다.
            //
            // 제곱은 그대로 둔다. 흰 페인트가 아스팔트보다 확실히 밝게 남는
            // 것이 이 층의 목적이기 때문이다.
            const double t = std::clamp((c.intensity - lo) / span, 0.0, 1.0);
            const double v = 58.0 + 190.0 * t * t;
            cv::fillConvexPoly(img_, poly, 4, cv::Scalar(v, v, v), cv::LINE_8);
            const double zz = (M * (c3 - eye)).z();
            if (zz > 1e-3) cv::fillConvexPoly(zbuf_, poly, 4, cv::Scalar(zz), cv::LINE_8);
        }
    }

    // 노면. **밝기로 칠한 납작한 판** 이다.
    //
    // 큐브로 그리면 도로가 검은 블록 밭이 되고, 안 그리면 차선과 주차선이
    // 통째로 사라진다. 노면 복셀은 부피가 아니라 **표면** 이므로 세워진 육면체가
    // 아니라 지면에 누운 사각형으로 그리는 것이 맞고, 색은 높이가 아니라
    // **관측된 밝기** 여야 한다 - 노면에서 높이는 어디나 같고, 정보는 전부
    // 밝기에 있다.
    void roadSurface(const std::vector<Splat>& pts, float voxel, const Orbit& orb,
                     double f, const Eigen::Vector3d& up_d) {
        const Eigen::Vector3d up = up_d.normalized();
        Eigen::Vector3d a = up.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = up.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = up.cross(a).normalized();
        const double h = 0.5 * voxel;

        // 밝기 범위를 실제 분포에 맞춘다. 0~255 를 그대로 쓰면 아스팔트가
        // 전부 어두운 회색 한 덩어리가 되어 흰 선이 안 두드러진다.
        int lo = 255, hi = 0;
        for (const auto& s : pts) {
            lo = std::min(lo, static_cast<int>(s.intensity));
            hi = std::max(hi, static_cast<int>(s.intensity));
        }
        const double span = std::max(24, hi - lo);

        for (const auto& s : pts) {
            const Eigen::Vector3d c = s.p.cast<double>();
            const Eigen::Vector3d q[4] = {c - a * h - b * h, c + a * h - b * h,
                                          c + a * h + b * h, c - a * h + b * h};
            cv::Point poly[4];
            bool ok = true;
            for (int i = 0; i < 4 && ok; ++i) ok = project3(q[i], orb, f, poly[i]);
            if (!ok) continue;
            const double w = std::max({std::abs(poly[0].x - poly[2].x),
                                       std::abs(poly[0].y - poly[2].y)});
            if (w > img_.cols * 0.5) continue;
            // 밝기를 정규화해 회색으로. 흰 페인트는 밝게 남고 아스팔트는
            // 배경으로 가라앉는다.
            const double t = std::clamp((s.intensity - lo) / span, 0.0, 1.0);
            const double v = 26.0 + 210.0 * t * t;    // 제곱으로 대비를 세운다
            cv::fillConvexPoly(img_, poly, 4, cv::Scalar(v, v, v), cv::LINE_8);
            double zz;
            const Eigen::Matrix3d M = orb.basis();
            const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
            zz = (M * (c - eye)).z();
            if (zz > 1e-3) {
                cv::fillConvexPoly(zbuf_, poly, 4, cv::Scalar(zz), cv::LINE_8);
            }
        }
    }

    // 삼각형 표면을 z 버퍼로 그린다.
    //
    // 화가 알고리즘으로는 안 된다 - 표면은 자기 자신과 교차하는 순서가 없어서,
    // 정렬만으로는 뒷면이 앞면을 덮는 자리가 반드시 생긴다. 픽셀마다 깊이를
    // 비교해야 표면이 표면으로 보인다.
    //
    // 음영은 헤드라이트 램버트다. 고정 광원을 쓰면 회전할 때 어떤 면은 계속
    // 어둡게 남아 형상이 안 읽힌다. 카메라를 광원으로 두면 늘 지금 보는
    // 방향에서 밝다.
    void surface(const SurfMesh& m, const Orbit& orb, double f,
                 const cv::Scalar& base) {
        if (m.tri.empty()) return;
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const double cx = img_.cols * 0.5, cy = img_.rows * 0.5;

        struct P { cv::Point2f s; double z; bool ok; };
        std::vector<P> pv(m.vert.size());
        for (std::size_t i = 0; i < m.vert.size(); ++i) {
            const Eigen::Vector3d v = M * (m.vert[i].cast<double>() - eye);
            pv[i].z = v.z();
            pv[i].ok = v.z() > 1e-3;
            if (pv[i].ok) {
                pv[i].s = {static_cast<float>(cx + f * v.x() / v.z()),
                           static_cast<float>(cy - f * v.y() / v.z())};
            }
        }
        const Eigen::Vector3d fwd = M.row(2).transpose();

        for (const auto& t : m.tri) {
            const P& a = pv[t[0]]; const P& b = pv[t[1]]; const P& c = pv[t[2]];
            if (!a.ok || !b.ok || !c.ok) continue;
            // 화면 경계 상자
            int x0 = static_cast<int>(std::floor(std::min({a.s.x, b.s.x, c.s.x})));
            int x1 = static_cast<int>(std::ceil (std::max({a.s.x, b.s.x, c.s.x})));
            int y0 = static_cast<int>(std::floor(std::min({a.s.y, b.s.y, c.s.y})));
            int y1 = static_cast<int>(std::ceil (std::max({a.s.y, b.s.y, c.s.y})));
            if (x1 < 0 || y1 < 0 || x0 >= img_.cols || y0 >= img_.rows) continue;
            // 투영이 튄 삼각형은 화면을 가로지른다. 그리지 않는다.
            if ((x1 - x0) > img_.cols / 2 || (y1 - y0) > img_.rows / 2) continue;
            x0 = std::max(0, x0); y0 = std::max(0, y0);
            x1 = std::min(img_.cols - 1, x1); y1 = std::min(img_.rows - 1, y1);

            const double d = (b.s.x - a.s.x) * (c.s.y - a.s.y)
                           - (c.s.x - a.s.x) * (b.s.y - a.s.y);
            if (std::abs(d) < 1e-6) continue;

            // 면 법선은 세 정점 법선의 평균. 헤드라이트와의 내적이 밝기다.
            Eigen::Vector3f n = m.norm[t[0]] + m.norm[t[1]] + m.norm[t[2]];
            double lam = 0.35;
            if (n.norm() > 1e-9f) {
                n.normalize();
                lam = 0.35 + 0.65 * std::abs(n.cast<double>().dot(fwd));
            }
            const cv::Vec3b col(static_cast<std::uint8_t>(base[0] * lam),
                                static_cast<std::uint8_t>(base[1] * lam),
                                static_cast<std::uint8_t>(base[2] * lam));

            for (int y = y0; y <= y1; ++y) {
                float* zr = zbuf_.ptr<float>(y);
                auto* pr = img_.ptr<cv::Vec3b>(y);
                for (int x = x0; x <= x1; ++x) {
                    const double px = x + 0.5, py = y + 0.5;
                    const double w0 = ((b.s.x - a.s.x) * (py - a.s.y)
                                     - (px - a.s.x) * (b.s.y - a.s.y)) / d;
                    const double w1 = ((px - a.s.x) * (c.s.y - a.s.y)
                                     - (c.s.x - a.s.x) * (py - a.s.y)) / d;
                    if (w0 < 0.0 || w1 < 0.0 || w0 + w1 > 1.0) continue;
                    const double z = a.z + w1 * (b.z - a.z) + w0 * (c.z - a.z);
                    if (z <= 1e-3 || z >= zr[x]) continue;
                    zr[x] = static_cast<float>(z);
                    pr[x] = col;
                }
            }
        }
    }

    // Eye-Dome Lighting.
    //
    // Boucheny (2009) 가 만들고 CloudCompare / Potree / ArcGIS 가 쓰는 기법이다.
    // 원리는 순수 화면공간이다: 이웃 픽셀보다 **뒤에 있는** 픽셀을 어둡게 칠한다.
    // 법선도 조명도 지오메트리도 필요 없고 깊이 버퍼만 있으면 된다.
    //
    //   res   = (1/N) * sum_i max(0, D(p) - D(p + r*n_i))
    //   shade = exp(-res * 300 * strength)
    //
    // 성긴 점군이 "3D 모델처럼" 보이는 이유의 거의 전부가 이것이다 - 사람의
    // 시각은 깊이 불연속의 음영 대비로 형상을 읽는데, EDL 이 정확히 그것만
    // 공급한다. 데이터도 알고리즘도 건드리지 않는 후처리 한 패스다.
    //
    // **D 는 선형 깊이가 아니라 log2 깊이다.** 이것을 놓치면 안 된다. 로그를
    // 쓰면 차이가 깊이의 **비율** 이 되어 거리 스케일에 불변이 되는데, 선형 z
    // 를 그대로 쓰면 1 m 부터 55 m 까지 오가는 이 장면에서 근거리는 새까맣고
    // 원거리는 밋밋해진다.
    void eyeDomeLighting(double strength = 0.4, double radius = 1.4) {
        const int W = img_.cols, H = img_.rows;
        if (W < 4 || H < 4) return;

        // log2 깊이를 한 번에 만든다. 배경(무한대)은 0 으로 표시한다.
        cv::Mat d(H, W, CV_32F);
        for (int y = 0; y < H; ++y) {
            const float* z = zbuf_.ptr<float>(y);
            float* o = d.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                o[x] = (z[x] >= std::numeric_limits<float>::max() * 0.5f || z[x] <= 0.0f)
                     ? 0.0f : std::log2(z[x]);
            }
        }

        static const int nx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int ny[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        const int r = std::max(1, static_cast<int>(std::lround(radius)));
        const double k = 300.0 * strength;

        cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& rows) {
            for (int y = rows.start; y < rows.end; ++y) {
                auto* px = img_.ptr<cv::Vec3b>(y);
                const float* dc = d.ptr<float>(y);
                for (int x = 0; x < W; ++x) {
                    const float dp = dc[x];
                    if (dp == 0.0f) continue;           // 배경은 칠하지 않는다
                    double sum = 0.0;
                    for (int i = 0; i < 8; ++i) {
                        const int sx = x + nx[i] * r, sy = y + ny[i] * r;
                        if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
                        const float dn = d.ptr<float>(sy)[sx];
                        // 이웃이 배경이면 실루엣이다. 큰 값을 더해 외곽에
                        // 어두운 윤곽선을 만든다 - EDL 특유의 "종이에 그린 듯한"
                        // 인상이 여기서 나온다.
                        sum += (dn == 0.0f) ? 100.0 : std::max(0.0f, dp - dn);
                    }
                    const double res = sum * 0.125;
                    if (res <= 0.0) continue;           // 평탄면은 exp 를 건너뛴다
                    const double shade = std::exp(-res * k);
                    if (shade >= 0.999) continue;
                    auto& c = px[x];
                    c[0] = static_cast<std::uint8_t>(c[0] * shade);
                    c[1] = static_cast<std::uint8_t>(c[1] * shade);
                    c[2] = static_cast<std::uint8_t>(c[2] * shade);
                }
            }
        });
    }

    // 복셀을 **속이 찬 육면체** 로 그린다. OctoMap / RViz 가 보여 주는 그 그림이다.
    //
    // 점으로 찍으면 같은 지도가 성기게 보인다. 복셀은 부피를 가진 칸인데 그
    // 중심 하나만 찍으면 칸과 칸 사이가 빈 것처럼 읽히고, 벽이 벽으로 보이지
    // 않는다. 칸을 칸 크기대로 채우면 그 순간 면이 된다 - 데이터는 그대로이고
    // 그리는 방법만 바뀐 것인데, 있는 것을 있는 크기로 그린 쪽이 맞다.
    //
    // 면마다 밝기를 달리한다. 단색으로 채우면 육면체가 육각형 얼룩이 되어
    // 입체가 사라진다. 윗면을 밝게, 옆면을 어둡게 하면 같은 색으로도 깊이가
    // 읽힌다 - 조명 계산이 아니라 면의 방향을 색으로 옮기는 것이다.
    void voxelCubes(const std::vector<Splat>& pts, float voxel, const Orbit& orb,
                    double f, const Eigen::Vector3d& up, double fade) {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const double cx = img_.cols * 0.5, cy = img_.rows * 0.5;
        const double h = 0.5 * voxel;

        // 화면 좌표 + 카메라 깊이
        auto proj = [&](const Eigen::Vector3d& p, cv::Point& out, double& z) {
            const Eigen::Vector3d v = M * (p - eye);
            z = v.z();
            if (z < 1e-3) return false;
            out = {static_cast<int>(cx + f * v.x() / z),
                   static_cast<int>(cy - f * v.y() / z)};
            return true;
        };

        // 뒤에서 앞으로 그린다. 화가 알고리즘이면 z 버퍼 없이도 가려짐이
        // 맞아떨어지고, 폴리곤마다 픽셀 깊이를 검사하는 것보다 훨씬 싸다.
        // 정렬은 버킷으로 한다 - 20 만 개를 완전정렬할 이유가 없다.
        struct Item { const Splat* s; double z; };
        std::vector<Item> vis;
        vis.reserve(pts.size());
        for (const auto& s : pts) {
            const Eigen::Vector3d v = M * (s.p.cast<double>() - eye);
            if (v.z() < 1e-3) continue;
            // 화면에서 2 px 도 안 되는 칸은 큐브로 그릴 이유가 없다.
            const double px = f * voxel / v.z();
            // 화면에서 작아지는 칸은 큐브로 그릴 값이 없다. 점은 이미
            // 깔려 있으므로 사라지지 않는다.
            if (px < 2.0) continue;
            const double u = cx + f * v.x() / v.z(), w = cy - f * v.y() / v.z();
            if (u < -px || w < -px || u > img_.cols + px || w > img_.rows + px) continue;
            vis.push_back({&s, v.z()});
        }
        std::sort(vis.begin(), vis.end(),
                  [](const Item& a, const Item& b) { return a.z > b.z; });

        // 복셀 격자의 세 축. 월드 축을 그대로 쓰면 큐브가 지면과 어긋나 보인다.
        Eigen::Vector3d ax = up.normalized();
        Eigen::Vector3d a1 = ax.cross(Eigen::Vector3d::UnitZ());
        if (a1.norm() < 1e-6) a1 = ax.cross(Eigen::Vector3d::UnitX());
        a1.normalize();
        const Eigen::Vector3d a2 = ax.cross(a1).normalized();

        // 카메라를 향한 세 면만 그린다. 나머지 셋은 어차피 가려진다.
        for (const auto& it : vis) {
            const Eigen::Vector3d c = it.s->p.cast<double>();
            const Eigen::Vector3d to_eye = (eye - c);
            cv::Scalar base = turbo(it.s->depth_norm);
            const double age_a = 1.0 - fade * static_cast<double>(it.s->age);

            // 축마다 카메라 쪽 면을 고른다.
            const Eigen::Vector3d axes[3] = {ax, a1, a2};
            const double shade[3] = {1.00, 0.72, 0.55};   // 윗면 / 옆면 / 옆면
            for (int k = 0; k < 3; ++k) {
                const double sgn = (to_eye.dot(axes[k]) >= 0.0) ? 1.0 : -1.0;
                const Eigen::Vector3d n = axes[k] * sgn;
                const Eigen::Vector3d u1 = axes[(k + 1) % 3] * h;
                const Eigen::Vector3d u2 = axes[(k + 2) % 3] * h;
                const Eigen::Vector3d fc = c + n * h;
                const Eigen::Vector3d q[4] = {fc - u1 - u2, fc + u1 - u2,
                                              fc + u1 + u2, fc - u1 + u2};
                cv::Point poly[4];
                bool ok = true;
                double zz;
                for (int i = 0; i < 4 && ok; ++i) ok = proj(q[i], poly[i], zz);
                if (!ok) continue;
                const double a = age_a * shade[k];
                const cv::Scalar col{base[0] * a + C_VOID[0] * (1 - a),
                                     base[1] * a + C_VOID[1] * (1 - a),
                                     base[2] * a + C_VOID[2] * (1 - a)};
                cv::fillConvexPoly(img_, poly, 4, col, cv::LINE_8);
                // **깊이도 같이 쓴다.** 화가 알고리즘이라 색은 뒤에서 앞으로
                // 덮이는데, EDL 은 깊이 버퍼를 읽으므로 색만 채우면 그 자리의
                // 깊이가 비어 있다. 뒤에서 앞으로 쓰므로 마지막에 남는 값이
                // 가장 가까운 면이 된다 - 정렬이 이미 그 일을 해 준다.
                cv::fillConvexPoly(zbuf_, poly, 4, cv::Scalar(it.z), cv::LINE_8);
            }
        }
    }

    // --- 클래스별 3 차원 형상 ---
    //
    // 지금까지 물체는 전부 같은 와이어프레임 상자였고, 자동차인지 사람인지는
    // 색으로만 갈렸다. 그러면 "무엇을 인식했는가" 가 화면에서 확인되지 않는다 -
    // 상자는 검출기가 무엇을 내놓았든 똑같이 생겼기 때문이다.
    //
    // 형상이 클래스를 말하게 한다. 자동차는 차체 위에 캐빈이 얹힌 실루엣이고,
    // 사람은 좁고 높은 몸통에 머리가 붙는다. 멀리서 실루엣만 봐도 갈린다.

    // --- 깊이를 지키는 채우기 ---
    //
    // **이 클래스에서 3D 로 보이지 않던 것의 원인은 전부 여기 하나였다.**
    //
    // 지금까지 클래스 형상(건물·담장·나무)과 물체 모형은 cv::fillConvexPoly 로
    // 화면에 직접 칠했다. 그리는 순서는 unordered_map 의 순회 순서, 즉 해시
    // 순서다. 그러면 100 m 뒤의 건물이 눈앞의 건물 위에 칠해지는 일이 매
    // 프레임 무작위로 일어난다 - 화면이 "떠 있는 판자 더미" 로 보이던 것이
    // 그것이다. 형상이 나빠서가 아니라 가려짐이 없어서였다.
    //
    // 점군(draw)·복셀(voxelCubes)·삼각형 표면(surface)·노면(roadTexture)은
    // 이미 z 버퍼를 쓰고 있었다. 클래스 형상만 쓰지 않아서, 지도에서 가장 큰
    // 면적을 차지하는 바로 그것이 깊이 없이 떠 있었다.
    //
    // 화소마다 깊이를 비교하고 통과한 자리에만 색과 깊이를 함께 쓴다. EDL 이
    // 읽는 것도 이 깊이라, 채우는 순간 외곽선과 음영이 같이 따라온다.
    struct SPt { double x, y, z; bool ok; };

    SPt proj(const Eigen::Vector3d& p, const Eigen::Matrix3d& M,
             const Eigen::Vector3d& eye, double f) const {
        const Eigen::Vector3d v = M * (p - eye);
        SPt s{0, 0, v.z(), v.z() > 1e-3};
        if (s.ok) {
            s.x = img_.cols * 0.5 + f * v.x() / v.z();
            s.y = img_.rows * 0.5 - f * v.y() / v.z();
        }
        return s;
    }

    // 삼각형 하나. surface() 가 쓰던 래스터라이저와 같은 것이다.
    void rasterTri(const SPt& a, const SPt& b, const SPt& c, const cv::Vec3b& col) {
        if (!a.ok || !b.ok || !c.ok) return;
        int x0 = static_cast<int>(std::floor(std::min({a.x, b.x, c.x})));
        int x1 = static_cast<int>(std::ceil (std::max({a.x, b.x, c.x})));
        int y0 = static_cast<int>(std::floor(std::min({a.y, b.y, c.y})));
        int y1 = static_cast<int>(std::ceil (std::max({a.y, b.y, c.y})));
        if (x1 < 0 || y1 < 0 || x0 >= img_.cols || y0 >= img_.rows) return;
        // 투영이 튄 삼각형은 화면을 가로지른다. 그리지 않는다.
        if ((x1 - x0) > img_.cols || (y1 - y0) > img_.rows) return;
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(img_.cols - 1, x1); y1 = std::min(img_.rows - 1, y1);
        const double d = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if (std::abs(d) < 1e-9) return;
        for (int y = y0; y <= y1; ++y) {
            float* zr = zbuf_.ptr<float>(y);
            auto* pr = img_.ptr<cv::Vec3b>(y);
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                const double w0 = ((b.x - a.x) * (py - a.y)
                                 - (px - a.x) * (b.y - a.y)) / d;
                const double w1 = ((px - a.x) * (c.y - a.y)
                                 - (c.x - a.x) * (py - a.y)) / d;
                if (w0 < 0.0 || w1 < 0.0 || w0 + w1 > 1.0) continue;
                const double z = a.z + w1 * (b.z - a.z) + w0 * (c.z - a.z);
                if (z <= 1e-3 || z >= zr[x]) continue;
                zr[x] = static_cast<float>(z);
                pr[x] = col;
            }
        }
    }

    // 볼록 다각형을 부채꼴 삼각형으로 쪼개 깊이와 함께 채운다.
    void fillPoly3(const Eigen::Vector3d* q, int n, const Orbit& orb, double f,
                   const cv::Scalar& col) {
        if (n < 3) return;
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        std::vector<SPt> s(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) s[static_cast<std::size_t>(i)] = proj(q[i], M, eye, f);
        const cv::Vec3b c8(static_cast<std::uint8_t>(std::clamp(col[0], 0.0, 255.0)),
                           static_cast<std::uint8_t>(std::clamp(col[1], 0.0, 255.0)),
                           static_cast<std::uint8_t>(std::clamp(col[2], 0.0, 255.0)));
        for (int i = 1; i + 1 < n; ++i) {
            rasterTri(s[0], s[static_cast<std::size_t>(i)],
                      s[static_cast<std::size_t>(i + 1)], c8);
        }
    }

    // 방향이 있는 속이 찬 상자. 세 축과 반크기를 받아 카메라 쪽 세 면을 채운다.
    //
    // 카메라를 향한 세 면만 그린다 - 나머지 셋은 어차피 자기 앞면에 가려지고,
    // 깊이 검사가 그것을 보장한다.
    void solidBox(const Eigen::Vector3d& c, const Eigen::Vector3d& ex,
                  const Eigen::Vector3d& ey, const Eigen::Vector3d& ez,
                  const Orbit& orb, double f, const cv::Scalar& base,
                  double bright = 1.0) {
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const Eigen::Vector3d axes[3] = {ex, ey, ez};
        const double shade[3] = {0.95, 0.70, 0.52};
        for (int k = 0; k < 3; ++k) {
            const double sgn = ((eye - c).dot(axes[k]) >= 0.0) ? 1.0 : -1.0;
            const Eigen::Vector3d n  = axes[k] * sgn;
            const Eigen::Vector3d u1 = axes[(k + 1) % 3];
            const Eigen::Vector3d u2 = axes[(k + 2) % 3];
            const Eigen::Vector3d fc = c + n;
            const Eigen::Vector3d q[4] = {fc - u1 - u2, fc + u1 - u2,
                                          fc + u1 + u2, fc - u1 + u2};
            const double a = std::clamp(shade[k] * bright, 0.0, 1.0);
            fillPoly3(q, 4, orb, f,
                      cv::Scalar(base[0] * a, base[1] * a, base[2] * a));
        }
    }

    // 자동차. 차체 + 그 위에 얹힌 캐빈. 진행 방향으로 눕힌다.
    void carModel(const Eigen::Vector3d& c, const Eigen::Vector3d& size,
                  const Eigen::Vector3d& fwd, const Eigen::Vector3d& up,
                  const Orbit& orb, double f, const cv::Scalar& col) {
        Eigen::Vector3d u = up.normalized();
        Eigen::Vector3d fw = fwd - u * fwd.dot(u);
        if (fw.norm() < 1e-6) fw = Eigen::Vector3d::UnitX();
        fw.normalize();
        const Eigen::Vector3d rt = u.cross(fw).normalized();

        const double L = std::max(0.6, size.maxCoeff());          // 길이
        const double W = std::max(0.4, std::min(size.x(), size.z()));
        const double H = std::max(0.4, size.y());
        // 차체: 낮고 길다. 지면에서 살짝 띄운다.
        solidBox(c - u * (H * 0.15), fw * (L * 0.5), u * (H * 0.30),
                 rt * (W * 0.5), orb, f, col, 1.0);
        // 캐빈: 짧고 위에. 이 두 덩이의 비율이 "자동차" 를 만든다.
        solidBox(c + u * (H * 0.34), fw * (L * 0.28), u * (H * 0.22),
                 rt * (W * 0.42), orb, f, col, 0.78);
    }

    // 단위 구를 삼각형으로. 팔면체를 두 번 쪼갠다 - 128 면.
    //
    // 한 번만 만들어 두고 나무마다 이동·확대해서 쓴다. 나무 한 그루가 128
    // 삼각형이면 화면에 100 그루가 있어도 12800 개이고, 이미 30 만 점을
    // 래스터하고 있는 파이프라인에서 그것은 반올림 오차다.
    static const std::vector<std::array<Eigen::Vector3f, 3>>& unitSphere() {
        static const std::vector<std::array<Eigen::Vector3f, 3>> tris = [] {
            std::vector<std::array<Eigen::Vector3f, 3>> t;
            const Eigen::Vector3f v[6] = {{1,0,0},{-1,0,0},{0,1,0},
                                          {0,-1,0},{0,0,1},{0,0,-1}};
            const int face[8][3] = {{0,2,4},{2,1,4},{1,3,4},{3,0,4},
                                    {2,0,5},{1,2,5},{3,1,5},{0,3,5}};
            for (const auto& fc : face) {
                t.push_back({v[fc[0]], v[fc[1]], v[fc[2]]});
            }
            // 두 번 쪼갠다. 쪼갤 때마다 새 꼭짓점을 구면으로 밀어낸다.
            for (int it = 0; it < 2; ++it) {
                std::vector<std::array<Eigen::Vector3f, 3>> next;
                next.reserve(t.size() * 4);
                for (const auto& tr : t) {
                    const Eigen::Vector3f ab = (tr[0] + tr[1]).normalized();
                    const Eigen::Vector3f bc = (tr[1] + tr[2]).normalized();
                    const Eigen::Vector3f ca = (tr[2] + tr[0]).normalized();
                    next.push_back({tr[0], ab, ca});
                    next.push_back({ab, tr[1], bc});
                    next.push_back({ca, bc, tr[2]});
                    next.push_back({ab, bc, ca});
                }
                t.swap(next);
            }
            return t;
        }();
        return tris;
    }

    // 나무. **줄기 + 수관** 을 진짜 입체로 세운다.
    //
    // 앞선 판은 수관을 동심원 몇 겹으로 칠했다. 정면에서는 구처럼 보였지만
    // 그것은 화면에 그린 원반이라 깊이가 없었다 - 뒤에 있는 나무가 앞 건물을
    // 덮었고, 카메라를 돌려도 수관의 실루엣이 변하지 않았다. 부피가 있는
    // 것처럼 **칠한** 것과 부피가 **있는** 것은 다르다.
    //
    // 구를 삼각형으로 쪼개면 그 차이가 사라진다. 법선은 구면 위치 그대로이고
    // (구의 법선은 반지름 방향이다), 음영은 카메라를 광원으로 둔 램버트다 -
    // 고정 광원이면 회전할 때 어떤 면이 계속 어두워 형상이 안 읽힌다.
    void treeModel(const Eigen::Vector3f& foot, float height, float rad,
                   const Eigen::Vector3f& up, const Orbit& orb, double f,
                   const cv::Scalar& col) {
        Eigen::Vector3f a = up.cross(Eigen::Vector3f::UnitZ());
        if (a.norm() < 1e-6f) a = up.cross(Eigen::Vector3f::UnitX());
        a.normalize();
        const Eigen::Vector3f b = up.cross(a).normalized();

        // **수관의 크기는 나무 높이에서 온다.**
        //
        // 앞선 판은 반지름을 격자칸에서만 가져왔다. 그러면 8 m 짜리 나무가
        // 반지름 1 m 짜리 공을 꼭대기에 얹은 6 m 바늘이 되고, 화면에는 검은
        // 가시 두 개가 꽂힌 것처럼 보였다 - 실제로 그렇게 나왔다.
        //
        // 나무는 수관이 몸통이고 줄기는 그 아래 짧은 받침이다. 수관을 높이의
        // 위쪽 4 분의 3 에 걸치게 하면 그 비례가 저절로 맞는다.
        const float canopy_ry = std::max(0.30f, height * 0.40f);   // 수직 반지름
        const float canopy_c  = height * 0.58f;                    // 중심 높이
        const float canopy_r  = std::max(rad, canopy_ry * 0.55f);  // 수평 반지름
        const Eigen::Vector3f cen = foot + up * canopy_c;
        const float trunk_h = std::max(0.2f, canopy_c - canopy_ry * 0.55f);

        // 줄기: 가는 상자. 원기둥으로 그릴 값이 없다.
        {
            const float tw = std::max(0.05f, canopy_r * 0.14f);
            solidBox((foot + up * (trunk_h * 0.5f)).cast<double>(),
                     (a * tw).cast<double>(),
                     (up * (trunk_h * 0.5f)).cast<double>(),
                     (b * tw).cast<double>(),
                     orb, f, cv::Scalar(col[0] * 0.42, col[1] * 0.40, col[2] * 0.36), 1.0);
        }

        // 수관. 위아래로 살짝 눌러 공이 아니라 나무로 읽히게 한다.
        const Eigen::Matrix3d M = orb.basis();
        const Eigen::Vector3d eye = orb.center - M.row(2).transpose() * orb.dist;
        const Eigen::Vector3d fwd = M.row(2).transpose();
        // 화면에서 2 px 도 안 되는 수관은 삼각형 128 개를 돌릴 값이 없다.
        {
            const double zc = (M * (cen.cast<double>() - eye)).z();
            if (zc < 1e-3) return;
            if (f * canopy_r / zc < 2.0) return;
        }
        const Eigen::Vector3f ex = a * canopy_r;
        const Eigen::Vector3f ey = up * canopy_ry;
        const Eigen::Vector3f ez = b * canopy_r;
        // 위쪽 잎은 하늘을, 아래쪽 잎은 그늘을 본다. 균일하게 칠하면 수관이
        // 초록 공 하나로 뭉친다.
        for (const auto& tr : unitSphere()) {
            Eigen::Vector3f n = (tr[0] + tr[1] + tr[2]) / 3.0f;
            if (n.norm() < 1e-6f) continue;
            n.normalize();
            const Eigen::Vector3f wn =
                (ex * n.x() + ey * n.y() + ez * n.z());
            // 뒷면은 자기 앞면에 가려진다. 그리지 않는다.
            const Eigen::Vector3d wc = (cen + wn).cast<double>();
            if ((eye - wc).dot(wn.cast<double>()) < 0.0) continue;
            const double lam = 0.34 + 0.52 * std::abs(wn.normalized().cast<double>().dot(fwd))
                             + 0.14 * std::clamp(static_cast<double>(n.dot(up)), 0.0, 1.0);
            const Eigen::Vector3d q[3] = {
                (cen + ex * tr[0].x() + ey * tr[0].y() + ez * tr[0].z()).cast<double>(),
                (cen + ex * tr[1].x() + ey * tr[1].y() + ez * tr[1].z()).cast<double>(),
                (cen + ex * tr[2].x() + ey * tr[2].y() + ez * tr[2].z()).cast<double>()};
            fillPoly3(q, 3, orb, f,
                      cv::Scalar(col[0] * lam, col[1] * lam, col[2] * lam));
        }
    }

    // 사람 골격.
    //
    // **관절을 추정한 것이 아니다.** 이 저장소에는 자세 추정기가 없고, 있는
    // 것은 YOLO 의 person 상자 하나뿐이다. 여기서 그리는 것은 그 상자에 사람
    // 비례를 맞춰 넣은 **도식** 이고, 팔다리의 각도는 관측이 아니라 그림이다.
    //
    // 그래도 상자보다 낫다: 멀리서도 사람으로 읽히고, 자동차와 절대 헷갈리지
    // 않는다. 다만 화면이 "관절을 봤다" 고 말하게 두면 안 되므로 라벨에
    // schematic 을 붙인다 - 그리지 않은 것을 그린 척하지 않는 쪽이 규칙이다.
    void personSkeleton(const Eigen::Vector3d& c, const Eigen::Vector3d& size,
                        const Eigen::Vector3d& up, const Eigen::Vector3d& face,
                        const Orbit& orb, double f, const cv::Scalar& col) {
        const Eigen::Vector3d u = up.normalized();
        Eigen::Vector3d fw = face - u * face.dot(u);
        if (fw.norm() < 1e-6) fw = Eigen::Vector3d::UnitX();
        fw.normalize();
        const Eigen::Vector3d rt = u.cross(fw).normalized();

        const double H = std::clamp(size.maxCoeff(), 0.9, 2.2);
        const double W = std::clamp(std::min(size.x(), size.z()), 0.25, 0.7);
        // 인체 비례. 발밑을 원점으로 잡는다.
        const Eigen::Vector3d foot = c - u * (H * 0.5);
        const auto P = [&](double h, double side, double front = 0.0) {
            return foot + u * (H * h) + rt * (W * side) + fw * (W * front);
        };
        const Eigen::Vector3d head   = P(0.94, 0.0);
        const Eigen::Vector3d neck   = P(0.83, 0.0);
        const Eigen::Vector3d sh_l   = P(0.82, -0.62), sh_r = P(0.82, 0.62);
        const Eigen::Vector3d el_l   = P(0.63, -0.72), el_r = P(0.63, 0.72);
        const Eigen::Vector3d hd_l   = P(0.46, -0.66), hd_r = P(0.46, 0.66);
        const Eigen::Vector3d pelvis = P(0.52, 0.0);
        const Eigen::Vector3d hip_l  = P(0.50, -0.30), hip_r = P(0.50, 0.30);
        const Eigen::Vector3d kn_l   = P(0.27, -0.32), kn_r = P(0.27, 0.32);
        const Eigen::Vector3d ft_l   = P(0.02, -0.32), ft_r = P(0.02, 0.32);

        const std::pair<const Eigen::Vector3d*, const Eigen::Vector3d*> bones[] = {
            {&neck, &pelvis},                                  // 척추
            {&sh_l, &sh_r},                                    // 어깨
            {&sh_l, &el_l}, {&el_l, &hd_l},                    // 왼팔
            {&sh_r, &el_r}, {&el_r, &hd_r},                    // 오른팔
            {&hip_l, &hip_r},                                  // 골반
            {&hip_l, &kn_l}, {&kn_l, &ft_l},                   // 왼다리
            {&hip_r, &kn_r}, {&kn_r, &ft_r},                   // 오른다리
        };
        for (const auto& b : bones) {
            cv::Point a1, b1;
            if (project3(*b.first, orb, f, a1) && project3(*b.second, orb, f, b1)) {
                const int m = 4 * std::max(img_.cols, img_.rows);
                if (std::abs(a1.x) > m || std::abs(b1.x) > m) continue;
                cv::line(img_, a1, b1, col, 2, cv::LINE_AA);
            }
        }
        // 머리
        cv::Point hp, np;
        if (project3(head, orb, f, hp) && project3(neck, orb, f, np)) {
            const int r = std::clamp(static_cast<int>(
                std::hypot(hp.x - np.x, hp.y - np.y) * 0.85), 2, 40);
            cv::circle(img_, hp, r, col, 2, cv::LINE_AA);
        }
    }

    // 사람. 좁고 높은 몸통 + 머리. 자동차와 실루엣이 겹치지 않는다.
    void personModel(const Eigen::Vector3d& c, const Eigen::Vector3d& size,
                     const Eigen::Vector3d& up, const Orbit& orb, double f,
                     const cv::Scalar& col) {
        const Eigen::Vector3d u = up.normalized();
        Eigen::Vector3d a = u.cross(Eigen::Vector3d::UnitZ());
        if (a.norm() < 1e-6) a = u.cross(Eigen::Vector3d::UnitX());
        a.normalize();
        const Eigen::Vector3d b = u.cross(a).normalized();
        const double H = std::max(0.8, size.maxCoeff());
        const double W = std::max(0.20, std::min(0.55, size.minCoeff()));
        // 몸통
        solidBox(c - u * (H * 0.10), a * (W * 0.5), u * (H * 0.36), b * (W * 0.35),
                 orb, f, col, 1.0);
        // 머리
        cv::Point hp;
        if (project3(c + u * (H * 0.40), orb, f, hp)) {
            const Eigen::Vector3d probe = c + u * (H * 0.40) + a * (W * 0.5);
            cv::Point pp;
            const int r = project3(probe, orb, f, pp)
                        ? std::clamp(static_cast<int>(std::hypot(pp.x - hp.x,
                                                                 pp.y - hp.y)), 2, 40)
                        : 3;
            cv::circle(img_, hp, r, col, cv::FILLED, cv::LINE_AA);
        }
    }

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
            col = {col[0] * a + C_VOID[0] * (1 - a),
                   col[1] * a + C_VOID[1] * (1 - a),
                   col[2] * a + C_VOID[2] * (1 - a)};

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

    // 장면 라벨을 **입체 모형** 으로 그린다.
    //
    // 앞선 판은 이웃한 같은 클래스의 대표점 네 개를 사각형으로 채웠다. 그
    // 사각형은 기둥 **꼭대기** 들을 이은 것이라 지붕 껍질 한 겹이었고,
    // 지면까지 내려오는 몸통이 없었다 - 그래서 건물이 공중에 뜬 판자로
    // 보였다. 게다가 대표점의 높이는 칸마다 조금씩 다르므로 그 껍질은
    // 울퉁불퉁했고, 이웃이 셋 다 있어야 채워지므로 가장자리에는 구멍이 났다.
    //
    // 기둥은 원래 **지면에서 꼭대기까지의 부피** 다. 그것을 그대로 세우면
    // 껍질이 아니라 몸통이 생기고, 격자 간격과 밑면 크기가 같으므로 이웃한
    // 기둥끼리 저절로 맞붙어 연속된 벽면이 된다. 이을 것을 찾을 필요가
    // 없어지고, 구멍도 생기지 않는다.
    void stuffVectors(const std::unordered_map<std::int64_t, Column>& cols,
                      const Eigen::Vector3d& up_d, float cell,
                      const Orbit& orb, double f, int layer = 0,
                      const Eigen::Vector3d& ego = Eigen::Vector3d::Zero(),
                      double radius = 0.0) {
        const float r2f = static_cast<float>(radius * radius);
        const Eigen::Vector3f egof = ego.cast<float>();
        const GroundGrid g(up_d.cast<float>(), cell);
        const Eigen::Vector3f& up = g.up;

        // **지붕선은 이웃과 함께 정한다.**
        //
        // 기둥마다 제 꼭대기를 그대로 세우면 같은 벽면이 칸마다 다른 높이로
        // 서고, 지도가 건물이 아니라 막대그래프가 된다 - 실제로 그렇게 나왔다.
        // 한 칸의 top 은 "그 칸에서 마지막으로 무엇을 봤는가" 일 뿐이라
        // 관측이 성긴 칸은 낮게, 잡음이 튄 칸은 높게 나온다.
        //
        // 5x5 이웃 중 **같은 클래스** 인 칸들에서 고른다. 클래스가 다른 이웃을
        // 섞으면 건물이 옆 나무 높이로 끌려간다.
        //
        // **중앙값이 아니라 상위 분위수를 쓴다.** 관측된 꼭대기는 진짜 높이의
        // **하한** 이기 때문이다: 스쳐 지나가며 본 칸, 앞차에 가린 칸, 시야
        // 밖으로 나간 칸은 전부 실제보다 낮게 나오고, 실제보다 높게 나오는
        // 경로는 topRun 이 이미 막았다. 한쪽으로만 치우친 오차에 중앙값을
        // 쓰면 벽 전체가 가장 덜 본 칸을 따라 내려앉는다.
        //
        // p70 은 "이 벽면을 가장 잘 본 이웃 몇 칸이 본 높이" 이고, 그것은
        // 여전히 관측이다 - 없는 높이를 지어내는 것이 아니라 같은 벽의 더
        // 나은 관측을 옆 칸에 나눠 주는 것이다.
        //
        // 같은 클래스 이웃의 수도 함께 돌려준다. 홀로 선 칸을 가려내는 데
        // 쓴다 - 이웃 없이 혼자 서 있는 1 m 짜리 건물은 건물이 아니다.
        auto smoothTop = [&](const Column& c, float own, int& nsame) {
            float v[25];
            int n = 0;
            for (int di = -2; di <= 2; ++di) {
                for (int dj = -2; dj <= 2; ++dj) {
                    const auto it = cols.find(GroundGrid::key(c.i + di, c.j + dj));
                    if (it == cols.end() || it->second.cls != c.cls) continue;
                    const int tb = std::min(it->second.topBin(), it->second.topRun());
                    if (tb < 0) continue;
                    v[n++] = static_cast<float>(tb + 1) * kBinH;
                }
            }
            nsame = std::max(0, n - 1);            // 자기 자신을 뺀 수
            if (n < 4) return own;
            const int q = n * 7 / 10;
            std::nth_element(v, v + q, v + n);
            return v[q];
        };

        for (const auto& [k, c] : cols) {
            if (c.cls == Stuff::Unknown) continue;
            if (r2f > 0.0f && (c.rep - egof).squaredNorm() > r2f) continue;
            // 레이어 격리: 4 는 구조물만, 5 는 지면만.
            if (layer == 4 && c.cls == Stuff::Ground) continue;
            if (layer == 5 && c.cls != Stuff::Ground) continue;
            const cv::Scalar col = stuffColor(c.cls);

            // **지면도 면이다.** 점 하나만 찍으면 노면이 화면에서 사라져,
            // 무엇이 길이고 무엇이 허공인지 구분되지 않는다. 격자칸을 칸
            // 크기대로 채우면 도로가 도로로 보이고, 그 위에 선 구조의
            // 발치가 어디인지도 읽힌다.
            // **지면 칸은 그리지 않는다.** 바닥은 아래에 깔린 기준 격자가
            // 맡는다 - 관측 칸을 하나씩 그리면 성긴 곳은 구멍이, 잡음이 있는
            // 곳은 어긋난 타일이 생겨 평평한 바닥이 누더기가 된다.
            if (c.cls == Stuff::Ground) continue;

            // **형상이 클래스를 말한다.**
            //
            // 지금까지 건물도 나무도 담장도 같은 가는 선이었다. 위치는 맞는데
            // 무엇인지가 화면에서 읽히지 않으니, 좌우의 건물이 하늘에 뜬 잡음
            // 처럼 보였다 - 실제로 그렇게 보고되었다. 건물은 덩어리로, 나무는
            // 줄기 위 수관으로, 담장은 낮은 판으로 그린다.
            const int tb = std::min(c.topBin(), c.topRun());
            if (tb < 0) continue;
            // **발치는 자기 관측에서, 지붕은 이웃과 함께.** 지면 높이는 이미
            // 반경 안의 최저점에서 왔으므로 칸마다 흔들리지 않는다. 흔들리는
            // 것은 꼭대기 쪽이고, 그것만 편다.
            const float h_raw = static_cast<float>(tb + 1) * kBinH;
            // 발치는 그 칸에 쓰인 지면 높이 그대로다. rep 에서 높이를 빼면
            // 안 된다 - rep 는 topBin 을 기준으로 놓였는데 여기서 쓰는 top 은
            // 그보다 낮을 수 있어, 그만큼 건물이 공중에 뜬다.
            const Eigen::Vector3f foot = c.rep + up * (c.ground - c.rep.dot(up));
            int nsame = 0;
            const float h = smoothTop(c, h_raw, nsame);

            // **홀로 선 칸은 구조물이 아니다.**
            //
            // 건물도 담장도 한 칸으로 끝나지 않는다. 이웃 없이 혼자 서 있는
            // 1 m 짜리 기둥은 스테레오 잡음이 한 칸에 뭉친 것이고, 그것을
            // 세우면 지도 곳곳에 까닭 없는 바늘이 꽂힌다 - 항공뷰에서 도로
            // 바깥에 흩뿌려져 있던 것이 전부 그것이었다.
            //
            // 나무와 기둥은 예외다. 가로등도 홀로 선 나무도 원래 한 칸이다.
            if ((c.cls == Stuff::Building || c.cls == Stuff::Fence) && nsame < 3) {
                continue;
            }

            // 지면에서 꼭대기까지의 중간. 모든 몸통이 여기를 중심으로 선다.
            const Eigen::Vector3d mid = (foot + up * (h * 0.5f)).cast<double>();
            const Eigen::Vector3d hup = (up * (h * 0.5f)).cast<double>();
            const float half = cell * 0.5f;

            if (c.cls == Stuff::Building) {
                // **건물은 지면에서 꼭대기까지 세운 하나의 덩어리다.**
                //
                // 앞선 판은 그 칸의 법선에 수직인 평면 조각 하나를 반 높이로
                // 세웠다. 벽면 한 조각으로는 맞지만, 조각들이 각자의 법선을
                // 따라 제멋대로 돌아서서 화면에는 흩어진 판자로 보였다 -
                // 구조 텐서의 법선은 칸마다 잡음만큼 흔들리기 때문이다.
                //
                // 밑면을 격자칸에 맞추면 그 흔들림이 사라진다. 이웃한 기둥의
                // 밑면과 변이 맞닿으므로 벽면이 저절로 이어지고, 세 면 중
                // 카메라 쪽만 보이므로 평평한 벽으로 읽힌다.
                solidBox(mid, (g.a * half).cast<double>(), hup,
                         (g.b * half).cast<double>(), orb, f, col, 1.0);
                continue;
            }
            if (c.cls == Stuff::Fence) {
                // 담장: 낮고 얇은 판. **두께 방향을 그 칸의 법선에 맞춘다.**
                // 격자축에 고정하면 도로와 비스듬한 담장이 계단으로 남는다.
                Eigen::Vector3f n = c.normal - up * c.normal.dot(up);
                if (n.norm() < 1e-3f) n = g.b;
                n.normalize();
                const Eigen::Vector3f tang = up.cross(n).normalized();
                solidBox(mid, (tang * half).cast<double>(), hup,
                         (n * (half * 0.30f)).cast<double>(), orb, f, col, 0.92);
                continue;
            }
            if (c.cls == Stuff::Vegetation) {
                // **한 칸에 한 그루가 아니다.**
                //
                // 수관 하나는 격자칸 여러 개를 덮는다. 칸마다 나무를 세우면
                // 가로수 한 그루가 작은 나무 아홉 그루의 무더기가 되어,
                // 멀리서 보면 초록 자갈밭처럼 보인다.
                //
                // 군집의 대표만 세운다: 5x5 이웃 중 가장 높은 나무 칸이 그
                // 자리의 나무다. 높이가 같으면 키로 갈라 매 프레임 같은 칸이
                // 뽑히게 한다 - 아니면 나무가 프레임마다 자리를 옮긴다.
                //
                // 3x3 으로 잡으면 2 m 마다 한 그루가 서는데 수관 지름이 그보다
                // 크므로 전부 겹쳐 한 덩이 초록 둔덕이 된다. 5x5 면 간격이
                // 수관 지름과 비슷해져 그루가 그루로 보인다.
                int nveg = 0;
                bool seed = true;
                for (int di = -2; di <= 2 && seed; ++di) {
                    for (int dj = -2; dj <= 2; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const auto it = cols.find(GroundGrid::key(c.i + di, c.j + dj));
                        if (it == cols.end() || it->second.cls != Stuff::Vegetation) continue;
                        ++nveg;
                        const int nb_top = std::min(it->second.topBin(),
                                                    it->second.topRun());
                        if (nb_top > tb || (nb_top == tb && it->first > k)) {
                            seed = false;
                            break;
                        }
                    }
                }
                if (!seed) continue;
                // 수관 반지름. 군집이 넓을수록 크되, 나무 높이를 넘지 않는다.
                // nveg 는 5x5 에서 최대 24 이므로 계수도 그 범위에 맞춘다.
                const float spread = cell * (0.60f + 0.045f * static_cast<float>(nveg));
                const float rad = std::clamp(std::min(spread, h * 0.40f), 0.35f, 2.5f);
                treeModel(foot, h, rad, up, orb, f, col);
                continue;
            }
            if (c.cls == Stuff::Pole) {
                // 기둥: 가는 사각기둥. 선으로 그리면 굵기가 거리와 무관해져
                // 멀리 있는 가로등이 가까운 것과 같은 굵기로 남는다.
                const float pw = std::max(0.07f, cell * 0.10f);
                solidBox(mid, (g.a * pw).cast<double>(), hup,
                         (g.b * pw).cast<double>(), orb, f, col, 1.0);
                continue;
            }
        }
    }

    // 예측을 그린다. **관측과 절대로 같아 보이면 안 된다.**
    //
    // 예측한 구조와 실제로 본 구조가 화면에서 구분되지 않으면, 그림은 있지도
    // 않은 세계를 보여 주면서 그것을 관측이라고 말하는 것이 된다. 그래서
    // 파선으로, 배경 쪽으로 눌러서, 신뢰도만큼만 진하게 그린다. 채점이 끝난
    // 것은 결과대로 색을 바꾼다 - 맞은 예측과 틀린 예측이 자리째로 보인다.
    void predictions(const std::unordered_map<std::int64_t, Prediction>& preds,
                     const Eigen::Vector3d& up, const Orbit& orb, double f,
                     const Eigen::Vector3d& ego, double draw_radius, float cell) {
        const double r2 = draw_radius * draw_radius;
        const GroundGrid g(up.cast<float>(), cell);

        // **예측도 면으로 잇는다.**
        //
        // 관측만 사각형으로 잇고 예측은 낱개 세로 파선으로 두었더니, 예측이
        // 공간이 아니라 울타리 말뚝처럼 보였다. 예측한 것도 벽이고 나무줄이면
        // 이어져 있어야 그 모양이 읽힌다.
        //
        // 다만 **선의 종류는 다르게 유지한다.** 예측이 관측과 같은 실선이 되면
        // 화면이 본 적 없는 세계를 관측이라고 말하게 된다. 이어붙이되 파선이다.
        auto dashed = [&](const cv::Point& a, const cv::Point& b,
                          const cv::Scalar& col) {
            const int m = 4 * std::max(img_.cols, img_.rows);
            if (std::abs(a.x) > m || std::abs(a.y) > m ||
                std::abs(b.x) > m || std::abs(b.y) > m) return;
            const double dx = b.x - a.x, dy = b.y - a.y;
            const int steps = std::max(1, static_cast<int>(std::hypot(dx, dy) / 5.0));
            for (int i = 0; i < steps; i += 2) {
                const double t0 = static_cast<double>(i) / steps;
                const double t1 = std::min(1.0, static_cast<double>(i + 1) / steps);
                cv::line(img_,
                         {static_cast<int>(a.x + dx * t0), static_cast<int>(a.y + dy * t0)},
                         {static_cast<int>(a.x + dx * t1), static_cast<int>(a.y + dy * t1)},
                         col, 1, cv::LINE_AA);
            }
        };

        for (const auto& [k, pr] : preds) {
            // 멀리 있는 예측은 그리지 않는다. 4 만 개를 전부 파선으로 그리면
            // 프레임당 62 ms 가 든다 (실측) - 그 대부분은 화면에서 한 점에
            // 뭉쳐 아무 것도 말해 주지 않는다.
            if ((pr.p.cast<double>() - ego).squaredNorm() > r2) continue;
            cv::Point pa, pb;
            if (!project3(pr.p.cast<double>(), orb, f, pa)) continue;
            const Eigen::Vector3d base =
                pr.p.cast<double>() - up.normalized() * pr.height;
            if (!project3(base, orb, f, pb)) continue;

            cv::Scalar col = (pr.grade > 0) ? C_GOOD
                           : (pr.grade < 0) ? C_BAD : stuffColor(pr.cls);
            // 미채점 예측은 신뢰도만큼만 진하다. 100 m 앞은 거의 배경이다.
            const double a = (pr.grade != 0) ? 0.85
                                             : 0.25 + 0.45 * pr.conf;
            col = {col[0] * a + C_VOID[0] * (1 - a),
                   col[1] * a + C_VOID[1] * (1 - a),
                   col[2] * a + C_VOID[2] * (1 - a)};

            // 세로 파선 - 그 자리에 얼마나 높은 것을 예상하는가.
            dashed(pb, pa, col);

            // 이웃 예측과 잇는다. 같은 클래스일 때만, 그리고 사각형을 닫고
            // 대각선을 그어 삼각형으로 만든다 - 관측 쪽과 같은 규칙이다.
            const Prediction* na = nullptr;
            const Prediction* nb2 = nullptr;
            const Prediction* nab = nullptr;
            {
                auto get = [&](const Eigen::Vector3f& d) -> const Prediction* {
                    const auto it = preds.find(g.key(pr.p + d));
                    return (it != preds.end() && it->second.cls == pr.cls)
                         ? &it->second : nullptr;
                };
                na  = get(g.a * cell);
                nb2 = get(g.b * cell);
                nab = get(g.a * cell + g.b * cell);
            }
            cv::Point q_a, q_b, q_ab;
            const bool ok_a  = na  && project3(na->p.cast<double>(),  orb, f, q_a);
            const bool ok_b  = nb2 && project3(nb2->p.cast<double>(), orb, f, q_b);
            const bool ok_ab = nab && project3(nab->p.cast<double>(), orb, f, q_ab);
            if (ok_a)  dashed(pa, q_a, col);
            if (ok_b)  dashed(pa, q_b, col);
            if (ok_a && ok_ab) dashed(q_a, q_ab, col);
            if (ok_b && ok_ab) dashed(q_b, q_ab, col);
            if (ok_ab) dashed(pa, q_ab, col);
        }
    }

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
            col = {col[0] * a + C_VOID[0] * (1 - a),
                   col[1] * a + C_VOID[1] * (1 - a),
                   col[2] * a + C_VOID[2] * (1 - a)};
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

bool loadManifest(const fs::path& p, std::vector<Seq>& seqs,
                  bool drive_only) {
    std::ifstream f(p);
    if (!f) return false;
    std::map<std::string, std::size_t> idx;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line, '\t');
        if (c.empty()) continue;
        if (c[0] == "SEQ" && c.size() >= 6) {
            // **차량 주행 시퀀스만 싣는다.**
            //
            // 이 뷰어의 화면은 도로 장면을 전제로 맞춰져 있다 - 자차를 차로
            // 그리고, 노면 텍스처를 깔고, 회랑을 외삽하고, 높이 상한을 지붕에
            // 맞춘다. 손에 든 카메라로 방 안을 도는 TUM 을 같은 화면에 넣으면
            // 그 전제가 전부 어긋난 그림이 나온다. --all 로 되돌릴 수 있다.
            if (drive_only && c[2] != "kitti") continue;
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
void backProject(const cv::Mat& depth16, const cv::Mat& gray,
                 const wme_tools::DatasetCalib& cal,
                 const Eigen::Isometry3d& T_world_cam, int stride,
                 double dmin, double dmax, double cmin, double cmax,
                 const Eigen::Vector3d& up, double ego_h,
                 double h_lo, double h_hi,
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
            const Eigen::Vector3d p_w = T_world_cam * p_cam;

            // **하늘에는 아무 것도 없다.**
            //
            // 하늘은 텍스처가 없으므로 스테레오 정합이 아무 시차나 골라낸다.
            // 그 시차를 역투영하면 도로 위 수십 m 에 구조가 생기고, 화면에는
            // 뚫려 있어야 할 곳이 지붕처럼 덮인다 - 실제로 그렇게 나왔다.
            //
            // 자차 높이를 기준으로 위아래를 자른다. 도로 장면에서 자차보다
            // 12 m 위에 있는 것은 건물 꼭대기까지이고, 그보다 높으면 관측이
            // 아니라 정합 실패다. 아래쪽도 자른다 - 노면 밑은 없다.
            const double h = p_w.dot(up) - ego_h;
            if (h < h_lo || h > h_hi) continue;

            Splat s;
            s.p = p_w.cast<float>();
            // 색 범위는 **유효 범위와 다르다.** KITTI 의 유효 범위는 3~80 m 지만
            // 실제 점의 대부분은 5~30 m 에 있어서, 유효 범위로 정규화하면 전부
            // turbo 의 파란 끝에 몰려 색이 정보를 잃는다.
            s.depth_norm = static_cast<float>(
                std::clamp((z - cmin) / std::max(1e-6, cmax - cmin), 0.0, 1.0));
            s.age = 0.0f;
            s.range = static_cast<float>(z);   // 관측 당시 거리
            // 같은 화소의 밝기를 함께 들고 온다. 깊이와 밝기는 같은 화소에서
            // 나오므로 재투영도 보간도 필요 없다.
            if (!gray.empty() && v < gray.rows && u < gray.cols) {
                s.intensity = gray.at<std::uint8_t>(v, u);
            }
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
    std::string dump_feat;
    // 기본은 차량 주행 시퀀스만. --all 이면 전부 싣는다.
    bool drive_only = true;
    int shot_frame = 0;
    // 복셀 상한. 넘으면 복셀을 키워 다시 합치므로 지도가 사라지지는
    // 않고 성겨진다. 260k 는 KITTI 00 에서 0.3 m -> 1.2 m 까지 두 번
    // 거칠어져, 라이다 지도라면 있어야 할 밀도가 남지 않았다.
    int cloud_cap = 900000;
    int start_cam = 0;      // 0 1인칭 / 1 3인칭 / 2 지도 / 3 항공

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--manifest") manifest = argv[i + 1];
        else if (k == "--seq") start_seq = std::atoi(argv[i + 1]);
        else if (k == "--autoplay") autoplay = std::atoi(argv[i + 1]) != 0;
        else if (k == "--screenshot") shot_path = argv[i + 1];
        else if (k == "--frame") shot_frame = std::atoi(argv[i + 1]);
        else if (k == "--cloud-cap") cloud_cap = std::atoi(argv[i + 1]);
        else if (k == "--dump-features") dump_feat = argv[i + 1];
        else if (k == "--all") drive_only = false;
        else if (k == "--cam") start_cam = std::atoi(argv[i + 1]);
    }

    const fs::path scene_dir = fs::path(manifest).parent_path();
    std::vector<Seq> seqs;
    if (!loadManifest(manifest, seqs, drive_only)) {
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
    // 헤드리스도 0 부터 재생한다. 예측 채점이 프레임 진행에
    // 의존하므로, 목표 프레임으로 건너뛰면 채점이 영영 안 된다.
    int frame = 0;
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
    Eigen::Vector3d ego_pos = Eigen::Vector3d::Zero();   // 지금 프레임의 자차 위치 (정답)
    Eigen::Vector3d heading_dir = Eigen::Vector3d::Zero();  // 진행 방향 (1/3 인칭 시선)
    // 마지막으로 라벨을 계산한 자리. 이만큼 움직이기 전엔 다시 안 한다.
    Eigen::Vector3d label_at[2] = {Eigen::Vector3d::Constant(1e9),
                                   Eigen::Vector3d::Constant(1e9)};
    // 패널별 시선 중심. 각 패널은 **자기가 추정한** 위치에 걸린다 - 정답에
    // 걸면 드리프트가 큰 쪽의 지도가 화면 밖으로 나간다.
    Eigen::Vector3d pan_center[2] = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    // 패널별 시선 방향. **그 패널의 추정 궤적에서** 뽑는다 - 정답에서
    // 뽑으면 드리프트만큼 카메라가 계속 비틀린다.
    Eigen::Vector3d pan_head[2] = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    // 운동 유형. 포즈만 내놓는 것과 무엇이 일어났는지 말하는 것은 다르다.
    std::string mot_kind[2] = {"", ""};
    double mot_yaw[2] = {0.0, 0.0};      // 누적 요각 (rad)
    int    mot_turns[2] = {0, 0};        // 완전히 돈 바퀴 수
    int    mot_at[2] = {-1, -1};         // 마지막으로 판정한 포즈 인덱스
    bool   user_zoomed = false;    // W/S 를 눌렀으면 자동 거리 조정을 멈춘다
    // 원시 표면 격자는 기본 꺼짐. 거친 복셀에서 이 격자는 가는 선 수만 개가
    // 되고, 그러면 좌우에 선 건물이 하늘에 뜬 잡음처럼 보인다 - 라벨 형상이
    // 그 자리를 대신한다. M 으로 켤 수 있다.
    bool   show_mesh = false;      // 표면 격자 (M 으로 토글)
    cv::Mat canvas(WIN_H, WIN_W, CV_8UC3);
    CloudView cloud[2];
    VoxelMap acc[2];
    // 표면 격자용 굵은 복셀. acc 를 그대로 이으면 선분이 300 만 개가
    // 되어 프레임을 못 채운다. 구조를 보여 주는 데는 성긴 격자로 충분하다.
    VoxelMap mesh[2];
    std::vector<MemoryObject> mem[2];
    // 지도에서 유도한 장면 라벨 (지면/건물/담장/나무/기둥).
    std::unordered_map<std::int64_t, Column> stuff[2];
    // 노면 텍스처. 부피 지도와 따로 가는 2 차원 정밀 격자다.
    std::unordered_map<std::int64_t, RoadCell> road[2];
    // 삼각형 표면. 자차가 의미 있게 움직였을 때만 다시 만든다.
    SurfMesh surf[2];
    Eigen::Vector3d surf_at[2] = {Eigen::Vector3d::Constant(1e9),
                                  Eigen::Vector3d::Constant(1e9)};
    // 아이소서피스는 기본 꺼짐. 라벨 형상(건물 평면, 나무 입체, 기둥)이
    // 같은 자리를 이미 그리는데 둘을 겹치면 매끄러운 형상 위로 거친 메시가
    // 삐져나온다. 메시는 잡음만큼 거칠고 형상은 그렇지 않으므로, 깔끔한
    // 화면에서는 형상이 이긴다. G 로 켜서 비교할 수 있다.
    bool show_surf = false;        // G 로 토글 - 삼각형 표면
    Eigen::Vector3f road_a{Eigen::Vector3f::UnitX()};
    Eigen::Vector3f road_b{Eigen::Vector3f::UnitZ()};
    // 오버레이는 기본 꺼짐. 전부 켜면 지도가 안 보이고, 사람이
    // 보려던 것은 지도다. L / O 로 필요할 때 켠다.
    // 라벨을 기본으로 켠다. 형상이 있어야 무엇을 인식했는지가 보인다.
    bool show_stuff = true;        // L 로 토글
    // 앞을 내다본 것과 그 성적표.
    std::unordered_map<std::int64_t, Prediction> pred[2];
    PredictScore pscore[2];
    bool show_pred = false;        // O 로 토글
    // **기본은 의미 형상만 그리는 깔끔한 3D 맵이다.**
    //
    // 원시 점군과 의미 형상을 같이 그리면 점이 형상을 덮는다. 나무를 입체로
    // 만들어도 그 위에 점 수만 개가 뿌려지면 나무로 안 보이고, 건물 평면도
    // 점에 묻힌다. 지도가 "무엇이 있는가" 를 말하려면 그 답만 그려야 한다.
    //
    // 원시 점군은 C 로 켠다 - 분류가 틀렸는지 확인할 때는 그것이 근거다.
    bool show_cubes = false;       // C 로 토글 - 원시 복셀/점군
    bool show_edl = true;          // E 로 토글 - Eye-Dome Lighting
    // 레이어 격리. 전부 한 화면에 겹치면 무엇을 보고 있는지 알 수
    // 없다. 한 층씩 떼어 보는 것이 벤치마크에서는 기본이다.
    //   0 전체 / 1 지도만 / 2 차량만 / 3 사람만 / 4 구조물만 / 5 지면만
    int  layer = 0;                // T 로 순환
    // 단계별 소요 시간 (ms). 랙의 원인을 추측하지 않기 위한 계측이다.
    double prof_label[2] = {0, 0}, prof_vec[2] = {0, 0};
    double prof_pred[2] = {0, 0}, prof_pdraw[2] = {0, 0};
    double prof_cloud[2] = {0, 0}, prof_frame = 0;
    int acc_frame[2] = {-1, -1};
    std::vector<double> err_series[2];

    // 카드 기하
    const int card_w = (WIN_W - 3 * PAD) / 2;
    const int card_y = TOPBAR_H + PAD;
    const int card_h = WIN_H - BOTTOM_H - card_y - PAD;
    const int vp_h = card_h - 250;

    while (true) {
        const auto t_frame0 = std::chrono::steady_clock::now();
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

            // 진행 방향. 한 프레임 차분은 잡음이라 시선이 매 프레임 튄다 -
            // 1 인칭에서는 그것이 곧 멀미다. 몇 프레임 뒤를 본다.
            if (j > 0) {
                const int j0 = std::max(0, j - 10);
                const Eigen::Vector3d d =
                    s.gt[static_cast<std::size_t>(j)].p - s.gt[static_cast<std::size_t>(j0)].p;
                if (d.norm() > 1e-3) {
                    // 부드럽게 따라간다. 코너에서 화면이 홱 돌지 않게.
                    heading_dir = (heading_dir.squaredNorm() < 1e-12)
                                ? d.normalized()
                                : (heading_dir * 0.85 + d.normalized() * 0.15).normalized();
                }
            }

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
                // 노면 격자의 두 축. world_up 과 함께 한 번만 정한다.
                {
                    Eigen::Vector3f u = orb.world_up.cast<float>().normalized();
                    Eigen::Vector3f aa = u.cross(Eigen::Vector3f::UnitZ());
                    if (aa.norm() < 1e-6f) aa = u.cross(Eigen::Vector3f::UnitX());
                    road_a = aa.normalized();
                    road_b = u.cross(road_a).normalized();
                }
                mesh[0].voxel = mesh[1].voxel =
                    (s.dataset == "kitti") ? 2.2f : 0.16f;
            }

            // 카메라 모드별 시선각. 항공뷰는 거의 수직으로 내려다보되 완전한
            // 90 도는 피한다 - 정확히 수직이면 지면 위 높이가 전부 한 점에
            // 겹쳐 물체의 높이 정보가 사라진다.
            //   0 FIRST  1 인칭. 눈이 인식 시점에 있고 진행 방향을 본다
            //   1 THIRD  3 인칭. 자차 뒤 위에서 따라간다 - 이동이 보인다
            //   2 MAP    비스듬히 내려다보는 고정 방위 지도
            //   3 BIRD   거의 수직. 배치와 궤적 형상이 가장 잘 읽힌다
            //
            // 기본이 1 인칭인 이유: 실내 시퀀스에서 위에서 내려다보면 방
            // 하나가 점 덩어리로만 보여 무엇을 보고 있는지 알 수 없다.
            // 사람이 그 자리에 서 있을 때 보이는 것이 기준이어야 한다.
            const double want_pitch = (cam_mode == 3) ? 1.30
                                    : (cam_mode == 2) ? 0.85
                                    : (cam_mode == 1) ? 0.35 : 0.02;
            // headless 는 루프를 한 번만 돈다. 보간만 하면 목표각에
            // 도달하지 못한 그림이 저장된다.
            orb.pitch += (want_pitch - orb.pitch) * (headless ? 1.0 : 0.25);
            if (!user_zoomed) {
                // 1 인칭은 눈이 자차에 있어야 하므로 궤도 반지름이 거의 0 이다.
                // 3 인칭은 자차가 화면 아래쪽에 오도록 조금 뒤에 선다.
                if (cam_mode == 0)      orb.dist = base_dist * 0.06;
                // 3 인칭은 **지도 반경보다 멀리** 선다. 24.5 m 에서 반경 55 m
                // 짜리 지도를 보면 프레임이 통째로 지도가 되고, 그것이 "하늘이
                // 채워졌다" 로 읽혔다. 라이다 시각화가 늘 높고 멀리서 보는
                // 이유는 취향이 아니라 이것이다 - 구조는 전체가 보여야 구조다.
                else if (cam_mode == 1) orb.dist = base_dist * 0.95;
                else if (cam_mode == 3) orb.dist = base_dist * 1.1;
                else                    orb.dist = base_dist;
            }
            // 시선 방향은 **패널마다** 자기 추정에서 뽑아 아래에서 넣는다.
            // 여기서 정답 궤적의 방향을 넣으면 두 패널이 같은 곳을 보게 되고,
            // 드리프트가 있는 쪽은 자기 지도를 비스듬히 보게 된다.
            orb.heading = Eigen::Vector3d::Zero();
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

            // **각 패널은 자기 인식 시점에 걸린다.**
            //
            // 두 시스템은 서로 다른 포즈를 추정하고, 각자의 지도는 **자기
            // 추정 포즈로** 역투영해 쌓은 것이다. 그런데 화면 중심을 정답
            // 포즈에 두면, 드리프트가 큰 쪽은 지도가 통째로 화면 밖으로
            // 밀려난다 - walking_xyz 에서 ATE 가 124 cm 인 패널의 표면 격자가
            // 오른쪽 구석에 몰려 있던 것이 그 결과다. 화면 밖으로 나간 것은
            // 잘못 그린 것이 아니라 **정답 좌표계로 보고 있었기 때문** 이다.
            //
            // 각 패널을 자기 추정 위치에 걸면, 그 시스템이 자기 세계에서
            // 무엇을 보고 있는지가 보인다. 두 지도가 어긋나 있다는 사실은
            // 아래 궤적(정답 회색 대 추정 색)이 이미 말해 준다.
            Orbit ob = orb;
            Eigen::Vector3d ego_k = ego_pos;
            {
                const double st = s.rgb.empty() ? 0.0
                                : s.rgb[static_cast<std::size_t>(frame)].first;
                double bd = 0.25;
                int pj = -1;
                for (std::size_t j = 0; j < run.traj.size(); ++j) {
                    const double d = std::abs(run.traj[j].t - st);
                    if (d < bd) { bd = d; pj = static_cast<int>(j); }
                }
                if (pj >= 0) {
                    ego_k = run.aligned[static_cast<std::size_t>(pj)].translation();

                    // **1 인칭에서는 눈이 포즈에 정확히 있어야 한다.**
                    //
                    // 지도 뷰에서는 중심을 평활해도 된다 - 지도가 조금 늦게
                    // 따라올 뿐이다. 그런데 1 인칭에서 눈을 평활하면 눈이
                    // 자기 몸보다 뒤에 있게 되고, 화면이 앞뒤로 미끄러진다.
                    // 좌표를 읽으려는 사람에게는 그것이 곧 흔들림이다.
                    ob.center = (cam_mode == 0 || acc_frame[k] < 0)
                              ? ego_k : pan_center[k] * 0.80 + ego_k * 0.20;

                    // **시선은 자세에서 나온다, 이동 방향이 아니라.**
                    //
                    // 이동 방향을 시선으로 쓰면 차량처럼 앞으로만 가는 경우에만
                    // 맞다. 손에 들고 위아래로 흔들거나 제자리에서 360 도 도는
                    // 시퀀스에서는 이동 벡터가 의미가 없거나 잡음이고, 그것을
                    // 시선으로 쓰면 카메라가 회전만 해도 시야가 엉뚱한 데를
                    // 향한다. 화면에서는 **세계가 따라 움직이는 것처럼** 보이고,
                    // 그러면 이 뷰어는 "세계를 기억한다" 를 보여 주지 못한다.
                    //
                    // 추정 포즈의 회전은 카메라가 실제로 어디를 향하는지다.
                    // 그것을 쓰면 카메라가 어떻게 움직이든 세계는 제자리에
                    // 있고, 움직이는 것은 시점뿐이다 - 그것이 세계 모델이다.
                    const Eigen::Matrix3d R =
                        run.aligned[static_cast<std::size_t>(pj)].rotation();
                    // 카메라 광축은 카메라 좌표계의 +z 다.
                    const Eigen::Vector3d look = R * Eigen::Vector3d::UnitZ();
                    if (look.norm() > 1e-6) {
                        // 평활은 약하게만. 세게 걸면 회전이 늦게 따라와
                        // 그것 자체가 미끄러짐으로 보인다.
                        pan_head[k] = (pan_head[k].squaredNorm() < 1e-12)
                                    ? look.normalized()
                                    : (pan_head[k] * 0.55 + look.normalized() * 0.45)
                                          .normalized();
                    }

                    // --- 운동 유형 판정 ---
                    //
                    // 포즈만 내놓는 것은 기존 SLAM 이 하는 일이다. 무엇이
                    // 일어났는지 - 전진인지, 옆걸음인지, 제자리 회전인지,
                    // 한 바퀴를 돌았는지 - 를 말할 수 있어야 세계 모델이다.
                    // 판정은 **직전 포즈와의 상대 변환** 에서 나온다.
                    if (pj > 0 && pj != mot_at[k]) {
                        const Eigen::Isometry3d& Tp =
                            run.aligned[static_cast<std::size_t>(pj - 1)];
                        const Eigen::Isometry3d& Tc =
                            run.aligned[static_cast<std::size_t>(pj)];
                        const Eigen::Isometry3d rel = Tp.inverse() * Tc;
                        // 상대 이동을 **카메라 좌표계에서** 본다. 월드에서 보면
                        // "앞으로" 가 시퀀스마다 다른 축이 된다.
                        const Eigen::Vector3d t = rel.translation();
                        const Eigen::AngleAxisd aa(rel.rotation());
                        const double ang = std::abs(aa.angle());

                        // 누적 요각. 월드 up 둘레로 돈 각을 부호까지 더한다 -
                        // 360 도 회전은 이 값으로만 알 수 있다.
                        const double dyaw = aa.angle() * aa.axis().dot(orb.world_up);
                        mot_yaw[k] += dyaw;
                        if (std::abs(mot_yaw[k]) >= 2.0 * kPiV) {
                            mot_turns[k] += (mot_yaw[k] > 0 ? 1 : -1);
                            mot_yaw[k] -= (mot_yaw[k] > 0 ? 1 : -1) * 2.0 * kPiV;
                        }

                        // 회전이 지배적인가. 시야 한 화면(약 0.9 rad)에 견줘
                        // 판단한다 - 각도와 거리를 그냥 비교할 수는 없다.
                        const double rot_px = ang / 0.9;
                        const double tr_m = t.norm();
                        if (rot_px < 0.02 && tr_m < 0.005) {
                            mot_kind[k] = "still";
                        } else if (rot_px > tr_m * 2.0) {
                            mot_kind[k] = "rotating in place";
                        } else {
                            // 카메라계 축: x 오른쪽, y 아래, z 앞
                            const double ax = std::abs(t.x()), ay = std::abs(t.y()),
                                         az = std::abs(t.z());
                            if (az >= ax && az >= ay) {
                                mot_kind[k] = (t.z() > 0) ? "forward" : "backward";
                            } else if (ax >= ay) {
                                mot_kind[k] = (t.x() > 0) ? "strafe right" : "strafe left";
                            } else {
                                mot_kind[k] = (t.y() > 0) ? "down" : "up";
                            }
                            // 옆·위아래로 가면서 돌면 대각선이다.
                            if (rot_px > tr_m * 0.6) mot_kind[k] += " + turning";
                        }
                        mot_at[k] = pj;
                    }
                }
                pan_center[k] = ob.center;
                ob.heading = (cam_mode <= 1) ? pan_head[k] : Eigen::Vector3d::Zero();
            }

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

            // 검출 상자로 받아들일 최대 크기. **장면 규모에서 나온다.**
            //
            // 검출 상자 안이 배경이면 깊이 중앙값이 엉뚱한 값을 잡아 물체가
            // 터무니없이 커진다. 14 m 는 KITTI 의 트럭/버스를 통과시키려고 둔
            // 값인데, 그걸 TUM 에 그대로 쓰면 5 m 짜리 사무실 안에 14 m 짜리
            // "사람" 이 생겨 화면을 통째로 덮는다 - walking_xyz 에서 지도가
            // 거대한 주황 와이어프레임에 파묻힌 것이 그것이다.
            //
            // 2.5 m 는 사람이 설 수 있는 크기의 상한이다. 그보다 큰 person 은
            // 관측이 아니라 깊이 실패다.
            const double box_max = (s.dataset == "kitti") ? 14.0 : 2.5;

            // 점군 누적. 프레임이 뒤로 가면 다시 쌓는다.
            if (kDrawCloud) {
                if (!has_memory) {
                    // 기억이 없으므로 매 프레임 지운다. 지금 프레임만 남는다.
                    acc[k].clear();
                    mesh[k].clear();
                    mem[k].clear();
                    road[k].clear();
                    acc_frame[k] = frame - 1;
                } else if (acc_frame[k] > frame || acc_frame[k] < 0) {
                    acc[k].clear();
                    mesh[k].clear();
                    road[k].clear();
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
                    // 이 프레임의 자차 높이. 하늘/노면밑 걸러내기의 기준이다.
                    const double ego_hh =
                        run.aligned[static_cast<std::size_t>(pi)].translation()
                            .dot(ob.world_up);
                    // 도로 장면은 위로 12 m (건물 꼭대기), 아래로 4 m.
                    // 실내는 천장이 3 m 를 넘지 않는다.
                    // 실측: 12 m 상한에서도 8 m 위에 32463 개가 남았고, 그것이
                    // 화면에서 하늘을 덮은 덩어리였다. KITTI 00 은 2 층
                    // 주택가라 지붕이 8 m 안쪽이다 - 그보다 위에 있는 것은
                    // 건물이 아니라 하늘에서 온 시차다.
                    // 이 숫자는 **맞바꿈** 이다. 12 m 에서는 8 m 위에 32463 개가
                    // 남아 하늘을 덮었고, 7.5 m 로 자르니 그것이 489 개로 줄었지만
                    // 2 층 지붕까지 같이 잘려 건물 칸이 382 에서 144 로 떨어졌다.
                    // 높이 하나로 하늘과 지붕을 가르려는 것이 애초에 무리다 -
                    // 하늘 점은 높아서 생기는 것이 아니라 텍스처가 없어서 생긴다.
                    // 그래서 9 m 로 완화해 보고 **재서** 정했다:
                    //
                    //   상한   8 m 초과 복셀   건물 칸
                    //   12.0 m      32463        382
                    //    9.0 m       8921        158
                    //    7.5 m        489        144
                    //
                    // 9 m 는 하늘 잡음이 18 배로 늘면서 건물은 14 칸밖에 못
                    // 되찾는다. 7.5 m 위쪽은 압도적으로 하늘이지 지붕이 아니다.
                    // 근본 해결은 시차 신뢰도를 깊이와 함께 들고 오는 것이지
                    // 높이를 자르는 것이 아니고, 그건 StereoDepth 쪽 일이다.
                    const double hlo = (s.dataset == "kitti") ? -3.0 : -2.0;
                    const double hhi = (s.dataset == "kitti") ?  7.5 :  2.2;
                    // 같은 프레임의 회색 영상. 노면 표시는 기하가 아니라
                    // 밝기로만 존재하므로 이것 없이는 주차선을 그릴 수 없다.
                    const cv::Mat gimg = cv::imread(
                        s.rgb[static_cast<std::size_t>(fi)].second,
                        cv::IMREAD_GRAYSCALE);
                    backProject(d16, gimg, s.calib,
                                run.aligned[static_cast<std::size_t>(pi)],
                                stride,
                                s.calib.depth_min, s.calib.depth_max, cmin, cmax,
                                ob.world_up, ego_hh, hlo, hhi, pts);
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
                    // 상한이 6 m 면 2 층 이상이 전부 같은 빨강으로 뭉쳐 도시가 평평해
                    // 보인다. 라이다 지도가 높이를 색으로 읽히게 하는 것은
                    // 범위가 실제 구조 높이를 덮기 때문이다.
                    const double h1 = (s.dataset == "kitti") ? 14.0 :  2.4;
                    // **시점 섹터.** 어디에서 봤는지를 16 칸으로 접는다. 자차
                    // 위치를 5 m 로 양자화하므로 같은 자리에서 여러 프레임 본
                    // 것은 한 시점으로 세어진다 - 그것이 요점이다. 스테레오
                    // 유령은 같은 오매칭이 인접 프레임에서 그대로 재현되므로
                    // 관측 횟수로는 안 걸리고, **시점이 바뀌어야** 사라진다.
                    //
                    // **격자 크기는 장면 규모에서 나온다.** 5 m 는 도로에서
                    // 맞지만 방 안에서는 전 구간이 한 섹터가 되어, 서로 다른
                    // 시점 조건을 영영 못 채우고 지도가 통째로 사라진다 -
                    // TUM 에서 실제로 그랬다.
                    const double sec_m = (s.dataset == "kitti") ? 5.0 : 0.4;
                    const Eigen::Vector3d epos =
                        run.aligned[static_cast<std::size_t>(pi)].translation();
                    const int sector = static_cast<int>(
                        (static_cast<long long>(std::floor(epos.x() / sec_m)) * 7 +
                         static_cast<long long>(std::floor(epos.z() / sec_m)) * 13) & 15);
                    const Eigen::Vector3f up_f = ob.world_up.cast<float>();
                    const float ego_h = static_cast<float>(ego_k.dot(ob.world_up));
                    for (auto& p : pts) {
                        const float h = p.p.dot(up_f) - ego_h;
                        p.depth_norm = static_cast<float>(
                            std::clamp((h - h0) / (h1 - h0), 0.0, 1.0));
                        p.seen = fi;
                        // **가까이서 본 것이 먼저 들어간다.** 가까운 관측이
                        // 자리를 잡은 뒤에 주변의 먼 유령을 걷어내야, 방금
                        // 넣은 좋은 값까지 같이 지우는 일이 없다.
                        acc[k].insert(p, static_cast<std::size_t>(cloud_cap), sector);
                        mesh[k].insert(p, 120000, sector);
                    }

                    // **노면은 정밀 격자에 따로 쌓는다.**
                    //
                    // 부피 복셀은 0.3 m 라 15 cm 짜리 주차선을 담을 수 없다.
                    // 노면만 0.1 m 2 차원 격자로 받으면 같은 메모리로 세 배
                    // 촘촘해지고, 흰 선이 칸보다 굵어져 비로소 보인다.
                    {
                        const float e0 = static_cast<float>(ego_hh);
                        const float rinv = 1.0f / kRoadCell;
                        for (const auto& p : pts) {
                            const float rel = p.p.dot(up_f) - e0;
                            // 자차 발밑 높이 ± 40 cm 가 노면이다. 연석과 차량
                            // 밑동은 여기서 빠져 부피 지도에 남는다.
                            const float floor_rel = (s.dataset == "kitti") ? -1.65f : -0.8f;
                            if (std::abs(rel - floor_rel) > 0.40f) continue;
                            const std::int64_t rk = roadKey(p.p, road_a, road_b, rinv);
                            auto& rc = road[k][rk];
                            // 가까이서 본 관측이 이긴다. 노면 표시는 원거리
                            // 에서 몇 화소뿐이라 멀리서 본 값은 번져 있다.
                            if (rc.hits == 0 || p.range < rc.range) {
                                rc.h = p.p.dot(up_f);
                                rc.range = p.range;
                                rc.intensity = p.intensity;
                            }
                            if (rc.hits < 255) ++rc.hits;
                        }
                    }
                    // 가까운 관측만 정정 권한을 갖는다. 30 m 에서 본 것으로
                    // 60 m 짜리를 지우면, 둘 다 못 믿을 값인데 하나가 다른
                    // 하나를 심판하는 꼴이 된다.
                    const float trust_r = (s.dataset == "kitti") ? 18.0f : 2.0f;
                    for (const auto& p : pts) {
                        if (p.range > trust_r) continue;
                        acc[k].supersede(p.p, p.range, 1);
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
                            if (!(mx > 0.2) || mx > box_max) continue;
                            const Eigen::Vector3d wc = Tw * b.center;

                            // **결합은 마지막으로 본 자리로 한다, 평균이 아니라.**
                            //
                            // 평균과 비교하면 걸어가는 사람은 평균이 뒤처지면서
                            // 거리가 병합 반경을 넘고, 같은 사람이 짧은 조각
                            // 수십 개로 쪼개진다. 조각마다 관측이 서너 번뿐이라
                            // 동적 판정에 필요한 관측 수에 영영 도달하지 못한다 -
                            // 실제로 walking_xyz 에서 상자 1076 개가 물체 49 개로
                            // 흩어지고 동적 판정은 하나도 서지 않았다.
                            //
                            // 평균은 **그리는** 데 쓰는 값이다 (정지 물체가 관측
                            // 잡음으로 떨지 않게). 결합은 추적이므로 최근 위치가
                            // 맞다. 두 목적에 같은 숫자를 쓴 것이 잘못이었다.
                            MemoryObject* hit = nullptr;
                            double bestd = merge_r;
                            for (auto& m : mem[k]) {
                                if (m.cls != b.cls) continue;
                                const double d = ((m.count > 0 ? m.cur_c : m.center()) - wc).norm();
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

                        // 동적으로 판정된 물체가 지나온 자리를 지운다.
                        //
                        // 판정이 뒤늦게 서더라도 trail 에는 처음부터의 자리가
                        // 다 들어 있으므로, 그 시점까지 쌓인 것도 같이 지워진다.
                        // "지금부터 안 쌓는다" 로는 이미 남은 잔상이 안 없어진다.
                        for (auto& m : mem[k]) {
                            if (!m.dynamic || m.trail.empty()) continue;
                            const Eigen::Vector3d sz = m.size();
                            for (const auto& c : m.trail) {
                                eraseBox(acc[k], c, sz);
                                eraseBox(mesh[k], c, sz);
                            }
                            // 마지막 자리만 남긴다. 다음 프레임이면 그것도
                            // 과거가 되므로 한 번 더 지워진다.
                            const Eigen::Vector3d last = m.trail.back();
                            m.trail.clear();
                            m.trail.push_back(last);
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
                const auto t_c0 = std::chrono::steady_clock::now();
                // 먼저 점으로 전부 깔고, 그 위에 가까운 칸을 큐브로 채운다.
                // 큐브만 그리면 2 px 미만으로 작아지는 먼 구조가 통째로
                // 사라져 "저기엔 아무 것도 없다" 로 읽힌다 - 없는 것과 너무
                // 작아 못 그린 것은 다르다.
                // **지도는 자차 주변 반경까지만 그린다.**
                //
                // 8 m 위 복셀은 756680 개 중 489 개(0.06 %)뿐인데도 화면 위쪽이
                // 꽉 차 보였다. 그것은 하늘 잡음이 아니라 **멀리까지 이어진
                // 지도** 다 - 시선각이 낮으면 지평선이 높이 걸리고 원거리
                // 지도가 프레임을 다 채운다.
                //
                // 라이다 뷰어가 예외 없이 센서 주변 일정 반경만 그리는 이유가
                // 이것이다. 지도가 사라지는 것이 아니라, 지금 볼 수 있는 만큼만
                // 보여 주는 것이다.
                const double map_r = (s.dataset == "kitti") ? 55.0 : 7.0;
                const double map_r2 = map_r * map_r;
                std::vector<Splat> near_pts, ground_pts;
                near_pts.reserve(flat.size() / 2 + 1);
                ground_pts.reserve(flat.size() / 2 + 1);
                const GroundGrid gg(ob.world_up.cast<float>(),
                                    (s.dataset == "kitti") ? 1.0f : 0.5f);
                // 기준 지면 높이. **점을 나누기 전에** 정해야 한다 - 라벨이
                // 없는 칸은 이 값으로 노면 여부를 판정하기 때문이다.
                double plane_h;
                {
                    std::vector<float> gh;
                    const Eigen::Vector3f e0 = ego_k.cast<float>();
                    const float rr = static_cast<float>(map_r * map_r) * 0.25f;
                    for (const auto& [ck, c] : stuff[k]) {
                        if (c.cls != Stuff::Ground) continue;
                        if ((c.rep - e0).squaredNorm() > rr) continue;
                        gh.push_back(c.ground);
                    }
                    if (!gh.empty()) {
                        std::nth_element(gh.begin(), gh.begin() + gh.size() / 2, gh.end());
                        plane_h = gh[gh.size() / 2];
                    } else {
                        plane_h = ego_k.dot(ob.world_up)
                                - ((s.dataset == "kitti") ? 1.65 : 0.8);
                    }
                }
                {
                    const Eigen::Vector3f e = ego_k.cast<float>();
                    // 노면 복셀은 따로 모은다. 큐브로 그리면 도로가 검은
                    // 블록 밭이 되고(실제로 그렇게 나왔다), 버리면 차선과
                    // 주차선이 통째로 사라진다. 밝기로 칠한 납작한 면이 답이다.
                    ground_pts.clear();
                    for (const auto& sp : flat) {
                        if ((sp.p - e).squaredNorm() > map_r2) continue;

                        // **근거 없는 점은 그리지 않는다.**
                        //
                        // 공간: 26 이웃 중 셋도 안 차 있으면 표면이 아니라
                        // SGBM 오매칭의 산탄이다 (PCL 의 Radius Outlier Removal
                        // 과 같은 판정, 해시 복셀이라 kd-tree 없이 된다).
                        //
                        // 시간: 세 번 이상, 서로 다른 두 시점 이상에서 봐야
                        // 인정한다. 유령은 한 시점에서만 재현되므로 이 조건이
                        // 공간 판정이 못 잡는 밀집 덩어리를 걷어낸다.
                        // **기억이 없는 쪽에는 증거를 요구할 수 없다.**
                        //
                        // 시간 게이트는 "세 번, 두 시점" 을 요구하는데, 왼쪽
                        // 패널은 설계상 매 프레임 지도를 비운다. 지워지는
                        // 지도는 그 증거를 영영 못 쌓으므로 게이트가 패널을
                        // 통째로 지워 버렸다 - 화면이 비는 것이 "ORB 가 아무
                        // 것도 못 봤다" 로 읽히지만 사실은 내가 만든 조건이
                        // 그 파이프라인에는 성립할 수 없었던 것이다.
                        //
                        // 누적하는 쪽에만 건다. 대조군을 불리하게 만드는 잣대는
                        // 비교를 망가뜨린다.
                        if (has_memory) {
                            if (sp.nbr < 3) continue;
                            if (!sp.confirmed()) continue;
                        }

                        // **노면인가.** 라벨에 기대지 않는다 - 실내에서는
                        // 라벨의 대부분이 미상이라 노면 판정이 통째로 빠지고,
                        // 그러면 도로가 검은 큐브 밭이 된다. 그 칸의 지면
                        // 높이(라벨이 있으면 그것, 없으면 격자면)에서 40 cm
                        // 안쪽이면 노면이다.
                        {
                            const auto it = stuff[k].find(gg.key(sp.p));
                            const float gh = (it != stuff[k].end())
                                           ? it->second.ground
                                           : static_cast<float>(plane_h);
                            const float rel = sp.p.dot(gg.up) - gh;
                            if (rel < 0.40f) { ground_pts.push_back(sp); continue; }
                        }
                        near_pts.push_back(sp);
                    }
                }
                // **바닥 격자를 먼저 깐다.** 지도보다 나중에 그리면 격자선이
                // 구조 위를 가로질러 지나간다.
                cloud[k].groundPlane(ego_k, plane_h, ob.world_up, ob, vp.width * 0.9,
                                     (s.dataset == "kitti") ? 5.0 : 0.5, map_r);

                // **노면은 밝기로 칠한다.**
                //
                // 차선도 주차선도 횡단보도도 높이 차이가 0 이다. 기하로는
                // 존재하지 않고 흰 페인트라는 사실로만 존재하므로, 깊이만 쓰는
                // 지도에서는 원리적으로 보일 수가 없다. 같은 화소의 카메라
                // 밝기를 들고 오면 그대로 나온다 - 라이다 지도가 intensity 로
                // 차선을 보여 주는 것과 같은 이야기다.
                if (layer == 0 || layer == 1 || layer == 5) {
                    // 노면 텍스처는 부피 지도보다 짧은 반경만 그린다. 0.1 m
                    // 칸은 멀어지면 화면에서 사라지므로 그릴 값이 없고, 그
                    // 자리는 아래 격자면이 이미 채우고 있다.
                    cloud[k].roadTexture(road[k], road_a, road_b, ob.world_up,
                                         ob, vp.width * 0.9, ego_k,
                                         std::min(map_r, (s.dataset == "kitti")
                                                         ? 28.0 : 5.0));
                }
                const bool L_map = (layer == 0 || layer == 1);

                // **지나온 곳 전부를 하나의 표면으로.**
                //
                // 큐브를 쌓으면 지도는 블록 더미로 남는다. 진짜 3D 모형은 닫힌
                // 삼각형 표면이고, 그것이 있어야 "지도" 가 "모형" 이 된다.
                // 표면이 켜지면 큐브는 그리지 않는다 - 같은 것을 두 번 그리면
                // 표면 위로 블록 모서리가 삐져나온다.
                if (show_surf && has_memory && L_map) {
                    if ((ego_k - surf_at[k]).norm() > acc[k].voxel * 4.0) {
                        // 표면은 자차 주변만 만든다. 먼 곳은 화면에서
                        // 삼각형이 픽셀보다 작아 큐브와 구분되지 않는다.
                        // 노면 위 40 cm 부터가 구조물이다. 그 아래는 도로이고,
                        // 도로는 밝기 텍스처가 이미 맡고 있다.
                        surfaceNets(acc[k].cells, acc[k].voxel, ego_k,
                                    std::min(map_r, 22.0), ob.world_up,
                                    plane_h, 0.40, stuff[k], gg, surf[k]);
                        surf_at[k] = ego_k;
                    }
                    cloud[k].surface(surf[k], ob, vp.width * 0.9,
                                     cv::Scalar(196, 186, 170));
                }
                // 원시 점군은 근거를 보고 싶을 때만. 기본 화면은 형상이다.
                if (L_map && show_cubes) {
                    cloud[k].draw(near_pts, ob, vp.width * 0.9, 0.62);
                }
                if (show_cubes && !show_surf && L_map && false) {
                    cloud[k].voxelCubes(near_pts, acc[k].voxel, ob, vp.width * 0.9,
                                        ob.world_up, 0.45);
                }
                prof_cloud[k] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_c0).count();
                // 점 위에 표면 격자를 덧그린다. 순서가 중요하다 - 먼저 그리면
                // 점군이 격자를 덮어 아무 것도 이어져 보이지 않는다.
                // **라벨과 원시 격자는 서로 독립이다.**
                //
                // 라벨 그리기가 show_mesh 안에 중첩돼 있었다. 원시 격자를
                // 끄자 라벨도 같이 꺼졌고, 진단은 지면 0 건물 0 나무 0 을
                // 찍었다 - 분류가 안 된 것이 아니라 아예 돌지 않은 것이다.
                // 하나의 토글이 두 가지를 끄면 그 중 하나는 반드시 조용히
                // 사라진다.
                if (has_memory) {
                    if (show_stuff) {
                        // **지도가 무엇으로 이루어졌는지** 를 그린다.
                        //
                        // 입력은 mesh 가 아니라 acc 다. 표면 격자의 복셀은
                        // KITTI 에서 2.2 m 인데, 그 해상도로는 줄기 위의 빈
                        // 공간도 벽면의 연속성도 표현되지 않는다 - 분류가
                        // 기대는 형상 자체가 사라진다.
                        //
                        // 지면 격자칸은 장면 규모에서 정한다. 도로변 건물은
                        // 1 m 칸이면 벽면이 여러 칸에 걸쳐 이어지고, 실내는
                        // 0.25 m 라야 가구와 벽이 갈린다.
                        // **격자칸은 복셀보다 커야 한다.** 0.25 m 칸에 0.3 m
                        // 복셀이면 칸마다 복셀이 하나뿐이라, 구조 텐서가 요구
                        // 하는 점 수가 영영 안 모이고 전부 미상이 된다 - TUM
                        // 에서 라벨 240 개 중 236 개가 미상이었다.
                        const float ccell = (s.dataset == "kitti") ? 1.0f : 0.5f;
                        const auto t_lbl = std::chrono::steady_clock::now();
                        // 작업 반경. 자차가 지금 보고 있는 범위만 새로
                        // 분류하고 나머지는 이전 라벨을 재사용한다.
                        const float work_r = (s.dataset == "kitti") ? 45.0f : 6.0f;
                        // **자차가 의미 있게 움직였을 때만 다시 분류한다.**
                        //
                        // 프레임당 1 m 남짓 움직이는데 반경 45 m 를 매번 다시
                        // 계산하면 같은 답을 45 번 구하는 셈이다. 실측 50.7 ms
                        // 가 통째로 그 낭비였다. 이동량 기준이라 정지 구간에서
                        // 저절로 멈추고, 빠른 구간에서는 저절로 자주 돈다 -
                        // 프레임 수로 세면 속도에 따라 성기거나 낭비가 된다.
                        const double moved = (ego_k - label_at[k]).norm();
                        if (moved > work_r * 0.06 || stuff[k].empty()) {
                            labelScene(acc[k].cells, ob.world_up, ccell,
                                       acc[k].voxel, ego_k, work_r, stuff[k]);
                            label_at[k] = ego_k;
                        }
                        const auto t_vec = std::chrono::steady_clock::now();
                        if (layer == 0 || layer == 4 || layer == 5) {
                            cloud[k].stuffVectors(stuff[k], ob.world_up, ccell,
                                                  ob, vp.width * 0.9, layer,
                                                  ego_k, map_r);
                        }
                        const auto t_end = std::chrono::steady_clock::now();
                        prof_label[k] = std::chrono::duration<double, std::milli>(
                            t_vec - t_lbl).count();
                        prof_vec[k] = std::chrono::duration<double, std::milli>(
                            t_end - t_vec).count();

                        // 앞을 내다보고, 지나온 자리는 채점한다.
                        //
                        // 순서가 중요하다: **채점이 먼저** 다. 이번 프레임에
                        // 새로 세운 예측이 곧바로 채점 대상이 되면, 아직
                        // 지나가지도 않은 것을 판정하게 된다.
                        if (show_pred) {
                            // 진행 방향은 최근 궤적에서 잡는다. 한 프레임
                            // 차분은 잡음이라 방향이 매 프레임 튄다.
                            Eigen::Vector3d dir = Eigen::Vector3d::Zero();
                            {
                                const int back = 12;
                                const int i1 = std::min<int>(
                                    static_cast<int>(run.aligned.size()) - 1,
                                    std::max(0, static_cast<int>(
                                        run.aligned.size() * (frame + 1) / nframes) - 1));
                                const int i0 = std::max(0, i1 - back);
                                if (i1 > i0) {
                                    dir = run.aligned[static_cast<std::size_t>(i1)].translation()
                                        - run.aligned[static_cast<std::size_t>(i0)].translation();
                                }
                            }
                            const auto t_p0 = std::chrono::steady_clock::now();
                            if (dir.norm() > 0.2) {
                                gradePredictions(pred[k], stuff[k], ego_k, dir,
                                                 ob.world_up, ccell, pscore[k]);
                                predictAhead(stuff[k], ego_k, dir, ob.world_up,
                                             ccell, frame, pscore[k], pred[k]);
                            }
                            const auto t_p1 = std::chrono::steady_clock::now();
                            cloud[k].predictions(pred[k], ob.world_up, ob,
                                                 vp.width * 0.9, ego_k,
                                                 (s.dataset == "kitti") ? 130.0 : 14.0,
                                                 ccell);
                            const auto t_p2 = std::chrono::steady_clock::now();
                            prof_pred[k] = std::chrono::duration<double, std::milli>(
                                t_p1 - t_p0).count();
                            prof_pdraw[k] = std::chrono::duration<double, std::milli>(
                                t_p2 - t_p1).count();
                        }
                    }
                    if (show_mesh) {
                        for (auto& v : mesh[k].cells) {
                            v.second.age = static_cast<float>(std::clamp(
                                1.0 - static_cast<double>(v.second.seen) / span, 0.0, 1.0));
                        }
                        cloud[k].lattice(mesh[k].cells, mesh[k].voxel, ob,
                                         vp.width * 0.9, 0.72);
                    }
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
                cloud[k].rangeRings(ego_k, ob.world_up, ob, f3, step, 5,
                                    cv::Scalar(78, 62, 50));
            }

            for (std::size_t i = 1; i < s.gt.size(); ++i) {
                cloud[k].line3(s.gt[i - 1].p, s.gt[i].p, ob, f3, C_GT, 1);
            }
            {
                const double frac = static_cast<double>(frame + 1) / nframes;
                const int upto = std::max(1, static_cast<int>(
                    frac * static_cast<double>(run.aligned.size())));
                for (int i = 1; i < upto; ++i) {
                    cloud[k].line3(run.aligned[static_cast<std::size_t>(i - 1)].translation(),
                                   run.aligned[static_cast<std::size_t>(i)].translation(),
                                   ob, f3, accent, 2);
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

                    // **없는 데이터는 그리지 않는다.**
                    //
                    // 의미 정보(상자/평면)는 내보내기 프레임에만 있다. 장면
                    // 내보내기는 stride 4 로 150 프레임까지만 돌았으므로
                    // walking_xyz 에서는 596 번 프레임이 마지막이다. 그런데
                    // "가장 가까운" 프레임을 고르는 규칙은 820 번에서도 596 을
                    // 골라 주고, 그러면 **7.5 초 전의 관측을 그때의 포즈로**
                    // 지금 화면에 그리게 된다. 그것이 지도 오른쪽에 따로 떠
                    // 있던 커다란 상자 무더기의 정체다.
                    //
                    // 오래된 관측을 지금 것처럼 그리는 것은 드리프트를 보여
                    // 주려던 화면이 스스로 없는 구조를 지어내는 일이다.
                    // 내보내기 간격의 두 배를 넘으면 아무 것도 그리지 않는다.
                    if (snap >= 0) {
                        // 내보내기 간격은 데이터에서 읽는다 - 도구의 기본값을
                        // 여기에 또 적어 두면 둘이 어긋날 때 조용히 틀린다.
                        int stride = 1 << 30, prev = -1;
                        for (const auto& b : s.boxes) {
                            if (prev >= 0 && b.frame > prev) {
                                stride = std::min(stride, b.frame - prev);
                            }
                            prev = std::max(prev, b.frame);
                        }
                        if (stride == (1 << 30)) stride = 1;
                        if (best > 2 * stride) snap = -1;
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
                        // 사람 골격은 자리를 크게 차지하므로 관측이 충분한
                        // 것만 그린다. 한두 번 본 검출이 전부 골격으로 서면
                        // 서로 겹쳐 엉킨 덩어리가 된다 - 실제로 그랬다.
                        const bool is_person = (m.cls == "person");
                        if (m.cls.empty() || m.count < (is_person ? 5 : 2)) continue;
                        // 레이어 격리. 2 는 차량만, 3 은 사람만, 4/5 는
                        // 구조물/지면 레이어이므로 물체를 그리지 않는다.
                        if (layer == 4 || layer == 5) continue;
                        if (layer == 3 && !is_person) continue;
                        if (layer == 2 && !(m.cls == "car" || m.cls == "truck" ||
                                            m.cls == "bus" || m.cls == "motorcycle" ||
                                            m.cls == "bicycle" || m.cls == "train")) continue;
                        const bool vehicle = (m.cls == "car" || m.cls == "truck" ||
                                              m.cls == "bus" || m.cls == "motorcycle" ||
                                              m.cls == "bicycle" || m.cls == "train");
                        const bool person = (m.cls == "person");

                        // **움직인다고 판정된 것은 지도의 일부가 아니다.**
                        // 지나온 자리에 남기지 않고, 지금 있는 자리에만 그린다.
                        // 그 자리에 쌓였던 복셀은 이미 지워졌다.
                        if (m.dynamic) {
                            // 최근에 못 본 동적 물체는 그리지 않는다. 마지막으로
                            // 본 자리에 세워 두면 그것이 바로 잔상이다.
                            if (frame - m.seen > 12) continue;
                            // 움직이는 것도 무엇인지 알아볼 수 있어야 한다.
                            // 이동 방향은 첫 관측에서 지금까지의 변위다.
                            const Eigen::Vector3d mv = m.cur_c - m.first_c;
                            if (vehicle) {
                                cloud[k].carModel(m.center(), m.size(),
                                                  mv.norm() > 0.2 ? mv : pan_head[k],
                                                  ob.world_up, ob, f3, C_WARN);
                            } else if (person) {
                                cloud[k].personSkeleton(m.center(), m.size(),
                                                        ob.world_up,
                                                        mv.norm() > 0.1 ? mv : pan_head[k],
                                                        ob, f3, C_WARN);
                            } else {
                                cloud[k].footprint(Eigen::Isometry3d::Identity(),
                                                   m.center(), m.size(),
                                                   ob.world_up, ob, f3, C_WARN, 2);
                            }
                            cv::Point at;
                            const Eigen::Vector3d top =
                                m.center() - ob.world_up * (m.size().maxCoeff() * 0.5 + 0.12);
                            if (cloud[k].project3(top, ob, f3, at)) {
                                at.x += vp.x; at.y += vp.y - 6;
                                if (at.x > vp.x && at.x < vp.x + vp.width - 90 &&
                                    at.y > vp.y + 12 && at.y < vp.y + vp.height) {
                                    text(canvas, m.cls + (person ? " moving (schematic)" : " moving"),
                                         at, T_LABEL, C_WARN, 1);
                                }
                            }
                            continue;
                        }

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
                        // **형상이 클래스를 말한다.** 이미 월드 좌표이므로
                        // 항등 변환으로 그린다.
                        if (vehicle) {
                            const Eigen::Vector3d mv = m.cur_c - m.first_c;
                            cloud[k].carModel(m.center(), m.size(),
                                              mv.norm() > 0.3 ? mv : pan_head[k],
                                              ob.world_up, ob, f3, col);
                        } else if (person) {
                            const Eigen::Vector3d mv2 = m.cur_c - m.first_c;
                            cloud[k].personSkeleton(m.center(), m.size(), ob.world_up,
                                                    mv2.norm() > 0.1 ? mv2 : pan_head[k],
                                                    ob, f3, col);
                        } else {
                            cloud[k].footprint(Eigen::Isometry3d::Identity(), m.center(),
                                               m.size(), ob.world_up, ob, f3, col, 1);
                        }
                    }

                    if (snap >= 0) {
                        for (const auto& q : s.planes) {
                            if (q.frame != snap || q.conf < 0.25) continue;
                            // extent 는 평면 위 점들의 **평균 산포** 다. 그대로
                            // 반지름으로 쓰면 실제 패치보다 훨씬 커 보인다.
                            //
                            // 게다가 planeQuad 의 네 꼭짓점은 c ± (a±b)·half 라
                            // 대각선이 2·√2·half 다. 실내 상한 1.2 m 는 한 변
                            // 3.4 m 짜리 사각형이 되고, 방 자체가 3 m 인 장면에서는
                            // 평면 열 개가 지도를 통째로 덮어 버린다 - 실제로
                            // walking_xyz 에서 점군도 물체도 보이지 않았다.
                            //
                            // 평면은 Tier 2 의 진단이지 화면의 주인공이 아니다.
                            // 장면 규모에 맞춰 줄이고 색도 배경 쪽으로 내린다.
                            const double half = std::clamp(q.extent, 0.15,
                                           s.dataset == "kitti" ? 5.0 : 0.45);
                            cloud[k].planeQuad(Tsnap, q.centroid, q.normal, half, ob, f3,
                                               cv::Scalar(74, 62, 52));
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
                            if (!(mx > 0.2) || mx > box_max) continue;
                            const bool vehicle = (b.cls == "car" || b.cls == "truck" ||
                                                  b.cls == "bus" || b.cls == "motorcycle" ||
                                                  b.cls == "bicycle" || b.cls == "train");
                            const bool person = (b.cls == "person");
                            const cv::Scalar col = vehicle ? accent
                                                 : (person ? C_WARN : C_INK3);

                            cloud[k].box3(Tsnap, b.center, b.size, ob, f3, col, 1);
                            // 지면 발자국. 항공뷰에서는 이것이 물체의 본체다.
                            cloud[k].footprint(Tsnap, b.center, b.size, ob.world_up,
                                               ob, f3, col, 2);

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
                                    cloud[k].arrow3(e, e + (e - a) * 3.0, ob, f3, col, 2);
                                }
                            }

                            // 클래스 라벨. 무엇을 잡았는지 화면에서 읽혀야
                            // "물체를 인식한다" 는 주장이 확인 가능해진다.
                            cv::Point at;
                            const Eigen::Vector3d top =
                                Tsnap * (b.center - Eigen::Vector3d(0, 0.5 * b.size.y(), 0));
                            if (cloud[k].project3(top, ob, f3, at)) {
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
                    const double sc3 = std::max(0.05, ob.dist * 0.03);
                    const Eigen::Vector3d o = T.translation();
                    const Eigen::Vector3d c[4] = {
                        T * Eigen::Vector3d(-sc3, -sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d( sc3, -sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d( sc3,  sc3 * 0.75, sc3 * 1.4),
                        T * Eigen::Vector3d(-sc3,  sc3 * 0.75, sc3 * 1.4)};
                    for (int e = 0; e < 4; ++e) {
                        cloud[k].line3(o, c[e], ob, f3, C_INK, 1);
                        cloud[k].line3(c[e], c[(e + 1) % 4], ob, f3, C_INK, 1);
                    }

                    // **자차도 형상으로 그린다.**
                    //
                    // 라이다 시각화가 한가운데에 차를 놓는 것은 장식이 아니다 -
                    // 그 차가 척도이고 방향이다. 프러스텀 하나만 있으면 지도의
                    // 크기도 자차가 어느 쪽을 향하는지도 화면에서 읽히지 않는다.
                    // 1 인칭에서는 눈이 차 안에 있으므로 그리지 않는다.
                    // 카메라가 모델보다 가까우면 그리지 않는다. 1.7 m 짜리
                    // 사람을 1.75 m 앞에서 그리면 화면이 통째로 그 사람이 되고,
                    // 지도를 보여 주려던 자리에 자차만 남는다 - TUM 3 인칭에서
                    // 실제로 흰 기둥 하나가 패널을 가렸다.
                    // **KITTI 에서만 자차를 그린다.**
                    //
                    // KITTI 는 차량에 실린 카메라이므로 그 자리에 차가 있는
                    // 것이 사실이고, 차체가 곧 척도와 진행 방향이 된다.
                    //
                    // TUM 은 손에 든 카메라다. 거기에 사람 몸을 그리면 관측된
                    // 적 없는 것을 지어내는 것이고, 3 m 짜리 방에서 1.7 m
                    // 짜리 몸은 화면을 통째로 가린다 - 실제로 그렇게 나왔다.
                    // 카메라는 프러스텀이 이미 나타내고 있다.
                    if (cam_mode != 0 && s.dataset == "kitti" && ob.dist > 12.0) {
                        const Eigen::Vector3d fw = (pan_head[k].squaredNorm() > 1e-12)
                                                 ? pan_head[k]
                                                 : (T.rotation() * Eigen::Vector3d::UnitZ());
                        cloud[k].carModel(o + ob.world_up * 0.7,
                                          Eigen::Vector3d(1.8, 1.5, 4.2),
                                          fw, ob.world_up, ob, f3, C_INK);
                    }
                }
            }
            // EDL 은 모든 3D 패스가 끝난 뒤에 건다. 깊이 버퍼가 다
            // 채워져 있어야 이웃 비교가 의미를 갖는다.
            if (show_edl) cloud[k].eyeDomeLighting(0.4, 1.4);
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

                    // 장면 라벨 범례. **개수를 같이 적는다** - 색만 있으면
                    // 그 클래스가 실제로 나왔는지 화면에서 알 수 없고,
                    // 0 개인 범례는 있지도 않은 능력을 광고한다.
                    if (show_stuff && !stuff[k].empty()) {
                        int n[6] = {0, 0, 0, 0, 0, 0};
                        for (const auto& [ck, c] : stuff[k]) {
                            n[static_cast<int>(c.cls)]++;
                        }
                        label(canvas, "scene labels from map geometry (not the detector)",
                              {vp.x + 14, vp.y + 126}, C_INK3, T_MICRO, 1);
                        // **무엇이 일어났는가.** 포즈만 내놓는 것은 기존 SLAM
                        // 이 하는 일이고, 그 위에 얹히는 것이 이 한 줄이다.
                        {
                            std::ostringstream mo;
                            mo << "motion: " << (mot_kind[k].empty() ? "-" : mot_kind[k])
                               << "   yaw " << static_cast<int>(mot_yaw[k] * 180.0 / kPiV)
                               << " deg";
                            if (mot_turns[k] != 0) {
                                mo << "   full turns " << mot_turns[k];
                            }
                            label(canvas, mo.str(), {vp.x + 14, vp.y + 180},
                                  (mot_turns[k] != 0) ? C_WARN : C_INK2, T_MICRO, 1);
                        }
                        // **예측은 채점 결과와 함께만 말한다.** 적중률 없이
                        // "100 m 앞을 예측한다" 만 적으면 그림이 주장을 대신하게
                        // 된다 - 이 저장소가 반복해 기록한 실패다.
                        if (show_pred) {
                            const double pr = pscore[k].precision();
                            const double rc = pscore[k].recall();
                            std::ostringstream o;
                            o << "lookahead 100 m: extrapolated, not generated  |  "
                              << pred[k].size() << " open  |  hit " << pscore[k].hit
                              << "  wrong " << pscore[k].miss
                              << "  missed " << pscore[k].missed << "  |  ";
                            if (std::isfinite(pr)) {
                                o << "precision " << static_cast<int>(pr * 100) << "%";
                            } else {
                                o << "precision --";
                            }
                            if (std::isfinite(rc)) {
                                o << "  recall " << static_cast<int>(rc * 100) << "%";
                            }
                            o << "  |  corridor " << std::fixed << std::setprecision(1)
                              << pscore[k].lateral << " m";
                            label(canvas, o.str(), {vp.x + 14, vp.y + 162},
                                  C_INK2, T_MICRO, 1);
                        }
                        int lx = vp.x + 14;
                        for (const Stuff sc : {Stuff::Ground, Stuff::Building,
                                               Stuff::Fence, Stuff::Vegetation,
                                               Stuff::Pole}) {
                            const int cnt = n[static_cast<int>(sc)];
                            if (cnt == 0) continue;
                            fill(canvas, {lx, vp.y + 136, 10, 10}, stuffColor(sc));
                            const std::string t = std::string(stuffName(sc)) + " " +
                                                  std::to_string(cnt);
                            label(canvas, t, {lx + 15, vp.y + 145}, C_INK2, T_MICRO, 1);
                            lx += 22 + textW(t, T_MICRO, 1);
                        }
                    }
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
            // 어느 레이어를 보고 있는지 화면에 적는다. 격리된 화면과
            // 아무 것도 없는 화면은 그림만으로는 구분되지 않는다.
            static const char* kLayer[6] = {"ALL LAYERS", "MAP ONLY", "VEHICLES ONLY",
                                            "PEOPLE ONLY", "STRUCTURE ONLY",
                                            "GROUND ONLY"};
            const char* mode = (cam_mode == 0) ? "FIRST PERSON"
                             : (cam_mode == 1) ? "THIRD PERSON"
                             : (cam_mode == 2) ? "MAP" : "BIRD'S EYE";
            label(canvas, "camera / layer", {W_MODE_X, fy + 40}, C_INK3);
            text(canvas, std::string(mode) + "   ", {W_MODE_X, fy + 62}, T_BODY, C_INK2);
            text(canvas, kLayer[layer],
                 {W_MODE_X + textW(std::string(mode) + "   ", T_BODY, 1), fy + 62},
                 T_BODY, (layer == 0) ? C_INK3 : C_WARN);
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
            text(canvas, "A/D orbit  W/S zoom  V camera  T layer  C cubes  L labels  O lookahead  R restart  F shot  Q quit",
                 {hx, fy + 126}, T_BODY, C_INK2);
        }
        {
            const std::string src =
                "ATE / RPE / ms-per-frame read from results/bench/viewer.tsv "
                "(computed by python/tools/bench_run.py) - not recomputed here";
            text(canvas, src, {WIN_W - PAD - textW(src, T_MICRO, 1), WIN_H - 16},
                 T_MICRO, C_INK3);
        }

        // 예측 채점은 **여러 프레임을 지나가야** 일어난다. 자차가 예측한
        // 자리를 통과해야 거기 무엇이 있었는지 볼 수 있기 때문이다. 한 장만
        // 찍는 헤드리스로는 영원히 0 - 0 이 나오고, 그것은 "예측이 틀렸다" 가
        // 아니라 "아직 안 가 봤다" 다. 그래서 재생하며 도는 모드를 따로 둔다.
        if (headless && frame < shot_frame) {
            ++frame;
            continue;
        }

        if (headless) {
            // 왜 비었는지 사후에 물어볼 수 있어야 한다. 화면이 검은 것과
            // 데이터가 없는 것은 그림만 봐서는 구분되지 않는다.
            for (int k = 0; k < 2; ++k) {
                std::cerr << "  패널 " << k << ": 복셀 " << acc[k].cells.size()
                          << " (" << acc[k].voxel << " m, coarsen " << acc[k].grown
                          << ", 원거리 정정 " << acc[k].revised << ")"
                          << ", 라벨 " << [&] {
                                 int n[6] = {0,0,0,0,0,0};
                                 for (const auto& [ck, c] : stuff[k]) {
                                     n[static_cast<int>(c.cls)]++;
                                 }
                                 std::ostringstream o;
                                 o << "지면 " << n[1] << " 건물 " << n[2]
                                   << " 담장 " << n[3] << " 나무 " << n[4]
                                   << " 기둥 " << n[5] << " 미상 " << n[0];
                                 return o.str();
                             }()
                          << [&] {
                                 // 임계값을 눈대중으로 정하지 않기 위한 덤프.
                                 // 분포를 보고 빈 구간에서 고르는 것이
                                 // 25.22 의 발산 문턱을 정한 방법이다.
                                 if (k != 1 || dump_feat.empty()) return std::string();
                                 std::ofstream o(dump_feat);
                                 o << "top_m\tplanarity\tlinearity\tscatter\tvert\tn\n";
                                 for (const auto& [ck, c] : stuff[k]) {
                                     if (c.n < 6) continue;
                                     o << (c.topBin() + 1) * kBinH << '\t' << c.planarity
                                       << '\t' << c.linearity << '\t' << c.scatter
                                       << '\t' << c.vert << '\t' << c.n << '\n';
                                 }
                                 return "  특징 덤프: " + dump_feat + "\n";
                             }()
                          << [&] {
                                 // 하늘에 무엇이 남아 있는지 **재서** 말한다.
                                 // 자차 높이 기준 상대 높이의 분포다.
                                 if (acc[k].cells.empty()) return std::string();
                                 std::vector<float> hs;
                                 hs.reserve(acc[k].cells.size());
                                 const Eigen::Vector3f upf = orb.world_up.cast<float>();
                                 const float e0 = static_cast<float>(ego_pos.dot(orb.world_up));
                                 for (const auto& [ck, v] : acc[k].cells) {
                                     hs.push_back(v.p.dot(upf) - e0);
                                 }
                                 std::sort(hs.begin(), hs.end());
                                 auto q = [&](double t) {
                                     return hs[std::min(hs.size() - 1,
                                         static_cast<std::size_t>(t * hs.size()))];
                                 };
                                 std::ostringstream o;
                                 o << "\n           높이(자차기준 m): 최저 " << std::fixed
                                   << std::setprecision(1) << hs.front()
                                   << "  p50 " << q(0.50) << "  p90 " << q(0.90)
                                   << "  p99 " << q(0.99) << "  최고 " << hs.back()
                                   << "  (8 m 초과 " << std::count_if(hs.begin(), hs.end(),
                                        [](float x) { return x > 8.0f; }) << " 개)";
                                 return o.str();
                             }()
                          << ", 예측 " << [&] {
                                 std::ostringstream o;
                                 o << pred[k].size() << " (적중 " << pscore[k].hit
                                   << " 빗나감 " << pscore[k].miss
                                   << " 놓침 " << pscore[k].missed;
                                 const double pr = pscore[k].precision();
                                 const double rc = pscore[k].recall();
                                 if (std::isfinite(pr)) {
                                     o << ", 정밀도 " << static_cast<int>(pr * 100) << "%";
                                 }
                                 if (std::isfinite(rc)) {
                                     o << " 재현율 " << static_cast<int>(rc * 100) << "%";
                                 }
                                 o << ", 회랑 " << std::fixed << std::setprecision(1)
                                   << pscore[k].lateral << " m)";
                                 return o.str();
                             }()
                          << "\n           시간(ms): 점군 " << std::fixed
                          << std::setprecision(1) << prof_cloud[k]
                          << "  라벨 " << prof_label[k]
                          << "  벡터 " << prof_vec[k]
                          << "  예측 " << prof_pred[k]
                          << "  예측그리기 " << prof_pdraw[k]
                          << "  프레임전체 " << prof_frame
                          << [&] {
                                 // 주차 차량이 실제로 몇 개나 기억되는가.
                                 // "물체 127 개" 로는 그 안에 차가 몇 대인지,
                                 // 몇 번이나 본 것인지 알 수 없다.
                                 if (mem[k].empty()) return std::string();
                                 std::map<std::string, std::array<int, 3>> by;
                                 for (const auto& m : mem[k]) {
                                     auto& e = by[m.cls];
                                     e[0]++;                       // 전체
                                     if (m.dynamic) e[1]++;        // 동적
                                     if (m.count >= 5) e[2]++;     // 충분히 본 것
                                 }
                                 std::ostringstream o;
                                 o << "\n           기억 물체 내역:";
                                 for (const auto& [cls, e] : by) {
                                     o << "  " << cls << " " << e[0]
                                       << "(정지 " << (e[0] - e[1])
                                       << ", 5회+ " << e[2] << ")";
                                 }
                                 return o.str();
                             }()
                          << ", 기억물체 " << mem[k].size()
                          << " (동적 " << std::count_if(mem[k].begin(), mem[k].end(),
                                          [](const MemoryObject& m) { return m.dynamic; })
                          << ")"
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

        prof_frame = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_frame0).count();
        cv::imshow(win, canvas);
        const int key = cv::waitKey(playing ? 1 : 0);

        if (key == 'q' || key == 'Q' || key == 27) break;
        else if (key == ' ') playing = !playing;
        else if (key == 'r' || key == 'R') { frame = 0; acc_frame[0] = acc_frame[1] = -1;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            pred[0].clear(); pred[1].clear();
            road[0].clear(); road[1].clear();
            surf[0].clear(); surf[1].clear();
            surf_at[0].setConstant(1e9); surf_at[1].setConstant(1e9);
            pscore[0].clear(); pscore[1].clear();
            mot_yaw[0] = mot_yaw[1] = 0.0; mot_turns[0] = mot_turns[1] = 0;
            mot_at[0] = mot_at[1] = -1; mot_kind[0] = mot_kind[1] = "";
            pan_head[0].setZero(); pan_head[1].setZero();
            mesh[0].clear(); mesh[1].clear(); }
        else if (key == 'v' || key == 'V') { cam_mode = (cam_mode + 1) % 4; user_zoomed = false; }
        else if (key == 'm' || key == 'M') show_mesh = !show_mesh;
        else if (key == 'l' || key == 'L') show_stuff = !show_stuff;
        // 예측 토글은 O 다. P 는 **이전 시퀀스** 로 이미 잡혀 있었고, 여기에
        // 토글을 얹으니 아래 분기가 영영 안 닿아 시퀀스 되감기가 통째로
        // 죽었다 - if/else 사슬에서 같은 키를 두 번 쓰면 뒤엣것은 없는 코드다.
        else if (key == 'o' || key == 'O') show_pred = !show_pred;
        else if (key == 'c' || key == 'C') show_cubes = !show_cubes;
        else if (key == 't' || key == 'T') layer = (layer + 1) % 6;
        else if (key == 'e' || key == 'E') show_edl = !show_edl;
        else if (key == 'g' || key == 'G') {
            show_surf = !show_surf;
            surf_at[0].setConstant(1e9); surf_at[1].setConstant(1e9);
        }
        else if (key == 'n' || key == 'N') {
            si = (si + 1) % static_cast<int>(seqs.size());
            frame = 0; acc_frame[0] = acc_frame[1] = -1; orb.dist = 0.0; user_zoomed = false;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            pred[0].clear(); pred[1].clear();
            road[0].clear(); road[1].clear();
            surf[0].clear(); surf[1].clear();
            surf_at[0].setConstant(1e9); surf_at[1].setConstant(1e9);
            pscore[0].clear(); pscore[1].clear();
            mot_yaw[0] = mot_yaw[1] = 0.0; mot_turns[0] = mot_turns[1] = 0;
            mot_at[0] = mot_at[1] = -1; mot_kind[0] = mot_kind[1] = "";
            pan_head[0].setZero(); pan_head[1].setZero();
            mesh[0].clear(); mesh[1].clear();
        } else if (key == 'p' || key == 'P') {
            si = (si + static_cast<int>(seqs.size()) - 1) % static_cast<int>(seqs.size());
            frame = 0; acc_frame[0] = acc_frame[1] = -1; orb.dist = 0.0; user_zoomed = false;
            mem[0].clear(); mem[1].clear(); acc[0].clear(); acc[1].clear();
            pred[0].clear(); pred[1].clear();
            road[0].clear(); road[1].clear();
            surf[0].clear(); surf[1].clear();
            surf_at[0].setConstant(1e9); surf_at[1].setConstant(1e9);
            pscore[0].clear(); pscore[1].clear();
            mot_yaw[0] = mot_yaw[1] = 0.0; mot_turns[0] = mot_turns[1] = 0;
            mot_at[0] = mot_at[1] = -1; mot_kind[0] = mot_kind[1] = "";
            pan_head[0].setZero(); pan_head[1].setZero();
            mesh[0].clear(); mesh[1].clear();
        } else if (key == '1') {
            pick[0] = (pick[0] + 1) % static_cast<int>(s.systems.size());
            acc_frame[0] = -1; mem[0].clear(); acc[0].clear(); mesh[0].clear();
            pred[0].clear(); pscore[0].clear();
        } else if (key == '2') {
            pick[1] = (pick[1] + 1) % static_cast<int>(s.systems.size());
            acc_frame[1] = -1; mem[1].clear(); acc[1].clear(); mesh[1].clear();
            pred[1].clear(); pscore[1].clear();
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
