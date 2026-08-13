// WorldVision-SLAM — 360° 등장방형(equirectangular) 영상 -> 원근 뷰 TUM 배치
//
// 왜 이 도구가 필요한가
// ---------------------
// 파이프라인 전체(StereoDepth, DirectAligner, tum_odometry, scene_export, ...)는
// **핀홀** 을 가정한다. u = fx*X/Z + cx 가 성립하고 직선이 직선으로 온다는
// 가정이다. 360° 카메라가 내는 등장방형 영상은 화소 좌표가 (경도, 위도)라서
// 그 가정을 만족하지 않는다. 어떤 (fx, fy, cx, cy) 로도 근사되지 않는다 -
// 중심 근처에서만 그럴듯하고 가장자리로 갈수록 발산한다.
//
// 그런데 등장방형 영상을 그대로 넣어도 **아무것도 실패하지 않는다**. 영상은
// 그냥 8비트 배열이고 calib.txt 는 아무 숫자나 받는다. 특징점은 잡히고 정합도
// 되고 궤적도 나온다. 틀린 궤적이 나온다. 이 저장소가 반복해서 잡아 온 실패
// 양상이고(dataset_calib.hpp 머리말의 freiburg 경로 추측이 같은 종류다),
// 그래서 변환을 하류에 맡기지 않고 여기서 한 번에 끝낸다.
//
// 만드는 것:
//   <out>/rgb/<이름>.png        잘라낸 원근 뷰
//   <out>/rgb.txt               타임스탬프 인덱스
//   <out>/calib.txt             유도된 fx fy cx cy width height (+ 뷰 파라미터)
//   <out>/depth/, depth.txt     --right 로 360 스테레오 쌍을 준 경우에만
//
// **깊이는 기본적으로 만들지 않는다.** 등장방형 한 장에서는 깊이가 나오지
// 않기 때문이다. 스케일을 모르는 단안 영상이고, 잘라내기는 정보를 더하지
// 않는다 - 광선의 방향을 다시 매길 뿐이다. 그래서 이 도구의 기본 출력은
// rgb 전용 배치이고, wme_tum_odometry / wme_tum_fusion 은 그것을 못 먹는다.
// 끝에서 그 사실을 크게 찍는다. 조용히 0 깊이를 채워 넣으면 하류가 "깊이가
// 결측인 프레임" 으로 오해하고 계속 굴러간다.
//
// 깊이가 나오는 유일한 경로는 **360 스테레오 리그** 다 (--right + --baseline).
// 그 경우에도 아무 방향이나 되는 게 아니다. 아래 checkRectified() 의 유도 참조 -
// 베이스라인이 파노라마 X 축이면 정렬 스테레오가 되는 것은 yaw 가 0 또는 180
// 도인 뷰뿐이고, yaw 90 도에서는 베이스라인이 광축과 나란해져 시차 자체가
// 깊이 정보를 잃는다. 이건 구현 한계가 아니라 기하다.
//
// 좌표/화소 규약과 내부파라미터 유도는 python/wme/reference/equirect.py 머리말에
// 한 번만 적어 두었다. 그 파일이 이 코드의 numpy 오라클이고,
// python/tools/equirect_validate.py 가 둘을 같은 입력으로 돌려 비교한다.
//
// 사용:
//   wme_equirect_convert --in <등장방형 파일 또는 폴더> --out <출력경로>
//                        [--yaw 0] [--pitch 0] [--hfov 90]
//                        [--width 1024] [--height 768]
//                        [--interp cubic] [--stride 1] [--max-frames 0]
//                        [--fps 10] [--times <파일>]
//                        [--right <오른쪽 등장방형>] [--baseline <m>]
//                        [--baseline-yaw 0] [--min-depth 4.0]

#include "wme/perception/StereoDepth.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
// std::tolower 는 <cctype> 에 있다. MSVC 는 다른 헤더를 타고 딸려 들어와
// 없어도 컴파일되지만 GCC 는 아니다 - 리눅스 CI 에서만 터지는 자리다.
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

double deg2rad(double d) { return d * kPi / 180.0; }
double rad2deg(double r) { return r * 180.0 / kPi; }

// 잘라낸 원근 뷰의 핀홀 내부파라미터. 유도는 reference/equirect.py 머리말.
//
//   영상은 화소 인덱스 -0.5 부터 W-0.5 까지, 즉 폭 W 화소를 덮는다.
//   시야를 좌우 대칭으로 두면 cx = (W-1)/2 이고 중심에서 경계까지가 W/2 화소다.
//   그 경계가 광축과 이루는 각이 hfov/2 이므로
//       tan(hfov/2) = (W/2) / fx   =>   fx = (W/2) / tan(hfov/2)
//   세로도 같다. hfov 만 주면 정사각 화소(fy = fx)를 택하고 vfov 는 유도한다.
struct PinholeView {
    int    width{0}, height{0};
    double fx{0}, fy{0}, cx{0}, cy{0};
    double yaw_deg{0}, pitch_deg{0};
    double hfov_deg{0}, vfov_deg{0};
};

bool deriveIntrinsics(int W, int H, double hfov_deg, double yaw_deg, double pitch_deg,
                      PinholeView& out) {
    if (W < 2 || H < 2) {
        std::cerr << "출력 해상도가 너무 작다: " << W << "x" << H << "\n";
        return false;
    }
    // 180 도에서 tan 이 발산한다. 핀홀은 반구 이상을 담을 수 없다 - 구현
    // 한계가 아니라 투영의 성질이다. 넓게 보려면 뷰를 여러 장 잘라야 한다.
    if (!(hfov_deg > 0.0 && hfov_deg < 180.0)) {
        std::cerr << "hfov 는 (0, 180) 도여야 한다: " << hfov_deg << "\n";
        return false;
    }
    out.width = W;
    out.height = H;
    out.fx = (W / 2.0) / std::tan(deg2rad(hfov_deg) / 2.0);
    out.fy = out.fx;                       // 정사각 화소
    out.cx = (W - 1) / 2.0;
    out.cy = (H - 1) / 2.0;
    out.yaw_deg = yaw_deg;
    out.pitch_deg = pitch_deg;
    out.hfov_deg = hfov_deg;
    out.vfov_deg = rad2deg(2.0 * std::atan((H / 2.0) / out.fy));
    return true;
}

// R_pano_cam = R_y(yaw) * R_x(pitch), roll = 0. 행 우선 3x3.
// 곱 순서 근거는 reference/equirect.py 머리말 - 이 순서라야 yaw 로 돌린 뒤에도
// 영상의 세로축이 자오선 안에 남는다(roll 이 섞이지 않는다).
void viewRotation(double yaw_deg, double pitch_deg, double R[9]) {
    const double y = deg2rad(yaw_deg), p = deg2rad(pitch_deg);
    const double cy = std::cos(y), sy = std::sin(y);
    const double cp = std::cos(p), sp = std::sin(p);
    // Ry = [[cy,0,sy],[0,1,0],[-sy,0,cy]],  Rx = [[1,0,0],[0,cp,-sp],[0,sp,cp]]
    R[0] = cy;  R[1] = sy * sp;  R[2] = sy * cp;
    R[3] = 0.0; R[4] = cp;       R[5] = -sp;
    R[6] = -sy; R[7] = cy * sp;  R[8] = cy * cp;
}

// 가로 감기 여유. 등장방형은 경도 방향으로 이어져 있어서 lam = -pi 근처 화소의
// 보간 이웃이 반대쪽 끝에 있다. 그냥 복제 경계로 두면 이음매에 세로줄이 남는다.
// INTER_CUBIC 이 한쪽 2 화소, INTER_LANCZOS4 가 4 화소를 보므로 그 두 배인 8.
constexpr int kWrapPad = 8;

// 원근 뷰의 각 화소가 등장방형의 어느 화소에서 오는지. map 값은 **화소 인덱스**
// 좌표이고, map_x 에는 kWrapPad 가 이미 더해져 있다 (패딩된 원본에 바로 쓴다).
void buildMaps(const PinholeView& v, int src_w, int src_h,
               cv::Mat& map_x, cv::Mat& map_y) {
    double R[9];
    viewRotation(v.yaw_deg, v.pitch_deg, R);

    map_x.create(v.height, v.width, CV_32F);
    map_y.create(v.height, v.width, CV_32F);

    for (int j = 0; j < v.height; ++j) {
        float* mx = map_x.ptr<float>(j);
        float* my = map_y.ptr<float>(j);
        const double yn = (j - v.cy) / v.fy;
        for (int i = 0; i < v.width; ++i) {
            const double xn = (i - v.cx) / v.fx;
            // 카메라 광선 -> 정규화 -> 파노라마계
            const double inv = 1.0 / std::sqrt(xn * xn + yn * yn + 1.0);
            const double dx = xn * inv, dy = yn * inv, dz = inv;
            const double px = R[0] * dx + R[1] * dy + R[2] * dz;
            const double py = R[3] * dx + R[4] * dy + R[5] * dz;
            const double pz = R[6] * dx + R[7] * dy + R[8] * dz;

            const double lam = std::atan2(px, pz);
            // asin 인자를 자른다. 정규화 뒤라도 반올림으로 |py| 가 1 을 아주
            // 조금 넘을 수 있고, 그러면 NaN 이 remap 까지 조용히 흘러간다.
            const double phi = std::asin(std::min(1.0, std::max(-1.0, -py)));

            mx[i] = static_cast<float>((lam + kPi) / (2.0 * kPi) * src_w - 0.5
                                       + kWrapPad);
            my[i] = static_cast<float>((kPi / 2.0 - phi) / kPi * src_h - 0.5);
        }
    }
}

// 가로만 감는다. cv::copyMakeBorder 는 BORDER_WRAP 을 받지 않으므로(예외를
// 던진다) 직접 이어 붙인다. 세로는 remap 의 BORDER_REPLICATE 에 맡긴다 -
// 광선이 극 바깥으로 나가지 않아 복제로 충분하고, 극을 감으면(경도 반대편으로
// 넘어가는 것이 옳다) 오히려 틀린 화소를 가져온다.
cv::Mat wrapPad(const cv::Mat& src) {
    cv::Mat out(src.rows, src.cols + 2 * kWrapPad, src.type());
    src(cv::Rect(src.cols - kWrapPad, 0, kWrapPad, src.rows))
        .copyTo(out(cv::Rect(0, 0, kWrapPad, src.rows)));
    src.copyTo(out(cv::Rect(kWrapPad, 0, src.cols, src.rows)));
    src(cv::Rect(0, 0, kWrapPad, src.rows))
        .copyTo(out(cv::Rect(src.cols + kWrapPad, 0, kWrapPad, src.rows)));
    return out;
}

struct Frame {
    double      stamp{0.0};
    fs::path    path;
    std::string name;   // 확장자 없는 파일 이름 (출력 이름으로 재사용)
};

bool isImage(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" ||
           e == ".tif" || e == ".tiff" || e == ".ppm";
}

// 입력은 단일 파일이거나, 영상이 든 폴더이거나, rgb/ + rgb.txt 를 가진 TUM 배치다.
// 셋 다 받는다 - 360 카메라의 출력 형태가 제각각이라 하나만 받으면 그 변환을
// 사용자가 손으로 하게 되고 그때 순서가 어긋난다.
std::vector<Frame> listInput(const fs::path& in, std::string& how) {
    std::vector<Frame> out;
    if (fs::is_regular_file(in)) {
        how = "단일 파일";
        out.push_back({0.0, in, in.stem().string()});
        return out;
    }
    if (!fs::is_directory(in)) return out;

    fs::path img_dir = in;
    if (fs::exists(in / "rgb") && fs::is_directory(in / "rgb")) img_dir = in / "rgb";

    for (const auto& e : fs::directory_iterator(img_dir)) {
        if (e.is_regular_file() && isImage(e.path()))
            out.push_back({0.0, e.path(), e.path().stem().string()});
    }
    std::sort(out.begin(), out.end(),
              [](const Frame& a, const Frame& b) { return a.name < b.name; });
    how = (img_dir == in) ? "폴더" : "TUM 배치의 rgb/";
    return out;
}

// 타임스탬프. 파일이 있으면 파일이 이긴다 (dataset_calib.hpp 와 같은 원칙).
// 없으면 --fps 로 만들고, 그것이 합성값이라는 것을 반드시 찍는다.
bool assignStamps(std::vector<Frame>& frames, const fs::path& in,
                  const std::string& times_file, double fps) {
    std::vector<double> t;
    fs::path src;
    if (!times_file.empty())                       src = times_file;
    else if (fs::exists(in / "rgb.txt"))           src = in / "rgb.txt";
    else if (fs::exists(in / "times.txt"))         src = in / "times.txt";

    if (!src.empty()) {
        std::ifstream f(src);
        if (!f) { std::cerr << "타임스탬프 파일을 열 수 없다: " << src << "\n"; return false; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            double v;
            if (ss >> v) t.push_back(v);   // "t 파일명" 도 "t" 도 첫 칸이 시각
        }
        if (t.size() < frames.size()) {
            std::cerr << "타임스탬프가 모자란다: " << t.size() << " 개, 프레임 "
                      << frames.size() << " 개 (" << src << ")\n";
            return false;
        }
        for (std::size_t i = 0; i < frames.size(); ++i) frames[i].stamp = t[i];
        std::cout << "타임스탬프: " << src.string() << "\n";
        return true;
    }

    if (!(fps > 0.0)) { std::cerr << "--fps 는 양수여야 한다: " << fps << "\n"; return false; }
    for (std::size_t i = 0; i < frames.size(); ++i)
        frames[i].stamp = static_cast<double>(i) / fps;
    std::cout << "타임스탬프: 파일이 없어 --fps " << fps
              << " 로 만든 **합성값** 이다. 실제 촬영 간격이 아니다.\n";
    return true;
}

// 360 스테레오 리그에서 이 뷰가 정렬(rectified) 스테레오인가.
//
// **베이스라인이 파노라마의 어느 방위를 향하는지는 데이터가 정한다.**
//
// 아래 유도는 원래 그것을 X 축(방위 0 도)이라고 **가정** 하고 있었다. 그런데
// 그 가정은 리그마다 다르고, 틀리면 이 검사가 정확히 거꾸로 답한다 - 실측:
// TartanAir V2 의 lcam/rcam 등장방형 쌍은 베이스라인이 90 도 어긋나 있어서
// 쓸 수 있는 뷰가 yaw 90/270 이고 못 쓰는 뷰가 yaw 0/180 인데, 가정을 박아
// 둔 코드는 **yaw 0 을 통과시키고 yaw 90 을 거부했다.** 시키는 대로 하면
// 시차가 깊이를 아예 담지 않는 뷰에서 SGBM 을 돌려 놓고 성공을 보고한다.
// 조용한 그럴듯한 오답이고, 이 저장소가 결함으로 규정한 바로 그것이다.
//
// 그래서 방위를 --baseline-yaw 로 받는다. 기본값은 0 도(옛 가정)지만 이제는
// 가정이 화면에 찍히므로, 틀린 리그를 쓰면 답이 조용히 틀리는 대신 눈에 띈다.
// 모르면 재는 방법은 아래 유도가 그대로 알려 준다 - dv 가 0 이 되는 yaw 를
// 찾으면 그것이 베이스라인 방위다.
//
// 베이스라인이 방위 a 를 향하면 t = B (cos a, 0, sin a) 이고, 아래 유도의
// yaw 는 (yaw - a) 로 바뀐다. a = 0 이면 원래 식과 같다.
//
// 베이스라인이 파노라마 X 축, t = (B, 0, 0) 이라 하자. 카메라계로 옮기면
//   t_cam = R^T t = ( B cos(yaw), B sin(pitch) sin(yaw), B cos(pitch) sin(yaw) )
// 정적 점의 두 영상 사이 이동은 t_cam = (tx, ty, tz) 에 대해
//   du = -(fx/Z) tx + (u-cx) tz/Z
//   dv = -(fy/Z) ty + (v-cy) tz/Z
// 이다. SGBM 은 dv = 0 (수평 epipolar) 을 전제하므로 세로 성분이 문제다:
//   dv = (B sin(yaw) / Z) * ( -fy sin(pitch) + (v-cy) cos(pitch) )
// yaw = 0 또는 180 도면 sin(yaw) = 0 이라 dv 가 정확히 0 이다. pitch 는
// 베이스라인 축을 도는 회전이라 아무 값이나 된다.
//
// 그 밖의 yaw 에서는 dv 가 0 이 아니고, 얼마나 나쁜지는 장면에 달렸다 -
// 가장 가까운 거리 Z_min 과 영상 크기가 정한다. 최악은 |v-cy| = H/2 에서
//   |dv|max = (B |sin yaw| / Z_min) * ( fy |sin pitch| + (H/2) cos pitch )
// 이 값이 정합의 허용 오차(SGBM 은 같은 행에서만 찾으므로 반 화소)를 넘으면
// 시차는 깊이가 아니라 잡음이다. 임의 상수를 박지 않고 여기서 계산한다.
constexpr double kMaxVerticalDisparityPx = 0.5;

bool checkRectified(const PinholeView& v, double baseline_m, double min_depth_m,
                    double baseline_yaw_deg, double& dv_max_px) {
    const double rel = v.yaw_deg - baseline_yaw_deg;
    const double s = std::abs(std::sin(deg2rad(rel)));
    const double sp = std::abs(std::sin(deg2rad(v.pitch_deg)));
    const double cp = std::abs(std::cos(deg2rad(v.pitch_deg)));
    dv_max_px = (baseline_m * s / min_depth_m) * (v.fy * sp + (v.height / 2.0) * cp);
    return dv_max_px <= kMaxVerticalDisparityPx;
}

std::string pad6(int i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d", i);
    return buf;
}

int interpFlag(const std::string& s) {
    if (s == "nearest")  return cv::INTER_NEAREST;
    if (s == "linear")   return cv::INTER_LINEAR;
    if (s == "cubic")    return cv::INTER_CUBIC;
    if (s == "lanczos4") return cv::INTER_LANCZOS4;
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string in_path, out_path, right_path, times_file;
    std::string interp = "cubic";
    double yaw = 0.0, pitch = 0.0, hfov = 90.0;
    int width = 1024, height = 768;
    int stride = 1, max_frames = 0;
    double fps = 10.0;
    double baseline = 0.0;
    // KITTI 도로 장면의 최근접 거리를 기본으로 쓴다 (kitti_convert.cpp 와 같은
    // 근거). 이 값이 시차 탐색 범위와 위 정렬 판정을 동시에 정한다.
    double min_depth = 4.0;
    // 베이스라인이 향하는 파노라마 방위(도). 기본 0 은 옛 가정이고, 리그마다
    // 다르다 - checkRectified 주석에 왜 이것이 조용한 오답을 만들었는지 있다.
    double baseline_yaw = 0.0;
    bool   baseline_yaw_given = false;

    // 인자는 전부 --키 값 쌍이다. 홀수면 마지막 하나가 짝 없이 남았다는 뜻이고,
    // i+1 < argc 조건은 그것을 조용히 버린다 - 오타난 플래그가 기본값으로
    // 굴러가는 자리라 여기서 막는다.
    if ((argc - 1) % 2 != 0) {
        std::cerr << "인자가 --키 값 쌍이 아니다. 마지막 인자 '" << argv[argc - 1]
                  << "' 에 값이 없다.\n";
        return 2;
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        const std::string val = argv[i + 1];
        if      (k == "--in")          in_path = val;
        else if (k == "--out")         out_path = val;
        else if (k == "--right")       right_path = val;
        else if (k == "--times")       times_file = val;
        else if (k == "--interp")      interp = val;
        else if (k == "--yaw")         yaw = std::atof(val.c_str());
        else if (k == "--pitch")       pitch = std::atof(val.c_str());
        else if (k == "--hfov")        hfov = std::atof(val.c_str());
        else if (k == "--width")       width = std::atoi(val.c_str());
        else if (k == "--height")      height = std::atoi(val.c_str());
        else if (k == "--stride")      stride = std::max(1, std::atoi(val.c_str()));
        else if (k == "--max-frames")  max_frames = std::atoi(val.c_str());
        else if (k == "--fps")         fps = std::atof(val.c_str());
        else if (k == "--baseline")    baseline = std::atof(val.c_str());
        else if (k == "--baseline-yaw") { baseline_yaw = std::atof(val.c_str());
                                          baseline_yaw_given = true; }
        else if (k == "--min-depth")   min_depth = std::atof(val.c_str());
        else {
            // 모르는 옵션을 조용히 무시하면 오타가 기본값으로 굴러간다.
            std::cerr << "모르는 옵션: " << k << "\n";
            return 2;
        }
    }
    if (in_path.empty() || out_path.empty()) {
        std::cerr <<
            "사용법: wme_equirect_convert --in <등장방형 파일/폴더> --out <출력경로>\n"
            "          [--yaw 0] [--pitch 0] [--hfov 90]\n"
            "          [--width 1024] [--height 768] [--interp cubic]\n"
            "          [--stride 1] [--max-frames 0] [--fps 10] [--times <파일>]\n"
            "          [--right <오른쪽 등장방형>] [--baseline <m>] [--baseline-yaw 0]\n"
            "          [--min-depth 4.0]\n"
            "\n"
            "  --right/--baseline 을 주면 360 스테레오 리그로 보고 깊이를 만든다.\n"
            "  주지 않으면 rgb 전용 배치가 나오고, 그것은 RGB-D 러너가 못 먹는다.\n";
        return 2;
    }
    const int flag = interpFlag(interp);
    if (flag < 0) { std::cerr << "모르는 보간법: " << interp << "\n"; return 2; }
    if (!(min_depth > 0.0)) { std::cerr << "--min-depth 는 양수여야 한다\n"; return 2; }

    const bool stereo = !right_path.empty();
    if (stereo && !(baseline > 0.0)) {
        std::cerr << "--right 를 주면 --baseline (m) 도 줘야 한다. 베이스라인 없이는\n"
                     "시차가 깊이로 변환되지 않는다.\n";
        return 2;
    }
    if (right_path.empty() && baseline > 0.0) {
        std::cerr << "--baseline 만 주는 것은 의미가 없다. --right 도 줘라.\n";
        return 2;
    }

    PinholeView view;
    if (!deriveIntrinsics(width, height, hfov, yaw, pitch, view)) return 2;

    std::string how;
    std::vector<Frame> frames = listInput(in_path, how);
    if (frames.empty()) {
        std::cerr << "입력 영상이 없다: " << in_path << "\n";
        return 1;
    }
    std::vector<Frame> rframes;
    if (stereo) {
        std::string rhow;
        rframes = listInput(right_path, rhow);
        if (rframes.size() != frames.size()) {
            std::cerr << "좌/우 프레임 수가 다르다: " << frames.size() << " vs "
                      << rframes.size() << ". 짝이 맞지 않으면 시차는 무의미하다.\n";
            return 1;
        }
    }
    if (!assignStamps(frames, in_path, times_file, fps)) return 1;

    std::cout << "입력: " << in_path << " (" << how << ", " << frames.size() << " 프레임)\n"
              << "뷰: yaw " << yaw << "도  pitch " << pitch << "도  hfov " << hfov
              << "도 -> vfov " << std::setprecision(6) << view.vfov_deg << "도\n"
              << "내부파라미터(유도): " << width << "x" << height
              << "  f=(" << view.fx << ", " << view.fy << ")"
              << "  c=(" << view.cx << ", " << view.cy << ")\n";

    // 첫 영상으로 원본 크기를 확정한다. 등장방형은 가로:세로 = 2:1 이어야
    // 한다 - 아니면 경도/위도 축척이 달라 위 규약이 통째로 틀린다. 조용히
    // 진행하면 결과가 세로로 늘어난 채 "돌아간다".
    cv::Mat first = cv::imread(frames[0].path.string(), cv::IMREAD_UNCHANGED);
    if (first.empty()) {
        std::cerr << "영상을 읽지 못했다: " << frames[0].path << "\n";
        return 1;
    }
    const int src_w = first.cols, src_h = first.rows;
    if (src_w != 2 * src_h) {
        std::cerr << "등장방형이 아니다: " << src_w << "x" << src_h
                  << " (가로 = 2 x 세로 여야 한다). 잘라낸 뷰의 기하가 틀린다.\n";
        return 1;
    }
    std::cout << "원본: " << src_w << "x" << src_h
              << "  적도 각해상도 " << (360.0 / src_w) << "도/화소"
              << "  -> 뷰 " << (view.hfov_deg / view.width) << "도/화소\n";
    // 잘라낸 뷰가 원본보다 촘촘하면 없는 정보를 확대하는 것이다. 실패는
    // 아니지만 흐릿한 영상이 "고해상도" 로 보이므로 반드시 알린다.
    if (view.hfov_deg / view.width < 360.0 / src_w) {
        std::cout << "  경고: 뷰가 원본보다 촘촘하다. 업샘플링이라 새 정보는 없다.\n";
    }

    cv::Mat map_x, map_y;
    buildMaps(view, src_w, src_h, map_x, map_y);

    // 스테레오 준비
    double dv_max = 0.0;
    bool swap_lr = false;
    double eff_baseline = 0.0;
    std::unique_ptr<wme::StereoDepth> stereo_depth;
    wme::StereoDepthConfig scfg;
    constexpr double kDepthScale = 256.0;   // 80 m * 256 = 20480 < 65535

    if (stereo) {
        // **가정을 조용히 두지 않는다.** 이 한 줄이 없어서 다른 리그에서
        // 정확히 거꾸로 판정한 적이 있다 (checkRectified 주석 참조).
        std::cout << "스테레오: 베이스라인 방위를 " << baseline_yaw
                  << " 도로 가정한다"
                  << (baseline_yaw_given ? " (--baseline-yaw 로 지정됨)"
                                         : " (기본값 - 리그가 다르면 --baseline-yaw 로 알려라)")
                  << ". dv = 0 인 뷰는 yaw " << baseline_yaw << " 와 "
                  << (baseline_yaw + 180.0) << " 도다.\n";
        if (!checkRectified(view, baseline, min_depth, baseline_yaw, dv_max)) {
            std::cerr << "\n이 뷰는 정렬 스테레오가 아니다.\n"
                      << "  yaw " << yaw << "도, pitch " << pitch << "도, B "
                      << baseline << " m, 최근접 " << min_depth << " m 에서\n"
                      << "  세로 시차가 최대 " << dv_max << " px (허용 "
                      << kMaxVerticalDisparityPx << " px).\n"
                      << "  베이스라인 방위가 " << baseline_yaw
                      << " 도일 때 dv = 0 인 것은 yaw 가 " << baseline_yaw
                      << " 또는 " << (baseline_yaw + 180.0) << " 도인 뷰뿐이다\n"
                      << "  (소스의 checkRectified 유도 참조). 그 방위에서 90 도\n"
                      << "  떨어진 뷰에서는 베이스라인이 광축과 나란해져 시차가\n"
                      << "  깊이 정보를 아예 잃는다.\n"
                      << "  **리그의 베이스라인 방위가 " << baseline_yaw
                      << " 도가 아니라면 --baseline-yaw 로 알려라** - 그러지 않으면\n"
                      << "  이 검사가 거꾸로 답한다. 깊이 없이 쓰려면 --right 를 빼라.\n";
            return 1;
        }
        // t_cam,x = B cos(yaw - a). 이것이 음수면 --right 로 준 쪽이 실제로는
        // 왼쪽이다. SGBM 은 기준 영상이 왼쪽이어야 하므로 바꾼다.
        const double tx = baseline * std::cos(deg2rad(yaw - baseline_yaw));
        swap_lr = tx < 0.0;
        eff_baseline = std::abs(tx);

        scfg.focal_px = view.fx;
        scfg.baseline_m = eff_baseline;
        scfg.num_disparities =
            wme::StereoDepth::requiredDisparities(view.fx, eff_baseline, min_depth);
        scfg.block_size = 5;
        scfg.min_depth_m = min_depth * 0.8;
        scfg.max_depth_m = 80.0;
        scfg.max_depth_sigma_m = 3.0;
        stereo_depth = std::make_unique<wme::StereoDepth>(scfg);

        std::cout << "스테레오: B=" << eff_baseline << " m"
                  << (swap_lr ? " (yaw 180도라 좌우를 바꿔 넣는다)" : "")
                  << "  세로시차 최대 " << dv_max << " px"
                  << "  시차범위 " << scfg.num_disparities
                  << " (최근접 " << (view.fx * eff_baseline /
                                     (scfg.num_disparities - 1)) << " m)\n";
    }

    const fs::path out = out_path;
    fs::create_directories(out / "rgb");
    if (stereo) fs::create_directories(out / "depth");

    std::ofstream rgb_idx(out / "rgb.txt");
    if (!rgb_idx) { std::cerr << "rgb.txt 를 열 수 없다: " << out << "\n"; return 1; }
    rgb_idx << "# timestamp filename\n" << std::fixed << std::setprecision(6);

    std::ofstream dep_idx;
    if (stereo) {
        dep_idx.open(out / "depth.txt");
        if (!dep_idx) { std::cerr << "depth.txt 를 열 수 없다\n"; return 1; }
        dep_idx << "# timestamp filename\n" << std::fixed << std::setprecision(6);
    }

    int written = 0;
    double sum_valid = 0.0, sum_clip = 0.0;

    for (std::size_t i = 0; i < frames.size(); i += static_cast<std::size_t>(stride)) {
        cv::Mat src = cv::imread(frames[i].path.string(),
                                 stereo ? cv::IMREAD_GRAYSCALE : cv::IMREAD_UNCHANGED);
        if (src.empty()) {
            std::cerr << "영상을 읽지 못했다: " << frames[i].path << "\n";
            return 1;
        }
        if (src.cols != src_w || src.rows != src_h) {
            std::cerr << "프레임마다 원본 크기가 다르다: " << frames[i].path
                      << " (" << src.cols << "x" << src.rows << ")\n";
            return 1;
        }

        cv::Mat left;
        cv::remap(wrapPad(src), left, map_x, map_y, flag, cv::BORDER_REPLICATE);

        const std::string name = pad6(written);
        const std::string rel_rgb = "rgb/" + name + ".png";
        if (!cv::imwrite((out / rel_rgb).string(), left)) {
            std::cerr << "PNG 쓰기 실패: " << rel_rgb << "\n";
            return 1;
        }
        rgb_idx << frames[i].stamp << " " << rel_rgb << "\n";

        if (stereo) {
            cv::Mat rsrc = cv::imread(rframes[i].path.string(), cv::IMREAD_GRAYSCALE);
            if (rsrc.empty() || rsrc.cols != src_w || rsrc.rows != src_h) {
                std::cerr << "오른쪽 영상을 읽지 못했거나 크기가 다르다: "
                          << rframes[i].path << "\n";
                return 1;
            }
            cv::Mat right;
            cv::remap(wrapPad(rsrc), right, map_x, map_y, flag, cv::BORDER_REPLICATE);

            const wme::StereoDepthResult sr = swap_lr ? stereo_depth->compute(right, left)
                                                      : stereo_depth->compute(left, right);
            sum_valid += sr.valid_ratio;
            sum_clip += sr.clipped_ratio;

            cv::Mat d16(sr.depth.size(), CV_16U);
            for (int y = 0; y < sr.depth.rows; ++y) {
                const float* zp = sr.depth.ptr<float>(y);
                std::uint16_t* op = d16.ptr<std::uint16_t>(y);
                for (int x = 0; x < sr.depth.cols; ++x) {
                    const double val = zp[x] * kDepthScale;
                    op[x] = (zp[x] > 0.0f && val < 65535.0)
                                ? static_cast<std::uint16_t>(val + 0.5) : 0;
                }
            }
            const std::string rel_dep = "depth/" + name + ".png";
            if (!cv::imwrite((out / rel_dep).string(), d16)) {
                std::cerr << "PNG 쓰기 실패: " << rel_dep << "\n";
                return 1;
            }
            dep_idx << frames[i].stamp << " " << rel_dep << "\n";
        }

        ++written;
        if (written % 100 == 0) std::cout << "  " << written << " 프레임\n" << std::flush;
        if (max_frames > 0 && written >= max_frames) break;
    }

    // calib.txt. dataset_calib.hpp 가 읽는 키가 먼저 오고, 뷰 파라미터는 그
    // 뒤에 같은 형식으로 붙인다 - 파서가 모르는 키를 그냥 넘기므로 안전하고,
    // 무엇을 잘라낸 배치인지가 산출물 안에 남는다. 재현에 필요한 값이다.
    {
        std::ofstream cf(out / "calib.txt");
        if (!cf) { std::cerr << "calib.txt 를 쓸 수 없다\n"; return 1; }
        cf << "# wme_equirect_convert: 등장방형 " << src_w << "x" << src_h
           << " 에서 잘라낸 원근 뷰\n"
           << "# fx = (W/2)/tan(hfov/2), cx = (W-1)/2 (유도는 소스 머리말)\n"
           << "# 왜곡은 정확히 0 이다. 재매핑으로 만든 이상적 핀홀이라\n"
           << "# 렌즈 왜곡이 들어올 자리가 없다 (원본이 이미 보정된 파노라마라면).\n"
           << std::setprecision(9)
           << "fx: " << view.fx << "\n"
           << "fy: " << view.fy << "\n"
           << "cx: " << view.cx << "\n"
           << "cy: " << view.cy << "\n"
           << "width: " << view.width << "\n"
           << "height: " << view.height << "\n"
           << "dist: 0 0 0 0 0\n";
        if (stereo) {
            cf << "depth_scale: " << kDepthScale << "\n"
               << "depth_min: " << scfg.min_depth_m << "\n"
               << "depth_max: " << scfg.max_depth_m << "\n"
               << "# 깊이는 360 스테레오 쌍의 StereoSGBM **추정값** 이다. 측정값이 아니다.\n"
               << "stereo_baseline: " << eff_baseline << "\n";
        }
        cf << "# --- 잘라내기 파라미터 (재현용) ---\n"
           << "equirect_src_width: " << src_w << "\n"
           << "equirect_src_height: " << src_h << "\n"
           << "equirect_yaw_deg: " << view.yaw_deg << "\n"
           << "equirect_pitch_deg: " << view.pitch_deg << "\n"
           << "equirect_hfov_deg: " << view.hfov_deg << "\n"
           << "equirect_vfov_deg: " << view.vfov_deg << "\n"
           << "equirect_interp: " << interp << "\n";
        if (!cf) { std::cerr << "calib.txt 쓰기 실패\n"; return 1; }
    }

    std::cout << "\n변환 완료: " << out.string() << "\n"
              << "  프레임 " << written << " / " << frames.size()
              << " (stride " << stride << ")\n";
    if (stereo) {
        std::cout << "  평균 유효깊이 " << (100.0 * sum_valid / std::max(1, written)) << "%"
                  << "  상한클리핑 " << (100.0 * sum_clip / std::max(1, written)) << "%\n";
        if (sum_clip / std::max(1, written) > 0.02)
            std::cout << "  경고: 상한 클리핑이 2% 를 넘는다. --min-depth 를 낮춰라.\n";
    }
    std::cout << "  정답궤적 없음. groundtruth.txt 를 만들지 않았다 -\n"
              << "    빈 파일을 남기면 하류가 '정답이 있다' 고 믿는다.\n";

    if (!stereo) {
        // 여기가 이 도구에서 제일 중요한 출력이다. 조용히 넘어가면 사용자는
        // rgb 전용 배치를 RGB-D 러너에 넣고 원인을 못 찾는다.
        std::cout << "\n"
                  << "  ** 깊이 없음 **  이 배치는 rgb 전용이다.\n"
                  << "     wme_tum_odometry / wme_tum_fusion / wme_scene_export 는\n"
                  << "     depth/ 와 depth.txt 를 요구하므로 이 배치를 못 먹는다.\n"
                  << "     360 영상 한 장에는 깊이 정보가 없고, 잘라내기는 정보를\n"
                  << "     더하지 않는다 - 광선을 다시 매길 뿐이다.\n"
                  << "     깊이가 필요하면 360 스테레오 리그(--right --baseline)를 쓰거나,\n"
                  << "     깊이를 안 쓰는 단안 경로에만 이 배치를 써라.\n";
    }
    return 0;
}
