// wme_env_probe - 영상/카메라를 받아 환경 상태와 tier 가중치를 실시간 출력한다.
// 적응 로직이 실제 촬영본에서 어떻게 반응하는지 눈으로 확인하는 용도.
//
//   wme_env_probe <video|카메라인덱스> [--csv out.csv] [--show]

#include "wme/perception/EnvironmentAnalyzer.hpp"
#include "wme/perception/ImageQualityEngine.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

using namespace wme;

namespace {

void printUsage() {
    std::puts("usage: wme_env_probe <video path | camera index> [--csv <file>] [--show]");
}

// 상태를 한 줄로 요약해 터미널에 갱신 출력
void printState(std::uint64_t n, double ms, const ImageQuality& q, const EnvironmentState& e) {
    std::printf(
        "\r[%6llu] %5.1fms | IQS %.2f sharp %.2f noise %4.1f blur %4.1f | "
        "%-7s %-11s %-5s | vis %.2f rel %.2f | a=[%.2f %.2f %.2f] prior %.2f   ",
        static_cast<unsigned long long>(n), ms,
        q.score, q.sharpness, q.noise_sigma, q.blur_extent_px,
        std::string(toString(e.scene)).c_str(),
        std::string(toString(e.lighting)).c_str(),
        std::string(toString(e.weather)).c_str(),
        e.visibility, e.sensor_reliability,
        e.tier.photometric, e.tier.constellation, e.tier.structural, e.tier.motion_prior);
    std::fflush(stdout);
}

// 화면에 상태 오버레이
void drawOverlay(cv::Mat& canvas, const ImageQuality& q, const EnvironmentState& e) {
    const auto put = [&canvas](const std::string& text, int row) {
        cv::putText(canvas, text, {12, 26 + row * 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    {0, 0, 0}, 3, cv::LINE_AA);
        cv::putText(canvas, text, {12, 26 + row * 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    {80, 255, 120}, 1, cv::LINE_AA);
    };
    char buf[192];

    std::snprintf(buf, sizeof(buf), "%s / %s / %s",
                  std::string(toString(e.scene)).c_str(),
                  std::string(toString(e.lighting)).c_str(),
                  std::string(toString(e.weather)).c_str());
    put(buf, 0);

    std::snprintf(buf, sizeof(buf), "IQS %.2f  vis %.2f  health %.2f  rel %.2f",
                  q.score, e.visibility, e.camera_health, e.sensor_reliability);
    put(buf, 1);

    std::snprintf(buf, sizeof(buf), "tier  photo %.2f  const %.2f  struct %.2f  prior %.2f",
                  e.tier.photometric, e.tier.constellation, e.tier.structural, e.tier.motion_prior);
    put(buf, 2);

    std::snprintf(buf, sizeof(buf), "dark %.2f haze %.2f rain %.2f snow %.2f blur %.2f dirt %.2f",
                  e.evidence.darkness, e.evidence.haze, e.evidence.rain_streak,
                  e.evidence.snow_particle, e.evidence.motion_blur, e.evidence.lens_dirt);
    put(buf, 3);

    // 가중 맵을 우하단에 축소 표시
    if (!q.weight_map.empty()) {
        cv::Mat vis;
        q.weight_map.convertTo(vis, CV_8U, 255.0);
        cv::applyColorMap(vis, vis, cv::COLORMAP_VIRIDIS);
        const int w = canvas.cols / 4;
        cv::resize(vis, vis, cv::Size(w, w * vis.rows / vis.cols));
        if (vis.rows < canvas.rows && vis.cols < canvas.cols) {
            vis.copyTo(canvas(cv::Rect(canvas.cols - vis.cols - 8,
                                       canvas.rows - vis.rows - 8, vis.cols, vis.rows)));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { printUsage(); return 1; }

    const std::string source = argv[1];
    std::string csv_path;
    bool show = false;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--show") show = true;
        else { printUsage(); return 1; }
    }

    cv::VideoCapture cap;
    const bool is_index = !source.empty() &&
                          source.find_first_not_of("0123456789") == std::string::npos;
    if (is_index) cap.open(std::stoi(source));
    else          cap.open(source);

    if (!cap.isOpened()) {
        std::fprintf(stderr, "입력을 열 수 없음: %s\n", source.c_str());
        return 2;
    }

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "frame,t,iqs,sharpness,noise_sigma,blur_px,darkness,haze,rain,snow,"
               "motion_blur,lens_dirt,visibility,camera_health,sensor_reliability,"
               "alpha_photo,alpha_const,alpha_struct,motion_prior,lighting,weather,scene\n";
    }

    ImageQualityEngine  iq;
    EnvironmentAnalyzer env({.analysis_width = 192, .history_size = 9, .update_hz = 5.0});

    cv::Mat bgr;
    std::uint64_t n = 0;
    const auto t0 = std::chrono::steady_clock::now();

    while (cap.read(bgr) && !bgr.empty()) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - t0).count();

        Frame f;
        f.id    = FrameId(n);
        f.stamp = Timestamp::fromSeconds(t);
        f.rgb   = bgr;
        cv::cvtColor(bgr, f.gray, cv::COLOR_BGR2GRAY);
        f.intrinsics.fx = f.intrinsics.fy = bgr.cols * 0.9;   // 실측 캘리브레이션 없을 때 근사
        f.intrinsics.cx = bgr.cols * 0.5;
        f.intrinsics.cy = bgr.rows * 0.5;
        f.intrinsics.width  = bgr.cols;
        f.intrinsics.height = bgr.rows;
        f.sensor = SensorKind::Monocular;

        const auto tick = std::chrono::steady_clock::now();
        const ImageQuality q = iq.evaluate(f);
        const EnvironmentState& e = env.update(f, q);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tick).count();

        printState(n, ms, q, e);

        if (csv.is_open()) {
            csv << n << ',' << t << ',' << q.score << ',' << q.sharpness << ','
                << q.noise_sigma << ',' << q.blur_extent_px << ','
                << e.evidence.darkness << ',' << e.evidence.haze << ','
                << e.evidence.rain_streak << ',' << e.evidence.snow_particle << ','
                << e.evidence.motion_blur << ',' << e.evidence.lens_dirt << ','
                << e.visibility << ',' << e.camera_health << ',' << e.sensor_reliability << ','
                << e.tier.photometric << ',' << e.tier.constellation << ','
                << e.tier.structural << ',' << e.tier.motion_prior << ','
                << toString(e.lighting) << ',' << toString(e.weather) << ','
                << toString(e.scene) << '\n';
        }

        if (show) {
            cv::Mat canvas = bgr.clone();
            drawOverlay(canvas, q, e);
            cv::imshow("wme_env_probe", canvas);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;
        }
        ++n;
    }

    std::puts("");
    if (csv.is_open()) std::printf("CSV 저장: %s (%llu 프레임)\n",
                                   csv_path.c_str(), static_cast<unsigned long long>(n));
    return 0;
}
