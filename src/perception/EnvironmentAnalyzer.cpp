#include "wme/perception/EnvironmentAnalyzer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wme {
namespace {

inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// 지수이동평균
inline void ema(double& acc, double sample, double alpha) {
    acc = (1.0 - alpha) * acc + alpha * sample;
}

void resizeToWidth(const cv::Mat& src, cv::Mat& dst, int target_w) {
    if (src.cols <= target_w) { src.copyTo(dst); return; }
    const double s = static_cast<double>(target_w) / static_cast<double>(src.cols);
    cv::resize(src, dst, cv::Size(), s, s, cv::INTER_AREA);
}

// 히스테리시스 판정: 켜질 때와 꺼질 때 임계가 다르다.
// 인자명은 leave 다 - exit 로 두면 <cstdlib> 의 ::exit 를 가려 gcc -Wshadow 가 뜬다.
bool hysteresis(bool current, double score, double enter, double leave) {
    return current ? (score > leave) : (score > enter);
}

}  // namespace

std::string_view toString(Lighting v) noexcept {
    switch (v) {
        case Lighting::Bright:      return "Bright";
        case Lighting::Normal:      return "Normal";
        case Lighting::LowLight:    return "LowLight";
        case Lighting::Dark:        return "Dark";
        case Lighting::Backlit:     return "Backlit";
        case Lighting::HighDynamic: return "HighDynamic";
        default:                    return "Unknown";
    }
}

std::string_view toString(Weather v) noexcept {
    switch (v) {
        case Weather::Clear: return "Clear";
        case Weather::Rain:  return "Rain";
        case Weather::Snow:  return "Snow";
        case Weather::Fog:   return "Fog";
        case Weather::Smoke: return "Smoke";
        case Weather::Dust:  return "Dust";
        default:             return "Unknown";
    }
}

std::string_view toString(SceneKind v) noexcept {
    switch (v) {
        case SceneKind::Indoor:  return "Indoor";
        case SceneKind::Outdoor: return "Outdoor";
        default:                 return "Unknown";
    }
}

std::string EnvironmentState::summary() const {
    std::string s;
    s += std::string(toString(scene)) + "/" + std::string(toString(lighting)) +
         "/" + std::string(toString(weather));
    s += " vis=" + std::to_string(visibility).substr(0, 4);
    s += " rel=" + std::to_string(sensor_reliability).substr(0, 4);
    s += " a=[" + std::to_string(tier.photometric).substr(0, 4) + "," +
                  std::to_string(tier.constellation).substr(0, 4) + "," +
                  std::to_string(tier.structural).substr(0, 4) + "]";
    return s;
}

// ---------------------------------------------------------------------------

EnvironmentAnalyzer::EnvironmentAnalyzer(EnvironmentConfig config) : cfg_(config) {
    if (cfg_.history_size % 2 == 0) ++cfg_.history_size;   // 중앙값을 위해 홀수 강제
}

void EnvironmentAnalyzer::reset() {
    history_.clear();
    history_head_  = 0;
    history_count_ = 0;
    prev_gray_.release();
    temporal_median_.release();
    state_       = EnvironmentState{};
    indoorness_ema_ = 0.5;
    last_update_ = Timestamp{};
    initialized_ = false;
}

void EnvironmentAnalyzer::setDynamicLevel(double ratio) {
    state_.dynamic_level = clamp01(ratio);
}

double EnvironmentAnalyzer::estimateHaze(const cv::Mat& bgr_small) const {
    // Dark Channel Prior: 맑은 야외 영상은 국소 패치의 최소 채널값이 0 에 가깝다.
    // 안개/연무는 대기광이 더해져 이 값이 전반적으로 올라간다.
    if (bgr_small.empty() || bgr_small.channels() != 3) return 0.0;

    cv::Mat min_ch(bgr_small.size(), CV_8U);
    for (int y = 0; y < bgr_small.rows; ++y) {
        const cv::Vec3b* src = bgr_small.ptr<cv::Vec3b>(y);
        std::uint8_t*    dst = min_ch.ptr<std::uint8_t>(y);
        for (int x = 0; x < bgr_small.cols; ++x) {
            dst[x] = std::min({src[x][0], src[x][1], src[x][2]});
        }
    }

    cv::Mat dark;
    const cv::Mat se = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(cfg_.dcp_patch, cfg_.dcp_patch));
    cv::erode(min_ch, dark, se);   // 패치 최소값 = 침식

    const double mean_dark = cv::mean(dark)[0] / 255.0;

    // 안개는 대비도 함께 떨어뜨린다. 두 근거를 결합해 흰 벽 오검출을 줄인다.
    cv::Scalar mu, sigma;
    cv::meanStdDev(min_ch, mu, sigma);
    const double low_contrast = clamp01(1.0 - sigma[0] / 40.0);

    return clamp01(std::sqrt(clamp01(mean_dark / 0.55) * low_contrast));
}

void EnvironmentAnalyzer::analyzeTransientParticles(const cv::Mat& gray_f,
                                                    EnvironmentEvidence& ev) {
    // 비/눈은 배경 대비 "밝은 쪽으로만" 나타나고 곧 사라진다.
    // 시간적 중앙값 대비 양의 잔차를 뽑으면 배경 움직임과 분리된다.
    // 링 버퍼에 덮어쓴다. clone() 은 프레임마다 Mat 을 새로 잡았다.
    const auto cap = static_cast<std::size_t>(std::max(1, cfg_.history_size));
    if (history_.size() != cap) {
        history_.assign(cap, cv::Mat{});
        history_head_  = 0;
        history_count_ = 0;
    }
    gray_f.copyTo(history_[(history_head_ + history_count_) % cap]);
    if (history_count_ < cap) ++history_count_;
    else                      history_head_ = (history_head_ + 1) % cap;

    if (history_count_ < 3) { ev.rain_streak = 0.0; ev.snow_particle = 0.0; return; }

    // 픽셀별 시간 중앙값
    const int n = static_cast<int>(history_count_);
    temporal_median_.create(gray_f.size(), CV_32F);
    median_scratch_.resize(static_cast<std::size_t>(n));
    std::vector<float>& buf = median_scratch_;

    for (int y = 0; y < gray_f.rows; ++y) {
        float* med = temporal_median_.ptr<float>(y);
        for (int x = 0; x < gray_f.cols; ++x) {
            for (int i = 0; i < n; ++i) {
                buf[static_cast<std::size_t>(i)] =
                    history_[(history_head_ + static_cast<std::size_t>(i)) % cap]
                        .ptr<float>(y)[x];
            }
            std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
            med[x] = buf[static_cast<std::size_t>(n / 2)];
        }
    }

    cv::Mat residual = gray_f - temporal_median_;
    cv::threshold(residual, residual, 8.0, 0.0, cv::THRESH_TOZERO);   // 양의 잔차만

    const double particle_ratio =
        static_cast<double>(cv::countNonZero(residual)) / static_cast<double>(residual.total());

    if (particle_ratio < 1e-4) { ev.rain_streak = 0.0; ev.snow_particle = 0.0; return; }

    // 잔차의 형태로 비와 눈을 구분한다.
    // 비 = 가늘고 방향이 일정한 줄무늬 -> 구조텐서 이방성 높음
    // 눈 = 둥글고 방향성 없음 -> 이방성 낮음
    cv::Mat rgx, rgy;
    cv::Scharr(residual, rgx, CV_32F, 1, 0);
    cv::Scharr(residual, rgy, CV_32F, 0, 1);

    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int y = 0; y < rgx.rows; ++y) {
        const float* px = rgx.ptr<float>(y);
        const float* py = rgy.ptr<float>(y);
        for (int x = 0; x < rgx.cols; ++x) {
            sxx += static_cast<double>(px[x]) * px[x];
            syy += static_cast<double>(py[x]) * py[x];
            sxy += static_cast<double>(px[x]) * py[x];
        }
    }
    const double tr   = sxx + syy;
    double anisotropy = 0.0;
    if (tr > kEps) {
        const double det  = sxx * syy - sxy * sxy;
        const double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
        const double l1 = tr * 0.5 + disc, l2 = tr * 0.5 - disc;
        anisotropy = (l1 > kEps) ? clamp01((l1 - l2) / l1) : 0.0;
    }

    // 입자 밀도를 0..1 로 (5% 이상이면 극심)
    const double density = clamp01(particle_ratio / 0.05);

    ev.rain_streak   = density * anisotropy;
    ev.snow_particle = density * (1.0 - anisotropy);
}

double EnvironmentAnalyzer::estimateShake(const cv::Mat& gray_f) {
    if (prev_gray_.empty() || prev_gray_.size() != gray_f.size()) {
        gray_f.copyTo(prev_gray_);
        return 0.0;
    }
    // 위상상관으로 전역 병진 성분을 구한다. 큰 값 = 흔들림 또는 급회전.
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(prev_gray_, gray_f, cv::noArray(), &response);
    gray_f.copyTo(prev_gray_);

    const double mag = std::hypot(shift.x, shift.y);
    // 저해상도 기준 8 px 이동을 극심으로 본다
    return clamp01(mag / 8.0) * clamp01(response * 2.0);
}

double EnvironmentAnalyzer::estimateBacklight(const cv::Mat& gray_f) const {
    // 역광: 주변부가 밝고 중앙이 어둡다.
    const int mx = gray_f.cols / 4, my = gray_f.rows / 4;
    const cv::Rect center(mx, my, gray_f.cols - 2 * mx, gray_f.rows - 2 * my);

    const double mean_all    = cv::mean(gray_f)[0];
    const double mean_center = cv::mean(gray_f(center))[0];

    const double total_sum  = mean_all * static_cast<double>(gray_f.total());
    const double center_sum = mean_center * static_cast<double>(center.area());
    const double border_n   = static_cast<double>(gray_f.total() - static_cast<std::size_t>(center.area()));
    if (border_n < 1.0) return 0.0;

    const double mean_border = (total_sum - center_sum) / border_n;
    return clamp01((mean_border - mean_center) / 60.0);
}

double EnvironmentAnalyzer::estimateSpecular(const cv::Mat& bgr_small) const {
    // 반사/젖은 바닥: 하단부에 채도 낮고 휘도 높은 픽셀이 몰린다.
    if (bgr_small.empty() || bgr_small.channels() != 3) return 0.0;

    cv::Mat hsv;
    cv::cvtColor(bgr_small, hsv, cv::COLOR_BGR2HSV);

    const int y0 = hsv.rows / 2;
    int specular = 0, total = 0;
    for (int y = y0; y < hsv.rows; ++y) {
        const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            ++total;
            if (row[x][2] > 210 && row[x][1] < 40) ++specular;
        }
    }
    if (total == 0) return 0.0;
    return clamp01(static_cast<double>(specular) / static_cast<double>(total) / 0.15);
}

double EnvironmentAnalyzer::estimateShadow(const cv::Mat& gray_f) const {
    // 강한 그림자: 휘도 분포가 이중봉이고 두 봉우리 사이가 비어 있다.
    constexpr int kBins = 32;
    std::array<int, kBins> hist{};
    for (int y = 0; y < gray_f.rows; ++y) {
        const float* row = gray_f.ptr<float>(y);
        for (int x = 0; x < gray_f.cols; ++x) {
            const int b = std::clamp(static_cast<int>(row[x] * kBins / 256.0f), 0, kBins - 1);
            ++hist[static_cast<std::size_t>(b)];
        }
    }
    const auto total = static_cast<double>(gray_f.total());
    if (total < 1.0) return 0.0;

    // 최대 두 봉우리와 그 사이 최저점의 깊이로 이중봉성을 측정
    int p1 = 0;
    for (int i = 1; i < kBins; ++i) if (hist[static_cast<std::size_t>(i)] > hist[static_cast<std::size_t>(p1)]) p1 = i;

    int p2 = -1;
    for (int i = 0; i < kBins; ++i) {
        if (std::abs(i - p1) < 6) continue;   // 같은 봉우리 어깨 제외
        if (p2 < 0 || hist[static_cast<std::size_t>(i)] > hist[static_cast<std::size_t>(p2)]) p2 = i;
    }
    if (p2 < 0) return 0.0;

    const int lo = std::min(p1, p2), hi = std::max(p1, p2);
    int valley = hist[static_cast<std::size_t>(lo)];
    for (int i = lo; i <= hi; ++i) valley = std::min(valley, hist[static_cast<std::size_t>(i)]);

    const double peak_min = static_cast<double>(std::min(hist[static_cast<std::size_t>(p1)], hist[static_cast<std::size_t>(p2)]));
    if (peak_min < total * 0.02) return 0.0;

    const double depth     = 1.0 - static_cast<double>(valley) / peak_min;
    const double separation = static_cast<double>(hi - lo) / static_cast<double>(kBins);
    return clamp01(depth * separation * 2.0);
}

double EnvironmentAnalyzer::estimateTexturePoverty(const cv::Mat& gray_f,
                                                   double noise_sigma) const {
    // Scharr 게인 32 를 나눠 gray/px 로 맞춘다. 정규화 없이 임계 20 을 쓰면
    // 실제로는 0.6 gray/px 라는 터무니없이 낮은 기준이 된다.
    cv::Mat gx, gy, mag;
    cv::Scharr(gray_f, gx, CV_32F, 1, 0, 1.0 / 32.0);
    cv::Scharr(gray_f, gy, CV_32F, 0, 1, 1.0 / 32.0);
    cv::magnitude(gx, gy, mag);

    // 잡음만으로도 그래디언트는 생긴다. 정규화 Scharr 의 잡음 이득은
    // sqrt(sum w^2)/32 = sqrt(236)/32 = 0.48 (축당), 크기로는 약 0.68*sigma.
    // 이 몫을 빼지 않으면 잡음이 텍스처로 계산된다. 실측에서 열화 영상
    // (sigma=12) 의 texture_poverty 가 깨끗한 영상보다 낮게 나왔다 - 완전한 역전.
    const double noise_grad = 0.68 * std::max(0.0, noise_sigma);
    const auto thresh = static_cast<float>(
        std::max(cfg_.texture_min_gradient, 2.0 * noise_grad));

    int weak = 0;
    for (int y = 0; y < mag.rows; ++y) {
        const float* row = mag.ptr<float>(y);
        for (int x = 0; x < mag.cols; ++x) if (row[x] < thresh) ++weak;
    }
    return clamp01(static_cast<double>(weak) / static_cast<double>(mag.total()));
}

double EnvironmentAnalyzer::estimateSceneComplexity(const cv::Mat& gray_f) const {
    // 그래디언트 방향 히스토그램의 엔트로피. 구조가 다양할수록 높다.
    cv::Mat gx, gy;
    cv::Scharr(gray_f, gx, CV_32F, 1, 0, 1.0 / 32.0);
    cv::Scharr(gray_f, gy, CV_32F, 0, 1, 1.0 / 32.0);

    constexpr int kBins = 18;   // 10도 간격
    std::array<double, kBins> hist{};
    double total = 0.0;

    for (int y = 0; y < gx.rows; ++y) {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        for (int x = 0; x < gx.cols; ++x) {
            const double m = std::hypot(static_cast<double>(px[x]), static_cast<double>(py[x]));
            if (m < cfg_.texture_min_gradient) continue;
            // float 인자를 그대로 넘기면 atan2f 가 잡힌다. 수평 엣지에서
            // atan2f(-y,0) = -1.57079637 이라 +kPi(double) 하면 pi/2 를 미세하게
            // 밑돌아 bin 8 로 떨어지고, 반대 부호는 bin 9 로 간다 - 같은 방향이
            // 부호에 따라 두 빈으로 쪼개진다. 실내 장면의 지배적 방향이 바로
            // 여기라 엔트로피가 체계적으로 부풀었다 (체커 0.473 -> 0.578).
            double a = std::atan2(static_cast<double>(py[x]), static_cast<double>(px[x]));
            if (a < 0.0) a += kPi;   // 방향은 180도 주기
            // clamp 가 아니라 wrap. a == kPi 는 a == 0 과 같은 방향인데 clamp 는
            // 반대쪽 끝 빈(17)에 넣는다. 세로 줄무늬 한 방향짜리 영상이
            // 빈 0 과 17 로 갈라져 복잡도 0.24 를 보고했다 - 정답은 0 이다.
            int b = static_cast<int>(a / kPi * kBins);
            b = ((b % kBins) + kBins) % kBins;
            hist[static_cast<std::size_t>(b)] += m;
            total += m;
        }
    }
    if (total < kEps) return 0.0;

    double entropy = 0.0;
    for (double h : hist) {
        const double p = h / total;
        if (p > 1e-9) entropy -= p * std::log(p);
    }
    return clamp01(entropy / std::log(static_cast<double>(kBins)));
}

double EnvironmentAnalyzer::estimateIndoorness(const cv::Mat& bgr_small) const {
    // 실내/실외 판별은 단일 단서로 불가능하므로 약한 근거 3개를 합산한다.
    //  1) 상단부가 하단부보다 크게 밝으면 하늘 -> 실외
    //  2) 상단부 청색 우세 -> 실외
    //  3) 휘도 동적범위가 넓으면 실외
    if (bgr_small.empty() || bgr_small.channels() != 3) return 0.5;

    const cv::Rect top(0, 0, bgr_small.cols, std::max(1, bgr_small.rows / 4));
    const cv::Rect bot(0, bgr_small.rows * 3 / 4, bgr_small.cols,
                       std::max(1, bgr_small.rows - bgr_small.rows * 3 / 4));

    const cv::Scalar mt = cv::mean(bgr_small(top));
    const cv::Scalar mb = cv::mean(bgr_small(bot));

    const double lum_t = 0.114 * mt[0] + 0.587 * mt[1] + 0.299 * mt[2];
    const double lum_b = 0.114 * mb[0] + 0.587 * mb[1] + 0.299 * mb[2];

    const double sky_bright = clamp01((lum_t - lum_b) / 70.0);
    const double sky_blue   = clamp01((mt[0] - 0.5 * (mt[1] + mt[2])) / 30.0);

    cv::Mat gray;
    cv::cvtColor(bgr_small, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mu, sd;
    cv::meanStdDev(gray, mu, sd);
    const double wide_range = clamp01(sd[0] / 70.0);

    const double outdoorness = clamp01(0.45 * sky_bright + 0.35 * sky_blue + 0.20 * wide_range);
    return 1.0 - outdoorness;   // indoorness
}

void EnvironmentAnalyzer::computeEvidence(const Frame& frame, const ImageQuality& quality,
                                          EnvironmentEvidence& ev) {
    cv::Mat gray_small, bgr_small;
    if (!frame.rgb.empty()) {
        resizeToWidth(frame.rgb, bgr_small, cfg_.analysis_width);
        cv::cvtColor(bgr_small, gray_small, cv::COLOR_BGR2GRAY);
    } else {
        resizeToWidth(frame.gray, gray_small, cfg_.analysis_width);
    }
    if (gray_small.empty()) return;

    cv::Mat gray_f;
    gray_small.convertTo(gray_f, CV_32F);

    const double a = cfg_.evidence_ema;

    // 저조도: 휘도가 낮으면 직접 증거, 게인 노이즈가 크면 보강 증거.
    // 가중합(0.7*lum + 0.3*noise*lum)은 두 군데서 틀린다. 노이즈가 0 이면
    // darkness 가 0.7 을 못 넘어 Dark 라벨(>0.75)이 도달 불가능하고,
    // 노이즈 항에 lum 이 곱해져 있어 자동노출이 어두운 장면을 밝게 올린 경우 -
    // 노이즈 채널이 존재하는 이유인 바로 그 경우 - 에 0 으로 지워진다.
    // noisy-OR 로 바꾸면 둘 다 각자 증거로 서고 함께면 더 강해진다.
    const double dark_by_lum   = clamp01((cfg_.dark_brightness - quality.brightness) / cfg_.dark_brightness);
    const double dark_by_noise = clamp01(quality.noise_sigma / 12.0);
    ema(ev.darkness, clamp01(1.0 - (1.0 - dark_by_lum) * (1.0 - 0.5 * dark_by_noise)), a);

    ema(ev.backlight, estimateBacklight(gray_f), a);
    ema(ev.haze, estimateHaze(bgr_small), a);
    ema(ev.specular, estimateSpecular(bgr_small), a);
    ema(ev.shadow_strength, estimateShadow(gray_f), a);
    ema(ev.texture_poverty, estimateTexturePoverty(gray_f, quality.noise_sigma), a);
    ema(ev.camera_shake, estimateShake(gray_f), a);

    ema(ev.motion_blur, clamp01(1.0 - quality.blur_free), a);
    ema(ev.noise, clamp01(1.0 - quality.noise_free), a);
    ema(ev.lens_dirt, clamp01(1.0 - quality.occlusion_free), a);

    // 물방울은 렌즈 오염의 부분집합 중 국소적/원형인 경우. 오염도가 높은데
    // 전체가 아니라 일부 영역에만 몰려 있으면 물방울로 본다.
    ev.water_drop = clamp01(ev.lens_dirt * (1.0 - ev.lens_dirt) * 4.0);

    EnvironmentEvidence particles{};
    analyzeTransientParticles(gray_f, particles);
    ema(ev.rain_streak, particles.rain_streak, a);
    ema(ev.snow_particle, particles.snow_particle, a);

    // 먼지: 연무와 유사하나 채도가 더 낮고 입자성이 있다
    ev.dust = clamp01(ev.haze * ev.snow_particle * 2.0);

    state_.scene_complexity = estimateSceneComplexity(gray_f);
    ema(indoorness_ema_, estimateIndoorness(bgr_small), a);
}

void EnvironmentAnalyzer::resolveLabels(const ImageQuality& quality) {
    const auto& ev = state_.evidence;

    // --- 실내/실외 ---
    const bool was_indoor = (state_.scene == SceneKind::Indoor);
    const bool is_indoor  = hysteresis(was_indoor, indoorness_ema_,
                                       cfg_.enter_threshold, cfg_.exit_threshold);
    state_.scene = is_indoor ? SceneKind::Indoor : SceneKind::Outdoor;

    // --- 조명 ---
    // 우선순위: 매우 어두움 > 역광 > 고동적범위 > 저조도 > 밝음 > 보통
    if (ev.darkness > 0.75) {
        state_.lighting = Lighting::Dark;
    } else if (ev.backlight > cfg_.enter_threshold) {
        state_.lighting = Lighting::Backlit;
    } else if (quality.saturated_high_ratio > 0.06 && quality.saturated_low_ratio > 0.06) {
        state_.lighting = Lighting::HighDynamic;
    } else if (ev.darkness > cfg_.exit_threshold) {
        state_.lighting = Lighting::LowLight;
    } else if (quality.brightness > cfg_.bright_brightness) {
        state_.lighting = Lighting::Bright;
    } else {
        state_.lighting = Lighting::Normal;
    }

    // --- 날씨 --- 가장 강한 증거를 채택하되 최소 강도를 넘어야 한다.
    // 안개와 연기는 같은 산란 증거를 공유한다. 실내면 연기, 실외면 안개로 귀속한다.
    struct Cand { Weather w; double s; };
    const std::array<Cand, 5> cands{{
        {Weather::Fog,   ev.haze * (1.0 - indoorness_ema_)},
        {Weather::Smoke, ev.haze * indoorness_ema_},
        {Weather::Rain,  ev.rain_streak},
        {Weather::Snow,  ev.snow_particle},
        {Weather::Dust,  ev.dust},
    }};
    const auto best = std::max_element(cands.begin(), cands.end(),
                                       [](const Cand& x, const Cand& y) { return x.s < y.s; });

    const bool was_clear = (state_.weather == Weather::Clear || state_.weather == Weather::Unknown);
    if (hysteresis(!was_clear, best->s, cfg_.enter_threshold, cfg_.exit_threshold)) {
        state_.weather = best->w;
    } else {
        state_.weather = Weather::Clear;
    }
}

EnvironmentState deriveAdaptationFrom(const EnvironmentEvidence& ev) {
    EnvironmentState s;
    s.evidence = ev;

    // --- 가시도 ---
    // 안개/비/눈/먼지가 유효 가시거리를 줄인다. 곱으로 결합해 동시 악화를 반영.
    s.visibility = clamp01((1.0 - 0.85 * ev.haze) *
                           (1.0 - 0.40 * ev.rain_streak) *
                           (1.0 - 0.45 * ev.snow_particle) *
                           (1.0 - 0.30 * ev.dust) *
                           (1.0 - 0.50 * ev.darkness));

    // --- 카메라 물리 상태 ---
    s.camera_health = clamp01((1.0 - 0.7 * ev.lens_dirt) *
                              (1.0 - 0.5 * ev.water_drop) *
                              (1.0 - 0.4 * ev.noise));

    // --- 종합 센서 신뢰도 ---
    s.sensor_reliability = clamp01(0.45 * s.visibility +
                                   0.30 * s.camera_health +
                                   0.25 * (1.0 - ev.motion_blur));

    // --- 3-tier 정보 가중치 -------------------------------------------------
    // 설계 근거는 docs/02-correspondence-problem.md 5장 표와 일치한다.

    // 측광(ECDA): 블러/노이즈/저조도/안개에 직접 무너진다. 곱으로 감쇠.
    // texture_poverty 와 camera_shake 도 반드시 포함해야 한다 -
    // 텍스처가 없으면 측광 잔차가 포즈를 구속하지 못하고(랭크 부족),
    // 흔들림이 크면 coarse-to-fine 수렴 반경을 넘긴다.
    s.tier.photometric = clamp01((1.0 - 0.90 * ev.motion_blur) *
                                 (1.0 - 0.85 * ev.haze) *
                                 (1.0 - 0.85 * ev.texture_poverty) *
                                 (1.0 - 0.70 * ev.darkness) *
                                 (1.0 - 0.45 * ev.rain_streak) *
                                 (1.0 - 0.60 * ev.noise) *
                                 (1.0 - 0.50 * ev.lens_dirt) *
                                 (1.0 - 0.35 * ev.camera_shake));

    // 성좌(TCG): YOLO 는 저조도/안개에서도 큰 물체를 잡는다. 감쇠가 훨씬 완만하다.
    // 다만 가시도가 극단적으로 낮으면 검출 자체가 사라진다.
    s.tier.constellation = clamp01(0.35 + 0.65 * std::pow(s.visibility, 0.45));

    // 구조(SPA): 외관과 무관하므로 깊이 센서가 살아 있는 한 유지된다.
    // 텍스처가 빈곤할수록 상대적 중요도가 오히려 올라간다.
    s.tier.structural = clamp01(0.40 + 0.60 * ev.texture_poverty +
                                0.30 * (1.0 - s.tier.photometric));

    // 관측 정보가 전반적으로 부족하면 운동 사전분포 의존도를 올린다.
    const double observability = clamp01(0.5 * s.tier.photometric +
                                         0.3 * s.tier.constellation +
                                         0.2 * s.tier.structural);
    s.tier.motion_prior = clamp01(0.05 + 0.95 * (1.0 - observability));

    // --- 시간 정책 ---------------------------------------------------------
    // 나쁜 환경일수록 오래 기억하고 늦게 버린다. 관측이 희소해지므로
    // 같은 기준을 유지하면 트랙이 조기에 소멸한다.
    const double adversity = 1.0 - s.sensor_reliability;
    s.memory_retention_scale  = 1.0 + 3.0 * adversity;
    s.track_persistence_scale = 1.0 + 2.5 * adversity + 1.5 * ev.motion_blur;

    // 어두우면 YOLO 신뢰도가 전반적으로 낮아지므로 임계를 내린다.
    // 대신 Confidence Engine 이 사후확률로 오검출을 걸러낸다.
    s.detection_threshold_scale =
        std::max(0.35, clamp01(1.0 - 0.45 * ev.darkness - 0.25 * ev.haze));

    return s;
}

void EnvironmentAnalyzer::deriveAdaptation() {
    // 순수 함수가 계산한 값만 가져온다. 라벨/장면복잡도/동적비율은 여기서 건드리지 않는다.
    const EnvironmentState a = deriveAdaptationFrom(state_.evidence);

    state_.visibility               = a.visibility;
    state_.camera_health            = a.camera_health;
    state_.sensor_reliability       = a.sensor_reliability;
    state_.tier                     = a.tier;
    state_.memory_retention_scale   = a.memory_retention_scale;
    state_.track_persistence_scale  = a.track_persistence_scale;
    state_.detection_threshold_scale = a.detection_threshold_scale;
}

const EnvironmentState& EnvironmentAnalyzer::update(const Frame& frame,
                                                    const ImageQuality& quality) {
    // 주기 제한. 첫 프레임은 무조건 평가한다.
    const double period = 1.0 / std::max(0.1, cfg_.update_hz);
    if (initialized_ && last_update_.valid() && frame.stamp.valid() &&
        (frame.stamp - last_update_) < period) {
        return state_;
    }

    computeEvidence(frame, quality, state_.evidence);
    resolveLabels(quality);
    deriveAdaptation();

    state_.stamp = frame.stamp;
    last_update_ = frame.stamp;
    initialized_ = true;
    return state_;
}

}  // namespace wme
