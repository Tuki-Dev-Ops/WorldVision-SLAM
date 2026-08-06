#include "wme/perception/ImageQualityEngine.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace wme {
namespace {

// 0..1 로 자르기
inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// 긴 변이 target 이 되도록 축소 (확대는 하지 않음)
void resizeToWidth(const cv::Mat& src, cv::Mat& dst, int target_w) {
    if (src.cols <= target_w) { dst = src; return; }
    const double s = static_cast<double>(target_w) / static_cast<double>(src.cols);
    cv::resize(src, dst, cv::Size(), s, s, cv::INTER_AREA);
}

}  // namespace

ImageQualityEngine::ImageQualityEngine(ImageQualityConfig config) : cfg_(config) {}

void ImageQualityEngine::resetLensModel() {
    lens_accum_.release();
    lens_occlusion_.release();
    lens_frames_ = 0;
}

double ImageQualityEngine::estimateNoiseSigma(const cv::Mat& gray32f) {
    // Immerkaer(1996): 라플라시안 차분 마스크의 절대응답 평균에서 sigma 를 얻는다.
    // 구조가 있는 영역도 포함되므로 상위 응답을 잘라 편향을 줄인다.
    static const cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
         1.f, -2.f,  1.f,
        -2.f,  4.f, -2.f,
         1.f, -2.f,  1.f);

    cv::filter2D(gray32f, noise_response_, CV_32F, kernel, cv::Point(-1, -1), 0.0,
                 cv::BORDER_REPLICATE);
    // cv::abs 는 MatExpr 이라 결과 Mat 을 새로 잡는다. 어차피 내부만 쓰므로
    // 절댓값은 아래 수집 루프에서 취한다 (float 절댓값이라 값은 동일하다).

    // 중앙값 기반 추정으로 엣지 이상치에 견디게 함
    std::vector<float>& vals = noise_vals_;
    vals.clear();
    vals.reserve(static_cast<std::size_t>(noise_response_.total()));
    for (int y = 1; y < noise_response_.rows - 1; ++y) {
        const float* row = noise_response_.ptr<float>(y);
        for (int x = 1; x < noise_response_.cols - 1; ++x) vals.push_back(std::abs(row[x]));
    }
    if (vals.empty()) return 0.0;

    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    const double median = vals[mid];

    // sigma = median * sqrt(pi/2) / (6 * 0.6745 보정)  -> 상수는 균일노이즈 실측 보정치
    constexpr double kScale = 1.2533141 / 6.0 / 0.6745;
    return median * kScale;
}

void ImageQualityEngine::estimateBlur(const cv::Mat& gx, const cv::Mat& gy,
                                      ImageQuality& out) const {
    // 그래디언트 구조텐서의 고유값 비로 방향성을, 총 에너지로 블러 정도를 본다.
    // 모션블러는 특정 방향의 그래디언트만 살아남으므로 이방성이 커진다.
    double sxx = 0.0, syy = 0.0, sxy = 0.0, energy = 0.0;

    for (int y = 0; y < gx.rows; ++y) {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        for (int x = 0; x < gx.cols; ++x) {
            const double a = px[x], b = py[x];
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
            energy += a * a + b * b;
        }
    }
    const double n = static_cast<double>(gx.total());
    if (n < 1.0) return;

    sxx /= n; syy /= n; sxy /= n; energy /= n;

    // 2x2 대칭 텐서 고유값
    const double tr   = sxx + syy;
    const double det  = sxx * syy - sxy * sxy;
    const double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
    const double l1   = tr * 0.5 + disc;   // major
    const double l2   = tr * 0.5 - disc;   // minor

    const double anisotropy = (l1 > kEps) ? clamp01((l1 - l2) / l1) : 0.0;

    // 선명도: 그래디언트 에너지를 기준값으로 정규화
    out.sharpness = clamp01(std::sqrt(energy) / std::sqrt(cfg_.sharpness_ref));

    // 블러 길이: 에너지가 낮을수록, 이방성이 클수록 길게 추정
    const double energy_deficit = 1.0 - out.sharpness;
    out.blur_extent_px = cfg_.blur_extent_max * energy_deficit * (0.4 + 0.6 * anisotropy);
    out.blur_direction_rad = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
    out.blur_free = clamp01(1.0 - out.blur_extent_px / cfg_.blur_extent_max);
}

void ImageQualityEngine::analyzeHistogram(const cv::Mat& gray8u, ImageQuality& out) const {
    constexpr int kBins = 256;
    int hist[kBins] = {0};

    for (int y = 0; y < gray8u.rows; ++y) {
        const std::uint8_t* row = gray8u.ptr<std::uint8_t>(y);
        for (int x = 0; x < gray8u.cols; ++x) ++hist[row[x]];
    }
    const double total = static_cast<double>(gray8u.total());
    if (total < 1.0) return;

    // 평균/표준편차
    double mean = 0.0;
    for (int i = 0; i < kBins; ++i) mean += static_cast<double>(i) * hist[i];
    mean /= total;

    double var = 0.0;
    for (int i = 0; i < kBins; ++i) {
        const double d = static_cast<double>(i) - mean;
        var += d * d * hist[i];
    }
    var /= total;

    out.brightness = clamp01(mean / 255.0);
    out.contrast   = clamp01(std::sqrt(var) / 64.0);   // sigma 64 를 만점 기준으로

    // 포화 비율
    int high = 0, low = 0;
    for (int i = static_cast<int>(cfg_.sat_high_thresh); i < kBins; ++i) high += hist[i];
    for (int i = 0; i <= static_cast<int>(cfg_.sat_low_thresh); ++i) low += hist[i];
    out.saturated_high_ratio = static_cast<double>(high) / total;
    out.saturated_low_ratio  = static_cast<double>(low) / total;

    // 노출 적정도: 평균이 중간이고 포화가 적을수록 좋음
    const double mid_penalty = std::abs(out.brightness - 0.45) / 0.45;
    const double sat_penalty = 2.0 * (out.saturated_high_ratio + out.saturated_low_ratio);
    out.exposure = clamp01(1.0 - 0.6 * mid_penalty - sat_penalty);

    // 동적범위: 누적 1%~99% 구간 폭
    const double lo_target = total * 0.01, hi_target = total * 0.99;
    int lo_bin = 0, hi_bin = kBins - 1;
    double acc = 0.0;
    for (int i = 0; i < kBins; ++i) {
        acc += hist[i];
        if (acc >= lo_target) { lo_bin = i; break; }
    }
    acc = 0.0;
    for (int i = kBins - 1; i >= 0; --i) {
        acc += hist[i];
        if (acc >= total - hi_target) { hi_bin = i; break; }
    }
    out.dynamic_range = clamp01(static_cast<double>(hi_bin - lo_bin) / 255.0);
}

void ImageQualityEngine::updateLensModel(const cv::Mat& grad_mag_small) {
    // 렌즈 오염/물방울은 카메라가 움직여도 화면상 같은 자리에서 계속 흐리다.
    // 따라서 "그래디언트의 장기 상위분위수"가 낮은 픽셀 = 오염 후보.
    // 지수이동 최댓값을 쓰면 장면이 잠깐 평탄한 벽을 볼 때의 오검출을 피할 수 있다.
    if (lens_accum_.empty() || lens_accum_.size() != grad_mag_small.size()) {
        grad_mag_small.copyTo(lens_accum_);
        lens_occlusion_ = cv::Mat::zeros(grad_mag_small.size(), CV_32F);
        lens_frames_ = 1;
        return;
    }

    constexpr float kDecay = 0.995f;   // 상위값을 천천히 잊는다
    for (int y = 0; y < lens_accum_.rows; ++y) {
        float*       acc = lens_accum_.ptr<float>(y);
        const float* cur = grad_mag_small.ptr<float>(y);
        for (int x = 0; x < lens_accum_.cols; ++x) {
            acc[x] = std::max(cur[x], acc[x] * kDecay);
        }
    }
    ++lens_frames_;

    // 충분한 관측이 쌓이기 전에는 판정하지 않는다
    if (lens_frames_ < 60) {
        lens_occlusion_.setTo(0.0f);
        return;
    }

    // 전역 중앙값 대비 상대적으로 낮은 장기최댓값을 오염도로 환산
    std::vector<float>& vals = lens_vals_;
    vals.clear();
    vals.reserve(static_cast<std::size_t>(lens_accum_.total()));
    for (int y = 0; y < lens_accum_.rows; ++y) {
        const float* row = lens_accum_.ptr<float>(y);
        vals.insert(vals.end(), row, row + lens_accum_.cols);
    }
    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    const float thresh = std::max(1.0f, vals[mid] * 0.25f);

    for (int y = 0; y < lens_accum_.rows; ++y) {
        const float* acc = lens_accum_.ptr<float>(y);
        float*       occ = lens_occlusion_.ptr<float>(y);
        for (int x = 0; x < lens_accum_.cols; ++x) {
            occ[x] = std::clamp(1.0f - acc[x] / thresh, 0.0f, 1.0f);
        }
    }
    // 산발적 오검출 제거
    cv::medianBlur(lens_occlusion_, lens_occlusion_, 3);
}

void ImageQualityEngine::buildWeightMap(const cv::Mat& gray32f, const cv::Mat& grad_mag,
                                        ImageQuality& out) {
    // 픽셀 가중 = 포화 감쇠 x 렌즈오염 감쇠 x 국소 정보량.
    // 작업 버퍼는 멤버다. create 는 크기/타입이 같으면 아무것도 하지 않으므로
    // 두 번째 프레임부터 할당이 없다.
    cv::Mat& w = weight_scratch_;
    w.create(gray32f.size(), CV_32F);

    const float hi = static_cast<float>(cfg_.sat_high_thresh);
    const float lo = static_cast<float>(cfg_.sat_low_thresh);

    for (int y = 0; y < gray32f.rows; ++y) {
        const float* g = gray32f.ptr<float>(y);
        const float* m = grad_mag.ptr<float>(y);
        float*       o = w.ptr<float>(y);
        for (int x = 0; x < gray32f.cols; ++x) {
            // 포화 영역은 측광 정보가 0 이다
            float sat = 1.0f;
            if (g[x] > hi)      sat = std::max(0.0f, 1.0f - (g[x] - hi) / (255.0f - hi + 1e-3f));
            else if (g[x] < lo) sat = std::max(0.0f, g[x] / (lo + 1e-3f));

            // 그래디언트가 있어야 측광 잔차가 포즈를 구속한다
            const float info = m[x] / (m[x] + 8.0f);
            o[x] = sat * info;
        }
    }

    // 렌즈 오염 감쇠
    if (!lens_occlusion_.empty()) {
        cv::resize(lens_occlusion_, occ_scratch_, w.size(), 0, 0, cv::INTER_LINEAR);
        for (int y = 0; y < w.rows; ++y) {
            float*       o = w.ptr<float>(y);
            const float* c = occ_scratch_.ptr<float>(y);
            for (int x = 0; x < w.cols; ++x) o[x] *= (1.0f - c[x]);
        }
    }

    if (cfg_.weight_map_width > 0 && w.cols > cfg_.weight_map_width) {
        const int h = std::max(1, w.rows * cfg_.weight_map_width / w.cols);
        cv::resize(w, out.weight_map, cv::Size(cfg_.weight_map_width, h), 0, 0, cv::INTER_AREA);
    } else {
        // w 는 재사용 버퍼라 다음 프레임에 덮어쓴다. 얕은 대입으로 넘기면
        // 반환된 가중맵이 그때 같이 바뀐다. 값은 같고 소유만 분리한다.
        w.copyTo(out.weight_map);
    }
}

ImageQuality ImageQualityEngine::evaluate(const Frame& frame) {
    ImageQuality q;

    cv::Mat gray = frame.gray;
    if (gray.empty()) {
        if (frame.rgb.empty()) return q;
        cv::cvtColor(frame.rgb, gray, cv::COLOR_BGR2GRAY);
    }

    resizeToWidth(gray, small_gray_, cfg_.analysis_width);
    small_gray_.convertTo(small_f_, CV_32F);

    // Scharr 커널의 게인은 32 다. 정규화하지 않으면 그래디언트가 수백 단위가 되어
    // sharpness_ref(900) 도, buildWeightMap 의 m/(m+8) 도 항상 포화한다.
    // 실제로 그 상태에서는 선명도가 어떤 영상에서든 1.0 이었고 모션블러 채널과
    // 정보량 가중이 통째로 죽어 있었다. 1/32 을 곱해 단위를 gray/px 로 맞춘다.
    constexpr double kScharrGain = 1.0 / 32.0;
    cv::Scharr(small_f_, gx_, CV_32F, 1, 0, kScharrGain);
    cv::Scharr(small_f_, gy_, CV_32F, 0, 1, kScharrGain);
    cv::magnitude(gx_, gy_, mag_);

    analyzeHistogram(small_gray_, q);
    estimateBlur(gx_, gy_, q);

    q.noise_sigma = estimateNoiseSigma(small_f_);
    q.noise_free  = clamp01(1.0 - q.noise_sigma / cfg_.noise_sigma_max);

    updateLensModel(mag_);
    q.occlusion_free = 1.0;
    if (!lens_occlusion_.empty()) {
        q.occlusion_free = clamp01(1.0 - cv::mean(lens_occlusion_)[0]);
    }

    buildWeightMap(small_f_, mag_, q);

    // 종합 점수. 곱이 아니라 가중합인 이유: 한 지표가 0 이어도 나머지 정보로
    // 부분적 추정이 가능하며, 하드 게이팅은 상위 엔진이 임계로 처리한다.
    q.score = clamp01(cfg_.w_sharpness * q.sharpness +
                      cfg_.w_exposure  * q.exposure  +
                      cfg_.w_contrast  * q.contrast  +
                      cfg_.w_noise     * q.noise_free +
                      cfg_.w_occlusion * q.occlusion_free);
    return q;
}

}  // namespace wme
