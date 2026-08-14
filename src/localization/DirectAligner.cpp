#include "wme/localization/DirectAligner.hpp"

#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace wme {
namespace {

// 경계 여유 1픽셀을 두고 이중선형 샘플. 세 맵을 같은 가중치로 한 번에 읽는다.
inline bool sample3(const cv::Mat& a, const cv::Mat& b, const cv::Mat& c,
                    float x, float y, float& va, float& vb, float& vc) {
    if (x < 1.0f || y < 1.0f ||
        x >= static_cast<float>(a.cols - 2) || y >= static_cast<float>(a.rows - 2)) {
        return false;
    }
    const int   ix = static_cast<int>(x), iy = static_cast<int>(y);
    const float fx = x - static_cast<float>(ix), fy = y - static_cast<float>(iy);

    const float w00 = (1.f - fx) * (1.f - fy), w10 = fx * (1.f - fy);
    const float w01 = (1.f - fx) * fy,         w11 = fx * fy;

    const auto lerp = [&](const cv::Mat& m) {
        const float* r0 = m.ptr<float>(iy) + ix;
        const float* r1 = m.ptr<float>(iy + 1) + ix;
        return w00 * r0[0] + w10 * r0[1] + w01 * r1[0] + w11 * r1[1];
    };
    va = lerp(a); vb = lerp(b); vc = lerp(c);
    return true;
}

}  // namespace

DirectAligner::DirectAligner(DirectAlignerConfig config, ThreadPool* pool)
    : cfg_(config), pool_(pool) {
    cfg_.pyramid_levels    = std::max(1, cfg_.pyramid_levels);
    cfg_.reduction_blocks  = std::max<std::size_t>(1, cfg_.reduction_blocks);
    cfg_.info_cluster_grid = std::clamp(cfg_.info_cluster_grid, 1, 64);
}

void DirectAligner::buildDepthPyramid(const cv::Mat& depth, int levels) {
    depth_pyramid_.resize(static_cast<std::size_t>(levels));
    depth.convertTo(depth_pyramid_[0], CV_32F);

    // 깊이는 평균내면 물리적으로 틀린 값이 나온다 - 엣지에서 전경과 배경 사이
    // 허공에 점이 생긴다. 그렇다고 INTER_NEAREST 를 쓰면 좌표 규약이 어긋난다:
    // 밝기는 INTER_AREA 라 출력 x 가 입력 2x+0.5 를 대표하는데, 최근접은 2x 를
    // 집는다. 레벨마다 반 픽셀씩 밀리고, 평면에서는 무해하지만 깊이 경계에서는
    // 전경 깊이를 배경 픽셀에 붙여 버린다. 실장면에서 이 점들이 곧 ECDA 가 고르는
    // 고그래디언트 픽셀이다.
    //
    // 2x2 블록의 유효값 중 중앙값을 고른다. 중심이 2x+0.5 로 밝기와 맞고,
    // 실제 관측된 값만 쓰므로 없는 깊이를 지어내지 않는다.
    for (int l = 1; l < levels; ++l) {
        const cv::Mat& src = depth_pyramid_[static_cast<std::size_t>(l - 1)];
        cv::Mat& dst = depth_pyramid_[static_cast<std::size_t>(l)];
        dst.create(src.rows / 2, src.cols / 2, CV_32F);

        for (int y = 0; y < dst.rows; ++y) {
            const float* r0 = src.ptr<float>(2 * y);
            const float* r1 = src.ptr<float>(2 * y + 1);
            float*       o  = dst.ptr<float>(y);
            for (int x = 0; x < dst.cols; ++x) {
                float v[4];
                int n = 0;
                if (r0[2 * x]     > 0.f) v[n++] = r0[2 * x];
                if (r0[2 * x + 1] > 0.f) v[n++] = r0[2 * x + 1];
                if (r1[2 * x]     > 0.f) v[n++] = r1[2 * x];
                if (r1[2 * x + 1] > 0.f) v[n++] = r1[2 * x + 1];
                if (n == 0) { o[x] = 0.f; continue; }
                std::sort(v, v + n);
                o[x] = v[n / 2];
            }
        }
    }
}

bool DirectAligner::depthIsLocallyFlat(const cv::Mat& D, int x, int y, float d) const {
    // 4-이웃 깊이가 상대 임계 안에 모두 들어와야 한다. 하나라도 무효이거나
    // 크게 벌어지면 경계로 본다. 상대값인 이유는 깊이 잡음이 z^2 에 비례해서다.
    if (x <= 0 || y <= 0 || x + 1 >= D.cols || y + 1 >= D.rows) return false;

    const float tol = static_cast<float>(cfg_.depth_edge_ratio) * d;
    const float n[4] = {D.at<float>(y, x - 1), D.at<float>(y, x + 1),
                        D.at<float>(y - 1, x), D.at<float>(y + 1, x)};
    for (float v : n) {
        if (!(v > 0.f)) return false;
        if (std::abs(v - d) > tol) return false;
    }
    return true;
}

void DirectAligner::selectPoints(const ImagePyramid& pyr, const cv::Mat& static_mask,
                                 const ImageQuality* quality, int level) {
    points_.clear();

    const auto  li   = static_cast<std::size_t>(level);
    const cv::Mat& I = pyr.gray[li];
    const cv::Mat& M = pyr.grad_mag[li];
    const cv::Mat& D = depth_pyramid_[li];

    if (quality_pyramid_.size() <= li) quality_pyramid_.resize(li + 1);
    if (mask_pyramid_.size()    <= li) mask_pyramid_.resize(li + 1);
    cv::Mat& quality_weight = quality_pyramid_[li];
    cv::Mat& mask           = mask_pyramid_[li];

    // 품질 가중맵을 현재 레벨 해상도로 맞춘다.
    // release() 하면 안 된다 - 다음 resize 가 같은 크기여도 버퍼를 새로 잡는다.
    // "비었다" 는 사실은 플래그로 남기고 버퍼는 유지한다.
    has_quality_ = (quality != nullptr && !quality->weight_map.empty());
    if (has_quality_) {
        cv::resize(quality->weight_map, quality_weight, I.size(), 0, 0, cv::INTER_LINEAR);
    }

    // 동적 객체 마스크 (0 = 동적). 고전 직접법이 못 풀던 부분을 토큰이 해결한다.
    has_mask_ = !static_mask.empty();
    if (has_mask_) {
        cv::resize(static_mask, mask, I.size(), 0, 0, cv::INTER_NEAREST);
    }

    // 격자 크기는 반드시 레벨에 따라 줄인다. 고정 크기를 쓰면 거친 레벨의
    // 셀 수가 min_points 아래로 떨어져 그 레벨이 통째로 건너뛰어진다.
    // 320x240 에 grid_cell=8 이면 레벨 3(40x30)은 셀이 15개뿐이라 항상 버려지고,
    // 결과적으로 수렴 반경을 넓혀주는 바로 그 레벨들이 사라진다.
    // 이 결함은 한동안 '직접법의 본질적 회전/병진 모호성'으로 오진되어 있었다.
    // Python 참조 구현(python/wme/localization/ecda.py)에서 실측 검증된 수정이다.
    const int cell = std::max(2, cfg_.grid_cell >> level);
    const auto min_grad = static_cast<float>(cfg_.min_gradient);

    for (int cy = 0; cy + 1 < I.rows; cy += cell) {
        const int y_end = std::min(cy + cell, I.rows - 1);
        for (int cx = 0; cx + 1 < I.cols; cx += cell) {
            const int x_end = std::min(cx + cell, I.cols - 1);

            // 셀 안에서 가장 강한 그래디언트 1점만 취한다.
            // 텍스처가 몰린 영역이 전체 시스템을 지배하는 것을 막고
            // 공간적으로 고르게 퍼진 점 집합이 조건수를 좋게 만든다.
            float best_score = min_grad;
            int   best_x = -1, best_y = -1;

            for (int y = std::max(1, cy); y < y_end; ++y) {
                const float*        mrow = M.ptr<float>(y);
                const float*        drow = D.ptr<float>(y);
                const std::uint8_t* krow = has_mask_ ? mask.ptr<std::uint8_t>(y) : nullptr;
                const float*        wrow = has_quality_ ? quality_weight.ptr<float>(y) : nullptr;

                for (int x = std::max(1, cx); x < x_end; ++x) {
                    const float d = drow[x];
                    if (!(d > static_cast<float>(cfg_.min_depth)) ||
                        d > static_cast<float>(cfg_.max_depth)) continue;
                    if (krow != nullptr && krow[x] == 0) continue;

                    // 깊이 경계는 버린다.
                    // ECDA 는 셀에서 가장 강한 그래디언트를 고르는데, 실장면에서
                    // 그건 대개 물체 윤곽이고 거기 깊이는 전경/배경 중 무엇이
                    // 잡혔는지 알 수 없다. 그 점의 3D 위치가 통째로 틀리면
                    // 잔차가 아니라 포즈가 끌려간다. 합성 평면 장면에는 이런
                    // 자리가 아예 없어서 여기서만 드러나는 결함이다.
                    if (!depthIsLocallyFlat(D, x, y, d)) continue;

                    // 품질 가중을 선택 단계에서부터 반영한다.
                    // 포화된 픽셀은 그래디언트가 커 보여도 정보가 없다.
                    const float w = (wrow != nullptr) ? wrow[x] : 1.0f;
                    if (w < 0.05f) continue;

                    const float score = mrow[x] * w;
                    if (score > best_score) {
                        best_score = score; best_x = x; best_y = y;
                    }
                }
            }
            if (best_x < 0) continue;

            AlignPoint p;
            p.u         = static_cast<float>(best_x);
            p.v         = static_cast<float>(best_y);
            p.intensity = I.at<float>(best_y, best_x);
            p.depth     = D.at<float>(best_y, best_x);
            p.weight    = has_quality_ ? quality_weight.at<float>(best_y, best_x) : 1.0f;
            points_.push_back(p);
        }
    }

    // 상한 초과 시 균등 간격으로 솎아낸다 (공간 분포를 유지하기 위해 무작위 아님)
    if (points_.size() > cfg_.max_points) {
        const std::size_t stride = (points_.size() + cfg_.max_points - 1) / cfg_.max_points;
        thin_scratch_.clear();
        thin_scratch_.reserve(cfg_.max_points);
        for (std::size_t i = 0; i < points_.size(); i += stride) thin_scratch_.push_back(points_[i]);
        points_.swap(thin_scratch_);   // 두 버퍼가 서로 자리를 바꿔 가며 재사용된다
    }
}

void DirectAligner::accumulate(const cv::Mat& cur_gray, const cv::Mat& cur_gx,
                               const cv::Mat& cur_gy, const CameraIntrinsics& K,
                               const SE3& T, double affine_a, double affine_b,
                               double huber, double health, double photo_sigma,
                               std::size_t lo, std::size_t hi, Accumulator& acc,
                               int cluster_grid, double depth_sigma_scale) const {
    const double fx = K.fx, fy = K.fy, cx = K.cx, cy = K.cy;
    const double min_z = cfg_.min_depth;

    // 클러스터는 ref 영상 좌표 기준 격자다. 워프된 위치가 아니라 원래 위치로
    // 나눠야 "이웃 픽셀끼리 잔차가 같이 틀린다" 는 구조를 담을 수 있다.
    const double cw = (cluster_grid > 0)
                          ? static_cast<double>(cur_gray.cols) / cluster_grid : 0.0;
    const double ch = (cluster_grid > 0)
                          ? static_cast<double>(cur_gray.rows) / cluster_grid : 0.0;

    Eigen::Matrix<double, 8, 1> J;

    for (std::size_t i = lo; i < hi; ++i) {
        const AlignPoint& p = points_[i];

        // ref 카메라 좌표로 역투영
        const double d = p.depth;
        const Vec3 P((static_cast<double>(p.u) - cx) * d / fx,
                     (static_cast<double>(p.v) - cy) * d / fy,
                     d);
        const Vec3 Q = T * P;
        if (Q.z() < min_z) continue;

        const double inv_z = 1.0 / Q.z();
        const float  u2 = static_cast<float>(fx * Q.x() * inv_z + cx);
        const float  v2 = static_cast<float>(fy * Q.y() * inv_z + cy);

        float I2 = 0.f, gx = 0.f, gy = 0.f;
        if (!sample3(cur_gray, cur_gx, cur_gy, u2, v2, I2, gx, gy)) continue;

        const double r = static_cast<double>(I2) -
                         (affine_a * static_cast<double>(p.intensity) + affine_b);
        const double abs_r = std::abs(r);
        abs_res_[i] = static_cast<float>(abs_r);

        // Huber: 큰 잔차의 영향을 선형으로 제한
        const double w_huber = (abs_r <= huber) ? 1.0 : huber / abs_r;
        double w = static_cast<double>(p.weight) * w_huber;
        if (w <= 0.0) continue;

        // d(pixel)/d(point) 에 이미지 그래디언트를 곱한 1x3 행
        const double gx_fx = static_cast<double>(gx) * fx * inv_z;
        const double gy_fy = static_cast<double>(gy) * fy * inv_z;
        const Vec3 Jp(gx_fx, gy_fy, -(gx_fx * Q.x() + gy_fy * Q.y()) * inv_z);

        // 깊이 불확실성을 잔차 분산으로 옮긴다.
        //   Q = R*P + t,  P = d * ray  ->  dQ/dd = R*P/d = (Q - t)/d
        //   dr/dd = Jp . (Q - t)/d,   sigma_d = c * d^2
        //   var_depth = (dr/dd)^2 * sigma_d^2 = (Jp.(Q-t))^2 * c^2 * d^2
        // 무게는 sigma_I^2 / (sigma_I^2 + var_depth). 깊이가 확실하면 1 로 가고,
        // 먼 점처럼 깊이가 잔차를 지배하면 0 으로 간다.
        //
        // 이것은 이상치 제거가 아니다. 이상치는 huber 가 잔차 크기로 자르고,
        // 여기는 잔차가 **작아도** 깊이가 못 미더우면 무게를 내린다 - 두 신호가
        // 다르다. 먼 벽의 잔차는 대개 작다.
        //
        // depth_sigma_scale 은 이 프레임이 **측정한** 깊이 오차가 센서 모델보다
        // 얼마나 큰지다 (align() 의 해당 주석). 1 이면 모델 그대로.
        if (photo_sigma > 0.0 && cfg_.depth_sigma_rel > 0.0) {
            const double dr_dd = Jp.dot(Q - T.translation()) / d;
            const double sigma_d = depth_sigma_scale * cfg_.depth_sigma_rel * d * d;
            const double var_depth = (dr_dd * sigma_d) * (dr_dd * sigma_d);
            const double photo_var = photo_sigma * photo_sigma;
            w *= photo_var / (photo_var + var_depth);
            if (w <= 0.0) continue;
        }

        // 좌측 섭동 T <- exp(dxi) T 에 대해
        //   dQ/d(rho) = I,  dQ/d(phi) = -skew(Q)
        // 따라서 회전 성분 자코비안은 Q x Jp 가 된다.
        J.segment<3>(0) = Jp;
        J.segment<3>(3) = Q.cross(Jp);
        J(6) = -static_cast<double>(p.intensity);
        J(7) = -1.0;

        acc.H.noalias() += w * (J * J.transpose());
        acc.b.noalias() += w * J * r;
        acc.chi2 += w * r * r;
        acc.wsum += w;
        ++acc.used;

        if (cluster_grid > 0) {
            const int gx_i = std::clamp(static_cast<int>(p.u / cw), 0, cluster_grid - 1);
            const int gy_i = std::clamp(static_cast<int>(p.v / ch), 0, cluster_grid - 1);
            const auto c = static_cast<std::size_t>(gy_i * cluster_grid + gx_i);
            scores_[c].noalias() += w * J * r;
            ++score_count_[c];
        }

        // inlier 는 반드시 *고정된 물리 임계* 로 센다.
        // 적응형 커널 임계로 세면 동어반복이 된다 - 잔차가 커지면 임계도 같이
        // 넓어져 비율이 그대로 높게 유지된다. 실측에서 이 값은 실제 오차와
        // 오히려 반대로 움직였다(TUM fr3_sitting lift 0.00). 건강 신호로 쓰려면
        // 기준이 상황에 따라 움직이면 안 된다.
        if (abs_r <= health) ++acc.inliers;
    }
}

double DirectAligner::robustDelta(double noise_sigma) const {
    res_scratch_.clear();
    for (float v : abs_res_) {
        if (v >= 0.f) res_scratch_.push_back(v);
    }
    // 표본이 적으면 산포 추정 자체가 이상치에 휘둘린다. 그럴 땐 L2 로 둔다.
    if (res_scratch_.size() < 32) return std::numeric_limits<double>::infinity();

    const std::size_t mid = res_scratch_.size() / 2;
    std::nth_element(res_scratch_.begin(),
                     res_scratch_.begin() + static_cast<std::ptrdiff_t>(mid),
                     res_scratch_.end());

    // 아핀 보정 후 잔차는 0 중심이므로 median(|r|) 이 곧 MAD 다.
    const double sigma = 1.4826 * static_cast<double>(res_scratch_[mid]);

    // 잔차가 잡음 수준보다 훨씬 크면 아직 해에 도달하지 못한 것이고,
    // 그 상태의 "이상치" 판정은 의미가 없다. 임계를 그만큼 넓혀 L2 로 돌린다.
    const double noise = std::max(cfg_.huber_min_delta, noise_sigma);
    const double relax = std::max(1.0, sigma / (cfg_.huber_noise_ratio * noise));

    return std::max(cfg_.huber_min_delta, cfg_.huber_k * sigma * relax);
}

Mat6 DirectAligner::poseInformation(const Accumulator& acc, double sensor_var) const {
    // 아핀을 포함한 8x8 헤시안. 사전분포까지 넣어야 실제로 푼 문제와 같아진다.
    Eigen::Matrix<double, 8, 8> H8 = acc.H;
    if (cfg_.estimate_affine) {
        H8(6, 6) += cfg_.affine_prior_weight;
        H8(7, 7) += cfg_.affine_prior_weight;
    } else {
        H8.row(6).setZero(); H8.col(6).setZero(); H8(6, 6) = 1.0;
        H8.row(7).setZero(); H8.col(7).setZero(); H8(7, 7) = 1.0;
    }

    // --- 샌드위치 -----------------------------------------------------------
    // 픽셀 잔차는 독립이 아니다. 한 표면의 깊이 오차, 반사율/노출 불일치,
    // 리샘플 오차는 이웃 픽셀에서 같은 방향으로 나타난다. 독립을 가정한
    // Lambda = H/sigma^2 는 N 개 관측을 N 개 증거로 세므로 그만큼 과신한다.
    //
    // 상관 구조를 모델로 정하지 않고 데이터에서 재는 방법이 M-추정의 샌드위치다.
    //   score_c = sum_{i in c} w_i J_i r_i          (해에서 sum_c score_c ~ 0)
    //   M       = G/(G-1) * sum_c score_c score_c^T
    //   Cov     = H^-1 M H^-1
    // 클러스터 *안* 의 상관은 형태를 가리지 않고 그대로 반영되고, 클러스터
    // *사이* 만 독립으로 본다. 잔차가 실제로 독립이면 M -> sigma^2 sum w^2 J J^T
    // 가 되어 기존 식으로 되돌아간다. 항을 더한 게 아니라 가정을 하나 뺀 것이다.
    if (cfg_.information_model == InformationModel::ClusterRobust) {
        int g_ne = 0;
        for (std::uint32_t n : score_count_) if (n > 0) ++g_ne;

        if (g_ne >= cfg_.info_min_clusters) {
            Eigen::Matrix<double, 8, 8> M = Eigen::Matrix<double, 8, 8>::Zero();
            for (std::size_t c = 0; c < scores_.size(); ++c) {
                if (score_count_[c] == 0) continue;
                M.noalias() += scores_[c] * scores_[c].transpose();
            }
            M *= static_cast<double>(g_ne) / static_cast<double>(g_ne - 1);
            if (!cfg_.estimate_affine) { M.row(6).setZero(); M.col(6).setZero();
                                         M.row(7).setZero(); M.col(7).setZero(); }

            const Eigen::LDLT<Eigen::Matrix<double, 8, 8>> ldlt(H8);
            if (ldlt.info() == Eigen::Success) {
                // Cov = H^-1 M H^-1. M 이 대칭이므로 solve 두 번으로 충분하다.
                const Eigen::Matrix<double, 8, 8> A = ldlt.solve(M);
                Eigen::Matrix<double, 8, 8> Cov = ldlt.solve(A.transpose());
                Cov = 0.5 * (Cov + Cov.transpose()).eval();

                // 아핀 주변화 = 공분산의 포즈 블록을 그냥 떼는 것.
                // (정보행렬 쪽 Schur 보수와 같은 양이지만 이쪽이 정의 그대로다.)
                const Mat6 Cp = Cov.block<6, 6>(0, 0);
                Eigen::SelfAdjointEigenSolver<Mat6> es(Cp);
                if (es.info() == Eigen::Success) {
                    // 부호만 보면 안 되고 조건수를 봐야 한다. 클러스터가 모자라면
                    // M 의 랭크가 8 아래로 떨어지는데, 그때 Cp 의 최소 고유값은
                    // 0 이 아니라 반올림 오차만큼의 *양수* 로 나온다. 부호 검사는
                    // 통과하고 역행렬에서 그 방향에 무한한 확신이 실린다.
                    // 실측: 클러스터 4 개에서 ANEES 가 1e16 까지 튀었다.
                    const double lo = es.eigenvalues()(0);
                    const double hi = es.eigenvalues()(5);
                    if (hi > 0.0 && lo > 1e-9 * hi) return Cp.inverse();
                }
                // 여기까지 오면 아래 독립 가정 경로로 떨어진다.
            }
        }
    }

    // --- Lambda = H_pose / (chi2 / N_eff) -----------------------------------
    // 아핀을 주변화한 포즈 정보 (Schur 보수).
    Mat6 Lambda = acc.H.block<6, 6>(0, 0);
    if (cfg_.estimate_affine) {
        const Eigen::Matrix2d Haa = H8.block<2, 2>(6, 6);
        const Eigen::Matrix<double, 6, 2> Hpa = acc.H.block<6, 2>(0, 6);
        Lambda.noalias() -= Hpa * Haa.inverse() * Hpa.transpose();
    }

    // 분모는 "독립 관측 하나가 갖는 잔차 분산" 이다.
    //   ResidualVariance : N_eff = sum(w)            픽셀 하나하나가 독립 증거
    //   EffectiveSample  : N_eff = effective_samples 프레임 전체가 관측 nu 개
    //   CoherentFrame    : 분산 자체를 상수로 둔다 (coherent_sigma 주석)
    double var = std::max(1e-6, sensor_var);
    if (cfg_.information_model == InformationModel::CoherentFrame) {
        // 프레임 = 관측 하나, 그 관측의 잔차 분산은 chi2/N 이 아니라 상수다.
        // N 은 rmse 와 같은 분모(acc.used)를 써야 한다. photometric_rmse 가
        // chi2/used 이므로 이렇게 두면 EffectiveSample 대비 바뀌는 것이
        // "rmse^2 -> coherent_sigma^2" 하나뿐이 되어 비교가 깨끗하다.
        const double n = static_cast<double>(std::max<std::size_t>(1, acc.used));
        var = std::max(var, n * cfg_.coherent_sigma * cfg_.coherent_sigma);
    } else if (cfg_.information_model != InformationModel::SensorVariance) {
        const double n_eff =
            (cfg_.information_model == InformationModel::ResidualVariance)
                ? acc.wsum
                : std::max(1e-9, cfg_.effective_samples);
        if (n_eff > 0.0) {
            // 센서 잡음을 바닥으로 깐다. 잔차 분산이 측정된 센서 잡음보다 작을
            // 수는 없고, 이 바닥이 없으면 합성 장면처럼 잔차가 0 에 가까울 때
            // Lambda 가 발산한다.
            var = std::max(var, acc.chi2 / n_eff);
        }
    }
    return Lambda / var;
}

Result<AlignmentResult> DirectAligner::align(const Frame& ref, const Frame& cur, const SE3& init,
                                             const ImageQuality* quality,
                                             const EnvironmentState* env,
                                             const MotionPrior* prior) {
    if (!ref.valid() || !cur.valid()) {
        return {ErrorCode::InvalidArgument, "프레임이 유효하지 않음"};
    }
    if (!ref.hasDepth()) {
        return {ErrorCode::InsufficientData, "ECDA 는 ref 프레임의 깊이를 요구"};
    }

    // --- 로버스트 커널 중재 --------------------------------------------------
    // 같은 프레임을 완화 커널과 비완화 커널로 각각 풀고, 측광이 보지 않은 관측
    // (cur 깊이맵)이 더 낫다고 하는 쪽을 택한다. 근거는 헤더 robust_arbitration.
    //
    // 재귀로 부르는 이유: 두 번의 풀이는 완전히 같은 코드여야 한다. 여기서 루프를
    // 복제하면 두 경로가 언젠가 어긋나고, 그 차이가 커널 차이로 오독된다.
    // 내부 버퍼(points_, blocks_ ...)는 중첩 호출이 끝난 뒤에야 다시 쓰이므로
    // 결과를 먼저 복사해 두면 안전하다.
    if (cfg_.robust_arbitration) {
        DirectAlignerConfig relaxed = cfg_, tight = cfg_;
        relaxed.robust_arbitration = false;
        tight.robust_arbitration = false;
        // 완화를 끄는 것은 noise_ratio 를 무한대로 두는 것과 같다.
        // relax = max(1, sigma / (inf * noise)) = 1 이므로 delta = huber_k * sigma,
        // 즉 로버스트 통계가 원래 의도한 임계 그대로다.
        tight.huber_noise_ratio = std::numeric_limits<double>::infinity();

        cfg_ = relaxed;
        Result<AlignmentResult> r_relaxed = align(ref, cur, init, quality, env);
        cfg_ = relaxed;
        cfg_.robust_arbitration = true;

        // --- 두 번째 풀이를 물어볼 이유가 있을 때만 --------------------------
        // 아래 교체 조건의 (1) 은 c_relaxed > floor 다. 그것이 거짓이면 대안이
        // 무엇이든 교체가 일어나지 않으므로 tight 를 푼 결과는 쓰이지 않는다.
        // 즉 이 게이트는 최적화가 아니라 **교체 조건 (1) 을 앞으로 옮긴 것**
        // 이고, 걸러지는 프레임은 정의상 교체가 불가능한 프레임뿐이다.
        // 궤적은 비트 단위로 같다.
        //
        // c_relaxed > floor 는 depth_sigma_scale(rho) > 1 과 같은 판정이다 -
        // rho = max(1, c/floor) 이므로. 실측 발동률은 kitti_04 39 %,
        // kitti_00/05/07 은 9~14 % 라 대부분의 프레임에서 풀이가 한 번으로 준다.
        //
        // 풀리지 않은 경우(!ok)는 값이 없어 판정할 수 없다. 그때는 아래의
        // "완화가 실패하면 tight 를 쓴다" 경로가 살아 있어야 하므로 풀어 본다.
        const bool ask_tight =
            !r_relaxed.ok() ||
            r_relaxed.value().depth_consistency >
                r_relaxed.value().depthNoiseFloor(cfg_.depth_sigma_rel);
        if (!ask_tight) {
            // 심판이 물어볼 필요가 없었다는 사실 자체를 남긴다. tight 는 풀지
            // 않았으므로 -1(판정 불가)로 둔다 - 0 으로 적으면 "풀었는데 0" 과
            // 구분되지 않는다.
            r_relaxed.value().arb_c_relaxed = r_relaxed.value().depth_consistency;
            return r_relaxed;
        }

        cfg_ = tight;
        Result<AlignmentResult> r_tight = align(ref, cur, init, quality, env);
        cfg_ = relaxed;
        cfg_.robust_arbitration = true;

        // --- 세 번째 후보: 운동 사전분포 ------------------------------------
        // 여기까지 온 프레임은 정의상 rho > 1 이다 (게이트가 그 판정이다).
        // 그 rho 로 사전분포를 키워 같은 프레임을 한 번 더 푼다. 채택 여부는
        // 아래 심판이 정하므로, 사전분포 세기가 틀려도 틀린 답이 들어오지는
        // 않는다. 근거는 헤더 motion_prior_arbitration.
        Result<AlignmentResult> r_prior{ErrorCode::InsufficientData, "사전분포 후보 없음"};
        bool have_prior_cand = false;
        if (cfg_.motion_prior_arbitration && prior != nullptr && prior->valid() &&
            r_relaxed.ok()) {
            prior_     = prior;
            prior_rho_ = r_relaxed.value().depth_sigma_scale;
            cfg_ = relaxed;
            r_prior = align(ref, cur, init, quality, env);
            prior_     = nullptr;
            prior_rho_ = 1.0;
            cfg_ = relaxed;
            cfg_.robust_arbitration = true;
            have_prior_cand = r_prior.ok();
        }

        // 심판이 판정하지 못하면 기존 동작(완화)을 그대로 둔다. "모른다" 를
        // "저쪽이 낫다" 로 접지 않는다.
        if (r_relaxed.ok() && r_tight.ok()) {
            const double c_relaxed = r_relaxed.value().depth_consistency;
            const double c_tight   = r_tight.value().depth_consistency;
            const double delta_t = (r_tight.value().T_cur_ref.translation() -
                                    r_relaxed.value().T_cur_ref.translation()).norm();
            for (auto* v : {&r_relaxed, &r_tight}) {
                v->value().arb_delta_t   = delta_t;
                v->value().arb_c_relaxed = c_relaxed;
                v->value().arb_c_tight   = c_tight;
            }
            // 교체 조건은 두 개고 둘 다 같은 바닥에 대한 판정이다.
            //   (1) 기본 답이 바닥을 넘었다            - 물어볼 이유가 있다
            //   (2) 대안이 바닥 안에 들어온다          - 답할 자격이 있다
            //   (3) 둘의 차이가 중앙값의 표준오차 위다 - 구분되는 둘이다
            //
            // (2) 를 "그냥 더 낫다" 로 느슨하게 두면 안 된다는 것이 실측이다.
            // 나쁜 둘 중 덜 나쁜 쪽을 고르는 일이 생기고, 그 한 번이 되돌아오지
            // 않는다: kitti_07 은 320~379 프레임에서 차가 서 있는데(정답 프레임간
            // 이동 0.01~0.36 m), 감속이 시작되는 한 프레임에서 잘못 고른 뒤 그
            // 구간 전체가 발산해 ATE 108 -> 5014 cm 가 됐다. 기본 경로는 같은
            // 구간을 depth_consistency 0.0075~0.0082 로 멀쩡히 지난다.
            // 바닥을 넘은 두 후보 사이에서는 이 채널이 순위를 매길 근거가 없다.
            //
            // (3) 이 없으면 구분되지 않는 둘 중 하나를 잡음으로 고른다.
            // 근거와 실측은 depthMedianResolution 주석에 있다.
            const double floor   = r_relaxed.value().depthNoiseFloor(cfg_.depth_sigma_rel);
            const double resolve =
                r_relaxed.value().depthMedianResolution(cfg_.depth_sigma_rel);

            const double c_prior = have_prior_cand ? r_prior.value().depth_consistency : -1.0;
            for (auto* v : {&r_relaxed, &r_tight}) v->value().arb_c_prior = c_prior;

            // 후보가 셋이 되어도 조건 (1)(2)(3) 은 그대로다. 다만 조건(2)
            // "대안이 바닥 안에 들어온다" 를 어디까지 요구할 것인가가 남는다.
            //
            // 원래 근거는 "바닥을 넘은 두 후보 사이에서는 이 채널이 순위를 매길
            // 근거가 없다" 였고, 실측 실패(kitti_07 108 -> 5014)가 그것을 뒷받침
            // 했다. 그런데 그 실패의 후보들은 포즈가 1~10 mm 차이에 c 가 소수
            // 넷째 자리까지 같았다 - 구분되지 않는 둘이었다. 그 뒤에 들어온 조건(3)
            // 이 정확히 그것을 잡는다.
            //
            // 다만 조건(3)의 분해능 depthMedianResolution() 은
            //   se = sqrt(2)*sqrt(pi/2) * floor / sqrt(N)
            // 이고, 여기서 floor 는 **모델** 산포다. 이 프레임이 실제로 잰 산포는
            // 그 rho 배이므로 중앙값의 표준오차도 rho 배다. 바닥 밖에서 순위를
            // 매기려면 그 자에 맞춰 바를 rho 배로 올려야 한다 - 안 올리면 rho 가
            // 큰(=가장 못 믿을) 프레임에서 바가 가장 낮아지는 거꾸로가 된다.
            //
            // 시도했고 버린 것: N 을 표본 수가 아니라 공간 독립 단위 수
            // (info_cluster_grid^2 = 64)로 바꿔 바를 50 배 더 올려 봤다. 통계적
            // 근거는 더 낫지만(깊이 오차는 화소가 아니라 면 단위로 상관된다)
            // 실측이 나빠졌다: kitti_04 222 -> 211, kitti_05 147 -> 144 로 좋아진
            // 대신 kitti_07 110 -> 187 이었다. 유도되지 않은 보정을 얹어 결과가
            // 나빠졌으므로 넣지 않는다.
            const double rho_c = (floor > 0.0) ? std::max(1.0, c_relaxed / floor) : 1.0;
            const double resolve_above = rho_c * resolve;
            const auto passes = [&](double c) {
                if (c < 0.0) return false;
                if (c <= floor) return c_relaxed - c > resolve;
                // 바닥 밖: 실제 산포로 잰 분해능을 넘어야 한다.
                return cfg_.arb_rank_above_floor && (c_relaxed - c > resolve_above);
            };
            const bool tight_ok = c_relaxed > floor && passes(c_tight);
            const bool prior_ok = c_relaxed > floor && have_prior_cand && passes(c_prior);

            if (tight_ok || prior_ok) {
                // 둘 다 통과했으면 심판이 더 낫다고 한 쪽. 다만 그 차이도 구분
                // 가능해야 한다 - 아니면 기존 동작(조임)을 유지한다. 자는 둘이
                // 놓인 자리에 맞춘다.
                const double tie = (c_tight <= floor && c_prior <= floor) ? resolve
                                                                         : resolve_above;
                const bool take_prior =
                    prior_ok && (!tight_ok || c_tight - c_prior > tie);
                AlignmentResult v = take_prior ? r_prior.value() : r_tight.value();
                v.arbitrated   = true;
                v.arb_choice   = take_prior ? 2 : 1;
                v.arb_delta_t  = delta_t;
                v.arb_c_relaxed = c_relaxed;
                v.arb_c_tight   = c_tight;
                v.arb_c_prior   = c_prior;
                return Result<AlignmentResult>{v};
            }
        } else if (!r_relaxed.ok() && r_tight.ok()) {
            // 완화 쪽이 아예 풀리지 않았으면 풀린 쪽을 쓴다.
            return r_tight;
        }
        return r_relaxed;
    }

    // 로버스트 임계의 기준이 되는 물리 잡음. 품질 엔진이 측정한 값을 그대로 쓴다.
    const double noise_sigma = (quality != nullptr && quality->noise_sigma > 0.0)
                                   ? quality->noise_sigma
                                   : cfg_.huber_min_delta;

    // inlier 집계용 고정 임계. 최적화 중 임계가 어떻게 움직이든 이 값은 그대로다.
    const double health_delta =
        cfg_.health_k * std::max(cfg_.huber_min_delta, noise_sigma);

    const int levels = cfg_.pyramid_levels;

    // 피라미드가 미리 만들어져 있지 않으면 여기서 만든다.
    // 실제 파이프라인에서는 프레임 생성 시 한 번만 만들어 재사용한다.
    ImagePyramid ref_owned, cur_owned;
    const ImagePyramid* rp = &ref.pyramid;
    const ImagePyramid* cp = &cur.pyramid;
    if (rp->levels() < levels) {
        ref_owned = ImagePyramid::build(ref.gray, ref.intrinsics, levels);
        rp = &ref_owned;
    }
    if (cp->levels() < levels) {
        cur_owned = ImagePyramid::build(cur.gray, cur.intrinsics, levels);
        cp = &cur_owned;
    }

    buildDepthPyramid(ref.depth, levels);

    SE3    T = init;
    double a = 1.0, b = 0.0;

    AlignmentResult result;
    points_per_level_.assign(static_cast<std::size_t>(levels), 0);

    Accumulator final_acc;
    bool any_level_converged = false;

    // final_acc 를 낸 레벨과 그때의 커널 임계. 아래에서 깊이 잡음 배율을 바꿔
    // 정보행렬만 다시 쌓을 때 **같은 조건** 으로 쌓아야 비교가 성립한다.
    int    final_level = -1;
    double final_huber = std::numeric_limits<double>::infinity();

    // 거친 레벨부터 세밀한 레벨로. 큰 변위는 거친 레벨에서만 잡힌다.
    for (int level = levels - 1; level >= 0; --level) {
        selectPoints(*rp, ref.static_mask, quality, level);
        points_per_level_[static_cast<std::size_t>(level)] = points_.size();

        if (points_.size() < cfg_.min_points) continue;   // 이 레벨은 건너뛴다

        const auto li = static_cast<std::size_t>(level);
        const cv::Mat& cg  = cp->gray[li];
        const cv::Mat& cgx = cp->grad_x[li];
        const cv::Mat& cgy = cp->grad_y[li];
        const CameraIntrinsics& K = rp->intrinsics[li];

        const std::size_t nblocks = std::min(cfg_.reduction_blocks, points_.size());
        const std::size_t bsize   = (points_.size() + nblocks - 1) / nblocks;

        // parallelFor 는 std::function 을 받는다. 참조 캡처가 많은 람다를 그대로
        // 넘기면 그 임시 std::function 이 내부 버퍼를 넘겨 호출마다 힙을 잡는다
        // (align 한 번에 parallelFor 가 ~175 회 불린다). 인자를 한 구조체로 묶어
        // 캡처를 포인터 하나로 줄이면 임시 객체가 버퍼 안에 들어간다.
        struct BlockJob {
            const DirectAligner*     self;
            const cv::Mat*           cg;
            const cv::Mat*           cgx;
            const cv::Mat*           cgy;
            const CameraIntrinsics*  K;
            const SE3*               T;
            double                   a, b, huber, health, ps;
            std::size_t              bsize, npoints;
            std::vector<Accumulator>* blocks;

            void run(std::size_t lo_b, std::size_t hi_b) const {
                for (std::size_t k = lo_b; k < hi_b; ++k) {
                    const std::size_t lo = k * bsize;
                    const std::size_t hi = std::min(lo + bsize, npoints);
                    if (lo < hi) {
                        self->accumulate(*cg, *cgx, *cgy, *K, *T, a, b, huber, health, ps,
                                         lo, hi, (*blocks)[k]);
                    }
                }
            }
        };

        // 고정 블록 수로 나눠 누산 순서를 워커 수와 무관하게 만든다 (결정적 재현).
        const auto accumulateAll = [&](const SE3& Tx, double ax, double bx, double hub) {
            blocks_.assign(nblocks, Accumulator{});
            abs_res_.assign(points_.size(), -1.f);
            const BlockJob job{this,  &cg,   &cgx,  &cgy, &K,  &Tx,
                               ax,    bx,    hub,   health_delta, noise_sigma,
                               bsize, points_.size(), &blocks_};
            const auto body = [&job](std::size_t lo_b, std::size_t hi_b) {
                job.run(lo_b, hi_b);
            };
            if (pool_ != nullptr) pool_->parallelFor(0, nblocks, body, 1);
            else                  body(0, nblocks);

            Accumulator total;
            for (const auto& blk : blocks_) total += blk;   // 항상 블록 인덱스 순서로 합산
            return total;
        };

        // 첫 누산은 L2 로 돌려 잔차 산포를 재고, 그 산포로 임계를 정한 뒤
        // 같은 임계로 다시 재서 이후 비교의 기준을 통일한다.
        Accumulator acc = accumulateAll(T, a, b, std::numeric_limits<double>::infinity());
        if (acc.used < cfg_.min_points) continue;

        double huber = robustDelta(noise_sigma);
        acc = accumulateAll(T, a, b, huber);

        // --- 운동 사전분포 항 -----------------------------------------------
        // 음의 로그우도로 쓰면
        //     chi2(T) / (2 sigma_I^2)  +  (1/2) e^T Lambda_p e,
        //     e = log(T * T_pred^-1),  Lambda_p = diag(1/sigma_i^2)
        // 이고, 누산기의 H 는 (1/2)chi2 의 헤시안이므로 양변에 sigma_I^2 를 곱해
        //     H += sigma_I^2 Lambda_p,   g += sigma_I^2 Lambda_p e
        // 가 된다. sigma_I^2 는 상수가 아니라 이 레벨이 실제로 남긴 잔차 분산
        // chi2/wsum 이다 (센서 바닥으로 하한). poseInformation 이 정보행렬을
        // 만들 때 쓰는 분모와 같은 양이라, 사전분포와 측광이 같은 자로 재진다.
        //
        // 여기에 rho^2 를 더 곱한다. rho 는 이 프레임의 깊이가 센서 모델보다 몇
        // 배 나쁜지이고, 깊이가 rho 배 나쁘면 그 깊이로 만든 포즈 구속도 rho 배
        // 부정확하다 - 측광 정보를 rho^2 로 나누는 것과 사전분포에 rho^2 를
        // 곱하는 것은 같은 연산이다. rho = 1 이면 이 항은 사실상 아무 일도 하지
        // 않는다(H 대각의 1 % 미만). 그래서 이 후보는 rho > 1 일 때만 만든다.
        //
        // 좌섭동 규약(T <- exp(dx) T)에서 de/ddx = I 를 1차로 쓴다. 위의
        // 레벨간 투영이 delta = (T * T_start^-1).log() 로 같은 규약을 쓴다.
        const bool use_prior = (prior_ != nullptr && prior_->valid());
        Vec6 prior_lam = Vec6::Zero();
        double prior_w = 0.0;
        if (use_prior) {
            const double sensor_var = (quality != nullptr) ? quality->photometricVariance() : 1.0;
            const double var = std::max(sensor_var,
                                        acc.chi2 / std::max(1e-9, acc.wsum));
            prior_w = var * prior_rho_ * prior_rho_;
            for (int k = 0; k < 6; ++k) {
                const double s = std::max(1e-9, prior_->sigma(k));
                prior_lam(k) = 1.0 / (s * s);
            }
        }
        // 사전분포 비용. 수용 판정이 이 항을 빼먹으면 LM 이 전체 목적함수를
        // 올리는 걸음도 받아들인다.
        const auto priorCost = [&](const SE3& Tx) {
            if (!use_prior) return 0.0;
            const Vec6 e = (Tx * prior_->T_pred.inverse()).log();
            return prior_w * e.dot(prior_lam.cwiseProduct(e));
        };

        const SE3 T_level_start = T;
        double lambda = cfg_.lambda_init;

        for (int it = 0; it < cfg_.max_iterations; ++it) {
            ++result.iterations;

            Eigen::Matrix<double, 8, 8> H = acc.H;
            Eigen::Matrix<double, 8, 1> g = acc.b;

            if (cfg_.estimate_affine) {
                // a=1, b=0 사전분포. 텍스처가 약할 때 아핀이 포즈를 먹어치우는 걸 막는다.
                H(6, 6) += cfg_.affine_prior_weight;
                H(7, 7) += cfg_.affine_prior_weight;
                g(6) += cfg_.affine_prior_weight * (a - 1.0);
                g(7) += cfg_.affine_prior_weight * b;
            } else {
                // 아핀을 고정하려면 해당 자유도를 잠근다
                H.row(6).setZero(); H.col(6).setZero(); H(6, 6) = 1.0; g(6) = 0.0;
                H.row(7).setZero(); H.col(7).setZero(); H(7, 7) = 1.0; g(7) = 0.0;
            }

            if (use_prior) {
                const Vec6 e = (T * prior_->T_pred.inverse()).log();
                for (int k = 0; k < 6; ++k) {
                    H(k, k) += prior_w * prior_lam(k);
                    g(k)    += prior_w * prior_lam(k) * e(k);
                }
            }

            // LM 감쇠는 대각 스케일에 비례시켜 단위 차이(병진/회전/밝기)를 흡수한다.
            // 다만 순수 상대 감쇠는 랭크 부족 방향을 감쇠하지 못하므로,
            // 최대 대각 성분에 비례한 바닥을 깔아 약한 방향을 묶는다.
            //
            // 바닥은 반드시 블록별로 잡는다. 포즈 대각은 (intensity/m)^2 라 1e7 대이고
            // 밝기 오프셋 대각은 sum(w) 라 1e3 대다. 한 덩어리로 비교하면 오프셋이
            // 단위 때문에 '약한 방향' 으로 오인돼 감쇠되고, 노출 변화를 못 흡수한다.
            const auto floorFor = [&](int lo, int hi) {
                double m = 0.0;
                for (int k = lo; k < hi; ++k) m = std::max(m, H(k, k));
                return std::max(1e-9, cfg_.damping_floor_ratio * m);
            };
            const double floor_pose   = floorFor(0, 6);
            const double floor_affine = floorFor(6, 8);
            for (int k = 0; k < 6; ++k) H(k, k) += lambda * std::max(H(k, k), floor_pose);
            for (int k = 6; k < 8; ++k) H(k, k) += lambda * std::max(H(k, k), floor_affine);

            const Eigen::LDLT<Eigen::Matrix<double, 8, 8>> ldlt(H);
            if (ldlt.info() != Eigen::Success) { lambda *= cfg_.lambda_up; continue; }

            const Eigen::Matrix<double, 8, 1> dx = -ldlt.solve(g);
            if (!dx.allFinite()) { lambda *= cfg_.lambda_up; continue; }

            const SE3    T_new = SE3::exp(dx.segment<6>(0)) * T;
            const double a_new = cfg_.estimate_affine ? a + dx(6) : a;
            const double b_new = cfg_.estimate_affine ? b + dx(7) : b;

            // 후보는 반드시 같은 임계로 잰다. 임계가 다르면 비용 비교가 무의미하다.
            Accumulator acc_new = accumulateAll(T_new, a_new, b_new, huber);

            // 평균 잔차로 비교한다. 워프에 따라 유효 점 수가 달라지므로
            // chi2 총합을 그대로 비교하면 시야를 벗어나는 방향으로 최적화된다.
            const double cost_old = (acc.chi2 + priorCost(T)) /
                                    static_cast<double>(std::max<std::size_t>(1, acc.used));
            const double cost_new = (acc_new.chi2 + priorCost(T_new)) /
                                    static_cast<double>(std::max<std::size_t>(1, acc_new.used));

            if (acc_new.used >= cfg_.min_points && cost_new < cost_old) {
                T = T_new; a = a_new; b = b_new;
                acc = acc_new;
                lambda = std::max(1e-9, lambda * cfg_.lambda_down);
                any_level_converged = true;
                if (dx.norm() < cfg_.convergence_delta) break;

                // abs_res_ 에는 방금 T_new 에서 잰 잔차가 그대로 남아 있다.
                // 정렬이 좋아지면 산포가 줄고 임계도 따라 좁아진다.
                // 임계가 바뀌면 비교 기준도 바뀌므로 acc 를 새 임계로 다시 잰다.
                const double h_new = robustDelta(noise_sigma);
                if (h_new < huber * 0.9) {
                    huber = h_new;
                    acc = accumulateAll(T, a, b, huber);
                }
            } else {
                lambda *= cfg_.lambda_up;
                if (lambda > 1e8) break;   // 더 줄일 여지가 없다
            }
        }

        // ClusterRobust 만 클러스터별 스코어가 필요하다. 그때는 H 와 스코어가
        // 같은 패스에서 나와야 같은 점 집합을 가리키므로 최종 해에서 단일
        // 스레드로 한 번 더 훑는다 (LM 반복 한 번에도 못 미치는 비용).
        // 나머지 모델은 acc 만으로 정보행렬이 나오므로 패스를 돌리지 않는다.
        Accumulator fin = acc;
        if (cfg_.information_model == InformationModel::ClusterRobust) {
            scores_.assign(static_cast<std::size_t>(cfg_.info_cluster_grid) *
                               static_cast<std::size_t>(cfg_.info_cluster_grid),
                           Eigen::Matrix<double, 8, 1>::Zero());
            score_count_.assign(scores_.size(), 0u);
            abs_res_.assign(points_.size(), -1.f);

            fin = Accumulator{};
            accumulate(cg, cgx, cgy, K, T, a, b, huber, health_delta, noise_sigma,
                       0, points_.size(), fin, cfg_.info_cluster_grid);
        }

        // --- 관측 가능 부분공간으로 투영 ------------------------------------
        // 이 레벨이 실제로 구속한 방향으로만 포즈를 옮긴다. 나머지 축의 값은
        // 증거가 아니라 감쇠와 수치오차의 산물이므로 다음 레벨에 넘기지 않는다.
        //
        // 레벨 0 은 제외한다. 이 게이트의 목적은 "다음 레벨에 무엇을 넘길지" 이고,
        // 레벨 0 의 소비자는 팩터그래프다. 그쪽은 정보행렬 Lambda 로 약한 방향을
        // 이미 알 수 있으므로, 여기서 또 깎으면 퇴화를 두 번 세는 셈이 된다.
        // 실제로 레벨 0 에도 걸었더니 평면 장면에서 해가 정확히 절반으로 깎였다.
        if (level > 0) {
            // 아핀을 주변화해 포즈만의 정보를 남긴다.
            Mat6 Lam = acc.H.block<6, 6>(0, 0);
            if (cfg_.estimate_affine) {
                Eigen::Matrix2d Haa = acc.H.block<2, 2>(6, 6);
                Haa(0, 0) += cfg_.affine_prior_weight;
                Haa(1, 1) += cfg_.affine_prior_weight;
                const Eigen::Matrix<double, 6, 2> Hpa = acc.H.block<6, 2>(0, 6);
                Lam.noalias() -= Hpa * Haa.inverse() * Hpa.transpose();
            }
            Lam = 0.5 * (Lam + Lam.transpose()).eval();

            // 병진(m)과 회전(rad)은 단위가 달라 고유값을 그대로 비교할 수 없다.
            // 대각으로 정규화하면 단위에 무관한 랭크 판정이 된다.
            Vec6 s_inv;   // delta -> 정규화 좌표
            for (int k = 0; k < 6; ++k) s_inv(k) = std::sqrt(std::max(1e-12, Lam(k, k)));
            const Vec6 s = s_inv.cwiseInverse();
            const Mat6 Lam_n = s.asDiagonal() * Lam * s.asDiagonal();

            Eigen::SelfAdjointEigenSolver<Mat6> es_l(Lam_n);
            if (es_l.info() == Eigen::Success) {
                const Vec6 ev = es_l.eigenvalues();       // 오름차순
                const double thresh = ev(5) * cfg_.level_observable_ratio;

                // 좌측 섭동 규약: T = exp(delta) * T_start
                const Vec6 delta   = (T * T_level_start.inverse()).log();
                const Vec6 delta_n = s_inv.asDiagonal() * delta;

                Vec6 keep = Vec6::Zero();
                for (int k = 0; k < 6; ++k) {
                    if (ev(k) < thresh) continue;
                    const Vec6 v = es_l.eigenvectors().col(k);
                    keep.noalias() += v * v.dot(delta_n);
                }
                T = SE3::exp(s.asDiagonal() * keep) * T_level_start;
            }
        }

        final_acc   = fin;
        final_level = level;
        final_huber = huber;
    }

    if (!any_level_converged || final_acc.used < cfg_.min_points) {
        return {ErrorCode::DidNotConverge, "유효 측광 잔차가 부족해 정렬 실패"};
    }

    result.T_cur_ref  = T;
    result.affine_a   = a;
    result.affine_b   = b;
    result.point_count  = final_acc.used;
    result.inlier_count = final_acc.inliers;
    result.inlier_ratio = static_cast<double>(final_acc.inliers) /
                          static_cast<double>(std::max<std::size_t>(1, final_acc.used));
    result.photometric_rmse =
        std::sqrt(final_acc.chi2 / static_cast<double>(std::max<std::size_t>(1, final_acc.used)));

    // --- 기하 정합성 --------------------------------------------------------
    // 정렬에 쓰이지 않은 cur 의 깊이맵과 대조한다. 측광 채널이 포화해도 이쪽은
    // 상한이 없다 (헤더 주석과 06-results.md 26 참조).
    if (!ref.depth.empty() && !cur.depth.empty() &&
        ref.depth.type() == CV_32F && cur.depth.type() == CV_32F) {
        const auto& K = cur.intrinsics;
        // 유효 깊이 범위는 데이터셋의 것을 쓴다. 여기 0.1/8.0 이 박혀 있었다 -
        // tools/dataset_calib.hpp 머리말이 기록한 바로 그 상수이고, 두 도구에서는
        // 걷어냈는데 라이브러리에는 남아 있었다. KITTI 는 유효 깊이가 3.2~48 m 라
        // 이 검사가 3.2~8 m 구간, 즉 카메라 바로 앞 노면 몇 미터만 표본으로 삼았다.
        // 장면의 임의 표본이 아니라 한쪽으로 치우친 표본이면, 이 채널이 "측광과
        // 독립인 관측" 이라는 계약을 지키지 못한다. 정렬이 점을 고를 때 쓰는
        // 범위(cfg_.min_depth/max_depth)와 같은 값을 쓴다.
        const auto z_lo = static_cast<float>(cfg_.min_depth);
        const auto z_hi = static_cast<float>(cfg_.max_depth);
        std::vector<double> rel;
        rel.reserve(2048);
        // 표본 깊이도 같이 모은다. 이 통계의 잡음 바닥이 깊이에 의존하므로
        // (depthNoiseFloor 주석) 어느 깊이에서 잰 값인지가 결과와 함께 있어야 한다.
        std::vector<double> zs;
        zs.reserve(2048);
        std::size_t outliers = 0;
        // 성긴 격자로 충분하다 - 추정이 아니라 검사다. 6 화소 간격이면
        // 640x480 에서 ~8500 표본이고 비용은 정렬의 1 % 미만이다.
        for (int v = 4; v + 4 < ref.depth.rows; v += 6) {
            const auto* zr_row = ref.depth.ptr<float>(v);
            for (int u = 4; u + 4 < ref.depth.cols; u += 6) {
                const float zr = zr_row[u];
                if (!(zr > z_lo) || zr > z_hi) continue;
                const Vec3 p_ref((u - K.cx) * zr / K.fx, (v - K.cy) * zr / K.fy, zr);
                const Vec3 p_cur = result.T_cur_ref * p_ref;
                if (!(p_cur.z() > 0.1)) continue;
                const int ui = static_cast<int>(std::lround(K.fx * p_cur.x() / p_cur.z() + K.cx));
                const int vi = static_cast<int>(std::lround(K.fy * p_cur.y() / p_cur.z() + K.cy));
                if (ui < 0 || vi < 0 || ui >= cur.depth.cols || vi >= cur.depth.rows) continue;
                const float zm = cur.depth.ptr<float>(vi)[ui];
                if (!(zm > z_lo) || zm > z_hi) continue;
                const double r = std::abs(p_cur.z() - zm) / zm;
                rel.push_back(r);
                zs.push_back(zm);
                if (r > 0.10) ++outliers;
            }
        }
        // 표본이 적으면 판정하지 않는다. 적은 표본의 중앙값을 신뢰도로 내보내면
        // 깊이가 거의 없는 장면에서 근거 없는 확신이 된다.
        if (rel.size() >= 50) {
            std::nth_element(rel.begin(), rel.begin() + rel.size() / 2, rel.end());
            result.depth_consistency   = rel[rel.size() / 2];
            result.depth_outlier_ratio = static_cast<double>(outliers) /
                                         static_cast<double>(rel.size());
            std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
            result.depth_median_z = zs[zs.size() / 2];
            result.depth_sample_count = rel.size();
        }
    }

    // --- 깊이 잡음을 모델값이 아니라 측정값으로 -----------------------------
    //
    // 여기까지 정보행렬은 sigma_Z = c*Z^2 라는 **모델** 위에 서 있었다. c 는
    // 센서 상수(스테레오라면 sigma_d/(f*B))이므로 프레임마다 같다. 그래서 깊이가
    // 실제로 그 모델보다 나쁜 프레임에서도 정보행렬은 같은 확신을 보고했다.
    //
    // 실측이 그 결과를 잰다 (kitti_04, 133 프레임):
    //   depth_incons 와 프레임간 수직오차의 상관   +0.95
    //   observable_dof                            133 프레임 내내 상수 6
    //   cond                                      +0.18
    // 즉 이 실패를 보는 채널은 있는데 퇴화 보고가 그것을 안 읽고 있었다.
    //
    // 읽는 방법은 새로 만들지 않는다. depth_consistency 는 이미 "정렬에 쓰이지
    // 않은 관측(cur 깊이)과의 상대 불일치" 이고, 그 값이 센서 잡음만으로 설명
    // 가능한 한계 depthNoiseFloor() = sqrt(2)*c*Z 도 이미 유도되어 있다. 둘의 비
    //
    //     rho = depth_consistency / (sqrt(2) * c * Z)
    //
    // 는 "이 프레임의 깊이 오차가 센서 모델보다 몇 배인가" 다. 무차원이고,
    // 적합한 상수가 들어 있지 않다. rho > 1 이면 sigma_Z 를 rho 배로 키워
    // **같은 식** 으로 정보행렬을 다시 쌓는다.
    //
    //   var_depth,i = (dr_i/dd)^2 * (rho * c * d_i^2)^2
    //   w_i        *= sigma_I^2 / (sigma_I^2 + var_depth,i)
    //
    // 왜 이것이 "depth_incons 를 DOF 에 곱하는 것" 과 다른가. 곱셈은 방향을
    // 구분하지 못한다. 여기서는 rho 가 점별 sigma_Z 로 들어가고, 그 무게가
    // dr/dd = Jp.(Q-t)/d - 즉 **그 점의 잔차가 깊이에 얼마나 민감한가** - 에
    // 따라 다르게 걸린다. 먼 점과 시선 방향으로 기울어진 점이 먼저 빠지므로
    // 정보의 감소가 방향에 따라 다르게 나타난다. 정보행렬은 여전히 "관측이
    // 실제로 얼마나 구속하는가" 를 재고 있고, 다만 그 관측의 잡음을 모델값이
    // 아니라 이 프레임이 잰 값으로 쓴다.
    //
    // 한 방향으로만 쓴다. rho 는 1 아래로 내려가지 않는다 - 측정이 모델보다
    // 좋게 나왔다고 확신을 키우면, 표본 잡음이 확신으로 바뀐다.
    //
    // 추정은 건드리지 않는다. 이 재누산은 T 가 확정된 뒤에 돌고 결과는
    // 정보행렬에만 들어간다 (T, a, b, rmse, inlier 는 그대로다).
    double rho = 1.0;
    if (cfg_.depth_sigma_rel > 0.0 && result.depth_consistency >= 0.0 &&
        final_level >= 0 && quality != nullptr) {
        const double floor = result.depthNoiseFloor(cfg_.depth_sigma_rel);
        if (floor > 0.0) rho = std::max(1.0, result.depth_consistency / floor);
    }
    result.depth_sigma_scale = rho;

    if (rho > 1.0) {
        // final_acc 를 낸 레벨을 그대로 다시 고른다. 레벨이 바뀌면 점 집합이
        // 바뀌어 rho 의 효과와 레벨 차이가 섞인다.
        const auto fl = static_cast<std::size_t>(final_level);
        selectPoints(*rp, ref.static_mask, quality, final_level);
        if (points_.size() >= cfg_.min_points) {
            Accumulator re;
            abs_res_.assign(points_.size(), -1.f);
            if (cfg_.information_model == InformationModel::ClusterRobust) {
                scores_.assign(static_cast<std::size_t>(cfg_.info_cluster_grid) *
                                   static_cast<std::size_t>(cfg_.info_cluster_grid),
                               Eigen::Matrix<double, 8, 1>::Zero());
                score_count_.assign(scores_.size(), 0u);
            }
            accumulate(cp->gray[fl], cp->grad_x[fl], cp->grad_y[fl], rp->intrinsics[fl],
                       T, a, b, final_huber, health_delta, noise_sigma,
                       0, points_.size(), re,
                       cfg_.information_model == InformationModel::ClusterRobust
                           ? cfg_.info_cluster_grid : 0,
                       rho);
            // 점이 다 빠져 버리면 판정할 근거가 없다. 그때는 모델값 그대로 둔다 -
            // "재 보니 아무것도 안 남았다" 를 "정보가 0 이다" 로 접으면 안 된다.
            if (re.used >= cfg_.min_points) final_acc = re;
            else                            result.depth_sigma_scale = 1.0;
        } else {
            result.depth_sigma_scale = 1.0;
        }
    }

    // --- 정보행렬 -----------------------------------------------------------
    // 아핀 자유도는 주변화한다. 그대로 두면 노출 변화에 대한 불확실성이
    // 포즈 확신도로 잘못 새어 들어간다. 잡음 모델은 cfg_.information_model.
    const double sigma2 = (quality != nullptr) ? quality->photometricVariance() : 1.0;
    Mat6 Lambda = poseInformation(final_acc, sigma2);

    // 환경 가중 alpha_0(E). 팩터그래프가 다른 tier 와 섞을 때 쓰는 정보 질량.
    if (env != nullptr) Lambda *= std::max(1e-6, env->tier.photometric);

    // J*J^T 를 수만 개 누산하면 부동소수점 때문에 미세한 비대칭이 남는다.
    // 수학적으로 대칭이 보장된 양이므로 명시적으로 대칭화한다.
    Lambda = 0.5 * (Lambda + Lambda.transpose()).eval();

    result.information = Lambda;

    // --- 퇴화 진단 ----------------------------------------------------------
    // 병진(m)과 회전(rad)은 단위가 달라 고유값을 그대로 비교할 수 없다.
    // 대각 정규화 후 스펙트럼을 보면 단위에 무관한 랭크 판정이 된다.
    Vec6 scale;
    for (int k = 0; k < 6; ++k) scale(k) = 1.0 / std::sqrt(std::max(1e-12, Lambda(k, k)));
    const Mat6 Lambda_n = scale.asDiagonal() * Lambda * scale.asDiagonal();

    Eigen::SelfAdjointEigenSolver<Mat6> es(Lambda_n);
    if (es.info() == Eigen::Success) {
        // Eigen 은 오름차순으로 준다. 내림차순으로 뒤집어 저장한다.
        const Vec6 ev = es.eigenvalues();
        for (int k = 0; k < 6; ++k) result.eigenvalues(k) = ev(5 - k);

        const double max_ev = std::max(1e-12, result.eigenvalues(0));
        result.observable_dof = 0;
        for (int k = 0; k < 6; ++k) {
            if (result.eigenvalues(k) > max_ev * cfg_.degeneracy_ratio) ++result.observable_dof;
        }
        // 같은 스펙트럼을 문턱 없이 읽은 유효 자유도 (effective_dof 주석).
        // 위의 계단이 KITTI 에서 한 번도 밟히지 않는다는 실측이 이 항의 근거다.
        const double sum_ev = std::max(1e-300, ev.sum());
        double h = 0.0;
        for (int k = 0; k < 6; ++k) {
            const double p = std::max(1e-300, ev(k) / sum_ev);
            h -= p * std::log(p);
        }
        result.effective_dof = std::exp(h);
        // 가장 약한 축 = 최소 고유값의 고유벡터. Tier 2 가 채워야 할 방향.
        result.weakest_direction = es.eigenvectors().col(0);
    }

    if (result.observable_dof < 6) {
        return Result<AlignmentResult>::degradedValue(
            result, static_cast<double>(result.observable_dof) / 6.0,
            "관측 불가 자유도 존재 - 구조 제약 필요");
    }
    if (result.inlier_ratio < 0.5) {
        return Result<AlignmentResult>::degradedValue(result, result.inlier_ratio,
                                                      "인라이어 비율 낮음");
    }
    // 기하 정합성. 위의 두 판정은 **측광 채널 안에서** 나오므로, 측광이 만족한
    // 채로 포즈가 틀린 경우를 잡지 못한다 - 13.3 과 23.3 이 잰 것이 정확히
    // 그 상태다. 이 검사만 정렬에 쓰이지 않은 관측(cur 깊이)에서 온다.
    //
    // 판정 불가(-1)는 여기서 degrade 하지 않는다. "모른다" 를 "나쁘다" 로
    // 접으면 깊이가 없는 정상 장면이 전부 열등해진다. 대신 depthReliability()
    // 가 그 경우를 0.5(중립)로 돌려주고, 그 구분은 소비자의 몫이다.
    if (result.depth_consistency >= 0.0 &&
        !result.depthConsistent()) {
        return Result<AlignmentResult>::degradedValue(
            result, result.depthReliability(),
            "기하 정합성 낮음 - 추정 포즈가 관측된 깊이와 어긋난다");
    }
    return result;
}

}  // namespace wme
