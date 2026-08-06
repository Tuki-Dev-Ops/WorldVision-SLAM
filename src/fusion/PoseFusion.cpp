#include "wme/fusion/PoseFusion.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace wme::fusion {
namespace {

// ad(xi) = [[skew(phi), skew(rho)], [0, skew(phi)]]  (xi = [rho, phi])
Mat6 adjointAlgebra(const Vec6& xi) {
    const Mat3 P = skew(Vec3(xi.head<3>()));
    const Mat3 W = skew(Vec3(xi.tail<3>()));
    Mat6 A = Mat6::Zero();
    A.block<3, 3>(0, 0) = W;
    A.block<3, 3>(0, 3) = P;
    A.block<3, 3>(3, 3) = W;
    return A;
}

bool finite(const Mat6& m) { return m.allFinite(); }

// 대칭화 + 반정부호 보정. 누산 과정의 반올림으로 아주 작은 음의 고유값이
// 생기면 그 방향에 음의 확신이 실린다. 정보행렬로 쓰기 전에 잘라낸다.
Mat6 symmetrize(const Mat6& m) { return (0.5 * (m + m.transpose())).eval(); }

struct Spectrum {
    Vec6 values{Vec6::Zero()};      // 내림차순
    Mat6 vectors{Mat6::Zero()};     // 열 k 가 values(k)
    int  rank{0};
};

Spectrum analyse(const Mat6& m, double ratio) {
    Spectrum out;
    Eigen::SelfAdjointEigenSolver<Mat6> es(symmetrize(m));
    if (es.info() != Eigen::Success) return out;
    for (int k = 0; k < 6; ++k) {
        out.values(k)      = es.eigenvalues()(5 - k);
        out.vectors.col(k) = es.eigenvectors().col(5 - k);
    }
    const double max_ev = std::max(1e-300, out.values(0));
    for (int k = 0; k < 6; ++k) {
        if (out.values(k) > max_ev * ratio) ++out.rank;
    }
    return out;
}

}  // namespace

std::string_view toString(Tier t) noexcept {
    switch (t) {
        case Tier::Photometric:   return "photometric";
        case Tier::Constellation: return "constellation";
        case Tier::Structural:    return "structural";
    }
    return "unknown";
}

std::string_view toString(Abstain a) noexcept {
    switch (a) {
        case Abstain::None:            return "none";
        case Abstain::Disabled:        return "disabled";
        case Abstain::NoInput:         return "no_input";
        case Abstain::Failed:          return "failed";
        case Abstain::ZeroInformation: return "zero_information";
        case Abstain::ZeroWeight:      return "zero_weight";
    }
    return "unknown";
}

Mat6 se3LeftJacobian(const Vec6& xi) {
    // J_l = sum_{n>=0} ad(xi)^n / (n+1)!
    const Mat6 A = adjointAlgebra(xi);
    Mat6 J = Mat6::Identity();
    Mat6 term = Mat6::Identity();
    double fact = 1.0;                       // (n+1)!
    for (int n = 1; n <= 30; ++n) {
        term = (term * A).eval();
        fact *= static_cast<double>(n + 1);
        const Mat6 add = term / fact;
        J += add;
        if (add.norm() < 1e-18 * std::max(1.0, J.norm())) break;
    }
    return J;
}

Mat6 se3LeftJacobianInverse(const Vec6& xi) {
    // J_l 은 블록 상삼각이고 대각 블록이 SO(3) 좌측 자코비안이라 항상 가역이다
    // (|phi| <= pi 에서 det J_l(phi) != 0). 급수의 역을 직접 구한다.
    return se3LeftJacobian(xi).inverse();
}

Mat6 transportInformation(const Mat6& information, const SE3& T_from, const SE3& T_to) {
    const Vec6 eps = (T_to * T_from.inverse()).log();
    const Mat6 Ji  = se3LeftJacobianInverse(eps);
    return symmetrize(Ji.transpose() * information * Ji);
}

double translationFraction(const Vec6& direction) {
    const double n2 = direction.squaredNorm();
    if (n2 <= 0.0) return 0.0;
    return direction.head<3>().squaredNorm() / n2;
}

Complementarity complementarity(const Mat6& a, const Mat6& b) {
    Complementarity out;
    const Spectrum sa = analyse(a, 1e-3);
    if (sa.values(0) <= 0.0) return out;

    out.strong_direction = sa.vectors.col(0);
    out.weak_direction   = sa.vectors.col(5);

    out.a_strong = out.strong_direction.dot(a * out.strong_direction);
    out.a_weak   = out.weak_direction.dot(a * out.weak_direction);
    out.b_strong = out.strong_direction.dot(b * out.strong_direction);
    out.b_weak   = out.weak_direction.dot(b * out.weak_direction);

    // A 의 약축 정보가 정확히 0 이면 비율이 무한대가 된다. 그 경우는
    // "채울 여지가 무한" 이 맞지만 숫자로 내보낼 수 없으므로 유효하지 않다고 말한다.
    if (!(out.a_weak > 0.0) || !(out.a_strong > 0.0) || !(out.b_strong > 0.0)) return out;

    out.fill_ratio = (out.b_weak / out.a_weak) / (out.b_strong / out.a_strong);
    out.valid = std::isfinite(out.fill_ratio);
    return out;
}

Result<FusionResult> fuse(const std::vector<TierEstimate>& estimates, const FusionConfig& cfg) {
    FusionResult out;

    // --- 유효 tier 선별 ----------------------------------------------------
    // 기권은 전부 이유와 함께 남긴다. 아무 것도 안 적고 가중치 0 으로 빠지면
    // "융합했다" 는 말이 무엇을 뜻하는지 밖에서 알 수 없다.
    struct Active {
        std::size_t idx{0};
        Mat6        lambda{Mat6::Zero()};   // alpha*kappa 적용 후
    };
    std::vector<Active> active;
    active.reserve(estimates.size());

    for (std::size_t i = 0; i < estimates.size(); ++i) {
        const TierEstimate& e = estimates[i];
        const auto slot = static_cast<std::size_t>(e.tier);
        if (slot >= kTierCount) continue;
        TierContribution& c = out.tiers[slot];
        c.alpha       = e.alpha;
        c.calibration = e.calibration;

        if (!e.available) { c.reason = e.reason; continue; }
        const double w = e.alpha * e.calibration;
        if (!(w > 0.0)) { c.reason = Abstain::ZeroWeight; continue; }
        if (!finite(e.information) || e.information.trace() <= 0.0) {
            c.reason = Abstain::ZeroInformation;
            continue;
        }
        Active a;
        a.idx    = i;
        a.lambda = symmetrize(w * e.information);
        active.push_back(a);
        c.used       = true;
        c.reason     = Abstain::None;
        c.info_trace = a.lambda.trace();
    }

    out.contributing_tiers = static_cast<int>(active.size());
    if (active.empty()) {
        return {ErrorCode::NotAvailable, "기여 가능한 tier 가 없다"};
    }

    // --- 초기값: 정보가 가장 큰 tier ---------------------------------------
    // 임의로 첫 번째를 쓰면 랭크 부족한 SPA 에서 출발해 최적화가 그 널공간을
    // 헤매게 된다. 가장 많이 아는 쪽에서 시작하는 것이 그 다음 반복에 유리하다.
    std::size_t seed = 0;
    for (std::size_t k = 1; k < active.size(); ++k) {
        if (active[k].lambda.trace() > active[seed].lambda.trace()) seed = k;
    }
    SE3 T = estimates[active[seed].idx].T_cur_ref;

    // --- 가우스-뉴턴 --------------------------------------------------------
    // eps_k(T) = log(T * T_k^-1),  d eps_k / d delta = J_l^-1(eps_k)  (T <- exp(delta) T)
    for (int it = 0; it < cfg.max_iterations; ++it) {
        Mat6 H = Mat6::Zero();
        Vec6 g = Vec6::Zero();
        for (const Active& a : active) {
            const Vec6 eps = (T * estimates[a.idx].T_cur_ref.inverse()).log();
            const Mat6 J   = se3LeftJacobianInverse(eps);
            const Mat6 JtL = J.transpose() * a.lambda;
            H.noalias() += JtL * J;
            g.noalias() += JtL * eps;
        }
        H = symmetrize(H);

        // 관측 불가 축으로는 움직이지 않는다. 절단 최소제곱.
        const Spectrum sp = analyse(H, cfg.update_observable_ratio);
        if (sp.values(0) <= 0.0) break;
        const double thresh = sp.values(0) * cfg.update_observable_ratio;
        Vec6 delta = Vec6::Zero();
        for (int k = 0; k < 6; ++k) {
            if (sp.values(k) <= thresh) continue;
            delta.noalias() -= (sp.vectors.col(k).dot(g) / sp.values(k)) * sp.vectors.col(k);
        }
        if (!delta.allFinite()) break;

        T = SE3::exp(delta) * T;
        T.normalize();
        out.iterations = it + 1;
        if (delta.norm() < cfg.convergence_delta) break;
    }

    // --- 결과 정보행렬 ------------------------------------------------------
    // information       : 융합점 T 로 옮겨 붙인 것. 실제로 이 해가 갖는 정보다.
    // information_naive : 문서 식 그대로의 단순합. 두 값의 차이가 곧
    //                     "선형화점이 달랐다" 의 크기이며, 그것을 보여주려고 남긴다.
    Mat6 total = Mat6::Zero();
    Mat6 naive = Mat6::Zero();
    for (const Active& a : active) {
        const Vec6 eps = (T * estimates[a.idx].T_cur_ref.inverse()).log();
        const Mat6 J   = se3LeftJacobianInverse(eps);
        total.noalias() += J.transpose() * a.lambda * J;
        naive           += a.lambda;

        TierContribution& c = out.tiers[static_cast<std::size_t>(estimates[a.idx].tier)];
        c.residual  = eps;
        c.self_nees = eps.dot(a.lambda * eps);
    }
    out.T_cur_ref         = T;
    out.information       = symmetrize(total);
    out.information_naive = symmetrize(naive);

    const double tr = std::max(1e-300, out.information.trace());
    for (auto& c : out.tiers) c.info_share = c.used ? (c.info_trace / tr) : 0.0;

    const Spectrum sp = analyse(out.information, cfg.degeneracy_ratio);
    out.eigenvalues       = sp.values;
    out.observable_dof    = sp.rank;
    out.weakest_direction = sp.vectors.col(5);

    if (out.observable_dof < 6) {
        return Result<FusionResult>::degradedValue(
            out, static_cast<double>(out.observable_dof) / 6.0,
            "융합 후에도 관측 불가 자유도가 남았다");
    }
    return out;
}

}  // namespace wme::fusion
