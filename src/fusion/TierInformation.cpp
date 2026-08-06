#include "wme/fusion/TierInformation.hpp"

#include <algorithm>
#include <cmath>

namespace wme::fusion {

std::array<double, kTierCount> tierWeights(const EnvironmentState& env, WeightMode mode) {
    if (mode == WeightMode::Uniform) return {1.0, 1.0, 1.0};
    return {env.tier.photometric, env.tier.constellation, env.tier.structural};
}

Mat6 constellationInformation(const std::vector<ConstellationPair>& pairs,
                              const SE3& T_cur_ref) {
    Mat6 lambda = Mat6::Zero();
    if (pairs.size() < 2) return lambda;

    for (const ConstellationPair& c : pairs) {
        const double s = std::max(1e-4, c.sigma);       // 0 sigma 는 무한 확신이 된다
        const double w = 1.0 / (s * s);
        const Vec3   q = T_cur_ref * c.p_ref;           // cur 좌표계로 옮긴 ref 점

        // r = q - p_cur, 좌측섭동 T <- exp([rho, phi]) T 이면 dq = rho + phi x q
        // 이므로 dr/d[rho, phi] = [I, -skew(q)].
        Eigen::Matrix<double, 3, 6> J;
        J.leftCols<3>()  = Mat3::Identity();
        J.rightCols<3>() = -skew(q);
        lambda.noalias() += w * (J.transpose() * J);
    }
    return (0.5 * (lambda + lambda.transpose())).eval();
}

}  // namespace wme::fusion
