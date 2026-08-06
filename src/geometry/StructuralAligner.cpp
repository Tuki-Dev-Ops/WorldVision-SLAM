#include "wme/geometry/StructuralAligner.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wme {

namespace {

// 대각 정규화 후 고유분해. 병진(m)과 회전(rad)은 단위가 달라 고유값을 그대로
// 비교할 수 없다. 정규화하면 단위에 무관한 랭크 판정이 된다.
// scale 은 정규화 좌표 -> 접선 좌표 변환이라 고유벡터를 되돌릴 때 쓴다.
struct Spectrum {
    Vec6 eigenvalues{Vec6::Zero()};     // 내림차순
    Mat6 eigenvectors{Mat6::Zero()};    // 열 k 가 eigenvalues(k) 에 대응, 접선 좌표
    int  rank{0};
};

[[nodiscard]] Spectrum analyse(const Mat6& Lam, double ratio) {
    Spectrum out;
    Vec6 scale;
    for (int k = 0; k < 6; ++k) scale(k) = 1.0 / std::sqrt(std::max(1e-12, Lam(k, k)));
    const Mat6 Lam_n = scale.asDiagonal() * Lam * scale.asDiagonal();

    Eigen::SelfAdjointEigenSolver<Mat6> es(Lam_n);
    if (es.info() != Eigen::Success) return out;

    const Vec6 ev = es.eigenvalues();   // 오름차순
    for (int k = 0; k < 6; ++k) {
        out.eigenvalues(k) = ev(5 - k);
        // 정규화 좌표의 고유벡터를 접선 좌표로 되돌린다 (delta = S * delta_n).
        Vec6 v = scale.asDiagonal() * es.eigenvectors().col(5 - k);
        const double n = v.norm();
        out.eigenvectors.col(k) = (n > 1e-12) ? Vec6(v / n) : Vec6::Zero();
    }
    const double max_ev = std::max(1e-12, out.eigenvalues(0));
    for (int k = 0; k < 6; ++k) {
        if (out.eigenvalues(k) > max_ev * ratio) ++out.rank;
    }
    return out;
}

}  // namespace

StructuralAligner::StructuralAligner(StructuralAlignerConfig config) : cfg_(config) {}

std::vector<PlaneMatch> StructuralAligner::match(const std::vector<Plane>& reference,
                                                 const std::vector<Plane>& current,
                                                 const SE3& init) const {
    std::vector<PlaneMatch> matches;
    if (reference.empty() || current.empty()) return matches;

    // 큰 평면부터 짝지어 애매한 대응이 먼저 자리를 차지하지 않게 한다.
    std::vector<std::size_t> order(reference.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&reference](std::size_t a, std::size_t b) {
        const double ca = reference[a].confidence(), cb = reference[b].confidence();
        if (ca != cb) return ca > cb;
        return a < b;
    });

    const Mat3 R = init.rotation().matrix();
    const Vec3 t = init.translation();
    std::vector<std::uint8_t> used(current.size(), 0);

    for (const std::size_t ri : order) {
        const Plane moved = reference[ri].transformed(R, t);

        std::size_t best = current.size();
        double best_angle = cfg_.max_normal_angle;
        double best_dd = 0.0;

        for (std::size_t ci = 0; ci < current.size(); ++ci) {
            if (used[ci] != 0) continue;
            const double cos_a = std::clamp(moved.normal.dot(current[ci].normal), -1.0, 1.0);
            const double angle = std::acos(cos_a);
            if (angle > cfg_.max_normal_angle) continue;
            const double dd = std::abs(moved.distance - current[ci].distance);
            if (dd > cfg_.max_distance_diff) continue;
            if (angle < best_angle) { best = ci; best_angle = angle; best_dd = dd; }
        }

        if (best < current.size()) {
            used[best] = 1;
            PlaneMatch m;
            m.ref_index     = ri;
            m.cur_index     = best;
            m.angle         = best_angle;
            m.distance_diff = best_dd;
            m.weight        = reference[ri].confidence() * current[best].confidence();
            matches.push_back(m);
        }
    }
    return matches;
}

Result<StructuralAlignmentResult> StructuralAligner::align(const std::vector<Plane>& reference,
                                                           const std::vector<Plane>& current,
                                                           const SE3& init,
                                                           double alpha_structural) const {
    StructuralAlignmentResult result;
    result.T_cur_ref = init;
    result.matches   = match(reference, current, init);

    const std::size_t m = result.matches.size();
    if (m < std::max<std::size_t>(1, cfg_.min_matches)) {
        return {ErrorCode::InsufficientData, "평면 대응이 부족해 SPA 추정 불가"};
    }

    // --- 회전: 법선끼리의 Kabsch (중심화 없이, 방향만) ---------------------
    // min_R sum w |R n_ref - n_cur|^2  <=>  max_R trace(R^T M), M = sum w n_cur n_ref^T
    double w_sum = 0.0;
    Mat3 M = Mat3::Zero();
    for (const PlaneMatch& mt : result.matches) {
        const double w = mt.weight;
        w_sum += w;
        M.noalias() += w * current[mt.cur_index].normal * reference[mt.ref_index].normal.transpose();
    }
    if (w_sum <= 0.0) {
        return {ErrorCode::Degenerate, "평면 신뢰도가 0 이라 가중치가 없음"};
    }

    // 구속되지 않은 회전축은 init 을 유지한다. 이 항이 없으면 평면 하나짜리
    // 장면에서 SVD 가 법선 둘레 임의 회전을 골라 확신 있는 오답이 된다.
    M.noalias() += (cfg_.rotation_prior_weight * w_sum) * init.rotation().matrix();

    Eigen::JacobiSVD<Mat3> svd_r(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd_r.matrixU();
    const Mat3 V = svd_r.matrixV();
    if ((U * V.transpose()).determinant() < 0.0) U.col(2) *= -1.0;   // 반사 제거
    const Mat3 R = U * V.transpose();

    // --- 병진 -------------------------------------------------------------
    // ref 좌표의 점 p 는 cur 좌표에서 q = R p + t 다. 평면 n_r·p = d_r 을
    // 대입하면 (R n_r)·q = d_r + (R n_r)·t 이므로
    //
    //     n_cur · t = d_cur - d_ref
    //
    // 부호를 뒤집으면 병진이 정확히 반대로 나온다. 항등 정합에서는 양변이 0 이라
    // 드러나지 않으므로, 반드시 병진이 있는 케이스로 검증해야 한다.
    MatX A(static_cast<int>(m), 3);
    VecX b(static_cast<int>(m));
    for (std::size_t i = 0; i < m; ++i) {
        const PlaneMatch& mt = result.matches[i];
        const double sw = std::sqrt(mt.weight);
        A.row(static_cast<int>(i)) = sw * current[mt.cur_index].normal.transpose();
        b(static_cast<int>(i)) = sw * (current[mt.cur_index].distance
                                       - reference[mt.ref_index].distance);
    }

    // 절단 최소노름 해.
    // 완전한 최소제곱은 거의 영에 가까운 특이값 방향까지 살려 두고 그 방향의
    // 잡음을 크게 증폭한다 (Python 참조 구현 실측: 병진 오차 40 m 초과).
    // 랭크 판정과 **같은** 문턱으로 자른다 - "관측되지 않는다" 고 보고한 방향으로
    // 해가 움직이면 보고와 결과가 모순된다.
    Eigen::JacobiSVD<MatX> svd_t(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& sv = svd_t.singularValues();
    const double s_max = (sv.size() > 0) ? sv(0) : 0.0;
    // A^T A 의 고유값은 sv^2 이므로, degeneracy_ratio 를 특이값에 적용하려면 제곱근이다
    const double tau = std::sqrt(cfg_.degeneracy_ratio) * s_max;

    Vec3 t = Vec3::Zero();
    int trans_rank = 0;
    for (int k = 0; k < sv.size(); ++k) {
        if (sv(k) <= tau || sv(k) <= 0.0) continue;
        ++trans_rank;
        t.noalias() += (svd_t.matrixU().col(k).dot(b) / sv(k)) * svd_t.matrixV().col(k);
    }

    // --- 랭크 --------------------------------------------------------------
    // 회전 잔차 r = n_cur - exp(dphi) R n_ref 의 자코비안은 [n_cur]_x 이고,
    // J^T J = I - n n^T 다. 즉 평면 하나는 법선에 **수직인** 두 축을 구속하고
    // 법선 둘레 회전은 구속하지 않는다 - sum w n n^T (그 직교여공간) 이 아니다.
    // 그 둘을 바꾸면 관측 가능한 축과 불가능한 축이 정확히 뒤바뀐다.
    //
    // 병진 랭크는 위 절단에서 이미 나왔다. A 의 특이값과 A^T A 의 고유값은
    // 같은 것을 두 번 세는 셈이라 따로 만들지 않는다.
    Mat3 Omega_r = Mat3::Zero();   // 회전 정보 (가중, sigma 미적용)
    for (const PlaneMatch& mt : result.matches) {
        const Vec3& n = current[mt.cur_index].normal;
        Omega_r.noalias() += mt.weight * (Mat3::Identity() - n * n.transpose());
    }

    Eigen::SelfAdjointEigenSolver<Mat3> es_r(Omega_r);
    int rot_rank = 0;
    if (es_r.info() == Eigen::Success) {
        const double max_ev = std::max(1e-12, es_r.eigenvalues()(2));
        for (int k = 0; k < 3; ++k) {
            if (es_r.eigenvalues()(k) > max_ev * cfg_.degeneracy_ratio) ++rot_rank;
        }
    }

    if (rot_rank == 0 && trans_rank == 0) {
        return {ErrorCode::Degenerate, "평면 법선이 어떤 자유도도 구속하지 못함"};
    }

    result.T_cur_ref       = SE3(R, t);
    result.rotation_rank   = rot_rank;
    result.translation_rank = trans_rank;

    // --- 잔차 --------------------------------------------------------------
    // 랭크가 6 이어도 대응이 틀렸으면 여기가 커진다. 랭크는 "구속했는가" 를,
    // 잔차는 "맞았는가" 를 말한다. 둘 다 없으면 조용한 오답을 못 잡는다.
    double n_sq = 0.0, o_sq = 0.0;
    for (const PlaneMatch& mt : result.matches) {
        const Plane& pr = reference[mt.ref_index];
        const Plane& pc = current[mt.cur_index];
        n_sq += mt.weight * (pc.normal - R * pr.normal).squaredNorm();
        const double e = pc.normal.dot(t) - (pc.distance - pr.distance);
        o_sq += mt.weight * e * e;
    }
    result.normal_rms = std::sqrt(n_sq / w_sum);
    result.offset_rms = std::sqrt(o_sq / w_sum);

    // --- 정보행렬 ----------------------------------------------------------
    // 접선 규약은 SE3 와 같은 xi = [rho(3), phi(3)], 좌측 섭동 T <- exp(xi) T 다.
    // 좌측 섭동에서 병진 변화는 rho 뿐 아니라 회전에서도 온다: dt = rho + phi x t.
    // 따라서 오프셋 잔차의 자코비안은 [n^T, (t x n)^T] 이고, 그 교차항을 빼면
    // 회전 불확실성이 병진 확신도로 잘못 새어 들어간다. t = 0 일 때만 블록대각이다.
    Mat6 Lam = Mat6::Zero();
    const double inv_st2 = 1.0 / (cfg_.translation_sigma * cfg_.translation_sigma);
    const double inv_sr2 = 1.0 / (cfg_.rotation_sigma * cfg_.rotation_sigma);

    for (const PlaneMatch& mt : result.matches) {
        const Vec3& n = current[mt.cur_index].normal;
        Vec6 J;
        J.head<3>() = n;
        J.tail<3>() = t.cross(n);
        Lam.noalias() += (mt.weight * inv_st2) * (J * J.transpose());
    }
    Lam.block<3, 3>(3, 3).noalias() += inv_sr2 * Omega_r;
    Lam *= std::max(alpha_structural, 1e-6);
    Lam = (0.5 * (Lam + Lam.transpose())).eval();
    result.information = Lam;

    // --- 퇴화 진단 ----------------------------------------------------------
    const Spectrum sp = analyse(Lam, cfg_.degeneracy_ratio);
    result.eigenvalues       = sp.eigenvalues;
    result.observable_dof    = sp.rank;
    result.weakest_direction = sp.eigenvectors.col(5);

    if (result.observable_dof < 6) {
        return Result<StructuralAlignmentResult>::degradedValue(
            result, static_cast<double>(result.observable_dof) / 6.0,
            "관측 불가 자유도 존재 - 다른 tier 가 채워야 함");
    }
    return result;
}

std::vector<Vec6> unobservableDirections(const StructuralAlignmentResult& r, double ratio) {
    std::vector<Vec6> out;
    const Spectrum sp = analyse(r.information, ratio);
    if (sp.eigenvalues(0) <= 1e-12) {
        // 정보가 전혀 없다. 여섯 축 전부 관측 불가다.
        for (int k = 0; k < 6; ++k) out.push_back(Vec6::Unit(k));
        return out;
    }
    const double thresh = sp.eigenvalues(0) * ratio;
    for (int k = 0; k < 6; ++k) {
        if (sp.eigenvalues(k) <= thresh) out.push_back(sp.eigenvectors.col(k));
    }
    return out;
}

}  // namespace wme
