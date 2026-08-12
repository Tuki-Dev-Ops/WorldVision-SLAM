#include "wme/token/ConstellationIndex.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

namespace wme {
namespace {

// 64비트 혼합 해시 (splitmix64 finalizer)
inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// 비트셋 인접행렬 헬퍼
struct BitAdj {
    std::size_t n{0}, words{0};
    std::vector<std::uint64_t> bits;

    void resize(std::size_t count) {
        n = count;
        words = (count + 63) / 64;
        bits.assign(n * words, 0ULL);
    }
    void set(std::size_t i, std::size_t j) { bits[i * words + (j >> 6)] |= (1ULL << (j & 63)); }
    [[nodiscard]] bool test(std::size_t i, std::size_t j) const {
        return (bits[i * words + (j >> 6)] >> (j & 63)) & 1ULL;
    }
    [[nodiscard]] const std::uint64_t* row(std::size_t i) const { return bits.data() + i * words; }
};

inline std::size_t popcountAll(const std::uint64_t* w, std::size_t words) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < words; ++i) c += static_cast<std::size_t>(std::popcount(w[i]));
    return c;
}

// 순위용 점수 격자.
//
// 점수는 Kabsch 의 SVD 를 거친 rms 로 만들어진다. 그래서 수학적으로 완전히
// 동률인 후보들도 마지막 1 ULP 가 갈린다. 실측: 같은 방을 평행이동만 다르게
// 세 번 넣고 질의하면 세 후보 모두 정확 정합인데 rms 가 5.567e-16 / 5.574e-16
// / 6.863e-16 으로 나오고 점수가 0x1.ffffffffffffap-1 대 0x1.ffffffffffff9p-1
// 이 된다. score 를 == 로 비교하면 아래 place_id 타이브레이크가 이 1 ULP 때문에
// 발동하지 못하고, 순위가 SVD 반올림 잡음에 걸린다 - 같은 지도를 다른 BLAS 나
// 다른 컴파일러로 돌리면 다른 장소가 루프 클로저로 뽑힌다. 격자에 올린 뒤
// 비교하면 그 잡음이 사라지고 동률이 실제로 동률로 판정된다.
//
// 간격은 점수의 물리적 분해능보다 열 자리 아래다: 인라이어 하나가 바뀌면
// 점수는 1e-2 규모로 움직이므로 진짜 차이는 그대로 남는다.
inline constexpr double kScoreQuantum = 1e-12;

// 양자화한 순위 키. 격자값끼리의 비교는 전순서라 std::sort 의 strict weak
// ordering 을 만족한다 - |a-b| < eps 식 비교는 추이적이지 않아 쓸 수 없다
// (동치가 전이되지 않아 정렬이 미정의 동작이 된다).
//
// std::round 가 아니라 floor(x+0.5) 인 이유: 참조 구현의 파이썬 round() 는
// 짝수 반올림이라 정확히 절반인 값에서 std::round 와 갈린다. 두 구현이 같은
// 답을 내야 하므로 양쪽 다 floor(x+0.5) 로 맞춘다. 점수는 [0,1] 로 clamp 되어
// 있어 음수 처리는 고려하지 않아도 된다.
inline double rankKey(double score) { return std::floor(score / kScoreQuantum + 0.5); }

// Bron-Kerbosch (피벗 + 예산 제한). 최대 클리크 하나만 찾으면 되므로
// 현재 최선보다 커질 수 없는 가지는 잘라낸다.
struct CliqueSolver {
    const BitAdj&              adj;
    std::size_t                budget;
    std::vector<std::uint32_t> best;
    std::vector<std::uint32_t> current;
    std::size_t                visited{0};

    CliqueSolver(const BitAdj& a, std::size_t b) : adj(a), budget(b) {}

    void run() {
        std::vector<std::uint64_t> P(adj.words, 0ULL);
        for (std::size_t i = 0; i < adj.n; ++i) P[i >> 6] |= (1ULL << (i & 63));
        std::vector<std::uint64_t> X(adj.words, 0ULL);
        expand(P, X);
    }

    void expand(std::vector<std::uint64_t> P, std::vector<std::uint64_t> X) {
        if (++visited > budget) return;

        const std::size_t p_count = popcountAll(P.data(), adj.words);
        if (p_count == 0) {
            if (popcountAll(X.data(), adj.words) == 0 && current.size() > best.size()) best = current;
            return;
        }
        // 가지치기: 남은 후보를 다 넣어도 최선을 못 넘으면 중단
        if (current.size() + p_count <= best.size()) return;

        // 피벗: P∪X 중 P 와의 인접이 가장 많은 정점
        std::size_t pivot = adj.n, best_deg = 0;
        for (std::size_t u = 0; u < adj.n; ++u) {
            const bool in_p = (P[u >> 6] >> (u & 63)) & 1ULL;
            const bool in_x = (X[u >> 6] >> (u & 63)) & 1ULL;
            if (!in_p && !in_x) continue;
            std::size_t deg = 0;
            const std::uint64_t* r = adj.row(u);
            for (std::size_t w = 0; w < adj.words; ++w) deg += static_cast<std::size_t>(std::popcount(P[w] & r[w]));
            if (pivot == adj.n || deg > best_deg) { pivot = u; best_deg = deg; }
        }

        // P \ N(pivot) 순회
        std::vector<std::uint64_t> cand(adj.words);
        const std::uint64_t* pr = (pivot < adj.n) ? adj.row(pivot) : nullptr;
        for (std::size_t w = 0; w < adj.words; ++w) cand[w] = pr ? (P[w] & ~pr[w]) : P[w];

        for (std::size_t w = 0; w < adj.words; ++w) {
            std::uint64_t bits = cand[w];
            while (bits) {
                const std::size_t v = w * 64 + static_cast<std::size_t>(std::countr_zero(bits));
                bits &= bits - 1;

                const std::uint64_t* r = adj.row(v);
                std::vector<std::uint64_t> nP(adj.words), nX(adj.words);
                for (std::size_t k = 0; k < adj.words; ++k) {
                    nP[k] = P[k] & r[k];
                    nX[k] = X[k] & r[k];
                }
                current.push_back(static_cast<std::uint32_t>(v));
                expand(std::move(nP), std::move(nX));
                current.pop_back();

                P[v >> 6] &= ~(1ULL << (v & 63));
                X[v >> 6] |= (1ULL << (v & 63));
                if (visited > budget) return;
            }
        }
    }
};

}  // namespace

// ---------------------------------------------------------------------------

Result<SE3> kabsch(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
    if (src.size() != dst.size() || src.size() < 3) {
        return {ErrorCode::InsufficientData, "kabsch 는 3개 이상의 대응이 필요"};
    }
    const auto n = static_cast<double>(src.size());

    Vec3 mu_s = Vec3::Zero(), mu_d = Vec3::Zero();
    for (std::size_t i = 0; i < src.size(); ++i) { mu_s += src[i]; mu_d += dst[i]; }
    mu_s /= n; mu_d /= n;

    Mat3 H = Mat3::Zero();
    for (std::size_t i = 0; i < src.size(); ++i) {
        H.noalias() += (src[i] - mu_s) * (dst[i] - mu_d).transpose();
    }

    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 R = svd.matrixV() * svd.matrixU().transpose();

    // det = -1 이면 반사. 최소 특이값에 대응하는 축을 뒤집어 회전으로 되돌린다.
    if (R.determinant() < 0.0) {
        Mat3 V = svd.matrixV();
        V.col(2) *= -1.0;
        R = V * svd.matrixU().transpose();
    }

    // 퇴화 검사: 점들이 한 직선 위에 있으면 회전이 결정되지 않는다
    const auto& sv = svd.singularValues();
    if (sv(1) < 1e-6 * std::max(1.0, sv(0))) {
        return {ErrorCode::Degenerate, "대응점이 공선 배치"};
    }

    return SE3(SO3(R), mu_d - R * mu_s);
}

// ---------------------------------------------------------------------------

ConstellationIndex::ConstellationIndex(ConstellationConfig config) : cfg_(config) {}

void ConstellationIndex::clear() {
    places_.clear();
    inverted_.clear();
    place_lookup_.clear();
    next_place_id_ = 1;
}

const Place* ConstellationIndex::place(std::uint64_t id) const {
    const auto it = place_lookup_.find(id);
    return (it == place_lookup_.end()) ? nullptr : &places_[it->second];
}

int ConstellationIndex::distanceBin(double d) const {
    if (d < cfg_.min_distance || d > cfg_.max_distance) return -1;
    // 로그 스케일 구간화: 근거리 정밀, 원거리 관대
    const double t = std::log(d / cfg_.min_distance) /
                     std::log(cfg_.max_distance / cfg_.min_distance);
    return std::clamp(static_cast<int>(t * cfg_.distance_bins), 0, cfg_.distance_bins - 1);
}

std::uint64_t ConstellationIndex::pairKey(int class_a, int class_b, double distance) const {
    const int bin = distanceBin(distance);
    if (bin < 0) return 0;
    // 클래스 순서 정규화로 순서 불변성 확보
    const auto lo = static_cast<std::uint64_t>(std::min(class_a, class_b) & 0xFFFF);
    const auto hi = static_cast<std::uint64_t>(std::max(class_a, class_b) & 0xFFFF);
    return mix64((lo << 32) | (hi << 16) | static_cast<std::uint64_t>(bin));
}

std::vector<std::uint64_t> ConstellationIndex::signature(
    const std::vector<ConstellationNode>& nodes) const {
    std::vector<std::uint64_t> sig;
    if (nodes.size() < 2) return sig;
    sig.reserve(nodes.size() * (nodes.size() - 1) / 2);

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1; j < nodes.size(); ++j) {
            const double d = (nodes[i].position - nodes[j].position).norm();
            const std::uint64_t k = pairKey(nodes[i].class_id, nodes[j].class_id, d);
            if (k != 0) sig.push_back(k);
        }
    }
    return sig;
}

std::uint64_t ConstellationIndex::insert(KeyframeId kf, Timestamp stamp, const SE3& anchor,
                                         std::vector<ConstellationNode> nodes,
                                         std::optional<Vec3> gravity) {
    if (nodes.size() < cfg_.min_nodes) return 0;
    if (nodes.size() > cfg_.max_nodes) nodes.resize(cfg_.max_nodes);

    Place p;
    p.place_id = next_place_id_++;
    p.keyframe = kf;
    p.stamp    = stamp;
    p.anchor   = anchor;
    p.nodes    = std::move(nodes);
    if (gravity && gravity->norm() > kEps) p.gravity = gravity->normalized();

    const auto index = static_cast<std::uint32_t>(places_.size());
    const auto sig   = signature(p.nodes);

    places_.push_back(std::move(p));
    place_lookup_[places_.back().place_id] = index;

    // 같은 키가 여러 번 나와도 장소는 한 번만 등록 (투표 편향 방지)
    std::vector<std::uint64_t> unique_keys(sig);
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
    for (std::uint64_t k : unique_keys) inverted_[k].push_back(index);

    return places_.back().place_id;
}

std::vector<std::pair<std::uint64_t, double>> ConstellationIndex::retrieve(
    const std::vector<std::uint64_t>& sig) const {

    std::unordered_map<std::uint32_t, double> votes;
    votes.reserve(places_.size() / 4 + 8);

    // idf 가중 투표: 흔한 쌍(어디에나 있는 '의자-의자 2m')은 변별력이 없다
    const double n_places = static_cast<double>(std::max<std::size_t>(1, places_.size()));
    std::vector<std::uint64_t> unique_keys(sig);
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());

    for (std::uint64_t k : unique_keys) {
        const auto it = inverted_.find(k);
        if (it == inverted_.end()) continue;
        const double df  = static_cast<double>(it->second.size());
        const double idf = std::log(1.0 + n_places / df);
        for (std::uint32_t idx : it->second) votes[idx] += idf;
    }

    std::vector<std::pair<std::uint64_t, double>> ranked;
    ranked.reserve(votes.size());
    for (const auto& [idx, score] : votes) {
        // 장소 크기로 정규화하지 않으면 객체가 많은 장소가 항상 이긴다
        const double norm = std::sqrt(static_cast<double>(places_[idx].nodes.size()));
        ranked.emplace_back(places_[idx].place_id, score / norm);
    }
    // ranked 는 unordered_map 을 훑어 만들었으므로 초기 순서가 해시 순회 순서다.
    // 점수만으로 정렬하면 동점 구간에 그 순서가 그대로 남고, 그 뒤의
    // top_candidates 절단이 무엇을 자를지가 해시에 달리게 된다.
    // place_id 로 타이브레이크하면 전순서가 되어 초기 순서와 무관해진다.
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    if (ranked.size() > cfg_.top_candidates) ranked.resize(cfg_.top_candidates);
    return ranked;
}

Result<ConstellationMatch> ConstellationIndex::verify(
    const std::vector<ConstellationNode>& query, const Place& place,
    const std::optional<Vec3>& query_gravity) const {

    // 양쪽 프레임의 중력을 모두 알 때만 카이랄리티를 적용한다.
    // 한쪽 벡터로 두 프레임을 재면 정상 대응이 통째로 걸러진다.
    const bool use_chirality = cfg_.use_chirality && query_gravity && place.gravity;

    // 1) 클래스 일관 대응 후보 생성
    struct Pair { std::uint16_t q, p; };
    std::vector<Pair> pairs;
    pairs.reserve(cfg_.max_pairs);

    const auto nq = static_cast<std::uint16_t>(std::min<std::size_t>(query.size(), 0xFFFF));
    const auto np = static_cast<std::uint16_t>(std::min<std::size_t>(place.nodes.size(), 0xFFFF));
    for (std::uint16_t i = 0; i < nq && pairs.size() < cfg_.max_pairs; ++i) {
        for (std::uint16_t j = 0; j < np && pairs.size() < cfg_.max_pairs; ++j) {
            if (query[i].class_id == place.nodes[j].class_id) pairs.push_back({i, j});
        }
    }
    if (pairs.size() < cfg_.min_inliers) {
        return {ErrorCode::InsufficientData, "클래스 일치 대응 부족"};
    }

    // 2) 쌍거리 일관성 그래프. 두 대응이 동시에 성립 가능하면 간선을 잇는다.
    BitAdj adj;
    adj.resize(pairs.size());

    for (std::size_t a = 0; a < pairs.size(); ++a) {
        for (std::size_t b = a + 1; b < pairs.size(); ++b) {
            // 같은 노드를 두 번 쓰는 대응은 배타적
            if (pairs[a].q == pairs[b].q || pairs[a].p == pairs[b].p) continue;

            const double dq = (query[pairs[a].q].position - query[pairs[b].q].position).norm();
            const double dp = (place.nodes[pairs[a].p].position -
                               place.nodes[pairs[b].p].position).norm();

            // 거리차의 표준편차는 네 노드 분산의 제곱합근이다.
            // 단순 합은 지나치게 관대해 오대응을 통과시킨다.
            const double var = query[pairs[a].q].sigma * query[pairs[a].q].sigma +
                               query[pairs[b].q].sigma * query[pairs[b].q].sigma +
                               place.nodes[pairs[a].p].sigma * place.nodes[pairs[a].p].sigma +
                               place.nodes[pairs[b].p].sigma * place.nodes[pairs[b].p].sigma;
            const double tol = cfg_.distance_tolerance +
                               cfg_.relative_tolerance * std::max(dq, dp) +
                               cfg_.sigma_gate * std::sqrt(var);

            if (std::abs(dq - dp) > tol) continue;

            // 카이랄리티: 중력축 기준 상하 배치가 뒤집힌 대응을 배제.
            // 각 프레임의 높이는 그 프레임 자신의 중력벡터로 재야 한다.
            if (use_chirality) {
                const Vec3 vq = query[pairs[b].q].position - query[pairs[a].q].position;
                const Vec3 vp = place.nodes[pairs[b].p].position - place.nodes[pairs[a].p].position;
                const double hq = vq.dot(*query_gravity);
                const double hp = vp.dot(*place.gravity);
                // 높이차가 유의미한데 부호가 반대면 불일치
                if (std::abs(hq) > 0.3 && std::abs(hp) > 0.3 && hq * hp < 0.0) continue;
            }
            adj.set(a, b);
            adj.set(b, a);
        }
    }

    // 3) 최대 클리크 = 최대 상호일관 대응 집합
    CliqueSolver solver(adj, cfg_.clique_budget);
    solver.run();

    if (solver.best.size() < cfg_.min_inliers) {
        return {ErrorCode::InsufficientData, "일관 대응 클리크가 너무 작음"};
    }

    // 4) Kabsch 로 SE(3) 복원
    std::vector<Vec3> src, dst;
    src.reserve(solver.best.size());
    dst.reserve(solver.best.size());
    for (std::uint32_t idx : solver.best) {
        src.push_back(query[pairs[idx].q].position);
        dst.push_back(place.nodes[pairs[idx].p].position);
    }

    auto T = kabsch(src, dst);
    if (!T) return {T.error().code, T.error().detail};

    // 5) 잔차 검증 + 카이제곱 게이트
    //
    // 카이제곱은 노드 자신의 위치 분산으로 정규화한 잔차다. RMS 는 미터 단위라
    // 정밀한 근거리 노드와 거친 원거리 노드를 같은 저울에 올린다.
    // 자유도 = 3*n - 6 (Kabsch 가 6 DoF 를 소모).
    double sse = 0.0;
    double chi2 = 0.0;
    std::size_t gated = 0;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const Vec3   r  = T.value() * src[i] - dst[i];
        const double r2 = r.squaredNorm();
        sse += r2;

        const std::uint32_t idx = solver.best[i];
        const double sq = query[pairs[idx].q].sigma;
        const double sp = place.nodes[pairs[idx].p].sigma;
        const double s  = sq + sp + 0.05;
        if (r2 / (s * s) < cfg_.chi2_gate) ++gated;

        // 축당 분산은 두 노드 분산의 합. 바닥값을 두어 sigma=0 발산을 막는다.
        const double var = std::max(1e-4, sq * sq + sp * sp);
        chi2 += r2 / var;
    }
    const double rms = std::sqrt(sse / static_cast<double>(src.size()));

    if (rms > cfg_.max_rms_error || gated < cfg_.min_inliers) {
        return {ErrorCode::DidNotConverge, "정합 잔차 과대"};
    }

    ConstellationMatch m;
    m.place_id  = place.place_id;
    m.keyframe  = place.keyframe;
    m.transform = T.value();
    m.rms_error = rms;

    // 점수: 인라이어 수(변별력)와 잔차(정확도)를 함께 반영
    const double inlier_term = static_cast<double>(gated) /
                               static_cast<double>(std::min(query.size(), place.nodes.size()));
    const double error_term  = 1.0 - std::min(1.0, rms / cfg_.max_rms_error);
    m.score = std::clamp(std::sqrt(std::max(0.0, inlier_term) * error_term), 0.0, 1.0);

    m.n_query_nodes = query.size();
    m.n_place_nodes = place.nodes.size();
    m.n_inliers     = gated;
    // 설명률: 질의 성좌 중 이 장소가 설명한 비율. 8개 중 4개만 설명한 정합은
    // 4개 중 4개를 설명한 정합과 같은 rms 를 가질 수 있지만 훨씬 약한 증거다.
    m.explained = query.empty() ? 0.0
                                : static_cast<double>(gated) / static_cast<double>(query.size());
    const double dof = std::max(1.0, 3.0 * static_cast<double>(src.size()) - 6.0);
    m.chi2_dof  = chi2 / dof;

    m.correspondences.reserve(solver.best.size());
    for (std::uint32_t idx : solver.best) {
        m.correspondences.emplace_back(query[pairs[idx].q].id, place.nodes[pairs[idx].p].id);
    }
    return m;
}

std::optional<SE3> ConstellationIndex::worldPose(const ConstellationMatch& m) const {
    const Place* p = place(m.place_id);
    if (p == nullptr) return std::nullopt;
    return p->anchor * m.transform;   // 질의 프레임의 월드 포즈
}

void ConstellationIndex::annotate(std::vector<ConstellationMatch>& all) const {
    if (all.empty()) return;

    std::vector<std::optional<SE3>> poses;
    poses.reserve(all.size());
    for (const auto& m : all) poses.push_back(worldPose(m));

    // 1) 후보를 월드 포즈로 탐욕 군집화. all 은 점수 내림차순이므로 각 군집의
    //    대표는 그 군집에서 가장 점수가 높은 후보가 된다.
    std::vector<std::size_t> cluster_of(all.size(), 0);
    std::vector<std::size_t> reps;              // 군집 대표 인덱스
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (!poses[i]) { cluster_of[i] = reps.size(); reps.push_back(i); continue; }
        std::size_t hit = reps.size();
        for (std::size_t c = 0; c < reps.size(); ++c) {
            const auto& rp = poses[reps[c]];
            if (!rp) continue;
            const Vec2 d = poses[i]->distanceTo(*rp);   // (병진 m, 회전 rad)
            if (d.x() <= cfg_.pose_agree_radius &&
                d.y() * kRad2Deg <= cfg_.pose_agree_deg) { hit = c; break; }
        }
        if (hit == reps.size()) reps.push_back(i);
        cluster_of[i] = hit;
    }

    // 2) 군집 질량 = 점수합. 개수가 아니라 질량으로 재야 약한 후보 여럿이
    //    강한 후보 하나를 이기지 못한다.
    std::vector<double>      mass(reps.size(), 0.0);
    std::vector<std::size_t> size(reps.size(), 0);
    for (std::size_t i = 0; i < all.size(); ++i) {
        mass[cluster_of[i]] += all[i].score;
        ++size[cluster_of[i]];
    }

    // 3) 각 후보에 자기 군집과 최강 경쟁 군집을 붙인다.
    for (std::size_t i = 0; i < all.size(); ++i) {
        auto&             m  = all[i];
        const std::size_t ci = cluster_of[i];
        m.agree_count = size[ci];
        m.support     = mass[ci];
        m.rival_mass  = 0.0;
        m.pose_margin = kNoRivalMargin;

        for (std::size_t c = 0; c < reps.size(); ++c) {
            if (c == ci) continue;
            if (mass[c] > m.rival_mass) m.rival_mass = mass[c];
            if (poses[i] && poses[reps[c]]) {
                m.pose_margin = std::min(m.pose_margin,
                                         poses[i]->distanceTo(*poses[reps[c]]).x());
            }
        }

        // 신뢰도 = 자기일관성 x 포즈 공간 우세 x 정규화 잔차.
        //
        // 세 항이 서로 다른 것을 잰다.
        //  - score  : 대응이 몇 개나 붙었고 잔차가 미터 단위로 얼마나 작은가.
        //  - 우세   : 다른 장소에서 독립적으로 풀린 포즈가 같은 곳을 가리키는가.
        //             score/rms/카이제곱은 한 후보 안에서만 계산되므로 "옳은
        //             성좌인가"를 모른다 - 동일한 모니터 두 대에 어긋나게 붙인
        //             대응도 완벽히 자기일관적이다. 이 항이 그 구멍을 메운다.
        //             질량비 mass/(mass+rival) 로 0..1 에 넣는다. 절대 질량을
        //             쓰면 지도 샘플링 밀도를 재게 되므로 반드시 비로 쓴다.
        //  - 카이제곱: 노드 자신의 공분산으로 정규화한 잔차. rms 는 미터 단위라
        //             정밀한 근거리 노드와 거친 원거리 노드를 같은 저울에 올린다.
        //
        // TUM 5개 시퀀스 실측에서 corr(신호, 포즈오차):
        //   score 단독 -0.52/-0.39/-0.54/-0.08 (xyz/desk/sitting/walking)
        //   score x 우세 -0.52/+0.11/-0.66/+0.02
        //   위 셋의 곱   -0.58/-0.56/-0.72/-0.10
        // 우세 항 단독은 desk 에서 오히려 해롭고 카이제곱이 그것을 되돌린다.
        // walking(동적 장면)은 어느 조합으로도 예측되지 않는다 - 그것이 결과다.
        const double denom     = m.support + m.rival_mass;
        const double dominance = (denom > kEps) ? (m.support / denom) : 0.0;
        const double chi2_term =
            1.0 / (1.0 + std::max(0.0, m.chi2_dof) /
                             std::max(kEps, cfg_.chi2_confidence_scale));
        m.confidence = std::clamp(m.score * dominance * chi2_term, 0.0, 1.0);
    }
}

std::vector<ConstellationMatch> ConstellationIndex::queryAll(
    const std::vector<ConstellationNode>& nodes, std::optional<Vec3> gravity) const {

    std::vector<ConstellationMatch> results;
    if (nodes.size() < cfg_.min_nodes || places_.empty()) return results;

    if (gravity && gravity->norm() > kEps) gravity = gravity->normalized();
    else                                   gravity.reset();

    for (const auto& [pid, vote] : retrieve(signature(nodes))) {
        const Place* p = place(pid);
        if (p == nullptr) continue;
        auto m = verify(nodes, *p, gravity);
        if (m) results.push_back(std::move(m).value());
    }
    // 여기서도 동점 타이브레이크가 필요하다. query() 는 all.front() 를 대표로
    // 뽑고 annotate() 는 이 순서로 군집 대표를 정하므로, 동점이 남으면 후보
    // 생성 순서(= retrieve 의 순서)가 결과로 새어 나간다.
    //
    // 점수는 격자에 올려서 비교한다. 원시 double 을 == 로 재면 SVD 반올림
    // 잡음 1 ULP 가 타이브레이크를 건너뛰게 만든다 (kScoreQuantum 주석 참고).
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        const double ka = rankKey(a.score);
        const double kb = rankKey(b.score);
        if (ka != kb) return ka > kb;
        return a.place_id < b.place_id;
    });
    annotate(results);
    return results;
}

Result<ConstellationMatch> ConstellationIndex::query(
    const std::vector<ConstellationNode>& nodes, std::optional<Vec3> gravity) const {

    if (nodes.size() < cfg_.min_nodes) {
        return {ErrorCode::InsufficientData, "성좌 구성 객체 부족"};
    }
    auto all = queryAll(nodes, gravity);
    if (all.empty()) return {ErrorCode::NotAvailable, "일치하는 장소 없음"};

    // --- 모호성 판정: 포즈 공간의 마진 -------------------------------------
    //
    // 예전 규칙은 score2 > 0.85*score1 이면 기각이었다. 그 규칙은 "점수가 붙으면
    // 지각적 혼동" 이라고 가정하지만, 5프레임 간격으로 등록한 지도에서는 이웃
    // 키프레임이 같은 장면을 보므로 점수가 붙는 것이 정상이다. fr1_xyz 에서
    // 이 규칙은 정대응 36개를 36개 모두 기각했다 (재현율 0 %).
    //
    // 실제로 물어야 할 것은 "붙은 후보들이 같은 포즈를 가리키는가" 다. annotate()
    // 가 후보를 월드 포즈로 군집화해 두었으므로 여기서는 질량비만 본다.
    //   - 이웃 키프레임: 서로 다른 anchor x 서로 다른 transform -> 한 군집. 채택.
    //   - 동일한 복도 두 개: 같은 transform x 50 m 떨어진 anchor -> 두 군집, 질량
    //     동률 -> 기각.
    // 점수 공간에서는 두 상황이 똑같이 보이고, 포즈 공간에서는 다르게 보인다.
    //
    // 대표는 여전히 최고점 후보다. "가장 무거운 군집의 대표를 고른다"도 시험했고
    // 어느 시퀀스에서도 이기지 못했다 (fr3_walking 에서는 정밀도 42 -> 28 %).
    // 동적 장면에서는 이웃 장소들이 같은 오답에 합의하기 때문에 질량이 증거가
    // 되지 못한다. 그래서 질량은 *기각* 에만 쓰고 *선택* 에는 쓰지 않는다.
    const auto& top = all.front();

    if (top.agree_count < cfg_.min_agree) {
        return {ErrorCode::NotAvailable, "포즈 군집이 너무 작음"};
    }
    if (top.rival_mass > kEps && top.support < cfg_.pose_dominance * top.rival_mass) {
        return {ErrorCode::NotAvailable, "포즈 군집 경쟁 - 지각적 혼동 가능"};
    }
    if (top.confidence < cfg_.min_confidence) {
        return {ErrorCode::NotAvailable, "신뢰도 미달"};
    }
    return top;
}

std::vector<ConstellationNode> ConstellationIndex::buildFrom(
    const std::vector<WorldTokenPtr>& tokens, const SE3& reference, std::size_t max_nodes) {

    std::vector<ConstellationNode> nodes;
    nodes.reserve(tokens.size());

    const SE3 world_to_ref = reference.inverse();
    for (const auto& t : tokens) {
        if (!t || !t->isStableLandmark()) continue;   // 움직이는 물체는 장소를 정의하지 못한다
        ConstellationNode n;
        n.id       = t->id;
        n.class_id = t->class_id;
        n.position = world_to_ref * t->position;
        n.sigma    = t->positionSigma();
        nodes.push_back(n);
    }

    // 위치 정밀도가 높은 순으로 상한만큼 취한다
    std::sort(nodes.begin(), nodes.end(),
              [](const auto& a, const auto& b) { return a.sigma < b.sigma; });
    if (nodes.size() > max_nodes) nodes.resize(max_nodes);
    return nodes;
}

}  // namespace wme
