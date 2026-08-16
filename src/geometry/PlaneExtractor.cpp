#include "wme/geometry/PlaneExtractor.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace wme {

namespace {

// 격자 클러스터 키. (법선 방향 3, 거리 1) 을 int64 하나로 접는다.
// 성분당 16 비트. 범위를 벗어나는 점은 애초에 깊이 게이트에서 걸린다.
constexpr std::int64_t kKeyOffset = 2048;
constexpr std::int64_t kKeyMax    = 60000;

[[nodiscard]] bool packKey(std::int64_t kx, std::int64_t ky, std::int64_t kz,
                           std::int64_t kd, std::int64_t& out) {
    const std::int64_t v[4] = {kx + kKeyOffset, ky + kKeyOffset,
                               kz + kKeyOffset, kd + kKeyOffset};
    for (const std::int64_t x : v) {
        if (x < 0 || x > kKeyMax) return false;
    }
    out = (v[0] << 48) | (v[1] << 32) | (v[2] << 16) | v[3];
    return true;
}

}  // namespace

// --- Plane -----------------------------------------------------------------

double Plane::fitDegradation() const {
    return 1.0 + 20.0 * rms;
}

double Plane::confidence() const {
    const double size = std::min(1.0, static_cast<double>(inliers) / 400.0);
    return size / fitDegradation();
}

Plane Plane::transformed(const Mat3& R, const Vec3& t) const {
    Plane p = *this;
    p.normal   = R * normal;
    p.distance = distance + p.normal.dot(t);
    p.centroid = R * centroid + t;
    return p;
}

std::size_t NormalField::validCount() const {
    std::size_t n = 0;
    for (const std::uint8_t v : valid) n += (v != 0) ? 1u : 0u;
    return n;
}

// --- PlaneExtractor --------------------------------------------------------

PlaneExtractor::PlaneExtractor(PlaneExtractorConfig config) : cfg_(config) {}

Result<NormalField> PlaneExtractor::estimateNormals(const cv::Mat& depth,
                                                    const CameraIntrinsics& K) {
    if (depth.empty() || depth.type() != CV_32F) {
        return {ErrorCode::InvalidArgument, "깊이맵은 비어있지 않은 CV_32F 이어야 함"};
    }
    if (!K.valid()) return {ErrorCode::InvalidArgument, "내부 파라미터 무효"};
    const int stride = std::max(1, cfg_.stride);

    // 서브샘플 격자. 법선은 (i, j) 와 오른쪽/아래 이웃의 외적이라 한 줄씩 짧다.
    const int gh = (depth.rows + stride - 1) / stride;
    const int gw = (depth.cols + stride - 1) / stride;
    if (gh < 2 || gw < 2) {
        return {ErrorCode::InsufficientData, "격자가 너무 작아 법선을 만들 수 없음"};
    }

    // 역투영. 유한하지 않은 깊이는 0 으로 눌러 두고 마스크로 걸러낸다.
    std::vector<Vec3> grid(static_cast<std::size_t>(gh) * gw);
    std::vector<double> gd(grid.size());
    for (int i = 0; i < gh; ++i) {
        const int v = i * stride;
        const auto* row = depth.ptr<float>(v);
        for (int j = 0; j < gw; ++j) {
            const int u = j * stride;
            const float raw = row[u];
            const double d = std::isfinite(raw) ? static_cast<double>(raw) : 0.0;
            const std::size_t k = static_cast<std::size_t>(i) * gw + j;
            gd[k] = d;
            grid[k] = Vec3((u - K.cx) * d / K.fx, (v - K.cy) * d / K.fy, d);
        }
    }

    NormalField field;
    field.rows = gh - 1;
    field.cols = gw - 1;
    const std::size_t n = static_cast<std::size_t>(field.rows) * field.cols;
    field.normal.assign(n, Vec3::Zero());
    field.point.assign(n, Vec3::Zero());
    field.valid.assign(n, 0);

    for (int i = 0; i < field.rows; ++i) {
        for (int j = 0; j < field.cols; ++j) {
            const std::size_t k0 = static_cast<std::size_t>(i) * gw + j;
            const Vec3& p  = grid[k0];
            const Vec3 a = grid[k0 + 1] - p;              // 오른쪽 이웃
            const Vec3 b = grid[k0 + gw] - p;             // 아래 이웃
            const Vec3 cr = a.cross(b);
            const double len = cr.norm();

            const std::size_t k = static_cast<std::size_t>(i) * field.cols + j;
            field.point[k] = p;
            if (len < 1e-9) continue;

            const double d0 = gd[k0], dr = gd[k0 + 1], dd = gd[k0 + gw];
            const bool continuous =
                std::abs(dr - d0) < cfg_.depth_discontinuity &&
                std::abs(dd - d0) < cfg_.depth_discontinuity &&
                d0 > cfg_.min_depth && d0 < cfg_.max_depth;

            Vec3 nrm = cr / len;
            // Hesse 표준형 n·p = d, d > 0 에 맞춘다. 법선은 카메라 반대쪽을 향한다.
            if (nrm.dot(p) < 0.0) nrm = -nrm;

            field.normal[k] = nrm;
            field.valid[k]  = continuous ? 1u : 0u;
        }
    }
    return field;
}

std::size_t PlaneExtractor::gather(const NormalField& field) {
    points_.clear();
    normals_.clear();
    dists_.clear();
    for (std::size_t k = 0; k < field.size(); ++k) {
        if (field.valid[k] == 0) continue;
        const double dist = field.normal[k].dot(field.point[k]);
        // 원점을 지나는 평면은 Hesse 규약이 성립하지 않고, 너무 먼 것은 잡음이다
        if (dist <= cfg_.min_depth * 0.5 || dist >= cfg_.max_depth) continue;
        points_.push_back(field.point[k]);
        normals_.push_back(field.normal[k]);
        dists_.push_back(dist);
    }
    return points_.size();
}

bool PlaneExtractor::fit(const std::vector<std::uint32_t>& idx, Plane& out) const {
    if (idx.size() < 3) return false;

    Vec3 centroid = Vec3::Zero();
    for (const std::uint32_t i : idx) centroid += points_[i];
    centroid /= static_cast<double>(idx.size());

    Mat3 cov = Mat3::Zero();
    double spread = 0.0;
    for (const std::uint32_t i : idx) {
        const Vec3 c = points_[i] - centroid;
        cov.noalias() += c * c.transpose();
        spread += c.norm();
    }

    Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
    if (es.info() != Eigen::Success) return false;

    // 고유값은 오름차순. 특이값 s_i = sqrt(lambda_i).
    const Vec3 lam = es.eigenvalues().cwiseMax(0.0);
    const double s_min = std::sqrt(lam(0)), s_mid = std::sqrt(lam(1));

    // 최소 특이값이 중간 특이값과 비슷하면 평면이 아니다 (구·모서리).
    if (s_mid < 1e-9 || s_min / s_mid > cfg_.planarity_ratio) return false;

    Vec3 normal = es.eigenvectors().col(0);
    double distance = centroid.dot(normal);
    if (distance < 0.0) { normal = -normal; distance = -distance; }

    double sq = 0.0;
    for (const std::uint32_t i : idx) {
        const double r = points_[i].dot(normal) - distance;
        sq += r * r;
    }

    out.normal   = normal;
    out.distance = distance;
    out.centroid = centroid;
    out.inliers  = idx.size();
    out.extent   = spread / static_cast<double>(idx.size());
    out.rms      = std::sqrt(sq / static_cast<double>(idx.size()));

    // --- 오프셋 불확실성 -----------------------------------------------------
    //
    // 1. 깊이 잡음이 평면 잔차에 어떻게 들어오는가.
    //    sigma_z = c z^2 이고 그 잡음은 **시선 방향**이다. 화소 시선을 m
    //    (m.z = 1) 이라 하면 점은 p = z m 이고 잡음은 dp = dz * m 이다. 평면
    //    잔차 r = n.p - d 에는 법선 성분만 남으므로
    //
    //        sigma_r = sigma_z * |n.m| = c z^2 * (d / z) = c * z * d
    //
    //    이다 (평면 위에서 n.p = d 이므로 n.m = n.p / z = d / z).
    //    즉 점 하나의 유효 잡음은 깊이에 **선형**이고 평면까지의 거리에
    //    비례한다. z 는 centroid.z() 다 - centroid 가 인라이어의 평균이므로
    //    그 z 성분이 곧 평균 깊이다.
    //
    // 2. 표본 수 N 으로 나누지 **않는다**.
    //    독립 표본이면 sigma_r/sqrt(N) 이어야 하지만 이 깊이맵들의 잡음은
    //    패치 전체에 결맞아 있다. 증거는 이 함수가 방금 계산한 rms 자신이다:
    //    KITTI 21 m 평면에서 유도값 sigma_r = 0.32 m 인데 같은 평면의 rms 는
    //    중앙값 0 이다(refine_threshold 0.04 m 안에 전부 들어온다). 화소마다
    //    독립인 0.32 m 잡음이 있었다면 그럴 수 없다. SGBM 도 구조광도 시차를
    //    공간적으로 매끄럽게 만들어서, 패치 **안**의 상대 오차는 작고 패치
    //    **전체**가 통째로 밀린다. 그 통째 이동은 평균화되지 않는다.
    //
    //    실측이 확인한다. 정답 포즈로 두 프레임의 같은 평면을 맞대고 남는
    //    오프셋 차이의 중앙값을 예측 sigma 로 나누면 (6 시퀀스, 961 대응)
    //        sigma_r        : 0.37 0.23 0.25 1.36 0.69 0.46   시퀀스 간 산포 5.9 배
    //        sigma_r/sqrt(N): 7.3  5.8  4.5  62   49   34     산포 13.6 배, 값도 5~60 배 어긋남
    //        현재 상수 0.05 : 2.7  2.1  2.1  0.55 0.096 0.069 산포 39 배
    //    평균화는 일어나지 않고, 유도값은 상수보다 산포를 7 배 줄인다.
    //
    //    **결맞음의 형태까지 맞힌 것은 아니다.** 잡음이 정확히 반지름 방향
    //    배율(z' = z(1 + c z g))이라면 평면은 기울어지면서 그 기울기를 적합이
    //    흡수하고, 오프셋에 남는 몫은 c z d 가 아니라 정확히 c d^2 n_z 다
    //    (정면 n_z=1 이면 c z^2, 광축과 나란한 벽 n_z=0 이면 0). 그 형태를 같은
    //    실측에 대고 재면 산포가 8.3 배로 5.9 배보다 **나쁘다**. 즉 실제 오차는
    //    순수한 반지름 배율도 아니고 화소별 독립도 아니다. 여기서는 유도한
    //    점 잡음 sigma_r 을 평균화되지 않는 바닥으로 쓰는 형태를 택했고, 그것이
    //    시험한 형태들 중 실데이터에서 가장 안정적이었다. 이 한계는 미해결이다.
    //
    // 3. 적합 잔차를 직교로 더한다.
    //    rms 는 이 평면 자신이 잰 **비결맞은** 산포다. 유도가 예측한 것보다
    //    실제로 더 지저분한 평면(곡면, 잘못 묶인 클러스터)은 그것으로 스스로
    //    신고한다. 유도값과 실측값 중 큰 쪽이 이기는 구조라 검산이 내장된다.
    //
    // c 를 모르면(depth_sigma_rel <= 0) 유도할 근거가 없으므로 0(미상)으로 둔다.
    // DirectAligner::depthNoiseFloor 이 같은 자리에서 같은 선택을 한다.
    if (cfg_.depth_sigma_rel > 0.0 && centroid.z() > 0.0) {
        const double sigma_r = cfg_.depth_sigma_rel * centroid.z() * distance;
        out.sigma_offset = std::sqrt(sigma_r * sigma_r + out.rms * out.rms);
    } else {
        out.sigma_offset = 0.0;
    }
    return true;
}

Result<std::vector<Plane>> PlaneExtractor::extract(const cv::Mat& depth,
                                                   const CameraIntrinsics& K) {
    auto fr = estimateNormals(depth, K);
    if (!fr.ok()) return fr.error();
    field_ = std::move(fr).value();

    if (gather(field_) == 0) {
        return {ErrorCode::InsufficientData, "유효 깊이 이웃이 없어 법선을 만들지 못함"};
    }

    // (법선 방향, 거리) 격자로 후보를 모은다. 정확한 값은 뒤에서 재적합한다.
    std::unordered_map<std::int64_t, std::vector<std::uint32_t>> cells;
    cells.reserve(points_.size() / 4 + 1);
    for (std::uint32_t i = 0; i < points_.size(); ++i) {
        const Vec3& n = normals_[i];
        std::int64_t key = 0;
        const bool okk = packKey(std::llround(n.x() * cfg_.normal_bins),
                                 std::llround(n.y() * cfg_.normal_bins),
                                 std::llround(n.z() * cfg_.normal_bins),
                                 std::llround(dists_[i] / cfg_.distance_bin), key);
        if (!okk) continue;
        cells[key].push_back(i);
    }

    // unordered_map 순회 순서는 미정이다. 결정적 결과를 위해 (개수 내림, 키 오름)
    // 으로 정렬한다 - 같은 입력에 같은 평면 집합이 나와야 한다.
    std::vector<std::pair<std::int64_t, const std::vector<std::uint32_t>*>> order;
    order.reserve(cells.size());
    for (const auto& kv : cells) order.emplace_back(kv.first, &kv.second);
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        if (a.second->size() != b.second->size()) return a.second->size() > b.second->size();
        return a.first < b.first;
    });

    used_.assign(points_.size(), 0);
    std::vector<Plane> planes;

    for (const auto& [key, members] : order) {
        (void)key;
        if (planes.size() >= cfg_.max_planes) break;
        if (members->size() < cfg_.min_inliers) break;   // 개수 내림차순이라 이후도 전부 미달

        scratch_idx_.clear();
        for (const std::uint32_t i : *members) {
            if (used_[i] == 0) scratch_idx_.push_back(i);
        }
        if (scratch_idx_.size() < cfg_.min_inliers) continue;

        Plane plane;
        if (!fit(scratch_idx_, plane)) continue;
        std::vector<std::uint32_t> mask = scratch_idx_;

        // 재적합: 격자 경계에서 잘린 인라이어를 되찾는다.
        for (int it = 0; it < cfg_.refine_iterations; ++it) {
            scratch_idx_.clear();
            for (std::uint32_t i = 0; i < points_.size(); ++i) {
                if (used_[i] != 0) continue;
                if (std::abs(points_[i].dot(plane.normal) - plane.distance) < cfg_.refine_threshold) {
                    scratch_idx_.push_back(i);
                }
            }
            if (scratch_idx_.size() < cfg_.min_inliers) break;
            Plane refined;
            if (!fit(scratch_idx_, refined)) break;
            plane = refined;
            mask  = scratch_idx_;
        }

        if (plane.inliers < cfg_.min_inliers) continue;
        for (const std::uint32_t i : mask) used_[i] = 1;
        planes.push_back(plane);
    }

    std::sort(planes.begin(), planes.end(), [](const Plane& a, const Plane& b) {
        const double ca = a.confidence(), cb = b.confidence();
        if (ca != cb) return ca > cb;
        return a.inliers > b.inliers;
    });
    return planes;
}

// --- 중력 ------------------------------------------------------------------

Result<Vec3> dominantGravity(const std::vector<Plane>& planes, double min_confidence,
                             double min_verticality) {
    const Plane* best = nullptr;
    double best_score = 0.0;

    for (const Plane& pl : planes) {
        if (pl.confidence() < min_confidence) continue;
        // 카메라 좌표계는 y 가 아래다. 바닥/천장은 법선의 y 성분이 지배적이다.
        const double vertical = std::abs(pl.normal.y());
        if (vertical < min_verticality) continue;
        const double score = vertical * pl.confidence() * static_cast<double>(pl.inliers);
        if (score > best_score) { best = &pl; best_score = score; }
    }

    if (best == nullptr) {
        return {ErrorCode::NotAvailable, "수평면이 없어 중력 방향을 정할 수 없음"};
    }
    // 중력은 아래(+y) 방향
    return (best->normal.y() > 0.0) ? best->normal : Vec3(-best->normal);
}

}  // namespace wme
