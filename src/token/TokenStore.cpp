#include "wme/token/TokenStore.hpp"

#include "wme/core/Assignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace wme {
namespace {

inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// 클래스 이름으로 어포던스를 추정한다.
// 학습된 분류기를 추가로 붙이는 것은 제약 위반이므로 COCO 어휘에 대한
// 명시적 표를 쓴다. 새 어휘를 쓰면 이 표를 확장하면 된다.
Affordance affordanceFor(const std::string& name) {
    static const std::unordered_map<std::string, Affordance> kTable = {
        {"person",     Affordance::Agent | Affordance::Obstacle},
        {"car",        Affordance::Agent | Affordance::Obstacle | Affordance::Hazard},
        {"bus",        Affordance::Agent | Affordance::Obstacle | Affordance::Hazard},
        {"truck",      Affordance::Agent | Affordance::Obstacle | Affordance::Hazard},
        {"bicycle",    Affordance::Agent | Affordance::Obstacle},
        {"motorcycle", Affordance::Agent | Affordance::Obstacle},
        {"dog",        Affordance::Agent | Affordance::Obstacle},
        {"cat",        Affordance::Agent | Affordance::Obstacle},

        {"chair",      Affordance::Sittable | Affordance::Obstacle | Affordance::Supportive},
        {"couch",      Affordance::Sittable | Affordance::Obstacle | Affordance::Supportive},
        {"bench",      Affordance::Sittable | Affordance::Obstacle | Affordance::Supportive},
        {"bed",        Affordance::Sittable | Affordance::Obstacle | Affordance::Supportive},

        {"dining table", Affordance::Supportive | Affordance::Obstacle},
        {"desk",         Affordance::Supportive | Affordance::Obstacle},
        {"tv",           Affordance::Obstacle},
        {"laptop",       Affordance::Graspable | Affordance::Openable},
        {"book",         Affordance::Graspable},
        {"cup",          Affordance::Graspable | Affordance::Container},
        {"bottle",       Affordance::Graspable | Affordance::Container},
        {"bowl",         Affordance::Graspable | Affordance::Container},
        {"vase",         Affordance::Graspable | Affordance::Container},
        {"backpack",     Affordance::Graspable | Affordance::Container},
        {"suitcase",     Affordance::Graspable | Affordance::Container},
        {"refrigerator", Affordance::Openable | Affordance::Container | Affordance::Obstacle},
        {"oven",         Affordance::Openable | Affordance::Container | Affordance::Obstacle},
        {"microwave",    Affordance::Openable | Affordance::Container},
        {"door",         Affordance::Passage | Affordance::Openable},
        {"stairs",       Affordance::Traversable | Affordance::Hazard},
    };
    const auto it = kTable.find(name);
    return (it == kTable.end()) ? Affordance::Obstacle : it->second;
}

// 박스 중앙부의 깊이 중앙값. 가장자리는 배경을 물기 쉬워 제외한다.
// vals 는 호출자가 들고 있는 재사용 버퍼다 (검출마다 roi.area() 만큼 새로
// 잡으면 프레임당 수십 KB 가 된다).
bool medianDepthInBox(const cv::Mat& depth, const cv::Rect& roi, double min_d, double max_d,
                      double& out_median, double& out_valid_ratio,
                      std::vector<float>& vals) {
    if (depth.empty() || roi.width <= 0 || roi.height <= 0) return false;

    vals.clear();
    vals.reserve(static_cast<std::size_t>(roi.area()));
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const float* row = depth.ptr<float>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const float d = row[x];
            if (std::isfinite(d) && d > min_d && d < max_d) vals.push_back(d);
        }
    }
    out_valid_ratio = static_cast<double>(vals.size()) / std::max(1.0, static_cast<double>(roi.area()));
    if (vals.size() < 8) return false;

    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    out_median = vals[mid];
    return true;
}

cv::Rect clampRect(const cv::Rect2f& box, int w, int h, double shrink) {
    const float cx = box.x + box.width * 0.5f;
    const float cy = box.y + box.height * 0.5f;
    const float hw = box.width  * 0.5f * static_cast<float>(shrink);
    const float hh = box.height * 0.5f * static_cast<float>(shrink);

    int x0 = std::max(0, static_cast<int>(std::floor(cx - hw)));
    int y0 = std::max(0, static_cast<int>(std::floor(cy - hh)));
    int x1 = std::min(w, static_cast<int>(std::ceil(cx + hw)));
    int y1 = std::min(h, static_cast<int>(std::ceil(cy + hh)));
    if (x1 <= x0 || y1 <= y0) return {};
    return {x0, y0, x1 - x0, y1 - y0};
}

}  // namespace

TokenStore::TokenStore(TokenStoreConfig config, ConfidenceConfig confidence)
    : cfg_(config), confidence_(confidence) {}

void TokenStore::clear() {
    tokens_.clear();
    next_id_    = 1;
    last_stamp_ = Timestamp{};
}

WorldTokenPtr TokenStore::find(TokenId id) const {
    const auto it = tokens_.find(id);
    return (it == tokens_.end()) ? nullptr : it->second;
}

void TokenStore::collectSorted(std::vector<WorldTokenPtr>& out) const {
    out.clear();
    out.reserve(tokens_.size());
    for (const auto& [id, t] : tokens_) out.push_back(t);
    // 해시 순회 순서는 비결정적이므로 ID 로 정렬해 재현성을 확보한다
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a->id < b->id; });
}

std::vector<WorldTokenPtr> TokenStore::allTokens() const {
    std::vector<WorldTokenPtr> out;
    collectSorted(out);
    return out;
}

std::vector<WorldTokenPtr> TokenStore::activeTokens() const {
    std::vector<WorldTokenPtr> out;
    for (auto& t : allTokens()) {
        if (t->lifecycle == TokenLifecycle::Active || t->lifecycle == TokenLifecycle::Occluded) {
            out.push_back(t);
        }
    }
    return out;
}

std::vector<WorldTokenPtr> TokenStore::stableLandmarks() const {
    std::vector<WorldTokenPtr> out;
    for (auto& t : allTokens()) {
        if (t->isStableLandmark()) out.push_back(t);
    }
    return out;
}

// ---------------------------------------------------------------------------

TokenStore::Measurement TokenStore::measure(const Detection& det, const Frame& frame) const {
    Measurement m;
    const CameraIntrinsics& K = frame.intrinsics;

    const double cx_px = det.box.x + det.box.width * 0.5;
    const double cy_px = det.box.y + det.box.height * 0.5;

    double depth = 0.0, valid_ratio = 0.0;
    const cv::Rect roi = clampRect(det.box, K.width, K.height, cfg_.depth_sample_shrink);

    if (frame.hasDepth() && roi.area() > 0 &&
        medianDepthInBox(frame.depth, roi, 0.1, 60.0, depth, valid_ratio, depth_scratch_)) {
        m.has_depth = true;
    } else {
        // 깊이가 없으면 베어링만 안다. 위치를 지어내지 않고 불확실성을 크게 준다.
        depth = 3.0;
        m.has_depth = false;
    }

    // 공분산: 깊이 방향은 z^2 에 비례, 횡방향은 픽셀 잡음이 깊이로 확대된다.
    if (m.has_depth) {
        const double sz = cfg_.depth_noise_coeff * depth * depth;
        const double sx = cfg_.bearing_noise_px * depth / K.fx;
        const double sy = cfg_.bearing_noise_px * depth / K.fy;
        m.cov_cam = Vec3(sx * sx, sy * sy, sz * sz).asDiagonal();

        // 박스 크기와 깊이로 물리 치수를 추정한다 (축정렬 반치수)
        m.extent = Vec3(0.5 * det.box.width  * depth / K.fx,
                        0.5 * det.box.height * depth / K.fy,
                        0.25 * (det.box.width * depth / K.fx + det.box.height * depth / K.fy));

        // 깊이 센서는 *보이는 표면* 을 잰다. 그 값을 무게중심 깊이로 쓰면
        // 객체가 통째로 카메라 쪽으로 당겨지고, 그 편향은 관측을 아무리 모아도
        // 사라지지 않아 융합할수록 과신하게 된다.
        // extent 는 반치수이므로(위 주석) 볼록 물체의 무게중심은 표면 + 반치수,
        // 즉 계수 1.0 이 기하학적 기본값이다.
        depth += cfg_.depth_centroid_offset * m.extent.z();

        // extent 자체가 보정 전 깊이로 계산돼 있어 약간 작게 잡힌다.
        // 4 m / 0.6 m 물체에서 잔차 0.02 m 수준으로, 깊이 잡음 모델 안에 묻힌다.
    } else {
        const double s = cfg_.no_depth_sigma;
        m.cov_cam = Vec3(s * s * 0.25, s * s * 0.25, s * s).asDiagonal();
        m.extent  = Vec3(0.2, 0.2, 0.2);
    }

    m.position_cam = K.backproject(Vec2(cx_px, cy_px), depth);

    // 박스가 화면 경계에 붙어 있으면 잘린 관측이다. 치수를 믿으면 안 된다.
    const double margin = 3.0;
    const bool touches = det.box.x <= margin || det.box.y <= margin ||
                         det.box.x + det.box.width  >= K.width  - margin ||
                         det.box.y + det.box.height >= K.height - margin;
    m.visible_ratio = touches ? 0.5 : clamp01(0.5 + 0.5 * valid_ratio);
    return m;
}

bool TokenStore::projectToken(const WorldToken& token, const Frame& frame,
                              const SE3& T_cam_world, cv::Rect2f& out_box,
                              double& out_depth) const {
    const CameraIntrinsics& K = frame.intrinsics;
    const Vec3 p_cam = T_cam_world * token.position;
    if (p_cam.z() < 0.1) return false;

    out_depth = p_cam.z();

    // 3D AABB 8 꼭짓점을 투영해 2D 박스를 만든다. 중심만 투영하면
    // 큰 물체가 화면 가장자리에서 사라진 것으로 잘못 판정된다.
    const Vec3 e = token.extent.cwiseMax(Vec3(0.05, 0.05, 0.05));
    double min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;
    bool any = false;

    for (int i = 0; i < 8; ++i) {
        const Vec3 corner(((i & 1) ? e.x() : -e.x()),
                          ((i & 2) ? e.y() : -e.y()),
                          ((i & 4) ? e.z() : -e.z()));
        const Vec3 q = T_cam_world * (token.position + corner);
        if (q.z() < 0.05) continue;
        const Vec2 px = K.project(q);
        min_x = std::min(min_x, px.x()); max_x = std::max(max_x, px.x());
        min_y = std::min(min_y, px.y()); max_y = std::max(max_y, px.y());
        any = true;
    }
    if (!any) return false;

    out_box = cv::Rect2f(static_cast<float>(min_x), static_cast<float>(min_y),
                         static_cast<float>(max_x - min_x), static_cast<float>(max_y - min_y));

    // 화면과 전혀 겹치지 않으면 시야 밖
    const cv::Rect2f screen(0.f, 0.f, static_cast<float>(K.width), static_cast<float>(K.height));
    return boxIoU(out_box, screen) > 0.0f ||
           (out_box.x < screen.width && out_box.y < screen.height &&
            out_box.x + out_box.width > 0.f && out_box.y + out_box.height > 0.f);
}

void TokenStore::associate(const DetectionSet& detections, const std::vector<Measurement>& meas,
                           const std::vector<WorldTokenPtr>& candidates, const Frame& frame,
                           const SE3& T_cam_world, const SE3& T_world_cam,
                           std::vector<int>& det_to_token, std::vector<double>& margins,
                           std::vector<int>& near_token, IntegrationReport& report) const {
    const std::size_t nd = detections.items.size();
    const std::size_t nt = candidates.size();

    det_to_token.assign(nd, -1);
    margins.assign(nd, -1.0);
    near_token.assign(nd, -1);
    if (nd == 0) return;
    if (nt == 0) { report.det_no_candidate += nd; return; }

    std::vector<double>& cost = cost_scratch_;
    cost.assign(nd * nt, kInfeasible);

    // 검출별로 "거리 게이트는 통과했지만 카이제곱 게이트에서 떨어진 최근접 후보"
    // 를 기억한다. 이것이 있으면 게이트가 좁아서 토큰이 죽는 것이고,
    // 없으면 애초에 후보가 없었던 것이다. 둘은 고칠 곳이 다르다.
    std::vector<double>& near_dist = near_dist_scratch_;
    std::vector<double>& near_maha = near_maha_scratch_;
    std::vector<char>&   feasible  = feasible_scratch_;
    near_dist.assign(nd, kInfeasible);
    near_maha.assign(nd, kInfeasible);
    feasible.assign(nd, 0);

    // 진단용: 최근접 후보와의 혁신 벡터와 그때의 dt
    std::vector<Vec3>   near_diff(cfg_.collect_assoc_probes ? nd : 0, Vec3::Zero());
    std::vector<double> near_dt(cfg_.collect_assoc_probes ? nd : 0, 0.0);

    const Mat3 R_wc = T_world_cam.rotation().matrix();

    for (std::size_t d = 0; d < nd; ++d) {
        const Detection& det = detections.items[d];
        const Vec3 z_world = T_world_cam * meas[d].position_cam;

        // 관측 공분산은 카메라 좌표계 값이다 (깊이축이 횡축보다 훨씬 크다).
        // 게이트가 재는 혁신은 월드 좌표이므로 여기서 회전시켜야 한다.
        // 안 시키면 큰 깊이 불확실성이 엉뚱한 월드 축에 붙는다 - 카메라가
        // 거의 안 도는 xyz 에서는 티가 안 나고, 회전이 큰 시퀀스에서 게이트가
        // 실제 운동을 이상치로 거부한다. fusePosition 은 같은 자리에서 이미
        // R*cov*R^T 를 쓴다 (10.2 의 양 불일치와 같은 계열).
        const Mat3 R_world = R_wc * meas[d].cov_cam * R_wc.transpose();

        for (std::size_t t = 0; t < nt; ++t) {
            const WorldToken& tok = *candidates[t];

            if (!cfg_.allow_cross_class && tok.class_id != det.class_id) continue;

            // 예측 위치를 기준으로 비교한다. 등속 물체를 마지막 관측 위치와
            // 비교하면 빠르게 움직이는 대상에서 정체성이 끊긴다.
            const double dt = (tok.last_seen.valid() && detections.stamp.valid())
                                  ? std::max(0.0, detections.stamp - tok.last_seen)
                                  : 0.0;
            const Vec3 predicted = tok.position + tok.velocity * dt;

            const Vec3 diff = z_world - predicted;
            const double dist = diff.norm();
            if (dist > cfg_.max_association_distance) continue;

            // 마할라노비스 게이트: 불확실한 토큰은 더 넓게 받아들인다.
            // 여기에 기동 여유를 더한다. 등속 예측이 틀릴 수 있는 폭은
            // 융합 공분산 안에 없다 - 없으면 게이트가 실제 운동을 거부한다.
            const double slack = cfg_.assoc_maneuver_speed * std::max(dt, 1e-2);

            Mat3 S = tok.position_cov + R_world;
            S.diagonal().array() += slack * slack + 1e-6;
            const double maha = diff.dot(S.inverse() * diff);
            if (dist < near_dist[d]) {
                near_dist[d]  = dist;
                near_maha[d]  = maha;
                near_token[d] = static_cast<int>(t);
                if (cfg_.collect_assoc_probes) { near_diff[d] = diff; near_dt[d] = dt; }
            }
            if (maha > cfg_.max_association_mahalanobis) continue;
            feasible[d] = 1;

            double c = cfg_.distance_weight * maha;

            // 이미지 겹침은 깊이가 나쁠 때 특히 유효한 보조 근거다
            cv::Rect2f proj;
            double proj_depth = 0.0;
            if (projectToken(tok, frame, T_cam_world, proj, proj_depth)) {
                const float iou = boxIoU(proj, det.box);
                c += cfg_.iou_weight * (1.0 - static_cast<double>(iou));
            } else {
                c += cfg_.iou_weight;   // 투영 실패는 약한 불이익
            }
            cost[d * nt + t] = c;
        }
    }

    const AssignmentResult assign = solveAssignment(cost, nd, nt);
    det_to_token = assign.row_to_col;

    // 정체성 확신을 위해 1등/2등 격차를 계산한다.
    // 격차가 작다는 것은 "같은 의자 두 개 중 어느 쪽인지 모른다"는 뜻이다.
    for (std::size_t d = 0; d < nd; ++d) {
        if (det_to_token[d] < 0) {
            if (near_dist[d] >= kInfeasible) {
                ++report.det_no_candidate;
            } else if (!feasible[d]) {
                ++report.det_gated_out;
                report.gated_maha_sum += near_maha[d];
                report.gated_dist_sum += near_dist[d];
            } else {
                ++report.det_unassigned;
            }
            continue;
        }
        const double best = cost[d * nt + static_cast<std::size_t>(det_to_token[d])];
        double second = kInfeasible;
        for (std::size_t t = 0; t < nt; ++t) {
            if (static_cast<int>(t) == det_to_token[d]) continue;
            second = std::min(second, cost[d * nt + t]);
        }
        margins[d] = (second >= kInfeasible) ? 10.0 : std::max(0.0, second - best);
    }

    if (!cfg_.collect_assoc_probes) return;
    const Mat3 R_cam_world = T_cam_world.rotation().matrix();
    for (std::size_t d = 0; d < nd; ++d) {
        if (near_dist[d] >= kInfeasible) continue;   // 후보 자체가 없었다
        AssocProbe p;
        p.class_id   = detections.items[d].class_id;
        p.agent      = has(affordanceFor(detections.items[d].class_name), Affordance::Agent);
        p.matched    = det_to_token[d] >= 0;
        p.gated      = !feasible[d];
        p.dt         = near_dt[d];
        p.dist       = near_dist[d];
        p.maha       = near_maha[d];
        p.diff_world = near_diff[d];
        p.diff_cam   = R_cam_world * near_diff[d];
        p.range      = meas[d].position_cam.z();
        p.meas_sigma = std::sqrt(std::max(0.0, meas[d].cov_cam.trace() / 3.0));
        if (det_to_token[d] >= 0) {
            p.tok_sigma = candidates[static_cast<std::size_t>(det_to_token[d])]->positionSigma();
        }
        report.probes.push_back(p);
    }
}

void TokenStore::fusePosition(WorldToken& token, const Vec3& z_world, const Mat3& R_world,
                              double dt) const {
    // 정보필터 융합. 관측 불확실성이 큰 쪽이 자연스럽게 덜 반영된다.
    Mat3 P = token.position_cov;

    // 예측 단계: 속도 불확실성이 시간에 비례해 위치 불확실성을 키운다
    if (dt > 0.0) {
        P += token.velocity_cov * dt * dt;
    }

    const Mat3 P_inv = P.inverse();
    const Mat3 R_inv = R_world.inverse();
    const Mat3 new_cov = (P_inv + R_inv).inverse();
    const Vec3 predicted = token.position + token.velocity * dt;
    const Vec3 new_pos = new_cov * (P_inv * predicted + R_inv * z_world);

    // 속도는 융합 후 위치 변화로부터 갱신한다 (관측 잡음 증폭을 막기 위해 EMA)
    if (dt > 1e-3) {
        const Vec3 v_obs = (new_pos - token.position) / dt;
        const double alpha = 0.35;
        const Vec3 v_prev = token.velocity;
        token.velocity = (1.0 - alpha) * v_prev + alpha * v_obs;
        token.acceleration = (token.velocity - v_prev) / dt;
        token.velocity_cov = (new_cov + token.position_cov) / (dt * dt);
    }

    token.position     = new_pos;
    token.position_cov = new_cov;
}

void TokenStore::updateLifecycle(WorldToken& token, Timestamp now,
                                 const EnvironmentState& env) const {
    const double silence = token.silenceSeconds(now);

    // 환경이 나쁘면 더 오래 기다린다. 안개 속에서 놓친 것을 사라졌다고
    // 결론내리면 지도가 계속 무너진다.
    const double occl_timeout = cfg_.occluded_timeout_s * env.track_persistence_scale;
    const double dorm_timeout = cfg_.dormant_timeout_s * env.memory_retention_scale;

    // 존재 믿음이 무너지면 "있던 자리에서 사라짐"으로 확정한다.
    // 이것만이 유일하게 부재를 주장하는 전이다.
    if (token.existence_belief < cfg_.existence_retire_threshold) {
        token.lifecycle = TokenLifecycle::Retired;
        return;
    }
    if (token.existence_belief < cfg_.existence_displace_threshold &&
        token.lifecycle != TokenLifecycle::Provisional) {
        token.lifecycle = TokenLifecycle::Displaced;
        return;
    }

    if (token.miss_count == 0) {
        token.lifecycle = (token.observation_count >= cfg_.observations_to_activate)
                              ? TokenLifecycle::Active
                              : TokenLifecycle::Provisional;
        return;
    }

    // 관측이 적은 임시 토큰은 빨리 정리한다. 오검출이 세계를 오염시키지 않게.
    if (token.lifecycle == TokenLifecycle::Provisional && silence > occl_timeout) {
        token.lifecycle = TokenLifecycle::Retired;
        return;
    }
    if (silence <= occl_timeout)      token.lifecycle = TokenLifecycle::Occluded;
    else if (silence <= dorm_timeout) token.lifecycle = TokenLifecycle::Dormant;
    else                              token.lifecycle = TokenLifecycle::Retired;
}

Result<IntegrationReport> TokenStore::integrate(const DetectionSet& detections,
                                                const Frame& frame, const SE3& T_world_cam,
                                                const EnvironmentState& env) {
    if (!frame.intrinsics.valid()) {
        return {ErrorCode::InvalidArgument, "카메라 내부 파라미터 무효"};
    }
    const Timestamp now = detections.stamp.valid() ? detections.stamp : frame.stamp;
    if (!now.valid()) return {ErrorCode::InvalidArgument, "시각 무효"};

    IntegrationReport report;
    const SE3 T_cam_world = T_world_cam.inverse();

    // 1) 검출별 3D 관측 생성 (아래 임시 배열은 전부 프레임 간 재사용 버퍼다)
    std::vector<Measurement>& meas = meas_scratch_;
    meas.clear();
    meas.reserve(detections.items.size());
    for (const auto& det : detections.items) meas.push_back(measure(det, frame));

    // 2) 연관 후보: 은퇴하지 않은 모든 토큰. 시야 밖 토큰도 후보로 둔다
    //    (카메라가 되돌아왔을 때 새 ID 를 만들지 않기 위해).
    collectSorted(sorted_scratch_);
    std::vector<WorldTokenPtr>& candidates = candidate_scratch_;
    candidates.clear();
    for (auto& t : sorted_scratch_) {
        if (t->isAlive()) candidates.push_back(t);
    }

    std::vector<int>&    det_to_token = det_to_token_scratch_;
    std::vector<double>& margins      = margin_scratch_;
    std::vector<int>&    near_token   = near_token_scratch_;
    associate(detections, meas, candidates, frame, T_cam_world, T_world_cam,
              det_to_token, margins, near_token, report);

    // 어떤 후보가 이번 프레임에 검출을 받았는지 먼저 확정한다. 아래 생성
    // 분기에서 "이 새 토큰이 잃어버린 트랙을 대체하는가" 를 물어야 하는데,
    // 루프 안에서 채우면 아직 도달하지 않은 검출의 매칭이 보이지 않는다.
    std::vector<char>& token_matched = matched_scratch_;
    token_matched.assign(candidates.size(), 0);
    for (const int ti : det_to_token) {
        if (ti >= 0) token_matched[static_cast<std::size_t>(ti)] = 1;
    }

    // 3) 매칭된 검출 갱신
    for (std::size_t d = 0; d < detections.items.size(); ++d) {
        const Detection& det = detections.items[d];
        const Vec3 z_world = T_world_cam * meas[d].position_cam;
        const Mat3 R_world = T_world_cam.rotation().matrix() * meas[d].cov_cam *
                             T_world_cam.rotation().matrix().transpose();

        Observation obs;
        obs.stamp              = now;
        obs.frame              = detections.frame;
        obs.box                = det.box;
        obs.detection_conf     = det.confidence;
        obs.position_cam       = meas[d].position_cam;
        obs.position_cov       = meas[d].cov_cam;
        obs.visible_ratio      = meas[d].visible_ratio;
        obs.sensor_reliability = env.sensor_reliability;
        obs.image_quality      = env.sensor_reliability;

        if (det_to_token[d] >= 0) {
            const auto ti = static_cast<std::size_t>(det_to_token[d]);
            WorldToken& tok = *candidates[ti];
            token_matched[ti] = 1;

            const double dt = tok.last_seen.valid() ? std::max(0.0, now - tok.last_seen) : 0.0;

            fusePosition(tok, z_world, R_world, dt);

            tok.meas_world = z_world;
            tok.meas_sigma = std::sqrt(std::max(0.0, R_world.trace() / 3.0));

            tok.box            = det.box;
            tok.mask           = det.mask;
            tok.detection_conf = det.confidence;
            tok.visible_ratio  = meas[d].visible_ratio;
            tok.occlusion      = 1.0 - meas[d].visible_ratio;
            tok.last_seen      = now;
            ++tok.observation_count;
            if (!det.class_scores.empty()) tok.class_distribution = det.class_scores;

            // 치수는 급변하면 안 된다. 잘린 관측이 물체를 작게 만들지 않도록.
            if (meas[d].has_depth && meas[d].visible_ratio > 0.7) {
                tok.extent = 0.7 * tok.extent + 0.3 * meas[d].extent;
            }

            tok.pushObservation(obs, cfg_.history_capacity);
            tok.pushTrajectory(tok.position, cfg_.trajectory_capacity);

            confidence_.onObserved(tok, obs, env, margins[d]);

            // 정적 판정은 누적 창으로 한다. 인접 프레임 변위(= 융합 보정량)는
            // 물체의 운동이 아니라 필터의 동역학을 재는 값이고, 30 Hz 에서는
            // 간격이 최소 판정 시간에도 못 미쳐 아예 버려졌다.
            // 기준점을 붙들었다가 충분한 시간이 지나면 그 사이 이동을 본다.
            if (!tok.static_ref_stamp.valid()) {
                tok.static_ref_position = tok.position;
                tok.static_ref_meas     = z_world;
                tok.static_ref_stamp    = now;
            } else {
                const double window = now - tok.static_ref_stamp;
                if (window >= confidence_.config().static_min_dt) {
                    // 창 양 끝의 원시 관측 차이를 넘긴다. 융합 위치의 차이는
                    // 필터의 동역학이지 물체의 운동이 아니고, 그 분포는
                    // ConfidenceEngine 이 쓰는 잡음 모델과 다른 통계다.
                    confidence_.updateStaticBelief(tok, z_world - tok.static_ref_meas,
                                                   window, env);
                    // 이제 이 토큰의 static_belief 는 관측에 근거한다.
                    tok.static_evidenced = true;
                    // 비교용: 같은 창에서 융합 위치는 얼마나 움직였는가.
                    tok.static_diag.fused_disp =
                        (tok.position - tok.static_ref_position).norm();
                    tok.static_ref_position = tok.position;
                    tok.static_ref_meas     = z_world;
                    tok.static_ref_stamp    = now;
                }
            }
            updateLifecycle(tok, now, env);
            ++report.matched;
        } else {
            // 4) 새 토큰 생성
            auto tok = std::make_shared<WorldToken>();
            tok->id         = allocateId();
            tok->class_id   = det.class_id;
            tok->class_name = det.class_name;
            tok->class_distribution = det.class_scores;
            tok->position     = z_world;
            tok->position_cov = R_world;
            tok->extent       = meas[d].extent;
            tok->box          = det.box;
            tok->mask         = det.mask;
            tok->detection_conf = det.confidence;
            tok->visible_ratio  = meas[d].visible_ratio;
            tok->first_seen   = now;
            tok->last_seen    = now;
            tok->observation_count = 1;
            tok->lifecycle    = TokenLifecycle::Provisional;
            tok->affordance   = affordanceFor(det.class_name);
            tok->semantic_label = det.class_name;

            // 사전분포: 자율 개체 클래스는 처음부터 동적으로 본다
            tok->static_belief = has(tok->affordance, Affordance::Agent) ? 0.15 : 0.6;
            tok->static_prior  = tok->static_belief;

            // 연관이 끊겨 새 토큰이 생겼는데 바로 그 자리에 같은 클래스의
            // *판정된* 토큰이 있으면 운동 상태를 물려받는다. 정체성을 주장하는
            // 것이 아니다 - 운동 상태는 트랙 ID 의 속성이 아니라 그 자리에
            // 있는 것의 속성이고, 연관이 한 번 끊겼다고 해서 걷던 사람이
            // 갑자기 미판정 상태로 돌아가야 할 이유는 없다.
            // 이 항이 없으면 수명이 짧은 트랙(walking_halfsphere: 중앙값 5 관측)
            // 이 영원히 판정 전 상태에 머물러 마스킹에서 통째로 빠진다.
            // 실측: 이 항이 없을 때 fr3_walking_halfsphere 10.55 -> 19.24 cm.
            // 물려받을 판정이 없으면(=아무도 판정된 적 없으면) 아무 일도 일어나지
            // 않는다. 그래서 발산의 초기 구간 - 판정이 하나도 없는 그때 - 에는
            // 여전히 마스킹이 걸리지 않고, 되먹임의 시작점이 막힌다.
            // 물려받는 조건은 "그 자리의 트랙을 대체한다" 이다. 원본 토큰이
            // 이번 프레임에 다른 검출과 매칭됐다면 그것은 아직 살아 있는 다른
            // 개체이고, 이 검출은 대체가 아니라 별개다 - 그때 판정을 복사하는
            // 것은 "옆에 있는 것과 같을 것이다" 라는 훨씬 약한 주장이 된다.
            // 실측: 이 조건 없이 복사하면 fr3_sitting_halfsphere 8.05 -> 20.80 cm.
            const int src_idx = near_token[d];
            if (src_idx >= 0 && !token_matched[static_cast<std::size_t>(src_idx)]) {
                const WorldToken& src = *candidates[static_cast<std::size_t>(src_idx)];
                if (src.static_evidenced) {
                    tok->static_belief    = src.static_belief;
                    tok->static_evidenced = true;
                }
            }
            tok->static_ref_position = z_world;
            tok->static_ref_meas     = z_world;
            tok->static_ref_stamp    = now;
            tok->meas_world = z_world;
            tok->meas_sigma = std::sqrt(std::max(0.0, R_world.trace() / 3.0));
            tok->existence_belief = 0.5;
            tok->identity_belief  = 0.5;

            tok->pushObservation(obs, cfg_.history_capacity);
            tok->pushTrajectory(tok->position, cfg_.trajectory_capacity);

            confidence_.onObserved(*tok, obs, env, -1.0);
            tokens_.emplace(tok->id, tok);
            ++report.created;
        }
    }

    // 5) 매칭되지 않은 토큰 처리
    for (std::size_t t = 0; t < candidates.size(); ++t) {
        if (token_matched[t]) continue;
        WorldToken& tok = *candidates[t];

        // 관측이 끊긴 동안 자율 개체의 "정적" 주장은 사전분포로 되돌아간다.
        // 기준점은 그대로 둔다 - 창이 길어질 뿐이고 dt 도 같이 늘어나므로
        // 판정 자체는 여전히 옳다. 오히려 긴 창이 분리도에 유리하다.
        const double idle = last_stamp_.valid() ? std::max(0.0, now - last_stamp_) : 0.0;
        confidence_.decayStaticBelief(tok, idle);

        cv::Rect2f proj;
        double proj_depth = 0.0;
        const bool in_view = projectToken(tok, frame, T_cam_world, proj, proj_depth);

        if (in_view) {
            // 보일 것으로 예측됐는데 없다 -> 부재의 증거
            const double expected_visibility = clamp01(1.0 - tok.occlusion);
            confidence_.onMissed(tok, expected_visibility, env);
            ++report.missed;
        } else {
            // 시야 밖이다 -> 부재의 증거가 아니다
            const double dt = last_stamp_.valid() ? std::max(0.0, now - last_stamp_) : 0.0;
            confidence_.onOutOfView(tok, dt);
            ++tok.miss_count;
        }
        updateLifecycle(tok, now, env);
    }

    // 6) 은퇴 토큰 제거. Memory Engine 이 이력을 가져간 뒤 호출된다는 전제다.
    for (auto it = tokens_.begin(); it != tokens_.end();) {
        if (it->second->lifecycle == TokenLifecycle::Retired) {
            it = tokens_.erase(it);
            ++report.retired;
        } else {
            ++it;
        }
    }

    // 7) 동적 객체 화면 점유율. EnvironmentAnalyzer 에 되먹임된다.
    //
    // 합산은 ID 순으로 한다. tokens_ 를 그대로 훑으면 덧셈 묶음이 해시 순회
    // 순서에 딸려 가고, 그 순서는 어떤 표준 라이브러리도 보장하지 않는다.
    //
    // 다만 실측한 크기는 정직하게 적어 둔다: 더하는 값은 float 박스 면적
    // (가수 24 비트, 지수 2^7~2^18)이고 항이 수십 개뿐이라 부분합이 double 에
    // 정확히 들어간다. 그래서 지금 값들로는 순서를 뒤집어도 비트가 같다
    // (역순 합산으로 확인). 이 수정이 없앤 것은 관측된 오차가 아니라
    // 값 범위가 넓어지면 그때 드러날 의존성이다.
    double dynamic_area = 0.0;
    const double screen_area = static_cast<double>(frame.intrinsics.width) *
                               static_cast<double>(frame.intrinsics.height);
    collectSorted(sorted_scratch_);
    for (const auto& t : sorted_scratch_) {
        if (t->isDynamic() && t->miss_count == 0) dynamic_area += t->box.area();
    }
    report.dynamic_area_ratio = clamp01(dynamic_area / std::max(1.0, screen_area));

    last_stamp_ = now;
    return report;
}

cv::Mat TokenStore::buildStaticMask(const Frame& frame, const SE3& T_world_cam,
                                    MaskReport* report) const {
    const CameraIntrinsics& K = frame.intrinsics;
    if (!K.valid()) return {};
    // 포즈는 더 이상 필요 없다 - 마스크는 이번 프레임의 검출 박스로만 그린다.
    // 시그니처는 호출부 호환을 위해 유지한다.
    (void)T_world_cam;

    cv::Mat mask(K.height, K.width, CV_8UC1, cv::Scalar(255));

    // 출처별 면적을 따로 세려면 각 토큰이 새로 지운 픽셀만 세야 한다.
    // 전체 마스크에 겹쳐 그리고 나서 합을 내면 중복이 이중 계산된다.
    const double total = std::max(1.0, static_cast<double>(K.width) * K.height);
    double a_obs = 0.0, a_stale = 0.0, a_unjudged = 0.0;
    std::size_t n_masking = 0, n_withheld = 0;

    for (const auto& [id, t] : tokens_) {
        if (!t->isDynamic() || !t->isAlive()) continue;

        // 이번 프레임에 검출과 연관되지 않은 토큰은 지우지 않는다.
        // 연관이 끊긴 토큰의 투영 박스는 등속 모델의 예측이지 관측이 아니고,
        // 그 자리에 실제로 있는 것은 대개 정적 배경이다. 실측: 지운 면적의
        // 70 %(sitting) / 86 %(walking) 가 이번 프레임에 관측되지 않은 토큰에서
        // 나왔다 (docs/06-results.md 16 장).
        if (t->miss_count > 0) continue;

        // 근거 없는 믿음으로는 지우지 않는다. static_evidenced 가 거짓이면 그
        // 토큰의 static_belief 는 클래스 표에서 온 사전분포일 뿐이고, 그것으로
        // 화면을 지우는 것은 14.1 이 기각한 클래스 마스킹 그 자체다 - 한 단계
        // 아래에서 같은 일을 하고 있었다.
        // 실측(fr3_sitting_halfsphere): 지운 면적 64.0 % 중 32.5 pp 가 판정이
        // 한 번도 안 돈 토큰에서 나왔다. 그 결과 ECDA 에 프레임당 130 점만
        // 남아 259 프레임 중 141 번 정렬에 실패했다 (마스킹 없으면 1356 점,
        // 실패 0, ATE 1.50 cm). 마스크가 스스로의 입력인 포즈를 무너뜨리고,
        // 포즈가 무너지니 연관이 끊기고, 연관이 끊기니 모든 토큰이 영원히
        // 판정 전 상태로 남는 되먹임이다.
        // 이 규칙은 그 되먹임의 시작점을 막고 실패 모드를 유계로 만든다:
        // 아무 토큰도 판정을 못 받은 상태에서는 마스킹이 0 이므로 결과가
        // mask OFF 로 퇴화한다. 마스킹이 아무것도 안 하는 것보다 나쁠 수 없다.
        if (!t->static_evidenced) {
            const cv::Rect r = clampRect(t->box, K.width, K.height, cfg_.dynamic_mask_dilate);
            a_unjudged += static_cast<double>(std::max(0, r.area()));
            ++n_withheld;
            continue;
        }

        cv::Mat before;
        const bool account = (report != nullptr);
        if (account) before = mask.clone();

        bool painted = false;
        // 세그멘테이션 마스크가 있으면 그것이 정확하다
        if (!t->mask.empty() && t->mask.size() == mask.size()) {
            mask.setTo(0, t->mask);
            painted = true;
        } else {
            // 이번 프레임 검출 박스를 쓴다. 3D AABB 투영은 융합 위치의 예측이라
            // 빠르게 움직이는 대상에서 뒤처지고, 그러면 사람은 남고 배경이
            // 지워진다 - 정확히 반대로 동작한다. 실측: 투영 박스로 마스킹하면
            // fr3_walking ATE 10.02 cm, 검출 박스로 마스킹하면 그보다 낫다.
            // (여기 도달하는 토큰은 이미 miss_count == 0 이므로 박스는 이번
            //  프레임의 관측이다.)
            const cv::Rect r = clampRect(t->box, K.width, K.height, cfg_.dynamic_mask_dilate);
            if (r.area() > 0) { mask(r).setTo(0); painted = true; }
        }
        if (!painted) continue;
        ++n_masking;
        if (!account) continue;

        const double added = static_cast<double>(cv::countNonZero(before) - cv::countNonZero(mask));
        if (t->miss_count > 0) a_stale += added;
        else                   a_obs   += added;
    }

    if (report) {
        report->masked_ratio = (total - cv::countNonZero(mask)) / total;
        report->from_observed  = a_obs / total;
        report->from_stale     = a_stale / total;
        report->withheld_unjudged = a_unjudged / total;
        report->n_masking      = n_masking;
        report->n_withheld     = n_withheld;
    }
    return mask;
}

Status TokenStore::merge(TokenId keep_id, TokenId absorb_id) {
    if (keep_id == absorb_id) return {ErrorCode::InvalidArgument, "동일 토큰 병합"};

    const auto ka = tokens_.find(keep_id);
    const auto ab = tokens_.find(absorb_id);
    if (ka == tokens_.end() || ab == tokens_.end()) {
        return {ErrorCode::NotAvailable, "병합 대상 토큰 없음"};
    }

    WorldToken& keep = *ka->second;
    const WorldToken& absorbed = *ab->second;

    // 위치는 정보 가중 평균. 더 확신하는 쪽이 이긴다.
    const Mat3 Pk = keep.position_cov, Pa = absorbed.position_cov;
    const Mat3 Pk_inv = Pk.inverse(), Pa_inv = Pa.inverse();
    const Mat3 new_cov = (Pk_inv + Pa_inv).inverse();
    keep.position = new_cov * (Pk_inv * keep.position + Pa_inv * absorbed.position);
    keep.position_cov = new_cov;

    ConfidenceEngine::mergeBeliefs(keep, absorbed, confidence_.config());
    // 병합된 믿음의 근거는 두 토큰 중 하나라도 판정을 받았으면 존재한다.
    keep.static_evidenced = keep.static_evidenced || absorbed.static_evidenced;

    // 이력을 시간순으로 합친다. 루프 클로저의 진짜 값어치가 여기에 있다 -
    // 같은 개체의 두 관측 이력이 하나로 이어진다.
    for (const auto& obs : absorbed.history) keep.history.push_back(obs);
    std::sort(keep.history.begin(), keep.history.end(),
              [](const Observation& a, const Observation& b) { return a.stamp < b.stamp; });
    while (keep.history.size() > cfg_.history_capacity) keep.history.pop_front();

    tokens_.erase(ab);
    return {};
}

}  // namespace wme
