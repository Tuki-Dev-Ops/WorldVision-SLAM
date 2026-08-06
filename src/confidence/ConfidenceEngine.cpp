#include "wme/confidence/ConfidenceEngine.hpp"

#include <algorithm>
#include <cmath>

namespace wme {
namespace {

inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

// 확률을 0/1 에서 떼어놓는다. 로그오즈가 발산하면 그 믿음은 영원히 갱신 불가가 된다.
inline double safeProb(double p) { return std::clamp(p, 1e-6, 1.0 - 1e-6); }

}  // namespace

ConfidenceEngine::ConfidenceEngine(ConfidenceConfig config) : cfg_(config) {}

double ConfidenceEngine::toLogOdds(double p) {
    const double s = safeProb(p);
    return std::log(s / (1.0 - s));
}

double ConfidenceEngine::toProbability(double l) {
    return 1.0 / (1.0 + std::exp(-l));
}

void ConfidenceEngine::applyEvidence(double& belief, double log_ratio, double reliability) const {
    // 신뢰도가 낮으면 증거의 세기 자체가 줄어든다.
    // 방향을 바꾸는 게 아니라 정보량을 줄이는 것이 요점이다.
    const double scaled = log_ratio * cfg_.evidence_gain * clamp01(reliability);
    const double l = std::clamp(toLogOdds(belief) + scaled, cfg_.logodds_min, cfg_.logodds_max);
    belief = toProbability(l);
}

void ConfidenceEngine::onObserved(WorldToken& token, const Observation& obs,
                                  const EnvironmentState& env, double assoc_margin) const {
    // --- 존재 -------------------------------------------------------------
    // 우도비 = P(검출 | 실재) / P(검출 | 부재).
    // 검출 신뢰도가 낮으면 p_detect 를 낮춰 우도비를 1 쪽으로 민다.
    const double conf   = clamp01(static_cast<double>(obs.detection_conf));
    const double p_det  = safeProb(cfg_.p_detect_visible * conf);
    const double p_fa   = safeProb(cfg_.p_false_alarm);
    const double ratio  = std::log(p_det / p_fa);

    // 환경 신뢰도와 이 관측 자체의 품질을 함께 반영한다
    const double reliability = clamp01(env.sensor_reliability) *
                               clamp01(obs.sensor_reliability) *
                               clamp01(0.3 + 0.7 * obs.image_quality);

    applyEvidence(token.existence_belief, ratio, reliability);

    // --- 정체성 -----------------------------------------------------------
    // 최선/차선 연관 비용의 격차가 클수록 "다른 개체와 헷갈렸을" 확률이 낮다.
    // 격차 정보가 없으면(단일 후보) 약한 긍정 증거만 준다.
    if (assoc_margin >= 0.0) {
        const double margin = assoc_margin / std::max(1e-6, cfg_.identity_margin_scale);
        // margin=0 이면 우도비 0(정보 없음), 크면 양의 증거로 포화
        applyEvidence(token.identity_belief, std::log1p(margin) * 2.0, reliability);
    } else {
        applyEvidence(token.identity_belief, 0.4, reliability);
    }

    token.sensor_reliability = reliability;
    token.miss_count = 0;
}

void ConfidenceEngine::onMissed(WorldToken& token, double expected_visibility,
                                const EnvironmentState& env) const {
    ++token.miss_count;

    // 가려졌거나 시야 밖이면 부재의 증거가 아니다.
    // 이 구분을 안 하면 사람 뒤에 잠깐 가린 의자가 사라진다.
    if (expected_visibility < cfg_.min_visibility_for_penalty) return;

    const double p_det = safeProb(cfg_.p_detect_visible * clamp01(expected_visibility));
    const double p_fa  = safeProb(cfg_.p_false_alarm);

    // 우도비 = P(미검출 | 실재) / P(미검출 | 부재)  < 1 이므로 음의 증거
    const double ratio = std::log((1.0 - p_det) / (1.0 - p_fa));

    // 악조건에서는 미검출이 부재를 뜻하지 않는다. 신뢰도로 그대로 감쇠된다.
    applyEvidence(token.existence_belief, ratio, clamp01(env.sensor_reliability));
}

void ConfidenceEngine::onOutOfView(WorldToken& token, double dt) const {
    // 존재 믿음은 유지한다. 안 보인다는 것은 없다는 뜻이 아니다.
    // 다만 다시 만났을 때 같은 개체라고 주장할 근거는 시간이 지날수록 약해진다.
    const double decay = std::exp(-cfg_.identity_decay * std::max(0.0, dt));
    token.identity_belief = 0.5 + (token.identity_belief - 0.5) * decay;
}

void ConfidenceEngine::updateStaticBelief(WorldToken& token, const Vec3& displacement,
                                          double dt, const EnvironmentState& env) const {
    if (dt < cfg_.static_min_dt) return;

    // 두 가설의 우도비를 그대로 쓴다. 이전 구현은 [-0.31, +0.38] 로 유계인
    // 임의 함수여서, 아무리 결정적인 관측이어도 증거의 세기가 상수에 묶였다.
    // 실측: sitting 과 walking 모두 갱신의 84 % 가 양의 증거였고 - 부호가
    // 전혀 구분되지 않았다 - 믿음은 사실상 토큰이 몇 번 살아남았는지만 셌다.
    //
    //   H_s (정적) : 변위 ~ N(0, sigma_d^2 I),  sigma_d = 독립 관측 두 번의 차
    //   H_d (동적) : 변위 ~ N(0, sigma_m^2 I),  sigma_m^2 = sigma_d^2 + (v*dt)^2
    //
    // 척도는 *관측* 불확실성이다. positionSigma() 는 융합 추정의 불확실성이라
    // 관측 수에 따라 줄어드는 반면 변위는 그렇지 않다 - 예측하는 통계가 다르다
    // (10.2 의 반복된 실수). 깊이 없는 토큰은 meas_sigma 가 커져 판정을
    // 스스로 포기하는데, positionSigma 는 융합으로 줄어들어 그러지 못한다.
    const double sigma_d = std::max(cfg_.motion_noise_floor,
                                    std::sqrt(2.0) * token.meas_sigma);
    const double travel  = cfg_.dynamic_speed_ref * dt;
    const double sigma_m = std::sqrt(sigma_d * sigma_d + travel * travel);

    const double d2 = displacement.squaredNorm();
    // 3 자유도 등방 가우시안의 로그우도비. 변위 0 에서 3*log(sigma_m/sigma_d) 로
    // 포화한다 - 안 움직였다는 것은 상한이 있는 증거다 (동적 물체도 잠시 멈춘다).
    // 아래로는 무계인데, 그 상태로 두면 이상치 하나가 채널을 지운다. 바로 아래
    // 혼합 모델이 그 하한을 정한다.
    const double llr = 3.0 * std::log(sigma_m / sigma_d) +
                       0.5 * d2 * (1.0 / (sigma_m * sigma_m) - 1.0 / (sigma_d * sigma_d));
    const double z = displacement.norm() / sigma_d;

    // 정적 가설에 꼬리를 붙인다: p(d|H_s) = (1-e)N(0,sigma_d^2) + e N(0,sigma_m^2).
    // 우도비는 log(e) 아래로 내려가지 않는다 - 이상치 관측 한 번이 결론을
    // 뒤집을 수는 있어도 지워 버릴 수는 없어야 한다.
    const double eps = std::clamp(cfg_.static_outlier_rate, 0.0, 0.5);
    const double m   = std::max(llr, 0.0);
    const double ratio = (eps <= 0.0)
        ? llr
        : m + std::log((1.0 - eps) * std::exp(llr - m) + eps * std::exp(-m));

    // 자율 개체가 "정적" 이라고 주장하려면 더 많은 증거가 필요하다.
    // 동적이라는 증거는 그대로 받는다 - 비대칭이 맞다. 움직이는 것을 놓치는
    // 대가가 멈춘 것을 못 믿는 대가보다 크다.
    const bool agent = has(token.affordance, Affordance::Agent);
    const double gain = (agent && ratio > 0.0) ? cfg_.agent_static_evidence_gain : 1.0;

    applyEvidence(token.static_belief, ratio * gain, clamp01(env.sensor_reliability));

    ++token.static_diag.updates;
    token.static_diag.disp  = displacement.norm();
    token.static_diag.sigma = sigma_d;
    token.static_diag.z     = z;
    token.static_diag.ratio = ratio;

    // 자율 개체는 "지금 멈춰 있어도 곧 움직일 수 있다". 여기서 상한을 씌우면
    // 안 된다 - 씌우면 아무리 오래 정지한 것을 관측해도 갱신이 무의미해진다.
    // 실측 대가: TUM fr3_sitting 에서 앉아 있는 사람이 통째로 마스킹되어
    // ATE 가 1.01 -> 15.84 cm 로 무너졌다. 그 사람들은 벽만큼 좋은 랜드마크다.
    // 그 요구는 상한이 아니라 유효기간으로 다룬다 - decayStaticBelief 참조.
}

void ConfidenceEngine::decayStaticBelief(WorldToken& token, double dt) const {
    // 자율 개체가 아니면 정적 주장은 시간이 지나도 약해지지 않는다.
    // 책상은 안 보이는 동안 걸어다니지 않는다.
    if (!has(token.affordance, Affordance::Agent)) return;
    if (dt <= 0.0 || cfg_.agent_static_halflife <= 0.0) return;

    // 사전분포 쪽으로 로그오즈를 되돌린다. 관측이 끊긴 사람은 "정적" 이라는
    // 주장의 근거가 시간과 함께 사라진다 - 신호등 앞의 차는 언젠가 출발한다.
    const double l0 = toLogOdds(token.static_prior);
    const double k  = std::pow(0.5, dt / cfg_.agent_static_halflife);
    token.static_belief = toProbability(l0 + (toLogOdds(token.static_belief) - l0) * k);
}

void ConfidenceEngine::mergeBeliefs(WorldToken& keep, const WorldToken& absorbed,
                                    const ConfidenceConfig& cfg) {
    // 독립 관측 가정: 로그오즈를 더한다. 같은 증거를 두 번 세지 않도록
    // 호출자는 실제로 서로 다른 관측 이력을 가진 토큰만 병합해야 한다.
    const auto combine = [&cfg](double a, double b) {
        const double l = std::clamp(toLogOdds(a) + toLogOdds(b), cfg.logodds_min, cfg.logodds_max);
        return toProbability(l);
    };
    keep.existence_belief = combine(keep.existence_belief, absorbed.existence_belief);
    keep.static_belief    = combine(keep.static_belief, absorbed.static_belief);

    // 정체성은 더하지 않는다. 병합 자체가 "동일 개체"라는 주장이므로
    // 두 확신을 곱해 보수적으로 취급한다.
    keep.identity_belief = clamp01(keep.identity_belief * absorbed.identity_belief +
                                   0.5 * std::abs(keep.identity_belief - absorbed.identity_belief));

    keep.observation_count += absorbed.observation_count;
    if (absorbed.first_seen.valid() &&
        (!keep.first_seen.valid() || absorbed.first_seen < keep.first_seen)) {
        keep.first_seen = absorbed.first_seen;
    }
    if (absorbed.last_seen > keep.last_seen) keep.last_seen = absorbed.last_seen;
}

}  // namespace wme
