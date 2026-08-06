#pragma once

// 모든 믿음을 로그오즈 공간에서 베이즈 갱신한다.
//
// 왜 세 가지 믿음을 따로 두는가: 이들은 따로 실패한다.
//   existence - "의자가 저기 있다"      (오검출/소멸)
//   identity  - "그게 아까 그 의자다"   (연관 오류)
//   static    - "그 의자는 안 움직인다" (동적 마스킹/랜드마크 자격)
// 하나의 스칼라 confidence 로 뭉치면 "확실히 존재하지만 동일 개체인지 모르는"
// 상태를 표현할 수 없고, 그 상태가 루프 클로저에서 지도를 망가뜨린다.
//
// 핵심 설계: 증거의 세기를 센서 신뢰도로 곱해 스케일한다. 안개 속 관측은
// 방향이 맞더라도 정보량이 작아야 한다. 별도 분기 없이 이 곱 하나로 처리된다.

#include "wme/core/Types.hpp"
#include "wme/perception/EnvironmentState.hpp"
#include "wme/token/WorldToken.hpp"

namespace wme {

struct ConfidenceConfig {
    // 검출 센서 모델
    double p_detect_visible{0.90};    // 실재하고 보일 때 검출될 확률
    double p_false_alarm{0.03};       // 실재하지 않는데 검출될 확률

    // 로그오즈 포화 한계. 무한히 확신하면 반증을 받아들이지 못한다.
    // "지도를 재시작하지 않고 수리한다"는 원칙이 여기에 걸려 있다.
    double logodds_min{-4.0};
    double logodds_max{ 4.0};

    // 연속 프레임의 검출은 독립 관측이 아니다. 같은 오검출을 100번 봐도
    // 증거 100개가 아니다. 이 이득으로 프레임 간 상관을 반영해 감쇠하지 않으면
    // 두세 프레임 만에 포화해 증거의 세기를 전혀 구분하지 못한다.
    double evidence_gain{0.25};

    // 자율 개체(사람/차량)가 "정적" 이라고 주장할 때만 증거를 약화시킨다.
    // 동적 증거는 감쇠하지 않는다 - 움직이는 것을 놓치는 대가가 더 크다.
    // 금지가 아니라 회의로 다뤄야 한다. 상한을 씌우면 채널이 통째로 죽는다.
    double agent_static_evidence_gain{0.5};

    // 미관측 페널티는 "보일 것으로 예측됐는데 안 보였다"에만 적용한다.
    // 시야 밖이나 가려짐은 부재의 증거가 아니다.
    double min_visibility_for_penalty{0.35};

    // 정체성
    double identity_margin_scale{1.5};   // 1등/2등 비용 격차 정규화 기준
    double identity_decay{0.02};         // 미관측 시 정체성 확신 감쇠

    // --- 동적/정적 판정 ---------------------------------------------------
    // 판정 창. 운동은 v*T 로 자라고 관측 잡음은 T 와 무관하므로, 분리도는
    // 오로지 이 값이 정한다. 0.05 s(= 30 Hz 두 프레임)에서 실측 분리비는
    // 1.20(sitting) / 1.44(walking) 로 사실상 구분이 없었고, 0.5 s 에서
    // 2.26 / 3.90 이 된다 (docs/06-results.md 16 장의 창길이 곡선).
    double static_min_dt{0.50};

    // 이 창에서 "정지한 물체" 가 실제로 보이는 변위의 표준편차 하한 (m).
    // 카메라 포즈 오차가 공통모드로 섞여 들어오므로 순수 관측 지터보다 크다.
    // 실측(고정된 tv/keyboard/chair, T=0.5 s): 변위 중앙값 26 mm(sitting) /
    // 51 mm(walking) -> sigma 17 / 33 mm. 두 시퀀스를 덮는 값으로 잡는다.
    double motion_noise_floor{0.035};

    // 동적 가설이 예측하는 속도 (m/s). 사람의 느린 보행. 이보다 느린 것을
    // 놓치는 대가보다 앉아 있는 사람을 지우는 대가가 크다는 것이 14.1 의 실측이다.
    double dynamic_speed_ref{0.7};

    // 정적 가설의 꼬리 무게. 검출 박스 중심의 3D 위치는 이상치를 낸다 -
    // 깊이 중앙값이 몸통과 배경 사이를 오가고, 박스가 팔이나 의자를 물었다 놓는다.
    // 이 항이 없으면 우도비가 아래로 무계라서, 이상치 한 번(실측 -453 nat)이
    // 10 초치 증거를 한 번에 지우고 믿음을 포화 하한에 박아 놓는다.
    // 실측 근거: *고정된* 물체(tv/keyboard/chair)의 판정 창조차 모델의 3 sigma 를
    // 16 %(sitting) / 37 %(walking) / 29 %(fr1_xyz) 넘는다. 이 물체들의 실제
    // 운동은 0 이므로 그 비율은 전부 관측 과정의 이상치다.
    double static_outlier_rate{0.2};

    // 자율 개체의 "정적" 주장 반감기 (s). 신호등 앞의 차 문제는 금지가 아니라
    // 유효기간으로 다룬다 - 관측이 끊기면 주장은 사전분포로 돌아간다.
    double agent_static_halflife{4.0};
};

class ConfidenceEngine {
public:
    explicit ConfidenceEngine(ConfidenceConfig config = {});

    // 관측이 연관되었을 때. assoc_margin 은 (차선 비용 - 최선 비용), 없으면 음수.
    void onObserved(WorldToken& token, const Observation& obs,
                    const EnvironmentState& env, double assoc_margin = -1.0) const;

    // 보일 것으로 예측됐으나 검출되지 않았을 때.
    // expected_visibility 는 Prediction Engine 의 가시성 예측값 (0..1).
    void onMissed(WorldToken& token, double expected_visibility,
                  const EnvironmentState& env) const;

    // 시야를 벗어났을 때. 존재 믿음은 건드리지 않고 정체성만 서서히 감쇠한다.
    void onOutOfView(WorldToken& token, double dt) const;

    // 관측된 변위로 정적/동적 믿음을 갱신한다.
    // displacement 는 창 양 끝의 *원시 관측* 차이여야 한다. 융합 위치의 차이를
    // 넣으면 필터가 이미 매끄럽게 만든 값을 재게 되고, 그 값의 분포는 아래에서
    // 쓰는 잡음 모델(독립 관측 두 번의 차 -> sqrt(2)*sigma)과 다른 통계다.
    void updateStaticBelief(WorldToken& token, const Vec3& displacement, double dt,
                            const EnvironmentState& env) const;

    // 관측이 끊긴 동안 자율 개체의 정적 주장을 사전분포로 되돌린다.
    void decayStaticBelief(WorldToken& token, double dt) const;

    // 두 토큰을 병합할 때 믿음을 결합한다 (루프 클로저 후 중복 제거).
    // 독립 관측 가정 하에 로그오즈를 더하고 포화 한계로 자른다.
    static void mergeBeliefs(WorldToken& keep, const WorldToken& absorbed,
                             const ConfidenceConfig& cfg);

    [[nodiscard]] const ConfidenceConfig& config() const { return cfg_; }

    // 확률 <-> 로그오즈
    [[nodiscard]] static double toLogOdds(double p);
    [[nodiscard]] static double toProbability(double l);

private:
    // 증거를 신뢰도로 스케일해 적용하고 포화 한계로 자른다
    void applyEvidence(double& belief, double log_ratio, double reliability) const;

    ConfidenceConfig cfg_;
};

}  // namespace wme
