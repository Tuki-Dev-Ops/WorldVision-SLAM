#pragma once

// tier 별 정보행렬을 만드는 것과 alpha_k(E) 를 고르는 것.
//
// Tier 0(ECDA)와 Tier 2(SPA)는 스스로 6x6 정보행렬을 보고한다. Tier 1(TCG)은
// 그렇지 않다 - ConstellationMatch 는 점수/신뢰도/rms 만 준다. 그래서 여기서
// 만든다. 손으로 정한 대각행렬을 쓰지 않고, Kabsch 대응 자체의 기하에서 뽑는다.
//
//   잔차 r_i = (T p_i^ref) - p_i^cur,   좌측섭동 자코비안 J_i = [I, -skew(q_i)]
//   Lambda_TCG = sum_i (1/sigma_i^2) J_i^T J_i,   q_i = T p_i^ref
//
// 이렇게 해야 랭크가 사실을 말한다. 객체 두 개면 그 축 둘레 회전이 구속되지
// 않고, 한 줄로 늘어선 객체들도 마찬가지다. 대각행렬은 그 사실을 지운다.
//
// 한계: sigma_i 는 노드의 위치 표준편차이고, 그 자체가 보정되지 않은 잡음
// 모델(TokenStore 의 깊이/베어링 모델)이다. 즉 Lambda_TCG 의 절대 스케일은
// 검증되지 않았다. 상대 스케일은 fusion_eval.py 가 ANEES 로 측정한다.

#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"
#include "wme/fusion/PoseFusion.hpp"
#include "wme/perception/EnvironmentState.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace wme::fusion {

// alpha_k 를 어디서 가져올 것인가.
enum class WeightMode {
    Environment,   // EnvironmentState::tier (손으로 설계된 스케줄)
    Uniform,       // 전부 1. 대조군 - 스케줄이 균등가중보다 나은지 재려면 필요하다
};

// alpha_0..alpha_2 를 순서대로 돌려준다.
[[nodiscard]] std::array<double, kTierCount> tierWeights(const EnvironmentState& env,
                                                         WeightMode mode);

// 성좌 대응 하나. 두 좌표계에서 본 같은 객체의 3D 위치.
struct ConstellationPair {
    Vec3   p_ref{Vec3::Zero()};   // ref 카메라 좌표
    Vec3   p_cur{Vec3::Zero()};   // cur 카메라 좌표
    double sigma{0.05};           // 위치 표준편차 (m). 두 관측의 결합
};

// 대응 집합에서 T_cur_ref 의 정보행렬을 만든다.
// T_cur_ref 는 이미 Kabsch 로 구해진 해여야 한다 - 여기서는 그 해에서
// 선형화만 한다. 대응이 2개 미만이면 0 행렬.
[[nodiscard]] Mat6 constellationInformation(const std::vector<ConstellationPair>& pairs,
                                            const SE3& T_cur_ref);

}  // namespace wme::fusion
