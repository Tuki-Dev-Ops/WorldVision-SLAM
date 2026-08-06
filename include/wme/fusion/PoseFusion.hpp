#pragma once

// 3-tier 상대포즈 융합 - docs/02-correspondence-problem.md 4장의 융합 규칙.
//
//   Lambda_total = a0(E) Lambda_ECDA + a1(E) Lambda_TCG + a2(E) Lambda_SPA
//
// 세 tier 는 팩터 종류가 다르지 않다. 전부 같은 (ref, cur) 쌍의 상대포즈를
// 추정하고 정보행렬 크기만 다르다. 여기에 야간/안개 분기는 없다 - alpha_k(E)
// 스칼라 하나가 정보량을 재배분하고 추정기가 나머지를 알아서 한다.
//
// **포즈를 평균내지 않는다.** 각 tier 는 자기 추정점 T_k 에서 선형화된 정보
// Lambda_k 를 준다. 서로 다른 선형화점에서 나온 정보를 그냥 더하면 틀리므로,
// 잔차 eps_k = log(T T_k^-1) 의 자코비안 J_l^-1(eps_k) 로 옮겨 붙인다.
//
//   min_T  sum_k 1/2 eps_k^T (a_k Lambda_k) eps_k
//
// 이것은 "T_k 를 평균낸다" 와 다르다. 회전이 섞이면 결과가 달라지고, 랭크
// 부족한 tier(SPA)가 자기가 구속하지 않는 축으로 해를 끌지 않는다.
//
// 한계 (측정 전에 적어 둔다):
//   - tier 들이 독립이라고 가정하고 정보를 더한다. ECDA 와 SPA 는 같은
//     깊이맵을 쓰므로 이는 사실이 아니다. docs/06-results.md 10.1 의 상관
//     오류와 같은 종류이며, Lambda_total 은 그만큼 과신이다. 추정치 자체는
//     상대 가중에만 영향을 받으므로 이 과신은 보고값의 문제다.
//   - alpha_k(E) 는 손으로 설계된 값이다 (docs/04-unified-objective.md 1장
//     표의 row 3). 보정된 우도가 아니다.
//
// 스레드 안전성: 상태 없음. 순수 함수.

#include "wme/core/Result.hpp"
#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace wme::fusion {

enum class Tier : int {
    Photometric   = 0,   // Tier 0 ECDA
    Constellation = 1,   // Tier 1 TCG
    Structural    = 2,   // Tier 2 SPA
};
inline constexpr int kTierCount = 3;

[[nodiscard]] std::string_view toString(Tier t) noexcept;

// tier 가 기여하지 못한 이유. "조용히 0 가중치로 빠지는 것" 을 막기 위해
// 반드시 하나를 남긴다 - 기권이 보이지 않으면 융합 결과를 읽을 수 없다.
enum class Abstain : std::uint8_t {
    None = 0,
    Disabled,        // ablation 스위치로 껐다
    NoInput,         // 입력이 없다 (평면 0개 / 객체 부족)
    Failed,          // 추정 시도했으나 실패
    ZeroInformation, // 정보행렬이 0 또는 비유한
    ZeroWeight,      // alpha_k(E) 가 0 이 되었다
};

[[nodiscard]] std::string_view toString(Abstain a) noexcept;

// 한 tier 가 같은 (ref, cur) 쌍에 대해 낸 추정.
struct TierEstimate {
    Tier    tier{Tier::Photometric};
    SE3     T_cur_ref;                     // ref 좌표계 -> cur 좌표계
    Mat6    information{Mat6::Zero()};     // xi = [rho, phi], 좌측섭동, T_cur_ref 에서 선형화
    double  alpha{1.0};                    // alpha_k(E)
    double  calibration{1.0};              // 상대 보정 배율 kappa_k (기본 1 = 원본 그대로)
    bool    available{false};              // false = 기권
    Abstain reason{Abstain::NoInput};
};

struct FusionConfig {
    int    max_iterations{10};
    double convergence_delta{1e-10};   // 갱신량 노름

    // 랭크 **보고** 문턱. 최대 고유값 대비 이 비율 미만이면 관측 불가로 본다.
    // SPA/DirectAligner 와 같은 값을 쓴다 - 세 곳이 다른 문턱으로 "관측 불가" 를
    // 말하면 상보성 판정이 문턱 차이를 재게 된다.
    double degeneracy_ratio{1e-3};

    // 해를 구할 때의 절단 문턱. 위와 **달라야 한다**.
    //
    // 융합에서 이 둘을 같은 값으로 두면 아키텍처의 주장 자체가 지워진다.
    // tier 사이 정보 스케일 차이가 실측으로 10^6 배 이상이므로, ECDA 가 못 보는
    // 축을 SPA 가 채운 성분은 전체 최대 고유값 대비 10^-7 쯤이 된다. 1e-3 으로
    // 자르면 그 성분이 통째로 버려지고 - 즉 "구멍을 메웠다" 는 주장이 코드에서
    // 조용히 제거된 채로 그럴듯한 숫자가 나온다 (docs/06-results.md 10.4).
    //
    // 약하게 구속된 방향은 퇴화가 아니라 그냥 약한 것이고, 그 축의 해는 그 축을
    // 아는 유일한 tier 가 정하는 것이 맞다. 여기서 절단은 0 나눗셈을 막는
    // 수치 안전장치일 뿐이므로 배정밀도 한계 근처로 둔다.
    double update_observable_ratio{1e-12};
};

// tier 하나가 최종 해에 실제로 무엇을 했는가.
struct TierContribution {
    bool    used{false};
    Abstain reason{Abstain::NoInput};
    double  alpha{0.0};
    double  calibration{1.0};
    double  info_trace{0.0};        // tr(alpha*kappa*Lambda_k). 기여의 절대 크기
    double  info_share{0.0};        // 전체 trace 대비 비중 0..1
    Vec6    residual{Vec6::Zero()}; // log(T_fused * T_k^-1). 이 tier 가 얼마나 끌려갔나
    double  self_nees{0.0};         // residual^T Lambda_k residual
};

struct FusionResult {
    SE3  T_cur_ref;
    Mat6 information{Mat6::Zero()};        // 융합점으로 옮긴 총 정보 (독립 가정)
    Mat6 information_naive{Mat6::Zero()};  // 문서 식 그대로의 단순합. 둘의 차이가 선형화점 차이다
    Vec6 eigenvalues{Vec6::Zero()};        // information 고유값 (내림차순)
    Vec6 weakest_direction{Vec6::Zero()};
    int  observable_dof{0};
    int  iterations{0};
    int  contributing_tiers{0};

    std::array<TierContribution, kTierCount> tiers{};

    [[nodiscard]] bool fullRank() const { return observable_dof == 6; }
    [[nodiscard]] bool contributed(Tier t) const {
        return tiers[static_cast<std::size_t>(t)].used;
    }
};

// 같은 (ref, cur) 쌍의 tier 추정들을 정보로 융합한다.
// 기여 가능한 tier 가 하나도 없으면 NotAvailable.
// 총 정보가 랭크 부족이면 성공하되 degraded 로 표시한다.
[[nodiscard]] Result<FusionResult> fuse(const std::vector<TierEstimate>& estimates,
                                        const FusionConfig& cfg = {});

// --- 선형화점 이동 ---------------------------------------------------------
// SE(3) 좌측 자코비안. 정의식 J_l = sum_n ad(xi)^n / (n+1)! 을 그대로 더한다.
// 닫힌 형태(Barfoot Q 행렬)를 쓰지 않는 이유는 부호를 틀리기 쉬워서다 -
// docs/06-results.md 7.1 이 정확히 그 사고의 기록이다. |phi| <= pi 에서
// 급수는 25항이면 기계정밀도로 수렴하며, 유한차분으로 검증된다.
[[nodiscard]] Mat6 se3LeftJacobian(const Vec6& xi);
[[nodiscard]] Mat6 se3LeftJacobianInverse(const Vec6& xi);

// 정보행렬을 T_from 에서 T_to 의 접선으로 옮긴다.
// eps = log(T_to * T_from^-1) 일 때 Lambda' = J^-T Lambda J^-1, J = J_l(eps).
[[nodiscard]] Mat6 transportInformation(const Mat6& information, const SE3& T_from,
                                        const SE3& T_to);

// --- 퇴화 상보성 -----------------------------------------------------------
// "B 가 A 가 못 보는 축을 채우는가" 를 스케일 불변으로 잰다.
//
// 절대 크기 비교는 무의미하다 - tier 마다 정보 스케일이 10^N 배 다르기 때문에
// 큰 쪽이 항상 이긴다. 물어야 할 것은 방향의 편향이다:
//
//   fill_ratio = (u_weak^T B u_weak / u_weak^T A u_weak)
//              / (u_strong^T B u_strong / u_strong^T A u_strong)
//
// A 와 B 를 각각 상수배해도 값이 변하지 않는다.
//
// 중립 기준은 1 이고, 그 뜻은 "B 가 A 와 같은 모양" 이다 (B = c*A 이면 정확히 1).
// 즉 fill_ratio 는 융합이 조건수를 얼마나 개선하는지를 재는 값이다. 1 보다 크면
// B 가 A 의 약축에 상대적으로 더 많은 정보를 싣는다는 뜻이고, 그것이 상보성이다.
// 등방인 B 는 fill_ratio = cond(A) 를 준다 - 아무 방향성이 없어도 약축을 채우긴
// 하기 때문이며, 그것이 올바른 거동이다.
struct Complementarity {
    Vec6   weak_direction{Vec6::Zero()};    // A 의 최약축 (단위벡터)
    Vec6   strong_direction{Vec6::Zero()};  // A 의 최강축
    double a_weak{0.0}, a_strong{0.0};
    double b_weak{0.0}, b_strong{0.0};
    double fill_ratio{0.0};                 // > 1 이면 상보적
    bool   valid{false};
};

[[nodiscard]] Complementarity complementarity(const Mat6& a, const Mat6& b);

// 접선 방향의 병진 성분 비율. 최약축이 "복도 축 병진" 인지 "회전" 인지를
// 구분해야 채워진 축이 정말 비어 있던 축인지 말할 수 있다.
[[nodiscard]] double translationFraction(const Vec6& direction);

}  // namespace wme::fusion
