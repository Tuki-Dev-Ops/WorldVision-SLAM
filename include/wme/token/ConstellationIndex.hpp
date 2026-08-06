#pragma once

// Token Constellation Geometry (TCG) - 기술자 없는 재지역화 / 루프 클로저.
//
// 아이디어: 장소는 텍스처 패치의 집합이 아니라 "특정 배치로 놓인 객체들"이다.
// 클래스 다중집합 + 쌍거리 스펙트럼 + 카이랄리티 워드로 회전/평행이동/순서에
// 불변인 서명을 만들고, 역색인으로 후보를 뽑은 뒤 클래스 일관 그래프 매칭으로
// 검증한다. 대응이 나오면 Kabsch 로 SE(3)를 닫힌 형태로 얻는다.
//
// 설계 근거와 BoW 대비 비교는 docs/02-correspondence-problem.md 3장 Tier 1 참조.

#include "wme/core/Result.hpp"
#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"
#include "wme/token/WorldToken.hpp"

#include <optional>
#include <unordered_map>
#include <vector>

namespace wme {

// 성좌를 구성하는 최소 정보. 전체 토큰을 복사하지 않는다.
struct ConstellationNode {
    TokenId id{};
    int     class_id{-1};
    Vec3    position{Vec3::Zero()};   // 성좌 기준 좌표계
    double  sigma{0.05};              // 위치 표준편차 (m)
};

// 색인에 등록된 하나의 장소
struct Place {
    std::uint64_t                  place_id{0};
    KeyframeId                     keyframe{};
    Timestamp                      stamp{};
    SE3                            anchor;   // 성좌 좌표계 -> 월드
    std::vector<ConstellationNode> nodes;

    // 이 장소 좌표계에서의 중력 방향. 카이랄리티 판정에 쓴다.
    // 질의 프레임과 좌표계가 다르므로 같은 벡터를 공유하면 안 된다.
    std::optional<Vec3> gravity;
};

struct ConstellationConfig {
    // 거리 구간화: 로그 스케일. 근거리 해상도를 높이고 원거리는 관대하게.
    double min_distance{0.20};      // m. 이보다 가까운 쌍은 무시 (같은 물체 중복)
    double max_distance{25.0};      // m
    int    distance_bins{24};

    std::size_t min_nodes{4};       // 성좌 성립 최소 객체 수
    std::size_t max_nodes{40};      // 상위 N 개만 사용 (계산량 상한)

    std::size_t top_candidates{8};  // 역색인 투표 상위 후보 수
    std::size_t max_pairs{512};     // 대응 후보 상한
    std::size_t clique_budget{200000};  // 최대 클리크 탐색 노드 예산

    // 허용오차 = 고정 base + 거리 비례(스케일 오차) + sigma_gate x 3-sigma(관측 잡음).
    // base 를 크게 잡으면 큰 방에서 오대응 클리크가 정대응과 같은 크기로 자라
    // 정확한 데이터에서도 틀린 변환이 나온다.
    double distance_tolerance{0.12};    // m
    double relative_tolerance{0.03};
    double sigma_gate{3.0};
    std::size_t min_inliers{4};         // Kabsch 최소 대응 수
    double max_rms_error{0.45};         // m. 정합 후 잔차 상한
    double chi2_gate{9.21};             // 자유도 2, 99% 신뢰

    bool use_chirality{true};           // 중력축 기반 거울 모호성 제거

    // --- 모호성 판정 -------------------------------------------------------
    // 점수비(score2 > 0.85*score1)만으로 자르면 조밀하게 샘플링한 지도에서
    // 정대응이 전멸한다. 이웃 키프레임이 비슷한 점수를 받는 것은 지각적 혼동이
    // 아니라 정상이다. 물어야 할 것은 "상위 후보들이 같은 *포즈* 를 가리키는가".
    //
    // 그래서 후보를 월드 포즈로 군집화하고 군집 질량(점수합)을 비교한다.
    // 점수 공간이 아니라 포즈 공간의 마진이다.
    double pose_agree_radius{0.50};      // m. 이 안이면 같은 포즈 군집
    double pose_agree_deg{25.0};         // deg
    double pose_dominance{1.5};          // 1위 군집 질량 / 2위 군집 질량 하한
    std::size_t min_agree{1};            // 군집 최소 크기 (자기 포함)

    // 신뢰도의 카이제곱 항 스케일. 1/(1 + chi2_dof/scale) 로 감쇠시킨다.
    // TUM 5개 시퀀스에서 corr(신뢰도, 포즈오차) 를 최대화하는 값이 10 근방이었다.
    double chi2_confidence_scale{10.0};

    // 채택 신뢰도 하한. TUM 실측 곡선에서 정밀도 92 % / 지도밖 오탐 0 인 지점.
    // 0 으로 두면 모호성 규칙만으로 판정한다 (정밀도 ~69 %).
    double min_confidence{0.55};
};

// 정합 결과
struct ConstellationMatch {
    std::uint64_t place_id{0};
    KeyframeId    keyframe{};
    SE3           transform;                 // query 좌표계 -> place 좌표계
    double        rms_error{0.0};
    double        score{0.0};                // 0..1 자기일관성 점수
    std::vector<std::pair<TokenId, TokenId>> correspondences;  // (query, place)

    // --- 신뢰도 원자료 -----------------------------------------------------
    // score/rms 는 "대응이 서로 일관적인가" 만 잰다. 동일한 모니터 두 대에
    // 어긋나게 붙인 대응도 완벽히 자기일관적이다. 아래 값들은 그 한계를
    // 벗어나기 위한 재료이며, 조합은 confidence 가 맡는다.
    std::size_t n_query_nodes{0};   // 질의 성좌 크기
    std::size_t n_place_nodes{0};
    std::size_t n_inliers{0};       // 카이제곱 통과 대응 수
    double      explained{0.0};     // n_inliers / n_query_nodes
    double      chi2_dof{0.0};      // 노드 공분산 정규화 잔차, 자유도당

    // 아래 값들은 후보 집합 전체를 봐야 정해지므로 queryAll/query 가 채운다.
    // 후보들을 월드 포즈로 군집화하고, 각 군집의 질량(점수합)을 비교한다.
    std::size_t agree_count{1};     // 같은 포즈 군집의 후보 수 (자기 포함)
    double      support{0.0};       // 자기 군집의 질량 = 점수합
    double      rival_mass{0.0};    // 가장 무거운 *다른* 군집의 질량
    double      pose_margin{0.0};   // 그 경쟁 군집까지의 포즈 거리 (m)
    double      confidence{0.0};    // 최종 채택 신뢰도 0..1
};

// 경쟁 후보가 없을 때의 pose_margin 표시값. 무한대를 CSV 로 내보내지 않는다.
inline constexpr double kNoRivalMargin = 999.0;

class ConstellationIndex {
public:
    explicit ConstellationIndex(ConstellationConfig config = {});

    // 장소 등록. 키프레임 생성 시 호출한다.
    // nodes 는 성좌 기준(보통 키프레임 카메라) 좌표계여야 하고,
    // gravity 도 *그 좌표계에서의* 중력 방향이어야 한다. 없으면 카이랄리티 미적용.
    std::uint64_t insert(KeyframeId kf, Timestamp stamp, const SE3& anchor,
                         std::vector<ConstellationNode> nodes,
                         std::optional<Vec3> gravity = std::nullopt);

    // 질의. gravity 는 질의 프레임 좌표계 기준.
    // 실패 시 ErrorCode::InsufficientData / NotAvailable.
    [[nodiscard]] Result<ConstellationMatch> query(
        const std::vector<ConstellationNode>& nodes,
        std::optional<Vec3> gravity = std::nullopt) const;

    // 상위 후보를 모두 반환 (루프 클로저 다중 가설 검증용)
    [[nodiscard]] std::vector<ConstellationMatch> queryAll(
        const std::vector<ConstellationNode>& nodes,
        std::optional<Vec3> gravity = std::nullopt) const;

    // 활성 토큰 집합에서 성좌를 구성한다. 안정 랜드마크만 채택.
    [[nodiscard]] static std::vector<ConstellationNode> buildFrom(
        const std::vector<WorldTokenPtr>& tokens, const SE3& reference,
        std::size_t max_nodes);

    void   clear();
    [[nodiscard]] std::size_t placeCount() const { return places_.size(); }
    [[nodiscard]] const Place* place(std::uint64_t id) const;
    [[nodiscard]] const ConstellationConfig& config() const { return cfg_; }

private:
    // 쌍 -> 역색인 키
    [[nodiscard]] std::uint64_t pairKey(int class_a, int class_b, double distance) const;
    [[nodiscard]] int distanceBin(double d) const;

    // 성좌의 모든 쌍 키 (중복 포함)
    [[nodiscard]] std::vector<std::uint64_t> signature(
        const std::vector<ConstellationNode>& nodes) const;

    // 역색인 투표로 후보 장소를 뽑는다
    [[nodiscard]] std::vector<std::pair<std::uint64_t, double>> retrieve(
        const std::vector<std::uint64_t>& sig) const;

    // 클래스 일관 대응 탐색 + Kabsch 검증
    [[nodiscard]] Result<ConstellationMatch> verify(
        const std::vector<ConstellationNode>& query,
        const Place& place,
        const std::optional<Vec3>& query_gravity) const;

    // 후보 집합 전체를 보고 포즈 합의 통계(agree/support/margin/confidence)를 채운다.
    // 후보 하나만 봐서는 알 수 없는 양이라 verify 가 아니라 여기서 계산한다.
    void annotate(std::vector<ConstellationMatch>& all) const;

    // 후보의 월드 포즈. anchor 를 곱해야 서로 비교할 수 있다.
    [[nodiscard]] std::optional<SE3> worldPose(const ConstellationMatch& m) const;

    ConstellationConfig cfg_;

    std::vector<Place>  places_;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> inverted_;  // key -> place index
    std::unordered_map<std::uint64_t, std::uint32_t> place_lookup_;
    std::uint64_t next_place_id_{1};
};

// Kabsch/Umeyama. 스케일 없이 회전+평행이동만 추정한다.
// src -> dst 로 보내는 SE(3) 를 반환. 반사행렬을 배제한다.
[[nodiscard]] Result<SE3> kabsch(const std::vector<Vec3>& src, const std::vector<Vec3>& dst);

}  // namespace wme
