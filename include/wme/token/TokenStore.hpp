#pragma once

// World Token Engine.
// 검출을 세계에 통합하고, 정체성을 유지하며, 생애주기를 관리한다.
//
// 기술자 없이 정체성을 유지하는 방법: 3D 위치 + 클래스 + 예측 + 박스 겹침을
// 결합한 비용으로 전역 최적 할당을 푼다. 외형 기술자가 없으므로 오히려
// 기하와 예측에 정직하게 의존하게 되고, 조명 변화에 무너지지 않는다.
//
// 스레드 안전성: 쓰기(integrate)는 단일 스레드. 읽기는 스냅샷을 통해서만.

#include "wme/confidence/ConfidenceEngine.hpp"
#include "wme/core/Frame.hpp"
#include "wme/core/Result.hpp"
#include "wme/core/SE3.hpp"
#include "wme/perception/Detection.hpp"
#include "wme/perception/EnvironmentState.hpp"
#include "wme/token/WorldToken.hpp"

#include <unordered_map>
#include <vector>

namespace wme {

struct TokenStoreConfig {
    // --- 연관 --------------------------------------------------------------
    double max_association_distance{2.0};   // m. 이보다 멀면 후보에서 제외
    double max_association_mahalanobis{9.0};// 카이제곱 게이트 (자유도 3, 97%)

    // 게이트 전용 기동 여유 (m/s). 게이트를 융합 공분산만으로 세우면 안 된다:
    // 정보필터는 상관된 관측을 독립으로 세어 공분산을 무너뜨리고, 그러면
    // 실제 운동이 전부 이상치로 거부된다.
    // 실측: 게이트에서 탈락한 최근접 동일클래스 후보가 174 mm(sitting) /
    // 185 mm(walking) 거리에 있었고 마할라노비스는 474 / 535 였다 - 융합
    // 공분산이 사람 박스 중심을 sigma 8 mm 로 안다고 주장한 것이다. 그 결과
    // 자율개체 토큰의 수명 중앙값이 1~2 관측이었다 (docs/06-results.md 16 장).
    // 이 값은 halfsphere 스윕에서도 유효하다. 포즈가 성한 상태로 잰
    // fr3_sitting_halfsphere 의 혁신 중앙값은 26 mm (자율개체) / 22 mm (정적
    // 클래스), 마할라노비스 중앙값 0.1, 게이트 통과 95 % 였고, 프레임간 카메라
    // 회전과 혁신의 상관은 +0.08 에 그쳤다. 회전이 게이트를 깨뜨린다는 가설은
    // 그 자리에서 기각된다 - 깨뜨리는 것은 포즈 발산이다
    // (docs/06-results.md 17.2 는 발산한 런에서 잰 값으로 반대 결론을 냈다).
    double assoc_maneuver_speed{2.0};
    double iou_weight{0.8};                 // 이미지 겹침 항 가중
    double distance_weight{1.0};            // 3D 거리 항 가중
    bool   allow_cross_class{false};        // 클래스가 다른 연관 허용 여부

    // --- 3D 추정 -----------------------------------------------------------
    double depth_noise_coeff{0.006};        // sigma_z = coeff * z^2 (스테레오/ToF 공통형)
    double bearing_noise_px{2.0};           // 박스 중심 픽셀 잡음
    double no_depth_sigma{5.0};             // 깊이 없을 때 위치 표준편차 (m)
    double depth_sample_shrink{0.5};        // 박스 중앙 이 비율만 깊이 샘플에 사용

    // 관측 깊이(보이는 표면) -> 무게중심 깊이 보정 계수.
    // 단위 주의: extent 는 *반치수* 이므로 centroid = surface + k * extent.z() 이고
    // 볼록 물체의 기하학적 값은 k = 1.0 이다. 전체 치수 기준 계수를 그대로
    // 넣으면 2 배 과보정된다 (실제로 0.778 을 넣어 4 m 물체에서 0.23 m 편향이 났다).
    // 이 항이 없으면 객체가 반치수만큼 카메라 쪽으로 당겨지고, 그 편향은
    // 관측 수로 줄지 않아 융합할수록 과신한다.
    // 오목/얇은 물체는 1.0 보다 작으므로 실센서에서는 재보정한다.
    // Python 참조(wme/graph/factors.py)는 같은 자리에 0.778 을 쓰는데, 그쪽은
    // 마스크 *전체* 의 깊이 중앙값이라 경사면이 섞여 근접면보다 깊다.
    // 여기는 depth_sample_shrink 로 박스 중앙만 재므로 근접면에 가깝다.
    // 계수는 샘플 영역에 딸린 값이다 - 두 값을 서로 옮기면 안 된다.
    double depth_centroid_offset{1.0};

    // --- 생애주기 ----------------------------------------------------------
    std::uint32_t observations_to_activate{3};
    double occluded_timeout_s{1.5};         // Active -> Occluded 후 유지 시간
    double dormant_timeout_s{45.0};         // Dormant 유지 시간
    double existence_retire_threshold{0.12};
    double existence_displace_threshold{0.25};

    // --- 이력 --------------------------------------------------------------
    std::size_t history_capacity{64};
    std::size_t trajectory_capacity{256};

    // --- 동적 마스크 --------------------------------------------------------
    double dynamic_mask_dilate{1.15};       // 마스크를 박스보다 이 배율로 키운다

    // --- 진단 ---------------------------------------------------------------
    // 검출별 연관 표본을 남긴다. 프레임마다 벡터를 채우므로 기본은 끈다.
    bool collect_assoc_probes{false};
};

// 검출 하나에 대한 연관 표본. 혁신(innovation)이 물체의 운동인지 카메라 포즈
// 오차인지는 같은 프레임의 정적 클래스와 자율 개체를 갈라 봐야만 나뉜다 -
// 두 쪽이 같이 크면 그것은 물체가 아니라 포즈다.
struct AssocProbe {
    int    class_id{-1};
    bool   agent{false};
    bool   matched{false};
    bool   gated{false};        // 거리 게이트는 통과, 카이제곱에서 탈락
    double dt{0.0};
    double dist{0.0};           // |z_world - predicted| (최근접 동일클래스 후보)
    double maha{0.0};
    Vec3   diff_world{Vec3::Zero()};
    Vec3   diff_cam{Vec3::Zero()};   // 카메라 축 분해 (x,y 횡 / z 깊이)
    double range{0.0};               // 관측의 카메라 거리 (m)
    double tok_sigma{0.0};
    double meas_sigma{0.0};
};

// 한 프레임 통합 결과 요약 (진단/시각화용)
struct IntegrationReport {
    std::size_t matched{0};
    std::size_t created{0};
    std::size_t missed{0};
    std::size_t retired{0};
    double      dynamic_area_ratio{0.0};    // EnvironmentAnalyzer 에 주입할 값

    // 연관 실패의 원인 분해. 새 토큰이 왜 생겼는지 세지 않으면 "증거가 느리다"
    // 와 "토큰이 못 산다" 를 구분할 수 없고, 그 둘은 고칠 곳이 다르다.
    std::size_t det_no_candidate{0};   // 같은 클래스 후보가 거리 게이트 안에 없었다
    std::size_t det_gated_out{0};      // 후보는 있었으나 카이제곱 게이트에서 탈락
    std::size_t det_unassigned{0};     // 게이트는 통과했는데 할당에서 밀렸다
    double      gated_maha_sum{0.0};   // 탈락한 최근접 후보의 마할라노비스 합
    double      gated_dist_sum{0.0};   // 같은 후보까지의 실제 거리 합 (m)

    std::vector<AssocProbe> probes;    // collect_assoc_probes 일 때만 채운다
};

class TokenStore {
public:
    explicit TokenStore(TokenStoreConfig config = {}, ConfidenceConfig confidence = {});

    // 검출 집합을 세계 상태에 통합한다.
    // T_world_cam 은 이 프레임의 카메라 포즈 (Localization Engine 출력).
    Result<IntegrationReport> integrate(const DetectionSet& detections, const Frame& frame,
                                        const SE3& T_world_cam, const EnvironmentState& env);

    // --- 조회 -------------------------------------------------------------
    [[nodiscard]] std::vector<WorldTokenPtr> activeTokens() const;
    [[nodiscard]] std::vector<WorldTokenPtr> allTokens() const;
    [[nodiscard]] std::vector<WorldTokenPtr> stableLandmarks() const;
    [[nodiscard]] WorldTokenPtr find(TokenId id) const;
    [[nodiscard]] std::size_t size() const { return tokens_.size(); }

    // 마스크가 어디서 왔는지. 지운 면적의 출처를 나누지 않으면 "믿음이 낮아서
    // 지웠다" 와 "판정이 한 번도 안 돌아 사전분포 그대로라서 지웠다" 가 섞인다.
    struct MaskReport {
        double masked_ratio{0.0};
        double from_observed{0.0};   // 이번 프레임 관측된 토큰이 지운 비율
        double from_stale{0.0};      // 관측이 끊긴 토큰이 지운 비율
        // 판정이 한 번도 안 돈 토큰이 지우려다 보류된 비율 (박스 면적 기준).
        // 사전분포만으로 지우려던 몫이 얼마였는지가 여기 남는다.
        double withheld_unjudged{0.0};
        std::size_t n_masking{0};
        std::size_t n_withheld{0};
    };

    // ECDA 가 쓸 정적 픽셀 마스크 (0 = 동적). 직접법의 고질적 약점을 여기서 없앤다.
    [[nodiscard]] cv::Mat buildStaticMask(const Frame& frame, const SE3& T_world_cam,
                                          MaskReport* report = nullptr) const;

    // 루프 클로저 후 동일 개체로 판명된 두 토큰을 병합한다.
    Status merge(TokenId keep, TokenId absorb);

    void clear();

    [[nodiscard]] const TokenStoreConfig& config() const { return cfg_; }
    [[nodiscard]] const ConfidenceEngine& confidence() const { return confidence_; }

private:
    // 검출로부터 카메라 좌표 위치와 공분산을 추정한다
    struct Measurement {
        Vec3 position_cam{Vec3::Zero()};
        Mat3 cov_cam{Mat3::Identity()};
        Vec3 extent{Vec3::Zero()};
        bool has_depth{false};
        double visible_ratio{1.0};
    };
    [[nodiscard]] Measurement measure(const Detection& det, const Frame& frame) const;

    // 토큰의 현재 예측 위치를 이미지에 투영해 박스를 만든다
    [[nodiscard]] bool projectToken(const WorldToken& token, const Frame& frame,
                                    const SE3& T_cam_world, cv::Rect2f& out_box,
                                    double& out_depth) const;

    // 연관 비용 행렬 구성 + 헝가리안
    // near_token 은 검출별 "거리 게이트 안의 최근접 동일클래스 후보" 색인이다.
    // 연관에는 쓰이지 않고, 연관이 끊겼을 때 운동 상태 판정을 물려받는 데 쓴다.
    void associate(const DetectionSet& detections, const std::vector<Measurement>& meas,
                   const std::vector<WorldTokenPtr>& candidates, const Frame& frame,
                   const SE3& T_cam_world, const SE3& T_world_cam,
                   std::vector<int>& det_to_token, std::vector<double>& margins,
                   std::vector<int>& near_token, IntegrationReport& report) const;

    // 정보필터 기반 위치 융합
    void fusePosition(WorldToken& token, const Vec3& z_world, const Mat3& R_world, double dt) const;

    // 생애주기 전이
    void updateLifecycle(WorldToken& token, Timestamp now, const EnvironmentState& env) const;

    TokenId allocateId() { return TokenId(next_id_++); }

    TokenStoreConfig  cfg_;
    ConfidenceEngine  confidence_;

    // ID 오름차순으로 정렬된 토큰 목록을 out 에 채운다.
    // 해시 순회 순서가 결과로 새지 않게 하는 유일한 통로이며, 재사용 버퍼를
    // 받으므로 프레임마다 벡터를 새로 잡지 않는다.
    void collectSorted(std::vector<WorldTokenPtr>& out) const;

    std::unordered_map<TokenId, WorldTokenPtr> tokens_;
    std::uint64_t next_id_{1};
    Timestamp     last_stamp_{};

    // --- 프레임 간 재사용 버퍼 ---------------------------------------------
    // integrate/associate 는 매 프레임 같은 모양의 임시 배열을 쓴다.
    // const 경로(associate)에서도 쓰이므로 mutable 이다. 인스턴스 비공유가
    // 전제이므로(헤더 상단 스레드 안전성 계약) 락은 필요 없다.
    mutable std::vector<float>         depth_scratch_;    // 박스 내 깊이 중앙값용
    mutable std::vector<double>        cost_scratch_;     // 연관 비용 행렬
    mutable std::vector<double>        near_dist_scratch_;
    mutable std::vector<double>        near_maha_scratch_;
    mutable std::vector<char>          feasible_scratch_;
    mutable std::vector<WorldTokenPtr> sorted_scratch_;   // collectSorted 결과
    std::vector<Measurement>           meas_scratch_;
    std::vector<WorldTokenPtr>         candidate_scratch_;
    std::vector<int>                   det_to_token_scratch_;
    std::vector<int>                   near_token_scratch_;
    std::vector<double>                margin_scratch_;
    std::vector<char>                  matched_scratch_;
};

}  // namespace wme
