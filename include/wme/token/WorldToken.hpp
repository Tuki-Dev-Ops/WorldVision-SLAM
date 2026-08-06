#pragma once

// World Token: WME 의 유일한 객체 표현.
// 바운딩 박스는 엔진 경계를 넘지 않는다. YOLO 출력은 즉시 토큰으로 승격되고,
// 이후 모든 엔진(추적/의미/기억/예측/계획)은 같은 토큰 인스턴스를 참조한다.
//
// 이 단일화가 WME 의 핵심 주장이다. 기존 SLAM 은 랜드마크/트랙/의미노드/
// 예측대상을 각각 다른 자료구조로 들고 그 사이 매핑을 유지하느라 깨진다.

#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wme {

// 객체 생애 단계. 관측이 끊겨도 즉시 삭제하지 않는다.
enum class TokenLifecycle : std::uint8_t {
    Provisional,  // 관측 1~2회. 아직 세계에 편입 안 됨
    Active,       // 현재 관측 중
    Occluded,     // 가려짐. 위치 신뢰 유지
    Dormant,      // 시야 밖. 기억에만 존재
    Displaced,    // 있던 자리에서 사라짐이 확인됨
    Retired       // 폐기 예정
};

// 이 토큰이 로봇에게 무엇을 허용하는가
enum class Affordance : std::uint32_t {
    None       = 0,
    Traversable= 1u << 0,   // 밟고 지나갈 수 있음
    Obstacle   = 1u << 1,
    Supportive = 1u << 2,   // 위에 물건을 올릴 수 있음
    Container  = 1u << 3,
    Graspable  = 1u << 4,
    Openable   = 1u << 5,
    Sittable   = 1u << 6,
    Passage    = 1u << 7,   // 문/통로
    Agent      = 1u << 8,   // 사람/동물/차량 등 자율 개체
    Hazard     = 1u << 9
};

inline Affordance operator|(Affordance a, Affordance b) {
    return static_cast<Affordance>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
inline bool has(Affordance set, Affordance flag) {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0;
}

// 한 번의 관측 기록. 원본을 절대 덮어쓰지 않는 것이 원칙이다.
struct Observation {
    Timestamp   stamp{};
    FrameId     frame{};
    cv::Rect2f  box;              // 이미지 좌표
    float       detection_conf{0.f};
    Vec3        position_cam;     // 관측 시점 카메라 좌표
    Mat3        position_cov;     // 카메라 좌표 공분산
    double      visible_ratio{1.0};
    double      image_quality{1.0};
    double      sensor_reliability{1.0};
};

// 예측 결과. 관측과 물리적으로 분리해 보관한다.
struct PredictionState {
    Timestamp horizon{};        // 예측 대상 시각
    Vec3      position{Vec3::Zero()};
    Mat3      position_cov{Mat3::Identity()};
    Vec3      velocity{Vec3::Zero()};
    double    visibility_prob{0.0};
    double    reliability{0.0};  // 예측기 자체 신뢰도
};

class WorldToken {
public:
    // --- 정체성 -----------------------------------------------------------
    TokenId     id{};                 // 세계 전역 영속 ID
    int         class_id{-1};
    std::string class_name;

    // 클래스 확률 분포. argmax 만 들고 다니면 재식별 시 클래스 오분류를 복구할 수 없다.
    std::vector<float> class_distribution;

    // --- 기하 (월드 좌표) ---------------------------------------------------
    Vec3 position{Vec3::Zero()};
    Mat3 position_cov{Mat3::Identity()};
    Quat orientation{Quat::Identity()};
    Vec3 extent{Vec3::Zero()};        // 축정렬 반치수 (m)

    Vec3 velocity{Vec3::Zero()};
    Vec3 acceleration{Vec3::Zero()};
    Mat3 velocity_cov{Mat3::Identity()};

    // --- 현재 프레임 관측 ---------------------------------------------------
    cv::Rect2f  box;
    cv::Mat     mask;                 // CV_8UC1, 있으면 사용
    float       detection_conf{0.f};
    double      visible_ratio{1.0};   // 1 = 완전 가시
    double      occlusion{0.0};

    // --- 의미 --------------------------------------------------------------
    std::string semantic_label;       // class_name 보다 구체적일 수 있음 (예: "의자#3")
    Affordance  affordance{Affordance::None};
    NodeId      graph_node{};         // World Graph 상 대응 노드

    // --- 신뢰도 ------------------------------------------------------------
    double existence_belief{0.5};     // 이 객체가 실재할 사후확률
    double identity_belief{0.5};      // 동일 개체라는 사후확률
    double static_belief{0.5};        // 정적 객체일 사후확률 (동적 마스킹에 사용)
    double static_prior{0.5};         // 생성 시 사전분포. 관측이 끊기면 여기로 돌아간다

    // static_belief 가 관측에 근거하는가. false 면 그 값은 클래스 표에서 온
    // 사전분포일 뿐이고, 그것으로 화면을 지우면 14.1 이 기각한 클래스 마스킹을
    // 한 단계 아래에서 다시 하는 것이다. 판정이 한 번 돌면 참이 되고,
    // 연관이 끊겨 새 토큰이 생길 때 같은 자리의 같은 클래스로부터 물려받는다 -
    // 운동 상태는 트랙 ID 의 속성이 아니라 그 자리에 있는 것의 속성이다.
    bool static_evidenced{false};

    // 정적/동적 판정용 기준점. 인접 프레임 변위는 잡음에 묻히므로, 일정 시간이
    // 지날 때까지 기준을 붙들고 그 사이의 이동을 본다.
    // 프레임 간격이 짧다는 이유로 관측을 버리면 30 Hz 에서는 판정 자체가 아예
    // 돌지 않는다 - 실제로 그랬다 (docs/06-results.md 15 장).
    Vec3      static_ref_position{Vec3::Zero()};
    Timestamp static_ref_stamp{};

    // 융합 전 원시 관측 (월드) 과 그 표준편차. 융합 위치는 필터가 이미 매끄럽게
    // 만든 값이라 운동을 재는 데 쓰면 필터의 동역학을 재게 된다. 판정 입력이
    // 실제로 무엇이었는지 사후에 가리려면 원시값이 남아 있어야 한다.
    Vec3   meas_world{Vec3::Zero()};
    double meas_sigma{0.0};
    Vec3   static_ref_meas{Vec3::Zero()};

    // 정적 판정 진단. 믿음이 안 오르는 이유를 "증거가 약하다" 와 "갱신이 적다"
    // 중 하나로 돌리려면 둘을 따로 세어야 한다.
    struct StaticDiag {
        std::uint32_t updates{0};   // updateStaticBelief 실행 횟수
        double disp{0.0};           // 판정에 쓴 창 변위 (원시 관측 기준, m)
        double fused_disp{0.0};     // 같은 창의 융합 위치 변위 (비교용, m)
        double sigma{0.0};          // 판정에 쓰인 sigma (m)
        double z{0.0};
        double ratio{0.0};          // 로그오즈 증거 (게인 적용 전)
    };
    StaticDiag static_diag;

    // --- 생애 --------------------------------------------------------------
    TokenLifecycle lifecycle{TokenLifecycle::Provisional};
    Timestamp      first_seen{};
    Timestamp      last_seen{};
    std::uint32_t  observation_count{0};
    std::uint32_t  miss_count{0};     // 연속 미관측 횟수

    // --- 이력 --------------------------------------------------------------
    std::deque<Observation> history;      // 링 버퍼. 용량은 환경에 따라 가변
    std::deque<Vec3>        trajectory;   // 월드 좌표 궤적

    // --- 예측 (관측과 분리) --------------------------------------------------
    PredictionState prediction;

    // --- 맥락 --------------------------------------------------------------
    double sensor_reliability{1.0};   // 마지막 관측 시 환경 신뢰도

    // ----------------------------------------------------------------------
    // Displaced 는 "그 자리에 없는 것 같다" 는 위치 가설이지, 추론을 멈출 이유가
    // 아니다. 제외했더니 연관 후보에서 빠져 부재 증거를 더 받지 못했고,
    // 존재믿음이 displace 임계 바로 아래(0.247)에 얼어붙어 은퇴 임계(0.12)에
    // 영영 못 닿았다. 제거는 Retired 만 대상이므로 지도에 영구 유령이 남는다.
    // 후보로 남겨야 다른 자리에서 다시 찾을 수도 있다 - 그게 Displaced 의 목적이다.
    [[nodiscard]] bool isAlive() const {
        return lifecycle != TokenLifecycle::Retired;
    }

    // 정적 랜드마크로 신뢰할 만한가. TCG 성좌 구성 자격.
    [[nodiscard]] bool isStableLandmark() const {
        return static_belief > 0.7 && existence_belief > 0.6 &&
               observation_count >= 3 && positionSigma() < 0.5;
    }

    // 측광 정렬에서 제외해야 하는가.
    //
    // 판단은 오직 믿음으로 한다. 클래스로 override 하면 안 된다 - "person" 은
    // 클래스이지 운동 상태가 아니고, 앉아 있는 사람은 벽만큼 좋은 랜드마크다.
    // 자율 개체라는 사실은 이미 두 군데에 반영되어 있다:
    //   - 생성 시 사전분포 static_belief = 0.15
    //   - 정적 증거에 대한 감쇠 (agent_static_evidence_gain)
    // 여기서 또 한 번 강제하면 그 두 경로가 통째로 무의미해진다.
    // 클래스로 일괄 마스킹했을 때의 실측 대가는 docs/06-results.md 14 장에 있다.
    [[nodiscard]] bool isDynamic() const {
        return static_belief < 0.4;
    }

    [[nodiscard]] double positionSigma() const {
        return std::sqrt(std::max(0.0, position_cov.trace() / 3.0));
    }

    [[nodiscard]] double ageSeconds(Timestamp now) const {
        return first_seen.valid() ? (now - first_seen) : 0.0;
    }

    [[nodiscard]] double silenceSeconds(Timestamp now) const {
        return last_seen.valid() ? (now - last_seen) : 0.0;
    }

    void pushObservation(Observation obs, std::size_t capacity) {
        history.push_back(std::move(obs));
        while (history.size() > capacity) history.pop_front();
    }

    void pushTrajectory(const Vec3& p, std::size_t capacity) {
        trajectory.push_back(p);
        while (trajectory.size() > capacity) trajectory.pop_front();
    }
};

using WorldTokenPtr = std::shared_ptr<WorldToken>;

}  // namespace wme
