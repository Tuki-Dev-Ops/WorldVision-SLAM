#pragma once

// SPA - Structural Primitive Alignment (Tier 2).
//
// 평면으로 상대 포즈를 구속한다. 외관과 무관하므로 텍스처가 없거나 조명이
// 무너진 곳에서도 살아남는다 - 측광(Tier 0)이 죽는 바로 그 조건이다.
//
// 세 tier 는 팩터 종류가 다르지 않다. 전부 상대포즈 팩터로 들어가고 정보행렬
// Lambda 만 다르다. SPA 의 특징은 그 Lambda 가 **본질적으로 랭크 부족**일 수
// 있다는 것이다.
//
//   - 평면 하나: 회전 2 DOF (법선 둘레 회전은 관측 불가) + 병진 1 DOF
//   - 서로 다른 방향의 평면 둘: 회전 3 DOF, 병진은 법선이 만드는 부분공간만
//   - 복도(평행한 벽 둘 + 바닥): 회전 3 DOF, 복도 축 병진은 관측 불가
//
// 그 사실을 숨기지 않고 랭크로 보고하는 것이 이 모듈의 계약이다.
// docs/02-correspondence-problem.md 3장 Tier 2 의 '퇴화 복구' 역할이 여기서 나온다.
//
// 역할 한계: 평면 대응을 각도/거리로 찾으므로 초기 추정이 어느 정도 맞다고
// 가정한다. SPA 는 정제기이지 부트스트랩이 아니다.
//
// 스레드 안전성: 인스턴스 비공유.

#include "wme/core/Result.hpp"
#include "wme/core/SE3.hpp"
#include "wme/core/Types.hpp"
#include "wme/geometry/PlaneExtractor.hpp"

#include <cstddef>
#include <vector>

namespace wme {

struct StructuralAlignerConfig {
    double max_normal_angle{0.35};     // rad, 대응 판정
    double max_distance_diff{0.6};     // m
    std::size_t min_matches{2};

    // 랭크 판정. 최대 고유값 대비 이 비율 미만이면 관측 불가로 본다.
    // 병진 절단 최소제곱도 같은 문턱을 쓴다 - 랭크로 "없다" 고 말한 방향으로
    // 해가 움직이면 보고와 결과가 어긋난다.
    double degeneracy_ratio{1e-3};

    // 정보행렬 스케일. 평면 법선/거리 추정 정확도에서 온다.
    double rotation_sigma{0.02};       // rad
    double translation_sigma{0.05};    // m

    // 회전 해의 정칙화. 법선이 한 방향뿐이면 Kabsch 해는 그 축 둘레로 1-매개변수
    // 족이라 SVD 가 임의의 한 점을 고른다. init 쪽으로 아주 약하게 당겨
    // "구속되지 않은 축은 움직이지 않는다" 를 강제한다. 완전 랭크에서는
    // 상대 크기가 1e-6 이라 해에 영향이 없다.
    double rotation_prior_weight{1e-6};
};

// 평면 대응 하나. 인덱스는 align() 에 넘긴 벡터 기준.
struct PlaneMatch {
    std::size_t ref_index{0};
    std::size_t cur_index{0};
    double angle{0.0};             // rad, init 로 옮긴 뒤의 법선 사잇각
    double distance_diff{0.0};     // m
    double weight{0.0};            // ref.confidence * cur.confidence
};

struct StructuralAlignmentResult {
    SE3    T_cur_ref;                        // ref 좌표계 -> cur 좌표계
    Mat6   information{Mat6::Zero()};        // [rho(3), phi(3)] 순서. SE3 접선 규약과 동일
    Vec6   eigenvalues{Vec6::Zero()};        // 대각 정규화 후 고유값 (내림차순, 무차원)
    Vec6   weakest_direction{Vec6::Zero()};  // 가장 약하게 구속된 접선 방향 (단위벡터)
    int    observable_dof{0};                // 0..6
    int    rotation_rank{0};                 // 0..3
    int    translation_rank{0};              // 0..3

    std::vector<PlaneMatch> matches;

    // 해를 넣었을 때 실제로 남는 잔차. 랭크가 6 이어도 이게 크면 대응이 틀린 것이다.
    double normal_rms{0.0};                  // |n_cur - R n_ref| 의 가중 RMS (무차원)
    double offset_rms{0.0};                  // |n_cur·t - (d_cur - d_ref)| 의 가중 RMS (m)

    [[nodiscard]] bool fullRank() const { return observable_dof == 6; }
};

// 관측되지 않는 접선 방향들. Tier 0(ECDA)이 채워야 할 축을 알려주는 용도다.
// 융합 규칙이 서로의 구멍을 메우려면 어디가 구멍인지 말할 수 있어야 한다.
// 좌표축이 아니라 실제 고유벡터를 돌려준다 - 퇴화 축이 축정렬이라는 보장은 없다.
[[nodiscard]] std::vector<Vec6> unobservableDirections(const StructuralAlignmentResult& r,
                                                       double ratio = 1e-3);

class StructuralAligner {
public:
    explicit StructuralAligner(StructuralAlignerConfig config = {});

    // 법선 각도와 거리로 평면을 짝짓는다.
    // init 을 주면 그것으로 reference 를 옮긴 뒤 비교한다. 초기 추정이 없으면
    // 항등으로 가정하므로 큰 운동에서는 대응을 못 찾는다 - SPA 가 정제기인 이유다.
    [[nodiscard]] std::vector<PlaneMatch> match(const std::vector<Plane>& reference,
                                                const std::vector<Plane>& current,
                                                const SE3& init = SE3::identity()) const;

    // 평면 대응으로 ref -> cur 상대 포즈를 구한다.
    // alpha_structural 은 융합 규칙의 alpha_2(E). 항을 더하지 않고 정보량만 조절한다.
    // 랭크가 6 미만이면 성공하되 degraded 로 표시한다 - 값은 유효하지만
    // 관측되지 않은 축으로는 움직이지 않았다는 뜻이다.
    [[nodiscard]] Result<StructuralAlignmentResult> align(const std::vector<Plane>& reference,
                                                          const std::vector<Plane>& current,
                                                          const SE3& init = SE3::identity(),
                                                          double alpha_structural = 1.0) const;

    [[nodiscard]] const StructuralAlignerConfig& config() const { return cfg_; }

private:
    StructuralAlignerConfig cfg_;
};

}  // namespace wme
