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

    // 정보행렬 스케일.
    //
    // rotation_sigma 는 **잘 맞는 평면 둘**(fitDegradation = 1)의 합성 법선 잔차
    // |n_cur - R n_ref| 의 표준편차다. 대응 하나의 실제 sigma 는
    //
    //     sigma_n = rotation_sigma * sqrt((f_ref^2 + f_cur^2) / 2),
    //     f = Plane::fitDegradation()^2 = (1 + 20 * rms)^2
    //
    // 다. 1/2 로 정규화했으므로 f = 1 인 평면 둘이면 정확히 rotation_sigma 로
    // 환원된다 - 실측으로 맞춘 절대 크기가 배율 도입으로 흔들리지 않는다.
    //
    // **크기는 유도되지 않았다.** 유도를 시도했고 실데이터가 기각했다:
    //   결맞은 깊이 오차가 만드는 기울기 c*d*sqrt(1-n_z^2)(116 배), 표면 거칠기
    //   rms/extent(4159 배), sigma_r/(sqrt(N)*extent)(135 배), 그리고 새 상수도
    //   지수도 없는 유도형 sigma_offset/(c*z*d)(12.7 배) 넷 다 시퀀스 간 산포가
    //   상수(11.4 배)보다 나빴다. 법선 sigma 를 유도하는 길은 찾지 못했다.
    //
    // 그래도 크기는 검산했다. 6 시퀀스 961 대응에서 정답 포즈로 잰 합성 법선
    // 잔차의 중앙값을 0.02*sqrt(2) 로 나누면 0.26 ~ 2.94, 기하평균 0.55 다
    // (가우시안이면 0.6745). 즉 0.02 rad 는 실측과 1.4 배 안에서 맞는다.
    //
    // **배율 f 는 유도가 아니라 적합이다.** 감추지 않는다. 961 대응에서
    // median(|법선 잔차| / 예측) 의 시퀀스 간 산포와 시퀀스 안 순위상관:
    //     상수                    11.4 배   rho -0.08
    //     1 + 20*rms               9.3 배   rho +0.22
    //     (1 + 20*rms)^2           7.5 배   rho +0.22   <- 채택
    //     sqrt(1 + 20*rms)        10.3 배   rho +0.22
    //     (1+20rms)(1+rms/extent)  9.1 배   rho +0.22
    //     1/confidence            12.8 배   rho +0.18
    //     1/min(1, N/400)         16.1 배   rho -0.10   <- 표본 수 항은 해롭다
    //     1 + rms/extent          11.1 배   rho +0.11
    //     z/d                     11.4 배   rho -0.21
    //     1/sqrt(1 - n_z^2)       34.2 배   rho +0.09
    //     sigma_offset/(c*z*d)    12.7 배   rho +0.02
    // 표본 수 항이 해롭다는 것이 이 표의 핵심이다 - 옛 코드가 회전 정보에 곱하던
    // confidence 에는 그 항이 들어 있었다. 적합 항만 남긴다.
    //
    // 지수는 두 지표를 함께 봐야 정해진다. 7 시퀀스 끝단 측정에서
    //     p:            0      1      2      3
    //     회전 W 산포  45.0   24.4   14.5   10.9   (단조 - 이것만으로는 못 고른다)
    //     ANEES 거리   1.128  0.976  0.980  1.084  (p ~ 1..2 에서 최소)
    // 회전 산포는 p 에 단조라 최적점이 없다. 1 차 지표인 ANEES 가 p=3 에서
    // 되돌아서므로 p <= 2 이고, 그 중 회전 산포가 낮은 p=2 를 골랐다.
    // **6 시퀀스에서 고른 지수라 표본이 얇다.** 새 상수를 만들지 않는 쪽을
    // 택하려면 p=1 이고 (회전 산포 24.4, ANEES 0.976), 그 숫자도 위에 있다.
    double rotation_sigma{0.02};       // rad

    // 평면의 sigma_offset 이 미상(0)일 때만 쓰는 **되돌림 값**이다. 정상 경로에서는
    // PlaneExtractor 가 유도한 평면별 sigma 가 이것을 대신한다.
    // 합성 잔차 |n_cur.t - (d_cur - d_ref)| 의 표준편차라는 뜻이다.
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

    // 이 대응의 오프셋 잔차 sigma (m). 두 평면의 sigma_offset 을 직교로 합친
    // 것이고, 한쪽이라도 미상이면 config 의 translation_sigma 로 물러선다.
    // 병진 최소제곱의 가중치와 정보행렬이 **둘 다** 이 값에서 온다 - 추정기와
    // 정보가 다른 가중을 쓰면 Lambda 는 그 추정기의 공분산이 아니게 된다.
    double sigma_offset{0.0};

    // 이 대응의 법선 잔차 sigma (rad). Kabsch 가중과 회전 정보가 둘 다 여기서
    // 온다. 유도는 StructuralAlignerConfig::rotation_sigma 주석에 있다.
    double sigma_normal{0.0};

    // ref.confidence * cur.confidence. **정보량이 아니다** - 대응 순서와 진단용.
    // 26.x 까지 이것이 정보행렬에 곱해졌고, 무차원인데다 inliers 400 에서
    // 포화해서 실제 벽은 전부 1 이었다.
    double weight{0.0};
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
