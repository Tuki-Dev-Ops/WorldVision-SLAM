#pragma once

// ECDA - Environment-Conditioned Direct Alignment (Tier 0).
//
// 측광 잔차를 직접 최소화해 상대 포즈를 구한다. 키포인트도 기술자도 쓰지 않는다.
// 고전 직접법(DSO/LSD)과 다른 점 세 가지:
//   1. 픽셀 가중치가 잔차 통계가 아니라 물리 모델에서 온다
//      (포화/블러/노이즈 = ImageQuality, 동적객체 = YOLO 토큰 마스크)
//   2. 정보행렬이 실데이터에서 보정되어 있다. 잔차를 독립으로 세지 않고
//      유효 표본수를 명시한다 (InformationModel). TUM 다섯 시퀀스 ANEES
//      146 087 -> 9.4, 프레임별 95% 게이트 통과 0/1086 -> 994/1086
//   3. 퇴화를 "발견"하지 않고 "보고"한다. 고유값 스펙트럼으로 관측 불가능한
//      자유도를 명시해 Tier 2(SPA)가 정확히 그 축만 채우게 한다
//
// 상세 설계 근거: docs/02-correspondence-problem.md 3장 Tier 0.
//
// 스레드 안전성: 인스턴스 비공유. 내부 버퍼를 프레임 간 재사용한다.

#include "wme/core/Frame.hpp"
#include "wme/core/Result.hpp"
#include "wme/core/SE3.hpp"
#include "wme/core/ThreadPool.hpp"
#include "wme/perception/EnvironmentState.hpp"
#include "wme/perception/ImageQuality.hpp"

#include <vector>

namespace wme {

// 보고할 포즈 정보행렬을 어떤 잡음 모델로 만들 것인가.
//
// 앞의 셋은 같은 식이고 유효 표본수 N_eff 만 다르다.
//
//   Lambda = H_pose / (chi2 / N_eff)
//
// chi2 는 해에서의 가중 잔차 제곱합이므로 chi2/N_eff 는 "독립 관측 하나가 갖는
// 잔차 분산" 이다. 즉 이 모델들은 "이 프레임의 측광 잔차가 독립 관측 몇 개어치
// 인가" 라는 한 가지 질문에만 서로 다르게 답한다. TUM 실측 ANEES (수용구간
// [5.5, 6.5], 정확도는 셋 다 완전히 동일 - 정보행렬은 추정에 관여하지 않는다):
//
//   SensorVariance     5 425 ~ 146 087
//   ResidualVariance     845 ~   8 621
//   EffectiveSample      3.7 ~     9.4
enum class InformationModel {
    // N_eff 를 쓰지 않고 Lambda = H / photometricVariance().
    // 잔차가 (i) 서로 독립이고 (ii) 그 분산이 센서 잡음과 같다고 본다.
    // 실측에서 (ii) 부터 틀린다 - 해에서의 잔차 분산이 센서 잡음의 3~34 배다.
    SensorVariance,

    // N_eff = sum(w). 가중된 픽셀 하나하나가 독립 증거라는 뜻.
    // 가정 (ii) 를 뺀다. (i) 은 남는다.
    ResidualVariance,

    // N_eff = effective_samples. 프레임 전체가 독립 관측 nu 개라는 뜻.
    // 가정 (i) 도 뺀다. 근거는 서브샘플링 실측 (effective_samples 주석).
    // 한계: nu 가 카메라마다 다르다. fr2 에서 필요한 nu 는 0.10~0.16 이다.
    EffectiveSample,

    // 위 셋과 분모의 성격이 다르다. N_eff 뿐 아니라 "한 관측의 잔차 분산" 도
    // 바꾼다.
    //
    //   Lambda = (H_pose / N) / coherent_sigma^2
    //
    // 즉 분모가 chi2/N_eff (이 프레임이 실제로 달성한 잔차 분산) 가 아니라
    // 상수 coherent_sigma^2 다. 근거는 coherent_sigma 주석.
    CoherentFrame,

    // 샌드위치. 공간 클러스터별 스코어의 표본공분산으로 중간항을 실측한다.
    //   M = G/(G-1) sum_c score_c score_c^T,  Cov = H^-1 M H^-1
    // (i) 을 "클러스터 안은 임의 상관, 클러스터 사이는 독립" 으로 약화한다.
    // 실측 결과 이 방식이 걷어내는 상관은 ResidualVariance 대비 1.4~2.4 배뿐이고
    // (fr1_360 에서는 오히려 나빠진다), 클러스터를 16 개에서 64 개로 바꿔도
    // 결과가 1.5 배 안에서 같다 - 즉 한 프레임 *안* 에서 볼 수 있는 상관은
    // 이미 다 걷은 것이고, 남은 과신은 프레임 단위 항이라 여기서는 안 보인다.
    // 진단용으로 남겨 둔다.
    ClusterRobust,
};

struct DirectAlignerConfig {
    // 640x480 기준 실측 수렴 반경 (평면 텍스처, 항등 초기화):
    //   레벨 1: 0 m / 3: 0.02 m / 4: 0.10 m / 5: 0.18 m (= 레벨0 기준 36 px)
    // 레벨 4 는 40x30 에 300점이라 비용이 거의 없는데 반경이 두 배가 된다.
    int    pyramid_levels{5};
    int    max_iterations{30};       // 레벨당 상한
    double convergence_delta{1e-6};  // 갱신량 노름 수렴 판정

    // 픽셀 선택
    int    grid_cell{8};             // 격자 셀 크기. 셀당 최상위 그래디언트 1점
    double min_gradient{6.0};        // 이보다 약한 픽셀은 포즈를 구속하지 못한다
    std::size_t max_points{6000};    // 레벨당 상한
    std::size_t min_points{60};      // 이보다 적으면 추정 포기

    // 깊이 유효 범위 (m)
    double min_depth{0.15};
    double max_depth{40.0};

    // 깊이 불확실성 계수 c: sigma_Z = c * Z^2  (단위 1/m). 0 이면 끈다.
    //
    // 왜 Z^2 인가. 스테레오는 Z = f*B/d 이므로 sigma_Z = Z^2/(f*B) * sigma_d,
    // 즉 c = sigma_d/(f*B). 구조광(Kinect)도 실측이 Z^2 에 비례해 c ~ 1.4e-3.
    // 두 센서가 같은 꼴이라 계수 하나로 덮인다.
    //
    // 왜 필요한가. ECDA 는 깊이를 참으로 믿고 3D 점을 만든다. 실내(TUM, 최대
    // ~8 m)에서는 그 가정이 거의 맞지만, KITTI 에서 60 m 점의 깊이 오차는
    // +-4 m 이고 6 m 점은 +-4 cm 다 - 1000 배 차이인데 같은 무게로 들어간다.
    // 실측(06-results.md 25.20): 깊이 상한을 40 m 에서 60 m 로 올리면
    // RANSAC-PnP 기준선은 좋아지고(146->129 cm) ECDA 는 나빠졌다(172->224 cm).
    // 기준선은 그 점들을 이상치로 버리고 ECDA 는 버릴 방법이 없었다.
    double depth_sigma_rel{0.0};

    // 4-이웃 깊이가 이 상대차를 넘으면 경계로 보고 그 점을 버린다.
    // 근거: ECDA 가 고르는 최강 그래디언트는 실장면에서 대개 물체 윤곽이고,
    // 거기 깊이는 전경/배경 중 무엇이 잡혔는지 알 수 없다. 3D 위치가 틀린 점은
    // 잔차를 키우는 게 아니라 포즈를 끌고 간다.
    // 상대값인 이유: 깊이 잡음은 z^2 에 비례한다.
    //
    // 다만 실측으로는 이득이 확인되지 않았다. TUM fr1_xyz 에서 0.02 / 0.05 /
    // 해제가 각각 ATE 1.55 / 1.56 / 1.56 cm 로 차이가 없다. 이 항이 필요하다고
    // 본 진단은 틀렸고, 실제 원인은 운동 사전분포 부재였다(15.18 -> 2.32 cm).
    // 물리적 근거는 남아 있으므로 가장 덜 개입하는 값으로 두되, 동적 물체가
    // 많은 시퀀스에서 재검증하기 전까지는 검증된 항이 아니다.
    double depth_edge_ratio{0.05};

    // 로버스트 커널.
    // 임계를 intensity 고정값으로 두면 안 된다. 초기 오차가 클 때 잔차가 큰 점이
    // 곧 큰 변위 정보를 지닌 점인데, 고정 임계는 그 점들만 골라 죽인다.
    // 실측(huber_delta=12, 평면 장면): 수렴 반경이 0.10 m -> 0.03 m 로 좁아지고
    // 0.06 m 에서는 초기 오차의 2.0 배로 발산했다. 같은 장면에서 12 -> 40 으로
    // 올리면 순수 L2 와 결과가 같아진다. 즉 이 설정의 커널은 해롭기만 했다.
    //
    // 대신 매 반복 잔차 분포의 로버스트 산포에 비례시킨다.
    //   sigma = 1.4826 * median(|r|)
    //
    // 다만 sigma 에 그냥 비례시키면 부족하다. 크게 어긋난 상태의 잔차 분포는
    // "정상치 다수 + 이상치 소수" 가 아니라 전부 어긋남이라, 로버스트 통계의
    // 전제 자체가 깨진다. 그 상태에서 상위 절반을 깎으면 정확히 큰 변위 정보를
    // 버리게 된다 (실측: 0.10 m 에서 오차비 0.005 -> 1.22).
    //
    // 커널은 잔차가 실제 센서 잡음 수준에 근접했을 때만 의미가 있다.
    //   delta = huber_k * sigma * max(1, sigma / (noise_ratio * noise_sigma))
    // 잔차가 잡음 수준이면 delta = 1.345 * sigma 로 제대로 로버스트하고,
    // 잡음의 10 배면 delta 가 그만큼 넓어져 사실상 L2 가 된다.
    // 스케줄을 손으로 정하지 않고, ImageQuality 가 측정한 물리량에서 나온다.
    double huber_k{1.345};           // sigma 배수. 가우시안 95% 효율 상수
    double huber_noise_ratio{2.0};   // 이 배수 이내면 커널을 완전히 조인다
    double huber_min_delta{2.0};     // 잡음 추정이 없을 때의 대체 잡음 수준 (intensity)

    // inlier 판정용 고정 임계 = health_k * max(측정 잡음, huber_min_delta).
    // 커널 임계와 분리해야 한다. 적응형 임계로 inlier 를 세면 잔차가 커질 때
    // 임계도 같이 넓어져 비율이 유지되고, 건강 신호가 동어반복이 된다.
    double health_k{4.0};

    // 밝기 아핀 모델 I_cur ~ a*I_ref + b. 노출/게인 변화를 흡수한다.
    bool   estimate_affine{true};
    double affine_prior_weight{1e2}; // a=1,b=0 으로 끌어당기는 사전분포

    // 퇴화 판정: 최대 고유값 대비 이 비율 미만이면 관측 불가 축으로 본다
    double degeneracy_ratio{1e-3};

    // 레벨 간 전달 시 관측 가능으로 인정할 최소 고유값 비율.
    // 각 레벨은 자기가 실제로 구속한 방향으로만 포즈를 움직여야 한다.
    // 이 게이트가 없으면 거친 레벨(20x15, 66점으로 8개 파라미터)이 구속하지도
    // 못한 축의 값을 그대로 다음 레벨에 넘긴다. 실측: 레벨을 4->5 로 늘리면
    // 오차가 0.28 m -> 0.51 m 로 나빠졌다. 진단용 degeneracy_ratio 와 목적이
    // 다르므로 따로 둔다 - 이쪽은 보고가 아니라 갱신을 막는 값이다.
    double level_observable_ratio{1e-2};

    // 레벤버그-마쿼트 감쇠
    double lambda_init{1e-4};
    double lambda_up{10.0};
    double lambda_down{0.3};

    // 감쇠 바닥. 최대 대각 성분 대비 비율.
    // 순수 상대 감쇠(lambda * H(k,k))는 랭크 부족 방향을 전혀 감쇠하지 못한다.
    // H(k,k) 가 0 에 가까우면 감쇠도 0 이라 그 방향 해가 무한정 커진다.
    // 거친 피라미드 레벨에서 텍스처가 완만한 램프에 가까워지면 실제로 일어나며,
    // 레벨 4~5 를 켰을 때 포즈가 0.58 m 로 튀던 원인이다.
    // H 는 전부 (intensity/단위)^2 라 대각끼리 비교는 "단위당 영상이 얼마나 변하나"
    // 를 비교하는 것이 되어 의미가 있다. 다만 m 와 rad 의 단위 차이만큼 스케일
    // 의존성이 남는다 (fx=500, 깊이 2.5 m 기준 두 축이 같은 자릿수).
    double damping_floor_ratio{1e-4};

    // 결정적 누산을 위한 고정 블록 수. 워커 수와 무관하게 합산 순서가 같다.
    std::size_t reduction_blocks{16};

    // --- 정보행렬 ----------------------------------------------------------
    // 기본은 EffectiveSample. 근거는 실측이다: 기존 SensorVariance 로 낸 Lambda 는
    // TUM 다섯 시퀀스에서 ANEES 5 425 ~ 146 087 (수용구간 [5.5, 6.5]) 였고
    // 1086 프레임 중 0 개가 프레임별 95% 게이트를 통과했다. 정확도는 좋았으므로
    // (상대 병진오차 중앙 5~16 mm) 부정확한 게 아니라 과신이었다.
    // 기본이 EffectiveSample 에서 CoherentFrame 으로 바뀌었다. EffectiveSample 은
    // 다섯 시퀀스 한 카메라에서 ANEES 3.7~9.4 였지만 다른 카메라(fr2)에서
    // 38.8/58.6 으로 깨진다. 12 시퀀스 3 카메라 기준 최악 이탈 9.76x -> 2.43x.
    // 정확도는 어느 쪽도 건드리지 않는다 - 정보행렬은 순수 출력이고 그건
    // InformationModelDoesNotChangeTheEstimate 가 지킨다.
    InformationModel information_model{InformationModel::CoherentFrame};

    // 유효 표본수 nu. "이 프레임의 측광 잔차 전체가 독립 관측 몇 개어치인가".
    //
    // 1 로 두는 근거는 서브샘플링 실측이다. 점 집합을 1/2, 1/4, 1/8, 1/16 로
    // 솎아 다섯 시퀀스를 다시 돌리면 (python/tools/tum_nees.py --scaling):
    //     Lambda ~ N^+1.00    당연하다. 점마다 J J^T 를 더한다
    //     오차^2 ~ N^-0.00    전혀 줄지 않는다
    //     ANEES  ~ N^+0.97    그래서 과신이 N 에 그대로 비례한다
    // 점을 16 배 넣어도 fr1_xyz 상대 병진오차 중앙값은 5.29 -> 5.11 mm 로
    // 그대로다. 즉 한 프레임의 측광 잔차는 점 개수와 무관하게 사실상 한 번의
    // 관측이다. 상관 모델(N_eff = N/kappa)로는 이 모양이 나오지 않는다 -
    // 그쪽은 N 에 비례해 줄어야 하고, 실제로 그런 항은 남아 있지 않다.
    //
    // nu=1 에서 다섯 시퀀스 ANEES 는 3.67 / 4.29 / 4.74 / 8.00 / 9.39 이고
    // (즉 필요한 nu 는 0.61~1.63, 과신과 과소신뢰 양쪽에 걸친다) 서브샘플링
    // 다섯 단계에서도 같은 범위에 머문다. 1 은 그 범위의 한가운데이자
    // "프레임 = 관측 하나" 라는 물리적 경계값이라 맞춘 값이 아니다.
    //
    // 한계: 이 값은 잔차가 실제로 상관되어 있다는 사실에 기대고 있다. 측광
    // 잔차가 정말 독립인 데이터(합성 장면 + 화이트 노이즈)에서는 점 개수만큼
    // 보수적이 된다. 그런 데이터에서는 ResidualVariance 가 맞다.
    double effective_samples{1.0};

    // 한 프레임이라는 관측 하나가 갖는 잔차 분산 (intensity). CoherentFrame 전용.
    //
    // EffectiveSample 은 그 분산으로 "이 프레임이 실제로 달성한 잔차 분산"
    // chi2/N 을 쓴다. 12 시퀀스 실측은 그게 틀렸다고 말한다. 프레임마다 필요한
    // 분산 var_req = (NEES/6) * chi2 를 뽑아 var = N^a * rmse^p 꼴로 맞춰 보면
    // 시퀀스 사이에서는 p = 0 이 최적이고 p 를 키울수록 단조롭게 나빠진다
    // (최악 이탈 2.43x -> 8.13x, p=2 가 곧 EffectiveSample 이다). 즉 달성한
    // 잔차 크기는 어느 시퀀스가 더 부정확한지에 대해 아무것도 말해주지 않는다.
    // 잔차가 가장 작은 fr2 가 가장 과신하는 이유가 이것이다 (rmse 7.5 대 9.6~22.8,
    // ANEES 38.8/58.6).
    //
    // 남는 식은
    //     Lambda = (H / N) / coherent_sigma^2
    // 즉 "프레임은 평균적인 점 하나를 잡음 coherent_sigma 로 관측한 것과 같은
    // 정보를 준다". N 이 약분되므로 15.1 의 서브샘플링 불변성은 그대로다
    // (솎은 실행에서 ANEES ~ N^-0.02 ~ +0.01, 기존 모델은 -0.02 ~ -0.10).
    //
    // 값 15.5 는 12 시퀀스 3 카메라에서 맞춘 상수다. 맞춘 값이라는 사실은
    // 숨기지 않는다. 다만 nu 와 달리 leave-one-out 이 흔들리지 않는다:
    // 11 개로 맞춘 값이 15.0~16.25 범위에 있고 뺀 시퀀스의 ANEES 는 2.27~15.74
    // (기존 모델은 2.56~58.58). 카메라 사이에서 깨지지 않는다는 뜻이지
    // 물리적으로 유도된 값이라는 뜻은 아니다.
    //
    // 이 상수를 관측으로 대체하려면 프레임 단위 공통오차를 밖에서 재야 한다.
    // 내부 중복(점 솎기, 3프레임 사이클)으로는 안 된다 - 사이클 잔차가 실제
    // 오차보다 두 자릿수 작다. 근거는 tests/test_direct_aligner.cpp 의 주석과
    // docs 참조.
    double coherent_sigma{15.5};

    // 스코어 클러스터 격자. 이미지를 GxG 로 나눈다 (클러스터 수 = G^2).
    // ClusterRobust 에서만 쓰인다. 실측: 격자 4(16개)와 8(64개)의 결과가
    // 1.5 배 안에서 같고, 3(9개)에서는 M 이 랭크 부족이 되어 ANEES 가
    // 1e4~1e5 로 튄다. 즉 이 값은 16~64 구간에서 답을 좌우하지 않는다.
    int info_cluster_grid{8};

    // 비어있지 않은 클러스터가 이보다 적으면 샌드위치를 쓰지 않는다.
    // 8x8 인 M 을 추정하는데 클러스터가 모자라면 Cov 가 특이해지고, 그 방향에
    // 무한한 확신이 실린다. 실측: 클러스터 4 개에서 ANEES 가 1e16, 9 개에서
    // 1e5 까지 튀었다. 9 개도 통과시키지 않도록 32 로 둔다.
    int info_min_clusters{32};
};

struct AlignmentResult {
    SE3    T_cur_ref;                    // ref 좌표계 -> cur 좌표계
    Mat6   information{Mat6::Zero()};    // 포즈 정보행렬 (물리 스케일 적용)
    Vec6   eigenvalues{Vec6::Zero()};    // 정보행렬 고유값 (내림차순)
    int    observable_dof{0};            // 관측 가능한 자유도 수 (0..6)
    Vec6   weakest_direction{Vec6::Zero()};  // 가장 약하게 구속된 접선 방향

    double affine_a{1.0};
    double affine_b{0.0};

    double photometric_rmse{0.0};
    double inlier_ratio{0.0};
    std::size_t point_count{0};
    std::size_t inlier_count{0};
    int    iterations{0};

    // --- 기하 정합성: 측광과 **독립인** 실패 신호 -------------------------
    //
    // ref 의 3D 점을 추정 포즈로 cur 에 투영하고, 그 화소에서 cur 이 실제로
    // 보고한 깊이와 비교한 상대 차이의 중앙값. cur 의 깊이맵은 정렬에 쓰이지
    // 않으므로(ref 깊이 + cur 밝기만 쓴다) 이것은 추정에 참여하지 않은 관측이다.
    //
    // 왜 필요한가. 06-results.md 13.3 은 측광 자기평가가 동적 장면에서 눈멀고,
    // 23.3 은 안개 속에서 엔진이 바닥값의 11.6 배로 표류하면서 실패를 **0 회**
    // 보고한다는 것을 쟀다. 26절이 그 이유를 잰다: 측광 잔차는 밝기 범위에,
    // inlier 비는 [0,1] 에 갇혀 있어 실패가 커져도 4.3 배 / 1.9 배에서 포화한다.
    // 깊이 불일치는 미터 단위 물리량이라 상한이 없고, 실측에서 오차 11.6 배에
    // 대해 10.4 배로 따라 올라간다.
    //
    // -1 = 표본 부족(깊이 구멍 등)으로 판정하지 않음. 0 을 쓰면 "완벽히
    // 일치한다" 와 구분되지 않는다.
    double depth_consistency{-1.0};      // median |z_pred - z_meas| / z_meas
    double depth_outlier_ratio{-1.0};    // 상대차 > 10 % 인 표본 비율

    [[nodiscard]] bool fullRank() const { return observable_dof == 6; }

    // 기하적으로 신뢰할 수 있는가. 판정 불가(-1)는 "괜찮다"로 읽지 않는다.
    //
    // 기본값 0.0057 은 적합된 값이다. 10 개 시퀀스 3 카메라에서 "진리값 상대
    // 병진오차 상위 10 %" 를 실패로 두고 Youden J 를 최대화한 문턱의 중앙값이며,
    // 시퀀스별 범위는 0.0035~0.0117 (3.4x), 카메라별 중앙값은 fr1 0.0055 /
    // fr2 0.0035 / fr3 0.0070 이다. `tools/depth_gate_calib.py` 가 다시 만든다.
    //
    // **처음 넣었던 0.02 는 자리표시자였고 너무 헐거웠다.** 정상 프레임의
    // 중앙값이 0.004 라서 0.02 문턱은 사실상 아무 것도 걸러내지 않는 - 즉
    // 언제나 true 를 돌려주는 게이트였다.
    //
    // 한계는 그대로 남는다: fr2 에서는 정상/실패 분포가 거의 겹쳐(비 1.02,
    // J 0.14) 어떤 문턱도 분리하지 못한다. 이 신호는 카메라에 따라 **분리력
    // 자체가** 달라진다.
    static constexpr double kDepthConsistencyGate = 0.0057;

    [[nodiscard]] bool depthConsistent(double max_rel = kDepthConsistencyGate) const {
        return depth_consistency >= 0.0 && depth_consistency <= max_rel;
    }

    // 기하 정합성에서 온 신뢰도 0..1. 판정 불가면 1 을 돌려주지 않는다 -
    // "모른다" 를 "괜찮다" 로 접으면 깊이 없는 장면이 최고 신뢰도를 받는다.
    [[nodiscard]] double depthReliability(
        double max_rel = kDepthConsistencyGate) const {
        if (depth_consistency < 0.0) return 0.5;      // 판정 불가 = 중립
        if (depth_consistency <= max_rel) return 1.0;
        return std::max(0.05, max_rel / depth_consistency);
    }
};

class DirectAligner {
public:
    explicit DirectAligner(DirectAlignerConfig config = {}, ThreadPool* pool = nullptr);

    // ref -> cur 상대 포즈 추정.
    // ref 는 깊이를 가져야 한다(RGB-D/스테레오). init 은 초기 추정 (보통 등속 예측).
    // quality/env 가 주어지면 픽셀 가중과 정보 스케일에 반영된다.
    [[nodiscard]] Result<AlignmentResult> align(const Frame& ref, const Frame& cur,
                                                const SE3& init,
                                                const ImageQuality* quality = nullptr,
                                                const EnvironmentState* env = nullptr);

    [[nodiscard]] const DirectAlignerConfig& config() const { return cfg_; }

    // 마지막 정렬에서 사용한 레벨별 점 개수 (진단용)
    [[nodiscard]] const std::vector<std::size_t>& pointsPerLevel() const { return points_per_level_; }

private:
    // 정렬에 쓰이는 한 점. ref 프레임 기준.
    struct AlignPoint {
        float u{0.f}, v{0.f};    // 해당 레벨 픽셀 좌표
        float intensity{0.f};    // ref 밝기
        float depth{0.f};        // ref 깊이 (m)
        float weight{1.f};       // 품질/마스크 기반 정적 가중
    };

    // 8x8 정규방정식 누산기 (포즈 6 + 아핀 2)
    struct Accumulator {
        Eigen::Matrix<double, 8, 8> H{Eigen::Matrix<double, 8, 8>::Zero()};
        Eigen::Matrix<double, 8, 1> b{Eigen::Matrix<double, 8, 1>::Zero()};
        double        chi2{0.0};
        double        wsum{0.0};      // sum(w). 잔차 분산의 분모
        std::size_t   inliers{0};
        std::size_t   used{0};

        void operator+=(const Accumulator& o) {
            H += o.H; b += o.b; chi2 += o.chi2; wsum += o.wsum;
            inliers += o.inliers; used += o.used;
        }
        void reset() { *this = Accumulator{}; }
    };

    // 깊이 피라미드. 최근접 축소만 쓴다 (깊이 평균은 물리적으로 무의미).
    void buildDepthPyramid(const cv::Mat& depth, int levels);

    // 레벨별 픽셀 선택. 격자 셀당 최상위 그래디언트 1점으로 공간 분포를 강제한다.
    // Frame 을 통째로 받으면 호출마다 피라미드 벡터 5개가 복사된다. 실제로 읽는
    // 두 가지(피라미드, 정적 마스크)만 넘긴다.
    void selectPoints(const ImagePyramid& pyr, const cv::Mat& static_mask,
                      const ImageQuality* quality, int level);

    // 깊이 경계 판정. 경계 픽셀의 깊이는 전경/배경 중 무엇인지 알 수 없다.
    [[nodiscard]] bool depthIsLocallyFlat(const cv::Mat& D, int x, int y, float d) const;

    // 잔차/자코비안 누산. [lo, hi) 구간만 처리한다.
    // huber 는 호출자가 정한다 (잔차 산포에 따라 매 반복 달라진다).
    // health 는 그와 별개로 고정된 물리 임계. inlier 집계에만 쓴다.
    //
    // cluster_grid > 0 이면 공간 클러스터별 스코어 sum w_i J_i r_i 를 scores_ 에
    // 같이 쌓는다. scores_ 는 블록별로 나뉘지 않으므로 이 경로는 반드시 단일
    // 스레드 1패스(lo=0, hi=N)로만 호출한다. 정보행렬 산출에만 쓰인다.
    void accumulate(const cv::Mat& cur_gray, const cv::Mat& cur_gx, const cv::Mat& cur_gy,
                    const CameraIntrinsics& K, const SE3& T,
                    double affine_a, double affine_b, double huber, double health,
                    double photo_sigma,
                    std::size_t lo, std::size_t hi, Accumulator& acc,
                    int cluster_grid = 0) const;

    // 마지막 누산의 |잔차| 분포에서 로버스트 임계를 만든다.
    // noise_sigma 는 측정된 영상 잡음(intensity). 점이 너무 적으면 L2 로 둔다.
    [[nodiscard]] double robustDelta(double noise_sigma) const;

    // 아핀을 주변화한 6x6 포즈 정보행렬. 모델에 따라 세 방식 중 하나.
    [[nodiscard]] Mat6 poseInformation(const Accumulator& acc, double sensor_var) const;

    DirectAlignerConfig cfg_;
    ThreadPool*         pool_;

    std::vector<cv::Mat>    depth_pyramid_;
    std::vector<AlignPoint> points_;             // 현재 레벨 점 집합
    std::vector<AlignPoint> thin_scratch_;       // max_points 초과 시 솎기용 (swap 대상)
    std::vector<Accumulator> blocks_;            // 결정적 병렬 누산용
    std::vector<std::size_t> points_per_level_;

    // 레벨별로 따로 들고 있어야 cv::resize 가 버퍼를 재사용한다. 한 장으로
    // 돌려쓰면 레벨마다 크기가 달라 매번 새로 잡는다.
    std::vector<cv::Mat>    quality_pyramid_;    // 레벨 크기로 리샘플된 품질 가중맵
    std::vector<cv::Mat>    mask_pyramid_;       // 레벨 크기로 리샘플된 정적 마스크
    bool                    has_quality_{false}; // 이번 호출에 가중맵이 있는가
    bool                    has_mask_{false};    // 이번 호출에 정적 마스크가 있는가

    // 점별 |잔차|. 블록마다 서로 겹치지 않는 구간만 쓰므로 락이 필요 없다.
    // 쓰이지 않은 점은 -1 로 남는다.
    mutable std::vector<float> abs_res_;
    mutable std::vector<float> res_scratch_;     // 중앙값 계산용 재사용 버퍼

    // 클러스터별 스코어와 표본 수. accumulateScores 가 채운다.
    mutable std::vector<Eigen::Matrix<double, 8, 1>> scores_;
    mutable std::vector<std::uint32_t>               score_count_;
};

}  // namespace wme
