#pragma once

// 깊이맵에서 평면을 뽑는다. Tier 2(SPA)의 재료다.
//
// 평면은 외관과 무관하므로 텍스처가 없거나 조명이 무너진 곳에서도 살아남는다 -
// 측광(Tier 0)이 죽는 바로 그 조건이다. 법선은 깊이 이웃의 외적으로 구한다.
// 학습 모델이 아니라 미분이므로 YOLO-only 제약과 무관하다
// (docs/02-correspondence-problem.md 1.1 의 그래디언트 논증과 같은 근거).
//
// 부호 규약이 이 파일에서 가장 사고가 잦은 지점이다. 평면은 Hesse 표준형
// n·p = d, d > 0 으로 저장하므로 **법선은 카메라 반대쪽을 향한다**. 표면 법선
// (카메라 쪽)과 반대라, 둘을 섞으면 모든 평면이 걸러져 추출 결과가 0 이 된다.
// Python 참조 구현에서 실제로 발생했던 실패이며, 여기서는 unit test 로 못박는다.
//
// 한계: 축정렬 격자 클러스터링이라 곡면이나 아주 얇은 구조는 못 잡는다.
// 방/복도의 벽·바닥·천장을 잡는 것이 목적이다.
//
// 스레드 안전성: 인스턴스 비공유. 내부 버퍼를 프레임 간 재사용한다.

#include "wme/core/Result.hpp"
#include "wme/core/Types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace wme {

// 카메라 좌표계의 평면. n·p = d, |n| = 1, d > 0 (카메라 앞).
struct Plane {
    Vec3   normal{Vec3::UnitZ()};
    double distance{0.0};
    Vec3   centroid{Vec3::Zero()};
    double extent{0.0};          // 평면 위 점들의 평균 산포 (m). 넓을수록 회전 구속이 강하다
    double rms{0.0};             // 적합 잔차 (m)
    std::size_t inliers{0};      // 서브샘플 격자 기준 점 수

    // 넓고 잘 맞는 평면일수록 믿을 만하다. 대응 가중치로 그대로 쓰인다.
    [[nodiscard]] double confidence() const;

    [[nodiscard]] double signedDistance(const Vec3& p) const {
        return p.dot(normal) - distance;
    }

    // 다른 좌표계로 옮긴다. n' = R n, d' = d + n'·t.
    [[nodiscard]] Plane transformed(const Mat3& R, const Vec3& t) const;
};

struct PlaneExtractorConfig {
    int    stride{4};                  // 깊이맵 서브샘플링. 법선은 이 격자에서 계산된다
    double min_depth{0.2};
    double max_depth{15.0};

    int    normal_bins{12};            // 법선 방향 격자 해상도
    double distance_bin{0.15};         // m
    std::size_t min_inliers{150};      // 서브샘플 격자 기준

    double refine_threshold{0.04};     // m, 재적합 시 인라이어 판정
    int    refine_iterations{2};
    std::size_t max_planes{8};

    // 이웃 깊이 차이가 이보다 크면 경계로 보고 법선을 만들지 않는다.
    // 이게 없으면 물체 실루엣에서 비스듬한 가짜 평면이 생긴다.
    double depth_discontinuity{0.15};

    // 최소 특이값이 중간 특이값 대비 이 비율을 넘으면 평면이 아니다 (구·모서리).
    double planarity_ratio{0.3};
};

// 법선 추정 결과. 진단과 단위 검증용으로 노출한다 - 법선이 틀리면 그 아래
// 전부가 조용히 틀리므로, 중간 산출물을 볼 수 없으면 검증이 불가능하다.
struct NormalField {
    int rows{0}, cols{0};                  // 서브샘플 격자 - 1
    std::vector<Vec3>  normal;             // Hesse 규약 (카메라 반대쪽)
    std::vector<Vec3>  point;              // 대응 3D 점 (카메라 좌표)
    std::vector<std::uint8_t> valid;       // 0 = 불연속/무효

    [[nodiscard]] std::size_t size() const { return normal.size(); }
    [[nodiscard]] std::size_t validCount() const;
};

class PlaneExtractor {
public:
    explicit PlaneExtractor(PlaneExtractorConfig config = {});

    // 깊이맵(CV_32F, 미터)에서 지배 평면들을 신뢰도 내림차순으로 뽑는다.
    // 유효 깊이가 전혀 없으면 InsufficientData. 평면이 없으면 빈 벡터(성공)다 -
    // "깊이가 없다" 와 "평면이 없다" 는 다른 사실이라 구분해서 보고한다.
    [[nodiscard]] Result<std::vector<Plane>> extract(const cv::Mat& depth,
                                                     const CameraIntrinsics& K);

    // 법선장만 계산한다. extract() 가 내부적으로 쓰는 것과 동일한 경로다.
    [[nodiscard]] Result<NormalField> estimateNormals(const cv::Mat& depth,
                                                      const CameraIntrinsics& K);

    [[nodiscard]] const PlaneExtractorConfig& config() const { return cfg_; }

private:
    // 유효 격자점을 (점, 법선, 거리) 로 평탄화한다. 반환은 점 개수.
    std::size_t gather(const NormalField& field);

    // 주어진 인덱스 집합에 SVD 최소제곱 평면 적합. 평면이 아니면 false.
    [[nodiscard]] bool fit(const std::vector<std::uint32_t>& idx, Plane& out) const;

    PlaneExtractorConfig cfg_;

    NormalField        field_;
    std::vector<Vec3>  points_;      // 유효 점만 모은 것
    std::vector<Vec3>  normals_;
    std::vector<double> dists_;
    std::vector<std::uint8_t> used_;
    std::vector<std::uint32_t> scratch_idx_;
};

// 가장 넓은 수평면의 법선을 중력 방향으로 삼는다.
// IMU 가 없을 때 ConstellationIndex 의 카이랄리티가 이 값을 필요로 한다.
// 수평면이 없으면 NotAvailable - 지어내지 않는다.
// 반환은 카메라 좌표계에서 '아래'(+y) 를 가리키는 단위벡터.
[[nodiscard]] Result<Vec3> dominantGravity(const std::vector<Plane>& planes,
                                           double min_confidence = 0.2,
                                           double min_verticality = 0.85);

}  // namespace wme
