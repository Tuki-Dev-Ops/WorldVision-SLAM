#pragma once

// 최적 이분 할당 (Jonker-Volgenant 계열 헝가리안, O(n^2 m)).
//
// WME 에서 데이터 연관은 기술자 매칭이 아니라 비용 최소화 문제다. 비용은
// 3D 거리 / 클래스 일치 / 박스 겹침 / 예측 오차의 결합이며, 탐욕적 매칭은
// 군집한 동일 클래스 객체(의자 6개가 붙어 있는 회의실)에서 정체성을 섞는다.
// 전역 최적 할당이 그 실패를 없앤다.
//
// 비용이 kInfeasible 이상인 쌍은 금지된 할당으로 취급한다.

#include <limits>
#include <vector>

namespace wme {

inline constexpr double kInfeasible = 1e18;

struct AssignmentResult {
    // row_to_col[i] = i번 행에 배정된 열, 미배정이면 -1
    std::vector<int> row_to_col;
    std::vector<int> col_to_row;
    double           total_cost{0.0};
    std::size_t      matched{0};
};

// cost 는 rows x cols 의 행 우선 밀집 행렬.
// 직사각 행렬을 지원하며 남는 행/열은 미배정(-1)으로 남는다.
[[nodiscard]] AssignmentResult solveAssignment(const std::vector<double>& cost,
                                               std::size_t rows, std::size_t cols);

}  // namespace wme
