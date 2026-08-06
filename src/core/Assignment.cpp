#include "wme/core/Assignment.hpp"

#include <algorithm>
#include <cmath>

namespace wme {
namespace {

// 최단경로 증강 방식 헝가리안.
// 내부적으로 정사각으로 패딩하지 않고, 행마다 한 번씩 증강경로를 찾는다.
// 잠재값(u, v)을 유지해 각 증강이 O(n*m) 이 되도록 한다.
struct Solver {
    std::size_t n, m;                 // n = rows, m = cols (n <= m 가정)
    const std::vector<double>& c;
    bool transposed;

    std::vector<double> u, v;         // 쌍대 잠재값
    std::vector<int>    match_col;    // 열 -> 행
    std::vector<int>    way;          // 교대 경로 복원용

    Solver(std::size_t rows, std::size_t cols, const std::vector<double>& cost, bool t)
        : n(rows), m(cols), c(cost), transposed(t),
          u(rows + 1, 0.0), v(cols + 1, 0.0),
          match_col(cols + 1, -1), way(cols + 1, 0) {}

    [[nodiscard]] double at(std::size_t row, std::size_t col) const {
        return transposed ? c[col * n + row] : c[row * m + col];
    }

    void run() {
        constexpr double kInf = std::numeric_limits<double>::infinity();

        // 행마다 새로 잡던 두 버퍼를 밖으로 뺀다. 행 시작마다 값만 다시 채우므로
        // 계산은 그대로이고 할당만 n 회에서 1 회로 줄어든다.
        std::vector<double> minv(m + 1, kInf);
        std::vector<char>   used(m + 1, 0);

        for (std::size_t i = 1; i <= n; ++i) {
            match_col[0] = static_cast<int>(i);
            std::size_t j0 = 0;

            std::fill(minv.begin(), minv.end(), kInf);
            std::fill(used.begin(), used.end(), static_cast<char>(0));

            do {
                used[j0] = 1;
                const auto i0 = static_cast<std::size_t>(match_col[j0]);
                double     delta = kInf;
                std::size_t j1 = 0;

                for (std::size_t j = 1; j <= m; ++j) {
                    if (used[j]) continue;
                    const double cur = at(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = static_cast<int>(j0); }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                if (!std::isfinite(delta)) break;   // 도달 가능한 열이 없다

                for (std::size_t j = 0; j <= m; ++j) {
                    if (used[j]) {
                        u[static_cast<std::size_t>(match_col[j])] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (match_col[j0] != -1);

            // 증강경로를 따라 매칭을 뒤집는다
            while (j0 != 0) {
                const auto j1 = static_cast<std::size_t>(way[j0]);
                match_col[j0] = match_col[j1];
                j0 = j1;
            }
        }
    }
};

}  // namespace

AssignmentResult solveAssignment(const std::vector<double>& cost,
                                 std::size_t rows, std::size_t cols) {
    AssignmentResult out;
    out.row_to_col.assign(rows, -1);
    out.col_to_row.assign(cols, -1);
    if (rows == 0 || cols == 0 || cost.size() != rows * cols) return out;

    // 알고리즘은 행 수 <= 열 수를 가정하므로 필요하면 전치해서 푼다
    const bool transposed = rows > cols;
    Solver solver(transposed ? cols : rows, transposed ? rows : cols, cost, transposed);
    solver.run();

    for (std::size_t j = 1; j <= solver.m; ++j) {
        const int i = solver.match_col[j];
        if (i <= 0) continue;

        const std::size_t r = transposed ? (j - 1) : static_cast<std::size_t>(i - 1);
        const std::size_t k = transposed ? static_cast<std::size_t>(i - 1) : (j - 1);
        if (r >= rows || k >= cols) continue;

        // 금지 쌍이 억지로 배정됐다면 버린다.
        // 큰 비용으로 막아도 다른 선택지가 없으면 알고리즘은 그 쌍을 고른다.
        const double pair_cost = cost[r * cols + k];
        if (pair_cost >= kInfeasible) continue;

        out.row_to_col[r] = static_cast<int>(k);
        out.col_to_row[k] = static_cast<int>(r);
        out.total_cost += pair_cost;
        ++out.matched;
    }
    return out;
}

}  // namespace wme
