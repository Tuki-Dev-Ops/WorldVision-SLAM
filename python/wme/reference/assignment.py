"""전역 최적 이분 할당 참조 구현.

C++ src/core/Assignment.cpp 와 같은 의미를 가져야 한다:
  - 직사각 행렬 지원, 남는 행/열은 -1
  - 비용이 INFEASIBLE 이상인 쌍은 절대 배정하지 않는다
    (게이트를 통과 못한 연관은 억지로 맺지 말고 새 토큰을 만들어야 한다)
"""

from __future__ import annotations

import numpy as np

INFEASIBLE = 1e18


def solve_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """헝가리안(최단경로 증강).

    Returns:
        row_to_col: (rows,) 배정된 열 인덱스, 미배정은 -1
        col_to_row: (cols,) 배정된 행 인덱스, 미배정은 -1
        total_cost: 배정된 쌍의 비용 합
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.size == 0:
        rows = cost.shape[0] if cost.ndim == 2 else 0
        cols = cost.shape[1] if cost.ndim == 2 else 0
        return np.full(rows, -1, dtype=int), np.full(cols, -1, dtype=int), 0.0

    rows, cols = cost.shape
    transposed = rows > cols
    C = cost.T if transposed else cost
    n, m = C.shape                      # n <= m 보장

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    match_col = np.full(m + 1, -1, dtype=int)   # 열 -> 행
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        match_col[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = match_col[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not np.isfinite(delta):
                break

            for j in range(m + 1):
                if used[j]:
                    u[match_col[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match_col[j0] == -1:
                break

        # 증강경로를 따라 매칭을 뒤집는다
        while j0 != 0:
            j1 = way[j0]
            match_col[j0] = match_col[j1]
            j0 = j1

    row_to_col = np.full(rows, -1, dtype=int)
    col_to_row = np.full(cols, -1, dtype=int)
    total = 0.0

    for j in range(1, m + 1):
        i = match_col[j]
        if i <= 0:
            continue
        r = (j - 1) if transposed else (i - 1)
        k = (i - 1) if transposed else (j - 1)
        if r >= rows or k >= cols:
            continue
        # 금지 쌍이 억지로 배정됐다면 버린다
        if cost[r, k] >= INFEASIBLE:
            continue
        row_to_col[r] = k
        col_to_row[k] = r
        total += float(cost[r, k])

    return row_to_col, col_to_row, total
