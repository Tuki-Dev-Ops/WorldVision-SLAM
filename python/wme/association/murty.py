"""Murty 의 k-최적 할당.

MHT 의 필수 부품이다. 최적 할당 하나만으로는 다중 가설을 만들 수 없고,
차선·차차선 할당까지 있어야 "지금은 애매하니 판단을 미룬다"가 가능해진다.

원리: 최적해를 구한 뒤, 그 해의 각 쌍을 하나씩 금지한 부분문제로 공간을 분할한다.
분할된 부분문제의 최적해들 중 가장 좋은 것이 전체 2등이고, 재귀적으로 k등까지 나온다.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..reference.assignment import INFEASIBLE, solve_assignment


@dataclass(order=True)
class _Node:
    cost: float
    order: int                                        # 동점 시 결정적 순서
    solution: np.ndarray = field(compare=False)
    forced: tuple[tuple[int, int], ...] = field(compare=False, default=())
    forbidden: tuple[tuple[int, int], ...] = field(compare=False, default=())


def _constrained(cost: np.ndarray, forced, forbidden) -> np.ndarray:
    """강제/금지 제약을 비용행렬에 반영한다."""
    c = cost.copy()
    for i, j in forbidden:
        c[i, j] = INFEASIBLE
    for i, j in forced:
        # 강제 쌍: 해당 행/열의 다른 선택지를 모두 막는다
        keep = c[i, j]
        c[i, :] = INFEASIBLE
        c[:, j] = INFEASIBLE
        c[i, j] = keep
    return c


def _total(cost: np.ndarray, solution: np.ndarray) -> float:
    total = 0.0
    for i, j in enumerate(solution):
        if j >= 0:
            total += float(cost[i, j])
    return total


def murty_k_best(cost: np.ndarray, k: int = 5) -> list[tuple[np.ndarray, float]]:
    """비용 오름차순으로 최대 k 개의 할당을 반환한다.

    Returns:
        [(row_to_col, total_cost), ...]
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.size == 0 or k <= 0:
        return []

    r2c, _, total = solve_assignment(cost)
    if not np.any(r2c >= 0) and cost.shape[0] > 0:
        return [(r2c, 0.0)]

    counter = 0
    heap: list[_Node] = [_Node(total, counter, r2c)]
    results: list[tuple[np.ndarray, float]] = []
    seen: set[bytes] = set()

    while heap and len(results) < k:
        node = heapq.heappop(heap)

        key = node.solution.tobytes()
        if key in seen:
            continue
        seen.add(key)
        results.append((node.solution.copy(), node.cost))

        # 이 해의 쌍들을 앞에서부터 하나씩 금지하며 공간을 분할한다.
        # 앞선 쌍들은 강제로 고정해 부분공간이 서로 겹치지 않게 한다.
        forced = list(node.forced)
        for i, j in enumerate(node.solution):
            if j < 0:
                continue
            if (i, j) in node.forced:
                continue

            forbidden = node.forbidden + ((i, int(j)),)
            sub = _constrained(cost, forced, forbidden)
            sub_sol, _, _ = solve_assignment(sub)

            # 부모와 같은 수의 행이 배정된 해만 유효하다.
            # 미배정 행은 비용에 0 을 기여하므로, 이 검사가 없으면 부분해가
            # 더 싸 보여서 k-최적 순서가 깨진다.
            if np.sum(sub_sol >= 0) == np.sum(node.solution >= 0):
                counter += 1
                heapq.heappush(heap, _Node(_total(cost, sub_sol), counter,
                                           sub_sol, tuple(forced), forbidden))
            forced.append((i, int(j)))

    return results
