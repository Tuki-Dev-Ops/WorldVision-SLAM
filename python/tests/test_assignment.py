"""이분 할당 검증. C++ tests/test_assignment.cpp 와 같은 케이스."""

import itertools

import numpy as np

from wme.reference.assignment import INFEASIBLE, solve_assignment

RNG = np.random.default_rng(9001)


def test_empty_inputs():
    r2c, c2r, total = solve_assignment(np.zeros((0, 0)))
    assert len(r2c) == 0 and len(c2r) == 0 and total == 0.0


def test_identity_matrix_matches_diagonal():
    n = 5
    cost = np.ones((n, n))
    np.fill_diagonal(cost, 0.0)
    r2c, _, total = solve_assignment(cost)
    assert total == 0.0
    assert list(r2c) == list(range(n))


def test_beats_greedy_on_trap_case():
    """탐욕이 실패하는 고전 구조. 전역 최적은 5."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 100.0]])
    r2c, _, total = solve_assignment(cost)
    assert total == 5.0
    assert list(r2c) == [1, 0]


def test_rectangular_more_columns():
    cost = np.array([[9.0, 1.0, 9.0, 9.0],
                     [9.0, 9.0, 9.0, 2.0]])
    r2c, _, total = solve_assignment(cost)
    assert list(r2c) == [1, 3]
    assert total == 3.0


def test_rectangular_more_rows():
    cost = np.array([[5.0, 9.0],
                     [1.0, 9.0],
                     [9.0, 9.0],
                     [9.0, 2.0]])
    r2c, _, _ = solve_assignment(cost)
    assert r2c[1] == 0 and r2c[3] == 1
    assert r2c[0] == -1 and r2c[2] == -1


def test_infeasible_pairs_never_assigned():
    """게이트를 못 넘은 쌍은 다른 선택지가 없어도 미배정으로 남아야 한다."""
    cost = np.array([[INFEASIBLE, INFEASIBLE],
                     [INFEASIBLE, 1.0]])
    r2c, _, _ = solve_assignment(cost)
    assert r2c[0] == -1 and r2c[1] == 1


def test_all_infeasible():
    r2c, _, total = solve_assignment(np.full((3, 3), INFEASIBLE))
    assert all(c == -1 for c in r2c)
    assert total == 0.0


def test_matches_brute_force():
    for _ in range(150):
        n = 5
        cost = RNG.uniform(0, 10, (n, n))
        best = min(sum(cost[i, p[i]] for i in range(n))
                   for p in itertools.permutations(range(n)))
        _, _, total = solve_assignment(cost)
        assert np.isclose(total, best, atol=1e-9)


def test_row_and_column_maps_are_consistent():
    cost = RNG.uniform(0, 5, (7, 9))
    r2c, c2r, _ = solve_assignment(cost)
    for i, j in enumerate(r2c):
        if j >= 0:
            assert c2r[j] == i


def test_is_deterministic():
    cost = RNG.uniform(0, 5, (12, 15))
    a = solve_assignment(cost)
    b = solve_assignment(cost)
    assert list(a[0]) == list(b[0])
    assert np.isclose(a[2], b[2])
