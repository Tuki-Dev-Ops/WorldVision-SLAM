"""의존성 없는 무도함수 최적화.

scipy 를 쓰지 않는다 - 평가/보정 코드가 무거운 의존성을 끌면 CI 에서 돌지 않고,
돌지 않는 실험은 없는 것과 같다. Nelder-Mead 면 파라미터 4~6개짜리 잡음모델
적합에는 충분하다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def minimize_nelder_mead(fn: Callable[[np.ndarray], float], x0: np.ndarray,
                         step: float = 0.25, max_iter: int = 2000,
                         tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Nelder-Mead 단체법.

    Args:
        fn: 최소화할 함수. 발산하면 inf 를 돌려주면 된다.
        x0: 초기값 (보통 로그공간 파라미터)
        step: 초기 단체 크기

    Returns:
        (최적 파라미터, 최적 함수값)
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size

    # 초기 단체: x0 와 각 축으로 step 만큼 이동한 점들
    simplex = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    values = np.array([fn(p) for p in simplex])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]

        # 수렴: 최선/최악 함수값 차이가 무시할 수준
        spread = abs(values[-1] - values[0])
        if spread < tol * (abs(values[0]) + tol):
            break

        centroid = simplex[:-1].mean(axis=0)

        reflected = centroid + alpha * (centroid - simplex[-1])
        f_r = fn(reflected)

        if values[0] <= f_r < values[-2]:
            simplex[-1], values[-1] = reflected, f_r
            continue

        if f_r < values[0]:
            expanded = centroid + gamma * (reflected - centroid)
            f_e = fn(expanded)
            if f_e < f_r:
                simplex[-1], values[-1] = expanded, f_e
            else:
                simplex[-1], values[-1] = reflected, f_r
            continue

        contracted = centroid + rho * (simplex[-1] - centroid)
        f_c = fn(contracted)
        if f_c < values[-1]:
            simplex[-1], values[-1] = contracted, f_c
            continue

        # 축소: 최선점 쪽으로 전체를 당긴다
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
        values[1:] = np.array([fn(p) for p in simplex[1:]])

    best = int(np.argmin(values))
    return simplex[best], float(values[best])
