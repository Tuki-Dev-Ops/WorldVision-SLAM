"""카이제곱 분포 유틸.

scipy 없이 동작해야 한다. 평가 코드가 무거운 의존성을 끌고 오면 CI 에서
돌지 않게 되고, 돌지 않는 평가는 없는 것과 같다.

정칙 불완전감마함수를 급수/연분수로 계산한다 (Numerical Recipes 방식).
"""

from __future__ import annotations

import math

_MAX_ITER = 300
_EPS = 3e-14
_TINY = 1e-300


def _gamma_p_series(a: float, x: float) -> float:
    """급수 전개. x < a + 1 에서 빠르게 수렴한다."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_continued_fraction(a: float, x: float) -> float:
    """연분수 전개(수정 Lentz). x >= a + 1 에서 사용."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gammainc_lower(a: float, x: float) -> float:
    """정칙 하부 불완전감마 P(a, x). 0 <= P <= 1."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("a > 0, x >= 0 이어야 함")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_p_series(a, x)
    return 1.0 - _gamma_q_continued_fraction(a, x)


def chi2_cdf(x: float, dof: float) -> float:
    """자유도 dof 카이제곱 분포의 누적분포함수."""
    if x <= 0.0:
        return 0.0
    return gammainc_lower(dof * 0.5, x * 0.5)


def chi2_ppf(p: float, dof: float) -> float:
    """분위수. 이분법으로 역함수를 구한다 - 안정적이고 충분히 빠르다."""
    if not 0.0 < p < 1.0:
        raise ValueError("0 < p < 1 이어야 함")

    # 상한을 지수적으로 늘려 해를 감싼다
    lo, hi = 0.0, max(1.0, dof)
    while chi2_cdf(hi, dof) < p:
        hi *= 2.0
        if hi > 1e12:
            return hi

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def chi2_interval(dof: float, confidence: float = 0.95) -> tuple[float, float]:
    """양측 신뢰구간 [lo, hi]."""
    alpha = 1.0 - confidence
    return chi2_ppf(alpha * 0.5, dof), chi2_ppf(1.0 - alpha * 0.5, dof)
