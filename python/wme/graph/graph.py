"""팩터그래프와 레벤버그-마쿼트 해결기.

이것이 "하나의 목적 함수" 질문에 대한 실제 답이다. 단일 스칼라를 통째로
최소화하는 게 아니라, 인수분해된 결합 사후분포를 다양체 위에서 푼다.
docs/04-unified-objective.md 5.1 참조.

밀집 행렬을 쓴다. 참조 구현의 목적은 정확성이지 속도가 아니고, 희소 Cholesky /
Schur 보수는 C++ 쪽 관심사다. 변수 수가 수천을 넘으면 느려진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .factors import Factor
from .variables import Variable


@dataclass
class OptimizationResult:
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    reason: str = ""

    @property
    def cost_reduction(self) -> float:
        if self.initial_cost <= 0.0:
            return 0.0
        return 1.0 - self.final_cost / self.initial_cost

    def summary(self) -> str:
        return (f"iters {self.iterations}  cost {self.initial_cost:.4g} -> "
                f"{self.final_cost:.4g} ({self.cost_reduction:.1%} 감소)  {self.reason}")


@dataclass
class SolverOptions:
    max_iterations: int = 50
    lambda_init: float = 1e-4
    lambda_up: float = 8.0
    lambda_down: float = 0.33
    lambda_max: float = 1e10
    step_tolerance: float = 1e-9
    cost_tolerance: float = 1e-10
    verbose: bool = False


class FactorGraph:
    def __init__(self):
        self._variables: dict[str, Variable] = {}
        self._factors: list[Factor] = []
        self._fixed: set[str] = set()

    # --- 구성 -------------------------------------------------------------

    def add_variable(self, key: str, variable: Variable) -> None:
        if key in self._variables:
            raise ValueError(f"변수 중복: {key}")
        self._variables[key] = variable

    def add_factor(self, factor: Factor) -> None:
        for k in factor.keys:
            if k not in self._variables:
                raise ValueError(f"팩터가 미등록 변수를 참조: {k}")
        self._factors.append(factor)

    def fix(self, key: str) -> None:
        """변수를 상수로 고정한다. 게이지 고정이나 참값 앵커에 쓴다."""
        if key not in self._variables:
            raise ValueError(f"미등록 변수: {key}")
        self._fixed.add(key)

    def value(self, key: str) -> Variable:
        return self._variables[key]

    @property
    def variables(self) -> dict[str, Variable]:
        return self._variables

    @property
    def factors(self) -> list[Factor]:
        return self._factors

    # --- 내부 -------------------------------------------------------------

    def _ordering(self) -> tuple[dict[str, int], int]:
        """자유 변수의 시작 인덱스와 전체 차원."""
        offsets, total = {}, 0
        for key, var in self._variables.items():
            if key in self._fixed:
                continue
            offsets[key] = total
            total += var.dim
        return offsets, total

    def cost(self, values: dict[str, Variable] | None = None) -> float:
        """가중 잔차 제곱합. 로버스트 커널은 IRLS 가중으로 반영한다."""
        values = values or self._variables
        total = 0.0
        for f in self._factors:
            r = f.residual(values)
            Lam = f.information()
            chi2 = float(r @ Lam @ r)
            if f.kernel is not None:
                chi2 *= f.kernel.weight(chi2)
            total += chi2
        return total

    def _build_system(self, values: dict[str, Variable], offsets: dict[str, int],
                      total: int) -> tuple[np.ndarray, np.ndarray]:
        H = np.zeros((total, total))
        b = np.zeros(total)

        for f in self._factors:
            r = f.residual(values)
            Lam = f.information()

            weight = 1.0
            if f.kernel is not None:
                weight = f.kernel.weight(float(r @ Lam @ r))
            if weight <= 0.0:
                continue
            W = Lam * weight

            J_all = f.jacobians(values)
            if J_all is None:
                J_all = f.numeric_jacobians(values)

            # 고정 변수는 시스템에서 빠지되 잔차에는 기여한다
            active = [(k, J) for k, J in zip(f.keys, J_all) if k not in self._fixed]

            for key_i, J_i in active:
                oi, di = offsets[key_i], J_i.shape[1]
                JtW = J_i.T @ W
                b[oi:oi + di] += JtW @ r
                for key_j, J_j in active:
                    oj, dj = offsets[key_j], J_j.shape[1]
                    H[oi:oi + di, oj:oj + dj] += JtW @ J_j
        return H, b

    # --- 최적화 -----------------------------------------------------------

    def optimize(self, options: SolverOptions | None = None) -> OptimizationResult:
        opts = options or SolverOptions()
        offsets, total = self._ordering()
        if total == 0:
            return OptimizationResult(True, 0, 0.0, 0.0, "자유 변수 없음")

        cost = self.cost()
        initial = cost
        lam = opts.lambda_init

        for it in range(1, opts.max_iterations + 1):
            H, b = self._build_system(self._variables, offsets, total)

            # 감쇠는 대각 스케일에 비례시킨다. 포즈(rad/m)와 위치(m)의
            # 단위가 달라 고정 감쇠를 쓰면 한쪽만 움직인다.
            #
            # 다만 하한을 전체 스케일에 비례시켜야 한다. 랭크 부족 팩터
            # (Tier 2 의 평면 제약이 대표적)만 붙은 방향은 diag(H) 가 0 이라
            # 절대 하한 1e-12 로는 사실상 감쇠가 걸리지 않고, 그 방향으로
            # 해가 무한정 날아간다. 실제로 SPA 단독 구성에서 ATE 가 32 m 까지
            # 발산했고 원인이 이것이었다.
            diag = np.diag(H)
            floor = max(float(diag.max()), 1.0) * 1e-9
            damped = H + lam * np.diag(np.maximum(diag, floor))
            try:
                delta = -np.linalg.solve(damped, b)
            except np.linalg.LinAlgError:
                lam *= opts.lambda_up
                if lam > opts.lambda_max:
                    return OptimizationResult(False, it, initial, cost, "H 특이 - 게이지 미고정?")
                continue

            if not np.all(np.isfinite(delta)):
                lam *= opts.lambda_up
                continue

            candidate = {k: (v.retract(delta[offsets[k]:offsets[k] + v.dim])
                             if k in offsets else v)
                         for k, v in self._variables.items()}
            new_cost = self.cost(candidate)

            if new_cost < cost:
                step = float(np.linalg.norm(delta))
                improvement = cost - new_cost
                self._variables = candidate
                cost = new_cost
                lam = max(1e-12, lam * opts.lambda_down)

                if opts.verbose:
                    print(f"  it {it:3d}  cost {cost:.6g}  |dx| {step:.3g}  lam {lam:.2g}")
                if step < opts.step_tolerance:
                    return OptimizationResult(True, it, initial, cost, "스텝 수렴")
                if improvement < opts.cost_tolerance * max(1.0, cost):
                    return OptimizationResult(True, it, initial, cost, "비용 수렴")
            else:
                lam *= opts.lambda_up
                if lam > opts.lambda_max:
                    return OptimizationResult(True, it, initial, cost, "감쇠 상한 - 국소최소")

        return OptimizationResult(False, opts.max_iterations, initial, cost, "반복 상한")

    # --- 불확실성 ---------------------------------------------------------

    def covariance(self, keys: list[str] | None = None) -> dict[str, np.ndarray]:
        """주변 공분산. H 를 역행렬 취해 해당 블록을 뽑는다.

        조건부가 아니라 주변 공분산이어야 한다. 다른 변수를 안다고 가정하면
        과신이 된다 - M3 에서 객체 치수를 주변화해야 했던 것과 같은 이유다.
        """
        offsets, total = self._ordering()
        if total == 0:
            return {}

        H, _ = self._build_system(self._variables, offsets, total)
        try:
            P = np.linalg.inv(H + np.eye(total) * 1e-12)
        except np.linalg.LinAlgError:
            raise ValueError("정보행렬이 특이하다 - 게이지가 고정되지 않았을 수 있다")

        out = {}
        for key in (keys if keys is not None else list(offsets)):
            if key not in offsets:
                continue
            o = offsets[key]
            d = self._variables[key].dim
            out[key] = P[o:o + d, o:o + d]
        return out

    # --- 진단 -------------------------------------------------------------

    def degeneracy(self) -> tuple[np.ndarray, int]:
        """정보행렬의 고유값 스펙트럼과 관측 가능한 자유도 수.

        ECDA 와 같은 원칙: 퇴화를 발견하는 게 아니라 보고한다.
        단위가 섞여 있으므로 대각 정규화 후 판정한다.
        """
        offsets, total = self._ordering()
        if total == 0:
            return np.zeros(0), 0

        H, _ = self._build_system(self._variables, offsets, total)
        scale = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-12))
        Hn = H * scale[:, None] * scale[None, :]

        eig = np.linalg.eigvalsh(Hn)[::-1]
        observable = int(np.sum(eig > max(eig[0], 1e-12) * 1e-6)) if eig.size else 0
        return eig, observable

    def __repr__(self) -> str:
        return (f"FactorGraph({len(self._variables)} vars, "
                f"{len(self._factors)} factors, {len(self._fixed)} fixed)")
