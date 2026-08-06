"""M3 게이트 실험: Sigma_k(E) 를 적합할 수 있는가, 그 결과가 보정되는가.

docs/05-research-program.md 5장의 M3. 여기서 실패하면 "하나의 사후분포" 논지가
근거를 잃고, 프로그램은 TCG 기여로 좁혀야 한다.

프로토콜
  훈련 조건에서 잡음모델을 적합하고, **보지 않은 조건**에서 평가한다.
  같은 조건에서 평가하면 아무것도 증명되지 않는다.

비교 대상 (전부 같은 추정기, 잡음모델만 교체)
  fixed      - 맑은 조건 적합값 고정 (현행 SLAM 관행)
  scheduled  - 손수 만든 부풀림 스케줄 (현행 WME)
  calibrated - sigma(z, E) 를 데이터에서 적합 (제안)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.metrics import NeesResult, nees
from ..sim.scenarios import Sequence, constant_condition, run, static_room
from ..sim.world import CameraTrajectory
from .estimator import collect_samples, estimate_objects, object_level_residuals
from .noise import (
    CalibratedNoise, FixedNoise, NoiseModel, ScheduledNoise,
    fit_calibrated, fit_fixed, fit_systematic,
)
from .shape import (
    estimate_objects_jointly, object_level_residuals_joint, refit_noise_under_joint_model,
)


@dataclass
class ConditionResult:
    severity: float
    level: float
    nees: NeesResult
    rmse: float
    bias_norm: float
    objects: int
    # 공통 편향을 제거한 뒤의 NEES. 이 값이 일관적인데 위가 아니라면,
    # 남은 과신은 공분산 오설정이 아니라 관측 모델의 *편향* 때문이다.
    # 편향은 공분산을 키워 덮을 수 없다 - 모델을 고쳐야 한다.
    debiased_nees: NeesResult | None = None

    @property
    def normalised_anees(self) -> float:
        return self.nees.anees / self.nees.dof

    @property
    def debiased_anees(self) -> float:
        if self.debiased_nees is None:
            return float("nan")
        return self.debiased_nees.anees / self.debiased_nees.dof


@dataclass
class ModelResult:
    name: str
    model: NoiseModel
    per_condition: list[ConditionResult] = field(default_factory=list)

    @property
    def consistent_fraction(self) -> float:
        if not self.per_condition:
            return 0.0
        return float(np.mean([c.nees.consistent for c in self.per_condition]))

    @property
    def worst_anees(self) -> float:
        return max((c.normalised_anees for c in self.per_condition), default=float("nan"))

    @property
    def mean_rmse(self) -> float:
        if not self.per_condition:
            return float("nan")
        return float(np.mean([c.rmse for c in self.per_condition]))

    def table(self) -> list[str]:
        rows = [f"  {'level':>6} {'sev':>6} {'ANEES/d':>9} {'debias':>8} {'band':>14} "
                f"{'rmse':>7} {'bias':>7}  verdict"]
        for c in self.per_condition:
            band = f"[{c.nees.lower:.2f},{c.nees.upper:.2f}]"
            verdict = ("ok" if c.nees.consistent
                       else ("OVERCONF" if c.nees.overconfident else "underconf"))
            rows.append(f"  {c.level:6.2f} {c.severity:6.3f} {c.normalised_anees:9.3f} "
                        f"{c.debiased_anees:8.3f} {band:>14} {c.rmse:7.4f} "
                        f"{c.bias_norm:7.4f}  {verdict}")
        return rows


@dataclass
class M3Report:
    channel: str
    train_levels: list[float]
    test_levels: list[float]
    models: list[ModelResult] = field(default_factory=list)
    truth_note: str = ""
    stage1: CalibratedNoise | None = None
    joint_systematic: CalibratedNoise | None = None

    # 정확도 기준: 최선 대비 이 배수를 넘으면 실격
    accuracy_tolerance: float = 1.5

    def best_rmse(self) -> float:
        vals = [m.mean_rmse for m in self.models if np.isfinite(m.mean_rmse)]
        return min(vals) if vals else float("nan")

    def passing_models(self) -> list[ModelResult]:
        """일관성 *그리고* 정확도를 함께 만족하는 구성.

        일관성만 요구하면 게이트가 뚫린다 - 공분산을 한없이 키우면 어떤
        엉터리 추정기도 '일관적'이 된다. 실제로 이 실험에서 그 일이 일어났고
        (RMSE 4.2 m 짜리 추정기가 100% 일관), 그래서 정확도 조건을 넣었다.
        """
        limit = self.best_rmse() * self.accuracy_tolerance
        return [m for m in self.models
                if m.consistent_fraction == 1.0 and m.mean_rmse <= limit]

    def passed(self) -> bool:
        """느슨하게 바꾸지 않는다. 통과 못 하면 통과 못 한 것이다."""
        return len(self.passing_models()) > 0

    def by_name(self, name: str) -> ModelResult | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def lines(self) -> list[str]:
        out = [
            "=" * 78,
            f"M3 게이트: 잡음모델 보정 ({self.channel})",
            "=" * 78,
            f"훈련 조건: {self.train_levels}",
            f"평가 조건: {self.test_levels}   <- 훈련에서 보지 못한 값",
            "",
        ]
        for m in self.models:
            out.append(f"[{m.name}]  {m.model}")
            out.extend(m.table())
            out.append(f"  -> 일관 비율 {m.consistent_fraction:.0%}, "
                       f"최악 ANEES/d {m.worst_anees:.2f}, "
                       f"평균 RMSE {m.mean_rmse:.4f} m")
            out.append("")
        if self.truth_note:
            out.append(self.truth_note)
            out.append("")

        out.append(f"게이트 = 일관성 100% AND 평균 RMSE <= 최선의 "
                   f"{self.accuracy_tolerance:g}배 ({self.best_rmse():.4f} m 기준)")
        winners = ", ".join(m.name for m in self.passing_models()) or "없음"
        out.append(f"통과 구성: {winners}")
        out.append(f"게이트 통과: {'YES' if self.passed() else 'NO'}")
        out.append("=" * 78)
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


def _make_sequence(channel: str, level: float, seed: int, n_frames: int,
                   laps: float) -> Sequence:
    """조건만 다르고 세계와 궤적은 동일한 시퀀스."""
    world = static_room(14, seed=seed)
    traj = CameraTrajectory.loop(radius=3.2, laps=laps, n=n_frames)
    return run(world, traj, constant_condition(**{channel: level}),
               seed=seed, name=f"{channel}={level:.2f}")


def _evaluate(model: NoiseModel, sequences: list[tuple[float, Sequence]],
              min_obs: int, joint_extent: bool = False) -> list[ConditionResult]:
    estimate = estimate_objects_jointly if joint_extent else estimate_objects
    out = []
    for level, seq in sequences:
        result = estimate(seq, model)
        try:
            errors, covs = result.nees_inputs(min_obs)
        except ValueError:
            continue
        # 공통 편향 제거: 편향이 원인인지 공분산이 원인인지 가른다
        debiased = errors - errors.mean(axis=0)
        out.append(ConditionResult(
            severity=seq.severity,
            level=level,
            nees=nees(errors, covs),
            rmse=float(np.sqrt((np.linalg.norm(errors, axis=1) ** 2).mean())),
            bias_norm=float(np.linalg.norm(errors.mean(axis=0))),
            objects=len(errors),
            debiased_nees=nees(debiased, covs),
        ))
    return out


def run_m3(channel: str = "haze",
           train_levels: list[float] | None = None,
           test_levels: list[float] | None = None,
           seed: int = 0, n_frames: int = 240, laps: float = 2.0,
           min_observations: int = 8, include_joint: bool = True) -> M3Report:
    """M3 실험을 실행한다.

    laps 를 2 로 두는 이유: 객체당 관측 수가 충분해야 NEES 가 의미를 갖는다.
    관측이 적으면 공분산이 커서 어떤 모델이든 일관해 보인다.
    """
    train_levels = train_levels if train_levels is not None else [0.0, 0.3, 0.6, 0.9]
    test_levels = test_levels if test_levels is not None else [0.15, 0.45, 0.75, 1.0]

    train_seqs = [(lv, _make_sequence(channel, lv, seed, n_frames, laps))
                  for lv in train_levels]
    test_seqs = [(lv, _make_sequence(channel, lv, seed + 100, n_frames, laps))
                 for lv in test_levels]

    cam = train_seqs[0][1].camera

    # 1) fixed - 맑은 조건에서만 적합한다. 현장에서 실제로 하는 방식.
    clear_samples = collect_samples(train_seqs[0][1], cam)
    fixed = fit_fixed(clear_samples, cam)

    # 2) scheduled - 같은 기준값에 손수 만든 부풀림
    scheduled = ScheduledNoise(base=fixed, inflate_power=1.0)

    # 3) calibrated - 2단계 적합
    #    1단계: 관측 단위 잔차에서 sigma(z, E)
    #    2단계: 객체 단위 잔차에서 계통 성분 sigma_sys(E)
    #    2단계가 없으면 융합할수록 과신한다. 1단계 잔차만으로는 계통 성분이
    #    독립 잡음과 구분되지 않아 보이지 않는다.
    all_samples = [s for _, seq in train_seqs for s in collect_samples(seq, cam)]
    stage1 = fit_calibrated(all_samples, cam, condition_dependent=True)

    object_level = [r for _, seq in train_seqs
                    for r in object_level_residuals(seq, stage1, min_observations)]
    calibrated = fit_systematic(stage1, object_level, cam, condition_dependent=True)

    report = M3Report(channel, train_levels, test_levels)
    report.stage1 = stage1
    for name, model in [("fixed", fixed), ("scheduled", scheduled),
                        ("calibrated (obs-only)", stage1),
                        ("calibrated + sys", calibrated)]:
        report.models.append(
            ModelResult(name, model, _evaluate(model, test_seqs, min_observations)))

    # 관측 모델 자체를 고친 경우: 크기를 상태에 넣고 주변화한다.
    # 계통 성분을 예산에 넣는 대신 원인을 제거하는 접근이라, 계통 항 없이
    # 1단계 모델만으로 일관해야 한다. 그게 안 되면 원인이 더 남아 있다는 뜻.
    if include_joint:
        # 잡음모델은 그것이 쓰일 관측 모델과 함께 보정해야 한다.
        # 편향된 모델로 적합한 stage1 을 그대로 쓰면 이번엔 과대평가가 된다.
        joint_noise = refit_noise_under_joint_model(
            [seq for _, seq in train_seqs], stage1, cam)

        joint_res = [r for _, seq in train_seqs
                     for r in object_level_residuals_joint(seq, joint_noise, min_observations)]
        joint_calibrated = fit_systematic(joint_noise, joint_res, cam,
                                          condition_dependent=True)

        report.models.append(ModelResult(
            "joint extent (stage1 noise)", stage1,
            _evaluate(stage1, test_seqs, min_observations, joint_extent=True)))
        report.models.append(ModelResult(
            "joint extent + refit noise", joint_calibrated,
            _evaluate(joint_calibrated, test_seqs, min_observations, joint_extent=True)))
        report.joint_systematic = joint_calibrated

    # 시뮬레이터의 참 파라미터와 비교해 적합이 실제로 복원했는지 본다
    from ..sim.sensor import SensorConfig
    cfg = SensorConfig()
    report.truth_note = (
        f"시뮬레이터 참 파라미터: c_d={cfg.depth_noise_coeff:.4f}, "
        f"g_d={cfg.depth_noise_growth:.2f}, c_px={cfg.base_box_noise_px:.2f}, "
        f"g_px={cfg.box_noise_growth:.2f}\n"
        f"1단계 적합:  c_d={stage1.c_d:.4f}, g_d={stage1.g_d:.2f}, "
        f"c_px={stage1.c_px:.2f}, g_px={stage1.g_px:.2f}\n"
        f"2단계 적합:  s_sys={calibrated.s_sys:.4f} m, k_geom={calibrated.k_geom:.4f}, "
        f"g_sys={calibrated.g_sys:.2f}\n"
        "c_px 가 참값보다 큰 것은 정상이다 - 박스 네 변의 잡음이 중심으로 전파되고,\n"
        "박스 중심이 무게중심의 투영과 정확히 일치하지 않는 모델링 오차도 흡수한다.\n"
        "sigma_sys 는 시뮬레이터에 없는 파라미터다. 순수하게 관측 모델의 불완전성에서\n"
        "나오며, 관측 단위 잔차만 봐서는 존재 자체가 보이지 않는다.")
    return report
