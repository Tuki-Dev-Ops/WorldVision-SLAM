"""월드 모델 SLAM 평가 지표.

ATE/RPE 는 WME 가 개선한다고 주장하는 것을 하나도 측정하지 못한다. 여기 있는
지표들이 실제 주장에 대응한다.

  불확실성 보정  - NEES / ANEES / ECE      <- 가장 중요하고, 대부분의 시스템이 실패한다
  열화 곡선      - 조건 강도별 성능
  정체성         - IDS, 중복 객체율
  변화 이해      - 이동/제거/추가 검출과 지연
  예측           - 시계별 오차 + 예측 NLL

설계 원칙: 단일 스칼라를 내지 않는다. "안개 0.8 에서 ATE 0.3 m" 는
"ATE 0.12 m" 보다 훨씬 많은 것을 말해준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stats import chi2_interval


# ===========================================================================
# 불확실성 보정
# ===========================================================================

@dataclass
class NeesResult:
    per_sample: np.ndarray        # 표본별 NEES
    anees: float                  # 평균 NEES (자유도로 나누지 않은 값)
    dof: int
    lower: float                  # 일관성 구간 하한 (자유도로 정규화)
    upper: float
    consistent: bool
    overconfident: bool           # anees/dof > upper -> 공분산이 실제보다 작다

    def summary(self) -> str:
        verdict = ("consistent" if self.consistent
                   else ("OVERCONFIDENT" if self.overconfident else "underconfident"))
        return (f"ANEES {self.anees / self.dof:.3f} (dof={self.dof}, "
                f"95% band [{self.lower:.3f}, {self.upper:.3f}]) -> {verdict}")


def nees(errors: np.ndarray, covariances: np.ndarray,
         confidence: float = 0.95) -> NeesResult:
    """Normalised Estimation Error Squared.

    NEES_i = e_i^T P_i^-1 e_i 이고, 추정기가 일관되면 E[NEES] = dof 이다.
    ANEES/dof 가 1 보다 크게 벗어나면 공분산이 과소평가된 것 - 즉 과신이다.
    계획기가 이 값을 믿고 움직이면 위험해지므로, ATE 보다 배포에 더 중요하다.

    Args:
        errors: (N, d) 추정오차 (참값 - 추정값)
        covariances: (N, d, d) 보고된 공분산
    """
    errors = np.asarray(errors, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if errors.ndim != 2:
        raise ValueError("errors 는 (N, d) 여야 함")
    n, d = errors.shape
    if covariances.shape != (n, d, d):
        raise ValueError(f"covariances 는 ({n}, {d}, {d}) 여야 함")
    if n == 0:
        raise ValueError("표본이 없음")

    per_sample = np.empty(n)
    for i in range(n):
        # 수치적으로 특이한 공분산은 조용히 통과시키면 안 된다
        P = covariances[i] + np.eye(d) * 1e-12
        per_sample[i] = float(errors[i] @ np.linalg.solve(P, errors[i]))

    anees = float(per_sample.mean())

    # N*ANEES ~ chi2(N*d) 이므로 구간을 N*d 로 정규화한다
    lo, hi = chi2_interval(n * d, confidence)
    lower, upper = lo / (n * d), hi / (n * d)
    normalised = anees / d

    return NeesResult(
        per_sample=per_sample,
        anees=anees,
        dof=d,
        lower=lower,
        upper=upper,
        consistent=lower <= normalised <= upper,
        overconfident=normalised > upper,
    )


@dataclass
class CalibrationResult:
    ece: float                    # Expected Calibration Error
    mce: float                    # Maximum Calibration Error
    bin_confidence: np.ndarray    # 구간별 평균 예측 확률
    bin_accuracy: np.ndarray      # 구간별 실제 빈도
    bin_count: np.ndarray
    brier: float

    def summary(self) -> str:
        return f"ECE {self.ece:.4f}  MCE {self.mce:.4f}  Brier {self.brier:.4f}"


def calibration(probabilities: np.ndarray, outcomes: np.ndarray,
                bins: int = 10) -> CalibrationResult:
    """이진 믿음의 보정도.

    existence_belief 0.9 를 준 객체들 중 실제로 90% 가 존재해야 한다.
    그렇지 않으면 그 확률은 확률이 아니라 점수다.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(outcomes, dtype=float).ravel()
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities 와 outcomes 크기가 맞지 않음")

    edges = np.linspace(0.0, 1.0, bins + 1)
    conf = np.zeros(bins)
    acc = np.zeros(bins)
    cnt = np.zeros(bins, dtype=int)

    # 마지막 구간은 오른쪽 끝을 포함해야 p=1.0 이 버려지지 않는다
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b] > 0:
            conf[b] = float(p[m].mean())
            acc[b] = float(y[m].mean())

    w = cnt / max(1, cnt.sum())
    gaps = np.abs(acc - conf)
    return CalibrationResult(
        ece=float(np.sum(w * gaps)),
        mce=float(gaps[cnt > 0].max()) if np.any(cnt > 0) else 0.0,
        bin_confidence=conf,
        bin_accuracy=acc,
        bin_count=cnt,
        brier=float(np.mean((p - y) ** 2)),
    )


# ===========================================================================
# 열화 곡선
# ===========================================================================

@dataclass
class DegradationCurve:
    severity: np.ndarray          # 조건 강도 0..1
    value: np.ndarray             # 해당 강도에서의 성능 지표
    failed: np.ndarray            # 치명적 실패(추적 상실 후 미복구) 여부
    metric_name: str = "ate_rmse"

    # 치명적 실패 구간에 대입할 절대 벌점.
    # 시스템 자신의 값에서 유도하면 안 된다 - 잘 하다가 무너지는 시스템이
    # 꾸준히 버티는 시스템보다 좋아 보이게 된다. 비교하려는 모든 시스템이
    # 같은 값을 써야 한다.
    failure_value: float = 1.0

    @property
    def failure_severity(self) -> float:
        """처음으로 치명적 실패가 난 강도. 없으면 inf."""
        f = np.flatnonzero(self.failed)
        return float(self.severity[f[0]]) if f.size else float("inf")

    def area_under_curve(self) -> float:
        """강도 구간에 대한 평균 성능. 낮을수록 좋다.

        단일 조건의 값만 보고하면 '어디서 무너지는가' 가 사라진다.
        비교는 이 곡선 전체로 해야 한다.
        """
        v = np.where(self.failed, self.failure_value, np.asarray(self.value, dtype=float))
        if np.any(np.isnan(v)):
            v = np.where(np.isnan(v), self.failure_value, v)
        return float(np.trapezoid(v, self.severity)) if len(v) > 1 else float(v[0])

    def summary(self) -> str:
        fs = self.failure_severity
        fs_txt = "none" if np.isinf(fs) else f"{fs:.2f}"
        return (f"{self.metric_name}: AUC {self.area_under_curve():.4f}  "
                f"first failure at severity {fs_txt}")


def degradation_curve(severities: np.ndarray, values: np.ndarray,
                      failures: np.ndarray | None = None,
                      metric_name: str = "ate_rmse",
                      failure_value: float = 1.0) -> DegradationCurve:
    """조건 강도별 성능 곡선.

    failure_value 는 절대 스케일의 실패 벌점이다. 여러 시스템을 비교할 때는
    반드시 같은 값을 써야 하며, 지표의 단위에 맞춰 정해야 한다
    (ATE 라면 '이 이상은 무의미' 수준의 미터값).
    """
    s = np.asarray(severities, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(s)
    f = (np.zeros(len(s), dtype=bool) if failures is None
         else np.asarray(failures, dtype=bool))
    return DegradationCurve(s[order], v[order], f[order], metric_name, failure_value)


def recovery_latency(timestamps: np.ndarray, tracking_ok: np.ndarray,
                     condition_cleared_at: float) -> float:
    """조건이 개선된 뒤 추적을 되찾기까지 걸린 시간.

    '지도를 재시작하지 않고 수리한다'는 주장은 이 값으로만 검증된다.
    끝까지 복구 못 하면 inf.
    """
    t = np.asarray(timestamps, dtype=float)
    ok = np.asarray(tracking_ok, dtype=bool)
    after = t >= condition_cleared_at
    if not np.any(after):
        return float("inf")
    idx = np.flatnonzero(after & ok)
    return float(t[idx[0]] - condition_cleared_at) if idx.size else float("inf")


# ===========================================================================
# 정체성 / 연관
# ===========================================================================

@dataclass
class IdentityResult:
    id_switches: int
    fragmentations: int           # 같은 객체 트랙이 끊긴 횟수
    duplicate_rate: float         # 하나의 참 객체에 붙은 여분 ID 비율
    mostly_tracked: float         # 생존구간 80% 이상 추적된 객체 비율
    mostly_lost: float            # 20% 미만
    association_accuracy: float
    matched: int
    total: int

    def summary(self) -> str:
        return (f"IDS {self.id_switches}  frag {self.fragmentations}  "
                f"dup {self.duplicate_rate:.3f}  MT {self.mostly_tracked:.3f}  "
                f"assoc {self.association_accuracy:.3f}")


def identity_metrics(gt_tracks: list[dict[int, int]]) -> IdentityResult:
    """프레임별 {참 객체 id -> 추정 토큰 id} 매핑으로부터 정체성 지표를 낸다.

    관측되지 않은 객체는 해당 프레임 dict 에서 빠져 있으면 된다.

    duplicate_rate 는 WME 고유 주장에 대응한다: 루프 클로저에서 성좌 매칭이
    이력을 *병합* 해야지 같은 물체를 새 ID 로 복제하면 안 된다.
    """
    if not gt_tracks:
        raise ValueError("프레임이 없음")

    seen: dict[int, set[int]] = {}
    last: dict[int, int] = {}
    switches = 0
    frags = 0
    obs_count: dict[int, int] = {}
    life: dict[int, int] = {}
    matched = 0
    total = 0

    # 각 참 객체의 생존구간 = 처음 관측 ~ 마지막 관측
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    for f, frame in enumerate(gt_tracks):
        for gid in frame:
            first_seen.setdefault(gid, f)
            last_seen[gid] = f

    active: dict[int, bool] = {}
    for frame in gt_tracks:
        for gid, tid in frame.items():
            total += 1
            if tid is None or tid < 0:
                active[gid] = False
                continue
            matched += 1
            obs_count[gid] = obs_count.get(gid, 0) + 1
            seen.setdefault(gid, set()).add(tid)

            if gid in last and last[gid] != tid:
                switches += 1
            if not active.get(gid, False) and gid in last:
                frags += 1          # 끊겼다가 다시 붙음
            last[gid] = tid
            active[gid] = True

        for gid in list(active):
            if gid not in frame:
                active[gid] = False

    for gid in first_seen:
        life[gid] = last_seen[gid] - first_seen[gid] + 1

    ratios = np.array([obs_count.get(g, 0) / life[g] for g in life], dtype=float)
    extra = sum(max(0, len(ids) - 1) for ids in seen.values())

    return IdentityResult(
        id_switches=switches,
        fragmentations=frags,
        duplicate_rate=extra / max(1, len(seen)),
        mostly_tracked=float(np.mean(ratios >= 0.8)) if ratios.size else 0.0,
        mostly_lost=float(np.mean(ratios < 0.2)) if ratios.size else 0.0,
        association_accuracy=matched / max(1, total),
        matched=matched,
        total=total,
    )


# ===========================================================================
# 변화 이해
# ===========================================================================

@dataclass
class ChangeEvent:
    object_id: int
    kind: str                     # "moved" | "removed" | "added"
    time: float


@dataclass
class ChangeResult:
    precision: dict[str, float]
    recall: dict[str, float]
    latency: dict[str, float]     # 종류별 평균 검출 지연 (s)
    false_change_rate: float      # 실제 변화가 없는데 변화라 보고한 비율
    detected: int
    missed: int
    spurious: int

    def summary(self) -> str:
        parts = [f"{k} P{self.precision.get(k, 0):.2f}/R{self.recall.get(k, 0):.2f}"
                 for k in sorted(set(self.precision) | set(self.recall))]
        return "  ".join(parts) + f"  false-change {self.false_change_rate:.3f}"


def change_metrics(ground_truth: list[ChangeEvent], reported: list[ChangeEvent],
                   time_tolerance: float = 5.0) -> ChangeResult:
    """변화 검출 성능.

    false_change_rate 가 핵심이다. 안개가 끼었을 뿐인데 세계가 변했다고
    보고하는 시스템은 쓸 수 없다. 이 지표는 순수 센서 열화 구간에서 재야 한다.
    """
    kinds = sorted({e.kind for e in ground_truth} | {e.kind for e in reported})
    used = [False] * len(reported)

    tp = {k: 0 for k in kinds}
    latencies: dict[str, list[float]] = {k: [] for k in kinds}

    for gt in ground_truth:
        best, best_dt = -1, None
        for i, r in enumerate(reported):
            if used[i] or r.kind != gt.kind or r.object_id != gt.object_id:
                continue
            dt = r.time - gt.time
            # 변화 이전에 보고했다면 그것은 검출이 아니다
            if dt < 0.0 or dt > time_tolerance:
                continue
            if best_dt is None or dt < best_dt:
                best, best_dt = i, dt
        if best >= 0:
            used[best] = True
            tp[gt.kind] += 1
            latencies[gt.kind].append(best_dt)

    gt_count = {k: sum(1 for e in ground_truth if e.kind == k) for k in kinds}
    rep_count = {k: sum(1 for e in reported if e.kind == k) for k in kinds}
    spurious = sum(1 for u in used if not u)

    return ChangeResult(
        precision={k: tp[k] / rep_count[k] if rep_count[k] else 0.0 for k in kinds},
        recall={k: tp[k] / gt_count[k] if gt_count[k] else 0.0 for k in kinds},
        latency={k: float(np.mean(v)) if v else float("inf") for k, v in latencies.items()},
        false_change_rate=spurious / max(1, len(reported)),
        detected=sum(tp.values()),
        missed=len(ground_truth) - sum(tp.values()),
        spurious=spurious,
    )


# ===========================================================================
# 예측
# ===========================================================================

@dataclass
class PredictionResult:
    ade: dict[float, float]       # 시계별 평균 변위 오차 (m)
    fde: dict[float, float]       # 시계별 최종 변위 오차
    nll: dict[float, float]       # 예측 음의 로그우도
    count: dict[float, int]

    def summary(self) -> str:
        return "  ".join(f"{h}s: ADE {self.ade[h]:.3f} NLL {self.nll[h]:.2f}"
                         for h in sorted(self.ade))


def prediction_metrics(predictions: dict[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
                       ) -> PredictionResult:
    """시계별 예측 품질.

    ADE 만 보면 안 된다. 예측 *불확실성* 이 정직한지가 더 중요하고, 그것은
    NLL 로만 드러난다. 자신 있게 틀린 예측은 계획기를 사고로 이끈다.

    Args:
        predictions: {horizon_s: [(예측 평균, 예측 공분산, 참값), ...]}
    """
    ade, fde, nll, cnt = {}, {}, {}, {}

    for horizon, samples in predictions.items():
        if not samples:
            continue
        errs, nlls = [], []
        for mean, cov, truth in samples:
            mean = np.asarray(mean, dtype=float).ravel()
            truth = np.asarray(truth, dtype=float).ravel()
            d = mean.size
            P = np.asarray(cov, dtype=float).reshape(d, d) + np.eye(d) * 1e-12

            e = truth - mean
            errs.append(float(np.linalg.norm(e)))

            sign, logdet = np.linalg.slogdet(P)
            if sign <= 0:
                continue        # 유효하지 않은 공분산은 NLL 에서 제외
            nlls.append(0.5 * (float(e @ np.linalg.solve(P, e))
                               + logdet + d * np.log(2.0 * np.pi)))

        ade[horizon] = float(np.mean(errs))
        fde[horizon] = float(errs[-1])
        nll[horizon] = float(np.mean(nlls)) if nlls else float("inf")
        cnt[horizon] = len(samples)

    return PredictionResult(ade, fde, nll, cnt)


# ===========================================================================
# 통합 리포트
# ===========================================================================

@dataclass
class WorldModelReport:
    """한 시퀀스에 대한 전체 평가. 단일 스칼라로 요약하지 않는다."""
    name: str
    ate_rmse: float | None = None
    rpe_trans_rmse: float | None = None
    pose_nees: NeesResult | None = None
    existence_calibration: CalibrationResult | None = None
    identity: IdentityResult | None = None
    change: ChangeResult | None = None
    prediction: PredictionResult | None = None
    degradation: DegradationCurve | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"=== {self.name} ==="]
        if self.ate_rmse is not None:
            out.append(f"ATE rmse           {self.ate_rmse:.4f} m")
        if self.rpe_trans_rmse is not None:
            out.append(f"RPE trans rmse     {self.rpe_trans_rmse:.4f} m")
        if self.pose_nees is not None:
            out.append(f"pose consistency   {self.pose_nees.summary()}")
        if self.existence_calibration is not None:
            out.append(f"existence calib    {self.existence_calibration.summary()}")
        if self.identity is not None:
            out.append(f"identity           {self.identity.summary()}")
        if self.change is not None:
            out.append(f"change             {self.change.summary()}")
        if self.prediction is not None:
            out.append(f"prediction         {self.prediction.summary()}")
        if self.degradation is not None:
            out.append(f"degradation        {self.degradation.summary()}")
        for k, v in sorted(self.extras.items()):
            out.append(f"{k:18s} {v:.4f}")
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())
