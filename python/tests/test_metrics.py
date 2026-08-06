"""평가 지표 검증.

지표가 틀리면 모든 실험 결과가 틀린다. 해석적으로 답을 아는 케이스와
분포에서 직접 표본을 뽑아 만든 케이스로 고정한다.
M1 게이트: 이 파일이 통과해야 설계 단계로 넘어갈 수 있다.
"""

import numpy as np
import pytest

from wme.eval.metrics import (
    ChangeEvent, calibration, change_metrics, degradation_curve,
    identity_metrics, nees, prediction_metrics, recovery_latency,
)
from wme.eval.stats import chi2_cdf, chi2_interval, chi2_ppf

RNG = np.random.default_rng(20260803)


# --- 카이제곱 (scipy 없이) -------------------------------------------------

@pytest.mark.parametrize("p,dof,expected", [
    (0.95, 1, 3.841459), (0.95, 2, 5.991465), (0.95, 10, 18.307038),
    (0.05, 1, 0.0039321400), (0.50, 1, 0.454936), (0.975, 6, 14.449375),
    (0.025, 6, 1.237344), (0.99, 30, 50.892181),
])
def test_chi2_ppf_matches_published_tables(p, dof, expected):
    assert chi2_ppf(p, dof) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("x,dof,expected", [
    (1.0, 1, 0.682689), (3.841459, 1, 0.95), (5.991465, 2, 0.95),
    (0.0, 3, 0.0), (10.0, 5, 0.924765),
])
def test_chi2_cdf_matches_published_tables(x, dof, expected):
    assert chi2_cdf(x, dof) == pytest.approx(expected, abs=1e-5)


def test_chi2_cdf_ppf_round_trip():
    for dof in [1, 2, 3, 6, 12, 50, 300]:
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            assert chi2_cdf(chi2_ppf(p, dof), dof) == pytest.approx(p, abs=1e-8)


def test_chi2_interval_is_ordered_and_brackets_mean():
    for dof in [1, 5, 30, 200]:
        lo, hi = chi2_interval(dof)
        assert lo < dof < hi


# --- NEES -----------------------------------------------------------------

def _sample_errors(cov, n):
    """공분산 cov 인 정규분포에서 오차를 뽑는다."""
    L = np.linalg.cholesky(cov)
    return RNG.standard_normal((n, len(cov))) @ L.T


def test_nees_is_consistent_for_a_correct_filter():
    """보고 공분산이 실제 오차 분포와 같으면 일관성 판정을 통과해야 한다."""
    P = np.diag([0.04, 0.09, 0.01])
    n = 3000
    errors = _sample_errors(P, n)
    covs = np.repeat(P[None], n, axis=0)

    r = nees(errors, covs)
    assert r.consistent, r.summary()
    assert not r.overconfident
    assert r.anees / r.dof == pytest.approx(1.0, abs=0.1)


def test_nees_flags_overconfidence():
    """실제보다 작은 공분산을 보고하면 과신으로 잡혀야 한다.

    이것이 실제 SLAM 시스템의 지배적 실패 모드다.
    """
    P_true = np.diag([0.04, 0.09, 0.01])
    P_reported = P_true / 9.0            # 표준편차를 3배 과소보고
    n = 2000

    r = nees(_sample_errors(P_true, n), np.repeat(P_reported[None], n, axis=0))
    assert not r.consistent
    assert r.overconfident
    assert r.anees / r.dof == pytest.approx(9.0, rel=0.15)


def test_nees_flags_underconfidence():
    P_true = np.diag([0.04, 0.09, 0.01])
    P_reported = P_true * 4.0
    n = 2000

    r = nees(_sample_errors(P_true, n), np.repeat(P_reported[None], n, axis=0))
    assert not r.consistent
    assert not r.overconfident


def test_nees_handles_correlated_covariance():
    P = np.array([[0.05, 0.02, 0.0],
                  [0.02, 0.08, 0.01],
                  [0.0, 0.01, 0.03]])
    n = 3000
    r = nees(_sample_errors(P, n), np.repeat(P[None], n, axis=0))
    assert r.consistent, r.summary()


def test_nees_band_tightens_with_more_samples():
    """표본이 많을수록 일관성 구간이 좁아져야 한다."""
    P = np.eye(3) * 0.01
    widths = []
    for n in [50, 500, 5000]:
        r = nees(_sample_errors(P, n), np.repeat(P[None], n, axis=0))
        widths.append(r.upper - r.lower)
    assert widths[0] > widths[1] > widths[2]


def test_nees_rejects_malformed_input():
    with pytest.raises(ValueError):
        nees(np.zeros((5, 3)), np.zeros((5, 2, 2)))
    with pytest.raises(ValueError):
        nees(np.zeros((0, 3)), np.zeros((0, 3, 3)))


# --- 보정 -----------------------------------------------------------------

def test_calibration_of_perfectly_calibrated_predictions():
    """확률 p 를 준 사건이 실제로 p 빈도로 일어나면 ECE ~ 0."""
    p = RNG.uniform(0.0, 1.0, 20000)
    y = (RNG.uniform(0.0, 1.0, 20000) < p).astype(float)

    r = calibration(p, y, bins=10)
    assert r.ece < 0.02, r.summary()


def test_calibration_detects_overconfidence():
    """0.95 라고 말해놓고 60% 만 맞으면 잡혀야 한다."""
    n = 5000
    p = np.full(n, 0.95)
    y = (RNG.uniform(0, 1, n) < 0.60).astype(float)

    r = calibration(p, y, bins=10)
    assert r.ece == pytest.approx(0.35, abs=0.03)
    assert r.brier > 0.2


def test_calibration_includes_probability_one():
    """p=1.0 이 마지막 구간에서 누락되면 안 된다."""
    r = calibration(np.array([1.0, 1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0]))
    assert r.bin_count.sum() == 4
    assert r.ece == pytest.approx(0.0, abs=1e-12)


def test_calibration_rejects_mismatched_input():
    with pytest.raises(ValueError):
        calibration(np.zeros(5), np.zeros(3))


# --- 정체성 ---------------------------------------------------------------

def test_identity_perfect_tracking():
    frames = [{1: 100, 2: 200} for _ in range(20)]
    r = identity_metrics(frames)
    assert r.id_switches == 0
    assert r.duplicate_rate == 0.0
    assert r.mostly_tracked == 1.0
    assert r.association_accuracy == 1.0


def test_identity_counts_switches():
    frames = [{1: 100} for _ in range(5)] + [{1: 101} for _ in range(5)]
    r = identity_metrics(frames)
    assert r.id_switches == 1
    assert r.duplicate_rate == pytest.approx(1.0)   # 참 객체 1개에 여분 ID 1개


def test_identity_counts_fragmentation():
    """추적이 끊겼다가 다시 붙는 것을 세야 한다."""
    frames = ([{1: 100}] * 5) + ([{}] * 3) + ([{1: 100}] * 5)
    r = identity_metrics(frames)
    assert r.fragmentations == 1
    assert r.id_switches == 0, "같은 ID 로 복귀했으므로 스위치가 아니다"


def test_identity_duplicate_rate_catches_loop_closure_failure():
    """WME 고유 주장: 루프 클로저는 이력을 병합해야지 객체를 복제하면 안 된다."""
    merged = [{1: 100}] * 10 + [{1: 100}] * 10
    duplicated = [{1: 100}] * 10 + [{1: 999}] * 10

    assert identity_metrics(merged).duplicate_rate == 0.0
    assert identity_metrics(duplicated).duplicate_rate > 0.0


def test_identity_mostly_lost():
    frames = [{1: 100}] + [{}] * 19
    frames[19] = {1: 100}
    r = identity_metrics(frames)
    assert r.mostly_lost == 1.0
    assert r.mostly_tracked == 0.0


def test_identity_unmatched_detections_lower_accuracy():
    frames = [{1: 100, 2: -1} for _ in range(10)]
    r = identity_metrics(frames)
    assert r.association_accuracy == pytest.approx(0.5)


# --- 변화 -----------------------------------------------------------------

def test_change_perfect_detection():
    gt = [ChangeEvent(1, "moved", 10.0), ChangeEvent(2, "removed", 10.0)]
    rep = [ChangeEvent(1, "moved", 11.0), ChangeEvent(2, "removed", 12.0)]

    r = change_metrics(gt, rep, time_tolerance=5.0)
    assert r.recall["moved"] == 1.0 and r.recall["removed"] == 1.0
    assert r.precision["moved"] == 1.0
    assert r.latency["moved"] == pytest.approx(1.0)
    assert r.false_change_rate == 0.0


def test_change_ignores_reports_before_the_change():
    """변화가 일어나기 전에 보고한 것은 검출이 아니라 오보다."""
    gt = [ChangeEvent(1, "moved", 10.0)]
    rep = [ChangeEvent(1, "moved", 9.0)]

    r = change_metrics(gt, rep)
    assert r.recall["moved"] == 0.0
    assert r.false_change_rate == 1.0


def test_change_respects_time_tolerance():
    gt = [ChangeEvent(1, "moved", 10.0)]
    assert change_metrics(gt, [ChangeEvent(1, "moved", 30.0)], time_tolerance=5.0).detected == 0
    assert change_metrics(gt, [ChangeEvent(1, "moved", 30.0)], time_tolerance=25.0).detected == 1


def test_change_does_not_confuse_types():
    """이동과 제거는 다른 행동을 요구한다. 섞으면 안 된다."""
    gt = [ChangeEvent(1, "moved", 10.0)]
    r = change_metrics(gt, [ChangeEvent(1, "removed", 11.0)])
    assert r.recall["moved"] == 0.0
    assert r.spurious == 1


def test_false_change_rate_under_pure_degradation():
    """실제 변화가 없는데 보고하면 전부 오보여야 한다.

    안개가 꼈을 뿐인데 세계가 변했다고 말하는 시스템은 쓸 수 없다.
    """
    r = change_metrics([], [ChangeEvent(i, "moved", 10.0) for i in range(4)])
    assert r.false_change_rate == 1.0
    assert r.spurious == 4


def test_change_each_report_matches_at_most_one_gt():
    gt = [ChangeEvent(1, "moved", 10.0), ChangeEvent(1, "moved", 12.0)]
    r = change_metrics(gt, [ChangeEvent(1, "moved", 11.0)], time_tolerance=5.0)
    assert r.detected == 1
    assert r.missed == 1


# --- 예측 -----------------------------------------------------------------

def test_prediction_zero_error_for_perfect_forecast():
    samples = [(np.array([1.0, 2.0, 3.0]), np.eye(3) * 0.01, np.array([1.0, 2.0, 3.0]))
               for _ in range(10)]
    r = prediction_metrics({1.0: samples})
    assert r.ade[1.0] == pytest.approx(0.0)


def test_prediction_nll_punishes_confident_wrong_forecasts():
    """평균 오차가 같아도 과신한 예측이 더 큰 벌점을 받아야 한다."""
    truth = np.array([1.0, 0.0, 0.0])
    mean = np.zeros(3)

    humble = [(mean, np.eye(3) * 1.0, truth)]
    confident = [(mean, np.eye(3) * 0.01, truth)]

    r_h = prediction_metrics({1.0: humble})
    r_c = prediction_metrics({1.0: confident})

    assert r_h.ade[1.0] == pytest.approx(r_c.ade[1.0]), "ADE 는 같아야 한다"
    assert r_c.nll[1.0] > r_h.nll[1.0], "NLL 만이 과신을 잡아낸다"


def test_prediction_error_grows_with_horizon():
    def samples(sigma):
        return [(np.zeros(3), np.eye(3) * 0.1, RNG.normal(0, sigma, 3)) for _ in range(200)]

    r = prediction_metrics({0.5: samples(0.05), 1.0: samples(0.15), 2.0: samples(0.4)})
    assert r.ade[0.5] < r.ade[1.0] < r.ade[2.0]


# --- 열화 곡선 -------------------------------------------------------------

def test_degradation_curve_sorts_and_integrates():
    c = degradation_curve([1.0, 0.0, 0.5], [0.3, 0.1, 0.2])
    assert list(c.severity) == [0.0, 0.5, 1.0]
    assert list(c.value) == [0.1, 0.2, 0.3]
    assert c.area_under_curve() == pytest.approx(0.2)


def test_degradation_curve_reports_first_failure():
    c = degradation_curve([0.0, 0.25, 0.5, 0.75, 1.0],
                          [0.1, 0.12, 0.2, np.nan, np.nan],
                          failures=[False, False, False, True, True])
    assert c.failure_severity == pytest.approx(0.75)


def test_degradation_curve_no_failure():
    c = degradation_curve([0.0, 0.5, 1.0], [0.1, 0.2, 0.3])
    assert np.isinf(c.failure_severity)


def test_degradation_auc_penalises_failure():
    """잘 하다가 무너지는 시스템이 꾸준히 버티는 시스템보다 좋아 보이면 안 된다.

    실패 벌점을 시스템 자신의 값에서 유도하면 정확히 그 역전이 일어난다.
    절대 스케일 failure_value 를 쓰는 이유.
    """
    graceful = degradation_curve([0.0, 0.5, 1.0], [0.1, 0.2, 0.4], failure_value=1.0)
    brittle = degradation_curve([0.0, 0.5, 1.0], [0.1, 0.15, np.nan],
                                failures=[False, False, True], failure_value=1.0)
    assert brittle.area_under_curve() > graceful.area_under_curve()


def test_degradation_auc_uses_the_same_scale_for_all_systems():
    """failure_value 가 다르면 비교가 성립하지 않는다. 명시적 파라미터인 이유."""
    a = degradation_curve([0.0, 1.0], [0.1, np.nan], failures=[False, True],
                          failure_value=1.0)
    b = degradation_curve([0.0, 1.0], [0.1, np.nan], failures=[False, True],
                          failure_value=5.0)
    assert b.area_under_curve() > a.area_under_curve()


# --- 복구 -----------------------------------------------------------------

def test_recovery_latency():
    t = np.arange(0.0, 10.0, 0.5)
    ok = t >= 6.0                      # 6초에 추적 복구
    assert recovery_latency(t, ok, condition_cleared_at=5.0) == pytest.approx(1.0)


def test_recovery_never_happens():
    t = np.arange(0.0, 10.0, 0.5)
    assert np.isinf(recovery_latency(t, np.zeros(len(t), dtype=bool), 5.0))
