"""실데이터 NEES 파이프라인 검증.

이 파일이 지키는 것은 결론이 아니라 *측정 도구* 다. docs/06-results.md 10.4:
알고리즘이 틀렸다고 결론짓기 전에, 측정이 판별력을 갖는지부터 확인해야 한다.
일부러 틀린 공분산을 넣었을 때 ANEES 가 정해진 방향으로 움직이지 않으면
tools/tum_nees.py 가 내는 숫자는 아무 의미가 없다.

마지막 두 테스트만 실데이터를 요구하고, 없으면 건너뛴다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from wme.reference.geometry import SE3, so3_exp  # noqa: E402

tum_nees = pytest.importorskip("tum_nees")

DATA = Path(__file__).resolve().parents[2] / "data"
DIAG = Path(__file__).resolve().parents[2] / "results" / "nees"
SEQ = "rgbd_dataset_freiburg1_xyz"


def _synthetic(n: int, cov_scale: float, seed: int = 3):
    """참 공분산 P 를 알고, 보고된 공분산은 그 cov_scale 배인 표본."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((6, 6))
    P = A @ A.T + 6.0 * np.eye(6)
    e = (np.linalg.cholesky(P) @ rng.standard_normal((6, n))).T
    reported = np.repeat(np.linalg.inv(P * cov_scale)[None], n, axis=0)
    return e, reported


# ---------------------------------------------------------------------------
# 규약
# ---------------------------------------------------------------------------

def test_error_uses_left_perturbation():
    """e = log(T_est * T_gt^-1) 는 T_est = exp(e) * T_gt 와 같아야 한다.

    정보행렬이 좌측 섭동에 대해 정의되어 있으므로, 오차를 우측 섭동으로
    만들면 다른 좌표계의 벡터를 같은 행렬로 정규화하게 된다.
    """
    T_gt = SE3(so3_exp(np.array([0.3, -0.2, 0.1])), np.array([0.5, -0.2, 0.3]))
    xi = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
    T_est = SE3.exp(xi) @ T_gt
    e = (T_est @ T_gt.inverse()).log()
    assert np.allclose(e, xi, atol=1e-9)

    # 우측 섭동은 일반적으로 다른 벡터다 - 둘을 섞으면 조용히 틀린다
    e_right = (T_gt.inverse() @ T_est).log()
    assert not np.allclose(e_right, xi, atol=1e-6)


def test_information_unpack_is_symmetric():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((6, 6))
    M = M @ M.T
    row = {}
    k = 0
    for i in range(6):
        for j in range(i, 6):
            row[tum_nees.INFO_COLUMNS[k]] = f"{M[i, j]:.17g}"
            k += 1
    out = tum_nees.unpack_information(row)
    assert np.allclose(out, M)
    assert np.allclose(out, out.T)


def test_groundtruth_interpolation_beats_nearest():
    """등속 운동에서 보간은 정확히 맞고 최근접은 못 맞춘다.

    TUM 진리값은 100 Hz 라 최근접 오차가 최대 5 ms 다. 재려는 상대 포즈
    오차가 mm 단위이므로 이 차이가 결과를 좌우한다.
    """
    stamps = np.arange(0.0, 1.0, 0.01)
    poses = [SE3(np.eye(3), np.array([1.0 * t, 0.0, 0.0])) for t in stamps]
    p = tum_nees.interpolate_pose(stamps, poses, 0.1234)
    assert p is not None
    assert abs(p.t[0] - 0.1234) < 1e-9

    # 구간 밖 / 큰 공백은 값을 지어내지 않고 None
    assert tum_nees.interpolate_pose(stamps, poses, 5.0) is None
    sparse = np.array([0.0, 1.0])
    assert tum_nees.interpolate_pose(sparse, [poses[0], poses[-1]], 0.5,
                                     max_gap=0.05) is None


# ---------------------------------------------------------------------------
# 판별력 - 이게 통과하지 못하면 실데이터 숫자는 읽을 가치가 없다
# ---------------------------------------------------------------------------

def test_calibrated_covariance_gives_anees_six():
    e, L = _synthetic(4000, 1.0)
    r = tum_nees.anees_report(tum_nees.nees_full(e, L), 6)
    assert r["consistent"], r
    assert abs(r["anees"] - 6.0) < 0.5


@pytest.mark.parametrize("scale,expect", [(0.1, 60.0), (0.01, 600.0),
                                          (10.0, 0.6), (100.0, 0.06)])
def test_wrong_covariance_moves_anees_the_right_way(scale, expect):
    """공분산을 s 배 하면 ANEES 는 정확히 1/s 배가 되어야 한다.

    부풀린 쪽과 줄인 쪽을 둘 다 넣는다. 한쪽만 확인하면 부호가 뒤집힌
    파이프라인도 통과한다.
    """
    e, L = _synthetic(4000, scale)
    r = tum_nees.anees_report(tum_nees.nees_full(e, L), 6)
    assert abs(r["anees"] / expect - 1.0) < 0.1, r
    assert not r["consistent"]
    assert r["overconfident"] == (scale < 1.0)


def test_acceptance_band_tightens_with_sample_count():
    """수용구간은 표본 수에 따라 좁아져야 한다.

    n 을 빼고 "ANEES 1.2 면 거의 맞다" 라고 말하는 것을 막는 성질이다.
    """
    wide = tum_nees.anees_report(np.full(10, 6.0), 6)
    tight = tum_nees.anees_report(np.full(1000, 6.0), 6)
    assert (tight["hi"] - tight["lo"]) < 0.2 * (wide["hi"] - wide["lo"])


def test_block_nees_uses_marginal_not_conditional():
    """블록 NEES 는 P 의 부분블록이어야 하고 Lambda 의 부분블록이면 안 된다.

    상관이 있는 공분산에서 둘은 서로 다르다. 조건부(Lambda 부분블록)를 쓰면
    "회전을 안다고 가정했을 때의 병진 불확실성" 이라는 다른 질문에 답하게
    되고, 그 답은 항상 더 자신 있어 보인다.
    """
    rng = np.random.default_rng(11)
    A = rng.standard_normal((6, 6))
    P = A @ A.T + 3.0 * np.eye(6)
    L = np.linalg.inv(P)
    n = 6000
    e = (np.linalg.cholesky(P) @ rng.standard_normal((6, n))).T
    infos = np.repeat(L[None], n, axis=0)

    v, ok = tum_nees.nees_block(e, infos, slice(0, 3))
    assert ok.all()
    r = tum_nees.anees_report(v[ok], 3)
    assert r["consistent"], r

    # 조건부 버전은 같은 데이터에서 일관되지 않는다 (과소신뢰 쪽으로 틀린다)
    cond = np.einsum("ni,ij,nj->n", e[:, :3], L[:3, :3], e[:, :3])
    assert not tum_nees.anees_report(cond, 3)["consistent"]


def test_selftest_passes():
    assert tum_nees.selftest() == 0


# ---------------------------------------------------------------------------
# 점 개수 스케일링 - 상관과 비수축 항을 가르는 유일한 측정
# ---------------------------------------------------------------------------

def _runs(kind: str, seed: int = 4):
    """N 을 바꿔가며 만든 가짜 실행들.

    indep : 오차 분산이 1/N 로 준다 (잔차가 독립일 때의 모습)
    corr  : 1/N 로 주되 상수배만큼 과신 (유효표본수 N/8)
    sys   : 오차가 N 과 무관 (docs/04-unified-objective.md 5.2.1(c) Sigma_sys)
    """
    rng = np.random.default_rng(seed)
    out = []
    for npt in (100.0, 200.0, 400.0, 800.0, 1600.0):
        info = np.repeat((npt * np.eye(6))[None], 500, axis=0)
        s = {"indep": 1.0 / np.sqrt(npt),
             "corr": np.sqrt(8.0) / np.sqrt(npt),
             "sys": 1.0}[kind]
        err = s * rng.standard_normal((500, 6))
        out.append((f"n{int(npt)}", [
            tum_nees.FrameData(0.0, err[i], info[i],
                               float(np.linalg.norm(err[i, :3])), 0.0, 0.0, 0.0,
                               points=npt)
            for i in range(500)]))
    return out


def _anees_slope(lines):
    return float("\n".join(lines).split("ANEES~N^")[1].split()[0])


@pytest.mark.parametrize("kind,want", [("indep", 0.0), ("corr", 0.0),
                                       ("sys", 1.0)])
def test_scaling_separates_correlation_from_non_shrinking_term(kind, want):
    """상관은 ANEES 기울기 0, 비수축 항은 1 이어야 한다.

    이 구분이 이 측정의 존재 이유다. corr 은 8 배 과신이지만 기울기는 0 이다 -
    과신의 *크기* 가 아니라 N 을 따라가는지가 원인을 가른다. 두 경우가 같은
    기울기를 준다면 --scaling 이 내는 숫자는 아무것도 말해주지 않는다.
    """
    assert abs(_anees_slope(tum_nees.scaling_report(_runs(kind))) - want) < 0.15


def test_scaling_refuses_to_report_slope_from_two_runs():
    """실행이 모자라면 기울기를 지어내지 않는다."""
    lines = tum_nees.scaling_report(_runs("sys")[:2])
    assert not any("ANEES~N^" in ln for ln in lines)


def test_loglog_slope_recovers_known_exponent():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    assert abs(tum_nees.loglog_slope(x, x ** -0.5) + 0.5) < 1e-9
    assert np.isnan(tum_nees.loglog_slope(np.array([1.0]), np.array([1.0])))


# ---------------------------------------------------------------------------
# 실데이터. 없으면 건너뛴다.
# ---------------------------------------------------------------------------

_diag = DIAG / f"{SEQ}_diag.csv"
_needs_data = pytest.mark.skipif(
    not (_diag.exists() and (DATA / SEQ / "groundtruth.txt").exists()),
    reason="TUM 시퀀스 또는 정보행렬이 담긴 진단 CSV 가 없음")


@_needs_data
def test_real_information_matrix_is_within_an_order_of_magnitude():
    """실측 결과의 회귀 방지. 양쪽 방향 모두.

    이 테스트는 원래 "최소 100 배 과신" 을 고정하고 있었다. 그 상태는 고쳐졌다
    (InformationModel::EffectiveSample, ANEES 3.7~9.4). 이제 고정할 것은 반대다:
    다시 과신으로 돌아가도 안 되고, 공분산을 부풀려 일관성을 사도 안 된다
    (docs/06-results.md 10.3 - 일관성만 보면 게임이 된다).

    정확한 값은 고정하지 않는다. 정렬기가 바뀌면 움직여야 하는 값이다.
    고정하는 것은 자릿수와, 그것이 정확도와 함께 성립한다는 사실뿐이다.
    """
    frames, _ = tum_nees.collect(DATA / SEQ, _diag)
    assert len(frames) > 50
    a = tum_nees.analyse(frames)

    # 6 의 한 자릿수 안. 아래로 벗어나면 부풀려 산 일관성이다.
    assert 0.6 < a["full"]["anees"] < 60.0, a["full"]

    # 일관성만 보면 안 된다 - 정확도가 같이 성립해야 의미가 있다
    assert a["trans_err_med"] < 0.05

    # 프레임별 게이트도 같이 본다. 평균만 맞고 분포가 틀린 경우를 잡는다.
    assert a["frac_within_95"] > 0.5


@_needs_data
def test_real_data_pipeline_discriminates():
    """실제 Lambda 를 그대로 쓰되 오차만 그 Lambda 에서 뽑아 넣으면 6 이 나와야.

    실패하면 오차 계산이나 정보행렬 해석이 틀린 것이지 추정기 문제가 아니다.
    실데이터 숫자를 읽기 전에 반드시 통과해야 하는 관문.
    """
    frames, _ = tum_nees.collect(DATA / SEQ, _diag)
    rng = np.random.default_rng(1)
    syn = []
    for f in frames:
        w, V = np.linalg.eigh(0.5 * (f.information + f.information.T))
        if w.min() <= 0:
            continue
        e = V @ (rng.standard_normal(6) / np.sqrt(w))
        syn.append(tum_nees.FrameData(f.stamp, e, f.information, 0.0, 0.0,
                                      f.motion, f.dt))
    assert len(syn) > 50
    r = tum_nees.analyse(syn)["full"]
    assert r["consistent"], r
