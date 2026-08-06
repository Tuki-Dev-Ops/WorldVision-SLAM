"""C++ fusion::fuse 대 numpy 오라클.

docs/06-results.md 26 절이 남긴 공백을 메운다: `PoseFusion` / `SPA` /
`TierInformation` 은 바인딩에 아예 없어서 차분 커버리지가 **0** 이었고, 그런데
18절(3-tier 융합), 21절(alpha 적합과 chi-square 게이트), 23.4(열화 기울기)의
결론이 전부 이 코드 위에 서 있었다.

오라클은 `tools/fusion_replay.py` 다. 임시로 쓴 것이 아니라, C++ 이 낸 40 개
ablation 궤적을 최악 상대오차 **0.002 %** 로 재현한 것이 이미 확인된 포트다
(21절의 게이트). 그 재현은 궤적 수준이었고 - 여기서는 프레임 하나하나의
`fuse()` 출력을 직접 맞대 본다.

19.3 의 교훈을 따라 모든 테스트는 **판별 가드**를 함께 갖는다: 입력이 코드에
닿지 않으면(기여 tier 0 개 등) 조용히 통과하지 않고 실패한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from wme import HAS_NATIVE  # noqa: E402

if HAS_NATIVE:
    from wme import _core as core  # noqa: E402

import fusion_replay as fr  # noqa: E402
from wme.reference.geometry import SE3 as PySE3  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="_core 미빌드")

TIERS = ["t0", "t1", "t2"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _spd(rng: np.random.Generator, scale: float) -> np.ndarray:
    """무작위 대칭양정부호 6x6. tier 사이 스케일 차이를 흉내낸다."""
    A = rng.normal(size=(6, 6))
    return scale * (A @ A.T + 6.0 * np.eye(6))


def _pose(rng: np.random.Generator, mag: float) -> tuple:
    xi = rng.normal(size=6) * mag
    py = PySE3.exp(xi)
    cpp = core.SE3.exp(xi)
    return py, cpp


def _estimates(seed: int, n_avail: int = 3, scales=(1e4, 1e3, 1e1)):
    """같은 입력을 두 구현에 각각 만들어 준다."""
    rng = _rng(seed)
    py_est, cpp_est = [], []
    tiers = [core.Tier.Photometric, core.Tier.Constellation, core.Tier.Structural]
    for k in range(3):
        py_T, cpp_T = _pose(rng, 0.03)
        info = _spd(rng, scales[k])
        avail = k < n_avail
        alpha = float(rng.uniform(0.4, 1.0))

        e = core.TierEstimate()
        e.tier = tiers[k]
        e.T_cur_ref = cpp_T
        e.information = info
        e.alpha = alpha
        e.calibration = 1.0
        e.available = avail
        cpp_est.append(e)
        py_est.append((avail, py_T, info, alpha))
    return py_est, cpp_est


def _fuse_both(seed: int, n_avail: int = 3, scales=(1e4, 1e3, 1e1)):
    py_est, cpp_est = _estimates(seed, n_avail, scales)
    ok, res = core.fuse(cpp_est)
    out = fr.fuse(py_est)
    return ok, res, out


# ===========================================================================
# 1. 융합 포즈
# ===========================================================================

@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_fused_pose_matches(seed: int):
    ok, res, out = _fuse_both(seed)
    assert ok and out.ok
    # 판별 가드: 세 tier 가 실제로 기여했는지 확인한다. 하나도 안 붙었는데
    # 두 구현이 "항등" 으로 일치하면 그건 검증이 아니다 (19.3).
    assert res.contributing_tiers == 3, "기여 tier 가 3 이 아니면 이 비교는 무의미"

    t_cpp = np.asarray(res.T_cur_ref.t)
    t_py = out.T_cur_ref.t
    assert np.allclose(t_cpp, t_py, rtol=0, atol=1e-9), f"{t_cpp} vs {t_py}"

    R_cpp = np.asarray(res.T_cur_ref.R)
    assert np.allclose(R_cpp, out.T_cur_ref.R, rtol=0, atol=1e-9)


@pytest.mark.parametrize("seed", [4, 5])
def test_fused_information_matches(seed: int):
    ok, res, out = _fuse_both(seed)
    assert ok and out.ok
    L_cpp = np.asarray(res.information)
    # 정보행렬은 1e1~1e5 로 자릿수가 넓다. 절대 허용오차만 쓰면 큰 성분에서
    # 무의미하게 헐거워지므로 상대 비교를 쓴다.
    assert np.allclose(L_cpp, out.information, rtol=1e-9, atol=0)


# ===========================================================================
# 2. 기권 경로 - 19.3 이 지적한 "도달하지 않는 분기"
# ===========================================================================

def test_single_tier_returns_that_tier_unchanged():
    """tier 하나면 융합은 항등이어야 한다. 이 성질이 깨지면 나머지가 다 흔들린다."""
    py_est, cpp_est = _estimates(21, n_avail=1)
    ok, res = core.fuse(cpp_est)
    assert ok and res.contributing_tiers == 1
    t = np.asarray(res.T_cur_ref.t)
    assert np.allclose(t, py_est[0][1].t, rtol=0, atol=1e-12)


def test_no_available_tier_abstains_in_both():
    py_est, cpp_est = _estimates(22, n_avail=0)
    ok, _ = core.fuse(cpp_est)
    out = fr.fuse(py_est)
    assert ok is False and out.ok is False


def test_zero_weight_is_abstention_not_zero_information():
    """alpha=0 은 '정보가 0' 이 아니라 '기권' 이다. 둘을 섞으면 기여율 통계가
    조용히 틀어진다."""
    _, cpp_est = _estimates(23)
    cpp_est[1].alpha = 0.0
    ok, res = core.fuse(cpp_est)
    assert ok
    assert res.contributing_tiers == 2
    assert res.tiers[1].used is False
    assert res.tiers[1].reason == core.Abstain.ZeroWeight


# ===========================================================================
# 3. 스케일 차이 - 18.3 이 "1e6 배 차이" 라고 적은 그 상황
# ===========================================================================

def test_extreme_scale_gap_still_matches():
    """tier 사이 정보 스케일이 1e6 배 벌어져도 두 구현이 같은 답을 내야 한다.
    18.3 은 이 구간에서 절단 문턱을 잘못 잡으면 약한 tier 의 기여가 통째로
    사라진다고 적었다 - 그때 두 구현이 갈리면 여기서 잡힌다."""
    ok, res, out = _fuse_both(31, scales=(1e7, 1e1, 1e1))
    assert ok and out.ok
    assert res.contributing_tiers == 3
    assert np.allclose(np.asarray(res.T_cur_ref.t),
                       out.T_cur_ref.t, rtol=0, atol=1e-9)


# ===========================================================================
# 4. SE(3) 좌측 자코비안 - 융합이 서 있는 토대
# ===========================================================================

@pytest.mark.parametrize("seed", [41, 42, 43])
def test_left_jacobian_matches(seed: int):
    xi = _rng(seed).normal(size=6) * 0.2
    assert np.allclose(np.asarray(core.se3_left_jacobian(xi)),
                       fr.se3_left_jacobian(xi), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(core.se3_left_jacobian_inverse(xi)),
                       fr.se3_left_jacobian_inverse(xi), rtol=1e-9, atol=0)


def test_left_jacobian_inverse_is_actually_the_inverse():
    """판별 가드. 위 테스트는 두 구현이 **같이** 틀려도 통과한다."""
    xi = _rng(44).normal(size=6) * 0.2
    J = np.asarray(core.se3_left_jacobian(xi))
    Ji = np.asarray(core.se3_left_jacobian_inverse(xi))
    assert np.allclose(J @ Ji, np.eye(6), rtol=0, atol=1e-12)


def test_left_jacobian_at_zero_is_identity():
    assert np.allclose(np.asarray(core.se3_left_jacobian(np.zeros(6))),
                       np.eye(6), rtol=0, atol=1e-15)
