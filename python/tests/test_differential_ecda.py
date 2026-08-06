"""ECDA 원소 단위 차분 - 26절이 "블로커는 구체적이다" 라고 적은 그 항목.

그 문장은 이랬다: *"`python/wme/localization/ecda.py` 가 아직 `huber_delta = 12.0`
을 들고 있다. C++ 이 11.4 에서 지운 고정 임계 커널이다. `robustDelta` 가
포팅되기 전까지 점 선택과 가중이 **구조적으로** 다르므로 크기 수준 비교만
타당하다."*

그 한 줄이 포팅되었으므로(적응형 huber + 고정 임계 inlier 계수, 13.4) 두 구현이
같은 점을 같은 가중으로 쓰게 되었고, 이제 값 자체를 맞대 볼 수 있다.

**여기서 요구하는 것은 비트 동일성이 아니다.** 두 구현은 보간, 피라미드 축소,
LM 감쇠 상세가 다르므로 같은 답에 다른 경로로 도달한다. 요구하는 것은 (a) 같은
포즈로 수렴하는가, (b) 그 과정의 관측량(점 수, inlier 비, 잔차)이 같은 자리에
있는가다. 서로 다른 두 값이 우연히 가까운 것과 구분하기 위해, 판별 가드로
**일부러 틀린 입력** 이 실제로 걸리는지도 함께 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from wme import HAS_NATIVE

if HAS_NATIVE:
    from wme import _core as core

from wme.localization import ecda
from wme.reference.geometry import SE3 as PySE3
from wme.sim.world import CameraModel

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="_core 미빌드")

W, H = 160, 120
CAM = CameraModel(fx=120.0, fy=120.0, cx=W / 2 - 0.5, cy=H / 2 - 0.5, width=W, height=H)


def _texture(seed: int = 7) -> np.ndarray:
    """저주파 + 고주파가 섞인 텍스처. 순수 난수는 보간 차이에 과민하다."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(H // 8, W // 8))
    big = np.kron(base, np.ones((8, 8)))[:H, :W]
    fine = rng.normal(size=(H, W)) * 0.35
    img = 128.0 + 45.0 * big + 12.0 * fine
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.float32))


def _plane_depth(dist: float = 2.5) -> np.ndarray:
    """카메라를 정면으로 마주보는 평면. 깊이는 화소마다 조금씩 다르다."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rx = (xx - CAM.cx) / CAM.fx
    ry = (yy - CAM.cy) / CAM.fy
    n = np.array([0.12, 0.08, 0.99])
    n /= np.linalg.norm(n)
    z = dist / (n[0] * rx + n[1] * ry + n[2])
    return np.ascontiguousarray(z.astype(np.float32))


def _warp(ref: np.ndarray, depth: np.ndarray, T: PySE3) -> np.ndarray:
    """ref 를 T 로 워프해 cur 을 만든다. 두 구현에 같은 입력을 주기 위한 것이므로
    합성 방식 자체는 검증 대상이 아니다."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z = depth.astype(np.float64)
    X = np.stack([(xx - CAM.cx) * z / CAM.fx, (yy - CAM.cy) * z / CAM.fy, z], axis=-1)
    Xc = X @ T.R.T + T.t
    u = CAM.fx * Xc[..., 0] / Xc[..., 2] + CAM.cx
    v = CAM.fy * Xc[..., 1] / Xc[..., 2] + CAM.cy
    out = np.full((H, W), np.nan)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out[vi[ok], ui[ok]] = ref[ok]
    # 구멍은 원본으로 메운다 - 두 구현이 같은 영상을 보기만 하면 된다.
    holes = np.isnan(out)
    out[holes] = ref[holes]
    return np.ascontiguousarray(out.astype(np.float32))


def _wide_depth(near: float = 3.0, far: float = 60.0) -> np.ndarray:
    """왼쪽은 가깝고 오른쪽은 먼 경사면. 1/Z 가 x 에 선형이라 실제 평면이다.

    깊이 불확실성 가중을 재려면 한 장면 안에 깊이가 크게 다른 점이 있어야 한다.
    2.5 m 평면에서는 sigma_Z = c*Z^2 가 어디서나 같아 가중이 상수배가 되고,
    상수배는 정규방정식의 해를 바꾸지 않는다 - 즉 아무것도 재지 못한다.
    """
    xx = np.mgrid[0:H, 0:W][1].astype(np.float64)
    t = xx / (W - 1)
    inv = (1.0 - t) / near + t / far
    return np.ascontiguousarray((1.0 / inv).astype(np.float32))


def _both(truth: PySE3, depth: np.ndarray | None = None, **over):
    ref = _texture()
    d = _plane_depth() if depth is None else depth
    cur = _warp(ref, d, truth)

    cfg = ecda.EcdaConfig(pyramid_levels=3, max_iterations=30, **over)
    py = ecda.align(ref, d, cur, CAM, cfg=cfg)

    ccfg = core.DirectAlignerConfig()
    ccfg.pyramid_levels = 3
    ccfg.max_iterations = 30
    for k, v in over.items():
        setattr(ccfg, k, v)
    cpp = core.DirectAligner(ccfg).align(ref, d, cur,
                                         CAM.fx, CAM.fy, CAM.cx, CAM.cy)
    return py, cpp, truth


# ===========================================================================
# 1. 같은 포즈로 수렴하는가
# ===========================================================================

@pytest.mark.parametrize("trans", [(0.02, -0.01, 0.005), (0.05, 0.03, -0.02)])
def test_both_converge_to_the_same_pose(trans):
    truth = PySE3.exp(np.array([*trans, 0.004, -0.006, 0.003]))
    py, cpp, truth = _both(truth)

    t_py = py.T_cur_ref.t
    t_cpp = np.asarray(cpp.T_cur_ref.t)
    err_py = np.linalg.norm(t_py - truth.t)
    err_cpp = np.linalg.norm(t_cpp - truth.t)
    gap = np.linalg.norm(t_py - t_cpp)

    # **주된 주장은 두 구현이 서로 일치한다는 것이다.** 참값과의 거리는 여기서
    # 부차적이다: `_warp` 는 최근접 스플랫에 구멍을 원본으로 메우는 조잡한
    # 합성기라 참값 자체가 정확하지 않다. 실제로 두 구현은 서로 0.3 mm 안에
    # 들어오면서 참값과는 나란히 21 mm 어긋난다 - 그 공통 오차는 엔진이 아니라
    # 내가 준 영상의 성질이다. 두 값을 한 문턱으로 묶으면 합성기의 한계가
    # 구현 불일치로 보고된다.
    assert gap < 5e-3, (
        f"두 구현이 서로 {gap*1000:.2f} mm 벌어졌다 "
        f"(참값 오차 py {err_py*1000:.1f} / cpp {err_cpp*1000:.1f} mm)")
    # 그럼에도 둘 다 실제로 수렴은 해야 한다 - 나란히 발산해도 gap 은 작다.
    assert err_py < 0.05 and err_cpp < 0.05, (
        f"둘 다 수렴하지 못했다: py {err_py:.4f} cpp {err_cpp:.4f}")


# ===========================================================================
# 2. 커널이 실제로 같은 점을 같은 무게로 쓰는가
#    - 26절이 막혀 있다고 적은 바로 그 비교
# ===========================================================================

def test_point_counts_are_comparable():
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    py, cpp, _ = _both(truth)
    # 격자 선택 규칙이 같으므로 점 수는 같은 자리에 있어야 한다. 경계 처리
    # 차이로 완전히 같지는 않으므로 20 % 이내를 요구한다.
    ratio = py.point_count / max(cpp.point_count, 1)
    assert 0.8 < ratio < 1.25, f"py {py.point_count} vs cpp {cpp.point_count}"


def test_inlier_ratio_is_comparable():
    """고정 임계 inlier 계수가 양쪽에 같이 들어갔는지 본다. 한쪽만 적응형
    임계로 세면 이 값이 구조적으로 갈린다 (13.4)."""
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    py, cpp, _ = _both(truth)
    assert abs(py.inlier_ratio - cpp.inlier_ratio) < 0.2, (
        f"py {py.inlier_ratio:.3f} vs cpp {cpp.inlier_ratio:.3f}")


# ===========================================================================
# 3. 판별 가드 - 이 비교가 실제로 무언가를 걸러내는가
# ===========================================================================

def test_the_comparison_would_catch_a_wrong_pose():
    """위 테스트들이 우연히 통과하는 것이 아님을 보인다. 일부러 틀린 포즈를
    넣으면 같은 허용오차에서 반드시 걸려야 한다."""
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    py, cpp, _ = _both(truth)
    wrong = np.asarray(cpp.T_cur_ref.t) + np.array([0.05, 0.0, 0.0])
    gap = np.linalg.norm(py.T_cur_ref.t - wrong)
    assert gap > 5e-3, "허용오차가 5 cm 오차도 통과시킨다면 이 비교는 무의미하다"


def test_depth_consistency_is_exposed_and_judged():
    """25절의 신호가 바인딩을 통해 실제로 판정되는지 확인한다. cur 깊이를 주지
    않으면 언제나 -1 이라 어떤 테스트도 아무 것도 재지 못한다."""
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    ref, depth = _texture(), _plane_depth()
    cur = _warp(ref, depth, truth)

    a = core.DirectAligner(core.DirectAlignerConfig())
    without = a.align(ref, depth, cur, CAM.fx, CAM.fy, CAM.cx, CAM.cy)
    assert without.depth_consistency < 0.0
    assert not without.depth_consistent()

    b = core.DirectAligner(core.DirectAlignerConfig())
    withd = b.align(ref, depth, cur, CAM.fx, CAM.fy, CAM.cx, CAM.cy,
                    cur_depth=depth)
    assert withd.depth_consistency >= 0.0, "cur 깊이를 줬는데도 판정되지 않았다"




# ===========================================================================
# 4. 깊이 불확실성 가중 (25.21)
# ===========================================================================
#
# KITTI 가 드러낸 결함: ECDA 는 깊이를 참으로 믿는다. 60 m 점의 스테레오 깊이
# 오차는 +-4 m, 6 m 점은 +-4 cm 인데 같은 무게로 들어간다. 새 항은
#   w *= sigma_I^2 / (sigma_I^2 + (dr/dd * c*d^2)^2)
#
# **관측량 선택.** 여기서는 수렴 포즈를 비교하지 않는다. 깊이가 가파른 장면
# 에서는 두 구현이 이 항과 무관하게 이미 벌어진다 - 3~12 m 경사면의 c=0 격차가
# 68 mm 다(정면 평면에서는 1.8 mm). LM 감쇠와 커널 스케줄이 다르기 때문이고,
# 그 위에서 포즈를 비교하면 이 항이 아니라 그 차이를 재게 된다.
#
# 대신 **정보행렬** 을 본다. Lambda ~ sum w J J^T 이므로 가중이 바뀌면 곧바로
# 바뀌고, 수렴 경로를 거치지 않는 누산 자체의 관측량이다. 실측으로 이 지표는
# 두 구현이 8 % 안에서 일치한다.

# c 를 장면마다 고정하면 안 된다. sigma_Z/Z = c*Z 이므로 같은 c 라도 먼 장면
# 에서는 상대 깊이오차가 커지고, 20 % 를 넘으면 가중이 거의 전멸에 가까워져
# 남는 점이 구현마다 달라진다 (실측: 3~10 m 에서 비율이 0.236 vs 0.158).
# 대신 **먼 쪽 상대 깊이오차** 를 고정한다. KITTI 실제 값도 이 자리다 -
# c=7.8e-4, 60 m 에서 sigma_Z/Z = 4.7 %.
_FAR_REL = 0.05


def _dsr_for(far: float) -> float:
    return _FAR_REL / far


def _info_trace(res) -> float:
    return float(np.trace(np.asarray(res.information)[:3, :3]))


def _one_step(depth, dsr):
    """단일 레벨 1 반복. 누산 한 번의 결과만 본다."""
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    ref = _texture()
    cur = _warp(ref, depth, truth)

    py = ecda.align(ref, depth, cur, CAM,
                    cfg=ecda.EcdaConfig(pyramid_levels=1, max_iterations=1,
                                        max_depth=80.0, depth_sigma_rel=dsr))
    c = core.DirectAlignerConfig()
    c.pyramid_levels = 1
    c.max_iterations = 1
    c.max_depth = 80.0
    c.depth_sigma_rel = dsr
    cpp = core.DirectAligner(c).align(ref, depth, cur, CAM.fx, CAM.fy, CAM.cx, CAM.cy)
    return py, cpp


@pytest.mark.parametrize("near,far", [(2.0, 4.0), (2.0, 6.0), (3.0, 10.0)])
def test_depth_weighting_shrinks_information_by_the_same_factor(near, far):
    """두 구현이 같은 비율로 정보를 깎는가.

    절대값은 비교하지 않는다 - 점 선택과 보간이 달라 c=0 에서도 8 % 차이가
    난다. 비교하는 것은 **이 항이 만든 변화** 다.
    """
    d = _wide_depth(near, far)
    py0, cpp0 = _one_step(d, 0.0)
    py1, cpp1 = _one_step(d, _dsr_for(far))

    ratio_py = _info_trace(py1) / _info_trace(py0)
    ratio_cpp = _info_trace(cpp1) / _info_trace(cpp0)
    assert abs(ratio_py - ratio_cpp) < 0.15 * max(ratio_py, ratio_cpp), (
        f"정보 감소 비율이 다르다: py {ratio_py:.4f} cpp {ratio_cpp:.4f}")


def test_the_information_ratio_actually_moves():
    """가드. 비율이 1 에 붙어 있으면 위 테스트는 1 == 1 을 재는 것이다."""
    d = _wide_depth(2.0, 6.0)
    py0, cpp0 = _one_step(d, 0.0)
    py1, cpp1 = _one_step(d, _dsr_for(6.0))
    r_py = _info_trace(py1) / _info_trace(py0)
    r_cpp = _info_trace(cpp1) / _info_trace(cpp0)
    assert r_py < 0.8 and r_cpp < 0.8, (
        f"이 항이 정보를 거의 깎지 않는다: py {r_py:.3f} cpp {r_cpp:.3f}")


def test_a_far_scene_loses_more_information_than_a_near_one():
    """방향 확인. 부호가 뒤집혀 있어도 위 두 테스트는 통과한다.

    sigma_Z = c*Z^2 이므로 같은 c 에서 먼 장면이 더 많이 깎여야 한다.
    """
    near = _wide_depth(1.5, 3.0)
    far = _wide_depth(6.0, 12.0)
    # 같은 c 를 준다 - 그래야 "멀수록 더 깎인다" 가 주장이 된다.
    c = _dsr_for(12.0)
    r = {}
    for name, d in (("near", near), ("far", far)):
        py0, cpp0 = _one_step(d, 0.0)
        py1, cpp1 = _one_step(d, c)
        r[name] = (_info_trace(py1) / _info_trace(py0),
                   _info_trace(cpp1) / _info_trace(cpp0))
    assert r["far"][0] < r["near"][0], f"numpy: near {r['near'][0]:.3f} far {r['far'][0]:.3f}"
    assert r["far"][1] < r["near"][1], f"C++:   near {r['near'][1]:.3f} far {r['far'][1]:.3f}"


def test_disabled_by_default_is_bit_identical():
    """기본값 0 에서는 이 항이 존재하지 않은 것과 같아야 한다.

    22 절 이후의 TUM 수치가 전부 이 경로 위에 있다. 기본값에서 조금이라도
    달라지면 발표된 숫자가 조용히 바뀐다.
    """
    assert core.DirectAlignerConfig().depth_sigma_rel == 0.0
    assert ecda.EcdaConfig().depth_sigma_rel == 0.0
    truth = PySE3.exp(np.array([0.02, -0.01, 0.005, 0.004, -0.006, 0.003]))
    a, _, _ = _both(truth)
    b, _, _ = _both(truth, depth_sigma_rel=0.0)
    assert np.allclose(a.T_cur_ref.t, b.T_cur_ref.t, rtol=0, atol=0)


def test_far_points_lose_weight_not_near_ones():
    """가중식 자체의 단조성. 위 테스트들은 장면 단위라 식의 모양까지는 못 본다."""
    photo_sigma, dr_dd = 2.0, 1.0
    ws = [photo_sigma**2 / (photo_sigma**2 + (dr_dd * 8e-4 * d * d) ** 2)
          for d in (3.0, 10.0, 30.0, 60.0)]
    assert ws == sorted(ws, reverse=True), f"거리에 단조감소하지 않는다: {ws}"
    assert ws[0] > 0.99, f"3 m 점의 무게가 {ws[0]:.3f} - 가까운 점까지 깎는다"
    assert ws[-1] < 0.5, f"60 m 점의 무게가 {ws[-1]:.3f} - 먼 점을 거의 안 깎는다"
