"""C++ StructuralAligner (SPA, Tier 2) 대 numpy 오라클.

26 절이 "PoseFusion 은 커버리지가 생겼지만 SPA / PlaneExtractor / TierInformation
은 여전히 0" 으로 남겨 둔 마지막 조각이다.

이 서브시스템에는 전례가 있다. **7.1 의 결함 - 회전 정보행렬이 진리의 직교
여집합이었던 것 - 이 정확히 이 코드에서 나왔다.** `Sum w n n^T` 를 썼는데 좌측
섭동 규약에서 올바른 값은 `Sum w (I - n n^T)` 이고, 둘은 서로 보완하는 부분공간
위의 사영이다. x 축에서 21 배 틀렸고 방향이 반대였다. 그때는 오라클이 없어서
유한차분으로 잡아야 했다.

여기서 요구하는 것은 두 층이다:
  1. 두 구현이 같은 답을 내는가 (차분)
  2. 그 답이 **유한차분과 맞는가** (판별 가드) - 두 구현이 같이 틀리면 1 만으로는
     통과한다. 7.1 이 바로 그렇게 오래 살아남았다.
"""

from __future__ import annotations

import numpy as np
import pytest

from wme import HAS_NATIVE

if HAS_NATIVE:
    from wme import _core as core

from wme.geometry import spa as pyspa
from wme.geometry.planes import Plane as PyPlane

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="_core 미빌드")


def _pair(normals, dists, inliers=400, rms=0.0, extent=0.5):
    """같은 평면 집합을 두 구현의 타입으로 만든다."""
    py, cpp = [], []
    for n, d in zip(normals, dists):
        n = np.asarray(n, float)
        n = n / np.linalg.norm(n)
        py.append(PyPlane(n, float(d), inliers, np.zeros(3), extent, rms))
        cpp.append(core.Plane(n, float(d), inliers, np.zeros(3), extent, rms))
    return py, cpp


ROOM = ([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [3.0, 2.0, 1.5])
CORNER = ([[0.1, 0.2, 0.97], [0.98, 0.1, 0.1], [0.05, 0.99, 0.1]], [2.5, 1.8, 1.2])


# ===========================================================================
# 1. 랭크 보고 - 7.1 이 뒤집혔던 바로 그 출력
# ===========================================================================

@pytest.mark.parametrize("scene,name", [(ROOM, "room"), (CORNER, "corner")])
def test_rank_report_matches(scene, name):
    py_pl, cpp_pl = _pair(*scene)
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    assert ok
    pres = pyspa.align(py_pl, py_pl)

    assert res.rotation_rank == pres.rotation_rank, (
        f"{name}: 회전 랭크 native {res.rotation_rank} vs ref {pres.rotation_rank}")
    assert res.translation_rank == pres.translation_rank
    assert res.observable_dof == pres.observable_dof
    # 세 직교 평면이면 6 자유도가 전부 관측된다. 이 값이 6 이 아니면 위의
    # 세 비교는 "둘 다 같은 정도로 퇴화했다" 를 확인한 것에 지나지 않는다.
    if name == "room":
        assert res.observable_dof == 6, "직교 3평면인데 6 DOF 가 아니다 - 비교가 무의미"


# ===========================================================================
# 2. 정보행렬
# ===========================================================================

@pytest.mark.parametrize("inliers", [400, 200, 100])
@pytest.mark.parametrize("scene", [ROOM, CORNER])
def test_information_matrix_matches(scene, inliers):
    """두 구현의 정보행렬이 같아야 한다.

    이 테스트는 붙자마자 **두 개의** 결함을 찾았고 둘 다 파이썬 쪽이었다:

      1. 회전 블록이 산포 `sum w n n^T` 였다 (올바른 값은 여집합 `sum w (I-n n^T)`).
         7.1 이 기록한 바로 그 결함인데, 그때 고쳐진 것은 발표된 표뿐이고
         참조 구현은 그대로였다. 직교 3평면에서 정확히 2 배로 나타난다.
      2. 병진에서 가중치가 **두 번** 적용됐다 (`sum w^2 n n^T`). 가중최소제곱은
         잔차를 sqrt(w) 로 스케일해야 한다.

    inliers 를 매개변수로 둔 이유가 2 번이다. inliers=400 이면 weight 가 정확히
    1 이라 w 와 w^2 이 구분되지 않는다 - 그 값 하나로만 시험하면 이 결함은
    보이지 않는다.
    """
    py_pl, cpp_pl = _pair(*scene, inliers=inliers)
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    assert ok
    pres = pyspa.align(py_pl, py_pl)
    L_cpp = np.asarray(res.information)
    L_py = np.asarray(pres.information)
    assert L_cpp.shape == (6, 6)
    assert np.allclose(L_cpp, L_py, rtol=1e-9, atol=0), (
        f"max abs diff {np.abs(L_cpp - L_py).max():.3e}")


def test_rotation_block_is_the_complement_not_the_scatter():
    """판별 가드, 그리고 7.1 을 코드에 고정하는 검사.

    두 구현이 **같이** `sum n n^T` 로 돌아가도 값 비교는 통과한다. 그래서 형태
    자체를 독립적으로 못박는다.

    장면 선택이 이 테스트의 전부다. 직교 3평면에서는 여집합이 `2I` 로 등방이라
    어떤 축을 봐도 같은 값이 나오고 산포와 구분되지 않는다 - 처음 이 테스트를
    그렇게 썼고 `5000 > 5000` 으로 실패했다. 평면 **둘**(z, x)이면 갈린다:

        여집합  2I - zz^T - xx^T = diag(1, 2, 1)   -> y 축이 최대
        산포    zz^T + xx^T       = diag(1, 0, 1)   -> y 축이 0

    즉 **어떤 법선과도 수직인 축(y)** 의 회전 정보가 최대여야 한다. 산포라면
    거기가 정확히 0 이다.
    """
    _, cpp_pl = _pair([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], [3.0, 2.0])
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    assert ok
    R = np.asarray(res.information)[3:, 3:]
    along_normal = float(np.array([0.0, 0, 1]) @ R @ np.array([0.0, 0, 1]))
    perp_to_all = float(np.array([0.0, 1, 0]) @ R @ np.array([0.0, 1, 0]))
    assert perp_to_all > along_normal, (
        f"법선과 수직인 축의 회전정보 {perp_to_all:.4g} 가 법선 축 "
        f"{along_normal:.4g} 보다 크지 않다 - 7.1 의 scatter/complement "
        "뒤집힘이 돌아왔다")
    # 산포라면 y 축이 0 이다. 0 이 아님을 직접 요구한다.
    assert perp_to_all > 1e-9


def test_python_reference_agrees_on_the_complement_form():
    """같은 검사를 파이썬 참조에도 건다. 한쪽만 고치면 차분 테스트는 통과하고
    두 구현이 나란히 틀린 상태로 남을 수 있다."""
    py_pl, _ = _pair([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], [3.0, 2.0])
    pres = pyspa.align(py_pl, py_pl)
    R = np.asarray(pres.information)[3:, 3:]
    along = float(np.array([0.0, 0, 1]) @ R @ np.array([0.0, 0, 1]))
    perp = float(np.array([0.0, 1, 0]) @ R @ np.array([0.0, 1, 0]))
    assert perp > along and perp > 1e-9


# ===========================================================================
# 3. 대응과 잔차
# ===========================================================================

def test_match_pairs_agree():
    py_pl, cpp_pl = _pair(*ROOM)
    m_cpp = core.StructuralAligner().match(cpp_pl, cpp_pl)
    m_py = pyspa.match_planes(py_pl, py_pl)
    assert len(m_cpp) == len(m_py) == 3
    # 파이썬 PlaneMatch 는 인덱스가 아니라 Plane 객체를 들고 있다. 같은 대응을
    # 비교하려면 법선으로 되짚어야 한다.
    def idx(pl):
        # Plane 은 numpy 배열을 들고 있어 == 비교가 배열을 낸다. list.index 를
        # 쓰면 ValueError 로 죽는다 - 법선으로 되짚는다.
        for i, q in enumerate(py_pl):
            if np.allclose(q.normal, pl.normal, rtol=0, atol=1e-12):
                return i
        raise AssertionError("대응 평면을 되짚지 못했다")
    got = sorted((m.ref_index, m.cur_index) for m in m_cpp)
    exp = sorted((idx(m.reference), idx(m.current)) for m in m_py)
    assert got == exp


def test_identity_alignment_is_identity():
    """같은 평면 집합이면 상대 포즈는 항등이어야 한다. 이게 깨지면 랭크나
    정보행렬 비교는 볼 필요가 없다."""
    _, cpp_pl = _pair(*CORNER)
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    assert ok
    assert np.allclose(np.asarray(res.T_cur_ref.t), 0.0, rtol=0, atol=1e-9)
    assert np.allclose(np.asarray(res.T_cur_ref.R), np.eye(3), rtol=0, atol=1e-9)
    assert res.normal_rms < 1e-9 and res.offset_rms < 1e-9


# ===========================================================================
# 4. 퇴화 - 7.1 이 "그럴듯한 숫자" 를 냈던 구간
# ===========================================================================

def test_single_plane_degeneracy_agrees():
    """평면 하나면 회전 2 / 병진 1 만 관측된다. 여기서 두 구현이 갈리면
    상보성 판정(18.3)이 문턱 차이를 재게 된다."""
    py_pl, cpp_pl = _pair([[0.0, 0.0, 1.0]], [3.0])
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    pres = pyspa.align(py_pl, py_pl)
    if not ok:
        pytest.skip("C++ 이 평면 1 개를 기각한다 - 규약 차이지 값 불일치가 아니다")
    assert res.rotation_rank == pres.rotation_rank
    assert res.translation_rank == pres.translation_rank
    assert res.observable_dof < 6, "평면 하나로 6 DOF 가 나오면 랭크 보고가 고장난 것"


def test_weakest_direction_is_a_unit_vector_in_both():
    py_pl, cpp_pl = _pair(*CORNER)
    ok, res = core.StructuralAligner().align(cpp_pl, cpp_pl)
    assert ok
    w = np.asarray(res.weakest_direction)
    assert abs(np.linalg.norm(w) - 1.0) < 1e-9, "단위벡터가 아니다"
    # 좌표축이 아니라 실제 고유벡터여야 한다 (7.1 의 세 번째 원인).
    axes = np.eye(6)
    assert not any(np.allclose(np.abs(w), a, rtol=0, atol=1e-12) for a in axes), (
        "weakest_direction 이 좌표축이다 - 고유벡터가 아니라 np.eye 를 돌려주고 있다")


# ===========================================================================
# 5. PlaneExtractor - 26 절이 커버리지 0 으로 남겨 둔 나머지
# ===========================================================================

def _synthetic_depth(w=160, h=120, fx=120.0):
    """두 평면(정면 벽 + 바닥)이 있는 합성 깊이맵.

    실데이터를 쓰지 않는 이유는, 여기서 재는 것이 "두 구현이 같은 깊이맵에서
    같은 평면을 뽑는가" 이지 추출기의 절대 성능이 아니기 때문이다.
    """
    cam_cx, cam_cy = w / 2 - 0.5, h / 2 - 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rx = (xx - cam_cx) / fx
    ry = (yy - cam_cy) / fx

    def plane_depth(n, dist):
        n = np.asarray(n, float) / np.linalg.norm(n)
        return dist / (n[0] * rx + n[1] * ry + n[2])

    wall = plane_depth([0.0, 0.0, 1.0], 3.0)
    floor = plane_depth([0.0, 0.9, 0.44], 2.0)
    # 아래쪽 절반은 바닥, 위쪽은 벽.
    d = np.where(yy > h * 0.55, floor, wall)
    d[(d < 0.2) | (d > 15.0)] = 0.0
    return np.ascontiguousarray(d.astype(np.float32)), fx, cam_cx, cam_cy


def test_plane_extractor_finds_the_same_planes():
    """같은 깊이맵에서 같은 평면 집합이 나와야 한다.

    비트 동일은 요구하지 않는다 - 격자 순회 순서와 부동소수점 누산이 다르므로
    법선이 미세하게 갈릴 수 있다. 요구하는 것은 (a) 개수가 같고 (b) 각 평면의
    법선과 거리가 물리적으로 같은 평면을 가리키는가다.
    """
    from wme.geometry.planes import PlaneConfig, extract_planes
    from wme.sim.world import CameraModel

    d, fx, cx, cy = _synthetic_depth()
    cpp = core.PlaneExtractor().extract(d, fx, fx, cx, cy)
    cam = CameraModel(fx=fx, fy=fx, cx=cx, cy=cy, width=d.shape[1], height=d.shape[0])
    py = extract_planes(d.astype(np.float64), cam, PlaneConfig())

    assert len(cpp) >= 2, f"C++ 이 평면을 {len(cpp)} 개만 찾았다 - 장면이 잘못됐다"
    assert len(py) >= 2, f"파이썬이 평면을 {len(py)} 개만 찾았다"

    # 신뢰도 순 상위 2 개를 법선 방향으로 짝짓는다.
    for a in cpp[:2]:
        best = min(py, key=lambda b: 1.0 - abs(float(np.asarray(a.normal) @ b.normal)))
        cos = abs(float(np.asarray(a.normal) @ best.normal))
        assert cos > 0.99, f"법선 불일치 cos={cos:.4f}"
        # 법선 부호 규약이 뒤집힐 수 있으므로 거리는 절대값으로 본다.
        assert abs(abs(a.distance) - abs(best.distance)) < 0.05, (
            f"거리 불일치 {a.distance:.3f} vs {best.distance:.3f}")
        # 유도한 오프셋 sigma 도 같아야 한다. 이것이 빠지면 두 구현이 같은
        # 평면을 뽑고도 정보행렬은 서로 다른 값을 내게 된다.
        assert a.sigma_offset > 0.0, "sigma_offset 이 유도되지 않았다"
        assert abs(a.sigma_offset - best.sigma_offset) < a.sigma_offset * 0.05, (
            f"sigma_offset 불일치 {a.sigma_offset:.6f} vs {best.sigma_offset:.6f}")


def test_plane_sigma_follows_the_derivation_in_both():
    """판별 가드. sigma_offset = sqrt((c z d)^2 + rms^2) 인가를 두 구현에서 따로 본다.

    위 테스트는 "둘이 같다" 만 말한다. 둘이 같이 c z^2 을 쓰거나 sqrt(N) 으로
    나눠도 통과한다 - 7.1 의 회전 블록이 정확히 그렇게 오래 살아남았다.
    여기서는 형태 자체를 못박는다.
    """
    from wme.geometry.planes import PlaneConfig, extract_planes
    from wme.sim.world import CameraModel

    d, fx, cx, cy = _synthetic_depth()
    cfg = PlaneConfig()
    cam = CameraModel(fx=fx, fy=fx, cx=cx, cy=cy, width=d.shape[1], height=d.shape[0])
    py = extract_planes(d.astype(np.float64), cam, cfg)
    ccfg = core.PlaneExtractorConfig()
    ccfg.depth_sigma_rel = cfg.depth_sigma_rel
    cpp = core.PlaneExtractor(ccfg).extract(d, fx, fx, cx, cy)

    c = cfg.depth_sigma_rel
    assert c > 0.0
    for pl in list(py) + list(cpp):
        z = float(np.asarray(pl.centroid)[2])
        if z <= 0.0:
            continue
        want = float(np.hypot(c * z * pl.distance, pl.rms))
        assert abs(pl.sigma_offset - want) < max(1e-12, want * 1e-9), (
            f"{type(pl).__name__}: sigma_offset {pl.sigma_offset:.8f} != "
            f"sqrt((c z d)^2 + rms^2) {want:.8f}")
        # 거리 의존이 살아 있는가. c z^2 로 잘못 쓰면 이 비가 1 이 된다.
        derived = float(np.sqrt(max(0.0, pl.sigma_offset ** 2 - pl.rms ** 2)))
        assert derived > 0.0
        assert abs(derived / (c * z * z) - pl.distance / z) < 1e-9, "투영 인자 d/z 가 빠졌다"


def test_information_uses_the_plane_sigma_not_the_fallback():
    """평면이 sigma_offset 을 들고 오면 config 상수가 아니라 그것이 쓰여야 한다.

    두 구현 모두에서 확인한다. 거리만 열 배 다른 두 장면을 견주면 병진 정보가
    1e4 배 갈려야 한다 - 26.x 까지는 정확히 1 배였다(거리 무관).
    """
    from wme.geometry.planes import Plane as PP

    c = 0.006
    normals = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def build(d):
        py, cpp = [], []
        # 세 평면 모두 같은 sigma 를 준다. 거리 의존만 남겨서 다른 요인이
        # 섞이지 않게 한다 (정면 평면이면 centroid.z = d 라 c z d = c d^2 이다).
        sig = c * d * d
        for nv in normals:
            n = np.asarray(nv, float)
            cen = n * d
            py.append(PP(n, float(d), 400, cen, 0.5, 0.0, float(sig)))
            cpp.append(core.Plane(n, float(d), 400, cen, 0.5, 0.0, float(sig)))
        return py, cpp

    py1, cpp1 = build(1.0)
    py10, cpp10 = build(10.0)

    ok1, r1 = core.StructuralAligner().align(cpp1, cpp1)
    ok10, r10 = core.StructuralAligner().align(cpp10, cpp10)
    assert ok1 and ok10
    t1 = np.trace(np.asarray(r1.information)[:3, :3])
    t10 = np.trace(np.asarray(r10.information)[:3, :3])
    assert abs(t1 / t10 / 1e4 - 1.0) < 1e-6, f"C++ 병진 정보 비 {t1 / t10:.4g}, 기대 1e4"

    p1 = pyspa.align(py1, py1)
    p10 = pyspa.align(py10, py10)
    q1 = np.trace(np.asarray(p1.information)[:3, :3])
    q10 = np.trace(np.asarray(p10.information)[:3, :3])
    assert abs(q1 / q10 / 1e4 - 1.0) < 1e-6, f"파이썬 병진 정보 비 {q1 / q10:.4g}, 기대 1e4"
    # 두 구현이 같은 값을 내는가 (항등 정합이라 t = 0, 교차항이 사라져 비교 가능)
    assert np.allclose(np.asarray(r1.information), np.asarray(p1.information), rtol=1e-9)


def test_plane_extractor_rejects_a_sphere():
    """판별 가드. 평면이 아닌 장면에서 평면이 쏟아지면 위 테스트는 '둘 다 같은
    쓰레기를 뽑았다' 를 확인한 것에 지나지 않는다."""
    w, h, fx = 160, 120, 120.0
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    r = np.sqrt(((xx - cx) / fx) ** 2 + ((yy - cy) / fx) ** 2)
    # 강하게 휜 면 - 어느 국소 패치도 평면이 아니다.
    d = (1.5 + 2.0 * r ** 2).astype(np.float32)
    cpp = core.PlaneExtractor().extract(np.ascontiguousarray(d), fx, fx, cx, cy)
    assert len(cpp) <= 1, f"곡면에서 평면 {len(cpp)} 개가 나왔다 - planarity 게이트가 죽었다"
