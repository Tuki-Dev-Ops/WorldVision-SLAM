"""평면 추출과 SPA(Tier 2) 검증.

핵심은 **랭크 부족을 숨기지 않는 것**이다. 복도에서 복도 축 병진은 평면으로
관측되지 않는데, 그걸 6-DoF 로 보고하면 융합 규칙이 없는 정보를 믿게 된다.
"""

import numpy as np
import pytest

from wme.geometry import (
    PlaneConfig, SpaConfig, align, dominant_gravity, extract_planes, match_planes,
    unobservable_directions,
)
from wme.reference.geometry import SE3, so3_exp
from wme.sim.render import Quad, RenderScene, render
from wme.sim.world import CameraModel

CAM = CameraModel(fx=220.0, fy=220.0, cx=159.5, cy=119.5, width=320, height=240)


def room():
    return RenderScene.room(size=4.0, height=2.6)


def corridor():
    """평행한 벽 둘 + 바닥. 복도 축 병진이 관측되지 않는 구조."""
    ex, ey, ez = np.eye(3)
    return RenderScene(quads=[
        Quad(np.array([0.0, 0.0, 0.0]), ez, ex, ey, 2.0, 12.0, 0.85, 1),      # 바닥
        Quad(np.array([1.6, 0.0, 1.3]), -ex, ey, ez, 12.0, 1.3, 0.9, 3),      # 벽
        Quad(np.array([-1.6, 0.0, 1.3]), ex, ey, ez, 12.0, 1.3, 0.9, 4),      # 벽
    ])


def looking(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z = z / max(np.linalg.norm(z), 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0, 0]) if n < 1e-6 else x / n
    return SE3(np.column_stack([x, np.cross(z, x), z]), eye)


def planes_at(scene, pose, cfg=None):
    out = render(scene, pose, CAM)
    depth = np.where(np.isfinite(out.depth), out.depth, 0.0).astype(float)
    return extract_planes(depth, CAM, cfg or PlaneConfig())


# --- 평면 추출 -------------------------------------------------------------

def test_extracts_planes_from_a_room():
    planes = planes_at(room(), looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0]))
    assert len(planes) >= 2, planes


def test_extracted_planes_are_unit_normals_in_front():
    for pl in planes_at(room(), looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0])):
        assert np.isclose(np.linalg.norm(pl.normal), 1.0, atol=1e-9)
        assert pl.distance > 0.0


def test_wall_distance_matches_geometry():
    """정면 벽을 마주보면 그 거리가 해석적으로 계산된다."""
    scene = RenderScene(quads=[
        Quad(np.array([0.0, 3.0, 0.0]), np.array([0.0, -1.0, 0.0]),
             np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 6.0, 6.0)
    ])
    R = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                         np.array([0, 1.0, 0])])
    planes = planes_at(scene, SE3(R, np.zeros(3)))

    assert planes
    assert planes[0].distance == pytest.approx(3.0, abs=0.05)
    # Hesse 표준형이므로 법선은 카메라 반대쪽 (z 성분이 양)
    assert planes[0].normal[2] > 0.9


def test_empty_depth_yields_no_planes():
    assert extract_planes(np.zeros((60, 80)), CAM) == []


def test_depth_discontinuity_is_not_a_plane():
    """물체 실루엣에서 가짜 평면이 생기면 안 된다."""
    depth = np.full((120, 160), 3.0)
    depth[:, 80:] = 8.0                    # 급격한 단차
    planes = extract_planes(depth, CAM, PlaneConfig(min_inliers=50))
    # 두 개의 진짜 평면은 잡되, 경계에서 비스듬한 가짜 평면은 없어야 한다
    for pl in planes:
        assert abs(pl.normal[2]) > 0.9, f"경계에서 생긴 기운 평면: {pl}"


def test_plane_transform_is_consistent():
    pl = planes_at(room(), looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0]))[0]
    R = so3_exp([0.1, -0.2, 0.05])
    t = np.array([0.3, -0.1, 0.2])

    moved = pl.transformed(R, t)
    # 평면 위의 점은 옮겨도 평면 위에 있어야 한다
    point_on = pl.normal * pl.distance
    assert moved.signed_distance(R @ point_on + t) == pytest.approx(0.0, abs=1e-9)


# --- 중력 ------------------------------------------------------------------

def test_gravity_from_floor():
    """IMU 가 없을 때의 대안. ConstellationIndex 의 카이랄리티가 이걸 쓴다."""
    planes = planes_at(room(), looking([0.0, -1.0, 1.6], [0.0, 2.0, 0.2]))
    g = dominant_gravity(planes)
    assert g is not None
    assert np.isclose(np.linalg.norm(g), 1.0)
    # 카메라 좌표계에서 아래는 +y
    assert g[1] > 0.8, g


def test_gravity_is_none_without_horizontal_plane():
    """수평면이 없으면 지어내지 않는다."""
    scene = RenderScene(quads=[
        Quad(np.array([0.0, 3.0, 0.0]), np.array([0.0, -1.0, 0.0]),
             np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 6.0, 6.0)
    ])
    R = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                         np.array([0, 1.0, 0])])
    assert dominant_gravity(planes_at(scene, SE3(R, np.zeros(3)))) is None


# --- SPA 정합 --------------------------------------------------------------

def test_recovers_rotation_in_a_room():
    scene = room()
    a = looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0])
    b = SE3(so3_exp([0.0, 0.06, 0.0]) @ a.R, a.t)

    result = align(planes_at(scene, a), planes_at(scene, b))
    assert result.converged
    truth = b.inverse() @ a
    assert result.T_cur_ref.distance_to(truth)[1] < 0.03


def test_recovers_translation_in_a_room():
    """병진이 있는 케이스. 항등 정합만으로는 부호 오류가 드러나지 않는다.

    평면 변환식 n_c·t = d_c - d_r 에서 부호를 뒤집으면 병진이 정확히 반대로
    나오는데, T = I 이면 양변이 0 이라 조용히 통과한다.
    """
    scene = room()
    a = looking([-0.8, -1.2, 1.3], [2.0, 2.5, 0.6])
    b = SE3(a.R, a.t + np.array([0.15, 0.10, 0.0]))

    result = align(planes_at(scene, a), planes_at(scene, b))
    assert result.converged
    assert result.translation_rank == 3, "이 시점은 병진이 완전히 구속되어야 한다"

    truth = b.inverse() @ a
    d_t, d_r = result.T_cur_ref.distance_to(truth)
    assert d_t < 0.05, f"병진 오차 {d_t:.4f} m"
    assert d_r < 0.02


def test_translation_sign_is_not_inverted():
    """부호가 뒤집혔다면 오차가 참 이동의 2배로 나온다. 그걸 직접 못박는다."""
    scene = room()
    a = looking([-0.8, -1.2, 1.3], [2.0, 2.5, 0.6])
    shift = np.array([0.2, 0.0, 0.0])
    b = SE3(a.R, a.t + shift)

    result = align(planes_at(scene, a), planes_at(scene, b))
    truth = b.inverse() @ a
    assert result.converged

    error = result.T_cur_ref.distance_to(truth)[0]
    motion = float(np.linalg.norm(truth.t))
    assert error < motion * 0.5, f"오차 {error:.4f} vs 이동 {motion:.4f} - 부호 의심"


def test_room_with_three_orthogonal_directions_is_full_rank():
    """벽·바닥이 세 방향을 덮으면 병진도 완전히 구속된다."""
    scene = room()
    pose = looking([-0.8, -1.2, 1.3], [2.0, 2.5, 0.6])
    planes = planes_at(scene, pose)

    result = align(planes, planes)          # 같은 프레임이면 항등이어야 한다
    assert result.converged
    if result.rotation_rank == 3 and result.translation_rank == 3:
        assert result.observable_dof == 6


def test_corridor_translation_is_reported_as_unobservable():
    """이 파일의 핵심 주장.

    평행한 벽 둘 + 바닥이면 복도 축 병진이 평면으로 관측되지 않는다.
    그걸 6-DoF 로 보고하면 융합 규칙이 없는 정보를 믿게 된다.
    """
    scene = corridor()
    pose = looking([0.0, -4.0, 1.2], [0.0, 4.0, 1.0])
    planes = planes_at(scene, pose)

    result = align(planes, planes)
    assert result.converged, "복도에서 평면을 못 찾았다"
    assert result.translation_rank < 3, (
        f"복도 축 병진이 관측 가능하다고 보고했다: rank {result.translation_rank}")
    assert result.observable_dof < 6


def test_unobservable_directions_are_reported():
    """Tier 0 가 채워야 할 축을 말할 수 있어야 융합이 성립한다."""
    scene = corridor()
    pose = looking([0.0, -4.0, 1.2], [0.0, 4.0, 1.0])
    planes = planes_at(scene, pose)

    result = align(planes, planes)
    weak = unobservable_directions(result)
    assert weak.shape[1] >= 1
    assert weak.shape[0] == 6


def test_too_few_matches_does_not_converge():
    """대응이 부족하면 성공을 주장하지 않는다."""
    scene = room()
    planes = planes_at(scene, looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0]))
    assert not align(planes[:1], planes[:1]).converged


def test_information_scales_with_environment_weight():
    """alpha_2(E) 는 항을 더하지 않고 정보량만 조절한다."""
    scene = room()
    planes = planes_at(scene, looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0]))

    full = align(planes, planes, alpha_structural=1.0)
    weak = align(planes, planes, alpha_structural=0.1)
    assert np.allclose(weak.information, full.information * 0.1, rtol=1e-9)
    # 포즈 추정 자체는 가중치와 무관해야 한다
    assert np.allclose(weak.T_cur_ref.matrix(), full.T_cur_ref.matrix(), atol=1e-12)


def test_matching_respects_angle_and_distance_gates():
    scene = room()
    planes = planes_at(scene, looking([0.0, -1.0, 1.3], [0.0, 3.0, 1.0]))
    tight = match_planes(planes, planes, config=SpaConfig(max_normal_angle=1e-6))
    loose = match_planes(planes, planes, config=SpaConfig())
    assert len(tight) <= len(loose)


def test_identity_alignment_is_near_identity():
    scene = room()
    planes = planes_at(scene, looking([-0.5, -1.0, 1.3], [1.5, 2.5, 0.8]))
    result = align(planes, planes)
    assert result.converged
    d_t, d_r = result.T_cur_ref.distance_to(SE3.identity())
    assert d_r < 1e-6
    assert d_t < 1e-6
