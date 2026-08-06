"""렌더러 검증.

렌더러가 기하학적으로 틀리면 ECDA 검증 전체가 무의미하다. 투영-역투영
왕복과 두 시점 사이의 워프 일관성으로 고정한다.
"""

import numpy as np
import pytest

from wme.reference.environment import Evidence
from wme.reference.geometry import SE3, so3_exp
from wme.sim.render import (
    Box, Quad, RenderScene, apply_degradation, procedural_texture, render, render_frame,
)
from wme.sim.world import CameraModel

CAM = CameraModel(fx=200.0, fy=200.0, cx=127.5, cy=95.5, width=256, height=192)


def room_scene():
    return RenderScene.room(size=4.0, height=2.4)


def inside_pose(t=None, rot=None):
    return SE3(so3_exp(rot if rot is not None else [0.0, 0.0, 0.0]),
               np.array(t if t is not None else [0.0, 0.0, 1.2]))


# --- 텍스처 ---------------------------------------------------------------

def test_texture_is_deterministic_and_bounded():
    p = np.random.default_rng(0).uniform(-5, 5, (500, 3))
    a = procedural_texture(p, 3)
    b = procedural_texture(p, 3)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_texture_has_gradient_everywhere():
    """직접정렬은 그래디언트가 없는 곳에서 아무것도 못 한다."""
    x = np.linspace(-3, 3, 400)
    p = np.stack([x, np.zeros_like(x), np.full_like(x, 1.0)], axis=1)
    d = np.abs(np.diff(procedural_texture(p, 1)))
    # 400 표본 중 평탄 구간이 극히 일부여야 한다
    assert np.mean(d < 1e-4) < 0.05


def test_texture_differs_by_seed():
    p = np.random.default_rng(1).uniform(-3, 3, (200, 3))
    assert not np.allclose(procedural_texture(p, 1), procedural_texture(p, 2))


# --- 기하 -----------------------------------------------------------------

def test_quad_intersection_depth_is_exact():
    """z=1 규약이므로 t 가 곧 z-깊이여야 한다."""
    floor = Quad(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                 np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 5.0, 5.0)
    origin = np.array([0.0, 0.0, 2.0])
    dirs = np.array([[0.0, 0.0, -1.0]])
    assert floor.intersect(origin, dirs)[0] == pytest.approx(2.0)


def test_quad_bounds_are_respected():
    small = Quad(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                 np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.5, 0.5)
    origin = np.array([0.0, 0.0, 2.0])
    inside = small.intersect(origin, np.array([[0.0, 0.0, -1.0]]))
    outside = small.intersect(origin, np.array([[1.0, 0.0, -1.0]]))
    assert np.isfinite(inside[0])
    assert not np.isfinite(outside[0])


def test_box_intersection_hits_near_face():
    box = Box(np.array([0.0, 0.0, 5.0]), np.array([1.0, 1.0, 1.0]))
    t = box.intersect(np.zeros(3), np.array([[0.0, 0.0, 1.0]]))
    assert t[0] == pytest.approx(4.0)


def test_box_miss_returns_inf():
    box = Box(np.array([0.0, 0.0, 5.0]), np.array([0.5, 0.5, 0.5]))
    t = box.intersect(np.zeros(3), np.array([[5.0, 0.0, 1.0]]))
    assert not np.isfinite(t[0])


def test_depth_matches_analytic_distance_to_wall():
    """정면 벽까지의 깊이는 해석적으로 계산된다."""
    scene = RenderScene(quads=[
        Quad(np.array([0.0, 3.0, 0.0]), np.array([0.0, -1.0, 0.0]),
             np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 5.0, 5.0)
    ])
    # 카메라 z축이 +y 를 향하도록
    R = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                         np.array([0, 1.0, 0])])
    out = render(scene, SE3(R, np.array([0.0, 0.0, 0.0])), CAM)

    centre = out.depth[CAM.height // 2, CAM.width // 2]
    assert centre == pytest.approx(3.0, abs=0.02)


def test_reprojection_between_two_views_is_consistent():
    """한 시점의 픽셀을 역투영해 다른 시점으로 투영하면 그 깊이와 맞아야 한다.

    이 성질이 깨지면 ECDA 가 무엇을 최소화하든 의미가 없다.
    """
    scene = room_scene()
    pose_a = inside_pose([0.0, 0.0, 1.2])
    pose_b = inside_pose([0.25, 0.1, 1.2], [0.0, 0.0, 0.05])

    a = render(scene, pose_a, CAM)
    b = render(scene, pose_b, CAM)

    T_ba = pose_b.inverse() @ pose_a
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(400):
        y = int(rng.integers(20, CAM.height - 20))
        x = int(rng.integers(20, CAM.width - 20))
        d = a.depth[y, x]
        if not np.isfinite(d) or d > 20.0:
            continue
        p_cam = np.array([(x - CAM.cx) * d / CAM.fx, (y - CAM.cy) * d / CAM.fy, d])
        q = T_ba @ p_cam
        if q[2] <= 0.2:
            continue
        u = CAM.fx * q[0] / q[2] + CAM.cx
        v = CAM.fy * q[1] / q[2] + CAM.cy
        if not (2 <= u < CAM.width - 2 and 2 <= v < CAM.height - 2):
            continue
        observed = b.depth[int(round(v)), int(round(u))]
        if not np.isfinite(observed):
            continue
        # 가려짐이 있을 수 있으므로 대부분이 맞으면 된다
        if abs(observed - q[2]) < 0.05:
            checked += 1
    assert checked > 150, f"일관된 픽셀이 {checked}개뿐"


def test_object_ids_are_recorded():
    scene = RenderScene(quads=room_scene().quads,
                        boxes=[Box(np.array([0.0, 2.0, 1.0]),
                                   np.array([0.4, 0.4, 0.4]), object_id=7)])
    R = np.column_stack([np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                         np.array([0, 1.0, 0])])
    out = render(scene, SE3(R, np.array([0.0, 0.0, 1.0])), CAM)
    assert 7 in np.unique(out.object_ids)


# --- 열화 -----------------------------------------------------------------

def test_fog_is_depth_dependent():
    """안개는 전역 스칼라가 아니다. 먼 픽셀이 먼저 대기광에 묻힌다.

    이 성질이 없으면 alpha_0(E) 검증이 무의미해진다.
    """
    scene = room_scene()
    pose = inside_pose([0.0, 0.0, 1.2], [0.0, 0.3, 0.0])
    raw = render(scene, pose, CAM)

    rng = np.random.default_rng(0)
    foggy = apply_degradation(raw, Evidence(haze=0.8), rng)

    d = raw.depth
    finite = np.isfinite(d)
    near = finite & (d < np.nanpercentile(d[finite], 25))
    far = finite & (d > np.nanpercentile(d[finite], 75))

    # 원본 대비 변화량이 먼 쪽에서 더 커야 한다
    near_shift = np.mean(np.abs(foggy[near] - raw.gray[near]))
    far_shift = np.mean(np.abs(foggy[far] - raw.gray[far]))
    assert far_shift > near_shift * 1.3


def test_fog_reduces_contrast():
    scene = room_scene()
    raw = render(scene, inside_pose([0.0, 0.0, 1.2], [0.0, 0.3, 0.0]), CAM)
    rng = np.random.default_rng(0)
    foggy = apply_degradation(raw, Evidence(haze=0.9), rng)
    assert foggy.std() < raw.gray.std()


def test_darkness_reduces_brightness():
    scene = room_scene()
    raw = render(scene, inside_pose(), CAM)
    dark = apply_degradation(raw, Evidence(darkness=0.8), np.random.default_rng(0))
    assert dark.mean() < raw.gray.mean() * 0.5


def test_motion_blur_monotonically_reduces_gradient_energy():
    """블러 강도에 따라 그래디언트 에너지가 단조 감소해야 한다.

    감소 폭의 절대값은 텍스처 스펙트럼에 달려 있어 고정할 수 없다.
    물리적으로 요구되는 성질은 단조성이므로 그것을 검증한다.
    """
    scene = room_scene()
    raw = render(scene, inside_pose([0.0, 0.0, 1.2], [0.0, 0.3, 0.0]), CAM)

    def grad_energy(img):
        return float(np.mean(np.abs(np.diff(img, axis=1))))

    energies = [grad_energy(apply_degradation(raw, Evidence(motion_blur=a),
                                              np.random.default_rng(0)))
                for a in (0.0, 0.25, 0.5, 0.75, 1.0)]

    assert all(a >= b for a, b in zip(energies, energies[1:])), energies
    assert energies[-1] < energies[0] * 0.75, "극심한 블러인데 변화가 미미하다"


def test_lens_smudge_is_screen_fixed():
    """렌즈 오염은 카메라가 움직여도 화면상 같은 자리에 남아야 한다."""
    scene = room_scene()
    rng = np.random.default_rng(0)
    a = apply_degradation(render(scene, inside_pose([0.0, 0.0, 1.2]), CAM),
                          Evidence(lens_dirt=0.9), rng)
    b = apply_degradation(render(scene, inside_pose([0.6, 0.3, 1.2]), CAM),
                          Evidence(lens_dirt=0.9), rng)
    clean_a = render(scene, inside_pose([0.0, 0.0, 1.2]), CAM).gray
    clean_b = render(scene, inside_pose([0.6, 0.3, 1.2]), CAM).gray

    h, w = a.shape
    box = (slice(int(h * 0.5), int(h * 0.75)), slice(int(w * 0.18), int(w * 0.4)))
    # 두 시점 모두 같은 영역에서 원본 대비 변화가 커야 한다
    assert np.mean(np.abs(a[box] - clean_a[box])) > np.mean(np.abs(a - clean_a))
    assert np.mean(np.abs(b[box] - clean_b[box])) > np.mean(np.abs(b - clean_b))


def test_render_frame_is_deterministic():
    scene = room_scene()
    pose = inside_pose([0.1, 0.0, 1.2])
    a = render_frame(scene, pose, CAM, Evidence(darkness=0.5), seed=3)
    b = render_frame(scene, pose, CAM, Evidence(darkness=0.5), seed=3)
    assert np.array_equal(a.gray, b.gray)
