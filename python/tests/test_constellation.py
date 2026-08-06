"""TCG 검증. C++ tests/test_constellation.cpp 와 같은 시나리오."""

import numpy as np
import pytest

from wme.reference.constellation import ConstellationIndex, Config, Node
from wme.reference.geometry import SE3, so3_exp

RNG = np.random.default_rng(4242)


def make_room(n=12, num_classes=5, extent=4.0, sigma=0.03):
    return [
        Node(token_id=i + 1,
             class_id=i % num_classes,
             position=np.array([RNG.uniform(-extent, extent),
                                RNG.uniform(-extent, extent),
                                RNG.uniform(-extent, extent) * 0.3]),
             sigma=sigma)
        for i in range(n)
    ]


def view_from(nodes, T, noise=0.0):
    out = []
    for n in nodes:
        p = T @ n.position
        if noise > 0.0:
            p = p + RNG.normal(0.0, noise, 3)
        out.append(Node(n.token_id, n.class_id, p, max(n.sigma, noise)))
    return out


def test_relocalizes_exact_match():
    idx = ConstellationIndex()
    room = make_room(12, 5, 4.0)
    idx.insert(1, 1.0, SE3.identity(), room)

    T = SE3(so3_exp([0.1, 0.4, -0.2]), np.array([1.5, -2.0, 0.3]))
    m = idx.query(view_from(room, T.inverse()))

    assert m is not None
    d_t, d_r = m.transform.distance_to(T)
    assert d_t < 0.05, d_t
    assert d_r < 0.02, d_r
    assert m.score > 0.5


def test_robust_to_position_noise():
    idx = ConstellationIndex()
    room = make_room(16, 6, 5.0, sigma=0.05)
    idx.insert(7, 2.0, SE3.identity(), room)

    T = SE3(so3_exp([-0.2, 0.15, 0.5]), np.array([-0.8, 1.2, -0.4]))
    m = idx.query(view_from(room, T.inverse(), noise=0.08))

    assert m is not None
    d_t, d_r = m.transform.distance_to(T)
    assert d_t < 0.30, d_t
    assert d_r < 0.10, d_r


def test_handles_partial_observation():
    """절반만 보여도 재지역화되어야 한다. 전부 보이는 경우는 실제로 드물다."""
    idx = ConstellationIndex()
    room = make_room(20, 7, 6.0)
    idx.insert(3, 1.0, SE3.identity(), room)

    T = SE3(so3_exp([0.0, 0.6, 0.0]), np.array([3.0, 0.0, 0.0]))
    m = idx.query(view_from(room, T.inverse(), noise=0.03)[:10])

    assert m is not None
    assert m.transform.distance_to(T)[0] < 0.25


def test_survives_spurious_detections():
    """오검출이 섞여도 클리크 탐색이 걸러내야 한다."""
    idx = ConstellationIndex()
    room = make_room(14, 5, 5.0)
    idx.insert(9, 1.0, SE3.identity(), room)

    T = SE3(so3_exp([0.2, -0.3, 0.1]), np.array([1.0, 1.0, 0.0]))
    observed = view_from(room, T.inverse(), noise=0.04)
    observed += [Node(1000 + i, i % 5, RNG.uniform(-6, 6, 3), 0.2) for i in range(5)]

    m = idx.query(observed)
    assert m is not None
    assert m.transform.distance_to(T)[0] < 0.35


def test_rejects_unknown_place():
    idx = ConstellationIndex()
    idx.insert(1, 1.0, SE3.identity(), make_room(12, 5, 4.0))
    assert idx.query(make_room(12, 5, 4.0)) is None


def test_rejects_ambiguous_candidates():
    """동일 배치가 두 곳이면 하나를 골라 주장하면 안 된다."""
    idx = ConstellationIndex()
    corridor = make_room(10, 3, 5.0)
    idx.insert(1, 1.0, SE3.identity(), corridor)
    idx.insert(2, 2.0, SE3(np.eye(3), np.array([50.0, 0.0, 0.0])), corridor)

    assert idx.query(corridor) is None
    # 다만 상위 계층이 판단할 수 있게 두 후보 모두 보고되어야 한다
    assert len(idx.query_all(corridor)) >= 2


def test_neighbouring_keyframes_are_not_ambiguous():
    """점수가 붙는 것과 지각적 혼동은 다르다.

    5프레임 간격으로 등록한 지도에서는 이웃 키프레임이 같은 장면을 보므로
    상위 후보들의 점수가 붙는 것이 정상이다. score2 > 0.85*score1 규칙은 그것을
    혼동으로 오인해 fr1_xyz 에서 정대응 36개를 36개 모두 기각했다(재현율 0 %).

    두 키프레임은 anchor 도 transform 도 다르지만 월드 포즈는 같은 곳을
    가리키므로 한 군집이 되어야 한다.
    """
    idx = ConstellationIndex()
    room = make_room(14, 5, 4.0)
    for shift in (0.0, 0.30):
        offset = np.array([shift, 0.0, 0.0])
        local = [Node(n.token_id, n.class_id, n.position - offset, n.sigma) for n in room]
        idx.insert(int(shift * 10) + 1, 1.0,
                   SE3(np.eye(3), np.array([shift, 0.0, 0.0])), local)

    cand = idx.query_all(room)
    assert len(cand) == 2
    # 점수 공간에서는 옛 규칙이 발동할 만큼 붙어 있어야 측정이 성립한다
    assert cand[1].score > 0.85 * cand[0].score
    # 포즈 공간에서는 한 군집이다
    assert cand[0].agree_count == 2
    assert cand[0].rival_mass == 0.0

    m = idx.query(room)
    assert m is not None, "이웃 키프레임은 기각 대상이 아니다"


def test_ambiguity_is_decided_in_pose_space_not_score_space():
    """두 상황은 점수로는 구분되지 않고 포즈로는 구분된다."""
    room = make_room(12, 4, 4.0)

    shift = np.array([50.0, 0.0, 0.0])

    # 이웃 키프레임: anchor 도 노드 좌표도 옮겨서 월드 포즈는 같은 곳
    near = ConstellationIndex()
    near.insert(1, 1.0, SE3.identity(), room)
    near.insert(2, 2.0, SE3(np.eye(3), shift),
                [Node(n.token_id, n.class_id, n.position - shift, n.sigma) for n in room])

    # 동일한 방 두 개: 노드 좌표는 같고 anchor 만 50 m 떨어져 있다
    far = ConstellationIndex()
    far.insert(1, 1.0, SE3.identity(), room)
    far.insert(2, 2.0, SE3(np.eye(3), shift), room)

    near_c, far_c = near.query_all(room), far.query_all(room)
    assert len(near_c) == len(far_c) == 2
    # 점수는 같은 모양이다
    assert near_c[0].score == pytest.approx(far_c[0].score, abs=1e-9)
    assert near_c[1].score == pytest.approx(far_c[1].score, abs=1e-9)
    # 포즈 군집은 다르다
    assert near_c[0].agree_count == 2 and far_c[0].agree_count == 1
    assert near.query(room) is not None
    assert far.query(room) is None


def test_confidence_combines_selfconsistency_dominance_and_chi2():
    """세 항이 서로 다른 것을 잰다. 하나라도 빠지면 신뢰도가 순위를 못 매긴다."""
    idx = ConstellationIndex()
    room = make_room(12, 5, 4.0)
    idx.insert(1, 1.0, SE3.identity(), room)

    clean = idx.query_all(room)[0]
    assert clean.confidence == pytest.approx(clean.score / (1.0 + clean.chi2_dof / 10.0),
                                             rel=1e-9)
    assert clean.confidence > 0.9

    # 잔차를 키우면 chi2 항이 신뢰도를 끌어내린다 (score 만으로는 못 본다)
    noisy = idx.query_all(view_from(room, SE3.identity(), noise=0.05))[0]
    assert noisy.chi2_dof > clean.chi2_dof
    assert noisy.confidence < clean.confidence


def test_requires_minimum_nodes():
    idx = ConstellationIndex()
    idx.insert(1, 1.0, SE3.identity(), make_room(12, 5, 4.0))
    assert idx.query(make_room(3, 3, 2.0)) is None


def test_scales_to_many_places():
    idx = ConstellationIndex()
    target = make_room(15, 8, 5.0)
    for i in range(200):
        idx.insert(i, float(i), SE3.identity(), make_room(15, 8, 5.0))
    pid = idx.insert(999, 999.0, SE3.identity(), target)

    T = SE3(so3_exp([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.0]))
    m = idx.query(view_from(target, T.inverse(), noise=0.03))

    assert m is not None
    assert m.place_id == pid


def test_signature_is_rigid_invariant():
    """서명은 강체변환에 불변이어야 한다. TCG 의 존재 근거."""
    idx = ConstellationIndex()
    room = make_room(10, 4, 4.0)
    T = SE3(so3_exp([0.7, -0.3, 1.1]), np.array([5.0, -3.0, 2.0]))

    assert sorted(idx.signature(room)) == sorted(idx.signature(view_from(room, T)))


def test_signature_is_order_invariant():
    idx = ConstellationIndex()
    room = make_room(10, 4, 4.0)
    shuffled = list(room)
    RNG.shuffle(shuffled)
    assert sorted(idx.signature(room)) == sorted(idx.signature(shuffled))


def test_correspondences_identify_the_same_objects():
    """BoW 가 절대 줄 수 없는 것: '그 의자가 아까 그 의자다'."""
    idx = ConstellationIndex()
    room = make_room(12, 5, 4.0)
    idx.insert(1, 1.0, SE3.identity(), room)

    T = SE3(so3_exp([0.05, 0.2, -0.1]), np.array([0.8, -0.5, 0.1]))
    m = idx.query(view_from(room, T.inverse(), noise=0.02))

    assert m is not None
    assert len(m.correspondences) >= 4
    # query 토큰 id 와 place 토큰 id 가 같은 개체를 가리켜야 한다
    for q_id, p_id in m.correspondences:
        assert q_id == p_id, f"{q_id} != {p_id}"


@pytest.mark.parametrize("num_classes", [1, 2, 3, 8])
def test_works_across_class_diversity(num_classes):
    """클래스가 한 종류뿐이면 대응 후보가 n^2 로 폭증하는 최악 케이스."""
    idx = ConstellationIndex()
    room = make_room(10, num_classes, 5.0)
    idx.insert(1, 1.0, SE3.identity(), room)

    T = SE3(so3_exp([0.05, 0.05, 0.02]), np.array([0.3, 0.2, 0.0]))
    m = idx.query(view_from(room, T.inverse(), noise=0.02))

    assert m is not None, f"클래스 {num_classes}종에서 실패"
    assert m.transform.distance_to(T)[0] < 0.3


DOWN = np.array([0.0, 0.0, -1.0])


def test_chirality_rejects_mirrored_layout():
    """거울 배치는 쌍거리 스펙트럼이 같다. 카이랄리티가 이를 걸러야 한다."""
    idx = ConstellationIndex(Config(use_chirality=True))

    room = [
        Node(1, 0, np.array([0.0, 0.0, 0.0])),
        Node(2, 1, np.array([2.0, 0.0, 1.5])),
        Node(3, 2, np.array([0.0, 2.0, -1.5])),
        Node(4, 3, np.array([2.0, 2.0, 1.0])),
        Node(5, 4, np.array([1.0, 1.0, -1.2])),
    ]
    idx.insert(1, 1.0, SE3.identity(), room, gravity=DOWN)

    mirrored = [Node(n.token_id, n.class_id, n.position * np.array([1.0, 1.0, -1.0]), n.sigma)
                for n in room]
    assert idx.query(mirrored, gravity=DOWN) is None


def test_chirality_uses_each_frames_own_gravity():
    """회귀 테스트.

    질의 프레임은 등록 시점 대비 회전되어 있으므로, 그 프레임에서의 중력은
    더 이상 [0,0,-1] 이 아니다. 두 프레임에 같은 벡터를 쓰면 정상 대응이
    통째로 걸러져 클리크가 쪼그라들고 틀린 변환이 나온다.
    """
    idx = ConstellationIndex(Config(use_chirality=True))
    room = make_room(12, 5, 4.0)
    idx.insert(1, 1.0, SE3.identity(), room, gravity=DOWN)

    T = SE3(so3_exp([0.1, 0.4, -0.2]), np.array([1.5, -2.0, 0.3]))
    q_nodes = view_from(room, T.inverse())
    q_gravity = T.inverse().R @ DOWN        # 질의 프레임에서 본 중력

    m = idx.query(q_nodes, gravity=q_gravity)
    assert m is not None
    assert len(m.correspondences) == len(room), "정상 대응이 카이랄리티에 걸렸다"
    assert m.transform.distance_to(T)[0] < 1e-6

    # 같은 벡터를 그대로 쓰면 (버그가 있던 방식) 대응이 줄어든다
    bad = idx.query(q_nodes, gravity=DOWN)
    assert bad is None or len(bad.correspondences) < len(room)


def test_gravity_is_optional():
    """중력 추정이 없으면 카이랄리티 없이도 동작해야 한다 (IMU 없는 구성)."""
    idx = ConstellationIndex()
    room = make_room(12, 5, 4.0)
    idx.insert(1, 1.0, SE3.identity(), room)      # gravity 미지정

    T = SE3(so3_exp([0.1, 0.4, -0.2]), np.array([1.5, -2.0, 0.3]))
    m = idx.query(view_from(room, T.inverse()))
    assert m is not None
    assert m.transform.distance_to(T)[0] < 1e-6
