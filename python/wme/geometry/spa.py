"""SPA - Structural Primitive Alignment (Tier 2).

평면으로 상대 포즈를 구속한다. 외관과 무관하므로 텍스처가 없거나 조명이
무너진 곳에서도 살아남는다 - 측광(Tier 0)이 죽는 바로 그 조건이다.

세 tier 는 팩터 종류가 다르지 않다. 전부 BetweenPoseFactor 로 들어가고
정보행렬 Lambda 만 다르다. SPA 의 특징은 그 Lambda 가 **본질적으로 랭크
부족**일 수 있다는 것이다.

  - 회전: 서로 다른 방향의 평면 3개면 완전히 구속된다
  - 병진: 평면 법선이 만드는 부분공간에서만 구속된다

복도(평행한 벽 둘 + 바닥)에서는 복도 축 방향 병진이 관측되지 않는다.
그 사실을 숨기지 않고 랭크로 보고하는 것이 이 모듈의 계약이다.
docs/02 3장 Tier 2 의 '퇴화 복구' 역할이 여기서 나온다.

역할 한계도 적어 둔다: 여기 정합은 초기 추정이 어느 정도 맞다고 가정한다.
평면 대응을 각도/거리로 찾기 때문이다. SPA 는 정제기이지 부트스트랩이 아니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..reference.geometry import SE3, so3_log
from .planes import Plane


@dataclass
class SpaConfig:
    max_normal_angle: float = 0.35        # rad, 대응 판정
    max_distance_diff: float = 0.6        # m
    min_matches: int = 2
    min_confidence: float = 0.15

    # 랭크 판정. 대각 정규화 후 최대 고유값 대비 비율.
    degeneracy_ratio: float = 1e-3

    # 정보행렬 스케일. 근거는 C++ StructuralAlignerConfig 의 같은 이름 주석이다.
    # rotation_sigma 는 유도되지 않았다(유도 시도가 실데이터에 기각됐다).
    # translation_sigma 는 평면의 sigma_offset 이 미상일 때만 쓰는 되돌림 값이다.
    rotation_sigma: float = 0.02          # rad
    translation_sigma: float = 0.05       # m


@dataclass
class PlaneMatch:
    reference: Plane
    current: Plane
    angle: float
    distance_diff: float
    # 미상일 때 쓸 되돌림 값. match_planes 가 config 에서 채운다.
    fallback_sigma: float = 0.05
    rotation_sigma: float = 0.02

    @property
    def weight(self) -> float:
        """confidence 곱. **정보량이 아니다** - 대응 순서와 진단용이다."""
        return float(self.reference.confidence * self.current.confidence)

    @property
    def sigma_normal(self) -> float:
        """이 대응의 법선 잔차 sigma (rad).

        두 평면의 몫이 독립으로 더해진다. 1/2 로 정규화해서 잘 맞는 평면
        둘(f = 1)이면 정확히 rotation_sigma 로 환원된다 - 실측으로 맞춘 절대
        크기를 배율이 흔들지 않는다. C++ 과 같은 식이어야 한다.
        """
        fr = self.reference.fit_degradation ** 2
        fc = self.current.fit_degradation ** 2
        return float(self.rotation_sigma * math.sqrt(0.5 * (fr * fr + fc * fc)))

    @property
    def sigma_offset(self) -> float:
        """이 대응의 오프셋 잔차 sigma (m).

        두 평면의 오프셋 오차가 각각 한 번씩 독립으로 들어오므로 분산이 더해진다.
        한쪽이라도 미상(0)이면 절반만 유도된 sigma 가 되므로 전체를 되돌림
        상수로 간다.
        """
        a, b = self.reference.sigma_offset, self.current.sigma_offset
        if a > 0.0 and b > 0.0:
            return float(math.sqrt(a * a + b * b))
        return float(self.fallback_sigma)


@dataclass
class SpaResult:
    T_cur_ref: SE3
    information: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(6))
    observable_dof: int = 0
    weakest_direction: np.ndarray = field(default_factory=lambda: np.zeros(6))
    matches: list[PlaneMatch] = field(default_factory=list)
    rotation_rank: int = 0
    translation_rank: int = 0
    converged: bool = False

    @property
    def full_rank(self) -> bool:
        return self.observable_dof == 6

    def summary(self) -> str:
        return (f"SPA {len(self.matches)} matches, "
                f"rot rank {self.rotation_rank}/3, trans rank {self.translation_rank}/3, "
                f"observable {self.observable_dof}/6")


def match_planes(reference: list[Plane], current: list[Plane],
                 init: SE3 | None = None,
                 config: SpaConfig | None = None) -> list[PlaneMatch]:
    """법선 각도와 거리로 평면을 짝짓는다.

    init 을 주면 그것으로 reference 를 옮긴 뒤 비교한다. 초기 추정이 없으면
    항등으로 가정하므로, 큰 운동에서는 대응을 못 찾는다 - SPA 가 정제기인 이유다.
    """
    cfg = config or SpaConfig()
    T = init or SE3.identity()

    matches: list[PlaneMatch] = []
    used: set[int] = set()

    # 큰 평면부터 짝지어 애매한 대응이 먼저 자리를 차지하지 않게 한다
    for ref in sorted(reference, key=lambda p: -p.confidence):
        moved = ref.transformed(T.R, T.t)
        best, best_angle, best_dd = None, cfg.max_normal_angle, None

        for i, cur in enumerate(current):
            if i in used:
                continue
            cos = float(np.clip(moved.normal @ cur.normal, -1.0, 1.0))
            angle = math.acos(cos)
            if angle > cfg.max_normal_angle:
                continue
            dd = abs(moved.distance - cur.distance)
            if dd > cfg.max_distance_diff:
                continue
            if angle < best_angle:
                best, best_angle, best_dd = i, angle, dd

        if best is not None:
            used.add(best)
            matches.append(PlaneMatch(ref, current[best], best_angle, best_dd,
                                      cfg.translation_sigma, cfg.rotation_sigma))
    return matches


def align(reference: list[Plane], current: list[Plane],
          init: SE3 | None = None, config: SpaConfig | None = None,
          alpha_structural: float = 1.0) -> SpaResult:
    """평면 대응으로 ref -> cur 상대 포즈를 구한다."""
    cfg = config or SpaConfig()
    matches = match_planes(reference, current, init, cfg)
    result = SpaResult(T_cur_ref=init or SE3.identity(), matches=matches)

    if len(matches) < cfg.min_matches:
        return result

    if cfg.rotation_sigma <= 0.0 or cfg.translation_sigma <= 0.0:
        return result

    # 가중치는 전부 **역분산**이다. 무차원 신뢰도가 아니다.
    #
    # 회전 sigma 는 평면의 적합 잔차로 스케일된다. 근거와 후보 비교는
    # C++ StructuralAlignerConfig::rotation_sigma 주석이다.
    sig_n = np.array([m.sigma_normal for m in matches])
    w = 1.0 / (sig_n ** 2)
    # 병진은 대응마다 다르다. 여기가 Lambda 에 거리가 들어오는 자리다.
    sig_t = np.array([m.sigma_offset for m in matches])
    w_t = 1.0 / (sig_t ** 2)
    n_ref = np.array([m.reference.normal for m in matches])
    n_cur = np.array([m.current.normal for m in matches])

    # --- 회전: 법선끼리의 Kabsch (중심화 없이, 방향만) ---
    H = (n_ref * w[:, None]).T @ n_cur
    try:
        U, s, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return result

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        V = Vt.T.copy()
        V[:, 2] *= -1.0
        R = V @ U.T

    # 회전 정보. 잔차 n_cur - R n_ref 의 좌측섭동 자코비안은 J = [n]x 이므로
    #
    #     J^T J = -[n]x [n]x = (n^T n) I - n n^T = I - n n^T   (단위 n)
    #
    # **산포행렬 n n^T 가 아니다.** 둘은 서로 직교하는 부분공간 위의 사영이라,
    # 산포를 쓰면 회전이 구속되지 **않는** 축에 정보를 싣고 구속되는 축에는
    # 아무것도 싣지 않는다. 06-results.md 7.1 이 이 결함을 기록하면서 발표된 표는
    # C++ 포트로 다시 만들어 고쳤지만, **이 파이썬 참조는 고쳐지지 않은 채로
    # 남아 있었다.** 26.x 의 차분 오라클이 붙자마자 회전 블록 2 배 불일치로
    # 드러났다 - 직교 3평면에서 산포는 wI, 여집합은 2wI 다.
    rot_info = np.einsum("i,jk->jk", w, np.eye(3)) - (n_cur * w[:, None]).T @ n_cur
    rot_eig = np.linalg.eigvalsh(rot_info)[::-1]
    rot_rank = int(np.sum(rot_eig > max(rot_eig[0], 1e-12) * cfg.degeneracy_ratio))

    # --- 병진 ---
    # ref 좌표의 점 p 는 cur 좌표에서 q = R p + t 다. 평면 n_r·p = d_r 을
    # 대입하면 (R n_r)·q = d_r + (R n_r)·t 이므로
    #
    #     n_cur · t = d_cur - d_ref
    #
    # 부호를 뒤집으면 병진이 정확히 반대로 나온다. 항등 정합에서는 양변이
    # 0 이라 드러나지 않으므로, 반드시 병진이 있는 케이스로 검증해야 한다.
    d_ref = np.array([m.reference.distance for m in matches])
    d_cur = np.array([m.current.distance for m in matches])
    b = d_cur - d_ref

    # 가중최소제곱은 잔차를 sqrt(w) 로 스케일해야 정규방정식이 sum w n n^T 가
    # 된다. w 로 스케일하면 sum w^2 n n^T 가 되어 **가중치가 두 번 적용**된다.
    # inliers=400(=weight 1) 로만 시험하면 w 와 w^2 이 같아 보이지 않는다 -
    # 이 결함이 오래 남은 이유이고, 26.x 의 첫 테스트도 같은 함정에 빠질 뻔했다.
    sw = np.sqrt(w_t)
    A = n_cur * sw[:, None]
    ATA = A.T @ A
    trans_eig = np.linalg.eigvalsh(ATA)[::-1]
    trans_rank = int(np.sum(trans_eig > max(trans_eig[0], 1e-12) * cfg.degeneracy_ratio))

    # 절단 최소노름 해.
    #
    # rcond=None 은 기계정밀도 기준이라 거의 영에 가까운 특이값 방향까지
    # 살려 두고, 그 방향의 잡음을 크게 증폭한다. 실측에서 병진 오차가
    # 40 m 넘게 튀었고 원인이 이것이었다. 랭크 판정과 같은 문턱으로 잘라
    # "관측되지 않는 방향으로는 움직이지 않는다" 는 설계 의도를 지킨다.
    try:
        t, *_ = np.linalg.lstsq(A, b * sw, rcond=math.sqrt(cfg.degeneracy_ratio))
    except np.linalg.LinAlgError:
        return result

    result.T_cur_ref = SE3(R, t)
    result.rotation_rank = rot_rank
    result.translation_rank = trans_rank
    result.converged = True

    # --- 정보행렬 ---
    # ATA 와 rot_info 는 이미 역분산으로 가중되어 있다. 병진 최소제곱이 쓴
    # 가중과 정확히 같은 값이어야 Lambda 가 그 추정기의 공분산이 된다.
    #
    # 주의: 이 참조 구현은 t x n 교차항을 넣지 않는다. C++ 는 넣는다
    # (좌측 섭동에서 dt = rho + phi x t). 항등 정합(t = 0)에서만 두 형태가
    # 같으므로 차등 테스트는 그 자리에서만 두 정보행렬을 맞대야 한다.
    Lam = np.zeros((6, 6))
    Lam[:3, :3] = ATA
    Lam[3:, 3:] = rot_info
    Lam *= max(alpha_structural, 1e-6)
    result.information = Lam

    # 퇴화 진단. 병진(m)과 회전(rad)은 단위가 달라 대각 정규화 후 본다.
    diag = np.maximum(np.diag(Lam), 1e-12)
    scale = 1.0 / np.sqrt(diag)
    Lam_n = Lam * scale[:, None] * scale[None, :]

    eig, vecs = np.linalg.eigh(Lam_n)
    order = np.argsort(eig)[::-1]
    eig, vecs = eig[order], vecs[:, order]

    result.eigenvalues = eig
    result.observable_dof = int(np.sum(eig > max(eig[0], 1e-12) * cfg.degeneracy_ratio))
    result.weakest_direction = vecs[:, -1]
    return result


def unobservable_directions(result: SpaResult, ratio: float = 1e-3) -> np.ndarray:
    """관측되지 않는 접선 방향들 (열벡터).

    Tier 0(ECDA)이 채워야 할 축을 알려주는 용도다. 융합 규칙이 서로의
    구멍을 메우려면 어디가 구멍인지 말할 수 있어야 한다.
    """
    if not result.converged:
        return np.eye(6)
    eig = result.eigenvalues
    weak = eig <= max(eig[0], 1e-12) * ratio
    if not np.any(weak):
        return np.zeros((6, 0))
    # eigenvalues 는 내림차순이므로 뒤쪽이 약한 축이다
    return np.eye(6)[:, np.flatnonzero(weak)]
