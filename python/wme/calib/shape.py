"""위치와 크기를 함께 추정하는 관측 모델.

M3 에서 드러난 문제: 검출 박스의 중심은 무게중심의 투영이 아니다. 편향은

    du ~ fx * e_x * e_z / Cz^2       (미터 환산 ~ e_x * e_z / Cz)

이고, 모든 관측에 공통으로 실려 융합으로 줄지 않는다.

편향을 빼서 고칠 수는 없다. 깊이 방향 크기 e_z 가 단일 뷰에서 관측 불가능하기
때문이다. 올바른 해법은 크기를 상태에 넣어 위치와 함께 추정하고, 미지의 크기가
남기는 불확실성을 주변화로 위치 공분산에 반영하는 것이다.

이것은 객체 중심 SLAM 이 가져야 할 형태이기도 하다 - WorldToken 은 이미
extent 를 들고 있지만 지금은 EMA 로 대충 갱신될 뿐 추정 대상이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reference.geometry import SE3
from ..sim.scenarios import Sequence
from ..sim.sensor import severity_of
from ..sim.world import CameraModel
from .estimator import EstimationResult, ObjectEstimate, box_center
from .noise import CalibratedNoise, NoiseModel, ObjectResidual

# 8개 꼭짓점의 부호 조합
_SIGNS = np.array([[sx, sy, sz]
                   for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])

# 정규화 잔차가 이 값을 넘으면 수렴 실패로 보고 해를 버린다.
# 모델이 맞으면 1 근처여야 하므로 여유를 크게 준 값이다.
_CHI2_REJECT = 5.0


def predict_observation(position_world: np.ndarray, extent: np.ndarray,
                        T_cam_world: SE3, cam: CameraModel) -> np.ndarray | None:
    """(위치, 크기) -> (box_x, box_y, box_w, box_h, depth) 예측.

    관측을 (x, y, w, h) 로 두는 이유: 시뮬레이터가 네 값에 독립 잡음을 주므로
    잔차 공분산이 대각이 된다. (u_min, u_max) 로 두면 상관이 생겨 R 이 복잡해진다.
    """
    p_cam = T_cam_world @ position_world
    corners = p_cam + _SIGNS * extent

    if np.any(corners[:, 2] <= 1e-3):
        return None

    u = cam.fx * corners[:, 0] / corners[:, 2] + cam.cx
    v = cam.fy * corners[:, 1] / corners[:, 2] + cam.cy

    u_lo, u_hi = float(u.min()), float(u.max())
    v_lo, v_hi = float(v.min()), float(v.max())
    return np.array([u_lo, v_lo, u_hi - u_lo, v_hi - v_lo, float(p_cam[2])])


def _numeric_jacobian(state: np.ndarray, T_cam_world: SE3, cam: CameraModel,
                      eps: float = 1e-5) -> np.ndarray | None:
    """중앙차분 야코비안 (5x6).

    해석적 야코비안을 쓸 수도 있지만 min/max 선택 때문에 조각별로 갈라진다.
    수치미분이 그 구조를 자동으로 따라가고 오류 여지가 적다.
    """
    base = predict_observation(state[:3], state[3:], T_cam_world, cam)
    if base is None:
        return None

    J = np.zeros((5, 6))
    for k in range(6):
        step = np.zeros(6)
        step[k] = eps
        plus = predict_observation(state[:3] + step[:3], state[3:] + step[3:],
                                   T_cam_world, cam)
        minus = predict_observation(state[:3] - step[:3], state[3:] - step[3:],
                                    T_cam_world, cam)
        if plus is None or minus is None:
            return None
        J[:, k] = (plus - minus) / (2.0 * eps)
    return J


@dataclass
class _Observation:
    z: np.ndarray                 # (5,) 관측
    T_world_cam: SE3              # 월드 <- 카메라
    T_cam_world: SE3              # 카메라 <- 월드 (순방향 모델이 쓰는 방향)
    severity: float
    depth: float


def _weight(model: NoiseModel, o: _Observation) -> np.ndarray:
    s_px = model.sigma_pixel(o.severity)
    s_d = model.sigma_depth(max(o.depth, 0.1), o.severity)
    return np.diag([1.0 / s_px ** 2] * 4 + [1.0 / s_d ** 2])


def _cost_at(state: np.ndarray, obs: list[_Observation], model: NoiseModel,
             cam: CameraModel) -> float:
    total, used = 0.0, 0
    for o in obs:
        pred = predict_observation(state[:3], state[3:], o.T_cam_world, cam)
        if pred is None:
            return float("inf")
        r = pred - o.z
        total += float(r @ _weight(model, o) @ r)
        used += 1
    return total if used >= 3 else float("inf")


def _normal_equations(state: np.ndarray, obs: list[_Observation], model: NoiseModel,
                      cam: CameraModel) -> tuple[np.ndarray, np.ndarray, int]:
    H = np.zeros((6, 6))
    b = np.zeros(6)
    used = 0
    for o in obs:
        pred = predict_observation(state[:3], state[3:], o.T_cam_world, cam)
        if pred is None:
            continue
        J = _numeric_jacobian(state, o.T_cam_world, cam)
        if J is None:
            continue
        W = _weight(model, o)
        r = pred - o.z
        H += J.T @ W @ J
        b += J.T @ W @ r
        used += 1
    return H, b, used


def _solve_object(obs: list[_Observation], init_state: np.ndarray,
                  model: NoiseModel, cam: CameraModel,
                  max_iter: int = 40) -> tuple[np.ndarray, np.ndarray] | None:
    """레벤버그-마쿼트로 (위치3, 크기3) 을 추정한다.

    수용 판정은 반드시 *후보* 상태의 비용으로 해야 한다. 직전 반복의 비용과
    비교하면 평가하지 않은 스텝을 받아들이게 되어 발산한다.

    Returns:
        (상태, 6x6 정보행렬) 또는 None
    """
    state = init_state.copy()
    cost = _cost_at(state, obs, model, cam)
    if not np.isfinite(cost):
        return None

    lam = 1e-3
    for _ in range(max_iter):
        H, b, used = _normal_equations(state, obs, model, cam)
        if used < 3:
            return None

        damped = H + lam * np.diag(np.maximum(np.diag(H), 1e-9))
        try:
            delta = -np.linalg.solve(damped, b)
        except np.linalg.LinAlgError:
            lam *= 10.0
            if lam > 1e10:
                break
            continue
        if not np.all(np.isfinite(delta)):
            lam *= 10.0
            continue

        candidate = state + delta
        candidate[3:] = np.clip(candidate[3:], 0.02, 3.0)   # 크기는 양수

        cand_cost = _cost_at(candidate, obs, model, cam)
        if cand_cost < cost:
            converged = np.linalg.norm(delta) < 1e-8
            state, cost = candidate, cand_cost
            lam = max(1e-9, lam * 0.4)
            if converged:
                break
        else:
            lam *= 5.0
            if lam > 1e10:
                break

    H, _, used = _normal_equations(state, obs, model, cam)
    if used < 3:
        return None

    # 수렴 자기진단. 정규화 잔차가 기대치를 크게 벗어나면 해를 신뢰할 수 없다.
    # 발산한 해를 조용히 내보내면 하나가 전체 통계를 지배한다.
    # 참값을 쓰지 않고 잔차만 보므로 실제 시스템도 할 수 있는 판정이다.
    dof = max(1, 5 * used - 6)
    if cost / dof > _CHI2_REJECT:
        return None
    return state, H


def _group_observations(seq: Sequence, cam: CameraModel
                        ) -> dict[int, list[_Observation]]:
    """정적 객체별 관측 묶음. 참 연관을 쓴다(연관 오류와 분리하기 위해)."""
    grouped: dict[int, list[_Observation]] = {}
    for det, truth in zip(seq.detections, seq.truths):
        sev = severity_of(truth.evidence)
        T_cam_world = truth.T_world_cam.inverse()
        for i, item in enumerate(det.items):
            oid = truth.detection_to_object[i]
            if oid < 0:
                continue
            obj = seq.world.by_id(oid)
            if obj is None or obj.is_agent:
                continue
            d = truth.detection_depth[i]
            if d <= 0.1:
                continue

            x, y, w, h = item.box
            grouped.setdefault(oid, []).append(_Observation(
                z=np.array([x, y, w, h, d]),
                T_world_cam=truth.T_world_cam,
                T_cam_world=T_cam_world,
                severity=sev,
                depth=d,
            ))
    return grouped


def _initial_state(obs: list[_Observation], cam: CameraModel) -> np.ndarray:
    """편향 섞인 중심 역투영 + 박스 크기에서 얻은 대략적 치수."""
    o0 = obs[len(obs) // 2]
    u, v = box_center((o0.z[0], o0.z[1], o0.z[2], o0.z[3]))
    d0 = float(o0.depth)
    p_cam = np.array([(u - cam.cx) * d0 / cam.fx, (v - cam.cy) * d0 / cam.fy, d0])

    ex = max(0.05, 0.5 * o0.z[2] * d0 / cam.fx)
    ey = max(0.05, 0.5 * o0.z[3] * d0 / cam.fy)
    return np.concatenate([o0.T_world_cam @ p_cam, [ex, ey, 0.5 * (ex + ey)]])


def estimate_objects_jointly(seq: Sequence, model: NoiseModel,
                             cam: CameraModel | None = None) -> EstimationResult:
    """위치와 크기를 함께 추정하고, 크기를 주변화해 위치 공분산을 낸다.

    주변화가 핵심이다. 크기를 안다고 가정(조건화)하면 위치 공분산이 작아지지만
    실제로는 모르므로 과신이 된다. Schur 보수로 크기 블록을 소거해야
    "크기를 몰라서 생기는 위치 불확실성"이 제대로 반영된다.
    """
    cam = cam or seq.camera
    result = EstimationResult()

    grouped = _group_observations(seq, cam)
    for truth in seq.truths:
        for oid in truth.detection_to_object:
            if oid >= 0 and oid in grouped:
                result.truth[oid] = truth.object_positions[oid]

    for oid, obs in grouped.items():
        solved = _solve_object(obs, _initial_state(obs, cam), model, cam)
        if solved is None:
            continue
        state, H = solved

        # 크기 블록 주변화: Lambda_CC = H_CC - H_Ce H_ee^-1 H_eC
        H_cc, H_ce, H_ee = H[:3, :3], H[:3, 3:], H[3:, 3:]
        try:
            Lambda = H_cc - H_ce @ np.linalg.solve(H_ee + np.eye(3) * 1e-9, H_ce.T)
            P = np.linalg.inv(Lambda + np.eye(3) * 1e-12)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(P)) or np.any(np.diag(P) <= 0.0):
            continue

        n = len(obs)
        mean_box = float(np.mean([max(o.z[2], o.z[3]) for o in obs]))
        mean_depth = float(np.mean([o.depth for o in obs]))

        Sigma_sys = model.systematic_covariance(
            float(np.mean([o.severity for o in obs])), mean_box, mean_depth, cam)

        result.estimates[oid] = ObjectEstimate(
            oid, state[:3], P + Sigma_sys, n,
            mean_box_px=mean_box, mean_depth=mean_depth)

    return result


def refit_noise_under_joint_model(sequences: list[Sequence], model: CalibratedNoise,
                                  cam: CameraModel, iterations: int = 2
                                  ) -> CalibratedNoise:
    """공동 추정기의 잔차로 잡음 파라미터를 다시 적합한다.

    잡음모델은 그것이 쓰일 관측 모델과 함께 보정해야 한다. 편향이 있는 모델로
    적합하면 잡음 파라미터가 편향을 흡수하고, 나중에 모델을 고치면 이번엔
    과대평가가 된다. 실제로 M3 에서 c_px 가 참값의 4배로 나왔다.

    닭과 달걀 문제라 EM 처럼 반복한다: 풀고 -> 잔차로 재적합 -> 다시 풀고.
    """
    current = model
    for _ in range(max(1, iterations)):
        px_res: list[tuple[float, float]] = []      # (severity, |픽셀 잔차|)
        d_res: list[tuple[float, float]] = []       # (severity, |깊이 잔차| / z^2)

        for seq in sequences:
            grouped = _group_observations(seq, cam)
            for oid, obs in grouped.items():
                solved = _solve_object(obs, _initial_state(obs, cam), current, cam)
                if solved is None:
                    continue
                state, _ = solved
                for o in obs:
                    pred = predict_observation(state[:3], state[3:], o.T_cam_world, cam)
                    if pred is None:
                        continue
                    r = pred - o.z
                    for k in range(4):
                        px_res.append((o.severity, abs(float(r[k]))))
                    d_res.append((o.severity, abs(float(r[4])) / max(o.depth ** 2, 1e-6)))

        if len(px_res) < 50 or len(d_res) < 20:
            break
        current = CalibratedNoise(
            **_fit_linear_sigma(px_res, "px"),
            **_fit_linear_sigma(d_res, "d"),
            s_sys=current.s_sys, k_geom=current.k_geom, g_sys=current.g_sys,
        )
    return current


def _fit_linear_sigma(residuals: list[tuple[float, float]], kind: str) -> dict:
    """sigma(s) = c * (1 + g*s) 를 강도 구간별 RMS 로부터 최소제곱 적합.

    최적화기를 쓰지 않는다 - 구간별 RMS 는 sigma 의 직접 추정치이고,
    거기에 직선을 맞추면 끝난다.
    """
    arr = np.array(residuals)
    sev, val = arr[:, 0], arr[:, 1]

    edges = np.linspace(sev.min(), sev.max() + 1e-9, 6)
    centres, rms = [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (sev >= lo) & (sev < hi)
        if m.sum() < 10:
            continue
        centres.append(float(sev[m].mean()))
        # |r| 의 RMS 가 곧 sigma 추정치 (평균이 0 인 가우시안 가정)
        rms.append(float(np.sqrt((val[m] ** 2).mean())))

    if len(centres) < 2:
        c = float(np.sqrt((val ** 2).mean()))
        return ({"c_px": c, "g_px": 0.0} if kind == "px" else {"c_d": c, "g_d": 0.0})

    A = np.column_stack([np.ones(len(centres)), np.array(centres)])
    coef, *_ = np.linalg.lstsq(A, np.array(rms), rcond=None)
    c = max(float(coef[0]), 1e-6)
    g = max(float(coef[1]) / c, 0.0)
    return ({"c_px": c, "g_px": g} if kind == "px" else {"c_d": c, "g_d": g})


def object_level_residuals_joint(seq: Sequence, model: NoiseModel,
                                 min_observations: int = 8) -> list[ObjectResidual]:
    """공동 추정기 기준 객체 단위 잔차 (계통 성분 적합용)."""
    result = estimate_objects_jointly(seq, model)
    sev = seq.severity
    out = []
    for oid, est in sorted(result.estimates.items()):
        if est.observations < min_observations or oid not in result.truth:
            continue
        out.append(ObjectResidual(
            error=est.error_against(result.truth[oid]),
            covariance=est.covariance,
            severity=sev,
            mean_box_px=est.mean_box_px,
            mean_depth=est.mean_depth,
        ))
    return out
