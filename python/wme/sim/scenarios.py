"""표준 시나리오.

각 시나리오는 WME 의 특정 주장 하나에 대응한다. 시나리오가 대응하지 않는
주장은 검증되지 않은 주장이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..eval.metrics import ChangeEvent
from ..reference.environment import Evidence
from ..yolo import COCO_CLASSES, DetectionSet
from .sensor import FrameTruth, SensorConfig, SimSensor
from .world import CameraModel, CameraTrajectory, SimObject, SimWorld

ConditionFn = Callable[[float], Evidence]


@dataclass
class Sequence:
    """실행된 시퀀스 하나. 참값이 전부 들어 있다."""
    name: str
    world: SimWorld
    trajectory: CameraTrajectory
    detections: list[DetectionSet]
    truths: list[FrameTruth]
    changes: list[ChangeEvent] = field(default_factory=list)
    camera: CameraModel = field(default_factory=CameraModel)

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def severity(self) -> float:
        from .sensor import severity_of
        return float(np.mean([severity_of(t.evidence) for t in self.truths]))


# --- 조건 스케줄 -----------------------------------------------------------

def constant_condition(**kwargs) -> ConditionFn:
    ev = Evidence(**kwargs)
    return lambda _t: ev


def ramp_condition(channel: str, start: float, end: float,
                   t0: float, t1: float) -> ConditionFn:
    """한 채널을 시간에 따라 선형 증감. 복구 지연 측정에 쓴다."""
    def fn(t: float) -> Evidence:
        a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        return Evidence(**{channel: start + a * (end - start)})
    return fn


def burst_condition(channel: str, level: float,
                    t_start: float, t_end: float) -> ConditionFn:
    """구간 동안만 열화. 조건이 걷힌 뒤 복구되는지 보는 용도."""
    def fn(t: float) -> Evidence:
        return Evidence(**{channel: level if t_start <= t < t_end else 0.0})
    return fn


# --- 세계 구성 -------------------------------------------------------------

# 실내 장면에 흔한 COCO 클래스. 어포던스 표와 어휘를 맞춘다.
_INDOOR = [
    (56, "chair"), (56, "chair"), (56, "chair"),
    (60, "dining table"), (57, "couch"), (62, "tv"),
    (41, "cup"), (73, "book"), (75, "vase"), (58, "potted plant"),
    (72, "refrigerator"), (66, "keyboard"),
]


def static_room(n_objects: int = 12, extent: float = 4.0, seed: int = 0) -> SimWorld:
    """정적 실내 장면. 성좌 재지역화의 기본 무대."""
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n_objects):
        cid, name = _INDOOR[i % len(_INDOOR)]
        # 방 가장자리에 두어 카메라가 중앙을 돌 때 잘 보이게 한다
        angle = 2.0 * np.pi * i / n_objects + rng.normal(0.0, 0.15)
        r = extent * rng.uniform(0.55, 1.0)
        objs.append(SimObject(
            object_id=i + 1,
            class_id=cid,
            class_name=name,
            position=np.array([r * np.cos(angle), r * np.sin(angle),
                               rng.uniform(0.2, 1.4)]),
            extent=np.array([0.25, 0.25, 0.3]) * rng.uniform(0.8, 1.6),
        ))
    return SimWorld(objs)


def add_walkers(world: SimWorld, count: int, duration: float,
                extent: float = 4.0, seed: int = 1) -> SimWorld:
    """사람을 추가한다. 동적 마스킹과 정적/동적 판정 검증용."""
    rng = np.random.default_rng(seed)
    objs = list(world.objects)
    base_id = max((o.object_id for o in world.objects), default=0) + 1

    for k in range(count):
        a0 = rng.uniform(0, 2 * np.pi)
        a1 = a0 + rng.uniform(np.pi * 0.5, np.pi * 1.5)
        r = extent * 0.5
        start = np.array([r * np.cos(a0), r * np.sin(a0), 0.9])
        end = np.array([r * np.cos(a1), r * np.sin(a1), 0.9])
        objs.append(SimObject(
            object_id=base_id + k,
            class_id=0,
            class_name="person",
            position=start,
            extent=np.array([0.3, 0.3, 0.9]),
            is_agent=True,
            waypoints=[(1.0, start), (1.0 + duration, end)],
        ))
    return SimWorld(objs)


def apply_changes(world: SimWorld, gap_start: float,
                  moved: int = 1, removed: int = 1, added: int = 1,
                  extent: float = 4.0, seed: int = 2
                  ) -> tuple[SimWorld, list[ChangeEvent]]:
    """두 방문 사이에 장면을 바꾼다.

    '무엇이 바뀌었는가' 는 두 관측 사이에만 정의된다. 변화 종류를 구분해
    참값을 남기는 것이 요점이다 - 이동/제거/추가는 서로 다른 행동을 요구한다.
    """
    rng = np.random.default_rng(seed)
    objs = list(world.objects)
    events: list[ChangeEvent] = []

    # 정적 객체만 후보로 삼는다. 사람이 움직인 건 '변화'가 아니다.
    static_ids = [o.object_id for o in objs if not o.is_agent]
    rng.shuffle(static_ids)
    cursor = 0

    for _ in range(moved):
        if cursor >= len(static_ids):
            break
        oid = static_ids[cursor]; cursor += 1
        obj = next(o for o in objs if o.object_id == oid)
        shifted = obj.position + np.array([rng.uniform(1.0, 2.0) * rng.choice([-1, 1]),
                                           rng.uniform(1.0, 2.0) * rng.choice([-1, 1]),
                                           0.0])
        # 간격 동안 순간이동한 것처럼: 이전엔 원위치, 이후엔 새 위치
        obj.waypoints = [(0.0, obj.position),
                         (gap_start, obj.position),
                         (gap_start + 1e-3, shifted),
                         (1e9, shifted)]
        events.append(ChangeEvent(oid, "moved", gap_start))

    for _ in range(removed):
        if cursor >= len(static_ids):
            break
        oid = static_ids[cursor]; cursor += 1
        obj = next(o for o in objs if o.object_id == oid)
        obj.present_until = gap_start
        events.append(ChangeEvent(oid, "removed", gap_start))

    base_id = max((o.object_id for o in objs), default=0) + 1
    for k in range(added):
        cid, name = _INDOOR[(k + 3) % len(_INDOOR)]
        angle = rng.uniform(0, 2 * np.pi)
        r = extent * rng.uniform(0.55, 1.0)
        oid = base_id + k
        objs.append(SimObject(
            object_id=oid, class_id=cid, class_name=name,
            position=np.array([r * np.cos(angle), r * np.sin(angle), rng.uniform(0.3, 1.2)]),
            extent=np.array([0.3, 0.3, 0.35]),
            present_from=gap_start,
        ))
        events.append(ChangeEvent(oid, "added", gap_start))

    return SimWorld(objs), events


# --- 실행 -----------------------------------------------------------------

def run(world: SimWorld, trajectory: CameraTrajectory,
        condition: ConditionFn | None = None,
        camera: CameraModel | None = None,
        sensor_config: SensorConfig | None = None,
        seed: int = 0, name: str = "sequence",
        changes: list[ChangeEvent] | None = None) -> Sequence:
    """궤적을 따라 관측을 생성한다. 같은 seed 면 완전히 동일한 결과."""
    camera = camera or CameraModel()
    sensor = SimSensor(camera, sensor_config, seed)
    condition = condition or constant_condition()

    detections, truths = [], []
    for i, (t, pose) in enumerate(zip(trajectory.stamps, trajectory.poses)):
        ev = condition(float(t))
        det, truth = sensor.observe(world, pose, float(t), ev, i, COCO_CLASSES)
        detections.append(det)
        truths.append(truth)

    return Sequence(name, world, trajectory, detections, truths,
                    changes or [], camera)


# --- 시나리오 --------------------------------------------------------------

def scenario_static(seed: int = 0, **condition_kwargs) -> Sequence:
    """정적 장면 주회. 기준선."""
    world = static_room(12, seed=seed)
    traj = CameraTrajectory.loop(radius=3.0, laps=1.0, n=160)
    return run(world, traj, constant_condition(**condition_kwargs),
               seed=seed, name="static")


def scenario_dynamic(seed: int = 0, walkers: int = 3, **condition_kwargs) -> Sequence:
    """사람이 걸어다니는 장면. 동적 마스킹과 정적/동적 판정 검증."""
    world = add_walkers(static_room(10, seed=seed), walkers, duration=8.0, seed=seed + 1)
    traj = CameraTrajectory.loop(radius=3.0, laps=1.0, n=160)
    return run(world, traj, constant_condition(**condition_kwargs),
               seed=seed, name="dynamic")


def scenario_revisit_with_changes(seed: int = 0, **condition_kwargs) -> Sequence:
    """재방문 + 장면 변화. 변화 검출과 루프 클로저 병합 검증."""
    base = static_room(12, seed=seed)
    traj = CameraTrajectory.revisit(radius=3.0, n_per_visit=120, gap_seconds=30.0)
    gap_start = float(traj.stamps[119]) + 1.0

    world, events = apply_changes(base, gap_start, seed=seed + 2)
    return run(world, traj, constant_condition(**condition_kwargs),
               seed=seed, name="revisit_changes", changes=events)


def scenario_condition_sweep(channel: str = "haze", levels: int = 6,
                             seed: int = 0) -> list[Sequence]:
    """동일 궤적 / 동일 세계 / 조건만 변화.

    Sigma_k(E) 를 적합하려면 반드시 필요한 데이터다. 실제 데이터셋에는 없다.
    """
    world = static_room(12, seed=seed)
    traj = CameraTrajectory.loop(radius=3.0, laps=1.0, n=160)

    out = []
    for i in range(levels):
        level = i / max(1, levels - 1)
        out.append(run(world, traj, constant_condition(**{channel: level}),
                       seed=seed, name=f"{channel}={level:.2f}"))
    return out


def scenario_degradation_burst(seed: int = 0, channel: str = "haze",
                               level: float = 0.9) -> Sequence:
    """중간 구간만 심하게 열화. 조건이 걷힌 뒤 복구되는지 본다.

    '지도를 재시작하지 않고 수리한다'는 주장은 이 시나리오로만 검증된다.
    """
    world = static_room(12, seed=seed)
    traj = CameraTrajectory.loop(radius=3.0, laps=2.0, n=300)
    t0, t1 = float(traj.stamps[100]), float(traj.stamps[200])
    return run(world, traj, burst_condition(channel, level, t0, t1),
               seed=seed, name=f"burst_{channel}")
