"""M5: 데이터 연관 전략 비교.

docs/04-unified-objective.md 5.4 가 "가장 깊은 구조적 결함"으로 지목한 지점을
실제로 측정한다. 지금까지의 모든 실험은 참 연관을 써서 이 문제를 피해 왔다.

검증할 주장
  1. 탐욕적 매칭은 붙어 있는 동일 클래스 객체에서 정체성을 섞는다.
  2. 프레임별 전역 최적(현행 방식)은 개선되지만 되돌릴 수 없는 결정을 한다.
  3. MHT 는 그 결정을 미루고, 객체 수준에서는 계산량이 감당된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.metrics import identity_metrics
from ..sim.scenarios import Sequence
from ..sim.sensor import severity_of
from ..sim.world import CameraModel
from .trackers import (
    AssociationConfig, DeferredTracker, GreedyTracker, HungarianTracker,
    Measurement, MhtTracker, Tracker,
)


@dataclass
class AssociationResult:
    name: str
    accuracy: float               # 참 연관과 일치한 비율
    id_switches: int
    duplicate_rate: float
    tracks_created: int
    true_objects: int
    deferred: int = 0
    hypotheses: int = 0

    @property
    def track_inflation(self) -> float:
        """참 객체 1개당 만들어진 트랙 수. 1.0 이 이상적."""
        return self.tracks_created / max(1, self.true_objects)

    def row(self) -> str:
        return (f"  {self.name:<12} {self.accuracy:8.3f} {self.id_switches:6d} "
                f"{self.duplicate_rate:8.3f} {self.tracks_created:7d} "
                f"{self.track_inflation:8.2f} {self.deferred:6d}")


def _measurements(seq: Sequence, frame_index: int, cam: CameraModel,
                  sigma_px: float, depth_coeff: float
                  ) -> tuple[list[Measurement], list[int]]:
    """한 프레임의 월드 좌표 관측과 그 참 객체 ID."""
    det = seq.detections[frame_index]
    truth = seq.truths[frame_index]
    sev = severity_of(truth.evidence)

    out, gt = [], []
    for i, item in enumerate(det.items):
        d = truth.detection_depth[i]
        if d <= 0.1:
            continue
        x, y, w, h = item.box
        u, v = x + w * 0.5, y + h * 0.5
        p_cam = np.array([(u - cam.cx) * d / cam.fx, (v - cam.cy) * d / cam.fy, d])

        s_px = sigma_px * (1.0 + 4.0 * sev)
        s_d = depth_coeff * d * d * (1.0 + 3.0 * sev)
        J = np.array([[d / cam.fx, 0.0, (u - cam.cx) / cam.fx],
                      [0.0, d / cam.fy, (v - cam.cy) / cam.fy],
                      [0.0, 0.0, 1.0]])
        R_cam = J @ np.diag([s_px ** 2, s_px ** 2, s_d ** 2]) @ J.T

        R_world = truth.T_world_cam.R @ R_cam @ truth.T_world_cam.R.T
        out.append(Measurement(truth.T_world_cam @ p_cam, R_world, item.class_id))
        gt.append(truth.detection_to_object[i])
    return out, gt


def run_tracker(tracker: Tracker, seq: Sequence, sigma_px: float = 2.0,
                depth_coeff: float = 0.006,
                retrospective: bool = False) -> AssociationResult:
    """시퀀스 하나를 처리하고 정체성 지표를 낸다.

    retrospective=True 는 MHT 전용이다. MHT 는 필터가 아니라 평활기라서,
    온라인 출력만 보면 그 이점을 측정할 수 없다 - 나중에 뒤집힌 결정이
    반영되지 않기 때문이다.
    """
    cam = seq.camera
    frames: list[dict[int, int]] = []
    gt_per_frame: list[list[int]] = []
    online: list[list[int]] = []
    matched = total = 0
    deferred = 0

    for fi in range(len(seq.detections)):
        meas, gt = _measurements(seq, fi, cam, sigma_px, depth_coeff)
        if not meas:
            continue
        assigned = tracker.update(meas, fi)
        tracker.prune(fi)
        gt_per_frame.append(gt)
        online.append(assigned)

    # 평활 이력이 있으면 그것을 쓴다. 프레임 수가 맞을 때만 유효하다.
    sequences = online
    if retrospective and hasattr(tracker, "retrospective"):
        hist = [a for _f, a in tracker.retrospective()]
        if len(hist) == len(online):
            sequences = hist

    for gt, assigned in zip(gt_per_frame, sequences):
        frame_map: dict[int, int] = {}
        for gid, tid in zip(gt, assigned):
            if gid < 0:                      # 오검출은 정체성 평가 대상이 아니다
                continue
            total += 1
            if tid < 0:
                deferred += 1
                continue
            matched += 1
            frame_map[gid] = tid
        frames.append(frame_map)

    # 확정 트랙만 평가한다. 한두 번 나타났다 사라진 오검출 잔재까지 세면
    # 전략 간 차이가 쓰레기 트랙 수에 파묻힌다.
    confirmed_ids = set(tracker.confirmed)
    frames = [{g: t for g, t in f.items() if t in confirmed_ids} for f in frames]
    frames = [f for f in frames if f]

    ident = identity_metrics(frames) if frames else None
    true_objects = len({g for f in frames for g in f})

    return AssociationResult(
        name=tracker.name.replace("Tracker", ""),
        accuracy=matched / max(1, total),
        id_switches=ident.id_switches if ident else 0,
        duplicate_rate=ident.duplicate_rate if ident else 0.0,
        tracks_created=len(confirmed_ids),
        true_objects=true_objects,
        deferred=deferred,
        hypotheses=getattr(tracker, "hypothesis_count", 0),
    )


@dataclass
class M5Report:
    scenario: str
    results: list[AssociationResult] = field(default_factory=list)

    def by_name(self, name: str) -> AssociationResult | None:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def lines(self) -> list[str]:
        out = [f"--- {self.scenario} ---",
               f"  {'strategy':<12} {'accuracy':>8} {'IDS':>6} {'dup':>8} "
               f"{'tracks':>7} {'inflate':>8} {'defer':>6}"]
        out.extend(r.row() for r in self.results)
        return out

    def __str__(self) -> str:
        return "\n".join(self.lines())


def compare_strategies(seq: Sequence, cfg: AssociationConfig | None = None,
                       scenario: str = "scenario") -> M5Report:
    cfg = cfg or AssociationConfig()
    report = M5Report(scenario)
    for cls in (GreedyTracker, HungarianTracker, DeferredTracker, MhtTracker):
        report.results.append(run_tracker(cls(cfg), seq))

    # MHT 는 온라인/평활 두 가지로 재는 것이 정직하다. 평활 이득을 온라인
    # 지표에 섞어 보고하면 실시간 성능을 과대평가하게 된다.
    smoothed = run_tracker(MhtTracker(cfg), seq, retrospective=True)
    smoothed.name = "Mht (smooth)"
    report.results.append(smoothed)
    return report
