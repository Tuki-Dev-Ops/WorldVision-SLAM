"""검출 -> 트랙 -> 믿음 -> 스냅샷 -> 변화 의 전 구간 파이프라인.

"무엇이 바뀌었는가" 를 주장이 아니라 수치로 만드는 경로다. 참값은 시뮬레이터가
주고(scenario_revisit_with_changes), 채점은 eval.metrics.change_metrics 가 한다.

카메라 포즈는 참값을 쓴다. 포즈 오차가 섞이면 변화 검출 실패가 연관 탓인지
포즈 탓인지 구분할 수 없기 때문이다 - M3 와 같은 분리 원칙이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..association.trackers import (
    AssociationConfig, HungarianTracker, Measurement, TrackState,
)
from ..eval.metrics import ChangeEvent
from ..reference.confidence import Beliefs, ConfidenceConfig, ConfidenceEngine
from ..sim.scenarios import Sequence
from ..sim.sensor import severity_of
from ..sim.world import CameraModel
from .change import ChangeConfig, ChangeDetector
from .state import TokenBelief, WorldSnapshot, WorldState


@dataclass
class PipelineConfig:
    association: AssociationConfig = field(default_factory=AssociationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)

    sigma_px: float = 2.0
    depth_coeff: float = 0.006
    snapshot_interval: float = 2.0        # s

    # 시야 안인데 검출되지 않은 트랙만 부재의 증거로 센다.
    # 시야 밖은 부재의 증거가 아니다 (manifesto 약속 3).
    frustum_margin_px: float = 8.0

    # 장거리 재식별: 위치 게이트를 벗어난 곳에서 다시 나타난 같은 클래스 객체를
    # 이어붙인다. 이것이 없으면 '이동' 이 '제거 + 추가' 로 쪼개져 보고된다.
    enable_long_range_reid: bool = True
    reid_max_distance: float = 3.0        # m
    reid_min_observations: int = 4
    reid_extent_tolerance: float = 0.5    # 상대 오차


@dataclass
class PipelineResult:
    state: WorldState
    snapshots: list[WorldSnapshot]
    detected: list[ChangeEvent] = field(default_factory=list)
    frames_processed: int = 0
    reid_links: dict[int, int] = field(default_factory=dict)   # 새 트랙 -> 원래 트랙

    # 트랙 ID -> 참 객체 ID (다수결). 채점 전용이며 추정에는 쓰이지 않는다.
    # 이 매핑이 없으면 검출은 트랙 번호로, 참값은 객체 번호로 나와
    # 아무리 잘 맞춰도 재현율이 0 으로 나온다.
    track_to_object: dict[int, int] = field(default_factory=dict)

    def as_object_events(self) -> list[ChangeEvent]:
        """검출 사건의 ID 를 참 객체 ID 로 옮긴다. 매핑이 없으면 버린다."""
        out = []
        for e in self.detected:
            oid = self.track_to_object.get(e.object_id)
            if oid is not None:
                out.append(ChangeEvent(oid, e.kind, e.time))
        return out


def measurements_from_frame(seq: Sequence, index: int, cam: CameraModel,
                            sigma_px: float, depth_coeff: float
                            ) -> tuple[list[Measurement], list[int], list[np.ndarray]]:
    """한 프레임의 월드 좌표 관측 + 참 객체 ID + 대략적 치수."""
    det = seq.detections[index]
    truth = seq.truths[index]
    sev = severity_of(truth.evidence)

    out, gt, extents = [], [], []
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
        R = truth.T_world_cam.R @ R_cam @ truth.T_world_cam.R.T

        out.append(Measurement(truth.T_world_cam @ p_cam, R, item.class_id))
        gt.append(truth.detection_to_object[i])
        extents.append(np.array([0.5 * w * d / cam.fx, 0.5 * h * d / cam.fy,
                                 0.25 * (w * d / cam.fx + h * d / cam.fy)]))
    return out, gt, extents


class WorldPipeline:
    """관측을 세계 상태로 통합하고 변화를 낸다."""

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.tracker = HungarianTracker(self.cfg.association)
        self.confidence = ConfidenceEngine(self.cfg.confidence)
        self.state = WorldState()
        self.detector = ChangeDetector(self.cfg.change)

        self._beliefs: dict[int, Beliefs] = {}
        self._meta: dict[int, dict] = {}          # track_id -> class/extent/시각
        # 정적 판정 창의 기준점 (원시 관측 + 시각). 프레임 간격이 아니라
        # 이 창으로 판정한다 - 16.2 의 창길이 곡선 참조.
        self._static_ref: dict[int, dict] = {}
        self._reid: dict[int, int] = {}
        # 채점용 다수결 집계. 추정 경로에서는 절대 읽지 않는다.
        self._gt_votes: dict[int, dict[int, int]] = {}

    # --- 내부 -------------------------------------------------------------

    def _in_view(self, position: np.ndarray, truth, cam: CameraModel) -> bool:
        p_cam = truth.T_world_cam.inverse() @ position
        if p_cam[2] < 0.2:
            return False
        u = cam.fx * p_cam[0] / p_cam[2] + cam.cx
        v = cam.fy * p_cam[1] / p_cam[2] + cam.cy
        m = self.cfg.frustum_margin_px
        return m <= u < cam.width - m and m <= v < cam.height - m

    def _try_reid(self, new_id: int, track: TrackState) -> int | None:
        """장거리 재식별.

        위치 게이트 밖에서 같은 클래스 객체가 새로 나타났고, 원래 트랙이
        그 사이 사라졌다면 같은 개체가 옮겨간 것일 수 있다. 이것을 잇지 않으면
        '이동' 이 '제거 + 추가' 두 사건으로 쪼개져 보고된다 - 계획기에게는
        전혀 다른 뜻이다.

        보수적으로 건다: 클래스가 같고, 치수가 비슷하고, 후보가 유일할 때만.
        """
        cfg = self.cfg
        if not cfg.enable_long_range_reid:
            return None
        meta_new = self._meta.get(new_id)
        if meta_new is None or track.count < cfg.reid_min_observations:
            return None

        candidates = []
        for tid, meta in self._meta.items():
            if tid == new_id or tid in self._reid.values():
                continue
            if meta["class_id"] != meta_new["class_id"]:
                continue
            b = self._beliefs.get(tid)
            # 원래 트랙이 아직 살아 있으면 옮겨간 게 아니다
            if b is None or b.existence > cfg.change.removed_existence:
                continue
            if meta["observations"] < cfg.reid_min_observations:
                continue

            d = float(np.linalg.norm(meta["position"] - track.position))
            if d > cfg.reid_max_distance:
                continue
            e_ref, e_new = meta["extent"], meta_new["extent"]
            rel = float(np.linalg.norm(e_new - e_ref) / max(np.linalg.norm(e_ref), 1e-6))
            if rel > cfg.reid_extent_tolerance:
                continue
            candidates.append((d, tid))

        # 후보가 여럿이면 잇지 않는다. 애매한 재식별은 정체성을 섞는다.
        if len(candidates) != 1:
            return None
        return candidates[0][1]

    # --- 처리 -------------------------------------------------------------

    def process(self, seq: Sequence) -> PipelineResult:
        cfg = self.cfg
        cam = seq.camera
        snapshots: list[WorldSnapshot] = []
        last_commit = -np.inf

        for fi in range(len(seq.detections)):
            truth = seq.truths[fi]
            stamp = float(truth.stamp)
            sev = severity_of(truth.evidence)
            reliability = max(0.05, 1.0 - sev)

            meas, gt_ids, extents = measurements_from_frame(
                seq, fi, cam, cfg.sigma_px, cfg.depth_coeff)

            assigned = self.tracker.update(meas, fi) if meas else []
            self.tracker.prune(fi)

            updated: set[int] = set()
            for m, tid, ext, gid in zip(meas, assigned, extents, gt_ids):
                if tid < 0:
                    continue
                updated.add(tid)
                if gid >= 0:
                    votes = self._gt_votes.setdefault(tid, {})
                    votes[gid] = votes.get(gid, 0) + 1
                b = self._beliefs.setdefault(tid, Beliefs())
                prev = self._meta.get(tid)

                self.confidence.on_observed(b, 0.85, reliability)

                # 정적 판정은 인접 프레임이 아니라 누적 창으로 한다.
                # 20 Hz 에서 프레임 간격은 판정 최소 시간(0.5 s)에 못 미쳐
                # 그대로 넘기면 갱신이 한 번도 일어나지 않는다 - 채널이 조용히
                # 죽는다. 기준점을 붙들었다가 창이 찼을 때 그 사이 이동을 본다.
                # 넘기는 값은 창 양 끝의 *원시 관측* 차이다. 융합 위치의 차이는
                # 필터의 동역학이지 물체의 운동이 아니다 (16.3).
                ref = self._static_ref.get(tid)
                if ref is None:
                    self._static_ref[tid] = {"position": m.z, "stamp": stamp}
                else:
                    window = stamp - ref["stamp"]
                    if window >= self.confidence.cfg.static_min_dt:
                        # 척도는 관측 불확실성이다. 융합 추정의 것을 쓰면 관측
                        # 수에 따라 줄어들어 창 변위의 통계와 어긋난다 (16.3).
                        b.meas_sigma = float(np.sqrt(np.trace(m.R) / 3.0))
                        disp = float(np.linalg.norm(m.z - ref["position"]))
                        self.confidence.update_static(b, disp, window, reliability)
                        self._static_ref[tid] = {"position": m.z, "stamp": stamp}

                self._meta[tid] = {
                    "class_id": m.class_id,
                    "extent": ext if prev is None else 0.7 * prev["extent"] + 0.3 * ext,
                    "position": m.z,
                    "stamp": stamp,
                    "first_seen": prev["first_seen"] if prev else stamp,
                    "observations": (prev["observations"] + 1) if prev else 1,
                }

                # 새로 확정된 트랙이면 장거리 재식별을 시도한다
                track = self.tracker.tracks.get(tid)
                if track is not None and tid not in self._reid:
                    origin = self._try_reid(tid, track)
                    if origin is not None:
                        self._reid[tid] = origin

            # 시야 안인데 안 보인 트랙 - 이것만이 부재의 증거다
            for tid, track in self.tracker.tracks.items():
                if tid in updated:
                    continue
                b = self._beliefs.get(tid)
                if b is None:
                    continue
                if self._in_view(track.position, truth, cam):
                    self.confidence.on_missed(b, 1.0, reliability)
                else:
                    self.confidence.on_out_of_view(b, 1.0 / 30.0)

            # 믿음을 세계 상태에 반영
            for tid, track in self.tracker.tracks.items():
                b = self._beliefs.get(tid)
                meta = self._meta.get(tid)
                if b is None or meta is None:
                    continue
                lifecycle = ("displaced" if b.existence < cfg.change.removed_existence
                             else ("active" if tid in updated else "occluded"))
                self.state.put(TokenBelief(
                    token_id=self._reid.get(tid, tid),
                    class_id=meta["class_id"],
                    position=track.position,
                    covariance=track.covariance,
                    extent=meta["extent"],
                    existence=b.existence,
                    identity=b.identity,
                    static_belief=b.static,
                    lifecycle=lifecycle,
                    first_seen=meta["first_seen"],
                    last_seen=meta["stamp"],
                    observation_count=meta["observations"],
                ))

            if stamp - last_commit >= cfg.snapshot_interval:
                snapshots.append(self.state.commit(stamp))
                last_commit = stamp

        if not snapshots or snapshots[-1].stamp < self.state.stamp:
            snapshots.append(self.state.commit(max(self.state.stamp + 1e-6,
                                                   float(seq.truths[-1].stamp))))

        # 다수결로 트랙 -> 참 객체 매핑을 만든다. 재식별로 이어붙인 트랙은
        # 원래 트랙의 이름으로 보고되므로 매핑도 그쪽으로 접어준다.
        mapping: dict[int, int] = {}
        for tid, votes in self._gt_votes.items():
            if votes:
                mapping[self._reid.get(tid, tid)] = max(votes, key=votes.get)

        return PipelineResult(self.state, snapshots, [], len(seq.detections),
                              dict(self._reid), mapping)


def detect_revisit_changes(seq: Sequence, config: PipelineConfig | None = None,
                           gap_seconds: float = 5.0) -> PipelineResult:
    """재방문 시퀀스에서 변화를 검출한다.

    시간 간격이 가장 큰 지점을 기준으로 방문 1 / 방문 2 를 나눈다. 변화는 두
    관측 *사이* 에만 정의되므로, 간격 이전의 마지막 스냅샷과 이후의 마지막
    스냅샷을 비교하는 것이 옳다.
    """
    pipeline = WorldPipeline(config)
    result = pipeline.process(seq)

    stamps = np.array([s.stamp for s in result.snapshots])
    if len(stamps) < 2:
        return result

    gaps = np.diff(stamps)
    split = int(np.argmax(gaps))
    if gaps[split] < gap_seconds:
        return result       # 재방문 구조가 아니다

    before = result.snapshots[split]
    after = result.snapshots[-1]
    result.detected = pipeline.detector.compare(before, after)
    return result
