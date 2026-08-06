"""YOLO 런타임 브리지.

WME 에서 허용된 유일한 인지 모델이다. C++ TensorRT 백엔드가 준비되기 전까지
Python 쪽에서 검출을 만들어 엔진에 주입한다.

후처리(디코딩 + letterbox 역변환 + 클래스별 NMS)는 백엔드와 무관하게 순수
numpy 로 구현했다. 모델 파일 없이도 테스트할 수 있고, C++ 구현과 같은
테스트 벡터로 차등 검증이 가능하다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    box: tuple[float, float, float, float]     # x, y, w, h (원본 이미지 좌표)
    confidence: float
    class_scores: np.ndarray | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.box
        return x, y, x + w, y + h


@dataclass
class DetectionSet:
    stamp: float
    frame_id: int
    items: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.items)


# --- 후처리 ----------------------------------------------------------------

def box_iou(a: tuple[float, float, float, float],
            b: tuple[float, float, float, float]) -> float:
    """xywh 두 박스의 IoU."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0

    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def non_max_suppression(dets: list[Detection], iou_threshold: float = 0.45,
                        max_output: int = 300) -> list[Detection]:
    """클래스별 NMS.

    서로 다른 클래스는 억제하지 않는다. 사람이 든 컵이 사람에 의해 지워지면 안 된다.
    동점은 입력 인덱스 순으로 깨서 결정적 결과를 보장한다.
    """
    if not dets:
        return []

    order = sorted(range(len(dets)), key=lambda i: (-dets[i].confidence, i))
    suppressed = [False] * len(dets)
    kept: list[Detection] = []

    for oi, i in enumerate(order):
        if suppressed[i] or len(kept) >= max_output:
            continue
        kept.append(dets[i])
        for j in order[oi + 1:]:
            if suppressed[j] or dets[j].class_id != dets[i].class_id:
                continue
            if box_iou(dets[i].box, dets[j].box) > iou_threshold:
                suppressed[j] = True
    return kept


def letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int
                     ) -> tuple[float, float, float]:
    """종횡비 유지 리사이즈의 (scale, pad_x, pad_y).

    letterbox 를 쓰면서 역변환을 빼먹는 것이 YOLO 통합에서 가장 흔한 버그다.
    """
    scale = min(dst_w / src_w, dst_h / src_h)
    pad_x = (dst_w - src_w * scale) * 0.5
    pad_y = (dst_h - src_h * scale) * 0.5
    return scale, pad_x, pad_y


def decode_yolov8(output: np.ndarray, src_size: tuple[int, int],
                  input_size: tuple[int, int], class_names: list[str] | None = None,
                  conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                  max_output: int = 300, keep_scores: bool = False) -> list[Detection]:
    """YOLOv8 원시 출력 -> Detection 목록.

    Args:
        output: (4 + nc, N) 또는 (1, 4 + nc, N). 박스는 입력 해상도 기준 cxcywh.
        src_size: (width, height) 원본 이미지
        input_size: (width, height) 모델 입력
        keep_scores: 클래스 확률 분포 전체를 보존할지.
            WME 는 기본으로 보존한다 - argmax 만 남기면 재식별 시 오분류를
            되돌릴 수 없기 때문이다.
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] < 5:
        raise ValueError(f"예상 밖 출력 형태: {arr.shape}")

    boxes = arr[:4].T                       # (N, 4) cxcywh
    scores = arr[4:].T                      # (N, nc)

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    src_w, src_h = src_size
    in_w, in_h = input_size
    scale, pad_x, pad_y = letterbox_params(src_w, src_h, in_w, in_h)

    # letterbox 역변환 후 원본 좌표로
    cx = (boxes[:, 0] - pad_x) / scale
    cy = (boxes[:, 1] - pad_y) / scale
    w = boxes[:, 2] / scale
    h = boxes[:, 3] / scale
    x = cx - w * 0.5
    y = cy - h * 0.5

    # 화면 밖으로 나간 박스를 잘라낸다
    x2 = np.clip(x + w, 0.0, src_w)
    y2 = np.clip(y + h, 0.0, src_h)
    x = np.clip(x, 0.0, src_w)
    y = np.clip(y, 0.0, src_h)

    dets: list[Detection] = []
    for i in range(len(x)):
        bw, bh = float(x2[i] - x[i]), float(y2[i] - y[i])
        if bw <= 1.0 or bh <= 1.0:
            continue
        cid = int(class_ids[i])
        name = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        dets.append(Detection(
            class_id=cid,
            class_name=name,
            box=(float(x[i]), float(y[i]), bw, bh),
            confidence=float(confidences[i]),
            class_scores=scores[i].copy() if keep_scores else None,
        ))

    return non_max_suppression(dets, iou_threshold, max_output)


# --- 백엔드 ----------------------------------------------------------------

class YoloBackend(Protocol):
    """C++ IYoloRuntime 과 같은 경계."""

    def infer(self, image: np.ndarray, stamp: float, frame_id: int) -> DetectionSet: ...
    def set_confidence_scale(self, scale: float) -> None: ...


class _BaseBackend:
    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 class_names: list[str] | None = None):
        self._base_conf = conf_threshold
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names

    def set_confidence_scale(self, scale: float) -> None:
        """EnvironmentState.detection_threshold_scale 을 반영한다.

        어두우면 임계를 낮춰 검출을 살리고, 늘어난 오검출은 Confidence Engine 이
        사후확률로 걸러낸다. 임계를 고정하면 야간에 토큰이 통째로 사라진다.
        """
        self.conf_threshold = self._base_conf * max(0.05, float(scale))


class OnnxYolo(_BaseBackend):
    """ONNX Runtime 백엔드. onnxruntime 이 설치되어 있어야 한다."""

    def __init__(self, model_path: str, input_size: tuple[int, int] = (640, 640),
                 providers: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        import onnxruntime as ort           # 지연 import - 없어도 모듈 로드는 되게

        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2

        src_h, src_w = image.shape[:2]
        in_w, in_h = self.input_size
        scale, pad_x, pad_y = letterbox_params(src_w, src_h, in_w, in_h)

        resized = cv2.resize(image, (int(round(src_w * scale)), int(round(src_h * scale))))
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        y0, x0 = int(round(pad_y)), int(round(pad_x))
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR -> RGB
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

    def infer(self, image: np.ndarray, stamp: float, frame_id: int) -> DetectionSet:
        import time

        t0 = time.perf_counter()
        blob = self._preprocess(image)
        out = self.session.run(None, {self.input_name: blob})[0]
        elapsed = (time.perf_counter() - t0) * 1e3

        src_h, src_w = image.shape[:2]
        items = decode_yolov8(out, (src_w, src_h), self.input_size,
                              self.class_names, self.conf_threshold, self.iou_threshold)
        return DetectionSet(stamp, frame_id, items, elapsed)


class UltralyticsYolo(_BaseBackend):
    """ultralytics 백엔드. 세그멘테이션 모델이면 마스크도 함께 나온다."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device

    def infer(self, image: np.ndarray, stamp: float, frame_id: int) -> DetectionSet:
        import time

        t0 = time.perf_counter()
        results = self.model.predict(image, conf=self.conf_threshold,
                                     iou=self.iou_threshold, device=self.device,
                                     verbose=False)
        elapsed = (time.perf_counter() - t0) * 1e3

        items: list[Detection] = []
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                cid = int(b.cls.item())
                items.append(Detection(
                    class_id=cid,
                    class_name=names.get(cid, str(cid)),
                    box=(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(b.conf.item()),
                ))
        return DetectionSet(stamp, frame_id, items, elapsed)


class ScriptedYolo(_BaseBackend):
    """미리 정해진 검출을 재생하는 백엔드.

    결정적 재현 테스트와 엔진 통합 테스트에 쓴다. 모델 추론이 끼면
    SLAM 버그가 재현되지 않는다.
    """

    def __init__(self, script: dict[int, list[Detection]], **kwargs):
        super().__init__(**kwargs)
        self.script = script

    def infer(self, image: np.ndarray, stamp: float, frame_id: int) -> DetectionSet:
        items = [d for d in self.script.get(frame_id, []) if d.confidence >= self.conf_threshold]
        return DetectionSet(stamp, frame_id, items, 0.0)


# COCO 80 클래스. WME 어포던스 표가 이 어휘를 전제한다.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
