"""YOLO 후처리 검증. 모델 파일 없이 순수 numpy 경로만 확인한다.

letterbox 역변환 누락이 YOLO 통합에서 가장 흔한 버그라 집중적으로 검증한다.
"""

import numpy as np
import pytest

from wme.yolo import (
    COCO_CLASSES, Detection, ScriptedYolo, box_iou, decode_yolov8,
    letterbox_params, non_max_suppression,
)


def det(cid, x, y, w, h, conf):
    return Detection(cid, str(cid), (x, y, w, h), conf)


# --- IoU / NMS -------------------------------------------------------------

def test_box_iou_known_values():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_nms_suppresses_overlapping_same_class():
    dets = [det(1, 100, 100, 50, 50, c) for c in (0.9, 0.8, 0.7)]
    kept = non_max_suppression(dets, 0.5)
    assert len(kept) == 1
    assert kept[0].confidence == pytest.approx(0.9)


def test_nms_keeps_overlapping_different_classes():
    """사람이 든 컵이 사람에 의해 지워지면 안 된다."""
    dets = [det(0, 100, 100, 80, 200, 0.9), det(41, 110, 150, 30, 30, 0.6)]
    assert len(non_max_suppression(dets, 0.3)) == 2


def test_nms_keeps_non_overlapping():
    dets = [det(1, 0, 0, 40, 40, 0.9), det(1, 200, 200, 40, 40, 0.8)]
    assert len(non_max_suppression(dets, 0.5)) == 2


def test_nms_is_deterministic_on_ties():
    dets = [det(1, i * 500, 0, 40, 40, 0.5) for i in range(4)]
    a = non_max_suppression(dets, 0.5)
    b = non_max_suppression(dets, 0.5)
    assert [d.box for d in a] == [d.box for d in b]


def test_nms_respects_max_output():
    dets = [det(1, i * 500, 0, 40, 40, 0.9 - i * 0.01) for i in range(10)]
    assert len(non_max_suppression(dets, 0.5, max_output=3)) == 3


def test_nms_empty():
    assert non_max_suppression([], 0.5) == []


# --- letterbox -------------------------------------------------------------

def test_letterbox_preserves_aspect_ratio():
    scale, pad_x, pad_y = letterbox_params(1920, 1080, 640, 640)
    assert scale == pytest.approx(640 / 1920)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx((640 - 1080 * scale) / 2)


def test_letterbox_square_source_has_no_padding():
    scale, pad_x, pad_y = letterbox_params(800, 800, 640, 640)
    assert scale == pytest.approx(0.8)
    assert pad_x == pytest.approx(0.0) and pad_y == pytest.approx(0.0)


# --- 디코딩 ----------------------------------------------------------------

def make_output(boxes, class_scores, num_classes=80):
    """(4 + nc, N) 원시 출력을 만든다. 박스는 입력 해상도 기준 cxcywh."""
    n = len(boxes)
    arr = np.zeros((4 + num_classes, n), dtype=np.float32)
    for i, (cx, cy, w, h) in enumerate(boxes):
        arr[:4, i] = (cx, cy, w, h)
    for i, (cid, score) in enumerate(class_scores):
        arr[4 + cid, i] = score
    return arr


def test_decode_maps_back_to_source_coordinates():
    """letterbox 역변환이 정확해야 3D 위치 추정이 맞는다."""
    src_w, src_h = 1280, 720
    in_w, in_h = 640, 640
    scale, pad_x, pad_y = letterbox_params(src_w, src_h, in_w, in_h)

    # 원본에서 (400, 300) 중심, 200x100 크기인 박스
    true_cx, true_cy, true_w, true_h = 400.0, 300.0, 200.0, 100.0
    model_box = (true_cx * scale + pad_x, true_cy * scale + pad_y,
                 true_w * scale, true_h * scale)

    out = make_output([model_box], [(56, 0.91)])
    dets = decode_yolov8(out, (src_w, src_h), (in_w, in_h), COCO_CLASSES)

    assert len(dets) == 1
    x, y, w, h = dets[0].box
    assert x + w / 2 == pytest.approx(true_cx, abs=1e-3)
    assert y + h / 2 == pytest.approx(true_cy, abs=1e-3)
    assert w == pytest.approx(true_w, abs=1e-3)
    assert h == pytest.approx(true_h, abs=1e-3)
    assert dets[0].class_id == 56
    assert dets[0].class_name == "chair"
    assert dets[0].confidence == pytest.approx(0.91, abs=1e-6)


def test_decode_applies_confidence_threshold():
    out = make_output([(320, 320, 100, 100), (100, 100, 50, 50)],
                      [(0, 0.9), (1, 0.1)])
    dets = decode_yolov8(out, (640, 640), (640, 640), conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].class_id == 0


def test_decode_preserves_class_distribution_by_default():
    """argmax 만 남기면 재식별 시 오분류를 되돌릴 수 없다."""
    out = make_output([(320, 320, 100, 100)], [(0, 0.6)])
    out[4 + 1, 0] = 0.35            # 2순위 클래스

    dets = decode_yolov8(out, (640, 640), (640, 640), keep_scores=True)
    assert dets[0].class_scores is not None
    assert dets[0].class_scores[1] == pytest.approx(0.35, abs=1e-6)

    lean = decode_yolov8(out, (640, 640), (640, 640), keep_scores=False)
    assert lean[0].class_scores is None


def test_decode_clips_boxes_to_image():
    out = make_output([(20, 20, 200, 200)], [(0, 0.9)])
    dets = decode_yolov8(out, (640, 640), (640, 640))
    x, y, w, h = dets[0].box
    assert x >= 0.0 and y >= 0.0
    assert x + w <= 640.0 and y + h <= 640.0


def test_decode_drops_degenerate_boxes():
    out = make_output([(320, 320, 0.5, 0.5)], [(0, 0.9)])
    assert decode_yolov8(out, (640, 640), (640, 640)) == []


def test_decode_accepts_batched_output():
    out = make_output([(320, 320, 100, 100)], [(0, 0.9)])
    batched = out[None]
    assert len(decode_yolov8(batched, (640, 640), (640, 640))) == 1


def test_decode_rejects_malformed_output():
    with pytest.raises(ValueError):
        decode_yolov8(np.zeros((3, 10)), (640, 640), (640, 640))


def test_decode_empty_when_nothing_passes():
    out = make_output([(320, 320, 100, 100)], [(0, 0.01)])
    assert decode_yolov8(out, (640, 640), (640, 640), conf_threshold=0.25) == []


# --- 백엔드 ----------------------------------------------------------------

def test_confidence_scale_lowers_threshold():
    """어두우면 임계를 낮춰 검출을 살린다. 오검출은 Confidence Engine 이 걸러낸다."""
    backend = ScriptedYolo({}, conf_threshold=0.5)
    assert backend.conf_threshold == pytest.approx(0.5)

    backend.set_confidence_scale(0.4)
    assert backend.conf_threshold == pytest.approx(0.2)

    backend.set_confidence_scale(1.0)            # 기준값 대비이지 누적이 아니다
    assert backend.conf_threshold == pytest.approx(0.5)


def test_scripted_backend_is_deterministic():
    script = {
        0: [det(0, 10, 10, 50, 50, 0.9)],
        1: [det(0, 12, 10, 50, 50, 0.8), det(56, 200, 100, 60, 60, 0.7)],
    }
    backend = ScriptedYolo(script, conf_threshold=0.25)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    assert len(backend.infer(img, 1.0, 0)) == 1
    assert len(backend.infer(img, 1.1, 1)) == 2
    assert len(backend.infer(img, 1.2, 99)) == 0


def test_scripted_backend_respects_threshold():
    backend = ScriptedYolo({0: [det(0, 10, 10, 50, 50, 0.3)]}, conf_threshold=0.5)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert len(backend.infer(img, 1.0, 0)) == 0

    backend.set_confidence_scale(0.5)            # 임계 0.25 로 하향
    assert len(backend.infer(img, 1.0, 0)) == 1


def test_coco_vocabulary_is_complete():
    assert len(COCO_CLASSES) == 80
    for name in ("person", "chair", "dining table", "cup", "car"):
        assert name in COCO_CLASSES
