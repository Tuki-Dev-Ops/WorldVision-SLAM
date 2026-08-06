#!/usr/bin/env python3
"""StereoSGBM 프런트엔드를 TUM 의 **실측 깊이** 에 대고 잰다.

왜 이게 필요한가
----------------
`tests/test_stereo_depth.cpp` 는 합성 텍스처로 규약을 잠근다 - 16 배 고정소수점,
f*B, 무효 픽셀 표기. 하지만 그 장면에는 실제 영상이 가진 것이 없다: 무텍스처
벽, 깊이 불연속, 반복 패턴, 노출 차이. 합성 시험만으로 "KITTI 에서 쓸 만하다"
고 말할 수 없다.

여기서는 TUM RGB-D 의 실제 프레임과 실제 깊이를 쓴다. 깊이는 센서에서 온
값이므로 정답으로 쓸 수 있다. 그 깊이로 왼쪽 영상을 워프해 가상 우안을 만들고,
SGBM 이 그 깊이를 되찾는지 본다.

정직하게 말해 둘 것
-------------------
가상 우안은 비교 대상인 바로 그 깊이로 만들었다. 그래서 이 측정은
**실제 스테레오 정확도가 아니라 정합기의 자기일관성** 이다. 진짜 스테레오가
겪는 두 가지가 빠져 있다:
  - 폐색: 왼쪽에만 보이는 픽셀. 워프는 이웃 값으로 늘려 채우므로 대응이
    "있는 것처럼" 된다. 실제로는 없다.
  - 광도 불일치: 두 카메라의 노출/비네팅 차이. 여기서는 정확히 0 이다.
둘 다 SGBM 을 **유리하게** 만든다. 따라서 여기 나오는 오차는 하한이다.
그래도 무텍스처 영역과 깊이 불연속은 실제 그대로라서, 합성 시험이 답하지
못하는 "이 장면에서 몇 %의 픽셀에 깊이가 생기는가" 는 여기서 답이 나온다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "tests"))
import conftest  # noqa: E402,F401  (os.add_dll_directory)

import cv2  # noqa: E402
from wme import _core  # noqa: E402

# TUM freiburg1 기본 내부파라미터와 깊이 스케일
TUM_FX = 517.306408
TUM_DEPTH_SCALE = 5000.0


def load_pairs(seq: Path, count: int, stride: int):
    """rgb.txt / depth.txt 를 타임스탬프로 맞춰 (rgb, depth) 경로 쌍을 낸다."""
    def read(name):
        out = []
        for line in (seq / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            t, p = line.split()
            out.append((float(t), seq / p))
        return out

    rgb, depth = read("rgb.txt"), read("depth.txt")
    dts = np.array([t for t, _ in depth])
    pairs = []
    for i in range(0, len(rgb), stride):
        t, rp = rgb[i]
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) > 0.02:
            continue
        pairs.append((rp, depth[j][1]))
        if len(pairs) >= count:
            break
    return pairs


def synth_right(gray: np.ndarray, depth_m: np.ndarray, fx: float, baseline: float):
    """실측 깊이로 왼쪽을 오른쪽으로 워프. 깊이가 없는 화소는 남겨 둔다."""
    h, w = gray.shape
    d = np.zeros_like(depth_m, dtype=np.float32)
    ok = depth_m > 0
    d[ok] = (fx * baseline / depth_m[ok]).astype(np.float32)

    xx = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    yy = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    right = cv2.remap(gray, xx + d, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return right, ok


def run(seq: Path, count: int, stride: int, baseline: float, cfg_over: dict,
        auto_range: bool = False, scene_min_depth: float = 1.0):
    pairs = load_pairs(seq, count, stride)
    if not pairs:
        raise SystemExit(f"{seq.name}: rgb/depth 쌍을 찾지 못했다")

    cfg = _core.StereoDepthConfig()
    cfg.focal_px = TUM_FX
    cfg.baseline_m = baseline
    cfg.num_disparities = (
        _core.StereoDepth.required_disparities(TUM_FX, baseline, scene_min_depth)
        if auto_range else 128)
    cfg.block_size = 5
    cfg.min_depth_m = 0.3
    cfg.max_depth_m = 12.0
    cfg.max_depth_sigma_m = 0.20      # 실내: 20 cm 오차 예산
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    stereo = _core.StereoDepth(cfg)

    rel, absd, cover, gt_cover, ratios, clipped = [], [], [], [], [], []
    min_repr = float("nan")
    for rp, dp in pairs:
        bgr = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gt = raw.astype(np.float32) / TUM_DEPTH_SCALE

        right, ok = synth_right(gray, gt, TUM_FX, baseline)
        res = stereo.compute(np.ascontiguousarray(gray), np.ascontiguousarray(right))
        z = np.asarray(res.depth)
        clipped.append(res.clipped_ratio)
        min_repr = res.min_representable_depth_m

        # 시차만큼 왼쪽 띠는 대응이 없다. 정직하게 잘라 낸다.
        m = np.zeros_like(ok)
        m[:, cfg.num_disparities + 8:] = True
        both = m & ok & (z > 0)
        gt_only = m & ok

        gt_cover.append(float(gt_only.mean()))
        cover.append(float(both.sum()) / max(1, int(gt_only.sum())))
        if both.sum() < 100:
            continue
        e = np.abs(z[both] - gt[both])
        absd.append(float(np.median(e)))
        rel.append(float(np.median(e / gt[both])))
        ratios.append(float(np.median(z[both] / gt[both])))

    def med(v):
        return float(np.median(v)) if v else float("nan")

    return {
        "sequence": seq.name,
        "frames": len(pairs),
        "baseline_m": baseline,
        "num_disparities": cfg.num_disparities,
        "min_representable_depth_m": min_repr,
        "clipped_ratio": med(clipped),
        "min_valid_disparity_px": _core.StereoDepth.min_valid_disparity(cfg),
        "gt_depth_coverage": med(gt_cover),
        "stereo_coverage_of_gt": med(cover),
        "median_abs_err_m": med(absd),
        "median_rel_err": med(rel),
        "median_scale_ratio": med(ratios),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--baselines", nargs="*", type=float, default=[0.10, 0.20, 0.54])
    ap.add_argument("--auto-range", action="store_true",
                    help="num_disparities 를 (베이스라인, 최근접거리) 로부터 정한다")
    ap.add_argument("--scene-min-depth", type=float, default=1.0,
                    help="--auto-range 가 쓰는 장면 최근접 거리 (m)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stereo_validate.json")
    args = ap.parse_args()

    data = ROOT / "data"
    names = args.sequences or ["rgbd_dataset_freiburg1_desk",
                               "rgbd_dataset_freiburg1_room",
                               "rgbd_dataset_freiburg3_structure_texture_far",
                               "rgbd_dataset_freiburg3_nostructure_notexture_far"]
    rows = []
    for n in names:
        seq = data / n
        if not seq.exists():
            print(f"건너뜀 (없음): {n}")
            continue
        for b in args.baselines:
            r = run(seq, args.frames, args.stride, b, {},
                    auto_range=args.auto_range, scene_min_depth=args.scene_min_depth)
            rows.append(r)
            print(f"{r['sequence'].replace('rgbd_dataset_', ''):38s} "
                  f"B={b:.2f} D={r['num_disparities']:3d} "
                  f"Zmin={r['min_representable_depth_m']:5.2f}m  "
                  f"복원 {r['stereo_coverage_of_gt']*100:5.1f}%  "
                  f"오차 {r['median_abs_err_m']*100:6.2f}cm "
                  f"({r['median_rel_err']*100:5.2f}%)  "
                  f"스케일 {r['median_scale_ratio']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
