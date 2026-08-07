#!/usr/bin/env python3
"""모든 시퀀스의 의미 구조(평면 + 물체 상자)를 뽑아 뷰어가 읽을 파일로 만든다.

`wme_scene_export` 를 시퀀스마다 한 번씩 부른다. 뷰어는 결과를
`results/bench/scene_<name>.tsv` 에서 읽는다.

**벤치마크와 동시에 돌리지 말 것.** 벤치는 프레임당 시간을 실측해 보고하는데,
같은 기계에서 YOLO 추론이 돌면 그 수치가 오염된다. 측정값을 재는 도구와
측정을 방해하는 도구를 같이 돌리면 안 된다.

사용: python tools/scene_export_all.py [--stride 4] [--frames 150]
      [--no-yolo] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENCV_BIN = Path("C:/opencv-dl/opencv/build/x64/vc16/bin")
ORT_BIN = ROOT / "third_party" / "onnxruntime-win-x64-1.22.0" / "lib"


def find_exe() -> Path:
    for c in ("build/win/tools", "build/msvc/tools", "build/tools"):
        for name in ("wme_scene_export.exe", "wme_scene_export"):
            p = ROOT / c / name
            if p.exists():
                return p
    raise SystemExit("wme_scene_export 를 찾지 못했다 - 먼저 빌드해야 한다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frames", type=int, default=150,
                    help="시퀀스당 내보낼 프레임 수 상한 (0 = 전부)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--no-yolo", action="store_true", help="평면만 뽑는다")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    exe = find_exe()
    out_dir = ROOT / "results" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ROOT / "models" / "yolo11n.onnx"
    use_yolo = not args.no_yolo and model.exists()
    if not args.no_yolo and not model.exists():
        # 조용히 끄지 않는다. 상자가 없는 화면과 검출기를 못 찾은 화면은
        # 구분되지 않으므로, 어느 쪽인지 여기서 말해 준다.
        print(f"경고: {model} 이 없다 - 평면만 뽑는다", file=sys.stderr)

    env = dict(os.environ)
    extra = os.pathsep.join(str(p) for p in (OPENCV_BIN, ORT_BIN) if p.exists())
    if extra:
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")

    seqs = sorted(
        p for pat in ("rgbd_dataset_*", "kitti_*")
        for p in (ROOT / "data").glob(pat)
        if (p / "rgb.txt").exists() and (p / "depth.txt").exists())
    if args.only:
        want = set(args.only)
        seqs = [p for p in seqs
                if p.name in want or p.name.replace("rgbd_dataset_", "") in want]
    if not seqs:
        raise SystemExit("시퀀스를 찾지 못했다")

    t_all = time.time()
    ok = fail = skip = 0
    for i, seq in enumerate(seqs, 1):
        name = seq.name.replace("rgbd_dataset_", "")
        out = out_dir / f"scene_{name}.tsv"
        if out.exists() and not args.force:
            print(f"[{i}/{len(seqs)}] {name}: 이미 있다 - 건너뜀", flush=True)
            skip += 1
            continue

        cmd = [str(exe), str(seq), str(out), "--stride", str(args.stride)]
        if args.frames:
            cmd += ["--max-frames", str(args.frames)]
        if use_yolo:
            cmd += ["--yolo", str(model), "--conf", str(args.conf)]

        print(f"[{i}/{len(seqs)}] {name}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        dt = time.time() - t0
        last = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
        if r.returncode == 0:
            print(f"    {last[-1] if last else ''}  ({dt:.0f}초)", flush=True)
            ok += 1
        else:
            print(f"    실패 rc={r.returncode}: {(r.stderr or '')[-200:]}",
                  file=sys.stderr, flush=True)
            fail += 1

    print(f"\n전체 {(time.time()-t_all)/60:.1f}분 - 성공 {ok} / 건너뜀 {skip} / 실패 {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
