#!/usr/bin/env python3
"""KITTI odometry (grayscale) 내려받기 + 압축 해제.

TUM 은 실내 RGB-D 로, 깊이가 센서에서 직접 온다. KITTI 는 실외 스테레오라
깊이를 우리가 만들어야 하고, 궤적 규모가 100배 크다. 그래서 여기서 처음으로
- 깊이 프런트엔드(StereoSGBM)가 실제로 필요한 경로가 되고,
- 스케일 드리프트가 관측 가능한 규모가 된다.

  data_odometry_gray.zip   22.1 GB  (00~21 시퀀스, image_0/image_1)
  data_odometry_calib.zip   0.6 MB  (P0..P3, Tr)
  data_odometry_poses.zip   1.2 MB  (00~10 정답 궤적)

이어받기를 한다. 22 GB 를 처음부터 다시 받는 일이 없어야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
FILES = [
    ("data_odometry_calib.zip", 0.6),
    ("data_odometry_poses.zip", 1.2),
    ("data_odometry_gray.zip", 22093.3),
]

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "kitti"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        v = r.headers.get("Content-Length")
    return int(v) if v else None


def download(name: str, out: Path, log) -> bool:
    url = f"{BASE}/{name}"
    total = remote_size(url)
    have = out.stat().st_size if out.exists() else 0
    if total is not None and have == total:
        log(f"{name}: 이미 완료 ({human(have)})")
        return True
    if have > (total or 0):
        log(f"{name}: 로컬이 더 큼 ({human(have)} > {human(total or 0)}) - 다시 받는다")
        out.unlink()
        have = 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        log(f"{name}: {human(have)} 부터 이어받기")
    else:
        log(f"{name}: 새로 받기 ({human(total or 0)})")

    mode = "ab" if have else "wb"
    t0 = time.time()
    last = t0
    with urllib.request.urlopen(req, timeout=120) as r, out.open(mode) as f:
        # 206 이 아니면 서버가 Range 를 무시한 것이다. 이어붙이면 파일이 깨진다.
        if have and r.status != 206:
            f.close()
            out.unlink()
            log(f"{name}: 서버가 Range 를 거부(status={r.status}) - 처음부터")
            return download(name, out, log)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            now = time.time()
            if now - last > 20.0:
                rate = have / max(1e-6, now - t0)
                pct = 100.0 * have / total if total else 0.0
                eta = (total - have) / rate if total and rate > 0 else 0.0
                log(f"{name}: {human(have)} / {human(total or 0)} "
                    f"({pct:.1f}%) {human(rate)}/s ETA {eta/60:.0f}분")
                last = now
    ok = total is None or have == total
    log(f"{name}: {'완료' if ok else '불완전'} {human(have)} "
        f"({(time.time()-t0)/60:.1f}분)")
    return ok


def extract(zip_path: Path, dest: Path, log) -> None:
    """이미 풀린 항목은 건너뛴다. 22 GB 를 두 번 푸는 일이 없어야 한다."""
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        todo = []
        for m in members:
            if m.is_dir():
                continue
            p = dest / m.filename
            if p.exists() and p.stat().st_size == m.file_size:
                continue
            todo.append(m)
        log(f"{zip_path.name}: {len(todo)} / {len(members)} 항목 해제")
        for i, m in enumerate(todo):
            z.extract(m, dest)
            if i % 5000 == 0 and i:
                log(f"  {i}/{len(todo)}")
    log(f"{zip_path.name}: 해제 완료")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--small-only", action="store_true",
                    help="calib/poses 만 (22 GB 없이 배선 검증)")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    logf = args.dest / "fetch.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    files = FILES[:2] if args.small_only else FILES
    ok = True
    for name, _mb in files:
        zp = args.dest / name
        if not download(name, zp, log):
            log(f"{name}: 실패")
            ok = False
            continue
        if not args.no_extract:
            extract(zp, args.dest, log)
    log("전체 완료" if ok else "일부 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
