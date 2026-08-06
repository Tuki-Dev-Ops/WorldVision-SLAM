#!/usr/bin/env python3
"""벤치마크에 쓰는 TUM 시퀀스 전체를 **완전히** 받는다.

왜
--
`tum_fetch.py` 는 기본이 9 초 창이다. 디스크를 아끼려는 설계였지만, tar 안의
`rgb.txt` 는 시퀀스 전체 목록이라 인덱스의 대부분이 없는 파일을 가리키게
된다. 도구들은 그 프레임을 조용히 건너뛰므로 "1419 프레임 시퀀스" 로 적힌
결과가 실제로는 165 프레임짜리였다 (실측: 16 개 중 15 개가 불완전, 최소 6.3 %).

이 스크립트는 전체를 받고, `tum_fetch.py` 가 인덱스를 실제 파일에 맞춰
잘라 주므로 이후로는 그 불일치가 생기지 않는다.

용량: 16 시퀀스 합계 약 25 GB (압축 해제 후).
이미 완전한 시퀀스는 건너뛴다.

사용: python tools/tum_fetch_all.py [--data ../data] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SEQUENCES = [
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg1_plant",
    "rgbd_dataset_freiburg1_teddy",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg2_desk_with_person",
    "rgbd_dataset_freiburg3_structure_texture_far",
    "rgbd_dataset_freiburg3_structure_notexture_far",
    "rgbd_dataset_freiburg3_nostructure_texture_far",
    "rgbd_dataset_freiburg3_nostructure_notexture_far",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
]


def complete(seq: Path) -> bool:
    """인덱스가 약속한 파일이 전부 있는가."""
    for kind in ("rgb", "depth"):
        idx = seq / f"{kind}.txt"
        if not idx.exists():
            return False
        for line in idx.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2 and not (seq / p[1]).exists():
                return False
    # full.txt 가 남아 있다면 한 번 잘린 적이 있다는 뜻이다.
    return not (seq / "rgb.full.txt").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    names = args.only or SEQUENCES
    t_all = time.time()
    ok = fail = skip = 0

    for i, name in enumerate(names, 1):
        seq = data / name
        if not args.force and seq.exists() and complete(seq):
            print(f"[{i}/{len(names)}] {name}: 이미 완전하다 - 건너뜀", flush=True)
            skip += 1
            continue
        print(f"\n[{i}/{len(names)}] {name}: 전체 내려받기", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(HERE / "tum_fetch.py"), name,
             "--data", str(data), "--all"],
            cwd=str(HERE))
        dt = (time.time() - t0) / 60.0
        if r.returncode == 0:
            print(f"    완료 ({dt:.1f}분)", flush=True)
            ok += 1
        else:
            print(f"    실패 (returncode {r.returncode})", file=sys.stderr, flush=True)
            fail += 1

    print(f"\n전체 {(time.time()-t_all)/60:.1f}분 - 성공 {ok} / 건너뜀 {skip} / 실패 {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
