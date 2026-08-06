#!/usr/bin/env python3
"""데이터셋이 인덱스가 약속한 파일을 실제로 갖고 있는지 검사한다.

왜 필요한가
-----------
`rgb.txt` 는 프레임 목록이고 도구들은 그 목록을 따라간다. 파일이 없으면
`cv::imread` 가 빈 Mat 를 돌려주고, 도구는 그 프레임을 **조용히 건너뛴다.**
결과 궤적은 짧아지지만 ATE 는 여전히 계산되고 표에 숫자로 실린다 - 즉
"1419 프레임 시퀀스" 라고 적힌 줄이 실제로는 165 프레임의 결과일 수 있다.

실제로 뷰어에서 fr1_teddy 의 영상이 한 장도 안 뜨는 것을 보고 발견했다.
06-results.md 가 반복해 적어 둔 실패 양식이다: 실패가 조용하면 초록은
아무 것도 뜻하지 않는다.

사용:  python tools/check_datasets.py [--fix-index]
       --fix-index 는 없는 파일 줄을 rgb.txt/depth.txt 에서 지우는 대신
       <name>.missing 목록만 남긴다. 인덱스는 원본이므로 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_index(p: Path) -> list[tuple[float, str]]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            out.append((float(parts[0]), parts[1]))
    return out


def check(seq: Path, write_missing: bool) -> dict:
    row = {"name": seq.name}
    for kind in ("rgb", "depth"):
        idx = read_index(seq / f"{kind}.txt")
        missing = [rel for _, rel in idx if not (seq / rel).exists()]
        row[kind] = (len(idx), len(missing))
        if missing and write_missing:
            (seq / f"{kind}.missing").write_text("\n".join(missing) + "\n",
                                                 encoding="utf-8")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-index", action="store_true",
                    help="없는 파일 목록을 <kind>.missing 으로 남긴다")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    seqs = sorted(p for p in args.data.iterdir()
                  if p.is_dir() and (p / "rgb.txt").exists())
    if not seqs:
        raise SystemExit(f"{args.data} 에 시퀀스가 없다")

    bad = 0
    print(f"{'sequence':<48} {'rgb idx':>8} {'missing':>8} "
          f"{'depth idx':>10} {'missing':>8}  usable")
    for seq in seqs:
        r = check(seq, args.fix_index)
        ri, rm = r["rgb"]
        di, dm = r["depth"]
        usable = min(ri - rm, di - dm)
        frac = usable / max(1, min(ri, di))
        flag = ""
        if frac < 0.999:
            flag = "  <-- 인덱스가 약속한 프레임의 일부만 존재한다"
            bad += 1
        print(f"{r['name']:<48} {ri:>8} {rm:>8} {di:>10} {dm:>8}  "
              f"{usable:>5} ({100*frac:5.1f}%){flag}")

    if bad:
        print(f"\n{bad} 개 시퀀스가 불완전하다.", file=sys.stderr)
        print("이 시퀀스로 낸 수치는 '전체 시퀀스' 결과가 아니다 - "
              "다시 받거나, 결과에 실제 프레임 수를 함께 적어야 한다.",
              file=sys.stderr)
        return 1
    print("\n모든 시퀀스가 인덱스와 일치한다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
