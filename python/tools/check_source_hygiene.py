#!/usr/bin/env python3
"""소스에 플랫폼별로만 터지는 바이트가 없는지 검사한다.

왜 이 검사가 존재하는가
-----------------------
`tests/test_spa.cpp` 의 문자열 리터럴 **안쪽** 에 생 CR 바이트가 7 개 들어간
적이 있다. C++ 표준은 CR 을 줄바꿈으로 인정하고, GCC/Clang 은 실제로 그렇게
읽는다. MSVC 는 아니다. 그래서 그 파일은 **Windows 에서만 컴파일되는 소스** 가
되었고, 로컬 236 개 테스트가 전부 초록인 채로 리눅스 CI 에서 처음 터졌다.

06-results.md 가 반복해 적어 둔 실패 양식과 같은 것이다: 실패가 조용하면
초록은 아무 것도 뜻하지 않는다. 그러니 조용할 수 없게 만든다.

검사 항목
  1. 문자열 리터럴 안팎을 가리지 않고 **줄 끝이 아닌 CR**
  2. NUL 바이트
  3. UTF-8 로 디코드되지 않는 바이트 (MSVC 의 cp949 함정과 같은 자리)

BOM 은 검사하지 않는다. GCC 4.4+ 는 선행 UTF-8 BOM 을 건너뛰고 이 저장소는
MSVC 에 /utf-8 을 준다 - 즉 두 툴체인 모두에서 무해하다. 재 보지 않은 것을
결함으로 세우지 않는다.

사용:  python tools/check_source_hygiene.py [--fix]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".cpp", ".hpp", ".h", ".c", ".cc", ".py", ".cmake", ".txt", ".md", ".yml", ".yaml"}
SKIP_DIRS = {"build", "data", ".git", "__pycache__", "third_party", "results", "models"}


def iter_sources():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def check(path: Path) -> list[str]:
    data = path.read_bytes()
    out: list[str] = []

    if b"\x00" in data:
        out.append("NUL 바이트")

    # 줄 끝이 아닌 CR
    for i, b in enumerate(data):
        if b == 0x0D and not (i + 1 < len(data) and data[i + 1] == 0x0A):
            line = data[:i].count(b"\n") + 1
            out.append(f"{line} 행: 줄 끝이 아닌 CR (GCC 는 줄바꿈으로 읽고 MSVC 는 아니다)")

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        out.append(f"UTF-8 아님: {e}")

    return out


def fix(path: Path) -> int:
    data = path.read_bytes()
    out, n = bytearray(), 0
    for i, b in enumerate(data):
        if b == 0x0D and not (i + 1 < len(data) and data[i + 1] == 0x0A):
            n += 1
            continue
        out.append(b)
    if n:
        path.write_bytes(bytes(out))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="줄 끝 아닌 CR 을 제거한다")
    args = ap.parse_args()

    bad = 0
    for p in iter_sources():
        problems = check(p)
        if not problems:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if args.fix:
            n = fix(p)
            problems = check(p)
            print(f"{rel}: CR {n} 개 제거"
                  + (f", 남은 문제 {len(problems)}" if problems else ""))
        if problems:
            bad += 1
            print(f"{rel}:")
            for m in problems[:5]:
                print(f"    {m}")
            if len(problems) > 5:
                print(f"    ... 외 {len(problems) - 5} 건")

    if bad:
        print(f"\n{bad} 개 파일에 문제가 있다", file=sys.stderr)
        return 1
    print("소스 위생 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
