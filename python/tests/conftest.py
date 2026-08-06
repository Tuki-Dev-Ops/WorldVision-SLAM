"""차등 테스트 오라클(`wme._core`)을 임포트 가능한 상태로 만든다.

Windows 의 Python 3.8+ 는 확장 모듈의 의존 DLL 을 PATH 에서 찾지 않는다.
`_core` 는 OpenCV world DLL 에 의존하므로 그 디렉터리를 os.add_dll_directory 로
명시하지 않으면 임포트가 ImportError 로 실패한다. wme/__init__ 은 그 예외를
삼켜 HAS_NATIVE=False 로 떨어지고, 그 결과 test_differential.py 전체가 조용히
skip 된다 - 실패할 수 없는 테스트가 된다.

경로는 WME_DLL_DIRS(os.pathsep 구분)로 주고, 없으면 아래 후보를 훑는다.
이 파일은 wme 를 임포트하기 전에 실행되어야 하므로 여기서 wme 를 import 하지
않는다.
"""

from __future__ import annotations

import os
import sys

_CANDIDATES = (
    r"C:\opencv-dl\opencv\build\x64\vc16\bin",
    r"C:\opencv\build\x64\vc16\bin",
)


def _register_dll_dirs() -> None:
    if not hasattr(os, "add_dll_directory"):     # 비 Windows
        return

    explicit = os.environ.get("WME_DLL_DIRS", "")
    paths = [p for p in explicit.split(os.pathsep) if p]
    if not paths:
        paths = [p for p in _CANDIDATES if os.path.isdir(p)]

    for p in paths:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except OSError:                       # 이미 등록됐거나 접근 불가
                pass


if sys.platform == "win32":
    _register_dll_dirs()
