"""Tier 0 - 측광 직접정렬 (ECDA).

C++ src/localization/DirectAligner.cpp 의 참조 구현. C++ 쪽은 툴체인 부재로
한 번도 실행되지 않았으므로, 여기가 그 알고리즘의 유일한 검증 수단이다.
"""

from .ecda import (
    EcdaConfig,
    InformationModel,
    EcdaResult,
    align,
    build_pyramid,
    scale_intrinsics,
    select_points,
)

__all__ = [
    "EcdaConfig", "EcdaResult", "InformationModel", "align",
    "build_pyramid", "scale_intrinsics", "select_points",
]
