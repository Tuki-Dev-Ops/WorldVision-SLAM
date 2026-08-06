"""Geometry - 구조 프리미티브.

  planes.py  깊이맵에서 평면 추출. 외관과 무관하므로 측광이 죽는 곳에서 산다.
  spa.py     Tier 2 정합. 정보행렬이 본질적으로 랭크 부족일 수 있고,
             그 사실을 숨기지 않고 보고하는 것이 계약이다.
"""

from .planes import Plane, PlaneConfig, dominant_gravity, estimate_normals, extract_planes
from .spa import PlaneMatch, SpaConfig, SpaResult, align, match_planes, unobservable_directions

__all__ = [
    "Plane", "PlaneConfig", "extract_planes", "estimate_normals", "dominant_gravity",
    "SpaConfig", "SpaResult", "PlaneMatch", "align", "match_planes",
    "unobservable_directions",
]
