"""Planner - 신념층을 실제로 소비하는 층.

  risk.py    위험도. 별도 센서가 아니라 위치/공분산/예측/어포던스/커버리지의 파생량.
  search.py  의미 기반 객체 탐색. 어디를 볼지 세계 모델이 정한다.

두 모듈 모두 불확실성을 소비한다. 점추정만 쓰면 계획기는 자신이 무엇을
모르는지 모른 채 움직인다.
"""

from .risk import Coverage, RiskAssessment, RiskConfig, RiskEstimator
from .search import ObjectSearch, SearchCandidate, SearchConfig

__all__ = [
    "RiskEstimator", "RiskConfig", "RiskAssessment", "Coverage",
    "ObjectSearch", "SearchConfig", "SearchCandidate",
]
