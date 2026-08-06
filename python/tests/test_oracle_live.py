"""오라클 자체 점검.

차등 테스트는 이 저장소의 1차 정확성 기구다. 그 기구가 켜져 있는지를 확인하는
테스트가 없으면, 기구가 꺼진 것과 두 구현이 일치하는 것을 구분할 수 없다.

test_differential.py 는 `_core` 가 없으면 모듈 전체를 skip 한다. skip 은 통과로
보이므로, 확장이 오랫동안 빌드되지 않아도 `pytest -q` 는 계속 초록색이었다.
실제로 그 사이에 세 개의 알고리즘이 갈라졌고 41개 차등 테스트 중 0개가 실행됐다.

이 파일은 그 상태를 *보이게* 만든다. xfail 이므로 요약에 항상 한 줄 남고,
확장이 빌드되면 XPASS 로 바뀌어 제거를 요구한다.
"""

import pytest

from wme import HAS_NATIVE

# 차등 테스트가 반드시 덮어야 하는 C++ 표면.
# 여기 이름이 하나라도 빠지면 그 알고리즘은 오라클 없이 굴러가는 것이다.
REQUIRED_SURFACE = (
    "SE3", "SO3", "kabsch", "solve_assignment",
    "ConstellationIndex", "ConstellationConfig", "ConstellationMatch",
    "ConfidenceEngine", "ConfidenceConfig", "WorldToken",
    "derive_adaptation", "EnvironmentEvidence",
    "DirectAligner", "DirectAlignerConfig",
    "box_iou", "non_max_suppression",
)


@pytest.mark.xfail(not HAS_NATIVE, strict=False,
                   reason="wme._core 미빌드 - 차등 테스트 41개가 전부 skip 중이다. "
                          "cmake -DWME_BUILD_PYTHON=ON 후 재실행")
def test_differential_oracle_is_live():
    """오라클이 실제로 로드되는가. 이것이 실패하면 차등 테스트는 아무것도 재지 않는다."""
    assert HAS_NATIVE, "C++ 오라클 없음 - test_differential.py 전체가 무의미하다"


@pytest.mark.xfail(not HAS_NATIVE, strict=False, reason="wme._core 미빌드")
def test_oracle_exposes_every_differentially_tested_subsystem():
    """바인딩이 조용히 좁아지면 차등 테스트도 조용히 좁아진다."""
    from wme import core

    missing = [n for n in REQUIRED_SURFACE if not hasattr(core, n)]
    assert not missing, f"바인딩에서 빠진 심볼: {missing}"
