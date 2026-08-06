"""WME - World Model Engine, Python 계층.

세 가지 역할을 한다.

1. **차등 테스트 오라클** (`wme.reference`)
   C++ 엔진의 핵심 알고리즘을 numpy 로 다시 구현한 것. 같은 테스트 벡터에
   대해 두 구현이 다른 답을 내면 둘 중 하나가 틀린 것이다. SLAM 처럼 오차가
   조용히 누적되는 시스템에서는 이 교차검증이 단위 테스트보다 강력하다.

2. **평가 하네스** (`wme.eval`)
   ATE / RPE, TUM-RGBD 로더. 논문 수치와 비교 가능한 표준 정의를 따른다.

3. **YOLO 브리지** (`wme.yolo`)
   C++ TensorRT 백엔드가 준비되기 전까지 검출을 공급한다. 후처리는 순수
   numpy 라 모델 없이도 검증된다.

C++ 엔진 바인딩(`wme._core`)은 빌드했을 때만 존재한다:

    from wme import core          # 없으면 ImportError 대신 None
"""

from __future__ import annotations

import os
import warnings

__version__ = "0.1.0"

# 확장이 없을 때 조용히 넘어가면 차등 테스트 전체가 skip 되고, skip 은 초록으로
# 보인다. 실제로 그 상태로 오래 굴렀다 (docs/06-results.md 19장). 그래서:
#   1. 실패 사유를 CORE_IMPORT_ERROR 에 남긴다 - 사후에 물어볼 수 있어야 한다
#   2. 경고를 띄운다 - 아무도 안 물어봐도 한 줄은 나온다
#   3. WME_REQUIRE_NATIVE=1 이면 아예 터뜨린다 - CI 는 이걸 켠다
# 조용한 실패를 없애는 게 목적이므로 except 를 좁히지 않는다. 등록 순서
# 오류는 ImportError 가 아니라 TypeError 로도 나온 적이 있다.
CORE_IMPORT_ERROR: BaseException | None = None

try:                                    # pybind11 확장은 선택적이다
    from . import _core as core         # type: ignore[attr-defined]
    HAS_NATIVE = True
except Exception as exc:                # pragma: no cover
    core = None                         # type: ignore[assignment]
    HAS_NATIVE = False
    CORE_IMPORT_ERROR = exc
    if os.environ.get("WME_REQUIRE_NATIVE"):
        raise
    warnings.warn(
        f"wme._core 를 임포트하지 못했다 - 차등 테스트가 전부 skip 된다: "
        f"{type(exc).__name__}: {exc}",
        RuntimeWarning, stacklevel=2)

__all__ = ["core", "HAS_NATIVE", "CORE_IMPORT_ERROR", "__version__"]
