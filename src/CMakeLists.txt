# wme_math 는 OpenCV 를 쓰지 않는다.
# SE3/할당/스레드풀은 Eigen 만 있으면 되고, 핵심 수학이 영상 라이브러리를
# 끌고 오는 것은 그 자체로 설계 결함이다. 분리해 두면 OpenCV 가 없는
# 환경에서도 이 부분은 빌드하고 검증할 수 있다.
wme_add_library(wme_math
    SOURCES
        core/Result.cpp
        core/ThreadPool.cpp
        core/Assignment.cpp)

if(NOT WME_HAS_OPENCV)
    message(STATUS "  -> OpenCV 의존 타깃은 건너뛴다")
    return()
endif()

wme_add_library(wme_core
    SOURCES
        core/Frame.cpp
    DEPS
        wme::wme_math
        ${OpenCV_LIBS})
target_include_directories(wme_core PUBLIC ${OpenCV_INCLUDE_DIRS})

set(WME_PERCEPTION_SOURCES
    perception/ImageQualityEngine.cpp
    perception/EnvironmentAnalyzer.cpp
    perception/StereoDepth.cpp
    perception/Detection.cpp
    perception/YoloDecode.cpp
    perception/YoloRuntimeCv.cpp)

if(WME_HAS_ORT)
    list(APPEND WME_PERCEPTION_SOURCES perception/YoloRuntimeOrt.cpp)
endif()

wme_add_library(wme_perception
    SOURCES ${WME_PERCEPTION_SOURCES}
    DEPS
        wme::wme_core
        opencv_dnn)

if(WME_HAS_ORT)
    target_link_libraries(wme_perception PUBLIC wme_onnxruntime)
    target_compile_definitions(wme_perception PUBLIC WME_HAS_ORT=1)
endif()

wme_add_library(wme_confidence
    SOURCES
        confidence/ConfidenceEngine.cpp
    DEPS
        wme::wme_perception)

wme_add_library(wme_token
    SOURCES
        token/ConstellationIndex.cpp
        token/TokenStore.cpp
    DEPS
        wme::wme_confidence)

wme_add_library(wme_localization
    SOURCES
        localization/DirectAligner.cpp
    DEPS
        wme::wme_token)

# 통합 파사드
add_library(wme INTERFACE)
add_library(wme::wme ALIAS wme)
target_link_libraries(wme INTERFACE
    wme::wme_math
    wme::wme_core
    wme::wme_perception
    wme::wme_confidence
    wme::wme_token
    wme::wme_localization)

# Tier 2 (SPA). 깊이만 쓰므로 perception/token 계열에 의존하지 않는다 -
# 외관과 무관하다는 것이 이 tier 의 존재 이유라 의존성에서도 그래야 한다.
wme_add_library(wme_geometry
    SOURCES
        geometry/PlaneExtractor.cpp
        geometry/StructuralAligner.cpp
    DEPS
        wme::wme_core)

target_link_libraries(wme INTERFACE wme::wme_geometry)

# ---------------------------------------------------------------------------
# 3-tier 융합 (docs/02-correspondence-problem.md 4장)
# ---------------------------------------------------------------------------
# 세 tier 의 정보행렬을 하나의 상대포즈로 합친다. localization/geometry/token
# 어디에도 속하지 않는다 - 그 셋을 소비하는 쪽이므로 위에 둔다.
# perception 에 의존하는 이유는 alpha_k(E) 가 EnvironmentState 에서 오기 때문이다.
wme_add_library(wme_fusion
    SOURCES
        fusion/PoseFusion.cpp
        fusion/TierInformation.cpp
    DEPS
        wme::wme_perception)

target_link_libraries(wme INTERFACE wme::wme_fusion)
