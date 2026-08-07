include(GoogleTest)

# 테스트 시간 상한.
#
# 180 은 최적화된 빌드에서 잰 값이다. 새니타이저 + Debug 조합에서는 같은
# 코드가 10 배 가까이 느려진다 (실측: ECDA 정렬 12 초 -> 172 초). 같은 숫자를
# 그대로 쓰면 재는 것이 알고리즘이 아니라 계측 오버헤드가 된다.
#
# **명령줄에서는 못 고친다.** `ctest --timeout` 은 TIMEOUT 속성이 **없는**
# 테스트의 기본값일 뿐이라, 아래에서 속성을 붙이는 순간 무시된다.
# sanitizers.yml 의 `--timeout 900` 은 그래서 아무 일도 하지 않았고,
# DirectAligner.PyramidDepthExtendsConvergenceRadius 는 900 이 아니라
# 180.13 초에 죽었다 - 죽은 손잡이를 돌리고 있었던 것이다.
if(WME_ASAN)
    set(WME_TEST_TIMEOUT 900)
else()
    set(WME_TEST_TIMEOUT 180)
endif()
message(STATUS "테스트 시간 상한: ${WME_TEST_TIMEOUT} 초 (WME_ASAN=${WME_ASAN})")

function(wme_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE ${ARG_DEPS} GTest::gtest GTest::gtest_main)
    set_target_properties(${name} PROPERTIES FOLDER "wme/tests")
    wme_stage_runtime_dlls(${name})
    # 모델/데이터를 저장소 루트 기준 상대경로로 찾는 테스트가 있으므로
    # 작업 디렉터리를 고정한다. 빌드 위치에 따라 결과가 달라지면 안 된다.
    gtest_discover_tests(${name}
        DISCOVERY_MODE PRE_TEST
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        PROPERTIES TIMEOUT ${WME_TEST_TIMEOUT} WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endfunction()

# OpenCV 없이도 도는 테스트. SE3 규약과 헝가리안이 여기서 검증된다 -
# 둘 다 Python 참조 구현과 차등 검증된 알고리즘이라, 컴파일만 되면
# 곧바로 의미 있는 확인이 된다.
wme_add_test(test_math
    SOURCES test_se3.cpp test_threadpool.cpp test_assignment.cpp
    DEPS wme::wme_math)

if(NOT WME_HAS_OPENCV)
    return()
endif()

wme_add_test(test_perception
    SOURCES test_perception.cpp
    DEPS wme::wme_perception)

wme_add_test(test_yolo
    SOURCES test_yolo.cpp
    DEPS wme::wme_perception)

wme_add_test(test_stereo_depth
    SOURCES test_stereo_depth.cpp
    DEPS wme::wme_perception)

wme_add_test(test_confidence
    SOURCES test_confidence.cpp
    DEPS wme::wme_confidence)

wme_add_test(test_token
    SOURCES test_constellation.cpp test_tokenstore.cpp
    DEPS wme::wme_token)

wme_add_test(test_localization
    SOURCES test_direct_aligner.cpp
    DEPS wme::wme_localization)

wme_add_test(test_spa
    SOURCES test_spa.cpp
    DEPS wme::wme_geometry)

# 엔드투엔드 배선. 서브시스템 테스트가 전부 통과하는 상태에서도 단계 사이 연결이
# 끊겨 있을 수 있다 - 11.3, 14.2, 16.5 가 전부 그런 사고였다.
wme_add_test(test_pipeline
    SOURCES test_pipeline.cpp
    DEPS wme::wme_localization wme::wme_token wme::wme_perception
         wme::wme_confidence wme::wme_fusion)

# ---------------------------------------------------------------------------
# 결정성 / 할당 하네스
# ---------------------------------------------------------------------------
# test_determinism 은 "두 번 돌린 결과의 비트가 같다"를 주장한다.
# 그 주장은 하네스가 실패할 수 있다는 증거가 있어야만 의미가 있으므로,
# 같은 케이스 코드를 순서 의존성이 주입된 소스 사본으로 한 번 더 빌드한다
# (test_determinism_mutant). 대조군이 "다르다"를 확인하지 못하면 빌드가
# 아니라 테스트가 실패한다.

wme_add_test(test_determinism
    SOURCES test_determinism.cpp
    DEPS wme::wme_localization)

wme_add_test(test_allocation
    SOURCES test_allocation.cpp
    DEPS wme::wme_localization)

# 원본 소스를 읽어 지정한 앵커 한 곳을 순서 의존 코드로 바꾼 사본을 만든다.
# 앵커가 사라지면 변이가 적용되지 않고, 그러면 대조군이 "차이를 못 찾았다"로
# 실패한다. 즉 판별력 상실은 조용히 지나가지 않는다. 구성 단계에서 죽이지
# 않는 이유는 원본 소스를 고치는 쪽의 빌드까지 막지 않기 위해서다.
function(wme_make_mutant out_var relative_source anchor replacement)
    set(_src "${CMAKE_SOURCE_DIR}/src/${relative_source}")
    file(READ "${_src}" _text)
    string(FIND "${_text}" "${anchor}" _pos)
    if(_pos EQUAL -1)
        message(WARNING
            "결정성 대조군의 앵커를 ${relative_source} 에서 찾지 못했다. "
            "변이가 적용되지 않으므로 test_determinism_mutant 가 실패한다. "
            "소스가 바뀌었다면 tests/CMakeLists.txt 의 변이 정의를 갱신할 것. "
            "앵커: ${anchor}")
    endif()
    string(REPLACE "${anchor}" "${replacement}" _text "${_text}")
    get_filename_component(_name "${relative_source}" NAME)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/mutant/mutant_${_name}")
    file(WRITE "${_out}" "${_text}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_src}")
    set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

# M1: 축소 블록 수를 워커 수에 연동한다. 고정 블록(reduction_blocks=16) 설계를
#     되돌린 것과 같아서, 부동소수 합산 묶음이 워커 수에 따라 달라진다.
wme_make_mutant(WME_MUTANT_ALIGNER localization/DirectAligner.cpp
    "const std::size_t nblocks = std::min(cfg_.reduction_blocks, points_.size());"
    "const std::size_t nblocks = std::min<std::size_t>(pool_ != nullptr ? pool_->workerCount() + 1u : 1u, points_.size());")

# M2: NMS 의 동점 타이브레이크를 구현 정의 순서로 되돌린다.
wme_make_mutant(WME_MUTANT_DETECTION perception/Detection.cpp
    "std::stable_sort(order.begin(), order.end()"
    "std::sort(order.begin(), order.end()")

# M3: allTokens() 를 ID 가 아니라 shared_ptr 주소로 정렬한다.
#     해시 순회 순서가 결과로 새어 나가던 상태와 같다.
wme_make_mutant(WME_MUTANT_TOKENSTORE token/TokenStore.cpp
    "[](const auto& a, const auto& b) { return a->id < b->id; });"
    "[](const auto& a, const auto& b) { return a.get() < b.get(); });")

# 변이 사본이 원본 오브젝트보다 먼저 링크되도록 wme_localization 대신
# 그 의존성(wme_token)만 링크한다. 원본 DirectAligner/TokenStore/Detection
# 오브젝트는 심벌이 이미 해결되므로 라이브러리에서 끌려오지 않는다.
wme_add_test(test_determinism_mutant
    SOURCES
        test_determinism_mutant.cpp
        ${WME_MUTANT_ALIGNER}
        ${WME_MUTANT_DETECTION}
        ${WME_MUTANT_TOKENSTORE}
    DEPS wme::wme_token)

# ---------------------------------------------------------------------------
# 3-tier 융합
# ---------------------------------------------------------------------------
# SE(3) 좌측 자코비안은 급수로 구현했고 유한차분으로 건다. 닫힌 형태를 손으로
# 적다가 부호를 뒤집은 사고가 이미 한 번 있었다 (docs/06-results.md 7.1).
wme_add_test(test_fusion
    SOURCES test_fusion.cpp
    DEPS wme::wme_fusion)
