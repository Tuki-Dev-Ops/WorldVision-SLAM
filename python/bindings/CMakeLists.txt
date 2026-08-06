# pybind11 확장 모듈. python/wme/_core*.pyd(.so) 로 떨어진다.

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 2.11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    # pip 로 설치된 pybind11 의 cmake 디렉터리를 찾는다
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        RESULT_VARIABLE PYBIND11_LOOKUP
        ERROR_QUIET)
    if(NOT PYBIND11_LOOKUP EQUAL 0)
        message(FATAL_ERROR "pybind11 을 찾지 못함. `pip install pybind11` 후 재시도")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 2.11 REQUIRED CONFIG)
endif()

pybind11_add_module(wme_python MODULE wme_module.cpp)

set_target_properties(wme_python PROPERTIES
    OUTPUT_NAME "_core"
    FOLDER "wme/python"
    # 소스 트리에 바로 떨궈 `cd python && pytest` 가 바로 되게 한다
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../wme"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../wme")

foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(wme_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/../wme"
        RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/../wme")
endforeach()

target_link_libraries(wme_python PRIVATE
    wme::wme_localization
    wme::wme_token
    wme::wme_perception
    wme::wme_confidence
    # 융합 서브시스템. 06-results.md 26 절이 차분 커버리지 0 으로 적어 둔 곳이고,
    # 18/21/23.4 의 결과가 전부 이 위에 서 있다.
    wme::wme_fusion
    wme::wme_geometry
    wme::wme_core)

# 바인딩은 얇은 변환층이라 프로젝트 경고 정책을 그대로 적용하기 어렵다
if(NOT MSVC)
    target_compile_options(wme_python PRIVATE -Wno-old-style-cast -Wno-shadow)
endif()
