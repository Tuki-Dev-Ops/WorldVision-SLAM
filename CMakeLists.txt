cmake_minimum_required(VERSION 3.24)

project(WorldModelEngine
    VERSION 0.1.0
    DESCRIPTION "WME - World Model Engine"
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# 실행 파일 옆에 필요한 런타임 DLL 을 둔다. 윈도우 DLL 탐색 순서에서
# 앱 디렉터리가 System32 보다 앞서므로, 이름이 겹치는 시스템 DLL 을 피하는
# 유일하게 확실한 방법이다.
function(wme_stage_runtime_dlls target)
    if(WIN32 AND ORT_RUNTIME_DLL)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ORT_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${target}>"
            COMMENT "onnxruntime.dll 배치 (System32 판 회피)")
    endif()
endfunction()

option(WME_BUILD_TESTS      "Build unit tests"            ON)
option(WME_BUILD_BENCHMARKS "Build benchmarks"            ON)
option(WME_BUILD_TOOLS      "Build CLI tools"             ON)
option(WME_BUILD_PYTHON     "Build pybind11 bindings"     OFF)
option(WME_WITH_CUDA        "Enable CUDA acceleration"    OFF)
option(WME_WITH_TENSORRT    "Enable TensorRT YOLO backend" OFF)
option(WME_WITH_ONNXRUNTIME "Enable ONNX Runtime backend" ON)
option(WME_WITH_OPEN3D      "Enable Open3D mesh backend"  OFF)
option(WME_WITH_PCL         "Enable PCL point cloud ops"  OFF)
option(WME_WITH_VIZ         "Build ImGui debug visualizer" OFF)
option(WME_ASAN             "Address + UB sanitizer"      OFF)
option(WME_NATIVE_ARCH      "Compile for host CPU (-march=native)" OFF)

include(WmeCompilerFlags)
include(WmeDependencies)

# ---------------------------------------------------------------------------
# Interface target carrying project-wide settings
# ---------------------------------------------------------------------------
add_library(wme_settings INTERFACE)
add_library(wme::settings ALIAS wme_settings)
target_include_directories(wme_settings INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(wme_settings INTERFACE cxx_std_20)
target_compile_options(wme_settings INTERFACE ${WME_WARNING_FLAGS})
target_link_libraries(wme_settings INTERFACE Eigen3::Eigen spdlog::spdlog)

wme_apply_sanitizers(wme_settings)

add_subdirectory(src)

if(WME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(WME_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(WME_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(WME_BUILD_PYTHON)
    add_subdirectory(python/bindings)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "  World Model Engine ${PROJECT_VERSION}")
message(STATUS "  ---------------------------------------")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA            : ${WME_WITH_CUDA}")
message(STATUS "  TensorRT        : ${WME_WITH_TENSORRT}")
message(STATUS "  ONNX Runtime    : ${WME_WITH_ONNXRUNTIME}")
message(STATUS "  Open3D / PCL    : ${WME_WITH_OPEN3D} / ${WME_WITH_PCL}")
message(STATUS "  Tests / Bench   : ${WME_BUILD_TESTS} / ${WME_BUILD_BENCHMARKS}")
message(STATUS "  Python bindings : ${WME_BUILD_PYTHON}")
message(STATUS "")
