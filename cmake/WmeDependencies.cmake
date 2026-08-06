# 외부 의존성 탐색. 없으면 FetchContent 로 대체.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- 필수 ------------------------------------------------------------------
# wme_math 이 std::thread / std::mutex / std::condition_variable 을 쓴다.
# MSVC 는 스레드 지원이 CRT 에 들어 있어 아무것도 안 해도 링크되지만,
# gcc/clang 은 -pthread 가 필요하다. glibc 2.34+ 는 pthread 를 libc 에 합쳐서
# 요즘 배포판에서는 우연히 링크되지만, 그보다 낮거나 libc 가 다르면 깨진다.
find_package(Threads REQUIRED)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW ON)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
endif()

find_package(spdlog 1.12 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
        GIT_SHALLOW ON)
    FetchContent_MakeAvailable(spdlog)
endif()

# OpenCV 는 선택적이다.
# 핵심 수학(SE3/할당/스레드풀)은 Eigen 만 있으면 되므로, OpenCV 가 없는
# 환경에서도 그 부분은 빌드하고 테스트할 수 있어야 한다. 실제로 이 저장소는
# OpenCV 없이 먼저 컴파일됐다.
# dnn 은 YOLO 백엔드가 쓴다. 목록에서 빠지면 imported target 이 정의되지 않아
# 링커에 원시 이름이 그대로 넘어가고, world 빌드에서 LNK1104 로 터진다.
# features2d 는 고전 대조군(wme_tum_baseline)의 ORB 가 쓴다 - 위 함정을 그대로
# 한 번 더 밟아서 추가됐다.
find_package(OpenCV 4.8 QUIET COMPONENTS core imgproc imgcodecs videoio calib3d highgui dnn features2d)
if(OpenCV_FOUND)
    set(WME_HAS_OPENCV ON)
    message(STATUS "OpenCV ${OpenCV_VERSION} - 전체 타깃 빌드")
else()
    set(WME_HAS_OPENCV OFF)
    message(STATUS "OpenCV 미탐색 - wme_math 와 그 테스트만 빌드한다")
endif()

# --- 선택 ------------------------------------------------------------------
if(WME_WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 87 89)  # Turing~Ada, Jetson Orin(87) 포함
endif()

if(WME_WITH_TENSORRT)
    find_package(TensorRT REQUIRED)
endif()

# ONNX Runtime.
# 공식 배포본은 CMake 패키지를 담지 않으므로 헤더/라이브러리를 직접 찾는다.
# ORT_ROOT 로 위치를 알려 주거나, 표준 경로에 있으면 자동으로 잡힌다.
if(WME_WITH_ONNXRUNTIME)
    # opset 주의: YOLO11 의 공식 ONNX 는 opset 22 다. ORT 1.20 은 21 까지라
    # 로드 자체가 거부된다. 1.22 이상을 쓴다.
    set(_ort_hints
        ${ORT_ROOT} $ENV{ORT_ROOT}
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.22.0)

    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${_ort_hints} PATH_SUFFIXES include)
    find_library(ORT_LIBRARY NAMES onnxruntime
        HINTS ${_ort_hints} PATH_SUFFIXES lib)

    if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
        add_library(wme_onnxruntime INTERFACE)
        target_include_directories(wme_onnxruntime INTERFACE ${ORT_INCLUDE_DIR})
        target_link_libraries(wme_onnxruntime INTERFACE ${ORT_LIBRARY})
        set(WME_HAS_ORT ON)
        message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

        # 윈도우는 System32 에 자체 onnxruntime.dll(Windows ML)을 갖고 있고,
        # DLL 탐색 순서상 System32 가 PATH 보다 앞선다. ABI 가 다른 그 DLL 을
        # 물면 세션 생성에서 접근 위반으로 죽는다 - 실제로 0xc0000005 를 봤다.
        # 앱 디렉터리가 최우선이므로 실행 파일 옆에 우리 DLL 을 둔다.
        if(WIN32)
            get_filename_component(_ort_lib_dir "${ORT_LIBRARY}" DIRECTORY)
            find_file(ORT_RUNTIME_DLL onnxruntime.dll
                HINTS "${_ort_lib_dir}" "${_ort_lib_dir}/../bin"
                NO_SYSTEM_ENVIRONMENT_PATH NO_DEFAULT_PATH)
            if(NOT ORT_RUNTIME_DLL)
                message(WARNING "onnxruntime.dll 을 찾지 못했다 - System32 판을 물 수 있다")
            endif()
        endif()
    else()
        set(WME_HAS_ORT OFF)
        message(STATUS "onnxruntime 미탐색 - ORT YOLO 백엔드 비활성화 (OpenCV DNN 백엔드는 유지)")
    endif()
else()
    set(WME_HAS_ORT OFF)
endif()

if(WME_WITH_OPEN3D)
    find_package(Open3D REQUIRED)
endif()

if(WME_WITH_PCL)
    find_package(PCL 1.13 REQUIRED COMPONENTS common io filters segmentation)
endif()

# --- 테스트/벤치 -----------------------------------------------------------
if(WME_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW ON)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

if(WME_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        FetchContent_Declare(googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.0
            GIT_SHALLOW ON)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
endif()
