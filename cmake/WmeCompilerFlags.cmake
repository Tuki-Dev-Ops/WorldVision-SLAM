# 프로젝트 전역 경고/최적화 플래그 정의

include(CheckCXXCompilerFlag)

# 소스는 전부 UTF-8 이고 주석이 한국어다. 인코딩을 명시하지 않으면
# MSVC 는 BOM 없는 파일을 시스템 코드페이지(949)로 읽어 주석 끝의 역슬래시
# 바이트가 줄잇기로 해석되고, gcc/clang 은 로케일에 따라 입력 문자셋이 달라진다.
# 저장소 안에 BOM 있는 파일과 없는 파일이 섞여 있으므로 플래그로 못박는다.
if(MSVC)
    set(WME_WARNING_FLAGS /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor /bigobj /utf-8)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    set(WME_WARNING_FLAGS
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast
        -Wcast-align -Wunused -Woverloaded-virtual
        -Wnull-dereference -Wdouble-promotion -Wformat=2)

    # clang 은 15 이전에 -finput-charset 을 아예 거부하므로 지원 여부를 확인한다.
    check_cxx_compiler_flag("-finput-charset=UTF-8" WME_HAS_INPUT_CHARSET)
    if(WME_HAS_INPUT_CHARSET)
        list(APPEND WME_WARNING_FLAGS -finput-charset=UTF-8)
    endif()
    check_cxx_compiler_flag("-fexec-charset=UTF-8" WME_HAS_EXEC_CHARSET)
    if(WME_HAS_EXEC_CHARSET)
        list(APPEND WME_WARNING_FLAGS -fexec-charset=UTF-8)
    endif()
endif()

if(WME_NATIVE_ARCH AND NOT MSVC)
    list(APPEND WME_WARNING_FLAGS -march=native)
endif()

# 대상 타깃에 새니타이저 적용 (Debug 검증용)
function(wme_apply_sanitizers target)
    if(NOT WME_ASAN)
        return()
    endif()
    if(MSVC)
        target_compile_options(${target} INTERFACE /fsanitize=address)
    else()
        target_compile_options(${target} INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} INTERFACE -fsanitize=address,undefined)
    endif()
endfunction()

# WME 라이브러리 타깃 생성 헬퍼
function(wme_add_library name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
    add_library(${name} ${ARG_SOURCES})
    add_library(wme::${name} ALIAS ${name})
    # Threads::Threads 는 MSVC 에서 빈 값이고, gcc/clang 에서만 -pthread 를 붙인다.
    target_link_libraries(${name} PUBLIC wme::settings Threads::Threads ${ARG_DEPS})
    set_target_properties(${name} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        FOLDER "wme")
endfunction()
