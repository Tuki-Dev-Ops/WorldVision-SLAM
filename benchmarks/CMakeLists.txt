if(NOT WME_HAS_OPENCV)
    return()
endif()

function(wme_add_benchmark name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE ${ARG_DEPS} benchmark::benchmark)
    set_target_properties(${name} PROPERTIES FOLDER "wme/benchmarks")
endfunction()

wme_add_benchmark(bench_perception
    SOURCES bench_perception.cpp
    DEPS wme::wme_perception)

wme_add_benchmark(bench_constellation
    SOURCES bench_constellation.cpp
    DEPS wme::wme_token)

wme_add_benchmark(bench_localization
    SOURCES bench_localization.cpp
    DEPS wme::wme_localization)

