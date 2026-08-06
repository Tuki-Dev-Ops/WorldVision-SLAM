if(NOT WME_HAS_OPENCV)
    return()
endif()

add_executable(wme_env_probe env_probe.cpp)
target_link_libraries(wme_env_probe PRIVATE
    wme::wme_perception
    opencv_highgui
    opencv_videoio)
set_target_properties(wme_env_probe PROPERTIES FOLDER "wme/tools")

# 실데이터 오도메트리. 채점은 python 이 맡는다 - 추정기와 채점기를 분리해야
# 두 쪽의 버그가 서로를 가려 주지 않는다.
add_executable(wme_tum_odometry tum_odometry.cpp)
target_link_libraries(wme_tum_odometry PRIVATE
    wme::wme_localization
    wme::wme_perception
    wme::wme_token
    opencv_imgcodecs
    # 재측위는 지도 키프레임과 기술자로 정합한다 (--reloc).
    opencv_features2d
    opencv_calib3d)
set_target_properties(wme_tum_odometry PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_odometry)
wme_stage_runtime_dlls(wme_env_probe)


# 고전 특징점 대조군 (ORB + 기술자 정합 + RANSAC PnP). WME 가 쓰지 않겠다고
# 선언한 바로 그 파이프라인이므로, 이것이 정직한 비교 대상이다.
# ORB-SLAM3 가 아니다 - 루프 클로저도 BA 도 없다. 소스 주석 참조.
add_executable(wme_tum_baseline tum_baseline.cpp)
target_link_libraries(wme_tum_baseline PRIVATE
    wme::wme_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_features2d
    opencv_calib3d)
set_target_properties(wme_tum_baseline PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_baseline)


# 물리 기반 열화 생성기. TUM 에 없는 악조건을 만든다 - 투과율을 실측 깊이에서
# 뽑는 것이 렌더러 안개와의 유일하고 결정적인 차이다.
add_executable(wme_tum_degrade tum_degrade.cpp)
# wme::settings 는 include 경로가 아니라 **컴파일 플래그**를 나른다. 특히
# MSVC 의 /utf-8 이 여기 있다. 다른 도구는 wme:: 라이브러리를 통해 딸려
# 받지만 이 도구는 엔진을 안 쓰므로 직접 걸어야 한다 - 빼먹으면 MSVC 가
# 소스를 코드페이지 949 로 읽고, 한국어 주석 끝의 0x5C 바이트가 줄잇기가
# 되어 바로 다음 줄의 선언이 통째로 사라진다. WmeCompilerFlags.cmake 의
# 주석이 경고하는 바로 그 함정이고, 실제로 한 번 밟았다.
target_link_libraries(wme_tum_degrade PRIVATE
    wme::settings
    opencv_imgcodecs
    opencv_imgproc)
set_target_properties(wme_tum_degrade PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_degrade)


# 루프 클로저. 후보 제안과 백엔드는 두 방식이 공유하고 검증 방법만 다르다 -
# 그래야 최종 차이를 "기술자 대 성좌" 로 읽을 수 있다.
add_executable(wme_tum_loopclose tum_loopclose.cpp)
target_link_libraries(wme_tum_loopclose PRIVATE
    wme::wme_token
    wme::wme_perception
    opencv_imgcodecs
    opencv_imgproc
    opencv_features2d
    opencv_calib3d)
set_target_properties(wme_tum_loopclose PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_loopclose)


# TCG(Tier 1) 실데이터 재지역화. 추정만 하고 채점은 python/tools/tcg_eval.py 가 한다.
add_executable(wme_tum_relocalize tum_relocalize.cpp)
target_link_libraries(wme_tum_relocalize PRIVATE
    wme::wme_perception
    wme::wme_token
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d)
set_target_properties(wme_tum_relocalize PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_relocalize)


# 3-tier 융합. tum_odometry 와 같은 규약으로 TUM 을 읽고, 세 tier 의
# (T, Lambda) 를 기록한 뒤 ablation 별 궤적을 낸다. 채점은
# python/tools/fusion_eval.py 가 한다.
add_executable(wme_tum_fusion tum_fusion.cpp)
target_link_libraries(wme_tum_fusion PRIVATE
    wme::wme_fusion
    wme::wme_localization
    wme::wme_geometry
    wme::wme_perception
    wme::wme_token
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d)
set_target_properties(wme_tum_fusion PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tum_fusion)


# TCG 노드 밀도 측정. 24.2 가 실내 재측위 실패의 원인으로 지목한 밀도를,
# 고치기 전에 어느 손잡이가 움직이는지부터 잰다.
add_executable(wme_tcg_density tcg_density.cpp)
target_link_libraries(wme_tcg_density PRIVATE
    wme::wme_perception
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d)
set_target_properties(wme_tcg_density PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_tcg_density)


# 평면을 성좌 노드로 쓸 수 있는가. 개수만이 아니라 시점 간 centroid 안정성을
# 함께 잰다 - 25.11/25.12 가 양을 늘렸다가 품질로 실패한 자리를 반복하지 않기 위해.
add_executable(wme_plane_density plane_density.cpp)
target_link_libraries(wme_plane_density PRIVATE
    wme::wme_geometry
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d)
set_target_properties(wme_plane_density PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_plane_density)


# KITTI odometry -> TUM RGB-D 배치 변환. 스테레오 깊이를 여기서 만든다.
# 전용 러너를 쓰지 않는 이유는 kitti_convert.cpp 머리말에 적어 두었다.
add_executable(wme_kitti_convert kitti_convert.cpp)
target_link_libraries(wme_kitti_convert PRIVATE
    wme::wme_perception
    opencv_imgcodecs
    opencv_imgproc)
set_target_properties(wme_kitti_convert PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_kitti_convert)


# 벤치마크 뷰어 (실행 파일). 왼쪽 = 고전 기술자 파이프라인, 오른쪽 = WME.
# highgui 외에 새 의존성을 들이지 않는다. 지표는 results/bench/viewer.tsv 에서
# 읽고 여기서 다시 계산하지 않는다 - bench_viewer.cpp 머리말 참조.
add_executable(wme_bench_viewer bench_viewer.cpp)
target_link_libraries(wme_bench_viewer PRIVATE
    wme::wme_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc)
set_target_properties(wme_bench_viewer PROPERTIES FOLDER "wme/tools")
wme_stage_runtime_dlls(wme_bench_viewer)
