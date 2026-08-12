<div align="center">

# WorldVision-SLAM

### WME — World Model Engine

**기술자(descriptor) 없는 SLAM.** 특징점을 기억하는 대신 세계를 기억한다.

<br>

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen-3.4-1F425F)
![pybind11](https://img.shields.io/badge/pybind11-2.11-FFD43B)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.22-005CED?logo=onnx&logoColor=white)
![GoogleTest](https://img.shields.io/badge/GoogleTest-236%20passing-brightgreen)
![pytest](https://img.shields.io/badge/pytest-645%20cases-brightgreen)

<br>

**한국어** · [English](docs/readme/README.en.md) · [中文](docs/readme/README.zh-CN.md) · [日本語](docs/readme/README.ja.md)

</div>

---

## 프로젝트 배경

고전 시각 SLAM은 30년 가까이 같은 전제 위에 서 있다. **화소 주변을 숫자 벡터로 요약(기술자)하고, 그 벡터가 비슷하면 같은 점이라고 본다.** ORB, SIFT, BRIEF가 모두 이 전제를 공유한다.

이 전제는 조건이 좋을 때 매우 잘 동작하고, 조건이 나빠지면 **조용히** 무너진다. 안개가 끼면 기술자는 "매칭 실패"를 보고하지 않는다. 그럴듯한 오답을 낸다. 밤이 되면, 비가 오면, 카메라가 흔들리면 같은 일이 벌어진다. 실패가 시끄럽지 않기 때문에 상위 계층은 그것을 알아차릴 방법이 없다.

WME는 다른 질문에서 출발한다. **사람은 기술자를 매칭하지 않는다.** 어두운 방에 들어가도 책상이 어디 있는지 안다. 화소를 대조해서가 아니라, 그 방의 **모델**을 갖고 있어서다. 대응(correspondence) 문제를 화소 수준에서 풀지 않고, 세계 모델 수준에서 푼다.

그래서 이 저장소가 만드는 것은 "더 나은 기술자"가 아니라 **기술자를 쓰지 않는 파이프라인**이고, 그 파이프라인이 정말 기술자 파이프라인보다 나은지를 같은 데이터에서 나란히 재는 장치다.

### 이 프로젝트가 스스로에게 건 조건

주장은 쉽고 검증은 어렵다. 그래서 규칙을 먼저 정했다.

| 규칙 | 이유 |
|---|---|
| **모든 C++ 구현에 독립 numpy 오라클을 붙인다** | 두 구현이 같은 답을 내야 그 답을 믿는다. 한 코드가 자기를 검증하면 버그가 자기를 가린다 |
| **추정과 채점을 다른 코드가 한다** | 추정은 C++, ATE/RPE 채점은 Python. 같은 코드로 둘 다 하면 두 버그가 서로를 상쇄한다 |
| **대조군은 "우리가 안 쓴다"고 선언한 바로 그 기술자 파이프라인이다** | 임의의 약한 상대를 이기는 것은 의미가 없다 |
| **측정이 판별하는지를 먼저 확인한다** | 모든 입력에서 같은 값이 나오는 지표는 통과해도 아무것도 증명하지 않는다 |
| **실패는 시끄러워야 한다** | "저장했다"고 찍고 파일을 안 쓴 도구, 초록으로 끝나는 도달 불가 테스트 — 전부 결함으로 취급한다 |

이 규칙들이 실제로 무엇을 잡아냈는지는 [docs/06-results.md](docs/06-results.md)에 전부 기록되어 있다. 성공한 실험만이 아니라 **거부된 가설 다섯 개와 내가 만든 결함들**까지 남아 있다.

---

## 프로젝트 목적

### 1. 세 계층으로 대응 문제를 푼다

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

| 계층 | 이름 | 하는 일 |
|---|---|---|
| **Tier 0** | ECDA | 직접 측광 정렬. 화소 밝기 잔차를 최소화한다. 기술자 없음 |
| **Tier 1** | TCG | 토큰 성좌 기하. 물체(YOLO 검출)의 상대 배치로 위치를 잡는다 |
| **Tier 2** | SPA | 구조 정렬. 평면의 법선/거리로 퇴화 축을 메운다 |

`α_k(E)`는 손으로 쓴 분기가 아니다. **환경 증거 E**(어둠, 안개, 모션블러, 텍스처 빈곤 …)에서 계산된다. 야간용 코드도, 우천용 코드도 존재하지 않는다 — 열화는 각 정보원이 기여하는 **정보량의 감소**로만 표현된다.

### 2. 그 주장을 같은 데이터에서 나란히 잰다

`results/bench/index.html`은 **왼쪽에 기존 방식, 오른쪽에 WME**를 놓고 궤적·ATE·RPE·속도를 한 화면에서 비교하는 뷰어다. 20개 시퀀스가 들어 있다.

<div align="center">

| 데이터셋 | 시퀀스 | 대조군 | 결과 (단일 설정) |
|---|---|---|---|
| **TUM RGB-D** (실내, 손) | 16 | ORB+PnP, `cv2.Odometry` | **8 – 7** (둘 중 나은 쪽 기준) |
| **KITTI odometry** (실외, 차량) | 4 | ORB+PnP | **4 – 0** |

</div>

**TUM은 사실상 무승부다.** 시퀀스마다 좋은 변형을 고르면 10–5가 되지만, 그건 배포되는 시스템이 못 하는 선택이다. 위 표는 **모든 시퀀스에 같은 설정**(Tier 0)을 돌린 숫자다.

이 수치들은 두 번 크게 움직였고, 두 번 다 알고리즘이 아니라 측정 쪽이 원인이었다.

- **KITTI를 붙이자 2–2로 뒤집혔다.** ECDA가 스테레오 깊이를 참값으로 믿고 있었다 — 60 m 지점의 깊이 오차는 ±4 m인데 6 m 지점(±4 cm)과 같은 무게로 들어가고 있었다. 불확실성을 잔차 분산으로 옮기고(계수는 `c = σ_d/(f·B)`로 **유도**, 튜닝 아님) 다시 재니 4–0이 되었다. ([§25.20–25.21](docs/06-results.md))
- **TUM 16개 중 13개가 시퀀스의 일부만 채점하고 있었다** — 평균 35%, 최저 6.4%. 압축 해제가 중간에 끊겨도 프레임 인덱스는 멀쩡히 남았고, 로더는 있는 파일만 읽고 깨끗한 실행을 보고했다. 전부 다시 받아 재실행하니 판정 5개가 **양방향으로** 뒤집혔다. ([§25.22](docs/06-results.md))

> **그전까지 "WME가 더 낫다"고 적혀 있던 문장은 알고리즘의 성질이 아니라, 한 번은 데이터셋의 성질이었고 한 번은 데이터의 3분의 1만 본 결과였다.**

### 3. 무엇이 아닌지도 분명히 한다

- **ORB-SLAM3가 아니다.** 대조군은 ORB 검출 → 해밍 매칭 → RANSAC PnP까지다. 루프 클로저도 번들 조정도 없다. 비교는 **오도메트리 대 오도메트리**로만 유효하다
- 양쪽 모두 **번들 조정이 없다**. 포즈 그래프까지다
- 열화(안개) 실험은 실제 TUM 프레임에 **실측 깊이**로 산란 방정식을 적용해 만든 것이지, 자연 열화 데이터가 아니다

무엇이 확립되지 않았는지는 [docs/06-results.md §26](docs/06-results.md)에 목록으로 있다.

---

## 저장소 구조

```
WorldVision-SLAM/
├── include/wme/              공개 헤더 (25)
│   ├── core/                 SE3, Frame, Result, ThreadPool, Assignment
│   ├── localization/         DirectAligner            ← Tier 0 (ECDA)
│   ├── token/                TokenStore, ConstellationIndex, WorldToken
│   │                                                  ← Tier 1 (TCG)
│   ├── geometry/             StructuralAligner, PlaneExtractor
│   │                                                  ← Tier 2 (SPA)
│   ├── fusion/               PoseFusion, TierInformation
│   ├── perception/           ImageQualityEngine, EnvironmentAnalyzer,
│   │                         StereoDepth, YoloRuntime{Cv,Ort}
│   └── confidence/           ConfidenceEngine
│
├── src/                      구현 (19 파일, 4.3 kLOC / 헤더 포함 6.7 kLOC)
│
├── tools/                    실행 가능한 실험 바이너리 (13)
│   ├── tum_odometry.cpp      WME 오도메트리
│   ├── tum_baseline.cpp      ORB+PnP 대조군 ← "안 쓴다"고 선언한 바로 그것
│   ├── kitti_convert.cpp     KITTI → TUM 배치 + StereoSGBM 깊이
│   ├── tum_loopclose.cpp     대칭 루프 클로저 (ORB vs TCG)
│   ├── tum_degrade.cpp       실측 깊이 기반 산란 열화
│   ├── tum_fusion.cpp        3계층 융합 실행
│   ├── scene_export.cpp      검출 상자 사전 내보내기
│   ├── seg_export.cpp        SegFormer-B0 의미 분할 사전 내보내기 (ONNX)
│   ├── bench_viewer.cpp      벤치 결과 뷰어
│   └── …                     relocalize, tcg_density, plane_density, env_probe
│
├── tests/                    C++ 테스트 (16 파일, 236 케이스)
│
├── python/
│   ├── wme/
│   │   ├── reference/        ★ C++ 과 대조되는 numpy 오라클
│   │   │                       assignment, confidence, constellation,
│   │   │                       environment, environment_cues, geometry, tokens
│   │   ├── localization/     ecda.py — DirectAligner 의 오라클
│   │   ├── geometry/         spa.py, planes.py
│   │   ├── eval/             ATE / RPE / Umeyama, TUM 로더  ← 채점 전담
│   │   ├── graph/            포즈그래프, 팩터, 측광 SLAM
│   │   ├── sim/ world/       합성 장면, 세계 모델 상태/예측/변화탐지
│   │   ├── association/ calib/ planner/
│   │   └── yolo.py
│   ├── bindings/             pybind11 → wme._core
│   ├── tools/                실험·벤치 스크립트 (35)
│   │   ├── bench_run.py      두 시스템 실행 + 채점 → benchmark.json
│   │   ├── bench_report.py   → results/bench/index.html  (좌/우 비교 뷰어)
│   │   ├── fetch_kitti.py    KITTI 내려받기 (이어받기 지원)
│   │   └── …
│   └── tests/                Python 테스트 (27 파일, 645 케이스)
│
├── docs/
│   ├── 00-manifesto.md       왜 기술자를 버리는가
│   ├── 01-architecture.md    계층 구조
│   ├── 02-correspondence-problem.md   이론적 근거
│   ├── 03-roadmap.md
│   ├── 04-unified-objective.md
│   ├── 05-research-program.md
│   └── 06-results.md         ★ 모든 실측 결과와 실패 기록
│
├── results/bench/index.html  ★ 좌: 기존 모델 / 우: WME  비교 뷰어
├── cmake/                    의존성·경고 정책
└── .github/workflows/        linux, windows-msvc, sanitizers, python
```

---

## 데이터셋

두 데이터셋 모두 저장소에 **포함되어 있지 않다** (합계 51 GB). 스크립트로 재현한다.

### TUM RGB-D — 실내, 손에 든 카메라, 측정된 깊이

Kinect 구조광 센서라 깊이가 **측정값**이다. 16개 시퀀스를 쓴다.

```bash
python python/tools/tum_fetch.py
```

| 그룹 | 시퀀스 | 성격 |
|---|---|---|
| `freiburg1` | xyz, desk, room, 360, plant, teddy | 왜곡 큼(k1=0.26), 일반 실내 |
| `freiburg2` | desk, desk_with_person | 다른 카메라 — 일반화 확인용 |
| `freiburg3` | structure/nostructure × texture/notexture | 퇴화 조건 격리 |
| `freiburg3` | sitting_*, walking_* | 동적 물체 |

> 내부 파라미터와 왜곡 계수는 freiburg 그룹마다 다르다. 하나로 고정하면 다른 그룹에서 **실패가 아니라 그럴듯한 오차**가 나온다.

### KITTI odometry — 실외, 차량, 스테레오

깊이가 없다. **우리가 만들어야 한다** — 그래서 `StereoDepth`(OpenCV SGBM) 프런트엔드가 여기서 처음 필요해진다. 깊이는 측정값이 아니라 **추정값**이고, 그 구분이 §25.21의 핵심이다.

```bash
python python/tools/fetch_kitti.py        # 21.6 GB, 이어받기 지원
```

변환 시 시차 탐색 범위는 장면의 최근접 거리에서 **유도**한다. 이 값을 대충 두면 SGBM은 "범위 밖"이라고 말하지 않고 **그럴듯한 오답**을 낸다 (실측: 깊이 스케일 2.42배).

```bash
build/win/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
```

| 항목 | 값 |
|---|---|
| 내려받는 시퀀스 | 00–21 (22개, 정답 궤적은 00–10) |
| 현재 변환·평가된 시퀀스 | 00, 04, 05, 07 |
| 해상도 / 초점거리 / 베이스라인 | 1241×376 / 718.86 px / 0.537 m |
| 유효 깊이 비율 (SGBM) | 67 – 74 % |

---

## 실행 방법

### 1. 사전 준비

| 항목 | 버전 |
|---|---|
| 컴파일러 | MSVC 2022 / GCC 11+ / Clang 14+ (C++20) |
| CMake | 3.24 이상 |
| OpenCV | 4.8 이상 — `core imgproc imgcodecs videoio calib3d highgui dnn features2d` |
| Python | 3.10 이상 + `numpy scipy pytest pybind11` |
| (선택) ONNX Runtime | 1.22 — YOLO 토큰 마스킹용 |

> OpenCV 컴포넌트에서 `features2d`를 빼면 링크 단계에서만 터진다. `cmake/WmeDependencies.cmake`에 이 함정이 주석으로 적혀 있다.

### 2. 빌드

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows (MSVC BuildTools):

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build/win -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/win
```

### 3. 테스트 — 먼저 이것부터

```bash
ctest --test-dir build --output-on-failure     # C++  236 케이스
cd python && python -m pytest -q               # Python 645 케이스
```

Python 쪽 상당수는 **C++ ↔ numpy 차분 테스트**다. 여기가 초록이라는 것은 두 개의 독립 구현이 같은 답을 냈다는 뜻이고, 이 저장소에서 숫자를 믿는 근거는 그것뿐이다.

> **초록색을 그대로 믿으면 안 된다.** 차분 테스트는 `wme._core` 확장을 빌드하지 않으면 **전부 skip 된다**. 확장 없이 돌리면 645 개 중 409 통과 / 234 skip / 2 xfail 이 나오고, 아무것도 실패하지 않았으므로 초록으로 보인다 — 실제로 그렇게 41 개가 조용히 지나간 적이 있다 ([§19](docs/06-results.md)). 그래서 skip 한 줄에 `ImportError` 원문이 남고, `WME_REQUIRE_NATIVE=1` 을 켜면 skip 자체가 사라진다 (모듈이 없으면 임포트에서 터진다).
>
> 다만 **지금 CI 는 그 변수를 켜지 않는다.** `linux` 워크플로가 확장을 빌드한 뒤 차분 테스트를 돌리므로 현재는 실제로 실행되지만, 확장 빌드가 조용히 실패하면 같은 함정이 다시 열린다. 로컬에서 돌릴 때는 켜 두는 편이 안전하다.

```bash
cmake -S . -B build -DWME_BUILD_PYTHON=ON && cmake --build build
cd python && WME_REQUIRE_NATIVE=1 python -m pytest -q tests/test_differential.py   # 76 케이스
```

### 4. 벤치마크 실행 → 좌/우 비교 뷰어

```bash
python python/tools/bench_run.py               # 두 시스템 실행 + 채점
python python/tools/bench_report.py            # → results/bench/index.html
```

일부만 다시 돌릴 때:

```bash
python python/tools/bench_run.py --only kitti --merge        # KITTI 만, 나머지 보존
python python/tools/bench_run.py --skip-run                  # 재추정 없이 재채점
```

그리고 `results/bench/index.html`을 브라우저로 연다. **왼쪽 = 기존 모델(ORB+PnP), 오른쪽 = WME**, 아래에 궤적·ATE 시계열·RPE·프레임당 시간.

### 5. 단일 시퀀스 실행

```bash
# TUM
build/tools/wme_tum_odometry data/rgbd_dataset_freiburg1_xyz out.txt
build/tools/wme_tum_baseline data/rgbd_dataset_freiburg1_xyz orb.txt
python python/tools/tum_eval.py data/rgbd_dataset_freiburg1_xyz out.txt

# KITTI — 깊이 상한과 불확실성 계수는 데이터셋에서 온다
build/tools/wme_tum_odometry data/kitti_00 out.txt \
    --kf-dist 1.0 --depth-max 60 --depth-sigma-rel 7.8e-4
```

### 6. 그 밖의 실험

```bash
python python/tools/baseline_cv2.py       # 제3자 대조군 (cv2.Odometry)
python python/tools/bench_degrade.py      # 안개 스윕
python python/tools/stereo_validate.py    # 스테레오 깊이를 TUM 실측 깊이에 대고 검증
python python/tools/loop_optimize.py      # 포즈그래프 루프 클로저
build/tools/wme_tum_loopclose data/rgbd_dataset_freiburg1_room out.txt
```

---

<div align="center">

**결과 전문 · 실패 기록 · 거부된 가설 → [docs/06-results.md](docs/06-results.md)**

</div>
