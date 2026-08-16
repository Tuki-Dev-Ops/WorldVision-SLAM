<div align="center">

# WorldVision-SLAM

### WME — World Model Engine

**기술자(descriptor)를 기억하는 대신, 세계를 기억하는 SLAM.**

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

## WME는 무엇을 하려는가?

SLAM은 카메라나 LiDAR 같은 센서를 이용해 **내 위치와 주변 환경을 동시에 추정하는 기술**입니다.

기존의 많은 시각 SLAM 시스템은 영상에서 특징점을 찾고, 그 주변의 픽셀을 descriptor라는 숫자 벡터로 표현한 다음, 이전 프레임에서 본 것과 비슷한 descriptor를 찾아 서로 같은 지점을 연결합니다.

이 방법은 조명이 안정적이고 텍스처가 충분한 환경에서는 매우 잘 작동합니다.

문제는 환경이 조금만 어려워졌을 때입니다.

안개가 끼거나, 밤이 되거나, 모션 블러가 생기거나, 텍스처가 부족해지면 descriptor 자체가 불안정해집니다. 더 어려운 점은 **완전히 실패하는 것이 아니라 그럴듯한 오답을 만들어낼 수 있다는 것**입니다.

WME는 여기서 다른 질문을 던집니다.

> **"픽셀을 비교하지 않고도, 같은 장소라는 것을 알아낼 수 있다면 어떨까?"**

사람은 어두운 방에 들어갔을 때 픽셀이나 특징점을 하나씩 비교하지 않습니다.

책상, 의자, 벽, 문, 창문 같은 **세계의 구조와 관계**를 기억하고 있기 때문입니다.

WME가 만들고 싶은 SLAM도 비슷한 방향입니다.

특징점 자체를 오래 기억하는 대신,

* 어떤 물체가 있었는지
* 물체들이 서로 어떻게 배치되어 있었는지
* 바닥과 벽은 어디에 있었는지
* 관측을 얼마나 믿을 수 있는지
* 이전에 같은 장소에서 무엇을 보았는지

같은 정보를 이용해 **세계 자체를 표현하고 다시 찾아가는 것**을 목표로 합니다.

---

# 핵심 아이디어

WME의 대응(correspondence)은 하나의 방법에 의존하지 않습니다.

현재는 세 가지 서로 다른 정보원을 사용하고, 각 정보원이 현재 환경에서 얼마나 신뢰할 수 있는지를 정보량(information) 관점에서 계산해 결합합니다.

```text
                    Environment E
                         │
                         ▼
              ┌─────────────────────┐
              │ Environment Analyzer │
              └──────────┬──────────┘
                         │
             information weighting
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Tier 0          Tier 1          Tier 2
        ECDA            TCG             SPA
          │              │              │
       Photometric    World Tokens    Structure
       Alignment      / Objects       / Planes
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    Pose Fusion
                         │
                         ▼
                       Pose
```

### Tier 0 — ECDA

**Direct Photometric Alignment**

이미지의 밝기 차이를 직접 이용해 두 프레임을 정렬합니다.

descriptor나 feature matching을 사용하지 않습니다.

환경이 충분히 안정적이라면 가장 많은 픽셀 정보를 직접 활용할 수 있다는 장점이 있습니다.

---

### Tier 1 — TCG

**Token Constellation Geometry**

이미지에서 인식된 물체를 하나의 `World Token`으로 보고, 개별 특징점 대신 **물체들의 상대적인 배치**를 이용합니다.

예를 들어,

```text
        [Chair]

[Table]       [Person]

        [Monitor]
```

와 같은 배치를 하나의 구조로 기억합니다.

카메라가 이동해도 개별 픽셀은 달라질 수 있지만, 물체 사이의 관계는 비교적 안정적으로 유지될 수 있습니다.

현재 구현에서는 YOLO 기반 객체 검출을 이용해 토큰을 생성합니다.

---

### Tier 2 — SPA

**Structural Plane Alignment**

벽, 바닥, 천장과 같은 구조적 정보를 이용합니다.

특징점이 거의 없는 복도나 텍스처가 부족한 벽처럼 일반적인 feature 기반 방법이 어려운 환경에서도 **평면의 방향과 거리**는 유용한 정보를 제공할 수 있습니다.

---

## 세 정보를 어떻게 합치는가?

WME는 다음과 같이 각 정보원의 기여도를 하나의 정보 행렬로 합칩니다.

```text
Λ_total =
    α₀(E) · Λ_ECDA
  + α₁(E) · Λ_TCG
  + α₂(E) · Λ_SPA
```

여기서 중요한 부분은 `α`입니다.

WME에는

```text
if night:
    use token

if rain:
    use plane

if fog:
    disable photometric
```

같은 환경별 분기 코드를 넣지 않습니다.

대신 현재 환경에서 각각의 정보원이 얼마나 신뢰할 수 있는지를 측정하고, 그 결과에 따라 정보량을 조절합니다.

즉,

**"밤이면 A를 사용한다"가 아니라
"현재 관측에서 A가 제공하는 정보량이 줄었다"**

라는 방식으로 환경 변화를 다룹니다.

---

# 왜 기존 SLAM과 비교하는가?

새로운 알고리즘을 만들었다고 해서 그것이 실제로 더 좋은 것은 아닙니다.

그래서 WME에서는 비교 자체를 프로젝트의 중요한 부분으로 보고 있습니다.

대조군은 일부러 단순하게 만들었습니다.

```text
ORB
 ↓
Descriptor Matching
 ↓
RANSAC
 ↓
PnP
 ↓
Pose
```

즉, WME가 사용하지 않겠다고 한 바로 그 종류의 파이프라인을 비교 대상으로 삼습니다.

이를 통해 질문을 최대한 단순하게 만들었습니다.

> **"descriptor 기반 대응보다 world-level correspondence가 실제로 더 나은가?"**

현재 벤치마크에는 TUM RGB-D와 KITTI odometry가 포함되어 있으며, 동일한 데이터와 동일한 평가 방법으로 두 시스템을 비교합니다.

| Dataset | Sequences | Baseline | Current Result |
| --- | ---: | --- | ---: |
| TUM RGB-D | 16 | ORB + PnP, `cv2.Odometry` | 8 – 7 |
| KITTI Odometry | 4 | ORB + PnP | 4 – 0 |

TUM에서는 사실상 비슷한 수준의 결과가 나옵니다.

반면 KITTI에서는 현재 평가한 4개 시퀀스에서 WME가 앞서고 있습니다.

다만 이 숫자를 "WME가 SLAM을 해결했다"는 의미로 해석하지는 않습니다.

현재 결과는 **특정 데이터와 특정 설정에서의 실험 결과**이며, 일반적인 환경에서의 우월성을 증명하려면 더 많은 데이터와 비교가 필요합니다.

그리고 위 표는 모든 시퀀스에 같은 설정으로 돌린 숫자입니다. 시퀀스마다 가장 좋은 변형을 골라 쓰면 TUM은 10–5가 되지만, 그것은 실제로 배포되는 시스템이 할 수 없는 선택입니다.

---

# 이 프로젝트에서 특히 중요하게 보는 것

WME에서는 알고리즘만큼 **실험의 신뢰성**을 중요하게 보고 있습니다.

그래서 몇 가지 원칙을 정해두었습니다.

### 1. C++ 구현과 독립적인 Python Oracle

핵심 수학 연산은 C++ 구현과 별도로 NumPy 기반 reference implementation을 가지고 있습니다.

```text
C++ implementation
       │
       │
       ├──── compare ──── Python / NumPy Oracle
       │
       ▼
     Result
```

한 구현이 자기 자신을 검증하는 구조를 피하기 위해서입니다.

---

### 2. 추정과 평가를 분리

SLAM은 C++에서 실행하지만 ATE/RPE와 같은 평가는 Python에서 별도로 수행합니다.

```text
C++  →  trajectory

Python → ATE / RPE / alignment
```

추정 코드에 있는 버그가 평가 코드의 버그와 서로 상쇄되는 상황을 최대한 피하려는 목적입니다.

---

### 3. 측정 방법 자체도 검증

실험에서 가장 위험한 것은 알고리즘이 틀린 것이 아니라 **측정이 틀린 상태에서 결과를 믿는 것**입니다.

실제로 개발 과정에서 그런 문제가 여러 번 발견되었습니다.

예를 들어 KITTI에서는 깊이 오차에 대한 불확실성을 제대로 반영하지 않았을 때 결과가 2–2로 나왔습니다.

60 m 지점의 깊이 오차는 ±4 m인데 6 m 지점(±4 cm)과 같은 무게로 들어가고 있었습니다. 이 불확실성을 잔차 분산으로 옮긴 뒤 결과가 4–0으로 바뀌었습니다. 계수는 손으로 맞춘 값이 아니라 `c = σ_d / (f·B)`로 유도한 값입니다.

TUM에서도 일부 데이터셋의 압축이 중간에 끊어진 상태에서 프레임 인덱스만 정상적으로 남아 있어, 실제보다 훨씬 적은 데이터를 평가하고 있던 문제가 발견되었습니다.

16개 중 13개가 평균 35 %, 최저 6.4 %만 채점되고 있었습니다. 데이터를 다시 받아 평가한 결과 일부 판정이 양방향으로 바뀌었습니다.

이런 사례 때문에 WME에서는 **실험 결과뿐 아니라 실패 과정도 기록**합니다.

자세한 내용은 [`docs/06-results.md`](docs/06-results.md)에 정리되어 있습니다.

---

### 4. 실패는 조용하면 안 된다

가장 위험한 실패는 프로그램이 죽는 실패가 아니라, 그럴듯한 값을 내면서 계속 굴러가는 실패입니다.

개발 과정에서 실제로 이런 것들이 있었습니다.

* 정렬 비교자가 부동소수 잡음 1 ULP에 순위를 맡기고 있어서, 같은 지도를 다른 기계에서 돌리면 다른 장소가 루프 클로저 후보로 뽑혔습니다.
* 거리 구간화 함수가 NaN을 그대로 통과시켜 `static_cast<int>(NaN)`에 닿았고, 그 결과가 멀쩡해 보이는 0번 구간이 되었습니다.
* Unity 씬에 스크립트 참조가 끊긴 컴포넌트가 직렬화되어 실행할 때 `level0 is corrupted`로 죽었는데, 엔진은 매 빌드마다 원인을 로그에 적고 있었고 로그 필터가 그 줄을 지우고 있었습니다.

지금은 이런 자리마다 시끄럽게 실패하도록 검사를 넣어두고 있습니다.

---

# 현재 구현 범위

현재 WME는 단순한 SLAM 알고리즘 하나가 아니라, 실험부터 시각화까지 연결된 작은 연구 플랫폼에 가깝습니다.

```text
WorldVision-SLAM/
│
├── include/wme/
│   ├── core/
│   ├── localization/       # ECDA
│   ├── token/              # TCG
│   ├── geometry/           # SPA
│   ├── fusion/
│   ├── perception/
│   └── confidence/
│
├── src/
│
├── tools/
│   ├── tum_odometry.cpp
│   ├── tum_baseline.cpp
│   ├── kitti_convert.cpp
│   ├── equirect_convert.cpp
│   ├── tum_loopclose.cpp
│   ├── tum_degrade.cpp
│   ├── tum_fusion.cpp
│   ├── scene_export.cpp
│   ├── seg_export.cpp
│   ├── bench_viewer.cpp
│   └── ...
│
├── tests/
│
├── python/
│   ├── wme/
│   │   ├── reference/
│   │   ├── localization/
│   │   ├── geometry/
│   │   ├── eval/
│   │   ├── graph/
│   │   ├── sim/
│   │   └── ...
│   │
│   ├── bindings/
│   ├── tools/
│   └── tests/
│
├── docs/
│   ├── 00-manifesto.md
│   ├── 01-architecture.md
│   ├── 02-correspondence-problem.md
│   ├── 03-roadmap.md
│   ├── 04-unified-objective.md
│   ├── 05-research-program.md
│   ├── 06-results.md
│   └── 07-adverse-weather.md
│
├── results/
│   └── bench/
│
└── engine/
    ├── unity/
    └── unreal/
```

---

# 데이터셋

## TUM RGB-D

실내에서 손으로 카메라를 움직이는 환경입니다.

Kinect 구조광 센서의 측정 깊이를 사용할 수 있기 때문에 WME의 기본적인 오도메트리 성능을 검증하기에 적합합니다.

현재 16개 시퀀스를 사용합니다.

```bash
python python/tools/tum_fetch.py
```

다양한 텍스처와 움직임, 동적 객체가 포함되어 있어 일반적인 실내 환경부터 퇴화 조건까지 비교할 수 있습니다.

내부 파라미터와 왜곡 계수는 freiburg 그룹마다 다릅니다. 하나로 고정하면 다른 그룹에서 실패가 아니라 그럴듯한 오차가 나옵니다.

---

## KITTI Odometry

차량에서 촬영한 실외 데이터입니다.

TUM과 달리 직접적인 깊이 영상이 없기 때문에 스테레오 영상으로 깊이를 추정합니다.

```bash
python python/tools/fetch_kitti.py
build/win/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
```

시차 탐색 범위는 장면의 최근접 거리에서 유도합니다. 이 값을 대충 두면 SGBM은 범위 밖이라고 말하지 않고 그럴듯한 오답을 냅니다. 실측으로 깊이 스케일이 2.42배 어긋난 적이 있습니다.

현재 00, 04, 05, 07 네 시퀀스를 변환해 평가하고 있습니다.

### kitti_04는 다른 셋과 같은 무게로 보면 안 됩니다

텍스처가 부족한 시골길이고 프레임당 이동이 2.91 m로 큽니다. 여기서 궤적이 세로로 표류합니다.

| 시퀀스 | 수직오차 RMS | ATE |
| --- | ---: | ---: |
| kitti_00 | 0.26 m | 94 cm |
| kitti_05 | 0.17 m | 141 cm |
| kitti_07 | 0.34 m | 108 cm |
| kitti_04 | **6.14 m** | **955 cm** |

RPE는 1.5배밖에 나쁘지 않습니다. 즉 프레임 단위 잡음이 아니라 누적된 표류입니다.

이것은 화면에서 이렇게 보입니다.

```text
카메라 영상        평평한 길을 직진

세계 모델          자차가 노면에 박히거나 떠 있음
                  (내리막에서 박히고, 오르막에서 뜸)
```

원인이 이어지는 방식은 이렇습니다.

```text
궤적이 세로로 표류
      ↓
지도가 세로로 번짐          노면 열의 세로 퍼짐 1.03 m  (다른 시퀀스 0.04 m)
      ↓
지면 추정이 번진 무리의 아래쪽에 앉음
      ↓
자차가 지면에 박히거나 뜸    높이 폭 4.79 m           (다른 시퀀스 0.15 m)
      ↓
진짜 노면 점이 "미상"이 됨   미상 33.6 %              (다른 시퀀스 2.9–4.9 %)
```

이런 케이스도 단순히 "실패"라고 기록하지 않고, **왜 실패했는지 지도와 추정 상태까지 함께 확인**하는 것을 목표로 합니다.

그래서 Unity 뷰어는 시퀀스를 불러올 때 자차 높이의 p10 / p50 / p90을 재고, 폭이 중앙값보다 크면 로그에 Error를 남깁니다. 문턱은 따로 고른 값이 아니라 그 시퀀스 자신에게서 나온 두 수이고, 폭이 값보다 크다는 것은 자차가 어디에서는 지면 아래에 있고 어디에서는 두 배 위에 있다는 뜻입니다.

---

# 360° 카메라도 지원

일반적인 SLAM 파이프라인은 대부분 핀홀 카메라 모델을 전제로 합니다.

하지만 360° 카메라의 원본 영상은 등장방형(equirectangular) 투영을 사용하기 때문에 그대로 입력하면 기하 모델이 맞지 않습니다.

문제는 프로그램이 반드시 에러를 내는 것이 아니라는 점입니다.

**정상적으로 실행되면서 잘못된 결과를 만들 수 있습니다.**

그래서 WME에서는 360° 영상을 먼저 원근 뷰로 변환한 뒤 기존 SLAM 파이프라인에 넣습니다.

```bash
build/win/tools/wme_equirect_convert \
    --in <360폴더> \
    --out data/pano_front \
    --yaw 0 \
    --pitch 0 \
    --hfov 90 \
    --width 640 \
    --height 480
```

내부 파라미터는 상수로 넣지 않고 뷰 파라미터에서 유도합니다.

```text
cx = (W - 1) / 2
fx = (W / 2) / tan(hfov / 2)
```

변환 과정은 합성 데이터와 실제 360° 카메라 데이터를 이용해 별도로 검증하고 있습니다.

| 항목 | 결과 |
| --- | ---: |
| C++ ↔ NumPy Oracle | 최대 1 LSB |
| 체커보드 재투영 RMS | 0.078 px |
| 같은 장면을 핀홀로 직접 렌더한 대조군 | 0.095 px |
| 360° Stereo 유효 깊이 | 92.4 % |
| Stereo 상대오차 중앙값 | 0.36 % |
| C++ ↔ Oracle (실사진 7개 뷰) | 최대 4 LSB |

## 합성 검증으로는 답할 수 없던 것

경도가 증가하는 방향, 즉 세계가 좌우로 뒤집혀 있는지 여부는 합성 데이터로는 원리적으로 확인할 수 없었습니다.

검증 스크립트가 같은 규약으로 원본을 만들고 같은 규약으로 다시 읽기 때문에, 규약이 통째로 뒤집혀 있어도 오차 0으로 통과합니다. 그리고 이 문제는 `--yaw` 값으로 흡수되지도 않습니다.

그래서 제조사가 다른 소비자용 360 카메라 두 대(GoPro Max, RICOH THETA SC)의 실제 촬영본으로 확인했습니다.

* 간판 글자가 정상으로 읽힙니다. 거울상이면 바로 드러납니다.
* EXIF의 GPS와 UTC 시각으로 계산한 태양 방위가 영상에서 잰 위치와 맞습니다.
* 길의 소실 방향이 OpenStreetMap의 실제 도로 방위와 3.5° 안에서 일치합니다. 좌우 반전을 가정하면 132° 어긋납니다.

## 스테레오로 깊이를 만들 때의 제약

360 스테레오 리그를 쓰면 깊이까지 만들 수 있지만, 아무 방향에서나 되는 것은 아닙니다.

베이스라인과 나란한 방향에서는 시차가 깊이 정보를 잃고, 그와 직교하는 방향에서는 세로 시차가 생겨 SGBM이 쓸 수 없습니다.

```bash
build/win/tools/wme_equirect_convert \
    --in <좌> --right <우> \
    --baseline 0.30 --baseline-yaw 0 \
    --out data/pano_stereo --yaw 0 --hfov 90
```

`--baseline-yaw`는 베이스라인이 향하는 파노라마 방위입니다. 기본값 0은 어디까지나 가정이고 리그마다 다릅니다.

실제로 TartanAir의 리그는 이 방위가 90° 어긋나 있어서, 가정을 그대로 둔 상태에서는 정렬 판정이 정확히 거꾸로 나왔습니다. 시차가 깊이를 담지 않는 뷰를 통과시키고, 쓸 수 있는 뷰를 거부했습니다. 지금은 스테레오를 쓸 때마다 이 가정을 화면에 출력합니다.

## 실제 360 시퀀스 받기

아카이브를 통째로 받지 않고 HTTP 범위 요청으로 필요한 프레임만 꺼냅니다. 실측으로 3프레임에 21 MB이고, 아카이브 전체는 6.89 GB입니다.

```bash
python python/tools/fetch_tartanair_360.py --out data/pano_seq --frames 50
```

---

# 시각화

**결과를 보는 곳은 Unity 빌드 하나입니다.** SLAM이 만들어낸 세계를 직접 돌아다니면서 확인합니다.

C++ 쪽에도 뷰어 창이 있지만 그것은 개발용 계측기이고, 결과를 보려고 띄울 필요는 없습니다. C++ 도구가 파이프라인에서 하는 일은 **씬을 파일로 내보내는 것**이고, 그건 아래 한 줄이면 끝납니다.

```text
SLAM
 ↓
World Model
 ↓
JSON / WVPC        ← C++ 가 여기까지 만든다 (headless)
 ↓
Unity
 ↓
3D World           ← 여기서 본다
```

## 씬 내보내기 (한 번)

```bash
# Windows: OpenCV DLL 경로를 PATH 에 넣어야 한다
$env:PATH = "C:\opencv-dl\opencv\build\x64\vc16\bin;" + $env:PATH

build/win/tools/wme_bench_viewer --manifest results/bench/viewer.tsv \
    --seq 0 --frame 399 --voxel 0.15 \
    --export-json results/scene/kitti_00.json \
    --screenshot  /tmp/shot.png
```

`--export-json`을 주면 창을 띄우지 않고 `.json`과 `.wvpc`를 만들고 끝납니다. `.wvpc`는 확장자만 바꿔 자동으로 함께 나옵니다.

DLL 경로를 빼면 `exit -1073741515`로 끝나는데, 종료 코드만 보면 크래시처럼 보이지만 실제로는 실행 자체가 시작되지 않은 것입니다.

`--frame`에 시퀀스 길이 이상을 주면 조용히 무한 루프합니다. `nframes`는 kitti_00 / 05 / 07이 400, kitti_04가 136이므로 각각 399, 135가 최대입니다.

---

## Unity Viewer

여기가 결과를 보는 곳입니다. 위에서 만든

```text
.json
.wvpc
```

파일을 Unity가 읽어들입니다.

WASD 이동, 마우스 시점, 점프, 비행 등의 기본적인 탐색 기능을 제공합니다.

```powershell
powershell -ExecutionPolicy Bypass `
    -File engine/unity/build.ps1 -Run
```

오른쪽 위 목록에서 시퀀스를 고릅니다. 화면 이름은 `Data_Set_01`~`05`이고 그 아래에 산출물 이름이 함께 뜹니다. 대응표는 `engine/unity/wvdata_names.tsv` 한 곳입니다.

데이터만 바꾸는 경우 Unity를 다시 빌드할 필요도 없습니다. 실행 파일 옆의 `wvdata/`를 교체하면 됩니다.

```powershell
Copy-Item results/scene/*.json,results/scene/*.wvpc build/unity/wvdata/ -Force
```

Unity 없이 C# 코드만 컴파일 검사할 수도 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File engine/unity/Check/check.ps1
```

씬 파일은 직접 편집하지 않습니다. `build.ps1`이 매 빌드마다 씬을 지우고 임포터가 코드로 다시 세우기 때문에, 씬을 바꾸려면 임포터를 고쳐야 합니다.

한 `.cs` 파일에는 `MonoBehaviour`를 하나만 둡니다. 파일 하나에 `MonoScript` 에셋은 하나뿐이라 나머지 클래스는 씬이 가리킬 대상이 없고, 에디터 안에서는 정상으로 보이다가 직렬화할 때 참조가 끊깁니다. 그것이 `level0 is corrupted`의 원인이었습니다. 지금은 임포터가 빌드 직전에 저장된 씬을 검사해서 막습니다.

---

## Unreal

같은 `.json`을 읽는 임포터가 `engine/unreal/`에 있습니다. 엔진 없이 스텁으로 검증할 수 있습니다.

```bash
python engine/unreal/Check/check.py
```

---

# 테스트

현재 테스트는 두 층으로 구성되어 있습니다.

```text
C++ tests
236 cases

Python tests
645 cases
```

특히 Python 테스트의 상당 부분은 C++ 구현과 NumPy reference implementation의 결과를 직접 비교합니다.

```text
C++ Result
     │
     ├──────────────┐
     │              │
     ▼              ▼
  expected       NumPy
                 Oracle
     │              │
     └──── compare ─┘
```

테스트를 실행하려면 먼저 Python extension을 포함해 빌드합니다.

```bash
cmake -S . -B build -DWME_BUILD_PYTHON=ON
cmake --build build

ctest --test-dir build --output-on-failure

cd python
python -m pytest -q
```

Native extension이 없는 상태에서 테스트가 조용히 skip되는 문제도 발견되어 현재는 이를 명시적으로 처리합니다.

즉,

> **테스트가 통과했다는 것과 테스트가 실제로 실행되었다는 것은 다르다.**

확장이 없으면 `RuntimeWarning`이 오류로 올라가 수집 단계에서 멈춥니다. 확장이 없는 상태가 의도된 것이라면 `WME_NATIVE_OPTIONAL=1`로 선언해야 하고, 그때 645개 중 409 통과 / 234 skip / 2 xfail이 나옵니다. 반대로 `WME_REQUIRE_NATIVE=1`은 임포트 자체를 실패시킵니다.

CI에서도 두 잡이 각자의 계약을 명시합니다. `linux`는 차분 테스트 직전에 `assert HAS_NATIVE`를 두고, `python`은 확장을 일부러 빌드하지 않는 잡이라 `WME_NATIVE_OPTIONAL=1`을 선언합니다.

---

# 벤치마크

전체 벤치마크는 다음과 같이 실행합니다.

```bash
python python/tools/bench_run.py
python python/tools/bench_report.py
```

그러면 다음 결과를 한 화면에서 비교할 수 있습니다.

```text
┌──────────────────────┬──────────────────────┐
│ Existing Pipeline    │ WorldVision-SLAM     │
│                      │                      │
│ ORB + PnP            │ ECDA + TCG + SPA     │
│                      │                      │
│ Trajectory           │ Trajectory           │
│ ATE                  │ ATE                  │
│ RPE                  │ RPE                  │
│ Runtime              │ Runtime              │
└──────────────────────┴──────────────────────┘
```

결과 뷰어는 `results/bench/index.html`에 생성됩니다.

일부만 다시 돌릴 수도 있습니다.

```bash
python python/tools/bench_run.py --only kitti --merge   # KITTI 만, 나머지 보존
python python/tools/bench_run.py --skip-run             # 재추정 없이 재채점
```

---

# 현재 결과를 어떻게 봐야 하는가

WME는 아직 "기존 SLAM을 대체했다"고 말할 단계는 아닙니다.

현재까지의 결과는 오히려 다음을 보여주는 단계에 가깝습니다.

**1. descriptor 없이도 visual correspondence를 구성할 수 있다.**

**2. 서로 다른 정보원들을 하나의 정보량 기반 모델로 결합할 수 있다.**

**3. 환경이 나빠질수록 특정 정보원에 의존하지 않고 다른 정보원으로 무게를 이동시킬 수 있다.**

**4. 하지만 이 방식이 모든 환경에서 기존 방법보다 우수하다는 것은 아직 증명되지 않았다.**

특히 실제 비·눈·안개 환경, 더 다양한 카메라, 장시간 주행, 대규모 루프 클로저 등은 아직 더 검증해야 합니다.

현재 안개 실험은 실제 TUM 프레임에 실측 깊이로 산란 방정식을 적용해 만든 합성 열화이며 자연 열화 데이터가 아닙니다. 실제 악천후 공개 데이터셋을 조사한 결과, 스테레오 영상과 정답 포즈를 동시에 제공하는 것이 생각보다 적다는 점도 확인했습니다. 조사 내용은 [`docs/07-adverse-weather.md`](docs/07-adverse-weather.md)에 정리되어 있습니다.

그래서 WME에서는 **결과보다 아직 해결하지 못한 문제를 함께 공개하는 것**을 중요하게 생각합니다.

---

# Roadmap

WME의 방향은 단순히 "descriptor를 다른 것으로 교체하는 것"에 있지 않습니다.

장기적으로는 SLAM의 지도를 **기하 정보만 저장하는 공간에서 세계에 대한 기억으로 확장하는 것**을 목표로 합니다.

```text
현재

Image
  ↓
Correspondence
  ↓
Pose
  ↓
Geometry Map
```

에서

```text
목표

Observation
     ↓
World Understanding
     ↓
World Model
     ↓
Memory
     ↓
Prediction
     ↓
Localization
```

으로 확장하는 것입니다.

이를 위해 앞으로는

* Semantic World Model
* Temporal Scene Memory
* Object-level Mapping
* Dynamic Object Modeling
* Uncertainty-aware Memory
* Hypothesis / Prediction Layer
* Adaptive SLAM Policy
* Long-term Localization
* Adverse Weather SLAM
* Loop Closure
* Large-scale World Representation

등을 단계적으로 연구할 예정입니다.

---

# 철학

WME를 한 문장으로 설명하면 이렇습니다.

> **SLAM이 특징점을 기억해야 하는가?
> 아니면 세계를 기억해야 하는가?**

이 프로젝트는 두 번째 질문에서 시작했습니다.

아직 답을 완성한 것은 아닙니다.

그래서 이 저장소에는 성공한 결과뿐만 아니라 잘못된 가설, 잘못된 측정, 데이터셋 문제, 구현상의 결함도 함께 남겨두고 있습니다.

좋은 SLAM을 만드는 것만큼,

**무엇을 믿을 수 있는지 정확하게 아는 것**

이 중요하다고 생각하기 때문입니다.

---

## Documentation

* [`00-manifesto.md`](docs/00-manifesto.md) — 왜 descriptor를 버리려 하는가
* [`01-architecture.md`](docs/01-architecture.md) — WME 전체 구조
* [`02-correspondence-problem.md`](docs/02-correspondence-problem.md) — 대응 문제에 대한 이론적 배경
* [`03-roadmap.md`](docs/03-roadmap.md) — 개발 및 연구 방향
* [`04-unified-objective.md`](docs/04-unified-objective.md) — 통합 목적함수
* [`05-research-program.md`](docs/05-research-program.md) — 연구 프로그램
* [`06-results.md`](docs/06-results.md) — 전체 실험 결과와 실패 기록
* [`07-adverse-weather.md`](docs/07-adverse-weather.md) — 악천후 환경 연구

---

<div align="center">

**WorldVision-SLAM**

*Don't match the pixels. Understand the world.*

</div>
