<div align="center">

# WorldVision-SLAM

### WME — World Model Engine

**Descriptor-free SLAM.** It remembers the world, not the feature points.

<br>

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C?logo=cmake&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen-3.4-1F425F)
![pybind11](https://img.shields.io/badge/pybind11-2.11-FFD43B)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.22-005CED?logo=onnx&logoColor=white)
![GoogleTest](https://img.shields.io/badge/GoogleTest-236%20passing-brightgreen)
![pytest](https://img.shields.io/badge/pytest-639%20passing-brightgreen)

<br>

[한국어](../../README.md) · **English** · [中文](README.zh-CN.md) · [日本語](README.ja.md)

</div>

---

## Background

Classical visual SLAM has rested on the same premise for nearly thirty years:
**summarise the neighbourhood of a pixel as a numeric vector (a descriptor), and treat similar
vectors as the same point.** ORB, SIFT and BRIEF all share it.

The premise works very well in good conditions and collapses **quietly** in bad ones. Under fog a
descriptor does not report "no match" — it returns a plausible wrong one. The same happens at
night, in rain, and when the camera shakes. Because the failure is not loud, nothing above it has
any way to notice.

WME starts from a different question. **People do not match descriptors.** You can walk into a
dark room and still know where the desk is — not by comparing pixels, but because you hold a
**model** of that room. So the correspondence problem is not solved at the pixel level; it is
solved at the level of a world model.

What this repository builds is therefore not a better descriptor but a **pipeline that does not
use descriptors**, together with the apparatus to check, on the same data and side by side,
whether it is actually better than one that does.

### The conditions this project set for itself

Claims are cheap and verification is not, so the rules came first.

| Rule | Why |
|---|---|
| **Every C++ implementation gets an independent numpy oracle** | Two implementations agreeing is what makes an answer believable. Code that checks itself hides its own bugs |
| **Estimation and scoring are different code** | Estimation in C++, ATE/RPE scoring in Python. One codebase doing both lets two bugs cancel |
| **The control is exactly the descriptor pipeline we claim not to use** | Beating an arbitrary weak opponent means nothing |
| **Check that the measurement discriminates, first** | A metric that returns the same value for every input proves nothing when it passes |
| **Failure must be loud** | A tool that prints "saved" without writing the file, a green test that is unreachable — both are treated as defects |

What these rules actually caught is recorded in full in [docs/06-results.md](../06-results.md) —
not only the experiments that worked, but **five rejected hypotheses and the defects I introduced
myself**.

---

## Purpose

### 1. Solve correspondence in three tiers

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

| Tier | Name | What it does |
|---|---|---|
| **Tier 0** | ECDA | Direct photometric alignment. Minimises pixel intensity residuals. No descriptors |
| **Tier 1** | TCG | Token constellation geometry. Localises from the relative arrangement of objects (YOLO detections) |
| **Tier 2** | SPA | Structural alignment. Fills degenerate axes using plane normals and offsets |

`α_k(E)` is not a hand-written branch. It is computed from **environmental evidence E** (darkness,
haze, motion blur, texture poverty, …). There is no night-time code path and no rain code path —
degradation is expressed *only* as a reduction in the information each source contributes.

### 2. Measure that claim on the same data, side by side

`results/bench/index.html` puts **the classical system on the left and WME on the right** and
compares trajectory, ATE, RPE and speed on one screen. There is also a native application,
`wme_bench_viewer.exe`, which additionally reconstructs the depth map in 3D through each system's
own estimated pose — drift shows up as a smeared point cloud, not just a number.

<div align="center">

| Dataset | Sequences | Control | Result (one configuration) |
|---|---|---|---|
| **TUM RGB-D** (indoor, handheld) | 16 | ORB+PnP, `cv2.Odometry` | **8 – 7** (against the better of the two) |
| **KITTI odometry** (outdoor, vehicle) | 4 | ORB+PnP | **4 – 0** |

</div>

**TUM is a tie.** Picking the better WME variant per sequence reads 10–5, but that is a choice no
deployed system gets to make; the table above runs **the same configuration** (Tier 0) on
everything.

These numbers moved twice, and both times the cause was measurement rather than algorithms.

- **Adding KITTI flipped it to 2–2.** ECDA trusted stereo depth as exact — a point at 60 m
  carries ±4 m of depth error and was weighted like a 6 m point at ±4 cm. Moving that uncertainty
  into the residual variance (coefficient **derived** as `c = σ_d/(f·B)`, not tuned) brought it
  to 4–0. ([§25.20–25.21](../06-results.md))
- **13 of the 16 TUM sequences were being scored on a fraction of themselves** — 35 % on average,
  6.4 % at worst. An interrupted extraction left the frame index intact while the frames were
  missing, so every loader read what existed and reported a clean run. Re-fetched and re-run,
  five verdicts flipped — **in both directions**. ([§25.22](../06-results.md))

> **Every sentence that had read "WME is better" was, once, a property of TUM rather than of the
> algorithms — and once, a result computed on a third of the data.**

### 3. Be equally clear about what it is not

- **This is not ORB-SLAM3.** The control runs ORB detection → Hamming matching → RANSAC PnP and
  stops there. No loop closure, no bundle adjustment. The comparison is valid **only as odometry
  against odometry**
- **Neither side has bundle adjustment.** Pose graph is as far as it goes
- The degradation (haze) experiments apply the scattering equation to real TUM frames using
  **measured depth** — they are not naturally degraded data

What is *not* established is listed in [docs/06-results.md §26](../06-results.md).

---

## Repository layout

```
WorldVision-SLAM/
├── include/wme/              public headers (27)
│   ├── core/                 SE3, Frame, Result, ThreadPool, Assignment
│   ├── localization/         DirectAligner            <- Tier 0 (ECDA)
│   ├── token/                TokenStore, ConstellationIndex, WorldToken
│   │                                                  <- Tier 1 (TCG)
│   ├── geometry/             StructuralAligner, PlaneExtractor
│   │                                                  <- Tier 2 (SPA)
│   ├── fusion/               PoseFusion, TierInformation
│   ├── perception/           ImageQualityEngine, EnvironmentAnalyzer,
│   │                         StereoDepth, YoloRuntime{Cv,Ort}
│   └── confidence/           ConfidenceEngine
│
├── src/                      implementation (~17 kLOC)
│
├── tools/                    runnable experiment binaries
│   ├── tum_odometry.cpp      WME odometry
│   ├── tum_baseline.cpp      ORB+PnP control <- exactly what we claim not to use
│   ├── kitti_convert.cpp     KITTI -> TUM layout + StereoSGBM depth
│   ├── bench_viewer.cpp      side-by-side benchmark application
│   ├── tum_loopclose.cpp     symmetric loop closure (ORB vs TCG)
│   ├── tum_degrade.cpp       scattering degradation on measured depth
│   └── ...                   fusion, relocalize, tcg_density, plane_density
│
├── tests/                    C++ tests (236 cases)
│
├── python/
│   ├── wme/
│   │   ├── reference/        * numpy oracles the C++ is compared against
│   │   ├── localization/     ecda.py - the DirectAligner oracle
│   │   ├── geometry/         spa.py, planes.py
│   │   ├── eval/             ATE / RPE / Umeyama, TUM loader  <- scoring only
│   │   ├── graph/            pose graph, factors, photometric SLAM
│   │   ├── sim/ world/       synthetic scenes, world-model state/prediction
│   │   └── association/ calib/ planner/
│   ├── bindings/             pybind11 -> wme._core
│   ├── tools/                experiment and benchmark scripts (32)
│   └── tests/                Python tests (639 cases)
│
├── docs/06-results.md        * every measurement and every failure
├── results/bench/index.html  * left: classical / right: WME
└── .github/workflows/        linux, windows-msvc, sanitizers, python
```

---

## Datasets

Neither dataset is **included** in the repository (51 GB combined). Both are reproducible from
scripts.

### TUM RGB-D — indoor, handheld, measured depth

A Kinect structured-light sensor, so depth is a **measurement**. 16 sequences.

```bash
python python/tools/tum_fetch_all.py     # all 16, complete
python python/tools/check_datasets.py    # verify index matches disk
```

> `tum_fetch.py` defaults to a 9-second window. Run `--all` for the whole sequence — and note
> that the tool now trims `rgb.txt`/`depth.txt` to the files that actually exist. Leaving the full
> index in place while holding only part of the images made every tool skip the gaps silently, so
> a run labelled "1419 frames" had really processed 165.

> Intrinsics and distortion differ per freiburg group. Pinning one set makes the others produce
> **a plausible error rather than a failure**.

### KITTI odometry — outdoor, vehicle, stereo

There is no depth; **we have to produce it** — which is where the `StereoDepth` (OpenCV SGBM)
front-end first becomes necessary. Depth here is an **estimate**, not a measurement, and that
distinction is the whole of §25.21.

```bash
python python/tools/fetch_kitti.py       # 21.6 GB, resumable
build/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
```

The disparity search range is **derived** from the scene's nearest depth. Leave it at a default
and SGBM will not say "out of range" — it returns a **plausible wrong answer** (measured: depth
scaled by 2.42×).

| Item | Value |
|---|---|
| Sequences downloaded | 00–21 (ground truth for 00–10) |
| Converted and evaluated so far | 00, 04, 05, 07 |
| Resolution / focal / baseline | 1241×376 / 718.86 px / 0.537 m |
| Valid depth ratio (SGBM) | 67 – 74 % |

---

## Running it

### 1. Prerequisites

| Item | Version |
|---|---|
| Compiler | MSVC 2022 / GCC 11+ / Clang 14+ (C++20) |
| CMake | 3.24 or newer |
| OpenCV | 4.8+ — `core imgproc imgcodecs videoio calib3d highgui dnn features2d` |
| Python | 3.10+ with `numpy scipy pytest pybind11` |
| (optional) ONNX Runtime | 1.22 — for YOLO token masking |

> Omitting `features2d` from the OpenCV components fails only at link time.
> `cmake/WmeDependencies.cmake` documents that trap in a comment.

### 2. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. Test — start here

```bash
ctest --test-dir build --output-on-failure     # C++    236 cases
cd python && python -m pytest -q               # Python 639 cases
```

Much of the Python side is **C++ ↔ numpy differential testing**. Green here means two independent
implementations produced the same answer, and that is the only reason any number in this
repository is believable.

### 4. Benchmark → side-by-side viewer

```bash
python python/tools/bench_run.py               # run both systems and score them
python python/tools/bench_report.py            # -> results/bench/index.html
python python/tools/bench_export.py            # -> results/bench/viewer.tsv
build/tools/wme_bench_viewer                   # native application
```

Partial re-runs:

```bash
python python/tools/bench_run.py --only kitti --merge   # KITTI only, keep the rest
python python/tools/bench_run.py --skip-run             # rescore without re-estimating
```

Viewer controls: `SPACE` play · `,` `.` step · `N`/`P` sequence · `1`/`2` swap the model shown in
each panel · `A`/`D` orbit · `W`/`S` zoom · `F` screenshot · `Q` quit.

The viewer **does not compute ATE/RPE.** It displays what `bench_run.py` computed, because two
implementations of a metric is how the screen and the document start disagreeing.

### 5. A single sequence

```bash
# TUM
build/tools/wme_tum_odometry data/rgbd_dataset_freiburg1_xyz out.txt
build/tools/wme_tum_baseline data/rgbd_dataset_freiburg1_xyz orb.txt
python python/tools/tum_eval.py data/rgbd_dataset_freiburg1_xyz out.txt

# KITTI - depth ceiling and uncertainty coefficient come from the dataset
build/tools/wme_tum_odometry data/kitti_00 out.txt \
    --kf-dist 1.0 --depth-max 60 --depth-sigma-rel 7.8e-4
```

### 6. Other experiments

```bash
python python/tools/baseline_cv2.py       # third-party control (cv2.Odometry)
python python/tools/bench_degrade.py      # haze sweep
python python/tools/stereo_validate.py    # stereo depth against TUM's measured depth
python python/tools/loop_optimize.py      # pose-graph loop closure
```

---

<div align="center">

**Full results, failures and rejected hypotheses → [docs/06-results.md](../06-results.md)**

</div>
