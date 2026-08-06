# World Model Engine (WME)

A perception engine whose central object is a continuously evolving **World State**, not a
map. Localization is one estimator inside it, not its purpose.

WME asks five questions where SLAM asks one:

> What exists? · What changed? · What will change? · What is reliable? · Where am I?

**YOLO is the only perception network.** There is no ORB, SIFT, SURF, AKAZE, SuperPoint,
LoFTR, LightGlue, SAM, DINO, or CLIP anywhere in this codebase — no keypoint detector, no
descriptor, no descriptor matching. That constraint is not a limitation worked around; it
is what forces objects to become the primitive of the system, which is the whole thesis.

## Start here

| Document | What it covers |
|---|---|
| [docs/06-results.md](docs/06-results.md) | **Start here.** Every measured result, with its control and its limitation |
| [results/bench/index.html](results/bench/index.html) | **Run the comparison.** Side-by-side viewer: classical ORB+PnP on the left, WME on the right, 12 TUM sequences |
| [docs/00-manifesto.md](docs/00-manifesto.md) | Why a World State instead of a map; the six commitments; what would falsify the approach |
| [docs/02-correspondence-problem.md](docs/02-correspondence-problem.md) | **The core research argument.** How you estimate pose with no descriptors, and why that is better |
| [docs/04-unified-objective.md](docs/04-unified-objective.md) | Is "one objective function" the right unification? Verdict and the correct formulation |
| [docs/05-research-program.md](docs/05-research-program.md) | The plan to get from engine to publishable result |
| [docs/01-architecture.md](docs/01-architecture.md) | Layer map, execution model, concurrency discipline |
| [docs/03-roadmap.md](docs/03-roadmap.md) | Exactly what is implemented vs. designed vs. not started |

### The one result to read

Fusing all three correspondence tiers helps **more as conditions degrade** — 6.8 % → 15.1 % →
27.4 % → **53.6 %** ATE improvement over photometric-only as haze goes 0 → 0.9. There is no
branch on weather anywhere in the fusion code; the behaviour comes entirely from the
environment scaling the information matrices. Details and caveats in
[docs/06-results.md](docs/06-results.md) §1.

**That result is simulation. On real data the slope is real and the magnitude is not.**
Degraded real data was manufactured by applying the scattering equation to TUM frames using
their *measured* depth (§23), and across that sweep the fusion gain does grow with haze without
any weather branch — but **53.6 % becomes 11 %**, and it only turns positive after Tier 0 has
already diverged to 44–105 cm. At haze = 0 fusion is net *harmful* on four of five sequences
(§18), and fitting the weight schedule does not rescue it: the fit chooses to switch the other
tiers off (§21). Read §1, §18, §21 and §23 together or not at all.

## The idea in one page

Removing descriptors removes the entire classical correspondence pipeline. WME replaces it
with three tiers fused in one factor graph, weighted by an explicit model of the
environment:

| Tier | Name | Basis | Role |
|---|---|---|---|
| 0 | **ECDA** | photometric field alignment, dynamic pixels masked by YOLO | frame-rate relative pose |
| 1 | **TCG** | object constellations — class multiset + pairwise distance spectrum + chirality | relocalization, loop closure, unbounded baseline |
| 2 | **SPA** | planes, lines, gravity | degeneracy repair in textureless scenes |

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA + Λ_prior(E)
```

Adaptation to night, rain, fog, and motion blur is this single equation. Degrading a
sensing modality reduces the information it contributes; the estimator does the rest.
There is no `if (night)` branch anywhere.

**Token Constellation Geometry** is the primary original contribution. A place is not a bag
of visual words — it is a set of objects in a specific arrangement. That signature is
rigid-invariant by construction, survives darkness and fog (YOLO still fires on large
objects when texture is gone), and yields pose in closed form via Kabsch. It also returns
something a descriptor match never can: *this chair is the same chair*, which is a world-model
update rather than just a pose.

## Repository layout

```
include/wme/          public headers — contracts, thread-safety, complexity documented
  core/               Types, SE3, Result, Frame, ThreadPool, Assignment
  perception/         EnvironmentState, ImageQuality, Detection, their engines
  confidence/         ConfidenceEngine — Bayesian existence/identity/static beliefs
  token/              WorldToken, TokenStore, ConstellationIndex
  localization/       DirectAligner (ECDA, Tier 0)
src/                  implementations, mirroring include/
tests/                GoogleTest — analytic cases + property tests
benchmarks/           Google Benchmark — p99 latency, scaling curves
tools/                wme_env_probe — live environment readout
python/
  wme/reference/      numpy reimplementation of the core algorithms (test oracle)
  wme/eval/           ATE / RPE / Umeyama, TUM-RGBD loader
  wme/yolo.py         YOLO bridge — decode + NMS in pure numpy, ONNX/ultralytics backends
  bindings/           pybind11 module exposing the C++ engine as wme._core
  tests/              pytest, incl. C++ ↔ Python differential tests
docs/                 design rationale and research argument
cmake/                toolchain, dependencies, helpers
```

## Run the benchmark

```bash
cmake --build build/win                      # needs OpenCV on PATH at run time
cd python
python tools/bench_run.py                    # both systems, 12 sequences -> benchmark.json
python tools/bench_report.py                 # -> results/bench/index.html
```

`bench_run.py` drives `wme_tum_baseline` (ORB + descriptor matching + RANSAC PnP) and
`wme_tum_odometry` (WME) over the same frames with the same intrinsics, undistortion, depth
scale and keyframe rule, then scores both. Estimation is C++, scoring is Python, deliberately
split. Over 16 TUM sequences against the ORB control the tally is 12–4; against **the better of
two independent controls** — adding `cv2.Odometry`, OpenCV's published dense RGB-D odometry,
which nobody here wrote — it is **9–6** ([docs/06-results.md](docs/06-results.md) §22.4). A
second control moved the verdict without a single WME number changing; the count of controls is
itself an experimental parameter.

What survives both: WME wins **decisively on dynamic content** (6.48 cm vs 19.34 / 52.89 on
`walking_xyz`) and on long traverses, and loses on clean short static scenes and on textureless
structure. Two sequences are unscorable because one system diverged or produced nothing —
winning against a failure is not winning (§22.5).

**Then a second dataset family arrived and the tally moved.** A stereo depth front-end
(`StereoDepth`, OpenCV SGBM) and `wme_kitti_convert` let the same two binaries run on KITTI
odometry. Over four sequences (394–589 m each) it came out **2–2**, not 9–6
([docs/06-results.md](docs/06-results.md) §25.20) — the indoor result was not a property of the
algorithms, it was a property of TUM.

The gap KITTI exposed was concrete: ECDA treated stereo depth as exact, so a 60 m point with
±4 m of error was weighted like a 6 m point with ±4 cm. Moving that uncertainty into the residual
variance — one coefficient, `σ_Z = c·Z²`, **derived** from `c = σ_d/(f·B)` rather than tuned —
takes KITTI to **4–0** (§25.21). It is off by default (`depth_sigma_rel = 0`), it does nothing
indoors where `Z ≤ 8 m`, and it makes the highway sequence 2.5× worse because that scene has no
near structure at all.

```bash
python tools/fetch_kitti.py                                   # 21.6 GB, resumable
build/win/tools/wme_kitti_convert data/kitti/dataset 00 data/kitti_00 --stride 2
build/win/tools/wme_tum_odometry data/kitti_00 out.txt --kf-dist 1.0
```

```bash
python tools/baseline_cv2.py                 # third-party control, same data path
```

```bash
# degradation sweep — scattering applied to real frames using real depth
python tools/bench_degrade.py                # 23: neither system degrades gracefully

# loop closure — same back-end, ORB vs TCG place recognition
python tools/loop_optimize.py <seq> <odom.txt> <edges.csv> <out.txt>   # 24
```

**What the whole comparison says.** WME's descriptor-free front-end is the real result — 2.2×
better than the classical front-end over a full 45 s traverse. What sits on top of it is not yet
earning its place: three-tier fusion is net harmful except where everything has already failed
(§23.4), and the degradation robustness that motivated dropping descriptors does not appear
(§23.3). The object tier *does* close loops (+30.7 %), but finds 15× fewer than the descriptor
control and locates them 7× less precisely (§24.2).

**Does the engine know when it is wrong?** Not from the photometric channel — a residual is
bounded by the intensity range and an inlier ratio by [0,1], so neither can report a failure
that grows without limit. `depth_consistency` compares the estimate against the depth map
alignment never reads, and tracks a divergence of 11.6× the do-nothing floor at **10.4×** where
the photometric signals saturate at 4.3× (§25). `align()` now degrades its own result on
geometric inconsistency, with a threshold fitted across 3 cameras.

**Acting on it by reweighting fusion makes things worse on 8 of 15 configurations** (§25.7):
down-weighting Tier 0 does not create accuracy, it hands weight to tiers that are 3–15× less
accurate. Acting on it by **replacing the keyframe** — which needs no second estimator — is net
positive: 4 better, 2 worse, 2 tied, helping on the four hardest sequences and costing on the two
easiest (§25.8). Adding map-anchored **relocalization** on top reaches 15.72 cm on
`fr3_walking_xyz` from a 20.72 cm base — 24 % for the whole chain.

**With loop closure, the WME full system reaches 15.06 cm against the classical 20.67 cm**
(§24.1) — though that lead comes from the front-end: ORB's loop closure finds 44 loops to TCG's
3 and locates them 7× more precisely. §24 originally reported TCG *degrading* ATE; that was two
defects of mine (chirality silently disabled, match transform inverted) and is corrected in
§25.11.

**Under haze, relocalization succeeds 0 times in 358 attempts** (§25.9): it matches descriptors
against map keyframes captured in the same haze, so both sides of the comparison are degraded by
the same cause. Detection and recovery are separate problems — an independent channel for
*detecting* failure does not give you one for *recovering* from it. The only recovery path in
this architecture that does not live in the photometric channel is the object constellation, and
§24.2 measures that one as unable to relocalize indoors.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_constellation
```

Requires a C++20 compiler, CMake ≥ 3.24, OpenCV ≥ 4.8. Eigen, spdlog, GoogleTest, and
Google Benchmark are fetched automatically if not found. CUDA, TensorRT, ONNX Runtime,
Open3D, and PCL are optional and off by default.

> **The C++ has not been compiled.** The machine it was authored on has no C++ toolchain
> (`cmake`, `cl`, `g++`, `nvcc` all absent). Expect to fix compile errors on first build.
> Standing up CI is the first task in
> [the roadmap](docs/03-roadmap.md#cross-cutting-requirements-not-yet-met).

## Python layer — and why it exists

Since the C++ could not be run, the core algorithms were reimplemented in numpy and
executed. `wme.reference` is not a convenience wrapper; it is a **differential-testing
oracle**. Two independent implementations of the same algorithm must agree on the same
input — if they disagree, one is wrong, and a third source (an analytic answer or synthetic
ground truth) decides which.

```bash
cd python
pip install -e ".[dev]"
pytest -q                      # 130 tests, no C++ needed

# build the bindings to also run the 41 C++ ↔ Python differential tests
cmake -S .. -B ../build -DWME_BUILD_PYTHON=ON && cmake --build ../build
pytest -q
```

This immediately found **three real defects in the C++** that a clean compile would not have
caught — most importantly, chirality checking that measured heights in two differently
rotated frames against the same gravity vector, which silently returned poses ~0.15 m off on
noise-free data. Full write-up in [docs/03-roadmap.md](docs/03-roadmap.md#phase-2b--python-layer-done).

The layer also carries the evaluation harness (ATE/RPE, TUM-RGBD loader) and the YOLO
bridge — decoding, letterbox inversion, and class-wise NMS in pure numpy, so detections can
be produced before the TensorRT backend exists.

## Try the environment engine

```bash
./build/tools/wme_env_probe /path/to/video.mp4 --show --csv env.csv
./build/tools/wme_env_probe 0 --show          # webcam
```

Prints, per frame: image quality score, noise σ, blur extent, the environment labels, and
the live tier weights `α₀ α₁ α₂`. Point it at a dark or hazy clip and watch the photometric
weight collapse while the constellation weight holds — that is the adaptation mechanism,
visible directly.

## Current state

**Implemented and tested:** core + Lie group math, thread pool, image quality, environment
analyzer, **ECDA** (Tier 0 direct alignment), **TCG** (Tier 1 constellation index), Hungarian
association, confidence engine, and the token store with its full lifecycle.

**Not yet built:** the YOLO backend (all token tests feed synthetic detections), the factor
graph that fuses the three tiers, SPA (Tier 2), dense geometry, semantics, memory,
prediction, planner, and Unity.

Two things are deliberately *not* claimed:

1. **ECDA is verified only on synthetic planar warps.** That proves the Jacobians, pyramid,
   affine brightness model, and dynamic-mask benefit are correct. It does not prove
   behaviour on real depth noise. TUM-RGBD is the next milestone.
2. **The fusion weights `α_k(E)` have now been fitted on real data, and fitting them does
   not help.** Every leave-one-out fit lands at `α₁, α₂ ≤ 0.01` — the fit chooses to stop
   fusing ([docs/06-results.md](docs/06-results.md) §21). What repairs fusion instead is a
   one-parameter χ² consistency gate, which beats Tier-0-only on 4 of 5 held-out sequences —
   but by reducing the *signed* component of the error, not by filling degenerate directions.
   The complementarity mechanism this architecture claims remains unsupported outside
   simulation.

[docs/03-roadmap.md](docs/03-roadmap.md) has the per-component status table and the full list
of what is not verified.

Code comments are in Korean; documentation is in English.
