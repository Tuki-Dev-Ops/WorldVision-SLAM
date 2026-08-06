# Implementation Roadmap and Current Status

WME is a multi-quarter engineering program, not a single deliverable. This document states
exactly what exists, what is designed, and what remains — so that nothing in this repository
looks more finished than it is.

## Status legend

- **DONE** — implemented, unit-tested, benchmarked
- **SPEC** — designed in docs, interface defined, implementation pending
- **TODO** — design not yet written

## Phase 1 — Foundation and perception  *(current)*

| Component | Status | Notes |
|---|---|---|
| `wme_core` types, `SE3`/`SO3` | **DONE** | Lie group ops with small-angle Taylor branches, adjoint identity tested |
| `Result<T>` / `Status` | **DONE** | typed errors + `Degraded` channel feeding the Confidence Engine |
| `Frame`, `ImagePyramid` | **DONE** | Scharr gradients, pyramid-consistent intrinsics |
| `ThreadPool` (work stealing) | **DONE** | `parallelFor` verified for exact single coverage |
| `ImageQualityEngine` | **DONE** | Immerkaer noise, structure-tensor blur, per-pixel weight map, lens-dirt accumulation. Scharr gain-32 normalisation added at first run — without it sharpness, blur extent and the info weight were all saturated (`06-results.md` §11.1) |
| `EnvironmentAnalyzer` | **DONE** | dark-channel haze, temporal-residual rain/snow separation, hysteresis labels, tier weights. Darkness is a noisy-OR (the weighted sum made `Dark` unreachable); texture poverty is noise-corrected (§11.1–11.2) |
| `WorldToken` | **DONE** | full representation as specified |
| `ConstellationIndex` (TCG) | **DONE** | inverted index + class-consistent max-clique + Kabsch + chi-square gate |
| `wme_env_probe` tool | **DONE** | live environment/tier-weight readout with CSV export |
| `wme_tum_odometry` tool | **DONE** | TUM RGB-D → ECDA → TUM-format trajectory; undistortion + constant-velocity prior. Scoring lives in Python on purpose |

## Phase 2 — Localization and token integration  *(in progress)*

| Component | Status | Notes |
|---|---|---|
| **ECDA** direct alignment (Tier 0) | **DONE** | LM coarse-to-fine, affine brightness, quality/dynamic-weighted residuals, marginalised information matrix, unit-free degeneracy spectrum, deterministic block reduction. First execution reworked three things: noise-referenced graduated Huber, per-block damping floor, and per-level projection onto the observable subspace. Measured cold-start convergence radius 0.18 m at 5 levels vs 0.02 m at 3 (`06-results.md` §11.4–11.7) |
| `solveAssignment` (Hungarian) | **DONE** | global optimal association; verified against brute force |
| `Detection` + class-wise NMS | **DONE** | deterministic tie-breaking |
| `ConfidenceEngine` | **DONE** | log-odds existence / identity / static beliefs, saturation bounds, reliability-scaled evidence |
| `TokenStore` | **DONE** | detection → token promotion, information-filter fusion, lifecycle FSM, static-mask export, merge. `Displaced` was a permanent-ghost state — excluded from association, so it stopped accruing absence evidence and froze forever above the retire threshold (§11.3) |
| `IYoloRuntime` interface | **DONE** | 두 백엔드가 전처리/디코딩(`YoloDecode`)을 공유해, 결과가 갈리면 추론 엔진 차이로 좁혀진다 |
| YOLO ONNX Runtime 백엔드 | **DONE** | ORT 1.22 + YOLO11n. CPU 60–150 ms/frame. OpenCV DNN 백엔드는 YOLO11 의 opset 22 / C2PSA 를 못 읽어 대안으로만 남는다 |
| YOLO TensorRT 백엔드 | **SPEC** | batched inference, CUDA stream, GPU NMS. 실시간 목표는 여기서 나온다 |
| `DepthEstimator` (stereo) | **SPEC** | SGM on GPU; mono falls back to inverse-depth filters |
| `PlaneExtractor` / **SPA** (Tier 2) | **SPEC** | region-growing on normals, gravity estimate |
| Factor graph + pose graph | **SPEC** | three-tier fusion per `02-correspondence-problem.md` §4 |
| Keyframe selection + BA | **SPEC** | sliding window + marginalisation |
| Drift recovery / relocalization loop | **SPEC** | TCG-driven, no map reset |

**Key risk in this phase:** the fusion weights `α_k(E)` are currently a hand-designed
calibration. They must be fit on real adverse-condition data before any accuracy claim is
publishable. Until then, treat them as a structural hypothesis, not a result.

**Second risk:** ECDA is verified only against synthetic planar homography warps. That
proves the Jacobians, the pyramid, the affine model, and the dynamic-mask benefit are
correct, but it does not prove behaviour on real sensor data with real depth noise. TUM-RGBD
is the next milestone.

## Phase 2b — Python layer  *(done)*

Built because no C++ toolchain was available to compile or run the engine. Rather than ship
unverified C++, the core algorithms were reimplemented in numpy and executed. This is
standard differential testing, and it immediately paid for itself.

| Component | Status | Notes |
|---|---|---|
| `wme.reference` — numpy oracle | **DONE** | SE3/Kabsch, Hungarian, TCG, confidence, tier weights |
| `wme.eval` — ATE / RPE / Umeyama | **DONE** | Sturm et al. (IROS 2012) definitions |
| `wme.eval.tum` — TUM-RGBD loader | **DONE** | rgb/depth/groundtruth time association |
| `wme.yolo` — detection bridge | **DONE** | decode + letterbox inverse + class-wise NMS in pure numpy; ONNX / ultralytics / scripted backends |
| `python/bindings` — pybind11 `_core` | **DONE** (code) | not compiled; `-DWME_BUILD_PYTHON=ON` |
| `tests/test_differential.py` | **DONE** | 41 C++ ↔ Python cross-checks, auto-skipped until `_core` is built |

**130 Python tests pass.** Running them found three defects that were present in the C++ and
would have survived a clean compile:

1. **Chirality used one gravity vector for two different frames.**
   `ConstellationIndex` compared heights in the query frame and the place frame against the
   same fixed `[0,0,-1]`. The query frame is rotated relative to the place, so its gravity is
   a different vector. Correct correspondences were being rejected — a 12-node exact match
   collapsed to an 8-node clique and returned a transform 0.148 m off *on noise-free data*.
   Fixed by storing gravity per `Place` and taking a query-frame gravity argument; chirality
   now applies only when both are known. Regression tests added on both sides.

2. **Log-odds saturated after two observations.**
   Per-observation evidence was ~3.3 nats against a ±4.0 clamp, so `existence_belief` maxed
   out almost immediately and the engine could no longer distinguish a 0.95-confidence
   detection from a 0.30 one, or a clear observation from a foggy one. Fixed with
   `evidence_gain = 0.25`, which is also the physically correct move: consecutive frames are
   strongly correlated, so treating each as an independent observation overcounts.

3. **`texture_poverty` and `camera_shake` were dead evidence channels.**
   Both were estimated by `EnvironmentAnalyzer` and never consumed. A textureless corridor —
   the canonical direct-method failure — produced `α₀ = 1.0`, full confidence in the exact
   estimator that cannot work there. Both now enter the photometric weight.

None of these would have been caught by compiling. (1) in particular produces plausible,
slightly-wrong poses — the failure mode that is hardest to find in a SLAM system.

## Phase 2c — Evaluation and simulation (M1)  *(done)*

Moved ahead of the design work deliberately. See
[05-research-program.md](05-research-program.md) §1 — you cannot design toward a capability
you cannot measure, and ATE measures none of WME's claims.

| Component | Status | Notes |
|---|---|---|
| `wme.eval.stats` | **DONE** | χ² cdf/ppf via incomplete gamma — no scipy, so evaluation runs in CI |
| `wme.eval.metrics` — NEES / ANEES | **DONE** | χ² consistency band; flags over- vs under-confidence |
| — calibration (ECE / MCE / Brier) | **DONE** | for existence beliefs |
| — degradation curves | **DONE** | AUC with an **absolute** failure penalty |
| — identity (IDS, frag, duplicate rate) | **DONE** | duplicate rate targets the loop-closure-merge claim |
| — change metrics (moved/removed/added) | **DONE** | typed, with latency and false-change rate |
| — prediction (ADE/FDE/NLL) | **DONE** | NLL is what catches confidently-wrong forecasts |
| `wme.sim` — world, trajectories, sensor | **DONE** | condition-dependent detection/box/depth/FA/confusion noise |
| `wme.sim` — scenarios | **DONE** | static, dynamic, revisit-with-changes, condition sweep, degradation burst |
| `tools/sim_report.py` | **DONE** | end-to-end sanity report |

**200 Python tests pass.** Writing them found one more real defect, this time in a metric:

4. **Degradation AUC penalised failure on a system-relative scale.** The failure penalty was
   derived from the system's own best non-failed value, so a system that performed well and
   then collapsed scored *better* than one that degraded gracefully — the exact inversion the
   metric exists to prevent. Fixed with an explicit absolute `failure_value` that all
   compared systems must share.

Measured behaviour of the sweep harness (haze 0 → 1, same world, same trajectory):

| level | severity | detection % | σ_z | α₀ | α₁ | α₂ | motion prior |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.00 | 94.5 | 0.183 | 1.000 | 1.000 | 0.400 | 0.164 |
| 0.4 | 0.36 | 79.9 | 0.379 | 0.660 | 0.889 | 0.502 | 0.338 |
| 0.8 | 0.72 | 66.9 | 0.569 | 0.320 | 0.739 | 0.604 | 0.523 |
| 1.0 | 0.90 | 58.0 | 0.665 | 0.150 | 0.627 | 0.655 | 0.626 |

α₀ collapses 6.7× while α₁ falls only 1.6×. That is the designed behaviour and it is now
*measured* rather than asserted — though note this only shows the weight schedule behaves as
intended, not that the schedule is correct. Correctness requires M3.

## Phase 2d — M3 calibration gate  *(run; partially passed)*

The decision point identified in [05-research-program.md](05-research-program.md) §5: can
`Σ_k(E)` be calibrated, and is the resulting uncertainty consistent? Run via
`python tools/m3_calibration.py`. Full analysis in
[04-unified-objective.md](04-unified-objective.md) §5.2.1.

| Component | Status |
|---|---|
| `wme.calib.optimize` — Nelder-Mead, no scipy | **DONE** |
| `wme.calib.noise` — Fixed / Scheduled / Calibrated models + MLE fitting | **DONE** |
| `wme.calib.estimator` — information-filter fusion, GT poses (isolates the measurement model) | **DONE** |
| `wme.calib.experiment` — train/test-on-unseen-conditions protocol | **DONE** |

**Result: passed, on the fourth attempt. The three failures were the substance.**

| configuration | ANEES/dof (unseen) | consistent | RMSE |
|---|---|---|---|
| fixed noise | 65 – 128 | 0 % | 0.129 m |
| hand-scheduled | 1.3 – 48.5 | 25 % | 0.129 m |
| calibrated, per-observation only | 38 – 58 | 0 % | 0.068 m |
| calibrated + systematic term | 1.9 – 2.8 | 0 % | 0.068 m |
| **joint extent + refit noise** | **0.85 – 1.27** | **100 %** | **0.037 m** |

The path there, in order:

1. **Per-observation parameters are recoverable, and recovering them alone is nearly
   worthless.** Observations are correlated through geometric modelling error, so fusion
   shrinks the reported covariance as `1/N` while that error does not shrink at all.
   Overconfidence grows *with the amount of evidence gathered*. This is very likely the
   mechanism behind SLAM's well-known overconfidence. Minimal reproduction in
   `test_unmodelled_bias_makes_fusion_overconfident`.
2. **Budgeting for the error removes 98 % of the overconfidence but not the cause**
   (ANEES 58 → 2). Maximum likelihood picked a size-dependent geometric form over constant
   and condition-dependent alternatives — the physically correct one — but an isotropic term
   cannot represent a view-dependent error.
3. **The gate itself was gameable and had to be fixed.** An intermediate configuration
   reported 100 % consistency with an RMSE of 4.2 m: it had simply inflated `σ_sys` to 2.5 m
   to cover a broken estimator. *Consistency alone is not a criterion* — an arbitrarily
   uncertain estimator always passes it. The gate now requires consistency **and** accuracy
   within 1.5× of the best configuration.
4. **The real fix was the measurement model, not the noise model.** A detection box centre is
   not the projection of a centroid; the bias is `≈ e_x·e_z/C_z`, and `e_z` is unobservable
   from a single view. So extent must be *estimated jointly with position* and then
   **marginalised** (Schur complement) so that not knowing the size properly inflates the
   position covariance. With that, `σ_sys` and `k_geom` both fit to **zero** — the error was
   removed rather than budgeted for.
5. **The noise model had to be refit under the corrected measurement model.** Parameters
   fitted under the biased model had absorbed the bias (`c_px = 8.55` against a true `2.0`),
   and became over-estimates once the model was fixed. After an EM-style refit:
   `c_px = 1.996` (true `2.00`), `g_px = 3.82` (true `4.00`), `c_d = 0.0056` (true `0.006`).

Consequences for the program:

- §5.2 of the objective document is revised twice: sensor reliability as a latent precision
  is necessary but not sufficient; error *correlation* matters, and correlated error is
  better removed than budgeted.
- **A noise model is only valid for the measurement model it was fitted under.** The two must
  be calibrated together.
- M3 is passed *under fixed ground-truth poses*. It must be re-run once the factor graph
  makes poses part of the estimate — that is a strictly harder problem.

## Phase 2e — Factor graph and M4  *(built; M4 nearly passed)*

The Phase 2 blocker: the layer that fuses tiers into one posterior. This is where
[04-unified-objective.md](04-unified-objective.md) §5.1 stops being a proposal.

| Component | Status | Notes |
|---|---|---|
| `wme.graph.variables` | **DONE** | manifold retraction; left perturbation, matching the C++ convention |
| `wme.graph.factors` | **DONE** | prior / between-pose / object-observation, Huber, numeric Jacobians by default |
| `wme.graph.graph` | **DONE** | LM on manifold, gauge fixing, marginal covariance, degeneracy spectrum |
| `wme.graph.slam` | **DONE** | joint poses + landmark positions + landmark extents |
| `tools/m4_slam.py` | **DONE** | M3 re-run with poses estimated rather than fixed |

Design points worth recording:

- **A tier is not a factor type.** ECDA and TCG both enter as `BetweenPoseFactor`; what
  differs is the magnitude of `Λ`. `BetweenPoseFactor.scaled(α)` is the whole of the
  environment adaptation at this layer — no branching, no mode switching.
- **Numeric Jacobians are the default.** Analytic-Jacobian errors produce plausible wrong
  answers and are the hardest SLAM bug class to find. Factors may override for speed.
- **Degeneracy is reported, not discovered.** `FactorGraph.degeneracy()` returns the
  unit-normalised eigen-spectrum and the observable DoF count; an unanchored pose graph
  correctly reports 6 unobservable directions instead of failing opaquely.

### M4 result — poses estimated (`python tools/m4_slam.py`)

| haze | ATE (odometry) | ATE (optimised) | landmark RMSE | pooled ANEES/dof | consistent |
|---|---|---|---|---|---|
| 0.00 | 0.297 m | 0.074 m | 0.072 m | 1.07 | ✅ |
| 0.25 | 0.297 m | 0.098 m | 0.108 m | 1.07 | ✅ |
| 0.50 | 0.297 m | 0.156 m | 0.164 m | 1.49 | ❌ mild |
| 0.75 | 0.297 m | 0.160 m | 0.179 m | 1.16 | ✅ |
| 1.00 | 0.297 m | 0.179 m | 0.236 m | 1.47 | ❌ mild |

Three observations.

1. **The graph does its job**: 40–75 % ATE reduction over integrated odometry, with the
   correction shrinking as conditions degrade — which is the correct behaviour, not a defect.
2. **Uncertainty stays close to calibrated with poses in the estimate.** Worst case is 1.25×
   overconfident in ANEES, i.e. ≈1.1× in σ. The starting point of this whole line of work was
   128× (≈11× in σ). M4 is not strictly passed, but the residual is at the level where
   resolving it requires careful treatment of estimator nonlinearity, not a modelling fix.
3. **The evaluation method had to be corrected before the result meant anything.** A single
   trial has ~14 landmarks; per-seed ANEES ranged 0.45–2.01 on *identical* settings. That
   spread is sample noise, and it is wide enough to make any single-trial verdict meaningless.
   Pooling across seeds was required. Stated generally: **a consistency test on a few dozen
   samples cannot distinguish a miscalibrated estimator from luck** — which is worth
   remembering, because published NEES plots are often single-run.

Remaining before M4 can be closed: a real ECDA photometric factor (the simulator renders no
pixels, so the between-pose factor is a stand-in), estimated rather than ground-truth data
association, and enough trials to resolve the residual 1.1×.

## Phase 2f — Data association and M5  *(built; partially validated)*

[04-unified-objective.md](04-unified-objective.md) §5.4 calls greedy-then-frozen association
"the deepest structural flaw" in SLAM. Every experiment up to this point sidestepped it by
using ground-truth association. `wme.association` implements the four strategies from that
section and measures them.

| Component | Status | Notes |
|---|---|---|
| `murty_k_best` | **DONE** | k-best assignment; verified against brute force on 3×3 |
| `GreedyTracker` / `HungarianTracker` | **DONE** | nearest-first vs. per-frame global optimum |
| `DeferredTracker` | **DONE** | commits only when the 1st/2nd margin is decisive |
| `MhtTracker` | **DONE** (implementation) | hypothesis-oriented, pruned to N-best |
| `tools/m5_association.py` | **DONE** | comparison across separation and condition |

### M5 result — crowded same-class objects, haze 0.5

| strategy | ID switches | duplicate rate | track inflation | deferred |
|---|---|---|---|---|
| Greedy | 19 | 0.60 | 1.10 | — |
| **Hungarian** | **7** | 0.30 | 1.00 | — |
| **Deferred** | **3** | 0.30 | 1.00 | 13.5 % of observations |
| MHT | 12 | 0.50 | 1.10 | — |

With well-separated objects of distinct classes, all strategies are identical and perfect —
correctly, since the assignment is then unique.

Confirmed:

1. **Global assignment beats greedy under genuine ambiguity** (19 → 7 switches). This
   validates the choice already made in `TokenStore`.
2. **Deferring beats committing** on identity (7 → 3) and pays for it by discarding 13.5 % of
   observations. The trade-off is real and now quantified rather than assumed.

### M5b — track-oriented MHT, and when multi-hypothesis inference actually helps

The first MHT implementation allocated track IDs *per hypothesis*, so a change of leading
hypothesis renamed every track and the identity metrics measured a reporting artefact. Rebuilt
track-oriented: track identity is **global and shared across hypotheses** — hypotheses differ in
*which detection attaches to which track*, not in what the tracks are. Two hypotheses that both
start a track from the same measurement now agree on its ID.

That made it possible to ask the real question, and the answer has a condition attached.

| scenario | Greedy | Hungarian | Deferred | MHT (online) | MHT (smoothed) |
|---|---|---|---|---|---|
| separated, distinct classes | 0 | 0 | 0 | 0 | 0 |
| crowded, **ambiguity never resolves** | 6 | **1** | 1 (16 deferred) | 10 | 14 |
| crowded, **ambiguity resolves later** | 27 | 37 | **6** (114 deferred) | 29 | **21** |

(ID switches; the last scenario approaches the objects so late observations are precise.)

4. **MHT is a smoother, not a filter, and this is measurable in both directions.**
   Retrospection improves it when later evidence exists (29 → 21) and *degrades* it when the
   ambiguity is permanent (10 → 14). Multi-hypothesis inference cannot manufacture information
   that the sensor never provided. The value of keeping hypotheses is entirely a function of
   whether the ambiguity is later resolvable — which is a property of the scene, not of the
   algorithm.
5. **MHT keeps the cleanest track set.** Track inflation 1.00 (exactly one track per object)
   against 1.10 for Hungarian and 1.20 for Greedy in the hard scenario. It does not create
   spurious tracks, because it never has to guess under uncertainty.
6. **MHT does not win on ID switches**, and should not be claimed to. Deferred remains the
   strongest identity strategy in every ambiguous scenario, at a cost that grows with the
   ambiguity — 40 % of observations discarded in the hardest case.
7. **Tractability at object scale is confirmed.** 100 frames × 20 hypotheses × k=4 runs in
   ~24 s of pure interpreted Python with numeric everything; the equivalent in the C++ engine
   is milliseconds. At keypoint scale this would be hopeless — which is the concrete payoff of
   the object-centric representation, and the part of §5.4 that does hold up.

**Revised §5.4 claim.** "Object-level MHT is tractable" is supported. "Object-level MHT gives
better association" is too strong: it gives a *cleaner track set* and *retrospectively better
history where evidence exists*, but a deferred-commitment filter beats it on identity while
discarding data. The publishable framing is the conditional one — that the benefit of
multi-hypothesis inference is predicted by whether ambiguity is later resolvable — because that
is what the experiment actually shows, in both directions.

Four things had to be fixed before any of these numbers meant anything, and all four are
worth recording because each silently produced plausible-looking output:

- **The "ambiguous" scenario was not ambiguous.** Objects were 0.70 m apart against a gate
  radius of ~0.35 m, so class and distance resolved every assignment uniquely. Every strategy
  scored identically and the experiment measured nothing. Ambiguity has to be constructed
  *relative to the gate*, not by intuition.
- **Track lifecycle was missing.** Without confirm/prune, each false alarm became a permanent
  track: inflation 2.7×, and real strategy differences were buried under junk. `TokenStore`
  already had this lifecycle; the experiment did not.
- **The systematic covariance floor from M3 was load-bearing.** Without it, track covariance
  shrinks as `1/N` while the modelling bias does not, the gate closes, valid measurements
  fall outside it, and association collapses. **Overconfidence is not merely a reporting
  problem — it actively breaks data association.** This is the sharpest practical consequence
  of the M3 finding.
- **A subtle bug in Murty's algorithm made MHT look terrible.** Sub-solutions that left a row
  unassigned scored artificially low (unassigned rows contribute zero cost), corrupting the
  k-best ordering. Fixing it moved MHT from 58 switches to 12. The lesson generalises: a
  k-best routine that returns *some* solutions in *roughly* increasing order will not announce
  its own failure.

## Phase 2g — Renderer and ECDA  *(done; first execution of Tier 0)*

The last stand-in removed. Until now the simulator produced only object-level detections, so
the photometric tier could not be tested and the factor graph substituted a relative-pose
factor for it. `wme.sim.render` produces pixels; `wme.localization.ecda` is the Python
reference for `src/localization/DirectAligner.cpp`, **which had never been executed** —
no C++ toolchain exists here, so this is the first time that algorithm has run at all.

| Component | Status | Notes |
|---|---|---|
| `wme.sim.render` | **DONE** | vectorised ray casting, textured quads + boxes, exact depth, object IDs |
| — degradation model | **DONE** | fog is **depth-dependent** (`I·e^{-βz} + A(1-e^{-βz})`), plus darkness, gain noise, directional blur, screen-fixed lens smudge |
| `wme.localization.ecda` | **DONE** | pyramid, grid-spread point selection, LM on SE(3) + affine brightness, affine marginalisation, degeneracy spectrum |

The algorithm works. Pose recovery from rendered pairs: translation error < 0.001 m, rotation
< 0.001 rad on well-conditioned motions; the affine model absorbs a 1.3× exposure change and
alignment fails without it; information scales exactly with `α₀(E)` and with photometric
variance; textureless scenes are reported degenerate.

Three findings, one of which refutes an assumption I had just added:

1. **Fog must be depth-dependent to be a valid test of `α₀(E)`.** Attenuating brightness by a
   global scalar produces a completely different failure mode from atmospheric scattering,
   where distant pixels wash out first. The renderer implements the physical model.
2. **`min_gradient` is scene-dependent, not a universal constant.** A close fronto-parallel
   surface yields max gradient 2.9 against the C++ default threshold of 6.0 — *zero* points
   selected, silent tracking loss. The information is there (lowering the threshold recovers
   200+ points); the fixed threshold is what fails. This is a real defect in the C++ config
   and needs an adaptive threshold.
3. **The selection grid must scale with the pyramid level — and misdiagnosing this cost the
   most time of anything in this project.**

   Rotation about the axis perpendicular to the optical axis drove ECDA into a local minimum
   that traded rotation for translation (0.17 m of spurious translation). Rank said 6. I added
   a condition-number diagnostic expecting it to catch the failure; it did not (730 for the
   failed case, 517 for the correct one). I recorded this as evidence of an intrinsic
   rotation/translation ambiguity that only the residual could detect.

   **That diagnosis was wrong.** The real cause was mundane: `grid_cell` was fixed across all
   pyramid levels, so at 320×240 with `grid_cell = 8`, level 3 (40×30) yields 15 candidate
   cells against `min_points = 60` — the level is skipped entirely. The coarse levels that
   provide the wide convergence basin were silently absent, so anything beyond a small motion
   failed. Scaling the cell as `grid_cell >> level` fixed it, and the "ambiguous" case now
   recovers to 0.02 m / 0.01 rad. The same defect was breaking photometric odometry in the
   SLAM integration below.

   This is a real defect in `src/localization/DirectAligner.cpp`, which has the same fixed
   grid. Regression test: `test_coarse_levels_must_scale_the_selection_grid`.

   What survives from the original claim, in weaker form: for genuinely diverged alignments
   the residual *and* the rank both flag the failure, but for this subtle one the rank did not.
   Gating on the residual is still necessary. The threshold, however, **cannot be absolute** —
   healthy alignments run rmse 0.2–3 in an empty room and 6–10 once objects add silhouette
   boundaries. `photometric_slam` gates relative to the sequence median instead.

## Phase 2h — Photometric SLAM integration  *(partial; one item open)*

`wme.graph.photometric_slam` replaces the stand-in relative-pose factor with real ECDA
odometry, and derives object detections from rendered masks so occlusion and truncation are
physically correct rather than modelled.

What works and is tested:

- **ECDA odometry on rendered sequences.** Median relative-pose error 0.0068 m against a true
  inter-frame motion of 0.0496 m; 29/29 alignments accepted. Degradation measurably worsens it.
- **Detections from rendered masks**, with truncation flags.
- **Relative-pose information calibration.** `J^T W J / σ²` treats neighbouring pixel residuals
  as independent when they are strongly correlated — the same error M3 found in object fusion,
  reappearing in the photometric layer. Calibrating against relative-pose NEES gives a scale
  of ≈0.18, i.e. the raw information was ~5.7× overconfident.

### Two more measurement-model mismatches, found by measuring rather than guessing

**The scenario did not exercise what it claimed to test.** With a short one-way sweep, the
photometric information exceeded the object information by 3300× (trace 3.3e8 vs 9.9e4), and
deleting every object factor changed the ATE by *zero* to four decimal places. That is not a
weighting bug — 40 frames of odometry accurate to 0.0068 m per step genuinely carries more
information than 8 objects localised to ~0.1 m. Objects can only help once drift has
accumulated and the same objects are re-observed. Switching to a there-and-back trajectory
made them contribute: **ATE 0.072 → 0.049 m, and removing the object factors returns it to
0.072.** Same class of error as M5's "the crowded scene was not crowded".

**Depth was measured on a different quantity than the model predicted.** Detections from
rendered masks report the median depth of the *visible surface*; the observation factor
predicted the *centroid* depth. Measured over 110 detections: bias = 0.778 · e_z, residual
σ = 0.021 m — a systematic 0.233 m offset, which is precisely the magnitude of the landmark
error that had been sitting there. Adding the offset term to the forward model:

| | landmark RMSE | pooled ANEES/dof |
|---|---|---|
| before | 0.26 m | 96 |
| after | 0.18 m | **17.5** |

This is the **third** instance of the same failure in this project — box centre vs. centroid
(M3), pixel residual correlation (Phase 2h), surface depth vs. centroid depth (here). The
pattern is worth stating as a rule: *whenever an estimator is overconfident or biased, check
first whether the forward model predicts the same statistic the sensor actually reports.*
Every time, it was cheaper to measure the discrepancy than to reason about it.

(Caveat recorded in the code: all objects in this scene share the same `e_z`, so
`0.778 · e_z` and a constant `0.233 m` are indistinguishable here. The proportional form was
chosen on geometric grounds and needs re-confirming on a scene with varied object sizes.)

### Current state of M4

| | ATE (odometry) | ATE (optimised) | improvement | landmark RMSE | ANEES/dof |
|---|---|---|---|---|---|
| clear | 0.072 m | 0.049 m | 32 % | 0.183 m | 17.5 |
| haze 0.4 | 0.135 m | 0.087 m | 36 % | 0.209 m | 5.3 |

The graph now demonstrably works — a third of the odometry drift is removed, and the objects
are what remove it. Uncertainty remains overconfident by 5–17×, down from 96×. The `xfail`
marker in `test_photometric_slam.py` stays until it is consistent.

M4 is **not closed**. The photometric tier is real and verified; the joint estimate improves
accuracy but does not yet report honest uncertainty. Remaining suspects, in order: pose error
propagating into landmarks without full accounting, the un-modelled 0.021 m residual in the
depth-offset relation, and the mask box being a *visible* silhouette rather than the full-box
projection when objects partially occlude one another.

## Phase 2i — Belief layer and M6  *(built; change detection measured)*

`wme.world` implements the part that distinguishes a world model from a map: a revisable
belief with versioned history, typed change detection, and forecasting kept structurally
separate from observation.

| Component | Status | Notes |
|---|---|---|
| `wme.world.state` | **DONE** | frozen snapshots, versioned; "what did the world look like 10 s ago" is a query, not a replay |
| `wme.world.change` | **DONE** | typed events: mis-estimated / moved / removed / added |
| `wme.world.prediction` | **DONE** | `Forecast` is a distinct type from `TokenBelief`, so a prediction *cannot* be written into an observation field |
| `wme.world.pipeline` | **DONE** | detections → tracks → beliefs → snapshots → changes |
| `tools/m6_change_detection.py` | **DONE** | measures P/R/latency and the false-change rate |

### M6 result

**The claim that mattered holds.** With no actual change and only the sensor degrading:

| haze | spurious events | objects | false-change rate |
|---|---|---|---|
| 0.0 | 0 | 72 | 0.000 |
| 0.3 | 0 | 72 | 0.000 |
| 0.6 | 0 | 72 | 0.000 |
| 0.9 | 0 | 72 | 0.000 |

A system that reports "the world changed" when fog rolls in cannot be attached to a planner.
The reason this holds is a single design choice: the move threshold is a Mahalanobis distance
against the *belief covariance*, not an absolute metric distance. When the sensor degrades the
covariance grows and the threshold widens with it. `test_absolute_threshold_would_fail_the_same_test`
is the control — the same data run against a fixed 0.15 m threshold produces false alarms in
volume.

**Real change detection works, with low recall.** Revisit scenario, 6 seeds:

| condition | kind | precision | recall |
|---|---|---|---|
| clear | moved | 1.00 | 0.67 |
| clear | removed | 1.00 | 0.50 |
| clear | added | 0.60 | 0.50 |
| haze 0.5 | moved | 1.00 | 0.17 |
| haze 0.5 | added | 1.00 | 0.33 |
| haze 0.5 | removed | 0.00 | 0.00 |

Precision is high and recall is low — the detector says little, and what it says is usually
right. That is the correct trade for this component: a false "the chair moved" sends a robot
to re-plan around nothing, while a missed change costs one more observation cycle. Degradation
collapses recall, which is honest rather than a defect: under haze 0.5 there is genuinely less
evidence, and the covariance-based gate reports that by staying silent.

Two structural facts came out of building this.

1. **"Moved" requires long-range re-identification, or it decomposes into "removed + added".**
   An object that shifts 1–2 m falls outside the association gate, so it becomes a new track;
   the old one dies. To a planner, "chair removed, different chair added" is a completely
   different statement from "chair moved". A conservative re-ID step (same class, similar
   extent, unique candidate, original track already dying) links them — 10–15 links per run.
   Without it the `moved` row above is empty by construction.
2. **The evaluation plumbing hid the result completely at first.** Detections carry *track*
   IDs; ground truth carries *simulator object* IDs. They never match, so the first run showed
   0/9 recall and 100 % spurious — indistinguishable from a broken detector. A majority-vote
   track→object mapping (scoring-only, never read by the estimator) fixed it. This is the same
   lesson as everywhere else in this project: before concluding the algorithm is wrong, check
   that the measurement is measuring what you think.

## Phase 2j — Memory Engine  *(done)*

`wme.world.memory` — history, consolidation, and forgetting. A memory, not a log; the three
things that make the difference are consolidation, a forgetting *policy*, and long-horizon
query.

**The central design decision: the unit of evidence is an episode, not a frame.**

A hundred frames from one pass are not a hundred independent observations. They share the
lighting, the viewpoint, and the error. Counting per frame makes an object glanced at once
for a long time more certain than an object confirmed across five separate visits — which is
plainly wrong, and it is the *fourth* appearance of the same correlation error in this
project (object fusion in M3, photometric residuals in Phase 2h, the pose chain in M4,
now memory). Here it is prevented by construction, and the test that proves it is:

| configuration | frames | visits | consolidated existence |
|---|---|---|---|
| one long stare | 200 | 1 | lower |
| five revisits | 25 | 5 | **higher** |

`test_visits_not_frames_determine_confidence` fails immediately if evidence is counted per
frame. `test_frame_count_alone_does_not_increase_confidence` pins the other side: 10 frames
and 100 frames in the *same* visit give identical confidence to 1e-9.

Two further properties, both tested:

- **Between-visit disagreement widens uncertainty**, even when each individual visit was
  precise. A tight estimate from one viewpoint must not erase the fact that other viewpoints
  disagreed — the covariance takes the larger of within-episode spread and between-episode
  spread.
- **Forgetting is by information value, not age.** An observation that confirmed what was
  already believed is worth less than one that surprised. Dropping surprising episodes first
  would erase exactly the record that the world changed. Adverse conditions extend retention,
  since sparse observation means less can be spared.

And the query contract the manifesto requires:

```
was_present(id, t)  ->  True   관측 중이었다
                        False  그 시각에는 관측되지 않았다
                        None   모른다 (기억의 범위 밖)
```

Conflating "absent" with "unknown" is the difference between a world model and a map.

## Phase 2k — World Graph  *(done)*

`wme.world.graph` — object relations. "cup on table", "bottle inside fridge", "chair near
couch", and connected regions.

**The design commitment: a relation is a posterior, not a boolean.** If a cup sits 0.12 m
above a tabletop and both positions have σ = 0.1 m, "on" is genuinely uncertain and the graph
must say so. Most scene-graph work asserts relations categorically, which is why such graphs
do not degrade when the perception does.

Three properties, each tested:

1. **Confidence is derived from the belief covariance.** The same geometry with σ = 0.02 m
   gives > 0.8 for `on`; with σ = 0.4 m it drops well below that.
   `test_ambiguous_gap_gives_intermediate_confidence` pins the middle: at the boundary the
   answer is neither 0 nor 1, which is the entire point of using a probability.
2. **Evidence is counted per visit, not per frame.** Fifty updates within one visit give a
   confidence identical to a single update, to 1e-12. This is the same correlation rule the
   Memory Engine enforces, and the fifth place in this project where it mattered.
3. **Relations are retractable.** Move the cup off the table while both remain visible and
   `on` decays below the assertion threshold. Crucially, the cup merely going *out of view*
   does **not** retract it — absence of observation is not evidence of absence, the same rule
   the Confidence Engine and Change Detector follow.

Supporting details: affordance is an explicit vocabulary table (adding a learned classifier
would violate the YOLO-only constraint); `near` uses surface-to-surface distance rather than
centre distance, so a large object is not artificially far from everything.

`WorldGraph.regions()` returns connected components of `near`. That is a **proxy** for rooms,
not room segmentation — real rooms need walls and planes from the Geometry Engine, which does
not exist yet. It is named `regions` rather than `rooms` for that reason.

## Phase 2l — Planner  *(done: risk + semantic search)*

`wme.planner` — the layer that actually consumes what the belief layer produces. Two
components, both chosen because they exercise the world model rather than being generic
robotics filler.

### Risk estimation

Risk is not a sensor, it is a derived quantity: position, covariance, velocity, forecast,
affordance, and coverage are all already there. Two commitments, each with a test that fails
if they are dropped.

1. **Uncertainty raises risk at the same mean distance.** An object 1.5 m away with σ = 0.01 m
   and one with σ = 0.8 m are not equally safe, and a planner on point estimates cannot tell
   them apart. Collision probability projects the covariance onto the approach direction, so
   uncertainty *along* the approach matters and uncertainty *across* it does not
   (`test_covariance_is_projected_onto_the_approach_direction`).
2. **Unobserved space is not free space.** Treating what you have not looked at as clear is
   the classic planner fatality. `Coverage` is built from camera poses (range + FOV) and the
   unobserved fraction enters the risk directly. Where no coverage is supplied the term is
   zero rather than invented.

Path risk takes the **worst** point, not the mean — averaging lets a short lethal segment
dissolve into a long safe one. `RiskAssessment` reports its decomposition and top
contributors, because a single scalar cannot tell a planner *why* something is dangerous.

### Semantic object search

"Find the cup." People do not sweep the floor; they look on tables. Candidates come from four
layers, and this is the only component that uses all of them at once:

| source | layer | prior |
|---|---|---|
| currently believed | `WorldState` | 0.95 |
| remembered position | `MemoryEngine` | 0.55, halving every 300 s |
| on a supporting surface | `WorldGraph` affordance | 0.25, reduced if already occupied |
| unobserved region | `Coverage` | 0.10 |

Ranked by `probability / (cost × risk)`. Risk enters the cost rather than as a separate term,
so "risky but certain" and "safe but hopeless" stay comparable.

Two defects were found by writing the tests:

- **Coverage scored too harshly.** A point 3 m away, dead centre, on a 6 m sensor scored 0.5 —
  the formula conflated *how well* something was observed with *whether* it was observed.
  Now it stays at 1 through the comfortable zone and falls off only near the limits.
- **Memory did not store the class name.** Resolving "is this remembered object a cup?"
  required looking the class ID up in the *current* snapshot — which fails in exactly the
  case that matters, searching for something that is no longer visible. `Episode` and
  `ObjectMemory` now carry `class_name`.

## Phase 2m — Tier 2 (SPA) and three-tier fusion  *(the central claim, measured)*

`wme.geometry` extracts planes from depth and aligns them (Tier 2). `wme.graph.fusion`
puts all three tiers into one graph. This is the first measurement of the claim the whole
architecture rests on:

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

### M7 result — ATE (m), 36 frames, 2 seeds

| haze | α₀ | α₁ | α₂ | T0 only | T0+T1 | T0+T2 | **T0+T1+T2** | gain vs T0 |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 0.40 | 0.098 | 0.090 | 0.099 | **0.091** | +6.8 % |
| 0.3 | 0.74 | 0.92 | 0.48 | 0.162 | 0.136 | 0.163 | **0.137** | +15.1 % |
| 0.6 | 0.49 | 0.82 | 0.55 | 0.261 | 0.192 | 0.252 | **0.190** | +27.4 % |
| 0.9 | 0.23 | 0.69 | 0.63 | 0.531 | 0.283 | 0.370 | **0.247** | **+53.6 %** |

**The improvement grows monotonically with degradation: 6.8 % → 15.1 % → 27.4 % → 53.6 %.**
That is precisely the claimed behaviour. In clear conditions the extra tiers add little — the
photometric term already has the information. As `α₀` collapses from 1.00 to 0.23 the other
tiers carry the estimate, and they do so through nothing but the information weights. There is
no branch on condition anywhere in the fusion code.

Secondary readings: TCG does most of the work (T0+T1 nearly matches the full system), and SPA
alone cannot determine a trajectory at all (ATE 1.28) — correctly, since planes leave several
DoF unobservable. Its value is as a *supplement*, which is what Tier 2 was designed to be.

### Three defects found, all silent

1. **SPA translation sign was inverted.** For a plane `n_r·p = d_r` and `q = Rp + t`, the
   correct relation is `n_cur·t = d_cur − d_ref`; the code had it reversed. Identity alignment
   gives 0 on both sides, so `test_identity_alignment_is_near_identity` passed happily and the
   rotation tests passed too. Only a case with actual translation exposes it — now
   `test_translation_sign_is_not_inverted` pins it explicitly.
2. **`lstsq(rcond=None)` amplified near-null directions.** Machine-precision cutoff keeps
   singular values that carry only noise, and SPA's information matrix is *designed* to be
   rank-deficient. Median translation error was 0.072 m with a **maximum of 40.9 m**. Truncating
   at the same threshold used for rank reporting fixed it, and it is what "do not move in
   unobservable directions" actually requires in code.
3. **LM damping provided no regularisation where `diag(H) ≈ 0`.** The floor was an absolute
   `1e-12` against diagonals of order `1e8`, so directions constrained only by rank-deficient
   factors were effectively undamped and the solution flew away — SPA-only diverged to ATE 32 m.
   The floor is now relative to the largest diagonal.

Plus one scenario defect of the now-familiar kind: **Tier 1 was silent because the scenario
starved it.** Objects spread over a wide arc put only ~3 in view at once against
`min_nodes = 4`, so 28 of 30 frames produced no constellation at all. Packing them into a
narrower arc raised the median to 6 and Tier 1 began contributing. The measurement was not
wrong; it was measuring nothing — the fourth time that has happened in this project.

### Also: gravity without an IMU

`dominant_gravity()` recovers gravity from the dominant horizontal plane (measured
`[0, 0.99, −0.15]` in camera frame). This closes a gap left open in Phase 1: chirality in
`ConstellationIndex` was disabled whenever gravity was unknown, and now it can be supplied
from geometry. Where no horizontal plane exists it returns `None` rather than inventing one.

## Phase 3 — Belief layer

| Component | Status |
|---|---|
| `ConfidenceEngine` (Bayesian existence / identity / static beliefs) | **DONE** |
| Token lifecycle + occlusion / out-of-view distinction | **DONE** (in `TokenStore`) |
| Long-term re-identification after Dormant | **SPEC** |
| `MemoryEngine` (history store, consolidation, environment-scaled forgetting) | **SPEC** |
| `PredictionEngine` (motion, visibility, occupancy forecast) | **SPEC** |
| `WorldState` copy-on-write snapshots + delta log | **SPEC** |

## Phase 4 — Semantics and mapping

| Component | Status |
|---|---|
| TSDF / voxel grid / occupancy, GPU integration | **SPEC** |
| Mesh generation, free-space extraction | **SPEC** |
| `SceneGraphBuilder`, spatial / support / container relation inference | **SPEC** |
| Room segmentation, `WorldGraph` | **SPEC** |

## Phase 5 — Application

| Component | Status |
|---|---|
| `Planner` (semantic navigation, object search, risk) | **SPEC** |
| ImGui debug visualiser | **SPEC** |
| Unity bridge protocol + client | **TODO** |
| Deterministic replay harness | **SPEC** |

## Phase 6 — Evaluation

| Item | Status |
|---|---|
| TUM-RGBD harness | **DONE** — `wme_tum_odometry` / `wme_tum_fusion` / `wme_tum_baseline`, 12 sequences, scoring split into python (`06-results.md` §12–§22) |
| Classical front-end baseline (ORB + PnP) | **DONE** — `tools/tum_baseline.cpp`, head-to-head on 12 sequences, 8–4 to WME (§22) |
| Side-by-side benchmark viewer | **DONE** — `python/tools/bench_run.py` → `bench_report.py` → `results/bench/index.html` |
| Per-tier ablations | **DONE** — 8 ablations × 5 sequences (§18), replayable offline (§21) |
| Loop closure + pose-graph back-end | **DONE** — `tools/tum_loopclose.cpp` + `loop_optimize.py`, symmetric back-end, ORB vs TCG verification (§24). ORB **+57 %**, TCG **+30.7 %**; WME full system 15.06 cm vs classical 20.67 cm |
| Adverse-condition set | **DONE (synthetic-on-real)** — `tools/tum_degrade.cpp` applies the scattering equation using TUM's *measured* depth; contrast tracks `exp(−βd)` to 2 % (§23) |
| Degradation sweep incl. `α_k(E)` slope | **DONE** — §1's slope confirmed in sign, deflated 53.6 % → 11 % (§23.4) |
| Third-party control (`cv2.Odometry`) | **DONE** — OpenCV 5's published dense RGB-D odometry, same data path; lands beside the hand-written ORB control on static scenes, so that control is not crippled (§22.4) |
| Structure/texture quartet | **DONE** — the four `fr3` sequences that vary structure and texture independently. WME 4.7× better with texture-only; ORB **diverges** on structure-without-texture (§22.4) |
| Baselines: **ORB-SLAM3, DSO, DROID-SLAM** | **TODO (measured blocker)** — ORB-SLAM3's `CMakeLists.txt` needs Pangolin + DBoW2 + g2o, uses POSIX-only flags and has **no WIN32/MSVC path**: a port, not a configure. Controls so far are a self-implemented ORB front-end and a third-party odometry |
| EuRoC / KITTI harness | **TODO (measured)** — EuRoC's data host times out from here (landing page resolves, `robotics.ethz.ch` does not). KITTI odometry is reachable at **21.6 GB** and is stereo, so it needs a depth front-end; `cv2.StereoSGBM` is available, making this work rather than a blocker |
| Oxford Robotcar night + rain | **TODO** — §23's degradation is synthetic-on-real; no naturally degraded data |

## Cross-cutting requirements not yet met

These apply to everything and are deliberately listed as outstanding:

1. **CI is written but has never run.** `.github/workflows/` holds four workflows — `linux.yml`
   (gcc + clang, minimal and full), `windows-msvc.yml`, `sanitizers.yml` (ASan/UBSan) and
   `python.yml`. **This directory is not a git repository**, so none of them has ever executed.
   The engine has been compiled by MSVC 19.44 only, and MSVC is far more permissive than gcc or
   clang about missing includes and two-phase name lookup — every diagnostic those two would
   raise is still unheard. Local state: **218 C++ tests, 525 Python tests, 1 xfail.** The SPA oracle added in
   §25.16 immediately found two defects in the Python reference — §7.1's scatter/complement
   inversion, never fixed there, and a doubly-applied weight — both now corrected.
   Writing a workflow is not standing up CI; the gap is `git init` and a remote.
2. **Allocation audited, partially.** Zero allocations on four of five hot paths;
   `TokenStore::integrate` remains at 27 (`06-results.md` §26). The `cv::Mat` counts are a lower
   bound — OpenCV's `AutoBuffer` and direct `malloc` are invisible to both counters.
3. **Determinism harness exists, narrowly.** Bit-identical across worker counts, repeated calls,
   live instances and processes for four components on one compiler, with mutation testing to
   prove the harness can fail. No recorded-log replay, no cross-machine golden file.
4. **`α_k(E)` fitted, and it does not help.** All three tiers now run on real data
   (`06-results.md` §18), and §21 fits the schedule offline from the recorded tier
   information: every leave-one-out fit lands at `α₁, α₂ ≤ 0.01` — the fit's answer is to
   stop fusing. The schedule was not the binding problem. A per-frame χ² consistency gate
   replaces it and beats Tier-0-only on 4 of 5 held-out sequences, but **not** by filling
   degenerate directions — selectivity stays at 1.0, so the architecture's complementarity
   mechanism remains unsupported outside simulation.
5. **TCG loop closure works; its limit is recall.** 3 accepted loops from 25 possible queries
   (ORB: 44 from 1292 proposed), edge accuracy 15.4 cm median (ORB: 2.14 cm) — and the pose graph
   improves ATE **+30.7 %**, so the WME full system beats the classical one 15.06 vs 20.67 cm.
   Object density binds (25 of 92 keyframes carry 4+ objects), and **both levers for it fail**:
   a temporal query window degrades edge accuracy 5× (§25.11), and a lower detection threshold
   *lowers* accepted loops 3 → 0 → 2 (§25.12). Node quality, not count, is the constraint.
   The alternative primitives are rejected too: plane centroids are fewer than objects (2.77 vs
   3.15) and move **1.5 m** between nearby viewpoints (§25.13); plane corners are stable by
   construction but there are **0.35 per keyframe**, since a corner needs three planes from a
   pool of 2.77 (§25.14). **Five candidates, all measured, all rejected — the constraint is
   landmark supply, not the algorithm** (§25.15): descriptors give ~1000 per frame, every WME
   primitive gives ~3, and a constellation needs 4.
   §24's earlier "TCG degrades ATE by 11.2 %" was two defects of mine — chirality silently off
   and the match transform inverted (§25.11).
6. **Real data covers five TUM windows, 9 s each, ECDA only** (`06-results.md` §12–13).
   Static scenes 1.0–2.5 cm ATE (11–23× the identity baseline); pure rotation 10.4 cm;
   **dynamic scenes 20.7 cm**. The sitting/walking pair isolates the dynamic cost at **20.6×**.
7. **동적 마스킹 미해결.** YOLO 백엔드가 붙어 `static_mask` 가 실제로 생산되지만
   (`06-results.md` §14), 클래스 기반 판정은 보행 3.96배 개선과 착석 15.7배 악화를
   맞바꾼다. 믿음 기반 대체는 8 초 창에서 증거가 충분히 쌓이지 않는다.
8. **Self-assessment: an independent channel exists, and nothing consumes it.** §13.3 measured
   every photometric signal blind on dynamic content and §23.3 measured the engine drifting to
   11.6× the do-nothing floor while reporting **zero** failed frames. §25 finds the reason —
   a photometric residual is bounded by the intensity range and an inlier ratio by [0,1], so
   neither can report an unbounded failure — and ships `depth_consistency`, which compares the
   estimate against the depth map alignment never reads and tracks that divergence at 10.4×
   where the photometric signals saturate at 4.3×. `align()` now returns a **degraded** result
   with a reliability when geometry disagrees, and the threshold is fitted across 10 sequences
   and 3 cameras (0.0057; the original 0.02 was inert — normal frames sit at 0.004). The honest
   limit: it is a good *relative* signal inside a sequence (lift 3.4–7.2) and a weak *absolute*
   one across them (Spearman +0.39 vs sequence ATE), fires on a third of clean frames at one
   cross-camera constant, and **does not separate at all on the fr2 camera**. Nothing downstream
   acts on the reliability yet.

## Build order and what is left

The highest-risk research claims were built first — `ConstellationIndex` (Tier 1) and
`ECDA` (Tier 0) — because if either failed, the architecture would need to change. Both are
implemented and tested against synthetic ground truth. What remains in Phase 2 is
conventional engineering plus the one piece that binds the tiers together:

```
  DONE                                    REMAINING
  ────                                    ─────────
  ImageQuality ─┐
  Environment  ─┼─► tier weights ──┐
                │                  │
  ECDA (T0) ────┼──────────────────┼──► FactorGraph ──► PoseGraph ──► BA
  TCG  (T1) ────┤                  │        ▲
  TokenStore ───┘                  │        │
       ▲                           └── SPA (T2) ◄── PlaneExtractor ◄── DepthEstimator
       │
  YoloRuntime backend (TensorRT/ORT)
```

The two blocking items are the **YOLO backend** (nothing produces real detections yet — all
token tests feed synthetic `DetectionSet`s) and the **factor graph** (the three tiers each
emit an information matrix, but nothing fuses them yet). Everything else in Phase 2 is
additive.
