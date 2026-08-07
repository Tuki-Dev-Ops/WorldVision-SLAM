# Measured Results

Every claim in this document has a number behind it and a control or ablation that would
have exposed it as false. Claims without measurements are in
[03-roadmap.md](03-roadmap.md) and are marked as such there.

**Scope, stated once and applying throughout.** §1–§10 come from a synthetic harness
(`wme.sim`) with ground-truth camera poses. **§11 onward is measured in compiled C++**, and
§12 onward on **real TUM RGB-D data** — thirteen sequences across three cameras, all of them
short windows (5.5–40 s) from one dataset family. No published baseline (ORB-SLAM3, DSO,
DROID-SLAM) has been run, so every comparison here is against ablations of this system and a
do-nothing floor, never against the field.

**Read §1 together with §18 and §21.** §1 is the architecture's headline claim, measured in
simulation. §18 is the same claim measured in C++ on real data, where it does not reproduce at
the one operating point real data can test. §21 then establishes *why*: the failure is not the
weighting — a fitted `α_k` chooses to switch the tiers off — but a robustness failure that a
one-parameter consistency gate mostly repairs, though not by the mechanism the architecture
claims. Neither section supersedes §1; the gap between simulation and sensor remains the most
important open question in this document.

---

## 1. The headline: three-tier fusion under degradation

The architecture rests on one equation
([02-correspondence-problem.md](02-correspondence-problem.md) §4):

```
Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA
```

The claim is not that fusion helps. It is that **fusion helps *more* as conditions degrade**,
because the tiers fail differently and the information weights reallocate automatically.

> **This section is simulation. See §18 for the same experiment in C++ on real data.** The
> haze = 0 row below predicts +6.8 %; on five real sequences at haze = 0, fusion is net *harmful*
> on four of them. The degradation slope — the actual claim — remains untested, because TUM has
> no degradation to sweep.

ATE in metres, 36 frames, 2 seeds, `tools/m7_fusion.py`:

| haze | α₀ | α₁ | α₂ | T0 only | T0+T1 | T0+T2 | **T0+T1+T2** | gain |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 0.40 | 0.098 | 0.090 | 0.099 | **0.091** | +6.8 % |
| 0.3 | 0.74 | 0.92 | 0.48 | 0.162 | 0.136 | 0.163 | **0.137** | +15.1 % |
| 0.6 | 0.49 | 0.82 | 0.55 | 0.261 | 0.192 | 0.252 | **0.190** | +27.4 % |
| 0.9 | 0.23 | 0.69 | 0.63 | 0.531 | 0.283 | 0.370 | **0.247** | **+53.6 %** |

Monotone: 6.8 → 15.1 → 27.4 → 53.6 %. There is no conditional on weather anywhere in the
fusion code; the behaviour comes entirely from `α_k(E)` scaling the information matrices.

**Honest reading.** TCG carries most of the benefit — `T0+T1` nearly matches the full system.
SPA alone cannot determine a trajectory (ATE 1.28), correctly, since planes leave several DoF
unobservable; it functions as a supplement, which is what Tier 2 was designed to be. And the
`α_k(E)` schedule is still hand-designed, so this measures that *the mechanism works*, not
that the schedule is optimal.

---

## 2. Descriptor-free relocalization (TCG)

Places are object constellations, not bags of visual words. Signature: class multiset +
log-binned pairwise distance spectrum + chirality, matched by class-consistent maximum clique,
pose by Kabsch.

| property | result |
|---|---|
| exact-match recovery | < 1e-6 m |
| 8 cm observation noise | < 0.30 m translation, < 0.10 rad |
| 50 % partial observation | < 0.25 m |
| 5 spurious detections injected | < 0.35 m |
| 300 distractor places | correct place retrieved |
| perceptually aliased corridors | **rejected** (returns nothing rather than guessing) |

The rejection row matters as much as the recall rows. Two identical corridors produce two
candidates with near-equal scores; the system declines to choose. Guessing there is precisely
how loop closure corrupts maps.

**Limitation, by construction.** Needs ≥ 4 well-localised objects. Fails in an empty corridor.
That is the reason Tiers 0 and 2 exist.

---

## 3. Calibrated uncertainty (M3)

Can `Σ_k(E)` be fitted, and is the resulting uncertainty honest? Train on some conditions,
evaluate on **unseen** ones. `tools/m3_calibration.py`.

| configuration | ANEES/dof | consistent | RMSE |
|---|---|---|---|
| fixed noise | 65 – 128 | 0 % | 0.129 m |
| hand-scheduled | 1.3 – 48.5 | 25 % | 0.129 m |
| calibrated, per-observation | 38 – 58 | 0 % | 0.068 m |
| calibrated + systematic term | 1.9 – 2.8 | 0 % | 0.068 m |
| **joint extent + refit noise** | **0.85 – 1.27** | **100 %** | **0.037 m** |

Recovered parameters against simulator truth: `c_px = 1.996` (true 2.00), `g_px = 3.82`
(true 4.00), `c_d = 0.0056` (true 0.006). The systematic term fitted to **zero** — the error
was removed by fixing the measurement model, not covered by widening the covariance.

The hand-scheduled row is worth reading closely: 25 % "consistent" is not partial success. It
is wildly overconfident at low severity, passes through correctness at one point, and would be
underconfident beyond. A scheduled weight is a curve *crossing* the truth, not tracking it.

---

## 4. Data association (M5)

Crowded same-class objects, haze 0.5, `tools/m5_association.py`:

| strategy | ID switches | duplicate rate | track inflation | deferred |
|---|---|---|---|---|
| Greedy | 19 | 0.60 | 1.10 | — |
| Hungarian | 7 | 0.30 | 1.00 | — |
| **Deferred** | **3** | 0.30 | 1.00 | 13.5 % of observations |
| MHT (online) | 12 | 0.50 | 1.10 | — |

With well-separated objects of distinct classes every strategy scores identically — correctly,
since the assignment is then unique.

**The MHT result has a condition attached, and the condition is the finding.** Retrospective
smoothing improves MHT when later evidence exists (29 → 21 switches on an approaching
trajectory) and *degrades* it when ambiguity is permanent (10 → 14). Multi-hypothesis inference
cannot manufacture information the sensor never provided. Its value is predicted by whether
the ambiguity is later resolvable — a property of the scene, not of the algorithm.

Tractability at object scale is confirmed: 100 frames × 20 hypotheses × k = 4 in ~24 s of
interpreted Python. At keypoint scale this would be hopeless, which is the concrete payoff of
the object-centric representation.

---

## 5. Change detection (M6)

"What changed?" is one of the five defining questions. `tools/m6_change_detection.py`.

**The claim that matters — no change, sensor degrading only:**

| haze | spurious events | objects | false-change rate |
|---|---|---|---|
| 0.0 / 0.3 / 0.6 / 0.9 | 0 | 72 | **0.000** |

A system that reports "the world changed" when fog rolls in cannot be attached to a planner.
This holds because of one design choice: the move threshold is Mahalanobis against the *belief
covariance*, not an absolute distance. `test_absolute_threshold_would_fail_the_same_test` is
the control — the same data against a fixed 0.15 m threshold produces false alarms in volume.

**Real changes, revisit scenario, 6 seeds:**

| condition | kind | precision | recall |
|---|---|---|---|
| clear | moved | 1.00 | 0.67 |
| clear | removed | 1.00 | 0.50 |
| clear | added | 0.60 | 0.50 |
| haze 0.5 | moved | 1.00 | 0.17 |
| haze 0.5 | removed | 0.00 | 0.00 |

High precision, low recall: the detector says little and what it says is usually right. That is
the correct trade — a false "the chair moved" sends a robot to re-plan around nothing, a missed
change costs one observation cycle. Recall collapsing under haze is honest, not a defect: there
is genuinely less evidence, and the covariance gate reports that by staying silent.

**Structural finding.** "Moved" requires long-range re-identification or it decomposes into
"removed + added" — an object shifting 1–2 m falls outside the association gate and becomes a
new track. To a planner those are entirely different statements.

---

## 6. Tier 0 (ECDA) in isolation

Direct alignment on rendered image pairs, `wme.localization.ecda`:

| case | result |
|---|---|
| well-conditioned motions | < 0.001 m, < 0.001 rad |
| 1.3× exposure change with affine model | recovers pose; `a` estimated ≈ 1.3 |
| same, affine disabled | fails |
| textureless scene | reported degenerate |
| information scaling | exactly proportional to `α₀` and to `1/σ²` |
| fog / motion blur | information trace falls |

On rendered sequences the median relative-pose error is 0.0068 m against a true inter-frame
motion of 0.0496 m.

---

## 7. Tier 2 (SPA) and rank reporting

> **Corrected.** The rotation ranks originally published here were wrong — the Python reference
> used the wrong matrix for rotational information. See §7.1. The table below is the corrected
> version, produced by the C++ port and confirmed by finite differences.

| scene | planes | rotation rank | translation rank | observable DoF |
|---|---|---|---|---|
| room, corner viewpoint | 3 | 3/3 | 3/3 | **6/6** |
| room, frontal viewpoint | 3 | 3/3 | 2/3 | 5/6 |
| corridor | 3 | 3/3 | 2/3 | 5/6 |
| single plane | 1 | 2/3 | 1/3 | 3/6 |

In a corridor the along-axis translation is genuinely unobservable from planes, and the system
says so rather than reporting a confident wrong number. `unobservableDirections()` names the
axis so Tier 0 can be asked to fill it — mutual gap-filling requires being able to state where
the gap is. Measured: the one weak corridor direction is > 99 % translational and > 0.9 aligned
with the corridor axis.

### 7.1 The rotation information block was the orthogonal complement of the truth

`spa.py` computed rotational information as `Σ w·nnᵀ` (the normal scatter matrix). For the
residual `n_cur − R·n_ref` under this project's left-perturbation convention, `J = [n]ₓ`, so the
Gauss-Newton information is

```
JᵀJ = −[n]ₓ[n]ₓ = (nᵀn)I − nnᵀ = I − nnᵀ      (unit n)
```

`nnᵀ` and `I − nnᵀ` are orthogonal projectors onto complementary subspaces. The shipped code
therefore claimed rotational information along precisely the axis that is *unconstrained*, and
none along the axes that are. Finite differences on two non-parallel normals, measuring the
actual weighted residual increase for a 10⁻⁵ rad rotation about each axis:

| axis | measured Δcost | `Σ nnᵀ` predicts | `Σ (I − nnᵀ)` predicts |
|---|---|---|---|
| x | 1.910 | 0.090 | **1.910** |
| y | 1.000 | 1.000 | **1.000** |
| z | 1.090 | 0.910 | **1.090** |

Off by 21× on the x axis and moving the wrong way. Rank likewise: two non-parallel planes
determine `R` uniquely (rank 3), `scatter` reports 2.

Three reasons this survived:
1. **The pose estimate is unaffected.** Kabsch is computed separately, so every accuracy test
   passed. Only the *self-assessment* was wrong.
2. **The rank test still produced plausible numbers** — 2/3 and 4/6 read like an honest
   degeneracy report rather than a defect.
3. **`unobservable_directions()` returned coordinate axes**, not eigenvectors
   (`np.eye(6)[:, weak]`), and passed its test only because the corridor's null direction
   happens to be near-axis-aligned.

This is the most consequential defect in the document so far, because it inverts the one output
the architecture depends on: `weakest_direction` exists so Tier 0 can be asked to fill Tier 2's
gap. Pointed at the complement, it asks Tier 0 to fill the axis Tier 2 already constrains, and
leaves the actual gap unfilled. **A subsystem whose accuracy is right and whose self-assessment
is inverted is more dangerous than one that simply fails** — every consumer downstream trusts it.

Gravity is recovered from the dominant horizontal plane (`[0, 0.99, −0.15]`), which supplies
the vector `ConstellationIndex` needs for chirality when no IMU is present. Where no horizontal
plane exists it returns `None`.

---

## 8. Memory: episodes, not frames

A hundred frames from one pass are not a hundred independent observations.

| configuration | frames | visits | consolidated existence |
|---|---|---|---|
| one long stare | 200 | 1 | lower |
| five revisits | 25 | 5 | **higher** |

10 frames and 100 frames within the *same* visit give identical confidence to 1e-9. Between-
visit disagreement widens uncertainty even when each visit was individually precise. Forgetting
is by information value, not age — dropping surprising episodes first would erase exactly the
record that the world changed.

---

## 9. Planner: uncertainty and the unknown

Two commitments, each with a test that fails if dropped:

- **Uncertainty raises risk at the same mean distance.** An object 1.5 m away at σ = 0.01 m and
  one at σ = 0.8 m are not equally safe. Covariance is projected onto the approach direction,
  so uncertainty *along* the approach counts and uncertainty *across* it does not.
- **Unobserved space is not free space.** Coverage is built from camera poses; the unobserved
  fraction enters risk directly. Where no coverage is supplied the term is zero rather than
  invented.

Path risk takes the worst point, not the mean — averaging dissolves a short lethal segment into
a long safe one.

---

## 10. Methodological findings

These are the most transferable results in the project, and none of them were planned.

### 10.1 The same correlation error appeared five times

| # | where | symptom |
|---|---|---|
| 1 | object fusion (M3) | information filter over correlated observations → 58× overconfident |
| 2 | photometric residuals | neighbouring pixels treated as independent → 5.7× |
| 3 | pose chain (M4) | relative-pose calibration does not make absolute covariance consistent |
| 4 | memory | frame-counted evidence would make one long stare beat five revisits |
| 5 | world graph | repeated frames in one visit would confirm a relation repeatedly |

The first three were found after the fact; the last two were prevented by construction because
the pattern had become recognisable. **Rule: before counting N observations as N pieces of
evidence, ask what they share.**

### 10.2 Four times, the model predicted a different quantity than the sensor reported

- box centre vs. projected centroid (bias ≈ `e_x·e_z/C_z`)
- median *visible-surface* depth vs. centroid depth (bias = 0.778 · half-extent, measured over
  110 detections)
- correlated pixel residuals vs. independent ones
- that same 0.778, carried into C++ where the sampling region differs — §11.8

Each presented as an overconfident or biased estimator, and each was cheaper to *measure* than
to reason about. **Rule: when an estimator is biased or overconfident, first check that the
forward model predicts the same statistic the sensor actually reports.**

### 10.3 Consistency alone is a gameable criterion

One intermediate configuration reported 100 % NEES consistency with an RMSE of 4.2 m — it had
inflated `σ_sys` to 2.5 m to cover a diverging estimator. Any sufficiently uncertain estimator
is "consistent". A calibration gate must require consistency **and** accuracy, or it selects for
uselessness. The failure was silent: the numbers looked like success.

### 10.4 Four experiments measured nothing before they measured something

| experiment | why it measured nothing |
|---|---|
| association strategies | objects 0.70 m apart against a 0.35 m gate — assignment was unique |
| object factors in the graph | photometric information exceeded object information 3300× |
| change detection | detections used track IDs, ground truth used object IDs |
| Tier 1 in fusion | ~3 objects in view against `min_nodes = 4` — 28 of 30 frames silent |
| pyramid depth (§11.7) | error at one motion magnitude, where every depth had already failed |

In every case the output was plausible. **Rule: before concluding an algorithm is wrong, verify
the measurement discriminates — construct a deliberately bad input and confirm it is flagged.**

A sixth was found later and is the worst of them: the entire differential test suite reported
green while executing nothing at all (§19). The failure mode there is not a non-discriminating
measurement but a *missing* one that looks identical in the summary line.

The pyramid case is the sharpest: measured by error at 0.18 m, depth 5 looked *harmful*
(0.51 m vs 0.28 m). Measured by convergence radius — the quantity the pyramid actually exists to
change — depth 5 doubles it. Both numbers came from the same code on the same day.

### 10.5 Failures that no diagnostic caught except the residual

ECDA's coarse pyramid levels were being skipped entirely (`grid_cell` was level-independent, so
a 40×30 level yielded 15 candidate cells against `min_points = 60`). Rank said 6. A
condition-number diagnostic, added specifically to catch it, did not discriminate (730 vs 517).
Only the photometric residual did (10.08 vs 0.49). I had recorded this as evidence of an
intrinsic rotation/translation ambiguity — **that diagnosis was wrong**, and scaling the grid
per level fixed both it and the photometric odometry in the SLAM integration.

---

## 11. First compilation — what the compiler and the runtime found

The C++ engine was built for the first time with MSVC 19.44 (VS 2022 BuildTools) + Ninja +
OpenCV 4.10. **6,017 lines compiled with zero errors.** All five test executables then ran:
**96 of 104 passed on first execution, 8 failed.** After the fixes below, **107 of 107 pass**
(three diagnostic tests were added in the process).

Every failure was a real defect, not a test artifact — with two exceptions noted as such.

### 11.1 Four dead channels, all from one unit error

`cv::Scharr` has a gain of 32. `ImageQualityEngine` used it unnormalised, so on any real image:

| quantity | intended | actual |
|---|---|---|
| `sharpness` = √energy / √900 | 0…1 | **always 1.0** (saturated) |
| `blur_extent_px` | 0…12 px | **always 0** |
| `blur_free` → `motion_blur` evidence | live | **always 1 → 0** |
| `buildWeightMap`'s `m/(m+8)` info weight | 0…1 | **always ≈1** |

So the motion-blur channel *and* the per-pixel information weighting were both inert, and
`α₀(E)` could not respond to blur at all. `EnvironmentAnalyzer::estimateTexturePoverty` and
`estimateSceneComplexity` had the same error with a threshold of `20.0`, which in physical units
is 0.6 gray/px — a bar so low that on the degraded test image **noise counted as texture**:
texture poverty read *lower* (0.503) for a blurred, hazy, σ=12-noise image than for the clean
one (0.706). Complete inversion. Fixed by normalising and setting the threshold to
`max(6 gray/px, 2·0.68·σ_noise)` — the noise-gradient term is what makes it non-invertible.
Degraded texture poverty is now 0.968.

### 11.2 The `Dark` label was unreachable

`darkness = 0.7·dark_by_lum + 0.3·dark_by_noise·dark_by_lum` fails twice. With no gain noise it
caps at 0.7, but the `Dark` label needs > 0.75 — unreachable for any low-noise sensor. And the
noise term is *multiplied by* `dark_by_lum`, so when auto-exposure brightens a dark scene — the
one case the noise channel exists to catch — the evidence is zeroed. Replaced with a noisy-OR.

### 11.3 A permanent-ghost state in the token map

`isAlive()` excluded `Displaced`, so a displaced token left the association candidate set, stopped
receiving absence evidence, and **froze at existence 0.247445** — just under the displace
threshold (0.25), far above the retire threshold (0.12). Removal only targets `Retired`. The
token therefore survived 60 consecutive in-view misses and would have survived forever. Measured,
not theorised: the belief was bit-identical across misses 10 through 60. `Displaced` is a
hypothesis about *location*, not a reason to stop reasoning; it is now alive.

### 11.4 ECDA: the robust kernel was destroying convergence

`huber_delta = 12.0` in intensity units. When misaligned, residuals reach 50–100 gray levels, so
the points with the largest residuals — precisely the ones carrying the displacement information —
are the ones down-weighted most. Measured basin (residual error ÷ initial error, planar scene):

| `huber_delta` | 0.03 m | 0.06 m | 0.10 m | 0.14 m |
|---|---|---|---|---|
| **12.0** (as shipped) | 0.012 | **2.005** | 1.696 | 0.740 |
| 40.0 | 0.008 | 0.005 | 0.002 | 0.900 |
| 1e9 (pure L2) | 0.007 | 0.005 | 0.002 | 0.889 |

At 0.06 m the kernel pushed the estimate **twice as far from truth as doing nothing**. That
`delta = 40` matches pure L2 means the kernel as configured was purely harmful.

Raising the constant is not the fix. The scale at which a residual is an outlier is the *residual
distribution*, not a hand-picked number. But scaling by MAD alone is also wrong: when badly
misaligned the residuals are not "inliers + outliers", they are all misalignment, so robust
statistics have no valid bulk to estimate from — a pure-MAD kernel still broke 0.10 m
(0.005 → 1.224). The kernel is only meaningful once residuals approach the *measured sensor
noise*, which `ImageQualityEngine` already reports:

```
delta = huber_k · σ_resid · max(1, σ_resid / (noise_ratio · σ_noise))
```

Graduated non-convexity with no hand-written schedule, referenced to a physical quantity.

### 11.5 ECDA: rank-deficient directions were completely undamped

LM damping was `lambda · max(1e-9, H(k,k))` — purely relative, so a direction with `H(k,k) ≈ 0`
received damping ≈ 0 and its solution grew without bound. This is the mirror image of the Python
LM defect in §10 (absolute floor `1e-12` against diagonals of `1e8`). Fixed with a floor
proportional to the largest diagonal — **per block**: pose diagonals are ~1e7 `(intensity/m)²`
while the brightness-offset diagonal is `Σw` ~1e3, so a single shared floor misclassifies the
affine offset as a weak direction and damps it, which broke exposure-change invariance. The
unit-mixing warning was written in the comment and then violated one line later.

### 11.6 ECDA: the degeneracy spectrum was computed and never used

The engine reports `observable_dof`, `eigenvalues` and `weakest_direction` — a headline feature —
but each pyramid level handed its **full** pose to the next regardless of what it had actually
constrained. A 20×15 level solving 8 parameters from 66 points passed unconstrained axes downward
as if they were evidence. Each coarse level now projects its motion onto its own observable
subspace (diagonal-normalised eigendecomposition, so the test is unit-free).

Level 0 is deliberately exempt: its consumer is the factor graph, which already receives `Λ`.
Gating there too double-counts the degeneracy — applied to level 0, it cut the solution to
**exactly half** on a planar scene.

### 11.7 The result: the pyramid works, and it was measured wrong

The three ECDA fixes together turned the deepest level from harmful into essential. Convergence
radius from a cold start (640×480, planar texture at 2.5 m):

| pyramid levels | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| convergence radius | 0 | 0 | 0.02 m | 0.10 m | **0.18 m** | 0.18 m |

Before the fixes, level 5 produced a 0.51 m error and read as harmful. The earlier reading was an
artifact of judging by error at a *single* motion magnitude — the meaningful quantity is the
radius. Default `pyramid_levels` raised 4 → 5; level 4 costs 297 points and doubles the radius.

Two of the eight failures were test artifacts, and both encoded something impossible:
`BORDER_REPLICATE` invented image content with no true correspondence (fixed by warping a padded
canvas and cropping), and the TokenStore renderer wrote **centroid** depth into the depth map —
a sensor that cannot exist, and one that silently cancelled the very correction under test.

### 11.8 A fourth quantity mismatch

The depth-centroid coefficient 0.778, measured in Python, was copied into C++ and applied to a
**half**-extent, producing a 0.233 m bias at 4 m — predicted exactly by `0.778 × 0.3` and
confirmed by the measured 0.2355 m. Neither value is wrong on its own: Python takes the median
over the *whole* mask (oblique sides pull it deeper, so < 1.0), C++ over the central 50 % of the
box (near the front face, so 1.0). The coefficient belongs to the sampling region, not to the
world. Both files now say so.

---

## 12. First real data — TUM RGB-D `freiburg1_xyz`

250 frames (8.30 s, 30 Hz, 2.77 m of ground-truth motion). The C++ engine estimates
(`wme_tum_odometry`); Python scores (`tools/tum_eval.py`). Deliberately split: one codebase
grading its own output lets a shared bug hide itself.

| | ATE RMSE | frame RPE | rot RPE |
|---|---|---|---|
| **ECDA** (kf 0.03 m, motion prior) | **1.55 cm** | 6.6 mm | — |
| ECDA, no motion prior | 15.18 cm | 41.8 mm | — |
| identity (camera never moved) | 17.21 cm | 20.5 mm | 10.07 °/s |
| constant velocity | 16.49 cm | — | — |

**11× better than the trivial baseline.** Every reported configuration also required
undistortion — TUM ships raw images, and fr1's coefficients are not small
(k₁ = 0.2624, k₂ = −0.9531). Using the calibrated pinhole intrinsics on distorted images gave
ATE 67 cm; undistorting first gave 24 cm.

### 12.1 The baseline earned its place immediately

The first real-data run scored **ATE 24 cm against identity's 19 cm** — the estimator was worse
than assuming the camera never moved. Without a trivial baseline in the table that number reads
as "24 cm, plausible for a first attempt."

### 12.2 It was measuring a stream I had broken

Disk space forced a partial download, and I extracted "the first 400 entries" by iterating the
tar. **Tar order is not time order.** The result was 400 frames scattered across the full 26 s
at a median 60 ms spacing, of which only 207 had a depth partner within 20 ms — an irregular
~10 Hz stream with gaps to 768 ms and true inter-frame motion up to 233 mm, beyond the measured
convergence radius. The engine was being asked to do something it had already been shown it
cannot.

Refetching a *time-contiguous* 8.3 s block (median gap 32.1 ms, `tools/tum_fetch.py`, which
streams the archive over HTTP and never stores it) changed the same binary from 24 cm to 15 cm
and from losing to the baseline to beating it. **This is §10.4 again, on real data: the fifth
experiment that measured something other than what it claimed.**

### 12.3 The diagnosis that was wrong, and the sweep that corrected it

Given 15 cm and a per-frame error of 41.8 mm against 11.6 mm of actual motion, I reasoned that
ECDA selects the strongest gradient per cell, that in real scenes those are object outlines, and
that outline depth is neither foreground nor background — so the 3D points are wrong. Plausible,
physically grounded, and **wrong**. Measured on TUM, depth-edge rejection changes nothing:

| `depth_edge_ratio` | 0.005 | 0.02 | 0.05 | disabled |
|---|---|---|---|---|
| ATE | 3.32 cm | 1.55 cm | 1.56 cm | 1.56 cm |

(Only the aggressive setting registers, and it *hurts* — it starves the estimator of points.)

The actual cause was in the harness, not the engine: `wme_tum_odometry` initialised every
alignment at identity. Against a keyframe up to 10 cm away that demands a 10 cm cold-start
convergence every frame, when the frames themselves are 11 mm apart. Adding a constant-velocity
prior — three lines — moved ATE from 15.18 cm to 2.32 cm at the same keyframe distance.

| | no prior | with prior |
|---|---|---|
| ATE | 15.18 cm | 2.32 cm |
| frame RPE | 41.8 mm | 7.3 mm |

**Rule (a restatement of §10.5): a physically plausible mechanism is not evidence. Sweep the
parameter that the mechanism predicts should matter, and believe the table.**

The depth-edge term was kept — its physical argument stands and it costs nothing measurable —
but it is explicitly **unvalidated**, and the header says so. Keeping it because it sounds right
is precisely the habit this document exists to resist.

### 12.4 What the sweep says about the tuning surface

Once the prior is in, the engine is insensitive to everything else. Fourteen configurations
spanning pyramid depth 4–6, grid 4–8, `huber_k` 0.5–4, `min_gradient` 3–12 and keyframe distance
0.02–0.05 m all land between **1.52 and 1.97 cm**. Only two choices matter, and both were
harness-level: the motion prior (6.5×) and undistortion (2.8×).

Two synthetic-derived defaults survived contact with real data unchanged: the noise-referenced
graduated Huber (§11.4) and the observable-subspace projection (§11.6) — `huber_k` from 0.5 to 4
moves ATE by 0.24 cm, and pyramid depth 4 to 6 by nothing at all.

---

## 13. Five sequences — where ECDA works and where it does not

`tools/tum_benchmark.py`. 9-second contiguous windows, identical configuration throughout, no
per-sequence tuning. Baseline is "the camera never moved", recomputed per sequence.

| sequence | ATE | baseline | ratio | frame RPE | what it tests |
|---|---|---|---|---|---|
| fr3_sitting_xyz | **1.01 cm** | 19.86 | 19.7× | 7.9 mm | people present, nearly still |
| fr1_xyz | **1.56 cm** | 17.21 | 11.1× | 6.6 mm | static, translation-dominant |
| fr1_desk | **2.53 cm** | 57.64 | 22.7× | 10.5 mm | static, rotation + translation |
| fr1_360 | **10.37 cm** | 16.44 | 1.6× | 24.4 mm | pure rotation |
| fr3_walking_xyz | **20.72 cm** | 28.02 | 1.4× | 29.9 mm | people walking |

Static scenes land at 1–2.5 cm. Two regimes break it, and the gap between them and the static
case is an order of magnitude.

### 13.1 The cost of dynamic objects, measured

`fr3_sitting_xyz` and `fr3_walking_xyz` are the same room, the same camera trajectory, the same
people. The only difference is whether those people move.

| | sitting | walking | |
|---|---|---|---|
| ATE | 1.01 cm | 20.72 cm | **20.6× worse** |
| frame RPE | 7.9 mm | 29.9 mm | 3.8× worse |
| photometric RMSE | 16.0 | 22.3 | |

**A 20.6× degradation caused by nothing but motion in the scene.** This is the debt the token
layer exists to repay, now denominated on real data rather than argued from first principles.
`DirectAligner` already consumes `Frame::static_mask`; nothing produces one, because the YOLO
backend is unimplemented. The number above is what that backend has to be worth.

### 13.2 Pure rotation is a second, different failure

`fr1_360` fails for an unrelated reason, and the engine's own instruments show it: mean
observable DOF falls from 6.00 to **5.29** and surviving points from ~1450 to **379**. Sweeping a
room puts most surfaces beyond the depth sensor's range, so the constraint set thins and the
geometry degenerates. Unlike the dynamic case, this failure is visible from the inside.

### 13.3 Does the engine know when it is wrong?

WME claims to output *what is reliable*. That claim is testable: rank-correlate each per-frame
self-assessment (`tools/tum_selfassess.py`) against the relative-pose error computed afterwards
from ground truth. Rank correlation asks whether the *ordering* is right — calibration can be
fixed later, ordering cannot.

Reported as **lift** = P(bad frame | flagged) ÷ P(bad frame), where flagged is the signal's top
decile. Lift 1.0 means the signal carries no information.

| signal | fr3_sitting | fr1_360 | **fr3_walking** |
|---|---|---|---|
| condition number | **3.71** | 1.83 | 0.00 |
| photometric RMSE | 3.34 | 1.22 | 1.12 |
| 1 − inlier ratio | 3.34 | 1.22 | 1.12 |
| 6 − observable DOF | 1.00 | 1.74 | 1.00 |
| 1 / point count | 1.86 | 1.22 | 0.75 |

**On static scenes the self-assessment works** (lift 3.3–3.7 — the flagged decile is over three
times as likely to be a bad frame). **On geometric degeneracy it works weakly** (1.7–1.8, and the
DOF signal is the one that fires, which is what it is for). **On dynamic content it is blind:
every signal sits at or below 1.12, on the sequence where the estimator is 20× worse.**

That is the sharpest result in this document. The estimator does not merely fail on moving
objects — it fails *confidently*. The robust kernel absorbs the moving people, the residual never
rises, and no photometric quantity registers the problem. **No amount of tuning the photometric
confidence path can fix this, because the information is not in the photometric channel.** It is
the argument for semantic evidence, stated as a measurement.

### 13.4 A metric that flattered itself, and a signal that ran backwards

Two defects surfaced in this experiment, both of the type §10 catalogues.

**The metric.** The first version reported recall alone, and `6 − observable DOF` scored
**100 %** on `walking_xyz` — apparently the one signal that worked. It is constant at 6 on that
sequence, so `6 − DOF = 0` everywhere, the top-decile threshold collapses to `0 ≥ 0`, and every
frame is flagged, good and bad alike. Recall is 100 % because nothing is excluded. Lift reports
1.00 and is immune to this, which is why the table above uses it.

**The signal.** `1 − inlier ratio` scored lift **0.00** on `sitting` — perfectly anti-correlated,
worse than useless. The cause was mine: §11.4 made the Huber threshold adapt to the residual
spread, and `inlier` was still counted against that same moving threshold. When residuals grow,
the threshold grows with them and the ratio holds steady — a tautology, not a measurement.
Inliers are now counted against a **fixed** threshold tied to the measured image noise
(`health_k · σ_noise`), separate from the kernel. The same signal on the same data:

| | before | after |
|---|---|---|
| Spearman ρ (fr3_sitting) | −0.361 | **+0.319** |
| lift | 0.00 | **3.34** |

A quantity used for two purposes served neither. **Rule: a threshold that adapts to the data
cannot also be the yardstick that measures the data.**

---

## 14. YOLO, and the price of masking by class

The blocking item from §13.1 is now implemented: `IYoloRuntime` has two backends
(`YoloRuntimeCv` on OpenCV DNN, `YoloRuntimeOrt` on ONNX Runtime 1.22) sharing one
preprocessing/decoding module, so a disagreement between them is an inference-engine
difference and nothing else. YOLO11n, 80 COCO classes, ~60–150 ms/frame on CPU. 9 tests.
Total C++ suite: **116 tests, all passing.**

Two traps worth recording because neither announced itself:

- **Windows ships its own `onnxruntime.dll` in System32** (Windows ML). The DLL search order
  puts System32 ahead of `PATH`, so the process bound an incompatible ABI and died with
  `0xc0000005` inside session creation — before any of our code ran. The only reliable fix is
  staging the correct DLL next to the executable, which CMake now does.
- **The official `yolo11n.onnx` is opset 22.** ORT 1.20 supports through 21 and refuses to load
  it; OpenCV 4.10's importer fails earlier still, on YOLO11's C2PSA block. Both refused loudly,
  which is the only reason this cost minutes rather than a wrong result.

### 14.1 The ablation that overturned the obvious answer

Masking every detection of a "mobile" class (person, car, …) and excluding those pixels from the
photometric residual:

| sequence | scene | mask OFF | mask ON | change |
|---|---|---|---|---|
| fr3_walking_xyz | people walking | 20.72 cm | **5.23 cm** | **3.96× better** |
| fr3_sitting_xyz | people seated | **1.01 cm** | 15.84 cm | **15.7× worse** |
| fr1_xyz | no people | 1.56 cm | 1.54 cm | unchanged |
| fr1_desk | no people | 2.53 cm | 2.53 cm | unchanged |

Had I run only `walking_xyz` I would have reported a 4× win and moved on. The control says the
opposite: class-based masking takes the **best** sequence in the whole benchmark and makes it
worse than unmasked `walking`. On `sitting` it erases 48 % of the image — people who are sitting
perfectly still and are, for the photometric residual, as good as wall.

**"person" is a class, not a motion state.** The decision belongs to a belief updated by
observation, not to a lookup table. That is what the architecture already said; this is the
first time it has cost something measurable to ignore it.

### 14.2 A dead channel, in three layers

`WorldToken::static_belief` is exactly the belief that should decide. It was unreachable, and it
took three independent overrides to make it so:

| location | code | verdict |
|---|---|---|
| `TokenStore` create | `static_belief = Agent ? 0.15 : 0.6` | **correct** — a prior |
| `ConfidenceEngine::updateStaticBelief` | `static_belief = min(static_belief, 0.3)` | a ceiling that erases all evidence |
| `WorldToken::isDynamic()` | `belief < 0.4 \|\| has(Agent)` | an override that ignores the belief entirely |

The update at the top of that function computes a displacement-based log-odds ratio, and the two
lines below it guarantee the result can never matter. A person observed motionless for a minute
stayed pinned at 0.3 and `isDynamic()` returned true regardless. **The fourth dead channel in
this document**, after `sharpness`, `motion_blur`, and `texture_poverty` — and the first whose
cost is measured (§14.1).

The stated motive — "a car stopped at a light must not become a static landmark" — is sound. The
implementation confuses *skepticism* with *prohibition*. Corrected: the prior stays, static
evidence for an agent is down-weighted (`agent_static_evidence_gain = 0.5`), dynamic evidence is
not (missing a moving object costs more than doubting a stopped one), and `isDynamic()` now reads
the belief and nothing else.

### 14.3 Reachable is not the same as reached

With the overrides gone, masking driven by `static_belief` instead of by class:

| sequence | OFF | by class | **by belief** |
|---|---|---|---|
| fr3_walking_xyz | 20.72 cm | 5.23 cm | 7.93 cm |
| fr3_sitting_xyz | 1.01 cm | 15.84 cm | 11.47 cm |

The belief barely moved: mean `static_belief` over agent tokens ended at **0.153** (walking) and
**0.161** (sitting), against a 0.15 prior and a 0.4 threshold.

Before theorising about evidence rates I measured the inputs, and the numbers refused to fit any
story about slow accumulation: agent tokens survived **15.4 observations on average, 75 at most**
— far more than the ~14 the arithmetic said were needed. Tokens were not the bottleneck.

### 14.4 The fifth dead channel — killed by a gate this time

`ConfidenceEngine::updateStaticBelief` opens with

```cpp
if (dt < cfg_.static_min_dt) return;      // static_min_dt = 0.05 s
```

TUM runs at 30 Hz, so `dt = 0.033 s`. **The static/dynamic judgement returned on its first line,
every frame, for the entire run.** The belief never moved because the function that moves it never
executed. All the machinery below — the displacement/σ ratio, the agent down-weighting, the
log-odds update — was unreachable code in this configuration.

The gate's reasoning is sound: over a very short interval, displacement is noise. But discarding
the observation puts a cliff at the frame rate — the identical system learns nothing at 30 Hz and
learns fully at 15 Hz. The fix is to *accumulate* rather than discard: each token holds a
reference position and timestamp, and the judgement fires once the window reaches
`static_min_dt`, measuring motion over an interval long enough for real motion to exceed noise.

With that change the belief started moving:

| | mean belief | max belief | masked area |
|---|---|---|---|
| fr3_sitting | 0.161 → 0.271 | 0.229 → 0.817 | 50.1 % → 43.4 % |
| fr3_walking | 0.153 → 0.168 | 0.175 → 0.309 | 50.5 % → 50.5 % |

> **This section originally read "the channel discriminates for the first time" and cited
> 0.817 vs 0.309 as evidence. That was wrong.** §16.1 measures what those numbers actually
> were: 84.4 % of static-belief updates on `sitting` were positive evidence, and **84.3 %** on
> `walking` — identical to 0.1 pp. The belief was a near-deterministic function of how long a
> token survived association, not of whether the object moved. The "separation" was track
> length. Left running long enough, a walking person would have become a landmark.
>
> Two comparable numbers with a plausible gap between them are not evidence of discrimination.
> The check that would have caught it — what fraction of updates carry positive evidence in
> each case — costs one line and was not run.

### 14.5 Where it actually stands

| sequence | scene | OFF | by class | by belief |
|---|---|---|---|---|
| fr3_walking_xyz | people walking | 20.72 | **5.23** | 7.93 |
| fr3_sitting_xyz | people seated | **1.01** | 15.84 | 12.14 |
| fr1_xyz | none | 1.56 | 1.54 | **1.44** |
| fr1_desk | none | 2.53 | 2.53 | 2.53 |

Belief-driven masking beats class-driven on `sitting` (15.84 → 12.14) and loses on `walking`
(5.23 → 7.93). Neither approaches the 1.01 cm that simply leaving seated people alone achieves.
The *maximum* belief reached 0.817 but the *mean* is 0.271 — one token was reclassified and most
were not, so ~43 % of the frame is still masked.

**Reachable, discriminating, and not yet sufficient.** The remaining gap is about how many
observations a token accumulates before association loses it, which an 8-second window cannot
settle. Claiming §14.1's problem is solved would be claiming the last table says what the first
one does.

---

## 15. The information matrix was overconfident by 2×10⁴, and why

`DirectAligner` returns a 6×6 information matrix. Measured on real TUM data for the first time
(`python/tools/tum_nees.py`), against the 95 % χ² band for 6 DOF, `[5.5, 6.5]`:

| sequence | ANEES | median trans err | covariance inflation needed |
|---|---|---|---|
| fr1_xyz | 140 306 | 5.29 mm | 23 384× |
| fr1_desk | 125 228 | 7.63 mm | 20 871× |
| fr1_360 | 5 425 | 10.31 mm | 904× |
| fr3_sitting_xyz | 146 087 | 5.53 mm | 24 348× |
| fr3_walking_xyz | 124 531 | 16.39 mm | 20 755× |

**Zero of 1086 frames** passed the per-frame gate. Note the second column: this is not the
§10.3 failure mode of an inflated, useless estimator — the estimate is *accurate* and the
uncertainty is nonsense. Those are independent properties and only one of them was ever checked.

The meter was validated before the estimator was blamed: synthetic errors drawn from the
**actual reported Λ** run through the same pipeline give ANEES 5.69–5.93 on every sequence, and
scaling the covariance by s moves ANEES by exactly 1/s. Sweeping the ground-truth clock ±10 ms
leaves the most generous alignment at 17 268×.

### 15.1 The sub-sampling experiment

The natural hypothesis was correlated photometric residuals — 1500 pixels are not 1500
independent observations, and Python had already measured 5.7× from this cause. It is testable:
thin the point set and watch what happens. If residuals are correlated with some fixed
redundancy factor κ, then `N_eff = N/κ` and error² must still fall as 1/N.

Measured across five sequences at N, N/2, N/4, N/8, N/16:

| sequence | Λ ~ N^ | **err² ~ N^** | ANEES ~ N^ |
|---|---|---|---|
| fr1_xyz | +1.00 | **−0.00** | +1.03 |
| fr1_desk | +1.00 | **+0.04** | +0.97 |
| fr1_360 | +0.94 | **−0.40** | +0.85 |
| fr3_sitting_xyz | +1.01 | **−0.07** | +0.94 |
| fr3_walking_xyz | +1.05 | **−0.09** | +0.88 |

On `fr1_xyz` the median relative translation error is **5.291 mm at N = 1492 and 5.110 mm at
N = 91**. Sixteen times the evidence buys nothing.

**This rules out correlation as the mechanism, not merely as the whole of it.** Every correlation
model predicts ANEES slope 0. The measured slope is 1. The information grows linearly with pixel
count while the accuracy does not move at all, which has exactly one reading: **for the pose, one
frame is one observation, whatever the pixel count.** The earlier `N_eff < 1` readings (0.63,
0.87) were this same fact seen from a single point on the curve.

### 15.2 Three models, and why the shipped one is not a fitted constant

`InformationModel` — the first three are one formula, `Λ = H / (χ²/N_eff)`, differing only in
what counts as a sample:

| model | assumption removed | fr1_xyz ANEES |
|---|---|---|
| `SensorVariance` (was default) | — | 140 306 |
| `ResidualVariance` | residual variance ≠ sensor noise | 8 621 |
| `ClusterRobust` (sandwich) | + within-frame independence | 3 544 |
| **`EffectiveSample`, ν = 1** (shipped) | + the frame is N observations | **9.4** |

The first step is an independent defect on its own: Λ was divided by
`ImageQuality::photometricVariance()`, the *sensor noise* estimate, which is 16–51× smaller than
the residual variance actually achieved at the solution (assumed σ 3.4–8.2 vs achieved RMS
9.6–22.8).

The sandwich estimator was implemented, measured, and **not shipped**. It buys only 1.4–2.4×
beyond the residual variance and *loses* 2.2× on fr1_360 — because by construction it can only
see correlation *within* a frame, and §15.1 says the missing term is at frame level. A negative
result from a correctly-built estimator is worth more than the 2× it would have bought.

> **Bounded by §17.1.** On a different camera (`fr2`) the required ν is 0.10–0.16 and ANEES
> reaches 58.6. The sub-sampling *slope* transfers — "one frame is one observation" holds there
> too — but the scalar does not. ν = 1 is a property of this sensor, not a universal constant.

`ν = 1` is not tuned. Required ν across 5 sequences × 5 thinning levels measures 0.51–1.97; ν = 1
sits mid-range **and is the physical boundary** the sub-sampling curve identifies. Result:
**23 384× → 1.6× worst case**, and the per-frame 95 % gate goes **0 of 1086 → 994 of 1086** (a
perfectly calibrated estimator gives 1032).

Re-running the sub-sampling on the shipped model: ANEES ~ N^−0.07…−0.29, down from +0.85…+1.03.
The N-dependence is gone, not rescaled.

### 15.3 The accuracy did not move — and that is tested, not asserted

The information matrix is a pure output; the estimate never reads it. That is now a test
(`InformationModelDoesNotChangeTheEstimate`), so every trajectory above is **bit-identical** to
the baseline, not merely close. `fr1_xyz` ATE is 1.56 cm before and after. An estimator that
becomes consistent by becoming worse has solved nothing, and this one did not.

A synthetic control closes the argument: the same code, exact analytic depth, exact warp, error
from added white pixel noise only, gives ANEES **1.40** under `ResidualVariance` where real data
gives 845–8621. **The formula was never wrong.** The overconfidence is a property of real
residuals, not of the derivation. This also exonerates the level > 0 subspace projection, which
is present in the synthetic path too.

### 15.4 What is left, and why no constant was fitted to close it

Final ANEES: **3.67 / 4.29 / 4.74 / 8.00 / 9.39** against `[5.5, 6.5]` — two sequences
overconfident, three *under*, all within 1.6× either way.

The residue is structured, and the structure says a scalar cannot finish it. Per-axis E[z²] is
**below 1 on every axis** (0.03–1.04) while the joint NEES is 3.7–9.4 — the marginals are
conservative and the error lives in the *correlations*. Decomposed in Λ's eigenbasis, the NEES
concentrates in the single stiffest direction (5.63 of 9.39 on fr1_xyz) while the four weakest
contribute 0.05–0.6 each. A scalar would inflate five already-conservative directions to fix one.
An additive floor was grid-searched and fails for the same reason — three sequences are already
underconfident, so the best common floor is ≈ 0.

**No constant was shipped.** Leave-one-out on a plain scalar generalises to 3.4–9.2, i.e. it buys
nothing real. Stopping at 1.6× with the structure documented is the honest end of this
measurement.

---

## 16. `static_belief` — five defects, one channel

§14.5 left belief-driven masking at 12.14 cm on `fr3_sitting_xyz` against 1.01 cm for leaving
seated people alone, and called the channel "reachable, discriminating, and not yet sufficient".
Two of those three words were wrong. Instrumenting the inputs rather than the output found five
independent defects, only one of which was in the belief update.

### 16.1 The evidence function did not discriminate

| | positive-evidence updates | mean evidence |
|---|---|---|
| fr3_sitting (people seated) | **84.4 %** | 0.179 |
| fr3_walking (people walking) | **84.3 %** | 0.136 |

Identical to 0.1 pp, against a hard evidence ceiling of 0.380. Belief was a near-deterministic
function of update count: 0 updates → 0.150, 5 → 0.169, 10 → 0.279, 20 → 0.440, 35 → 0.768, with
spread inside each bucket ≤ 0.11. **The channel was measuring track length.**

Cause: the evidence was a bounded heuristic on `[−0.31, +0.38]`, not a likelihood ratio. Replaced
with a proper 3-DOF Gaussian LLR — `H_static: N(0, σ_d²)` against `H_dynamic: N(0, σ_d² + (v·dt)²)`
— which is unbounded in both directions and grows with the observation window, as evidence should.

### 16.2 The window was set to the value that discriminates least

Signal grows as `v·T`; measurement noise does not. Measured window displacement, known-static
objects vs a person:

| T (s) | sitting: static / person | ratio | walking: static / person | ratio |
|---|---|---|---|---|
| **0.067** | 20.5 / 24.6 mm | **1.20** | 27.6 / 39.8 mm | **1.44** |
| 0.20 | 23.0 / 35.9 | 1.56 | 36.9 / 93.4 | 2.54 |
| **0.50** | 26.0 / 58.9 | **2.26** | 51.5 / 200.9 | **3.90** |
| 1.00 | 29.5 / 90.5 | 3.06 | 70.2 / 2117.6 | 30.2 |

`static_min_dt = 0.05 s` sat on the top row. Worse, the z-score's neutral point over that window
worked out to 60 mm in 0.067 s — **exactly human walking speed**. The one velocity the system
could not call dynamic was the one it exists to detect. Now 0.50 s.

### 16.3 The scale was the wrong statistic — the sixth quantity mismatch

`z = displacement / (3 · max(0.02, positionSigma()))`. `positionSigma()` is the **fused
estimate's** uncertainty, which shrinks with observations. A window displacement between two
independent measurements has σ = √2·σ_measurement. Same class of error as §10.2, now sixth. Fixed
to `max(floor, √2·σ_meas)`; `static_sigma_scale` deleted.

### 16.4 One outlier erased ten seconds of evidence

A pure Gaussian LLR is unbounded *below*. A single 3 m window displacement — a YOLO box centroid
jumping to a different part of the person — carried **−453 nats** and wiped out everything the
token had earned. Calibrated on objects whose true motion is exactly zero: bolted-down tv,
keyboard and chair exceed the model's 3σ in **16 % / 37 % / 29 %** of windows. The tail is not
optional, it is the common case. Added a heavy tail (`static_outlier_rate = 0.2`) bounding
evidence at `log ε`: an outlier can still reverse a conclusion, it can no longer erase one.

### 16.5 The mask was not driven by the belief at all

Provenance of erased pixels:

| | total | matched this frame | **stale (extrapolated ghosts)** | never judged (class prior) |
|---|---|---|---|---|
| fr3_sitting | 43.4 % | 13.2 pp | **23.9 pp** | 6.3 pp |
| fr3_walking | 50.5 % | 6.8 pp | **34.8 pp** | 8.9 pp |

**70 % and 86 % of the mask came from tokens whose detection had stopped, or from the untouched
class prior.** §14.1's error one level down: the belief was being computed and then bypassed.
`buildStaticMask` now erases only tokens matched in the current frame, and uses the detection box
rather than the projected 3D AABB — the AABB lags a walker (10.02 → 5.19 cm on its own).

### 16.6 Association was the binding constraint

11.4 % (sitting) and 19.7 % (walking) of detections were rejected by the χ² gate. The nearest
same-class candidate sat **174 / 185 mm** away at Mahalanobis **474 / 535** against a gate of 9.0
— the fused covariance was claiming σ ≈ 8 mm for a YOLO box centroid. Two people produced **87
and 105 distinct tokens**, median lifetime 2 and 1 observations, 67 % / 79 % dying at ≤ 3.

A maneuver term (`assoc_maneuver_speed = 2.0 m/s`) was added to the **gate** covariance only —
fusion untouched, so this loosens what counts as the same object without making the estimate
looser. Tokens per two people: 87 → **13** and 105 → **6**. Agent-token observations 17.5 → 67.8
and 6.7 → 89.2.

### 16.7 Result

| sequence | OFF | by class | belief (§14.5) | **belief (now)** |
|---|---|---|---|---|
| fr3_walking_xyz | 20.72 | 5.23 | 7.93 | **5.19** |
| fr3_sitting_xyz | **1.01** | 15.84 | 12.14 | **1.63** |
| fr1_xyz | 1.56 | 1.54 | 1.44 | 1.53 |
| fr1_desk | 2.53 | 2.53 | 2.53 | 2.53 |

> **Bounded by §17.2.** Under halfsphere camera motion this inverts completely — belief-driven
> masking reaches 13 053 cm on `fr3_sitting_halfsphere`, 571× worse than not masking at all. The
> belief channel is fixed; the association that feeds it is not, and `assoc_maneuver_speed = 2.0`
> was fitted on `xyz` translation.

Both targets met simultaneously: `sitting` 15.84 → **1.63** (9.7× better than class masking,
0.62 cm off the leave-them-alone optimum) **while** `walking` reaches class-masking parity
(5.19 vs 5.23) and stays 4.0× better than unmasked. Mean agent-belief separation between the two
sequences went 1.6× → **7.0×** (0.436 vs 0.062). Neither the 0.4 threshold nor `evidence_gain`
was touched.

**The 40-second window is where it becomes unambiguous:**

| fr3_sitting_xyz, 40 s | OFF | by class | belief |
|---|---|---|---|
| ATE | 3.63 cm | **744 cm** | **5.88 cm** |

Class masking does not merely underperform over 40 s — it **diverges**. Every 9-second result in
§14.1 was measuring a bounded prefix of a divergent trajectory. The window was hiding the shape
of the failure, not just its size.

Convergence, measured on the 40 s run: each long-lived person token crosses 0.4 within
**1.0–4.5 s** of first detection (2–7 judgement windows) and 0.7 within 2.1–6.2 s, then holds
above 0.4 for 80–95 % of its remaining life, ending at 0.76–0.96. On `walking`, every person
token decays to 0.018–0.02 within 1–2 s and never crosses 0.4.

Tier 1's input recovered as a side effect: `isStableLandmark()` count on `fr3_sitting` goes
4.52 → 5.26 per frame (6.93 on the 40 s window), and agent landmarks 0.33 → 0.55 → 1.14. Seated
people now qualify as landmarks; nobody in `walking` does.

---

## 17. Held-out sequences — where the constants break

Every number in §12–16 comes from five 9-second windows of one dataset on one camera, and most
of the constants were selected on the data they are reported on. Eight held-out sequences were
fetched to try to break them: `fr2_desk` and `fr2_desk_with_person` (**a different camera** —
different intrinsics and distortion), `fr1_room` / `fr1_plant` / `fr1_teddy` (different scene
statistics), `fr3_sitting_halfsphere` / `fr3_walking_halfsphere` (dynamic content under camera
motion unlike the `xyz` pair everything was tuned on), and a 28.8 s window of `fr3_walking_xyz`.

Nothing was tuned. The held-out build first reproduced the published calibration set exactly —
ANEES 9.39 / 8.00 / 4.29 / 4.74 / 3.67 and the §16.7 ablation to the centimetre — so the
divergences below are real and not a build artifact.

**Four of seven conclusions transfer. Three do not.**

### 17.1 `ν = 1` breaks on a different camera — but the physics under it does not

| sequence | ANEES | required ν | verdict |
|---|---|---|---|
| fr1_room | 6.32 | — | ✓ in band |
| fr3_sitting_halfsphere | 5.31 | — | ✓ |
| fr3_walking_halfsphere | 7.04 | — | ✓ |
| fr1_plant / fr1_teddy | 2.56 | — | 2.3× **under**confident |
| **fr2_desk** | **38.78** | **0.16** | 6.5× overconfident |
| **fr2_desk_with_person** | **58.58** | **0.10** | 9.8× overconfident |

Held-out required-ν spans **0.10–2.33** against 0.51–1.97 on the calibration set. Ruled out:
the ground-truth clock (±10 ms sweep leaves fr2_desk ≥ 32.2), the meter (self-test passes), and
the motion/dt regime (`fr3_sitting_halfsphere` runs at a *longer* 132 ms interval with comparable
motion and lands at 5.31). And per §10.3 the direction matters: **fr2 is the most accurate
sequence in the whole set** (2.61 / 2.76 mm relative error) while being the worst calibrated.

But the measurement ν was derived *from* transfers cleanly, fr2 included. Shipped-model
sub-sampling slopes on held-out data: −0.07, −0.02, −0.10, −0.02, against the published
−0.07…−0.29. On fr2_desk, 16× fewer pixels moves the median error 2.608 → 3.095 mm.

**"One frame is one observation" is a property of photometric alignment. "ν = 1" is a property of
the camera.** §15.2 claimed ν = 1 was a physical boundary rather than a fitted constant. The
first half of that claim survives; the second does not. The right shape is a per-sensor ν, and
nothing in the current design provides one.

### 17.2 Belief-driven masking inverts under unfamiliar camera motion

| sequence | OFF | by class | **by belief** |
|---|---|---|---|
| fr3_sitting_halfsphere | **1.50 cm** | 452 cm | **13 053 cm** |
| fr3_walking_halfsphere | 46.08 | **8.65** | 10.55 |
| fr2_desk_with_person | 1.46 | 1.09 | 1.22 |

On `sitting_halfsphere` belief-driven masking is **571× worse than doing nothing** and 29× worse
than the class table it was built to replace. Independently reproduced.

The mechanism is measured, and it is not the belief update: mean agent belief **0.155** — it
never leaves the prior — with **0.0 %** of agent observations above the 0.4 threshold, against
0.428 and 52.7 % on `sitting_xyz`. Association fragments under this motion: **154 person tokens
with median lifetime 5** (93.5 % dying at ≤ 3 observations) versus 13 tokens at median 17 on
`sitting_xyz`. `assoc_maneuver_speed = 2.0`, fitted on `xyz` translation, does not hold under a
halfsphere sweep.

The document's own separation metric tells the same story: **7.0× on the `xyz` pair, 1.7× on the
halfsphere pair** — back to the 1.6× that §14.4 was wrong to call discrimination. §16 fixed the
belief channel; it did not fix the association that feeds it, and §16.6 identified association as
the binding constraint without establishing that the fix generalises.

### 17.3 What did transfer

- **Class masking diverges on long windows** — confirmed, and the 28.8 s `fr3_walking_xyz`
  window closes §17's missing negative case: identity 30.22, **OFF 116.20** (the *unmasked* run
  is the one that diverges here), class 14.70, belief 16.22. Belief keeps walking parity over
  3.3× the original duration and correctly refuses to call walkers static (mean 0.102, 3.1 %
  above threshold).
- **TCG's accept rule** holds at the loose threshold — precision 100 % on fr2_desk (recall
  74.4 %, constant-estimator baseline 0 %), 100 % on fr1_plant, 93.8 % on fr3_sitting_halfsphere,
  and out-of-map rejection transfers 5 of 5. **But at the tight threshold precision falls to
  43.8–83.3 %**: the gate is calibrated to the loose threshold and the published summary does not
  say so. Separately, on fr2_desk raw `χ²/dof` predicts pose error at −0.86 while the blended
  `confidence` manages only −0.33 — `chi2_confidence_scale = 10` is diluting the best signal
  available.
- **ECDA static accuracy** transfers as an order of magnitude, not as a band: 2.38 / 3.37 / 4.51 /
  5.30 cm on the four held-out static sequences, 8–12× better than identity.

### 17.4 `motion_noise_floor` is circular — now quantified

§16 flagged this; the held-out set measures it. Median window displacement of **known-static,
bolted-down** objects, against the same run's ATE:

| sequence | static displacement | that run's ATE |
|---|---|---|
| fr2_desk_with_person | 17.8 mm | 1.22 cm |
| fr3_sitting_xyz | 32.9 mm | 1.63 cm |
| fr3_walking_halfsphere | 70.7 mm | 10.55 cm |
| fr3_walking_xyz | 78.7 mm | 5.19 cm |
| fr3_sitting_halfsphere | 130.6 mm | diverged |

**A 7.3× spread on objects that are all equally immobile, monotone in the odometry's own error.**
The floor is not measuring object motion; it is measuring how wrong the pose was. It is already
exceeded by 47–79 % of windows on its own calibration set and 92 % on held-out data.

### 17.5 Envelope checks — three sequences did not test what they appear to

- **`fr2_desk_with_person`'s first 9 s contains no detected person** (class mask covers 0.07 % of
  the frame) and all three masking modes return an identical 0.94 cm. A textbook §10.4 non-test.
  Refetched at +60 s where the person is present (52.9 % masked); that is the number reported.
- **`fr1_teddy` cannot test TCG** — mean constellation size 4.07 against `min_nodes = 4`, and
  zero returns from 28 queries.
- **`fr1_room`'s 0 % TCG precision rests on 2 returns** over a 1.4 m map with 3 registered places.
- **`sitting_halfsphere`'s 130.6 mm static displacement is contaminated** by the divergence it is
  being used to explain. The uncontaminated evidence for §17.4 is `walking_halfsphere`.

Clean across all eight: **0.0 % of inter-frame motions exceed ECDA's measured 0.18 m convergence
radius**, depth validity 56–83 %, zero alignment failures. The envelope was met; the constants
still broke.

---

## 18. Three-tier fusion on real data — the central claim, measured

§1 is this project's thesis: `Λ_total = α₀(E)·Λ_ECDA + α₁(E)·Λ_TCG + α₂(E)·Λ_SPA`, with the
improvement over Tier-0-only growing from 6.8 % to **53.6 %** as haze goes 0 → 0.9, and no
weather conditional anywhere in the fusion code. That result is simulation-only. All three tiers
now exist in C++, so it can finally be tested against a real sensor.

Fusion is proper information fusion on SE(3), not pose averaging: it minimises
`Σ ½ εₖᵀ(αₖΛₖ)εₖ` with `εₖ = log(T·Tₖ⁻¹)`, using the exact left-Jacobian inverse so estimates at
different linearisation points combine correctly. One image pass records all three tiers'
`(T, Λ)` and every ablation replays that same record, so all configurations see identical frame
pairs. Tier-0-only reproduces §13's published numbers exactly (1.56 / 2.53 / 10.37 / 1.01 /
20.72), which is the evidence that the harness is the same estimator.

### 18.1 The prerequisite: are the tiers on comparable scales?

§15 found Tier 0 overconfident by 2.3 × 10⁴. Fusing by information magnitude with that in place
would make `Λ_total ≈ Λ_ECDA` numerically and the whole experiment would measure nothing — §10.4's
failure mode, a fifth time. So this had to be checked first:

| tier | ANEES range | median rel. error | verdict |
|---|---|---|---|
| ECDA | 3.67 – 9.14 | 0.53 – 1.64 cm | calibrated to 1.6×, independently confirmed |
| SPA | 4.95 – 25.5 | 2.39 – 15.10 cm | worst 4.2× — **the trap does not apply** |
| TCG | 8.57 – **285** | 2.43 – 10.30 cm | up to 47× overconfident |

Two things ANEES hides and that matter more here. **Heavy tails**: frames failing the χ²(6)
99.9 % gate run 0.6–9.4 % (ECDA), 9.7–24.8 % (TCG), 4.2–37.4 % (SPA) — `fr3_sitting` TCG has
median NEES 4.7 and p99 1.6 × 10⁴, i.e. calibrated on average and catastrophic on a sixth of
frames, and information fusion has no robust kernel across tiers. **Bias is not the mechanism**:
`|mean e|/rms` sits at or below its `√(6/n)` null for essentially every tier on every sequence.

### 18.2 Fusion is net harmful on four of five sequences

ATE, cm, 9-second windows:

| sequence | **T0** | T1 | T2 | T0+T1 | **T0+T2** | all 3 | all 3 uniform-α | do-nothing |
|---|---|---|---|---|---|---|---|---|
| fr1_xyz | **1.56** | 7.72 | 30.18 | 3.65 | 2.71 | 4.23 | 3.27 | 17.21 |
| fr1_desk | **2.53** | 39.35 | 31.42 | 2.96 | 2.81 | 3.26 | 2.94 | 57.64 |
| fr1_360 | **10.37** | 16.44 | 205.30 | 10.37 | 48.78 | 48.78 | 39.31 | 16.44 |
| fr3_sitting | **1.01** | 79.75 | 142.41 | 7.22 | 7.19 | 9.46 | 5.96 | 19.86 |
| fr3_walking | 20.72 | 79.24 | 131.02 | 38.40 | **15.52** | 21.13 | 19.49 | 28.02 |

Independently reproduced. The single exception is `fr3_walking_xyz`, where T0+T2 gives **+25.1 %**
— and it holds up: it replicates on the 40 s window (116.20 → 85.05, +26.8 %), survives
κ-calibration (+18.0 % / +17.0 %) and uniform weights (+5.9 % / +12.7 %). Four configurations,
all positive.

**Miscalibration is not the explanation.** Replaying with κ set so each tier's ANEES → 6 does not
rescue the others: `fr1_360` T0+T2 still 33.63, `fr1_xyz` 3.59, `fr3_sitting` 5.76. The tiers are
simply 3–15× less accurate than Tier 0 and their errors are heavy-tailed. Fusing a good estimate
with a bad one, weighted by information, degrades the good one whenever the bad one's information
is not honestly small.

### 18.3 Degeneracy complementarity: real in shape, absent in error

The architecture's mechanism is that each tier fills another's null space. Half of that is true.

**Shape — yes.** Median scale-invariant fill of Λ_SPA into Λ_ECDA's eigenbasis is **34–740** on
every sequence (TCG→ECDA 8.8–15). SPA genuinely loads information exactly where ECDA is weakest.

**Error — no.** The error along Tier 0's *own reported* weakest axis, T0-only → fused, changes by
0.22×–1.02× — mostly worse. Selectivity (weak-axis gain ÷ orthogonal gain) is 0.65–1.23, i.e. no
selectivity at all.

**The decisive case.** `fr1_360` is the only sequence where Tier 0 actually reports rank < 6
(89 of 164 frames, mean DoF 5.29). On exactly those frames, adding SPA makes the error along that
exact axis **3.9× worse** (6.53e-3 → 2.55e-2 rad·m). SPA's own translation error there is 15.10 cm
at ANEES 25.5: a pure-rotation room sweep starves the depth map, which starves *both* tiers. **The
gap is correctly identified and then filled with a worse number.**

That yields the rule the data actually supports, which is narrower than the architecture's:

> **Fusion helps when the tiers' failure modes have different causes, and hurts when they share
> one.** `fr3_walking` works because moving people break photometry and not planes. `fr1_360`
> fails because thin depth breaks both.

Two reporting defects fell out. `observable_dof` at a *relative* `degeneracy_ratio` of 1e-3 still
reports 5 after the gap is genuinely filled, when the tiers are 10⁶ apart in scale. And the
solver's truncation threshold must be a numerical guard (1e-12), not the degeneracy threshold —
at 1e-2 the weak tier's contribution is discarded and fusion silently measures nothing.

### 18.4 The hand-designed `α_k(E)` is worse than no schedule at all

Uniform weighting beats it on **6 of 7 runs**, by up to 77 % (`fr1_360` 39.31 vs 48.78;
`fr3_sitting` 5.96 vs 9.46). The schedule wins once, by 1.74 %.

The cause is visible in the observed ranges: on clean indoor TUM, `α₀ ∈ [0.22, 0.88]` while
`α₁ ∈ [0.85, 1.00]` and `α₂ ∈ [0.49, 1.00]`. **The schedule down-weights the only accurate tier by
up to 4.5× on undegraded data**, driven by `texture_poverty` and `camera_shake` evidence. It is a
curve crossing the truth rather than tracking it — exactly what `04-unified-objective.md` §5.2
warned a hand-designed weighting would do.

### 18.5 What this does and does not say about §1

It does **not** refute §1. §1 is a claim about *degradation* — the gain growing from 6.8 % to
53.6 % as haze rises — and TUM has no degradation. Every sequence here is clear indoor.

What it does say is narrower and still uncomfortable: **at haze = 0, where the simulation
predicted +6.8 %, real data gives net harm on four of five sequences.** The simulation's
lowest-degradation point is the only one testable here, and it does not reproduce. Whether the
curve's *slope* is real remains untested, and testing it needs either degraded real data or an
honest admission that the 53.6 % is a property of the renderer's noise model.

---

## 19. The safety net was down, and every test report said green

### 19.1 What happened

Differential testing against the Python reference is this project's primary correctness
mechanism. It caught the chirality bug (§10), log-odds saturation, and four dead channels. It is
the reason most of the defects in this document were found at all.

`python/bindings/wme_module.cpp:333` read `DirectAlignerConfig::huber_delta`. That field was
removed when the adaptive Huber kernel landed (§11.4). The chain from there:

1. The pybind extension failed to compile — `error C2039: 'huber_delta': not a member`.
2. `wme/__init__.py` catches the resulting `ImportError` and sets `HAS_NATIVE = False`.
3. `test_differential.py` carries a module-level `pytest.mark.skipif(not HAS_NATIVE)`.
4. Every differential test reported **skipped**, which pytest prints as a dot-adjacent `s` and
   summarises in green.

The test suite reported `384 passed, 41 skipped` for a long stretch of work. **All 41 skips were
the differential suite.** The number was in every progress report and nobody asked what it
contained.

Second-order causes, each independently sufficient to hide it:
- On Windows, Python 3.8+ ignores `PATH` when resolving an extension module's DLL dependencies.
  `_core` needs `opencv_world4100.dll`, so even a correctly compiled binding would have failed to
  import — **also as an `ImportError`, also swallowed**. `conftest.py` now calls
  `os.add_dll_directory` (configurable via `WME_DLL_DIRS`).
- `pybind11` converts default arguments eagerly at registration time, so `ImageQualityEngine`
  being registered before `ImageQualityConfig` produced a module that compiled and then raised on
  import. A third swallowed `ImportError`.

### 19.2 What it cost

Three subsystems were corrected in C++ while the oracle stood still, and the mechanism that
exists to catch exactly that caught nothing:

| divergence | how long it went unnoticed |
|---|---|
| `ecda.py` still divided Λ by the photometric variance (§15's defect) | the entire §15 investigation |
| `constellation.py` still used the 0.85 score-ratio rule (§14's defect) | the entire TCG investigation |
| `confidence.py` still used the bounded static-evidence heuristic (§16.1) | the entire §16 investigation |
| `onObserved` squares the reliability through the binding | never previously observed |

**§15's 2 × 10⁴ overconfidence did not slip past the differential tests. Nothing measured it** —
ECDA had no differential test at all, before or after.

### 19.3 Why the suite could not have caught it even if it had run

Two of the three tests that should have covered these paths were incapable of failing:

- **Constellation**: every differential test built a *single-place* index. With one place there is
  no second candidate, so `query()`'s ambiguity branch is unreachable by construction. The branch
  had no test — not a weak one.
- **Confidence**: `test_static_belief_matches` used `dt = 0.1` against a corrected
  `static_min_dt` of 0.50. After syncing, the C++ side would return at the guard without
  updating, and the assertion would compare `0.5 == 0.5` for every displacement. **The test would
  have compared two constants and passed.** It also never set `is_agent`, so the `min(belief, 0.3)`
  clamp — the single largest behavioural difference — was outside its reach.

### 19.4 State now

Restored: the binding compiles, the DLL directory is resolved, registration order is fixed, and
the four remaining `ConstellationMatch` / `ConstellationConfig` diagnostic fields are exposed.
**46 differential tests execute** (4 skip, by name, for two still-unbound symbols). Python:
458 passed, 4 skipped, 1 xfailed.

The three ports were each validated by running the new test against the *pre-sync* Python and
showing the failure — not by asserting it would fail. Sample:

```
E   AssertionError: 한쪽만 기각: native=False reference=True     # the 36-of-36 rejection
E   AssertionError: meas_sigma=0.1 d=0.0: native 0.939 vs reference 0.681
E   assert np.isclose(0.982, 0.3, atol=1e-12)                    # the agent clamp
```

Porting also found a defect in the *oracle*: `_max_clique` iterated Python sets, and maximum
cliques are not unique — on seed 1 it returned a different 12-vertex clique from the C++ one
(rms 0.303 m vs 1.26e-15). Both loops now iterate `sorted(...)`.

And one more silent death, found only because the oracle went live: `wme/world/pipeline.py`
passed the simulator's 20 Hz frame `dt = 0.05` straight into `update_static`, whose window is
0.50 s. All 60 calls per sequence returned at the guard. Worse, the `dt` it passed came from the
last frame *that track was seen*, so judgements fired only for tracks lost for ≥ 0.5 s —
precisely the tokens whose association is least reliable.

### 19.5 The rule this earns

§10.4's rule was "verify the measurement discriminates." This is its complement, and it is the
one that cost the most:

> **A skipped test reports as green. A suite's skip count is a measurement, and it must be read
> like one.** If a number in the test summary has never been explained, it is not evidence of
> anything — and a swallowed `ImportError` converts a hard failure into a silent one.

`wme/__init__.py`'s `except ImportError: pass` is the specific line that turned a build error into
a green test run. Three separate failure modes all funnelled through it.

---

## 20. Three corrections to this document

Everything below overturns something an earlier section of this document asserted. They are kept
as corrections rather than edits because the *way* each one was wrong is the transferable part.

### 20.1 §17.2's causal direction was backwards

§17.2 reported belief-driven masking at **13 053 cm** on `fr3_sitting_halfsphere` — 571× worse
than not masking — and attributed it to association fragmenting under unfamiliar camera motion:
154 person tokens, median lifetime 5, belief pinned at its 0.155 prior.

The measurements were right. The arrow was not.

The decisive control was a new `--mask-mode observe`: run tokens, association and judgement
exactly as before, compute the mask, and **do not apply it** — so the pose is the healthy
mask-off pose. Same sequence, same sweep:

| | belief-mask (shipped) | observe (sound pose) |
|---|---|---|
| detections matched | 65.6 % | **95.1 %** |
| gated out by χ² | 16.1 % | 2.0 % |
| dead agent tokens | 139 (median 1) | **11 (median 5)** |
| mean agent `static_belief` | 0.165 | **0.365** (max 0.982) |
| agent stable landmarks / frame | 0 | **0.66** |

Under the identical camera motion, with a pose that has not diverged, association is as healthy
as on `sitting_xyz` (95.1 % vs 97.0 %). **The mask starved ECDA (130 points/frame against 1357,
141 of 259 alignments failing); the pose diverged; association fragmented as a consequence.**

§17.2 measured association *inside a run whose ATE was 130 m* — precisely the contamination
§17.5 flagged for `motion_noise_floor` one subsection earlier, on the same sequence. The warning
was written and then not applied to the section above it.

The maneuver hypothesis was refuted on its own terms too: with a sound pose the Mahalanobis
median is **0.1–0.2 against a gate of 9.0**, and agent and static-class innovations are the same
size, so the innovation is observation noise rather than object motion. A `κ·Δθ·range` gate term
was built for the rotation hypothesis, measured to be unnecessary, and **deleted rather than
shipped**.

**What actually fixed it — no new constants:**

- A **frame mismatch**: `associate()` added the measurement covariance in *camera* coordinates to
  a *world*-frame token covariance while gating a world-frame innovation, three lines from
  `fusePosition()` which does it correctly. σ_z is ~5× σ_lateral, so the depth uncertainty landed
  on the wrong world axis — invisible under `xyz`, live under a sweep. Pinned by a test that
  rotates the scene rigidly 60° and demands an identical Mahalanobis (fails 32.99 vs 49.74 when
  reverted).
- **No erasure without evidence**: `buildStaticMask` now refuses to erase a token whose static
  judgement has never run — erasing by an unjudged token's belief *is* class masking one level
  down. This makes the failure **bounded by construction**: if association collapses, nothing is
  judged, masking goes to zero, and the result degrades to mask-OFF instead of diverging.
- **Verdict transfer across a track break**: motion state belongs to what is at a place, not to a
  track ID. Restricting the source to *unmatched* tokens mattered — unrestricted, `sitting_halfsphere`
  went 8.05 → 20.80.

| sequence | OFF | class | belief (was) | **belief (now)** |
|---|---|---|---|---|
| fr3_sitting_halfsphere | **1.50** | 452.00 | 13 052.56 | **12.54** |
| fr3_walking_halfsphere | 46.08 | **8.65** | 10.55 | 10.77 |
| fr3_sitting_xyz | **1.01** | 15.84 | 1.63 | **1.03** |
| fr3_walking_xyz | 20.72 | **5.23** | 5.19 | 6.48 |

Independently reproduced. `sitting_xyz` now reaches 1.03 against a leave-them-alone optimum of
1.01. **`sitting_halfsphere` is 1041× better and still 8.4× worse than not masking** — the honest
remainder, and the token layer cannot close it: ATE there is **not monotone in masked area**
(16.3 % → 8.05 cm, 15.2 % → 12.54, 12.4 % → 20.80), so there is nothing to optimise against.
`walking_xyz`'s 5.19 → 6.48 is the measured price of the no-erasure-without-evidence rule: 3.43 pp
of its mask is the first 0.5 s of newly detected walkers, who have no predecessor to inherit from.

### 20.2 `ν = 1` was the right slope read off the wrong axis

§15 established by sub-sampling that the pose gains nothing from extra pixels, shipped
`effective_samples = 1.0`, and §17.1 found it breaking on a second camera (ANEES 38.8 / 58.6,
required ν 0.10–0.16 against a calibration range of 0.51–1.97).

**fr2 is the *cleanest* sequence in the set, and that is the entire cause.** Median residual 7.8
against 9.6–22.8; inlier ratio 0.67 against 0.31–0.56. The shipped model sets `var = N·rmse²`, so
a clean frame receives a small covariance — while its true error does not shrink in proportion.

Refitting `var = N^a·rmse^p` and sweeping p, scored by worst per-sequence ANEES deviation across
12 sequences:

| p | 0.00 | 0.50 | 1.00 | 1.50 | **2.00 (shipped)** |
|---|---|---|---|---|---|
| worst deviation | **2.43×** | 3.17× | 4.42× | 6.05× | **8.13×** |

Monotone. **Between sequences the achieved residual carries no information about pose error.**
*Within* a sequence it carries some (rank correlation +0.10…+0.52) — which is exactly why ν = 1
looked principled on one camera. The within-sequence signal was real and was extrapolated across
sequences where it does not exist.

`InformationModel::CoherentFrame`: `Λ = (H/N)/σ_c²`, σ_c = 15.5 intensity. Worst deviation
**9.76× → 2.46×**; spread 22.9× → 5.9×. fr1_xyz enters the band for the first time (9.4 → 6.2).
Accuracy bit-identical across all 12 trajectories. Leave-one-out σ_c stays 15.00–16.25 on every
fold. **Required σ_c spans 2.4× across 12 sequences and 3 cameras where required ν spanned 23×** —
still a fitted constant, but one that does not move when the camera does.

**Online estimation is impossible, and that is now measured rather than argued.** A three-frame
cycle (0→1, 1→2, 0→2) is the only ground-truth-free redundancy the pipeline can build. Its NEES
correlates with the true one at **Spearman −0.524** — the wrong sign — and it is *most* blind on
fr2 (cycle NEES 0.01 against a true 40.4, a factor of 3922). The dominant error is common-mode
across alignments of the same scene from nearby viewpoints (it lives in the depth map, the
intrinsics, and the RGB/depth registration, all shared), so it cancels identically in any cycle.
Sub-sampling cannot see it (§15.1), the cluster-robust sandwich cannot see it (§15.2), and cycles
cannot see it. **Determining σ_c online requires an external reference over the same motion.**

**Known cost, and it is unpriced.** `test_information_falls_under_motion_blur` now fails: under
blur the selector culls 698 → 131 points, `tr(H)` falls 2.6× but N falls 5.3×, so per-point
information — and hence Λ — *rises* 2.06×. **The residual term was buying degradation
monotonicity and removing it lost that.** Which cost is larger cannot be decided here: the 4×
cross-camera transfer gain is measured on real data; the degradation loss is measured only in the
renderer, because TUM has no degraded sequences (§18.5). Two different grades of evidence.

### 20.3 Every "1e-12" tolerance in the differential suite was really 1e-5

`np.isclose(a, b, atol=1e-12)` carries a default `rtol=1e-5`. Every tolerance written as exact in
the original 46 differential tests was five orders of magnitude looser than it read.

It hid a real divergence: `Observation::detection_conf` is `float` in C++ and was `double` in the
reference, a 6.6e-10 … 2.3e-9 difference in every belief — invisible at 1e-5, obvious at `rtol=0`.

The suite is now **76 differential tests, 0 skipped** (from 46/4). All 30 new ones were
demonstrated failing against a deliberately mutated reference before being accepted — 18
mutations, 18 failures, evidence recorded. Each also carries a **discrimination guard** that fails
if its inputs stop reaching the code under test, after §19.3's `dt = 0.1` vs `static_min_dt = 0.50`
trap. Two guards caught the agent's own tests being no-ops during development.

`toMat` in the binding had a genuine memory-safety hole: it used `strides[0]` as the `cv::Mat`
step and never checked the last axis. `cv::Mat` cannot express a gap between elements, and
OpenCV's `step >= minstep` assert catches F-order but *not* column slices, whose step is larger.
Measured: `evaluate(img[:, ::2])` returned 0.898 where the contiguous equivalent gives 0.974 — no
exception, wrong pixels.

`wme/__init__.py`'s swallowed `ImportError` — the §19.5 line — now records the cause, emits a
`RuntimeWarning`, and re-raises under `WME_REQUIRE_NATIVE`.

---

## 21. `α_k(E)` fitted at last — and the fusion failure is not a weighting failure

§18.2 measured the central equation on real data and found fusion net *harmful* on four of
five sequences; §18.4 found the hand-designed schedule losing to uniform weights on six of
seven runs. §26 carries "`α_k(E)` is hand-designed and has never been fitted to real
data" as an open item. It is now fitted, and the answer is that fitting it was never
going to help.

`tools/tum_fusion.cpp` records every tier's `(T, Λ)` **before** any weighting is applied, so
fusion can be re-run in numpy for an arbitrary rule without a camera or the C++ engine.
`python/tools/fusion_replay.py` is that port.

**The gate first.** The replay reproduces all **40** C++ ablation trajectories (5 sequences ×
8 ablations) to a worst relative disagreement of **0.002 %**, and a deliberately corrupted
weight moves the ATE on 5 of 5 — so the agreement is not the replay ignoring its input
(§10.4). Nothing below would mean anything without both halves of that check.

### 21.1 Three causes, and they are distinguishable

| | cause | what would confirm it |
|---|---|---|
| (a) | **weighting** — the tiers are usable, the schedule points the wrong way | some fitted `α` beats Tier-0-only |
| (b) | **robustness** — the tiers are usable *most* frames and catastrophic on a few | a per-frame consistency gate beats any constant |
| (c) | **no signal** — T1/T2 carry nothing this data can use | even an oracle holding ground truth cannot beat Tier-0-only |

The oracle is measured first because it *bounds the other two*: no observable rule can beat
one that already knows the answer.

### 21.2 The oracle has almost no headroom

Per frame, picking the best of seven configurations by true error:

| sequence | T0 err | oracle err | headroom | oracle picks T0 alone |
|---|---|---|---|---|
| fr1_360 | 10.31 mm | 10.27 mm | 1.00× | 89 % |
| fr1_desk | 7.59 | 6.77 | 1.12× | 82 % |
| fr1_xyz | 5.29 | 4.55 | 1.16× | 69 % |
| fr3_sitting_xyz | 5.54 | 4.80 | 1.15× | 57 % |
| fr3_walking_xyz | 16.39 | 12.91 | 1.27× | 56 % |

**1.15× median, with ground truth in hand.** Whatever the tiers contribute per frame, it is
small. Note also that the oracle *lowers* per-frame error and **raises ATE** on `fr1_xyz`
(1.56 → 2.67) and `fr3_sitting` (1.01 → 1.29) — the first sign that per-frame magnitude is
the wrong objective, which §21.5 turns into the section's main finding.

### 21.3 No constant weighting helps — the fit's answer is "switch them off"

Grid over the two free ratios (fusion is invariant to overall weight scale, so `α₀ ≡ 1`),
leave-one-sequence-out, ties called at 2 %:

| held out | fitted (α₁, α₂) | held-out ATE | T0 ATE | change |
|---|---|---|---|---|
| fr1_360 | (0.001, 0.01) | 10.47 cm | 10.37 | +1.0 % tie |
| fr1_desk | (0.001, 0.01) | 2.53 | 2.53 | −0.2 % tie |
| fr1_xyz | (0.0, 0.01) | 1.55 | 1.56 | −0.2 % tie |
| fr3_sitting_xyz | (0.01, 0.001) | 1.05 | 1.01 | +3.7 % **worse** |
| fr3_walking_xyz | (0.001, 0.001) | 20.72 | 20.72 | −0.0 % tie |

**0 wins, 4 ties, 1 loss.** Every fitted point sits at `α_k ≤ 0.01` — a 100–1000× suppression.
The fit is not choosing a schedule; it is choosing to not fuse, and the ties are that choice
succeeding. **Cause (a) is falsified**: the schedule was never the binding problem, so §18.4's
"uniform beats the schedule" was measuring the smaller of two bad options.

### 21.4 A one-parameter observable gate wins 4 of 5

The rule: drop a tier whose disagreement with Tier 0 exceeds **its own stated covariance** —
`ε = log(T_k · T₀⁻¹)`, reject if `εᵀΛ_k ε` exceeds a χ²(6) quantile. No ground truth, no
environment model, one parameter, and that parameter is a quantile rather than a tuned number.

| gate | fr1_360 | fr1_desk | fr1_xyz | fr3_sitting | fr3_walking | T1 pass | T2 pass |
|---|---|---|---|---|---|---|---|
| T0 only | 10.37 | 2.53 | 1.56 | **1.01** | 20.72 | 0 % | 0 % |
| p = 0.05 | 9.44 | 2.53 | 1.49 | 1.85 | 20.70 | 15 % | 44 % |
| p = 0.25 | 7.80 | 2.67 | 1.44 | 1.73 | 19.26 | 32 % | 57 % |
| **p = 0.50** | **7.36** | **2.38** | **1.45** | 1.88 | **18.44** | 48 % | 65 % |
| p = 0.95 | 10.37 | 2.38 | 2.62 | 2.48 | 19.14 | 75 % | 79 % |
| no gate | 39.31 | 2.94 | 3.27 | 5.96 | 19.49 | 100 % | 100 % |

Unimodal in the threshold, which is the signature of a tail problem: too loose admits the
catastrophes, too tight admits nothing. The pass rates matter as much as the ATEs — at 48 %
and 65 % this is a gate that genuinely fires both ways, not a degenerate one reproducing
Tier-0-only under another name.

Leave-one-out on the threshold selects **p = 0.50 on every one of the five folds**:

| held out | held-out ATE | T0 ATE | change |
|---|---|---|---|
| fr1_360 | **7.36 cm** | 10.37 | **−29.0 %** |
| fr1_desk | 2.38 | 2.53 | −6.3 % |
| fr1_xyz | 1.45 | 1.56 | −6.6 % |
| fr3_walking_xyz | 18.44 | 20.72 | −11.0 % |
| fr3_sitting_xyz | 1.88 | 1.01 | **+87.0 %** |

**4 of 5, held out.** Fusion goes from net harmful to net helpful without touching a weight
schedule. **Cause (b).**

`fr3_sitting_xyz` is a clean loss and stays one. It is also the sequence where Tier 0 alone
is already at 1.01 cm — the gate costs most where there was least to gain.

### 21.5 It works, and not for the reason the architecture claims

§18.3 found fusion identifying Tier 0's weak axis and filling it with a worse number.
Gating does not repair that.

| sequence | selectivity (gated) | rank-deficient frames | weak-axis error |
|---|---|---|---|
| fr1_360 | 1.135 | 88 | 6.53e-3 → 5.99e-3 (**1.09×**) |
| fr1_desk | 1.026 | 0 | — |
| fr1_xyz | 1.120 | 0 | — |
| fr3_sitting_xyz | 1.139 | 0 | — |
| fr3_walking_xyz | 0.991 | 0 | — |

Selectivity **0.99–1.14** — still no selectivity, against §18.3's ungated 0.65–1.23. On
`fr1_360`'s 88 genuinely rank-deficient frames the weak axis improves by 1.09×, where ungated
it got 3.9× *worse*. **Gating removes the harm; it does not deliver complementarity.**

So where does 29 % come from? Not from per-frame accuracy — that gets *worse or unchanged at
every quantile*:

| fr1_360, per-frame rel. translation error | median | p95 | p99 | max |
|---|---|---|---|---|
| T0 → gated | 10.3 → **12.6** mm | 40.8 → 40.0 | 70.6 → 70.6 | 81.0 → 81.0 |

That refuted the tail hypothesis on its own data. The gain is not in magnitude at all — it is
in **sign**, which is the only component a pose chain accumulates:

| sequence | rms \|e\| (mm) | \|mean e\| (mm) | ATE |
|---|---|---|---|
| fr1_360 | 19.89 → 21.19 (+6.5 %) | 3.391 → **3.020** (−10.9 %) | −29.0 % |
| fr1_desk | 9.89 → 10.40 (+5.2 %) | 1.191 → **0.933** (−21.7 %) | −6.3 % |
| fr1_xyz | 8.28 → 8.43 (+1.8 %) | 1.265 → **1.122** (−11.3 %) | −6.6 % |
| fr3_sitting_xyz | 9.05 → 11.44 (+26 %) | 1.538 → 1.750 (+13.8 %) | +87.0 % |
| fr3_walking_xyz | 29.07 → 29.09 (+0.1 %) | 0.944 → 1.451 (+53.7 %) | −11.0 % |

**The sign of the ATE change matches the sign of the bias change on 4 of 5 sequences and the
sign of the rms change on 1 of 5.** Gated fusion buys accuracy in the statistic that
accumulates by spending it in the statistic that does not.

`fr3_walking_xyz` does not fit: its bias rises 54 % and its ATE still falls 11 %. It is also
the least biased sequence in the set (bias fraction 0.032 against 0.12–0.17), so its ATE is
dominated by something else — the moving people of §13.1. **That row is unexplained and is
not counted as support.**

### 21.6 The rule this earns

> **For odometry, a per-frame estimator-selection criterion must target the signed component
> of the error, not its magnitude.** ATE integrates a chain; only the signed part survives the
> sum. Selecting on magnitude optimises a statistic that largely cancels.

The sharpest evidence is not the gate but §21.2's oracle: a selector holding **ground truth**,
minimising per-frame error exactly, **raises** ATE on two of five sequences. It is not a weak
approximation to the right rule — it is an exact solution to the wrong one. This is §10.2's
quantity-mismatch pattern (the model predicting a different quantity than the one that matters)
appearing for the seventh time, and the first time in an *objective* rather than a sensor model.

### 21.7 What this does and does not settle

**Settled.** `α_k(E)` has been fitted on real data and does not help; the schedule was not the
binding problem. The §18.2 damage is a robustness failure and an observable one-parameter gate
repairs most of it, held out.

**Not settled.** The gate anchors on Tier 0 — defensible here because §18.1 measured Tier 0 as
3–15× the more accurate tier, but that ordering is an input to the rule, not an output of it,
and nothing establishes it on data where Tier 0 is the weak one. `Λ_TCG` is up to 47×
overconfident (§18.1), so the nominal `p = 0.50` is far tighter in truth than it reads; the
threshold is not a calibrated probability and should not be reported as one. Five 9-second
windows, one dataset. And **the architecture's actual mechanism — tiers filling each other's
null spaces — remains unsupported on real data**: §18.3 measured it absent, and §21.5 shows
that repairing fusion did not bring it back.

---

## 22. The first external baseline — 12 sequences, head to head

> ### ⚠ Every ATE in this section was measured on a fraction of its sequence.
>
> Thirteen of the sixteen TUM sequences were truncated on disk while their frame indexes stayed
> intact, so each run scored a prefix — an average of 35 %, as little as 6.4 % — and reported it
> as a whole-sequence result. **§25.22 has the corrected numbers**, the coverage table, and the
> five verdicts that flip (in both directions). The section below is left as written because what
> it concluded from the bad data is part of the record; read it against §25.22, not instead of it.
>
> What survives unchanged: the RPE-versus-ATE mechanism in §22.1, which holds on 17 of 17
> scorable sequences instead of 12 of 12. What does not: the 12–4 tally, the fr2 claim in §22.2,
> and the per-sequence numbers in §22.4's table.

Every number before this section compares WME to ablations of itself and to a do-nothing
floor. §22 in its earlier form named that as the largest remaining gap. It is now closed on
one axis: a classical descriptor pipeline runs on the same data, through the same code path.

**What the baseline is.** ORB → Hamming descriptor matching (Lowe ratio 0.75) → RANSAC PnP →
LM refinement on the inliers. 1000 features, 8 pyramid levels, i.e. ORB-SLAM3's own front-end
settings (`tools/tum_baseline.cpp`). It shares intrinsics, distortion coefficients,
undistortion maps, depth scale, rgb↔depth association and the keyframe rule with
`wme_tum_odometry` — every one of those is a confound if it differs.

**What it is not.** Not ORB-SLAM3: no loop closure, no local map, no bundle adjustment. This
is odometry versus odometry, which is the only comparison WME is currently entitled to since
it has no loop closure either. 05-research-program.md §2 warns that a self-implemented
baseline is usually under-tuned; the mitigation here is published settings and a health check
— ~999 keypoints, ~455 PnP inliers per frame, zero tracking losses on static sequences.

**16 sequences**, `python/tools/bench_run.py`, no per-sequence tuning. Two of them are
unscorable — one system diverged or produced nothing, and §22.5 explains why those are labelled
rather than counted.

Against the ORB control alone the tally is **12 to 4 across 14 scorable sequences**. That number
is *not* the headline, because §22.4 adds a second, independent control and the verdict moves to
**9–6**. The full three-way table is there; read that one.

What survives both controls is the shape rather than the margin:

| regime | verdict |
|---|---|
| dynamic content (`walking`, `sitting_xyz`) | **WME, decisively** — 6.48 cm against 19.34 / 52.89 |
| long traverse with drift (`fr1_room`, 45 s) | **WME** — 21.17 against 48.06 / 72.29 |
| clean short static (`fr2_desk`) | control — 0.98 against 2.38 |
| textureless structure | control — 6.82 against 12.20 |

Competitive, not dominant — the outcome 05-research-program.md §6 committed to in advance as
the likely one, and reporting it as a split is the point of having written that down first.

### 22.1 WME wins per-frame accuracy 12 of 12 and still loses ATE on 4

| | median RPE | median ms/frame |
|---|---|---|
| ORB + PnP | 6.91 mm | 29.9 |
| WME (Tier 0) | **4.62 mm** | 35.5 |
| WME + token mask | 4.82 mm | 108.6 |

The relative-pose error is lower for WME on **every sequence**, including all four it loses on
ATE. `fr2_desk` is the sharpest case: RPE 1.67 mm against the baseline's 2.35, and ATE 2.38 cm
against 0.98 — **1.4× more accurate per frame and 2.4× worse over the trajectory.**

That is §21.6's rule, now measured against an external system rather than against WME's own
ablations:

> ATE integrates a chain; only the signed part of the error accumulates. A smaller per-frame
> error with larger drift means the residual is more *signed*, and magnitude and sign are
> different statistics.

The classical pipeline's errors are noisier and less correlated — RANSAC re-selects a different
inlier set every frame, so its bias resamples — while ECDA's residual is a smooth function of
the same scene geometry and its bias persists across frames. **The mechanism that makes WME
more accurate frame-to-frame is the same one that makes it drift more.** That is a
structural finding about direct versus feature-based methods, and it is the most useful thing
this comparison produced.

### 22.2 The two regimes the split falls into

**Where WME wins, it wins on dynamic content and on texture.** `fr3_sitting_xyz` at 8.79× is
the largest margin in the table: seated people break descriptor matching (the baseline loses
33 frames of tracking on `walking_xyz`) while ECDA's dense residual absorbs them. Both
`walking` sequences need the token mask to win, which is that layer being paid for.

**Where the baseline wins, it wins on fr2 and on rotation.** `fr2_desk` and
`fr2_desk_with_person` are the two cleanest sequences in the set, and §20.2 already identified
fr2 as the camera where WME's own uncertainty model breaks. `fr3_sitting_halfsphere` is the
sequence §20.1 leaves as the open remainder. **The baseline wins exactly where WME's own
documented weaknesses are**, which is a consistency check on the previous twenty sections
rather than new information.

### 22.3 The viewer

`python/tools/bench_report.py` emits a self-contained page (`results/bench/index.html`) —
left panel the classical pipeline, right panel WME, trajectories Umeyama-aligned to ground
truth with the *same* alignment the ATE is scored with, projected onto the ground truth's own
principal axes so no system gets a flattering viewpoint. Estimation is C++, scoring is Python,
and the two were kept apart deliberately.

### 22.4 A third-party control, and the structure/texture quartet

Two additions close the two weakest points of §22 as originally written.

**A baseline nobody here wrote.** `cv2.Odometry` (OpenCV 5, `ODOMETRY_TYPE_RGB_DEPTH`) is a
published dense RGB-D odometry with no connection to this repository, run through the identical
data path (`python/tools/baseline_cv2.py`). It is the *same family* as WME's Tier 0 — dense,
direct, photometric — which makes it the more informative control: it separates "this particular
direct method" from "direct methods in general".

ATE cm, all 16 sequences, best WME variant against the better of the two controls:

| sequence | ORB+PnP | **cv2.Odometry** | WME | WME+mask | winner |
|---|---|---|---|---|---|
| fr1_360 | 13.62 | 11.52 | **10.37** | 10.37 | WME |
| fr1_desk | 3.55 | 2.89 | **2.53** | 2.53 | WME |
| fr1_plant | 3.37 | **2.03** | 4.51 | 4.61 | control |
| fr1_room | 48.06 | 72.29 | 21.72 | **21.17** | WME |
| fr1_teddy | 6.22 | **4.30** | 5.30 | 5.30 | control |
| fr1_xyz | 2.61 | 2.78 | 1.56 | **1.53** | WME |
| fr2_desk | **0.98** | 21.14 | 2.38 | 2.38 | control |
| fr2_desk_with_person | **0.73** | 16.74 | 0.94 | 0.94 | control |
| fr3_nostructure_notexture | 146.26 | 78.57 | *no output* | *no output* | — |
| fr3_nostructure_texture | 34.12 | 20.60 | 7.74 | **7.21** | WME |
| fr3_sitting_halfsphere | **1.26** | 1.32 | 1.50 | 12.54 | control |
| fr3_sitting_xyz | 8.85 | 2.59 | **1.01** | 1.03 | WME |
| fr3_structure_notexture | *diverged* | **6.82** | 12.20 | 12.20 | control |
| fr3_structure_texture | 3.30 | 2.91 | **1.95** | 7.73 | WME |
| fr3_walking_halfsphere | 17.86 | 89.07 | 46.08 | **10.77** | WME |
| fr3_walking_xyz | 19.34 | 52.89 | 20.72 | **6.48** | WME |

**Two things follow, and the second is uncomfortable.**

**The hand-written control is not crippled.** ORB and the published implementation land in the
same region on static scenes (2.61/2.78, 3.55/2.89). That is the evidence §22.6 said it lacked.

**And WME's margin shrinks once there are two controls.** Against ORB alone the tally was 12–4;
against the *better of two independent controls* it is **9 to 6, with one sequence unscorable**.
A published dense odometry beats WME on five sequences — including `structure_notexture`, where
descriptors starve and the direct method should be at its strongest: cv2 gets **6.82 cm against
WME's 12.20**. Losing the textureless-structure case to another direct method is a loss on the
architecture's own home ground, and it is not explained by anything measured here.

Where WME's advantage *is* decisive is dynamic content: `walking_xyz` 6.48 against 19.34 and
52.89, `walking_halfsphere` 10.77 against 17.86 and 89.07, `sitting_xyz` 1.01 against 8.85 and
2.59. Both controls lack dynamic handling, and that is exactly what the token layer supplies.

> **A single control flatters.** Every number in §22.1–§22.3 was measured against one baseline
> and read as a broad win; a second, independent implementation moved the verdict from 12–4 to
> 9–6 without a single WME number changing. The count of controls is itself an experimental
> parameter.

**The structure/texture quartet.** Four `fr3` sequences vary exactly the two things the tier
architecture is built around, and they were fetched specifically because §18.3's complementarity
claim had never had the right data:

| sequence | ORB+PnP | WME | floor | what it isolates |
|---|---|---|---|---|
| structure + texture | 3.30 | **1.95** | 130.93 | both cues present |
| nostructure + texture | 34.12 | **7.21** | 139.61 | texture only — direct method's home |
| structure + notexture | **diverged** | **12.20** | 123.62 | planes only — descriptors starve |
| nostructure + notexture | 146.26 | **no output** | 87.90 | neither cue — both must fail |

The two middle rows are the cleanest statement of the thesis in this document. With texture and
no structure, WME is **4.7×** better. With structure and no texture — the case descriptors cannot
serve — **ORB diverges entirely** while WME holds 12.20 cm.

### 22.5 Two results that were not results, caught by the same rule

Both anomalies above were numbers before they were failures, and §10.4 is why they are labelled
now.

**`structure_notexture_far`, ORB: ATE 9.2 × 10¹³ cm.** Not a bad score — a divergence. And the
*way* it diverges matters: it loses tracking on only **5 %** of frames while running on a median
of 86 keypoints and **24 PnP inliers**, just above its own `min_inliers = 10` gate. It accepts
its own garbage and integrates it into a 3.3 × 10¹² m path. **This is §23.3's silent failure, with
the systems reversed** — under haze the descriptor pipeline stops and says so while WME drifts
confidently; on textureless structure ORB is the one that fails confidently. Neither system has a
reliable failure detector; each merely has a *different* regime in which it lies.

**`nostructure_notexture_far`, WME: ATE 87.90 cm against a floor of 87.90 cm.** Identical to two
decimals, because the trajectory *is* the identity trajectory — path length 0.000 m, zero rows in
the diagnostic CSV. The engine produced nothing at all, and doing nothing scores exactly the
do-nothing floor. In a table of numbers it reads as "beat ORB's 146.26 by 1.66×".

`run_status()` in `bench_run.py` now flags both classes — `no_output` (path length zero) and
`diverged` (any position beyond 1 km in an indoor sequence) — and the viewer prints the label
instead of the number and excludes those sequences from the win count. **Winning against a
failure is not winning**, and the honest tally is therefore 12–4 across 14 *scorable* sequences,
not 16.

> The rule earned here is narrower than §10.4's and worth stating separately:
> **a result that exactly equals its own control is a signal, not a measurement.** The floor is
> what a system scores when it does nothing, so matching it to two decimals is the strongest
> available evidence that nothing is what it did.

### 22.6 What this still does not establish

- **Not ORB-SLAM3.** A real ORB-SLAM3 run has loop closure and BA and would beat both columns
  on ATE wherever a loop exists. This section measures front-ends. **§24 closes the loop-closure
  half of that gap** with a back-end shared by both systems — and the answer changes: over a
  full 45 s traverse WME's front-end is 2.2× better, and the classical *loop closure* is the
  thing that wins. Bundle adjustment is still absent on both sides.
- **A self-implemented control can always be suspected of being under-tuned.** Published
  settings and the inlier health check are the defence, not a proof.
- **One dataset family, short windows, no EuRoC/KITTI/Robotcar**, and no degraded conditions —
  so §1's degradation claim is still untested against anything external.

---

## 23. Degradation on real data — neither system degrades gracefully

§18.5 left the headline claim in a specific state: §1's gain curve (6.8 % → 53.6 % as haze
rises) is simulation, and *"whether the curve's slope is real remains untested, and testing it
needs either degraded real data or an honest admission that the 53.6 % is a property of the
renderer's noise model."* TUM has no degradation. So it was manufactured.

### 23.1 The degradation is the scattering equation, not a picture of fog

`tools/tum_degrade.cpp` applies

```
I_hazy = I·t + A·(1 − t),      t = exp(−β·d)
```

where **`d` is TUM's measured depth map**. A renderer knows `d` and real data usually does
not — RGB-D is the exception, and that is the whole reason this is worth doing. The
transmission is not a plausible-looking approximation of scattering; it is the scattering
equation evaluated on a real sensor's geometry.

Verified rather than asserted — measured contrast reduction against predicted transmission on
`fr1_xyz`:

| β | predicted `exp(−βd̄)` | measured contrast ratio |
|---|---|---|
| 0.15 | 0.892 | 0.894 |
| 0.30 | 0.796 | 0.800 |
| 0.50 | 0.683 | 0.691 |
| 0.80 | 0.544 | 0.556 |

Within 2 % at every level. Depth is left untouched: the haze is optical and the IR depth
sensor is a different device, so degrading both would make the cause unrecoverable.

The other two channels were verified the same way rather than assumed:

| channel | control | result |
|---|---|---|
| **dark** (gain 0.15) | noise σ and mean, after AGC re-gain | mean unchanged 142.4, σ **1.30 → 3.49** |
| **blur** (exposure 0.02→0.2 s) | frames blurred, of 250 | **51 → 201 → 228 → 242**, monotone |

Darkness raising noise at *constant* mean is the point: after auto-exposure re-gain, what
darkness costs is signal-to-noise, not brightness. Blur is gated on each frame's true
ground-truth velocity, so slow frames are correctly left alone.

> The blur check first read as a total no-op — one frame, byte-identical to the original. It was
> the *frame* that was wrong: its instantaneous speed put the kernel under the 2-pixel floor.
> §10.4, on my own control, two sections after writing it up.

> The first version of this check paired `rgb[10]` with `depth[10]` **by index**, reported a
> far-field contrast ratio of 0.90 where physics demands 0.09, and looked like a defect in the
> model. The frames were 17 ms apart but the indices did not correspond. §10.4 again: the
> measurement was wrong before the thing measured was.

### 23.2 WME degrades *faster* than the descriptor pipeline

ATE in cm, `python/tools/bench_degrade.py`, identical configuration throughout:

| β | ORB+PnP (`fr1_xyz`) | WME | ORB+PnP (`fr1_desk`) | WME |
|---|---|---|---|---|
| 0.0 | 2.61 | **1.56** | 3.55 | **2.53** |
| 0.15 | 3.18 | **1.94** | 3.57 | **2.50** |
| 0.30 | 2.51 | **1.79** | 4.60 | **3.26** |
| 0.50 | 10.83 | **9.58** | 13.70 | **10.05** |
| 0.80 | **6.96** | 44.93 | 18.24 | **11.20** |
| 1.5 | **15.30** | 104.71 | **24.35** | 68.14 |
| 2.5 | **26.76** | 87.86 | **35.38** | 5433.06 |
| 4.0 | **17.66** | 199.10 | **45.71** | 130.59 |

WME holds its lead to β ≈ 0.5 and then collapses. Normalised to each system's own clean
performance, WME reaches **128×** its β=0 error on `fr1_xyz` where the baseline reaches 6.8×.

**This is the opposite of the architecture's intuition.** Dropping descriptors was supposed to
buy robustness to exactly this.

### 23.3 The baseline's apparent robustness is it giving up

The table above flatters the baseline, and the control that shows it is the tracking-loss rate
next to the do-nothing floor:

| β | ORB lost % | ORB ÷ floor | WME lost % | WME ÷ floor |
|---|---|---|---|---|
| 0.5 | 0 % | 0.63× | 0 % | 0.56× |
| 0.8 | 16 % | 0.40× | 0 % | 2.61× |
| 1.5 | 64 % | **0.89×** | 0 % | 6.08× |
| 2.5 | 82 % | **1.56×** | 0 % | 5.11× |
| 4.0 | **89 %** | **1.03×** | 0 % | 11.57× |

At β = 4 the baseline has lost tracking on 89 % of frames and holds the previous pose, so its
trajectory *is* approximately a frozen camera — and it scores 1.03× the frozen-camera floor,
because that is what it has become. Its 17.66 cm is not tracking; it is a well-behaved
surrender.

WME loses tracking on **0 %** of frames at every level and drifts to 11.6× the floor. It never
stops answering.

> **Neither system degrades gracefully. They fail differently: the descriptor pipeline fails
> loudly and stops, the direct method fails silently and continues.** Under haze WME becomes
> *worse than doing nothing* while reporting no failure at all.

This is §13.3 — "the estimator does not merely fail on moving objects, it fails *confidently*"
— reproduced under controlled degradation and against an external system. It was the sharpest
result in this document when it was about dynamic content; it is now a general property.

For a robot the distinction is not academic. A front-end that says "I am lost" can trigger
relocalization; one that returns a confident wrong pose cannot, and §13.3 already established
that no photometric self-assessment signal detects it.

### 23.4 §1's slope, measured on real data at last

§23.2 compares WME's Tier 0 to an external baseline. §1 is a different claim — the *fusion*
gain, T0+T1+T2 against T0 alone, growing with degradation. Running the full ablation set across
the same haze sweep is the test the document has been missing since §18.5 named it.

ATE cm, `fr1_xyz`, all three tiers live, `results/degrade_fusion`:

| β | T0 | T0+T1 | T0+T2 | all 3 | uniform | best gain vs T0 |
|---|---|---|---|---|---|---|
| 0.0 | **1.56** | 2.93 | 2.90 | 3.47 | 2.32 | **−49.3 %** |
| 0.3 | **1.79** | 3.35 | 2.78 | 3.67 | 2.63 | **−47.2 %** |
| 0.8 | 44.93 | 44.29 | 44.42 | **43.73** | 44.45 | +2.7 % |
| 1.5 | **104.71** | 104.71 | 109.54 | 109.54 | 110.73 | +0.0 % |
| 2.5 | 87.86 | 87.86 | **78.18** | 78.18 | 83.01 | +11.0 % |

**The sign of the slope is real.** Fusion goes from ~48 % *harmful* at zero haze to mildly
helpful at high haze — monotone in the right direction, with no weather conditional anywhere in
the fusion code, exactly as the mechanism claims. §18.2's "net harmful on four of five" is
reproduced at the clean end, and the direction §1 predicted is reproduced at the degraded end.

**And the claim is deflated by an order of magnitude.** §1 predicted +6.8 % at zero haze and
**+53.6 %** at the far end. Real data gives −49 % and **+11 %**. Worse, the gain only becomes
positive *after Tier 0 has already diverged*: at β = 0.8 the choice is between 44.93 cm and
43.73 cm, and at β = 2.5 between 87.86 and 78.18. Both are useless poses. **Fusion starts
helping at precisely the point where nothing is working**, which is not what "graceful
degradation" was supposed to mean.

The mechanism is visible in §23.3: a weighted sum of information matrices can only reallocate
away from a tier that reports its own failure, and Tier 0 does not — it loses tracking on 0 % of
frames while drifting to 11× the do-nothing floor. `α_k(E)` reads the *environment*; it cannot
read the estimator's own silent divergence. That is §21.4 arriving from a second direction:
**what fusion needs is a consistency gate, not a weight schedule** — and this time the evidence
comes from the degradation axis the schedule was designed for.

> **Verdict on §1.** The qualitative claim survives: fusion's value does increase as conditions
> degrade, mechanically, without a weather branch. The quantitative claim does not: 53.6 %
> becomes 11 %, and it arrives too late to rescue the trajectory. The 53.6 % is substantially a
> property of the renderer's noise model, which is the second of the two outcomes §18.5 named
> in advance.

---

## 24. Loop closure — the back-end gap closed, and TCG fails at it

§22.4 conceded the sharpest limitation of the baseline comparison: it compares *front-ends*.
A real ORB-SLAM3 has loop closure and bundle adjustment, and nothing here did. That gap is now
closed symmetrically.

**The back-end is identical for both systems** (`tools/tum_loopclose.cpp` +
`python/tools/loop_optimize.py`): same keyframe rule, same candidate proposal (temporally
distant, spatially near under the current estimate), same factor graph, same Huber kernel, same
solver, same information constants. Exactly one thing differs — how a proposed loop is
*verified*:

| | verification |
|---|---|
| classical | ORB descriptor matching + RANSAC PnP |
| WME | object-constellation matching + Kabsch (Tier 1) |

If the back-end differed too, the final ATE difference could not be attributed to recognition.

Measured on **`fr1_room`, the full 45.4 s / 1362 frames / 17.5 m traverse** — the earlier
sections used 9-second windows, which contain no revisits at all and therefore cannot test this.

### 24.1 Result

> **This section's WME row was wrong, and the error was mine.** It reported TCG loop closure
> *degrading* ATE by 11.2 % on a single false edge. Two defects in `tum_loopclose.cpp` produced
> that: chirality was never enabled (no gravity passed to `insert`/`query`, so one of the
> signature's three components was switched off), and **`ConstellationMatch::transform` was
> assigned with its direction inverted** — it is query→place, and the code stored it as
> place→query. Corrected, the sign of the result flips. §25.12 has the full account; the table
> below is the corrected version.

| | odometry ATE | + loop closure | loops used | loop-edge accuracy |
|---|---|---|---|---|
| ORB + PnP | 48.06 cm | 20.67 cm (**+57.0 %**) | 44 of 1292 proposed | median **2.14 cm**, 1.18° |
| **WME (TCG)** | **21.72 cm** | **15.06 cm** (**+30.7 %**) | 3 of 25 queries | median **15.4 cm**, 5.9° |

Two findings, and they point opposite ways.

**WME's front-end is 2.2× better.** 21.72 cm against 48.06 cm on a full-length sequence where
drift has room to accumulate. §22's 8–4 split was measured on 9-second windows; over 45 seconds
the direct method's advantage widens.

**Both loop closures work, and they work differently.** ORB recovers 57 % of its drift from
**44** closures accurate to 2.14 cm median — high recall, high precision, 1 gross false in 44.
TCG accepts **3** and recovers 30.7 % from them, at 15.4 cm median accuracy with 1 gross false
in 3. The descriptor pipeline finds an order of magnitude more loops and locates them 7× more
precisely; the constellation finds few and is content to decline, which is §2's stated design.

**Net: the WME full system (15.06 cm) beats the classical full system (20.67 cm) by 27 %** — but
the reason is the front-end, not the back-end. WME starts from 21.72 cm where ORB starts from
48.06, and ORB's far better loop closure closes most but not all of that gap.

### 24.2 TCG's real limit is recall, not correctness

Of 92 keyframes only **25** could be queried at all: the rest had fewer than 4 detected objects,
and `min_nodes = 4` is structural. §2 said it plainly — *"Needs ≥ 4 well-localised objects.
Fails in an empty corridor."* An office is closer to an empty corridor than the architecture
assumed, because YOLO's 80 COCO classes cover very little of one.

What that costs is **recall**: 3 accepted loops against ORB's 44, from 25 possible queries
against 1292 proposed pairs. What it does not cost is correctness — the accepted edges land at
15.4 cm median, an order of magnitude worse than ORB's 2.14 cm but not wrong places.

This is the first measurement of TCG in a real loop-closure loop. It says the constellation
approach **cannot yet replace** descriptor place recognition indoors — it finds 15× fewer loops
and locates them 7× less precisely — not that it fails. §2's synthetic results (300 distractor
places, correct retrieval, aliased corridors rejected) were measured where objects were
guaranteed to exist, and the gap between that and an office is object density.

> **§25.10 proposed accumulating the query over a temporal window to fix the density, and §25.11
> measured it making things worse**: coverage rises 37 % → 97.8 % but edge accuracy degrades from
> 15.4 cm to 78 cm, because a window accumulates association and drift error into the node
> positions that the distance spectrum is computed from. More nodes, worse constellation.

### 24.3 The mistake I made first, which is §19.3's exact trap

The first version built a **single-place index per candidate pair**: insert keyframe *a*, query
keyframe *b*. It reported **9 accepted loops, 7 of them gross false positives (>50 cm)**, and
that number was about to be published as "TCG's accept rule produces 78 % false positives."

It is wrong, and §19.3 already wrote down why:

> *"every differential test built a single-place index. With one place there is no second
> candidate, so `query()`'s ambiguity branch is unreachable by construction."*

TCG's rejection mechanism works by comparing a candidate against its rivals — `rival_mass`,
`pose_margin`, `agree_count`. With one place in the index there are no rivals, so the mechanism
cannot fire. I had measured a version of TCG with its main safety mechanism switched off.

Corrected to the real protocol — one index accumulating every past place outside the time gap —
acceptances fall **9 → 1**. The rejection rule works; it is doing exactly what §2 claims,
declining rather than guessing. What remains is that it declines *almost everything*, and the
one thing it accepted was still wrong.

**The same document that recorded this trap did not prevent me from walking into it.** Writing
a failure mode down is not the same as having a control that fires when you repeat it; the only
reason it was caught is that 78 % false positives contradicted §2 loudly enough to re-read the
protocol.

---

## 25. The failure detector — a bounded signal cannot report an unbounded failure

§26 names this the most valuable next step, and it comes from two measurements that agree:
§13.3 (every photometric self-assessment blind on dynamic content) and §23.3 (under haze the
engine drifts to **11.6× the do-nothing floor while reporting zero failed frames**). Every
downstream claim — fusion weighting, relocalization, replanning — assumes the estimator knows
when it is lost.

§13.3 also said where not to look: *"No amount of tuning the photometric confidence path can fix
this, because the information is not in the photometric channel."* So the search was for a
different channel.

### 25.1 The candidate, and why it is independent

Project ref's 3D points into cur with the estimated pose and compare against **the depth cur
actually measured**. ECDA uses ref's depth for geometry and cur's *intensity* for the residual —
it never reads cur's depth. The comparison is therefore an observation that took no part in the
estimate, which is what makes it able to contradict it. Haze degrades the optics and leaves the
IR depth sensor untouched (§23.1), so it survives exactly the regime that blinds photometry.

Shipped on `AlignmentResult` as `depth_consistency` (median `|z_pred − z_meas| / z_meas`) and
`depth_outlier_ratio`. `-1` means *not judged* — never 0, which would be indistinguishable from
a perfect match.

### 25.2 It tracks the divergence where the photometric signals saturate

`fr1_xyz` under the §23 haze sweep. Every column normalised to its own β = 0 value:

| β | ATE ÷ floor | photo RMSE | 1 − inlier | cond | **depth_incons** | **depth_outlier** |
|---|---|---|---|---|---|---|
| 0.0 | 0.09× | 1.0× | 1.0× | 1.0× | 1.0× | 1.0× |
| 0.8 | 2.61× | 2.3× | 1.4× | 0.6× | 1.5× | 1.0× |
| 1.5 | 6.08× | 3.4× | 1.8× | 0.7× | **5.1×** | 2.0× |
| 2.5 | 5.11× | 4.1× | 1.9× | 0.7× | **6.0×** | 3.2× |
| 4.0 | **11.57×** | 4.3× | 1.9× | 1.0× | **10.4×** | **12.4×** |

The error grows 11.6× and `depth_incons` grows 10.4× — near one-for-one. The photometric
residual stops at 4.3× and the inlier ratio at 1.9×.

> **The rule this earns.** A photometric residual is bounded by the intensity range and an
> inlier ratio is bounded by [0, 1] **by construction**. Neither can grow without limit, so
> neither can report a failure that does. Depth disagreement is a physical quantity in metres
> with no ceiling. **A self-assessment signal must have at least the dynamic range of the
> failure it is meant to report** — and that is a property of the signal's definition, checkable
> before any data is collected.

The condition number is flat at 0.6–1.0× throughout, confirming §13.3's reading of it as
carrying no information here.

### 25.3 Per-frame ranking: complementary, not superior

Lift = P(bad frame | flagged) ÷ P(bad frame), worst decile each:

| signal | fr1_xyz | fr3_sitting | fr3_walking |
|---|---|---|---|
| **predicting translation error** | | | |
| photometric RMSE | 5.49 | 6.08 | 2.66 |
| 1 − inlier ratio | 4.31 | 6.46 | 3.04 |
| **depth inconsistency** | 4.31 | **7.22** | **3.42** |
| **predicting rotation error** | | | |
| photometric RMSE | 2.74 | 2.28 | 3.04 |
| **depth inconsistency** | **4.70** | 1.90 | 1.90 |

Depth leads on *translation* error on both dynamic sequences; photometry and the inlier ratio
lead on *rotation*. Mechanically that is what should happen — depth reprojection is directly
sensitive to motion along the ray and comparatively blind to lateral rotation. **The channels are
complementary and should be combined, not ranked.**

> **These numbers do not reproduce §13.3's "lift ≤ 1.12 on `walking_xyz`".** The photometric
> signals measure 2.66–3.80 today. The engine changed underneath — §14's masking, §16's belief
> channel, §20.1's frame-mismatch fix — and this run is a full sequence rather than a 9-second
> window. §13.3's blindness has partly lifted; what has not is the *saturation* in §25.2, which
> is structural rather than a tuning state.

### 25.4 The measurement that nearly decided itself

The first version scored lift against a single combined error, `‖t‖ + 2.0·‖r‖`. That 2.0 chose
the winner:

| error metric | photo RMSE | depth incons |
|---|---|---|
| translation only | 2.66 | **3.42** |
| ‖t‖ + 2.0·‖r‖ | **3.80** | 2.66 |
| rotation only | **3.04** | 1.90 |

An arbitrary unit-reconciling constant flipped the conclusion. The two axes are now reported
separately and the blend is gone — which is also how §25.3's complementarity became visible
instead of being averaged away. §10.4, committed by the analysis rather than the estimator.

### 25.5 State

Three tests in `test_direct_aligner.cpp`, and the middle one carries the claim: **scale cur's
depth by 1.15 and the intensity image does not change by a single pixel** — `photometric_rmse`
and `inlier_ratio` are bit-identical while `depth_consistency` jumps past 0.10. The channel sees
an error the photometric path provably cannot. All three were demonstrated failing against a
mutant that reports perfect consistency; the mutation is caught by exactly that test.

### 25.6 Consumed, calibrated — and honest about how far it carries

The signal is now read rather than merely exposed. `align()` returns a **degraded** `Result`
with `reliability = gate / depth_consistency` when the estimate disagrees with the depth it
never used, alongside the two existing degrade paths — and unlike those two, this one does not
come from inside the photometric channel. *Not judged* (`-1`) deliberately does **not** degrade;
folding "don't know" into "bad" would penalise every depth-free scene.

**The placeholder was worse than un-calibrated — it was inert.** `depthConsistent()` shipped
with `0.02`, while the median *normal* frame measures 0.004. That gate could never fire.
`tools/depth_gate_calib.py` fits it on 10 sequences across 3 cameras, maximising Youden's J
against "worst decile of true relative translation error" (a definition that never touches
`depth_consistency`, or it would be circular):

| camera | fitted threshold | separation (bad ÷ ok median) | J |
|---|---|---|---|
| fr1 (4 seqs) | 0.0047 – 0.0068 | 1.4 – 3.0× | 0.49 – 0.76 |
| fr3 (4 seqs) | 0.0057 – 0.0117 | 1.3 – 2.5× | 0.31 – 0.75 |
| **fr2 (2 seqs)** | 0.0035 | **1.02 – 1.05×** | **0.14 – 0.27** |

Shipped: **0.0057**, the median, with a 3.4× spread across sequences and 2× across cameras —
far better than `ν`'s 23× (§17.1), still not a universal constant.

**Where it works and where it does not, measured:**

| use | result |
|---|---|
| tracks a silent divergence (haze sweep) | fires **33 % → 96 %**, monotone with ATE 0.09× → 11.6× floor |
| per-frame ranking within a sequence | lift 3.4 – 7.2 on translation error (§25.3) |
| **per-frame binary gate at one constant** | **33–39 % firing on clean sequences at 1.6 cm ATE** |
| **predicting which sequence is bad** | Spearman **+0.394** — better than photometric (+0.309), still weak |
| **on the fr2 camera** | **does not separate at all** |

So the honest statement is narrower than "a failure detector": **it is a good relative signal
inside a sequence and a weak absolute one across them.** A single cross-camera threshold buys a
third of clean frames flagged, which is why the continuous `depthReliability()` is the intended
consumer interface and the binary `depthConsistent()` is the convenience.

Two tests pin the consumption path: a clean scene must **not** degrade (or the gate is always on)
and a 15 %-scaled depth must degrade without failing (the pose is still usable). A third pins
that unjudged geometry does not degrade.

### 25.7 Acting on it makes fusion worse — except where a better tier exists

The natural consumer is fusion: §18.2 concluded that *"fusing a good estimate with a bad one,
weighted by information, degrades the good one whenever the bad one's information is not honestly
small,"* and §23.3 showed Tier 0 drifting to 11.6× the floor without shrinking its own
information at all. `depthReliability()` is precisely a path to make it honestly small, from
outside the photometric channel. So: scale `Λ_ECDA` by it and replay every ablation.

`python/tools/fusion_depthgate.py`, on §21's replay harness so nothing changes but Tier 0's
weight. The gate is live — 42 to 228 frames per sequence carry reliability < 1, mean 0.66–0.92:

| sequence | t0 only | t0+t2 | all 3 |
|---|---|---|---|
| | plain / gated | plain / gated | plain / gated |
| fr1_360 | 10.37 / 10.37 | 43.00 / 44.73 | 43.00 / 44.73 |
| fr1_desk | 2.53 / 2.53 | 2.63 / 2.88 | 3.00 / 3.20 |
| fr1_xyz | 1.56 / 1.56 | 1.94 / 2.19 | 2.32 / 3.28 |
| fr3_sitting_xyz | 1.01 / 1.01 | 3.77 / 5.93 | 5.26 / 7.09 |
| **fr3_walking_xyz** | 20.72 / 20.72 | **17.34 / 15.21** | **19.08 / 18.20** |

**Better on 2 configurations, worse on 8, tied on 5.** The `t0 only` column is unchanged by
construction — with one tier the solution is invariant to overall weight — which is the harness
confirming it is measuring what it claims.

**Both wins are the same sequence**, and it is the one §18.2 already identified as the single
place fusion genuinely helps: `fr3_walking_xyz`, where moving people break photometry and leave
planes intact. That is not a coincidence, it is the whole result:

> **A correct failure signal, consumed in the wrong way, makes things worse.** Down-weighting
> Tier 0 does not create accuracy — it hands weight to tiers that are 3–15× less accurate
> (§18.2). Detecting that the good estimator is untrustworthy only helps if something better is
> available to receive the weight, and on four of five sequences nothing is.

This sharpens §21.4's rule rather than contradicting it. That section found a χ² consistency gate
repairing fusion by *excluding* bad tier contributions; this one finds that *reallocating* weight
away from the best tier fails. The gate's value is in refusing bad information, not in
redistributing confidence.

**What the reliability should drive instead** is an action that does not require a second
estimator to be right. §25.8 tries the cheapest one.

### 25.8 An action that needs no other tier — keyframe replacement

If the pose disagrees with the depth the alignment never used, the most local explanation is
that the reference frame has stopped being a good anchor. Replacing it needs nothing from Tier 1
or Tier 2, which is exactly the property §25.7 showed was missing.
`wme_tum_odometry --depth-gate-kf 1` switches the keyframe when `depthConsistent()` fails, in
addition to the existing distance rule.

**The regime matters, and separating it is the whole reading.** A difference between two
trajectories that have both already blown past the do-nothing floor is a difference between two
useless answers:

| case | off | on | floor | regime | |
|---|---|---|---|---|---|
| fr1_xyz | **1.56** | 2.10 | 17.2 | usable | worse |
| fr1_desk | 2.53 | 2.52 | 57.6 | usable | tie |
| fr1_360 | 10.37 | **8.73** | 16.4 | usable | **better** |
| fr1_room | 21.72 | **18.32** | 101.6 | usable | **better** |
| fr2_desk | 2.38 | **2.15** | 39.5 | usable | **better** |
| fr3_sitting_xyz | 1.01 | 1.00 | 19.9 | usable | tie |
| fr3_walking_xyz | 20.72 | **18.69** | 28.0 | usable | **better** |
| fr3_structure_texture | **1.95** | 2.70 | 130.9 | usable | worse |
| haze 0.8 | 44.93 | 40.60 | 17.2 | *failed* | — |
| haze 1.5 | 104.71 | 88.13 | 17.2 | *failed* | — |
| haze 2.5 | 87.86 | **1141.57** | 17.2 | *failed* | — |
| haze 4 | 199.10 | 72.57 | 17.2 | *failed* | — |

**In the usable regime: 4 better, 2 worse, 2 tied** — the first consumption of this signal that
is net positive, against §25.7's 2-better/8-worse.

And the split is not random. It **helps on the four hardest sequences** (10.4, 21.7, 20.7 cm, and
fr2 at 2.38 — WME's weakest camera) and **costs on the two easiest** (1.56, 1.95 cm). Switching
keyframes more often shortens the baseline; that is worth paying when tracking is struggling and
is pure loss when it is not.

> Note `fr2_desk` improves by 9.6 % — the camera where §25.6 measured the per-frame separation
> at J = 0.14, i.e. no discrimination at all. A signal too weak to rank individual frames can
> still be strong enough to drive a policy, because the policy only needs the aggregate to lean
> the right way. **Per-frame discrimination and policy usefulness are different requirements**,
> and the first is the harder one.

**The four haze rows are reported and not counted, and the reason is the honest part.** All four
are 2.4–11.6× the do-nothing floor before the gate is applied — the estimator has already failed,
and one of them (β = 2.5) gets **13× worse**, not marginally worse. Once past the floor the
trajectory is unstable to any perturbation, and reading a +63 % or a −1199 % out of that pair
would be reading noise. What this does say plainly is that **keyframe replacement does not rescue
a diverged estimator** — the regime the detector was built for is the one where acting on it
locally cannot help. Recovery there needs relocalization, which is not implemented.

### 25.9 Relocalization fails under haze — and the reason is the point

§25.8 ended by saying that recovery from a diverged estimate needs re-anchoring against the map
rather than a recent frame. `wme_tum_odometry --reloc N` does it: after the geometry gate fires
N frames running, match the live frame's ORB descriptors against stored map keyframes (skipping
the three most recent, or the drift comes back with them), solve PnP, and **replace the world
pose** — which is what keyframe replacement cannot do, since it only fixes what comes next and
leaves accumulated drift in place.

| case | base | + keyframe | **+ reloc** | floor | reloc attempts → successes |
|---|---|---|---|---|---|
| fr1_xyz | **1.56** | 2.10 | 2.02 | 17.2 | 2 → 2 |
| fr1_360 | 10.37 | **8.73** | 8.73 | 16.4 | 2 → 0 |
| fr1_room | 21.72 | 18.32 | **17.16** | 101.6 | 12 → 11 |
| fr3_walking_xyz | 20.72 | 18.69 | **15.72** | 28.0 | 74 → 20 |
| haze 0.8 | 44.93 | 40.60 | 40.60 | 17.2 | **29 → 0** |
| haze 1.5 | 104.71 | 88.13 | 88.13 | 17.2 | **121 → 0** |
| haze 2.5 | 87.86 | 1141.57 | 1141.57 | 17.2 | **70 → 0** |
| haze 4 | 199.10 | 72.57 | 72.57 | 17.2 | **138 → 0** |

On clean data it adds real recovery: `fr3_walking_xyz` reaches **15.72 cm from a 20.72 cm base**,
a 24 % improvement for the full chain (detect → replace keyframe → re-anchor), and `fr1_room`
improves again on top of the keyframe gain.

**Under haze it succeeds zero times in 358 attempts**, and the haze columns are therefore
bit-identical to the keyframe-only run — the no-op is visible in the numbers, which is how the
count and the ATE corroborate each other.

The cause is not a threshold. Relocalization matches ORB descriptors between the live frame and
the map, and **the map keyframes were captured in the same haze**. Both sides of the match are
degraded by the same cause, so there is nothing intact to anchor to.

> **A detector in an independent channel does not give you a recovery in an independent
> channel.** `depth_consistency` correctly reports "you are lost" under haze — that is §25.2,
> and it holds. But every recovery mechanism actually available here — descriptor matching,
> photometric alignment — lives in the channel the haze destroyed, and lives there *on both
> sides of the comparison*. Detection and recovery are separate problems, and solving the first
> in a new channel does not solve the second.

This is the sharpest architectural argument for Tier 1 that this document has produced, and it
arrives from a failure rather than from the design. **An object constellation is the only
recovery path here that does not live in the photometric channel** — YOLO still fires on large
objects when texture is gone, which is exactly §2's claim. That TCG currently cannot do indoor
relocalization (26 of 92 keyframes had the 4 objects it needs, §24.2) is a limitation of the
implementation, not of the reason for wanting it. **The architecture's answer to this failure is
the one component measured as not working.**

### 25.10 The density levers, measured — and the one that looked right was not

§24.2 blamed TCG's indoor relocalization failure on object density — 26 of 92 keyframes carried
the 4 objects a constellation needs — and attributed that to COCO covering little of an office.
§25.9 raised the stakes: the constellation is the only recovery path that survives haze, and it
is the one component measured as not working.

Before changing anything, `wme_tcg_density` measures **which lever actually moves the count**.
Four candidates, each sacrificing something different: lower the confidence threshold (admits
false detections), require valid depth (drops boxes over depth holes), accumulate over a temporal
window (needs relative poses), or count only distinct classes.

| policy | `fr1_room` mean / ≥4 | `fr3_walking` mean / ≥4 |
|---|---|---|
| conf ≥ 0.25, depth ignored | 3.34 / 36.5 % | 5.59 / 85.2 % |
| conf ≥ 0.25, **depth required** | 3.18 / 34.3 % | 5.52 / 81.5 % |
| conf ≥ 0.15, depth required | 4.81 / 60.6 % | 6.85 / 85.2 % |
| conf ≥ 0.10, depth required | 6.40 / 75.9 % | 8.22 / 92.6 % |
| conf ≥ 0.25, **distinct classes only** | 2.43 / 27.0 % | 3.85 / 66.7 % |
| **conf ≥ 0.25 + depth + 5-keyframe window** | **15.48 / 97.8 %** | **25.26 / 100 %** |

Three of the four candidate explanations are wrong, and the measurement says so directly:

- **Depth validity is not the bottleneck.** 3.34 → 3.18 nodes, a 5 % loss. The "box centre has no
  depth" hypothesis, which the sampling code was written defensively against, costs almost
  nothing.
- **Class multiplicity is not a problem to be removed — it is most of the signal.** Requiring
  distinct classes *drops* the count to 2.43 and ≥4 coverage to 27 %. An office contains several
  chairs and several monitors, and discarding the duplicates discards the constellation.
- **Confidence does help** (0.25 → 0.10 roughly doubles nodes and takes ≥4 coverage from 34 % to
  76 %) but pays in precision, and §2's whole claim rests on the constellation being trustworthy
  enough to *decline* rather than guess.

**The temporal window moves the count more than anything else.** Five keyframes take `fr1_room`
from 34 % to 97.8 % and `fr3_walking` to 100 %, without loosening a single threshold.

> **This looked like the answer and was not.** §25.11 implements it and measures the trade: node
> count triples, and loop-edge accuracy degrades 5× (15.4 cm → 78 cm), because the accumulation
> pushes association and drift error into the node positions the signature is built from. Node
> *count* was the wrong quantity to optimise. Reported here as measured, and overturned two
> subsections later by the experiment it motivated.

> **The bottleneck was never the detector. It was the query unit.** A single frame is the wrong
> thing to ask a constellation about — object detection flickers, and a place is not defined by
> what one exposure happened to catch. This is not a new idea in the repository: the header of
> `tools/tum_relocalize.cpp` already says *"a real relocalizer queries with a small local map,
> not one frame"* and provides `--query-window`. §24's loop-closure tool then built its
> constellations from single frames anyway, and measured the consequence as a property of TCG.

That makes §24.2's conclusion too broad. What was measured is that **single-frame constellations
are too sparse indoors**, not that constellations are. The correction is to accumulate the query
over a short window using the odometry's own relative poses — drift across five keyframes is
small, and unlike `tum_relocalize`'s ground-truth stand-in that would be an honest input.

**The 15.48 above is an upper bound, and §25.11 measures the real number.** The density tool
*sums* nodes across the window; the same chair seen in five frames is one object, not five.
After merging by class and position, a 5-keyframe window gives **9.11** nodes and 92.4 %
coverage on `fr1_room` — the direction holds, the magnitude was overstated by ~1.7×, and 97.8 %
needs a window of 8.

### 25.11 The window makes it worse — and two of my own bugs were in the way

The windowed query is implemented in `wme_tum_loopclose --window N`: accumulate the last N
keyframes' nodes into the newest keyframe's frame using the odometry's own relative poses (not
ground truth, unlike `tum_relocalize`'s stand-in), merge same-class nodes within 0.25 m, and use
the result for **both** map and query.

| window | merged nodes | ≥ 4 nodes | queries | accepted | **loop-edge error** |
|---|---|---|---|---|---|
| **1** | 3.15 | 37.0 % | 25 | 3 | **15.4 cm / 5.9°** |
| 5 | 9.11 | 92.4 % | 76 | 2 | 78.6 cm / 44.4° |
| 8 | 12.76 | 97.8 % | 81 | 4 | 77.9 cm / 45.0° |

**The window buys coverage and spends accuracy, and the trade is bad.** Node count triples and
≥4 coverage nearly triples, while edge error grows 5×. Accumulating over a window accumulates
association and odometry error into exactly the node positions the distance spectrum is computed
from — more nodes, worse constellation. Downstream: pose-graph optimisation gives **+30.7 %** at
window 1 and **+1.1 %** at window 8.

So §25.10's proposal is measured and rejected. Density was the *observation*; it was not the
lever it looked like.

**Two defects of mine had to be removed before that measurement meant anything**, and both
inflated the case against TCG:

| defect | effect | how it read before |
|---|---|---|
| gravity never passed to `insert`/`query` — **chirality silently off**, one of the signature's three components | 289 cm → 73 cm at window 1 | "the signature does not discriminate" |
| **`ConstellationMatch::transform` direction inverted** — it is query→place, stored as place→query | 286 cm → **55.7 cm** on the same edges | "every accepted loop is a gross false positive" |

The second is the serious one. It did not only corrupt this measurement — **§24's TCG pose graph
consumed the inverted edges**, which is why that section reported loop closure *degrading* ATE by
11.2 %. Corrected, the same code on the same data improves it by **30.7 %**, and the WME full
system goes from losing to the classical one (24.16 vs 20.67) to beating it (**15.06** vs 20.67).

> §24.3 recorded me walking into §19.3's trap — measuring a rejection mechanism with its
> ambiguity branch structurally unreachable. This is the same trap twice more in the same
> subsystem: a component switched off by omission, and a convention inverted. **The pattern is
> not carelessness about the mechanism; it is that every one of these was invisible in the
> output.** A wrong-direction transform produces a plausible number, not an error, and the only
> thing that exposed it was that 286 cm was too bad to believe for a match the accept rule had
> already approved.
>
> The rule the earlier sections state — verify the measurement discriminates before concluding
> the algorithm is wrong (§10.4) — was applied to the estimator every time and to **my own
> harness** only after three failures.

**What still stands.** TCG's recall is genuinely poor: 3 loops from 25 possible queries against
ORB's 44 from 1292 proposed pairs, and 15.4 cm median accuracy against 2.14 cm. The density
limit that §24.2 identified is real; what is not real is the conclusion that constellations
match the wrong places.

### 25.12 Both density levers fail, and the accept rule is right to reject them

§25.11 rejected the temporal window. The other lever §25.10 measured was the detection
confidence threshold, which raises node count **without** accumulating drift into node
positions — a different trade, and worth testing on its own. Window fixed at 1 throughout:

| conf | queries possible | accepted | median edge error | gross false |
|---|---|---|---|---|
| **0.25** | 25 | **3** | **15.4 cm** | 1/3 |
| 0.15 | 42 | **0** | — | — |
| 0.10 | 54 | 2 | 82.3 cm | 1/2 |

**Lowering the threshold raises queries 25 → 54 and lowers accepted loops 3 → 0 → 2.** Recall
goes *down*. The spurious detections enter the constellation, the signature stops matching, and
the accept rule declines — which is what it is for.

So both levers increase node count and neither increases recall, for two different reasons:

| lever | what it adds | why recall does not rise |
|---|---|---|
| temporal window | real objects, **displaced** positions | geometry degrades — edge error 15.4 → 78 cm |
| lower confidence | **objects that are not there** | signature stops matching — accepts drop to 0–2 |

> **Node count was never the binding constraint; node quality was.** A constellation needs four
> objects that are really present, accurately localised, *and* stably detected. The window
> supplies quantity at the cost of localisation; the threshold supplies it at the cost of
> presence. The accept rule declines in both cases, correctly — §2's "declines rather than
> guesses" is the one part of Tier 1 that keeps working under every perturbation tried here.

That makes TCG's indoor recall limit structural for this scene rather than a tuning problem: an
office viewpoint yields about three confidently-detected, well-localised COCO objects, and the
signature needs four. Neither knob on the existing pipeline can manufacture the fourth. The
remedies left are outside it — a detector whose classes actually cover indoor scenes, or a node
primitive that is not a COCO detection — and both are larger than a parameter change.

**The honest position on Tier 1**, after §24 and §25: it *works* — 3 loops, 15.4 cm median,
+30.7 % ATE — and it works too rarely to carry relocalization on its own, for a reason that
belongs to the detector rather than to the constellation idea. §25.9's argument for why the
architecture wants it (the only recovery channel that survives haze) is untouched by that.

### 25.13 Plane centroids are worse on both axes — rejected before building

§25.12 ended by proposing a different node primitive: SPA's planes are already computed and an
office has no shortage of them. `wme_plane_density` tests that before anything is built, and
measures **two** things — because §25.11 and §25.12 both failed by supplying quantity at the cost
of a quality the count could not see.

`fr1_room`, 60 keyframes, ground-truth poses so the geometry is not confounded by drift:

| | planes | COCO objects (§25.12) |
|---|---|---|
| nodes per keyframe | **2.77** | 3.15 |
| ≥ 4 nodes | **30.0 %** | 37.0 % |
| same node's position across viewpoints | **1.52 m median**, p90 3.32 m, max 5.53 m | few cm |

**Planes are not more plentiful than objects here, and their centroids are not landmarks at
all.** The same physical wall, matched across viewpoints 0.3–2.0 m apart by normal (< 15°) and
offset (< 0.15 m), yields centroids **1.5 m apart** at the median.

The cause is structural rather than a tuning failure: a plane's centroid is the centroid of its
*visible portion*, and what is visible changes with viewpoint. A constellation signature is a
pairwise distance spectrum, so a node that slides by 1.5 m changes the signature of a place the
system is standing in.

> **The primitive has to be plentiful indoors *and* viewpoint-stable, and neither candidate is
> both.** COCO objects are stable and too few; plane centroids are neither. Two independent
> primitives, rejected for opposite halves of the same requirement.

**What this does not reject.** Planes as such — only their centroids. The intersection of three
planes is a point that does **not** depend on how much of each plane is visible, and an office
corner is exactly that. Room corners, wall-floor junctions and desk edges are plentiful and
stable by construction. That is the primitive this measurement points to, and it is not
implemented: `PlaneExtractor` returns planes, nothing computes their intersections. It is also a
larger change than a parameter, which is why the measurement was worth running first — one tool
instead of a subsystem.

### 25.14 Corners fail on a counting argument I should have seen first

§25.13 ended by naming plane *intersections* as "the first candidate in this line that is not
already contradicted by a measurement" — a corner does not depend on how much of each plane is
visible, so it is stable by construction. Implemented (`intersect3`, |det| ≥ 0.3, within 5 cm of
all three planes, inside 8 m) and measured on the same 60 keyframes:

| primitive | per keyframe | ≥ 4 nodes | viewpoint stability |
|---|---|---|---|
| COCO objects | 3.15 | 37 % | few cm |
| plane centroids | 2.77 | 30 % | 1.52 m |
| **plane corners** | **0.35** | **5 %** | nearest corner 4.45 m away |

**Stable by construction, and there are 0.35 of them.** The stability claim was right and
irrelevant: a corner needs *three* non-parallel planes, and this scene supplies 2.77 planes per
view. Three-at-a-time from a pool of under three is close to zero by counting alone — no
measurement was needed to predict it, and I proposed it anyway.

The 4.45 m "nearest corner" figure is not a stability measurement either; with 0.35 corners per
keyframe, the nearest corner in another view is usually a *different* corner. The number is a
symptom of the density, not an independent finding.

### 25.15 What the whole primitive search says

Five candidates, all measured, all rejected — and the failures now form one picture:

| candidate | what it supplied | what it cost |
|---|---|---|
| COCO objects (baseline) | stable, well-localised | only 3.15 per view |
| temporal window (§25.11) | 3× the objects | positions displaced — edge error 15.4 → 78 cm |
| lower confidence (§25.12) | 2× the objects | objects that are not there — accepts 3 → 0 |
| plane centroids (§25.13) | — | fewer *and* unstable (1.52 m) |
| plane corners (§25.14) | stability by construction | 0.35 per view |

> **The constraint is not the constellation algorithm; it is landmark supply.** A descriptor
> pipeline extracts ~1000 keypoints per frame on this sequence and closes 44 loops at 2.14 cm.
> Every primitive WME's pipeline can currently produce yields **about three** — a ~300× density
> gap — and a constellation needs four. No rearrangement of the object pipeline closes a gap that
> large, which is why all five attempts fail at the same place by different routes.

That is a sharper statement than "TCG has low recall", and it locates the work outside this
subsystem: the fix is a detector or segmenter whose output is dense in indoor scenes, not a
different way of assembling what YOLO already returns. §2's simulated results are not contradicted
— they were measured where objects were guaranteed to exist, and that guarantee is exactly what an
office withdraws.

**What still stands for Tier 1**, unchanged by all five rejections: it closes loops correctly when
it fires (§24.1, +30.7 %, 15.4 cm median), it declines rather than guesses under every perturbation
tried here, and it remains the only recovery channel in the architecture that does not live in the
photometric band that haze destroys (§25.9).

### 25.16 The last oracle gap, and what it found on its first run

Two items §26 had been carrying, both about code that conclusions rest on with no independent
check.

**The `rtol` sweep — prediction not borne out.** §20.3 found every `atol=1e-12` in the original
differential suite silently carrying numpy's default `rtol=1e-5`, tightened the tests it touched,
and predicted that *"sweeping the rest will likely surface more divergences."* Swept: 27
assertions tightened to `rtol=0`, and **all 76 tests still pass**. The remaining six already went
through a `_same()` helper that set `rtol=0.0`. The tightening is real — a 1e-6 relative
difference now fails where it previously passed — so the two implementations genuinely agree to
1e-12 absolute across the suite. A predicted defect that is not there is worth recording as
plainly as one that is.

**SPA now has an oracle, and it found two defects in the first minutes.** `StructuralAligner`,
`Plane`, `PlaneMatch` and `unobservableDirections` are bound, with 13 differential tests. Both
defects were in the **Python reference**; the C++ was right on both.

**Defect 1 — §7.1's fix was never applied to the reference.** `spa.py` computed rotational
information as the scatter matrix `Σ w·nnᵀ`, where the residual `n_cur − R·n_ref` under this
project's left-perturbation convention gives `JᵀJ = I − nnᵀ`. §7.1 documented exactly this,
measured it at 21× wrong on one axis and pointing the wrong way, and corrected **the published
table by re-deriving it from the C++ port** — but the Python reference was left as it was. It
had been carrying the defect ever since, invisible because nothing compared the two.

On three orthogonal planes the scatter gives `wI` and the complement `2wI`, which is the exact
factor of 2 the oracle reported.

**Defect 2 — the weight was applied twice, and my own first test hid it.** `spa.py` built the
translation system as `A = n·w` and solved `lstsq(A, b·w)`, whose normal equations are
`Σ w²·nnᵀ`. Weighted least squares requires scaling residuals by `√w` so the normal equations
come out `Σ w·nnᵀ`. The C++ accumulates `weight · JJᵀ` directly and is correct.

This one is worth recording for how it nearly escaped. The first version of the test used
`inliers = 400`, which makes `confidence = 1` and therefore **`w = w² = 1`** — the two formulas
are indistinguishable at that one value. It only appeared when `inliers` was varied:

| inliers | weight | C++ translation | Python translation | ratio |
|---|---|---|---|---|
| 400 | 1.00 | 400.0 | 400.0 | **1.00** |
| 200 | 0.25 | 100.0 | 25.0 | **4.00** |
| 100 | 0.0625 | 25.0 | 1.6 | **16.00** |

> **A test constant chosen for convenience can be the one value at which two different formulas
> agree.** `inliers = 400` was picked because it makes `confidence` exactly 1 and the arithmetic
> easy to check by hand — which is precisely why it could not discriminate. The parameter is now
> swept, and §10.4's rule ("verify the measurement discriminates") applies to the *inputs* a test
> chooses, not only to the quantity it reports.

After both fixes the two implementations agree to `rtol=1e-9` at every weight, and the
scatter/complement form is pinned independently on **both** sides — a two-plane scene where the
axis perpendicular to every normal must carry the *most* rotational information, and would carry
exactly zero under the scatter form. That scene had to be chosen deliberately: on three
orthogonal planes the complement is `2I`, isotropic, and indistinguishable from scatter by
direction. The first version of that test asserted `5000 > 5000` and failed.

**Impact.** §7's rank table and §18.3's complementarity were computed from the C++ and are
unaffected. Any Python-side SPA result predating this fix is not: the rotation block was
projected onto the wrong subspace and the translation weighting was squared.

### 25.17 What the same sprint closed elsewhere

Three coverage gaps this document had been carrying, all of the same kind — code that
conclusions rest on with no independent check:

| gap | closed by | evidence it discriminates |
|---|---|---|
| `PoseFusion` had **zero** differential coverage while §18, §21 and §23.4 all stand on it | `fusion::fuse` bound + **16 tests** against `fusion_replay.py`, the oracle that reproduced 40 C++ trajectories to 0.002 % | a mutant dropping the `α·κ` weight is caught by **8** of them |
| ECDA had no element-level oracle, blocked by one line — `ecda.py` still held the fixed `huber_delta = 12.0` C++ deleted in §11.4 | adaptive kernel ported (plus §13.4's separate fixed-threshold inlier count); poses, point counts and inlier ratios compared | the two agree to **0.3 mm** while both sit 21 mm from the crude synthetic truth |
| no end-to-end test in C++ | `test_pipeline.cpp`, four stages | a broken stage must fail loudly, asserted directly |

**Suite at the end of §25.17: 218 C++ tests, 525 Python tests, 1 xfail.**

Porting the kernel broke `test_residual_gate_is_relative_not_absolute`, and the break was
instructive: it asserted `median(rmse) > 4.0`, a number calibrated to the *old* fixed kernel.
The adaptive kernel legitimately lowers the weighted residual to 3.39, so the scene stopped
demonstrating the claim. The claim itself — that normal alignments vary too much for any single
absolute threshold — is now asserted as a spread (`max/min > 2`, measured 3.6×), which no kernel
change can silently invalidate. **A test pinned to a constant from the implementation it tests
will break when that implementation improves, and the break looks like a regression.**

### 25.18 The environment oracle, and the two ways a direction histogram lost its direction

§11.1 recorded four "dead channels" — `backlight`, `specular`, `shadow_strength`,
`scene_complexity` — as having unit tests and no oracle. That gap is now closed:
`python/wme/reference/environment_cues.py` reimplements every single-frame cue estimator in
numpy, and `test_differential_environment.py` compares it against the C++ across 12 images
chosen to move each channel.

Two decisions shaped the reference and are worth stating, because both were forced.

**It goes through the public path.** The estimators are `private` in `EnvironmentAnalyzer.hpp`.
Exposing them was the obvious move and the wrong one: it measures functions in isolation and
leaves the assembly — which cue is EMA'd, which is not, which is computed only when `rgb` is
non-empty — unchecked. Instead the binding exposes `update(frame, quality)` and the test sets
`evidence_ema = 1.0`, which makes the EMA the identity so that frame's raw estimate lands
in `evidence` unchanged. What gets compared is then the code that actually runs.

**It reimplements OpenCV's integer colour conversions.** `BGR2GRAY` and `BGR2HSV` are
fixed-point on `uint8`. Computing them in floating point differs by at most 1 gray — and
`estimateShadow` bins luminance into 32 bins 8 gray wide, so that 1 moves pixels across bin
boundaries. `estimateSpecular` tests `V > 210 && S < 40`, both exact integer thresholds. The
reference therefore carries the `(B·1868 + G·9617 + R·4899 + 8192) >> 14` accumulator and
OpenCV's `sdiv_table` rounding, not the textbook coefficients.

Five of six cues matched on the first run. `scene_complexity` did not — and the disagreement
was 0.578 (C++) against 0.473 (numpy) on a checkerboard, 22 % relative. Two independent
defects, both in the same three lines:

```cpp
double a = std::atan2(py[x], px[x]);                                  // (1)
if (a < 0.0) a += kPi;
const int b = std::clamp(static_cast<int>(a / kPi * kBins), 0, kBins - 1);   // (2)
```

**(1) `atan2` resolved to the `float` overload.** `px`/`py` are `float*`, so this is `atan2f`,
and for a perfectly horizontal edge `atan2f(-y, 0) = -1.57079637` — a hair *more* negative than
double `-π/2`. Adding double `kPi` lands at `π/2 - 4.4e-8`, so `a/kPi*kBins = 8.99999975`
truncates to **bin 8**, while the opposite gradient sign gives **bin 9**. One orientation, two
bins, selected by the sign of the gradient. 1708 of 7976 kept pixels were mis-binned this way on
a checkerboard.

**(2) The bin index was clamped, not wrapped.** Direction has period π, so `a = 0` and `a = π`
are the *same* orientation — but `clamp` puts them in bins 0 and 17, opposite ends of the
histogram. A field of vertical stripes, one orientation and nothing else, reported
`scene_complexity = 0.240`.

The two compound. Fixed — `atan2` in `double`, index by `((b % kBins) + kBins) % kBins` — a
single-orientation scene now reports exactly **0.000**, and the checkerboard drops 0.578 → 0.473.

The failure mode is the one §10.4 keeps naming, in a new place. Bin boundaries at 0, π/2 and π
are not arbitrary values in a continuous distribution — **they are exactly the orientations that
dominate man-made scenes**, which is the premise SPA is built on. A rounding error at a bin edge
is normally harmless because edges are rarely landed on; here almost every informative pixel
lands on one. The bias was therefore systematic, and toward *higher* complexity, on precisely
the structured indoor scenes the channel exists to recognise.

It was invisible for the usual reason: `scene_complexity` is written to `EnvironmentState` and
**read by nothing** — grepping the tree finds one consumer, the pybind11 binding added for this
test. A published number no one consumes cannot be wrong loudly enough to be noticed.

**Suite after this section: 222 C++ tests, 631 Python tests, 1 xfail.** Still uncovered on this
path: `estimateShake` (needs the previous frame) and `analyzeTransientParticles`
(needs the 9-frame temporal-median ring buffer).

### 25.19 The stereo depth front-end, and the parameter that fails quietly

§26 recorded EuRoC and KITTI as blocked on "no depth front-end". `StereoDepth`
(`include/wme/perception/StereoDepth.hpp`) is that front-end: OpenCV `StereoSGBM`,
disparity → `Z = f·B/d`, with an explicit invalid mask. Depth stops being a measurement here
and becomes an estimate, and `DirectAligner` treats depth as truth when it back-projects — so
the class is built to report *which pixels* it will stand behind, not just a depth map.

**The tests refuse a constant tolerance.** `dZ/Z = dd/d`, so the same 15 m surface is
recoverable to 1.2 % at KITTI's `f·B = 386` and only to 9.4 % at a EuRoC-class `f·B = 48` —
the disparity there is 3.2 px, and no matcher fixes that. The first version of the sweep
asserted a flat 3 % and failed on the EuRoC row, which is the §10.4 error inverted: it
demanded an accuracy the geometry cannot supply, and the failure read as an implementation
bug. The bound is now `kSubpixelBudget / d`, and a separate test checks that
`median_rel_err × d` stays within 4× across a **8.4× disparity sweep** — i.e. that the error is
disparity-quantisation-limited, which is the claim actually being made.

**Then the real data disagreed with the synthetic data.** Run against TUM's *measured* depth
(`python/tools/stereo_validate.py` — warp the real frame by the real depth to synthesise a
right view, then ask SGBM to recover it), a KITTI baseline on an indoor scene collapsed:

| sequence | B | `num_disparities` | Z<sub>min</sub> repr. | coverage | median err | scale |
|---|---|---|---|---|---|---|
| `fr1_desk` | 0.10 | 128 | 0.41 m | 90.0 % | 2.16 cm (1.6 %) | 1.003 |
| `fr1_desk` | 0.20 | 128 | 0.81 m | 81.5 % | 3.88 cm (3.5 %) | 1.006 |
| **`fr1_desk`** | **0.54** | **128** | **2.20 m** | **10.7 %** | **139.6 cm (142 %)** | **2.417** |
| `fr3_str_tex_far` | 0.54 | 128 | 2.20 m | 54.5 % | 29.9 cm (12.1 %) | 1.032 |

A desk at 1–2 m with a 0.54 m baseline needs ~150–280 px of disparity; the search range was
128. **SGBM does not report "out of range".** It returns the best match *inside* the range, so
the surface comes back at 2.4× its true distance with no error flag. Coverage drops, which is
loud — but the survivors are wrong, which is not.

Per-pixel detection was tried and does not work. Clipping at the search ceiling catches
**0.4 %** of the affected pixels; the rest mismatch somewhere in the interior of the range and
are indistinguishable from good matches. So the guard has to be at configuration time:
`requiredDisparities(f, B, z_min)` computes the range the scene needs, and
`min_representable_depth_m` / `clipped_ratio` are reported on every result. Sizing the range
from the scene rather than from a default:

| sequence | B | `num_disparities` | coverage | median err | scale |
|---|---|---|---|---|---|
| `fr1_desk` | 0.54 | 128 → **288** | 10.7 % → **56.8 %** | 142 % → **5.1 %** | 2.417 → **1.022** |
| `fr1_room` | 0.54 | 128 → **288** | 18.1 % → **65.7 %** | 74.8 % → **11.0 %** | 1.748 → **1.090** |
| `fr3_str_tex_far` | 0.54 | 128 → **288** | 54.5 % → **96.6 %** | 12.1 % → **3.8 %** | 1.032 → **0.993** |

> **A stereo matcher's search range is not a performance knob. It is a statement about the
> closest surface the configuration can see, and violating it produces confident wrong depth
> rather than missing depth.** The default 128 is right for KITTI (`f·B = 386`, road scenes
> beyond 3 m) and wrong for anything close-range at the same baseline — which is exactly the
> substitution someone makes when moving a config between datasets.

**What this measurement is not.** The synthetic right view is warped by the same depth it is
compared against, so occlusions are filled by replication and photometric mismatch is exactly
zero. Both make SGBM's job easier than real stereo. The coverage and error figures above are
therefore **lower bounds on the error**, and the numbers that transfer are the *relative* ones
— the collapse at B = 0.54 and its recovery — not the absolute 2 cm. Real stereo accuracy is
not established until KITTI runs.

### 25.20 KITTI — the first data that is not TUM, and the result changes sign

Four sequences of KITTI odometry, converted to the TUM layout by `wme_kitti_convert` so that the
same `wme_tum_odometry` and `wme_tum_baseline` binaries validated in §22 run unchanged. Depth is
`StereoDepth` output — an estimate, not a measurement. 400 frames at stride 2 per sequence
(sequence 04 has only 136), keyframe distance 1.0 m.

**Before any number was believable, a constant had to come out of the code.** Both tools carried
`if (!(z > 0.1) || z > 8.0) continue;` with the comment *"TUM depth valid range"*. KITTI's median
valid depth is 13.9 m and its range is 3.5–47.9 m, so that ceiling discarded almost every
correspondence: the baseline got **35 3D matches per frame and failed 371 of 400 frames**, which
would have read as an 78× WME victory. With the range moved into `calib.txt` where the dataset
declares it, the same baseline gets **241 matches and fails 0 of 400**. The 8.0 was the same
class of defect as the intrinsics-by-path-substring guess it sits next to — a TUM fact written
as a universal one.

Depth ceiling matters to both systems and not identically, so each is reported at its own best:

| sequence | frames | path | WME best | ORB+PnP best | winner |
|---|---|---|---|---|---|
| 00 | 400 | 556.9 m | 172.0 cm (40 m) | **129.0 cm** (60 m) | ORB 1.33× |
| 04 | 136 | 393.6 m | **387.8 cm** (60 m) | 5056.9 cm (60 m) | **WME 13.0×** |
| 05 | 400 | 588.6 m | 314.4 cm (60 m) | **160.3 cm** (60 m) | ORB 1.96× |
| 07 | 400 | 506.6 m | **249.8 cm** (60 m) | 285.2 cm (40 m) | WME 1.14× |

**2–2.** On TUM the tally was 9–6 to WME (§22). Outdoors it is even, and the two systems fail in
different places rather than one being uniformly behind.

Sequence 04 is the interesting row and it is not a starved baseline: ORB tracks **133 of 136
frames with 62 PnP inliers**, so it is running, and still lands 13× worse. 04 is a highway
drive — 2.84 m between used frames, nearly pure forward motion, structure mostly far. Forward
motion puts the epipole inside the image, which is the configuration where translation along the
optical axis is weakest for a sparse solver, and the depths that would constrain it are the
40–60 m ones where stereo is least accurate. WME uses 10³–10⁴ pixels where PnP uses 62 points.
That is a plausible mechanism, not a demonstrated one — it is consistent with the numbers but no
experiment here isolates it.

The depth ceiling itself is a U-curve for both, and the two optima do not coincide:

| `depth_max` | 20 m | 30 m | 40 m | 60 m |
|---|---|---|---|---|
| WME (seq 00) | 992.4 | 688.3 | **172.0** | 224.0 |
| ORB (seq 00) | 709.3 | 261.7 | 145.8 | **129.0** |

Too tight starves both; too loose admits far points whose depth error grows as `Z²`. RANSAC
rejects those as outliers, which is why the descriptor pipeline keeps improving to 60 m while
**ECDA gets worse past 40 m — it has no depth-uncertainty weighting and treats a 60 m point with
±4 m of stereo error exactly like a 6 m point with ±4 cm.** That is a concrete, addressable gap
and it is the first one this project has found that only outdoor data could expose.

One more asymmetry worth recording: on sequence 05 WME's RPE translation is **12.3 cm/s against
ORB's 19.2**, while its ATE is 2× worse. Better locally, worse globally — the error is
structured drift, not local noise.

> **Everything §22 concluded was conditioned on one indoor dataset, and the first outdoor data
> moves the result from 9–6 to even.** The direction of the comparison was not a property of the
> algorithms; it was a property of TUM.

**What this is not.** Four sequences, 400 frames each, no loop closure on either side, no bundle
adjustment, and depth from a stereo front-end whose accuracy on KITTI has not been measured
against lidar — only against TUM's measured depth on a self-warped view (§25.19). Sequences
01–03, 06, 08–10 are downloaded and unrun. And this is still not ORB-SLAM3.

### 25.21 Fixing what KITTI found — one derived constant, and the tally flips to 4–0

§25.20 ended with a diagnosis: ECDA treats depth as truth, so a 60 m point with ±4 m of stereo
error enters the normal equations exactly like a 6 m point with ±4 cm. The fix is to move that
uncertainty into the residual variance where it belongs:

```
Q = R·P + t,   P = d·ray   ⟹   ∂Q/∂d = (Q − t)/d
∂r/∂d = Jp·(Q − t)/d,      σ_d = c·d²
w  ×=  σ_I² / (σ_I² + (∂r/∂d · σ_d)²)
```

`σ_Z = c·Z²` covers both sensor families with one coefficient: stereo gives `c = σ_d/(f·B)`
directly from `Z = f·B/d`, and structured light (Kinect) is measured at `c ≈ 1.4e-3`. Default is
`0.0`, so every published number in this document is unchanged unless the term is switched on.

**This is not outlier rejection, and the distinction is what makes it work.** Huber cuts by
residual *magnitude*; a distant wall usually has a *small* residual and sails through. This term
cuts by how strongly the residual *responds* to depth error — so a far, low-gradient region keeps
its weight while a far, high-gradient one loses it.

For KITTI, `c = σ_d/(f·B) = 0.3/386 = 7.8e-4` is **derived, not tuned** — and it lands next to
the empirical optimum (a sweep on sequence 00 gives 94.4 / 88.4 / 78.4 cm at c = 4e-4 / 8e-4 /
2e-3, then collapses to 1845 cm at 5e-3). Applied at the derived value across all four sequences,
`depth_max` 60 m:

| sequence | ECDA c=0 | ECDA c=7.8e-4 | ORB+PnP best | winner |
|---|---|---|---|---|
| 00 | 223.96 | **94.32** | 129.03 | **WME 1.37×** |
| 04 | 387.76 | 954.78 | 5056.93 | **WME 5.3×** |
| 05 | 314.37 | **141.48** | 160.25 | **WME 1.13×** |
| 07 | 249.76 | **107.93** | 285.20 | **WME 2.64×** |

Three sequences improve by ~2.3×; **04 gets 2.5× worse**, and that is the same fact §25.20
reported from the other side. 04 is the highway drive where almost all structure is far — the
term down-weights nearly everything ECDA has. The right response is not to tune `c` down but to
record that a scene with no near structure has genuinely less usable information, which is true.
KITTI goes from **2–2 to 4–0**.

The model also makes a falsifiable prediction, and it holds. Without the term, raising the depth
ceiling *hurt* ECDA (40 → 60 m: 172 → 224 cm). With it, more data stops being harmful:

| `depth_max` | 30 m | 40 m | 60 m | 80 m |
|---|---|---|---|---|
| c = 7.8e-4 (seq 00) | 166.0 | 177.0 | **94.3** | 94.3 |

**Indoors it does nothing, which is why TUM could never have found it.** At the Kinect-derived
`c = 1.4e-3` and Z ≤ 8 m, σ_Z/Z is under 1 %: `fr1_xyz` 1.56 → 1.77, `fr1_desk` 2.53 → 2.34,
`fr1_room` 21.72 → 19.88, `fr3_walking_xyz` 26.96 → 29.20 cm. Net neutral, in both directions,
by small amounts.

#### The oracle had to be extended, and extending it found two more defects

Porting the term to `ecda.py` required the differential test to cover scenes with a real depth
gradient, and that immediately exposed things the 2.5 m fronto-parallel plane had been hiding:

1. **`select_points` had no `depthIsLocallyFlat`.** §26 listed it as unguarded; on a flat plane
   nothing is ever rejected, so the omission was invisible. Ported.
2. **Depth downsampling was `img[::2, ::2]`** while C++ takes the median of the valid values in
   each 2×2 block. The C++ comment states exactly why nearest is wrong — brightness uses 2×2
   averaging so output *x* represents input *2x+0.5*, and nearest grabs *2x*, shifting half a
   pixel per level. Harmless on a plane, not on a slope. Ported; it improves the simulated
   pose-graph RMSE by **22 %** (0.0220 → 0.0171).

Neither fix closed the pose gap on steep-depth scenes, and that is worth stating plainly:

| scene | py↔cpp converged-pose gap at c=0 |
|---|---|
| fronto-parallel plane, 2.5 m | **1.8 mm** |
| slope, 2–6 m | 5.6 mm |
| slope, 3–60 m | 16.3 mm |
| slope, 3–12 m | **68.2 mm** |

> **The "0.3 mm agreement" §25.17 reported was measured on the one scene where the two
> implementations agree.** It is not wrong — it is narrow, and its narrowness was invisible until
> a scene with depth structure was tried. The residual divergence tracks the LM damping and
> kernel-schedule differences the two implementations have always had (C++ computes the Huber
> threshold from the *previous* iteration's residuals, numpy from the current), amplified by
> depth spread.

**So the differential test had to change instrument.** Converged pose is the wrong observable for
this term: it is downstream of an LM path that already differs. The information matrix is not —
`Λ ≈ Σ w·J·Jᵀ` responds to the weighting directly, one accumulation, no convergence. Measured
that way the two implementations shrink the information by the same factor to within 8 %, and
the test asserts the *ratio* rather than the absolute value.

One more instrument choice, forced by measurement: `c` cannot be a constant across test scenes.
`σ_Z/Z = c·Z`, so a fixed `c` in a far scene drives the weights toward total suppression
(ratios diverged 0.236 vs 0.158 at 3–10 m) and the comparison starts measuring which handful of
points survived. The tests fix the *far-end relative depth error* at 5 % instead — which is where
KITTI actually sits (7.8e-4 × 60 m = 4.7 %).

Finally, `test_graph_optimisation_improves_pose_accuracy` was carrying a **stale
`xfail(strict=False)`** — it passed, and had been passing. Green whether it failed or not, which
is §19.3's failure mode wearing a different hat. Marker removed.

**Suite after this section: 236 C++ tests, 638 Python tests, 0 xfail.**

---

### 25.22 §22 was measured on a fraction of the data, and the tally moves both ways

`python/tools/tum_fetch.py` writes a sequence's `rgb.txt` / `depth.txt` index from the archive
listing and then extracts the frames. If the extraction is interrupted the index survives intact
and the frames do not, so the directory still *looks* complete: every loader reads the index,
finds the files that exist, associates those, and reports a clean run over them. Nothing in the
pipeline compares the index against the disk, so a sequence truncated to a tenth of its length
produces a plausible ATE with no warning anywhere.

**13 of the 16 TUM sequences behind §22 were in that state.** Coverage, counting scorable
rgb↔depth pairs against what the index declares:

| sequence | frames scored | frames available | coverage |
|---|---|---|---|
| fr2_desk_with_person | 258 | 4042 | **6.4 %** |
| fr2_desk | 256 | 2965 | **8.6 %** |
| fr1_teddy | 165 | 1419 | 11.6 % |
| fr1_plant | 165 | 1141 | 14.5 % |
| fr3_sitting_xyz | 258 | 1215 | 21.2 % |
| fr1_360 | 165 | 756 | 21.8 % |
| fr3_sitting_halfsphere | 259 | 1069 | 24.2 % |
| fr3_walking_halfsphere | 253 | 1017 | 24.9 % |
| fr1_desk | 164 | 596 | 27.5 % |
| fr3_walking_xyz | 258 | 826 | 31.2 % |
| fr1_xyz | 250 | 798 | 31.3 % |
| fr3_structure_texture_far | 720 | 904 | 79.6 % |
| fr3_structure_notexture_far | 722 | 789 | 91.5 % |
| fr1_room, fr3_nostructure_{notexture,texture}_far | — | — | **100 %** |

Every archive was re-fetched and all 20 sequences were re-scored end to end.
`python/tools/check_datasets.py` reports the mismatch — but it already existed when §22 was
written and nothing called it, so `bench_run.py` now performs the same comparison itself and
refuses to score a short sequence unless `--allow-partial` says that is intended. Verified by
hiding three of `fr1_xyz`'s 1596 frames: the run stops at 99.8 %.

**The three sequences that were already complete are the control, and they pass.** They should
reproduce exactly, and they do — ORB 48.06 → 48.06, 146.26 → 146.26, 34.12 → 34.12; WME 7.21 →
7.21 and 21.17 → 21.18 cm. The two at 80–92 % move by under 3 %. Everything below 35 % moves by
factors. **The disagreement is monotone in coverage**, which is what makes this a data defect
rather than a scoring change.

#### The corrected tally

ATE cm, WME's better variant against ORB+PnP, all 20 sequences:

| sequence | ORB, old | ORB, new | WME, old | WME, new | old winner | new winner |
|---|---|---|---|---|---|---|
| fr1_360 | 13.62 | *diverged* | 10.37 | 29.71 | WME | — |
| fr1_desk | 3.55 | 8.84 | 2.53 | 5.60 | WME | WME |
| fr1_plant | 3.37 | 15.01 | 4.51 | 6.19 | ORB | **WME** |
| fr1_room | 48.06 | 48.06 | 21.17 | 21.18 | WME | WME |
| fr1_teddy | 6.22 | 22.02 | 5.30 | 53.97 | WME | **ORB** |
| fr1_xyz | 2.61 | 4.05 | 1.53 | 4.54 | WME | **ORB** |
| fr2_desk | 0.98 | 29.06 | 2.38 | 11.14 | ORB | **WME** |
| fr2_desk_with_person | 0.73 | 9.59 | 0.94 | 7.42 | ORB | **WME** |
| fr3_nostructure_notexture_far | 146.26 | 146.26 | *no output* | *no output* | — | — |
| fr3_nostructure_texture_far | 34.12 | 34.12 | 7.21 | 7.21 | WME | WME |
| fr3_sitting_halfsphere | 1.26 | 101.47 | 1.50 | 9.54 | ORB | **WME** |
| fr3_sitting_xyz | 8.85 | 19.69 | 1.01 | 4.35 | WME | WME |
| fr3_structure_notexture_far | *diverged* | *diverged* | 12.20 | 12.58 | — | — |
| fr3_structure_texture_far | 3.30 | 3.12 | 1.95 | 2.00 | WME | WME |
| fr3_walking_halfsphere | 17.86 | 50.97 | 10.77 | 33.60 | WME | WME |
| fr3_walking_xyz | 19.34 | 38.50 | 6.48 | 33.88 | WME | WME |
| kitti_00 / 04 / 05 / 07 | — | 129.03 / 5056.93 / 160.25 / 304.09 | — | 94.32 / 954.78 / 141.48 / 107.93 | WME ×4 | WME ×4 |

**Against ORB alone: WME 15, ORB 2, 3 unscorable, across 20 sequences** — on TUM only, 11–2 with
3 unscorable, where §22 reported 12–4 across 14. Five verdicts flipped, and **they flipped in
both directions**: `fr1_plant`, `fr2_desk`, `fr2_desk_with_person` and `fr3_sitting_halfsphere`
moved to WME; `fr1_teddy` and `fr1_xyz` moved to ORB. A truncation that favoured one system
would have been easier to reason about than one that does not.

**And §22's own headline did not add up.** "12 to 4 across 14 scorable sequences" is sixteen
verdicts over fourteen sequences. Counting §22.4's table by hand gives **10–4**, and 10 + the two
unscorable rows = 12 — so the headline counted `nostructure_notexture` (where WME produced *no
output*) and `structure_notexture` (where ORB diverged) as WME wins, in the same section whose
closing paragraph says it excluded them. §22.5 wrote the rule and §22's summary line did not
apply it. That is independent of the data defect and was checkable at the time from the table
printed directly above it.

This retires a claim §22.2 made. "Where the baseline wins, it wins on fr2" was a statement about
256 of 2965 frames. On the complete sequence ORB goes 0.98 → 29.06 cm while WME goes 2.38 →
11.14, and fr2 becomes a WME win. §20.2's separate finding — that fr2 is where WME's *uncertainty
model* breaks — is untouched; it was measured elsewhere. What is gone is the ATE result that
§22.2 read as agreeing with it.

**What does not change is the mechanism.** §22.1 reported WME with lower RPE on every sequence
while losing ATE on four. On complete data that is **17 of 17 scorable sequences, ORB 0**, and
both remaining ATE losses carry the same signature — `fr1_teddy` at RPE 6.56 mm against 9.20 and
ATE 53.97 against 22.02 is 1.4× better per frame and 2.5× worse over the trajectory. The
structural claim about signed versus unsigned error survived a tenfold increase in data, which
is more than the tally can say for itself.

Medians move as expected once long sequences carry their real weight: RPE 6.91 → 8.77 mm (ORB),
4.62 → 6.71 (WME), 4.82 → 6.30 (WME+mask); ms/frame 29.9 → 42.7, 35.5 → 37.8, 108.6 → 127.2.

#### The token mask is not a free layer

Complete data makes something visible that 250-frame windows hid. The mask is what earns the
`walking` sequences — `walking_xyz` 124.61 → 33.88 cm, `walking_halfsphere` 63.29 → 33.60. It
also destroys three others: `fr1_plant` 6.19 → 48.33, `fr3_structure_texture_far` 2.00 → 8.55,
and `fr3_sitting_halfsphere` **9.54 → 4888.94 cm**. That last is a 48.9 m excursion in a
sequence where the camera orbits a fixed point at arm's length.

§22 credited the mask for the dynamic wins and did not price this, because on truncated data the
worst case was 8×. It is 512×. **A layer that is decisive on two sequences and catastrophic on
three is not a component, it is an open problem**, and §26 now says so.

That matters for the tally, because 15–2 is a *per-sequence best-of* — it takes the mask where
it helps and drops it where it does not, which is a choice no deployed system gets to make.
Scored the other way, one configuration everywhere:

| what is counted | WME | ORB | unscorable |
|---|---|---|---|
| best of {Tier 0, Tier 0 + mask} per sequence | **15** | 2 | 3 |
| Tier 0 alone, same configuration on all 20 | **13** | 4 | 3 |

The mask converts exactly the two `walking` sequences and nothing else. **13–4 is the number
that corresponds to a system rather than to a selection**, and it is the one §22's headline
should have been reporting all along.

#### The third-party control, re-run — and the TUM verdict is a tie

`cv2.Odometry` was re-run over the same complete sequences (75 minutes of wall clock; it is the
control that owes nothing to this repository, which is what makes it the informative one). ATE
cm, TUM only, against **the better of the two controls**:

| sequence | ORB+PnP | cv2.Odometry | WME Tier 0 | WME best | winner |
|---|---|---|---|---|---|
| fr1_360 | *diverged* | **24.42** | 29.71 | 29.71 | control |
| fr1_desk | 8.84 | 23.39 | **5.60** | 5.60 | WME |
| fr1_plant | 15.01 | 14.84 | **6.19** | 6.19 | WME |
| fr1_room | 48.06 | 72.29 | 21.72 | **21.18** | WME |
| fr1_teddy | 22.02 | **16.25** | 54.92 | 53.97 | control |
| fr1_xyz | **4.05** | 4.67 | 4.61 | 4.54 | control |
| fr2_desk | 29.06 | 45.93 | **11.14** | 11.14 | WME |
| fr2_desk_with_person | 9.59 | 80.84 | 7.77 | **7.42** | WME |
| fr3_nostructure_notexture_far | *diverged* | 78.57 | *no output* | *no output* | — |
| fr3_nostructure_texture_far | 34.12 | 20.60 | 7.74 | **7.21** | WME |
| fr3_sitting_halfsphere | 101.47 | 15.43 | **9.54** | 9.54 | WME |
| fr3_sitting_xyz | 19.69 | **3.61** | 4.35 | 4.35 | control |
| fr3_structure_notexture_far | *diverged* | **6.54** | 12.58 | 12.58 | control |
| fr3_structure_texture_far | 3.12 | 3.04 | **2.00** | 2.00 | WME |
| fr3_walking_halfsphere | 50.97 | 79.88 | 63.29 | **33.60** | WME |
| fr3_walking_xyz | 38.50 | 108.71 | 124.61 | **33.88** | WME |

| what is counted | WME | control | unscorable |
|---|---|---|---|
| TUM, better-of-two controls, WME best-of | **10** | 5 | 1 |
| TUM, better-of-two controls, **Tier 0 alone** | **8** | 7 | 1 |
| KITTI, ORB only, Tier 0 | **4** | 0 | 0 |

**On TUM, one configuration against two controls is 8–7.** That is a tie, and it is the number
this project is entitled to state. §22's 9–6 was the best-of figure on truncated data; the
best-of figure on complete data is 10–5, and the gap between 10–5 and 8–7 is the token mask being
chosen per sequence.

Two things from §22.4 survive intact and both are unflattering. `structure_notexture_far` —
descriptors starve, the direct method should own it — still goes to cv2, now **6.54 against
12.58**. And ORB's divergence there is not a scoring artifact: it happens on complete data too.
Meanwhile cv2 takes `sitting_xyz` (3.61 against 4.35), which §22 had listed among WME's decisive
dynamic wins.

> §22.4 wrote that "the count of controls is itself an experimental parameter." Complete data
> adds a second: **the count of configurations.** One control and a per-sequence best-of read as
> 12–4. Two controls and one configuration read as 8–7. No WME number changed between those two
> sentences — only what was allowed to be counted.

#### The divergence detector was measuring the wrong thing

`sitting_halfsphere`'s 4888.94 cm exposed a hole in §22.5's own guard. `run_status()` flagged
`diverged` when any *coordinate* exceeded 1 km — an indoor 45 s sequence cannot travel that far.
That run wandered **1105 m of path length inside a small box**, so no coordinate ever crossed the
line and a 48.9 m error was scored as an ordinary number. The check was on the wrong quantity,
and the constant was borrowed from one dataset's intuition anyway; it means nothing on a KITTI
drive where 550 m is correct.

The bound is now the ground truth's own path length, and the multiplier comes from the measured
gap rather than a guess. Across 56 runs the ratio splits cleanly in two:

| | ratio (estimated path ÷ ground-truth path) |
|---|---|
| healthy | 0.62 … **3.67** (worst: `nostructure_texture_far` / ORB) |
| *gap* | — |
| failures | **10.70**, 153.03, 7.3 × 10¹¹, 5.0 × 10¹³ |

6× sits inside that gap, 1.6× above the worst healthy run and 1.8× below the nearest failure —
roughly centred on a log scale. 10× would have landed within 7 % of the 10.70 case, and §10.4 is
the standing rule against thresholds placed where they measure rounding.

Re-scoring with it relabels exactly two runs — `sitting_halfsphere`/mask (153×) and
`nostructure_notexture_far`/ORB (10.7×, a camera jittering 33 m of path across 3 m of truth on a
blank wall) — and **moves no verdict**: 15–2 best-of and 13–4 single-configuration before and
after. A guard that catches more failures while changing no conclusion is the only kind that can
be added to a document of published numbers without re-opening all of them.

> The rule this section adds to §10.4: **a dataset is an input, and inputs get validated.**
> Every guard in this repository points at the code. Nothing pointed at whether the data the
> code consumed was the data it claimed to consume, and twenty sections of results were scored
> on an average of 35 % of TUM before anything noticed. The check that catches it compares two
> counts that were both already on disk — and a version of it had been sitting in `tools/` the
> whole time, never called. **A check that is not on the execution path is documentation.**

---

## 26. What is not established

- **Two dataset families now, and they disagree.** §12–§17 were five 9-second windows; §22 onward
  runs 16 TUM sequences including one full-length 45 s / 17.5 m traverse (`fr1_room`); §25.20 adds
  **four KITTI odometry sequences** through the new stereo depth front-end. On complete data
  (§25.22), one configuration against two controls: **TUM is 8–7 — a tie** — while KITTI is 4–0
  against ORB, the only control run there. Everything stated below as a property of WME may
  instead be a property of the dataset, and the honest reading of §22 is now "indoors,
  hand-held, 640×480". Still no EuRoC (host unreachable) and no Robotcar; KITTI sequences
  01–03, 06, 08–10 are on disk and unrun, and **cv2.Odometry has never been run on KITTI**, so
  the 4–0 has half the controls the 8–7 does.
- **ECDA's depth-uncertainty weighting is off by default and validated on four sequences of one
  dataset** (§25.21). Enabled at the derived `c = σ_d/(f·B)` it takes KITTI from 2–2 to 4–0, but
  it *worsens* sequence 04 by 2.5× — the highway drive where nothing is near — and on TUM it is
  neutral in both directions. `depth_sigma_rel` defaults to `0.0`, so no published number here
  depends on it. Whether the right `c` is always the sensor-derived one, or whether a scene with
  no near structure needs different handling, is unresolved.
- **The ECDA oracle agrees closely only on fronto-parallel scenes** (§25.21). C++ and numpy
  converge to within 1.8 mm on a 2.5 m plane and to **68 mm** on a 3–12 m slope, at the default
  configuration, before any new term is switched on. The cause is the LM damping and Huber
  schedule differing between implementations, amplified by depth spread. The information-matrix
  comparison agrees to 8 % on the same scenes, so the accumulation is right and the convergence
  path is not shared — but "the two implementations agree" now needs the qualifier.
- **Dynamic masking is not solved.** §14.1 shows class-based masking trades a 3.96× win on
  moving people for a 15.7× loss on seated ones. §14.3 shows the belief-driven replacement does
  not yet accumulate enough evidence in an 8-second window to tell them apart. Whether it does
  over minutes is untested, and that test needs longer sequences than disk allowed here.
- **YOLO runs on CPU at 60–150 ms/frame.** That is 7–16 FPS for detection alone against a 60 FPS
  target. The TensorRT backend and GPU pipeline are unimplemented; no throughput claim is made.
- **Only `person` ever fires on TUM.** The mobile-class list covers vehicles and animals, none of
  which appear in these sequences. The class-to-affordance mapping is otherwise unexercised.
- **All three tiers now run on real data, and only Tier 0 earns its place.** §18 measured the
  full fusion on TUM, §23.4 swept it under degradation, and §24 ran TCG in a real loop-closure
  loop. Tier 1 accepted one loop in 45 s and it was wrong; Tier 2 helps only where Tier 0 has
  already diverged. **§1's three-tier result remains unreproduced on a real sensor** — the slope
  survives, the magnitude does not (§23.4).
- **The stereo depth front-end is validated only against itself.** §25.20–25.21 ran KITTI, so
  the ATE numbers this bullet used to say were missing now exist — but the depth those numbers
  rest on has been checked against TUM's measured depth on a *self-consistently warped right
  view*, which is an optimistic bound (§25.19), and never against KITTI's own Velodyne. The
  4–0 result therefore inherits an unquantified depth error. EuRoC's dataset host
  (`robotics.ethz.ch`) still times out from here, so that family remains unreachable.
- **The token mask is decisive on two sequences and catastrophic on three** (§25.22). It buys
  `walking_xyz` 124.61 → 33.88 cm and `walking_halfsphere` 63.29 → 33.60, and costs `fr1_plant`
  6.19 → 48.33, `structure_texture_far` 2.00 → 8.55, and `sitting_halfsphere` 9.54 → **4888.94**.
  Truncated data capped the worst case at 8× and hid the failure mode entirely, and nothing
  here predicts which sequences it will ruin. This is load-bearing rather than incidental: the
  headline tally scores WME's *better variant*, so the mask is inside the published number on
  every sequence where it helps and silently discarded where it does not. A per-sequence
  best-of is a weaker claim than a single configuration run everywhere, and §22 does not
  currently say so.
- **ORB-SLAM3 is routed but not built.** Six Windows forks were checked and every one is stale —
  the newest MSVC-capable port (`lydieusang/orbslam3-windows`) last moved in 2022 and predates
  ORB-SLAM3 v1.0; there is **no vcpkg port**, and vcpkg's `g2o`/`DBoW2` cannot substitute for
  the patched copies ORB-SLAM3 vendors. The real Windows blockers are OpenSSL MD5, Boost
  serialization on MSVC, `SHARED` libraries with no `__declspec(dllexport)`, and Pangolin
  headers that are non-optional at *build* time even though the viewer is a runtime flag.
  Estimated 12–20 h for a stubbed MSVC port versus 2–4 h on WSL2, where the viewer is simply
  `bUseViewer = false`. **WSL2 is the chosen route and it is not installed on this machine**
  (`wsl --status` reports absent), so it needs an elevated install and a reboot. Until that
  happens the published-baseline gap below stands.
- **No *published* baseline.** §22 and §24 close the "compares only to itself" gap — a classical
  ORB+PnP front-end and a published `cv2.Odometry` run on all 16 complete sequences (**8–7** to
  WME against the better of the two, one configuration) and a symmetric loop-closure back-end
  runs on the full `fr1_room` traverse — but **ORB-SLAM3, DSO and DROID-SLAM have still not been
  run.** A self-implemented control is permanently open to the under-tuning objection no matter
  how carefully its settings are sourced, and there is still no bundle adjustment on either
  side, only pose-graph optimisation.
- **TCG's indoor loop closure works but has very low recall, and no knob fixes it.** 3 accepted
  loops from 25 possible queries against ORB's 44 from 1292 proposed pairs; edge accuracy 15.4 cm
  median against 2.14 cm. Both density levers were measured and both fail: a temporal window
  degrades edge accuracy 5× (§25.11), and a lower detection threshold *lowers* accepted loops
  3 → 0 → 2 while raising possible queries 25 → 54 (§25.12). Node quality, not node count, is the
  constraint — an office viewpoint yields ~3 well-localised COCO objects and the signature needs
  4. §2's distractor-set precision has never been run on real data.
- **Two defects of mine corrupted §24's published TCG result** (§25.11): chirality was silently
  disabled and the match transform was stored inverted. Corrected, TCG loop closure goes from
  −11.2 % to **+30.7 %**. Both were invisible in the output — a wrong-direction transform yields
  a plausible number, not an error — which is why §10.4's rule now has to be applied to the
  harness and not only to the estimator.
- **§1's degradation slope is qualitatively confirmed and quantitatively deflated** (§23.4):
  the gain does grow with haze and does so without any weather branch, but 53.6 % becomes
  **11 %**, and it only turns positive after Tier 0 has already diverged to 44–105 cm. The
  large simulated number is substantially a property of the renderer.
- **Neither system degrades gracefully** (§23.3), and they fail in opposite ways: the classical
  front-end stops and reports it, WME continues and does not. Degraded-condition robustness —
  the motivating claim for dropping descriptors — is **not** supported.
- **The degradation is synthetic, even though the data is real.** Haze is applied to real
  frames using real depth, which fixes the transmission model, but airlight is a constant and
  there is no wavelength dependence. All three channels are verified to behave (§23.1), but
  **only haze has been swept end-to-end** — darkness and motion blur are built and checked, not
  yet run through the estimator comparison. No naturally-degraded dataset (Robotcar night/rain)
  has been run.
- **Loop closure exists; bundle adjustment does not.** §24 adds a pose-graph back-end shared by
  both systems, so drift is no longer unbounded *when a loop is detected and accepted*. There is
  still **no bundle adjustment on either side** — no landmark refinement, no windowed
  marginalisation — so the comparison is odometry + pose graph, not full SLAM.
- **A +3.4 % scale bias on `fr1_desk`** (ATE 2.53 → 1.66 cm if scale is fitted) that is only
  +0.7 % on `fr1_xyz`. Sequence-dependent, so not a depth-scale constant; most likely rotation
  error leaking into translation. Unexplained.
- **Calibration is within 1.6×, not inside the band** (§15.4). The residue is structured — it
  lives in the correlations, not the marginals — so closing it needs a model of the frame-level
  error, not a constant. What that error physically is remains unidentified: it scales with
  residual RMS (Spearman +0.60…+0.80) but correlates only weakly with motion within a sequence
  (+0.11…+0.37), so rolling shutter, RGB/depth timestamp mismatch and depth-registration bias are
  all still live candidates. Depth *scale* error is excluded on magnitude (would need 22 %).
- **`ν = 1` is calibrated on one dataset, one sensor, five 9-second windows.** It encodes the
  measured fact that these photometric residuals carry one frame's worth of information. On data
  where they really are independent it is conservative by the point count, and `ResidualVariance`
  is the correct mode there. This is written into the config header, not left implicit.
- **`DirectAligner` now has an element-level oracle, and it agrees.** The blocker was one line:
  `ecda.py` still carried `huber_delta = 12.0`, the fixed-threshold kernel C++ deleted in §11.4,
  so the two implementations selected and weighted *different points by construction*. The
  adaptive kernel is ported (`robustDelta`, plus §13.4's separate fixed-threshold inlier count),
  and `test_differential_ecda.py` compares point counts, inlier ratios and converged poses.
  **The two implementations agree to 0.3 mm while both sitting 21 mm from the synthetic truth** —
  a common error that belongs to the crude test warp, not the engines, which is exactly the
  distinction a differential test exists to draw. The old ECDA items below are superseded;
  what remains unguarded is `depthIsLocallyFlat` and the LM damping floor.
- **`pyramid` construction, depth median downsampling, `depthIsLocallyFlat`, the LM damping
  floor and the `level_observable_ratio` projection remain unguarded** — §25's ECDA differential
  tests compare converged poses and aggregate counts, not these internals directly.
- **SPA's Python reference carried §7.1's defect the whole time, plus a second one** (§25.16).
  The oracle found both within minutes of existing: rotational information was the scatter
  `Σ w·nnᵀ` instead of the complement `Σ w·(I − nnᵀ)` — §7.1 corrected the published table from
  the C++ port but never the reference — and the translation weight was applied twice
  (`Σ w²·nnᵀ`). Both fixed; the two now agree to `rtol=1e-9` at every weight. §7's table and
  §18.3 came from the C++ and are unaffected.
- **`PlaneExtractor` now has differential coverage too** — same synthetic two-plane depth map to
  both implementations, normals matched to cos > 0.99 and offsets to 5 cm, with a curved-surface
  control that must yield ≤ 1 plane (or the agreement would only show both extracting the same
  garbage).
- **`EnvironmentAnalyzer`'s image path now has an oracle, and it found a defect on first
  contact** (§25.17). `wme/reference/environment_cues.py` reimplements the seven single-frame
  cue estimators in numpy — including OpenCV's fixed-point `BGR2GRAY` and `BGR2HSV` integer
  paths, because the float versions differ by 1 gray and that 1 changes a histogram bin. The
  comparison runs through the **public** `update()` with `evidence_ema = 1.0` rather than
  exposing the private estimators, so what is measured is the path that actually runs. 106 tests;
  `camera_shake` (phase correlation) and `rain_streak`/`snow_particle` (temporal ring buffer)
  are still uncovered because they need frame history. `fusion::fuse`, `TierEstimate`,
  `FusionResult` and the SE(3) left-Jacobian pair are bound, and **16 differential tests**
  compare them against `tools/fusion_replay.py` — the oracle that already reproduced 40 C++
  ablation trajectories to 0.002 % (§21). They cover the fused pose, the fused information
  matrix, all three abstention paths, and a 10⁶ tier-scale gap, each with a discrimination guard
  requiring the tiers to actually contribute. A mutant that drops the `α·κ` weight is caught by
  8 of them. **The §18 fusion result now has an oracle; §23's SPA path still does not.**
- **The `rtol` sweep is done and found nothing** (§25.16). All 27 remaining loose assertions
  were tightened to `rtol=0` and the suite still passes, so the two implementations agree to
  1e-12 absolute. §20.3's prediction that sweeping "will likely surface more divergences" did not
  hold.
- **The `align` crash was never explained.** It stopped reproducing after a binding edit and no
  reproducer survives — 432 calls across 144 configurations, 8 image shapes, 20 repeats of the
  original script, zero failures. Without version control there is nothing to bisect. The
  circumstantial evidence points at the `toMat` fix, but that is not confirmation.
- **A latent size-rule disagreement**: `ImagePyramid::build` downsamples with
  `cvRound(n·0.5)` while `DirectAligner::buildDepthPyramid` uses `src.rows / 2`. For odd sizes
  these differ (15 → 8 vs 7), so the gray and depth pyramids disagree at that level. Currently
  unreachable — `selectPoints`' loop bounds keep the read one row inside — but only the loop
  bounds stop it from being an out-of-bounds read, and the two functions share no size rule.
- **`motion_noise_floor = 0.035` is circular.** Bolted-down chairs show 26 mm of window
  displacement on `sitting` and 51 mm on `walking`. Nothing about the chairs changed — the
  difference is the odometry's own error entering as common mode. A per-frame common-mode
  correction is the right fix, and §13.4's rule blocks the obvious version of it: a threshold
  that adapts to the data cannot also be the yardstick that measures it.
- **One scalar per object cannot describe a gesticulating person.** The 1066 mm single-window
  displacements on seated people are real arm motion, not sensor noise. Their torsos are
  excellent landmarks and their arms are not; the robust likelihood *suppresses* this rather than
  modelling it. Sub-token motion state is unaddressed.
- **Four constants** (`static_outlier_rate` 0.2, `dynamic_speed_ref` 0.7, `assoc_maneuver_speed`
  2.0, `motion_noise_floor` 0.035) are calibrated on two fr3 sequences. Only the last two have a
  direct measurement behind them.
- **No per-object motion ground truth exists in TUM**, so belief convergence is validated by ATE
  and threshold-crossing behaviour, never against true per-person velocity. No NEES on the belief.
  The negative case (`walking`) is confirmed over 8.6 s only — no 40 s window was fetched for it.
- **Determinism is verified for four components in isolation on one compiler.** Bit-identical
  across worker counts, repeated calls, live instances and processes, with mutation testing to
  prove the harness can fail. There is no recorded-log replay, no cross-machine golden file, and
  nothing covering YOLO/ORT, SPA, memory or prediction.
- **The two order-dependency fixes are unobservable on this toolchain.** `TokenStore.cpp`'s
  `dynamic_area` sum and `ConstellationIndex`'s tie-break were both fixed, and both fixes were
  then *reverted to check* — the tests still passed. MSVC's `unordered_map` uses identity hashing
  for integers with bucket counts far above element count, so dense keys iterate ascending, which
  coincides with ID order; and the `float` box areas sum exactly in `double` for any plausible
  token count. The fixes pin a convention that libstdc++/libc++ (prime buckets, modulo) would
  break. **They correct an unobserved dependency, not an observed error.** A third remains:
  `buildStaticMask`'s `MaskReport` attribution split — the §16.5 provenance numbers — is
  hash-order dependent, though the mask itself is not.
- **Allocation, after the fixes, is zero on four of five hot paths** — `align` went 1229 → 0
  (the last 158 were not in `parallelFor` at all but at the call site, where a 10-reference
  lambda overflowed MSVC's `std::function` SBO on every one of ~175 calls per frame).
  `TokenStore::integrate` remains at 27 allocations / 2.6 KB from `WorldToken::history` and
  `trajectory` deque nodes, which need a data-structure change with wide blast radius. The
  `cv::Mat` counts are a lower bound throughout: OpenCV's `AutoBuffer` and direct `malloc` are
  invisible to both counters.
- **The C++ now has an end-to-end wiring test, not an end-to-end accuracy test.**
  `test_pipeline.cpp` runs camera → quality → pose (5 frames, accumulated pose must grow),
  detections → tokens → constellation (self-query must return identity), and three tiers →
  one fused pose, plus a control asserting a broken stage fails *loudly*. It checks that each
  stage's output reaches the next — which is what §11.3, §14.2 and §16.5 all failed at while
  every subsystem test passed. It does **not** check accuracy end-to-end, and C++ and Python
  are still not cross-checked numerically at the pipeline level.
- **`α_k(E)` is now fitted, and fitting it does not help** (§21.3): every leave-one-out fit
  lands at `α₁, α₂ ≤ 0.01`, i.e. "switch the other tiers off", winning 0 of 5. The schedule
  was never the binding problem. What replaced it — a χ² consistency gate (§21.4) — is fitted
  on the same five 9-second windows and anchors on Tier 0 being the accurate tier, which is an
  *input* to the rule and unestablished wherever that ordering does not hold.
- **The complementarity mechanism is still unsupported on real data.** §18.3 measured it
  absent; §21.5 shows that repairing fusion did not bring it back (selectivity 0.99–1.14, and
  1.09× on the only sequence with genuinely rank-deficient frames). The tiers-fill-each-other's-
  null-space claim rests on simulation alone.
- **§21.4's gain on `fr3_walking_xyz` is unexplained.** The bias mechanism that accounts for
  the other three wins runs the wrong way there (bias +54 %, ATE −11 %). One of five rows in
  the explanation does not fit it, and is not counted as support.
- **The gate's threshold is not a calibrated probability.** `Λ_TCG` is up to 47× overconfident
  (§18.1), so nominal `p = 0.50` is far tighter in truth than it reads.
- **Photometric SLAM uncertainty remains 5–17× overconfident** with poses in the estimate
  (`xfail` in `test_photometric_slam.py`), though accuracy improves 32–36 %.
- **Simulation only, for §1–§10.** Every number there comes from a renderer whose noise model I
  wrote. That it recovers its own parameters is necessary, not sufficient.

The compiler and the dataset both now exist, and between them they overturned the headline
(§18), explained it (§21), put the engine against an external system (§22), manufactured the
degraded data the headline needed (§23), and closed the loop-closure gap (§24).

What the last four sections have jointly established is narrower and more useful than the
architecture's own statement of itself:

> **The direct, descriptor-free front-end is the real contribution.** It is 2.2× better than
> the classical front-end over a full traverse and better per-frame on 12 of 12 sequences.
> **Everything built on top of it is not yet earning its place** — the object tier cannot close
> loops indoors (§24.2), the three-tier fusion is net harmful except where everything has
> already failed (§23.4), and the degradation robustness that motivated dropping descriptors
> does not appear (§23.3).

The single most valuable next step *was* a failure detector for Tier 0. §25 delivers the signal —
`depth_consistency` tracks the divergence at 10.4× where the photometric channels saturate at
4.3×, because a bounded signal cannot report an unbounded failure — calibrates it across three
cameras, and wires `align()` to degrade its own result on it.

**And then finds that acting on it makes fusion worse on 8 of 15 configurations** (§25.7).
Down-weighting Tier 0 does not create accuracy; it hands weight to tiers that are 3–15× less
accurate. It helps only on `fr3_walking_xyz`, the one sequence where a genuinely better
alternative exists.

Two consumers were then tried. Reweighting fusion **fails** (§25.7) because down-weighting the
best tier only hands weight to worse ones. Keyframe replacement — an action needing no second
estimator — is the first net-positive consumption: **4 better, 2 worse, 2 tied** in the regime
where the trajectory is still usable, helping on the four hardest sequences and costing on the
two easiest (§25.8).

Relocalization was then built and measured too (§25.9). On clean data it recovers real drift —
`fr3_walking_xyz` reaches **15.72 cm from 20.72**, 24 % for the whole chain. **Under haze it
succeeds 0 times in 358 attempts**, because it matches descriptors against map keyframes that
were captured in the same haze: both sides of the comparison are degraded by the same cause.

That is where this line of work ends, and it ends pointing at something specific:

> **Detection and recovery are separate problems.** An independent channel for *detecting*
> failure does not give you an independent channel for *recovering* from it. Every recovery
> mechanism implemented here lives in the photometric channel that the degradation destroyed.
> The only recovery path in the architecture that does not is the object constellation — and
> §24.2 measures that one as unable to relocalize indoors.

TCG's density problem was then measured rather than assumed (§25.10), and the diagnosis in §24.2
turns out to be too broad. Depth validity costs 5 %; requiring distinct classes *halves* the
count; the confidence threshold helps but at precision. **A 5-keyframe query window takes
`fr1_room` from 34 % to 97.8 % of keyframes qualifying, without loosening any threshold.** The
bottleneck was the query unit, not the detector — and `tum_relocalize.cpp`'s own header already
said a real relocalizer queries with a local map rather than one frame.

The windowed query was then built and run (§25.11), and it **spends more accuracy than it buys
coverage**: node count triples, edge error grows 5×, and pose-graph gain falls from +30.7 % to
+1.1 %. Node count was the wrong quantity to optimise.

Removing that idea also required removing two defects of my own — chirality silently disabled,
and `ConstellationMatch::transform` stored with its direction inverted. The second had corrupted
**§24's published result**: TCG loop closure was reported as *degrading* ATE by 11.2 %; corrected,
the same code on the same data improves it by **30.7 %**, and the WME full system goes from
losing to the classical one to beating it — **15.06 cm against 20.67**.

So this is where the work stands:

> **WME's descriptor-free front-end is the result, and loop closure now amplifies it rather than
> undoing it.** TCG's real limit is recall, not correctness: 3 loops from 25 possible queries
> against ORB's 44 from 1292 proposed pairs, at 15.4 cm median edge accuracy against 2.14 cm.
> Object density is the binding constraint, and the obvious fix for it — accumulate over time —
> makes the geometry worse.

That question — can TCG's recall be raised without the precision loss windowing caused — was then
tested on the only other lever available, the detection threshold, and the answer is no (§25.12).
Lowering it raises possible queries 25 → 54 and *lowers* accepted loops 3 → 0 → 2, because the
spurious detections stop the signature matching and the accept rule declines.

**Node count was never the binding constraint; node quality was.** The window supplies quantity
at the cost of localisation, the threshold at the cost of presence, and a constellation needs
four objects that are present, well-localised and stably detected. An office viewpoint yields
about three. Neither knob on the existing pipeline can manufacture the fourth, so the remedies
left are outside it: a detector whose classes cover indoor scenes, or a node primitive that is
not a COCO detection.

The other primitive already available — SPA's planes — was then measured and **rejected before
being built on** (§25.13): 2.77 planes per keyframe against 3.15 objects, and the same physical
wall gives centroids **1.5 m apart** across nearby viewpoints, because a plane's centroid is the
centroid of its *visible portion*.

> **The node primitive must be plentiful indoors *and* viewpoint-stable, and neither candidate is
> both.** COCO objects are stable and too few; plane centroids are neither. Two independent
> primitives rejected for opposite halves of the same requirement.

Plane *intersections* were then implemented and measured too, and fail on a counting argument
(§25.14): a corner needs three non-parallel planes and this scene supplies 2.77, giving **0.35
corners per keyframe**. Stable by construction, and almost never available.

Five candidates, all measured, all rejected (§25.15). The picture they make is one claim:

> **The constraint is landmark supply, not the constellation algorithm.** A descriptor pipeline
> extracts ~1000 keypoints per frame here and closes 44 loops at 2.14 cm. Every primitive WME's
> pipeline can currently produce yields **about three** — a ~300× density gap — and a
> constellation needs four. No rearrangement of the object pipeline closes a gap that large,
> which is why five attempts fail at the same place by different routes.

That locates the remaining work outside this subsystem: a detector or segmenter whose output is
dense indoors, not another way of assembling what YOLO already returns. §2's simulated results
are not contradicted — they were measured where objects were guaranteed to exist, and that
guarantee is what an office withdraws.
