# The Correspondence Problem under the YOLO-Only Constraint

> This is the single most important design document in WME. Every other subsystem
> is downstream of the decision made here.

## 1. The problem statement

The classical visual SLAM pipeline is:

```
image ──► keypoint detector ──► descriptor ──► matcher ──► PnP / epipolar ──► BA ──► pose
          (FAST/ORB/SIFT)      (BRIEF/SIFT)   (BF/FLANN/
                                               LightGlue)
```

The WME constraint removes **every stage before PnP**:

| Banned | Category |
|---|---|
| ORB, SIFT, SURF, AKAZE | hand-crafted detector + descriptor |
| SuperPoint, LoFTR, LightGlue, DISK | learned detector / matcher |
| SAM, DINO, CLIP, GroundingDINO, MaskRCNN | auxiliary perception networks |

What remains available:

1. **Raw pixel intensity** `I(u)` and its spatial derivatives `∇I(u)`.
2. **Brightness-constancy solvers** (differential optical flow) — these solve a PDE,
   they do not detect or describe anything.
3. **YOLO output**: class, box, mask, objectness, per-class logits.
4. **Depth**, where the sensor provides it (stereo / RGB-D / active).

### 1.1 Is a spatial gradient a "feature extractor"?

No, and the distinction matters legally and scientifically. A Sobel/Scharr derivative is
a linear filter defining the image's differential structure — it is the object on which
the photometric error is defined. It produces no keypoint set, no descriptor vector, and
no matching. Banning it would ban the brightness-constancy equation itself, i.e. ban
optical flow, i.e. ban every non-descriptor method ever published. WME therefore treats
`∇I` as a **primitive of the image**, not as a feature.

What WME does *not* do, and will not do: build a repeatable keypoint set, compute a
descriptor around it, or perform nearest-neighbour descriptor matching. There is no
`detect()`, no `compute()`, no `match()` anywhere in this codebase.

## 2. Why this constraint is a feature, not a bug

Descriptor-based SLAM has three failure modes that are *structural*, not tunable:

**(a) Descriptors are photometric, so they die exactly when you need them.**
ORB is a set of intensity comparisons. At night, in fog, under rain streaks, or through a
dirty lens, the comparisons randomise. Relocalization accuracy collapses precisely in the
conditions where a robot most needs to know where it is. Every "robust descriptor" paper is
an attempt to patch a representation that is fundamentally a local brightness pattern.

**(b) Descriptors have no identity, only appearance.**
Two identical chairs produce near-identical descriptors. Descriptor matching therefore
*cannot* distinguish "I have returned to the same chair" from "I am looking at the other
chair." Loop closure geometric verification exists solely to clean up this ambiguity, and
it is the dominant source of catastrophic map corruption.

**(c) Descriptors describe texture, and texture is not what changes.**
A world model must answer *what changed*. A corner of a floor tile is not a thing that can
change. An object is. Descriptor SLAM builds a map of texture patches and then, separately,
bolts semantics on top. The map and the meaning are never the same data structure.

The YOLO-only constraint forces the correspondence layer to be built out of **objects and
photometric fields** rather than texture patches. That is not a handicap; it is the correct
representation for a world model, and it is what makes the rest of WME possible.

## 3. WME's answer: a three-tier correspondence stack

WME does not have *a* correspondence method. It has three, operating at different spatial
and temporal scales, fused in a single factor graph with **environment-conditioned weights**.

```
                    ┌─────────────────────────────────────────┐
                    │        EnvironmentState + IQS           │
                    │   (drives all three tier weights)       │
                    └──────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
  ┌───────────┐            ┌──────────────┐          ┌────────────────┐
  │  Tier 0   │            │   Tier 1     │          │    Tier 2      │
  │   ECDA    │            │    TCG       │          │     SPA        │
  │ photometric│           │  semantic    │          │  structural    │
  │  field     │           │ constellation│          │  primitives    │
  ├───────────┤            ├──────────────┤          ├────────────────┤
  │ high rate │            │ low rate     │          │ mid rate       │
  │ short     │            │ unbounded    │          │ mid baseline   │
  │ baseline  │            │ baseline     │          │                │
  │ 6-DoF     │            │ 6-DoF +      │          │ 3-5 DoF        │
  │ relative  │            │ reloc + loop │          │ (constrains    │
  │           │            │ closure      │          │  rot + normal) │
  └───────────┘            └──────────────┘          └────────────────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Unified Factor Graph │
                        │  (Localization Engine)│
                        └──────────────────────┘
```

### Tier 0 — Environment-Conditioned Direct Alignment (ECDA)

**What it is.** Estimate relative pose `T ∈ SE(3)` by directly minimising photometric
residual over a set of high-gradient pixels with maintained inverse-depth estimates:

```
E(T, a, b) = Σ  w(u) · ρ_Huber( I₂( π(T · π⁻¹(u, d_u)) ) − (a·I₁(u) + b) )
             u∈P
```

with `(a, b)` an affine brightness model absorbing exposure and gain changes, `ρ_Huber` a
robust kernel, and a coarse-to-fine pyramid for basin-of-attraction.

**What is new.** Three things, and they are the reason ECDA is not "DSO again":

1. **The weight `w(u)` is derived from a physical model of the scene condition**, not from
   residual statistics alone:

   ```
   w(u) = w_grad(u) · w_quality(u) · w_dynamic(u) · w_env
   ```

   - `w_grad` — gradient magnitude saliency (standard).
   - `w_quality` — from the per-pixel output of the Image Quality Engine: local blur
     estimate, local saturation (blown highlights / crushed blacks carry zero photometric
     information), local noise variance. A pixel in a blown-out region gets `w ≈ 0` rather
     than contributing a residual that the Huber kernel must then fight.
   - `w_dynamic` — **from YOLO**. Every pixel inside the mask of a token whose motion model
     says it is moving is excluded. This is the direct-method killer that classical direct
     SLAM never solved: DSO/LSD-SLAM assume a static world and degrade badly with a person
     walking through frame. WME has an object-level dynamics model, so it can zero those
     pixels *before* they poison the linear system, rather than hoping robust statistics
     absorbs them.
   - `w_env` — a global scalar from `EnvironmentState`. Under fog, the photometric signal
     is attenuated by a known transmission model; ECDA's total information contribution is
     down-weighted accordingly so the factor graph automatically leans on Tier 1/2.

2. **Residuals carry calibrated covariance, not just a robust kernel.** ECDA outputs an
   information matrix `Λ = Jᵀ W J` scaled by a *measured* photometric noise model
   (estimated per-frame from the noise floor by the Image Quality Engine), so the factor
   graph gets a physically meaningful confidence rather than an arbitrary weight.

3. **Degeneracy is detected, not discovered.** The eigen-spectrum of `Λ` is analysed each
   frame. A textureless corridor is rank-deficient along its axis; ECDA reports *which*
   DoF are unobservable and emits a rank-deficient factor rather than a wrong full-rank
   one. This is what allows Tier 2 to fill in exactly the missing DoF.

**Why not just optical flow + PnP?** Because that reintroduces a discrete correspondence
set and its outliers. Direct alignment keeps the problem continuous and differentiable and
lets the photometric weighting above act per-pixel.

### Tier 1 — Token Constellation Geometry (TCG)

This is WME's primary original contribution and its answer to relocalization and loop
closure without descriptors.

**Core idea.** A place is not a bag of visual words. A place is *a set of objects in a
specific spatial arrangement*. That is a strictly stronger and more stable signature than
texture, and YOLO gives it to us directly.

**The constellation signature.** For a set of `n` co-visible tokens with 3D positions
`{x_i}` and class labels `{c_i}`, define a signature that is invariant to rigid transform
and to token ordering, and robust to partial observation:

- **Class multiset** `M = {{c_i}}` — a coarse, extremely cheap pre-filter.
- **Pairwise distance spectrum** — the multiset of `(c_i, c_j, ‖x_i − x_j‖)` triples,
  quantised into log-spaced distance bins. Rigid-invariant by construction.
- **Chirality word** — for each ordered class-distinct triple, the sign of the scalar
  triple product with the gravity vector. This breaks the mirror ambiguity that pure
  distance spectra suffer from, and gravity is available from IMU or from the Geometry
  Engine's dominant-plane normal.

The signature is hashed into an inverted index. Retrieval is a multiset intersection —
`O(k)` in the number of query bins, independent of map size in the common case.

**Verification and pose recovery.** Retrieval proposes candidate constellations. For each:

1. Build the class-consistency bipartite graph (a `chair` may only match a `chair`).
2. Find the maximum common subgraph under a distance-consistency constraint. Because class
   labels partition the correspondence space, the search is dramatically smaller than
   descriptor matching's — for a scene of 20 objects across 8 classes the hypothesis space
   is tractable by branch-and-bound, no RANSAC lottery required.
3. Recover `T` in closed form by Umeyama/Kabsch on the matched 3D centroids.
4. Accept on a chi-square test against the tokens' own position covariances.

**Why this is better than a bag of visual words.**

| | BoW / descriptor loop closure | Token Constellation Geometry |
|---|---|---|
| Signature basis | local texture | object identity + metric layout |
| Night / fog | fails (texture gone) | survives (YOLO still fires on large objects) |
| Viewpoint change | fails beyond ~40° | invariant by construction (3D, rigid-invariant) |
| Perceptual aliasing | severe (identical corridors) | reduced (class *composition* differs) |
| Pose from match | needs PnP + RANSAC | closed-form Kabsch |
| Map size scaling | vocabulary tree, retraining | inverted index, no training |
| Output | a pose | a pose **and** a semantic association |

That last row is the deep one. A descriptor loop closure tells you "you are here." A
constellation loop closure tells you "you are here, **and this chair is the same chair you
saw before**." The second statement is a world-model update. It is what allows the Memory
Engine to merge object histories across a loop instead of duplicating them.

**Honest limitations.** TCG needs ≥4 well-localised, class-distinct-ish tokens. It fails in
an empty white corridor. It is sensitive to 3D token position error, which is why token
positions carry full covariance and the chi-square gate is strict. It cannot relocalise in a
scene whose objects have all moved. These are real, and they are exactly why Tier 0 and
Tier 2 exist.

### Tier 2 — Structural Primitive Alignment (SPA)

Planes, lines, and the gravity direction, extracted by the Geometry Engine from depth /
multi-view. Constrains rotation strongly and translation along plane normals. Its role in
the stack is **degeneracy repair**: when ECDA reports rank deficiency along a corridor axis
and TCG has too few tokens, SPA's wall planes supply the missing rotational and lateral
constraint. Cheap, always available indoors, and completely independent of appearance.

## 4. The fusion rule

All three tiers emit factors into one graph. The adaptation described in the WME spec
("night → increase temporal memory", "fog → increase memory dependency") is implemented
here, concretely, as **information reallocation**:

```
Λ_total = α_0(E) · Λ_ECDA + α_1(E) · Λ_TCG + α_2(E) · Λ_SPA + Λ_prior(E)
```

where `α_k(E)` are functions of `EnvironmentState`. Examples of the learned/calibrated map:

| Condition | α₀ (photometric) | α₁ (constellation) | α₂ (structural) | prior |
|---|---|---|---|---|
| Bright indoor, static | 1.00 | 0.6 | 0.4 | low |
| Night, low light | 0.25 | 0.9 | 0.7 | high (motion prior) |
| Fog / smoke | 0.10 | 0.8 | 0.9 | high |
| Heavy rain | 0.55 | 0.9 | 0.5 | mid |
| Motion blur burst | 0.05 | 0.7 | 0.6 | very high |
| Textureless corridor | rank-deficient | 0.3 | 1.00 | mid |

This is the entire "Adaptive Engine" reduced to one honest equation. There is no magic —
degradation of a sensing modality is expressed as reduction of the information it
contributes, and the estimator handles the rest correctly by construction.

## 5. What this buys the World Model

The correspondence layer and the semantic layer are **the same layer**. Tokens are
simultaneously:

- the landmarks that constrain pose (Tier 1),
- the dynamic masks that protect pose (Tier 0's `w_dynamic`),
- the nodes of the World Graph,
- the entities the Memory Engine tracks,
- the objects the Prediction Engine forecasts,
- the affordances the Planner reasons over.

In descriptor SLAM these are six different data structures with a fragile mapping between
them. In WME there is one: `WorldToken`. That unification is the actual thesis of this
engine, and it is only reachable *because* the descriptor route was closed off.

## 6. Evaluation plan

To be defensible at a venue, the following comparisons are required:

- **Baselines**: ORB-SLAM3 (mono/stereo/RGB-D), DSO, DROID-SLAM, and an ablation of WME
  with each tier disabled.
- **Datasets**: TUM-RGBD (dynamic sequences especially), EuRoC, KITTI, Oxford Robotcar
  (night/rain traversals), and a purpose-collected adverse-condition set.
- **Headline claims to test**:
  1. WME relocalization recall under night/fog exceeds BoW baselines by a wide margin.
  2. WME ATE on dynamic sequences beats ORB-SLAM3 due to token-driven dynamic masking.
  3. WME degrades gracefully (no catastrophic loss) where baselines lose tracking.
  4. Constellation loop closure produces **zero** false positives at operating threshold on
     perceptually aliased sequences where BoW produces many.
- **Ablations that must be reported honestly**: performance in object-sparse scenes, where
  WME is expected to underperform ORB-SLAM3. Claiming otherwise would be dishonest and a
  reviewer will find it immediately.
