# Is "One Objective Function" the Right Unification?

> This document answers a design question: should geometry, semantics, temporal memory,
> uncertainty, sensor reliability, dynamic objects, and prediction be fused into a single
> objective function?

## 1. Verdict

**The unification is correct at the level of the *probabilistic model*. It is wrong at the
level of the *optimizer*, and it is dangerous at the level of *prediction*.**

Three separate claims, and they need to be kept apart:

| Reading of "one objective function" | Verdict |
|---|---|
| One **joint posterior** over all state, factored, with calibrated likelihoods | ✅ Correct. This is the thesis. |
| One **scalar cost** minimised monolithically over all variables every frame | ❌ Intractable, ill-posed, and unnecessary. |
| One **weighted sum of heterogeneous losses** with hand-tuned weights | ❌ This is the trap. It is what most "unified SLAM" papers actually are, and why they don't reproduce. |
| Prediction as a **term in the same objective** as observation | ❌ Actively harmful. Explained in §5.3. |

The distinction between rows 1 and 3 is the whole ballgame, and it is sharper than it looks:

> **A sum of log-likelihoods with calibrated noise models has no free weights — the weights
> are all exactly 1 and the units cancel. A sum of losses with tuned weights is a different
> object entirely, and it is not a posterior.**

WME today is row 3. The `α_k(E)` weights are hand-designed. Getting to row 1 is the
central remaining research task, and it is what would make this publishable rather than
merely engineered.

## 2. What actually limits SLAM

Before designing a unification, it is worth being honest about which limitations are real
and which are folklore. Ordered by how much they actually cost in deployment.

### 2.1 Data association is decided greedily and then frozen — **the deepest structural flaw**

Every SLAM system in production makes hard, irreversible association decisions: this
keypoint matches that landmark; this detection is that object; this place is that place.
The continuous optimisation that follows is exquisite, and it is conditioned on a discrete
choice that was made by a threshold.

When association is wrong, the continuous estimator does not just degrade — it *confidently*
converges to a wrong answer, because the residuals of a wrong association are small by
construction (that is why it was selected). This is the mechanism behind essentially every
catastrophic SLAM failure.

The correct object is a **joint discrete-continuous posterior**. This is known and largely
unsolved in practice. It is the single largest improvement opportunity in the field.

### 2.2 Beliefs are unimodal when the truth is multimodal

Loop closure is genuinely ambiguous in repetitive environments. The honest posterior is
bimodal. Every mainstream system collapses it to one mode immediately, either by accepting
(and corrupting the map) or rejecting (and losing the constraint).

Switchable constraints, max-mixtures, and dynamic covariance scaling are patches on this.
None of them maintain an actual multimodal belief across time.

### 2.3 Uncertainty is uncalibrated

Reported covariances are systematically overconfident by an order of magnitude or more.
Causes: fixed linearisation points, marginalisation inconsistency, dropped landmark
covariance, robust kernels that silently change the effective noise model, and noise
parameters set by hand once and never validated.

An overconfident SLAM system cannot be safely consumed by a planner. This matters more for
deployment than ATE does, and it is barely measured in the literature.

### 2.4 Sensor noise is treated as homoscedastic when it is not

`σ = 0.01 m` in a config file, for all conditions, forever. Real sensor noise varies by
orders of magnitude with range, incidence angle, exposure, motion, and weather. Every
condition-robustness paper is, at bottom, an attempt to compensate for a noise model that
should have been condition-dependent in the first place.

### 2.5 The map has no ontology of change

A map cannot distinguish:
- "I mis-estimated this surface"
- "This object moved"
- "This object was removed"
- "This is a different object that looks the same"

All four produce the same thing — a residual — and all four are handled the same way:
overwrite or reject. A world model must separate them, because they license entirely
different actions.

### 2.6 Dynamic objects are discarded rather than modelled

The standard pipeline detects dynamics and *removes* them. This throws away exactly the
information a robot acting in a populated world needs most. Worse, "dynamic" is conflated
across three genuinely different properties: **movable** (a chair), **moving** (a chair
being carried), **moved** (a chair that is not where it was).

### 2.7 Memory is a keyframe list

There is no consolidation, no forgetting policy, no episodic/semantic distinction, no
mechanism to answer "was this here yesterday?" Long-term autonomy fails on this, not on
odometry accuracy.

### 2.8 The system is retrospective

SLAM answers where you *were*. It cannot support active perception, occlusion reasoning, or
"look there next", because it has no forward model.

### 2.9 Determinism and evaluation are weak

Most SLAM code is not bit-reproducible, so bugs are not reproducible. ATE on TUM/EuRoC is a
weak proxy for anything a robot cares about, and near-saturated.

### 2.10 Not on this list, deliberately

Front-end feature quality, loop-closure recall on benchmark sequences, and BA solver speed
are largely *solved enough*. Improving them further has low marginal value. They are where
most papers go because they are measurable.

## 3. Which of these does "one objective function" actually fix?

| Limitation | Fixed by a single joint model? |
|---|---|
| 2.1 Greedy association | **Only if the discrete variables are in the model.** A continuous-only objective does not touch this. |
| 2.2 Unimodality | **No.** Requires an explicitly multimodal representation, not a bigger objective. |
| 2.3 Uncalibrated uncertainty | **Yes, necessarily** — a joint posterior forces every term to be a calibrated likelihood. This is the strongest argument for the approach. |
| 2.4 Homoscedastic noise | **Yes** — via a latent environment variable modulating precision. |
| 2.5 Ontology of change | Representation, not objective. Orthogonal. |
| 2.6 Dynamic objects | **Yes** — moving objects become state to estimate rather than outliers to reject. |
| 2.7 Memory | Orthogonal. |
| 2.8 Prediction | **Yes, but only if kept structurally separate** (§5.3). |

So: the unification genuinely fixes 2.3, 2.4, 2.6, and partially 2.1 and 2.8. It does
nothing for 2.2, 2.5, 2.7. That is a real, worthwhile scope — but it is four to five of the
eight, not all of them. Claiming otherwise would be overselling, and a reviewer will notice.

## 4. Why the monolithic-scalar reading fails

### 4.1 The state is not a vector

```
poses            T_{0:t} ∈ SE(3)^t        continuous manifold
token positions  x_i ∈ R^3                continuous
token velocities v_i ∈ R^3                continuous
association      a_t : detections → tokens discrete, combinatorial
existence        e_i ∈ {0, 1}              Bernoulli
class            c_i ∈ {1..K}              categorical
static/dynamic   s_i ∈ {0, 1}              Bernoulli
scene relations  G                         discrete structure
environment      E                         latent, continuous
```

No single differentiable scalar covers this. Gradient descent on `a_t` is not defined. Any
system claiming "one objective" over this state is, in practice, alternating: fix discretes,
optimise continuous, re-decide discretes. That is coordinate descent on a non-convex hybrid
problem, and it inherits every weakness of §2.1.

### 4.2 The units problem, and its actual resolution

Summing intensity², pixel², metre², and nats requires weights. Hand-tuned weights are not
identifiable without ground truth, do not transfer across sensors, and are the reason such
systems do not reproduce.

**But this problem dissolves if — and only if — every term is a proper log-likelihood with a
calibrated noise model.** Then:

```
-log p(Z | X) = Σ_k  ½ r_k(X)ᵀ Λ_k r_k(X)  + const
```

Every `Λ_k` is a physical precision (inverse covariance in the residual's own units), so
each term is dimensionless and the weights are all 1. Nothing is tuned.

This is the fork in the road:

- Calibrate every noise model → the unification is principled, and the "one objective"
  claim is true and defensible.
- Leave weights hand-set → you have a heuristic with extra steps, and the unification claim
  is not supported.

WME already has the right *shape* for this in ECDA: the information matrix is divided by a
*measured* photometric variance rather than scaled by a tuned constant. That pattern has to
be extended to every factor.

### 4.3 Joint optimisation over all timescales is both intractable and unnecessary

Frame-rate photometric terms number in the millions per minute. But given the keyframe
poses, they are conditionally independent of the map. Hierarchical inference with
marginalisation is not a computational compromise — **it is exact inference exploiting the
conditional independence structure**, up to linearisation.

This is worth stating explicitly because it reframes the multi-rate architecture from
"engineering pragmatism" to "correct given the graph structure."

## 5. The correct formulation

### 5.1 One posterior, factored

```
p(X_c, X_d, E | Z_{0:t})  ∝  p(E | Z_env) · p(X_d) · ∏_k φ_k(X_c, X_d ; Λ_k(E))
```

- `X_c` continuous: poses, token positions/velocities, planes
- `X_d` discrete: association, existence, class, static/dynamic, graph edges
- `E` latent environment
- `Λ_k(E)` precision of factor `k`, **modulated by** `E`

The seven elements map onto this cleanly:

| Element | Where it lives |
|---|---|
| Geometry | `X_c` + photometric / structural factors |
| Semantics | `X_d` (class, relations) + constellation factors coupling `X_d` to `X_c` |
| Temporal memory | the graph itself is the memory; marginalised priors carry the past |
| Uncertainty | the posterior, not a side channel |
| Sensor reliability | `E` → `Λ_k(E)`, a **precision**, never a term |
| Dynamic objects | `s_i ∈ X_d` + motion factors on `v_i` |
| Prediction | separate generative branch, §5.3 |

### 5.2 Sensor reliability is a precision, not a term — this is the key move

The tempting formulation is to add a reliability term to the cost. That is wrong: it makes
"be reliable" something the optimiser can trade against fitting the data, which is
meaningless.

The correct formulation is **hierarchical / heteroscedastic**: `E` is a latent variable that
governs the noise of the observation models.

```
z_k | X, E  ~  N( h_k(X),  Σ_k(E) )
```

Degradation then reduces the information a modality contributes, automatically and
correctly, with no branching and no tuning. This is exactly WME's existing "no `if (night)`"
principle — but stated as a hierarchical Bayesian model rather than as a heuristic weight
schedule, which is what upgrades it from engineering to a contribution.

It also yields something the heuristic version cannot: **`E` becomes inferable**. Given
residual statistics across modalities, you can *estimate* the current environment state
rather than only measuring it from images, and the two estimates cross-validate. A lens that
is dirty in a way the image statistics miss will still show up as photometric residuals that
are inconsistent with constellation residuals.

### 5.2.1 Measured: a calibrated per-observation noise model is *not sufficient*

This section was written as theory. It has now been tested (`wme.calib`, M3 gate,
`python tools/m3_calibration.py`), and the experiment changed it.

Setup: object positions fused by an information filter, camera poses fixed to ground truth
so the measurement model is isolated. Noise fitted on training conditions, evaluated on
**unseen** conditions. Consistency measured by ANEES against its χ² band.

| noise model | ANEES/dof on held-out conditions | consistent |
|---|---|---|
| fixed (fitted on clear, applied everywhere) | 65 – 128 | 0 % |
| hand-scheduled inflation | 1.3 – 48.5 | 25 % |
| calibrated `Σ(z, E)`, per-observation only | 38 – 58 | 0 % |
| calibrated, **two-stage** (adds correlated term) | 1.8 – 2.4 | 0 % |

Three things came out of this.

**(a) The per-observation parameters are recoverable.** Fitted `c_d = 0.00603` against a
true `0.006`, `g_d = 3.39` against a true `3.0`. So the parametric-calibration half of the
argument holds: you *can* fit `Σ(z, E)` from data and recover the physics.

**(b) That alone barely helps.** The per-observation-only calibrated model is still 38–58×
overconfident — no better than the naive fixed model. Fitting the marginal noise of each
measurement correctly does almost nothing for the consistency of the *fused* estimate.

The reason is that observations are **not independent**. Any error that is a deterministic
function of the geometry — here, the fact that a detection box centre is not the projection
of an object's centroid — is shared across all observations of that object. Fusion shrinks
the reported covariance as `1/N` while that component does not shrink at all. Overconfidence
therefore grows *in proportion to the number of observations*: the more evidence the system
gathers, the more wrong its uncertainty becomes.

This is, in all likelihood, the actual mechanism behind the order-of-magnitude
overconfidence that SLAM systems are known for. It is reproduced minimally in
`test_unmodelled_bias_makes_fusion_overconfident`.

**(c) The fix requires a second, correlated term, fitted at the object level.**

```
Σ_total(object) = ( Σ_i Σ_i(z_i, E)⁻¹ )⁻¹  +  Σ_sys(E, geometry)
                   └── shrinks as 1/N ──┘      └─ does not shrink ─┘
```

`Σ_sys` is invisible in per-observation residuals — there it is indistinguishable from
independent noise. It only appears when fused residuals fail to shrink as `1/√N`, so it must
be fitted in a second stage on object-level residuals. Doing so drops ANEES/dof from ~58 to
~2: **98 % of the overconfidence removed.**

The fit also identified `Σ_sys`'s form. Given three candidate terms (a constant floor, a
size-dependent geometric term `k·w²z/f²`, and a condition-dependent factor), maximum
likelihood drove the floor and the condition factor to zero and kept only the geometric
term. That is the correct answer — the modelling error is geometric in origin and does not
depend on weather.

**(d) Budgeting for the error was not enough; the model had to be fixed.** A residual factor
of ~2 remained, and debiasing recovered only 15–20 % of it — so it was covariance *shape*,
not an offset. An isotropic `Σ_sys` cannot represent an error whose direction depends on
viewing geometry.

The correct fix removes the error instead of budgeting for it. A detection box centre is not
the projection of a centroid; the offset is `≈ e_x·e_z / C_z`, and the depth extent `e_z` is
**unobservable from a single view**. So extent must become part of the state, estimated
jointly with position across views, and then **marginalised** — Schur complement, not
conditioning — so that not knowing the object's size correctly inflates the position
covariance rather than being silently assumed away.

With that change, and after refitting the noise under the corrected measurement model:

| | ANEES/dof | consistent | RMSE |
|---|---|---|---|
| box-centre + fitted `Σ_sys` | 1.9 – 2.8 | 0 % | 0.068 m |
| **joint extent + refit noise** | **0.85 – 1.27** | **100 %** | **0.037 m** |

Both `Σ_sys` and the geometric term fitted to **zero** — the systematic error was gone, not
absorbed. And the noise parameters came back to their true values: `c_px = 1.996` against a
true `2.00`, `g_px = 3.82` against `4.00`.

**(e) Two things the experiment taught that were not in the original design.**

*The noise model is only valid for the measurement model it was fitted under.* Parameters
fitted with the biased box-centre model had absorbed the bias (`c_px = 8.55` vs. a true
`2.00`) and became over-estimates once the model was corrected. Noise and measurement models
must be calibrated together, EM-style.

*Consistency alone is not a criterion, and an early version of this gate was gamed by our own
code.* One configuration reported 100 % NEES consistency with an RMSE of 4.2 m — it had
inflated `σ_sys` to 2.5 m to cover a diverging estimator. Any sufficiently uncertain
estimator is "consistent". A calibration gate must require consistency **and** accuracy, or
it selects for uselessness. This is worth stating loudly because the failure was silent: the
numbers looked like success.

**Revised claim for §5.2.** Sensor reliability as a latent precision is necessary but not
sufficient. The complete statement is:

> Every likelihood must be calibrated, **the correlation structure of its errors must be
> modelled**, and **the noise model must be fitted under the measurement model it will be used
> with**. A perfectly calibrated marginal noise model, fused under an independence assumption,
> is arbitrarily overconfident — and correlated model error should be eliminated by fixing the
> model, not covered by widening the covariance.

One further observation worth keeping: the hand-scheduled model was "consistent" at 25 % of
conditions. Inspecting the table shows why — it is wildly overconfident at low severity,
passes through correctness at one point, and would be underconfident beyond. A scheduled
weight is a curve crossing the truth, not tracking it. That is precisely the failure mode
that makes tuned weights non-transferable, and it is visible here in four rows.

### 5.3 Prediction must not be a term in the same objective

This is the one place where "unify everything" is actively harmful.

If a term `‖x_t − f(x_{t−1})‖²_Λ` sits in the same objective as the observation terms, then
the optimiser can reduce total cost by **making the present agree with the forecast**. With
a strong enough motion model, the system will smooth away genuine surprise — and genuine
surprise is precisely the signal that something changed. A world model that suppresses
surprise cannot answer "what changed?", which is one of its five defining questions.

The correct structure:

- Prediction supplies a **prior** for the next timestep: `p(X_{t+1} | X_t)`. Legitimate —
  this is the standard Markov prior and it *is* a factor.
- Prediction is also run **forward beyond the estimation horizon**, producing forecasts that
  are **scored** against later observations. The residual is *diagnostic output*, not
  something to minimise over the current state.
- The two must be different objects in code. WME's `WorldToken` already separates
  `prediction` from `position` for exactly this reason; the objective must respect the same
  boundary.

The distinction in one line: **a motion prior constrains the next state; a forecast is a
falsifiable claim about the future. Only the first belongs in the objective.**

### 5.4 The discrete variables are the hard part

This is where a real contribution is available, and where the current WME design is weakest:
`TokenStore` runs Hungarian assignment and freezes the result — textbook §2.1.

Three viable levels, in increasing ambition:

1. **Deferred association.** Keep the top-`k` assignment hypotheses per detection alive for
   `N` frames; commit only when the marginal is decisive. Cheap, and removes most single-frame
   association errors.
2. **Max-mixture factors.** Represent an ambiguous association as a mixture; the optimiser
   selects the mode. Keeps the graph Gaussian and fast. Well-suited to loop closure, where
   WME currently just rejects ambiguity outright.
3. **Explicit hybrid inference.** Maintain a posterior over association histories
   (MHT-style) with pruning. Expensive but principled, and object-level associations are far
   fewer than keypoint-level ones — **which is exactly why the object-centric representation
   makes this tractable where classical SLAM could not.**

Level 3 is the genuinely novel claim, and it is enabled by the YOLO-only constraint rather
than despite it: there are ~20 objects per scene, not ~2000 keypoints.

## 6. What changes in WME

| Current | Should become |
|---|---|
| `α_k(E)` hand-tuned scalar weights | calibrated `Σ_k(E)` fitted on adverse-condition data with held-out validation |
| Hungarian association, frozen | deferred / mixture association, committed on evidence |
| `query()` rejects ambiguous places | ambiguous places enter the graph as a mixture factor |
| ECDA information scaled by measured photometric variance | ✅ already correct — extend this pattern everywhere |
| `existence/identity/static` beliefs updated outside the graph | same beliefs, but as discrete variables coupled to the continuous estimate |
| Prediction stored separately from observation | ✅ already correct — preserve it in the objective |

Concretely, the immediate next implementation step is the factor graph, with:
- factors carrying real precisions, not weights
- robust kernels declared as explicit heavy-tailed likelihoods (so the effective noise model
  stays interpretable) rather than as opaque loss reshaping
- a mixture factor type from day one, so multimodality is representable rather than retrofitted

## 7. What to publish, and what not to claim

A paper that claims to simultaneously improve seven axes will be rejected. Reviewers read
that as unfocused. The system is the platform; the contributions are narrower.

Three defensible units, in order of strength:

1. **Descriptor-free object-constellation relocalization** (TCG). Sharp, novel, testable
   against BoW/NetVLAD under night/fog/viewpoint change. Strongest standalone claim.
   → CVPR / ICRA.
2. **Environment as a latent precision variable.** Show that a hierarchical heteroscedastic
   noise model outperforms both fixed noise and hand-scheduled weights, and that `E` can be
   *inferred* from cross-modal residual disagreement. → ICRA / RSS.
3. **Object-level hybrid discrete-continuous inference.** Argue that object-centric
   representation makes MHT-style association tractable, and show it fixes failures that
   frozen association cannot. → RSS / T-RO. Highest risk, highest value.

The unified system paper (T-RO / IJRR) comes after, and cites the three.

**Claims to avoid:** "we improve accuracy on all axes", "we unify everything", any ATE
improvement on saturated benchmarks presented as the headline. The honest headline is
*graceful degradation and calibrated uncertainty under conditions where baselines fail
catastrophically* — that is both true and more interesting.

## 8. How this gets falsified

| Claim | Test | Fails if |
|---|---|---|
| Calibrated likelihoods remove tuned weights | fit `Σ_k(E)` on train conditions, evaluate on held-out unseen conditions | performance requires per-condition retuning |
| `E` as precision beats weight scheduling | ablate: fixed noise / scheduled weights / learned `Σ_k(E)` | scheduling matches learned model |
| Uncertainty is calibrated | NEES / NIS consistency tests, reliability diagrams | χ² consistency fails (the usual outcome — most systems fail this badly) |
| Deferred association helps | inject association-ambiguous scenes; measure identity-switch rate and ATE | no measurable reduction vs. frozen Hungarian |
| Object-level MHT is tractable | measure hypothesis count and latency vs. scene object count | hypothesis growth is not controllable by pruning |
| Prediction separation preserves surprise | inject an unannounced scene change; measure detection latency | a joint objective detects it as fast (would falsify §5.3) |

Note that the NEES/NIS row is the one most likely to fail, and it is the one most worth
running first — it is cheap, and if WME's uncertainty is not calibrated, the entire
"one posterior" argument is undermined at its foundation.

## 9. Summary

The instinct is right: these seven elements are genuinely coupled, and estimating them
separately loses information. Unify them.

Unify them as **one factored posterior with calibrated likelihoods and a latent precision
variable**, inferred hierarchically across timescales, with discrete association variables
in the model rather than decided beforehand, and with forecasting kept structurally outside
the objective.

Do not unify them as one hand-weighted scalar cost minimised monolithically. That version is
what the field has already tried, and it is why "unified SLAM" has a reputation problem.

The measurable difference between the two is whether anything in your config file is a
weight rather than a physical noise parameter.
