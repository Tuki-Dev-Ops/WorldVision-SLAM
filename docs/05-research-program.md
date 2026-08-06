# Research Program: From Engine to Publishable Result

> Reviews the proposed six-step plan, corrects two things, and lays out the executable
> version with concrete milestones.

## 1. The proposed plan, reviewed

```
1. Implement classical SLAM from scratch
2. Analyse each module's bottlenecks
3. Design a new world-model structure and objective
4. Add new optimisation and memory structures
5. Benchmark against existing models
6. Propose as a new algorithm if improvement is confirmed
```

This is the correct *shape* of a systems-research program — build, measure, redesign,
compare, claim. Two corrections, one of them serious.

### Correction 1 (serious): evaluation must move from step 5 to step 1.5

**You cannot design toward a capability you cannot measure.** If the benchmark is not in
place before the design work, the design will drift toward whatever *is* measurable — which
today means ATE on TUM/EuRoC. ATE is saturated, and it does not measure a single thing WME
claims to improve:

| WME claims | Measured by ATE? |
|---|---|
| Graceful degradation under night / fog / blur | No |
| Calibrated uncertainty | No |
| Object identity persistence across occlusion and loops | No |
| Change detection ("what changed?") | No |
| Prediction quality | No |
| Recovery without map reset | No |

A system tuned to improve ATE will not improve any of these. Worse, at review time,
"we improve ATE by 4% on fr1_desk" is a rejection; the interesting claims will have no
numbers behind them because nobody built the measurement.

So the benchmark is not the last step. **It is the second step, and it is itself a
contribution** — there is no existing benchmark for world-model SLAM capabilities.

### Correction 2: do not reimplement the baselines

"Implement classical SLAM from scratch" reads two ways.

- **As a baseline for comparison: don't.** Use the official ORB-SLAM3 / DSO / DROID-SLAM
  releases with their published configurations. A self-implemented baseline that
  underperforms is the fastest way to lose a reviewer — the assumption will be that you
  crippled it, and that assumption is usually correct. Cost: months. Value: negative.
- **As understanding, and as ablations of your own system: yes, and it is essential.** The
  scientifically meaningful "classical comparison" is an ablation of WME itself:

  | Ablation | Isolates |
  |---|---|
  | fixed noise instead of `Σ_k(E)` | the heteroscedastic claim |
  | hand-scheduled `α_k(E)` weights | the calibration claim |
  | frozen Hungarian association | the deferred-association claim |
  | Tier 0 only (no TCG, no SPA) | direct-method baseline, internally consistent |
  | Tier 1 only | how far objects alone get you |
  | no dynamic masking | the token-masking claim |
  | BoW/NetVLAD instead of TCG | the constellation claim, head-to-head |

  These are far more convincing than an external baseline, because every confound is held
  fixed. External baselines establish *absolute* competitiveness; ablations establish
  *causal* attribution. A strong paper needs both, and the ablations carry the argument.

Step 2's "bottleneck analysis" then becomes: **instrument the official baselines** and
measure where they actually fail (tracking-loss events, association-error rate,
uncertainty miscalibration under conditions), rather than inferring bottlenecks from a
reimplementation.

## 2. The corrected program

```
0. Engine foundation                                       ← done (Phases 1–2b)
1. Evaluation framework + simulation benchmark             ← critical path, start now
2. Baseline instrumentation: where do they actually break?
3. Close the loop: factor graph, calibrated Σ_k(E)
4. New structures: hybrid association, memory, prediction
5. Full comparison: baselines + ablations, real + simulated
6. Claim only what survives
```

## 3. Why simulation is not optional

Real datasets cannot supply what WME needs to be evaluated on:

| Requirement | Real datasets | Simulation |
|---|---|---|
| Fog density sweep 0.0 → 1.0, same trajectory | impossible | trivial |
| Ground-truth object identity through occlusion | none | exact |
| A scene where exactly one object moved between visits | essentially none | scripted |
| Ground-truth "was this the same chair" | none | exact |
| Per-condition repeatability for calibration fitting | no | yes |
| Ground-truth future object trajectories | no | exact |

The controlled-condition sweep matters most: to *fit* `Σ_k(E)` you need many traversals of
the same trajectory under varied, known conditions. No dataset provides this. Simulation is
the only way to obtain the calibration data, and calibration is what separates the
principled formulation from a tuned heuristic (see [04-unified-objective.md](04-unified-objective.md) §4.2).

**Simulation is for calibration and controlled ablation. Real datasets are for external
validity.** Both, and neither substitutes for the other. Claims fitted in simulation must
be shown to transfer to TUM/KITTI/Robotcar, and that transfer test is itself a result.

### Simulator choice

| Option | Verdict |
|---|---|
| **Own lightweight synthetic generator** | Start here. Full control, deterministic, no dependencies, runs in CI. Sufficient for geometry/association/calibration studies. Implemented in `wme.sim`. |
| Habitat / AI2-THOR | Next. Photorealistic indoor, object annotations, scene-change scripting. Right for the semantic and change-detection claims. |
| CARLA | For outdoor/driving, weather is built in. Right for the adverse-condition claims at scale. |
| Isaac Sim | Only if physical interaction is needed. Heavy. |

The own-generator step is not throwaway: it is what lets calibration and association
studies run deterministically in CI, and it is the only harness where ground truth for
*every* quantity is available simultaneously.

## 4. Metrics that must exist before the design work

Beyond ATE/RPE, the following are required. All are implemented in `wme.eval.metrics`.

**Uncertainty calibration** — the most important, and the one most systems fail.
- NEES (Normalised Estimation Error Squared) with χ² consistency bounds
- ANEES over a sequence; a system reporting ANEES ≫ dof is overconfident
- Reliability diagram / expected calibration error for existence beliefs

**Degradation** — the headline claim.
- Performance vs. condition-severity curve, not a single number
- Catastrophic-failure rate (tracking lost and not recovered)
- Recovery latency after conditions improve

**Identity and association**
- Identity switches (IDS), mostly-tracked / mostly-lost (MOT-style, but in 3D world frame)
- Association error rate against ground-truth correspondence
- Post-loop-closure duplicate-object rate — WME's specific claim that constellation matches
  *merge* histories rather than duplicating them

**Change understanding**
- Change detection precision/recall, separated by change type: moved / removed / added
- Time-to-detect after a change
- False-change rate under pure sensor degradation (does fog look like the world changed?)

**Prediction**
- Displacement error at horizons (0.5 s, 1 s, 2 s)
- Predictive negative log-likelihood — measures whether predicted *uncertainty* is honest,
  not just the mean
- Surprise-detection latency for unannounced changes

**Systems**
- Bit-identical replay under varying thread counts
- p50/p95/p99 latency per module

## 5. Milestones

| # | Deliverable | Gate to pass before proceeding |
|---|---|---|
| M1 | `wme.sim` generator + `wme.eval.metrics` | metrics validated on synthetic cases with known answers |
| M2 | Baselines running on TUM/EuRoC + instrumented | reproduce published numbers within noise |
| M3 | Factor graph closing the loop, `Σ_k(E)` calibrated in sim | ANEES within χ² bounds — **if this fails, the formulation is wrong** |
| M4 | Hybrid association, memory, prediction | ablations show causal attribution |
| M5 | Full comparison: sim + TUM + EuRoC + Robotcar night/rain | degradation curves beat baselines where claimed; honest reporting where they don't |
| M6 | Paper(s) | only claims that survived M5 |

M3's gate is the real decision point. If calibrated uncertainty cannot be achieved, the
"one posterior" thesis is not supported and the program should pivot to the narrower TCG
contribution, which stands on its own.

## 6. Expected outcome, stated honestly in advance

Committing to this now, so the results are interpreted rather than rationalised:

- **Likely true:** WME degrades more gracefully than baselines under night/fog/blur; TCG
  relocalises where BoW fails; token masking beats direct methods on dynamic sequences;
  uncertainty is better calibrated than baselines (a low bar).
- **Likely false:** WME beats ORB-SLAM3 on ATE in clean, object-sparse sequences. It should
  not, and reporting it as a loss strengthens the paper.
- **Genuinely unknown:** whether `Σ_k(E)` calibrated in simulation transfers to real
  sensors; whether object-level MHT stays tractable in cluttered scenes.

If the "likely false" row turns out true, be suspicious of the experiment before being
pleased with it.
