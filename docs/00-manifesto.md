# WME Manifesto

## The question SLAM asks is too small

Simultaneous Localization and Mapping asks *"where am I, and what does the geometry around
me look like?"* It answers with a pose and a point cloud. For thirty years that was the
right question, because the robot's job was to not hit things.

A robot that acts in the world needs answers to five questions, and only the first is a
SLAM question:

1. **Where am I?** — pose
2. **What exists?** — an inventory of entities, not surfaces
3. **What changed?** — a diff against what was believed before
4. **What will change?** — a forecast, with its own uncertainty
5. **What is reliable?** — calibrated confidence in every one of the above

A point cloud cannot express (2), because a point is not a thing. It cannot express (3),
because when the world changes the cloud is simply *wrong* and gets overwritten. It cannot
express (4) at all. And (5) is usually a single scalar "tracking good / tracking lost."

WME's position is that these are not features to add on top of SLAM. They require a
different object at the centre of the system.

## The World State

WME has exactly one central object: a **continuously evolving World State**. It is not a
map that gets built and then read. It is a belief that is continuously revised.

Concretely, the difference:

| | Map | World State |
|---|---|---|
| Contents | geometry | entities, geometry, relations, forecasts, confidence |
| Update on contradiction | overwrite | revise belief, keep both hypotheses' evidence |
| An object disappearing | points get carved out | token transitions to `Displaced`, history retained |
| Query "what was here 10 s ago" | impossible | versioned snapshot |
| Failure of a sensor | tracking lost | one information source drops out; belief widens |
| Semantics | a separate layer | the same layer |

Everything in this engine exists to update that belief. Localization is not the goal; it is
one estimator among several, and when it fails the World State does not.

## Six commitments

**1. Objects are the primitive, not points.**
The `WorldToken` is the unit of the world. It is a landmark for pose estimation, a node in
the semantic graph, an entity in memory, a target for prediction, and an affordance for the
planner — one data structure serving all five roles. This is only possible because YOLO
detections are promoted to first-class entities immediately, and it is the reason the
YOLO-only constraint produces a *better* architecture rather than a compromised one.

**2. Prediction and observation are never mixed.**
A predicted position is stored in a different field from an observed one, always. The
moment a system writes a forecast into the same slot as a measurement, it has lost the
ability to tell you what it actually saw, and every downstream confidence estimate becomes
a lie. WME keeps them physically separate in `WorldToken`.

**3. Nothing disappears immediately.**
An object leaving the field of view is not evidence of absence. Tokens move through a
lifecycle — `Active → Occluded → Dormant → Displaced → Retired` — and only `Displaced`
means "we verified it is gone." How long each stage lasts is a function of environment: in
fog you wait much longer before concluding anything.

**4. Every quantity carries its uncertainty.**
Not a confidence score — a covariance or a posterior. `existence_belief`,
`identity_belief`, `static_belief` are separate because they fail separately: you can be
certain a chair exists, uncertain whether it is *that* chair, and uncertain whether it is
bolted down.

**5. Degradation is information loss, not mode switching.**
There is no `if (night) useNightMode()`. There is an `EnvironmentState` that reduces the
information weight each estimator contributes. The estimator then handles the rest
correctly by construction. This is a single equation instead of a combinatorial explosion
of hand-written condition branches, and it is why the engine handles fog-at-night-with-a-
dirty-lens without anyone having written that case.

**6. Repair, never restart.**
Losing tracking is not an exception. It is a period during which the pose factor
contributes nothing while every other belief keeps updating. When conditions improve, the
constellation index relocalises, the pose graph absorbs the correction, and object
histories are *merged* rather than duplicated. The map is never reset, because the world
did not reset.

## What would falsify this

A research claim that cannot fail is not a claim. WME is wrong if:

- Object-anchored relocalization does not beat appearance-based retrieval under night, fog,
  and large viewpoint change on real sequences.
- The unified token representation costs more in accuracy (in object-sparse scenes) than it
  gains in robustness and capability, across a representative benchmark suite.
- Environment-conditioned information weighting does not measurably outperform fixed
  weighting under changing conditions.

Each of these is measurable, and the evaluation plan in
[`02-correspondence-problem.md`](02-correspondence-problem.md#6-evaluation-plan) is designed
to test them rather than to confirm them.
