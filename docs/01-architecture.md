# WME System Architecture

## 1. The inversion

Classical SLAM is a **pipeline that produces a map**. Each stage hands a product to the
next; the map is the terminal artefact; semantics, if present, are painted on afterwards.

WME is a **set of estimators that write into a single shared belief**. There is no terminal
artefact. There is one object — the `WorldState` — and every engine is a process that
proposes revisions to it. This is the architectural expression of the core philosophy: the
World State is the only source of truth.

```
        CLASSICAL                              WME
   ┌───┐ ┌───┐ ┌───┐ ┌───┐              ┌─────────────────────┐
   │ A ├►│ B ├►│ C ├►│MAP│              │     WorldState      │
   └───┘ └───┘ └───┘ └───┘              │  (versioned belief) │
                                        └──▲──▲──▲──▲──▲──▲───┘
   map is the output                       │  │  │  │  │  │
                                          E1 E2 E3 E4 E5 E6
                                     every engine reads and revises
```

Consequences that fall out of this choice:

- **No engine owns data another engine needs.** They all read the same state, so there is no
  ordering constraint beyond data dependency, so the schedule can be a DAG rather than a
  chain, so it parallelises.
- **Failure is local.** If the Localization Engine loses tracking, it stops proposing pose
  revisions. The Memory, Prediction, and Semantic engines keep running on the last good
  belief. This is what "never restart the map, repair it" means mechanically.
- **Time is first class.** The state is versioned, so "what did the world look like 4
  seconds ago" is a query, not a replay. The Unity visualiser's rewind and the Memory
  Engine's history are the same mechanism.

## 2. Layer map

```
┌───────────────────────────────────────────────────────────────────────┐
│ L5  APPLICATION      Planner · Unity Visualizer · REST/gRPC surface   │
├───────────────────────────────────────────────────────────────────────┤
│ L4  BELIEF           WorldState · WorldGraph · Memory · Prediction    │
├───────────────────────────────────────────────────────────────────────┤
│ L3  ESTIMATION       Localization · Geometry · Tracking · Semantic    │
│                      Confidence                                       │
├───────────────────────────────────────────────────────────────────────┤
│ L2  REPRESENTATION   WorldToken · TokenStore · Constellation index    │
├───────────────────────────────────────────────────────────────────────┤
│ L1  PERCEPTION       YOLO runtime · Environment Analyzer · ImageQuality│
├───────────────────────────────────────────────────────────────────────┤
│ L0  CORE             SE3 · TaskGraph · ThreadPool · Frame · Logging   │
└───────────────────────────────────────────────────────────────────────┘
```

Dependencies point strictly downward. `wme_core` knows nothing about YOLO; `wme_perception`
knows nothing about pose; the Planner knows nothing about pixels. Enforced by CMake target
link boundaries, not by convention.

## 3. Execution model

WME is not frame-synchronous. Different engines run at rates matched to the physics they
model, which is the only way to hit 60 FPS while doing global optimisation.

| Engine | Rate | Thread | Rationale |
|---|---|---|---|
| Environment Analyzer | 5 Hz | perception | weather does not change in 16 ms |
| Image Quality | every frame | perception | per-frame weighting needs it |
| YOLO | 15–30 Hz | GPU stream | inference bound; tracking interpolates |
| ECDA (Tier 0) | every frame | tracking | pose must be frame-rate |
| Tracking | every frame | tracking | identity continuity |
| Geometry / TSDF | 10 Hz | mapping | integration is bandwidth bound |
| TCG retrieval | 2 Hz | mapping | loop closure is not urgent |
| Pose graph / BA | on keyframe | optimisation | expensive, asynchronous |
| Prediction | 10 Hz | belief | forecast horizon is ~1 s |
| World Graph | 5 Hz | belief | relations change slowly |

Coordination is by a **task graph with explicit data dependencies**, not by locks scattered
through the code. A node declares what it reads and what it writes; the scheduler derives
the parallelism. Engines never block on each other; they consume the most recent published
version of what they need.

### 3.1 Concurrency discipline

- `WorldState` uses **copy-on-write snapshots with atomic publish**. Readers take a shared
  pointer to an immutable snapshot and are never blocked by writers. Writers build a delta
  and publish. This is the only lock-free-read structure in the system and it is the reason
  the visualiser can run at display rate without stalling estimation.
- Mutable engine-internal state is single-threaded by construction — each engine is owned by
  exactly one task-graph node.
- Anything crossing a thread boundary is either an immutable snapshot or goes through an
  SPSC ring buffer. No shared mutable containers. No recursive mutexes anywhere.

## 4. Module inventory

```
wme_core          SE3/SO3, Frame, Timestamp, ThreadPool, TaskGraph, RingBuffer,
                  Result<T>, config, logging
wme_perception    EnvironmentAnalyzer, ImageQualityEngine, YoloRuntime (TensorRT/ORT)
wme_token         WorldToken, TokenStore, TokenLifecycle, ConstellationIndex
wme_geometry      DepthEstimator, PointCloud, VoxelGrid, TSDF, PlaneExtractor, MeshGen
wme_localization  ECDA, TCG, SPA, FactorGraph, PoseGraph, BundleAdjustment, DriftRecovery
wme_tracking      TokenTracker, MotionModel, ReID (geometric, non-descriptor), Occlusion
wme_semantic      SceneGraphBuilder, RelationInference, RoomSegmentation
wme_memory        HistoryStore, TemporalIndex, Consolidation, Forgetting
wme_prediction    MotionForecast, VisibilityForecast, OccupancyForecast
wme_confidence    BayesianUpdater, ReliabilityModel, ConfidencePropagation
wme_world         WorldState, WorldStateSnapshot, WorldGraph, DeltaLog
wme_planner       SemanticNavigator, ObjectSearch, RiskEstimator
wme_viz           ImGui debug UI, Unity bridge (shared memory + protocol)
```

## 5. The Unity bridge

Unity is a **client**, not a component. It never participates in estimation. The bridge is a
versioned binary protocol over shared memory (local) or TCP (remote), publishing
`WorldStateSnapshot` deltas. Because snapshots are immutable and versioned, rewind, pause,
and frame-stepping in Unity are implemented purely client-side against a retained ring of
snapshots — the engine does not need to know the visualiser exists.

The protocol specification is not yet written — see [03-roadmap.md](03-roadmap.md) Phase 5.

## 6. Error handling policy

- No exceptions across engine boundaries. Engines return `Result<T>` — an expected-style
  type carrying a typed error and a severity.
- A degraded engine reports `Degraded` with a reason; it does not throw and it does not
  silently return garbage. The Confidence Engine consumes those reports directly — a
  degraded producer is exactly a low-reliability sensor.
- Exceptions are reserved for programmer error (contract violation), and terminate.

## 7. What "production quality" means here, concretely

1. Every public header has a documented contract: preconditions, thread-safety class,
   allocation behaviour, and complexity.
2. No allocation in per-frame hot paths — arenas and preallocated pools, verified by a
   benchmark that asserts zero `operator new` calls in the tracking loop.
3. Every numeric routine has a test against an analytically known answer, plus a property
   test (e.g. `exp(log(T)) == T` on random SE(3)).
4. Every engine has a benchmark reporting p50/p95/p99 latency, because p99 is what breaks a
   real-time system and mean latency hides it.
5. Deterministic replay: given a recorded input log and a seed, the engine produces
   bit-identical output. Without this, no SLAM bug is ever reproducible.
