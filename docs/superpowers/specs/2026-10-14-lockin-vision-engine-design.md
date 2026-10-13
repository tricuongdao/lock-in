# LOCK-IN — Vision Engine Design

**Date:** 2026-10-14
**Status:** Design approved in conversation; awaiting written-spec review
**Scope:** Slice A only — the vision engine. No UI, no storage, no daemon.

---

## 1. Context

### 1.1 What LOCK-IN is

LOCK-IN is an open-source Windows desktop app for laptop users that watches the
webcam locally, detects when the user stops studying, and nudges them back with a
sound or animation. It tracks a daily streak and shows a dashboard of time
locked-in, time on the phone, time distracted, and a distraction counter. All
inference runs on the user's machine; no frame ever leaves it.

### 1.2 Why this slice first

The full product decomposes into five subsystems:

| # | Subsystem | Depends on |
|---|---|---|
| **A** | **Vision engine** | — |
| B | Session & stats core | A's event contract |
| C | App shell & UI | A, B |
| D | Enforcement (Windows Service, Ultra Lock-in) | B, C |
| E | Distribution (installer, signing, updates) | all |

A is the long pole and the only genuinely uncertain part. Everything else is
well-trodden engineering. A is also the only slice that can be fully validated
without a camera, a UI, a daemon, or an installer — by replaying recorded video
through it and scoring accuracy and CPU cost against explicit gates.

### 1.3 Decisions already made

| ID | Decision | Rationale |
|---|---|---|
| D1 | **Windows 11 only** | macOS support was explicitly dropped. This removes the LaunchDaemon, notarization, Apple Developer Program, and WKWebView divergence from the project. |
| D2 | **Open source, for strangers** | Real enforcement is required, so signing is on the critical path. |
| D3 | **Rust + Tauri 2 + `ort`** | Leanest binaries and no runtime dependency; best fit for the lightweight requirement. Tauri is not used in this slice. |
| D4 | **Cascaded tiered pipeline** | The only architecture that satisfies lightweight + fast + high precision together. A uniform single-pass design costs 3–5× the CPU. |
| D5 | **Rich attention profile (Tier 3)** | Face presence, head pose, gaze, eyes-closed, phone-in-use, other-people-present. |
| D6 | **High precision — silence over noise** | Applied to *both* measurement and interruption. A beep while the user legitimately blinks is what makes people uninstall a focus app. The accepted cost is that the dashboard under-reports. |
| D7 | **Vision engine first** | Highest risk, fully testable in isolation. |

---

## 2. Goals and non-goals

### Goals

- **G1** Emit a trustworthy `FocusState` on every tick and discrete `FocusEvent`s
  when a state has been held long enough to count.
- **G2** Meet explicit accuracy gates measured on a labelled corpus — the primary
  gate is false nudges per hour of focused work.
- **G3** Stay inside a stated power budget on the reference laptop, measured rather
  than asserted.
- **G4** Be testable with no camera and no display, by replaying recorded video.
- **G5** Have no dependency on UI, storage, or any Windows-specific API in the
  core crate.

### Non-goals (explicitly out of scope for this slice)

- Dashboard, charts, tray icon, onboarding UI
- SQLite persistence, streaks, daily rollups
- The Windows Service, Ultra Lock-in enforcement, tamper resistance
- Nudges: sound, animation, notification
- Installer, code signing, auto-update
- macOS or Linux
- Adaptive per-user learning. Calibration alone must carry v1; learning is a v2
  candidate and is deliberately excluded to keep this slice shippable.

---

## 3. Constraints

- **C1 — Camera access is user-session only.** A Windows Service runs in Session 0
  and cannot open the camera. Capture and inference therefore live in a
  user-session process. This constrains later slices, not this one, but the crate
  boundary is drawn so the engine can be hosted by either a tray app or a
  background user-session process without change.
- **C2 — Permissive licensing only.** The project ships to strangers, so every
  model and dependency must be MIT or Apache-2.0. See §13 for the one documented
  exception.
- **C3 — CPU is the baseline.** GPU acceleration via DirectML is an optimisation,
  never a requirement; the engine must hit its budget on CPU alone. **The Intel NPU
  is not reachable from this stack** (§12.1), so it is not a design target.
- **C6 — DirectML caps at ONNX opset 20.** Every model must be exported at opset 20
  or lower, or it silently loses iGPU acceleration.
- **C4 — Reference hardware.** Intel Core Ultra 5 125H, Windows 11. Budgets are
  measured against this machine and stated as budgets, not universal truths.
- **C5 — No frame touches disk or network.** Inference is in-memory only. Recorded
  clips exist solely for the test corpus, are produced deliberately by the
  maintainer, and are never written by the running engine.

---

## 4. Architecture

### 4.1 Component boundary

```
lockin-vision  (Rust library — no camera side effects, no DB, no UI, no service)
    capture/   FrameSource trait → CameraSource (nokhwa) | ReplaySource (video file)
    models/    ONNX session wrappers, one per model
    analyze/   NMS, solvePnP, EAR, gaze, phone↔person association
    fuse       → Observation
    state      → FocusState + FocusEvent
    schedule   adaptive rate control

lockin-cli     two binaries: `replay` (accuracy harness) and `probe` (live debug)
```

The engine consumes frames plus a clock and emits ticks. It knows nothing about
SQLite, Tauri, nudges, or Windows. This is what makes it testable without a camera
and reusable unchanged by every later slice.

The `FrameSource` trait is the seam that makes G4 achievable: `ReplaySource` feeds
a video file at a controlled rate, so the entire pipeline, including the state
machine and its dwell timers, runs identically in tests and in production.

### 4.2 The cascaded pipeline

| Stage | Cadence | Model | Produces |
|---|---|---|---|
| 0 · Motion gate | every tick, ~free | — | 64×48 grayscale frame difference. If the scene is static **and** a face was present on the previous tick, skip stages 1–3 entirely and re-emit the previous observation. |
| 1 · Face presence | 5 Hz | YuNet (227 KB, MIT) | 0..N face boxes with 5 landmarks each → face count, "is anyone at the desk" |
| 2 · Dense landmarks | 5 Hz, only when stage 1 found a face | Face Mesh, 478 points incl. iris (2.3–4.7 MB) | head pose via `solvePnP`, EAR, gaze vector |
| 3 · Person + phone | 1–2 Hz, independent cadence | NanoDet-Plus (2.3 MB, Apache-2.0) | person boxes (class 0), cell-phone boxes (class 67) |

Design intent behind the cadences:

- **Stage 3 runs on a slower clock than stages 1–2.** This is the single largest
  contributor to meeting the power budget. The 0.5–1 s latency it introduces is
  absorbed by the dwell windows in §7, which is precisely why a high-precision
  stance and a slow detector are compatible.
- **Stage 0 is what makes sitting still nearly free.** No scene change means no
  inference at all. A user reading a book in front of the screen costs almost
  nothing.
- Stages 1 and 2 are a **cropper dependency**: Face Mesh expects a cropped face, so
  if stage 1 finds no face, stage 2 cannot run. This is a real single point of
  failure and is tracked as risk R2.

### 4.3 Frame sources

`CameraSource` wraps `nokhwa` for live capture at 30 fps; frames are downsampled
before inference, and the capture rate is deliberately decoupled from the tick rate
so the camera thread never blocks the pipeline.

`ReplaySource` reads a video file and presents frames on a virtual clock, with an
option to run as fast as possible for batch evaluation or at real time for
interactive inspection.

---

## 5. Signal extraction

All derived quantities are computed in Rust in `analyze/`. Nothing is delegated to
a Python runtime.

| Signal | Method | Consumed by |
|---|---|---|
| Face count | Stage 1 box count | `OtherPresent`, `AwayFromDesk` |
| Head pose (yaw, pitch, roll) | `solvePnP` over the six canonical Face Mesh indices `33, 263, 1, 61, 291, 199` | `Distracted`, `GlancingAway` |
| Eye aspect ratio (EAR) | Six points per eye, standard EAR formula | `Drowsy` |
| Gaze vector | Normalised iris-centroid offset within each eye (iris points 468–477) | `GlancingAway`, `Distracted` |
| Phone posture | Association of class-67 boxes with class-0 boxes, region-classified against the head box from stage 2 | `PhoneInUse` |

### 5.1 Phone posture classification

This is the weakest signal in the pipeline and the design treats it accordingly.
For each person box, phone boxes whose centre falls inside it are associated, then
classified:

- centre falls within the head region derived from stage 2 landmarks → `AtHead`
  (handset to the ear)
- centre falls inside the person box below the head region → `InHand`
- otherwise → `Present` (a phone on the desk, which is **not** phone use)

`Present` deliberately does not enter `PhoneInUse`. This distinction is the single
most valuable idea taken from the prior art (§13) and it is what separates "a phone
exists in the frame" from "the user is using their phone".

---

## 6. Fusion — the Observation type

Each tick fuses stage outputs into one record:

```rust
struct Observation {
    tick_ms: u64,
    faces: u8,
    primary: Option<FaceSignals>,   // landmarks-derived, primary face only
    phone: PhoneSignal,             // None | Present | InHand | AtHead
    motion: f32,
    confidence: Confidence,         // per-signal confidences
}

struct FaceSignals {
    head: Euler,        // yaw, pitch, roll in degrees
    ear: f32,
    gaze: (f32, f32),   // normalised iris offset
}
```

`Observation` exists so the `probe` binary can render an overlay and so corpus
label review can compare raw signals against ground truth. It is diagnostic
output, not the engine's contract — `FocusState` and `FocusEvent` are.

---

## 7. The state machine

This is where D6 (high precision) is actually implemented. All four reference
repositories classify per frame, which is exactly why they are noisy.

### 7.1 States

```rust
enum FocusState {
    Starting,       // before any stable state exists: warm-up or calibration
    AwayFromDesk,   // no face for longer than the away threshold
    Focused,        // face present, pose and gaze in bounds, eyes open, no phone
    GlancingAway,   // out of bounds, but not yet long enough to count
    Distracted,     // out of bounds, held past the distraction threshold
    PhoneInUse,     // phone associated with the person and classified as in use
    Drowsy,         // EAR below the personal threshold
    OtherPresent,   // more than one face
}
```

### 7.2 Transition rules

- **Dwell, not thresholds.** `Focused → GlancingAway` is immediate, because it is
  cheap bookkeeping. `GlancingAway → Distracted` requires the deviation to be held
  for a threshold of roughly 2.0 s. A glance never becomes a distraction, and a
  glance never emits an event.
- **Asymmetric hysteresis.** Entry into a state requires sustained evidence; exit
  requires much less. This eliminates state flapping, which is the mechanism that
  produces burst-beeping in the prior art.
- **Latched minimum duration.** Once entered, `PhoneInUse` cannot drop out for at
  least ~3 s, so a hand occluding the phone does not generate start/stop churn.
- **All thresholds are profile-relative**, never absolute. The ±10° constants in
  the prior art do not transfer between people, lighting, or eyewear and are
  actively harmful without calibration (§8).

**Complete priority order on disagreement**, highest first. Every state appears, so
no combination of signals is unhandled:

> `PhoneInUse` › `Drowsy` › `Distracted` › `OtherPresent` › `GlancingAway` ›
> `AwayFromDesk` › `Focused` › `Starting`

A head-down phone user is in `PhoneInUse`, not `Distracted`. A second person
entering the frame moves the session to `OtherPresent` rather than being discarded
as noise, because another human in the room is a plausible distraction source that
later slices may want to surface.

**Starting values for every threshold.** These are what the implementation begins
with and what the corpus tunes. They are not settled truths — but stating them
numerically is what makes them calibrated constants rather than magic numbers.

| Transition | Dwell | Kind |
|---|---|---|
| Head deviation → `GlancingAway` | immediate | entry |
| `GlancingAway` → `Distracted` | 2.0 s | entry |
| `Distracted` → out | 0.6 s | exit |
| `PhoneInUse` entry | 2.0 s | entry |
| `PhoneInUse` latched minimum | 3.0 s | latch |
| `PhoneInUse` → out | 0.8 s | exit |
| `Drowsy` entry | 1.0 s | entry |
| `Drowsy` → out | 0.5 s | exit |
| No face → `AwayFromDesk` | 3.0 s | entry |
| Face returns → out of `AwayFromDesk` | 0.5 s | exit |
| Second face → `OtherPresent` | 5.0 s | entry |
| `OtherPresent` → out | 3.0 s | exit |
| Confidence floor for recording anything | 0.6 | — |

Entry dwells are deliberately far longer than exit dwells. That asymmetry is the
mechanism that eliminates flapping, and it is why the prior art's per-frame
classification cannot be fixed by tuning constants alone.

**Primary-face rule.** When several faces are present, the primary face is the
largest box, with ties broken by proximity to frame centre. Signals in §5 are
derived from the primary face only. `AwayFromDesk` requires *zero* faces; if the
primary face is lost but another remains, the state is `OtherPresent`, not
`AwayFromDesk`.

### 7.3 Low-confidence handling

Under D6, a tick whose confidence falls below the floor is **not recorded as a new
state**. Its time is carried forward to the last stable state.

This keeps every accounting bucket summing to wall-clock time — there is no
statistical hole — while never manufacturing a distraction out of weak evidence.
The honest cost, accepted with D6: the dashboard under-reports, and calibration is
the primary lever against that.

### 7.4 Events

**Measurement and interruption are separate outputs.** Every tick carries a
`FocusState` for time accounting. Only discrete `FocusEvent`s can ever trigger a
nudge. Later slices choose what to do with them; this slice guarantees they are
trustworthy.

```rust
enum FocusEvent {
    FocusGained   { at: Instant },
    DistractionStarted { at: Instant, reason: DistractionReason },
    DistractionEnded   { at: Instant, duration: Duration },
    PhoneUseStarted { at: Instant, posture: PhonePosture },
    PhoneUseEnded   { at: Instant, duration: Duration },
    DrowsyStarted { at: Instant },
    DrowsyEnded   { at: Instant, duration: Duration },
    AwayStarted   { at: Instant },
    AwayEnded     { at: Instant, duration: Duration },
    PeopleChanged { count: u8 },
}
```

An event is emitted only on a state *transition that has satisfied its dwell
requirement*. There is no event per frame, and no event for `GlancingAway`.

---

## 8. Calibration

Thresholds expressed in degrees and ratios do not transfer between people, lighting
conditions, or eyewear. Without calibration, D6 is unachievable.

First run produces a `CalibrationProfile` from a guided ~60 s capture:

- neutral head pose, giving the origin for yaw/pitch/roll
- eyes-open EAR baseline, giving a per-user drowsiness threshold
- per-user yaw and pitch bounds derived from the neutral pose

Every derived threshold in §7 is expressed relative to this profile. The profile is
a plain serialisable value; persisting it is slice B's concern, not this one. The
engine accepts a profile as input and emits a recommendation when none is present.

---

## 9. Power budget

Adaptive tick rate, driven by observed state rather than a fixed clock:

| Condition | Tick rate |
|---|---|
| Face present, scene changing | 5 Hz (base) |
| No face for 10 s | 1 Hz |
| No face for 60 s, or session locked/idle | engine sleeps entirely |

Stage 3 runs at 1–2 Hz regardless. Stage 0 short-circuits all inference on a static
scene.

**Target: under ~3 % of one core averaged** over a 30-minute replay session on the
reference machine, validated against a hard ceiling test in the harness rather than
asserted in prose.

Session idle and lock detection uses `GetLastInputInfo` and WTS session-change
notifications. This is the only Windows-specific code in the engine, and it sits
behind a trait so replay runs are unaffected.

---

## 10. Public API

```rust
pub struct Engine { /* owns a Box<dyn FrameSource> */ }

impl Engine {
    pub fn new(source: Box<dyn FrameSource>, profile: CalibrationProfile) -> Self;
    pub fn tick(&mut self, now: Instant) -> Tick;
}

pub struct Tick {
    pub state: FocusState,          // always present; drives time accounting
    pub observation: Observation,   // diagnostic; drives the probe overlay
    pub events: SmallVec<[FocusEvent; 2]>,
    pub confidence: Confidence,
}
```

**Frame ownership.** The engine owns its `FrameSource`, supplied at construction.
`tick` pulls the newest available frame when the current cadence calls for one, and
otherwise advances the state machine on the clock alone. Callers never pass frames
in — which is precisely what lets `ReplaySource` and `CameraSource` be swapped
without the caller changing, and what makes G4 achievable.

`tick` is the only entry point, and it is deterministic given the same frame
sequence and clock. The determinism gate in §11.3 depends on that property.

Deliberately small. The engine holds no reference to storage, UI, or any Windows
API outside the idle-detection trait named in §9.

---

## 11. Testing and acceptance

### 11.1 Replay harness

```
lockin-replay <video> <labels.jsonl> --out report.json
```

Emits per-state precision and recall, **false nudges per hour of focused work**,
CPU milliseconds per frame, peak RSS, and p95 tick latency.

The headline metric is false nudges per hour, because that is the product
requirement from D6 expressed as a number.

### 11.2 Labelled corpus

**The corpus is a deliverable of this slice, not an afterthought.** Roughly 20
two-minute clips covering: focused work, looking away, phone in hand, phone at ear,
drowsy, leaving the desk, two people present, wearing glasses, and poor lighting.

Without it, every accuracy claim in this project is an opinion. This is real work
and is scheduled as such. Recordings are gitignored: they are large and contain the
maintainer's face.

### 11.3 Acceptance gates

These are the committed gates for this slice. The corpus may justify revising a
number, but never removing a gate:

| Gate | Target |
|---|---|
| False nudges per hour of focused work | **≤ 1 per 4 hours** |
| Recall on sustained phone use | **≥ 90 %** |
| Average CPU, 30-minute replay | **< 3 % of one core** |
| Peak RSS | **< 250 MB** |
| Determinism | Identical report hash across repeated replays of the same input |

The determinism gate matters more than it looks: without it, regressions in a
probabilistic pipeline are undetectable.

---

## 12. Repository layout

```
lock-in/
├─ Cargo.toml                    # workspace
├─ crates/
│  ├─ lockin-vision/             # the engine (this slice)
│  │  └─ src/{capture,models,analyze,fuse,state,schedule}.rs
│  └─ lockin-cli/                # replay + probe binaries
├─ models/                       # fetch.ps1 + README.md + LICENSES/
├─ corpus/                       # labelled clips (gitignored)
└─ docs/superpowers/specs/       # this document
```

### 12.1 Verified technical dependencies

Versions verified against crates.io on 2026-10-14. Crates marked later-slice are
listed so the boundary decisions are visible now, not because this slice uses them.

| Purpose | Crate | Version | Notes |
|---|---|---|---|
| ONNX inference | `ort` | 2.0.0-rc.14 | **No stable release exists.** Pin `~2.0`. Use `load-dynamic` to avoid DLL-resolution failures. |
| GPU offload | `ort` feature `directml` | — | Bundled in every Windows prebuilt. Works with no extra runtime. |
| Camera capture | `nokhwa` | 0.10.11 | `input-msmf` + `decoding`. Media Foundation, pure Rust, no C++ dependency. |
| Session idle / lock | `windows` | 0.62.2 | Win32 called directly. No maintained wrapper crate exists. |
| Storage *(slice B)* | `rusqlite` | 0.40.2 | `bundled` feature; clean MSVC build, no external DLL. |
| Service host *(slice D)* | `windows-service` | 0.8.1 | Restart-on-failure is SCM configuration, not crate configuration. |
| App ↔ service IPC *(slice D)* | `tokio` named pipes | 1.53.3 | In-tree. **The default DACL grants Everyone and Anonymous read access** and must be replaced with an explicit security descriptor. |
| UI *(slice C)* | `tauri` | 2.12.3 | WebView2 is the only Windows renderer. |

**Rejected:** `opencv` (requires a multi-hour OpenCV C++ build), `kamera`
(unmaintained), `escapi` (abandoned since 2019, non-standard licence), `interprocess`
(unnecessary abstraction for one Windows-only channel), gRPC over localhost (port
exposure and firewall prompts for no benefit).

---

## 13. Model provenance and licensing

Verified by primary-source audit. The code/weights distinction is the whole point of
this table.

| Model | Code | Weights | Verdict |
|---|---|---|---|
| YuNet | MIT | **MIT** | ✅ Ship. 227 KB, official ONNX. |
| NanoDet-Plus | Apache-2.0 | **Apache-2.0** | ✅ Ship. 2.3 MB, official ONNX export, COCO person=0 and cell phone=67. |
| YOLOX-Nano | Apache-2.0 | Apache-2.0 | ✅ Fallback detector. 1.8 MB. |
| ONNX Runtime | MIT | — | ✅ Ship. |
| MediaPipe Face Mesh | Apache-2.0 | **UNVERIFIED** | ⚠️ See R1. |
| Ultralytics YOLO | AGPL-3.0 | **AGPL-3.0** | ❌ Excluded — would force all of LOCK-IN to AGPL. |
| InsightFace SCRFD / buffalo | MIT | **Non-commercial only** | ❌ Excluded. |
| PFLD | **No licence at all** | UNVERIFIED | ❌ Excluded. |

`models/fetch.ps1` downloads and hash-verifies each model; no multi-MB binary is
committed to git. `models/README.md` records for every model its source URL,
licence, sha256, and exact export or conversion steps.

### Prior art, and what is taken from it

None of the four reference repositories contribute runnable code — all are Python
single-script webcam demos. What is taken is models, mathematics, and design ideas:

- **`Cell_phone_usage_detection`** contributes the most valuable idea: a posture
  taxonomy distinguishing *phone in hand* and *phone at head* from *no phone*, rather
  than merely "a phone is visible". Its own implementation is unusable (80–220 ms
  per ROI, weights not committed, entry point imports a missing module) and it has
  no licence.
- **`ai-focus-detector`** contributes the EAR and iris-aspect-ratio formulas, the
  canonical landmark index sets, and above all the calibrate-then-threshold pattern.
  It has **no licence**, so its code cannot be copied; the ideas are reimplemented
  clean-room.
- **`distraction-system`** (MIT) contributes the six canonical `solvePnP`
  correspondence points. Its camera matrix has a swapped principal point, and its
  inline blocking beeps with no debounce are a worked example of exactly what §7
  exists to prevent.
- **`Person_Use_Phone_Detector`** (MIT) contributes confirmation of the COCO class
  indices. Its bare AABB overlap test is replaced by region classification (§5.1)
  plus temporal hysteresis.

### R1 — the MediaPipe weights licence gap

Google publishes MediaPipe's *code* under Apache-2.0 but no model-specific licence
for the `.tflite`/`.task` weights, and the model cards are not machine-readable. The
ONNX conversions in the PINTO model zoo are Apache-2.0 as *scripts*; the converted
weights inherit whatever Google granted, which is undocumented.

**Decision:** ship the ONNX export with attribution and record the gap explicitly in
`models/LICENSES/`, rather than hiding it or overstating the licence. This is normal
for open-source projects and is the only practical route, since YuNet's five points
are too sparse for head pose and cannot detect eyes-closed at all.

---

## 14. Risks

| ID | Risk | Mitigation |
|---|---|---|
| **R1** | MediaPipe Face Mesh weights licence unverified | Documented and attributed; tracked openly in `models/LICENSES/`. |
| **R2** | Face Mesh is a cropper model, so stage 2 depends entirely on stage 1. A missed face detection cascades into total signal loss. | Measure stage-1 recall on the corpus specifically; consider a fallback path and treat stage-1 miss rate as a first-class metric. |
| **R3** | Phone association is the weakest signal. COCO "cell phone" is a small-object class with a high miss rate, and a phone on the desk is easily mistaken for phone use. | Region classification plus the ~2 s dwell; `Present` never counts as use; report phone-use recall separately from other states. |
| **R4** | The labelled corpus is real work and may be skipped under time pressure, at which point all accuracy claims become unverifiable. | It is a declared deliverable of this slice and gates the acceptance criteria. |
| **R5** | ~~NPU offload may not be viable.~~ **Resolved: it is not.** `ort`'s prebuilt binaries omit OpenVINO; Intel's prebuilt OpenVINO EP tops out at ORT 1.24.1 while `ort` 2.0 requires ORT 1.31.0; DirectML is GPU-only and cannot reach the NPU. | Requirement dropped. The Arc iGPU via bundled DirectML is the acceleration target; CPU remains the baseline (C3). |
| **R8** | **`ort` has no stable release.** 2.0.0-rc.14 has been a release candidate for roughly two years and the 2.x API is explicitly unstable — the weakest link for a shipping product. | Pin `~2.0` and keep every ONNX Runtime call inside `models/`, so `tract` (pure-Rust, CPU-only) is a drop-in fallback if `ort` breaks. Revisit when 2.0 goes stable. |
| **R9** | DirectML permits **only one thread to call `Run` per session**, and disables memory-pattern and parallel execution. | Serialise inference per session; never share one session across threads. Model the constraint into the scheduler rather than discovering it under load. |
| **R6** | Thermal throttling on thin laptops changes latency mid-session and can starve the pipeline. | Adaptive tick rate already handles degradation; the harness records p95 latency and CPU over a full 30-minute run, not a short burst. |
| **R7** | Calibration quality dominates real-world precision; a poor profile silently degrades everything downstream. | Calibration is a guided, validated capture that rejects bad sessions rather than storing them. |

---

## 15. Deferred to later slices

Recorded here so they are not lost, and explicitly not built now:

- **Slice B:** SQLite schema, time accounting, distraction events, daily rollups,
  streak and timezone logic, export and delete-all.
- **Slice C:** Tauri shell, dashboard, onboarding and calibration UI, toggle and
  Ultra Lock-in timer, nudge sound and animation, tray, autostart.
- **Slice D:** Windows Service, root-owned unlock timestamp, watchdog restart,
  honest tamper-resistance tiers.
- **Slice E:** NSIS installer, code signing, auto-update, CI.
- **Later:** adaptive per-user learning from correction feedback; macOS port.
