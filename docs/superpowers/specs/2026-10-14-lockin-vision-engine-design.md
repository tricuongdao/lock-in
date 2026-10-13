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
| D5 | **Rich attention profile (Tier 3)** | Face presence, head pose, eyes-open/closed, phone-in-use, other-people-present. Gaze was dropped after verification showed the chosen landmark model carries no iris points — see §5.2. |
| D6 | **High precision — silence over noise** | Applied to *both* measurement and interruption. A beep while the user legitimately blinks is what makes people uninstall a focus app. The accepted cost is that the dashboard under-reports. |
| D7 | **Vision engine first** | Highest risk, fully testable in isolation. |

---

## 2. Goals and non-goals

### Goals

- **G1** Emit a trustworthy `FocusState` on every tick and discrete `FocusEvent`s
  when a state has been held long enough to count.
- **G2** Make the engine's behaviour reproducible and its resource cost measurable
  through a deterministic replay harness, with a live overlay for qualitative
  behavioural acceptance.
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
- **C3 — CPU is the baseline; the GPU is vendor-agnostic.** Acceleration runs
  through **DirectML**, which is vendor-neutral: it targets any Direct3D 12 adapter,
  so NVIDIA, AMD and Intel GPUs are all first-class, discrete *and* integrated. Every
  Windows 11 machine clears this bar, since Windows 11 already requires a WDDM 2.0+
  driver. CPU remains the guaranteed fallback, and the engine must still hit its
  budget on CPU alone. **The Intel NPU is not reachable from this stack** (§12.1) and
  is not a design target.
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
    analyze/   NMS, solvePnP, EAR, eye state, phone↔person association
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
| 2 · Dense landmarks | 5 Hz, only when stage 1 found a face | Face Mesh (2.3 MB), 468 points | head pose via `solvePnP`, EAR, eye boxes |
| 2b · Eye state | 5 Hz, only when stage 2 produced an eye box | open-closed-eye-0001 (45 KB, Apache-2.0) | eyes-open / eyes-closed, confirming EAR |
| 3 · Person + phone | 1–2 Hz, independent cadence | YOLOX-Nano (~3.6 MB, Apache-2.0, ready-made ONNX) | person boxes (class 0), cell-phone boxes (class 67) |

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

### 4.4 Execution provider strategy

Any GPU must work, integrated ones included — and equally, no machine may break
because of its GPU. Two facts shape this.

**DirectML is vendor-neutral.** It targets any Direct3D 12 adapter, so NVIDIA
(Kepler and newer), AMD (GCN and newer) and Intel (Arc, Iris Xe, UHD) are all served
by a single execution provider that ships inside `ort`'s prebuilt binaries. There is
nothing extra to download and no per-vendor SDK. It is in ONNX Runtime's *sustained
engineering* status and caps at **opset 20** (C6).

**A GPU is not automatically faster here.** These models are 2–5 MB and run at
1–5 Hz. For graphs this small, kernel-launch and transfer overhead can make an
integrated GPU *slower* than plain CPU. Preferring the GPU by default would
therefore be a pessimisation on some laptops, and on a battery-powered machine that
runs for hours it also costs power.

So the engine does not *choose* an execution provider — it **measures** one.

| Step | Behaviour |
|---|---|
| Enumerate | Query available providers. DirectML and CPU are always candidates. |
| Verify | Run each candidate over a sample of frames and compare its outputs against the CPU reference within tolerance. A provider that crashes or disagrees is discarded. |
| Benchmark | Time each surviving candidate, **per model**, over the sample. |
| Select | Pick the fastest verified configuration per model, not globally — it is entirely normal for the winner to be `face_detector: CPU, face_mesh: DirectML`. |
| Persist | Store the selection in the runtime profile, alongside §8's calibration profile. |
| Re-evaluate | On driver-version change, adapter change, or explicit user request. |

Consequences that fall out of this design:

- **Fallback is structural, not a special case.** Sessions list the chosen GPU
  provider *and* CPU, so ONNX Runtime partitions the graph and runs unsupported
  operators on CPU rather than failing.
- **No GPU is not an error.** Remote Desktop sessions, VMs and GPU-less machines
  have no D3D12 adapter. Session creation fails cleanly and the engine runs on CPU.
  This is a supported configuration and the corpus tests it.
- **A GPU can vanish mid-session.** Driver resets and TDR events surface as
  device-removed errors. The engine catches them, rebuilds the affected session on
  CPU, and continues — losing a few ticks, never the session.
- **Discrete GPUs are not preferred by default.** On a hybrid laptop the autotune
  may select the dGPU, but battery cost is weighed against latency gain, because
  this app runs for hours. A few milliseconds is not worth the power.
- **Stage 3 runs on its own thread and its own session.** This stops the slow
  detector blocking the fast path, and it satisfies DirectML's rule that only one
  thread may call `Run` per session (R9).

CUDA and TensorRT are deliberately **not** supported in v1: they are NVIDIA-only,
need multi-gigabyte runtimes, and would add nothing for graphs this small. The
provider abstraction leaves room to add them later without touching the state
machine.

---

## 5. Signal extraction

All derived quantities are computed in Rust in `analyze/`. Nothing is delegated to
a Python runtime.

| Signal | Method | Consumed by |
|---|---|---|
| Face count | Stage 1 box count | `OtherPresent`, `AwayFromDesk` |
| Head pose (yaw, pitch, roll) | `solvePnP` over the six canonical Face Mesh indices `33, 263, 1, 61, 291, 199` | `Distracted`, `GlancingAway` |
| Eye aspect ratio (EAR) | Six points per eye, standard EAR formula | `Drowsy` candidate |
| Eye state | 32×32 roll-corrected eye crop through open-closed-eye-0001 | `Drowsy` confirmation |
| Phone posture | Association of class-67 boxes with class-0 boxes, region-classified against the head box from stage 2 | `PhoneInUse` |

There is deliberately **no gaze vector**. See §5.2.

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

### 5.2 Verified model I/O and preprocessing

Verified against the PINTO and upstream sources. These are contracts the Rust
implementation must satisfy exactly, and **every model has a different preprocessing
regime** — sharing one preprocessing path between them is a bug waiting to happen.

| | Face Mesh (stage 2) | YOLOX-Nano (stage 3) |
|---|---|---|
| Input name | `input` | `images` |
| Input shape | `[N,3,192,192]` float32 | `[1,3,416,416]` float32 |
| Colour | **RGB** — reverse of OpenCV's BGR | **BGR** |
| Scale | `/255` → `[0,1]` | **0–255, no scaling** |
| Normalisation | none | mean `[103.53, 116.28, 123.675]`, std `[57.375, 57.12, 58.395]` |
| Geometry | resize **ignoring aspect ratio** | letterbox to a 32-multiple, pad value **114** |
| Outputs | `score [N,1]`, `final_landmarks [N,468,3]` **int32** | `output [1,3549,85]` float32 |

**Face Mesh requires a cropped face and cannot take a full frame.** It carries four
extra int32 inputs — `crop_x1`, `crop_y1`, `crop_width`, `crop_height` — because the
decode is fused into the graph; `_post` means post-processing is baked in. The graph
returns **absolute pixel coordinates in the source frame**, so no manual un-cropping
is needed; divide by frame dimensions to normalise.

Two consequences:

- **Stage 1 is load-bearing, not merely useful.** Its box is the crop rect, so a
  missed face detection means total signal loss for stage 2. This is R2, now
  confirmed structural rather than hypothetical, and it makes stage-1 recall a
  first-class metric.
- **The crop needs margin.** The upstream model card specifies a 25 % margin around
  the face and PINTO's documentation warns the cropped area "must not be too
  narrow". The crop rect is therefore the YuNet box expanded by a margin and clamped
  to frame bounds — a tuning parameter the corpus must exercise.

**There is no gaze vector, and this is a verified limitation, not an oversight.**
The export emits 468 landmarks; iris points 468–477 belong to MediaPipe's
`refine_landmarks` variant, which this graph does not contain. `GlancingAway` and
`Distracted` therefore rest on **head pose**, with EAR covering eyes-open. Eye
detection itself survives intact — landmarks 0–467 include full eye contours, so EAR
and `Drowsy` are unaffected. The richer export (PINTO 410, 256×256) may carry 478
points, but it ships with no README, no demo and no code, so its signature is
unverified and this design does not depend on it.

**YOLOX-Nano's output is fully raw** — there is no sigmoid in the graph.
Post-processing in Rust is two elementwise operations (`+ grid`, `exp`) scaled by
stride, then `sigmoid(obj) × sigmoid(cls)`, then class-agnostic NMS. NanoDet-Plus
would additionally require a four-level centre-prior grid and four softmax-over-
eight-bin DFL integrals per candidate point. That, plus shipping a ready-made ONNX
with no PyTorch export step, is why YOLOX-Nano is the primary detector.

### 5.3 Behaviour models

Two small classifiers, adopted after verification showed they beat the geometric
approach on their own signals. Both passed the test the rejected alternatives
failed: **their training domain matches our imagery.**

| | open-closed-eye-0001 (stage 2b) | Glasses-Detector (calibration) |
|---|---|---|
| Licence | Apache-2.0 (OpenVINO zoo, `models/public/`) | MIT |
| Size | 45 KB | 4.92 MB (`anyglasses` / `eyeglasses` head) |
| Input | `input.1` `[1,3,32,32]` f32, **BGR**, `(x−127)/255` | `images` `[1,3,256,256]` f32, **RGB**, ImageNet mean/std |
| Output | `19` `[1,2,1,1]` softmax — **0 = closed, 1 = open** | `logits` `[1,1]`, positive = glasses present |
| Cadence | every tick with an eye box | **once, at calibration** |
| Trained on | MRL Eye Dataset — close-up eye crops | ten real face-image corpora |

**Class order is inverted relative to the documentation.** OpenVINO's README states
`[open, closed]`, but its own accuracy-checker converter defines
`{0: 'closed', 1: 'open'}` and its gaze demo tests `out[0] < out[1]`. **The code is
authoritative: index 0 is closed.** Trusting the README would invert `Drowsy` into
"alert while asleep" — a silent, total failure of the signal. This is R14.

**Normalisation is mandatory, not cosmetic.** Raw 0–255 input yields **NaN**, not
merely degraded output.

**The eye box** comes from stage-2 landmarks: a square 1.8× the inter-landmark eye
distance, roll-corrected before cropping. That is the geometry OpenVINO's own gaze
demo uses, and it is what makes the domain match hold.

**Rejected alternatives, and why:**

- **PPC / PUC** (phone use) — the positive class is *phone pointed at the camera*,
  not phone held. Its `possession` samples carry filenames from PUC's
  `point_somewhere` corpus, and `--test_ratio` defaults to `0.0`, so its F1 of 0.99
  is in-distribution with no test set at all. Wrong semantics behind a good number.
- **SC** (sitting) — trained on Google AVA Actions film footage as a 32×24
  whole-body silhouette. A close-up webcam frame is head and shoulders, so the cue
  it learned is absent. `AwayFromDesk` keeps the geometric approach: person presence
  plus a desk region of interest.

Glasses detection runs **once at calibration** and stores a flag in the profile. It
is not a continuous signal, because stage 2b already covers eyes-closed robustness,
and paying 4.92 MB of inference per tick would buy nothing.

### 5.4 Head pose: why solvePnP, not a network

Head pose comes from `solvePnP` over six landmarks rather than a pretrained head-pose
network, despite several being available and permissively licensed.

- **The decision does not need better accuracy.** `GlancingAway` and `Distracted`
  turn on roughly 15–20° of deviation from the calibrated neutral. No head-pose
  network reports better than ~3.5° MAE on real imagery, and solvePnP costs nothing
  because the landmarks already exist.
- **Past the threshold, precision stops mattering.** PnP does degenerate beyond
  roughly ±25° yaw, as landmark error grows and the projection becomes
  ill-conditioned — but by then the decision to leave `Focused` has already been
  made. A more accurate network buys nothing at the point where we actually decide.
- The cheap permissive alternative, OpenVINO `head-pose-estimation-adas-0001`
  (Apache-2.0, 1.9 M params, real-imagery-trained, single-digit ms), was therefore
  **not adopted**. It would add a sixth model and a fifth preprocessing regime to
  improve a number we never consult.

Two things to verify during implementation, because both produce **systematic** error
rather than noise:

- **Camera intrinsics.** Monocular PnP absorbs a wrong focal length as a fixed
  pitch/yaw bias, which calibration would then bake in as a false neutral. The focal
  estimate deserves a real check, not a guess derived from frame width.
- **Profile-face landmarks.** Landmark quality past ±45° yaw determines whether PnP
  degrades gracefully or produces nonsense. The dwell windows mask this, but it is
  worth confirming deliberately rather than assuming.

### 5.5 Why there is no pretrained "focused vs distracted" model

Investigated thoroughly; the conclusion is that none exists that is **usable**, for
three independent reasons. Recorded here so it is not re-litigated.

**The student-engagement family is legally unusable.** DAiSEE is the obvious
candidate — 9,068 clips, 112 subjects, 1080p **laptop webcam** footage from dorms and
libraries, which is exactly our imagery domain. But its terms require agreeing to use
the images "for research purposes only" and not to provide them to second parties, so
**every DAiSEE-trained weight inherits that restriction**, whatever its model card
claims.

**Its accuracy was insufficient anyway.** The best published 4-class result is
**63.9%** on the official held-out split. That test set is 0.22% "very low" — four
clips — and the paper states that *"none of the methods are able to classify any
samples to the first two classes correctly."* A third of predictions wrong and biased
toward majority classes is not something to beep at a working user on.

**The widely-cited high numbers are not the same task.** Claims of 99.86% on
"subject-independent splits" are incompatible with ~30% measured label noise and a
61-clip minority class, and read as split leakage. The 98.82% binary engaged/not-
engaged figure is a different, far easier relabelling.

Two further traps in this space: Ultralytics-derived drowsiness models carry an **MIT
tag over AGPL-3.0 weights** — the same trap as the four original reference
repositories — and one popular "student engagement" model's 94.2% headline is
**training** accuracy over roughly 600 images of about 10 people.

The cautionary data point: DAiSEE's own authors benchmarked a commercial affect SDK
on real webcam video and found it reports **"attentive" while the subject's eyes are
closed.** The best-funded attempt at this exact task fails on a signal we compute for
free from landmarks.

### 5.6 Why gaze is permanently excluded

Gaze was dropped first because the chosen landmark model carries no iris points. Two
further findings make that a permanent exclusion rather than a deferred feature.

**Every usable pretrained gaze model is licence-poisoned.** Gaze360's licence forbids
commercial use of *"models trained on dataset"* — not merely its images. All released
L2CS-Net checkpoints and every MobileGaze model are Gaze360-trained, so their MIT and
Apache-2.0 code badges do **not** grant the weights, and the Arm ONNX mirror carries
the same defect. MPIIGaze and MPIIFaceGaze are CC BY-NC-SA 4.0 outright. The
`py-feat` project hit this and relabelled its L2CS mirror from MIT to `other` for
exactly this reason.

This matters beyond commerce: a non-commercial restriction is incompatible with the
OSI definition of open source, so shipping any of these would quietly convert LOCK-IN
from *open source* into *source-available, non-commercial* — a direct conflict with D2.

**Even with a licence, uncalibrated gaze fails on the case we care about.** The
geometry: a webcam sits ~5 cm above a screen whose centre is ~60 cm away, so screen
centre is about +5° pitch from the camera, while a phone held ~40 cm below the camera
is about −53°. The differentiator is ~58°. Against L2CS's 10.41° cross-dataset error
that is under 3σ. Worse, the error is **systematic rather than random**: MPIIGaze's
vertical gaze range spans only **[−1.5°, +20°]**, so ~−30° phone gaze is
out-of-distribution, and a bin classifier will confidently round toward the nearest
familiar angle — reporting "screen" while the user reads their phone, every frame.
Head pose does not rescue it, because held-phone and screen gazing produce similar
head pitch; the separation lives in the eyes.

Calibrated gaze at ~3.9° error would genuinely work, but only with per-user and
per-camera-height calibration *and* a licence-clean model, and no such model currently
exists. Gaze is therefore closed, not postponed.

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
    head: Euler,                // yaw, pitch, roll in degrees
    ear: f32,
    eye_closed: Option<bool>,   // stage 2b confirmation; None when no eye box
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
    Focused,        // face present, head pose in bounds, eyes open, no phone
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
- **`Drowsy` requires two independent signals to agree.** EAR below the profile
  threshold *and* the eye classifier reporting closed, both held for the entry
  dwell. Requiring agreement strictly reduces false positives, which is the trade D6
  asks for. It costs some `Drowsy` recall, and that cost is accepted deliberately
  rather than discovered later.

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
- **whether the user wears glasses**, from a single calibration-time classifier
  inference. Glasses are the documented failure mode for EAR, so this flag adapts
  the EAR threshold rather than being displayed and ignored.

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
lockin-replay <fixture-video> --out report.json
```

Emits per-state durations, CPU milliseconds per frame, peak RSS, p95 tick latency,
and a hash of the full state sequence. There are no labels to score against, so the
harness measures **mechanical** properties rather than accuracy.

### 11.2 Test fixture

A small **unlabelled** fixture: a handful of 15–30 second webcam clips showing a
face present, a face absent, and a hand-held object. **No labels and no per-frame
annotation.** Recorded once, in about a minute.

Its only job is to give the replay harness real input so the mechanical gates below
are measurable and reproducible. It is a test fixture, not an evaluation set, and
nothing in this spec claims accuracy measurements from it.

Recordings are gitignored: they are large and contain the maintainer's face.

### 11.3 Acceptance gates

All of the following are measurable and run in the harness or in CI:

| Gate | Target |
|---|---|
| Average CPU, 30-minute run | **< 3 % of one core** |
| Peak RSS | **< 250 MB** |
| Determinism | Identical output hash across repeated replays of the same fixture |
| GPU/CPU parity | Identical `FocusState` sequence on CPU and on DirectML over the fixture |
| GPU-less operation | Fixture passes with no D3D12 adapter present |
| Idle behaviour | Engine sleeps within 60 s of losing the face, and immediately on session lock |
| Eye class order | Unit test over known open-eye and closed-eye fixtures, so R14 cannot silently invert |
| No frame written to disk | Asserted by test: a replay run writes no frame data anywhere |

**Behavioural acceptance is qualitative and human-confirmed.** With no labelled
corpus, the engine's *accuracy* is not measured by any gate. Instead, a short manual
checklist is confirmed against the live `probe` overlay:

- sitting focused → reads `Focused`
- turning the head away for several seconds → `Distracted` *after* the dwell, not before
- picking up a phone and holding it → `PhoneInUse`; a phone lying on the desk → **not** `PhoneInUse`
- standing up and leaving → `AwayFromDesk`
- a second person entering frame → `OtherPresent`

This catches gross failures, which is what it is for. It does not measure precision,
and the spec does not pretend otherwise — see R4.

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
| GPU offload | `ort` feature `directml` | — | Bundled in every Windows prebuilt. **Vendor-neutral: NVIDIA, AMD and Intel, discrete and integrated.** No extra runtime. Caps at opset 20. |
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
| **YOLOX-Nano** | Apache-2.0 | **Apache-2.0** | ✅ **Primary detector.** ~3.6 MB. Megvii ship a **ready-made ONNX**, so no PyTorch export step is needed. Simplest Rust post-processing of the two (§5.2). COCO person=0, cell phone=67. |
| NanoDet-Plus | Apache-2.0 | Apache-2.0 | ✅ Alternative. ~4.7 MB FP32, but needs a PyTorch `.ckpt` run through `tools/export_onnx.py`, and its DFL decode is materially more Rust code. |
| open-closed-eye-0001 | Apache-2.0 | **Apache-2.0** | ✅ Ship. 45 KB. Trained on the MRL Eye Dataset — close-up eye crops, matching our domain. **Index 0 = closed**, contrary to OpenVINO's own README. |
| Glasses-Detector | MIT | **MIT** | ✅ Ship. 4.92 MB. Ten real face-image corpora, and it covers *transparent* glasses — the case that actually breaks EAR. Calibration-time only. |
| PPC / PUC | MIT | MIT | ❌ Rejected for phone use. Positive class is phone *pointed at the camera*, not held, and `--test_ratio` defaults to `0.0`, so no test set exists. |
| SC (Sitting) | MIT | MIT | ❌ Rejected. Trained on AVA film footage as a whole-body silhouette; a close-up webcam frame removes the cue it learned. |
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
| **R2** | **Confirmed structural, not hypothetical.** Face Mesh requires a cropped face and cannot take a full frame, so stage 2 depends entirely on stage 1 — a missed face detection is total signal loss. | Stage-1 recall is a first-class metric measured on the corpus. The crop rect carries a 25 % margin per the model card, and that margin is a corpus-tuned parameter (§5.2). |
| **R3** | Phone association is the weakest signal. COCO "cell phone" is a small-object class with a high miss rate, and a phone on the desk is easily mistaken for phone use. | Region classification plus the ~2 s dwell; `Present` never counts as use; report phone-use recall separately from other states. |
| **R4** | **Accuracy is asserted, not measured.** With no labelled corpus, the precision stance rests entirely on reasoning — dwell windows, asymmetric hysteresis, the confidence floor, and the EAR/classifier agreement gate — with no ground-truth evidence behind it. | Accepted deliberately. The replay harness is built to consume a labelled corpus the moment one exists, so adding one later requires no engine change. The qualitative checklist in §11.3 catches gross failures in the meantime. |
| **R5** | ~~NPU offload may not be viable.~~ **Resolved: it is not.** `ort`'s prebuilt binaries omit OpenVINO; Intel's prebuilt OpenVINO EP tops out at ORT 1.24.1 while `ort` 2.0 requires ORT 1.31.0; DirectML is GPU-only and cannot reach the NPU. | Requirement dropped. The Arc iGPU via bundled DirectML is the acceleration target; CPU remains the baseline (C3). |
| **R8** | **`ort` has no stable release.** 2.0.0-rc.14 has been a release candidate for roughly two years and the 2.x API is explicitly unstable — the weakest link for a shipping product. | Pin `~2.0` and keep every ONNX Runtime call inside `models/`, so `tract` (pure-Rust, CPU-only) is a drop-in fallback if `ort` breaks. Revisit when 2.0 goes stable. |
| **R9** | DirectML permits **only one thread to call `Run` per session**, and disables memory-pattern and parallel execution. | Serialise inference per session; never share one session across threads. Stage 3 gets its own session and thread (§4.4) so the constraint is designed in rather than discovered under load. |
| **R10** | **A GPU can be slower than the CPU for graphs this small**, and a particular driver can silently return wrong outputs. | Autotune measures and verifies rather than assumes, per model (§4.4). CPU stays the baseline (C3), and a provider that fails verification is dropped. |
| **R11** | A driver reset or TDR event mid-session destroys the inference session. | Device-removed errors are caught and the affected session is rebuilt on CPU. The session survives with a few lost ticks. |
| **R12** | DirectML is in *sustained engineering* at Microsoft; new Windows ML investment moved to WinML. | The provider abstraction (§4.4) confines any future swap to WinML, or a retreat to CPU, to the `models/` module. Revisit at each ONNX Runtime release. |
| **R13** | The models require **four incompatible preprocessing regimes** — RGB `/255` for Face Mesh, BGR `0–255` letterboxed for YOLOX, BGR `(x−127)/255` for the eye classifier, RGB with ImageNet mean/std for the glasses detector. Mixing them up yields plausible-looking but wrong output rather than a clean error. | Preprocessing lives beside each model wrapper in `models/`, never in a shared helper (§5.2, §5.3). A swapped colour order degrades detection visibly on the corpus, so the corpus catches it. |
| **R14** | **OpenVINO's documentation contradicts its own code on eye class order.** The README says `[open, closed]`; the accuracy-checker converter and the gaze demo both say index 0 is closed. A silent inversion turns `Drowsy` into "alert while asleep" — the signal appears to work while being exactly backwards. | Trust the code, not the README (§5.3). Assert the class order with a unit test over a known closed-eye and a known open-eye fixture, so an upstream model swap cannot silently invert it. Normalisation is asserted the same way, since raw 0–255 input yields NaN rather than a visible failure. |
| **R15** | PnP head pose carries **systematic** error when camera intrinsics are wrong — a bad focal length appears as a fixed pitch/yaw bias that calibration then bakes in as a false neutral. It also degenerates past roughly ±25° yaw. | Intrinsics get a real check rather than a guess from frame width. The 15–20° decision threshold is reached well before degeneracy, so accuracy past the decision point is not required (§5.4). |
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
