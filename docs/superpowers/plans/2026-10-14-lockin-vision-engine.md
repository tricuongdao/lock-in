# LOCK-IN Vision Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless Rust vision engine for LOCK-IN — a library that turns laptop-webcam frames into a trustworthy `FocusState` per tick and discrete `FocusEvent`s, plus the CLI harness that proves it behaves deterministically within budget.

**Architecture:** A cascaded tiered pipeline. A near-free motion gate short-circuits everything on a static scene; YuNet answers "is a face present" at 5 Hz; Face Mesh produces dense landmarks only when it is; a 45 KB eye-state classifier confirms eyes-closed; YOLOX-Nano finds people and phones at 1–2 Hz on its own thread. Their outputs fuse into an `Observation`, and a temporal state machine with dwell windows and asymmetric hysteresis turns observations into states and events. No UI, no storage, no daemon.

**Tech Stack:** Rust (stable), `ort` 2.0 (ONNX Runtime) with the bundled DirectML provider, `nokhwa` for Media Foundation capture, `image` for the fixture format, `windows` for idle/lock detection, `clap` for the CLIs.

**Spec:** `docs/superpowers/specs/2026-10-14-lockin-vision-engine-design.md`

## Global Constraints

- **Platform:** Windows 11 only. No macOS or Linux support, no `cfg` branches for them.
- **ONNX Runtime:** `ort = "~2.0"`. There is no stable release; pin the minor. All ONNX Runtime calls live in `models/` and nowhere else (R8).
- **Opset:** every model is exported at opset ≤ 20. DirectML caps there (C6).
- **Licences:** only MIT or Apache-2.0 models. Never Ultralytics-derived weights (AGPL), never InsightFace/SCRFD weights (non-commercial), never Gaze360- or MPIIGaze-derived weights (non-commercial). See spec §13.
- **Preprocessing:** four mutually incompatible regimes. Each lives **beside its own model wrapper**; a shared preprocessing helper is forbidden (R13, spec §5.2/§5.3).
- **Colour orders:** Face Mesh takes RGB `/255`; YOLOX takes BGR `0–255` letterboxed pad 114; the eye classifier takes BGR `(x−127)/255`; the glasses detector takes RGB with ImageNet mean/std.
- **No frame data on disk or network.** Inference is in-memory only. The only writer is `probe --record`, which the user invokes deliberately.
- **Determinism:** identical frame sequence and clock must produce an identical output hash. The harness asserts it.
- **Thresholds are profile-relative**, never absolute. The spec's starting values live in one `Thresholds` struct.
- **Camera access is user-session only.** Nothing in this crate may assume Session 0.

## Review Focus

Five input classes the spec implies but that no single task's tests naturally exercise. Each gets a test in the owning task, named below.

1. **Camera unavailable or held by another process.** A user with Zoom open still expects LOCK-IN to start, report clearly, and recover when the camera frees up. Owner: Task 5 (`test_camera_open_failure_is_not_fatal`).
2. **Prolonged absence of any face.** Someone leaves for lunch; the engine must sleep within 60 s, must not accrue "distracted" time, and must resume instantly on return. Owner: Task 17 (`test_away_time_does_not_accrue_distraction`).
3. **Extreme and profile head angles.** Past roughly ±25° yaw, PnP degenerates; a user turning to talk to someone must not produce NaN or a spurious `Focused`. Owner: Task 10 (`test_head_pose_is_bounded_and_never_nan`).
4. **Primary-face identity swapping between frames.** Two people in frame, boxes of similar size; if the primary face flip-flops, the state machine flaps and beeps. Owner: Task 14 (`test_primary_face_selection_is_stable_across_swaps`).
5. **Model output at or below the confidence floor.** A dark room or heavy motion blur drives detector scores down; the engine must carry the last stable state forward rather than inventing a distraction. Owner: Task 16 (`test_low_confidence_ticks_carry_forward`).

---

## Task 1: Workspace scaffold and the `Frame` type

**Files:**
- Create: `Cargo.toml` (workspace), `rust-toolchain.toml`, `crates/lockin-vision/Cargo.toml`, `crates/lockin-vision/src/lib.rs`, `crates/lockin-vision/src/frame.rs`
- Test: `crates/lockin-vision/src/frame.rs` (unit tests inline)

**Interfaces:**
- Consumes: nothing.
- Produces: `Frame { width: u32, height: u32, bgra: Vec<u8>, captured_at: Instant }`, with `Frame::new(width, height, bgra, captured_at) -> Result<Frame, FrameError>` and `Frame::luma_64x48(&self) -> [u8; 3072]`.

- [ ] **Step 1: Create the workspace and crate skeletons**

Workspace `Cargo.toml` lists members `crates/lockin-vision` and `crates/lockin-cli`, with `[workspace.dependencies]` holding `ort = "~2.0"`, `nokhwa = { version = "0.10", features = ["input-msmf", "decoding"] }`, `image = "0.25"`, `windows = "0.62"`, `clap = { version = "4", features = ["derive"] }`, `smallvec = "1"`, `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`, `thiserror = "2"`, `anyhow = "1"`. `rust-toolchain.toml` pins `channel = "stable"`.

- [ ] **Step 2: Write the failing test for `Frame::new` and `luma_64x48`**

```rust
#[test]
fn rejects_buffer_of_wrong_length() {
    let e = Frame::new(2, 2, vec![0u8; 15], Instant::now());
    assert!(matches!(e, Err(FrameError::BadBufferLength { expected: 16, got: 15 })));
}

#[test]
fn luma_is_deterministic_and_sized() {
    let f = Frame::new(64, 48, vec![7u8; 64 * 48 * 4], Instant::now()).unwrap();
    assert_eq!(f.luma_64x48(), f.luma_64x48());
    assert_eq!(f.luma_64x48().len(), 3072);
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p lockin-vision frame::`
Expected: FAIL — `cannot find function Frame::new`.

- [ ] **Step 4: Implement `Frame`**

`bgra` length must equal `width * height * 4`. `luma_64x48` nearest-neighbour downsamples and converts with Rec. 601 integer weights `(77*r + 150*g + 29*b) >> 8`. Nearest-neighbour, not area-average — the motion gate compares successive frames where sub-pixel accuracy is irrelevant, and a box filter is measurably slower.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p lockin-vision frame::`
Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml rust-toolchain.toml crates/
git commit -m "feat(vision): workspace scaffold and Frame type"
```

---

## Task 2: The `FrameSource` seam

**Files:**
- Create: `crates/lockin-vision/src/capture/mod.rs`, `crates/lockin-vision/src/capture/synthetic.rs`, `crates/lockin-vision/src/capture/replay.rs`
- Test: `crates/lockin-vision/src/capture/synthetic.rs`, `crates/lockin-vision/src/capture/replay.rs`

**Interfaces:**
- Consumes: `Frame` from Task 1.
- Produces: `trait FrameSource { fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError>; }`; `SyntheticSource::new(pattern: SyntheticPattern, count: usize, fps: u32)`; `ReplaySource::open(dir: &Path) -> Result<ReplaySource, CaptureError>`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn synthetic_source_yields_exact_count_then_none() { /* pattern Solid, count 3 -> 3 frames, then Ok(None) */ }

#[test]
fn synthetic_source_is_deterministic() { /* same seed -> byte-identical frames */ }

#[test]
fn replay_source_reads_numbered_images_in_order() { /* tempdir with 0001.png, 0002.png -> ordered */ }

#[test]
fn replay_source_errors_on_empty_directory() { /* Err(CaptureError::EmptyFixture) */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision capture::`
Expected: FAIL — unresolved imports.

- [ ] **Step 3: Implement the seam**

`SyntheticPattern` is `Solid(u8)` or `Checkerboard { cell: u32 }`. `ReplaySource::open` reads `*.png` and `*.jpg`, sorts lexicographically, and converts to BGRA8 via `image`. A replay frame's `captured_at` is the wall clock at read time, so replay timing is driven by the caller's tick rate rather than baked-in timestamps.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision capture::`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/capture/
git commit -m "feat(vision): FrameSource trait with synthetic and replay sources"
```

---

## Task 3: Motion gate (stage 0)

**Files:**
- Create: `crates/lockin-vision/src/motion.rs`
- Test: `crates/lockin-vision/src/motion.rs`

**Interfaces:**
- Consumes: `Frame::luma_64x48` from Task 1.
- Produces: `MotionGate::new(threshold: f32)`, `MotionGate::update(&mut self, luma: &[u8; 3072]) -> f32` returning mean absolute difference, and `MotionGate::is_static(&self) -> bool`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn identical_frames_report_zero_motion() { /* update twice with same luma -> 0.0, is_static true */ }

#[test]
fn a_changed_frame_reports_motion_above_threshold() { /* Solid(0) then Solid(255) -> > threshold, is_static false */ }

#[test]
fn first_update_is_large_so_the_first_tick_never_skips() { /* update once -> f32::MAX, is_static false */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision motion::`
Expected: FAIL.

- [ ] **Step 3: Implement `MotionGate`**

Mean absolute difference over all 3072 bytes. The first call returns `f32::MAX` so the first tick always runs inference — a cold start must never be treated as a static scene.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision motion::`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/motion.rs
git commit -m "feat(vision): motion gate for static-scene short-circuit"
```

---

## Task 4: Non-maximum suppression

**Files:**
- Create: `crates/lockin-vision/src/analyze/nms.rs`, `crates/lockin-vision/src/analyze/mod.rs`
- Test: `crates/lockin-vision/src/analyze/nms.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Rect { x: f32, y: f32, w: f32, h: f32 }` with `iou(&self, other: &Rect) -> f32` and `center(&self) -> (f32, f32)`; `Detection { rect: Rect, score: f32, class_id: u32 }`; `nms(dets: &mut Vec<Detection>, iou_threshold: f32, score_threshold: f32)`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn identical_rects_have_iou_one() { }

#[test]
fn disjoint_rects_have_iou_zero() { }

#[test]
fn nms_keeps_the_higher_scoring_of_two_overlapping_boxes() { }

#[test]
fn nms_keeps_boxes_belonging_to_different_classes() { /* two identical rects, class 0 and 67 -> both survive */ }

#[test]
fn nms_drops_detections_below_score_threshold() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::nms`
Expected: FAIL.

- [ ] **Step 3: Implement NMS**

Sort descending by score, greedily keep, suppress same-class overlaps above `iou_threshold`. Class-aware suppression is required: a phone box frequently overlaps its holder's person box, and class-agnostic NMS would delete one of them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::nms`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/
git commit -m "feat(vision): class-aware non-maximum suppression"
```

---

## Task 5: YuNet face detection (stage 1)

**Files:**
- Create: `crates/lockin-vision/src/models/mod.rs`, `crates/lockin-vision/src/models/yunet.rs`, `crates/lockin-vision/src/capture/camera.rs`
- Test: `crates/lockin-vision/src/models/yunet.rs`

**Interfaces:**
- Consumes: `Frame` (Task 1), `Detection`/`Rect`/`nms` (Task 4).
- Produces: `trait InferenceSession { fn run(&self, inputs: &[InputTensor]) -> Result<Vec<OutputTensor>, ModelError>; fn name(&self) -> &str; }`; `YuNet::new(session: Box<dyn InferenceSession>)`; `YuNet::detect(&self, frame: &Frame) -> Result<Vec<FaceBox>, ModelError>` where `FaceBox { rect: Rect, score: f32, landmarks: [(f32, f32); 5] }`; `CameraSource::open(index: u32) -> Result<CameraSource, CaptureError>`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn preprocessing_scales_to_unit_range_and_preserves_aspect() { /* known 4x2 frame -> letterboxed tensor, padding 0 */ }

#[test]
fn decode_maps_anchor_grid_to_frame_coordinates() { /* synthetic tensor -> rects inside frame bounds */ }

#[test]
fn detect_returns_empty_for_a_blank_frame_when_session_says_so() { /* FakeSession returns zeros -> no faces */ }

#[test]
fn camera_open_failure_is_not_fatal() { /* CameraSource::open(9999) -> Err, no panic */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::yunet`
Expected: FAIL.

- [ ] **Step 3: Implement the session trait, YuNet preprocessing, and decode**

`InferenceSession` is the seam that makes every model wrapper testable without ONNX files or a GPU. `InputTensor { name: String, shape: Vec<i64>, data: TensorData }` where `TensorData` is `F32(Vec<f32>)` or `I32(Vec<i32>)`.

YuNet **preprocessing** (this model's own regime, not shared): 320×240 input, aspect preserved with padding, BGR passed through, values scaled by `1/255`, NCHW. **Decode**: YuNet is anchor-free with strides 8/16/32, one score plus four box deltas plus five landmark pairs per cell; map back through the padding and scale, then `nms(.., 0.3, 0.6)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::yunet`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/ crates/lockin-vision/src/capture/camera.rs
git commit -m "feat(vision): YuNet face detection with testable session seam"
```

---

## Task 6: Face Mesh landmarks (stage 2)

**Files:**
- Create: `crates/lockin-vision/src/models/facemesh.rs`
- Test: `crates/lockin-vision/src/models/facemesh.rs`

**Interfaces:**
- Consumes: `InferenceSession`, `FaceBox`, `Rect` (Tasks 4, 5).
- Produces: `FaceMesh::new(session: Box<dyn InferenceSession>)`; `FaceMesh::landmarks(&self, frame: &Frame, face: &FaceBox) -> Result<Landmarks, ModelError>` where `Landmarks { points: [[i32; 3]; 468], score: f32 }`; `pub const CROP_MARGIN: f32 = 0.25`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn crop_rect_applies_25_percent_margin_and_clamps_to_frame() { /* face at edge -> crop stays inside bounds */ }

#[test]
fn crop_rect_is_never_narrower_than_the_face_box() { /* model card: crop must not be too narrow */ }

#[test]
fn preprocessing_is_rgb_unit_range_and_ignores_aspect_ratio() { /* non-square crop -> non-uniform scale */ }

#[test]
fn landmarks_come_back_as_absolute_pixels_not_normalised() { /* int32 in [0, width] */ }

#[test]
fn all_four_crop_inputs_are_passed_to_the_session() { /* FakeSession asserts 5 inputs: input + 4 crop int32 */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::facemesh`
Expected: FAIL.

- [ ] **Step 3: Implement the crop, preprocessing, and decode**

The graph takes **five** inputs: `input` `[1,3,192,192]` f32 plus `crop_x1`/`crop_y1`/`crop_width`/`crop_height` as `[1,1]` int32. Preprocessing is RGB (reverse of BGR), resized to 192×192 **ignoring aspect ratio**, `/255` into `[0,1]`, NCHW. The `_post` graph already fuses the decode, so `final_landmarks` arrives as int32 **absolute pixels in the source frame** — do not un-crop manually. Outputs are `score` `[1,1]` and `final_landmarks` `[1,468,3]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::facemesh`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/facemesh.rs
git commit -m "feat(vision): Face Mesh landmarks with crop-rect inputs"
```

---

## Task 7: Camera intrinsics

**Files:**
- Create: `crates/lockin-vision/src/analyze/intrinsics.rs`
- Test: `crates/lockin-vision/src/analyze/intrinsics.rs`

**Interfaces:**
- Consumes: `Frame` (Task 1).
- Produces: `CameraIntrinsics { fx: f32, fy: f32, cx: f32, cy: f32 }` with `CameraIntrinsics::from_frame(frame: &Frame) -> CameraIntrinsics` and `CameraIntrinsics::from_calibration(fx: f32, fy: f32, cx: f32, cy: f32) -> CameraIntrinsics`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn principal_point_is_the_frame_centre_not_a_swapped_axis() {
    let f = Frame::new(640, 480, vec![0; 640*480*4], Instant::now()).unwrap();
    let k = CameraIntrinsics::from_frame(&f);
    assert_eq!((k.cx, k.cy), (320.0, 240.0)); // the prior art swaps these
}

#[test]
fn default_focal_length_is_the_wider_dimension() { }

#[test]
fn from_calibration_overrides_the_estimate() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::intrinsics`
Expected: FAIL.

- [ ] **Step 3: Implement intrinsics**

Default estimate: `fx = fy = max(width, height)`, principal point at the true centre. This is a guess by construction, which is exactly what R15 warns about — the `probe` binary will report the estimate so it can be replaced with a real measurement, and `from_calibration` is the path that replaces it. `cx` is the horizontal centre and `cy` the vertical centre; the prior art's `distraction-system` confuses these and the test pins the correct assignment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::intrinsics`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/intrinsics.rs
git commit -m "feat(vision): camera intrinsics with a pinned principal point"
```

---

## Task 8: Head pose via solvePnP

**Files:**
- Create: `crates/lockin-vision/src/analyze/pose.rs`
- Test: `crates/lockin-vision/src/analyze/pose.rs`

**Interfaces:**
- Consumes: `Landmarks` (Task 6), `CameraIntrinsics` (Task 7).
- Produces: `Euler { yaw: f32, pitch: f32, roll: f32 }` (degrees); `HeadPose { euler: Euler, valid: bool }`; `estimate_head_pose(points: &[[i32; 3]; 468], k: &CameraIntrinsics) -> HeadPose`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn frontal_landmarks_produce_near_zero_angles() { /* canonical 3D points projected at centre -> |yaw|,|pitch| < 2.0 */ }

#[test]
fn head_pose_is_bounded_and_never_nan() {
    // degenerate: all 468 points at the same pixel, and at extreme offsets
    let p = [[0, 0, 0]; 468];
    let hp = estimate_head_pose(&p, &k(640, 480));
    assert!(hp.euler.yaw.is_finite() && hp.euler.pitch.is_finite() && hp.euler.roll.is_finite());
    assert!(hp.euler.yaw.abs() <= 90.0 && hp.euler.pitch.abs() <= 90.0);
    assert!(hp.valid == false || hp.euler.yaw.is_finite());
}

#[test]
fn a_yawed_head_reports_a_larger_yaw_than_a_frontal_one() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::pose`
Expected: FAIL.

- [ ] **Step 3: Implement pose estimation**

Six correspondence points, the canonical Face Mesh indices `33, 263, 1, 61, 291, 199`, against a generic 3D face model. Use `nalgebra` for the linear algebra: a DLT/EPnP initialisation followed by Gauss-Newton refinement over the six 2D–3D pairs, then convert the rotation matrix to Euler angles.

Guard rails, which the tests above pin: return `valid = false` and a zeroed `Euler` when the solve fails to converge or when the reprojection error exceeds a threshold. **Never** return NaN. Clamp `yaw` and `pitch` to ±90. `valid = false` is what Review Focus item 3 relies on — the state machine treats an invalid pose as no evidence rather than as "looking away".

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::pose`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/pose.rs
git commit -m "feat(vision): head pose via solvePnP with bounded output"
```

---

## Task 9: Eye aspect ratio and eye box

**Files:**
- Create: `crates/lockin-vision/src/analyze/eyes.rs`
- Test: `crates/lockin-vision/src/analyze/eyes.rs`

**Interfaces:**
- Consumes: `Landmarks` (Task 6), `Rect` (Task 4).
- Produces: `ear(points: &[[i32; 3]; 468]) -> f32`; `eye_box(points: &[[i32; 3]; 468]) -> Option<Rect>`; `pub const EYE_BOX_SCALE: f32 = 1.8;`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn ear_matches_the_formula_for_hand_built_landmarks() { /* known open-eye points -> expected ratio */ }

#[test]
fn closed_eye_has_lower_ear_than_open_eye() { }

#[test]
fn eye_box_is_square_and_scaled_1_8x_inter_landmark_distance() { }

#[test]
fn eye_box_is_none_when_the_landmark_index_is_missing() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::eyes`
Expected: FAIL.

- [ ] **Step 3: Implement EAR and the eye box**

EAR uses six points per eye — left `33, 160, 158, 133, 153, 144`, right `362, 385, 387, 263, 373, 380` — as `(‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖)`.

`eye_box` returns a **square** box `1.8 ×` the inter-landmark eye distance, centred between the two eyes' outer corners, roll-corrected to the eye line. That geometry is what OpenVINO's own gaze demo uses and it is what makes the eye classifier's domain match hold.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::eyes`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/eyes.rs
git commit -m "feat(vision): eye aspect ratio and roll-corrected eye box"
```

---

## Task 10: Eye-state classifier (stage 2b)

**Files:**
- Create: `crates/lockin-vision/src/models/eyestate.rs`
- Test: `crates/lockin-vision/src/models/eyestate.rs`

**Interfaces:**
- Consumes: `InferenceSession`, `Frame`, `Rect` (Tasks 1, 4, 5).
- Produces: `EyeStateClassifier::new(session: Box<dyn InferenceSession>)`; `EyeStateClassifier::classify(&self, frame: &Frame, eye_box: &Rect) -> Result<bool, ModelError>` returning `true` when **closed**.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn index_zero_is_closed_and_index_one_is_open() {
    // OpenVINO's README says [open, closed]; its converter and demo say 0 = closed.
    // Pin the CODE's convention so an upstream swap cannot silently invert Drowsy.
    assert_eq!(classify_from_probs([0.9, 0.1]), true);   // closed
    assert_eq!(classify_from_probs([0.1, 0.9]), false);  // open
}

#[test]
fn preprocessing_is_bgr_and_centred_on_127() { /* (x-127)/255, NOT /255 alone */ }

#[test]
fn normalisation_is_mandatory_raw_input_would_nan() {
    // assert the preprocessed tensor range is approximately [-0.498, 0.502]
}

#[test]
fn classify_crops_exactly_the_eye_box_at_32x32() { /* FakeSession asserts input shape [1,3,32,32] */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::eyestate`
Expected: FAIL.

- [ ] **Step 3: Implement the classifier**

This model's own regime: the eye box resized to 32×32, **BGR**, `(x − 127) / 255`. Input name `input.1`, output `19` shaped `[1,2,1,1]` softmax. Raw 0–255 input produces NaN, so normalisation is asserted by test, not assumed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::eyestate`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/eyestate.rs
git commit -m "feat(vision): eye-state classifier with pinned class order"
```

---

## Task 11: YOLOX-Nano person and phone detection (stage 3)

**Files:**
- Create: `crates/lockin-vision/src/models/yolox.rs`
- Test: `crates/lockin-vision/src/models/yolox.rs`

**Interfaces:**
- Consumes: `InferenceSession`, `Frame`, `Detection`/`nms` (Tasks 1, 4, 5).
- Produces: `YoloxNano::new(session: Box<dyn InferenceSession>)`; `YoloxNano::detect(&self, frame: &Frame) -> Result<Vec<Detection>, ModelError>`; `pub const PERSON_CLASS: u32 = 0;` `pub const CELL_PHONE_CLASS: u32 = 67;`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn preprocessing_letterboxes_to_416_with_pad_114_and_keeps_bgr_0_255() {
    // assert padded pixels equal 114 and the tensor is not divided by 255
}

#[test]
fn decode_applies_grid_offset_and_exp_scaling() { /* known raw tensor -> known boxes */ }

#[test]
fn decode_never_produces_nan_for_large_raw_values() { /* exp overflow guarded */ }

#[test]
fn detect_keeps_only_person_and_cell_phone_classes() { /* other 78 classes dropped */ }

#[test]
fn boxes_are_rescaled_out_of_the_letterbox_back_into_frame_coordinates() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::yolox`
Expected: FAIL.

- [ ] **Step 3: Implement preprocessing, decode, and NMS**

Regime: **BGR**, `0–255` with **no** division, letterboxed to a 32-multiple at 416×416 with pad value **114**, mean `[103.53, 116.28, 123.675]` and std `[57.375, 57.12, 58.395]` subtracted.

Decode: output is `[1, 3549, 85]` and fully raw, with **no sigmoid in the graph**. Per row: `xy = (raw[0..2] + grid) * stride`, `wh = exp(raw[2..4]) * stride`, `score = sigmoid(obj) * sigmoid(cls)`, then class-agnostic NMS at `nms_thr = 0.45`, `score_thr = 0.1`. Guard `exp` against overflow, since a garbage tensor must not produce infinity. Finally invert the letterbox to frame coordinates.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::yolox`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/yolox.rs
git commit -m "feat(vision): YOLOX-Nano detection with raw decode"
```

---

## Task 12: Phone posture classification

**Files:**
- Create: `crates/lockin-vision/src/analyze/phone.rs`
- Test: `crates/lockin-vision/src/analyze/phone.rs`

**Interfaces:**
- Consumes: `Detection`/`Rect` (Task 4), `Landmarks` (Task 6), `PERSON_CLASS`/`CELL_PHONE_CLASS` (Task 11).
- Produces: `PhoneSignal { None, Present, InHand, AtHead }`; `classify_phone(persons: &[Detection], phones: &[Detection], primary_face: Option<&FaceBox>) -> PhoneSignal`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_phone_on_the_desk_far_from_any_person_is_present_not_in_use() { /* -> Present */ }

#[test]
fn a_phone_inside_a_person_box_below_the_head_is_in_hand() { /* -> InHand */ }

#[test]
fn a_phone_inside_the_head_region_is_at_head() { /* -> AtHead */ }

#[test]
fn no_phone_detected_is_none() { }

#[test]
fn a_phone_on_the_desk_never_returns_in_hand() {
    // Review-adjacent: the single most valuable distinction in the design.
    // A phone lying flat within the person's box footprint but outside the head
    // region must not read as use.
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::phone`
Expected: FAIL.

- [ ] **Step 3: Implement classification**

For each person box, associate phone boxes whose **centre** falls inside it. Then classify against the head region derived from the primary face's landmarks: centre inside the head region → `AtHead`; centre inside the person box below the head region → `InHand`; otherwise → `Present`.

`Present` deliberately never escalates to phone use — that distinction is the most valuable idea taken from the prior art and the test above pins it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::phone`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/phone.rs
git commit -m "feat(vision): phone posture classification"
```

---

## Task 13: Glasses detection (calibration only)

**Files:**
- Create: `crates/lockin-vision/src/models/glasses.rs`
- Test: `crates/lockin-vision/src/models/glasses.rs`

**Interfaces:**
- Consumes: `InferenceSession`, `Frame`, `FaceBox` (Tasks 1, 5).
- Produces: `GlassesDetector::new(session: Box<dyn InferenceSession>)`; `GlassesDetector::detect(&self, frame: &Frame, face: &FaceBox) -> Result<bool, ModelError>`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn preprocessing_is_rgb_with_imagenet_mean_and_std() { /* 0.485/0.229 per channel */ }

#[test]
fn positive_logit_means_glasses_present() { }

#[test]
fn face_crop_is_resized_to_256_square() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::glasses`
Expected: FAIL.

- [ ] **Step 3: Implement the detector**

Its own regime — the fourth: face crop resized to **256×256**, **RGB**, ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Input `images` `[1,3,256,256]`, output `logits` `[1,1]`; positive means glasses present. Use the `anyglasses` head, not the sunglasses one — transparent glasses are the case that breaks EAR.

This runs **only during calibration**, never per tick.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::glasses`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/glasses.rs
git commit -m "feat(vision): glasses detector for calibration"
```

---

## Task 14: Primary-face selection

**Files:**
- Create: `crates/lockin-vision/src/analyze/primary.rs`
- Test: `crates/lockin-vision/src/analyze/primary.rs`

**Interfaces:**
- Consumes: `Rect` (Task 4), `FaceBox` (Task 5).
- Produces: `PrimarySelector::new()`; `PrimarySelector::select(&mut self, faces: &[FaceBox]) -> Option<usize>`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn largest_face_wins_when_sizes_differ_clearly() { }

#[test]
fn nearer_to_centre_wins_when_sizes_are_equal() { }

#[test]
fn primary_face_selection_is_stable_across_swaps() {
    // Review Focus 4. Two similar boxes trading places frame to frame must not
    // cause the selection to oscillate; hysteresis keeps the incumbent unless
    // a challenger is clearly larger.
    let mut s = PrimarySelector::new();
    let a = face(100.0, 100.0, 90.0);   // incumbent
    let b = face(300.0, 100.0, 92.0);   // 2% larger, must NOT steal
    assert_eq!(s.select(&[a.clone(), b.clone()]), Some(0));
    assert_eq!(s.select(&[b.clone(), a.clone()]), Some(0)); // index of a, still the incumbent
    let big = face(300.0, 100.0, 200.0); // 2x larger, must steal
    assert_eq!(s.select(&[a, big]), Some(1));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision analyze::primary`
Expected: FAIL.

- [ ] **Step 3: Implement sticky selection**

Rank by area, break ties by proximity to frame centre, but **keep the incumbent** unless a challenger's area exceeds it by a hysteresis factor of 1.25. Without this, two people of similar size cause the primary face — and therefore the head pose feeding the state machine — to oscillate, which shows up downstream as beeping.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision analyze::primary`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/analyze/primary.rs
git commit -m "feat(vision): sticky primary-face selection"
```

---

## Task 15: Fusion into `Observation`

**Files:**
- Create: `crates/lockin-vision/src/fuse.rs`
- Modify: `crates/lockin-vision/src/lib.rs`
- Test: `crates/lockin-vision/src/fuse.rs`

**Interfaces:**
- Consumes: everything from Tasks 4–14.
- Produces: `FaceSignals { head: HeadPose, ear: f32, eye_closed: Option<bool> }`; `PhoneSignal` (Task 12); `Confidence { overall: f32, face: f32, phone: f32, eye: f32 }`; `Observation { tick_ms: u64, faces: u8, primary: Option<FaceSignals>, phone: PhoneSignal, motion: f32, confidence: Confidence }`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn confidence_is_the_minimum_of_the_signal_confidences() { }

#[test]
fn an_invalid_head_pose_yields_no_primary_signals() { /* valid=false -> primary None */ }

#[test]
fn observation_is_serialisable_and_round_trips() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision fuse::`
Expected: FAIL.

- [ ] **Step 3: Implement `Observation` and its assembly**

`overall` confidence is the **minimum** of the contributing signals, not the mean: one untrustworthy signal should not be averaged away by two good ones, because the state machine's confidence floor reads `overall`. An invalid head pose produces `primary: None` rather than a zeroed `Euler`, so downstream code cannot mistake "unknown" for "looking straight ahead".

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision fuse::`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/fuse.rs crates/lockin-vision/src/lib.rs
git commit -m "feat(vision): fusion into Observation"
```

---

## Task 16: The temporal state machine

**Files:**
- Create: `crates/lockin-vision/src/state.rs`, `crates/lockin-vision/src/thresholds.rs`
- Test: `crates/lockin-vision/src/state.rs`

**Interfaces:**
- Consumes: `Observation`, `FaceSignals` (Task 15), `CalibrationProfile` (Task 18 — this task defines `Thresholds` and takes `&Thresholds`, so it has no build dependency on Task 18).
- Produces: `FocusState` (8 variants), `FocusEvent` (10 variants), `DistractionReason`, `Thresholds` (the spec's starting values), `StateMachine::new(thresholds: Thresholds)`, `StateMachine::update(&mut self, obs: &Observation, now: Instant) -> (FocusState, SmallVec<[FocusEvent; 2]>)`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_brief_glance_never_becomes_a_distraction() { /* 0.5 s deviation -> GlancingAway, no event */ }

#[test]
fn a_sustained_deviation_emits_distraction_started_once() { /* 3 s -> exactly one event */ }

#[test]
fn distraction_ended_reports_the_correct_duration() { }

#[test]
fn exit_dwell_is_shorter_than_entry_dwell() { /* asymmetry prevents flapping */ }

#[test]
fn phone_in_use_outranks_distracted_when_both_apply() { /* fixed priority */ }

#[test]
fn phone_in_use_is_latched_for_its_minimum_duration() { /* brief occlusion -> no PhoneUseEnded */ }

#[test]
fn low_confidence_ticks_carry_forward() {
    // Review Focus 5. Below the floor, time accrues to the last stable state and
    // no event fires; a dark frame must not invent a distraction.
    let mut sm = StateMachine::new(Thresholds::default());
    /* settle into Focused */
    let (s, ev) = sm.update(&dark_low_confidence_obs(), t);
    assert_eq!(s, FocusState::Focused);
    assert!(ev.is_empty());
}

#[test]
fn away_time_does_not_accrue_distraction() {
    // Review Focus 2. No face for 10 minutes -> AwayFromDesk, zero distraction events.
}

#[test]
fn no_faces_at_all_is_away_from_desk_but_losing_only_the_primary_is_other_present() { }

#[test]
fn drowsy_requires_ear_and_classifier_to_agree() { /* EAR alone -> no Drowsy */ }

#[test]
fn identical_observation_sequences_produce_identical_event_sequences() { /* determinism */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision state::`
Expected: FAIL.

- [ ] **Step 3: Implement `Thresholds` with the spec's starting values**

One struct, all values from spec §7.2: `glance_entry = 0.0 s` (immediate), `distracted_entry = 2.0 s`, `distracted_exit = 0.6 s`, `phone_entry = 2.0 s`, `phone_latch = 3.0 s`, `phone_exit = 0.8 s`, `drowsy_entry = 1.0 s`, `drowsy_exit = 0.5 s`, `away_entry = 3.0 s`, `away_exit = 0.5 s`, `other_present_entry = 5.0 s`, `other_present_exit = 3.0 s`, `confidence_floor = 0.6`. All are `Duration` except `confidence_floor`.

- [ ] **Step 4: Implement the state machine**

Derive a candidate state from the observation, then apply dwell and hysteresis before committing. The order of operations that the tests pin:

1. If `confidence.overall < confidence_floor`, return the current state, accrue time to it, emit no events. **Return early** — this is the carry-forward rule.
2. Otherwise rank the applicable candidate states by the fixed priority `PhoneInUse › Drowsy › Distracted › OtherPresent › GlancingAway › AwayFromDesk › Focused › Starting`.
3. A candidate must be held for its entry dwell before it becomes current. Leaving a state needs only its exit dwell.
4. `PhoneInUse` additionally holds for `phone_latch` before it may be left.
5. `Drowsy` requires `ear` below the profile threshold **and** `eye_closed == Some(true)`. Either alone is not enough.
6. Any state with **zero** faces is `AwayFromDesk`; losing only the primary face while others remain is `OtherPresent`.
7. Emit an event only on a committed transition that satisfied its dwell. `GlancingAway` never emits an event.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p lockin-vision state::`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add crates/lockin-vision/src/state.rs crates/lockin-vision/src/thresholds.rs
git commit -m "feat(vision): temporal state machine with dwell and hysteresis"
```

---

## Task 17: Scheduler and session state

**Files:**
- Create: `crates/lockin-vision/src/schedule.rs`
- Test: `crates/lockin-vision/src/schedule.rs`

**Interfaces:**
- Consumes: `FocusState` (Task 16).
- Produces: `trait SessionState { fn is_idle(&self) -> bool; fn is_locked(&self) -> bool; }`; `Scheduler::new()`; `Scheduler::tick_interval(&mut self, state: FocusState, now: Instant) -> Option<Duration>` where `None` means sleep; `WindowsSessionState` implementing the trait via `GetLastInputInfo` and WTS notifications.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_present_face_ticks_at_the_base_rate() { /* -> Some(200 ms) */ }

#[test]
fn ten_seconds_without_a_face_drops_to_one_hertz() { /* -> Some(1000 ms) */ }

#[test]
fn sixty_seconds_without_a_face_sleeps_the_engine() { /* -> None */ }

#[test]
fn a_locked_session_sleeps_immediately_even_with_a_face() { }

#[test]
fn returning_from_sleep_resumes_at_the_base_rate() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision schedule::`
Expected: FAIL.

- [ ] **Step 3: Implement the scheduler**

Base 200 ms (5 Hz) with a face present; 1 Hz after 10 s without; `None` after 60 s without, or immediately when `is_locked()`. `SessionState` is a trait, so replay passes a no-op implementation and the engine never touches Win32 during tests. `WindowsSessionState` is the only Win32 code in the crate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision schedule::`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/schedule.rs
git commit -m "feat(vision): adaptive scheduler behind a testable session-state trait"
```

---

## Task 18: Calibration

**Files:**
- Create: `crates/lockin-vision/src/calibration.rs`
- Test: `crates/lockin-vision/src/calibration.rs`

**Interfaces:**
- Consumes: `HeadPose`, `ear` (Tasks 8, 9), `GlassesDetector` (Task 13), `Thresholds` (Task 16).
- Produces: `CalibrationProfile { neutral_euler: Euler, ear_open: f32, yaw_bound: f32, pitch_bound: f32, wears_glasses: bool, confidence_floor: f32 }`; `Calibrator::new()`, `Calibrator::observe(&mut self, obs: &Observation)`, `Calibrator::finish(self) -> Result<CalibrationProfile, CalibrationError>`; `CalibrationProfile::thresholds(&self) -> Thresholds`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn rejects_a_session_with_too_few_valid_frames() { /* < 30 frames -> Err(TooFewFrames) */ }

#[test]
fn rejects_a_session_where_the_head_moved_too_much() { /* stddev > bound -> Err(MovedTooMuch) */ }

#[test]
fn an_accepted_session_averages_the_neutral_pose() { }

#[test]
fn ear_open_is_the_mean_of_the_observed_open_ear() { }

#[test]
fn bounds_are_derived_from_the_neutral_pose_not_hardcoded() { /* profile-relative, spec §8 */ }

#[test]
fn wearing_glasses_raises_the_drowsy_ear_margin() { /* flag changes thresholds, it is not decoration */ }

#[test]
fn profile_round_trips_through_json() { }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision calibration::`
Expected: FAIL.

- [ ] **Step 3: Implement the calibrator**

Accumulate ~60 s of observations. **Reject bad sessions rather than storing them** — that is R7's mitigation, and a poor profile silently degrades every downstream threshold. Reject on fewer than 30 valid frames or on excessive head movement, and surface the reason so the UI can ask the user to retry.

`wears_glasses` must change `ear_open`'s margin; a stored flag that nothing reads is worse than no flag.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision calibration::`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/calibration.rs
git commit -m "feat(vision): calibration profile with session validation"
```

---

## Task 19: Model loading and execution-provider autotune

**Files:**
- Create: `crates/lockin-vision/src/models/ort_session.rs`, `crates/lockin-vision/src/models/provider.rs`
- Test: `crates/lockin-vision/src/models/provider.rs`

**Interfaces:**
- Consumes: `InferenceSession` (Task 5).
- Produces: `OrtSession::from_file(path: &Path, providers: &[Provider]) -> Result<OrtSession, ModelError>`; `Provider { Cpu, DirectMl }`; `RuntimeProfile { per_model: HashMap<ModelId, Provider> }`; `autotune(candidates: &[Provider], sample: &[Frame], reference: &dyn Fn(Provider, &Frame) -> Vec<f32>) -> RuntimeProfile`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_provider_whose_output_disagrees_with_the_cpu_reference_is_discarded() { /* fake: DirectML returns garbage -> CPU chosen */ }

#[test]
fn selection_is_per_model_not_global() { /* fake: A faster on GPU, B faster on CPU -> mixed profile */ }

#[test]
fn a_provider_that_panics_during_verification_is_discarded_not_propagated() { /* catch -> CPU */ }

#[test]
fn a_slower_correct_provider_loses_to_a_faster_correct_one() { }

#[test]
fn cpu_is_always_a_candidate_even_with_no_gpu_present() { /* empty candidate list -> CPU */ }

#[test]
fn provider_dll_load_failure_degrades_to_cpu_without_panicking() { /* nonexistent path -> CPU session or clean Err */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision models::provider`
Expected: FAIL.

- [ ] **Step 3: Implement autotune behind the session seam**

`autotune` takes a closure rather than real sessions, so the whole selection policy is testable with no model files and no GPU. Verification compares each candidate's outputs against the CPU reference within tolerance; a crashing or disagreeing candidate is discarded. Benchmark over a sample of frames, pick the fastest **per model**, and persist.

`OrtSession::from_file` is the only place that touches `ort`. Build with `load-dynamic` to avoid DLL-resolution failures, and list the chosen provider **and** CPU so ONNX Runtime partitions the graph rather than failing. Catch device-removed errors and rebuild on CPU — a driver reset must cost a few ticks, not the session.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision models::provider`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-vision/src/models/ort_session.rs crates/lockin-vision/src/models/provider.rs
git commit -m "feat(vision): provider autotune with per-model selection"
```

---

## Task 20: The `replay` harness

**Files:**
- Create: `crates/lockin-cli/Cargo.toml`, `crates/lockin-cli/src/main.rs`, `crates/lockin-cli/src/replay.rs`, `crates/lockin-cli/src/report.rs`
- Test: `crates/lockin-cli/tests/replay_determinism.rs`

**Interfaces:**
- Consumes: the whole library.
- Produces: `lockin-replay <fixture-dir> --out <report.json>`; `Report { per_state_ms: BTreeMap<String, u64>, cpu_ms_per_frame: f64, peak_rss_bytes: u64, p95_tick_latency_ms: f64, state_sequence_hash: String, events: Vec<EventRecord> }`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn two_replays_of_the_same_fixture_produce_the_same_hash() {
    // asserts the determinism gate; runs against a synthetic fixture so it
    // needs no models and no camera
}

#[test]
fn the_fixture_fixture_is_never_written_to_disk() { /* snapshot the temp dir before and after */ }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lockin-cli --test replay_determinism`
Expected: FAIL — binary not found.

- [ ] **Step 3: Implement the harness**

Drive the engine from `ReplaySource` with a **virtual clock** derived from frame index and a fixed nominal fps, so timing is reproducible and independent of machine speed. Hash the full `FocusState` sequence with a stable hasher (FNV-1a, not `DefaultHasher`, which is not guaranteed stable across releases). Compute peak RSS from `PeakWorkingSetSize` via `GetProcessMemoryInfo`, and CPU time from `GetProcessTimes`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p lockin-cli --test replay_determinism`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-cli/
git commit -m "feat(cli): replay harness with determinism gate"
```

---

## Task 21: The `probe` binary

**Files:**
- Create: `crates/lockin-cli/src/probe.rs`, `crates/lockin-cli/src/record.rs`, `crates/lockin-cli/src/overlay.rs`
- Test: `crates/lockin-cli/tests/record_roundtrip.rs`

**Interfaces:**
- Consumes: `CameraSource`, `ReplaySource` (Tasks 2, 5), the engine.
- Produces: `lockin-probe [--record <dir>] [--frames <n>]`; `lockin-probe --calibrate`; `RecordWriter::write(&mut self, frame: &Frame) -> Result<(), io::Error>` writing `%04d.png`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn recorded_frames_replay_back_byte_for_byte() {
    // record N synthetic frames, reopen with ReplaySource, compare pixels
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lockin-cli --test record_roundtrip`
Expected: FAIL.

- [ ] **Step 3: Implement the probe**

`--record` is the fixture producer: it writes numbered PNGs, which is why the harness needs no video decoder. The live overlay prints the current `FocusState`, head pose, EAR, face count and the confidence value, and **prints the camera-intrinsics estimate** so R15's guess can be replaced with a real measurement. `--calibrate` walks the ~60 s calibration and writes the profile JSON.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p lockin-cli --test record_roundtrip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/lockin-cli/src/probe.rs crates/lockin-cli/src/record.rs crates/lockin-cli/src/overlay.rs
git commit -m "feat(cli): live probe with overlay and fixture recording"
```

---

## Task 22: Model provenance and fetch

**Files:**
- Create: `models/README.md`, `models/fetch.ps1`, `models/LICENSES/README.md`, `models/hashes.json`
- Create: `crates/lockin-vision/tests/models_present.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `models/fetch.ps1` downloading to `models/` and verifying sha256 against `models/hashes.json`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn every_model_named_in_the_manifest_exists_and_hashes_correctly() {
    // skips with a clear message when models/fetch.ps1 has not been run
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lockin-vision --test models_present`
Expected: FAIL — manifest missing.

- [ ] **Step 3: Write the manifest and fetch script**

`models/README.md` records, per model: source URL, licence, sha256, opset, exact input/output signature, and the export or conversion steps. Five entries: YuNet (MIT), Face Mesh 032 `face_mesh_Nx3x192x192_post.onnx` (Apache-2.0 exporter, weights unverified), YOLOX-Nano (Apache-2.0), open-closed-eye-0001 (Apache-2.0), Glasses-Detector (MIT).

`models/LICENSES/README.md` states the MediaPipe weights gap **explicitly** — Google publishes no model-specific licence for the `.tflite`/`.task` files, and the ONNX conversion inherits that ambiguity. Record it as a known risk rather than claiming Apache-2.0.

`fetch.ps1` downloads and verifies hashes; no model binary is committed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pwsh models/fetch.ps1; cargo test -p lockin-vision --test models_present`
Expected: PASS, 5 models verified.

- [ ] **Step 5: Commit**

```bash
git add models/ crates/lockin-vision/tests/models_present.rs
git commit -m "chore(models): provenance manifest and hash-verified fetch"
```

---

## Task 23: Wire the pipeline end to end

**Files:**
- Create: `crates/lockin-vision/src/engine.rs`
- Modify: `crates/lockin-vision/src/lib.rs`
- Test: `crates/lockin-vision/src/engine.rs`

**Interfaces:**
- Consumes: every prior task.
- Produces: `Engine::new(source: Box<dyn FrameSource>, profile: CalibrationProfile, sessions: ModelSessions) -> Engine`; `Engine::tick(&mut self, now: Instant) -> Tick`; `Tick { state: FocusState, observation: Observation, events: SmallVec<[FocusEvent; 2]>, confidence: Confidence }`.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_static_scene_skips_inference_after_the_first_tick() { /* FakeSessions counts run() calls */ }

#[test]
fn stage_two_does_not_run_when_no_face_is_detected() { /* cropper dependency, R2 */ }

#[test]
fn stage_three_runs_on_its_own_slower_cadence() { /* 5 ticks of stage 1 -> 1 of stage 3 */ }

#[test]
fn the_engine_sleeps_and_resumes_per_the_scheduler() { /* scheduler returns None -> tick is a no-op */ }

#[test]
fn identical_inputs_produce_identical_ticks() { /* determinism at the engine level */ }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p lockin-vision engine::`
Expected: FAIL.

- [ ] **Step 3: Implement `Engine::tick`**

The single orchestration point, in order: pull a frame if the scheduler's interval has elapsed; update the motion gate; short-circuit to the previous observation if static **and** a face was present last tick; run stage 1; run stage 2 only if a face exists; run stage 2b only if an eye box exists; run stage 3 when its cadence is due; select the primary face; classify phone posture; fuse; update the state machine. Collect events for return.

The `ModelSessions` struct holds `Box<dyn InferenceSession>` per model, so every test above runs with fakes and the whole engine is testable without ONNX files, a GPU, or a camera.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p lockin-vision engine::`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the whole suite and record the budget**

Run: `cargo test --workspace && cargo run -p lockin-cli --bin lockin-replay -- fixtures/basic --out report.json`
Expected: all tests PASS; `report.json` shows `cpu_ms_per_frame` consistent with **< 3 % of one core** and `peak_rss_bytes` **< 250 MB**. Record the actual numbers in the commit message. If the budget is missed, that is a finding to report, not a number to quietly relax.

- [ ] **Step 6: Commit**

```bash
git add crates/lockin-vision/src/engine.rs crates/lockin-vision/src/lib.rs
git commit -m "feat(vision): wire the cascaded pipeline end to end"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: §4.1 boundary → 1, 2, 23; §4.2 stages → 3, 5, 6, 10, 11; §4.3 sources → 2, 5; §4.4 providers → 19; §5 signals → 7, 8, 9; §5.1 phone → 12; §5.2 I/O → 6, 11; §5.3 behaviour models → 10, 13; §5.4 head pose → 8; §7 state machine → 16; §7.3 carry-forward → 16; §7.4 events → 16; §8 calibration → 18; §9 power budget → 17; §10 API → 23; §11 harness → 20; §11.2 fixture → 21; §11.3 gates → 20, 22, 23; §12 layout → 1; §13 provenance → 22. No gaps.

**Deliberately deferred, with reasons.** §5.5 and §5.6 are *investigations*, not features — they are closed by the spec and produce no task. Gaze and end-to-end attention models are excluded, so a task for either would contradict the spec.

**Type consistency.** `Rect`/`Detection` (4) are reused by 5, 11, 12, 14. `InferenceSession` (5) is the seam for 6, 10, 11, 13, 19, 23. `Landmarks` (6) feeds 8, 9, 12. `Observation` (15) feeds 16, 18. `Thresholds` (16) is consumed by 18 via `CalibrationProfile::thresholds()`, which keeps 18 from depending on 16's internals. `FocusState` (16) is read by 17 and 20.

**Review Focus coverage.** Item 1 → Task 5. Item 2 → Task 16. Item 3 → Task 8. Item 4 → Task 14. Item 5 → Task 16. Each has a named test in its owning task.

**Known gap, stated rather than hidden.** Tasks 5, 6, 10, 11 and 13 test preprocessing and decode as pure functions but cannot test real model behaviour without the weights as well as real imagery. The `models_present` integration test (Task 22) verifies the artefacts exist and hash correctly; it does not verify accuracy, because the maintainer's decision to drop the evaluation corpus removed the labelled test set. This is R4, accepted deliberately.
