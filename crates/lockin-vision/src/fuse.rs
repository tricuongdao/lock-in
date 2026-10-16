//! Fusion: one tick's stage outputs into one [`Observation`].
//!
//! This is the record every later task reads. The state machine's floor, the
//! calibrator's sampling and the `probe` overlay all take an `Observation`, so
//! the two decisions this module makes — what counts as an unusable reading, and
//! how much of the tick to trust — are the product's decisions wearing fusion's
//! clothes.
//!
//! # `overall` is the minimum, not the mean
//!
//! [`Confidence::overall`] is the **least** of the three signal confidences.
//! A mean would let one untrustworthy signal be averaged away by two healthy
//! ones: a frame whose dense landmarks are not a face — Face Mesh's logit is
//! `-7`, three orders of magnitude below the floor — still has a confident
//! detector box and a phone-free frame, and the mean of `0.93`, `1.0` and
//! `9.1e-4` is `0.64`, which clears Task 16's `0.6` floor. The minimum is
//! `9.1e-4`, which does not. Under D6 the honest answer to "may this tick be
//! acted on" is no, and a mean would say yes.
//!
//! # What each field means, and where its number comes from
//!
//! A field is the confidence of the **claim its signal makes this tick**. A
//! signal that claims nothing cannot mislead, so it is `1.0`; a signal that
//! makes a claim and cannot support it is `0.0`.
//!
//! | field | the claim | the number |
//! |---|---|---|
//! | `face` | "here are the wearer's face signals" | the primary box's [`FaceBox::score`] — a probability YuNet has already put through a `Sigmoid` **inside its graph**, so nothing here activates it a second time (R22) — or `0.0` when a face was detected and no reading could be built from it |
//! | `eye` | "here is the eye reading" | the logistic of Face Mesh's [`Landmarks::score`](crate::models::facemesh::Landmarks::score), which is a **raw logit** |
//! | `phone` | "the wearer is on the phone" | the strongest cell-phone detection's score, when the verdict is `InHand` or `AtHead` |
//!
//! **`Landmarks::score` is not a probability and must not be thresholded as
//! one.** The graph ends in a bare `Gemm` with no `Sigmoid` anywhere, so the
//! number is log-odds; the probability it encodes is its logistic,
//! `1 / (1 + exp(-x))`, which is the exact inverse of the logit function and the
//! only mapping that is. The normalisation is load-bearing rather than
//! decorative, because the floor is compared against it: `σ(0) = 0.5` is *below*
//! Task 16's `0.6` floor, and the floor is reached at a logit of
//! `ln 1.5 ≈ +0.41`. So the gate the `Landmarks::score` docs call natural —
//! `> 0.0` — lands the eye field at `0.5`, and the floor is the stricter of the
//! two by the same sign. A real face measures `+12` to `+18` and lands within an
//! ulp of `1.0`; a wall or a blank image measures `-7` to `-12` and lands below
//! `1e-3`; a crop too narrow to hold the face measures `-2.9` and lands at
//! `0.052`.
//!
//! # The three cases where a field is `1.0` rather than `0.0`
//!
//! Each is a place where zero would look conservative and would in fact break
//! the product, because the floor is read by the whole state machine and the
//! missing evidence is read by one branch of it:
//!
//! 1. **No face at all.** Zero faces claims nothing about the wearer, and
//!    `AwayFromDesk` is entered by *counting* zero-face ticks — after the floor
//!    check, which returns early. A zero here would make the spec's 3 s away
//!    entry unreachable, so `face` is `1.0` when `faces == 0`. A dark frame the
//!    detector cannot see into therefore reads as an empty desk, which is the
//!    honest reading of "no face detected" and the direction D6 accepts: D6
//!    forbids manufacturing a *distraction* out of weak evidence, and an empty
//!    desk is not a distraction.
//! 2. **No eye crop.** `eye_closed` is `None` when stage 2b had no usable box,
//!    and the eye field does not depend on stage 2b's answer: it is the dense
//!    stage's verdict on the landmarks the eye reading is *built from*, so a
//!    missing confirmation is expressed by the missing `Some`, not by a zero in
//!    the confidence. Zeroing it would carry every tick forward and silence
//!    `Focused`, `Distracted` and `PhoneInUse` time for a gap only `Drowsy`
//!    reads — and `Drowsy` needs `eye_closed == Some(true)` anyway, so a `None`
//!    can cost recall but can never manufacture a nap.
//! 3. **No phone, or a phone that is not in use.** `None` and `Present` claim
//!    no use, and only a claim of use can create a state. Suppressing the tick
//!    for a weak phone-class box would lose `Focused` time on every frame with a
//!    phone-shaped object on the desk — a *persistent* false positive is exactly
//!    how the engine would record nothing at all.
//!
//! # An invalid head pose is no signals, never a zeroed `Euler`
//!
//! [`FaceSignals::from_landmarks`] applies [`HeadPose::valid`], and it is the
//! only place `Observation::primary` is built from landmarks, so a rejected pose
//! reaches the record as `primary: None` and `confidence.face == 0.0` together.
//! The two say the same thing twice on purpose: `HeadPose`'s own contract is
//! that `valid == false` means *no evidence* and the state machine carries the
//! previous state forward, and a tick that a rejected pose left below the floor
//! is carried forward instead of being read as `OtherPresent` — "a colleague is
//! in the room" — which is what a `Some` with unknown angles would become.
//!
//! # What this module does not claim
//!
//! - **The phone field cannot see which box decided the verdict.** `InHand` and
//!   `AtHead` come from `classify_phone`, which reports the posture and not the
//!   box behind it, so the field takes the *strongest* cell-phone box in the
//!   frame. That is an upper bound on the score of whichever box produced the
//!   verdict, and it is the right bound in the direction that matters: the field
//!   is weak exactly when every phone box in the frame is weak. It is not exact
//!   when a confident phone — one on the desk — is in frame beside a weak one
//!   that claims use, and closing that needs `classify_phone` to report its
//!   attribution. Re-deriving the geometry here would duplicate Task 12's rule
//!   and let the two drift.
//! - **The eye field is not the classifier's own certainty.**
//!   `EyeStateClassifier::classify` returns a `bool`, and `ear` is a ratio
//!   rather than a score, so the only model verdict about the eye reading's
//!   inputs is Face Mesh's logit. Should a later task expose stage 2b's softmax,
//!   this field should become the minimum of the two.
//! - **Staleness is invisible here.** Stage 3 runs at 1–2 Hz against a 5 Hz
//!   tick, so a caller carries the last phone verdict forward — and with it the
//!   detections it was derived from, which is why [`StageOutputs::phones`] is
//!   "the boxes the verdict came from" and not "this tick's boxes". Nothing in
//!   `Detection` carries a timestamp, so fusion cannot discount an old verdict
//!   and does not pretend to.
//!
//! # Contract
//!
//! No I/O, no clock, no allocation, no randomness: every number in the record
//! comes from an argument, and the same arguments always produce the same
//! record. The work is a handful of comparisons and one exponential — see
//! [`fuse`], which is what the engine calls once per tick.

use crate::analyze::eyes::ear;
use crate::analyze::intrinsics::CameraIntrinsics;
use crate::analyze::nms::Detection;
use crate::analyze::phone::PhoneSignal;
use crate::analyze::pose::{HeadPose, estimate_head_pose};
use crate::models::facemesh::Landmarks;
use crate::models::yolox::CELL_PHONE_CLASS;
use crate::models::yunet::FaceBox;
use serde::{Deserialize, Serialize};

/// How many faces [`Observation::faces`] can report.
///
/// The count saturates rather than wrapping at it: a frame with 256 faces is not
/// a frame a webcam produces, but a wrapping `u8` would report **zero** — and
/// zero faces is `AwayFromDesk` to the state machine. A saturated count is wrong
/// by a few faces; a wrapped one is wrong about whether the wearer is at the
/// desk.
const MOST_FACES: u8 = u8::MAX;

/// The confidence of a signal that makes no claim this tick.
///
/// Named rather than written as `1.0` at each site, because the reasoning is the
/// same at all three and it is not "we are sure of the reading": it is that
/// there is no reading to be wrong about. See the module docs for the three
/// cases and for what zero would cost at each.
const NO_CLAIM: f32 = 1.0;

/// The confidence of a signal whose claim could not be supported at all.
///
/// The other end of the range from [`NO_CLAIM`], and the one that puts a tick
/// below Task 16's floor: a face was detected and no reading could be built from
/// it, or a verdict claims phone use with no phone box behind it.
const NO_EVIDENCE: f32 = 0.0;

/// The landmark-derived signals of one face: the pose, the eye ratio and the
/// classifier's verdict, together.
///
/// The head pose travels as a [`HeadPose`] rather than as a bare
/// [`Euler`](crate::analyze::pose::Euler) so that `valid` cannot be dropped on
/// the way here; this module's rule is that a `FaceSignals` is only ever built
/// from a pose that solved, so `head.valid` is `true` in every value
/// [`FaceSignals::from_landmarks`] returns. The fields are public because an
/// `Observation` is a record — the `probe` overlay reads it, and Task 16's
/// fixtures build it by hand — and the constructor is the path the pipeline
/// takes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaceSignals {
    /// Which way the face is pointing, in degrees. `valid` is `true` in every
    /// value this module produces; see [`FaceSignals::from_landmarks`].
    #[serde(with = "head_pose")]
    pub head: HeadPose,
    /// The eye aspect ratio: the mean of the two eyes' lid separations over
    /// their widths, `0.0` for an eye whose corners coincide.
    pub ear: f32,
    /// Stage 2b's verdict — `true` when the classifier saw closed eyes — or
    /// `None` when no eye box was usable and the classifier did not run.
    pub eye_closed: Option<bool>,
}

/// How far each signal of one tick can be trusted, and the least of them.
///
/// See the module docs for what each field means and where its number comes
/// from. Every field is finite and within `0.0..=1.0` whatever the models
/// return, which is what lets a consumer compare `overall` against a floor
/// without defending itself against a NaN — and a NaN left in a field would be
/// *dropped* by a minimum rather than propagated, so the guard is load-bearing
/// rather than decorative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence {
    /// The least of the three signal confidences. **Not the mean**: see the
    /// module docs.
    pub overall: f32,
    /// The confidence of the face reading.
    pub face: f32,
    /// The confidence of the phone verdict.
    pub phone: f32,
    /// The confidence of the eye reading.
    pub eye: f32,
}

/// One tick's diagnostic record: what the pipeline saw, and how much of it to
/// believe.
///
/// This is the type every later task reads — the state machine's input, the
/// calibrator's sample, the overlay's contents — and it is a *record*, not a
/// decision. `FocusState` and `FocusEvent` are the engine's contract; this is
/// what it saw on the way there.
///
/// `Copy` because the motion gate re-emits the previous observation on a static
/// scene, and no field of it allocates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Milliseconds from the engine's own origin, as the caller counted them.
    /// Fusion reads no clock, so this is a carried number and not a measurement
    /// made here.
    pub tick_ms: u64,
    /// How many faces the detector found this frame. The count saturates at
    /// `u8::MAX` rather than wrapping. Zero is the state machine's
    /// `AwayFromDesk` count.
    pub faces: u8,
    /// The primary face's landmark-derived signals, or `None` when there is no
    /// primary face **or** the one that was chosen produced no usable reading —
    /// an invalid head pose above all. See [`FaceSignals::from_landmarks`].
    pub primary: Option<FaceSignals>,
    /// The phone posture Task 12 classified for this tick.
    #[serde(with = "phone_signal")]
    pub phone: PhoneSignal,
    /// Stage 0's motion, in grey levels (`0.0..=255.0`), or `f32::MAX` for a
    /// gate that has not yet seen a previous frame. Carried verbatim: it is a
    /// measurement, not a confidence, and fusion does not sanitise it.
    pub motion: f32,
    /// How much of the above to believe.
    pub confidence: Confidence,
}

/// Everything one tick's stages produced, as the pipeline hands it to [`fuse`].
///
/// Borrowed throughout: fusion allocates nothing, and every slice here is
/// borrowed from a buffer the caller already holds.
#[derive(Debug, Clone, Copy)]
pub struct StageOutputs<'a> {
    /// Milliseconds from the engine's origin. Carried into
    /// [`Observation::tick_ms`] unchanged.
    pub tick_ms: u64,
    /// Stage 1's faces for this frame, in YuNet's order — strongest first. Only
    /// `len()` and `primary` are read: the signals come from the primary face
    /// alone, and the count is the frame's, not a per-face summary.
    pub faces: &'a [FaceBox],
    /// The primary face's box, from [`PrimarySelector::select`](crate::analyze::primary::PrimarySelector::select).
    /// `None` when there was no face to choose; no other field is read from it
    /// than its score.
    pub primary: Option<&'a FaceBox>,
    /// Stage 2's landmarks for the primary face, or `None` when stage 2 did not
    /// run or failed. A detected face with no landmarks is no reading at all.
    pub landmarks: Option<&'a Landmarks>,
    /// Stage 2b's verdict, or `None` when no eye box was usable.
    pub eye_closed: Option<bool>,
    /// The camera's intrinsics, which the pose solve needs to tell a rotation of
    /// the head from a translation of it.
    pub intrinsics: &'a CameraIntrinsics,
    /// The **cell-phone** boxes the phone verdict was derived from — stage 3's
    /// output on the tick the verdict was produced, which on a tick where the
    /// slower stage did not run is the previous result carried forward with the
    /// verdict.
    pub phones: &'a [Detection],
    /// Stage 3's posture verdict for those boxes.
    pub phone: PhoneSignal,
    /// Stage 0's motion, in grey levels.
    pub motion: f32,
}

impl FaceSignals {
    /// The signals of one face, or `None` when its landmarks do not carry a
    /// pose.
    ///
    /// The one place [`HeadPose::valid`] is applied between the solve and the
    /// record, so that "unknown" cannot reach a consumer as "looking straight
    /// ahead": a solve that failed, did not converge or could not explain the
    /// landmarks yields `None` here rather than a `HeadPose` with a zeroed
    /// `Euler` beside it.
    ///
    /// The eye ratio is measured from the same 468 points and travels with them
    /// whether or not stage 2b confirmed it — `ear` is a ratio of distances
    /// within one eye, and a missing confirmation is what `eye_closed: None`
    /// says. The eye **confidence** is not read here: see [`fuse`], which needs
    /// the dense stage's own verdict for that and takes it from the same
    /// landmarks.
    pub fn from_landmarks(
        landmarks: &Landmarks,
        k: &CameraIntrinsics,
        eye_closed: Option<bool>,
    ) -> Option<FaceSignals> {
        let head = estimate_head_pose(&landmarks.points, k);
        if !head.valid {
            return None;
        }
        Some(FaceSignals {
            head,
            ear: ear(&landmarks.points),
            eye_closed,
        })
    }
}

/// One tick's stage outputs as the engine's diagnostic record.
///
/// The order of the work is the whole rule, and it is short:
///
/// 1. count the faces, saturating;
/// 2. build the primary face's signals from its landmarks — which is where an
///    invalid head pose becomes `primary: None`;
/// 3. read one confidence per signal, from the table in the module docs;
/// 4. report the **least** of the three as `overall`.
///
/// No I/O, no clock, no allocation: everything in the record is one of the
/// arguments, carried or compared, and the same arguments always produce the
/// same record. Deterministic by construction, which is what lets Task 23's
/// engine-level determinism test mean anything.
pub fn fuse(stages: StageOutputs<'_>) -> Observation {
    let faces = stages.faces.len().min(MOST_FACES as usize) as u8;

    // Both halves are required for a reading: a face the selector chose, and
    // landmarks the pose solve accepts. A detected face with no landmarks is
    // not a reading either, which is why the two are folded together here
    // rather than checked at the call site.
    let signals = stages
        .primary
        .zip(stages.landmarks)
        .and_then(|(_, landmarks)| {
            FaceSignals::from_landmarks(landmarks, stages.intrinsics, stages.eye_closed)
        });

    // The landmarks the *eye* confidence rests on: the same ones, and only when
    // they became a reading. A rejected pose and absent landmarks are one fact
    // to a consumer — the record carries no eye reading — so the field cannot
    // depend on which of the two happened.
    let eye_landmarks = if signals.is_some() {
        stages.landmarks
    } else {
        None
    };

    let face = face_confidence(faces, stages.primary, signals.is_some());
    let phone = phone_confidence(stages.phone, stages.phones);
    let eye = eye_confidence(eye_landmarks);

    Observation {
        tick_ms: stages.tick_ms,
        faces,
        primary: signals,
        phone: stages.phone,
        motion: stages.motion,
        confidence: Confidence {
            // The least, not the mean: one untrustworthy signal must not be
            // averaged away by two healthy ones. See the module docs.
            overall: face.min(phone).min(eye),
            face,
            phone,
            eye,
        },
    }
}

/// The confidence of the face reading this tick.
///
/// Three cases, and the order they are tested in *is* the rule:
///
/// 1. **No faces at all** — nothing is claimed about the wearer, so nothing is
///    distrusted, and the tick stays above the floor so that the away dwell can
///    accrue. See the module docs.
/// 2. **A face and no reading** — the detector saw a face and the landmark
///    stage could not turn it into signals: a pose the solve refused, or a
///    stage 2 that did not run. That is no evidence, and it is what keeps a
///    rejected pose from being read downstream as a frame full of usable
///    signals.
/// 3. **A face and a reading** — the primary box's own score, sanitised.
fn face_confidence(faces: u8, primary: Option<&FaceBox>, readable: bool) -> f32 {
    if faces == 0 {
        return NO_CLAIM;
    }
    if !readable {
        return NO_EVIDENCE;
    }
    probability(primary.map_or(NO_EVIDENCE, |face| face.score))
}

/// The confidence of the eye reading: the dense stage's verdict on the
/// landmarks the reading is built from.
///
/// `None` — no reading at all — is [`NO_CLAIM`] rather than zero, and a `None`
/// `eye_closed` beside a reading is **not** zero either: the confidence is
/// about the landmarks, and stage 2b's missing confirmation is stated by the
/// missing `Some`, not by a number the whole state machine reads as a floor
/// breach. See the module docs for what zeroing it would cost.
fn eye_confidence(landmarks: Option<&Landmarks>) -> f32 {
    landmarks.map_or(NO_CLAIM, |landmarks| logistic(landmarks.score))
}

/// The confidence of the phone verdict.
///
/// Only a claim of **use** can create a state — `PhoneInUse` outranks every
/// other state, and a false one is the worst failure this product has (R3, D6)
/// — so only a claim of use can be undermined by a weak detection, and the
/// field is the strongest cell-phone box in the frame for it.
///
/// `None` and `Present` claim no use, so they are [`NO_CLAIM`] whatever the
/// detector's scores are. That is deliberate and it is what keeps a
/// phone-shaped object on the desk from silencing the engine: a weak box that
/// reads `Present` cannot manufacture anything, and suppressing the tick for it
/// would lose `Focused` time on every frame it appears in — for as long as it
/// sits there.
///
/// A verdict of use with **no** cell-phone box behind it is an inconsistent
/// input, and it is no evidence rather than no claim: the fold starts at
/// [`NO_EVIDENCE`], which is exactly what an empty slice deserves.
fn phone_confidence(phone: PhoneSignal, phones: &[Detection]) -> f32 {
    match phone {
        PhoneSignal::None | PhoneSignal::Present => NO_CLAIM,
        PhoneSignal::InHand | PhoneSignal::AtHead => phones
            .iter()
            // Only cell-phone boxes are phones, exactly as Task 12 reads them:
            // a person box is not a phone however confident it is, and however
            // it is placed in the slice.
            .filter(|detection| detection.class_id == CELL_PHONE_CLASS)
            .map(|detection| probability(detection.score))
            .fold(NO_EVIDENCE, f32::max),
    }
}

/// A detector score as a confidence: finite, and within `0.0..=1.0`.
///
/// `FaceBox::score` and `Detection::score` are probabilities and both are
/// documented as such — YuNet's own decode clamps — but the fields are public,
/// so a hand-built value or a session that misbehaved can put anything here. A
/// score that is not finite is no evidence rather than a clamped guess, and the
/// NaN case is the reason it is a guard and not a flourish: `f32::min`
/// *ignores* a NaN operand, so a NaN left in a field would be dropped by the
/// minimum and a record with no usable reading at all would clear the floor.
fn probability(score: f32) -> f32 {
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        NO_EVIDENCE
    }
}

/// A raw logit as the probability it encodes.
///
/// The logistic, `1 / (1 + exp(-x))`, which is the inverse of the logit
/// function the model's head computes — and the only mapping that is, which is
/// why this is the right normalisation for a **raw logit** and would be a
/// second activation for a probability. See the module docs for what the
/// documented values of Face Mesh's score become, and for where the floor
/// falls in logit space.
///
/// Total for every input: `exp` saturates rather than overflowing, so a large
/// positive logit gives `1.0` and a large negative one `0.0`, and a NaN logit —
/// what a broken decode would hand over — is no evidence rather than a NaN in
/// the record.
fn logistic(logit: f32) -> f32 {
    let probability = 1.0 / (1.0 + (-logit).exp());
    if probability.is_finite() {
        probability
    } else {
        NO_EVIDENCE
    }
}

/// [`PhoneSignal`] on the wire, by name.
///
/// Task 12's type is in another task's file and carries no serde derives, so the
/// conversion lives here. It is by **name** rather than by discriminant because
/// the record exists to be read: the `probe` overlay prints it and a corpus
/// review compares it against ground truth, and `"in_hand"` needs no key while
/// `2` does. The round-trip test takes all four variants through it, so two
/// verdicts that shared a spelling would fail rather than collapse quietly; an
/// unknown name is an error rather than a default, so a corpus with a typo fails
/// loudly too.
mod phone_signal {
    use super::PhoneSignal;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// The wire spelling of one verdict.
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Name {
        None,
        Present,
        InHand,
        AtHead,
    }

    pub fn serialize<S: Serializer>(
        signal: &PhoneSignal,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let name = match signal {
            PhoneSignal::None => Name::None,
            PhoneSignal::Present => Name::Present,
            PhoneSignal::InHand => Name::InHand,
            PhoneSignal::AtHead => Name::AtHead,
        };
        name.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<PhoneSignal, D::Error> {
        Ok(match Name::deserialize(deserializer)? {
            Name::None => PhoneSignal::None,
            Name::Present => PhoneSignal::Present,
            Name::InHand => PhoneSignal::InHand,
            Name::AtHead => PhoneSignal::AtHead,
        })
    }
}

/// [`HeadPose`] on the wire, as its four numbers.
///
/// `HeadPose` and [`Euler`](crate::analyze::pose::Euler) belong to `pose`'s diff
/// and carry no serde derives, so the conversion lives here. It carries `valid`
/// rather than dropping it, because a `FaceSignals` is a public record and a
/// caller can build one by hand.
///
/// **Deserialising restores `pose`'s invariant**: an invalid pose comes back as
/// [`HeadPose::INVALID`] whatever angles the input carried. `INVALID` is the
/// only shape in which an invalid pose exists — `valid == false` with a zeroed
/// `Euler` — so a record that satisfied the flag but not the shape would hand a
/// consumer that reads the angles before the flag a pose that was never solved.
mod head_pose {
    use crate::analyze::pose::{Euler, HeadPose};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// The four numbers of one pose, in the order a reader expects them.
    #[derive(Serialize, Deserialize)]
    struct Angles {
        yaw: f32,
        pitch: f32,
        roll: f32,
        valid: bool,
    }

    pub fn serialize<S: Serializer>(pose: &HeadPose, serializer: S) -> Result<S::Ok, S::Error> {
        Angles {
            yaw: pose.euler.yaw,
            pitch: pose.euler.pitch,
            roll: pose.euler.roll,
            valid: pose.valid,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<HeadPose, D::Error> {
        let angles = Angles::deserialize(deserializer)?;
        if !angles.valid {
            return Ok(HeadPose::INVALID);
        }
        Ok(HeadPose {
            euler: Euler {
                yaw: angles.yaw,
                pitch: angles.pitch,
                roll: angles.roll,
            },
            valid: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::nms::Rect;
    use crate::models::yolox::{CELL_PHONE_CLASS, PERSON_CLASS};

    /// The intrinsics Task 7 estimates for the reference 640x480 frame, as
    /// literals: the focal length is the frame's wider dimension and the
    /// principal point is its exact centre. Written out rather than built
    /// through `from_frame` so that a change to *that* estimate cannot silently
    /// re-tune these tests.
    fn k() -> CameraIntrinsics {
        CameraIntrinsics::from_calibration(640.0, 640.0, 320.0, 240.0)
    }

    /// A detected face scoring `score`.
    ///
    /// The box is the same for every fixture and its landmarks are zeroes:
    /// fusion reads no geometry but the score, so a fixture that varied them
    /// would be testing a term the rule does not have. The score is a
    /// probability and every fixture below passes one that is, except where the
    /// point is that a caller handed something that is not.
    fn face(score: f32) -> FaceBox {
        FaceBox {
            rect: Rect {
                x: 100.0,
                y: 100.0,
                w: 80.0,
                h: 80.0,
            },
            score,
            landmarks: [(0.0, 0.0); 5],
        }
    }

    /// A cell-phone detection scoring `score`.
    fn phone_box(score: f32) -> Detection {
        Detection {
            rect: Rect {
                x: 200.0,
                y: 300.0,
                w: 40.0,
                h: 80.0,
            },
            score,
            class_id: CELL_PHONE_CLASS,
        }
    }

    /// A person detection scoring `score`, for the tests that check the phone
    /// field reads the class and not the frame's strongest box.
    fn person_box(score: f32) -> Detection {
        Detection {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 640.0,
                h: 480.0,
            },
            score,
            class_id: PERSON_CLASS,
        }
    }

    /// The landmarks a frontal face 600 mm from the reference camera produces,
    /// with Face Mesh scoring `score`.
    ///
    /// Hand-projected through [`k`] rather than computed, so a reader can check
    /// every pixel: `u = 640 * X / (600 + Z) + 320`, `v = 640 * Y / (600 + Z) +
    /// 240`, rounded, for the six model points of
    /// [`MODEL`](crate::analyze::pose::MODEL).
    ///
    /// | index | landmark | `(X, Y, Z)` | pixel |
    /// |---|---|---|---|
    /// | `33` | outer eye corner, image left | `(-45, -35, 25)` | `(274, 204)` |
    /// | `263` | outer eye corner, image right | `(45, -35, 25)` | `(366, 204)` |
    /// | `1` | nose tip | `(0, 0, 0)` | `(320, 240)` |
    /// | `61` | mouth corner, image left | `(-26, 36, 22)` | `(293, 277)` |
    /// | `291` | mouth corner, image right | `(26, 36, 22)` | `(347, 277)` |
    /// | `199` | chin | `(0, 70, 12)` | `(320, 313)` |
    ///
    /// The twelve eye points are placed around the two projected outer corners
    /// so that the eye aspect ratio is exactly `0.6` for **both** eyes —
    /// `(numerator 24 + 24) / (2 * 40)` — which is why `FaceSignals::ear` is
    /// asserted against that literal below. `33` and `263` are the pose's points
    /// *and* the eyes' outer corners: the same anatomical points under both
    /// readings, as `eyes` and `pose` agree.
    ///
    /// Only the eighteen points either reading consults are placed; the other
    /// 450 stay at the origin with `z` at zero, which neither module reads.
    fn frontal(score: f32) -> Landmarks {
        let mut points = [[0i32; 3]; 468];
        // The six correspondence points of the pose solve.
        for (index, (x, y)) in [
            (33usize, (274i32, 204i32)),
            (263, (366, 204)),
            (1, (320, 240)),
            (61, (293, 277)),
            (291, (347, 277)),
            (199, (320, 313)),
        ] {
            points[index] = [x, y, 0];
        }
        // The remaining ten eye points: the two inner corners, and the four lid
        // points of each eye. Their x offsets are 13 and 27 px from the eye's
        // outer corner — a 40 px eye — and their y offsets are 12 px either side
        // of the eye line.
        for (index, (x, y)) in [
            (133usize, (314i32, 204i32)),
            (160, (287, 192)),
            (158, (301, 192)),
            (153, (301, 216)),
            (144, (287, 216)),
            (362, (326, 204)),
            (385, (339, 192)),
            (387, (353, 192)),
            (373, (353, 216)),
            (380, (339, 216)),
        ] {
            points[index] = [x, y, 0];
        }
        Landmarks { points, score }
    }

    /// Landmarks that decoded to nothing: all 468 points on one pixel, with a
    /// confident Face Mesh score beside them.
    ///
    /// Six coincident points are a constellation with no extent, which no
    /// rotation fits, so the pose solve refuses them — the shape a decode that
    /// produced garbage takes.
    fn landmarks_on_one_pixel() -> Landmarks {
        Landmarks {
            points: [[0i32; 3]; 468],
            score: 15.0,
        }
    }

    #[test]
    fn confidence_is_the_minimum_of_the_signal_confidences() {
        // Three ticks, each with a *different* signal as the weakest, so the
        // rule is pinned on every field rather than on whichever one happens to
        // be low. Every number is chosen so that the mean of the three differs
        // from the minimum, and in the second case so that the mean would clear
        // Task 16's `0.6` floor while the minimum does not.
        //
        // The eye field is Face Mesh's logit through its logistic: logit `17`
        // is `1 / (1 + exp(-17))`, and `exp(-17)` is under half an ulp of `1.0`,
        // so that is exactly `1.0` in `f32`; logit `0` is exactly `0.5`.
        //
        // (face score, Face Mesh logit, phone box score, expected eye, expected
        // overall, which signal the overall is)
        let cases = [
            (0.72, 17.0, 0.55, 1.0, 0.55, "the phone"),
            (0.40, 17.0, 0.55, 1.0, 0.40, "the face"),
            (0.72, 0.0, 0.88, 0.5, 0.50, "the eye"),
        ];

        for (score, logit, phone_score, eye, overall, which) in cases {
            let faces = [face(score)];
            let landmarks = frontal(logit);
            let phones = [phone_box(phone_score)];
            let obs = fuse(StageOutputs {
                tick_ms: 1_000,
                faces: &faces,
                primary: Some(&faces[0]),
                landmarks: Some(&landmarks),
                eye_closed: Some(false),
                intrinsics: &k(),
                phones: &phones,
                phone: PhoneSignal::InHand,
                motion: 3.5,
            });

            assert_eq!(obs.confidence.face, score, "the face field");
            assert_eq!(obs.confidence.phone, phone_score, "the phone field");
            assert_eq!(obs.confidence.eye, eye, "the eye field");
            assert_eq!(
                obs.confidence.overall, overall,
                "the least trustworthy signal is {which}"
            );

            // The premise of the assertion above: a mean would read a different
            // number, so the test is about the minimum rather than about three
            // values that happen to coincide.
            let mean = (score + phone_score + eye) / 3.0;
            assert!(
                mean > overall,
                "the mean of the three is {mean}, so only a minimum reads {overall}"
            );
        }
    }

    #[test]
    fn an_invalid_head_pose_yields_no_primary_signals() {
        // Landmarks with no extent at all: the six correspondence points are one
        // pixel, no rotation fits them, and the solve refuses. A refused pose has
        // to reach the record as *no signals* — a zeroed `Euler` beside
        // `valid: false` is what downstream code reads as "looking straight
        // ahead" when it forgets to check the flag.
        let faces = [face(0.93)];
        let refused = landmarks_on_one_pixel();
        let obs = fuse(StageOutputs {
            tick_ms: 200,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&refused),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 2.0,
        });

        assert_eq!(obs.faces, 1, "the detector still saw a face");
        assert_eq!(obs.primary, None, "a refused pose carries no signals");
        assert_eq!(
            obs.confidence.face, 0.0,
            "and the face reading is not trusted: nothing was read"
        );
        assert_eq!(obs.confidence.overall, 0.0);
        assert!(
            obs.confidence.overall < 0.6,
            "below Task 16's floor, so the tick is carried forward instead of \
             being read as a frame full of usable signals"
        );

        // The same for a tick whose stage 2 did not run at all: a detected face
        // with no landmarks behind it is no reading, and the two cases must
        // agree or the state machine sees two different kinds of "no signals".
        let obs = fuse(StageOutputs {
            tick_ms: 200,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: None,
            eye_closed: None,
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 2.0,
        });
        assert_eq!(obs.primary, None);
        assert_eq!(obs.confidence.face, 0.0);
        assert_eq!(obs.confidence.overall, 0.0);

        // The positive control, without which the two blocks above would pass
        // for a fixture that is unreadable for some other reason: the same box
        // and the same intrinsics with landmarks the solve accepts produce the
        // signals, the ratio the fixture was built for, and the classifier's
        // verdict carried through.
        let good = frontal(15.0);
        let obs = fuse(StageOutputs {
            tick_ms: 200,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&good),
            eye_closed: Some(true),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 2.0,
        });
        let signals = obs.primary.expect("a solved pose is face signals");
        assert!(
            signals.head.valid,
            "and the pose in them is the valid one: {}",
            signals.head.euler.yaw
        );
        assert_eq!(
            signals.ear, 0.6,
            "48 px of lid separation over a 40 px eye, twice"
        );
        assert_eq!(signals.eye_closed, Some(true));
        assert_eq!(obs.confidence.face, 0.93);
        assert_eq!(obs.confidence.overall, 0.93);
    }

    #[test]
    fn a_rejected_landmark_reading_is_not_averaged_away() {
        // Face Mesh's verdict on the crop it was given, at the value the module
        // documents for a wall or a blank image: a raw logit of `-7`. The
        // detector is confident, the phone field claims nothing, the six
        // landmarks still solve a pose — two healthy signals and one that says
        // the landmarks are not a face. The mean clears the floor; the minimum
        // does not, and the minimum is the rule because the floor is what
        // decides whether the tick may be acted on.
        let faces = [face(0.93)];
        let landmarks = frontal(-7.0);
        let obs = fuse(StageOutputs {
            tick_ms: 400,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: None,
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 1.0,
        });

        assert!(
            obs.primary.is_some(),
            "the pose itself solved: this is about the dense verdict, not the solve"
        );
        let eye = obs.confidence.eye;
        assert!(
            (eye - 0.000911).abs() < 1e-5,
            "the logistic of -7 is 9.11e-4, got {eye}"
        );
        let mean = (obs.confidence.face + obs.confidence.phone + eye) / 3.0;
        assert!(
            mean > 0.6,
            "the premise: the mean is {mean}, which clears Task 16's floor"
        );
        assert_eq!(obs.confidence.overall, eye, "the minimum is the eye field");
        assert!(
            obs.confidence.overall < 0.6,
            "so the tick is carried forward rather than acted on"
        );
    }

    #[test]
    fn a_frame_with_no_faces_is_trusted_enough_to_reach_away_from_desk() {
        // Task 16's carry-forward rule runs *before* its zero-face rule, so a
        // tick with no faces and no confidence would never accrue the 3 s that
        // `AwayFromDesk` needs — the spec's own state would be unreachable and
        // an empty desk would read as `Focused` for ever. Zero faces claims
        // nothing about the wearer, so nothing here is distrusted.
        let obs = fuse(StageOutputs {
            tick_ms: 600,
            faces: &[],
            primary: None,
            landmarks: None,
            eye_closed: None,
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 0.0,
        });

        assert_eq!(obs.faces, 0);
        assert_eq!(obs.primary, None);
        assert_eq!(
            obs.confidence,
            Confidence {
                overall: 1.0,
                face: 1.0,
                phone: 1.0,
                eye: 1.0,
            },
            "no face, no phone and no eye reading: nothing is claimed, so \
             nothing is distrusted"
        );
    }

    #[test]
    fn the_face_confidence_is_the_primarys_own_score() {
        // Three faces, and the primary is the middle one: 0.71, which is neither
        // the detector's best box (0.95), nor the first in the slice, nor a
        // function of the count. Every signal in §5 is derived from the primary
        // face alone, so the field is the primary's own score — and the count
        // still travels beside it, because zero faces is a different state.
        let faces = [face(0.95), face(0.71), face(0.88)];
        let landmarks = frontal(17.0);
        let obs = fuse(StageOutputs {
            tick_ms: 800,
            faces: &faces,
            primary: Some(&faces[1]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 4.0,
        });

        assert_eq!(obs.faces, 3, "the count is reported as it is");
        assert_eq!(obs.confidence.face, 0.71, "the primary's own score");
        assert_eq!(obs.confidence.overall, 0.71);
    }

    #[test]
    fn a_weak_phone_box_behind_a_claim_of_use_is_not_acted_on() {
        let faces = [face(0.90)];
        let landmarks = frontal(17.0);

        // The verdict claims use — the one phone verdict that can create a state
        // — and the strongest box behind it scores 0.41. The tick goes below the
        // floor, so Task 16 carries the previous state forward rather than
        // entering `PhoneInUse` on a box the detector is not sure about.
        let weak = [phone_box(0.41)];
        let obs = fuse(StageOutputs {
            tick_ms: 1_000,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &weak,
            phone: PhoneSignal::InHand,
            motion: 1.0,
        });
        assert_eq!(obs.confidence.phone, 0.41);
        assert_eq!(obs.confidence.overall, 0.41);
        assert!(
            obs.confidence.overall < 0.6,
            "a claim of use on a weak box is carried forward, not acted on"
        );

        // A confident box in the same frame clears the floor.
        let strong = [phone_box(0.88)];
        let obs = fuse(StageOutputs {
            tick_ms: 1_000,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &strong,
            phone: PhoneSignal::AtHead,
            motion: 1.0,
        });
        assert_eq!(obs.confidence.phone, 0.88);
        assert_eq!(obs.confidence.overall, 0.88);

        // A phone the verdict says is *not* in use never drags the tick down,
        // however unsure the detector is about it: `Present` claims no use and
        // cannot manufacture a state, so suppressing the tick for it would only
        // lose `Focused` time — on every frame, for as long as the object sits
        // on the desk.
        let unsured = [phone_box(0.31)];
        let obs = fuse(StageOutputs {
            tick_ms: 1_000,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &unsured,
            phone: PhoneSignal::Present,
            motion: 1.0,
        });
        assert_eq!(obs.confidence.phone, 1.0);
        assert_eq!(obs.confidence.overall, 0.90);

        // And the class is read, not the frame's strongest box: a person
        // detection scoring 0.99 is not the phone behind the verdict.
        let mixed = [person_box(0.99), phone_box(0.44)];
        let obs = fuse(StageOutputs {
            tick_ms: 1_000,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &mixed,
            phone: PhoneSignal::InHand,
            motion: 1.0,
        });
        assert_eq!(obs.confidence.phone, 0.44);
    }

    #[test]
    fn a_claim_of_use_with_no_phone_box_behind_it_is_not_trusted() {
        // An inconsistent input — the caller says `AtHead` and hands no
        // cell-phone box — and a claim of use with nothing behind it is exactly
        // the evidence D6 refuses to act on. The field is `NO_EVIDENCE` rather
        // than the `NO_CLAIM` that `PhoneSignal::None` gets, which is the whole
        // difference between "no phone" and "on the phone, allegedly".
        let faces = [face(0.95)];
        let landmarks = frontal(17.0);
        let obs = fuse(StageOutputs {
            tick_ms: 2_000,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::AtHead,
            motion: 1.0,
        });

        assert_eq!(obs.confidence.phone, 0.0);
        assert_eq!(obs.confidence.overall, 0.0);
    }

    #[test]
    fn a_missing_eye_box_does_not_zero_the_eye_confidence() {
        // Stage 2b did not run: the eye ratio is there but there was no usable
        // crop for the classifier to confirm it with, so `eye_closed` is `None`.
        // The eye field is the dense stage's verdict on the landmarks the eye
        // reading is built from, and it does not depend on stage 2b's answer —
        // a zero here would carry *every* tick forward and silence the states
        // that never read the classifier, for a gap only `Drowsy` reads.
        let faces = [face(0.90)];
        let landmarks = frontal(17.0);
        let unconfirmed = fuse(StageOutputs {
            tick_ms: 40,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: None,
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 0.5,
        });

        assert_eq!(
            unconfirmed.primary.expect("the pose solved").eye_closed,
            None,
            "the missing confirmation is in the signals, not in the confidence"
        );
        assert_eq!(unconfirmed.confidence.eye, 1.0);
        assert_eq!(unconfirmed.confidence.overall, 0.90);

        // The same tick with a classifier verdict reads the same eye
        // confidence: the field is about the landmarks, not about stage 2b.
        let confirmed = fuse(StageOutputs {
            tick_ms: 40,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(true),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 0.5,
        });
        assert_eq!(confirmed.confidence.eye, unconfirmed.confidence.eye);
        assert_eq!(
            confirmed.primary.expect("the pose solved").eye_closed,
            Some(true),
            "and the verdict itself is carried through unchanged"
        );
    }

    #[test]
    fn the_count_saturates_and_a_score_that_is_not_a_probability_is_not_trusted() {
        // 300 faces is not a frame a webcam produces. The point is the `u8`: a
        // wrapping cast would report 44 here and **zero** at 256, and zero faces
        // is `AwayFromDesk` to the state machine.
        let many: Vec<FaceBox> = (0..300).map(|_| face(0.9)).collect();
        let landmarks = frontal(17.0);
        let obs = fuse(StageOutputs {
            tick_ms: 1,
            faces: &many,
            primary: Some(&many[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 0.0,
        });
        assert_eq!(obs.faces, 255, "the count saturates rather than wrapping");

        // A score that is not a probability is not a confidence, and the NaN
        // case is the dangerous one: `f32::min` *ignores* a NaN operand, so a
        // NaN left in the field would be dropped by the minimum and a tick with
        // no usable face score at all would clear the floor.
        let not_a_number = [face(f32::NAN)];
        let obs = fuse(StageOutputs {
            tick_ms: 1,
            faces: &not_a_number,
            primary: Some(&not_a_number[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 0.0,
        });
        assert_eq!(obs.confidence.face, 0.0, "a NaN score is not evidence");
        assert_eq!(obs.confidence.overall, 0.0);

        // And a phone box scored above 1.0 is clamped to a probability rather
        // than allowed to lift the tick above every other signal.
        let over = [phone_box(1.5)];
        let faces = [face(0.9)];
        let obs = fuse(StageOutputs {
            tick_ms: 1,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &over,
            phone: PhoneSignal::InHand,
            motion: 0.0,
        });
        assert_eq!(obs.confidence.phone, 1.0);
        assert_eq!(obs.confidence.overall, 0.9);
    }

    #[test]
    fn the_report_carries_the_tick_and_the_motion_verbatim() {
        // Fusion reads no clock and sanitises no measurement: the tick is the
        // caller's number and the motion is stage 0's own reading in grey
        // levels — including the `f32::MAX` a gate with no previous frame
        // reports, which must not be quietly turned into something that looks
        // like a static scene.
        let faces = [face(0.9)];
        let landmarks = frontal(17.0);
        let cold = fuse(StageOutputs {
            tick_ms: 4_294_967_295,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: f32::MAX,
        });
        assert_eq!(cold.tick_ms, 4_294_967_295);
        assert_eq!(cold.motion, f32::MAX);

        let moving = fuse(StageOutputs {
            tick_ms: 12_345,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(false),
            intrinsics: &k(),
            phones: &[],
            phone: PhoneSignal::None,
            motion: 255.0,
        });
        assert_eq!(moving.tick_ms, 12_345);
        assert_eq!(moving.motion, 255.0);
    }

    #[test]
    fn observation_is_serialisable_and_round_trips() {
        // Every field carries a value a default or a dropped field would change,
        // and the comparison is the whole record rather than a field at a time.
        let faces = [face(0.81)];
        let landmarks = frontal(12.5);
        let phones = [phone_box(0.74)];
        let obs = fuse(StageOutputs {
            tick_ms: 12_345,
            faces: &faces,
            primary: Some(&faces[0]),
            landmarks: Some(&landmarks),
            eye_closed: Some(true),
            intrinsics: &k(),
            phones: &phones,
            phone: PhoneSignal::InHand,
            motion: 7.5,
        });
        let json = serde_json::to_string(&obs).expect("an Observation serialises");
        let back: Observation = serde_json::from_str(&json).expect("and deserialises");
        assert_eq!(back, obs, "the record survives a round trip: {json}");

        // The four phone verdicts stay distinguishable on the wire: a mapping
        // that collapsed two of them onto one name would round-trip each of them
        // back as the same variant, and the state machine reads that variant.
        for signal in [
            PhoneSignal::None,
            PhoneSignal::Present,
            PhoneSignal::InHand,
            PhoneSignal::AtHead,
        ] {
            let one = Observation {
                tick_ms: 1,
                faces: 0,
                primary: None,
                phone: signal,
                motion: 0.0,
                confidence: Confidence {
                    overall: 1.0,
                    face: 1.0,
                    phone: 1.0,
                    eye: 1.0,
                },
            };
            let json = serde_json::to_string(&one).expect("an Observation serialises");
            let back: Observation = serde_json::from_str(&json)
                .unwrap_or_else(|error| panic!("{signal:?} does not deserialise: {error}"));
            assert_eq!(back, one, "{signal:?} survives the round trip: {json}");
        }
    }

    #[test]
    fn a_record_whose_pose_is_not_valid_carries_no_angles() {
        // `HeadPose::INVALID` is the only shape in which `pose` returns an
        // invalid pose — a zeroed `Euler` beside `valid: false` — and a record
        // read back from a corpus has to satisfy the same invariant, or code
        // that reads the angles before the flag sees a pose that was never
        // solved. The JSON is written out rather than produced by the type, so
        // this pins the wire format as well: a renamed field or a phone verdict
        // spelled as a number fails here.
        let json = r#"{
            "tick_ms": 1,
            "faces": 1,
            "primary": {
                "head": {"yaw": 31.0, "pitch": -12.0, "roll": 4.0, "valid": false},
                "ear": 0.21,
                "eye_closed": null
            },
            "phone": "none",
            "motion": 0.0,
            "confidence": {"overall": 1.0, "face": 1.0, "phone": 1.0, "eye": 1.0}
        }"#;
        let obs: Observation = serde_json::from_str(json).expect("a well-formed record");
        let signals = obs.primary.expect("the record has a primary");
        assert_eq!(
            signals.head,
            HeadPose::INVALID,
            "an invalid pose is the zeroed one, whatever the JSON carried"
        );
        assert_eq!(signals.ear, 0.21, "and the rest of the record is kept");
    }
}
