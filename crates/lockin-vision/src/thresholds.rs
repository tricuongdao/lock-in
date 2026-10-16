//! Every number the state machine's decisions rest on, in one struct.
//!
//! Spec §7.2's table of starting values is this struct, verbatim, plus the three
//! numbers the table assumes and never states: the neutral head pose, the yaw and
//! pitch bounds that make "out of bounds" checkable, and the eye-aspect-ratio
//! threshold that makes `Drowsy` checkable. They live here rather than as
//! constants beside the code that reads them because §7.2's first rule is that
//! **all thresholds are profile-relative, never absolute** — and a constant is
//! not profile-relative. Task 18's `CalibrationProfile::thresholds` fills all
//! eighteen fields from a measured wearer; [`Thresholds::default`] is what the
//! engine starts with before any profile exists, which is the state the spec
//! calls "no profile present".
//!
//! # The dwells
//!
//! The two columns of §7.2's table are a design decision, not a rounding: an
//! **entry** dwell is deliberately several times its **exit** dwell. Entry asks
//! "is this sustained?" and exit asks "has it stopped?" — and the prior art's
//! per-frame classification cannot be rescued by tuning these numbers, because it
//! has no notion of either. The table's shape is what the state machine
//! implements; the numbers are what the corpus tunes.
//!
//! | transition | field | starting value |
//! |---|---|---|
//! | head deviation → `GlancingAway` | [`glance_entry`](Thresholds::glance_entry) | 0.0 s (immediate) |
//! | `GlancingAway` → `Distracted` | [`distracted_entry`](Thresholds::distracted_entry) | 2.0 s |
//! | `Distracted` → out | [`distracted_exit`](Thresholds::distracted_exit) | 0.6 s |
//! | `PhoneInUse` entry | [`phone_entry`](Thresholds::phone_entry) | 2.0 s |
//! | `PhoneInUse` latched minimum | [`phone_latch`](Thresholds::phone_latch) | 3.0 s |
//! | `PhoneInUse` → out | [`phone_exit`](Thresholds::phone_exit) | 0.8 s |
//! | `Drowsy` entry | [`drowsy_entry`](Thresholds::drowsy_entry) | 1.0 s |
//! | `Drowsy` → out | [`drowsy_exit`](Thresholds::drowsy_exit) | 0.5 s |
//! | no face → `AwayFromDesk` | [`away_entry`](Thresholds::away_entry) | 3.0 s |
//! | face returns → out of `AwayFromDesk` | [`away_exit`](Thresholds::away_exit) | 0.5 s |
//! | second face → `OtherPresent` | [`other_present_entry`](Thresholds::other_present_entry) | 5.0 s |
//! | `OtherPresent` → out | [`other_present_exit`](Thresholds::other_present_exit) | 3.0 s |
//! | confidence floor for recording anything | [`confidence_floor`](Thresholds::confidence_floor) | 0.6 |
//!
//! Two transitions in §7.2 have **no** threshold and therefore no field here:
//! `Focused → GlancingAway` is immediate (§7.2's first bullet) and `Starting`
//! has no condition at all, so both are exited the moment a candidate is
//! eligible. `GlancingAway` has no exit dwell either — its entry is immediate,
//! and the asymmetry in §7.2 lives at `Distracted`, whose exit is 0.6 s. The
//! state machine spells those three zeros out where it reads them.

use core::time::Duration;

/// Every threshold the state machine applies, in one value.
///
/// `Copy`, because the machine takes it by value at construction and never
/// mutates it; the fields are public because a profile *is* a bag of numbers and
/// Task 18 builds one field by field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// How long the head must be out of bounds before the deviation is reported
    /// at all. Zero in §7.2: a glance is cheap bookkeeping and is reported on the
    /// first tick that sees it.
    pub glance_entry: Duration,
    /// How long the deviation must be held before it is a distraction rather
    /// than a glance. This is the field that makes `Distracted` trustworthy: it
    /// is the deviation that is timed, not the state.
    pub distracted_entry: Duration,
    /// How long the deviation must be gone before `Distracted` is left. Shorter
    /// than [`distracted_entry`](Thresholds::distracted_entry) by design.
    pub distracted_exit: Duration,
    /// How long a claim of phone use must be held before `PhoneInUse` is
    /// entered. The expensive direction: a false `PhoneInUse` is the worst
    /// failure this product has (R3, D6), so it is gated on two seconds of
    /// evidence and, after that, on a latch.
    pub phone_entry: Duration,
    /// The minimum time `PhoneInUse` is held once entered, however quickly the
    /// claim disappears. A hand occluding the phone is a one-frame event; without
    /// the latch it produces start/stop churn in the events, which is what the
    /// spec calls burst-beeping.
    pub phone_latch: Duration,
    /// How long the claim must be gone, *after* the latch expires, before
    /// `PhoneInUse` is left.
    pub phone_exit: Duration,
    /// How long the eye-aspect ratio and the eye classifier must agree that the
    /// eyes are closed before `Drowsy` is entered.
    pub drowsy_entry: Duration,
    /// How long the agreement must be gone before `Drowsy` is left.
    pub drowsy_exit: Duration,
    /// How long there must be no face at all before `AwayFromDesk` is entered.
    /// Keyed on the face **count**: see `Observation::faces`.
    pub away_entry: Duration,
    /// How long a face must be present before `AwayFromDesk` is left.
    pub away_exit: Duration,
    /// How long a second face must be present before `OtherPresent` is entered.
    /// Long, because a spurious second detection is common and a colleague is
    /// not.
    pub other_present_entry: Duration,
    /// How long the second face must be gone before `OtherPresent` is left.
    pub other_present_exit: Duration,
    /// The least [`Confidence::overall`](crate::fuse::Confidence::overall) a tick
    /// may carry and still be recorded as a state. Below it the tick is carried
    /// forward to the last stable state and emits nothing (spec §7.3).
    pub confidence_floor: f32,
    /// The calibration session's mean yaw, in degrees, which "in bounds" is
    /// measured from. Zero until Task 18 measures the wearer.
    pub neutral_yaw: f32,
    /// The calibration session's mean pitch, in degrees.
    pub neutral_pitch: f32,
    /// How far from [`neutral_yaw`](Thresholds::neutral_yaw) the head may turn
    /// and still be reading the screen, in degrees. Inclusive: exactly the bound
    /// is in bounds.
    pub yaw_bound: f32,
    /// How far from [`neutral_pitch`](Thresholds::neutral_pitch) the head may
    /// tip, in degrees. Inclusive.
    pub pitch_bound: f32,
    /// The eye-aspect ratio at or below which an eye reads as closed for this
    /// wearer. Task 18 derives it from the session's open-eye baseline and the
    /// glasses flag; it is one of the two signals `Drowsy` requires.
    pub ear_threshold: f32,
}

impl Default for Thresholds {
    /// Spec §7.2's starting values, and the pre-calibration fallbacks.
    ///
    /// The eleven dwells and the floor are the spec's own numbers. The five
    /// profile-relative numbers are this task's, and they are chosen the way
    /// §7.2 asks: from the anthropometry the spec states rather than from the
    /// prior art's constants.
    ///
    /// - **The bounds are 20°.** §5.4 says `GlancingAway` and `Distracted` "turn
    ///   on roughly 15–20° of deviation from the calibrated neutral", so the
    ///   starting value is the top of that range: under D6 a wider bound costs
    ///   recall and buys precision, which is the direction the whole design
    ///   leans. The prior art's ±10° is deliberately not copied — §7.2 calls
    ///   those constants "actively harmful without calibration".
    /// - **The neutral is zero.** Degrees are already relative to the camera
    ///   axis, and a profile that has not been measured has no better estimate of
    ///   where the wearer's screen is. Task 18 replaces both fields.
    /// - **The eye threshold is 0.2.** §5.2's ratio "measures a few tenths for a
    ///   frontal adult face" and "falls towards 0" when closed, so 0.2 sits
    ///   inside the gap. It is the weaker of `Drowsy`'s two signals by
    ///   construction: the classifier must agree before anything happens.
    fn default() -> Thresholds {
        Thresholds {
            glance_entry: Duration::ZERO,
            distracted_entry: Duration::from_millis(2000),
            distracted_exit: Duration::from_millis(600),
            phone_entry: Duration::from_millis(2000),
            phone_latch: Duration::from_millis(3000),
            phone_exit: Duration::from_millis(800),
            drowsy_entry: Duration::from_millis(1000),
            drowsy_exit: Duration::from_millis(500),
            away_entry: Duration::from_millis(3000),
            away_exit: Duration::from_millis(500),
            other_present_entry: Duration::from_millis(5000),
            other_present_exit: Duration::from_millis(3000),
            confidence_floor: 0.6,
            neutral_yaw: 0.0,
            neutral_pitch: 0.0,
            yaw_bound: 20.0,
            pitch_bound: 20.0,
            ear_threshold: 0.2,
        }
    }
}
