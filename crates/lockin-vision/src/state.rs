//! The temporal state machine: one tick's signals in, one state and at most two
//! events out.
//!
//! This is the product's decision layer. Everything before it produced signals;
//! this turns signals into states and events, and it is where D6 (high
//! precision) is actually implemented. All four reference repositories classify
//! per frame, which is exactly why they are noisy: a single frame's head angle
//! is evidence, not a verdict. Here a verdict has to be **held** — a candidate
//! state becomes the current state only after its evidence has been continuous
//! for that transition's entry dwell, and a current state is left only after its
//! evidence has been gone for that transition's exit dwell. The entry dwells are
//! several times the exit dwells on purpose (spec §7.2): entry asks "is this
//! sustained?", exit asks "has it stopped?", and the asymmetry is what removes
//! the flapping that produces burst-beeping in the prior art.
//!
//! # The order of operations, and the test that pins each step
//!
//! 1. **Zero faces is `AwayFromDesk`.** The face count is a **count**, not a
//!    signal, so it is read before the confidence floor and is not gated by it
//!    (R53): `AwayFromDesk` must be reachable whatever confidence a zero-face
//!    tick carries. The transition still serves its full entry dwell, and a
//!    zero-face tick can reach no other state — a claim of use with nobody in
//!    frame has nobody to attribute it to.
//!    `no_faces_at_all_is_away_from_desk_but_losing_only_the_primary_is_other_present`,
//!    `away_time_does_not_accrue_distraction`.
//! 2. **On a tick with a face, below the floor, carry forward.**
//!    [`Confidence::overall`] under [`Thresholds::confidence_floor`] returns the
//!    current state, adds the tick's time to it and emits nothing, before any
//!    signal-derived transition is consulted (spec §7.3: the accounting buckets
//!    must still sum to wall-clock time, and weak evidence must never
//!    manufacture a distraction). `low_confidence_ticks_carry_forward`.
//! 3. **Rank the applicable transitions** by the spec's fixed order,
//!    `PhoneInUse › Drowsy › Distracted › OtherPresent › GlancingAway ›
//!    AwayFromDesk › Focused › Starting`, and take the highest. One table states
//!    the order (`rank`) and the candidate is its maximum, so the order cannot
//!    drift between two restatements of it.
//!    `phone_in_use_outranks_distracted_when_both_apply`,
//!    `other_present_holds_until_its_exit_dwell_and_reports_the_committed_population`.
//! 4. **Entry needs its dwell; exit needs only its own.** Each transition's
//!    evidence carries its own timer, so a candidate is *not eligible at all*
//!    until its dwell has elapsed — and leaving the current state takes the
//!    current state's exit dwell on top of that.
//!    `exit_dwell_is_shorter_than_entry_dwell`.
//! 5. **`PhoneInUse` is latched** for [`Thresholds::phone_latch`] once entered, so
//!    a hand occluding the phone cannot produce start/stop churn.
//!    `phone_in_use_is_latched_for_its_minimum_duration`.
//! 6. **`Drowsy` needs both signals** — the ratio below
//!    [`Thresholds::ear_threshold`] *and* the classifier reporting closed eyes
//!    ([`FaceSignals::eye_closed`] is `Some(true)`). Requiring agreement strictly
//!    reduces false positives, which is the trade D6 asks for; the recall it
//!    costs is accepted deliberately. `drowsy_requires_ear_and_classifier_to_agree`.
//! 7. **Events only on committed transitions.** An event names a transition that
//!    satisfied its dwell; there is no per-frame event and no event at all for
//!    `GlancingAway`. An event sequence is therefore identical for identical
//!    observation sequences. `a_brief_glance_never_becomes_a_distraction`,
//!    `identical_observation_sequences_produce_identical_event_sequences`.
//!
//! [`Confidence::overall`]: crate::fuse::Confidence::overall
//!
//! # Three things this module was handed, and how it treats them
//!
//! 1. **Zero faces and low confidence are different facts (R53).** Task 15 builds
//!    `Confidence::overall` as `1.0` on a tick with no faces — nothing is claimed
//!    about the wearer, so nothing is distrusted — and as `0.0` on a tick where a
//!    face was detected and no reading could be built from it. An earlier draft
//!    of this machine read the floor first and would therefore have needed that
//!    `1.0` to be a *guarantee*: a zero-face tick below the floor could never
//!    have accrued `AwayFromDesk`'s three seconds, and the state would have been
//!    unreachable. R53 moves the count branch in front of the floor instead, so
//!    nothing here depends on what confidence a zero-face tick carries — and a
//!    live tick that is under the floor *with* no face, which is what a weak
//!    carried-forward phone box produces once the wearer stands up, still
//!    reaches `AwayFromDesk`.
//! 2. **`primary.is_none()` does not mean "nobody is there".** It also means the
//!    primary role is held by a face that is not currently on screen, or that no
//!    box was usable. Presence here is the count, and a second face that is
//!    sustained for [`Thresholds::other_present_entry`] reads `OtherPresent` —
//!    never `AwayFromDesk`, which requires the count to be exactly zero.
//! 3. **`Drowsy` is an AND, not an OR.** See rule 6.
//!
//! # A below-floor tick with a face is inert, and that is a decision
//!
//! Rule 2 returns early, before any transition's timer is touched. A tick under
//! the floor therefore neither starts nor advances nor clears a dwell: it is not
//! evidence, and evidence is what the timers measure. The alternative reading —
//! that the timer runs on wall-clock time regardless of what was seen — would let
//! a dark frame finish a dwell that no observation supports. The consequence of
//! the decision is stated plainly: a dwell measures the wall-clock span over
//! which the condition was *seen* to hold, and a gap in what was seen cannot
//! shorten it. `low_confidence_ticks_carry_forward` pins both halves — the
//! carry-forward itself, and that five seconds of below-floor deviation accrue
//! nothing.
//!
//! # Two judgement calls the spec leaves open
//!
//! - **What a duration measures.** `DistractionEnded { at, duration }` and its
//!   siblings report time **in the state**, from the committed entry to the
//!   committed exit, so `duration == at - started.at` always holds for the pair
//!   and a consumer can check it. The entry dwell that preceded the commit is not
//!   counted; the exit dwell is, because the state was current until the exit
//!   committed. The alternative — measuring from the first deviating tick — would
//!   report a number no event could be reconciled with, since the machine never
//!   sees the beginning of a deviation that was already underway in its first
//!   frame.
//! - **What `PeopleChanged` is.** The population is reported on the *committed*
//!   transitions into and out of `OtherPresent`, with the count observed on the
//!   committing tick, and on the way out only when the count really fell below
//!   two. Reporting every change in the count would be a per-frame event, which
//!   rule 7 forbids, and it would report the spurious second detections that
//!   `other_present_entry`'s five seconds exist to reject. The cost of tying it
//!   to the state is that an arrival is not reported while a higher-priority
//!   state holds — a colleague appearing during a distraction produces no event
//!   until the distraction ends — and that follows from the fixed priority order
//!   rather than from this choice.
//!
//! # Contract
//!
//! No I/O, no allocation on the hot path, no clock of its own: `update` reads the
//! observation and the caller's [`Instant`], and the same arguments always
//! produce the same state and the same events. `Starting` is the initial state
//! and never a target: a tick with no evidence at all (no reading, one face or
//! none, no second face) satisfies no entry transition, and "no candidate" is how
//! this machine says a transition has not committed — which is exactly what an
//! entry dwell means. The two-event cap on the return value is the spec's
//! `SmallVec<[FocusEvent; 2]>`, and it is exactly sufficient: a transition emits
//! at most one event for the state it leaves and one for the state it enters.

use core::time::Duration;
use std::time::Instant;

use smallvec::SmallVec;

use crate::analyze::phone::PhoneSignal;
use crate::fuse::{FaceSignals, Observation};
use crate::thresholds::Thresholds;

/// What the wearer is doing, as one of eight states (spec §7.1).
///
/// Every tick produces one of these for time accounting; only the discrete
/// [`FocusEvent`]s can trigger a nudge. The variant order below is §7.1's; the
/// **priority** order used to resolve disagreement is stated in `rank` and is
/// not inferred from this declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FocusState {
    /// Before any stable state exists: warm-up or calibration. The initial
    /// state, and not a target — see the module docs.
    Starting,
    /// No face at all, held past [`Thresholds::away_entry`].
    AwayFromDesk,
    /// A face, head pose in bounds, eyes open, no phone.
    Focused,
    /// Out of bounds, but not yet long enough to count. Cheap bookkeeping, and
    /// the one state that never emits an event.
    GlancingAway,
    /// Out of bounds, held past [`Thresholds::distracted_entry`].
    Distracted,
    /// A phone associated with the wearer and classified as in use.
    PhoneInUse,
    /// The eye ratio below the profile's threshold *and* the classifier
    /// reporting closed eyes, held past [`Thresholds::drowsy_entry`].
    Drowsy,
    /// More than one face, held past [`Thresholds::other_present_entry`].
    OtherPresent,
}

/// A committed transition, as the engine's contract with everything upstream of
/// a nudge (spec §7.4).
///
/// Every variant carries the instant it was emitted, which is always the instant
/// of the tick that committed the transition. The `*Ended` variants carry the
/// time the state was current — see the module docs — so that
/// `duration == at - started.at` holds for every pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusEvent {
    /// The wearer became `Focused` — from `Starting`, from a glance, from a
    /// distraction, from the phone, or from having been away.
    FocusGained { at: Instant },
    /// The wearer entered `Distracted`, with the signal that put them there.
    DistractionStarted {
        at: Instant,
        reason: DistractionReason,
    },
    /// `Distracted` was left. `duration` is the time it was current.
    DistractionEnded { at: Instant, duration: Duration },
    /// The wearer entered `PhoneInUse`, with the posture that was observed on
    /// the committing tick: the spec's `PhonePosture`, which Task 12 spells
    /// [`PhoneSignal`].
    PhoneUseStarted { at: Instant, posture: PhoneSignal },
    /// `PhoneInUse` was left. `duration` is the time it was current, and it is
    /// never shorter than [`Thresholds::phone_latch`].
    PhoneUseEnded { at: Instant, duration: Duration },
    /// The wearer entered `Drowsy`.
    DrowsyStarted { at: Instant },
    /// `Drowsy` was left. `duration` is the time it was current.
    DrowsyEnded { at: Instant, duration: Duration },
    /// The wearer entered `AwayFromDesk`.
    AwayStarted { at: Instant },
    /// `AwayFromDesk` was left. `duration` is the time it was current.
    AwayEnded { at: Instant, duration: Duration },
    /// The number of people in frame changed, and the change was committed:
    /// `OtherPresent` was entered, or left with fewer than two faces. `count` is
    /// the face count on the committing tick. See the module docs.
    PeopleChanged { count: u8 },
}

/// Why the wearer was called distracted (spec §7.4).
///
/// One variant, deliberately. §5.2 records that the exported landmark graph has
/// no iris points, so there is no gaze vector and `GlancingAway`/`Distracted`
/// "rest on head pose" — a head outside the profile's bounds is the only
/// mechanism this slice can measure. The enum exists rather than a bare
/// `DistractionStarted { at }` so that a later slice can add a reason of its own
/// (a screen-content signal, a session-state signal) without changing the shape
/// of the event. An unused second variant today would be a claim with nothing
/// behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistractionReason {
    /// The head was held outside the profile's yaw or pitch bounds for
    /// [`Thresholds::distracted_entry`].
    HeadDeviation,
}

/// The priority order of §7.2, as a number: higher wins.
///
/// **The order is stated here rather than inferred from declaration order**, so
/// that reordering [`FocusState`] cannot silently change which state wins — the
/// same reasoning, and the same pattern, as `PhoneSignal`'s own severity. A
/// head-down phone user is `PhoneInUse` and not `Distracted`; a second person
/// outranks a glance, because another human in the room is a plausible
/// distraction source, and a sustained deviation outranks the second person,
/// because the wearer's own head is the stronger evidence about the wearer.
///
/// `Starting` is the floor of the order. Nothing ranks below it, so a candidate
/// can never be a lower-priority alternative to warm-up: see
/// `StateMachine::supports`.
fn rank(state: FocusState) -> u8 {
    match state {
        FocusState::Starting => 0,
        FocusState::Focused => 1,
        FocusState::AwayFromDesk => 2,
        FocusState::GlancingAway => 3,
        FocusState::OtherPresent => 4,
        FocusState::Distracted => 5,
        FocusState::Drowsy => 6,
        FocusState::PhoneInUse => 7,
    }
}

/// The state machine: one `Thresholds`, the current state, and the timers that
/// decide when it may change.
///
/// Every field is either the current commitment or the instant a condition
/// became continuously true, so the whole machine is a deterministic function of
/// the observations and instants it has been handed.
#[derive(Debug)]
pub struct StateMachine {
    /// Every dwell this machine applies. Taken by value at construction and
    /// never mutated, so a profile swap is a new machine rather than a partially
    /// re-tuned one.
    thresholds: Thresholds,
    /// The committed state.
    state: FocusState,
    /// When `state` was committed, or `None` while the machine is still
    /// `Starting`. This is the origin of every `*Ended` duration.
    since: Option<Instant>,
    /// Since when the head has been continuously out of bounds, if it is. Drives
    /// both `GlancingAway` (immediately) and `Distracted` (after
    /// [`Thresholds::distracted_entry`]): what is timed is the *deviation*, which
    /// is why a wearer who was already looking away when a phone call ended does
    /// not serve a fresh two seconds in `GlancingAway`.
    deviation: Option<Instant>,
    /// Since when there have been no faces at all.
    no_face: Option<Instant>,
    /// Since when there have been two or more faces.
    multi_face: Option<Instant>,
    /// Since when the tick has claimed phone **use** — `InHand` or `AtHead`, not
    /// `Present` and not `None`, which claim no use and so can create no state.
    phone_claim: Option<Instant>,
    /// Since when the ratio and the classifier have both said the eyes are
    /// closed.
    drowsy_claim: Option<Instant>,
    /// Since when the eligible candidate has differed from the current state
    /// without the state's own evidence cancelling the difference. When this
    /// reaches the current state's exit dwell, the candidate commits.
    leaving: Option<Instant>,
}

impl StateMachine {
    /// A machine in `Starting`, with no timer running.
    ///
    /// No instant is needed: the first tick supplies one, and every dwell is
    /// measured from an instant the machine has seen.
    pub fn new(thresholds: Thresholds) -> StateMachine {
        StateMachine {
            thresholds,
            state: FocusState::Starting,
            since: None,
            deviation: None,
            no_face: None,
            multi_face: None,
            phone_claim: None,
            drowsy_claim: None,
            leaving: None,
        }
    }

    /// One tick: the state it leaves the machine in, and the events it committed.
    ///
    /// The rules and their order are in the module docs. `now` is the caller's
    /// clock, which is why this is a pure function of its arguments; a
    /// non-monotonic `now` cannot panic (every comparison saturates) and simply
    /// fails to satisfy any dwell.
    pub fn update(
        &mut self,
        obs: &Observation,
        now: Instant,
    ) -> (FocusState, SmallVec<[FocusEvent; 2]>) {
        // Rule 1, and R53's ordering: the face count is a **count**, not a
        // signal, so it is read before the floor and is not gated by it. A
        // zero-face tick therefore accrues the away dwell whatever confidence it
        // carries, which is what makes `AwayFromDesk` reachable from a live tick
        // whose carried-forward phone box is weak.
        if obs.faces > 0 && obs.confidence.overall < self.thresholds.confidence_floor {
            // Rule 2: below the floor, the tick is not evidence. Nothing is
            // touched — not a condition timer, not the current state, which the
            // caller's own clock keeps accounting for — and nothing is emitted.
            return (self.state, SmallVec::new());
        }

        let evidence = self.evidence(obs);
        self.hold(&evidence, now);

        // Rule 3: the highest-ranked transition whose dwell has elapsed.
        let Some(next) = self.candidate(&evidence, now) else {
            // No transition is eligible: this tick neither contradicts the
            // current state nor supports another, so nothing is being left.
            self.leaving = None;
            return (self.state, SmallVec::new());
        };

        // Rules 3 and 4: leaving the current state costs its exit dwell, and a
        // candidate that is *lower* priority than a state whose own evidence
        // still holds does not even start that clock. The exception is a
        // higher-priority claim, which the fixed order says wins — on the current
        // state's terms, not immediately.
        if next == self.state
            || (self.supports(self.state, &evidence) && rank(next) < rank(self.state))
        {
            self.leaving = None;
        } else if self.leaving.is_none() {
            self.leaving = Some(now);
        }

        // `next` differs from the current state here — the branch above clears
        // `leaving` in exactly the cases that would leave nothing to commit to.
        let Some(left_at) = self.leaving else {
            return (self.state, SmallVec::new());
        };
        let leaving_for = now.saturating_duration_since(left_at);
        if leaving_for < self.exit_dwell(self.state) || !self.latch_expired(now) {
            return (self.state, SmallVec::new());
        }

        // Rule 7: a committed transition, and only a committed transition, emits.
        let events = self.transition_events(self.state, next, obs, now);
        self.state = next;
        self.since = Some(now);
        self.leaving = None;
        (next, events)
    }

    /// This tick's evidence, read once.
    ///
    /// The condition timers, the candidate table and [`StateMachine::supports`]
    /// all read the same record, so no two of them can disagree about what this
    /// tick said.
    fn evidence(&self, obs: &Observation) -> Evidence {
        let reading = self.reading(obs);
        let in_bounds = reading.is_some_and(|signals| self.in_bounds(signals));
        Evidence {
            in_bounds,
            // An unreadable head is not a deviating one: the deviation needs the
            // reading that would state it.
            deviating: reading.is_some() && !in_bounds,
            drowsy: reading.is_some_and(|signals| self.drowsy_agreement(signals)),
            faces: obs.faces,
            // A claim of use needs somebody to attribute the phone to. Without a
            // face there is no head region, no wearer and no hand, which is why
            // Task 12's classifier answers `None` for a frame it cannot attribute
            // — and why a carried-forward verdict on a zero-face tick is not a
            // claim this machine can act on.
            phone_use: obs.faces > 0 && uses_phone(obs.phone),
        }
    }

    /// The tick's reading, or `None` when there is nothing to read.
    ///
    /// Three ways to have no reading, and they are one fact to every rule here:
    /// no face at all; a pose the solve rejected ([`HeadPose::valid`] is `false`,
    /// which is the pose module's own "no evidence" state — reading it as a
    /// zeroed `Euler` would turn a failed solve into a head looking straight
    /// ahead); and an angle that is not a finite number, which compares false
    /// against every bound and would otherwise read as *deviating*, manufacturing
    /// a distraction out of a broken solve. The thresholds themselves are not
    /// defended the same way: a profile is a validated configuration (Task 18
    /// rejects the sessions it cannot measure), while a pose is a measurement.
    ///
    /// [`HeadPose::valid`]: crate::analyze::pose::HeadPose::valid
    fn reading(&self, obs: &Observation) -> Option<FaceSignals> {
        if obs.faces == 0 {
            return None;
        }
        obs.primary.filter(|signals| {
            signals.head.valid
                && signals.head.euler.yaw.is_finite()
                && signals.head.euler.pitch.is_finite()
        })
    }

    /// Whether the head is inside the profile's bounds.
    ///
    /// Inclusive at the bound: exactly `yaw_bound` degrees from neutral is in
    /// bounds. `roll` is not read — §8's profile carries yaw and pitch bounds,
    /// and a head tipped past its side is still facing the screen.
    fn in_bounds(&self, signals: FaceSignals) -> bool {
        (signals.head.euler.yaw - self.thresholds.neutral_yaw).abs() <= self.thresholds.yaw_bound
            && (signals.head.euler.pitch - self.thresholds.neutral_pitch).abs()
                <= self.thresholds.pitch_bound
    }

    /// Rule 6: the ratio below the profile's threshold **and** the classifier
    /// reporting closed eyes. Either alone is not drowsiness — see the module
    /// docs on the precision this buys and the recall it costs.
    fn drowsy_agreement(&self, signals: FaceSignals) -> bool {
        signals.ear < self.thresholds.ear_threshold && signals.eye_closed == Some(true)
    }

    /// Starts, keeps or clears every condition timer from this tick's evidence.
    ///
    /// One place, so that a condition cannot be forgotten by a branch that
    /// returns early — an omission there would leave a stale timer running and
    /// commit a transition nothing supports.
    fn hold(&mut self, evidence: &Evidence, now: Instant) {
        hold_since(&mut self.deviation, evidence.deviating, now);
        hold_since(&mut self.drowsy_claim, evidence.drowsy, now);
        hold_since(&mut self.phone_claim, evidence.phone_use, now);
        hold_since(&mut self.no_face, evidence.faces == 0, now);
        hold_since(&mut self.multi_face, evidence.faces > 1, now);
    }

    /// The state this tick would commit to, or `None` when no transition is
    /// eligible yet.
    ///
    /// The table is §7.2's priority order, and the maximum is taken by
    /// `rank` rather than by writing the order down twice: reordering the table
    /// without reordering `rank` would not compile, and reordering `rank` is a
    /// one-line change to the order itself.
    fn candidate(&self, evidence: &Evidence, now: Instant) -> Option<FocusState> {
        [
            (
                self.phone_claim,
                self.thresholds.phone_entry,
                FocusState::PhoneInUse,
            ),
            (
                self.drowsy_claim,
                self.thresholds.drowsy_entry,
                FocusState::Drowsy,
            ),
            (
                self.deviation,
                self.thresholds.distracted_entry,
                FocusState::Distracted,
            ),
            (
                self.multi_face,
                self.thresholds.other_present_entry,
                FocusState::OtherPresent,
            ),
            (
                self.deviation,
                self.thresholds.glance_entry,
                FocusState::GlancingAway,
            ),
            (
                self.no_face,
                self.thresholds.away_entry,
                FocusState::AwayFromDesk,
            ),
        ]
        .into_iter()
        .filter(|(since, dwell, _)| held_for(*since, *dwell, now))
        .map(|(_, _, state)| state)
        .chain(evidence.in_bounds.then_some(FocusState::Focused))
        .max_by_key(|state| rank(*state))
    }

    /// Whether this tick's evidence still supports `state`'s own condition.
    ///
    /// This is what stops the exit dwell from counting down through a tick that
    /// re-supports the state: a wearer who looked back for 0.2 s and then looked
    /// away again never stopped being distracted, and the exit clock starts again
    /// rather than having quietly run on behind the blip.
    fn supports(&self, state: FocusState, evidence: &Evidence) -> bool {
        match state {
            // Warm-up claims nothing about the wearer, so no tick can contradict
            // it. Nothing ranks below `Starting`, so a candidate can never be a
            // lower-priority alternative to it either: the arm exists to make the
            // match total, not because it decides anything.
            FocusState::Starting => true,
            FocusState::Focused => evidence.in_bounds,
            FocusState::GlancingAway | FocusState::Distracted => self.deviation.is_some(),
            FocusState::PhoneInUse => self.phone_claim.is_some(),
            FocusState::Drowsy => self.drowsy_claim.is_some(),
            FocusState::OtherPresent => self.multi_face.is_some(),
            FocusState::AwayFromDesk => self.no_face.is_some(),
        }
    }

    /// How long a differing candidate must persist before `state` may be left.
    ///
    /// §7.2's table states one for five states; `Focused`, `GlancingAway` and
    /// `Starting` have none, because they are the cheap direction — a glance is
    /// entered and left immediately, and focus is not a claim that has to be
    /// defended. The asymmetry the spec asks for lives in the five that do.
    fn exit_dwell(&self, state: FocusState) -> Duration {
        match state {
            FocusState::Distracted => self.thresholds.distracted_exit,
            FocusState::PhoneInUse => self.thresholds.phone_exit,
            FocusState::Drowsy => self.thresholds.drowsy_exit,
            FocusState::AwayFromDesk => self.thresholds.away_exit,
            FocusState::OtherPresent => self.thresholds.other_present_exit,
            FocusState::Focused | FocusState::GlancingAway | FocusState::Starting => Duration::ZERO,
        }
    }

    /// Rule 4: `PhoneInUse` cannot be left before its latch has run, however
    /// quickly its evidence disappeared.
    fn latch_expired(&self, now: Instant) -> bool {
        if self.state != FocusState::PhoneInUse {
            return true;
        }
        self.since
            .is_none_or(|since| now.saturating_duration_since(since) >= self.thresholds.phone_latch)
    }

    /// Rule 7: the events of one committed transition, in the order they
    /// happened — what ended, then what began.
    ///
    /// At most one event per match, so the two-slot vector the spec asks for is
    /// exactly enough and never spills. `GlancingAway` appears in neither match
    /// arm's output: the spec's one state that is never an event.
    fn transition_events(
        &self,
        from: FocusState,
        to: FocusState,
        obs: &Observation,
        now: Instant,
    ) -> SmallVec<[FocusEvent; 2]> {
        let mut events = SmallVec::new();
        let held = self
            .since
            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
        match from {
            FocusState::Distracted => events.push(FocusEvent::DistractionEnded {
                at: now,
                duration: held,
            }),
            FocusState::PhoneInUse => events.push(FocusEvent::PhoneUseEnded {
                at: now,
                duration: held,
            }),
            FocusState::Drowsy => events.push(FocusEvent::DrowsyEnded {
                at: now,
                duration: held,
            }),
            FocusState::AwayFromDesk => events.push(FocusEvent::AwayEnded {
                at: now,
                duration: held,
            }),
            // The population is reported on the way out only when it really
            // changed: leaving for a higher-priority state with both people still
            // in frame is not a change in who is in the room.
            FocusState::OtherPresent if obs.faces < 2 => {
                events.push(FocusEvent::PeopleChanged { count: obs.faces });
            }
            _ => {}
        }
        match to {
            FocusState::Focused => events.push(FocusEvent::FocusGained { at: now }),
            FocusState::Distracted => events.push(FocusEvent::DistractionStarted {
                at: now,
                reason: DistractionReason::HeadDeviation,
            }),
            FocusState::PhoneInUse => events.push(FocusEvent::PhoneUseStarted {
                at: now,
                // The claim is live on the committing tick by construction: the
                // candidate is only eligible while the timer the claim started is
                // still running.
                posture: obs.phone,
            }),
            FocusState::Drowsy => events.push(FocusEvent::DrowsyStarted { at: now }),
            FocusState::AwayFromDesk => events.push(FocusEvent::AwayStarted { at: now }),
            FocusState::OtherPresent => events.push(FocusEvent::PeopleChanged { count: obs.faces }),
            FocusState::GlancingAway | FocusState::Starting => {}
        }
        events
    }
}

/// One tick's evidence, as the rules read it.
///
/// A record rather than a set of locals so that the three readers of it — the
/// condition timers, the candidate table and [`StateMachine::supports`] — cannot
/// disagree about what this tick said.
struct Evidence {
    /// The head is inside the profile's bounds.
    in_bounds: bool,
    /// The head is outside them, which requires a reading.
    deviating: bool,
    /// The ratio and the classifier agree that the eyes are closed.
    drowsy: bool,
    /// How many faces the detector found.
    faces: u8,
    /// The tick claims the wearer is using a phone, and there is somebody to
    /// attribute that use to.
    phone_use: bool,
}

/// Whether a phone verdict claims **use**.
///
/// `InHand` and `AtHead` are claims about the wearer; `Present` is the desk
/// phone, which §5.1's rule and Task 12's classifier both exclude, and `None`
/// claims nothing at all. Only a claim of use can create a state.
fn uses_phone(signal: PhoneSignal) -> bool {
    matches!(signal, PhoneSignal::InHand | PhoneSignal::AtHead)
}

/// Starts, keeps or clears a condition's timer.
///
/// The timer holds the instant the condition most recently became true, so
/// "continuously held" is one `Option`, and a condition that stops being held is
/// forgotten rather than accumulated.
fn hold_since(slot: &mut Option<Instant>, held: bool, now: Instant) {
    match (held, *slot) {
        (true, None) => *slot = Some(now),
        (false, _) => *slot = None,
        (true, Some(_)) => {}
    }
}

/// Whether a condition held since `since` has served `dwell`.
///
/// A `now` before `since` — a caller whose clock went backwards — serves nothing
/// rather than panicking, which is the same reason every comparison in this file
/// saturates.
fn held_for(since: Option<Instant>, dwell: Duration, now: Instant) -> bool {
    since.is_some_and(|since| now.saturating_duration_since(since) >= dwell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::pose::{Euler, HeadPose};
    use crate::fuse::Confidence;

    /// The origin of every clock in this file.
    ///
    /// Taken from the real clock exactly once per test and never again: every
    /// tick and every asserted instant is this base plus an explicit offset, so a
    /// dwell assertion is arithmetic rather than a race with how long the test
    /// took to run. [`Instant`] has no other constructor, so this is the one call
    /// to a clock here.
    fn base() -> Instant {
        Instant::now()
    }

    /// A valid head pose, `yaw` and `pitch` degrees from the camera axis.
    ///
    /// `roll` is zero and is not read by any rule: the profile's bounds are yaw
    /// and pitch (spec §8), and a head tipped past its side is a real pose that
    /// says nothing about where the wearer is looking.
    fn pose(yaw: f32, pitch: f32) -> HeadPose {
        HeadPose {
            euler: Euler {
                yaw,
                pitch,
                roll: 0.0,
            },
            valid: true,
        }
    }

    /// One face's signals: where it is pointing, and the two eye readings.
    fn signals(yaw: f32, pitch: f32, ear: f32, eye_closed: Option<bool>) -> FaceSignals {
        FaceSignals {
            head: pose(yaw, pitch),
            ear,
            eye_closed,
        }
    }

    /// The eyes of an awake wearer: a 0.30 ratio, well above the 0.2 default, and
    /// the classifier agreeing they are open.
    fn awake() -> FaceSignals {
        signals(0.0, 0.0, 0.30, Some(false))
    }

    /// One tick, with **every** confidence field set to `overall`.
    ///
    /// The state machine reads `overall` and nothing else, and a fixture that
    /// varied the three signal confidences would be testing fields no rule here
    /// consults. `overall` is a parameter rather than a constant because the
    /// floor is the second rule: a fixture has to say which side of it the tick
    /// is on, and the tests that care are named for it.
    fn tick(
        faces: u8,
        primary: Option<FaceSignals>,
        phone: PhoneSignal,
        overall: f32,
    ) -> Observation {
        Observation {
            tick_ms: 0,
            faces,
            primary,
            phone,
            motion: 0.0,
            confidence: Confidence {
                overall,
                face: overall,
                phone: overall,
                eye: overall,
            },
        }
    }

    /// The ordinary tick: one face looking at the screen with its eyes open, no
    /// phone, trusted.
    fn looking() -> Observation {
        tick(1, Some(awake()), PhoneSignal::None, 1.0)
    }

    /// A trusted tick with the head turned `yaw` degrees, and a phone verdict.
    ///
    /// 40° is twice the default 20° bound, so "out of bounds" is not a borderline
    /// call in any test below.
    fn facing(yaw: f32, phone: PhoneSignal) -> Observation {
        tick(1, Some(signals(yaw, 0.0, 0.30, Some(false))), phone, 1.0)
    }

    /// The wearer's head out of bounds, no phone.
    fn turned_away() -> Observation {
        facing(40.0, PhoneSignal::None)
    }

    /// A trusted tick with the given eye readings and the head in bounds.
    fn eyes(ear: f32, eye_closed: Option<bool>) -> Observation {
        tick(
            1,
            Some(signals(0.0, 0.0, ear, eye_closed)),
            PhoneSignal::None,
            1.0,
        )
    }

    /// `count` faces, and the primary face's signals or none.
    fn crowd(count: u8, primary: Option<FaceSignals>) -> Observation {
        tick(count, primary, PhoneSignal::None, 1.0)
    }

    /// One tick with a confidence of `overall` in every field, built field by
    /// field.
    ///
    /// The two fixtures below need a specific combination, and the combination is
    /// the point: what a live pipeline produces on a zero-face tick is a
    /// `1.0` face field and a `1.0` eye field — nothing is claimed about a
    /// wearer who is not in frame — so the only field that can pull `overall`
    /// under the floor is the phone verdict.
    fn absent(faces: u8, phone: PhoneSignal, overall: f32) -> Observation {
        tick(faces, None, phone, overall)
    }

    /// Nobody at the desk, nothing claimed, trusted: what a live pipeline
    /// produces for an empty chair, and the fixture `AwayFromDesk` is reached
    /// through.
    fn empty() -> Observation {
        absent(0, PhoneSignal::None, 1.0)
    }

    /// Nobody at the desk, with the phone verdict Task 12 carried forward at full
    /// confidence.
    ///
    /// Stage 3 runs at 1–2 Hz against a 5 Hz tick, so a caller re-emits its last
    /// verdict — and `classify_phone` cannot revise it without a primary face to
    /// anchor the head region. A claim of use with nobody in frame is therefore
    /// reachable input, and rules 1 and 3 have to decide it between them.
    fn empty_claiming(phone: PhoneSignal) -> Observation {
        absent(0, phone, 1.0)
    }

    /// The same carried-forward claim behind a **weak** phone box: `overall` is
    /// the least of the three signal confidences, so one weak box under a claim
    /// of use is enough to put the whole tick below the floor while the face
    /// count says nobody is there.
    ///
    /// This is the live shape R53 exists for. Read the floor first and the away
    /// dwell can never accrue for a wearer who stands up with the phone in hand
    /// after a weak detection; read the count first and it accrues.
    fn empty_claiming_weakly(phone: PhoneSignal) -> Observation {
        absent(0, phone, 0.4)
    }

    /// The hand-built dark observation: a face was detected, no reading could be
    /// built from it, and `overall` is `0.0` — Task 15's "no evidence" tick.
    fn dark() -> Observation {
        tick(1, None, PhoneSignal::None, 0.0)
    }

    /// A deviating tick that is nonetheless below the floor.
    ///
    /// Nothing in the pipeline produces a confident detection with a weak
    /// *reading*: Task 15 zeroes the face field when a pose is rejected. This
    /// fixture exists to prove the floor is checked before the deviation is read,
    /// so it does not have to be reachable — the `0.2` is a hand-written number
    /// under the default `0.6`.
    fn dark_and_turned() -> Observation {
        tick(
            1,
            Some(signals(40.0, 0.0, 0.30, Some(false))),
            PhoneSignal::None,
            0.2,
        )
    }

    /// A machine driven by a clock this file controls, keeping every event it
    /// emitted.
    struct Run {
        machine: StateMachine,
        base: Instant,
        state: FocusState,
        events: Vec<FocusEvent>,
        states: Vec<FocusState>,
    }

    impl Run {
        /// A machine with the spec's starting thresholds, on its own clock.
        fn new() -> Run {
            Run::on(Thresholds::default(), base())
        }

        /// A machine on a clock supplied by the caller, which is what lets two
        /// runs be compared instant for instant.
        fn on(thresholds: Thresholds, base: Instant) -> Run {
            Run {
                machine: StateMachine::new(thresholds),
                base,
                state: FocusState::Starting,
                events: Vec::new(),
                states: Vec::new(),
            }
        }

        /// `base + ms`: the only instants any assertion in this file mentions.
        fn at(&self, ms: u64) -> Instant {
            self.base + Duration::from_millis(ms)
        }

        /// One tick `ms` after the base, returning that tick's own events.
        fn tick(&mut self, ms: u64, obs: &Observation) -> SmallVec<[FocusEvent; 2]> {
            let (state, events) = self.machine.update(obs, self.at(ms));
            self.state = state;
            self.events.extend(events.iter().copied());
            self.states.push(state);
            events
        }

        /// Feeds `obs` at `step_ms` intervals from `from_ms` to `to_ms`
        /// inclusive, returning the state after each tick.
        ///
        /// A driver, not an oracle: it asserts nothing, so the tests below can say
        /// what each span must look like.
        fn span(
            &mut self,
            from_ms: u64,
            to_ms: u64,
            step_ms: u64,
            obs: &Observation,
        ) -> Vec<FocusState> {
            let mut seen = Vec::new();
            let mut ms = from_ms;
            while ms <= to_ms {
                self.tick(ms, obs);
                seen.push(self.state);
                ms += step_ms;
            }
            seen
        }
    }

    /// The event's variant name, for the determinism test's "this sequence is not
    /// two empty vectors" check.
    ///
    /// Every arm has to return a distinct string: the test compares a
    /// de-duplicated list against a literal, so a copy-pasted arm collapses two
    /// kinds into one and fails rather than passing quietly.
    fn kind(event: &FocusEvent) -> &'static str {
        match event {
            FocusEvent::FocusGained { .. } => "focus_gained",
            FocusEvent::DistractionStarted { .. } => "distraction_started",
            FocusEvent::DistractionEnded { .. } => "distraction_ended",
            FocusEvent::PhoneUseStarted { .. } => "phone_use_started",
            FocusEvent::PhoneUseEnded { .. } => "phone_use_ended",
            FocusEvent::DrowsyStarted { .. } => "drowsy_started",
            FocusEvent::DrowsyEnded { .. } => "drowsy_ended",
            FocusEvent::AwayStarted { .. } => "away_started",
            FocusEvent::AwayEnded { .. } => "away_ended",
            FocusEvent::PeopleChanged { .. } => "people_changed",
        }
    }

    #[test]
    fn a_brief_glance_never_becomes_a_distraction() {
        // A glance is out of bounds and reported at once — "Focused →
        // GlancingAway is immediate, because it is cheap bookkeeping" — and it is
        // the one state that never emits an event. Half a second of it is four
        // times shorter than the 2.0 s that would make it a distraction.
        let mut run = Run::new();
        assert_eq!(
            run.tick(0, &looking()).as_slice(),
            [FocusEvent::FocusGained { at: run.at(0) }],
            "settling into Focused is itself a committed transition"
        );
        assert_eq!(run.state, FocusState::Focused);

        let seen = run.span(200, 700, 100, &turned_away());
        assert!(
            seen.iter().all(|state| *state == FocusState::GlancingAway),
            "the deviation is reported on the first tick that sees it: {seen:?}"
        );
        assert!(
            seen.iter().all(|state| *state != FocusState::Distracted),
            "0.5 s is a quarter of the entry dwell: {seen:?}"
        );
        assert_eq!(
            run.events,
            [FocusEvent::FocusGained { at: run.at(0) }],
            "a glance emits nothing, and nothing emits per frame"
        );
    }

    #[test]
    fn a_sustained_deviation_emits_distraction_started_once() {
        // The same signal, held: exactly one event, on the tick the dwell
        // completes, and nothing on the ticks after it.
        let mut run = Run::new();
        run.tick(0, &looking());

        let before = run.span(1000, 2800, 200, &turned_away());
        assert!(
            before
                .iter()
                .all(|state| *state == FocusState::GlancingAway),
            "1.8 s of deviation is still a glance: {before:?}"
        );
        assert!(run.events.len() == 1, "nothing yet: {:?}", run.events);

        assert_eq!(
            run.tick(3000, &turned_away()).as_slice(),
            [FocusEvent::DistractionStarted {
                at: run.at(3000),
                reason: DistractionReason::HeadDeviation,
            }],
            "exactly 2.0 s of deviation is the entry dwell"
        );
        assert_eq!(run.state, FocusState::Distracted);

        let after = run.span(3200, 4000, 200, &turned_away());
        assert!(
            after.iter().all(|state| *state == FocusState::Distracted),
            "the state is held while the deviation continues: {after:?}"
        );
        assert_eq!(
            run.events,
            [
                FocusEvent::FocusGained { at: run.at(0) },
                FocusEvent::DistractionStarted {
                    at: run.at(3000),
                    reason: DistractionReason::HeadDeviation,
                },
            ],
            "3 s of deviation, one event"
        );
    }

    #[test]
    fn distraction_ended_reports_the_correct_duration() {
        // The pair has to be reconcilable: `duration` is the time `Distracted`
        // was current, so it is exactly `ended.at - started.at`, and this test
        // asserts both that identity and the arithmetic behind it.
        let mut run = Run::new();
        run.tick(0, &looking());
        run.span(1000, 5000, 200, &turned_away());

        let started = run.at(3000);
        assert_eq!(run.state, FocusState::Distracted);

        let returning = run.span(5200, 5600, 200, &looking());
        assert!(
            returning
                .iter()
                .all(|state| *state == FocusState::Distracted),
            "0.4 s back in bounds is under the 0.6 s exit dwell: {returning:?}"
        );

        let closed = run.tick(5800, &looking());
        assert_eq!(
            closed.as_slice(),
            [
                FocusEvent::DistractionEnded {
                    at: run.at(5800),
                    duration: Duration::from_millis(2800),
                },
                FocusEvent::FocusGained { at: run.at(5800) },
            ],
            "the exit dwell completes at 5800, 0.6 s after the wearer looked back"
        );

        let Some(FocusEvent::DistractionEnded { at, duration }) = closed.first().copied() else {
            panic!("the tick that left Distracted must report the end: {closed:?}");
        };
        assert_eq!(
            duration,
            at.duration_since(started),
            "duration is the time in the state, so the pair reconciles"
        );
        assert_eq!(duration, Duration::from_millis(5800 - 3000));
    }

    #[test]
    fn exit_dwell_is_shorter_than_entry_dwell() {
        // The asymmetry, and the reason it exists: leaving a state must be
        // cheaper than entering it, or the state flaps.
        let mut run = Run::new();
        run.tick(0, &looking());
        run.span(1000, 2800, 200, &turned_away());

        assert_eq!(
            run.tick(2999, &turned_away()).len(),
            0,
            "1.999 s of deviation is not 2.000 s of it"
        );
        assert_eq!(run.state, FocusState::GlancingAway);

        run.tick(3000, &turned_away());
        assert_eq!(run.state, FocusState::Distracted);

        // A brief return to bounds, then the deviation again: the exit is
        // cancelled, so nothing at all is reported. Without that cancellation the
        // 0.2 s return would satisfy `distracted_exit` on a later tick and emit a
        // DistractionEnded for a distraction that never stopped.
        assert!(run.tick(3200, &looking()).is_empty());
        assert!(run.tick(3400, &turned_away()).is_empty());
        let resumed = run.span(3600, 5400, 200, &turned_away());
        assert!(
            resumed.iter().all(|state| *state == FocusState::Distracted),
            "the state never left: {resumed:?}"
        );
        assert_eq!(
            run.events.len(),
            2,
            "one FocusGained and one DistractionStarted, and nothing from the blip: {:?}",
            run.events
        );

        // The same 0.6 s that would not *enter* anything is enough to leave.
        let settling = run.span(5600, 6000, 200, &looking());
        assert!(
            settling
                .iter()
                .all(|state| *state == FocusState::Distracted),
            "0.4 s in bounds: {settling:?}"
        );
        assert_eq!(
            run.tick(6200, &looking()).as_slice(),
            [
                FocusEvent::DistractionEnded {
                    at: run.at(6200),
                    duration: Duration::from_millis(3200),
                },
                FocusEvent::FocusGained { at: run.at(6200) },
            ],
            "0.6 s in bounds is the exit dwell"
        );
        assert_eq!(
            run.events,
            [
                FocusEvent::FocusGained { at: run.at(0) },
                FocusEvent::DistractionStarted {
                    at: run.at(3000),
                    reason: DistractionReason::HeadDeviation,
                },
                FocusEvent::DistractionEnded {
                    at: run.at(6200),
                    duration: Duration::from_millis(3200),
                },
                FocusEvent::FocusGained { at: run.at(6200) },
            ],
            "2.0 s to enter, 0.6 s to leave, four events in the whole run"
        );
    }

    #[test]
    fn phone_in_use_outranks_distracted_when_both_apply() {
        // Both transitions become eligible on the same tick, and the fixed order
        // decides. This is spec §7.2's "a head-down phone user is in PhoneInUse,
        // not Distracted".
        let mut both = Run::new();
        both.tick(0, &looking());
        let early = both.span(1000, 2800, 200, &facing(40.0, PhoneSignal::InHand));
        assert!(
            early.iter().all(|state| *state == FocusState::GlancingAway),
            "neither has served its 2.0 s yet: {early:?}"
        );
        assert_eq!(
            both.tick(3000, &facing(40.0, PhoneSignal::InHand))
                .as_slice(),
            [FocusEvent::PhoneUseStarted {
                at: both.at(3000),
                posture: PhoneSignal::InHand,
            }],
            "the phone wins the tick it becomes eligible on"
        );
        assert_eq!(both.state, FocusState::PhoneInUse);
        both.span(3200, 4000, 200, &facing(40.0, PhoneSignal::InHand));
        assert!(
            !both.events.iter().any(|event| matches!(
                event,
                FocusEvent::DistractionStarted { .. } | FocusEvent::DistractionEnded { .. }
            )),
            "a deviation under a phone claim is never a distraction: {:?}",
            both.events
        );

        // The other order: distracted first, phone second. The higher-priority
        // claim still wins, but only after `Distracted`'s own exit dwell — a
        // current state is left on its own terms.
        let mut later = Run::new();
        later.tick(0, &looking());
        later.span(1000, 5000, 200, &turned_away());
        assert_eq!(later.state, FocusState::Distracted);

        let claim = later.span(5200, 7000, 200, &facing(40.0, PhoneSignal::InHand));
        assert!(
            claim.iter().all(|state| *state == FocusState::Distracted),
            "the phone claim has not served its 2.0 s entry yet: {claim:?}"
        );
        let waiting = later.span(7200, 7600, 200, &facing(40.0, PhoneSignal::InHand));
        assert!(
            waiting.iter().all(|state| *state == FocusState::Distracted),
            "the claim is eligible at 7200 but Distracted keeps its exit dwell: {waiting:?}"
        );
        assert_eq!(
            later
                .tick(7800, &facing(40.0, PhoneSignal::InHand))
                .as_slice(),
            [
                FocusEvent::DistractionEnded {
                    at: later.at(7800),
                    duration: Duration::from_millis(4800),
                },
                FocusEvent::PhoneUseStarted {
                    at: later.at(7800),
                    posture: PhoneSignal::InHand,
                },
            ],
            "0.6 s after the claim became eligible, both halves are reported"
        );
        assert_eq!(later.state, FocusState::PhoneInUse);
    }

    #[test]
    fn phone_in_use_is_latched_for_its_minimum_duration() {
        // A hand occludes the phone 1.6 s after the state was entered. The exit
        // dwell expires at 4400 and `PhoneUseEnded` must not fire: the latch runs
        // to 5000, and the event's own duration proves it did.
        let mut run = Run::new();
        run.tick(0, &facing(0.0, PhoneSignal::InHand));
        run.span(200, 1800, 200, &facing(0.0, PhoneSignal::InHand));
        assert_eq!(
            run.state,
            FocusState::Focused,
            "2.0 s of claim is the entry"
        );

        assert_eq!(
            run.tick(2000, &facing(0.0, PhoneSignal::InHand)).as_slice(),
            [FocusEvent::PhoneUseStarted {
                at: run.at(2000),
                posture: PhoneSignal::InHand,
            }],
            "entered at exactly 2.0 s"
        );

        run.span(2200, 3400, 200, &facing(0.0, PhoneSignal::InHand));
        let occluded = run.span(3600, 4800, 200, &facing(0.0, PhoneSignal::Present));
        assert!(
            occluded
                .iter()
                .all(|state| *state == FocusState::PhoneInUse),
            "1.2 s of occlusion cannot leave a latched state: {occluded:?}"
        );
        assert_eq!(
            run.events.len(),
            2,
            "no PhoneUseEnded before the latch expires: {:?}",
            run.events
        );

        assert_eq!(
            run.tick(5000, &facing(0.0, PhoneSignal::Present))
                .as_slice(),
            [
                FocusEvent::PhoneUseEnded {
                    at: run.at(5000),
                    duration: Duration::from_millis(3000),
                },
                FocusEvent::FocusGained { at: run.at(5000) },
            ],
            "the latch expires at 5000 and the exit dwell is already satisfied"
        );
        let ended = run
            .events
            .iter()
            .find_map(|event| match event {
                FocusEvent::PhoneUseEnded { duration, .. } => Some(*duration),
                _ => None,
            })
            .expect("the run must end the phone use");
        assert_eq!(
            ended,
            Thresholds::default().phone_latch,
            "the reported duration is exactly the latch, because the latch is what \
             ended it"
        );
    }

    #[test]
    fn low_confidence_ticks_carry_forward() {
        // Review Focus 5, and spec §7.3: a tick under the floor is not recorded
        // as a new state, its time is carried forward to the last stable state,
        // and it emits nothing. A dark frame must not invent a distraction.
        //
        // Every fixture here has a **face** in it, which is what makes the floor
        // the rule that applies: since R53 a zero-face tick is read by rule 1
        // whatever its confidence, so a dark *empty* desk is `AwayFromDesk` on
        // the count rather than carried forward. That case is pinned by
        // `away_time_does_not_accrue_distraction`.
        let mut run = Run::new();
        run.tick(0, &looking());
        assert_eq!(run.state, FocusState::Focused);

        let (state, events) = (run.state, run.tick(200, &dark()));
        assert_eq!(state, FocusState::Focused);
        assert!(events.is_empty(), "no evidence, no event: {events:?}");

        // Five seconds of below-floor *deviation*: the timer is not touched, so
        // the deviation is worth nothing to the distraction dwell.
        let dark_turned = run.span(400, 5400, 200, &dark_and_turned());
        assert!(
            dark_turned
                .iter()
                .all(|state| *state == FocusState::Focused),
            "a deviation nobody can vouch for accrues nothing: {dark_turned:?}"
        );

        // The dark stretch left no residue: the next trusted deviation serves its
        // full dwell from where it is first seen.
        let recovered = run.span(5600, 7400, 200, &turned_away());
        assert!(
            recovered
                .iter()
                .all(|state| *state == FocusState::GlancingAway),
            "1.8 s of trusted deviation: {recovered:?}"
        );
        assert_eq!(
            run.tick(7600, &turned_away()).as_slice(),
            [FocusEvent::DistractionStarted {
                at: run.at(7600),
                reason: DistractionReason::HeadDeviation,
            }],
            "and it becomes a distraction at 2.0 s"
        );
        assert_eq!(
            run.events,
            [
                FocusEvent::FocusGained { at: run.at(0) },
                FocusEvent::DistractionStarted {
                    at: run.at(7600),
                    reason: DistractionReason::HeadDeviation,
                },
            ],
            "the whole 7.6 s run produced two events, both committed"
        );
    }

    #[test]
    fn away_time_does_not_accrue_distraction() {
        // Review Focus 2. Ten minutes of empty desk: one state transition in and
        // one out, and not a single distraction event. The desk is not a
        // distraction, however long the wearer is away from it.
        let mut run = Run::new();
        run.tick(0, &looking());

        let leaving = run.span(200, 2800, 200, &empty());
        assert!(
            leaving.iter().all(|state| *state == FocusState::Focused),
            "2.6 s of an empty desk is not yet 3.0: {leaving:?}"
        );
        assert_eq!(
            run.tick(3200, &empty()).as_slice(),
            [FocusEvent::AwayStarted { at: run.at(3200) }],
            "the dwell runs from the first tick with no face, which was 200"
        );

        let gone = run.span(4200, 600_000, 1000, &empty());
        assert!(
            gone.iter().all(|state| *state == FocusState::AwayFromDesk),
            "an empty desk holds the state"
        );
        assert_eq!(
            run.events.len(),
            2,
            "ten minutes away produced no distraction event: {:?}",
            run.events
        );

        // The return is cheap — 0.5 s — and reports the whole time away.
        let returning = run.span(600_200, 600_600, 200, &looking());
        assert!(
            returning
                .iter()
                .all(|state| *state == FocusState::AwayFromDesk),
            "0.4 s back is under the 0.5 s exit dwell: {returning:?}"
        );
        assert_eq!(
            run.tick(600_700, &looking()).as_slice(),
            [
                FocusEvent::AwayEnded {
                    at: run.at(600_700),
                    duration: Duration::from_millis(597_500),
                },
                FocusEvent::FocusGained {
                    at: run.at(600_700),
                },
            ],
            "597.5 s away, reported as the time the state was current"
        );
    }

    #[test]
    fn no_faces_at_all_is_away_from_desk_but_losing_only_the_primary_is_other_present() {
        // Rule 1, and the handoff constraint behind it: presence is the
        // **count**. R53 is pinned in the first half — this tick is under the
        // floor, and it still reaches `AwayFromDesk` — and it also settles what a
        // carried-forward phone verdict can do with nobody in frame.
        let mut away = Run::new();
        away.tick(0, &looking());
        let claiming = away.span(200, 2800, 200, &empty_claiming_weakly(PhoneSignal::InHand));
        assert!(
            claiming.iter().all(|state| *state == FocusState::Focused),
            "2.6 s of an empty desk is not yet 3.0, whatever the confidence: {claiming:?}"
        );
        assert_eq!(
            away.tick(3200, &empty_claiming_weakly(PhoneSignal::InHand))
                .as_slice(),
            [FocusEvent::AwayStarted { at: away.at(3200) }],
            "the count is read before the floor, so a weak tick still gets there"
        );
        assert!(
            !away
                .events
                .iter()
                .any(|event| matches!(event, FocusEvent::PhoneUseStarted { .. })),
            "nobody is on the phone at an empty desk: {:?}",
            away.events
        );
        // The same claim at full confidence is the same story, and the state is
        // reached on the same tick: what decides is the count.
        let mut trusted = Run::new();
        trusted.tick(0, &looking());
        trusted.span(200, 2800, 200, &empty_claiming(PhoneSignal::InHand));
        assert_eq!(
            trusted
                .tick(3200, &empty_claiming(PhoneSignal::InHand))
                .as_slice(),
            [FocusEvent::AwayStarted {
                at: trusted.at(3200),
            }]
        );

        // The second half: two faces and **no primary reading**, which is what
        // Task 14's selector returns when the role is held by a face that is not
        // on screen. Keyed on `primary.is_none()` this would be `AwayFromDesk`
        // after 3.0 s; keyed on the count it is two people, and `OtherPresent`
        // after its own five.
        let mut crowded = Run::new();
        crowded.tick(0, &looking());
        let two_faces = crowded.span(200, 5000, 200, &crowd(2, None));
        assert!(
            two_faces.iter().all(|state| *state == FocusState::Focused),
            "2.4 s past the away dwell and the state is not away: {two_faces:?}"
        );
        assert!(
            two_faces
                .iter()
                .all(|state| *state != FocusState::AwayFromDesk),
            "losing the primary face is not losing the people: {two_faces:?}"
        );
        assert_eq!(
            crowded.tick(5200, &crowd(2, None)).as_slice(),
            [FocusEvent::PeopleChanged { count: 2 }],
            "five seconds of a second face is OtherPresent"
        );
        assert_eq!(crowded.state, FocusState::OtherPresent);
        assert_eq!(
            crowded.events.len(),
            2,
            "one FocusGained and one committed population change: {:?}",
            crowded.events
        );
    }

    #[test]
    fn other_present_holds_until_its_exit_dwell_and_reports_the_committed_population() {
        // A colleague is in the room and the wearer looks away: the deviation is
        // the higher-priority claim, but `OtherPresent` is left on its own terms,
        // and `other_present_exit` is three seconds of it.
        let mut yield_to_distraction = Run::new();
        yield_to_distraction.tick(0, &looking());
        yield_to_distraction.span(200, 5000, 200, &crowd(2, Some(awake())));
        assert_eq!(
            yield_to_distraction
                .tick(5200, &crowd(2, Some(awake())))
                .as_slice(),
            [FocusEvent::PeopleChanged { count: 2 }],
            "the colleague is committed at 5.0 s"
        );

        let deviating = crowd(2, Some(signals(40.0, 0.0, 0.30, Some(false))));
        let waiting = yield_to_distraction.span(5400, 10_200, 200, &deviating);
        assert!(
            waiting
                .iter()
                .all(|state| *state == FocusState::OtherPresent),
            "the deviation is 2.4 s old at 10200 and Distracted still waits for the \
             exit dwell: {waiting:?}"
        );
        assert_eq!(
            yield_to_distraction.tick(10_400, &deviating).as_slice(),
            [FocusEvent::DistractionStarted {
                at: yield_to_distraction.at(10_400),
                reason: DistractionReason::HeadDeviation,
            }],
            "exactly 3.0 s after the deviation became eligible"
        );
        assert!(
            !yield_to_distraction
                .events
                .iter()
                .any(|event| matches!(event, FocusEvent::PeopleChanged { count: 1 })),
            "the second person never left, so the population never changed: {:?}",
            yield_to_distraction.events
        );

        // And the way out: the colleague leaves, the wearer stays. The exit dwell
        // is three seconds, and the committed change reports the count that was
        // observed on the tick that committed it.
        let mut leaving = Run::new();
        leaving.tick(0, &looking());
        leaving.span(200, 5000, 200, &crowd(2, Some(awake())));
        leaving.tick(5200, &crowd(2, Some(awake())));
        assert_eq!(leaving.state, FocusState::OtherPresent);

        let alone = leaving.span(5400, 8200, 200, &looking());
        assert!(
            alone.iter().all(|state| *state == FocusState::OtherPresent),
            "2.8 s alone is under the 3.0 s exit dwell: {alone:?}"
        );
        assert_eq!(
            leaving.tick(8400, &looking()).as_slice(),
            [
                FocusEvent::PeopleChanged { count: 1 },
                FocusEvent::FocusGained {
                    at: leaving.at(8400),
                },
            ],
            "the population change and the focus are both committed on this tick"
        );
        assert_eq!(
            leaving.events,
            [
                FocusEvent::FocusGained { at: leaving.at(0) },
                FocusEvent::PeopleChanged { count: 2 },
                FocusEvent::PeopleChanged { count: 1 },
                FocusEvent::FocusGained {
                    at: leaving.at(8400),
                },
            ]
        );
    }

    #[test]
    fn drowsy_requires_ear_and_classifier_to_agree() {
        // Spec §7.2: "EAR below the profile threshold *and* the eye classifier
        // reporting closed, both held for the entry dwell". Either signal alone
        // is not enough, and this is the deliberate precision trade D6 asks for.
        let mut agreed = Run::new();
        agreed.tick(0, &eyes(0.10, Some(true)));
        let waiting = agreed.span(200, 800, 200, &eyes(0.10, Some(true)));
        assert!(
            waiting.iter().all(|state| *state == FocusState::Focused),
            "0.8 s of agreement is not 1.0: {waiting:?}"
        );
        assert_eq!(
            agreed.tick(1000, &eyes(0.10, Some(true))).as_slice(),
            [FocusEvent::DrowsyStarted {
                at: agreed.at(1000)
            }],
            "the entry dwell is 1.0 s"
        );

        let waking = agreed.span(1200, 1600, 200, &looking());
        assert!(
            waking.iter().all(|state| *state == FocusState::Drowsy),
            "0.4 s awake by the ratio is under the 0.5 s exit dwell: {waking:?}"
        );
        assert_eq!(
            agreed.tick(1700, &looking()).as_slice(),
            [
                FocusEvent::DrowsyEnded {
                    at: agreed.at(1700),
                    duration: Duration::from_millis(700),
                },
                FocusEvent::FocusGained {
                    at: agreed.at(1700),
                },
            ],
            "the exit dwell is 0.5 s, and the duration is the time in the state"
        );

        // Each signal alone, held for 2.4 s — more than twice the entry dwell, so
        // a machine that timed either one on its own would have entered `Drowsy`
        // twice over.
        for (ear, eye_closed, what) in [
            (0.10, Some(false), "the ratio alone"),
            (0.30, Some(true), "the classifier alone"),
            (0.10, None, "the ratio with no eye crop, so no confirmation"),
        ] {
            let mut run = Run::new();
            let alone = run.span(0, 2400, 200, &eyes(ear, eye_closed));
            assert!(
                alone.iter().all(|state| *state == FocusState::Focused),
                "{what} is not drowsiness: {alone:?}"
            );
            assert_eq!(
                run.events,
                [FocusEvent::FocusGained { at: run.at(0) }],
                "{what} emits no drowsy event"
            );
        }

        // Drowsy outranks Distracted, which is the order §7.2 states. The
        // wearer is out of bounds *and* drowsy from the first tick: the glance is
        // immediate and the agreement needs its second, so the sequence is a
        // glance, then `Drowsy` — and the deviation, which is 2.0 s old at 2000,
        // never becomes `Distracted` because `Drowsy` outranks it.
        let mut both = Run::new();
        let drowsy_and_turned = tick(
            1,
            Some(signals(40.0, 0.0, 0.10, Some(true))),
            PhoneSignal::None,
            1.0,
        );
        let held = both.span(0, 2400, 200, &drowsy_and_turned);
        assert!(
            held[..5]
                .iter()
                .all(|state| *state == FocusState::GlancingAway),
            "the glance is immediate and the agreement is not: {held:?}"
        );
        assert!(
            held[5..].iter().all(|state| *state == FocusState::Drowsy),
            "Drowsy is entered at 1.0 s and holds: {held:?}"
        );
        assert_eq!(
            both.events,
            [FocusEvent::DrowsyStarted { at: both.at(1000) }],
            "a deviating, drowsy wearer is drowsy, not distracted: {:?}",
            both.events
        );
    }

    #[test]
    fn a_glance_entry_threshold_holds_the_glance_back() {
        // `glance_entry` is 0.0 s in the spec, which makes the glance immediate
        // — but a threshold that nothing reads is not a threshold. Tuned to
        // 400 ms it must hold the glance back, and the state must be *held*, not
        // reported as something else.
        let thresholds = Thresholds {
            glance_entry: Duration::from_millis(400),
            ..Thresholds::default()
        };
        let mut run = Run::on(thresholds, base());
        run.tick(0, &looking());

        let early = run.span(1000, 1200, 200, &turned_away());
        assert!(
            early.iter().all(|state| *state == FocusState::Focused),
            "the deviation is not yet old enough to be reported: {early:?}"
        );
        assert_eq!(
            run.tick(1400, &turned_away()).len(),
            0,
            "reporting a glance is not itself an event"
        );
        assert_eq!(run.state, FocusState::GlancingAway);
        assert_eq!(
            run.events,
            [FocusEvent::FocusGained { at: run.at(0) }],
            "no event either way"
        );
    }

    #[test]
    fn identical_observation_sequences_produce_identical_event_sequences() {
        // Determinism gate (spec §11.3): the same observations on the same clock
        // produce the same states and the same events, instants included. Two
        // machines share one base so that the comparison is instant for instant,
        // and the script is long enough to exercise every rule in the module —
        // including the ones that emit nothing.
        let script: Vec<(u64, Observation)> = {
            let mut script = Vec::new();
            let mut push = |from: u64, to: u64, obs: Observation| {
                let mut ms = from;
                while ms <= to {
                    script.push((ms, obs));
                    ms += 200;
                }
            };
            push(0, 600, looking());
            push(800, 3200, turned_away());
            push(3400, 7000, facing(40.0, PhoneSignal::InHand));
            push(7200, 11_200, empty());
            push(11_400, 16_400, crowd(2, Some(awake())));
            push(16_600, 17_100, dark());
            push(17_300, 20_300, looking());
            script
        };

        let drive = |base: Instant| {
            let mut run = Run::on(Thresholds::default(), base);
            for (ms, obs) in &script {
                run.tick(*ms, obs);
            }
            (run.states, run.events)
        };

        let base = base();
        let (states, events) = drive(base);
        let (again_states, again_events) = drive(base);

        assert_eq!(events, again_events, "identical inputs, identical events");
        assert_eq!(states, again_states, "identical inputs, identical states");
        assert_eq!(
            events.len(),
            11,
            "the script visits every state the module can reach but Starting: {:?}",
            events
        );

        let mut kinds: Vec<&str> = events.iter().map(kind).collect();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(
            kinds,
            [
                "away_ended",
                "away_started",
                "distraction_ended",
                "distraction_started",
                "focus_gained",
                "people_changed",
                "phone_use_ended",
                "phone_use_started",
            ],
            "the comparison above is only worth making if the script is not two \
             empty sequences: {:?}",
            events
        );
    }

    #[test]
    fn the_starting_thresholds_are_the_specs_numbers() {
        // Spec §7.2's table, field by field, and the plan's step 3. The
        // behavioural tests above pin the numbers they read; this pins the ones
        // they do not (the latch and the floor are read only through the states
        // that need them, and the profile-relative fields have no profile yet).
        let t = Thresholds::default();
        assert_eq!(t.glance_entry, Duration::ZERO, "immediate");
        assert_eq!(t.distracted_entry, Duration::from_millis(2000));
        assert_eq!(t.distracted_exit, Duration::from_millis(600));
        assert_eq!(t.phone_entry, Duration::from_millis(2000));
        assert_eq!(t.phone_latch, Duration::from_millis(3000));
        assert_eq!(t.phone_exit, Duration::from_millis(800));
        assert_eq!(t.drowsy_entry, Duration::from_millis(1000));
        assert_eq!(t.drowsy_exit, Duration::from_millis(500));
        assert_eq!(t.away_entry, Duration::from_millis(3000));
        assert_eq!(t.away_exit, Duration::from_millis(500));
        assert_eq!(t.other_present_entry, Duration::from_millis(5000));
        assert_eq!(t.other_present_exit, Duration::from_millis(3000));
        assert_eq!(t.confidence_floor, 0.6);

        // The profile-relative five, which only Task 18 replaces.
        assert_eq!(t.neutral_yaw, 0.0);
        assert_eq!(t.neutral_pitch, 0.0);
        assert_eq!(t.yaw_bound, 20.0, "§5.4's 15–20°, at the precise end");
        assert_eq!(t.pitch_bound, 20.0);
        assert_eq!(
            t.ear_threshold, 0.2,
            "§5.2: a few tenths open, towards 0 shut"
        );
    }
}
