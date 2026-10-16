//! Primary-face selection: which of the faces in a frame the rest of the
//! engine is looking at.
//!
//! Everything downstream is derived from **one** face — the head pose, the eye
//! crops, the head region a phone is judged against — so this module answers a
//! question the pipeline cannot avoid: when several people are in frame, which
//! one is the wearer?
//!
//! The rule is the spec's: the **largest box**, ties broken by proximity to the
//! frame centre. On its own, that rule oscillates. A detector's box area jitters
//! by a few per cent from frame to frame, and two people of similar apparent
//! size trade the top rank several times a second; the primary face then flips,
//! and because the primary face feeds the head pose, the flip reaches the state
//! machine as a head that keeps turning — which the user hears as beeping. So
//! selection is **sticky**: the incumbent keeps the role unless a challenger is
//! *clearly* larger, by the hysteresis factor `HYSTERESIS`.
//!
//! # The state, and its update rule
//!
//! A selector holds one thing across calls: the box of the face it chose last
//! time. `select` is handed a fresh slice with no identity attached to any box,
//! so the incumbent is recognised by **geometry** — by **overlap** while the
//! face is still where it was, and by **proximity** when it has moved too far in
//! one tick for overlap to say anything. Slice *order* is never identity:
//! `YuNet::detect` returns faces strongest-first and the scores jitter every
//! frame, so index 0 this frame is routinely a different person next frame.
//!
//! In full, one call does this:
//!
//! 1. an empty slice answers `None` and leaves the incumbent alone — a detector
//!    that misses for a tick has not said the wearer left, and `AwayFromDesk`
//!    is the state machine's judgement (it counts faces, with its own dwell);
//! 2. the leader is the face the ranking puts first: largest area, ties to the
//!    face nearer the frame centre, ties to the lower index. A box with no
//!    usable geometry is not ranked at all, and a slice with no usable face has
//!    no primary;
//! 3. the incumbent is **continued** by the best-overlapping face, when one
//!    overlaps its remembered box by at least `INCUMBENT_IOU`, and otherwise by
//!    the face nearest that box's centre, when it is within `CONTINUATION_REACH`
//!    box-sides of it;
//! 4. a continuing face keeps the role unless the leader is a different face and
//!    clearly larger — by `HYSTERESIS` — than **both** its own area now and the
//!    area the role was held with last tick, so a face that momentarily shrinks
//!    cannot lower the bar it has to clear;
//! 5. with no continuation the role is **held, not free**: only a face clearly
//!    larger than the remembered box takes it, and otherwise this tick has no
//!    primary to report;
//! 6. the remembered box becomes the primary's box, so it tracks that face
//!    rather than drifting.
//!
//! # Where the role changes hands, in numbers
//!
//! For two 100-pixel boxes shifted `d` pixels apart the IoU is
//! `(100 - d) / (100 + d)`: 53 px gives 0.307 and keeps the overlap match, 54 px
//! gives 0.2987 and does not. So the *match* ends at about half a box width of
//! motion per tick — and the *role* does not. From there to `CONTINUATION_REACH`
//! (one box side) the nearest face continues it, and past that it is held.
//! Measured at a desk and 5 Hz, one box side per tick is roughly half a metre
//! per second of head motion: a lean costs the wearer nothing, and a move fast
//! enough to break even that keeps the role rather than giving it away.
//!
//! Holding the role is the deliberate answer to an ambiguity geometry cannot
//! settle: a face that has left the reach band may be the wearer, who moved very
//! fast and settled, or someone else, who walked in after the wearer left. The
//! two look identical. This module picks the answer that does not attribute a
//! stranger's head pose to the wearer — no primary until something is clearly
//! larger — and the state machine reads a present-`None` primary as
//! `OtherPresent`, which is what the spec prescribes when the primary face is
//! lost while faces remain. `AwayFromDesk` needs *zero* faces.
//!
//! # What the rule reads, and what it does not
//!
//! Only `FaceBox::rect`. Neither the detector's confidence nor the landmarks
//! enter the ranking: the spec's rule is "largest box", and a confidence term
//! would let a well-lit colleague outrank the wearer at the desk. Landmarks are
//! Face Mesh's input, not this decision's.
//!
//! The one thing the rule needs that a slice of boxes does not carry is the
//! **frame centre** the tie-break measures from. The interface the plan fixes —
//! `new()` with no frame — leaves the selector to assume one, so
//! `PrimarySelector::new` uses the crate's reference webcam frame, 640x480 (the
//! size `capture::synthetic` generates and every fixture uses), and
//! `PrimarySelector::from_frame_size` is the honest constructor for any other
//! size. The assumption can only ever change which of two *equal-area* faces
//! wins, which is the rarest path through this module.
//!
//! Nothing here allocates, reads a clock, or touches the filesystem: at 5 Hz
//! this runs 200 times a minute, and the whole selector is a box and a point.

use crate::analyze::nms::Rect;
use crate::models::yunet::FaceBox;

/// How much more area than the incumbent a challenger must have to take the
/// primary role from it.
///
/// `1.25` is the spec's figure, and it is a **hysteresis**: the incumbent is
/// kept for anything up to a quarter larger, which is far wider than the few
/// per cent a detector's box area wanders between frames and far narrower than
/// the several-fold jump of a second person leaning into frame. The comparison
/// against it is strict, so a challenger of exactly this multiple does not take
/// the role — a tie goes to stability.
const HYSTERESIS: f32 = 1.25;

/// How much of the incumbent's last box a face must still overlap to be taken
/// for that same face.
///
/// For two boxes of the same size that is a shift of just over half a box
/// width — 53 pixels for a 100-pixel box, whose IoU is `(100 - d) / (100 + d)`.
/// Motion past it is not a lost identity, only a lost *overlap*:
/// `CONTINUATION_REACH` takes over from there, so this constant decides which
/// of the two mechanisms recognises the wearer, not whether they are
/// recognised. A lower bar would let the role travel to a stranger standing
/// beside the wearer; a higher one would leave more of an ordinary lean to the
/// weaker, purely positional rule.
const INCUMBENT_IOU: f32 = 0.3;

/// How far a face may be from the incumbent's last box and still continue it,
/// in units of that box's longer side.
///
/// The fallback for motion too fast for `INCUMBENT_IOU`: a face that was in one
/// place last tick and is a box width away this tick is still best explained as
/// that same face moving, and the alternative — treating it as gone — would drop
/// the primary on any quick lean. At 5 Hz one box side per tick is about half a
/// metre per second of head motion at a desk, well past a lean and short of
/// someone walking out of frame. Past it the role is held rather than given
/// away: see the module docs for why that is the honest answer and not a
/// transfer.
const CONTINUATION_REACH: f32 = 1.0;

/// The frame size [`PrimarySelector::new`] assumes, in pixels: the reference
/// webcam frame the synthetic source generates and every fixture uses.
const REFERENCE_FRAME: (u32, u32) = (640, 480);

/// Picks the primary face, holding the previous choice across calls.
///
/// See the module docs for the rule and the update rule; this type exists so
/// that a frame-to-frame identity survives the fresh slice each call arrives
/// with.
#[derive(Debug)]
pub struct PrimarySelector {
    /// The box of the face this selector chose on its last call that had a
    /// primary, in frame coordinates. `None` until then, and unchanged by a
    /// call with no faces, a call with nothing usable to rank, and a call whose
    /// role is held by a face that is not on screen.
    incumbent: Option<Rect>,
    /// The point "proximity to the frame centre" is measured from.
    centre: (f32, f32),
}

impl PrimarySelector {
    /// A selector for the crate's reference frame, 640x480.
    ///
    /// The plan's interface is `new()` with no frame, so the frame centre the
    /// tie-break needs has to come from somewhere: this constructor takes the
    /// reference webcam size. It is a **guess about the frame, not a
    /// measurement of it** — a caller whose frames are another size wants
    /// [`PrimarySelector::from_frame_size`], and the cost of getting it wrong is
    /// confined to which of two *equal-area* faces wins.
    pub fn new() -> PrimarySelector {
        PrimarySelector::from_frame_size(REFERENCE_FRAME.0, REFERENCE_FRAME.1)
    }

    /// A selector for frames of `width` by `height` pixels.
    ///
    /// The tie-break measures from the exact centre of such a frame — the
    /// principal point this crate already assumes, in `CameraIntrinsics`, is the
    /// frame's centre. A frame with no pixels has its centre at the origin
    /// rather than a division by zero.
    pub fn from_frame_size(width: u32, height: u32) -> PrimarySelector {
        PrimarySelector {
            incumbent: None,
            centre: (width as f32 / 2.0, height as f32 / 2.0),
        }
    }

    /// Chooses the primary face in `faces`, returning its index in that slice.
    ///
    /// `None` means this tick has no primary, and there are three ways to get
    /// there: no faces at all, no face whose box has usable geometry, and a role
    /// held by a face that is not on screen. In all three the incumbent is left
    /// exactly as it was, so the next frame that does have a primary is matched
    /// against the last box the selector believed in.
    ///
    /// The index is a position in the slice that was passed, not an identity:
    /// the caller re-derives the face from it.
    pub fn select(&mut self, faces: &[FaceBox]) -> Option<usize> {
        // No faces is not a decision. The incumbent stays where it is, because
        // one missed detection is not the wearer leaving the desk.
        // `AwayFromDesk` is the state machine's to judge — it counts faces and
        // applies its own dwell — and a selector that cleared itself on every
        // dropout would hand the role to whoever else is in frame the first time
        // the wearer blinked.
        if faces.is_empty() {
            return None;
        }

        // Nothing usable to rank is the same kind of non-decision: a slice of
        // boxes whose geometry is not a number cannot produce a primary, and it
        // must not clear the role the previous frame established.
        let leader = self.top_ranked(faces)?;

        let chosen = match self.continuation(faces) {
            // A face continues the incumbent. It keeps the role unless the
            // leader is a *different* face and clearly larger than the area the
            // role is held with — the larger of that face's area now and the
            // remembered box's, so a face which momentarily shrinks cannot lower
            // the bar it has to clear. A challenger of exactly `HYSTERESIS`
            // times that area does not take it: ties go to stability.
            Some((holder, remembered)) => {
                let bar = area(&faces[holder].rect).max(area(&remembered));
                if holder == leader || area(&faces[leader].rect) <= bar * HYSTERESIS {
                    holder
                } else {
                    leader
                }
            }
            // No face in this slice is plausibly the incumbent, so the role is
            // held by a face that is not on screen. It is **not** free: only a
            // face clearly larger than the box holding it takes the role, and
            // otherwise there is no primary to report this tick.
            None => match self.incumbent {
                Some(held) if area(&faces[leader].rect) <= area(&held) * HYSTERESIS => return None,
                _ => leader,
            },
        };

        // The remembered box becomes the chosen face's *current* box, so it
        // follows that face as it moves. An empty slice, an unusable slice and a
        // held role all return before this.
        self.incumbent = Some(faces[chosen].rect);
        Some(chosen)
    }

    /// The face that continues the incumbent, and the box the role is held
    /// with, if a face does.
    ///
    /// Overlap first: the face overlapping the remembered box **most**,
    /// provided that overlap is at least `INCUMBENT_IOU`. The best overlap
    /// rather than the first one over the bar, because a detection that splits
    /// in two would otherwise hand the role to whichever half the detector
    /// happened to list first, and the slice's order is not identity.
    ///
    /// With no overlap worth the name — the face moved more than half its own
    /// width in one tick — proximity takes over: the face nearest the remembered
    /// box's centre, provided it is within `CONTINUATION_REACH` box-sides of it.
    /// A stranger across the frame is not within reach, and neither is a box
    /// whose geometry has no centre.
    fn continuation(&self, faces: &[FaceBox]) -> Option<(usize, Rect)> {
        let remembered = self.incumbent?;

        let mut overlapped: Option<(usize, f32)> = None;
        let mut nearest: Option<(usize, f32)> = None;
        for (index, face) in faces.iter().enumerate() {
            if !usable(&face.rect) {
                continue;
            }

            let overlap = remembered.iou(&face.rect);
            let better = match overlapped {
                None => true,
                Some((_, best)) => overlap > best,
            };
            // A NaN overlap fails the threshold and is skipped, which is what a
            // box with no usable geometry deserves; a face that is not the
            // incumbent must never inherit the role by accident.
            if overlap >= INCUMBENT_IOU && better {
                overlapped = Some((index, overlap));
            }

            let separation = separation_squared(&remembered, &face.rect);
            let closer = match nearest {
                None => true,
                Some((_, best)) => separation < best,
            };
            if closer {
                nearest = Some((index, separation));
            }
        }

        if let Some((index, _)) = overlapped {
            return Some((index, remembered));
        }
        let reach = CONTINUATION_REACH * remembered.w.max(remembered.h);
        nearest
            .filter(|(_, separation)| *separation <= reach * reach)
            .map(|(index, _)| (index, remembered))
    }

    /// The index of the face the ranking puts first, or `None` when no face has
    /// geometry the ranking can read.
    fn top_ranked(&self, faces: &[FaceBox]) -> Option<usize> {
        let mut leader: Option<usize> = None;
        for (index, face) in faces.iter().enumerate() {
            if !usable(&face.rect) {
                continue;
            }
            let outranks = match leader {
                Some(best) => self.outranks(face, &faces[best]),
                None => true,
            };
            if outranks {
                leader = Some(index);
            }
        }
        leader
    }

    /// Whether `candidate` outranks `leader`: more area first, then nearer the
    /// frame centre, and otherwise the earlier index, because a face that
    /// outranks nothing keeps the place it was found in.
    ///
    /// Both boxes have been through `usable`, so both areas and both distances
    /// are real numbers and neither comparison can be swallowed by a NaN.
    fn outranks(&self, candidate: &FaceBox, leader: &FaceBox) -> bool {
        let candidate_area = area(&candidate.rect);
        let leader_area = area(&leader.rect);
        if candidate_area != leader_area {
            return candidate_area > leader_area;
        }
        // Equal areas — exactly equal, which a detector produces for boxes of
        // the same pixel size. The nearer centre wins, and equal distances fall
        // through to the index the caller above started from.
        self.distance_squared(&candidate.rect) < self.distance_squared(&leader.rect)
    }

    /// Squared Euclidean distance from `rect`'s centre to the frame centre.
    ///
    /// Squared because only the *order* of two of these is ever read: the square
    /// root cannot change which is smaller, and at 5 Hz it is time spent for
    /// nothing.
    fn distance_squared(&self, rect: &Rect) -> f32 {
        let (x, y) = rect.center();
        let dx = x - self.centre.0;
        let dy = y - self.centre.1;
        dx * dx + dy * dy
    }
}

impl Default for PrimarySelector {
    /// The reference-frame selector [`PrimarySelector::new`] builds.
    fn default() -> PrimarySelector {
        PrimarySelector::new()
    }
}

/// Whether a box's geometry is a box: finite coordinates, non-negative extents.
///
/// The same invariants `YuNet::decode` states in its own `debug_assert`, and a
/// box that violates them is not ranked, not matched and not continued no matter
/// where in the slice it sits. It is reachable in principle: `clamp` propagates
/// a NaN rather than rejecting it, so a decode handed a garbage tensor produces
/// rects with no usable geometry, and every crop derived from one would be
/// garbage in turn. Filtering here is what makes the ranking total.
fn usable(rect: &Rect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.w.is_finite()
        && rect.h.is_finite()
        && rect.w >= 0.0
        && rect.h >= 0.0
}

/// A box's area in square frame pixels.
///
/// "Largest box" means this and not the longer side: a 130x130 face is bigger
/// than a 90x160 one, and a rule that compared the longest edge would say the
/// opposite. Every box that reaches a comparison has been through `usable`, so
/// both extents are finite and non-negative, and this is never a negative or a
/// not-a-number area that could win by a double negation.
fn area(rect: &Rect) -> f32 {
    rect.w * rect.h
}

/// Squared distance between two boxes' centres.
///
/// Squared because only the *order* of two of these is ever read, and because
/// the reach test compares it against a squared reach.
fn separation_squared(a: &Rect, b: &Rect) -> f32 {
    let ((ax, ay), (bx, by)) = (a.center(), b.center());
    let (dx, dy) = (ax - bx, ay - by);
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A face with the box a test chose, whatever that geometry is.
    ///
    /// The landmarks are zeroes and the confidence is uniform: the selector
    /// ranks boxes, so a fixture that varied either would be testing a term the
    /// rule does not have.
    fn any_face(rect: Rect) -> FaceBox {
        FaceBox {
            rect,
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        }
    }

    /// A face whose box is `w` by `h` with its top-left at `(x, y)`.
    fn box_face(x: f32, y: f32, w: f32, h: f32) -> FaceBox {
        any_face(Rect { x, y, w, h })
    }

    /// A face whose box is a `size`-square with its top-left at `(x, y)`.
    fn face(x: f32, y: f32, size: f32) -> FaceBox {
        box_face(x, y, size, size)
    }

    #[test]
    fn largest_face_wins_when_sizes_differ_clearly() {
        // The small face is the better-placed of the two: its centre is
        // (290, 240), 30 px from the reference frame centre (320, 240), where
        // the large one's is (500, 180), 180 px across and 60 px down. It is
        // also listed first, and the detector is more confident about it, and it
        // loses anyway, because the rule is "largest box" and nothing else in
        // this module outranks area.
        let small = FaceBox {
            score: 0.99,
            ..face(250.0, 200.0, 80.0)
        };
        let large = FaceBox {
            score: 0.80,
            ..face(420.0, 100.0, 160.0)
        };
        assert_eq!(area(&small.rect), 6400.0);
        assert_eq!(area(&large.rect), 25600.0, "four times the small face");
        assert_eq!(small.rect.center(), (290.0, 240.0));
        assert_eq!(large.rect.center(), (500.0, 180.0));

        // The premise, so that "the large box won" is a statement about area:
        // 900 is nearer the frame centre than 36000, so a rule that preferred
        // the centre — or the confidence — would answer 0.
        let to_centre = |rect: &Rect| {
            let (x, y) = rect.center();
            (x - 320.0).powi(2) + (y - 240.0).powi(2)
        };
        assert_eq!(to_centre(&small.rect), 900.0);
        assert_eq!(to_centre(&large.rect), 36000.0);

        let mut selector = PrimarySelector::new();
        assert_eq!(
            selector.select(&[small, large]),
            Some(1),
            "the 160 px box is at index 1 and wins on area"
        );

        // The other order, on a fresh selector: the same box wins, so the choice
        // is a property of the geometry and not of the position. The pair is
        // what rules out an index preference — "the first face wins" fails the
        // assertion above, "the last face wins" fails this one.
        let mut selector = PrimarySelector::new();
        let chosen = selector.select(&[large, small]).expect("a primary");
        assert_eq!(
            [large, small][chosen].rect,
            large.rect,
            "the same box, listed first this time"
        );
    }

    #[test]
    fn nearer_to_centre_wins_when_sizes_are_equal() {
        // Both boxes are 80x80 — 6400 px² each — so area cannot separate them
        // and the tie-break decides. Their centres are (320, 200) and
        // (560, 200): 40 px above the reference frame's centre (320, 240), and
        // 240 px to its right.
        let near = face(280.0, 160.0, 80.0);
        let far = face(520.0, 160.0, 80.0);
        assert_eq!(area(&near.rect), area(&far.rect), "equal areas");
        assert_eq!(near.rect.center(), (320.0, 200.0));
        assert_eq!(far.rect.center(), (560.0, 200.0));

        // The nearer box is listed second, so "first in the slice" is not the
        // answer: index 1 is the box 40 px from the centre.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[far, near]), Some(1));

        // The other order, on a fresh selector, and the weaker half of the pair:
        // on its own this assertion is satisfied by an area-only rule that
        // breaks ties towards the earlier face, which is why the assertion above
        // exists. Together they leave the tie-break as the only rule that
        // answers both.
        let mut selector = PrimarySelector::new();
        let chosen = selector.select(&[near, far]).expect("a primary");
        assert_eq!(
            [near, far][chosen].rect,
            near.rect,
            "the same box, listed first this time"
        );
    }

    #[test]
    fn primary_face_selection_is_stable_across_swaps() {
        // Review Focus 4, and the reason this module holds state at all: two
        // people of similar apparent size, with the detector's box area
        // jittering the way it does every frame. `b` is 92² = 8464 px² against
        // `a`'s 90² = 8100 — 4.5% more area, inside the hysteresis window — so
        // the role must not change hands however the two are ordered.
        let a = face(100.0, 100.0, 90.0);
        let b = face(300.0, 100.0, 92.0);
        assert!(
            area(&b.rect) > area(&a.rect),
            "the premise: b really is the larger box, so `a` keeping the role \
             is the hysteresis at work and not a smaller challenger losing"
        );

        // The wearer alone: `a` takes the role.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[a]), Some(0));

        // The colleague appears — the larger box — and does not steal it.
        assert_eq!(selector.select(&[a, b]), Some(0), "a is at index 0 here");
        // The same frame with the two boxes in the other order: still `a`,
        // which is at index 1 in this slice.
        assert_eq!(selector.select(&[b, a]), Some(1), "a is at index 1 here");
        assert_eq!(selector.select(&[a, b]), Some(0));

        // Frames of detector jitter around the same two people, in both orders,
        // with `b` staying inside the hysteresis window of `a`'s area. The
        // primary is `a`'s index in every slice and never alternates.
        for side in [89.0, 91.0, 88.5, 92.5, 90.5] {
            let rival = face(300.0, 100.0, side);
            assert!(
                area(&rival.rect) < area(&a.rect) * 1.25,
                "{side} px is inside the hysteresis window, so this frame asks \
                 whether the incumbent survives, not whether a giant appeared"
            );
            assert_eq!(selector.select(&[a, rival]), Some(0), "a is at index 0");
            assert_eq!(selector.select(&[rival, a]), Some(1), "a is at index 1");
        }

        // A challenger that is *clearly* larger does take the role: 200² =
        // 40000 px² is 4.9x `a`, far outside the window.
        let big = face(300.0, 100.0, 200.0);
        assert_eq!(selector.select(&[a, big]), Some(1), "big is at index 1");

        // And once `big` holds the role it keeps it. The next challenger is
        // 210² = 44100, a mere 1.10x `big`, and it is listed *first*, so a
        // selector that had not remembered `big` would answer 0 here.
        let nearly_big = face(300.0, 100.0, 210.0);
        assert!(area(&nearly_big.rect) > area(&big.rect));
        assert!(area(&nearly_big.rect) < area(&big.rect) * 1.25);
        assert_eq!(
            selector.select(&[nearly_big, big]),
            Some(1),
            "big is at index 1 here"
        );
        // And the other way round, where the incumbent is the larger of the two
        // and only an area-blind rule could answer 1.
        assert_eq!(
            selector.select(&[big, nearly_big]),
            Some(0),
            "big is at index 0 here"
        );
    }

    #[test]
    fn the_largest_area_wins_not_the_longest_side_or_the_centre() {
        // A tall box against a squarer one that is nonetheless *bigger*:
        // 130² = 16900 px² against 90x160 = 14400. The tall box has the longer
        // side and sits exactly on the frame centre — its centre is (320, 240)
        // — while the squarer one is out at (95, 365). Area is the rule, so the
        // squarer box wins both signals that are not the rule.
        let tall = box_face(275.0, 160.0, 90.0, 160.0);
        let square = box_face(30.0, 300.0, 130.0, 130.0);
        assert_eq!(area(&tall.rect), 14400.0);
        assert_eq!(area(&square.rect), 16900.0, "the bigger box, by area");
        assert_eq!(
            tall.rect.center(),
            (320.0, 240.0),
            "the tall box is the centred one"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(
            selector.select(&[tall, square]),
            Some(1),
            "the square box is at index 1"
        );
    }

    #[test]
    fn a_challenger_at_exactly_the_hysteresis_factor_does_not_steal() {
        // The boundary, on both sides of it. `a` is 100x100 = 10000 px², so the
        // hysteresis factor of 1.25 puts the bar at exactly 12500: a 125x100
        // challenger is *at* the bar and must not take the role, while 125.04 is
        // over it and must. The comparison is strict on purpose — a tie goes to
        // stability, which is what this module is for.
        let a = face(100.0, 100.0, 100.0);
        let at_the_bar = box_face(300.0, 100.0, 125.0, 100.0);
        let over_the_bar = box_face(300.0, 100.0, 125.04, 100.0);
        assert_eq!(area(&a.rect), 10000.0);
        assert_eq!(
            area(&at_the_bar.rect),
            12500.0,
            "exactly 1.25x the incumbent"
        );
        assert!(
            area(&over_the_bar.rect) > 12500.0,
            "and this one is past it"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[a]), Some(0));
        assert_eq!(
            selector.select(&[a, at_the_bar]),
            Some(0),
            "at the bar, the incumbent keeps the role"
        );
        assert_eq!(
            selector.select(&[a, over_the_bar]),
            Some(1),
            "past the bar, the challenger takes it"
        );
    }

    #[test]
    fn the_incumbent_keeps_the_role_against_a_nearer_equal_area_face() {
        // The centre tie-break ranks *candidates*; it never displaces the
        // incumbent, because taking the role from a face that is still in frame
        // requires strictly more area. Both boxes are 100x100 = 10000 px²: the
        // challenger's centre is exactly the frame centre (320, 240) and the
        // incumbent's is (70, 70), out at a corner.
        let corner = face(20.0, 20.0, 100.0);
        let middle = face(270.0, 190.0, 100.0);
        assert_eq!(area(&corner.rect), area(&middle.rect));
        assert_eq!(middle.rect.center(), (320.0, 240.0));

        let mut selector = PrimarySelector::new();
        assert_eq!(
            selector.select(&[corner]),
            Some(0),
            "the wearer at a corner"
        );
        assert_eq!(
            selector.select(&[middle, corner]),
            Some(1),
            "the centred face is nearer the centre and still does not steal"
        );
    }

    #[test]
    fn the_incumbent_is_recognised_by_overlap_not_by_rank() {
        // Two faces both overlap the incumbent's last box, so which one *is*
        // the incumbent is decided by how well each matches the remembered box
        // — not by which ranks higher and not by which the slice lists first.
        //
        // Frame 1 is the wearer, a 200x200 box at (100, 100).
        //
        // Frame 2 holds two 150x150 boxes of equal area, 22500 px²:
        //   * `x` at (200, 150) overlaps the remembered box by 15000 px² —
        //     IoU 15000/47500 = 0.316 — and its centre (275, 225) is nearer the
        //     frame centre than `y`'s, so `x` is the better *rank*;
        //   * `y` at (120, 120) sits inside the remembered box, IoU
        //     22500/40000 = 0.5625, with its centre at (195, 195).
        //
        // `y` is the incumbent, because it matches the remembered box best.
        // Equal areas close the door on a steal, so the answer is index 1.
        let wearer = face(100.0, 100.0, 200.0);
        let x = face(200.0, 150.0, 150.0);
        let y = face(120.0, 120.0, 150.0);
        assert_eq!(
            area(&x.rect),
            area(&y.rect),
            "equal areas: no steal is possible"
        );
        assert_eq!(wearer.rect.iou(&y.rect), 0.5625, "y matches well");
        assert_eq!(
            wearer.rect.iou(&x.rect),
            15000.0 / 47500.0,
            "and x, while above the match threshold, matches worse"
        );
        assert!(
            wearer.rect.iou(&x.rect) >= 0.3,
            "the premise: x is above the threshold too, so only *which* match \
             is chosen can explain the answer"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[x, y]),
            Some(1),
            "y at index 1 continues the incumbent; x only outranks it"
        );
    }

    #[test]
    fn overlap_beats_proximity_and_the_bar_decides_which_applies() {
        // The two ways a face is recognised can disagree about which face it
        // is, and pinning that disagreement is what fixes the threshold itself
        // rather than only the boundaries around it.
        //
        // The wearer's remembered box is 200x200 at (100, 100): area 40000,
        // centre (200, 200). Against it:
        //   * `overlapped` — 150x150 at (200, 150) — shares 15000 px², IoU
        //     15000/47500 = 0.316, and its centre is 6250 px² away;
        //   * `near_miss` — 150x150 at (250, 150) — shares 7500 px², IoU
        //     7500/55000 = 0.136, and its centre is 16250 px² away;
        //   * `centred` — 60x60 at (170, 170) — sits exactly on the remembered
        //     box's centre, separation 0, but shares only 3600/40000 = 0.09.
        let wearer = face(100.0, 100.0, 200.0);
        let overlapped = face(200.0, 150.0, 150.0);
        let near_miss = face(250.0, 150.0, 150.0);
        let centred = face(170.0, 170.0, 60.0);
        assert_eq!(wearer.rect.iou(&overlapped.rect), 15000.0 / 47500.0);
        assert_eq!(wearer.rect.iou(&near_miss.rect), 7500.0 / 55000.0);
        assert_eq!(wearer.rect.iou(&centred.rect), 0.09);
        assert_eq!(separation_squared(&wearer.rect, &overlapped.rect), 6250.0);
        assert_eq!(separation_squared(&wearer.rect, &near_miss.rect), 16250.0);
        assert_eq!(separation_squared(&wearer.rect, &centred.rect), 0.0);
        assert!(area(&overlapped.rect) > area(&centred.rect));

        // Both candidates are above the bar, so the better *overlap* wins even
        // though the other one is exactly where the wearer was: overlapping the
        // remembered box is a stronger claim than merely sitting at its centre.
        // Raising the bar past 0.316, or putting proximity first, answers 1.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[overlapped, centred]),
            Some(0),
            "overlapped at index 0 is the better match"
        );

        // Now the best overlap is *below* the bar — 0.136 against 0.3 — so
        // nothing matches and proximity decides, keeping the role on the face at
        // the remembered centre. A bar low enough to admit 0.136 would match
        // `near_miss` instead and answer 0.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[near_miss, centred]),
            Some(1),
            "past the bar, the nearest face at index 1 continues the incumbent"
        );
    }

    #[test]
    fn a_transient_shrink_does_not_lower_the_bar() {
        // The bar a challenger must clear is the larger of the incumbent's area
        // *now* and the area the role was held with last tick. Here the wearer's
        // box shrinks from 100² = 10000 to 80² = 6400 — leaning back, still
        // overlapping the remembered box by 0.64 — while a rival holds at
        // 90² = 8100. Against the shrunken area alone the rival would take the
        // role; against the remembered area it cannot.
        let wearer = face(100.0, 100.0, 100.0);
        let shrunken = face(110.0, 110.0, 80.0);
        let rival = face(400.0, 100.0, 90.0);
        assert_eq!(wearer.rect.iou(&shrunken.rect), 0.64, "still the same face");
        assert_eq!(area(&shrunken.rect), 6400.0);
        assert_eq!(area(&rival.rect), 8100.0);
        assert!(
            area(&rival.rect) > area(&shrunken.rect) * 1.25,
            "8100 > 8000: the shrunken area alone would let the rival in"
        );
        assert!(
            area(&rival.rect) < area(&wearer.rect) * 1.25,
            "8100 < 12500: the remembered area does not"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[rival, shrunken]),
            Some(1),
            "the shrunken wearer at index 1 keeps the role"
        );
    }

    #[test]
    fn a_face_moving_a_half_box_width_keeps_the_role() {
        // Where the role changes hands, in numbers, for two 100-px boxes: the
        // IoU is (100 - d) / (100 + d), so a shift of 53 px gives 0.307 and
        // keeps the overlap match while 54 px gives 0.2987 and does not — and
        // one box side, 100 px, is the reach of the proximity fallback that
        // takes over. Each band is pinned against a rival 1.10x the wearer's
        // area, inside the hysteresis window, so "the wearer kept the role" is
        // only the answer if the wearer was recognised at all.
        let wearer = face(100.0, 100.0, 100.0);
        let rival = face(400.0, 300.0, 105.0); // 11025 = 1.1025x the wearer
        assert!(area(&rival.rect) < area(&wearer.rect) * 1.25);

        let shifted = |dx: f32| face(100.0 + dx, 100.0, 100.0);
        let inside_the_bar = shifted(53.0);
        let past_the_bar = shifted(54.0);
        let at_the_reach = shifted(100.0);
        let past_the_reach = shifted(101.0);
        // A diagonal step of (60, 60): a squared separation of 7200 is inside
        // the squared reach of 10000, while a Manhattan separation of 120 would
        // be outside it.
        let diagonal = face(160.0, 160.0, 100.0);

        assert_eq!(wearer.rect.iou(&inside_the_bar.rect), 4700.0 / 15300.0);
        assert!(wearer.rect.iou(&inside_the_bar.rect) >= 0.3);
        assert_eq!(wearer.rect.iou(&past_the_bar.rect), 4600.0 / 15400.0);
        assert!(
            wearer.rect.iou(&past_the_bar.rect) < 0.3,
            "the overlap match really has ended here"
        );
        // Separation is what the fallback reads from this point on: 54 px is
        // 0.54 of a box side and inside the reach, 101 px is not, and the
        // diagonal step is inside it on both axes at once.
        assert_eq!(separation_squared(&wearer.rect, &past_the_bar.rect), 2916.0);
        assert_eq!(
            separation_squared(&wearer.rect, &at_the_reach.rect),
            10000.0
        );
        assert_eq!(
            separation_squared(&wearer.rect, &past_the_reach.rect),
            10201.0
        );
        assert_eq!(separation_squared(&wearer.rect, &diagonal.rect), 7200.0);
        let manhattan = |rect: &Rect| {
            let ((ax, ay), (bx, by)) = (wearer.rect.center(), rect.center());
            (ax - bx).abs() + (ay - by).abs()
        };
        assert!(
            manhattan(&diagonal.rect) > 100.0,
            "the premise: a Manhattan measure would put the diagonal step \
             outside the reach, so this band pins the metric as well"
        );

        for (band, moved, expected) in [
            ("inside the overlap bar", inside_the_bar, Some(1)),
            ("past the bar, inside the reach", past_the_bar, Some(1)),
            ("exactly at the reach", at_the_reach, Some(1)),
            ("a diagonal step inside the reach", diagonal, Some(1)),
            ("past the reach, role held", past_the_reach, None),
        ] {
            let mut selector = PrimarySelector::new();
            assert_eq!(selector.select(&[wearer]), Some(0));
            assert_eq!(
                selector.select(&[rival, moved]),
                expected,
                "{band}: the rival is at index 0 and the moved wearer at index 1"
            );
        }
    }

    #[test]
    fn the_moving_incumbent_is_followed() {
        // No fixture above moves the incumbent, which is exactly what storing
        // its *current* box after every call is for: a selector that kept only
        // its first box would lose a wearer who walked across the frame, because
        // every frame is compared against the last box it stored.
        //
        // The wearer steps 40 px per frame — an IoU of 0.4286 between
        // consecutive boxes, comfortably inside the overlap bar — for six
        // frames, 240 px in all, well past the one-box-side reach. A rival 1.10x
        // its area is listed *first* every frame, so the answer is the wearer's
        // index only if the role is kept and the box is tracked.
        const STEP: f32 = 40.0;
        const STEPS: usize = 6;
        assert!(
            STEP * STEPS as f32 > 100.0,
            "the walk must exceed one box side, or a selector that never \
             updated its remembered box could follow it anyway"
        );

        let rival = face(450.0, 300.0, 105.0);
        let first = face(100.0, 100.0, 100.0);
        assert!(area(&rival.rect) < area(&first.rect) * 1.25);

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[first]), Some(0), "the wearer alone");

        let mut previous = first;
        for step in 1..=STEPS {
            let moved = face(100.0 + STEP * step as f32, 100.0, 100.0);
            assert!(
                previous.rect.iou(&moved.rect) >= 0.3,
                "step {step}: consecutive boxes overlap enough to match"
            );
            assert_eq!(
                selector.select(&[rival, moved]),
                Some(1),
                "step {step}: the wearer at index 1 still holds the role"
            );
            previous = moved;
        }
    }

    #[test]
    fn an_unmatched_tick_holds_the_role_instead_of_handing_it_to_a_stranger() {
        // The wearer leaves the frame. A colleague is present and *is* the
        // largest face, but at 105² = 11025 it is under 1.25x the wearer's
        // 10000, and it is nowhere near where the wearer was — so the role is
        // held by a face that is not on screen and this tick has no primary. A
        // rule that ranked the slice by area whenever the incumbent could not be
        // found would hand the role over here, with no hysteresis at all.
        let wearer = face(100.0, 100.0, 100.0);
        let colleague = face(450.0, 60.0, 100.0);
        let stranger = face(500.0, 300.0, 105.0);
        assert!(
            area(&stranger.rect) < area(&wearer.rect) * 1.25,
            "11025 < 12500: the bar is the *remembered* box's area"
        );
        assert_eq!(wearer.rect.iou(&stranger.rect), 0.0, "nowhere near");
        assert!(
            separation_squared(&wearer.rect, &stranger.rect) > 100.0 * 100.0,
            "and outside the reach, which is what makes it a stranger"
        );
        assert!(separation_squared(&wearer.rect, &colleague.rect) > 100.0 * 100.0);

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[stranger, colleague]),
            None,
            "the role is held: the larger face at index 0 does not take it"
        );

        // The wearer reappearing before anything takes the role gets it back,
        // because the role was held rather than given away.
        assert_eq!(
            selector.select(&[wearer]),
            Some(0),
            "the wearer is still the incumbent"
        );

        // A face that *is* clearly larger does take the held role, and then
        // keeps it like any other incumbent.
        let giant = face(500.0, 300.0, 130.0); // 16900 > 12500
        assert!(area(&giant.rect) > area(&wearer.rect) * 1.25);
        assert_eq!(
            selector.select(&[giant, colleague]),
            Some(0),
            "giant is at index 0"
        );
        assert_eq!(
            selector.select(&[colleague, giant]),
            Some(1),
            "giant is at index 1"
        );
    }

    #[test]
    fn a_face_that_takes_the_held_role_keeps_it_against_the_old_wearer() {
        // The role passes to a face that is clearly larger than the remembered
        // box: 130x100 = 13000 against 10000, over the 1.25 bar. From then on
        // that face *is* the incumbent, so a wearer who comes back is a
        // challenger like any other — and 118² = 13924 is larger than the new
        // holder's area while staying inside its 1.25 window, so the role is not
        // reclaimed. A selector with no memory of the holder would answer 0 for
        // the returning, largest face; the holder is at index 2.
        let wearer = face(100.0, 100.0, 100.0);
        let colleague = face(300.0, 300.0, 100.0);
        let successor = box_face(450.0, 300.0, 130.0, 100.0);
        let returning = face(100.0, 100.0, 118.0);
        assert!(
            area(&successor.rect) > area(&wearer.rect) * 1.25,
            "13000 > 12500: it takes the held role"
        );
        assert!(
            area(&returning.rect) > area(&successor.rect),
            "13924 > 13000: the returning wearer is the largest face"
        );
        assert!(
            area(&returning.rect) < area(&successor.rect) * 1.25,
            "13924 < 16250: and still inside the window"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[colleague, successor]),
            Some(1),
            "the successor at index 1 takes the held role"
        );
        assert_eq!(
            selector.select(&[returning, colleague, successor]),
            Some(2),
            "the successor at index 2 keeps it"
        );
    }

    #[test]
    fn an_empty_slice_returns_none_and_keeps_the_incumbent() {
        // A detector that reports no face for a tick has not told us the wearer
        // left. The state machine owns "the desk is empty" — it counts faces and
        // applies `AwayFromDesk`'s dwell — so an empty slice answers `None` and
        // leaves the incumbent where it was, rather than handing the role to
        // whoever else is in frame the moment the wearer is missed once.
        let a = face(100.0, 100.0, 90.0);
        let b = face(300.0, 100.0, 96.0);

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[a]), Some(0));
        assert_eq!(selector.select(&[]), None, "no faces, no primary");
        assert_eq!(selector.select(&[]), None, "and it stays that way");

        assert!(
            area(&b.rect) > area(&a.rect),
            "the premise: b is the larger box, 96² = 9216 against 8100"
        );
        assert!(
            area(&b.rect) < area(&a.rect) * 1.25,
            "and inside the window, so only an incumbent `a` explains the answer"
        );
        assert_eq!(
            selector.select(&[b, a]),
            Some(1),
            "a held the role across the dropout; b, at index 0, does not take it"
        );
    }

    #[test]
    fn a_non_finite_box_is_never_the_primary() {
        // A decode handed a garbage tensor produces boxes with no usable
        // geometry — `clamp` propagates a NaN rather than rejecting it — and
        // `YuNet::decode`'s own invariants say the extents are non-negative.
        // Such a box is not ranked, not matched and not continued, wherever in
        // the slice it sits. Before this, one listed first could hold the role
        // forever: its area is not a number, so nothing could out-rank it, and
        // nothing could displace it either.
        let nan_origin = any_face(Rect {
            x: f32::NAN,
            y: 100.0,
            w: 90.0,
            h: 90.0,
        });
        let nan_extent = any_face(Rect {
            x: 100.0,
            y: 100.0,
            w: f32::NAN,
            h: 80.0,
        });
        let infinite = any_face(Rect {
            x: 100.0,
            y: 100.0,
            w: f32::INFINITY,
            h: 80.0,
        });
        let negative = any_face(Rect {
            x: 100.0,
            y: 100.0,
            w: -120.0,
            h: -120.0,
        });
        let real = face(300.0, 200.0, 100.0);
        assert!(
            area(&negative.rect) > area(&real.rect),
            "the premise: a double negative is a *positive* nominal area of \
             14400, larger than the real face's 10000 — only the extent \
             invariant keeps it out of the ranking"
        );

        for (name, broken) in [
            ("a NaN origin", nan_origin),
            ("a NaN extent", nan_extent),
            ("an infinite extent", infinite),
            ("negative extents", negative),
        ] {
            let mut selector = PrimarySelector::new();
            assert_eq!(
                selector.select(&[broken, real]),
                Some(1),
                "{name}: the real face at index 1 is the primary"
            );
        }

        // A slice with nothing usable in it has no primary, and the role is
        // still free for the next frame that has one.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[nan_origin, infinite, negative]), None);
        assert_eq!(
            selector.select(&[real]),
            Some(0),
            "a usable face still takes the role"
        );
    }

    #[test]
    fn the_tie_break_measures_euclidean_distance_to_the_centre() {
        // Two equal 80x80 boxes whose centres are offset from the frame centre
        // (320, 240) by (12, 0) and (8, 8). Euclidean distance separates them —
        // 144 against 128 — while a Manhattan distance would separate them the
        // other way, 12 against 16, so this pins the metric as well as the
        // tie-break.
        let on_axis = face(292.0, 200.0, 80.0); // centre (332, 240)
        let diagonal = face(288.0, 208.0, 80.0); // centre (328, 248)
        assert_eq!(area(&diagonal.rect), area(&on_axis.rect), "equal areas");
        assert_eq!(on_axis.rect.center(), (332.0, 240.0));
        assert_eq!(diagonal.rect.center(), (328.0, 248.0));

        let manhattan = |rect: &Rect| {
            let (x, y) = rect.center();
            (x - 320.0).abs() + (y - 240.0).abs()
        };
        assert!(
            manhattan(&on_axis.rect) < manhattan(&diagonal.rect),
            "the premise: a Manhattan metric would rank these the other way"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(
            selector.select(&[on_axis, diagonal]),
            Some(1),
            "under Euclidean distance the nearer box is at index 1"
        );
    }

    #[test]
    fn a_selector_can_be_built_for_another_frame_size() {
        // Two equal 100x100 boxes at (300, 300) and (700, 300): their centres
        // are (350, 350) and (750, 350).
        let left = face(300.0, 300.0, 100.0);
        let right = face(700.0, 300.0, 100.0);
        assert_eq!(area(&left.rect), area(&right.rect));

        // In a 1280x720 frame the centre is (640, 360): `right` is 110 px away
        // and `left` 290 px, so `right` wins.
        let mut wide = PrimarySelector::from_frame_size(1280, 720);
        assert_eq!(wide.select(&[left, right]), Some(1));

        // The same two boxes in the reference 640x480 frame, whose centre is
        // (320, 240): now `left` is 30 px and 110 px away where `right` is
        // 430 px and 110 px away, so the *other* box wins. Which frame the
        // boxes came from is what "the frame centre" means.
        let mut reference = PrimarySelector::new();
        assert_eq!(reference.select(&[left, right]), Some(0));

        // A frame with no pixels has its centre at the origin rather than a NaN
        // or a panic. `right` is the farther of the two from the reference
        // centre and the nearer to (0, 0), and it is listed *first* here: a
        // centre that came out NaN would leave it leading and the answer would
        // be 0.
        let mut degenerate = PrimarySelector::from_frame_size(0, 0);
        assert_eq!(
            degenerate.select(&[right, left]),
            Some(1),
            "the box nearer the origin is at index 1"
        );
    }
}
