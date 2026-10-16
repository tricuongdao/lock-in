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
//! so the incumbent is recognised by **geometry** — it is the face that
//! overlaps the remembered box most, when that overlap is at least
//! `INCUMBENT_IOU`. Slice *order* is never identity: `YuNet::detect` returns
//! faces strongest-first and the scores jitter every frame, so index 0 this
//! frame is routinely a different person next frame.
//!
//! In full, one call does this:
//!
//! 1. an empty slice answers `None` and leaves the incumbent alone — a detector
//!    that misses for a tick has not said the wearer left, and `AwayFromDesk`
//!    is the state machine's judgement (it counts faces, with its own dwell);
//! 2. the incumbent is matched to the slice by maximum overlap, if any face
//!    overlaps the remembered box by at least `INCUMBENT_IOU`;
//! 3. the leader is the face the ranking puts first: largest area, ties to the
//!    face nearer the frame centre, ties to the lower index;
//! 4. the primary is the incumbent when one was matched and the leader is
//!    either that same face or not more than `HYSTERESIS` times its area;
//!    otherwise the primary is the leader — a challenger that is clearly larger
//!    takes the role, and so does any face at all when the incumbent is gone;
//! 5. the remembered box becomes the primary's box, so it tracks that face
//!    rather than drifting.
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
/// Two people side by side overlap by about nothing, while the same person one
/// tick later overlaps themselves heavily; `0.3` is the same "these are the same
/// face" figure `YuNet`'s own suppression uses, and it tolerates a face that has
/// moved by over half its own width or shrunk to 55% of its size between calls.
/// A lower bar would let the role travel across the frame to a stranger; a
/// higher one would orphan the incumbent whenever the wearer moved quickly.
const INCUMBENT_IOU: f32 = 0.3;

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
    /// The box of the face this selector chose on its last non-empty call, in
    /// frame coordinates. `None` until then, and still `None` after a call with
    /// no faces — a dropout is not a departure.
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

    /// Chooses the primary face in `faces`, returning its index in that slice —
    /// or `None` when there is no face to choose.
    ///
    /// The index is a position in the slice that was passed, not an identity:
    /// the caller re-derives the face from it. Every call updates the
    /// remembered box, so the selector tracks a face rather than a position,
    /// and an empty slice is the one call that changes nothing.
    pub fn select(&mut self, faces: &[FaceBox]) -> Option<usize> {
        // No faces is not a decision. `None` says there is no primary; the
        // incumbent stays where it is, because one missed detection is not the
        // wearer leaving the desk. `AwayFromDesk` is the state machine's to
        // judge — it counts faces and applies its own dwell — and a selector
        // that cleared itself on every dropout would hand the role to whoever
        // else is in frame the first time the wearer blinked.
        if faces.is_empty() {
            return None;
        }

        let incumbent = self.match_incumbent(faces);
        let leader = self.top_ranked(faces);

        let chosen = match incumbent {
            // A face in this slice continues the incumbent, so it keeps the
            // role unless the leader is a *different* face and clearly larger.
            // A challenger of exactly `HYSTERESIS` times the incumbent's area
            // does not take it: ties go to stability.
            Some(incumbent) if incumbent != leader => {
                if area(&faces[leader].rect) > area(&faces[incumbent].rect) * HYSTERESIS {
                    leader
                } else {
                    incumbent
                }
            }
            // The incumbent is the leader as well: it is still the largest
            // face, or at least as well placed as any other, so there is
            // nothing to weigh.
            Some(incumbent) => incumbent,
            // No face in this slice resembles the incumbent, which means it
            // left the frame — the choice is free.
            None => leader,
        };

        // The remembered box becomes the chosen face's *current* box, so it
        // follows that face as it moves. Only an empty slice skips this.
        self.incumbent = Some(faces[chosen].rect);
        Some(chosen)
    }

    /// The face that continues the incumbent, if one does.
    ///
    /// The face overlapping the remembered box **most**, provided that overlap
    /// is at least `INCUMBENT_IOU`. The best overlap rather than the first one
    /// over the bar: a detection that splits in two would otherwise hand the
    /// role to whichever half the detector happened to list first, and the
    /// slice's order is not identity — `YuNet::detect` returns faces
    /// strongest-first and the scores jitter every frame.
    fn match_incumbent(&self, faces: &[FaceBox]) -> Option<usize> {
        let remembered = self.incumbent?;

        let mut matched: Option<(usize, f32)> = None;
        for (index, face) in faces.iter().enumerate() {
            let overlap = remembered.iou(&face.rect);
            let best_so_far = match matched {
                None => true,
                Some((_, best)) => overlap > best,
            };
            // A NaN overlap fails the threshold and is skipped, which is what a
            // box with no usable geometry deserves; a face that is not the
            // incumbent must never inherit the role by accident.
            if overlap >= INCUMBENT_IOU && best_so_far {
                matched = Some((index, overlap));
            }
        }
        matched.map(|(index, _)| index)
    }

    /// The index of the face the ranking puts first, on a slice that is not
    /// empty.
    fn top_ranked(&self, faces: &[FaceBox]) -> usize {
        let mut leader = 0;
        for index in 1..faces.len() {
            if self.outranks(&faces[index], &faces[leader]) {
                leader = index;
            }
        }
        leader
    }

    /// Whether `candidate` outranks `leader`: more area first, then nearer the
    /// frame centre, and otherwise the earlier index, because a face that
    /// outranks nothing keeps the place it was found in.
    ///
    /// The area comparison is two steps rather than one so that a NaN area
    /// *loses* instead of winning by accident: a NaN is unequal to everything,
    /// and every ordering comparison against it is false, so the inequality
    /// test routes it to a comparison it cannot pass.
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

/// A box's area in square frame pixels.
///
/// "Largest box" means this and not the longer side: a 130x130 face is bigger
/// than a 90x160 one, and a rule that compared the longest edge would say the
/// opposite. Detector extents are non-negative — `YuNet::decode` clamps every
/// box into the frame — so this is never a negative area that could win a
/// comparison by a double negation, and a box that somehow carried one ranks
/// below every real face.
fn area(rect: &Rect) -> f32 {
    rect.w * rect.h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A face whose box is a `size`-square with its top-left at `(x, y)`.
    ///
    /// The landmarks are zeroes and the confidence is uniform: the selector
    /// ranks boxes, so a fixture that varied either would be testing a term the
    /// rule does not have.
    fn face(x: f32, y: f32, size: f32) -> FaceBox {
        box_face(x, y, size, size)
    }

    /// A face whose box is `w` by `h` with its top-left at `(x, y)`.
    fn box_face(x: f32, y: f32, w: f32, h: f32) -> FaceBox {
        FaceBox {
            rect: Rect { x, y, w, h },
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        }
    }

    #[test]
    fn largest_face_wins_when_sizes_differ_clearly() {
        // The small face is nearer the frame centre — its centre is (140, 140)
        // against (500, 180) for a reference frame centre of (320, 240) — it is
        // listed first, and the detector is more confident about it. The large
        // face wins anyway, because the rule is "largest box" and nothing else
        // in this module outranks area.
        let small = FaceBox {
            score: 0.99,
            ..face(100.0, 100.0, 80.0)
        };
        let large = FaceBox {
            score: 0.80,
            ..face(420.0, 100.0, 160.0)
        };
        assert_eq!(area(&small.rect), 6400.0);
        assert_eq!(area(&large.rect), 25600.0, "four times the small face");

        let mut selector = PrimarySelector::new();
        assert_eq!(
            selector.select(&[small, large]),
            Some(1),
            "the 160 px box is at index 1 and wins on area"
        );

        // Permuted, and on a fresh selector, the same box wins: the answer is a
        // box, and the index merely says where in *this* slice it was found.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[large, small]), Some(0));
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

        // The nearer box is listed second, so "first in the slice" is not the
        // answer: index 1 is the box 40 px from the centre.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[far, near]), Some(1));

        // And on a fresh selector the same box wins from the other order, so
        // which face is primary is a property of the geometry rather than of
        // the slice it arrived in.
        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[near, far]), Some(0));
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

        // And once it holds the role it keeps it against `a` in either order,
        // which is the same stickiness applied to the new incumbent.
        assert_eq!(
            selector.select(&[big, a]),
            Some(0),
            "big is at index 0 here"
        );
        assert_eq!(selector.select(&[a, big]), Some(1));
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
    fn the_incumbent_is_dropped_when_it_leaves_the_frame() {
        // The wearer leaves. Neither remaining box resembles their last box —
        // both overlaps are zero — so the role is free and the ranking decides:
        // the larger of the two, which the slice lists *second*.
        let wearer = face(100.0, 100.0, 100.0);
        let small = face(300.0, 300.0, 100.0);
        let larger = box_face(450.0, 300.0, 110.0, 100.0);
        assert_eq!(wearer.rect.iou(&small.rect), 0.0);
        assert_eq!(wearer.rect.iou(&larger.rect), 0.0);
        assert!(
            area(&larger.rect) > area(&small.rect),
            "the premise: the second box is the larger one"
        );
        assert!(
            area(&larger.rect) < area(&small.rect) * 1.25,
            "and it is inside the hysteresis window, so a selector that still \
             held the wearer's role for `small` would keep holding it"
        );

        let mut selector = PrimarySelector::new();
        assert_eq!(selector.select(&[wearer]), Some(0));
        assert_eq!(
            selector.select(&[small, larger]),
            Some(1),
            "with the incumbent gone the choice is free: index 1, the larger box"
        );

        // And the wearer coming back does not get the role back for free: the
        // selector no longer holds their box, and the face now wearing the role
        // is not out-sized by them, so it keeps it.
        assert_eq!(
            selector.select(&[wearer, small, larger]),
            Some(2),
            "the larger box at index 2 still holds the role"
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

        // A frame with no pixels has its centre at the origin rather than a
        // division by zero or a panic, and the ranking still answers.
        let mut degenerate = PrimarySelector::from_frame_size(0, 0);
        assert_eq!(
            degenerate.select(&[left, right]),
            Some(0),
            "the box nearer the origin wins"
        );
    }
}
