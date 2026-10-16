//! Phone posture: telling a phone that is *in use* from a phone that merely
//! exists in the frame.
//!
//! [`classify_phone`] answers one question per frame — is the phone in this
//! frame being used? — and it answers with four states rather than two, because
//! the most valuable idea borrowed from the prior art is that "a phone is
//! visible" and "the wearer is on the phone" are different facts:
//!
//! | signal | meaning | counts as use |
//! |---|---|---|
//! | `None` | no phone in the frame | no |
//! | `Present` | a phone that belongs to no hand and no head | **no** |
//! | `InHand` | inside a person's box, below the head region | yes |
//! | `AtHead` | inside the head region — a handset at the ear | yes |
//!
//! `Present` never escalates. A phone on the desk is a phone on the desk in
//! every frame it appears in, and no amount of persistence promotes it: the
//! separation has to happen in the geometry, because Task 16's dwell cannot do
//! it. A desk phone that has not moved for two seconds and a phone that has
//! been held for two seconds are the same observation to a dwell.
//!
//! # The rule
//!
//! Three predicates, applied to the centre of each cell-phone box:
//!
//! 1. centre inside the head region (below) → `AtHead`;
//! 2. otherwise, centre inside a person box **and** below the head region →
//!    `InHand`;
//! 3. otherwise → `Present`.
//!
//! The *centre* is what is tested, never overlap. That is the whole of this
//! module's improvement over the prior art's bare AABB test: a phone lying on
//! the desk in front of the wearer overlaps the bottom of their person box, and
//! an overlap test calls that phone use. A centre is one point, so it either
//! belongs to the body or it does not.
//!
//! Only boxes of [`CELL_PHONE_CLASS`] are phones and only boxes of
//! [`PERSON_CLASS`] are bodies, so a mixed detection list is harmless: the
//! classes are read, not assumed from position in the slice.
//!
//! # The head region, and why it is the face box's geometry
//!
//! The region is the face box widened to **1.8 face widths**, raised by **0.5
//! face heights** above the box's top edge, and ending at the box's bottom edge:
//!
//! ```text
//!              ┌───────────────────┐  ← crown: face-box top − 0.5 h
//!              │                   │
//!         ┌────┤     face box      ├────┐  ← ear height: a handset is out here,
//!         │    │                   │    │    beside the face, past its edge
//!         └────┤                   ├────┘
//!              │                   │
//!              └───────────────────┘  ← chin: face-box bottom
//! ```
//!
//! Why a box at all, rather than the landmarks: the region has to be an
//! *enclosure of the head in the frame's axes*. `FaceBox.rect` is already that
//! enclosure whatever the head's roll — YuNet returns an axis-aligned box
//! around the face, so a tilted head moves the box with it — whereas a region
//! built from the eye vector would need to be rotated with the head to stay an
//! enclosure of a rectangle that cannot rotate. The landmarks' remaining
//! contribution would be a midline and a scale that the box already carries,
//! at the price of a second degenerate case (two eye points that coincide).
//!
//! Why those numbers:
//!
//! - **Width 1.8 w.** YuNet's box is the face, roughly cheekbone to cheekbone
//!   and brow to chin, so it is about 0.8 of the head's width; the ear canal
//!   sits at the head's silhouette edge, which is 0.6 w from the midline. A
//!   handset held at the ear is *outside* the face box — it is beside the face
//!   — and a partially occluded phone is boxed as the visible sliver beside the
//!   head, whose centre can land a further ~0.3 w out when the phone is angled
//!   towards the camera. ±0.9 w covers that. Wide is deliberate: the error it
//!   costs is a phone held beside the head reading `AtHead` instead of
//!   `InHand`, and both are use, while the error a tight region costs is the
//!   headline metric — *time on the phone* — reading `Present` through an
//!   entire call.
//! - **Crown 0.5 h above.** The face box stops at the brow; the head does not.
//!   Eyes sit about 0.61 of the way from chin to crown and the box's top edge
//!   about 0.7, so the crown is a little under half a face height above the box
//!   (0.44, rounded up for a box that has shrunk to the visible face under an
//!   occluding hand).
//! - **Bottom is the chin, and nothing below it.** Below the chin is jaw, neck
//!   and body, which is where the `InHand` region has to begin. A handset at
//!   the ear has its centre at or above the chin — the ear is level with the
//!   eyes, and a phone long enough to reach the jaw still centres well above it
//!   — so this costs nothing and is the boundary that keeps a desk phone out of
//!   `AtHead`.
//!
//! The region is deliberately **not** clipped to the person box, and `AtHead`
//! deliberately does **not** require one. See below.
//!
//! # Cadence: faces at 5 Hz, phones at 1–2 Hz
//!
//! YuNet and YOLOX both emit absolute frame pixels for the frame they were
//! given, so there is no coordinate mismatch to correct — but the two boxes in
//! one call are not from the same instant. Stage 3 runs on its own cadence
//! (§4 of the design), so a person box can be up to about half a second older
//! than the face box beside it. For a seated wearer that is a few pixels of
//! drift, and during a reach for the phone it is tens of pixels.
//!
//! This module cannot fix that and does not pretend to: it has no clock, by
//! design, and a timestamp would be a scheduling decision wearing geometry's
//! clothes. What the mismatch does *not* do is destroy the strongest signal,
//! and that is why `AtHead` is decided by the head region alone. The head
//! region comes from the face, so a YOLOX tick that missed the person — the
//! person class is the easy one, but a person box is not guaranteed — cannot
//! hide a handset at the ear. Only `InHand` needs a person box, so only
//! `InHand` is at the mercy of the slower detector's recall on a given tick;
//! the dwells in Task 16 (2 s entry, 3 s latch, 0.8 s exit) are four to six
//! times the phone cadence, so a single stale or missing tick cannot by itself
//! produce or end an event.
//!
//! # What this module deliberately does not claim
//!
//! - A phone **inside** a person box and below the head region is `InHand`,
//!   even if it is in fact lying on the desk. Told otherwise by nothing but
//!   boxes, this module cannot separate the two: a hand and a desk phone both
//!   put a cell-phone box inside the person's box below their chin. Erring to
//!   `Present` there would erase `InHand` entirely, and `InHand` is what the
//!   ~2 s entry dwell is spent on.
//! - A phone at head *height* but clear of the head region — held out to the
//!   side at shoulder height — is `Present`, not `InHand`: "in hand" is defined
//!   as below the head region, per §5.1. Holding a phone beside the face reads
//!   `AtHead` instead, which is still use.
//! - With **no primary face** there is no head region, so nothing can be "below"
//!   it: the answer is `None` or `Present` and never use. The alternative — an
//!   empty head region that makes the whole person box `InHand` — would let a
//!   desk phone inside a loose person box become phone use exactly when the
//!   face is lost, which is the false positive R3 names, and Task 16 treats a
//!   frame with zero faces as `AwayFromDesk` in any case. A face box with no
//!   interior counts as no face, for the same reason: a zero-height region has
//!   everything below it.
//! - Person boxes are not tied to faces. A phone in a *second* person's box is
//!   use, because the brief associates over every person box and this interface
//!   has no way to say which box is the wearer's. With one person in frame —
//!   the product's normal case — the two readings coincide.
//!
//! # Cost
//!
//! One pass over the phones and, per phone, one pass over the persons: a
//! handful of comparisons on the boxes a frame already has, no allocation, and
//! two divisions in `head_region` and none in the classification itself. It
//! runs at the phone detector's 1–2 Hz, not the face cadence.

use crate::analyze::nms::{Detection, Rect};
use crate::models::yolox::{CELL_PHONE_CLASS, PERSON_CLASS};
use crate::models::yunet::FaceBox;

/// The head region's width, in face-box widths.
///
/// See the module docs for the anthropometry behind 1.8.
const HEAD_WIDTH_RATIO: f32 = 1.8;

/// How far above the face box's top edge the head region reaches, in face-box
/// heights.
///
/// See the module docs for the anthropometry behind 0.5.
const CROWN_ABOVE_RATIO: f32 = 0.5;

/// What a frame's phone detections say about the wearer's phone posture.
///
/// Ordered from weakest claim to strongest: `None` < `Present` < `InHand` <
/// `AtHead`. When a frame holds more than one phone the strongest claim wins,
/// so a phone on the desk cannot mask a phone in the hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneSignal {
    /// No cell-phone box in the frame.
    None,
    /// A phone exists in the frame and is attributable to no hand and no head:
    /// the desk phone, which is **not** phone use.
    Present,
    /// A phone's centre is inside a person's box, below the head region: held
    /// in the hand.
    InHand,
    /// A phone's centre is inside the head region: a handset at the ear.
    AtHead,
}

impl PhoneSignal {
    /// How strong a claim of phone *use* this signal is, for choosing between
    /// several phones in one frame.
    ///
    /// The order is stated here rather than inferred from declaration order, so
    /// that reordering the enum cannot silently change which phone wins.
    fn severity(self) -> u8 {
        match self {
            PhoneSignal::None => 0,
            PhoneSignal::Present => 1,
            PhoneSignal::InHand => 2,
            PhoneSignal::AtHead => 3,
        }
    }
}

/// Classifies the wearer's phone posture from one frame's detections.
///
/// See the module docs for the rule, the head region's geometry and the
/// judgements this takes. In short: a phone box whose centre is inside the head
/// region is `AtHead`; one whose centre is inside a person box and below that
/// region is `InHand`; anything else is `Present`; and no phone at all is
/// `None` — as is a `primary_face` that is absent, which leaves use
/// unclaimable.
pub fn classify_phone(
    persons: &[Detection],
    phones: &[Detection],
    primary_face: Option<&FaceBox>,
) -> PhoneSignal {
    // No face, no head region — and a phone cannot be below a region that does
    // not exist, so no phone can be use. A face box with no interior is not a
    // head either, and filtering it here is what stops a degenerate decode from
    // producing a region that everything is below. See the module docs.
    let head = primary_face
        .filter(|face| face.rect.w > 0.0 && face.rect.h > 0.0)
        .map(head_region);

    // Strongest claim wins, so the answer does not depend on the order the
    // detector happened to return its phones in.
    let mut strongest = PhoneSignal::None;
    for phone in phones
        .iter()
        .filter(|phone| phone.class_id == CELL_PHONE_CLASS)
    {
        let (x, y) = phone.rect.center();
        let posture = if head.is_some_and(|head| contains(&head, x, y)) {
            PhoneSignal::AtHead
        } else if let Some(head) = head
            && below(&head, y)
            && persons
                .iter()
                .any(|person| person.class_id == PERSON_CLASS && contains(&person.rect, x, y))
        {
            PhoneSignal::InHand
        } else {
            PhoneSignal::Present
        };
        if posture.severity() > strongest.severity() {
            strongest = posture;
        }
    }
    strongest
}

/// The head region of `face`: the box widened and raised as the module docs
/// describe.
///
/// The bottom edge is exactly the face box's bottom edge — the chin — and the
/// top is exactly half a face height above the box's top, so the region's
/// vertical extent is expressed in the box's own terms rather than as absolute
/// offsets that could drift away from it.
///
/// A degenerate face box — no width, no height, or a NaN from a session that
/// produced one — is filtered out by the caller rather than handled here. A
/// zero-height region would make *everything* below it, which is the one thing
/// this module must never do by accident: a garbage box would become a claim of
/// phone use. What it costs is one comparison; what it buys is that
/// [`classify_phone`]'s answer for a face box that is not a box is the same as
/// its answer for no face at all.
fn head_region(face: &FaceBox) -> Rect {
    let face = face.rect;
    let width = face.w * HEAD_WIDTH_RATIO;
    Rect {
        x: face.center().0 - width / 2.0,
        y: face.y - face.h * CROWN_ABOVE_RATIO,
        w: width,
        h: face.h * (1.0 + CROWN_ABOVE_RATIO),
    }
}

/// Whether the point `(x, y)` is inside `rect`.
///
/// Half-open on both axes: the left and top edges are inside and the right and
/// bottom edges are not. That is the convention `Rect`'s own arithmetic already
/// uses — two boxes that meet along an edge overlap in nothing — and here it
/// decides the one boundary a desk phone can sit on, the bottom of a person
/// box, in favour of the desk.
///
/// Local to this module because it is the only containment test in the tree so
/// far, and because `analyze::nms` belongs to another task's diff.
fn contains(rect: &Rect, x: f32, y: f32) -> bool {
    x >= rect.x && x < rect.x + rect.w && y >= rect.y && y < rect.y + rect.h
}

/// Whether a point at height `y` is below `rect`.
///
/// The exact complement of `contains`'s bottom-edge test, so a point on the
/// edge is either inside the head region or below it and never neither.
fn below(rect: &Rect, y: f32) -> bool {
    y >= rect.y + rect.h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box in frame pixels, 1280x720 in the fixtures below.
    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// One detection. The score is fixed: nothing in this module ranks phones
    /// by confidence, and a fixture that varied it would suggest otherwise.
    fn detection(rect: Rect, class_id: u32) -> Detection {
        Detection {
            rect,
            score: 0.9,
            class_id,
        }
    }

    /// The wearer's person box, at a desk: `x` in 560..800, `y` in 120..600.
    ///
    /// Every fixture in this module is placed at hand-chosen coordinates so the
    /// expected signal follows from arithmetic a reader can check, not from
    /// re-running the classifier.
    fn wearer() -> Detection {
        detection(rect(560.0, 120.0, 240.0, 480.0), PERSON_CLASS)
    }

    /// The primary face, inside the wearer's box: `x` in 640..720, `y` in
    /// 140..240, so the face's midline is x = 680 and its chin is y = 240.
    ///
    /// Its landmarks are placed where such a face's landmarks would be, so a
    /// later change that derives the region from them instead of from the box
    /// is measured by these tests rather than by their absence.
    fn face() -> FaceBox {
        FaceBox {
            rect: rect(640.0, 140.0, 80.0, 100.0),
            score: 0.9,
            landmarks: [
                (660.0, 175.0), // right eye
                (700.0, 175.0), // left eye
                (680.0, 200.0), // nose tip
                (662.0, 220.0), // right mouth corner
                (698.0, 220.0), // left mouth corner
            ],
        }
    }

    /// A handset at the ear: centre (730, 205).
    ///
    /// 50 px right of the face's midline — past the face box's right edge at
    /// 720, inside the head region's right edge at 752 — and at eye height
    /// (the face's eyes are at y = 175), which is where an ear is.
    fn phone_at_the_ear() -> Detection {
        detection(rect(710.0, 160.0, 40.0, 90.0), CELL_PHONE_CLASS)
    }

    /// Held in the hand at chest height: centre (680, 455).
    ///
    /// Inside the wearer's box, 215 px below the chin at y = 240 — the middle
    /// of the torso, where a seated wearer's hand and phone are.
    fn phone_in_hand() -> Detection {
        detection(rect(650.0, 400.0, 60.0, 110.0), CELL_PHONE_CLASS)
    }

    /// Lying flat on the desk in front of the wearer: centre (665, 620).
    ///
    /// Its far ten rows (590..600) are inside the wearer's box — the near edge
    /// of a body is at the desk's occlusion line — while its centre is 20 px
    /// below that box's bottom edge at 600, out on the desk.
    fn phone_on_the_desk() -> Detection {
        detection(rect(620.0, 590.0, 90.0, 60.0), CELL_PHONE_CLASS)
    }

    /// Far from anybody, across the desk: centre (1060, 665). Right of the
    /// wearer's box (which ends at x = 800) and below the head region.
    fn phone_across_the_desk() -> Detection {
        detection(rect(1020.0, 640.0, 80.0, 50.0), CELL_PHONE_CLASS)
    }

    #[test]
    fn a_phone_on_the_desk_far_from_any_person_is_present_not_in_use() {
        assert_eq!(
            classify_phone(&[wearer()], &[phone_across_the_desk()], Some(&face())),
            PhoneSignal::Present,
            "a phone nobody is near is present, not use"
        );
        // Nobody in frame at all: the same answer, from the same reason.
        assert_eq!(
            classify_phone(&[], &[phone_across_the_desk()], Some(&face())),
            PhoneSignal::Present
        );
        // And a phone out on the desk does not change character because the
        // face was lost this tick.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_across_the_desk()], None),
            PhoneSignal::Present
        );
    }

    #[test]
    fn a_phone_inside_a_person_box_below_the_head_is_in_hand() {
        assert_eq!(
            classify_phone(&[wearer()], &[phone_in_hand()], Some(&face())),
            PhoneSignal::InHand
        );
    }

    #[test]
    fn a_phone_inside_the_head_region_is_at_head() {
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at_the_ear()], Some(&face())),
            PhoneSignal::AtHead
        );
    }

    #[test]
    fn no_phone_detected_is_none() {
        assert_eq!(
            classify_phone(&[wearer()], &[], Some(&face())),
            PhoneSignal::None,
            "a wearer with no phone in frame is not a phone signal"
        );
        assert_eq!(classify_phone(&[], &[], None), PhoneSignal::None);
    }

    #[test]
    fn a_phone_on_the_desk_never_returns_in_hand() {
        // The desk phone whose far edge overlaps the wearer's box: an overlap
        // test — the prior art's — associates it with the body and calls it
        // use. Its centre is out on the desk, so it is not use.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_on_the_desk()], Some(&face())),
            PhoneSignal::Present,
            "a phone lying flat in the wearer's footprint is not in their hand"
        );

        // Lying flat and clear of the box altogether, still on the desk.
        let flat = detection(rect(700.0, 610.0, 90.0, 50.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[flat], Some(&face())),
            PhoneSignal::Present,
            "centre (745, 635) is below the wearer's box"
        );

        // Centre exactly on the box's bottom edge at y = 600. Boxes here are
        // half-open — the edge belongs to what is beyond it — so the boundary
        // belongs to the desk. Reading the edge as inside would make every
        // phone that touches the body's outline a phone in its hand.
        let on_the_edge = detection(rect(670.0, 580.0, 60.0, 40.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[on_the_edge], Some(&face())),
            PhoneSignal::Present,
            "centre (700, 600) is on the wearer's boundary, not inside it"
        );
    }

    #[test]
    fn the_head_region_reaches_above_the_face_box_to_the_crown() {
        // Centre (700, 120): 20 px above the face box's top edge at 140, and
        // 30 px below the region's crown edge at 90. The region is the head,
        // not the face — YuNet's box stops at the brow.
        let above_the_face = detection(rect(680.0, 100.0, 40.0, 40.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[above_the_face], Some(&face())),
            PhoneSignal::AtHead
        );
    }

    #[test]
    fn a_phone_just_below_the_chin_is_in_hand_not_at_head() {
        // Centre (680, 260): 20 px below the chin at 240, inside the wearer's
        // box. The head region ends at the chin, so this is below it — a phone
        // raised to the jaw is in the hand, not at the head.
        let at_the_jaw = detection(rect(650.0, 205.0, 60.0, 110.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[at_the_jaw], Some(&face())),
            PhoneSignal::InHand
        );

        // Centre exactly *on* the chin, at (680, 240). The head region's lower
        // edge and the "below" test are complements — a point on the edge is
        // either inside the head or below it, never neither — so this is `InHand`
        // and not `Present`. A gap between the two predicates would silently
        // drop every phone that lands on the boundary.
        let on_the_chin = detection(rect(650.0, 185.0, 60.0, 110.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[on_the_chin], Some(&face())),
            PhoneSignal::InHand,
            "the chin line belongs to the body below it, not to neither region"
        );
    }

    #[test]
    fn a_phone_at_the_head_needs_no_person_box() {
        // YuNet runs at 5 Hz and YOLOX at 1-2 Hz, so a tick can carry a face
        // and a phone with no person box beside them. A handset at the ear is
        // then still a handset at the ear: the head region is what says so, and
        // it comes from the face.
        assert_eq!(
            classify_phone(&[], &[phone_at_the_ear()], Some(&face())),
            PhoneSignal::AtHead
        );
    }

    #[test]
    fn a_phone_below_the_head_but_outside_every_person_box_is_present() {
        // Chest height, beside the wearer: centre (850, 455) is right of the
        // wearer's box. "In hand" is a claim about a body, and there is no body
        // here.
        let beside_the_wearer = detection(rect(820.0, 400.0, 60.0, 110.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[beside_the_wearer], Some(&face())),
            PhoneSignal::Present
        );
    }

    #[test]
    fn no_primary_face_never_escalates_beyond_present() {
        // Without a face there is no head region, and without a head region
        // there is nothing for a phone to be below. Every posture collapses to
        // "a phone exists" rather than to a guess about use.
        for phone in [phone_at_the_ear(), phone_in_hand(), phone_on_the_desk()] {
            assert_eq!(
                classify_phone(&[wearer()], &[phone], None),
                PhoneSignal::Present,
                "{phone:?} without a primary face"
            );
        }
    }

    #[test]
    fn a_phone_beside_the_head_but_clear_of_it_is_not_at_head() {
        // Centre (790, 200): inside the wearer's box, at head height, but 110 px
        // right of the face's midline — 1.38 face widths, where the head's own
        // silhouette reaches about 0.62. At head *height* is not at the head,
        // and that is the difference between region classification and the
        // prior art's "inside the person box" test. The design's rule then
        // leaves this `Present` rather than `InHand`, which the module docs
        // state as a known hole rather than paper over.
        let out_to_the_side = detection(rect(760.0, 160.0, 60.0, 80.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[out_to_the_side], Some(&face())),
            PhoneSignal::Present
        );
    }

    #[test]
    fn a_phone_above_the_crown_is_not_at_head() {
        // Centre (700, 60): 80 px above the face box's top edge, and above the
        // region's crown edge at 90. The region has an upper bound as well as a
        // lower one — a phone on a shelf above the desk is not a handset at the
        // ear.
        let above_the_head = detection(rect(680.0, 40.0, 40.0, 40.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[above_the_head], Some(&face())),
            PhoneSignal::Present
        );
    }

    #[test]
    fn a_face_box_with_no_interior_is_no_face_at_all() {
        // A degenerate box is not a head. Treating its region as real is the
        // one way this module could invent use out of nothing: a zero-height
        // region has *everything* below it, so the wearer's chest phone would
        // come back `InHand` from a box that is not a box. Both degenerate
        // shapes read as no face, so the phones fall back to `Present` — and
        // neither panics, which is this crate's standing contract for geometry
        // fed by a model.
        let zero_area = FaceBox {
            rect: rect(680.0, 190.0, 0.0, 0.0),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let not_a_number = FaceBox {
            rect: rect(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        for face in [zero_area, not_a_number] {
            for phone in [phone_at_the_ear(), phone_in_hand()] {
                assert_eq!(
                    classify_phone(&[wearer()], &[phone], Some(&face)),
                    PhoneSignal::Present,
                    "{phone:?} under {face:?}"
                );
            }
        }
    }

    #[test]
    fn a_phone_on_the_desk_does_not_mask_a_phone_in_use() {
        assert_eq!(
            classify_phone(
                &[wearer()],
                &[phone_on_the_desk(), phone_in_hand()],
                Some(&face())
            ),
            PhoneSignal::InHand,
            "the desk phone is listed first and must not decide the frame"
        );
        assert_eq!(
            classify_phone(
                &[wearer()],
                &[phone_in_hand(), phone_at_the_ear()],
                Some(&face())
            ),
            PhoneSignal::AtHead,
            "and a handset at the ear outranks a phone in the hand"
        );
        assert_eq!(
            classify_phone(
                &[wearer()],
                &[phone_at_the_ear(), phone_in_hand()],
                Some(&face())
            ),
            PhoneSignal::AtHead,
            "in either order: the strongest claim wins"
        );
    }

    #[test]
    fn only_cell_phone_boxes_are_phones_and_only_person_boxes_are_bodies() {
        // A person-class box handed in as a phone is not a phone, however
        // phone-shaped its position inside the person's own box makes it look.
        let person_as_phone = detection(wearer().rect, PERSON_CLASS);
        assert_eq!(
            classify_phone(&[wearer()], &[person_as_phone], Some(&face())),
            PhoneSignal::None
        );

        // And a cell-phone box handed in as a person is not a body: the chest
        // phone below has no person box to be inside, so it is present rather
        // than in hand.
        let phone_as_person = detection(wearer().rect, CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[phone_as_person], &[phone_in_hand()], Some(&face())),
            PhoneSignal::Present
        );
    }

    #[test]
    fn a_phone_in_another_persons_hand_is_still_use() {
        // The brief associates over every person box, and nothing in this
        // interface ties a person box to a face. A colleague's phone inside the
        // colleague's box — x in 900..1100, so clear of the wearer — is
        // therefore use. Task 16 ranks `PhoneInUse` above `OtherPresent`, so a
        // second person on a phone is the design's reading and not this
        // module's invention.
        let colleague = detection(rect(900.0, 180.0, 200.0, 400.0), PERSON_CLASS);
        let colleagues_phone = detection(rect(940.0, 400.0, 60.0, 110.0), CELL_PHONE_CLASS);
        assert_eq!(
            classify_phone(&[wearer(), colleague], &[colleagues_phone], Some(&face())),
            PhoneSignal::InHand
        );
    }
}
