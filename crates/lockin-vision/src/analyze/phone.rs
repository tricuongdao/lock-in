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
//! | `InHand` | in the *wearer's* box, below the head region and above desk level | yes |
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
//! Three outcomes, in order, on the centre of each cell-phone box:
//!
//! 1. centre inside the head region (below) → `AtHead`;
//! 2. otherwise `InHand`, when the centre is inside the **wearer's** person box
//!    — the one the primary face belongs to, not any person's, see
//!    *Attribution* — **and** at or below the head region's bottom edge, below
//!    the chin, **and** above the box's `IN_HAND_MAX_RATIO` ceiling, up in the
//!    torso rather than down at desk level, see *Desk level*;
//! 3. otherwise → `Present`. No phone at all → `None`.
//!
//! The *centre* is what is tested, never overlap. That is half of this module's
//! improvement over the prior art's bare AABB test: a phone lying on the desk in
//! front of the wearer overlaps the bottom of their person box, and an overlap
//! test calls that phone use. A centre is one point, so it either belongs to the
//! body or it does not — and the vertical band above is the other half, because
//! containment alone still cannot tell a phone in a hand from a phone on the
//! desk it is resting on.
//!
//! Only boxes of [`CELL_PHONE_CLASS`] are phones and only boxes of
//! [`PERSON_CLASS`] are bodies, so a mixed detection list is harmless: the
//! classes are read, not assumed from position in the slice.
//!
//! # Desk level, and why `InHand` has a ceiling
//!
//! The chin bounds `InHand` from above. Nothing in §5.1 bounds it from below,
//! and that gap is not academic: for a seated wearer YOLOX draws the person box
//! around the visible body down to the desk's occlusion line, so the box covers
//! the desk surface in front of them, and a phone lying on that desk has its
//! centre inside the box and below the chin. Read literally, §5.1 calls that
//! phone use — R3's named false positive, in the form that hurts most, because a
//! desk phone is in frame on every tick and the 2 s entry dwell plus the 3 s
//! latch then convert a per-frame error into continuous `PhoneInUse`.
//!
//! Containment cannot separate them. A **vertical band** can, and does, because
//! the two postures sit in different parts of the same box: a phone in a hand is
//! up in the torso, and a desk is at the bottom. `IN_HAND_MAX_RATIO` is where
//! that band ends, and its own docs carry the number's justification. This is a
//! *refinement* of §5.1 rather than a contradiction of it — the association rule
//! is still centre-containment (R42), and the band is the fourth condition the
//! spec omitted because it was written from prior art that had no hand signal
//! either.
//!
//! # Attribution: whose phone is it
//!
//! LOCK-IN tracks one person — the one whose face is primary — so a phone in
//! somebody else's hand is not the wearer's phone use, and `InHand` requires the
//! **wearer's** box. Task 16's priority order makes this load-bearing rather
//! than cosmetic: `PhoneInUse` outranks `OtherPresent`, so a colleague's handset
//! read as `InHand` would not merely add noise, it would rank the session into
//! `PhoneInUse` and bill the wearer for the colleague's call.
//!
//! The wearer's box is the person box sharing the most **horizontal** extent
//! with the primary face (`wearers_box`). Horizontal because the obvious rule —
//! the box containing the face's centre — dies on a common detection: a YOLOX
//! box that stops at the shoulders contains no part of the face at all, and a
//! face-centre rule would then find no wearer and lose every `InHand` in the
//! frame. A head sits over its own shoulders, so a clipped box still shares all
//! of the face's columns, and a colleague clear of the face shares none of them.
//! IoU is rejected for the same reason: a neck-clipped box has zero overlap
//! *area* with the face, so every positive IoU threshold refuses exactly the box
//! attribution needs.
//!
//! `AtHead` needs no person box, and that is attribution by construction rather
//! than an omission: the head region *is* the wearer's own head, derived from
//! the wearer's own face, so a handset inside it is at the wearer's ear. The
//! person box would add nothing but the risk of losing a call to a missed
//! detection — see *Cadence*.
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
//!   towards the camera. ±0.9 w covers that. Wide is deliberate, and it survives
//!   review on this argument: the error a wide region costs is a phone held
//!   beside the face reading `AtHead` instead of `InHand`, and **both are use**,
//!   so it mislabels a posture without mis-measuring the metric; the error a
//!   tight region costs is *time on the phone* reading `Present` through an
//!   entire call, which is the product's headline number going to zero. The two
//!   errors are not symmetric, so the surplus band is bought knowingly. Both
//!   sides of it are bracketed by tests (0.85 w must be at the head, 0.95 w must
//!   not), so drift in either direction fails rather than hides.
//! - **Crown 0.5 h above.** The face box stops at the brow; the head does not.
//!   Eyes sit about 0.61 of the way from chin to crown and the box's top edge
//!   about 0.7, so the crown is a little under half a face height above the box
//!   (0.44, rounded up for a box that has shrunk to the visible face under an
//!   occluding hand). Bracketed at 0.4 h inside and 0.6 h outside.
//! - **Bottom is the chin, and nothing below it.** Below the chin is jaw, neck
//!   and body, which is where `InHand` begins. A handset at the ear has its
//!   centre at or above the chin — the ear is level with the eyes, and a phone
//!   long enough to reach the jaw still centres well above it — so this costs
//!   nothing and is the boundary that keeps a desk phone out of `AtHead`.
//!
//! The region is deliberately **not** clipped to the person box, and `AtHead`
//! deliberately does **not** require one. For a box clipped at the neck the two
//! do not even overlap — the region is the head, and the box starts at the
//! shoulders — which is an ordinary detection and not a contradiction, because
//! neither rule consults the other.
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
//! hide a handset at the ear. `InHand` needs both the wearer's box and, through
//! the chin, the face; both are derived from the current tick's face, so the
//! part of the signal that is at the mercy of the slower detector's recall is
//! the same part that is already the weaker claim. The dwells in Task 16 (2 s
//! entry, 3 s latch, 0.8 s exit) are four to six times the phone cadence, so a
//! single stale or missing tick cannot by itself produce or end an event.
//!
//! # What this module deliberately does not claim
//!
//! - A phone **held at desk level** — resting under a hand at the desk's edge,
//!   or left on the lap — reads `Present`. The band in *Desk level* is what
//!   buys the desk phone's exclusion, and it cannot be sharpened into "is a hand
//!   touching it" with the boxes this interface is given. The cost is bounded
//!   and known: the phone has to be below three quarters of the person box to be
//!   missed, and picking a phone up moves it out of that band immediately.
//! - A phone at head *height* but clear of the head region — held out to the
//!   side at shoulder height — is `Present`, not `InHand`: "in hand" is defined
//!   as below the head region and above desk level. Holding a phone beside the
//!   face reads `AtHead` instead, which is still use.
//! - With **no primary face** there is no head region and no wearer, so nothing
//!   can be use: the answer is `None` or `Present`. The alternative — an empty
//!   head region that makes the whole person box `InHand` — would let a desk
//!   phone inside a loose person box become phone use exactly when the face is
//!   lost, which is the false positive R3 names, and Task 16 treats a frame with
//!   zero faces as `AwayFromDesk` in any case. A face box with no interior
//!   counts as no face, for the same reason: a zero-height region has everything
//!   below it.
//! - Attribution is approximate by construction. `wearers_box` picks the box
//!   under the primary face by horizontal overlap, which is the right box for
//!   the ordinary frames this runs on and can be wrong when two people's boxes
//!   overlap the same face columns — a colleague leaning in over the wearer's
//!   shoulder. Nothing in this interface can resolve that; a person box carries
//!   no identity, and the face is the only thing that does.
//!
//! # Cost
//!
//! One pass over the persons to find the wearer, then one pass over the phones
//! with a constant number of comparisons each; no allocation, two divisions in
//! `head_region` and none afterwards. It runs at the phone detector's 1–2 Hz,
//! not the face cadence.

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

/// How far down a person box a phone may be and still be *in a hand*, as a
/// fraction of the box's height.
///
/// The chin says where the head ends; this says where the body's usable region
/// ends and the desk begins. Without it `InHand` is unbounded below the chin, so
/// a phone lying on the desk reads as phone use whenever the person box covers
/// the desk surface — which for a seated wearer it usually does, because YOLOX
/// draws its box around the visible body down to the occlusion line. That is
/// R3's named false positive, and it is the persistent kind: a desk phone is in
/// frame every tick, so the 2 s entry dwell and the 3 s latch convert a
/// per-frame error into continuous phone-use time.
///
/// Three quarters is roughly where a seated wearer's hand is not. A box running
/// from the head to the desk holds the torso in its upper two thirds and the lap
/// and desk surface below that. A phone held at the desk's edge, or left lying
/// on the lap, therefore reads `Present` — a recall cost taken deliberately,
/// because the alternative is the metric the product is built on counting a
/// phone nobody is holding.
///
/// The band can never be empty: its ceiling is `person.y + 0.75 * person.h`,
/// which is at or below the box's own top edge, and `InHand` already requires
/// the centre to be inside the box.
const IN_HAND_MAX_RATIO: f32 = 0.75;

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
    /// A phone's centre is inside the **wearer's** box, below the head region
    /// and above desk level: held in the hand.
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
/// See the module docs for the rule, the head region's geometry, the desk-level
/// band, the attribution rule and the judgements this takes. In short: a phone
/// box whose centre is inside the head region is `AtHead`; one whose centre is
/// inside the *wearer's* box, below that region and above desk level is
/// `InHand`; anything else is `Present`; and no phone at all is `None` — as is a
/// `primary_face` that is absent, which leaves use unclaimable.
pub fn classify_phone(
    persons: &[Detection],
    phones: &[Detection],
    primary_face: Option<&FaceBox>,
) -> PhoneSignal {
    // No face, no head region — and a phone cannot be below a region that does
    // not exist, so no phone can be use. A face box with no interior is not a
    // head either, and filtering it here is what stops a degenerate decode from
    // producing a region that everything is below. The same box names the
    // wearer, so it is filtered once and used for both. See the module docs.
    let face = primary_face.filter(|face| face.rect.w > 0.0 && face.rect.h > 0.0);
    let head = face.map(head_region);
    let wearer = face.and_then(|face| wearers_box(persons, face));

    // Strongest claim wins, so the answer does not depend on the order the
    // detector happened to return its phones in.
    let mut strongest = PhoneSignal::None;
    for phone in phones
        .iter()
        .filter(|phone| phone.class_id == CELL_PHONE_CLASS)
    {
        let (x, y) = phone.rect.center();
        let posture = if head.is_some_and(|head| contains(&head, x, y)) {
            // The head region is the wearer's own head, so `AtHead` is
            // attributed by construction and needs no person box: see the
            // module docs on the cadence mismatch.
            PhoneSignal::AtHead
        } else if let (Some(head), Some(wearer)) = (head, wearer)
            && contains(&wearer.rect, x, y)
            && below(&head, y)
            && in_hand_band(&wearer.rect, y)
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

/// The part of a person box a phone held *in a hand* occupies: the torso, down
/// to `IN_HAND_MAX_RATIO` of the box's height.
///
/// This is the lower bound `InHand` needs; `below` is the upper one, and the two
/// are **not** redundant even though both bound the same band, because they are
/// measured from different boxes. The tempting simplification runs "once
/// `contains` has failed and `y` sits below the head region, `y` is below the
/// box's own top edge anyway, so the chin check can go" — and that rests on the
/// person box starting at or above the head region, which is false for exactly
/// the detection attribution exists to survive: a box clipped at the shoulders
/// starts *below* the chin. What the chin check does, in every frame, is keep a
/// phone at head height but clear of the head region — held up beside the face —
/// out of `InHand`, which the band alone would not do: the box contains such a
/// phone and the band covers it.
fn in_hand_band(person: &Rect, y: f32) -> bool {
    y < person.y + person.h * IN_HAND_MAX_RATIO
}

/// The person box the primary face belongs to — the wearer's, as opposed to
/// anybody else's in the room.
///
/// The box sharing the most **horizontal** extent with the face, and `None`
/// when no person box shares any of the face's columns.
///
/// Horizontal only, deliberately. A YOLOX box that stops at the shoulders
/// contains no part of the face at all — the face is above its top edge — and
/// that is an ordinary detection, not a degenerate one; a rule keyed on the
/// face's *centre*, or on the face box being inside the person box, would find
/// no wearer in exactly that frame and lose every `InHand` with it. Sharing the
/// face's columns is what survives the clip: a head sits over its own shoulders,
/// and a colleague clear of the face shares none of them.
///
/// Deliberately not an IoU test either, for the same reason: a neck-clipped box
/// has *zero* overlap **area** with the face box, so any positive IoU threshold
/// refuses the one box this function exists to find. Horizontal extent is
/// measured on the axis that survives, which is the axis a head and its own
/// shoulders share.
///
/// Ties go to the first box in the frame's order, so a duplicated detection
/// cannot make the answer depend on which copy the detector emitted first.
fn wearers_box<'a>(persons: &'a [Detection], face: &FaceBox) -> Option<&'a Detection> {
    let (left, right) = (face.rect.x, face.rect.x + face.rect.w);
    let mut wearer: Option<(&Detection, f32)> = None;
    for person in persons
        .iter()
        .filter(|person| person.class_id == PERSON_CLASS)
    {
        let overlap = (person.rect.x + person.rect.w).min(right) - person.rect.x.max(left);
        if overlap > 0.0 && overlap > wearer.map_or(0.0, |(_, best)| best) {
            wearer = Some((person, overlap));
        }
    }
    wearer.map(|(person, _)| person)
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
    /// **The landmarks are inert.** No rule in this module reads them: the head
    /// region comes from `rect`, which is a deliberate and reviewed decision —
    /// an axis-aligned region cannot rotate with a rolled head, and five YuNet
    /// points cannot enclose one. They are YuNet's five points in their ordinary
    /// places rather than zeroes so that a fixture printed in a failure is
    /// recognisable as a face; an implementation that derived the region from
    /// them would **not** be measured by these tests, and an earlier draft of
    /// this comment wrongly claimed it would.
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

    /// A 40x40 cell-phone box centred on `(x, y)`.
    ///
    /// The boundary probes further down are placed by the coordinate the rule
    /// reads — the centre — so their arithmetic stays legible in the test
    /// instead of hiding in a corner of a box.
    fn phone_at(x: f32, y: f32) -> Detection {
        detection(rect(x - 20.0, y - 20.0, 40.0, 40.0), CELL_PHONE_CLASS)
    }

    /// A handset at the ear: centre (730, 205).
    ///
    /// 50 px right of the face's midline — past the face box's right edge at
    /// 720, inside the head region's right edge at 752 — and at eye height
    /// (the face's eyes are at y = 175), which is where an ear is.
    fn phone_at_the_ear() -> Detection {
        detection(rect(710.0, 160.0, 40.0, 90.0), CELL_PHONE_CLASS)
    }

    /// Held in the hand at chest height: centre (680, 420).
    ///
    /// Inside the wearer's box, 180 px below the chin at y = 240 and 60 px above
    /// the in-hand band's ceiling at y = 480 — a phone in a hand at a desk,
    /// which is what `InHand` is for.
    fn phone_in_hand() -> Detection {
        detection(rect(650.0, 365.0, 60.0, 110.0), CELL_PHONE_CLASS)
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

    /// The wearer's box as a detector that stopped at the shoulders draws it:
    /// `x` in 600..760, `y` in 250..600.
    ///
    /// It contains no part of the face — the face's centre (680, 190) is 60 px
    /// above its top edge — while sharing all 80 px of the face's columns. This
    /// is the shape that must not cost `InHand`, and the shape a rule keyed on
    /// the face's *centre* loses.
    fn wearer_clipped_at_the_neck() -> Detection {
        detection(rect(600.0, 250.0, 160.0, 350.0), PERSON_CLASS)
    }

    /// Somebody else in the room: `x` in 900..1100, `y` in 180..580 — clear of
    /// the wearer and of the face's columns.
    fn colleague() -> Detection {
        detection(rect(900.0, 180.0, 200.0, 400.0), PERSON_CLASS)
    }

    /// The colleague's phone, at the colleague's chest: centre (970, 455).
    ///
    /// Inside the colleague's box, below the chin and inside the in-hand band
    /// (`455 < 180 + 0.75 * 400 = 480`), so anything but `Present` for it is
    /// attribution rather than position.
    fn colleagues_phone() -> Detection {
        phone_at(970.0, 455.0)
    }

    /// A colleague leaning in from the wearer's right: `x` in 700..900.
    ///
    /// Their box shares 20 px of the primary face's columns (700..720) where the
    /// wearer's shares all 80, so both are candidates for attribution and only
    /// one of them is under the face.
    fn colleague_leaning_in() -> Detection {
        detection(rect(700.0, 180.0, 200.0, 400.0), PERSON_CLASS)
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
    fn a_phone_in_the_torso_band_inside_the_person_box_is_in_hand() {
        // Centre (680, 450): inside the wearer's box, below the chin at 240, and
        // 30 px above the in-hand band's ceiling at 480 — the lower half of that
        // band's bracket. With the ceiling below 0.6875 of the box the answer
        // becomes `Present`, so a region tightened until real hand-held phones
        // stop being seen fails a test instead of silently costing recall.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(680.0, 450.0)], Some(&face())),
            PhoneSignal::InHand,
            "a phone in a hand at the bottom of the torso is still in the hand"
        );
    }

    #[test]
    fn a_desk_phone_inside_the_person_box_is_present_not_in_hand() {
        // A phone lying on the desk in front of the wearer, with its centre
        // *inside* the person box — where a loose YOLOX box puts it, since the
        // box covers the desk surface the wearer is sitting at. Containment
        // cannot separate this from a phone in a hand; the vertical band can,
        // because a held phone is up in the torso and the desk is the bottom of
        // the box.
        //
        // Centre (680, 500): inside the box (`y` in 120..600) and below the chin
        // at 240, but 20 px below the band's ceiling at 480. This is the upper
        // half of the band's bracket: with the ceiling above 0.79 of the box
        // this reads `InHand`, which is the defect this test exists for.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(680.0, 500.0)], Some(&face())),
            PhoneSignal::Present,
            "a phone at desk level inside the wearer's box is not in their hand"
        );

        // And deeper: centre (680, 580), near the bottom edge at 600. A detector
        // that draws the wearer's box 25 px lower than this fixture still has
        // this phone inside it, which is what made the defect a coin flip on
        // where the box's bottom lands.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(680.0, 580.0)], Some(&face())),
            PhoneSignal::Present,
            "and neither is one on the desk itself"
        );
    }

    #[test]
    fn a_phone_inside_the_head_region_is_at_head() {
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at_the_ear()], Some(&face())),
            PhoneSignal::AtHead
        );

        // The region's lateral edge, just inside it: centre (748, 205) is 68 px
        // — 0.85 face widths — right of the midline, where the region's edge is
        // 72 px out. This is the lower half of the width bracket: with
        // `HEAD_WIDTH_RATIO` below 1.7 this centre falls outside the region and
        // the answer becomes `Present`, so drift towards a region too tight to
        // see a handset at the ear fails a test rather than only costing recall.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(748.0, 205.0)], Some(&face())),
            PhoneSignal::AtHead,
            "0.85 face widths from the midline is still the head"
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

        // Higher, and still inside: centre (700, 100) is 40 px above the face
        // box and 10 px inside the crown edge at 90. This is the lower half of
        // the crown bracket — below `CROWN_ABOVE_RATIO` 0.4 this reads
        // `Present`.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(700.0, 100.0)], Some(&face())),
            PhoneSignal::AtHead,
            "0.4 face heights above the box is still the head"
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

        // And just above the edge: centre (700, 80) is 60 px above the face box
        // and 10 px above the crown at 90. This is the upper half of the crown
        // bracket — above `CROWN_ABOVE_RATIO` 0.6 this reads `AtHead`, and a
        // crown that grew until it swallowed the shelf is exactly what the
        // earlier one-sided bracket could not catch.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(700.0, 80.0)], Some(&face())),
            PhoneSignal::Present,
            "0.6 face heights above the box is above the head"
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

        // And just past the edge: centre (756, 200) is 76 px — 0.95 face widths
        // — right of the midline, 4 px beyond the region's edge at 752. This is
        // the upper half of the width bracket: above `HEAD_WIDTH_RATIO` 1.9 this
        // reads `AtHead`, and a region wide enough to cover the body's shoulders
        // at head height is what the surplus band costs.
        assert_eq!(
            classify_phone(&[wearer()], &[phone_at(756.0, 200.0)], Some(&face())),
            PhoneSignal::Present,
            "0.95 face widths from the midline is clear of the head"
        );
    }

    #[test]
    fn a_face_box_with_no_interior_is_no_face_at_all() {
        // A degenerate box is not a head, and treating its region as real is the
        // way this module could invent use out of nothing: a zero-height region
        // has *everything* below it, so the wearer's chest phone would come back
        // `InHand` from a box that is not a box. Every degenerate shape reads as
        // no face, so the phones fall back to `Present` — and none of them
        // panics, which is this crate's standing contract for geometry fed by a
        // model.
        //
        // Four shapes, because they fail differently: the zero-height one is the
        // dangerous one (it is still wide enough to name a wearer and still says
        // everything is below it), the zero-width one leaves a one-dimensional
        // region, the fully collapsed one does both, and the NaN one is what a
        // session that misbehaved would hand over.
        let collapsed = FaceBox {
            rect: rect(680.0, 190.0, 0.0, 0.0),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let flat = FaceBox {
            rect: rect(640.0, 190.0, 80.0, 0.0),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let thin = FaceBox {
            rect: rect(680.0, 140.0, 0.0, 100.0),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        let not_a_number = FaceBox {
            rect: rect(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        for face in [collapsed, flat, thin, not_a_number] {
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
    fn a_phone_in_another_persons_hand_is_not_the_wearers_phone_use() {
        // The inversion of what this test asserted before review, and the
        // inversion is the point. LOCK-IN tracks one person — the one whose face
        // is primary — so a colleague's handset is not the wearer's phone use.
        // Task 16 ranks `PhoneInUse` above `OtherPresent`, so reading it as
        // `InHand` would not merely add noise: it would rank the session into
        // `PhoneInUse` and bill the wearer for the colleague's call.
        //
        // The phone at (970, 455) satisfies every positional condition — inside
        // the colleague's box (`y` in 180..580), below the chin, inside the
        // in-hand band (455 < 480) — so `Present` here is attribution and
        // nothing else.
        assert_eq!(
            classify_phone(
                &[wearer(), colleague()],
                &[colleagues_phone()],
                Some(&face())
            ),
            PhoneSignal::Present,
            "a phone in another person's box is not the wearer's phone use"
        );
    }

    #[test]
    fn the_wearers_box_is_found_when_the_detector_clips_it_at_the_neck() {
        // The approximation the attribution rule rests on: the wearer's box is
        // the one sharing the most *horizontal* extent with the primary face,
        // not the one containing the face's centre. A detector that stops at the
        // shoulders contains no part of the face at all, and a rule keyed on the
        // face's centre would find no wearer and lose every `InHand` — a recall
        // failure traded for a precision win, which is not the trade I1 asked
        // for.
        let clipped = wearer_clipped_at_the_neck();

        // A phone at the wearer's chest, inside the clipped box: in hand.
        assert_eq!(
            classify_phone(&[clipped], &[phone_at(680.0, 420.0)], Some(&face())),
            PhoneSignal::InHand,
            "a box clipped at the neck is still the wearer's box"
        );

        // The head region lies entirely *above* that box — it is the head, and
        // the box now starts at the shoulders — and a handset at the ear is
        // still `AtHead`, because `AtHead` is decided by the region and needs no
        // person box at all. This is the one shape where the two regions are
        // disjoint, and it is an ordinary detection, not a degenerate one.
        assert_eq!(
            classify_phone(&[clipped], &[phone_at_the_ear()], Some(&face())),
            PhoneSignal::AtHead
        );

        // And the colleague's box, which is wider than the wearer's but shares
        // none of the face's columns, must not steal the attribution: their
        // phone stays `Present` in the same frame.
        assert_eq!(
            classify_phone(
                &[clipped, colleague()],
                &[colleagues_phone()],
                Some(&face())
            ),
            PhoneSignal::Present,
            "the clipped wearer's box still wins the attribution"
        );
    }

    #[test]
    fn the_wearers_box_is_the_one_under_the_face_not_the_first_in_the_frame() {
        // Two boxes share the primary face's columns and the frame lists the
        // colleague's first — which is an order YOLOX's own score sorting can
        // produce, so the rule cannot depend on it. The wearer is the box under
        // *more* of the face: all 80 px of its columns against the leaning
        // colleague's 20.
        let leaning_in = [colleague_leaning_in(), wearer()];

        // The colleague's phone: centre (850, 420), inside the colleague's box
        // (`x` in 700..900, `y` in 180..580), below the chin, inside the in-hand
        // band (`420 < 480`). Positionally it is `phone_in_hand()`, and it is
        // `Present` only because the box around it is not the wearer's.
        assert_eq!(
            classify_phone(&leaning_in, &[phone_at(850.0, 420.0)], Some(&face())),
            PhoneSignal::Present,
            "the colleague's box is not the wearer's, however early it is listed"
        );

        // And the wearer's own phone, in the same frame, is still in hand — the
        // rule picks a box, it does not discard the frame.
        assert_eq!(
            classify_phone(&leaning_in, &[phone_in_hand()], Some(&face())),
            PhoneSignal::InHand
        );
    }
}
