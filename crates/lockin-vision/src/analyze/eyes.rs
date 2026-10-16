//! Eyes: how open they are, and where to crop them.
//!
//! Two things come out of the 468 landmarks here, and both feed `Drowsy`:
//! [`ear`], the eye aspect ratio whose fall is the drowsiness signal, and
//! [`eye_boxes`], the **two** squares the eye-state classifier is handed -- one
//! per eye. They are the two halves of the agreement gate: Task 16 fires
//! `Drowsy` only when the ratio is low **and** the classifier calls the crops
//! closed.
//!
//! # The ratio
//!
//! Six points per eye, in a fixed order, the classic construction: three on the
//! upper lid, three on the lower, and the ratio of their separation to the
//! eye's own width,
//!
//! ```text
//! EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
//! ```
//!
//! | eye | p1 | p2 | p3 | p4 | p5 | p6 |
//! |---|---|---|---|---|---|---|
//! | left (the subject's right) | `33` | `160` | `158` | `133` | `153` | `144` |
//! | right | `362` | `385` | `387` | `263` | `373` | `380` |
//!
//! `p1` and `p4` are the eye's corners, so the denominator is the eye's width;
//! the numerator is two lid separations, each between a point on the upper lid
//! and its opposite number on the lower. Dividing by two turns the pair into an
//! average height, which is what makes the result a ratio of two *lengths*: an
//! open eye measures a few tenths for a frontal adult face and a closed one falls
//! towards `0`, and -- this is the point of the division -- a wearer twice as
//! close to the camera measures the same thing, because both sides scale
//! together.
//!
//! [`ear`] returns the **mean of the two eyes**. The two are normally within a
//! few percent of each other and a wink or a blink that has not finished
//! crossing the face is not, so the mean is the quieter signal. The absolute
//! scale is a decision rather than a measurement: the spec's threshold is
//! *personal* -- Task 18's calibration measures the wearer's own eyes-open
//! baseline -- so what has to be stable is the statistic, not its units.
//!
//! # The boxes: one eye each, which is what the reference actually does
//!
//! [`eye_boxes`] returns a square **per eye**, and that is a correction to an
//! earlier reading of the reference, so it is worth stating exactly what the
//! reference is. OpenVINO's `open-closed-eye-0001` demo
//! (`eye_state_estimation` in the OpenVINO demos, and the model's own accuracy
//! checker) builds its crops like this:
//!
//! ```text
//! getEyeLandmarks()   -> four points: [leftOuter, leftInner, rightOuter, rightInner]
//! createBoundingBox(p1, p2, scale = 1.8)
//!                     -> scale * ||p1 - p2|| square, centred on (p1 + p2) / 2
//! estimate(frame)
//!                     -> left  crop = createBoundingBox(eyeLandmarks[0], eyeLandmarks[1])
//!                        right crop = createBoundingBox(eyeLandmarks[2], eyeLandmarks[3])
//! ```
//!
//! So the reference's `d` is the distance **between one eye's own two corners**
//! and its square is centred on **that one eye's** midpoint. It is a single-eye
//! close-up -- which is what the model was trained on (MRL Eye Dataset, eye
//! crops) and what its own three demo assets are. One crop spanning both eyes,
//! measured from outer corner to outer corner across the face, is roughly 2.5x
//! the reference's linear framing and centres on the nose bridge instead of an
//! eye; no amount of roll correction repairs that. This module therefore returns
//! two boxes and no longer has an inter-ocular one.
//!
//! The landmarks are the same four the reference uses:
//!
//! | eye | outer corner | inner corner | `d` |
//! |---|---|---|---|
//! | left (the subject's right) | `33` | `133` | `||p33 - p133||` |
//! | right | `362` | `263` | `||p362 - p263||` |
//!
//! `d` is a Euclidean distance, not a component of one: an eye line is a line,
//! and how the head is rolled must not change how much of the face a crop
//! covers. A box built from the corners' x and y separations independently would
//! be `|dx| + |dy|` -- a different, roll-dependent number, and 2.5x too big at
//! 45 degrees.
//!
//! # Roll: the box is axis-aligned, the *sampling* is not
//!
//! The reference rotates the crop by the head's roll before inference, and so
//! does this pipeline -- but in two different places, and the split is what
//! makes the interface work:
//!
//! - [`eye_boxes`] returns an axis-aligned [`Rect`], because that is the type
//!   the rest of the geometry stage speaks and because a rotation is a property
//!   of *sampling*, not of a rectangle. The square it describes is the
//!   reference's square in the eye's own **unrolled** frame: side `1.8d`,
//!   centred on the eye, not rotated.
//! - the head's roll, from Task 8's pose, is passed alongside them, and
//!   [`EyeStateClassifier::preprocess`](crate::models::eyestate::preprocess)
//!   takes it and samples the crop **rotated about the box's centre**. So the
//!   32x32 the graph sees is the eye upright, which is the reference's framing.
//!
//! A rotation is needed rather than merely nice. Sampling the unrotated square
//! on a rolled head delivers a rotated eye to a model trained on upright ones,
//! and the error grows with the roll: at 45 degrees the eye's long axis is
//! diagonal in the crop. There is no axis-aligned substitute for that -- the
//! tightest enclosing rectangle of a rotated square is *larger*, not
//! equivalent, which is why the correction lives in the sampler.
//!
//! The angle is the head's roll, not the eye line's own angle, and the two are
//! different measurements: the eye line tilts with the head, but it also tilts
//! with gaze and with facial asymmetry. The reference passes `headPoseAngles.z`
//! for exactly that reason, and Task 8 already computes it as
//! [`Euler::roll`](crate::analyze::pose::Euler::roll).
//!
//! # `z` is ignored, deliberately
//!
//! Each landmark arrives as `[x, y, z]` and **this module reads `x` and `y`
//! only** -- the third column is never touched by any function here.
//!
//! 1. **It is not in frame pixels.** Face Mesh's fused post-process scales `x`
//!    and `y` into the source frame and concatenates `z` raw, so `z` stays in
//!    the model's 192-unit input space. The ratified conversion is
//!    `z_frame_units = z * crop_width / 192`, and this module is not given a
//!    crop. Anything computed from `z` here would be wrong by the crop's own
//!    scale -- a factor of two or more.
//! 2. **Nothing here needs it.** A ratio of distances within one eye and a box
//!    around one eye are both planar measurements, and the eye's `z` extent is
//!    the network's weakest output: it is inferred from a single 192x192 crop
//!    with no stereo and no shading model. The eye contours this module reads
//!    are 2-D by construction, and the classifier's own crop is 2-D as well.
//!
//! Both are the same ruling [`pose`](crate::analyze::pose) reaches for the same
//! reason, and the tests below keep it honest: the fixtures give every landmark
//! a distinct non-zero `z` and expect the same numbers as if they were zero.

use crate::analyze::nms::Rect;

/// The side of an eye box as a multiple of that eye's own corner distance.
///
/// The reference's `createBoundingBox(p1, p2, scale)` is called with `1.8`, and
/// `d` there is one eye's corner-to-corner distance -- see the module docs.
pub const EYE_BOX_SCALE: f32 = 1.8;

/// Points in a decoded landmark set.
const LANDMARKS: usize = 468;

/// Coordinates per point: `x`, `y`, `z`.
const COORDINATES: usize = 3;

/// The six indices that make up one eye's ratio, in the order the formula names
/// them: `p1` and `p4` are the corners and the rest alternate between the two
/// lids.
///
/// The two eyes are named separately rather than as a nested array, because the
/// two are read by the same code and a mistake between them is invisible: both
/// eyes are present in every frame, so a swapped pair still produces a plausible
/// ratio. The test that moves one landmark at a time is what separates them.
const LEFT: [usize; 6] = [33, 160, 158, 133, 153, 144];

/// The right eye's six, in the same order. See [`LEFT`].
const RIGHT: [usize; 6] = [362, 385, 387, 263, 373, 380];

/// The two landmarks whose separation is an eye's own width, for the box this
/// module builds and for the ratio it computes: `[outer, inner]`.
///
/// The two eyes are not mirror images in this table and must not be written as
/// if they were: the left eye runs outward-to-inward (`33`, `133`) and so does
/// the right in *position*, but its indices run the other way (`362` is the
/// inner corner, `263` the outer). A table copied from one eye to the other
/// would measure the diagonal of the face instead.
const LEFT_EYE: [usize; 2] = [33, 133];

/// The right eye's `[outer, inner]`. See [`LEFT_EYE`].
const RIGHT_EYE: [usize; 2] = [362, 263];

/// The eye aspect ratio of one face's 468 landmarks.
///
/// The mean of the two eyes' ratios, each computed from the six indices in
/// `LEFT` and `RIGHT` -- see the module docs for the formula and for why the
/// mean is what is returned. Always finite and never negative: an `i32`
/// coordinate is exactly representable in `f32` up to 2^24 and rounds above it,
/// so no input can produce a NaN.
///
/// Zero when an eye's own two corners coincide. That is not a usable eye and
/// [`eye_boxes`] says so for that eye with a `None`; this function has no way to
/// say it, and the caller that gates on the ratio is gated on the boxes first.
pub fn ear(points: &[[i32; COORDINATES]; LANDMARKS]) -> f32 {
    0.5 * (ratio(points, &LEFT) + ratio(points, &RIGHT))
}

/// One eye's ratio: two lid separations over twice the eye's width.
fn ratio(points: &[[i32; COORDINATES]; LANDMARKS], eye: &[usize; 6]) -> f32 {
    let width = 2.0 * distance(points, eye[0], eye[3]);
    if width <= 0.0 {
        // The eye's two corners are the same pixel, so it has no width and the
        // ratio is undefined. Zero is what this returns rather than a panic or a
        // NaN: `ear` is called every tick from a camera thread, where a panic
        // takes the thread down, and the honest reading of "no measurement" is
        // the quiet one -- `Drowsy` needs the classifier to agree as well, and
        // this input is one `eye_boxes` has already refused with a `None`.
        return 0.0;
    }
    (distance(points, eye[1], eye[5]) + distance(points, eye[2], eye[4])) / width
}

/// A crop square for each eye, left first, or `None` for an eye whose own two
/// corners are unusable.
///
/// Each is the reference's `createBoundingBox(outer, inner, `[`EYE_BOX_SCALE`]`)`:
/// a square of side `1.8 * d`, where `d` is that eye's own corner distance,
/// centred on the midpoint of its own corner line. See the module docs for the
/// reference's construction and for why one box per eye is the whole point.
///
/// An eye's entry is `None` when its corners are the same pixel -- a landmark
/// set that decoded to nothing, or one eye whose corners landed together. There
/// is no distance to scale and no crop worth taking, and a zero-area [`Rect`]
/// would be worse than none, because the classifier would happily resample it
/// and hand back a verdict about two pixels. The two entries are independent, so
/// one unusable eye never costs the other its box.
///
/// A box's centre may be outside the frame -- a wearer at the edge of the
/// picture, or a roll that swings it -- which is the caller's problem rather
/// than this function's: the classifier's sampler clamps to the frame's edge, so
/// an off-frame box costs edge pixels rather than a panic. Clamping it here
/// instead would move the centre off the eye.
///
/// Use the head's roll in degrees with these, straight from Task 8's
/// [`Euler::roll`](crate::analyze::pose::Euler::roll) into
/// [`preprocess`](crate::models::eyestate::preprocess): the squares are
/// axis-aligned and the sampling is what carries the roll, so the sign
/// convention belongs to that function and this module's part is the two
/// squares.
pub fn eye_boxes(points: &[[i32; COORDINATES]; LANDMARKS]) -> [Option<Rect>; 2] {
    [box_for(points, LEFT_EYE), box_for(points, RIGHT_EYE)]
}

/// The square around one eye, from its `[outer, inner]` corner indices.
fn box_for(points: &[[i32; COORDINATES]; LANDMARKS], eye: [usize; 2]) -> Option<Rect> {
    let [outer, inner] = [points[eye[0]], points[eye[1]]];
    let (dx, dy) = ((inner[0] - outer[0]) as f32, (inner[1] - outer[1]) as f32);

    // Distance from the two corners, not from their x and y separations: an eye
    // line is a line, and how the head is rolled must not change how much of the
    // face the crop covers. The sum of squares is exact for `i32` -- the largest
    // possible gap is 2^32, whose square and the sum of two of them are both
    // well inside `f32`'s range -- so this is `0.0` exactly when the corners
    // coincide and positive otherwise, and the test is `<= 0.0` rather than a
    // tolerance because there is nothing here to be approximately zero.
    let squared = dx * dx + dy * dy;
    if !squared.is_finite() || squared <= 0.0 {
        return None;
    }
    let distance = squared.sqrt();

    // `< f32::MAX`, because `f32` saturates at that rather than at an infinity:
    // a square no float could hold is not a crop.
    let half = 0.5 * EYE_BOX_SCALE * distance;
    if !half.is_finite() || half >= f32::MAX {
        return None;
    }
    let (centre_x, centre_y) = (
        (outer[0] + inner[0]) as f32 * 0.5,
        (outer[1] + inner[1]) as f32 * 0.5,
    );

    Some(Rect {
        x: centre_x - half,
        y: centre_y - half,
        w: 2.0 * half,
        h: 2.0 * half,
    })
}

/// The Euclidean distance between two landmarks of one face, in frame pixels.
///
/// `x` and `y` only -- see the module docs on `z`. Written as `sqrt(dx^2 + dy^2)`
/// rather than through a hypot helper so that the arithmetic is the module's own,
/// and in `f32` from the moment the difference is taken: an `i32` that large is
/// not a pixel any camera produces, and the sum of two squares is finite for
/// every `i32` pair either way.
fn distance(points: &[[i32; COORDINATES]; LANDMARKS], from: usize, to: usize) -> f32 {
    let (dx, dy) = (
        (points[to][0] - points[from][0]) as f32,
        (points[to][1] - points[from][1]) as f32,
    );
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The left eye's six canonical indices, in the order the ratio names them.
    const LEFT: [usize; 6] = [33, 160, 158, 133, 153, 144];

    /// The right eye's six, the same way round.
    const RIGHT: [usize; 6] = [362, 385, 387, 263, 373, 380];

    /// The left eye's six indices, named by the role the formula gives each.
    const L_OUTER: usize = 33;
    const L_UPPER_OUTER: usize = 160;
    const L_UPPER_INNER: usize = 158;
    const L_INNER: usize = 133;
    const L_LOWER_INNER: usize = 153;
    const L_LOWER_OUTER: usize = 144;

    /// And the right eye's, the same six roles under different indices -- its
    /// outer corner is `263`, not `362`, and its `p1` is its *inner* corner.
    /// Nothing may assume the two sets share an order, and the moved-indices
    /// loop below is what holds that.
    const R_OUTER: usize = 263;
    const R_UPPER_OUTER: usize = 385;
    const R_UPPER_INNER: usize = 380;
    const R_INNER: usize = 362;
    const R_LOWER_INNER: usize = 373;
    const R_LOWER_OUTER: usize = 387;

    /// A tolerance for the fixture's own arithmetic. Every expectation below is
    /// hand-computed from whole-pixel distances, and a few of those distances are
    /// irrational, so the last `f32` bits are not what is being tested. The
    /// assertions that can be exact say `assert_eq!` instead.
    const EPSILON: f32 = 1e-3;

    /// The left eye's fixture: outer corner `33` at `(40, 100)`, inner corner
    /// `133` at `(80, 100)`, and four lid points `lid` px off that line.
    const LEFT_OUTER_AT: (i32, i32) = (40, 100);

    /// The left eye's fixture: inner corner.
    const LEFT_INNER_AT: (i32, i32) = (80, 100);

    /// 468 points at the origin: a decode that produced nothing usable.
    fn landmarks() -> [[i32; 3]; 468] {
        [[0; 3]; 468]
    }

    /// Places one landmark, in frame pixels, with `z` at zero -- this module
    /// ignores `z`, and [`left_eye_open`] is where the non-zero case is covered.
    fn place(points: &mut [[i32; 3]; 468], index: usize, x: i32, y: i32) {
        points[index] = [x, y, 0];
    }

    /// One eye's six points, in the left eye's own frame: the corners `40` px
    /// apart at x = 40 and x = 80 on row 100, and the four lid points `lid` px
    /// off that line -- `p2` (54) directly over `p6` (54), `p3` (66) over `p5`
    /// (66).
    ///
    /// Sharing an x is what makes each numerator term a single lid gap. Whether
    /// a *real* Face Mesh eye has that property is a separate question and not
    /// this fixture's job: the ratio is a function of six points, and any six in
    /// these roles exercise it.
    fn eye_shape(
        outer: usize,
        upper_outer: usize,
        upper_inner: usize,
        inner: usize,
        lower_inner: usize,
        lower_outer: usize,
        lid: i32,
    ) -> [(usize, i32, i32); 6] {
        [
            (outer, LEFT_OUTER_AT.0, LEFT_OUTER_AT.1),
            (upper_outer, 54, 100 - lid),
            (upper_inner, 66, 100 - lid),
            (inner, LEFT_INNER_AT.0, LEFT_INNER_AT.1),
            (lower_inner, 66, 100 + lid),
            (lower_outer, 54, 100 + lid),
        ]
    }

    /// Places one eye's six points.
    fn place_eye(points: &mut [[i32; 3]; 468], shape: &[(usize, i32, i32); 6]) {
        for &(index, x, y) in shape {
            place(points, index, x, y);
        }
    }

    /// Places one eye's points with `x` mirrored about `320` and `y` mirrored
    /// about `200`, which turns the left eye's frame into the right eye's: `33`
    /// at `(40, 100)` lands on `(280, 100)`, where `263` -- the right eye's outer
    /// corner -- takes it.
    ///
    /// Both axes are mirrored, which is what a face's midline reflection does to
    /// a point that is off-centre in both.
    fn place_eye_mirrored(points: &mut [[i32; 3]; 468], shape: &[(usize, i32, i32); 6]) {
        for &(index, x, y) in shape {
            place(points, index, 320 - x, 200 - y);
        }
    }

    /// The left eye open -- its lids 12 px off the line -- with a distinct
    /// non-zero `z` at every one of the six points.
    ///
    /// The `z`s are the point of the fixture: `x` and `y` are frame pixels and
    /// `z` is in Face Mesh's 192-unit space, and this module reads `x` and `y`
    /// only. A fixture with every `z` at zero would pass whether or not the
    /// implementation looked at them.
    fn left_eye_open(points: &mut [[i32; 3]; 468]) {
        place_eye(
            points,
            &eye_shape(
                L_OUTER,
                L_UPPER_OUTER,
                L_UPPER_INNER,
                L_INNER,
                L_LOWER_INNER,
                L_LOWER_OUTER,
                12,
            ),
        );
        let zs = [317, 12, -25, 4, 88, -3];
        for (index, z) in LEFT.into_iter().zip(zs) {
            points[index][2] = z;
        }
    }

    /// The right eye open: the left eye's shape, mirrored onto the face's other
    /// side, so its own outer corner `263` is at `(280, 100)` and its inner
    /// `362` at `(320, 100)`.
    ///
    /// The two eyes are congruent and 40 px wide each, and 200 px of face
    /// separate `33` from `263` -- but that inter-ocular distance is **not** what
    /// any box here is built from, which is the correction this module carries.
    fn right_eye_open(points: &mut [[i32; 3]; 468]) {
        place_eye_mirrored(
            points,
            &eye_shape(
                R_OUTER,
                R_UPPER_OUTER,
                R_UPPER_INNER,
                R_INNER,
                R_LOWER_INNER,
                R_LOWER_OUTER,
                12,
            ),
        );
    }

    /// Rotates `(x, y)` about `(160, 160)` by a whole number of degrees and
    /// writes it rounded to whole pixels.
    ///
    /// A whole-number angle is the point of it: `cos 0 = 1`, `sin 90 = 0` and
    /// `cos 180 = -1` exactly, and the quarter-turn cases keep whole pixels, so a
    /// rolled landmark is still an exact pixel and a rolled expectation is still
    /// a hand-computed literal.
    fn place_rotated(points: &mut [[i32; 3]; 468], index: usize, x: i32, y: i32, degrees: f32) {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let (dx, dy) = ((x - 160) as f32, (y - 160) as f32);
        place(
            points,
            index,
            (160.0 + dx * cos - dy * sin).round() as i32,
            (160.0 + dx * sin + dy * cos).round() as i32,
        );
    }

    /// The `n`th of an eye's six indices, in the order the formula names them.
    fn nth(indices: [usize; 6], n: usize) -> usize {
        indices[n - 1]
    }

    /// Whether two numbers agree to [`EPSILON`].
    fn close(got: f32, expected: f32) -> bool {
        (got - expected).abs() < EPSILON
    }

    #[test]
    fn ear_matches_the_formula_for_hand_built_landmarks() {
        // Points at coordinates chosen here and the ratio worked out by hand, so
        // the expectation is a literal rather than the formula's own output.
        //
        // The left eye: `p1` (40, 100) to `p4` (80, 100) is 40 px wide, so the
        // denominator `2 * ||p1-p4||` is 80. `p2` (54, 88) sits directly over
        // `p6` (54, 112) and `p3` (66, 88) over `p5` (66, 112), so each numerator
        // term is 24 px and the eye's own ratio is `(24 + 24) / 80 = 0.6`.
        //
        // `closed_eye_has_lower_ear_than_open_eye` reads the same eye at two
        // further lid openings.
        //
        // Only the left eye is placed here, so the right eye's six indices are
        // still at the origin and its ratio is the `0.0` that `ratio` returns for
        // an eye with no width. The mean of `0.6` and `0.0` is what this reads.
        let mut points = landmarks();
        left_eye_open(&mut points);
        assert_eq!(ear(&points), 0.3, "one 0.6 eye and one empty eye");

        // And now only the *right*, with the left untouched. This is the
        // assertion that separates a mean of the two eyes from the left eye
        // alone, which the block above cannot make: `ratio` on an all-zero eye is
        // `0.0`, so an implementation returning `ratio(points, &LEFT)`
        // unconditionally reads exactly the same 0.3 up there and passes it --
        // and then reads `0.0` here, where the two eyes' ratios are `0.0` and
        // 0.3. Averaging reads 0.15, which is neither.
        let mut right_only = landmarks();
        right_eye_open(&mut right_only);
        assert_eq!(
            ear(&right_only),
            0.15,
            "an empty left eye and a 0.3 right eye: the case a left-eye-only \
             reader gets wrong"
        );

        // Both eyes open, and the two do *not* read the same number: the mirrored
        // shape's lid pairs are 12 px apart where the left's are 24, so the right
        // eye reads a quarter of the left's 0.6. Measured, not assumed --
        // `right_eye_open`'s mirrored points put `p2` and `p6` on *different* rows
        // and different columns from each other, so that pair measures obliquely
        // where the left eye's pair is a vertical gap.
        let mut both = points;
        right_eye_open(&mut both);
        assert!(
            close(ear(&both), 0.45),
            "the mean of 0.6 and 0.3, got {}",
            ear(&both)
        );

        // And with only the *left* eye's lids half shut, which takes its own ratio
        // from 0.6 to 0.4 while the right eye stays at 0.3 -- so the mean moves
        // to 0.35. This is the assertion that makes the mean a function of both
        // eyes: a reader that reported one of them, or a constant, cannot follow
        // it from 0.45 to 0.35, and neither can one that missed the left eye's
        // lid movement.
        //
        // The left eye is the one to half shut because the right eye's ratio is
        // *invariant* to its own lid gap in this fixture: both of the pairs the
        // formula reads for that eye lie within a single lid -- `p2`/`p6` on the
        // upper and `p3`/`p5` on the lower -- so `lid` cancels out of both. That
        // is a property of the fixture's role table, not of the formula; a
        // fixture whose pairs straddled the eye line would not have it. Measured
        // rather than reasoned: the right eye reads 0.3 at every `lid` from 0 to
        // 12.
        let mut one = points;
        place_eye(
            &mut one,
            &eye_shape(
                L_OUTER,
                L_UPPER_OUTER,
                L_UPPER_INNER,
                L_INNER,
                L_LOWER_INNER,
                L_LOWER_OUTER,
                8,
            ),
        );
        right_eye_open(&mut one);
        assert!(
            close(ear(&one), 0.35),
            "the mean of 0.4 and 0.15, got {}",
            ear(&one)
        );

        // Finally every one of the twelve indices is read. Each is moved 20 px up,
        // one at a time, and the resulting ratio is a literal: the eye that did
        // not move keeps its own ratio, and the moved one's two terms are measured
        // from the coordinates above -- a corner move widens its denominator, a
        // lid move changes one of its pairs. The two eyes' literals differ because
        // the mirror changes *which* distances move and by how much, so a table
        // that reused one eye's values for the other could not pass.
        for (indices, moved_eye_ratios) in [
            (
                LEFT,
                [0.41833f32, 0.57500, 0.57500, 0.41833, 0.32500, 0.32500],
            ),
            (
                RIGHT,
                [0.34743, 1.20340, 1.05415, 0.34743, 1.05415, 1.20340],
            ),
        ] {
            for (n, expected) in moved_eye_ratios.into_iter().enumerate() {
                let index = nth(indices, n + 1);
                let mut moved = both;
                moved[index][1] = points[index][1] - 20;
                let got = ear(&moved);
                assert!(
                    close(got, expected),
                    "moving landmark {index} (position {}) should read {expected}, got {got}",
                    n + 1
                );
                assert!(
                    !close(got, 0.45),
                    "landmark {index} must actually be read: the ratio never left the \
                     unmoved value"
                );
            }
        }
    }

    #[test]
    fn closed_eye_has_lower_ear_than_open_eye() {
        // The same two corner points throughout, with only the four lid points
        // moving: the numerator shrinks and the denominator does not.
        let mut points = landmarks();
        left_eye_open(&mut points);
        let open = ear(&points);

        // The lids 8 px off the line instead of 12, so `(16 + 16) / 160` over the
        // two eyes, one of which is empty.
        place_eye(
            &mut points,
            &eye_shape(
                L_OUTER,
                L_UPPER_OUTER,
                L_UPPER_INNER,
                L_INNER,
                L_LOWER_INNER,
                L_LOWER_OUTER,
                8,
            ),
        );
        let half = ear(&points);

        // And shut: all six points on the eye line at y = 100, so both numerator
        // pairs collapse to zero while the 40 px baseline is untouched.
        place_eye(
            &mut points,
            &eye_shape(
                L_OUTER,
                L_UPPER_OUTER,
                L_UPPER_INNER,
                L_INNER,
                L_LOWER_INNER,
                L_LOWER_OUTER,
                0,
            ),
        );
        let shut = ear(&points);

        assert!(half < open, "lids halfway down read lower: {half} < {open}");
        assert!(shut < half, "and shut lower still: {shut} < {half}");
        // Three pins rather than two inequalities. An implementation returning
        // the lid gap, or the eye's width, or the difference of the two, is
        // monotone in the same direction and satisfies the inequalities alone.
        assert_eq!(open, 0.3, "the eye line is 40 px, the lids 12 px off it");
        assert_eq!(half, 0.2, "and 8 px off it");
        assert_eq!(shut, 0.0, "and on it: no gap to measure at all");
    }

    #[test]
    fn eye_box_is_square_and_scaled_1_8x_inter_landmark_distance() {
        // The whole face: the left eye's corners at (40, 100) and (80, 100), so
        // its own `d` is 40 and its box is `1.8 * 40 = 72` a side about their
        // midpoint (60, 100). The right eye's are at (240, 100) and (280, 100) --
        // also 40 apart -- so its box is 72 a side about (260, 100).
        //
        // Both eyes are 200 px apart, which is the number this test deliberately
        // does *not* use: the boxes are per eye, and an implementation built on
        // the inter-ocular distance would answer 360 here. Every number below is
        // exact in `f32`, so these are equalities.
        let mut points = landmarks();
        left_eye_open(&mut points);
        right_eye_open(&mut points);
        assert_eq!(
            points[263][0] - points[33][0],
            240,
            "the eyes are 240 px apart, and neither box may use that"
        );

        let [left, right] = eye_boxes(&points);
        let left = left.expect("two distinct corners are an eye box");
        let right = right.expect("and so are the other eye's");

        assert_eq!(left.w, 72.0, "1.8 * 40, one eye's own width");
        assert_eq!(left.h, 72.0, "and square");
        assert_eq!(left.center(), (60.0, 100.0), "about that eye's midpoint");
        assert_eq!(right.w, 72.0, "the other eye is the same width");
        assert_eq!(right.center(), (260.0, 100.0), "about its own midpoint");
        assert_eq!(right.x, 260.0 - 36.0, "and not about the nose bridge");

        // A *longer* eye, so the scale is a multiplication rather than a
        // constant: the left inner corner moves 15 px out and the side follows it
        // to `1.8 * 55`.
        let mut stretched = points;
        place(&mut stretched, L_INNER, 95, 100);
        let [left, _] = eye_boxes(&stretched);
        let left = left.expect("still an eye box");
        assert_eq!(left.w, 99.0, "1.8 * 55, not 1.8 * 40");
        assert_eq!(left.x, 67.5 - 49.5, "and it follows the midpoint");
    }

    #[test]
    fn eye_box_size_is_the_same_for_a_rolled_eye_line() {
        // The distance between the two corners is a Euclidean one, so it does not
        // change when the head rolls -- and neither may the crop. The three
        // separations below are `(40, 0)`, `(0, 40)` and `(24, 32)`: one flat, one
        // vertical, one genuinely diagonal, all with length exactly 40, so all
        // three must give a side of exactly 72.
        //
        // This is the assertion a component-wise implementation fails. `|dx| +
        // |dy|` is 40, 40 and 56 for the three, and the middle case alone already
        // shows the bug in the *opposite* direction from what a flat fixture can
        // reveal.
        for (dx, dy) in [(40, 0), (0, 40), (24, 32)] {
            let mut points = landmarks();
            place(&mut points, L_OUTER, 40, 100);
            place(&mut points, L_INNER, 40 + dx, 100 + dy);

            let [left, right] = eye_boxes(&points);
            let left = left.expect("two distinct corners are an eye box");
            assert!(
                right.is_none(),
                "the other eye is still at the origin and has no box"
            );
            assert_eq!(
                left.w, 72.0,
                "a {dx} by {dy} eye line is 40 px long and 1.8 * 40 is 72"
            );
            assert_eq!(left.h, left.w, "and the crop is square at every roll");
            assert_eq!(
                left.center(),
                (40.0 + dx as f32 / 2.0, 100.0 + dy as f32 / 2.0),
                "centred on the corners' midpoint, whatever the direction"
            );
        }

        // A rolled *head*, which moves every landmark: the whole face rotated 90
        // degrees about (160, 160). The left eye's corners land on exact pixels --
        // `(40, 100)` becomes `(220, 40)` and `(80, 100)` becomes `(220, 80)` --
        // so the box is still exactly 72 a side, now centred (220, 60).
        let mut face = landmarks();
        left_eye_open(&mut face);
        right_eye_open(&mut face);
        let mut rolled = landmarks();
        for (index, point) in face.iter().enumerate() {
            if *point != [0; 3] {
                place_rotated(&mut rolled, index, point[0], point[1], 90.0);
            }
        }

        let [left, right] = eye_boxes(&rolled);
        let left = left.expect("a rolled face still has an eye box");
        let right = right.expect("both of its eyes, in fact");
        assert_eq!(left.w, 72.0, "roll does not change the side");
        assert_eq!(
            left.center(),
            (220.0, 60.0),
            "or move the centre off the eye"
        );
        assert_eq!(right.w, 72.0, "the other eye rolls with it");
        assert_eq!(right.center(), (220.0, 260.0), "to its own new midpoint");
    }

    #[test]
    fn eye_box_is_none_when_the_landmark_index_is_missing() {
        // No decode at all: every index is at the origin, so each eye's corners
        // are *coincident* and neither has a distance to scale. A zero-area box
        // would be worse than none -- the classifier would happily crop it and
        // return a verdict about nothing.
        assert_eq!(
            eye_boxes(&landmarks()),
            [None, None],
            "a landmark set that decoded to nothing has no eye boxes"
        );

        // One eye placed and the other not, which is what a partial decode looks
        // like. The two entries are independent, so the eye that *is* there keeps
        // its box: `33` at (40, 100) and `133` at (80, 100) are 40 px apart, and
        // its square is still 72 a side about (60, 100). The eye that is not
        // there -- both its corners at the origin -- is `None`.
        let mut half = landmarks();
        left_eye_open(&mut half);
        let [left, right] = eye_boxes(&half);
        let left = left.expect("one eye is present and is an eye box");
        assert_eq!(left.w, 72.0, "the present eye's own width, not the face's");
        assert_eq!(left.center(), (60.0, 100.0), "about the present eye");
        assert!(
            right.is_none(),
            "a missing eye has no box, and must not borrow the other's"
        );

        // One eye's corners *touching*: a distance, but not an eye. Still a box,
        // and still square -- the only case refused is no distance at all.
        let mut touching = landmarks();
        place(&mut touching, L_OUTER, 100, 100);
        place(&mut touching, L_INNER, 101, 100);
        let [left, right] = eye_boxes(&touching);
        let left = left.expect("distinct corners are a usable pair");
        assert_eq!(left.w, 1.8, "however close they are");
        assert_eq!(left.w, left.h, "and it is still square");
        assert!(right.is_none(), "and the other eye still has nothing");
    }
}
