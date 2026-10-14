//! Eyes: how open they are, and where to crop them.
//!
//! Two numbers come out of the 468 landmarks here, and both feed `Drowsy`:
//! [`ear`], the eye aspect ratio whose fall is the drowsiness signal, and
//! [`eye_box`], the square the eye-state classifier is handed. They are the two
//! halves of the agreement gate -- Task 16 fires `Drowsy` only when the ratio is
//! low **and** the classifier calls the crop closed -- so the two are built from
//! the same few landmarks and are discussed together here.
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
//! # The box, and what "roll-corrected" can mean for a rectangle
//!
//! [`eye_box`] is the crop the eye-state classifier reads, and it exists to make
//! a domain match: `open-closed-eye-0001` was trained on **close-up eye crops**
//! from the MRL Eye Dataset, and a crop of the right size centred on the eyes is
//! what puts a webcam frame into that domain.
//!
//! OpenVINO's own gaze demo builds the same crop, and its geometry is where the
//! numbers come from. The demo takes the distance between the two outer corners
//! -- `33` and `263`, the same two landmarks the ratio uses as its `p1`s but
//! *across* the face -- computes the angle of the line joining them, and warps a
//! square of side `1.8 *` that distance, centred on their midpoint and rotated to
//! that angle, down to the 32x32 the classifier wants. So the demo's crop is a
//! **square whose diagonal is the eye line**: the corners sit at the middle of
//! its two vertical edges, and the square reaches `0.9 * d` from its centre in
//! every direction, which is exactly the distance to each corner. That is the
//! sense in which the box is roll-corrected: its size is measured along the line
//! between the eyes, not along the frame's axes, so a tilted head gets the same
//! crop as an upright one. A box whose *width* was `1.8 * d` instead would
//! contain the eye line only while it stayed near horizontal.
//!
//! What it cannot be is a *rotated* rectangle: [`Rect`] is four `f32`s and an
//! origin, and the classifier is handed that axis-aligned window. So this
//! function returns the square's axis-aligned enclosure -- the same centre, the
//! same side `1.8 * d`, and therefore the same pixels the demo's warp would
//! resample, read as a window instead of as a rotation. At a roll of 45 degrees
//! the eye line's ends land on the middle of the box's edges rather than in its
//! corners, which costs the crop some frame pixels at the extremes of the roll
//! and costs the classifier nothing it was trained to use: the eye still
//! occupies the same fraction of the 32x32 it is resized into.
//!
//! # `z` is ignored, deliberately
//!
//! Each landmark arrives as `[x, y, z]` and **this module reads `x` and `y`
//! only** -- the third column is never touched by either function.
//!
//! 1. **It is not in frame pixels.** Face Mesh's fused post-process scales `x`
//!    and `y` into the source frame and concatenates `z` raw, so `z` stays in
//!    the model's 192-unit input space. The ratified conversion is
//!    `z_frame_units = z * crop_width / 192`, and this module is not given a
//!    crop. Anything computed from `z` here would be wrong by the crop's own
//!    scale -- a factor of two or more.
//! 2. **Neither function needs it.** A ratio of distances within one eye and a
//!    box around the two eyes are both planar measurements, and the eye's `z`
//!    extent is the network's weakest output: it is inferred from a single
//!    192x192 crop with no stereo and no shading model. The eye contours this
//!    module reads are 2-D by construction, and the classifier's own crop is
//!    2-D as well.
//!
//! Both are the same ruling [`pose`](crate::analyze::pose) reaches for the same
//! reason, and the tests below keep it honest: the fixtures give every landmark
//! a distinct non-zero `z` and expect the same numbers as if they were zero.

use crate::analyze::nms::Rect;

/// The side of the eye box as a multiple of the inter-landmark eye distance.
///
/// OpenVINO's gaze demo crops `1.8 *` the distance between the outer eye corners
/// as the side of the square eye region, and the classifier was trained on crops
/// of that framing. See the module docs for how the square sits on the eye line.
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

/// Outer corner of the left eye: `p1` of [`LEFT`], and one end of the eye line.
const LEFT_OUTER: usize = 33;

/// Outer corner of the right eye: `p4` of [`RIGHT`], and the other end of it.
const RIGHT_OUTER: usize = 263;

/// The eye aspect ratio of one face's 468 landmarks.
///
/// The mean of the two eyes' ratios, each computed from the six indices the two
/// private index tables name -- see the module docs for the formula and for why
/// the mean is what is returned. Always finite and never negative: an `i32`
/// coordinate is exactly representable in `f32` up to 2^24 and rounds above it,
/// so no input can produce a NaN.
///
/// Zero when the eye's own two corners coincide. That is not a usable eye and
/// [`eye_box`] says so with a `None`; this function has no way to say it, and
/// the caller that gates on the ratio is gated on the box first.
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
        // this input is one `eye_box` has already refused with a `None`.
        return 0.0;
    }
    (distance(points, eye[1], eye[5]) + distance(points, eye[2], eye[4])) / width
}

/// Where to crop the eyes for the eye-state classifier, or `None`.
///
/// A square of side `EYE_BOX_SCALE` times the distance between the two outer
/// eye corners, centred on the midpoint of the line joining them. See the module
/// docs for why that square, and for what the axis-aligned rectangle can and
/// cannot carry of the roll.
///
/// `None` when the corners are the same pixel -- a landmark set that decoded to
/// nothing at all, or one whose eye corners landed together. There is no
/// inter-ocular distance to scale and no crop worth taking; a zero-area [`Rect`]
/// would be worse than none, because the classifier would happily resample it
/// and hand back a verdict about two pixels.
///
/// The centre may be outside the frame -- a wearer at the edge of the picture, or
/// a roll that swings it -- which is the caller's problem rather than this
/// function's: the classifier's own sampler clamps to the frame's edge, so an
/// off-frame box costs edge pixels rather than a panic. Clamping it here instead
/// would move the centre off the eyes.
pub fn eye_box(points: &[[i32; COORDINATES]; LANDMARKS]) -> Option<Rect> {
    let outer = points[LEFT_OUTER];
    let inner = points[RIGHT_OUTER];
    let (dx, dy) = ((inner[0] - outer[0]) as f32, (inner[1] - outer[1]) as f32);

    // Distance from the two corners, not from their x and y separations: an eye
    // line is a line, and which way it is rolled must not change how much of the
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
    /// outer corner is `263`, not `362`. Nothing may assume the two sets are the
    /// same order, and the moved-indices loop below is what holds that.
    const R_OUTER: usize = 263;
    const R_UPPER_OUTER: usize = 385;
    const R_UPPER_INNER: usize = 380;
    const R_INNER: usize = 362;
    const R_LOWER_INNER: usize = 373;
    const R_LOWER_OUTER: usize = 387;

    /// A tolerance for the fixture's own arithmetic. Every expectation below is
    /// hand-computed from whole-pixel distances, and a few of those distances are
    /// irrational, so the last `f32` bits are not what is being tested.
    const EPSILON: f32 = 1e-3;

    /// 468 points at the origin: a decode that produced nothing usable.
    fn landmarks() -> [[i32; 3]; 468] {
        [[0; 3]; 468]
    }

    /// Places one landmark, in frame pixels, with `z` at zero -- the module
    /// ignores `z`, and [`left_eye_open`] is where the non-zero case is covered.
    fn place(points: &mut [[i32; 3]; 468], index: usize, x: i32, y: i32) {
        points[index] = [x, y, 0];
    }

    /// One eye's six points, in the left eye's own frame: the outer corner at
    /// `(40, 100)`, the eye line 40 px long, and the four lid points `lid` px off
    /// it -- `p2` (54) directly over `p6` (54), and `p3` (66) over `p5` (66).
    ///
    /// Sharing an x is deliberate, and it is what makes both numerator terms a
    /// single lid gap. Face Mesh's real eye contours have four distinct x, so the
    /// real pairs are diagonal; that is the formula being *correct*, not a
    /// fixture to copy, because the ratio is a function of the points and any six
    /// points in these roles are a valid test of it.
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
            (outer, 40, 100),
            (upper_outer, 54, 100 - lid),
            (upper_inner, 66, 100 - lid),
            (inner, 80, 100),
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

    /// Places one eye's points with `x` mirrored about `320`, which turns the left
    /// eye's frame into the right eye's: `33` at `40` lands on `280`, where `263`
    /// -- the right eye's outer corner -- takes it.
    fn place_eye_mirrored(points: &mut [[i32; 3]; 468], shape: &[(usize, i32, i32); 6]) {
        for &(index, x, y) in shape {
            place(points, index, 320 - x, y);
        }
    }

    /// The left eye open, its lids 12 px off the line, with a distinct non-zero
    /// `z` at every one of the six points.
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

    /// The right eye open: the left eye's shape reflected about the face's
    /// vertical midline 160 px to its right, so its own outer corner `263` is
    /// 240 px from the left eye's `33` and the two eyes are congruent.
    ///
    /// | role | left | right |
    /// |---|---|---|
    /// | outer corner | `33` `(40, 100)` | `263` `(280, 100)` |
    /// | upper, outer | `160` `(54, 88)` | `385` `(266, 88)` |
    /// | upper, inner | `158` `(66, 88)` | `380` `(254, 88)` |
    /// | inner corner | `133` `(80, 100)` | `362` `(240, 100)` |
    /// | lower, inner | `153` `(66, 112)` | `373` `(254, 112)` |
    /// | lower, outer | `144` `(54, 112)` | `387` `(266, 112)` |
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

    /// Rotates `(x, y)` about the origin by a whole number of degrees and writes
    /// it rounded to whole pixels.
    ///
    /// A whole-number angle is the point of it: `cos 0 = 1`, `sin 90 = 1` and
    /// `cos 60 = 0.5` exactly, so a rolled landmark is still an exact pixel and a
    /// rolled expectation is still a hand-computed literal.
    fn place_rotated(points: &mut [[i32; 3]; 468], index: usize, x: i32, y: i32, degrees: f32) {
        let (sin, cos) = degrees.to_radians().sin_cos();
        let rotated_x = x as f32 * cos - y as f32 * sin;
        let rotated_y = x as f32 * sin + y as f32 * cos;
        place(
            points,
            index,
            rotated_x.round() as i32,
            rotated_y.round() as i32,
        );
    }

    /// The `n`th of an eye's six indices, in the order the formula names them.
    fn nth(indices: [usize; 6], n: usize) -> usize {
        indices[n - 1]
    }

    /// Whether two ratios agree to [`EPSILON`].
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
        // Each pair sharing an x is what makes that term the lid gap alone, and
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
        // unconditionally reads exactly the same 0.3 up there and passes it -- and
        // then reads `0.0` here, where the two eyes' ratios are `0.0` and `0.3`.
        // Averaging reads 0.15, which is neither.
        let mut right_only = landmarks();
        right_eye_open(&mut right_only);
        assert_eq!(
            ear(&right_only),
            0.15,
            "an empty left eye and a 0.3 right eye: the mirror case, and the one \
             a left-eye-only reader gets wrong"
        );

        // Both eyes fully open, and the two are *not* the same number: the mirror
        // reverses which of each lid pair is the upper point, which leaves the
        // right eye's pairs 12 px apart where the left's are 24, so it reads 0.3
        // against the left's 0.6 and the mean is 0.45.
        let mut both = points;
        right_eye_open(&mut both);
        assert!(
            close(ear(&both), 0.45),
            "the mean of 0.6 and 0.3, got {}",
            ear(&both)
        );

        // And with only the *left* eye's lids half shut, which takes its own ratio
        // from 0.6 to 0.4 while the right eye stays at 0.3 -- so the mean moves to
        // 0.35. This is the assertion that makes the mean a function of both eyes:
        // a reader that reported one of them, or a constant, cannot follow it from
        // 0.45 to 0.35, and neither can one that missed the left eye's lid move.
        //
        // The left eye is the one to half shut because the right eye's ratio is
        // *invariant* to its own lid gap: both of its pairs shorten together, so
        // `lid` cancels out of it. Measured, not assumed -- the right eye reads 0.3
        // at every lid from 0 to 12, so a fixture that half shut *it* would assert
        // 0.45 twice and test one thing.
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
            "the mean of 0.4 and 0.3, got {}",
            ear(&one)
        );

        // Finally every one of the twelve indices is read. Each is moved 20 px up,
        // one at a time, and the resulting ratio is a literal: the eye that did
        // not move keeps its own ratio, and the moved one's two terms are measured
        // from the coordinates above -- a corner move widens its denominator, a lid
        // move changes one of its pairs. The two eyes' literals differ because the
        // mirror changes *which* distances move and by how much, so a table that
        // reused one eye's values for the other could not pass.
        for (indices, moved_eye_ratios) in [
            (
                LEFT,
                [0.41833f32, 0.57500, 0.57500, 0.41833, 0.32500, 0.32500],
            ),
            (
                RIGHT,
                [0.34743, 1.05415, 1.20340, 0.34743, 1.20340, 1.05415],
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
                    !close(got, 0.6),
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

        // The lids 8 px off the line instead of 12, so `(16 + 16) / 160`.
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
        // The whole face: the left eye's outer corner `33` at (40, 100) and the
        // right eye's `263` at (280, 100), so the eye line is 240 px across and
        // its midpoint is (160, 100). The box is 1.8 * 240 = 432 a side -- every
        // number here is exact in `f32`, so the assertions can be equalities.
        let mut points = landmarks();
        left_eye_open(&mut points);
        right_eye_open(&mut points);
        assert_eq!(
            points[263][0] - points[33][0],
            240,
            "the fixture's eye distance is 240 px"
        );

        let box_ = eye_box(&points).expect("two distinct corners are an eye box");
        assert_eq!(box_.w, 432.0, "1.8 * 240");
        assert_eq!(box_.h, 432.0, "and square");
        assert_eq!(box_.x, 160.0 - 216.0, "the midpoint, less half the side");
        assert_eq!(box_.y, 100.0 - 216.0, "on both axes");
        assert_eq!(
            box_.center(),
            (160.0, 100.0),
            "which puts the centre between the two outer corners"
        );

        // A *longer* eye line, so the scale is a multiplication rather than a
        // constant: `263` moves 25 px further out and the side follows it to
        // 1.8 * 265.
        let mut stretched = points;
        place(&mut stretched, 263, 305, 100);
        let stretched_box = eye_box(&stretched).expect("still an eye box");
        assert_eq!(stretched_box.w, 477.0, "1.8 * 265, not 1.8 * 240");
        assert_eq!(
            stretched_box.x,
            172.5 - 238.5,
            "and it follows the midpoint"
        );
    }

    #[test]
    fn eye_box_follows_a_rolled_eye_line_without_going_off_square() {
        // Corner separations of `(16, 12)` and `(24, -32)`: both whole pixels,
        // both a Pythagorean distance (20 and 40), and neither along an axis. A
        // box built from the x and y separations separately gets both wrong, and
        // an axis-aligned fixture cannot show it: there the diagonal and the
        // separations are the same three numbers.
        for (offset_x, offset_y, distance) in [(16.0, 12.0, 20.0), (24.0, -32.0, 40.0)] {
            let mut points = landmarks();
            place(&mut points, 33, 40, 100);
            place(
                &mut points,
                263,
                40 + offset_x as i32,
                100 + offset_y as i32,
            );

            let box_ = eye_box(&points).expect("two distinct corners are an eye box");
            let midpoint = (40.0 + offset_x / 2.0, 100.0 + offset_y / 2.0);

            // The square whose *diagonal* is the eye line, which is the square
            // the eye line cannot leave: its half-diagonal is `0.9 * d`, and that
            // is what the box's own half-extents must be.
            let half = (box_.w + box_.h) / 4.0;
            assert!(
                close(half, 0.9 * distance),
                "half-diagonal {half} for an eye line {distance} px long"
            );
            assert!(
                close(box_.w, box_.h),
                "square on a rolled eye line too: {} by {}",
                box_.w,
                box_.h
            );
            assert!(
                close(box_.w, 1.8 * distance),
                "side {} for an eye line {distance} px long, angled ({offset_x}, {offset_y})",
                box_.w
            );

            // And the centre is the midpoint of the *line*, not the centre of the
            // axis-aligned span the two corners' x and y extents describe. Those
            // disagree whenever the line is rolled, and this is the assertion
            // that separates them.
            let (centre_x, centre_y) = box_.center();
            assert!(
                close(centre_x, midpoint.0) && close(centre_y, midpoint.1),
                "centre ({centre_x}, {centre_y}) against the midpoint {midpoint:?}"
            );
        }

        // A rolled *face*: the axis-aligned fixture above rotated 90 degrees about
        // the origin, so the eye line runs down the frame instead of across it and
        // both eyes roll with it. The corners land on exact pixels: `(40, 100)`
        // becomes `(-100, 40)`, so the box is still exactly 432 a side, around the
        // rolled midpoint `(-100, 160)`.
        let mut face = landmarks();
        left_eye_open(&mut face);
        right_eye_open(&mut face);
        let mut rolled = landmarks();
        for (index, point) in face.iter().enumerate() {
            if *point != [0; 3] {
                place_rotated(&mut rolled, index, point[0], point[1], 90.0);
            }
        }

        let box_ = eye_box(&rolled).expect("a rolled face still has an eye box");
        assert_eq!(box_.w, 432.0, "roll does not change the side");
        assert_eq!(box_.h, 432.0, "or the squareness");
        assert_eq!(box_.center(), (-100.0, 160.0), "the rolled midpoint");
    }

    #[test]
    fn eye_box_is_none_when_the_landmark_index_is_missing() {
        // No decode at all: every index is at the origin, so the two corners are
        // *coincident* and there is no inter-ocular distance to scale. A
        // zero-area box would be worse than none -- the classifier would happily
        // crop it and return a verdict about nothing.
        assert!(
            eye_box(&landmarks()).is_none(),
            "a landmark set that decoded to nothing has no eye box"
        );

        // One eye placed and the other not, which is what a partial decode looks
        // like: the corners are 40 px apart and this *is* a box. The distinction
        // is deliberate -- it is corner coincidence that is refused, not a
        // landmark set that happens to be sparse.
        let mut half = landmarks();
        left_eye_open(&mut half);
        assert!(
            eye_box(&half).is_some(),
            "one corner at (40, 100) and one at (80, 100) is still two corners"
        );

        // Two corners one pixel apart: a distance, but not an eye. Still a box,
        // and still square -- the only degenerate case refused is no distance at
        // all.
        let mut touching = landmarks();
        place(&mut touching, 33, 100, 100);
        place(&mut touching, 263, 101, 100);
        let box_ = eye_box(&touching).expect("distinct corners are a usable pair");
        assert_eq!(box_.w, 1.8, "however close they are");
        assert_eq!(box_.w, box_.h, "and it is still square");
    }
}
