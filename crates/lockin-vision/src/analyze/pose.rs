//! Head pose: which way the face is pointing, in degrees, from landmarks that
//! already exist.
//!
//! [`estimate_head_pose`] is the signal `GlancingAway` and `Distracted` are
//! built on, so the contract it has to keep is not "a number every frame" but
//! **"never a confidently wrong number"**:
//!
//! - It never returns NaN or infinity, and never panics, for any input at all —
//!   including the degenerate ones (`Frame::new(0, 0, …)` gives
//!   `fx == fy == 0.0`, and all 468 landmarks can arrive on one pixel).
//! - `yaw` and `pitch` are always within ±90°, and `roll` is always finite.
//! - When the solve does not converge, or the pose it found cannot actually
//!   explain the landmarks, it returns [`HeadPose::INVALID`] — `valid == false`
//!   with a **zeroed** [`Euler`] — rather than its best guess. `INVALID` is the
//!   only shape in which a pose is returned invalid, so "no evidence" cannot be
//!   confused with "looking straight ahead" by accident.
//!
//! That distinction is load-bearing downstream. Task 16's state machine treats
//! `valid == false` as *no evidence* and carries the previous state forward,
//! where a finite-but-wrong pose would be read as a real head movement.
//!
//! # The correspondence set, and why only six points
//!
//! Six canonical Face Mesh indices, [`INDICES`], against a generic 3D face
//! model, [`MODEL`]:
//!
//! | index | landmark | why it is here |
//! |---|---|---|
//! | `33` | outer corner of the subject's right eye | the widest baseline in x |
//! | `263` | outer corner of the subject's left eye | …and the other end of it |
//! | `1` | nose tip | the most anterior point: it separates yaw from pitch |
//! | `61` | the subject's right mouth corner | constrains roll low on the face |
//! | `291` | the subject's left mouth corner | …and the other end of it |
//! | `199` | chin | the longest baseline in y |
//!
//! This is the classic six-point `solvePnP` set, and it is six rather than 468
//! because the solve runs at 5 Hz on a laptop and buys nothing from the other
//! 462: a pose has six degrees of freedom and these six points span all three
//! axes, so the problem is exactly determined rather than over-determined.
//! Points that are nearly collinear or nearly coincident — which most of the
//! mesh is, locally — would add equations that are numerically the same
//! equation, worsening the conditioning of the system without adding
//! information.
//!
//! # The 3D model, and what it costs
//!
//! [`MODEL`] is a **generic** adult face in millimetres, not the wearer's: 90 mm
//! between the outer eye corners, 52 mm between the mouth corners, 105 mm from
//! the eye line to the chin, with the nose tip as the origin (so the model's
//! `z` is how far each point sits *behind* the nose tip). No measurement of the
//! wearer is available at this stage and none is needed: a monocular six-point
//! solve recovers the rotation from the *shape* of the landmark constellation,
//! and a face that is uniformly 10% larger than the model changes the recovered
//! distance, not the recovered angles.
//!
//! What a generic model does cost is a **systematic** error — the same bias in
//! every frame — which is the error class risk R15 records. Measured on
//! synthetic faces whose true pose is known, by projecting a face of a
//! different shape through the intrinsics and solving it against [`MODEL`]:
//!
//! - **Uniform** shape error is free: a face 5% larger in every axis solves to
//!   within 0.5°, because the distance absorbs it. Only the ratios matter.
//! - **Per-axis** shape error is not free. A face 10% wider and 5% shorter than
//!   the model — the realistic spread across adults — reads pitch at up to
//!   1.5× its true value (10.7° of error at 20° of true pitch) and yaw at about
//!   0.75×. The mechanism is the coupling between pitch and depth: the model's
//!   vertical extent and its nose protrusion both set the apparent distance,
//!   and a shape error in either is traded against rotation.
//! - That is a **gain** error rather than an offset, which is the survivable
//!   kind here. Sign and monotonicity survive, and a neutral pose still reads
//!   as neutral (0.2°). Task 18's calibration measures the wearer's neutral and
//!   Task 16 compares deviations against *that*, so the small offset is
//!   absorbed and only the gain remains — a 15° bound lands at about 10° of
//!   true deviation for the worst face above, and 18° for the reverse error.
//!   Worth knowing, and worth a per-wearer model in a later task, but it shifts
//!   a threshold rather than inverting a decision.
//!
//! # Angles
//!
//! Degrees, right-handed about the camera's axes, with `x` right and `y` down
//! as everywhere else in this crate, and `z` away from the camera:
//!
//! | angle | axis | positive means |
//! |---|---|---|
//! | `yaw` | `y` | the face turns toward the image's **right** |
//! | `pitch` | `x` | the face tips **down**, nose toward the image's bottom |
//! | `roll` | `z` | the face turns **clockwise** on screen |
//!
//! All three are zero when the face is square to the camera, which is what
//! makes `Euler::default()` the frontal pose. Each sign is stated in terms of
//! the image because that is what a reader can check against a frame; the
//! module's own tests pin each one against the projected landmarks.
//!
//! **The three angles are the principal representative of the rotation, which is
//! not always the triple a caller might have written down.** `yaw` is
//! `asin(R₂₀)` and `pitch` and `roll` are `atan2`s, so one rotation has more
//! than one spelling and this module always returns the one whose `yaw` and
//! `pitch` are inside ±90°. A head at yaw 100°, pitch −100°, roll 0° comes back
//! as yaw 80°, pitch 80°, roll 180° — *the same rotation*: its landmarks are the
//! same six pixels as that triple's, so the two are one input, and the answer
//! says what the head is doing. `roll` is not bounded, so it is the angle that
//! absorbs the difference. What is refused is a rotation whose principal
//! representative still does not fit — an extraction that needs a pitch past
//! ±90° — and `euler_of`'s recomposition check makes "the angles name the
//! rotation they were read from" a property of the return value rather than of
//! three reads that happened to look right.
//!
//! # `z` is ignored, deliberately
//!
//! Each landmark arrives as `[x, y, z]` and **this module reads `x` and `y`
//! only**. That is a decision, not an oversight, and it rests on two facts
//! about `z`:
//!
//! 1. **It is not in frame pixels.** The graph's fused post-process scales and
//!    offsets `x` and `y` into the source frame and concatenates `z` raw, so `z`
//!    stays in the model's 192-unit input space. The ratified conversion is
//!    `z_frame_units = z * crop_width / 192`, which needs the crop width — and
//!    [`estimate_head_pose`] is not given one. Any use of `z` here that skipped
//!    that factor would be wrong by the crop's own scale (×192/crop_width, a
//!    factor of two or more for a tight crop), and a wrong `z` is not noise: it
//!    is a systematic bias in exactly the angle the eyes cannot correct.
//! 2. **The solve does not need it.** The six-point correspondence is against
//!    [`MODEL`], whose `z` is already known; the landmark `z` would only enter
//!    if the face's own shape were being used *instead of* the model. It is not,
//!    because Face Mesh's `z` is the network's weakest output — it is inferred
//!    from a single 192×192 crop with no stereo and no shading model — so
//!    handing it the geometry would replace a fixed, known-bounded error with a
//!    learned, unbounded one.
//!
//! The consequence is that this solve is a six-point *2D–3D* solve: the rotation
//! is recovered from the shape of the constellation and the distance from its
//! apparent size. Depth variation across the face — the cue `z` would have
//! sharpened — is small at laptop distance (26 mm across a 600 mm range) and the
//! landmark rounding to whole pixels dominates it. A future dense-shape
//! refinement could use `z` *with* the crop width, which is a larger change than
//! this module: it needs the crop, a per-wearer model, and a residual weighting
//! that trusts `z` far less than `x` and `y`.

use crate::analyze::intrinsics::CameraIntrinsics;
use nalgebra::{
    Matrix3, Matrix3x4, Matrix4, Rotation3, SMatrix, SVD, SVector, UnitQuaternion, Vector3,
};

/// Points in a decoded landmark set.
const LANDMARKS: usize = 468;

/// Coordinates per point: `x`, `y`, `z`.
const COORDINATES: usize = 3;

/// How many correspondences the solve uses.
const CORRESPONDENCES: usize = 6;

/// The six canonical Face Mesh indices the solve reads, paired one for one with
/// [`MODEL`].
///
/// The order is the order [`MODEL`] is written in — eyes, nose, mouth, chin —
/// and the two are only meaningful zipped together.
pub const INDICES: [usize; CORRESPONDENCES] = [33, 263, 1, 61, 291, 199];

/// The generic 3D face those six indices are matched against, in millimetres.
///
/// Origin at the nose tip, `x` toward the subject's left (the image's right),
/// `y` down, `z` away from the camera — so the nose tip is the closest point of
/// the face and every other point's `z` is how far behind the nose tip it sits.
///
/// Average adult proportions, rounded to whole millimetres: 90 mm between the
/// outer eye corners, 52 mm between the mouth corners, 35 mm from the eye line
/// down to the nose tip, 105 mm from the eye line to the chin. The `z` column
/// follows the standard generic six-point model's ratios (the eye corners
/// ~25 mm behind the nose tip, the mouth ~22 mm, the chin ~12 mm). That column
/// is not decoration: with every `z` equal the six points would be coplanar and
/// the solve would inherit the planar ambiguity, where a rotation and its
/// mirror fit the same image equally well.
///
/// Only the ratios matter: the solve recovers a rotation and a distance, so
/// scaling every column scales the distance and leaves the angles alone. What
/// the absolute millimetres buy is the plausibility gate on that distance —
/// `NEAREST_MM` and `FARTHEST_MM` are meaningful numbers only because the
/// model is in millimetres.
pub const MODEL: [[f32; COORDINATES]; CORRESPONDENCES] = [
    [-45.0, -35.0, 25.0],
    [45.0, -35.0, 25.0],
    [0.0, 0.0, 0.0],
    [-26.0, 36.0, 22.0],
    [26.0, 36.0, 22.0],
    [0.0, 70.0, 12.0],
];

/// The closest a face can be and still be a face at a laptop: 50 mm.
const NEAREST_MM: f32 = 50.0;

/// The farthest: 10 m, past which nothing about a webcam face is measurable.
const FARTHEST_MM: f32 = 10_000.0;

/// The smallest spread of the six observed points, in pixels, that can carry a
/// pose at all.
///
/// The mean distance from their own centroid, so it is the face's apparent size
/// rather than its position. This is the module's accuracy floor, and it is in
/// pixels because the error it guards against is in pixels: the landmarks are
/// whole numbers, so a constellation only a few pixels across carries a
/// rotation whose resolution is the pixel grid itself.
///
/// Measured on frontal, yawed, pitched and rolled synthetic faces projected
/// through a 640×480 frame's intrinsics, where the rounding to whole pixels is
/// the only error:
///
/// | spread | equivalent distance | worst error | solved |
/// |---|---|---|---|
/// | 28.8 px | 1.0 m | 1.5° | 6/6 |
/// | 14.4 px | 2.0 m | 3.3° | 6/6 |
/// | 11.5 px | 2.5 m | 2.2° | 6/6 |
/// | 9.6 px | 3.0 m | 8.3° | 6/6 |
/// | 4.8 px | 6.0 m | 5.2° | 6/6 |
/// | 2.9 px | 10 m | **19.6°** | 2/6 |
///
/// Twelve pixels is a **margin above the erratic region**, not the best row in
/// that table: the 9.6 px row already carries 8.3° of error and the 11.5 px row
/// happens to be better than the 14.4 px one, because below about 12 px the
/// error stops falling smoothly and starts depending on where the rounding
/// lands. Twelve is the first whole number clear of that, and it is where this
/// floor sits: a face about 2.4 m from a 640×480 webcam, or 4.8 m from a
/// 1280×720 one. Past that the honest answer is no pose, because the landmarks
/// cannot support one — and a missing pose is what Task 16 reads as "no
/// evidence", where a wrong one is read as a head movement.
///
/// The floor is in pixels because the error it guards against is in pixels — the
/// landmarks are whole numbers — which is also what makes it scale with
/// resolution rather than with the frame.
const MIN_SPREAD_PX: f32 = 12.0;

/// How large a reprojection error is still a fit, as a fraction of the spread
/// [`spread_of`] measures.
///
/// Relative rather than absolute because both sides of the comparison scale
/// with the face's apparent size: a landmark 2 px out on a face 40 px across is
/// the same measurement error as one 6 px out on a face 120 px across.
///
/// 10% is where the two failure modes separate, measured on synthetic faces
/// through this module's own fixture. Each figure is the fitted residual as a
/// fraction of the face's spread:
///
/// | input | residual |
/// |---|---|
/// | the model exactly, rounded to whole pixels | < 0.015 |
/// | ±1 px on every one of the six points | 0.015 – 0.025 |
/// | a face 10% wider and 5% shorter than the model | ≈ 0.03 |
/// | one correspondence 30 px out of place on a 48 px face | > 0.10 |
///
/// So the gate sits about three times above the worst honest measurement in
/// that table, and rejects a constellation no rigid face produces. It is a
/// backstop against a landmark that has landed on the wrong feature — a mouth
/// corner picked up on a hand, say — not a measure of accuracy: the model
/// mismatch in the third row is a genuine *fit*, and tightening the gate to
/// reject it would reject most real wearers.
const MAX_RESIDUAL_FRACTION: f32 = 0.10;

/// Gauss-Newton iterations. Fixed and small: this runs at 5 Hz, and with six
/// correspondences and six parameters the step is quadratic once close.
///
/// Ten was the first choice and is not enough: at a steep pitch the problem is
/// ill-conditioned along one direction, the step creeps, and a head pitched 80°
/// came back with a 17 px residual — refused by the gate below, but refused for
/// a pose the solver is exact on at 75° and 85°. Measured over a sweep of steep
/// pitches and a 105-pose noisy sweep, sixteen iterations converge everything
/// the solver can converge and twenty changes nothing further; twenty is what
/// is spent, on a solve that costs microseconds.
const MAX_ITERATIONS: usize = 20;

/// A step smaller than this — radians mixed with millimetres — stops the
/// refinement early. Well below the landmark rounding floor, so it never
/// truncates a solve that is still improving.
const CONVERGED_STEP: f32 = 1e-6;

/// How many times a step may be halved before the refinement gives up on it.
///
/// Four attempts: the full step, then a half, a quarter and an eighth. Newton's
/// step is only meaningful where the residual is defined, and the linear
/// estimate can be far enough out that a full step carries a model point through
/// the camera plane and out the other side. Halving is the cheapest way back
/// inside. Measured: without it, a head pitched 80° — a pose the solver is
/// otherwise exact on, between 75° and 85° both working — was refused outright.
const MAX_BACKTRACKS: usize = 4;

/// The iteration cap every decomposition in this module runs under.
///
/// nalgebra's `svd` passes `max_niter = 0`, which its own documentation defines
/// as "the algorithm continues indefinitely until convergence" — and an
/// iteration whose convergence test compares against a NaN never converges.
/// That is not hypothetical here: a mutant of this module reached the SVD with a
/// NaN and hung a test process for nineteen minutes of CPU. `SVD::try_new` takes
/// the cap instead, so every decomposition in this module is bounded by
/// construction and a run that exceeds the cap is `None`, which the caller turns
/// into [`HeadPose::INVALID`].
///
/// A hundred is far above what these systems need — a 12×12 DLT normal matrix
/// and two 3×3 rotations, all well conditioned — and far below anything a
/// 5 Hz frame budget would notice.
const MAX_SVD_ITERATIONS: usize = 100;

/// How far the rotation rebuilt from a returned [`Euler`] may sit from the
/// rotation it was read off, element by element.
///
/// Three sine/cosine pairs and two products in `f32` are worth about `1e-7`; a
/// decomposition that took the wrong branch is wrong by `O(1)`. `1e-4` sits two
/// orders of magnitude clear of the arithmetic and four below any real error.
const ROUND_TRIP_TOLERANCE: f32 = 1e-4;

/// √3 — the mean distance Hartley normalisation targets in three dimensions, to
/// the precision `f32` has for it. Spelled out as a literal because
/// `f32::consts::SQRT_3` is still behind `more_float_constants` on the pinned
/// toolchain.
const SQRT_3: f32 = 1.732_050_8;

/// [`MODEL`]'s own mean radius in millimetres: the distance from its centroid
/// that the observed spread is the pixel-space image of.
///
/// A constant rather than a loop per frame, pinned by a test that recomputes it
/// from [`MODEL`] — editing the model without editing this fails that test
/// rather than quietly mis-scaling the fallback distance that uses it.
const MODEL_RADIUS_MM: f32 = 46.99;

/// Head rotation about the camera's three axes, in degrees.
///
/// Zero is the face square to the camera. See the module docs for the sign of
/// each axis; they are not interchangeable and the directions are pinned by the
/// module's tests.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Euler {
    /// Rotation about the camera's `y` axis: positive turns the face toward the
    /// image's right. Within ±90° in every pose this module returns.
    pub yaw: f32,
    /// Rotation about the camera's `x` axis: positive tips the face down.
    /// Within ±90° in every pose this module returns.
    pub pitch: f32,
    /// Rotation about the camera's `z` axis: positive turns the face clockwise
    /// on screen. Finite always; unbounded by design, because a head tipped
    /// past its side is a real pose.
    pub roll: f32,
}

/// One frame's head pose: the angles, and whether they mean anything.
///
/// The only way to be invalid is [`HeadPose::INVALID`], whose `Euler` is zeroed
/// — so a caller that ignores `valid` reads a frontal pose rather than a
/// half-written one, and a caller that honours it (as Task 15 and Task 16 do)
/// gets no evidence at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadPose {
    /// The pose, in degrees. See [`Euler`].
    pub euler: Euler,
    /// `false` when the solve failed, did not converge, or could not explain
    /// the landmarks. `euler` is zeroed whenever this is `false`.
    pub valid: bool,
}

impl HeadPose {
    /// The one invalid pose: no angles, and no claim.
    ///
    /// A single value rather than a struct literal at each failure site, so
    /// that "invalid poses carry a zeroed `Euler`" is a property of the type
    /// instead of a rule every failure path has to remember.
    pub const INVALID: HeadPose = HeadPose {
        euler: Euler {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        },
        valid: false,
    };
}

/// Estimates head pose from one face's 468 landmarks, given the camera's
/// intrinsics.
///
/// Reads six of the 468 points — [`INDICES`] — and ignores the rest, including
/// every `z` (see the module docs). See [`HeadPose`] for what `valid` means.
pub fn estimate_head_pose(
    points: &[[i32; COORDINATES]; LANDMARKS],
    k: &CameraIntrinsics,
) -> HeadPose {
    // Every rejection below lands on the same value, so no failure path can
    // invent a pose.
    let Some(rotation) = solve(points, k) else {
        return HeadPose::INVALID;
    };
    let Some(euler) = euler_of(&rotation) else {
        return HeadPose::INVALID;
    };
    HeadPose { euler, valid: true }
}

/// The rotation of the six correspondence points, or `None` when they do not
/// determine one.
///
/// Four gates, in order, each of which can only turn a solve into a rejection:
/// usable intrinsics, six points that are actually spread out, a linear
/// initialisation that decomposes into a rotation in front of the camera, and a
/// refinement whose residual is small enough to be a fit rather than a
/// compromise.
fn solve(
    points: &[[i32; COORDINATES]; LANDMARKS],
    k: &CameraIntrinsics,
) -> Option<UnitQuaternion<f32>> {
    // The intrinsics are the caller's statement about their camera and reach
    // here unvalidated. `from_frame` returns `fx == fy == 0.0` for a zero-area
    // frame, and `from_calibration` stores whatever it is handed, NaN included
    // — and every one of those divides into the projection below. A focal
    // length that is zero, negative, infinite or NaN is not a camera, so there
    // is no pose to report from it.
    //
    // Written as a rejection rather than as its positive form throughout this
    // module: a NaN compares false against everything, so `is_finite` first and
    // a plain comparison second is the only order in which a NaN is rejected
    // instead of sailing through.
    if !k.fx.is_finite()
        || !k.fy.is_finite()
        || !k.cx.is_finite()
        || !k.cy.is_finite()
        || k.fx <= 0.0
        || k.fy <= 0.0
    {
        return None;
    }

    let observed = observed_points(points);
    let spread = spread_of(&observed)?;
    let (rotation, translation) = initial_pose(&observed, k)?;

    // Two places to start the refinement from, and the better fit wins.
    //
    // The linear estimate is a good start almost everywhere and a bad one where
    // the problem is ill-conditioned: rotation and distance are what a monocular
    // solve trades against each other, and a face foreshortened to a third of
    // its width is where that trade is worst. Measured, on the fixture the
    // steep-pitch test uses: the linear estimate settles at a **17 px** residual
    // on a 3.2 px allowance, and raising the iteration budget to a hundred does
    // not move it — a different basin, not slow convergence.
    //
    // The second start needs no linear algebra at all: the face is at the
    // distance its apparent size implies, looking straight ahead, and the
    // refinement takes it from there. Measured on the same fixture, it converges
    // to **0.2 px**. It costs one division and at most one more bounded
    // refinement, and it only runs when the first attempt failed to fit.
    let mut best = refine(rotation, translation, &observed, k).filter(|(_, _, r)| r.is_finite());
    let fitted = best.map_or(f32::INFINITY, |(_, _, residual)| residual);
    if !fitted.is_finite() || fitted > MAX_RESIDUAL_FRACTION * spread {
        let (fallback, start) = weak_perspective_start(&observed, spread, k);
        if let Some((rotation, translation, residual)) = refine(fallback, start, &observed, k)
            && residual.is_finite()
            && residual < fitted
        {
            best = Some((rotation, translation, residual));
        }
    }
    let (rotation, translation, residual) = best?;

    // Does the pose actually explain the landmarks? Everything upstream fits
    // *some* pose to *any* six points; this is the only gate that asks whether
    // the fit is a fit. See `MAX_RESIDUAL_FRACTION`.
    if !residual.is_finite() || residual > MAX_RESIDUAL_FRACTION * spread {
        return None;
    }
    // And is it a head at a laptop? A constellation with no perspective in it —
    // six points spread over a hundred million pixels, say — is fitted exactly
    // by a face 0.00001 mm from the lens, at which point every angle in the
    // solve is arithmetic noise.
    if !translation.z.is_finite() || translation.z < NEAREST_MM || translation.z > FARTHEST_MM {
        return None;
    }
    Some(rotation)
}

/// The second starting point: a face square to the camera, at the distance its
/// apparent size implies.
///
/// `z ≈ f·X/x` — the weak-perspective distance, exact for a face square to the
/// camera and a usable starting point when it is not, because
/// [`MODEL_RADIUS_MM`] and the observed spread are the same statistic measured
/// in two spaces. The lateral parts of the translation follow from the face's
/// centroid landing where it does on the sensor.
///
/// Deliberately **not** seeded with the linear estimate's rotation: at the
/// configuration where this is needed, that rotation is the part that is wrong,
/// and starting from it is what the measurement above rules out.
fn weak_perspective_start(
    observed: &[[f32; 2]; CORRESPONDENCES],
    spread: f32,
    k: &CameraIntrinsics,
) -> (UnitQuaternion<f32>, Vector3<f32>) {
    let mut centre = [0.0f32; 2];
    for point in observed {
        centre[0] += point[0];
        centre[1] += point[1];
    }
    centre[0] /= CORRESPONDENCES as f32;
    centre[1] /= CORRESPONDENCES as f32;

    let depth = 0.5 * (k.fx + k.fy) * MODEL_RADIUS_MM / spread;
    (
        UnitQuaternion::identity(),
        Vector3::new(
            (centre[0] - k.cx) * depth / k.fx,
            (centre[1] - k.cy) * depth / k.fy,
            depth,
        ),
    )
}

/// The six correspondence points, `[x, y]` each, as the floating-point pixels
/// the solve works in.
///
/// `z` is not read: see the module docs. An `i32` is exactly representable in
/// `f32` below 2^24 and rounds above it, so nothing here can produce a
/// non-finite coordinate from any input.
fn observed_points(points: &[[i32; COORDINATES]; LANDMARKS]) -> [[f32; 2]; CORRESPONDENCES] {
    let mut observed = [[0.0f32; 2]; CORRESPONDENCES];
    for (seen, index) in observed.iter_mut().zip(INDICES) {
        *seen = [points[index][0] as f32, points[index][1] as f32];
    }
    observed
}

/// How large the six observed points are — the mean distance from their own
/// centroid, in pixels — or `None` when they are too small to carry a pose.
///
/// Size, not position: a face at the frame's corner is as solvable as one at
/// its centre. This is the gate that rejects 468 copies of one pixel, where the
/// constellation has no extent at all and every rotation fits it equally.
fn spread_of(observed: &[[f32; 2]; CORRESPONDENCES]) -> Option<f32> {
    let mut centre = [0.0f32; 2];
    for point in observed {
        centre[0] += point[0];
        centre[1] += point[1];
    }
    centre[0] /= CORRESPONDENCES as f32;
    centre[1] /= CORRESPONDENCES as f32;

    let mut spread = 0.0f32;
    for point in observed {
        spread += ((point[0] - centre[0]).powi(2) + (point[1] - centre[1]).powi(2)).sqrt();
    }
    spread /= CORRESPONDENCES as f32;

    if spread.is_finite() && spread >= MIN_SPREAD_PX {
        Some(spread)
    } else {
        None
    }
}

/// The linear initialisation: the 3×4 projection matrix that maps the model
/// onto the observed points, split back into a rotation and a translation using
/// the known intrinsics.
///
/// This is the six-point DLT, so the system is exactly determined — twelve
/// equations for the eleven degrees of freedom of a projection matrix up to
/// scale — and its solution is the null space of that matrix, read off an SVD.
/// Both sides are Hartley-normalised first: the DLT's error is algebraic rather
/// than geometric, and it is dominated by whichever coordinate is written
/// largest, so raw pixels (`u ≈ 320`) against raw millimetres (`X ≈ 45`) would
/// let the principal point decide the answer. Normalising puts both centroids
/// at their origins with unit-ish extent, which is what the SVD needs to
/// separate the null space from the noise floor.
///
/// `None` when the system has no usable null space, when the split does not
/// produce a rotation, or when the resulting head is behind the camera — all of
/// which mean the linear step has nothing worth refining.
fn initial_pose(
    observed: &[[f32; 2]; CORRESPONDENCES],
    k: &CameraIntrinsics,
) -> Option<(UnitQuaternion<f32>, Vector3<f32>)> {
    let (to_pixels, seen) = normalise_2d(observed)?;
    let (to_millimetres, model) = normalise_3d();

    let mut system = SMatrix::<f32, 12, 12>::zeros();
    for (row, (point, seen)) in model.iter().zip(&seen).enumerate() {
        let [x, y, z] = *point;
        let [u, v] = *seen;
        // `p · [X Y Z 1]ᵀ` along the row of `P` that produces `u`, and the same
        // for `v`: two rows per correspondence, the classic DLT layout.
        system.set_row(
            2 * row,
            &SMatrix::<f32, 1, 12>::from_row_slice(&[
                x,
                y,
                z,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -u * x,
                -u * y,
                -u * z,
                -u,
            ]),
        );
        system.set_row(
            2 * row + 1,
            &SMatrix::<f32, 1, 12>::from_row_slice(&[
                0.0,
                0.0,
                0.0,
                0.0,
                x,
                y,
                z,
                1.0,
                -v * x,
                -v * y,
                -v * z,
                -v,
            ]),
        );
    }
    // nalgebra's SVD iterates rather than solving in one pass, and an iteration
    // that reads a NaN has no termination argument at all: comparisons against a
    // NaN are false, so its convergence test never fires. Everything above is
    // finite for finite input, so this check cannot fire today — it is here
    // because the alternative to an `Option` on this line is a camera thread
    // that never returns, and it costs 144 comparisons on a solve that runs five
    // times a second.
    if !system.iter().all(|value| value.is_finite()) {
        return None;
    }
    // `try_new`, not `svd`: the cap is what makes this terminate. See
    // `MAX_SVD_ITERATIONS`.
    let decomposition = SVD::try_new(system, true, true, f32::EPSILON, MAX_SVD_ITERATIONS)?;
    let v_t = decomposition.v_t?;
    // The singular values come back largest first, so the null space is the
    // last row: the direction the system squeezes hardest.
    let null = v_t.row(11);
    let normalised = Matrix3x4::from_fn(|row, column| null[column + 4 * row]);

    let projection = to_pixels.try_inverse()? * normalised * to_millimetres;
    let with_intrinsics = inverse_intrinsics(k) * projection;
    // The same termination guard as above, for the two three-by-three
    // decompositions below: a projection matrix can be finite on paper and still
    // overflow through the multiplications that build it.
    if !with_intrinsics.iter().all(|value| value.is_finite()) {
        return None;
    }
    let turn = with_intrinsics.fixed_columns::<3>(0).into_owned();
    let offset = with_intrinsics.column(3).into_owned();

    // `K⁻¹P = λ[R | t]` for one unknown λ, and `R`'s singular values are all
    // one, so the mean singular value of the left block is λ's magnitude. Its
    // *sign* is the only thing the SVD leaves free, and it is fixed here by the
    // one fact about a head that is not in doubt: it is in front of the camera.
    let singular =
        SVD::try_new(turn, false, false, f32::EPSILON, MAX_SVD_ITERATIONS)?.singular_values;
    let magnitude = (singular[0] + singular[1] + singular[2]) / 3.0;
    if !magnitude.is_finite() || magnitude <= 0.0 || !offset.z.is_finite() || offset.z == 0.0 {
        return None;
    }
    let scale = if offset.z > 0.0 { 1.0 } else { -1.0 } / magnitude;

    let scaled = turn * scale;
    // The last input to a decomposition, and the only one that a projection
    // matrix's own arithmetic can turn into a NaN — `scale` divides by the
    // magnitude checked above, so it is finite *today*, and "provably
    // unreachable" is the argument this module already had falsified once.
    if !scale.is_finite() || !scaled.iter().all(|value| value.is_finite()) {
        return None;
    }
    // A three-by-three `try_new` cannot report a failure to converge: nalgebra
    // computes the 3×3 case by eigendecomposition and its dispatcher discards
    // that `Option` (an `unsafe` transmute in `try_new_unordered`). The cap
    // still bounds the work; a non-convergent result arrives as a matrix of
    // garbage instead of a `None`, which the finiteness checks on `rotation`
    // below are what catch.
    let decomposition = SVD::try_new(scaled, true, true, f32::EPSILON, MAX_SVD_ITERATIONS)?;
    let (mut u, v_t) = (decomposition.u?, decomposition.v_t?);
    let mut rotation = u * v_t;
    if rotation.determinant() < 0.0 {
        // A reflection is not a rotation: flipping the smallest singular
        // direction is the nearest proper rotation to `scaled`.
        for value in u.column_mut(2).iter_mut() {
            *value = -*value;
        }
        rotation = u * v_t;
    }
    if !rotation.iter().all(|value| value.is_finite()) {
        return None;
    }
    // A linear estimate that puts the far side of the model behind the camera
    // has no projection to refine against, and there is nothing here worth
    // inventing: the refinement's own backtracking handles a step that
    // overshoots, and an estimate that is *already* outside cannot be repaired
    // by a correction this function would have to guess at. A push-back along
    // the optical axis was written here and deleted — with the budget in
    // `MAX_ITERATIONS` and the backtracking in `refine`, no fixture in this
    // module needs it, and a branch no test can reach is worse than none.
    let rotation =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation));
    let translation = offset * scale;
    if !translation.iter().all(|value| value.is_finite()) || !projects(&rotation, &translation) {
        return None;
    }
    Some((rotation, translation))
}

/// `K⁻¹`, written out rather than inverted: `fx` and `fy` are known positive
/// and finite by the time anything calls this, so the inverse is exact
/// arithmetic with no failure mode to handle.
fn inverse_intrinsics(k: &CameraIntrinsics) -> Matrix3<f32> {
    Matrix3::new(
        1.0 / k.fx,
        0.0,
        -k.cx / k.fx,
        0.0,
        1.0 / k.fy,
        -k.cy / k.fy,
        0.0,
        0.0,
        1.0,
    )
}

/// Hartley normalisation of the observed points: centroid at the origin, mean
/// distance √2. Returns the similarity that does it and the points it produced.
///
/// The extent check is a **termination guard, not an arithmetic one**, and it
/// stays even though [`spread_of`] refuses anything narrower before this is
/// called. The scale below divides by that extent, and an extent of zero makes
/// every normalised point a NaN — which the decomposition then hands to
/// nalgebra's SVD, whose iteration does not terminate on a NaN. Measured: a
/// single-line change to how the spread is measured (from the constellation's
/// own centre to the frame's origin) let 468 coincident landmarks past the
/// floor and turned the solve into an infinite loop, with the test process
/// burning 19 minutes of CPU before it was killed. A wrong pose is a bug; a
/// solve that never returns is a frozen camera thread.
fn normalise_2d(
    observed: &[[f32; 2]; CORRESPONDENCES],
) -> Option<(Matrix3<f32>, [[f32; 2]; CORRESPONDENCES])> {
    let mut centre = [0.0f32; 2];
    for point in observed {
        centre[0] += point[0];
        centre[1] += point[1];
    }
    centre[0] /= CORRESPONDENCES as f32;
    centre[1] /= CORRESPONDENCES as f32;

    let mut extent = 0.0f32;
    for point in observed {
        extent += ((point[0] - centre[0]).powi(2) + (point[1] - centre[1]).powi(2)).sqrt();
    }
    extent /= CORRESPONDENCES as f32;
    if !extent.is_finite() || extent <= 0.0 {
        return None;
    }
    let scale = std::f32::consts::SQRT_2 / extent;

    let mut normalised = [[0.0f32; 2]; CORRESPONDENCES];
    for (into, point) in normalised.iter_mut().zip(observed) {
        *into = [
            (point[0] - centre[0]) * scale,
            (point[1] - centre[1]) * scale,
        ];
    }
    Some((
        Matrix3::new(
            scale,
            0.0,
            -scale * centre[0],
            0.0,
            scale,
            -scale * centre[1],
            0.0,
            0.0,
            1.0,
        ),
        normalised,
    ))
}
/// Hartley normalisation of [`MODEL`]: centroid at the origin, mean distance
/// √3. The same construction as [`normalise_2d`], a dimension up.
fn normalise_3d() -> (Matrix4<f32>, [[f32; COORDINATES]; CORRESPONDENCES]) {
    let mut centre = [0.0f32; COORDINATES];
    for point in MODEL {
        for (axis, value) in centre.iter_mut().enumerate() {
            *value += point[axis];
        }
    }
    for value in centre.iter_mut() {
        *value /= CORRESPONDENCES as f32;
    }

    let mut extent = 0.0f32;
    for point in MODEL {
        extent += ((point[0] - centre[0]).powi(2)
            + (point[1] - centre[1]).powi(2)
            + (point[2] - centre[2]).powi(2))
        .sqrt();
    }
    extent /= CORRESPONDENCES as f32;
    // `MODEL` is a compile-time constant with six distinct points, so this is a
    // fixed positive number and there is nothing to fall back to. The model's
    // own test pins the structure that keeps it so; a degenerate edit would send
    // an infinity into the scale below, and the finiteness guards in the
    // decomposition are what turn that into `None` rather than into a hang.
    debug_assert!(extent.is_finite() && extent > 0.0, "MODEL has no extent");
    let scale = SQRT_3 / extent;

    let mut normalised = [[0.0f32; COORDINATES]; CORRESPONDENCES];
    for (into, point) in normalised.iter_mut().zip(MODEL) {
        for (axis, value) in into.iter_mut().enumerate() {
            *value = (point[axis] - centre[axis]) * scale;
        }
    }
    (
        Matrix4::new(
            scale,
            0.0,
            0.0,
            -scale * centre[0],
            0.0,
            scale,
            0.0,
            -scale * centre[1],
            0.0,
            0.0,
            scale,
            -scale * centre[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        normalised,
    )
}

/// Gauss-Newton over the six parameters — a rotation increment and a
/// translation increment — minimising the reprojection error of the six model
/// points.
///
/// Fixed [`MAX_ITERATIONS`] and no line search: with six correspondences and
/// six parameters the system is exactly determined, so once the linear estimate
/// is in the right basin the step is Newton's and converges in a handful of
/// iterations. A step that does not converge is not retried — it is handed to
/// the residual gate in [`solve`], which is what decides whether the answer is
/// worth reporting.
///
/// Returns the refined pose and the RMS reprojection residual in pixels.
fn refine(
    mut rotation: UnitQuaternion<f32>,
    mut translation: Vector3<f32>,
    observed: &[[f32; 2]; CORRESPONDENCES],
    k: &CameraIntrinsics,
) -> Option<(UnitQuaternion<f32>, Vector3<f32>, f32)> {
    for _ in 0..MAX_ITERATIONS {
        let (jacobian, errors) = linearise(&rotation, &translation, observed, k)?;
        let normal = jacobian.transpose() * jacobian;
        let gradient = jacobian.transpose() * errors;
        // A ridge proportional to the diagonal, plus a floor for the directions
        // the diagonal does not cover. Small enough not to bend the step, large
        // enough that a rank-deficient normal matrix solves instead of
        // returning `None` — and 1e-12 of a residual measured in pixels is far
        // below the landmark rounding the answer already carries.
        let mut damped = normal;
        for axis in 0..6 {
            damped[(axis, axis)] = damped[(axis, axis)] * (1.0 + 1e-6) + 1e-12;
        }
        let step = damped.lu().solve(&(-gradient))?;
        if !step.iter().all(|value| value.is_finite()) {
            return None;
        }
        // `R ← R·exp(δ)`: the increment acts on the model, which is what
        // `linearise` differentiates against. A step that would leave the
        // region where the projection exists is halved rather than abandoned —
        // see `MAX_BACKTRACKS`.
        let increment = Vector3::new(step[0], step[1], step[2]);
        let advance = Vector3::new(step[3], step[4], step[5]);
        let mut moved = None;
        for (halvings, _) in (0..MAX_BACKTRACKS).enumerate() {
            let length = 2.0f32.powi(-(halvings as i32));
            let candidate = rotation * UnitQuaternion::from_scaled_axis(length * increment);
            let shifted = translation + length * advance;
            if projects(&candidate, &shifted) {
                moved = Some((candidate, shifted, length));
                break;
            }
        }
        let (candidate, shifted, length) = moved?;
        rotation = candidate;
        translation = shifted;
        if length * step.norm() < CONVERGED_STEP {
            break;
        }
    }

    // The residual at the pose actually returned, not at the one the last step
    // started from.
    let (_, errors) = linearise(&rotation, &translation, observed, k)?;
    let residual = errors.norm() / (2.0 * CORRESPONDENCES as f32).sqrt();
    Some((rotation, translation, residual))
}

/// Whether every model point is in front of the camera at this pose.
///
/// The cheap form of the check [`linearise`] makes for itself: this one only
/// decides whether a step is worth taking, and does not need the Jacobian that
/// comes with the answer.
fn projects(rotation: &UnitQuaternion<f32>, translation: &Vector3<f32>) -> bool {
    let matrix = rotation.to_rotation_matrix();
    let matrix = matrix.matrix();
    MODEL.iter().all(|model| {
        let depth = (matrix * Vector3::new(model[0], model[1], model[2]) + translation).z;
        depth.is_finite() && depth > 0.0
    })
}

/// The reprojection residual vector and its Jacobian at a pose: two rows per
/// correspondence, six columns — three for the rotation increment `δ` and three
/// for the translation increment.
///
/// `None` when any model point lands on or behind the camera, where the
/// projection divides by zero or flips sign and every number below it is
/// meaningless.
fn linearise(
    rotation: &UnitQuaternion<f32>,
    translation: &Vector3<f32>,
    observed: &[[f32; 2]; CORRESPONDENCES],
    k: &CameraIntrinsics,
) -> Option<(SMatrix<f32, 12, 6>, SVector<f32, 12>)> {
    let matrix = *rotation.to_rotation_matrix().matrix();
    let mut jacobian = SMatrix::<f32, 12, 6>::zeros();
    let mut residual = SVector::<f32, 12>::zeros();

    for (row, (model, seen)) in MODEL.iter().zip(observed).enumerate() {
        let point = matrix * Vector3::new(model[0], model[1], model[2]) + translation;
        // A point at the camera's own plane has no projection, and one behind
        // it projects to the mirror image of where it is: both are `None`, not
        // a large number.
        if !point.z.is_finite() || point.z <= 0.0 {
            return None;
        }
        let inverse = 1.0 / point.z;
        let projected = [
            k.fx * point.x * inverse + k.cx,
            k.fy * point.y * inverse + k.cy,
        ];
        residual[2 * row] = projected[0] - seen[0];
        residual[2 * row + 1] = projected[1] - seen[1];

        // d(u, v)/d(point): the pinhole projection's own derivative.
        let about_x = Vector3::new(k.fx * inverse, 0.0, -k.fx * point.x * inverse * inverse);
        let about_y = Vector3::new(0.0, k.fy * inverse, -k.fy * point.y * inverse * inverse);
        // d(point)/d(δ) = -R·[X]× for a right-multiplied increment.
        let skew = Matrix3::new(
            0.0, -model[2], model[1], model[2], 0.0, -model[0], -model[1], model[0], 0.0,
        );
        let about_rotation = -(matrix * skew);

        for axis in 0..3 {
            jacobian[(2 * row, axis)] = about_x.dot(&about_rotation.column(axis));
            jacobian[(2 * row + 1, axis)] = about_y.dot(&about_rotation.column(axis));
            // The translation enters the projection directly.
            jacobian[(2 * row, 3 + axis)] = about_x[axis];
            jacobian[(2 * row + 1, 3 + axis)] = about_y[axis];
        }
    }
    Some((jacobian, residual))
}

/// The three angles of a rotation, in degrees, or `None` when this
/// representation cannot state it honestly.
///
/// The rotation is split as `R = Rz(roll)·Ry(-yaw)·Rx(pitch)` — the composition
/// the module documents and this module's tests project — which makes
/// `R[2][0] = sin(yaw)`, `R[2][1] = sin(pitch)` and `R[1][0] = sin(roll)`.
///
/// A pitch past ±90° is the same rotation written the other way round: the
/// extraction would have to report either a number outside the ±90° the caller
/// is promised or a clamped one that is not the pose. Both are a plausible
/// wrong answer, so neither is returned.
fn euler_of(rotation: &UnitQuaternion<f32>) -> Option<Euler> {
    let matrix = rotation.to_rotation_matrix();
    let matrix = matrix.matrix();
    // `asin` is defined on `-1..=1`, and a unit quaternion's matrix is inside
    // that to within f32 rounding — a search of 25 million quaternions on a
    // 1e-4 grid never produced an out-of-range element. If one ever appeared,
    // `asin` would return a NaN and the finiteness gate below would refuse the
    // pose: no NaN leaves this function either way, which is the property that
    // matters. An explicit clamp here was deleted for being unreachable.
    let yaw = matrix[(2, 0)].asin().to_degrees();
    let pitch = matrix[(2, 1)].atan2(matrix[(2, 2)]).to_degrees();
    let roll = matrix[(1, 0)].atan2(matrix[(0, 0)]).to_degrees();
    if !yaw.is_finite() || !pitch.is_finite() || !roll.is_finite() || pitch.abs() > 90.0 {
        return None;
    }
    // The bound the caller is promised, applied last so that it holds on every
    // path out of this function. `asin` cannot leave ±90° and the gate above
    // refuses a pitch past it, so neither clamp can fire today; they are here
    // so that the bound is a property of the return value rather than of the
    // two functions that happen to produce it.
    let euler = Euler {
        yaw: yaw.clamp(-90.0, 90.0),
        pitch: pitch.clamp(-90.0, 90.0),
        roll,
    };
    // Three angles read off three matrix elements are not a decomposition until
    // they are put back together. Recomposing is the only check that looks at
    // the whole rotation rather than at the elements the extraction happened to
    // read, so it is what catches a branch taken the wrong side, a sign lost in
    // the conversion, or the pair of angles that trade off against each other
    // at a gimbal.
    if !composes_to(&euler, matrix) {
        return None;
    }
    Some(euler)
}

/// Whether the module's own composition of `euler` —
/// `Rz(roll)·Ry(-yaw)·Rx(pitch)`, the one the docs state and the tests project —
/// reproduces `matrix`.
fn composes_to(euler: &Euler, matrix: &Matrix3<f32>) -> bool {
    let (sy, cy) = euler.yaw.to_radians().sin_cos();
    let (sp, cp) = euler.pitch.to_radians().sin_cos();
    let (sr, cr) = euler.roll.to_radians().sin_cos();
    let about_x = Matrix3::new(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
    // `Ry(-yaw)`: the sign that makes positive yaw turn the face toward the
    // image's right.
    let about_y = Matrix3::new(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    let about_z = Matrix3::new(cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0);
    let rebuilt = about_z * about_y * about_x;
    rebuilt
        .iter()
        .zip(matrix.iter())
        .all(|(rebuilt, original)| (rebuilt - original).abs() <= ROUND_TRIP_TOLERANCE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The intrinsics Task 7 estimates for a 640×480 frame, as literals: the
    /// focal length is the frame's wider dimension and the principal point is
    /// its exact centre. Written out rather than built through `from_frame` so
    /// that a change to *that* estimate cannot silently re-tune these tests.
    fn k() -> CameraIntrinsics {
        CameraIntrinsics::from_calibration(640.0, 640.0, 320.0, 240.0)
    }

    /// `Euler` written as its three numbers.
    fn euler(yaw: f32, pitch: f32, roll: f32) -> Euler {
        Euler { yaw, pitch, roll }
    }

    /// The rotation the module documents, written out here in full — three
    /// sine/cosine pairs and three matrix products, in the test rather than
    /// called from the solve.
    ///
    /// `R = Rz(roll) · Ry(-yaw) · Rx(pitch)`: the sign on the `y` term is what
    /// makes positive `yaw` turn the face toward the image's right, and it is
    /// the only place in these tests where the convention is written down.
    fn rotation(angle: Euler) -> [[f32; COORDINATES]; 3] {
        let (sy, cy) = angle.yaw.to_radians().sin_cos();
        let (sp, cp) = angle.pitch.to_radians().sin_cos();
        let (sr, cr) = angle.roll.to_radians().sin_cos();
        let about_x = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]];
        let about_y = [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]];
        let about_z = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
        multiply(about_z, multiply(about_y, about_x))
    }

    /// The 3×3 product `a · b`, spelled out because the test has no matrix
    /// crate to borrow and three loops is clearer than a dependency.
    fn multiply(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for (row, out_row) in out.iter_mut().enumerate() {
            for (column, cell) in out_row.iter_mut().enumerate() {
                *cell = (0..3).map(|n| a[row][n] * b[n][column]).sum();
            }
        }
        out
    }

    /// The landmarks a face at a known pose would produce: the six model points
    /// placed at `translation`, rotated by `angle`, projected through `k` and
    /// rounded to the whole pixels the decode emits.
    ///
    /// Every one of the 462 points this solve does not read is left at the
    /// origin, and every `z` is left at zero — the solve consults neither, and
    /// `only_the_six_correspondence_indices_are_read` is what holds it to that.
    fn project_at(
        angle: Euler,
        translation: [f32; COORDINATES],
        k: &CameraIntrinsics,
    ) -> [[i32; COORDINATES]; LANDMARKS] {
        let r = rotation(angle);
        let mut points = [[0i32; COORDINATES]; LANDMARKS];
        for (slot, model) in INDICES.iter().zip(MODEL) {
            let mut camera = [0.0f32; COORDINATES];
            for (row, value) in camera.iter_mut().enumerate() {
                *value =
                    (0..COORDINATES).map(|n| r[row][n] * model[n]).sum::<f32>() + translation[row];
            }
            points[*slot] = [
                (k.fx * camera[0] / camera[2] + k.cx).round() as i32,
                (k.fy * camera[1] / camera[2] + k.cy).round() as i32,
                0,
            ];
        }
        points
    }

    /// The same, centred: a face half a metre away by default.
    fn project(
        angle: Euler,
        distance: f32,
        k: &CameraIntrinsics,
    ) -> [[i32; COORDINATES]; LANDMARKS] {
        project_at(angle, [0.0, 0.0, distance], k)
    }

    #[test]
    fn frontal_landmarks_produce_near_zero_angles() {
        // The six model points projected square-on through the intrinsics, at
        // three distances: a pose with no rotation must come back as a pose
        // with no rotation, and the distance must not leak into the angles.
        for distance in [400.0, 600.0, 900.0] {
            let pose = estimate_head_pose(&project(Euler::default(), distance, &k()), &k());
            assert!(
                pose.valid,
                "a frontal face {distance} mm away is a solvable pose, got {pose:?}"
            );
            assert!(
                pose.euler.yaw.abs() < 2.0 && pose.euler.pitch.abs() < 2.0,
                "square-on means square-on at {distance} mm, got {pose:?}"
            );
            assert!(
                pose.euler.roll.abs() < 2.0,
                "and level, at {distance} mm, got {pose:?}"
            );
        }
    }

    #[test]
    fn a_known_pose_is_recovered_on_each_axis_and_their_combination() {
        // The expectation for every case is the pose that was projected — a
        // number that exists before the solve runs and was not read off it.
        // Tolerance 2°: the fixture rounds each landmark to a whole pixel,
        // which on a face ~48 px across is worth a few tenths of a degree.
        let cases = [
            euler(20.0, 0.0, 0.0),
            euler(-20.0, 0.0, 0.0),
            euler(35.0, 0.0, 0.0),
            euler(0.0, 15.0, 0.0),
            euler(0.0, -15.0, 0.0),
            euler(0.0, 0.0, 12.0),
            euler(0.0, 0.0, -12.0),
            euler(20.0, 10.0, 5.0),
            euler(-25.0, -15.0, -8.0),
        ];
        for wanted in cases {
            let pose = estimate_head_pose(&project(wanted, 600.0, &k()), &k());
            assert!(pose.valid, "{wanted:?} is a solvable pose, got {pose:?}");
            // Asserted per axis so a failure says which angle is wrong.
            assert!(
                (pose.euler.yaw - wanted.yaw).abs() < 2.0,
                "yaw: wanted {}, got {} ({wanted:?})",
                wanted.yaw,
                pose.euler.yaw
            );
            assert!(
                (pose.euler.pitch - wanted.pitch).abs() < 2.0,
                "pitch: wanted {}, got {} ({wanted:?})",
                wanted.pitch,
                pose.euler.pitch
            );
            assert!(
                (pose.euler.roll - wanted.roll).abs() < 2.0,
                "roll: wanted {}, got {} ({wanted:?})",
                wanted.roll,
                pose.euler.roll
            );
        }
    }

    #[test]
    fn a_yawed_head_reports_a_larger_yaw_than_a_frontal_one() {
        let k = k();
        let frontal = estimate_head_pose(&project(Euler::default(), 600.0, &k), &k);
        assert!(frontal.valid);
        for yaw in [15.0, 25.0, 40.0] {
            let turned = estimate_head_pose(&project(euler(yaw, 0.0, 0.0), 600.0, &k), &k);
            assert!(
                turned.valid,
                "{yaw}° of yaw is a solvable pose, got {turned:?}"
            );
            assert!(
                turned.euler.yaw > frontal.euler.yaw + 5.0,
                "{yaw}° of yaw must report more than the frontal {}: got {}",
                frontal.euler.yaw,
                turned.euler.yaw
            );
            // And the size of the reading tracks the size of the turn, at least
            // in the direction that matters: a turn twice as far must not read
            // as a turn half as far.
            assert!(
                turned.euler.yaw > yaw / 2.0,
                "{yaw}° of yaw read as {}, which is not even half of the way there",
                turned.euler.yaw
            );
        }
        // Turning the other way is not the same answer with a different sign
        // lost: it is the mirror, and its sign is negative.
        let mirrored = estimate_head_pose(&project(euler(-25.0, 0.0, 0.0), 600.0, &k), &k);
        assert!(mirrored.euler.yaw < -5.0, "got {mirrored:?}");
    }

    #[test]
    fn head_pose_is_bounded_and_never_nan() {
        let k = k();
        // Six collinear correspondence points: distinct pixels, no area, and no
        // rigid face projects to a line. Distinct matters — this fixture is what
        // kills a solve that measures the constellation from the frame's origin
        // rather than from its own centre, which coincident points could not.
        let mut line = [[0i32; COORDINATES]; LANDMARKS];
        for (step, slot) in INDICES.iter().enumerate() {
            line[*slot] = [step as i32 * 90, step as i32 * 70, 0];
        }
        // Six points spread across the whole `i32` range — a "face" forty
        // million pixels wide, which is a face 0.000014 mm from the lens.
        let mut wild = [[0i32; COORDINATES]; LANDMARKS];
        let corners = [
            [-2_000_000_000, -2_000_000_000, 0],
            [2_000_000_000, -2_000_000_000, 0],
            [0, 0, 0],
            [-2_000_000_000, 2_000_000_000, 0],
            [2_000_000_000, 2_000_000_000, 0],
            [0, 2_000_000_000, 0],
        ];
        for (slot, corner) in INDICES.iter().zip(corners) {
            wild[*slot] = corner;
        }
        // A face so far away that its six landmarks land on a couple of pixels.
        let distant = project(Euler::default(), 30_000.0, &k);
        // A face 30 mm from the lens: the landmarks span two thousand pixels,
        // which is not a head at a laptop.
        let pressed = project_at(Euler::default(), [0.0, 0.0, 30.0], &k);
        // A frontal face eleven metres away, spanning two or three pixels. This
        // one is not a shape no camera could see — it is a perfectly ordinary
        // projection — but the rounding to whole pixels is a fifth of the
        // constellation, and the pose it implies is arithmetic noise: measured
        // at 19.6° of pitch for a face that is square to the camera.
        let across_the_room = project(Euler::default(), 11_000.0, &k);
        // The same distance through a **long lens**, where the floor cannot
        // help: a 4K frame's focal length is 3840 px, so a face 13 m away still
        // spans 13 px of spread and passes the floor comfortably. What refuses
        // it is the distance bound — which is the whole reason `FARTHEST_MM`
        // exists rather than being left to the floor.
        let long_lens = CameraIntrinsics::from_calibration(3840.0, 3840.0, 1920.0, 1080.0);
        let far_through_a_long_lens = project(Euler::default(), 13_000.0, &long_lens);
        let near_through_a_long_lens = project(Euler::default(), 700.0, &long_lens);
        let frontal = project(euler(20.0, 10.0, 5.0), 600.0, &k);

        let cases = [
            ("all 468 points at the origin", [[0, 0, 0]; LANDMARKS], k),
            (
                "all 468 points at one mid-frame pixel",
                [[320, 240, 0]; LANDMARKS],
                k,
            ),
            (
                "all 468 points at i32::MAX",
                [[i32::MAX; COORDINATES]; LANDMARKS],
                k,
            ),
            (
                "all 468 points at i32::MIN",
                [[i32::MIN; COORDINATES]; LANDMARKS],
                k,
            ),
            ("the six correspondence points on one line", line, k),
            ("six points spread across the i32 range", wild, k),
            ("a face too far away to have a pose", distant, k),
            ("a face 30 mm from the lens", pressed, k),
            ("a frontal face eleven metres away", across_the_room, k),
            (
                "a frontal face against NaN intrinsics",
                frontal,
                CameraIntrinsics::from_calibration(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            ),
            (
                "a frontal face against an infinite focal length",
                frontal,
                CameraIntrinsics::from_calibration(f32::INFINITY, f32::INFINITY, 320.0, 240.0),
            ),
            (
                "a frontal face against a negative focal length",
                frontal,
                CameraIntrinsics::from_calibration(-640.0, -640.0, 320.0, 240.0),
            ),
        ];
        for (what, points, k) in cases {
            let pose = estimate_head_pose(&points, &k);
            assert!(
                pose.euler.yaw.is_finite()
                    && pose.euler.pitch.is_finite()
                    && pose.euler.roll.is_finite(),
                "{what}: every angle must be finite, got {pose:?}"
            );
            assert!(
                pose.euler.yaw.abs() <= 90.0 && pose.euler.pitch.abs() <= 90.0,
                "{what}: yaw and pitch are bounded by 90°, got {pose:?}"
            );
            assert!(
                !pose.valid || pose.euler.yaw.is_finite(),
                "{what}: a valid pose is a finite one, got {pose:?}"
            );
            if !pose.valid {
                assert_eq!(
                    pose,
                    HeadPose::INVALID,
                    "{what}: a rejected pose carries no angles"
                );
            }
        }

        // None of these is a pose, so none of them may come back as one. A
        // solver that answers anyway is the failure this module exists to
        // prevent: a finite, plausible, wrong Euler is read downstream as a
        // real head movement.
        let unsolvable = [
            ("all 468 points at the origin", [[0, 0, 0]; LANDMARKS], k),
            (
                "468 copies of one mid-frame pixel",
                [[320, 240, 0]; LANDMARKS],
                k,
            ),
            (
                "all 468 points at i32::MAX",
                [[i32::MAX; COORDINATES]; LANDMARKS],
                k,
            ),
            ("the six correspondence points on one line", line, k),
            ("six points spread across the i32 range", wild, k),
            ("a face too far away to have a pose", distant, k),
            ("a face 30 mm from the lens", pressed, k),
            ("a frontal face eleven metres away", across_the_room, k),
            (
                "a face thirteen metres away through a long lens",
                far_through_a_long_lens,
                long_lens,
            ),
            (
                "zeroed intrinsics, as `from_frame` gives a zero-area frame",
                frontal,
                CameraIntrinsics::from_calibration(0.0, 0.0, 0.0, 0.0),
            ),
            (
                "NaN intrinsics",
                frontal,
                CameraIntrinsics::from_calibration(f32::NAN, f32::NAN, f32::NAN, f32::NAN),
            ),
            (
                "an infinite focal length",
                frontal,
                CameraIntrinsics::from_calibration(f32::INFINITY, f32::INFINITY, 320.0, 240.0),
            ),
            (
                "a negative focal length",
                frontal,
                CameraIntrinsics::from_calibration(-640.0, -640.0, 320.0, 240.0),
            ),
        ];
        for (what, points, k) in unsolvable {
            assert_eq!(
                estimate_head_pose(&points, &k),
                HeadPose::INVALID,
                "{what} has no pose, and must say so rather than guess"
            );
        }

        // The same landmarks against working intrinsics still solve: the
        // rejections above are about the input, not about a solver that has
        // stopped answering — and that includes the long lens, where the only
        // rejected fixture is the one thirteen metres out.
        assert!(estimate_head_pose(&frontal, &k).valid);
        assert!(
            estimate_head_pose(&near_through_a_long_lens, &long_lens).valid,
            "a 4K frame with a face 700 mm away is an ordinary pose"
        );
    }

    #[test]
    fn an_impossible_correspondence_set_is_rejected_and_a_rounded_one_is_not() {
        let k = k();
        let clean = project(euler(15.0, 5.0, 0.0), 600.0, &k);

        // One pixel on every point: the fixture's own rounding, which is not a
        // corruption. Rejecting this would reject most real frames — so the
        // assertions below are about a gate that discriminates, not one that
        // refuses everything.
        let mut rounded = clean;
        for (index, point) in rounded.iter_mut().enumerate() {
            if INDICES.contains(&index) {
                point[0] += 1;
            }
        }
        let pose = estimate_head_pose(&rounded, &k);
        assert!(
            pose.valid,
            "a one-pixel nudge is landmark noise, got {pose:?}"
        );
        assert!(
            (pose.euler.yaw - 15.0).abs() < 3.0,
            "and it barely moves the answer: got {}",
            pose.euler.yaw
        );

        // An occluded lower face: the two mouth corners and the chin land on
        // the nose, which is what Face Mesh does when a hand, a mug or a mask
        // hides them. The six points no longer describe a rigid face, so the
        // best pose available is several pixels out — measured, 13.3 px of RMS
        // residual against a 2.6 px allowance — and reporting that pose would
        // report a confident head movement that never happened.
        let occluded = |points: &mut [[i32; COORDINATES]; LANDMARKS]| {
            points[61] = points[1];
            points[291] = points[1];
            points[199] = points[1];
        };
        let mut collapsed = clean;
        occluded(&mut collapsed);
        assert_eq!(
            estimate_head_pose(&collapsed, &k),
            HeadPose::INVALID,
            "a lower face collapsed onto the nose has no pose"
        );

        // The same corruption on a face that is not at the centre of the frame.
        // The allowance is a fraction of the face's own *size*, so where the
        // face sits must not change the answer: a solve that measured the
        // constellation from the frame's origin instead of from its own centre
        // would give this one a 27 px allowance and swallow the corruption.
        let mut off_centre = project_at(euler(10.0, 5.0, 0.0), [200.0, 120.0, 600.0], &k);
        assert!(
            estimate_head_pose(&off_centre, &k).valid,
            "the off-centre face itself is an ordinary pose"
        );
        occluded(&mut off_centre);
        assert_eq!(
            estimate_head_pose(&off_centre, &k),
            HeadPose::INVALID,
            "an off-centre face has the same allowance as a centred one"
        );

        // And a single landmark far out of place: the chin 80 px below where
        // the other five put it. No rigid face fits that either, though this
        // one is refused earlier — the refinement cannot even evaluate a pose
        // for it — so it pins the *response* rather than one particular gate.
        let mut broken = clean;
        broken[199][1] += 80;
        assert_eq!(
            estimate_head_pose(&broken, &k),
            HeadPose::INVALID,
            "a chin 80 px from where the rest of the face puts it has no pose"
        );
    }

    #[test]
    fn a_noisy_face_is_still_solved() {
        // Real landmarks are not the exact projection of any model: they carry
        // the network's own error, a pixel or two at this resolution, and a
        // face is never quite the generic one. The solve has to keep answering
        // through that — a refusal is safe, but it costs the engine its only
        // head-pose evidence, and a gate that refuses honest measurements is as
        // useless as one that accepts dishonest ones. This is the other side of
        // `an_impossible_correspondence_set_is_rejected_and_a_rounded_one_is_not`.
        let k = k();
        let cases = [
            Euler::default(),
            euler(20.0, 0.0, 0.0),
            euler(-20.0, 0.0, 0.0),
            euler(0.0, 15.0, 0.0),
            euler(0.0, -15.0, 0.0),
            euler(0.0, 0.0, 15.0),
            euler(10.0, 10.0, -10.0),
        ];
        for wanted in cases {
            let mut points = project(wanted, 600.0, &k);
            // One pixel, in a fixed pattern, on every one of the six points:
            // deterministic, and the same size as the rounding the decode does
            // to every landmark it emits.
            for (step, index) in INDICES.iter().enumerate() {
                let bump = if step % 2 == 0 { 1 } else { -1 };
                points[*index][0] += bump;
                points[*index][1] -= bump;
            }
            let pose = estimate_head_pose(&points, &k);
            assert!(
                pose.valid,
                "a pixel of landmark noise is not a reason to give up ({wanted:?} -> {pose:?})"
            );
            for (what, got, want) in [
                ("yaw", pose.euler.yaw, wanted.yaw),
                ("pitch", pose.euler.pitch, wanted.pitch),
                ("roll", pose.euler.roll, wanted.roll),
            ] {
                assert!(
                    (got - want).abs() < 4.0,
                    "{what} under one pixel of noise: wanted {want}, got {got}"
                );
            }
        }
    }

    #[test]
    fn an_extreme_rotation_comes_back_as_the_same_rotation_or_not_at_all() {
        // A head turned and tipped past both of the angles this module promises
        // to report. The two fixtures below are the *same pixels* — measured,
        // all six correspondence points identical — because yaw 100°, pitch
        // −100°, roll 0° and yaw 80°, pitch 80°, roll 180° are one rotation
        // written two ways. So no implementation can accept one and refuse the
        // other: what comes back is the principal representative of that single
        // rotation, with |yaw| and |pitch| inside the promised ±90°, and the
        // claim it makes about the head has to still be true.
        let k = k();
        let past = project(euler(100.0, -100.0, 0.0), 600.0, &k);
        let inside = project(euler(80.0, 80.0, 180.0), 600.0, &k);
        for index in INDICES {
            assert_eq!(
                past[index], inside[index],
                "the two spellings of this rotation are the same landmarks"
            );
        }
        assert_eq!(
            estimate_head_pose(&past, &k),
            estimate_head_pose(&inside, &k),
            "and therefore the same answer"
        );

        for wanted in [
            euler(100.0, -100.0, 0.0),
            euler(80.0, 80.0, 180.0),
            euler(100.0, -100.0, 30.0),
            euler(89.0, 0.0, 0.0),
        ] {
            let pose = estimate_head_pose(&project(wanted, 600.0, &k), &k);
            assert!(
                pose.valid,
                "a rotation this module can state is reported: {wanted:?} -> {pose:?}"
            );
            assert!(
                pose.euler.yaw.abs() <= 90.0 && pose.euler.pitch.abs() <= 90.0,
                "and stated inside the promised range: {pose:?}"
            );
            // The returned angles name the rotation the fixture projected — not
            // a nearby one. The tolerance is the fixture's own rounding to whole
            // pixels, measured at 7e-3 of matrix element on this constellation,
            // where the module's own recomposition check inside `euler_of` runs
            // at 1e-4 because it never sees a rounded landmark.
            let projected = rotation(wanted);
            let rebuilt = rotation(pose.euler);
            let mut worst = 0.0f32;
            for row in 0..3 {
                for column in 0..3 {
                    worst = worst.max((projected[row][column] - rebuilt[row][column]).abs());
                }
            }
            assert!(
                worst < 0.05,
                "{wanted:?} came back as {:?}, which is a different rotation ({worst})",
                pose.euler
            );
        }

        // And the rotations this representation genuinely cannot state are still
        // refused rather than re-labelled: past the yaw singularity the aliased
        // triple needs a pitch outside ±90°, which is what the gate catches.
        assert_eq!(
            estimate_head_pose(&project(euler(95.0, 0.0, 0.0), 600.0, &k), &k),
            HeadPose::INVALID,
            "past the yaw singularity there is no triple inside the range"
        );
    }

    #[test]
    fn a_head_tipped_past_vertical_is_refused_rather_than_clamped() {
        // A face tipped 120° about the camera's x axis: chin up, the eyes
        // behind the nose. The solve recovers it — the landmarks are exactly
        // that projection — and the pose is real, but `pitch` cannot state it
        // inside the ±90° the caller is promised. Clamping to 90 would report
        // "looking straight up" for a head that is past that, which is the
        // plausible-looking wrong answer this module refuses; the honest answer
        // is that there is no pose here to report.
        let k = k();
        assert_eq!(
            estimate_head_pose(&project(euler(0.0, 120.0, 0.0), 600.0, &k), &k),
            HeadPose::INVALID,
            "a pitch past ±90° is not a pose this representation can state"
        );
        // Just inside the bound is still reportable, so the refusal above is
        // about the representation and not about steep pitches in general.
        let steep = estimate_head_pose(&project(euler(0.0, 80.0, 0.0), 600.0, &k), &k);
        assert!(steep.valid, "80° of pitch is a pose, got {steep:?}");
        assert!(
            (steep.euler.pitch - 80.0).abs() < 2.0,
            "and it is reported as itself: got {}",
            steep.euler.pitch
        );
    }

    #[test]
    fn only_the_six_correspondence_indices_are_read() {
        let k = k();
        let clean = project(euler(18.0, -6.0, 4.0), 550.0, &k);
        let wanted = estimate_head_pose(&clean, &k);

        // The other 462 points, moved somewhere absurd: no effect at all, bit
        // for bit. This is what kills a solve that reads `points[0..6]`, or
        // that ranks the mesh by some score, or that lets the eye *inner*
        // corners in.
        let mut elsewhere = clean;
        for (index, point) in elsewhere.iter_mut().enumerate() {
            if !INDICES.contains(&index) {
                *point = [index as i32 * 13 - 3000, 4000 - index as i32 * 7, 0];
            }
        }
        assert_eq!(
            estimate_head_pose(&elsewhere, &k),
            wanted,
            "the pose comes from the six correspondence points and nothing else"
        );

        // And `z` is one of the things not read: the six points' third column
        // set to values an order of magnitude past anything the model's own
        // `z` column holds changes nothing. See the module docs — this is the
        // deliberate half of the `z` decision, and it is pinned here.
        let mut with_z = clean;
        for (index, point) in with_z.iter_mut().enumerate() {
            if INDICES.contains(&index) {
                point[2] = 900 + index as i32;
            }
        }
        assert_eq!(
            estimate_head_pose(&with_z, &k),
            wanted,
            "landmark z is not frame pixels and is not used"
        );

        // The correspondence points are not decoration: moving one of them
        // does change the answer, so the assertions above are not the whole
        // solve ignoring its input.
        let mut moved = clean;
        moved[1][1] -= 4;
        assert_ne!(
            estimate_head_pose(&moved, &k),
            wanted,
            "a correspondence point that moves moves the pose"
        );
    }

    #[test]
    fn the_euler_signs_follow_the_face_in_the_image() {
        // The sign convention, checked against the pixels rather than against
        // the solve: for each axis, the landmarks are projected at a known
        // positive angle, the *image* is asked which way the face went, and the
        // solver is required to agree with it. A solver whose yaw sign is
        // flipped passes no assertion here, whatever its own arithmetic says.
        let k = k();
        let upward = |point: [i32; COORDINATES]| point[1];
        let sideways = |point: [i32; COORDINATES]| point[0];

        let turned = project(euler(25.0, 0.0, 0.0), 600.0, &k);
        let eye_middle = (sideways(turned[33]) + sideways(turned[263])) as f32 / 2.0;
        assert!(
            sideways(turned[1]) as f32 > eye_middle,
            "+25° yaw must put the nose right of the eye line: {} vs {eye_middle}",
            sideways(turned[1])
        );
        assert!(
            estimate_head_pose(&turned, &k).euler.yaw > 0.0,
            "and the solve must report it as positive yaw"
        );

        let tipped = project(euler(0.0, 25.0, 0.0), 600.0, &k);
        let eye_line = (upward(tipped[33]) + upward(tipped[263])) as f32 / 2.0;
        assert!(
            upward(tipped[1]) as f32 > eye_line,
            "+25° pitch must put the nose below the eye line: {} vs {eye_line}",
            upward(tipped[1])
        );
        assert!(
            estimate_head_pose(&tipped, &k).euler.pitch > 0.0,
            "and the solve must report it as positive pitch"
        );

        let tilted = project(euler(0.0, 0.0, 25.0), 600.0, &k);
        assert!(
            upward(tilted[263]) > upward(tilted[33]),
            "+25° roll must drop the image-right eye corner below the left one: {} vs {}",
            upward(tilted[263]),
            upward(tilted[33])
        );
        assert!(
            estimate_head_pose(&tilted, &k).euler.roll > 0.0,
            "and the solve must report it as positive roll"
        );
    }

    #[test]
    fn the_face_model_is_the_generic_symmetric_one_the_module_documents() {
        // The index set against the graph's own topology, written as literals:
        // 33 and 263 are the outer eye corners and 61/291 the mouth corners,
        // and the model below is only meaningful paired with them in this
        // order. Swapping the two eyes' entries satisfies every symmetry
        // assertion further down and would put the yaw sign inside out.
        assert_eq!(INDICES, [33, 263, 1, 61, 291, 199]);
        assert_eq!(INDICES.len(), MODEL.len());
        assert!(
            INDICES.iter().all(|index| *index < LANDMARKS),
            "every index must address a landmark the decode produces"
        );

        let at = |index: usize| MODEL[INDICES.iter().position(|slot| *slot == index).unwrap()];
        let (right_eye, left_eye) = (at(33), at(263));
        let (right_mouth, left_mouth) = (at(61), at(291));
        let (nose, chin) = (at(1), at(199));

        // The subject's right eye corner is on the image's left, which is
        // negative x, and its mirror is positive — the fact that fixes the yaw
        // sign.
        assert!(
            right_eye[0] < 0.0 && left_eye[0] > 0.0,
            "eyes are the right way round"
        );
        assert!(
            right_mouth[0] < 0.0 && left_mouth[0] > 0.0,
            "so are the mouth corners"
        );
        assert_eq!(
            (nose[0], chin[0]),
            (0.0, 0.0),
            "the nose and chin are on the midline"
        );
        // Symmetric about it, to the millimetre.
        assert_eq!(right_eye[0], -left_eye[0]);
        assert_eq!(right_eye[1], left_eye[1]);
        assert_eq!(right_mouth[0], -left_mouth[0]);
        assert_eq!(right_mouth[1], left_mouth[1]);

        // `y` is down: the eye line is above the nose tip, the mouth and chin
        // below it, and the chin lowest of all.
        assert!(
            right_eye[1] < nose[1] && left_eye[1] < nose[1],
            "eyes above the nose"
        );
        assert!(
            right_mouth[1] > nose[1] && chin[1] > right_mouth[1],
            "mouth and chin below"
        );
        // `z` is away from the camera: the nose tip is the closest point on the
        // face, and every other point sits behind it.
        assert_eq!(nose[2], 0.0, "the model's origin is the nose tip");
        assert!(
            MODEL.iter().all(|point| point[2] >= 0.0),
            "the nose tip protrudes: nothing is in front of it"
        );
        // A face is wider than it is deep, which is the property that makes the
        // six points a usable basis at all.
        let width = left_eye[0] - right_eye[0];
        let height = chin[1] - right_eye[1];
        assert!(
            width > 60.0 && height > 90.0 && width < height,
            "an adult face: {width} mm across the eyes, {height} mm from the eye line to the chin"
        );

        // And the model's own mean radius, which is a constant because the
        // fallback distance divides by it: recomputed here from the model it
        // describes, so editing the model without editing the constant fails
        // this rather than mis-scaling every fallback distance.
        let mut centre = [0.0f32; COORDINATES];
        for point in MODEL {
            for (axis, value) in centre.iter_mut().enumerate() {
                *value += point[axis];
            }
        }
        for value in centre.iter_mut() {
            *value /= CORRESPONDENCES as f32;
        }
        let mut radius = 0.0f32;
        for point in MODEL {
            radius += ((point[0] - centre[0]).powi(2)
                + (point[1] - centre[1]).powi(2)
                + (point[2] - centre[2]).powi(2))
            .sqrt();
        }
        radius /= CORRESPONDENCES as f32;
        assert!(
            (radius - MODEL_RADIUS_MM).abs() < 0.01,
            "MODEL_RADIUS_MM must be the model's own mean radius: it is {MODEL_RADIUS_MM}, the model says {radius}"
        );
    }

    #[test]
    fn the_principal_point_is_used_as_the_principal_point() {
        // The same face, in the same place, seen by two cameras that differ
        // only in where their optical axis meets the sensor. `u = fx·X/Z + cx`
        // makes the second camera's image exactly the first's, shifted by the
        // difference in `cx`/`cy` — so the pose is the same pose, and a solve
        // that ignores the principal point (or reads it as a translation of the
        // head) returns a different one for the same face.
        let wanted = euler(18.0, -12.0, 7.0);
        let centred = CameraIntrinsics::from_calibration(640.0, 640.0, 320.0, 240.0);
        let off_centre = CameraIntrinsics::from_calibration(640.0, 640.0, 420.0, 320.0);
        let face = project_at(wanted, [40.0, -25.0, 600.0], &centred);
        let mut moved = [[0i32; COORDINATES]; LANDMARKS];
        for (index, point) in face.iter().enumerate() {
            if INDICES.contains(&index) {
                moved[index] = [point[0] + 100, point[1] + 80, point[2]];
            }
        }

        let first = estimate_head_pose(&face, &centred);
        let second = estimate_head_pose(&moved, &off_centre);
        assert!(first.valid && second.valid, "{first:?} {second:?}");
        for (what, got, want) in [
            ("yaw", first.euler.yaw, wanted.yaw),
            ("pitch", first.euler.pitch, wanted.pitch),
            ("roll", first.euler.roll, wanted.roll),
            ("shifted yaw", second.euler.yaw, wanted.yaw),
            ("shifted pitch", second.euler.pitch, wanted.pitch),
            ("shifted roll", second.euler.roll, wanted.roll),
        ] {
            assert!((got - want).abs() < 2.0, "{what}: wanted {want}, got {got}");
        }
    }
}
