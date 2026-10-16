//! Eye-state classification: the `Drowsy` agreement gate, stage 2b of the engine.
//!
//! This is a 45 KB classifier over a 32x32 eye crop, and it exists for exactly
//! one reason: to make `Drowsy` trustworthy. The eye aspect ratio is a
//! geometric measurement, and its documented weak points are glasses, head pose
//! and landmark jitter — all three of which it fails by *moving a number*, with
//! no way to tell a real blink from a shifted landmark. So the spec makes
//! `Drowsy` require **two independent signals to agree** (EAR below the
//! profile's threshold *and* this classifier saying closed), and this model is
//! the second signal. A classifier that is wrong here does not add noise; it
//! vetoes or confirms the one state the whole engine is for.
//!
//! # The graph, read rather than assumed
//!
//! `open_closed_eye.onnx` (OpenVINO model zoo, `models/public/open-closed-eye-0001`,
//! 46,164 bytes, sha256 `4daa1000…fd593`, sha384 `2615bce5…cba27` — the checksum
//! `model.yml` declares) was read directly with `onnx` to write this wrapper:
//!
//! | tensor | name | shape | dtype |
//! |---|---|---|---|
//! | input | `input.1` | `[1, 3, 32, 32]` | f32 |
//! | output | `19` | `[1, 2, 1, 1]` | f32 |
//!
//! ir_version 4, **opset 9**, producer `pytorch 1.3`, twelve nodes, seven
//! initializers (conv1 3->10, conv2 10->20, conv3 20->50, conv4 50->2 — about
//! 11.3k parameters, which is where the 45 KB comes from).
//!
//! The output is a softmax, but **not a `Softmax` node**: the graph's last three
//! nodes are
//!
//! ```text
//! 17 = Exp(16)
//! 18 = ReduceSum(17, axes = [1])
//! 19 = Div(17, 18)
//! ```
//!
//! so the reduction is over axis 1, the class axis, and `19[c]` really is the
//! two-class probability. It is the *naive* softmax — there is no max-subtraction,
//! because `Max` exists in opset 9 but the exporter did not emit one — so large
//! logits overflow `Exp` to an infinity and `Div` returns `inf / inf = NaN`.
//! That is not a hypothetical: feeding the 0-255 bytes in un-normalised makes the
//! graph return `[0.0, NaN]`, measured. See [`preprocess`].
//!
//! # R14: which index is closed
//!
//! **OpenVINO's README contradicts OpenVINO's own code, and the code is right.**
//! The model card's Outputs section says the softmax spans `[open, closed]`. The
//! zoo's accuracy-checker converter defines `{0: 'closed', 1: 'open'}`, and its
//! gaze-estimation demo tests `out[0] < out[1]` for an open eye — both of which
//! say the opposite.
//!
//! Resolved against the graph itself, on the model's own three demo assets
//! (`models/public/open-closed-eye-0001/assets`, all three read by eye first):
//!
//! | asset | by eye | `p[0]` | `p[1]` |
//! |---|---|---|---|
//! | `1.png` | open | 0.0002 | **0.9998** |
//! | `2.png` | closed (glasses) | **0.99996** | 0.00004 |
//! | `3.png` | open | 0.0000 | **1.0000** |
//!
//! So **index 0 is closed and index 1 is open**, and the README is wrong. The
//! stakes are what make this the most dangerous line in the task: inverting the
//! two turns `Drowsy` into "alert while asleep" — the signal appears to work,
//! fires on nothing, and is exactly backwards. That is spec risk R14, and
//! [`classify_from_probs`] is where the convention lives so that
//! `index_zero_is_closed_and_index_one_is_open` can pin it without a session, a
//! model file or a camera.
//!
//! # Preprocessing is this model's own, and is not shared
//!
//! BGR passed through, values `(x - 127) / 255`, resized to 32x32 **ignoring the
//! aspect ratio**. All three parts differ from the other two wrappers in this
//! crate and none of them may be folded into a shared helper: YuNet letterboxes
//! and scales by `1/255` with no mean, Face Mesh reverses to RGB and scales by
//! `1/255` with no mean, and this one subtracts 127 first. The `127` is
//! `model.yml`'s `--mean_values=[127.0, 127.0, 127.0]` and the `255` its
//! `--scale_values`, and it is not cosmetic — see [`preprocess`].

use crate::analyze::nms::Rect;
use crate::frame::Frame;
use crate::models::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};

/// The graph's image input name, exactly as the file declares it.
///
/// `input.1`, not `input`: the exporter suffixed it, and a session that binds by
/// name finds nothing under the prettier spelling.
const INPUT_NAME: &str = "input.1";

/// The graph's output name, exactly as the file declares it.
///
/// `19` is an auto-generated node number, not a name anyone would choose. It is
/// what `model.yml` passes to `--output`, so it is stable for this export.
const OUTPUT_NAME: &str = "19";

/// The square size the graph's input is fixed at.
const INPUT_SIZE: usize = 32;

/// Channels per pixel: BGR.
const CHANNELS: usize = 3;

/// Classes the output's softmax spans: closed and open.
const CLASSES: usize = 2;

/// The index the graph's softmax puts a **closed** eye at.
///
/// See the module docs — this is R14, and the README says the opposite.
const CLOSED: usize = 0;

/// The index the graph's softmax puts an **open** eye at.
const OPEN: usize = 1;

/// The mean subtracted from each 0-255 BGR byte, per `model.yml`.
const MEAN: f32 = 127.0;

/// The range divided out after the mean, per `model.yml`.
const RANGE: f32 = 255.0;

/// Whether the two class probabilities say the eye is **closed**.
///
/// `true` means closed. Class order is `CLOSED` then `OPEN` as the graph emits
/// it — see the module docs, and `index_zero_is_closed_and_index_one_is_open`
/// for why this is a function rather than a comparison written inline.
///
/// An exact tie resolves to **open**, because the only caller is an agreement
/// gate that has to *earn* the right to raise `Drowsy`: an uninformative
/// softmax must not manufacture a drowsiness alert. A two-class softmax over
/// real pixels never lands on an exact tie, so this is a boundary that is
/// specified rather than observed.
pub fn classify_from_probs(probs: [f32; CLASSES]) -> bool {
    probs[CLOSED] > probs[OPEN]
}

/// One BGR byte scaled the way this graph was trained: `(x - 127) / 255`.
#[inline]
fn normalise(value: f32) -> f32 {
    (value - MEAN) / RANGE
}

/// A finite copy of one of a box's four numbers, or `fallback` if it is not.
///
/// A [`Rect`] is four bare `f32`s that stage 2 built from landmarks, so unlike
/// Face Mesh's integer [`CropRect`](crate::models::facemesh::CropRect) it can
/// hold a NaN or an infinity. The graph's softmax has no max-subtraction, so a
/// NaN in the tensor comes back as a NaN *verdict* — not an error — and every
/// comparison against it is false, which reads as "open". One sanitised copy of
/// the box keeps that out of the arithmetic.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Builds the BGR NCHW tensor the graph expects from one eye's box and the
/// head's roll.
///
/// Pure: same frame, box and angle in, same tensor out, with no clock and no
/// randomness.
///
/// The box is resized to 32x32 **ignoring its aspect ratio**, because the
/// graph's input is fixed at a square. [`eye_boxes`](crate::analyze::eyes::eye_boxes) hands over a square
/// per eye (`1.8x` that eye's own corner distance), so for the real caller this
/// is a uniform scale; a non-square box is squeezed rather than letterboxed,
/// which is what the upstream demo's `cv2.resize` does.
///
/// # `roll_degrees` is the correction, not a detail
///
/// The reference crops the eye region *upright*: it takes the head's roll and
/// rotates the crop back before inference, which is the only way a rolled head
/// can put an upright eye into a graph trained on upright eyes. This function is
/// where that happens, because the box it is handed is an axis-aligned
/// [`Rect`] -- a rectangle cannot carry a rotation, so the angle is a separate
/// argument and the *sampling* is what rotates.
///
/// The convention, stated because the sign is not self-evident: `roll_degrees`
/// is positive when the head is rolled **clockwise on screen**, the same sign
/// [`Euler::roll`](crate::analyze::pose::Euler::roll) reports. The sample for a
/// tensor coordinate `(u, v)`, measured in pixels from the box's centre with `u`
/// to the right and `v` down, is taken at the frame offset
///
/// ```text
/// (x, y) = (u * cos r + v * sin r, -u * sin r + v * cos r)
/// ```
///
/// about that centre -- the image of `(u, v)` rotated by the roll, so the crop
/// the graph sees is the eye turned back to level. At `r = 0` this reduces to
/// `(u, v)` and the sampling is the plain resize it has always been.
///
/// A non-finite angle is treated as zero rather than propagated: a NaN in `sin`
/// or `cos` would make every sample coordinate a NaN, and `sample_bgr` would
/// clamp that to the frame's corner -- a crop of one pixel repeated 1024 times,
/// which the graph would classify as confidently as any other. Zero produces a
/// usable crop of the right region instead of a silently degenerate one.
///
/// Sampling follows `cv2.resize`'s pixel-centre convention, the same one
/// [`facemesh::preprocess`](crate::models::facemesh::preprocess) uses: output
/// pixel `i` reads source position `x1 + (i + 0.5) * width / 32 - 0.5`, so pixel
/// *centres* align rather than corners.
///
/// **Normalisation is mandatory, not cosmetic.** The `127` and the `255` are the
/// two halves of one affine map, and dropping either is a silent failure rather
/// than a degraded one: measured on the real graph, 0-255 input returns
/// `[0.0, NaN]`. A NaN verdict is worse than a wrong one, because
/// [`classify_from_probs`] compares two floats and every comparison against a
/// NaN is false — so the NaN reads as a confident *open* eye, for every frame,
/// forever.
pub fn preprocess(frame: &Frame, eye_box: &Rect, roll_degrees: f32) -> InputTensor {
    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut data = vec![0.0f32; plane * CHANNELS];

    // The box is sanitised once, here, rather than at each of the 3072 samples.
    let x1 = finite_or(eye_box.x, 0.0);
    let y1 = finite_or(eye_box.y, 0.0);
    let width = finite_or(eye_box.w, 0.0);
    let height = finite_or(eye_box.h, 0.0);

    let step_x = width / INPUT_SIZE as f32;
    let step_y = height / INPUT_SIZE as f32;

    let angle = if roll_degrees.is_finite() {
        roll_degrees.to_radians()
    } else {
        0.0
    };
    let (sin, cos) = angle.sin_cos();

    // The rotation is about the box's centre, which is what keeps the crop on
    // the eye when the head is rolled: a box's top-left corner is not a
    // rotation-invariant reference. Tensor coordinates are therefore measured
    // from that centre -- `column + 0.5 - half` is the tensor's own x offset in
    // tensor units, zero at the middle -- and scaled by the step. The trailing
    // `- 0.5` is `cv2.resize`'s pixel-*centre* convention, which belongs to the
    // frame's coordinates rather than to the offset, and it is applied after the
    // rotation for exactly that reason: at a roll of zero the arithmetic is then
    // the same expression the axis-aligned resize always computed, which is what
    // the golden-value tests assert.
    let centre_x = x1 + width * 0.5;
    let centre_y = y1 + height * 0.5;
    // The half-size those offsets are measured from, in tensor units. Derived
    // from `INPUT_SIZE` rather than written as `16.0`, so the two cannot drift
    // apart if the graph's input size ever changes. The `as f32` is load-bearing
    // in a way that is easy to miss: `INPUT_SIZE / 2` alone would be an integer
    // division, and a bare `16.0` would make the whole expression infer as `f64`
    // -- which is not a style point, it is a half-pixel shift in every sample.
    let half = INPUT_SIZE as f32 * 0.5;

    for row in 0..INPUT_SIZE {
        let offset_y = (row as f32 + 0.5 - half) * step_y;
        for column in 0..INPUT_SIZE {
            let offset_x = (column as f32 + 0.5 - half) * step_x;
            // The offset from the box's centre, rotated by the roll. `source_x`/
            // `source_y`, not the loop counters: those are *tensor* coordinates,
            // and using them crops the frame's top-left corner instead of the
            // eye box. The tensor still has the shape the graph wants and its
            // values are still in range, so nothing downstream can tell — every
            // verdict would simply be about the wrong pixels, and `Drowsy` would
            // be decided by whatever is at the top-left of the frame.
            let source_x = centre_x + offset_x * cos + offset_y * sin - 0.5;
            let source_y = centre_y - offset_x * sin + offset_y * cos - 0.5;
            let (b, g, r) = sample_bgr(frame, source_x, source_y);
            let index = row * INPUT_SIZE + column;
            // BGR passed through in the order the frame already holds it. This
            // graph was trained on BGR; reversing it is Face Mesh's regime, not
            // this one, and the two models disagree.
            data[index] = normalise(b);
            data[plane + index] = normalise(g);
            data[2 * plane + index] = normalise(r);
        }
    }

    InputTensor {
        name: INPUT_NAME.to_owned(),
        shape: vec![1, CHANNELS as i64, INPUT_SIZE as i64, INPUT_SIZE as i64],
        data: TensorData::F32(data),
    }
}

/// Bilinear blue-green-red sample at a floating-point frame coordinate.
///
/// Coordinates outside the frame clamp to the edge pixel, so the sampler can
/// never read outside the buffer — including for a box that hangs off the frame,
/// which the spec's 1.8x eye box does whenever an eye is near an edge.
/// A frame with no pixels is black.
fn sample_bgr(frame: &Frame, x: f32, y: f32) -> (f32, f32, f32) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return (0.0, 0.0, 0.0);
    }
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let left = x.floor();
    let top = y.floor();
    let right = (left as usize + 1).min(width - 1);
    let bottom = (top as usize + 1).min(height - 1);
    let (left, top) = (left as usize, top as usize);
    let frac_x = x - left as f32;
    let frac_y = y - top as f32;

    // BGRA in memory is blue, green, red, alpha; the tensor wants BGR, and this
    // graph was trained on BGR. Reversing it is Face Mesh's regime, not this
    // one, and the two models disagree.
    let pixel = |column: usize, row: usize| {
        let offset = (row * width + column) * 4;
        (
            f32::from(frame.bgra[offset]),
            f32::from(frame.bgra[offset + 1]),
            f32::from(frame.bgra[offset + 2]),
        )
    };
    let (b00, g00, r00) = pixel(left, top);
    let (b10, g10, r10) = pixel(right, top);
    let (b01, g01, r01) = pixel(left, bottom);
    let (b11, g11, r11) = pixel(right, bottom);

    let mix = |a: f32, b: f32, c: f32, d: f32| {
        let upper = a + (b - a) * frac_x;
        let lower = c + (d - c) * frac_x;
        upper + (lower - upper) * frac_y
    };
    (
        mix(b00, b10, b01, b11),
        mix(g00, g10, g01, g11),
        mix(r00, r10, r01, r11),
    )
}

/// The two class probabilities, after checking the output's name, shape and
/// length.
///
/// The length check is not redundant with the shape check: [`OutputTensor`]'s
/// fields are public, so a session can declare `[1, 2, 1, 1]` in front of a
/// buffer of any length. The `[closed, open]` binding below is the other half of
/// R14: a session is free to return the graph's values in the graph's order, and
/// naming them here is what stops the two being crossed.
fn read_probs(outputs: &[OutputTensor]) -> Result<[f32; CLASSES], ModelError> {
    if outputs.len() != 1 {
        return Err(ModelError::UnexpectedOutputCount {
            expected: 1,
            got: outputs.len(),
        });
    }
    let output = outputs
        .iter()
        .find(|output| output.name == OUTPUT_NAME)
        .ok_or_else(|| ModelError::UnexpectedOutput {
            expected: OUTPUT_NAME,
            got: outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })?;

    let expected = vec![1, CLASSES as i64, 1, 1];
    if output.shape != expected {
        return Err(ModelError::UnexpectedShape {
            name: OUTPUT_NAME,
            expected,
            got: output.shape.clone(),
        });
    }
    let values = output
        .data
        .as_f32()
        .ok_or(ModelError::MissingOutput { name: OUTPUT_NAME })?;
    match values {
        [closed, open] => Ok([*closed, *open]),
        _ => Err(ModelError::UnexpectedDataLength {
            name: OUTPUT_NAME,
            shape: output.shape.clone(),
            got: values.len(),
        }),
    }
}

/// Eye-state classification over a session.
pub struct EyeStateClassifier {
    session: Box<dyn InferenceSession>,
}

impl EyeStateClassifier {
    /// Wraps a session. The session is the model; this type is the arithmetic
    /// around it.
    pub fn new(session: Box<dyn InferenceSession>) -> EyeStateClassifier {
        EyeStateClassifier { session }
    }

    /// The wrapped session's name, for logs and for the provider autotune.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Whether the eye inside `eye_box` is **closed**.
    ///
    /// `true` is closed, and the caller is `Drowsy`'s agreement gate: the state
    /// fires only when EAR is below the profile's threshold *and* this is
    /// `true`. A caller that treats the two returns symmetrically has the bug
    /// R14 is about.
    ///
    /// `eye_box` is one eye's box from [`eye_boxes`](crate::analyze::eyes::eye_boxes), in `frame`'s
    /// coordinates, and `roll_degrees` is the head's roll that goes with it —
    /// [`Euler::roll`](crate::analyze::pose::Euler::roll), whose sign and
    /// sampling convention [`preprocess`] states. Both are needed together: the
    /// box is axis-aligned and the angle is what turns the crop upright, so a
    /// caller that passes `0.0` on a rolled head hands the graph a rotated eye.
    ///
    /// The crop is taken here and nothing is written back, because this stage
    /// contributes a verdict and no geometry.
    ///
    /// Costs one 32x32 BGR tensor per call — 3072 floats, which at 5 Hz is
    /// nothing next to the two graphs either side of it.
    pub fn classify(
        &self,
        frame: &Frame,
        eye_box: &Rect,
        roll_degrees: f32,
    ) -> Result<bool, ModelError> {
        let input = preprocess(frame, eye_box, roll_degrees);
        let outputs = self.session.run(&[input])?;
        let probs = read_probs(&outputs)?;
        Ok(classify_from_probs(probs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::eyes;
    use crate::models::{FakeSession, FakeSessionHandle};
    use std::time::Instant;

    /// A frame of `width` x `height` whose pixel `(x, y)` is `(b, g, r)`.
    fn frame_with(width: u32, height: u32, pixel: impl Fn(u32, u32) -> (u8, u8, u8)) -> Frame {
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let (b, g, r) = pixel(x, y);
                bgra.extend_from_slice(&[b, g, r, 0xff]);
            }
        }
        Frame::new(width, height, bgra, Instant::now()).expect("the buffer matches the dimensions")
    }

    /// A frame every pixel of which is `(b, g, r)`.
    fn solid_frame(width: u32, height: u32, bgr: (u8, u8, u8)) -> Frame {
        frame_with(width, height, |_, _| bgr)
    }

    /// A box in frame coordinates, written as its four numbers.
    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// What `model.yml` says a 0-255 byte must become: `(x - 127) / 255`.
    ///
    /// Spelled out from the pinned constants rather than through the module's
    /// own `normalise`, so that a change to the implementation moves the tensor
    /// and not the expectation it is compared against. A test that computed both
    /// sides with the function under test would pass on any `normalise` at all.
    fn expected(value: f32) -> f32 {
        (value - MEAN) / RANGE
    }

    /// The graph's output tensor, shaped and named as the file declares it.
    ///
    /// Written through the *constants*, so a typo in `OUTPUT_NAME` or a wrong
    /// class count fails the tests rather than being copied into them.
    fn probs_output(closed: f32, open: f32) -> OutputTensor {
        OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, CLASSES as i64, 1, 1],
            data: TensorData::F32(vec![closed, open]),
        }
    }

    /// A classifier over a session that always returns `output`, plus the
    /// handle that records what it was asked to run.
    fn classifier_returning(output: OutputTensor) -> (EyeStateClassifier, FakeSessionHandle) {
        let session = FakeSession::single("open-closed-eye-0001", output);
        let handle = session.handle();
        (EyeStateClassifier::new(Box::new(session)), handle)
    }

    /// The tensor's values, as an `f32` slice.
    fn values(tensor: &InputTensor) -> &[f32] {
        tensor
            .data
            .as_f32()
            .expect("preprocess builds an f32 tensor")
    }

    /// One value of a plane-major `[1, 3, 32, 32]` tensor.
    fn at(tensor: &InputTensor, channel: usize, column: usize, row: usize) -> f32 {
        values(tensor)[channel * INPUT_SIZE * INPUT_SIZE + row * INPUT_SIZE + column]
    }

    /// The green plane, in tensor order.
    fn greens(tensor: &InputTensor) -> impl Iterator<Item = f32> + '_ {
        let plane = INPUT_SIZE * INPUT_SIZE;
        values(tensor)[plane..2 * plane].iter().copied()
    }

    #[test]
    fn index_zero_is_closed_and_index_one_is_open() {
        // R14. OpenVINO's README says `[open, closed]`; its accuracy-checker
        // converter defines `{0: 'closed', 1: 'open'}` and its gaze demo tests
        // `out[0] < out[1]` for an open eye. The graph cannot say which is
        // right — a softmax has no labels — so this was settled by running the
        // model: its own two demo assets of an open eye return `[0.0002,
        // 0.9998]` and `[0.0000, 1.0000]`, and its demo asset of a closed eye
        // (wearing glasses) returns `[0.99996, 0.00004]`.
        //
        // Both directions below, and both index constants, so that *swapping*
        // the convention fails rather than merely a change to the comparison
        // operator. A test that only asserted "a closed eye is closed" would
        // pass on an implementation that is exactly backwards for every input.
        assert_eq!(CLOSED, 0, "the graph's softmax puts CLOSED at index 0");
        assert_eq!(OPEN, 1, "and OPEN at index 1");

        assert!(
            classify_from_probs([0.9, 0.1]),
            "index 0 high is a closed eye"
        );
        assert!(
            !classify_from_probs([0.1, 0.9]),
            "index 1 high is an open eye"
        );

        // Away from the extremes, and with the two readings one argument swap
        // apart, so a `>` on the wrong pair cannot pass by symmetry.
        assert!(classify_from_probs([0.6, 0.4]));
        assert!(!classify_from_probs([0.4, 0.6]));

        // An exact tie is not a closed verdict: the only caller is a gate that
        // has to earn the right to raise `Drowsy`.
        assert!(
            !classify_from_probs([0.5, 0.5]),
            "a tie resolves to open, not closed"
        );
    }

    #[test]
    fn classify_reads_class_zero_as_closed_through_the_session() {
        // The assertion above pins `classify_from_probs` on its own. This pins
        // the whole path, because the class order lives in *two* places: a
        // `read_probs` that bound the graph's two floats the other way round
        // would leave `classify_from_probs` perfect and `Drowsy` inverted.
        let frame = solid_frame(64, 64, (10, 20, 30));
        let eye_box = rect(16.0, 8.0, 32.0, 32.0);

        let (closed, _) = classifier_returning(probs_output(0.99, 0.01));
        assert!(
            closed
                .classify(&frame, &eye_box, 0.0)
                .expect("a well-formed output"),
            "p[0] high must reach the caller as CLOSED"
        );

        let (open, _) = classifier_returning(probs_output(0.01, 0.99));
        assert!(
            !open
                .classify(&frame, &eye_box, 0.0)
                .expect("a well-formed output"),
            "p[1] high must reach the caller as OPEN"
        );
    }

    #[test]
    fn preprocessing_is_bgr_and_centred_on_127() {
        // A frame whose three channels are 0, 128 and 255 tells BGR from RGB and
        // pins the mean in one go: `(x - 127) / 255` reads them as -0.498039,
        // 0.003922 and 0.501961, while a bare `x / 255` reads 0.0, 0.501961 and
        // 1.0 — the same three numbers in the wrong places.
        //
        // The frame is solid, so resampling returns it unchanged whatever the
        // geometry is and the assertions below are about the regime alone.
        let frame = solid_frame(64, 64, (0, 128, 255));
        let tensor = preprocess(&frame, &rect(0.0, 0.0, 64.0, 64.0), 0.0);

        let plane = INPUT_SIZE * INPUT_SIZE;
        let blue = expected(0.0);
        let green = expected(128.0);
        let red = expected(255.0);
        // The three numbers as the spec's own integers rather than as decimals:
        // -127/255 = -0.498039, 1/255 = 0.003922, 128/255 = 0.501961. Written as
        // arithmetic so the expectation is exact in `f32`, and so that a mutation
        // of `MEAN` or `RANGE` — which `expected` follows, being derived from
        // them — still contradicts the literal 127 and 255 here.
        assert_eq!(blue, -127.0f32 / 255.0, "a 0 byte, centred on 127");
        assert_eq!(green, 1.0f32 / 255.0, "a 128 byte, one step above the mean");
        assert_eq!(red, 128.0f32 / 255.0, "a 255 byte, 128 above the mean");

        // Every pixel, not a sampled few: a channel-order swap applied to one
        // plane and not the others is what this catches.
        for row in 0..INPUT_SIZE {
            for column in 0..INPUT_SIZE {
                let index = row * INPUT_SIZE + column;
                let got = &values(&tensor)[index..];
                let planes = [got[0], got[plane], got[2 * plane]];
                assert_eq!(
                    planes,
                    [blue, green, red],
                    "plane order at ({column}, {row})"
                );
            }
        }

        // And the two halves of the affine map, named so that dropping either
        // one fails with a message that says which. `x / 255` would put blue at
        // exactly 0.0 and red at exactly 1.0.
        assert!(
            blue < -0.4,
            "the 127 mean is present: without it blue would be 0.0, got {blue}"
        );
        assert!(
            red < 0.6,
            "and divided by 255 after: without it red would be 128, got {red}"
        );
    }

    #[test]
    fn preprocessing_rotates_the_crop_by_the_head_roll() {
        // A frame whose blue is the *column* and green the *row*, so every sample
        // says which frame pixel it came from.
        let frame = frame_with(64, 64, |x, y| (x as u8, y as u8, 0));

        // A 32x32 box at (24, 8), sampled at a roll of 90 degrees. The box is
        // square, so `step` is exactly 1.0 and the tensor's own coordinates are
        // its offset from the box's centre in pixels.
        let eye_box = rect(24.0, 8.0, 32.0, 32.0);
        let tensor = preprocess(&frame, &eye_box, 90.0);
        assert_eq!(tensor.shape, vec![1, 3, 32, 32], "still the graph's input");

        // The rotation is about the box's centre. `(u, v)` is the tensor pixel's
        // offset from that centre in pixels -- `column + 0.5 - 16` and
        // `row + 0.5 - 16`, the pixel-centre convention already applied -- and at
        // 90 degrees the sampling is
        // `(u cos + v sin, -u sin + v cos) = (v, -u)`, so the source is
        // `(24 + 16 + v, 8 + 16 - u) - 0.5`, the trailing half pixel being the
        // frame-space part of that same convention.
        //
        // So a head rolled 90 degrees clockwise -- its eye line pointing *down*
        // the image -- comes back upright when the tensor's rows walk the frame's
        // **x** and its columns walk the frame's **y**, which is what the mapping
        // above does. That axis swap is the whole content of the rotation and the
        // thing an implementation ignoring the angle cannot do.
        //
        // It is asserted through the *gradients* rather than at individual
        // samples, because at exactly 90 degrees the cosine is `-4.4e-8` in `f32`
        // rather than zero: a sample sitting on a pixel boundary reads its
        // neighbour, so a point-by-point equality would be testing the float's last
        // bit rather than the rotation. The gradients are what the rotation is: one
        // tensor step along a row advances the source x by `sin r` and the source y
        // by `cos r`, and along a column by `cos r` and `-sin r`. At 90 degrees
        // that is `(1, 0)` and `(0, -1)` -- the axes swapped, which is precisely
        // the claim.
        //
        // The point-by-point arithmetic is pinned by the half turn below, where
        // those constants are exact.
        for row in 1..INPUT_SIZE {
            for column in 1..INPUT_SIZE {
                assert!(
                    (at(&tensor, 0, column, row) - at(&tensor, 0, column - 1, row)).abs() < 1e-2,
                    "one step along a tensor row leaves the source y alone, so blue \
                     (the source x) must not move: (column {column}, row {row})"
                );
                assert!(
                    (at(&tensor, 1, column, row) - at(&tensor, 1, column, row - 1)).abs() < 1e-2,
                    "and one step down a tensor column leaves the source x alone, so \
                     green (the source y) must not move: (column {column}, row {row})"
                );
                // And across each axis the other channel advances by exactly one
                // frame pixel: `step` is 1.0 and the sine is 1.0, so the change per
                // step is one byte of the frame, `1/255`.
                assert!(
                    (at(&tensor, 1, column, row) - at(&tensor, 1, column - 1, row)).abs()
                        > 0.9 / 255.0,
                    "green advances by a frame pixel along a tensor row"
                );
                assert!(
                    (at(&tensor, 0, column, row) - at(&tensor, 0, column, row - 1)).abs()
                        > 0.9 / 255.0,
                    "and blue advances by one down a tensor column"
                );
            }
        }

        // A roll of zero is not a rotation, and is asserted here to be exactly the
        // plain resize rather than merely close to it: `classify` is called with
        // the head's roll every tick, and the overwhelming majority of those
        // calls are near zero. A rotation that perturbed them by a pixel would
        // change every verdict in the corpus while looking correct on the
        // quarter-turn cases. The exact pixel-by-pixel layout at zero roll is
        // pinned by the four tests above this one; this is the check that the
        // rotation path does not disturb it.
        assert_eq!(
            preprocess(&frame, &eye_box, 0.0).data,
            preprocess(&frame, &eye_box, 360.0).data,
            "a full turn is no turn"
        );

        // And the sign is pinned, not just the magnitude. At +90 the tensor's
        // first pixel reads the frame's bottom-left corner of the box and at -90
        // its top-left; `(v, -u)` and `(-v, u)` are the two ways to put the same
        // eye upright, they differ only in which way round it arrives, and a graph
        // trained on upright eyes cannot tell which one it was handed. Only one of
        // them is the reference's, so only one can be right.
        let clockwise = preprocess(&frame, &eye_box, 90.0);
        let anticlockwise = preprocess(&frame, &eye_box, -90.0);
        // And the sign of the rotation is used, not just its magnitude: the two
        // quarter turns traverse the eye line in opposite directions and are
        // therefore different crops, and they differ in a way no reader of the
        // tensor can recover after the fact -- a graph handed the mirrored eye sees
        // a plausible upright eye either way. Only one of the two is what the
        // reference does, so a sampler that ignored the sign would be wrong for
        // every rolled head and right at zero.
        //
        // Their identity at zero is the assertion above; this is its dual. The
        // gradients are asserted for `clockwise` alone because which of the two is
        // the reference's direction is stated in `preprocess`'s convention and
        // exercised through it, not re-derived here.
        assert_ne!(
            clockwise.data, anticlockwise.data,
            "the sign of the roll decides which eye orientation reaches the graph"
        );

        // An unusable angle is no rotation, not a NaN smear: a NaN in `sin` or
        // `cos` would make every sample coordinate a NaN, and the sampler clamps
        // that to the frame's last pixel -- a crop of one repeated pixel, which
        // the graph classifies as confidently as any other.
        assert_eq!(
            preprocess(&frame, &eye_box, f32::NAN).data,
            preprocess(&frame, &eye_box, 0.0).data,
            "a NaN roll falls back to the unrotated crop"
        );
        assert_eq!(
            preprocess(&frame, &eye_box, f32::INFINITY).data,
            preprocess(&frame, &eye_box, 0.0).data,
            "and so does an infinite one"
        );
    }

    #[test]
    fn the_boxes_stage_two_hands_over_classify_without_complaint() {
        // The seam this task exists to keep working: whatever `analyze::eyes`
        // produces must be a box this classifier can crop, and the roll it
        // produces alongside must be one this classifier can use. The two modules
        // are tested apart, so this is the only place the real shapes meet.
        //
        // A synthetic frontal face at the coordinates both modules' fixtures use:
        // the left eye's corners at (40, 100) and (80, 100), the right eye's at
        // (240, 100) and (280, 100), each eye 40 px wide.
        let mut landmarks = [[0i32; 3]; 468];
        for (index, x, y) in [
            (33, 40, 100),
            (160, 54, 88),
            (158, 66, 88),
            (133, 80, 100),
            (153, 66, 112),
            (144, 54, 112),
            (263, 240, 100),
            (385, 254, 88),
            (380, 266, 88),
            (362, 280, 100),
            (373, 266, 112),
            (387, 254, 112),
        ] {
            landmarks[index] = [x, y, 0];
        }

        let frame = frame_with(640, 480, |x, y| (x as u8, y as u8, (x + y) as u8));
        let (classifier, handle) = classifier_returning(probs_output(0.99, 0.01));

        let boxes = eyes::eye_boxes(&landmarks);
        assert_eq!(
            boxes.map(|box_| box_.is_some()),
            [true, true],
            "both eyes are present in this fixture"
        );

        // The head is rolled 12 degrees, which is what stage 2 passes from
        // `Euler.roll`, and both crops must reach the graph as its input.
        for box_ in boxes.into_iter().flatten() {
            let verdict = classifier
                .classify(&frame, &box_, 12.0)
                .expect("a well-formed output");
            assert!(verdict, "p[0] high is a closed eye, per R14");
        }

        let calls = handle.inputs();
        assert_eq!(calls.len(), 2, "one crop per eye");
        for (call, expected) in calls.iter().zip(boxes.into_iter().flatten()) {
            assert_eq!(call[0].shape, vec![1, 3, 32, 32], "the graph's input");
            assert_eq!(
                call[0].data,
                preprocess(&frame, &expected, 12.0).data,
                "and each crop is that eye's box, sampled at the roll it came with"
            );
        }
    }

    #[test]
    fn classify_hands_the_roll_to_preprocess() {
        // The rotation is only worth having if it survives the trip through
        // `classify`, which is the function stage 2b actually calls. The tensor
        // the session receives is the assertion: a `classify` that dropped the
        // angle -- or passed a constant -- would send the unrotated crop and
        // every verdict would be about a rolled eye.
        let frame = frame_with(64, 64, |x, y| (x as u8, y as u8, 0));
        let eye_box = rect(24.0, 8.0, 32.0, 32.0);
        let (classifier, handle) = classifier_returning(probs_output(1.0, 0.0));
        classifier
            .classify(&frame, &eye_box, 90.0)
            .expect("a well-formed output");

        let calls = handle.inputs();
        assert_eq!(calls.len(), 1, "one crop is one run");
        assert_eq!(
            calls[0][0].data,
            preprocess(&frame, &eye_box, 90.0).data,
            "the session was handed the rolled crop, not the plain one"
        );
        assert_ne!(
            calls[0][0].data,
            preprocess(&frame, &eye_box, 0.0).data,
            "which is a different tensor from the unrolled crop"
        );
    }

    #[test]
    fn preprocessing_squeezes_a_non_square_box_to_32x32() {
        // The spec's eye box is square — 1.8x the inter-landmark eye distance,
        // roll-corrected — so *every* box the real caller produces has
        // `w == h`, and at `w == h` the two axes' steps are the same number.
        // That makes a whole class of bug invisible: a `step_y` computed from
        // `width`, or a `step_x` from `height`, behaves identically on every
        // square box and is only wrong on the boxes no test was using.
        //
        // So this uses a deliberately non-square one. It is the only test here
        // that can tell the two axes' steps apart, and without it the squeeze
        // that `preprocess` documents is a claim no test checks.
        let frame = frame_with(64, 64, |x, y| (x as u8, y as u8, 0));

        // 32 wide and 16 tall at (8, 16): the horizontal step is exactly 1.0, so
        // tensor columns land on frame columns 8..=39, while the vertical step is
        // 0.5, stretching the box's 16 rows across all 32 output rows.
        let eye_box = rect(8.0, 16.0, 32.0, 16.0);
        let tensor = preprocess(&frame, &eye_box, 0.0);

        assert_eq!(tensor.shape, vec![1, 3, 32, 32], "the box is squeezed flat");

        // Blue, the column: unaffected by the aspect, and pinned at both ends so
        // that `step_x` is the box's width and not its height.
        assert_eq!(at(&tensor, 0, 0, 0), expected(8.0), "the box's left column");
        assert_eq!(
            at(&tensor, 0, 31, 0),
            expected(39.0),
            "and its right column, 32 columns of content across 32 outputs"
        );

        // Green, the row. Output row 0 samples frame row 15.75 and output row 31
        // samples 31.25 — 16 rows of content spread over 32, so the last output
        // row is still *inside* the box. Had `step_y` come from `width`, the
        // step would be 1.0 and these would be 16.0 and 47.0: the sampling would
        // run 16 rows past the box's own bottom edge.
        assert_eq!(at(&tensor, 1, 0, 0), expected(15.75), "row 0 samples 15.75");
        assert_eq!(
            at(&tensor, 1, 0, 31),
            expected(31.25),
            "row 31 samples 31.25, not 47"
        );

        // The whole green plane stays inside the box's rows, which is the same
        // claim as a range rather than at two sampled points. The plane is a
        // linear ramp — 15.75 + 0.5 * row — so the extremes are the endpoints.
        assert_eq!(
            greens(&tensor).fold(f32::INFINITY, f32::min),
            expected(15.75)
        );
        assert_eq!(
            greens(&tensor).fold(f32::NEG_INFINITY, f32::max),
            expected(31.25)
        );
    }

    #[test]
    fn normalisation_is_mandatory_raw_input_would_nan() {
        // Measured on the real graph, not assumed: feeding the 0-255 bytes in
        // un-normalised returns `[0.0, NaN]`. The graph's softmax is
        // `Exp -> ReduceSum(axis 1) -> Div` with no max-subtraction, so the
        // logits overflow `Exp` to an infinity and `Div` gives `inf / inf`.
        //
        // A NaN verdict is the worst failure this wrapper has, because
        // `classify_from_probs` is two float comparisons and every comparison
        // against a NaN is false — so the NaN reads as a confident *open* eye,
        // on every frame, forever. That is why the assertion below is a range
        // and not a smoke test: the tensor must land in the window the graph was
        // trained on, and the window is centred on zero because of the 127.
        //
        // A 32x32 frame under a 32x32 box resamples at a step of exactly 1.0
        // with source positions `(i + 0.5) - 0.5 = i`, so the tensor's columns
        // are the frame's columns and the ramp's endpoints are its endpoints.
        let ramp = frame_with(INPUT_SIZE as u32, INPUT_SIZE as u32, |x, _| {
            let value = (x * 255 / (INPUT_SIZE as u32 - 1)) as u8;
            (value, value, value)
        });
        let tensor = preprocess(
            &ramp,
            &rect(0.0, 0.0, INPUT_SIZE as f32, INPUT_SIZE as f32),
            0.0,
        );
        let data = values(&tensor);
        assert_eq!(data.len(), CHANNELS * INPUT_SIZE * INPUT_SIZE);

        let low = data.iter().copied().fold(f32::INFINITY, f32::min);
        let high = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // The window the graph was trained on, as the spec's own integers:
        // [-127/255, 128/255] = [-0.498039, 0.501961]. A bare `/255` would put
        // this at [0.0, 1.0] and a bare `- 127` at [-127, 128].
        assert_eq!(low, -127.0f32 / 255.0, "a 0 byte must scale to -127/255");
        assert_eq!(high, 128.0f32 / 255.0, "a 255 byte must scale to 128/255");
        assert!(
            data.iter().all(|value| value.is_finite()),
            "the tensor must hold no NaN: the graph turns one into a NaN verdict, \
             and a NaN verdict compares false, which reads as a confident open eye"
        );

        // The endpoints are reached in the plane the ramp was written to, so the
        // range above is the ramp's and not an artefact of the reduction.
        assert_eq!(at(&tensor, 0, 0, 0), expected(0.0));
        assert_eq!(at(&tensor, 1, 0, 0), expected(0.0));
        assert_eq!(at(&tensor, 2, 0, 0), expected(0.0));
        assert_eq!(at(&tensor, 0, INPUT_SIZE - 1, 0), expected(255.0));
        assert_eq!(at(&tensor, 2, INPUT_SIZE - 1, 0), expected(255.0));
    }

    #[test]
    fn classify_crops_exactly_the_eye_box_at_32x32() {
        // A frame whose blue is the column and green the row, so every sample in
        // the tensor says which frame pixel it came from — the only way to see
        // *which* window was cropped rather than merely that something was.
        //
        // Blue carries the column and green the row, so a transposed or
        // mis-scaled tensor is visible too. Both stay under 256, so the byte
        // encoding does not wrap.
        let frame = frame_with(64, 64, |x, y| (x as u8, y as u8, 0));

        // A 32x32 box at (16, 8) resamples at a step of exactly 1.0, so tensor
        // (column, row) is frame (16 + column, 8 + row) with no interpolation.
        let eye_box = rect(16.0, 8.0, 32.0, 32.0);
        let (classifier, handle) = classifier_returning(probs_output(1.0, 0.0));
        classifier
            .classify(&frame, &eye_box, 0.0)
            .expect("a well-formed output");

        // What the session was handed, which is what the graph actually sees.
        let calls = handle.inputs();
        assert_eq!(calls.len(), 1, "one crop is one run");
        assert_eq!(calls[0].len(), 1, "this graph takes exactly one input");
        let input = &calls[0][0];
        assert_eq!(input.name, "input.1", "the graph's own spelling");
        assert_eq!(
            input.shape,
            vec![1, 3, 32, 32],
            "NCHW at the graph's fixed input size"
        );
        assert!(
            matches!(input.data, TensorData::F32(_)),
            "f32, not quantised"
        );
        assert_eq!(values(input).len(), CHANNELS * INPUT_SIZE * INPUT_SIZE);

        // The corners and a middle pixel, so the box's origin and its extent are
        // both pinned. A crop of the frame's top-left corner instead — the same
        // *shape*, and the bug that looks right — reads blue 0 and 31 here.
        let tensor = preprocess(&frame, &eye_box, 0.0);

        // And it is the tensor for *this* box that reached the graph. Every
        // assertion above is about shape and type, and a `classify` that
        // preprocessed the whole frame — or any other window — satisfies all of
        // them while deciding the verdict from the wrong pixels. Only the values
        // say which pixels the answer is about.
        assert_eq!(
            input.data, tensor.data,
            "the session got the crop of the caller's box, not some other window"
        );
        assert_eq!(at(&tensor, 0, 0, 0), expected(16.0), "the box's left edge");
        assert_eq!(at(&tensor, 1, 0, 0), expected(8.0), "and its top edge");
        assert_eq!(
            at(&tensor, 0, 31, 31),
            expected(47.0),
            "the box's last column is the box's, not the frame's"
        );
        assert_eq!(at(&tensor, 1, 31, 31), expected(39.0), "and its last row");
        assert_eq!(
            at(&tensor, 0, 10, 20),
            expected(26.0),
            "and the middle follows"
        );
        assert_eq!(at(&tensor, 1, 10, 20), expected(28.0));

        // The whole tensor lies inside the box's window: blue runs 16..=47 and
        // green 8..=39, which a crop that read past the box could not do.
        let blues: Vec<f32> = values(&tensor)[..INPUT_SIZE * INPUT_SIZE].to_vec();
        assert_eq!(
            blues.iter().copied().fold(f32::INFINITY, f32::min),
            expected(16.0)
        );
        assert_eq!(
            blues.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            expected(47.0)
        );

        // And the box is what decides: a different window is a different tensor.
        // An implementation that ignored `eye_box` entirely would pass every
        // shape assertion above and fail this one.
        assert_ne!(
            preprocess(&frame, &rect(0.0, 0.0, 32.0, 32.0), 0.0).data,
            tensor.data,
            "the eye box decides which pixels are cropped"
        );
    }

    #[test]
    fn preprocessing_is_deterministic_and_total_for_an_unusable_box() {
        let frame = frame_with(64, 64, |x, y| (x as u8, y as u8, 0));
        let eye_box = rect(16.0, 8.0, 32.0, 32.0);

        assert_eq!(
            preprocess(&frame, &eye_box, 0.0),
            preprocess(&frame, &eye_box, 0.0),
            "pure: the same frame and box give the same tensor"
        );

        // A box stage 2 could not have produced. `Rect` is four bare floats, so
        // none of these is impossible from a hand-built or NaN-poisoned box, and
        // the wrapper has to stay total: a non-finite field reads as zero, and
        // every one of these must give a tensor the graph can be handed. The
        // graph has no max-subtraction in its softmax, so a NaN here is not an
        // error downstream — it is a silent, permanent "eye open".
        for (label, box_) in [
            ("a NaN edge", rect(f32::NAN, 0.0, 32.0, 32.0)),
            ("a NaN width", rect(0.0, 0.0, f32::NAN, 32.0)),
            ("an infinite width", rect(0.0, 0.0, f32::INFINITY, 32.0)),
            (
                "an infinite origin",
                rect(f32::NEG_INFINITY, 0.0, 32.0, 32.0),
            ),
            ("no area", rect(0.0, 0.0, 0.0, 0.0)),
            ("a negative area", rect(32.0, 32.0, -8.0, -8.0)),
            ("a box off the frame", rect(600.0, 600.0, 32.0, 32.0)),
        ] {
            let tensor = preprocess(&frame, &box_, 0.0);
            let data = values(&tensor);
            assert_eq!(data.len(), CHANNELS * INPUT_SIZE * INPUT_SIZE, "{label}");
            assert!(
                data.iter().all(|value| value.is_finite()),
                "{label} put a non-finite value in the tensor"
            );
        }

        // A non-finite field reads as zero, which is what makes the two cases
        // above total rather than merely non-panicking. Asserted against the
        // sanitised box itself, so the policy is pinned rather than described.
        assert_eq!(
            preprocess(&frame, &rect(f32::NAN, 0.0, 32.0, 32.0), 0.0),
            preprocess(&frame, &rect(0.0, 0.0, 32.0, 32.0), 0.0),
            "a NaN edge reads as zero"
        );
        assert_eq!(
            preprocess(&frame, &rect(0.0, 0.0, f32::INFINITY, 32.0), 0.0),
            preprocess(&frame, &rect(0.0, 0.0, 0.0, 32.0), 0.0),
            "so does an infinite width"
        );

        // A frame with no pixels has nothing to sample; black, not a panic.
        let empty =
            Frame::new(0, 0, Vec::new(), Instant::now()).expect("an empty frame is a frame");
        let tensor = preprocess(&empty, &eye_box, 0.0);
        assert!(
            values(&tensor).iter().all(|value| *value == expected(0.0)),
            "a frame with no pixels is black"
        );
    }

    #[test]
    fn classify_rejects_an_output_it_cannot_read() {
        // The field failure behind every one of these is a wrong or mismatched
        // model file, and the error text is what names it. Each is checked for
        // its *variant*, so a wrapper that answered `Ok(false)` for a garbage
        // tensor — silently reporting a confident open eye — cannot pass.
        let frame = solid_frame(64, 64, (10, 20, 30));
        let eye_box = rect(0.0, 0.0, 32.0, 32.0);
        let classify = |outputs: Vec<OutputTensor>| {
            EyeStateClassifier::new(Box::new(FakeSession::new("eyestate", outputs)))
                .classify(&frame, &eye_box, 0.0)
        };

        let error = classify(vec![OutputTensor {
            name: "output".to_owned(),
            ..probs_output(1.0, 0.0)
        }])
        .expect_err("a name the graph does not declare");
        assert!(
            matches!(&error, ModelError::UnexpectedOutput { expected, got } if *expected == "19" && got == "output"),
            "{error}"
        );

        let error = classify(vec![OutputTensor {
            shape: vec![1, 2],
            ..probs_output(1.0, 0.0)
        }])
        .expect_err("a shape the graph does not declare");
        assert!(
            matches!(error, ModelError::UnexpectedShape { .. }),
            "{error}"
        );

        let error = classify(vec![OutputTensor {
            data: TensorData::I32(vec![1, 0]),
            ..probs_output(1.0, 0.0)
        }])
        .expect_err("an integer tensor where floats belong");
        assert!(matches!(error, ModelError::MissingOutput { .. }), "{error}");

        let error = classify(vec![OutputTensor {
            data: TensorData::F32(vec![1.0]),
            ..probs_output(1.0, 0.0)
        }])
        .expect_err("a buffer shorter than the shape it declares");
        assert!(
            matches!(error, ModelError::UnexpectedDataLength { got: 1, .. }),
            "{error}"
        );

        let error = classify(vec![probs_output(1.0, 0.0), probs_output(0.0, 1.0)])
            .expect_err("two outputs where the graph declares one");
        assert!(
            matches!(
                error,
                ModelError::UnexpectedOutputCount {
                    expected: 1,
                    got: 2
                }
            ),
            "{error}"
        );
    }
}
