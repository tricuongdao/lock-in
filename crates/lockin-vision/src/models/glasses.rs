//! Glasses detection: the calibration-time fact that adapts the EAR threshold.
//!
//! This wrapper answers one yes/no question, once, while the user calibrates:
//! **are they wearing glasses?** Nothing else in the crate consumes the answer
//! yet — Task 18 wires it into the profile, where it raises the eye-aspect-ratio
//! threshold, because eyewear is the *documented* failure mode for EAR. Rimless
//! and transparent lenses shift the landmark positions around the eye without
//! darkening anything, so the ratio moves with no other symptom, and `Drowsy`
//! then fires either always or never for that user. A flag nothing reads would
//! be worse than no flag, which is why Task 18 must consume this one.
//!
//! It runs **only during calibration, never per tick**. That is what makes the
//! cost irrelevant and the correctness decisive: one 256x256 forward pass per
//! calibration, in exchange for a threshold that is right for the whole session.
//!
//! # The graph, read rather than assumed
//!
//! `classification_anyglasses_medium.onnx` (PINTO_model_zoo folder
//! `489_Glasses-Detector`; the source model is Mantas Birškus's
//! `glasses-detector`, MIT, DOI 10.5281/zenodo.8126101) was read directly with
//! `onnx` **and then run** before a line of this module was written:
//!
//! | tensor | name | shape | dtype |
//! |---|---|---|---|
//! | input | `images` | `[1, 3, 256, 256]` | f32 |
//! | output | `logits` | `[1, 1]` | f32 |
//!
//! 5,039,935 bytes, sha256
//! `d4d5e6ea5c079e21c94f605c2e89a2cc803e98f96f4ef10501e1de693d2d46b7`,
//! ir_version 8, producer `pytorch 2.12.1`, **opset 17**, 186 nodes, 126
//! initializers, no functions. This matches the signature the task brief
//! recorded; unlike Tasks 5 and 11, the documentation and the graph agree here.
//!
//! # The output is a raw logit, and that is pinned
//!
//! The graph's last node is `Gemm(..., fc.weight, fc.bias) -> logits`. **There
//! is no `Sigmoid` and no `Softmax` anywhere in the graph** — `read_graph.py`
//! reports both lists empty — so `logits` is the pre-activation value, free to
//! be negative and free to exceed 1. Measured on real head crops, this model
//! returns `-10.474` for a face without glasses and `+6.543` for one with them.
//!
//! The upstream demo confirms it from the other side: its `classify_crop`
//! returns `logit, sigmoid(logit)` as *two* values, and its
//! `--sunglasses-threshold` of `0.75` is compared against the **sigmoid**, not
//! the logit. So a caller that assumed this output was already a probability and
//! thresholded it at 0.75 would in fact be demanding a logit of at least
//! `ln(3) = 1.0986` — it would silently discard every marginal positive.
//!
//! The decision boundary here is therefore the logit's zero, and [`glasses_present`]
//! is where that lives. Note honestly what that means: `logit > 0.0` and
//! `sigmoid(logit) > 0.5` are the *same predicate*, so a wrapper that applied a
//! sigmoid would be indistinguishable through this interface. What is pinned is
//! the boundary and the raw value's freedom to leave `[0, 1]` unpunished.
//!
//! # Which head, and why it is not the sunglasses one
//!
//! The bundle ships four classification heads with identical graph signatures —
//! `anyglasses`, `eyeglasses`, `sunglasses`, `shadows`, each in a `small` and a
//! `medium` variant. **The sunglasses head is the wrong one for this job** and
//! the brief says so; running it says the same thing in numbers. On seven real
//! portraits (four wearing transparent glasses, three wearing none), cropped by
//! the upstream wholebody-YOLO head detector:
//!
//! | head | glasses-wearers called positive | non-wearers called negative |
//! |---|---|---|
//! | `anyglasses_medium` | 4 / 4 | 3 / 3 |
//! | `eyeglasses_medium` | 4 / 4 | 3 / 3 |
//! | `sunglasses_medium` | **0 / 4** | 3 / 3 |
//!
//! The sunglasses head is not broken — it is *correct*: none of those four
//! subjects wears sunglasses. That is precisely the failure being avoided. A
//! user in ordinary spectacles is the user whose EAR the flag exists to correct,
//! and the sunglasses head reports "no glasses" for every one of them.
//!
//! `anyglasses` is the head this wrapper is documented against, because its
//! positives include both transparent and opaque eyewear: it is the superset,
//! and missing a wearer leaves the documented failure mode uncorrected, while a
//! false positive only makes `Drowsy` marginally easier to raise — and that
//! direction is already gated by the eye classifier, since `Drowsy` needs both
//! signals to agree. The `small` variants were rejected on measurement too:
//! `anyglasses_small` scored 6/7, calling the round-wire spectacles in
//! `glasses_jobs.jpg` a `-0.463`. `medium` is the one that got all seven.
//!
//! # Preprocessing is this model's own: the fourth distinct regime
//!
//! **RGB**, each byte scaled to `0-1` and then standardised by the ImageNet mean
//! `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, resized to
//! **256x256**. The order matters and is not a style choice: the `/255` is
//! applied *first*, then the mean is subtracted, because that is what the
//! upstream demo does —
//!
//! ```text
//! tensor = np.asarray(resized, dtype=np.float32) / 255.0
//! tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
//! ```
//!
//! — and because the other order deforms the input in a way the graph cannot
//! report. Measured on the same seven crops, applying the mean and std to the
//! raw 0-255 bytes instead drives `anyglasses_medium` to `-1559` and `-2877` on
//! the very images it otherwise calls confidently positive, taking it from 7/7
//! to 3/7. The three other wrappers in this crate each have their own regime and
//! none of them may be folded together: Face Mesh reverses to RGB and scales by
//! `1/255` with no mean; YOLOX takes raw BGR `0-255` letterboxed; the eye
//! classifier takes BGR `(x - 127) / 255`. Mixing any two of these produces
//! plausible-looking numbers rather than an error.
//!
//! # The resize squeezes; it does not letterbox
//!
//! The brief fixes the input at 256x256 and says nothing about the aspect ratio,
//! so the choice was made from what the model was trained on: the upstream demo
//! crops a box and calls `PIL.Image.resize((256, 256))`, and PIL's two-argument
//! form **stretches** — it does not preserve the aspect ratio. The training
//! pipeline therefore fed this graph photographic head crops squeezed to a
//! square, and the wrapper does the same rather than introducing a letterbox the
//! model has never seen. A face box is close to square in practice, so the two
//! differ by little; where they differ, the model's own convention wins.
//!
//! # The crop is the face box, and that was measured too
//!
//! Upstream feeds this classifier a wholebody-YOLO **Head** box; this crate
//! hands it a YuNet **FaceBox**, which is a smaller window. That is a claim
//! about accuracy no unit test can check, so it was checked against the real
//! graph first: on the same seven portraits, the unmodified YuNet face box
//! reaches the same 7/7 as the upstream head box, and the verdicts are stable
//! from 1.0x to 2.0x that box, so the wrapper does not expand the crop. Growing
//! it would be an unmeasured guess in the one direction that matters less than
//! being right about the face.
//!
//! Sampling is bilinear at `cv2.resize`'s pixel-centre positions, matching the
//! convention [`facemesh`](crate::models::facemesh) and
//! [`eyestate`](crate::models::eyestate) already use, so that a crop of a
//! 256-pixel box is close to the identity map. The crop is **not** rotated by
//! the head's roll: the interface has no angle to give it, upstream does not
//! rotate either, and photographic head crops at a moderate roll are what the
//! model was trained on.

use crate::analyze::nms::Rect;
use crate::frame::Frame;
use crate::models::yunet::FaceBox;
use crate::models::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};

/// The graph's image input name, exactly as the file declares it.
const INPUT_NAME: &str = "images";

/// The graph's output name, exactly as the file declares it.
const OUTPUT_NAME: &str = "logits";

/// The square size the graph's input is fixed at.
const INPUT_SIZE: usize = 256;

/// Half of it, as the `f32` the sampler measures tensor coordinates from.
///
/// A separate constant rather than a `* 0.5` at the use site, for the same
/// reason [`eyestate`](crate::models::eyestate) has one: `INPUT_SIZE as f32 *
/// 0.5` and `(INPUT_SIZE / 2) as f32` happen to agree for an even size and would
/// silently disagree -- by a half pixel in every sample -- for an odd one.
const HALF_INPUT_SIZE: f32 = INPUT_SIZE as f32 * 0.5;
const _: () = assert!(
    HALF_INPUT_SIZE as usize * 2 == INPUT_SIZE,
    "INPUT_SIZE must be even"
);

/// Channels per pixel: RGB.
const CHANNELS: usize = 3;

/// The byte range a channel is divided by before the mean is subtracted.
///
/// Half of one affine map, with [`MEAN`] and [`STD`]: the graph was trained on
/// `(x / 255 - mean) / std`, and dropping or reordering any of the three is a
/// silent deformation rather than an error -- see the module docs for what it
/// costs, measured.
const SCALE: f32 = 255.0;

/// The ImageNet mean, **in RGB order**, which is the order the tensor is built.
const MEAN: [f32; CHANNELS] = [0.485, 0.456, 0.406];

/// The ImageNet standard deviation, in RGB order.
const STD: [f32; CHANNELS] = [0.229, 0.224, 0.225];

/// Whether a `logits` value says the crop contains glasses.
///
/// `true` is glasses present. The boundary is the logit's zero, which is the
/// sigmoid's `0.5` — see the module docs for why the value is a logit and not a
/// probability, and why a caller must not threshold it at the demo's `0.75`.
///
/// A value of **exactly** `0.0` is *not* glasses, and that direction is chosen
/// rather than observed. The only consumer is a profile flag that raises the
/// eye-aspect-ratio threshold, which makes `Drowsy` easier to raise; like the
/// eye classifier's tie, this one has to earn that. A logit lands on exactly
/// zero with probability zero on real pixels, so this is a specified boundary
/// rather than a measured one.
pub fn glasses_present(logit: f32) -> bool {
    logit > 0.0
}

/// One RGB byte scaled the way this graph was trained:
/// `(x / 255 - mean) / std`.
///
/// The division comes *first* and is not commutative with the subtraction. The
/// three constants are per channel and in RGB order, so `channel` selects both
/// the mean and the std; swapping any two of them leaves a tensor of exactly the
/// right shape and range whose values mean something else.
#[inline]
fn normalise(value: f32, channel: usize) -> f32 {
    (value / SCALE - MEAN[channel]) / STD[channel]
}

/// A finite copy of one of a box's four numbers, or `fallback` if it is not.
///
/// A [`Rect`] is four bare `f32`s that stage 1 built from a tensor decode, so a
/// NaN or an infinity is not impossible in it. Sanitising once here keeps the
/// arithmetic below finite, which is what makes the sampler's clamping
/// sufficient for totality rather than merely usual.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// One frame coordinate clamped into the frame, with a NaN pinned to the edge.
///
/// `f32::clamp` propagates a NaN, and a NaN coordinate would make `floor` NaN
/// and `as usize` zero, so the sampled pixel, its fraction and the resulting
/// tensor value would all be NaN. Nothing above this produces one, but a tensor
/// full of NaN is a *verdict* from this graph rather than an error, so the
/// sampler does not rely on that argument holding forever.
#[inline]
fn clamped(value: f32, max: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, max)
    }
}

/// Builds the RGB NCHW tensor the graph expects from one face's box.
///
/// Pure: same frame and box in, same tensor out, with no clock and no
/// randomness anywhere.
///
/// The box is resized to 256x256 **ignoring its aspect ratio**, because the
/// graph's input is fixed at a square and that is what `PIL.Image.resize((256,
/// 256))` — the training pipeline's own resize — does. A non-square box is
/// therefore squeezed rather than letterboxed; see the module docs for why the
/// model's own convention wins over a letterbox it has never seen.
///
/// The crop is **not** rotated by the head's roll: this interface is given no
/// angle, upstream rotates nothing either, and photographic head crops at a
/// moderate roll are the regime the model was trained on. The eye classifier is
/// the crate's one wrapper that rotates, and it does so because its graph was
/// trained on upright eyes specifically.
///
/// Sampling is bilinear at `cv2.resize`'s pixel-centre positions — output pixel
/// `i` reads source position `x1 + (i + 0.5) * step - 0.5`, plus the box's
/// centre offset — the same convention
/// [`facemesh`](crate::models::facemesh) and
/// [`eyestate`](crate::models::eyestate) use. Coordinates outside the frame
/// clamp to the edge pixel, so the sampler can never read outside the buffer,
/// including for a box that hangs off the frame. A frame with no pixels is
/// black.
///
/// **Normalisation is mandatory, not cosmetic.** The `/255`, the mean and the
/// std are one affine map in a fixed order, and getting it wrong is a silent
/// deformation rather than an error: applied to raw 0-255 bytes instead of
/// 0-1, the shipping model calls four photographed pairs of glasses `-1559`,
/// `-1726`, `-1598` and `-2877` — 3/7 instead of 7/7.
pub fn preprocess(frame: &Frame, face_box: &Rect) -> InputTensor {
    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut data = vec![0.0f32; plane * CHANNELS];

    // The box is sanitised once, here, rather than at each of the 196,608
    // samples.
    let x1 = finite_or(face_box.x, 0.0);
    let y1 = finite_or(face_box.y, 0.0);
    let width = finite_or(face_box.w, 0.0);
    let height = finite_or(face_box.h, 0.0);

    let step_x = width / INPUT_SIZE as f32;
    let step_y = height / INPUT_SIZE as f32;

    // Tensor coordinates are measured from the box's centre, and the trailing
    // `- 0.5` applied afterwards is `cv2.resize`'s pixel-*centre* convention,
    // which belongs to the frame's coordinates rather than to the offset: at
    // that position the arithmetic is the plain resize every golden-value test
    // in this crate asserts.
    let centre_x = x1 + width * 0.5;
    let centre_y = y1 + height * 0.5;
    // Derived from `INPUT_SIZE` rather than written as `128.0`, so the two
    // cannot drift apart if the graph's input size ever changes. The `as f32` is
    // load-bearing: `INPUT_SIZE / 2` alone would be an integer division, and a
    // bare `128.0` would make the whole expression infer as `f64`, which is a
    // half-pixel shift in every sample rather than a style point.
    let half = HALF_INPUT_SIZE;

    for row in 0..INPUT_SIZE {
        let offset_y = (row as f32 + 0.5 - half) * step_y;
        for column in 0..INPUT_SIZE {
            let offset_x = (column as f32 + 0.5 - half) * step_x;
            // The frame coordinate this tensor pixel samples. `source_x` and
            // `source_y` rather than the loop counters: those are *tensor*
            // coordinates, and using them crops the frame's top-left corner
            // instead of the caller's box. The tensor would still have the shape
            // the graph wants and its values would still be in range, so nothing
            // downstream could tell — every verdict would simply be about the
            // wrong face.
            let source_x = centre_x + offset_x - 0.5;
            let source_y = centre_y + offset_y - 0.5;
            let (r, g, b) = sample_rgb(frame, source_x, source_y);
            let index = row * INPUT_SIZE + column;
            // RGB, in the order Face Mesh's graph also wants and the frame does
            // *not* hold: `bgra` is blue first. Plane 0 is the red byte. This is
            // not a free choice — two of this crate's four models take RGB and
            // two take BGR, and the two that take RGB were both trained on
            // `PIL.Image.convert("RGB")` output.
            data[index] = normalise(r, 0);
            data[plane + index] = normalise(g, 1);
            data[2 * plane + index] = normalise(b, 2);
        }
    }

    InputTensor {
        name: INPUT_NAME.to_owned(),
        shape: vec![1, CHANNELS as i64, INPUT_SIZE as i64, INPUT_SIZE as i64],
        data: TensorData::F32(data),
    }
}

/// Bilinear red-green-blue sample at a floating-point frame coordinate.
///
/// Channels are returned red first, which is the order this graph's tensor is
/// built in — not the order `frame.bgra` holds them, which is blue first. The
/// reversal is therefore here, at the one place pixels are read, rather than
/// being left to the caller's plane writes.
fn sample_rgb(frame: &Frame, x: f32, y: f32) -> (f32, f32, f32) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return (0.0, 0.0, 0.0);
    }
    let x = clamped(x, (width - 1) as f32);
    let y = clamped(y, (height - 1) as f32);
    let left = x.floor();
    let top = y.floor();
    let right = (left as usize + 1).min(width - 1);
    let bottom = (top as usize + 1).min(height - 1);
    let (left, top) = (left as usize, top as usize);
    let frac_x = x - left as f32;
    let frac_y = y - top as f32;

    let pixel = |column: usize, row: usize| {
        let offset = (row * width + column) * 4;
        (
            f32::from(frame.bgra[offset + 2]),
            f32::from(frame.bgra[offset + 1]),
            f32::from(frame.bgra[offset]),
        )
    };
    let (r00, g00, b00) = pixel(left, top);
    let (r10, g10, b10) = pixel(right, top);
    let (r01, g01, b01) = pixel(left, bottom);
    let (r11, g11, b11) = pixel(right, bottom);

    let mix = |a: f32, b: f32, c: f32, d: f32| {
        let upper = a + (b - a) * frac_x;
        let lower = c + (d - c) * frac_x;
        upper + (lower - upper) * frac_y
    };
    (
        mix(r00, r10, r01, r11),
        mix(g00, g10, g01, g11),
        mix(b00, b10, b01, b11),
    )
}

/// The single `logits` value, after checking the output's name, shape and
/// length.
///
/// The length check is not redundant with the shape check: [`OutputTensor`]'s
/// fields are public, so a session can declare `[1, 1]` in front of a buffer of
/// any length, and reading `values[0]` past the end of an empty one would panic
/// in a function that returns `Result`.
fn read_logit(outputs: &[OutputTensor]) -> Result<f32, ModelError> {
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

    let expected = vec![1, 1];
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
        [logit] => Ok(*logit),
        _ => Err(ModelError::UnexpectedDataLength {
            name: OUTPUT_NAME,
            shape: output.shape.clone(),
            got: values.len(),
        }),
    }
}

/// Glasses classification over a session.
///
/// Constructed once and called once, at calibration. It holds no state between
/// calls, so it may be shared or dropped immediately afterwards.
pub struct GlassesDetector {
    session: Box<dyn InferenceSession>,
}

impl GlassesDetector {
    /// Wraps a session. The session is the model; this type is the arithmetic
    /// around it.
    pub fn new(session: Box<dyn InferenceSession>) -> GlassesDetector {
        GlassesDetector { session }
    }

    /// The wrapped session's name, for logs and for the provider autotune.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Whether the person in `face` is wearing glasses.
    ///
    /// `true` is glasses present, and the caller is calibration: the answer
    /// becomes a profile flag that **raises** the eye-aspect-ratio threshold,
    /// because eyewear shifts the eye landmarks and moves the ratio with no
    /// other symptom. A caller that ignores this return leaves the documented
    /// failure mode in place for every user who wears glasses.
    ///
    /// `face` is one detection from stage 1, in `frame`'s coordinates, and its
    /// `rect` is the window that is cropped — measured to reach the same
    /// verdicts as the upstream wholebody-YOLO head box this model was
    /// demonstrated with, so no expansion is applied.
    ///
    /// Costs one 256x256 RGB tensor per call — 196,608 floats. That is the
    /// largest single input in this crate, and it is affordable precisely
    /// because this runs once: the per-tick stages either side of it build
    /// tensors a fraction of this size.
    pub fn detect(&self, frame: &Frame, face: &FaceBox) -> Result<bool, ModelError> {
        let input = preprocess(frame, &face.rect);
        let outputs = self.session.run(&[input])?;
        let logit = read_logit(&outputs)?;
        Ok(glasses_present(logit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A detected face around `rect`, with the parts this wrapper does not read
    /// filled in plausibly.
    fn face(rect: Rect) -> FaceBox {
        FaceBox {
            rect,
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        }
    }

    /// The graph's output tensor, named and shaped as the file declares it.
    ///
    /// Built through the module's own `OUTPUT_NAME`, so a typo in that constant
    /// fails the tests here rather than being copied into them.
    fn logits_output(logit: f32) -> OutputTensor {
        OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, 1],
            data: TensorData::F32(vec![logit]),
        }
    }

    /// A detector over a session that always returns `output`, plus the handle
    /// that records what it was asked to run.
    fn detector_returning(output: OutputTensor) -> (GlassesDetector, FakeSessionHandle) {
        let session = FakeSession::single("classification_anyglasses_medium", output);
        let handle = session.handle();
        (GlassesDetector::new(Box::new(session)), handle)
    }

    /// The tensor's values, as an `f32` slice.
    fn values(tensor: &InputTensor) -> &[f32] {
        tensor
            .data
            .as_f32()
            .expect("preprocess builds an f32 tensor")
    }

    /// One value of a plane-major `[1, 3, 256, 256]` tensor.
    fn at(tensor: &InputTensor, channel: usize, column: usize, row: usize) -> f32 {
        values(tensor)[channel * INPUT_SIZE * INPUT_SIZE + row * INPUT_SIZE + column]
    }

    /// One red/green/blue byte written as the graph was trained to read it.
    ///
    /// The ImageNet constants are spelled out here as literals rather than read
    /// from the module's `MEAN` and `STD`, so that changing those constants
    /// moves the tensor and not the expectation it is compared against. A helper
    /// that restated the production expression would stay green under any
    /// mutation of the production constants.
    fn norm(byte: f32) -> f32 {
        (byte / 255.0 - 0.485) / 0.229
    }

    /// The same, for the green plane, whose mean and std are the middle pair.
    fn norm_green(byte: f32) -> f32 {
        (byte / 255.0 - 0.456) / 0.224
    }

    /// The same, for the blue plane, whose mean and std are the last pair.
    fn norm_blue(byte: f32) -> f32 {
        (byte / 255.0 - 0.406) / 0.225
    }

    #[test]
    fn preprocessing_is_rgb_with_imagenet_mean_and_std() {
        // A frame whose three channels are 0, 128 and 255 tells RGB from BGR in
        // one go: red is the 255 byte, green the 128 and blue the 0.
        //
        // The frame is solid, so bilinear resampling returns it unchanged
        // whatever the geometry is, and every assertion below is about the
        // regime alone rather than about which pixels were cropped.
        let frame = solid_frame(96, 96, (0, 128, 255));
        let tensor = preprocess(&frame, &rect(0.0, 0.0, 96.0, 96.0));
        assert_eq!(
            tensor.shape,
            vec![1, 3, 256, 256],
            "NCHW at the graph's fixed input size"
        );

        let plane = INPUT_SIZE * INPUT_SIZE;
        // Hand-computed from the brief's own constants:
        //   red   = (255/255 - 0.485) / 0.229 =  0.515 / 0.229 =  2.248908
        //   green = (128/255 - 0.456) / 0.224 =  0.045961 / 0.224 =  0.205183
        //   blue  = (  0/255 - 0.406) / 0.225 = -0.406 / 0.225 = -1.804444
        let red = norm(255.0);
        let green = norm_green(128.0);
        let blue = norm_blue(0.0);
        // The three are distinct, so a channel-order swap cannot pass by
        // coincidence, and their decimals are named so a failure says which of
        // the three constants moved.
        assert!((red - 2.248908).abs() < 1e-5, "red plane: got {red}");
        assert!((green - 0.205183).abs() < 1e-5, "green plane: got {green}");
        assert!((blue + 1.804444).abs() < 1e-5, "blue plane: got {blue}");

        // Every pixel of every plane, not a sampled few: a channel-order swap
        // applied to one plane and not the others is exactly what this catches.
        for row in 0..INPUT_SIZE {
            for column in 0..INPUT_SIZE {
                let index = row * INPUT_SIZE + column;
                let got = &values(&tensor)[index..];
                assert_eq!(
                    [got[0], got[plane], got[2 * plane]],
                    [red, green, blue],
                    "plane order at ({column}, {row})"
                );
            }
        }

        // The halves of the affine map, named so that dropping one fails with a
        // message that says which. Without the `/255`, red would be
        // `(255 - 0.485) / 0.229 = 1111.4` -- and that is not hypothetical
        // either: on seven real head crops, applying the mean and std to the raw
        // 0-255 bytes took the shipping model from 7/7 to 3/7, driving the
        // glasses-wearers to -1559 and -2877.
        assert!(
            red < 3.0,
            "the /255 is present: applying mean and std to raw 0-255 bytes \
             would put red at 1111.4, got {red}"
        );
        assert!(
            blue < 0.0,
            "and the mean is subtracted after it: a bare x/255 would leave the \
             0 byte at exactly 0.0, got {blue}"
        );
        // Red is the channel the frame holds last, so a wrapper that passed BGR
        // through would put `norm_blue(255.0) = 2.889` in plane 0 instead.
        assert_ne!(
            red,
            (255.0f32 / 255.0 - 0.406) / 0.225,
            "plane 0 is the red byte, not the blue one"
        );
    }

    #[test]
    fn positive_logit_means_glasses_present() {
        // Real values from the shipping model, on real head crops: the graph's
        // `logits` is a raw logit, so these are free to be negative and free to
        // exceed 1, and they are what the threshold is a threshold *on*.
        assert!(glasses_present(6.543), "a photographed pair of glasses");
        assert!(glasses_present(3.280), "and a second one");
        assert!(
            glasses_present(0.407),
            "the weakest real positive measured: a wire-rimmed pair, and still \
             above the boundary"
        );
        assert!(!glasses_present(-5.141), "a real face with no glasses");
        assert!(!glasses_present(-10.474), "and a second one");

        // The boundary is the logit's zero, which is the sigmoid's 0.5. Pinned
        // from both sides and *at* zero, because the flag this feeds raises the
        // EAR threshold and therefore makes `Drowsy` easier to raise: like the
        // eye classifier's tie, it has to earn that. A value of exactly 0.0 is
        // not positive, so it is not glasses.
        assert!(glasses_present(0.001), "just above zero is present");
        assert!(
            !glasses_present(0.0),
            "a tie is not glasses: the flag makes Drowsy easier to raise"
        );
        assert!(!glasses_present(-0.001), "just below zero is absent");

        // And the same through the session, because the boundary living in
        // `glasses_present` is worth nothing if `detect` does not reach it.
        let frame = solid_frame(64, 64, (10, 20, 30));
        let window = face(rect(8.0, 8.0, 48.0, 48.0));

        let (present, _) = detector_returning(logits_output(6.543));
        assert!(
            present.detect(&frame, &window).expect("a well-formed output"),
            "a positive logit must reach the caller as glasses present"
        );

        let (absent, _) = detector_returning(logits_output(-10.474));
        assert!(
            !absent.detect(&frame, &window).expect("a well-formed output"),
            "a negative logit must reach the caller as absent"
        );

        // A raw logit leaves [0, 1] on both sides, and neither is a malformed
        // *probability* to be rejected. A wrapper that validated its output as a
        // probability would error on every confidently-classified face, which is
        // every face.
        assert!(
            detector_returning(logits_output(-10.474))
                .0
                .detect(&frame, &window)
                .is_ok(),
            "-10.474 is a logit, not an out-of-range probability"
        );
        assert!(
            detector_returning(logits_output(6.543))
                .0
                .detect(&frame, &window)
                .is_ok(),
            "and neither is 6.543"
        );
    }

    #[test]
    fn face_crop_is_resized_to_256_square() {
        // The frame's blue is the column and its green is the row, so every
        // sample in the tensor says which frame pixel it came from -- the only
        // way to see *which* window was cropped and *how* it was stretched,
        // rather than merely that something was.
        let frame = frame_with(128, 128, |x, y| (x as u8, y as u8, 0));

        // A deliberately non-square box: 64 wide by 32 tall at (16, 32). At the
        // graph's 256 this gives `step_x = 64/256 = 0.25` and
        // `step_y = 32/256 = 0.125`, so the two axes' steps differ by exactly
        // 2x -- which is what makes a `step_y` computed from the width visible.
        //
        // The source positions follow from the box's centre (48, 48) and the
        // pixel-centre convention `x1 + w/2 + (i + 0.5 - 128) * step - 0.5`:
        //
        //   column     0 -> 48 - 31.875 - 0.5 = 15.625    (blue)
        //   column   255 -> 48 + 31.875 - 0.5 = 79.375    (blue)
        //   row        0 -> 48 - 15.9375 - 0.5 = 31.5625  (green)
        //   row      255 -> 48 + 15.9375 - 0.5 = 63.4375  (green)
        //
        // The frame is a linear ramp in both channels, so bilinear
        // interpolation returns those fractional positions exactly.
        let window = rect(16.0, 32.0, 64.0, 32.0);
        let tensor = preprocess(&frame, &window);

        assert_eq!(
            tensor.shape,
            vec![1, 3, 256, 256],
            "the box is squeezed to the graph's square, not letterboxed"
        );
        assert_eq!(values(&tensor).len(), CHANNELS * INPUT_SIZE * INPUT_SIZE);

        // Blue -- the frame's column -- pinned at both ends. Had the box been
        // ignored, column 0 would sample the frame's left edge and read
        // `norm_blue(0.0)`; had the width come from the height, column 255 would
        // read `norm_blue(15.625 + 31.875) = norm_blue(47.5)`.
        assert_eq!(
            at(&tensor, 2, 0, 0),
            norm_blue(15.625),
            "the box's left edge, four pixels inside it"
        );
        assert_eq!(
            at(&tensor, 2, 255, 0),
            norm_blue(79.375),
            "and its right edge: 64 columns of content across 256 outputs"
        );

        // Green -- the frame's row -- and this is the squeeze. The box holds 32
        // rows spread over 256 outputs, so output row 255 is still *inside* the
        // box. Had `step_y` come from the box's width it would be 0.25, the
        // sampling would run 32 rows past the box's own bottom edge, and these
        // two numbers would be 31.5625 and 79.375 -- the same as the column's.
        assert_eq!(
            at(&tensor, 1, 0, 0),
            norm_green(31.5625),
            "row 0 samples 31.5625"
        );
        assert_eq!(
            at(&tensor, 1, 0, 255),
            norm_green(63.4375),
            "row 255 samples 63.4375, not 79.375"
        );

        // The whole green plane stays within the box's own rows, as a range
        // rather than at two sampled points. The plane is a linear ramp in the
        // row, so its extremes are its endpoints.
        let greens: Vec<f32> = values(&tensor)
            [INPUT_SIZE * INPUT_SIZE..2 * INPUT_SIZE * INPUT_SIZE]
            .to_vec();
        assert_eq!(
            greens.iter().copied().fold(f32::INFINITY, f32::min),
            norm_green(31.5625)
        );
        assert_eq!(
            greens.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            norm_green(63.4375)
        );

        // The red plane is the frame's third byte, which this frame holds at 0
        // everywhere: a constant, so a crop of any window would agree. It is
        // asserted to be constant precisely because that is what proves the
        // columns and rows above are not red leaking into the other planes.
        let reds: Vec<f32> = values(&tensor)[..INPUT_SIZE * INPUT_SIZE].to_vec();
        assert_eq!(
            reds.iter().copied().fold(f32::INFINITY, f32::min),
            norm(0.0)
        );
        assert_eq!(
            reds.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            norm(0.0)
        );
    }

    #[test]
    fn detect_crops_exactly_the_face_box_through_the_session() {
        // The seam this task exists to keep working: whatever stage 1 produces
        // must be a box this wrapper can crop, and the verdict must be about
        // that window. The shape assertions elsewhere would all pass on a
        // wrapper that cropped the whole frame.
        let frame = frame_with(128, 128, |x, y| (x as u8, y as u8, 0));
        let window = face(rect(16.0, 32.0, 64.0, 32.0));
        let (detector, handle) = detector_returning(logits_output(4.0));

        assert!(
            detector
                .detect(&frame, &window)
                .expect("a well-formed output"),
            "a positive logit is glasses present"
        );

        let calls = handle.inputs();
        assert_eq!(calls.len(), 1, "one crop is one run");
        assert_eq!(calls[0].len(), 1, "this graph takes exactly one input");
        let input = &calls[0][0];
        assert_eq!(input.name, INPUT_NAME, "the graph's own spelling");
        assert_eq!(input.shape, vec![1, 3, 256, 256]);
        assert!(
            matches!(input.data, TensorData::F32(_)),
            "f32, not quantised"
        );
        // The values, which are the only thing that says *which* pixels the
        // answer is about. A wrapper that preprocessed some other window
        // satisfies every assertion above.
        assert_eq!(
            input.data,
            preprocess(&frame, &window.rect).data,
            "the session got the crop of the caller's face box, not some other window"
        );
        assert_ne!(
            preprocess(&frame, &rect(0.0, 0.0, 64.0, 32.0)).data,
            input.data,
            "and the face box is what decides it"
        );

        // A different box is a different tensor, so `rect` is read and not
        // ignored or replaced by a constant.
        let other = face(rect(48.0, 64.0, 64.0, 32.0));
        let (second, second_handle) = detector_returning(logits_output(4.0));
        second.detect(&frame, &other).expect("a well-formed output");
        assert_ne!(
            second_handle.inputs()[0][0].data,
            input.data,
            "two faces crop to two different tensors"
        );
    }

    #[test]
    fn preprocessing_is_deterministic_and_total_for_an_unusable_box() {
        let frame = frame_with(128, 128, |x, y| (x as u8, y as u8, 0));
        let window = rect(16.0, 32.0, 64.0, 32.0);

        assert_eq!(
            preprocess(&frame, &window),
            preprocess(&frame, &window),
            "pure: the same frame and box give the same tensor"
        );

        // A box stage 1 could not have produced. `Rect` is four bare floats, so
        // none of these is impossible from a hand-built or NaN-poisoned box, and
        // the wrapper has to stay total: a non-finite field reads as zero, and
        // every one of these must still give a tensor the graph can be handed.
        for (label, box_) in [
            ("a NaN edge", rect(f32::NAN, 0.0, 64.0, 32.0)),
            ("a NaN width", rect(0.0, 0.0, f32::NAN, 32.0)),
            ("an infinite width", rect(0.0, 0.0, f32::INFINITY, 32.0)),
            (
                "an infinite origin",
                rect(f32::NEG_INFINITY, 0.0, 64.0, 32.0),
            ),
            ("no area", rect(0.0, 0.0, 0.0, 0.0)),
            ("a negative area", rect(64.0, 64.0, -16.0, -16.0)),
            ("a box off the frame", rect(4000.0, 4000.0, 64.0, 32.0)),
        ] {
            let tensor = preprocess(&frame, &box_);
            let data = values(&tensor);
            assert_eq!(data.len(), CHANNELS * INPUT_SIZE * INPUT_SIZE, "{label}");
            assert!(
                data.iter().all(|value| value.is_finite()),
                "{label} put a non-finite value in the tensor"
            );
        }

        // A non-finite field reads as zero, which is what makes the two boxes
        // above total rather than merely non-panicking. Asserted against the
        // sanitised box itself, so the policy is pinned rather than described.
        assert_eq!(
            preprocess(&frame, &rect(f32::NAN, 0.0, 64.0, 32.0)),
            preprocess(&frame, &rect(0.0, 0.0, 64.0, 32.0)),
            "a NaN edge reads as zero"
        );
        assert_eq!(
            preprocess(&frame, &rect(0.0, 0.0, f32::INFINITY, 32.0)),
            preprocess(&frame, &rect(0.0, 0.0, 0.0, 32.0)),
            "so does an infinite width"
        );

        // A frame with no pixels has nothing to sample; black, not a panic.
        let empty = Frame::new(0, 0, Vec::new(), Instant::now()).expect("an empty frame is legal");
        let tensor = preprocess(&empty, &window);
        assert!(
            values(&tensor).iter().all(|value| value.is_finite()),
            "an empty frame must not put a non-finite value in the tensor"
        );
        assert!(
            values(&tensor).iter().all(|value| *value < 0.0),
            "every channel of a black pixel is below the ImageNet mean"
        );
    }

    #[test]
    fn a_wrong_output_is_reported_rather_than_read() {
        // `OutputTensor`'s fields are public, so a session can hand back any
        // name, shape and buffer it likes. Each of these is a real export or
        // wiring mistake, and each must be an error rather than a verdict.
        let frame = solid_frame(64, 64, (10, 20, 30));
        let window = face(rect(8.0, 8.0, 48.0, 48.0));

        let call = |output: OutputTensor| {
            detector_returning(output)
                .0
                .detect(&frame, &window)
                .map_err(|error| error.to_string())
        };

        // The head is there under a different name -- a graph that exported
        // `output` instead of `logits`, which is what a re-export looks like.
        let wrong_name = call(OutputTensor {
            name: "output".to_owned(),
            shape: vec![1, 1],
            data: TensorData::F32(vec![1.0]),
        });
        assert!(
            wrong_name
                .as_ref()
                .is_err_and(|message| message.contains("output") && message.contains("logits")),
            "a misnamed output must name both, got {wrong_name:?}"
        );

        // The two-class shape: the classification bundle's *other* exports are
        // the same graph family, and a head read at the wrong width would put a
        // second class's value where this wrapper reads the first.
        let wrong_shape = call(OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, 2],
            data: TensorData::F32(vec![1.0, 2.0]),
        });
        assert!(
            wrong_shape.as_ref().is_err_and(|message| message.contains("[1, 2]")),
            "a wrong shape must report what it got, got {wrong_shape:?}"
        );

        // The shape is right and the buffer is short: `[1, 1]` promises one
        // value. Indexing it would panic, and a function returning `Result`
        // must not.
        let short = call(OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, 1],
            data: TensorData::F32(Vec::new()),
        });
        assert!(
            short.as_ref().is_err_and(|message| message.contains("holds 0")),
            "a short buffer must report its actual length, got {short:?}"
        );

        // Quantised, or any other non-float tensor: the graph is f32 and a
        // session that returns integers has been pointed at the wrong file.
        let integers = call(OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, 1],
            data: TensorData::I32(vec![1]),
        });
        assert!(
            integers
                .as_ref()
                .is_err_and(|message| message.contains("logits")),
            "an integer tensor must be refused by name, got {integers:?}"
        );

        // Two heads where the graph declares one: this wrapper reads the
        // verdict from a single value and has no rule for a second.
        let two = detector_returning_outputs(vec![logits_output(1.0), logits_output(-1.0)]);
        let two = two.detect(&frame, &window).map_err(|error| error.to_string());
        assert!(
            two.as_ref()
                .is_err_and(|message| message.contains("2 output") && message.contains("1")),
            "two outputs must be refused, naming the count, got {two:?}"
        );
    }

    /// A detector over a session returning several outputs, for the count check.
    fn detector_returning_outputs(outputs: Vec<OutputTensor>) -> GlassesDetector {
        GlassesDetector::new(Box::new(FakeSession::new("glasses", outputs)))
    }
}
