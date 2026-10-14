//! Face Mesh: 468 dense landmarks for one face, the second stage of the engine.
//!
//! YuNet says *where* a face is; this says what it is doing. The 468 points are
//! the input to head pose (Task 8), to the eye aspect ratio and the eye box
//! (Task 9) and to phone association (Task 12), so a decode that is subtly
//! wrong here is wrong in every stage after it.
//!
//! # The graph, read rather than assumed
//!
//! `face_mesh_Nx3x192x192_post.onnx` (PINTO_model_zoo 032, sha256
//! `ae6ada90…c04eb`) declares **five** inputs and two outputs:
//!
//! | tensor | shape | dtype |
//! |---|---|---|
//! | `input` | `[batch, 3, 192, 192]` | f32 |
//! | `crop_x1`, `crop_y1`, `crop_width`, `crop_height` | `[batch, 1]` | i32 |
//! | `score` | `[batch, 1]` | f32 |
//! | `final_landmarks` | `[batch, 468, 3]` | i32 |
//!
//! The crop rectangle is **four of the graph's five inputs**, not a Rust-side
//! convenience: the post-process is fused into the graph, and its last eight
//! nodes are
//!
//! ```text
//! x = cast_i32(crop_x1 + x_model * crop_width  / 192 + 0.5)
//! y = cast_i32(crop_y1 + y_model * crop_height / 192 + 0.5)
//! z = z_model
//! ```
//!
//! read node by node out of `post_process.onnx` and the merged graph (the
//! `192` is a constant-folded `1/192` reciprocal, the `0.5` a folded add). So
//! [`Landmarks::points`] arrive as **absolute pixels in the source frame** and
//! are handed back untouched — there is no un-cropping anywhere in this module,
//! and adding `crop_x1` a second time is the bug this file exists to prevent.
//!
//! Three other things the graph and the upstream demo settle, each of which the
//! task brief could only assert:
//!
//! - **RGB**, not the BGR the frame holds. `demo_video.py` line 658 feeds
//!   `cv2.resize(...)[:,:,::-1] / 255`, reversing a BGR frame into RGB. Feeding
//!   BGR to the real graph on a real face drops `score` from `18.1` to `11.9`.
//! - **192x192 ignoring the aspect ratio**: the input is fixed at a square and
//!   the fused decode divides by exactly that, so a 2:1 crop is squeezed, never
//!   letterboxed.
//! - **468 points and no iris points**: `final_landmarks` is `[N, 468, 3]`.
//!   MediaPipe's iris refinement (indices 468-477) is simply not in this model,
//!   which is why the engine has no gaze signal to read here.
//!
//! # What `score` actually is
//!
//! The upstream README calls `score` a "probability value". It is not: the
//! graph's last convolution is followed by a bare `Reshape`, with no `Sigmoid`
//! anywhere in its 214 nodes, and a real face measures `+18.1` while an empty
//! wall measures `-9.0` and a too-narrow crop measuring `-2.9`. It is an
//! unbounded **logit**, and it is returned as the graph produced it — applying a
//! sigmoid here would be inventing arithmetic the model never did, and would
//! make the demo's own `score > 0.95` gate mean something else. A caller that
//! wants a probability can take the sigmoid of it.

use crate::analyze::nms::Rect;
use crate::frame::Frame;
use crate::models::yunet::FaceBox;
use crate::models::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};

/// How much of the face box's own size is added on **each** side before
/// cropping, as the model card asks for.
///
/// The card's warning is the reason it exists: the crop "must not be too narrow",
/// because a crop tighter than the face cuts off the jaw and the outer brows and
/// the detector loses the points the ear and pose stages need. Measured on the
/// graph, cropping the left face of the research photo to its eyes and nose
/// alone drives `score` from `+18.1` to `-2.9`.
pub const CROP_MARGIN: f32 = 0.25;

/// The crop's size as a multiple of the face box's: a margin on both sides.
const CROP_SCALE: f32 = 1.0 + 2.0 * CROP_MARGIN;

/// The height and width the graph's image input is fixed at.
const INPUT_SIZE: usize = 192;

/// Points in `final_landmarks`'s second dimension.
const LANDMARKS: usize = 468;

/// `x`, `y` and `z` per point.
const COORDINATES: usize = 3;

/// The graph's five input names, in the order it declares them.
const INPUT_NAME: &str = "input";
const CROP_X1_NAME: &str = "crop_x1";
const CROP_Y1_NAME: &str = "crop_y1";
const CROP_WIDTH_NAME: &str = "crop_width";
const CROP_HEIGHT_NAME: &str = "crop_height";

/// The graph's two output names.
const SCORE_NAME: &str = "score";
const LANDMARKS_NAME: &str = "final_landmarks";

/// How many outputs the graph declares: `score` and `final_landmarks`.
const OUTPUT_COUNT: usize = 2;

/// The value `1/255` scales bytes into the unit range with.
const UNIT_RANGE: f32 = 1.0 / 255.0;

/// The rectangle of the source frame that is cropped, resized and fed to the
/// graph — and, because the decode is fused, the four extra inputs that put the
/// resulting landmarks back in frame coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropRect {
    /// Left edge in frame pixels.
    pub x1: i32,
    /// Top edge in frame pixels.
    pub y1: i32,
    /// Width in frame pixels; at least one for a frame with pixels.
    pub width: i32,
    /// Height in frame pixels.
    pub height: i32,
}

impl CropRect {
    /// The right edge, exclusive.
    pub fn x2(&self) -> i32 {
        self.x1 + self.width
    }

    /// The bottom edge, exclusive.
    pub fn y2(&self) -> i32 {
        self.y1 + self.height
    }
}

/// The crop rectangle for `face_box` in a `frame_width` x `frame_height` frame.
///
/// The box is grown by [`CROP_MARGIN`] of its own size on each side — 1.5x the
/// box, centred on it — and the window is then **slid** back inside the frame
/// rather than shrunk. Sliding is the point: a face in a corner still gets the
/// full margin on the three sides that have room, where shrinking would take it
/// off the side that does not and hand the model a crop barely wider than the
/// face. Only when the frame itself is smaller than the crop does the crop give
/// way, and then only down to the frame — which still contains the box.
///
/// Pure arithmetic over the box's own numbers; a box with no area or a frame
/// with no pixels yields the degenerate rect rather than a panic.
pub fn crop_rect(face_box: &Rect, frame_width: u32, frame_height: u32) -> CropRect {
    let width = crop_extent(face_box.w, frame_width);
    let height = crop_extent(face_box.h, frame_height);
    CropRect {
        x1: crop_origin(face_box.x, face_box.w, width, frame_width),
        y1: crop_origin(face_box.y, face_box.h, height, frame_height),
        width,
        height,
    }
}

/// One axis of the crop's size: the box's, scaled, held inside the frame and
/// never below one pixel.
fn crop_extent(size: f32, frame: u32) -> i32 {
    let frame = frame.min(i32::MAX as u32) as i32;
    if frame == 0 {
        // No pixels to crop. The graph gets a black tensor instead.
        return 0;
    }
    // A zero, negative or NaN extent — a `FaceBox` built by hand rather than by
    // YuNet — becomes a single pixel: the smallest thing that is still a crop.
    let scaled = (size * CROP_SCALE).round();
    let scaled = if scaled.is_finite() { scaled } else { 1.0 };
    scaled.clamp(1.0, frame as f32) as i32
}

/// One axis of the crop's origin, centring the crop on the box and then sliding
/// it inside the frame.
fn crop_origin(edge: f32, size: f32, extent: i32, frame: u32) -> i32 {
    let frame = frame.min(i32::MAX as u32) as i32;
    if extent <= 0 || frame <= 0 {
        return 0;
    }
    let centred = (edge + size / 2.0) - extent as f32 / 2.0;
    let centred = if centred.is_finite() { centred } else { 0.0 };
    // `extent <= frame`, so the upper bound is at least zero and the clamp
    // cannot be asked for an empty range.
    centred.clamp(0.0, (frame - extent) as f32) as i32
}

/// One `[1, 1]` i32 input: the graph's crop-rectangle scalars.
fn scalar_input(name: &str, value: i32) -> InputTensor {
    InputTensor {
        name: name.to_owned(),
        shape: vec![1, 1],
        data: TensorData::I32(vec![value]),
    }
}

/// Builds the RGB NCHW tensor Face Mesh expects from `crop` of `frame`.
///
/// Pure: same frame and crop in, same tensor out, with no clock and no
/// randomness.
///
/// The crop is resized to 192x192 **ignoring its aspect ratio** — the graph's
/// input is fixed at a square, and its fused decode divides by exactly 192 — so
/// a 2:1 crop is squeezed rather than letterboxed. Sampling follows
/// `cv2.resize`'s bilinear filter, the one the upstream demo actually runs:
/// output pixel `i` reads source position `x1 + (i + 0.5) * width / 192 - 0.5`,
/// so pixel *centres* align rather than corners.
pub fn preprocess(frame: &Frame, crop: CropRect) -> InputTensor {
    let plane = INPUT_SIZE * INPUT_SIZE;
    // One buffer for all three planes rather than three to copy in afterwards.
    let mut data = vec![0.0f32; plane * COORDINATES];
    let step_x = crop.width as f32 / INPUT_SIZE as f32;
    let step_y = crop.height as f32 / INPUT_SIZE as f32;

    for row in 0..INPUT_SIZE {
        let source_y = crop.y1 as f32 + (row as f32 + 0.5) * step_y - 0.5;
        for column in 0..INPUT_SIZE {
            let source_x = crop.x1 as f32 + (column as f32 + 0.5) * step_x - 0.5;
            let (r, g, b) = sample_rgb(frame, source_x, source_y);
            let index = row * INPUT_SIZE + column;
            // RGB, the order this graph was trained on — the reverse of the
            // BGR frame and of YuNet's own tensor. YuNet's sampler is the same
            // arithmetic with the channels the other way round; the two are
            // separate functions because the two models disagree.
            data[index] = r * UNIT_RANGE;
            data[plane + index] = g * UNIT_RANGE;
            data[2 * plane + index] = b * UNIT_RANGE;
        }
    }

    InputTensor {
        name: INPUT_NAME.to_owned(),
        shape: vec![1, COORDINATES as i64, INPUT_SIZE as i64, INPUT_SIZE as i64],
        data: TensorData::F32(data),
    }
}

/// Bilinear red-green-blue sample at a floating-point frame coordinate.
///
/// Coordinates outside the crop — its half-pixel borders, and a degenerate crop
/// whose `step` is zero — clamp to the frame's edge pixel, so the sampler can
/// never read outside the buffer. A frame with no pixels is black.
fn sample_rgb(frame: &Frame, x: f32, y: f32) -> (f32, f32, f32) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return (0.0, 0.0, 0.0);
    }
    // `NaN.clamp` is `NaN` and `NaN as usize` is `0`, so even a NaN coordinate
    // lands on a real pixel instead of panicking. The crop's own fields are
    // integers, so it cannot arise from `crop_rect`.
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
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

/// One face's dense landmarks, in the source frame's pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmarks {
    /// The 468 points, `[x, y, z]` each. `x` and `y` are **absolute pixels in
    /// the frame the face was cropped from** — the graph's fused post-process
    /// has already put them there — and `z` is the model's own relative depth,
    /// in the same arbitrary units the model was trained with, *not* scaled to
    /// the crop.
    ///
    /// The ordering is MediaPipe's face-mesh topology, which is what makes
    /// per-point indices (eye corners, chin) meaningful downstream. There are no
    /// iris points: indices 468-477 of the full 478-point topology are absent.
    pub points: [[i32; 3]; LANDMARKS],
    /// The graph's own confidence that it found a face to landmark.
    ///
    /// A **raw logit**, not a probability — see the module docs. Measured on the
    /// shipping model: `+12` to `+18` for a real face, `-7` to `-12` for a wall
    /// or a blank image, and `-2.9` for a crop too narrow to hold the face.
    pub score: f32,
}

/// Face Mesh over a session.
pub struct FaceMesh {
    session: Box<dyn InferenceSession>,
}

impl FaceMesh {
    /// Wraps a session. The session is the model; this type is the arithmetic
    /// around it.
    pub fn new(session: Box<dyn InferenceSession>) -> FaceMesh {
        FaceMesh { session }
    }

    /// The wrapped session's name, for logs and for the provider autotune.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Reads the 468 landmarks of `face` out of `frame`.
    ///
    /// `face` comes from [`crate::models::yunet::YuNet::detect`], so its box is
    /// already in `frame`'s coordinates; the crop is derived here and handed to
    /// the graph, which is what puts the returned points back in frame
    /// coordinates without any arithmetic in this crate.
    ///
    /// Costs one 192x192 RGB tensor built from the crop, so the crop should be
    /// one face rather than a crowd.
    pub fn landmarks(&self, frame: &Frame, face: &FaceBox) -> Result<Landmarks, ModelError> {
        let crop = crop_rect(&face.rect, frame.width, frame.height);
        let input = preprocess(frame, crop);
        // The graph's own declaration order, so a session that binds its inputs
        // positionally rather than by name still gets them right.
        let outputs = self.session.run(&[
            input,
            scalar_input(CROP_X1_NAME, crop.x1),
            scalar_input(CROP_Y1_NAME, crop.y1),
            scalar_input(CROP_WIDTH_NAME, crop.width),
            scalar_input(CROP_HEIGHT_NAME, crop.height),
        ])?;
        decode(&outputs)
    }
}

/// Reads the graph's two outputs into [`Landmarks`].
///
/// Nothing here maps, scales or offsets a point: `final_landmarks` is already in
/// frame pixels. See the module docs for the node-by-node derivation.
fn decode(outputs: &[OutputTensor]) -> Result<Landmarks, ModelError> {
    if outputs.len() != OUTPUT_COUNT {
        return Err(ModelError::UnexpectedOutputCount {
            expected: OUTPUT_COUNT,
            got: outputs.len(),
        });
    }
    Ok(Landmarks {
        points: read_landmarks(outputs)?,
        score: read_score(outputs)?,
    })
}

/// The graph's `score`, after checking its name, shape and length.
fn read_score(outputs: &[OutputTensor]) -> Result<f32, ModelError> {
    let output = output_named(outputs, SCORE_NAME)?;
    let expected = vec![1, 1];
    if output.shape != expected {
        return Err(ModelError::UnexpectedShape {
            name: SCORE_NAME,
            expected,
            got: output.shape.clone(),
        });
    }
    let values = output
        .data
        .as_f32()
        .ok_or(ModelError::MissingOutput { name: SCORE_NAME })?;
    match values.first() {
        Some(value) => Ok(*value),
        None => Err(ModelError::UnexpectedDataLength {
            name: SCORE_NAME,
            shape: output.shape.clone(),
            got: 0,
        }),
    }
}

/// The graph's `final_landmarks`, after checking its name, shape and length.
///
/// The length check is not redundant with the shape check: [`OutputTensor`]'s
/// fields are public, so a session — or a test — can hand back a declared shape
/// that its buffer does not fill, and copying by row after that would read past
/// the end of the slice, which panics in a function that promises a `Result`.
fn read_landmarks(outputs: &[OutputTensor]) -> Result<[[i32; COORDINATES]; LANDMARKS], ModelError> {
    let output = output_named(outputs, LANDMARKS_NAME)?;
    let expected = vec![1, LANDMARKS as i64, COORDINATES as i64];
    if output.shape != expected {
        return Err(ModelError::UnexpectedShape {
            name: LANDMARKS_NAME,
            expected,
            got: output.shape.clone(),
        });
    }
    let values = output.data.as_i32().ok_or(ModelError::MissingOutput {
        name: LANDMARKS_NAME,
    })?;
    if values.len() != LANDMARKS * COORDINATES {
        return Err(ModelError::UnexpectedDataLength {
            name: LANDMARKS_NAME,
            shape: output.shape.clone(),
            got: values.len(),
        });
    }
    let mut points = [[0i32; COORDINATES]; LANDMARKS];
    // `as_chunks` rather than an index per coordinate: it makes the row width
    // and the point count the same constant, so the two cannot drift apart.
    for (point, row) in points.iter_mut().zip(values.as_chunks::<COORDINATES>().0) {
        *point = *row;
    }
    Ok(points)
}

/// One output looked up by the name the graph declares it under.
fn output_named<'a>(
    outputs: &'a [OutputTensor],
    name: &'static str,
) -> Result<&'a OutputTensor, ModelError> {
    outputs
        .iter()
        .find(|output| output.name == name)
        .ok_or_else(|| ModelError::UnexpectedOutput {
            expected: name,
            got: outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })
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

    /// A box in frame coordinates, written as its four numbers.
    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    /// A face box on that rect, scored by YuNet rather than by Face Mesh: the
    /// two scores are different numbers and the wrapper must not confuse them.
    fn face(x: f32, y: f32, w: f32, h: f32) -> FaceBox {
        FaceBox {
            rect: rect(x, y, w, h),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        }
    }

    /// The two outputs the graph declares, carrying `score` and `points`.
    fn outputs(score: f32, points: [[i32; COORDINATES]; LANDMARKS]) -> Vec<OutputTensor> {
        vec![
            OutputTensor {
                name: SCORE_NAME.to_owned(),
                shape: vec![1, 1],
                data: TensorData::F32(vec![score]),
            },
            OutputTensor {
                name: LANDMARKS_NAME.to_owned(),
                shape: vec![1, LANDMARKS as i64, COORDINATES as i64],
                data: TensorData::I32(points.iter().flat_map(|point| *point).collect()),
            },
        ]
    }

    /// A boxed fake session returning `outputs`, plus the handle that reads back
    /// what the wrapper asked it to run.
    fn fake_session(outputs: Vec<OutputTensor>) -> (Box<dyn InferenceSession>, FakeSessionHandle) {
        let fake = FakeSession::new("facemesh", outputs);
        let handle = fake.handle();
        (Box::new(fake), handle)
    }

    /// A session returning one all-zero landmark set, for tests that only care
    /// what was sent *to* it.
    fn blank_session() -> (Box<dyn InferenceSession>, FakeSessionHandle) {
        fake_session(outputs(0.0, [[0; COORDINATES]; LANDMARKS]))
    }

    /// A 640x480 frame of one colour, and a face box well inside it.
    fn fixture() -> (Frame, FaceBox) {
        (
            frame_with(640, 480, |_, _| (0, 0, 0)),
            face(200.0, 100.0, 100.0, 80.0),
        )
    }

    #[test]
    fn the_pinned_constants_are_the_verified_graph_signature() {
        // Read out of `face_mesh_Nx3x192x192_post.onnx` (sha256
        // ae6ada9004f22be3ab6bab8cbfbc8b795f8595f69e5eea77b95cb2fc344c04eb)
        // with onnx, and confirmed by running it: `input` ['batch',3,192,192]
        // f32, four crop scalars ['batch',1] i32, and `final_landmarks`
        // ['batch',468,3] i32. These three numbers are that signature; changing
        // one silently breaks every landmark the engine produces.
        assert_eq!(CROP_MARGIN, 0.25, "the model card's margin, on each side");
        assert_eq!(INPUT_SIZE, 192, "the image input is fixed at 192x192");
        assert_eq!(LANDMARKS, 468, "468 points, and no iris points after them");
    }

    #[test]
    fn crop_rect_applies_25_percent_margin_and_clamps_to_frame() {
        // Away from any edge: a quarter of the box's own size off each side, so
        // 1.5x the box, centred on it. 25 of 100, 20 of 80.
        let centred = crop_rect(&rect(200.0, 100.0, 100.0, 80.0), 640, 480);
        assert_eq!(
            (centred.x1, centred.y1),
            (175, 80),
            "the margin comes off the origin: 200 - 25 and 100 - 20"
        );
        assert_eq!(
            (centred.width, centred.height),
            (150, 120),
            "and it is added on both sides, so the crop is 1.5x the box"
        );

        // A face in the top-left corner would grow to -25, -20.
        let top_left = crop_rect(&rect(0.0, 0.0, 100.0, 80.0), 640, 480);
        assert_eq!(
            (top_left.x1, top_left.y1),
            (0, 0),
            "the crop has to stay inside the frame"
        );
        assert_eq!(
            (top_left.width, top_left.height),
            (150, 120),
            "and it stays 1.5x the box by sliding rather than shrinking"
        );

        // ... and a face in the bottom-right corner has to give way on the high
        // side instead: 640 - 150 and 480 - 120.
        let bottom_right = crop_rect(&rect(540.0, 400.0, 100.0, 80.0), 640, 480);
        assert_eq!(
            (bottom_right.x1, bottom_right.y1),
            (490, 360),
            "the far corner clamps against the frame's far edge"
        );
        assert_eq!((bottom_right.width, bottom_right.height), (150, 120));

        // The four numbers are the four inputs the graph is handed, so the
        // rectangle has to be a real rectangle: origin plus extent is the edge
        // the decode divides by.
        assert_eq!((centred.x2(), centred.y2()), (325, 200));
    }

    #[test]
    fn crop_rect_is_never_narrower_than_the_face_box() {
        // The model card's warning: a narrow crop loses the jaw and the outer
        // brows. So wherever the box sits and whatever the frame's shape, the
        // crop still covers it — a square crop, a shrink against the frame edge
        // or an aspect-preserving fit would all fail one of these cases.
        let cases = [
            (640, 480, rect(0.0, 0.0, 100.0, 80.0)),
            (640, 480, rect(540.0, 400.0, 100.0, 80.0)),
            (640, 480, rect(590.0, 0.0, 50.0, 90.0)),
            (640, 480, rect(100.5, 40.25, 200.0, 50.0)),
            (640, 480, rect(300.0, 200.0, 40.0, 160.0)),
            (192, 192, rect(46.0, 46.0, 100.0, 100.0)),
            (120, 90, rect(0.0, 0.0, 100.0, 80.0)),
            (320, 240, rect(0.0, 0.0, 320.0, 240.0)),
        ];
        for (frame_width, frame_height, rect) in cases {
            let crop = crop_rect(&rect, frame_width, frame_height);
            assert!(
                crop.width > 0 && crop.height > 0,
                "{crop:?} must have pixels for the graph to resize"
            );
            assert!(
                crop.x1 >= 0
                    && crop.y1 >= 0
                    && crop.x2() <= frame_width as i32
                    && crop.y2() <= frame_height as i32,
                "{crop:?} leaves the {frame_width}x{frame_height} frame"
            );
            let (box_x2, box_y2) = (rect.x + rect.w, rect.y + rect.h);
            assert!(
                crop.x1 as f32 <= rect.x
                    && crop.y1 as f32 <= rect.y
                    && crop.x2() as f32 >= box_x2
                    && crop.y2() as f32 >= box_y2,
                "{crop:?} does not cover the face box {rect:?} it was grown from"
            );
            assert!(
                crop.width as f32 >= rect.w && crop.height as f32 >= rect.h,
                "{crop:?} is narrower than the {rect:?} it was grown from, which is \
                 exactly what the model card warns against"
            );
        }

        // The tight case, where the margin cannot fit at all: the crop gives way
        // down to the frame — and the frame is the box, so it is not one pixel
        // narrower than the face it came from.
        let tight = crop_rect(&rect(0.0, 0.0, 100.0, 80.0), 100, 80);
        assert_eq!(
            (tight.x1, tight.y1, tight.width, tight.height),
            (0, 0, 100, 80),
            "a 100x80 frame with a 100x80 face crops the whole frame"
        );

        // A box far wider than it is tall must not be squared off: the width
        // follows the box's own width.
        let wide = crop_rect(&rect(100.0, 100.0, 200.0, 50.0), 800, 600);
        assert_eq!(
            (wide.width, wide.height),
            (300, 75),
            "300 is 1.5 x 200, not 1.5 x the smaller side"
        );
    }

    #[test]
    fn preprocessing_is_rgb_unit_range_and_ignores_aspect_ratio() {
        // Distinct bytes per channel, so a BGR feed — YuNet's order, and the
        // order the frame itself holds — is unmistakable.
        let solid = frame_with(640, 480, |_, _| (10, 120, 250));
        let crop = crop_rect(&rect(200.0, 100.0, 100.0, 80.0), 640, 480);
        let tensor = preprocess(&solid, crop);

        assert_eq!(tensor.name, INPUT_NAME, "the graph's image input name");
        assert_eq!(
            tensor.shape,
            vec![1, 3, 192, 192],
            "NCHW, f32, at the size the graph is fixed at"
        );
        let values = tensor.data.as_f32().expect("an f32 image tensor");
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert_eq!(values.len(), 3 * plane, "the shape and the data agree");
        for index in 0..plane {
            assert!(
                (values[index] - 250.0 / 255.0).abs() < 1e-6,
                "plane 0 must be red, is {}",
                values[index]
            );
            assert!(
                (values[plane + index] - 120.0 / 255.0).abs() < 1e-6,
                "plane 1 must be green, is {}",
                values[plane + index]
            );
            assert!(
                (values[2 * plane + index] - 10.0 / 255.0).abs() < 1e-6,
                "plane 2 must be blue, is {}",
                values[2 * plane + index]
            );
        }
        assert!(
            values.iter().all(|value| (0.0..=1.0).contains(value)),
            "unit range: not 0..255, and not -1..1"
        );
        assert_eq!(
            preprocess(&solid, crop).data,
            tensor.data,
            "pure: the same frame and crop give the same tensor"
        );

        // A 2:1 crop of a frame whose red channel ramps with the column and
        // green with the row: both axes must be stretched across all 192 pixels,
        // which is what "resized without considering the aspect ratio" means. An
        // aspect-preserving fit would reach only column 191 and row 95 of the
        // crop and pad the rest.
        //
        // The crop starts at (160, 80) rather than at the frame's origin, so the
        // four numbers here also pin *which* pixels were sampled: reading the
        // frame from 0, or from the box instead of the crop, lands on different
        // ramp values.
        let ramp = frame_with(640, 480, |x, y| {
            (0, (y * 255 / 480) as u8, (x * 255 / 640) as u8)
        });
        let wide = crop_rect(&rect(200.0, 100.0, 160.0, 80.0), 640, 480);
        assert_eq!(
            (wide.x1, wide.y1, wide.width, wide.height),
            (160, 80, 240, 120),
            "a 160x80 box crops 240x120, starting 40 left of it and 20 above it"
        );
        let values = preprocess(&ramp, wide).data;
        let values = values.as_f32().expect("an f32 image tensor");
        let red = |column: usize, row: usize| values[row * INPUT_SIZE + column];
        let green = |column: usize, row: usize| values[plane + row * INPUT_SIZE + column];
        let unit = 1.0 / 255.0;
        // The crop's own pixels: column 160 is red 63 and column 399 is red 158;
        // row 80 is green 42 and row 199 is green 105. Sampling the frame from 0
        // instead of from the crop would land on red 94 and green 63.
        assert!(
            (red(0, 0) - 63.0 * unit).abs() < 2.0 * unit,
            "the first column samples the crop's left edge (x=160), is {}",
            red(0, 0) / unit
        );
        assert!(
            (red(191, 0) - 158.0 * unit).abs() < 2.0 * unit,
            "the last column samples the crop's last column (x=399), is {}",
            red(191, 0) / unit
        );
        assert!(
            (green(0, 0) - 42.0 * unit).abs() < 2.0 * unit,
            "the first row samples the crop's top edge (y=80), is {}",
            green(0, 0) / unit
        );
        assert!(
            (green(0, 191) - 105.0 * unit).abs() < 2.0 * unit,
            "the last row samples the crop's last row (y=199), is {}",
            green(0, 191) / unit
        );
    }

    #[test]
    fn landmarks_come_back_as_absolute_pixels_not_normalised() {
        // The graph's post-process is fused, so these are already frame pixels.
        // These points can only be read as pixels: they run past the 192-pixel
        // tensor, off the 640x480 frame, and negative — so normalising them
        // (x / 192), scaling them by the crop, or adding crop_x1 a second time
        // each change the answer.
        let mut points = [[0i32; COORDINATES]; LANDMARKS];
        for (index, point) in points.iter_mut().enumerate() {
            let index = index as i32;
            *point = [100 + index, 200 + index, -300 + index];
        }
        points[7] = [9999, -9999, 42];

        let (session, handle) = fake_session(outputs(7.25, points));
        let (frame, face) = fixture();
        let landmarks = FaceMesh::new(session)
            .landmarks(&frame, &face)
            .expect("two well-formed outputs are not an error");

        assert_eq!(landmarks.points[0], [100, 200, -300], "point 0 verbatim");
        assert_eq!(
            landmarks.points[7],
            [9999, -9999, 42],
            "a point outside the frame is not clamped to it"
        );
        assert_eq!(landmarks.points[100], [200, 300, -200], "and not offset");
        assert_eq!(
            landmarks.points[467],
            [567, 667, 167],
            "all 468 points, in order, to the last one"
        );
        assert_eq!(
            landmarks.score, 7.25,
            "the model's own score, not the 0.9 the face box was detected with"
        );
        assert_eq!(handle.call_count(), 1, "one face, one graph run");

        // A score the model reserves for "no face here" is still the model's
        // answer: the wrapper does not clamp it into a probability.
        let (session, _) = fake_session(outputs(-6.5, [[0; COORDINATES]; LANDMARKS]));
        let landmarks = FaceMesh::new(session).landmarks(&frame, &face).unwrap();
        assert_eq!(landmarks.score, -6.5, "reported as the graph produced it");
    }

    #[test]
    fn all_four_crop_inputs_are_passed_to_the_session() {
        // The crop is four of the graph's five inputs, so a wrapper that builds
        // a tensor and stops still gets landmarks back — in the wrong place.
        let (session, handle) = blank_session();
        let (frame, face) = fixture();
        FaceMesh::new(session).landmarks(&frame, &face).unwrap();

        let inputs = handle.last_inputs().expect("the graph ran");
        assert_eq!(
            inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                INPUT_NAME,
                CROP_X1_NAME,
                CROP_Y1_NAME,
                CROP_WIDTH_NAME,
                CROP_HEIGHT_NAME
            ],
            "all five inputs, in the order the graph declares them"
        );
        assert_eq!(inputs[0].shape, vec![1, 3, 192, 192]);
        assert_eq!(
            inputs[0].data.as_f32().map(<[f32]>::len),
            Some(3 * INPUT_SIZE * INPUT_SIZE),
            "the image tensor is f32 and the shape and data agree"
        );

        // The four scalars the decode divides by, as integers: a float tensor
        // here is a graph that will not run, and the wrong numbers are
        // landmarks in the wrong place.
        let value_of = |name: &str| {
            let input = inputs
                .iter()
                .find(|input| input.name == name)
                .unwrap_or_else(|| panic!("no input named {name}"));
            assert_eq!(input.shape, vec![1, 1], "{name} is [batch, 1]");
            let values = input
                .data
                .as_i32()
                .unwrap_or_else(|| panic!("{name} must be i32, not f32"));
            assert_eq!(values.len(), 1, "{name} carries one value");
            values[0]
        };
        assert_eq!(value_of(CROP_X1_NAME), 175, "200 - 25");
        assert_eq!(value_of(CROP_Y1_NAME), 80, "100 - 20");
        assert_eq!(value_of(CROP_WIDTH_NAME), 150, "100 x 1.5");
        assert_eq!(value_of(CROP_HEIGHT_NAME), 120, "80 x 1.5");
    }

    #[test]
    fn landmarks_reject_outputs_the_graph_could_not_have_produced() {
        let (frame, face) = fixture();
        let run = |outputs: Vec<OutputTensor>| {
            let (session, _) = fake_session(outputs);
            FaceMesh::new(session).landmarks(&frame, &face)
        };
        let well_formed = outputs(1.0, [[0; COORDINATES]; LANDMARKS]);

        // "The model did not answer" is not "the model found nothing": a
        // session returning the wrong number of tensors is an error.
        assert!(matches!(
            run(Vec::new()),
            Err(ModelError::UnexpectedOutputCount {
                expected: 2,
                got: 0
            })
        ));
        assert!(matches!(
            run(vec![well_formed[0].clone()]),
            Err(ModelError::UnexpectedOutputCount {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            run(vec![
                well_formed[0].clone(),
                well_formed[1].clone(),
                well_formed[1].clone()
            ]),
            Err(ModelError::UnexpectedOutputCount {
                expected: 2,
                got: 3
            })
        ));

        // A graph that emitted a different point count, or a truncated buffer
        // behind a correct shape, would misread every point after the gap.
        let mut short_shape = well_formed.clone();
        short_shape[1].shape = vec![1, 467, 3];
        short_shape[1].data = TensorData::I32(vec![0; 467 * 3]);
        assert!(matches!(
            run(short_shape),
            Err(ModelError::UnexpectedShape {
                name: "final_landmarks",
                ..
            })
        ));

        let mut short_data = well_formed.clone();
        short_data[1].data = TensorData::I32(vec![0; LANDMARKS * COORDINATES - 1]);
        assert!(
            matches!(
                run(short_data),
                Err(ModelError::UnexpectedDataLength {
                    name: "final_landmarks",
                    ..
                })
            ),
            "a buffer that does not fill its own shape must be an error, not a panic"
        );

        // And the element types are part of the signature: `final_landmarks` is
        // i32 and `score` is f32, in that order.
        let mut float_landmarks = well_formed.clone();
        float_landmarks[1].data = TensorData::F32(vec![0.0; LANDMARKS * COORDINATES]);
        assert!(matches!(
            run(float_landmarks),
            Err(ModelError::MissingOutput {
                name: "final_landmarks"
            })
        ));

        let mut integer_score = well_formed.clone();
        integer_score[0].data = TensorData::I32(vec![1]);
        assert!(matches!(
            run(integer_score),
            Err(ModelError::MissingOutput { name: "score" })
        ));

        let mut wide_score = well_formed;
        wide_score[0].shape = vec![1, 2];
        assert!(matches!(
            run(wide_score),
            Err(ModelError::UnexpectedShape { name: "score", .. })
        ));

        // Tensors under names this graph does not declare: YuNet's heads, say.
        let (session, _) = fake_session(vec![
            OutputTensor {
                name: "cls_8".to_owned(),
                shape: vec![1, 6400, 1],
                data: TensorData::F32(vec![0.0; 6400]),
            },
            OutputTensor {
                name: "bbox_8".to_owned(),
                shape: vec![1, 6400, 4],
                data: TensorData::F32(vec![0.0; 6400 * 4]),
            },
        ]);
        assert!(matches!(
            FaceMesh::new(session).landmarks(&frame, &face),
            Err(ModelError::UnexpectedOutput { .. })
        ));
    }

    #[test]
    fn a_frame_with_no_pixels_is_not_read_past_the_end() {
        // `Frame::new(0, 0, vec![])` is constructible, and a crop of nothing
        // must not index an empty buffer: the graph still gets its 192x192
        // tensor, and the wrapper still answers with what the session said.
        let empty = Frame::new(0, 0, Vec::new(), Instant::now()).expect("a zero-area frame");
        let crop = crop_rect(&rect(0.0, 0.0, 0.0, 0.0), 0, 0);
        assert_eq!(
            (crop.x1, crop.y1, crop.width, crop.height),
            (0, 0, 0, 0),
            "a frame with no pixels has no crop"
        );
        let values = preprocess(&empty, crop).data;
        let values = values.as_f32().expect("an f32 image tensor");
        assert_eq!(values.len(), 3 * INPUT_SIZE * INPUT_SIZE);
        assert!(
            values.iter().all(|value| *value == 0.0),
            "a frame with no pixels is black, not a panic"
        );

        let (session, _) = blank_session();
        let mesh = FaceMesh::new(session);
        assert!(mesh.landmarks(&empty, &face(0.0, 0.0, 0.0, 0.0)).is_ok());

        // The same for a box with no area in a frame that has pixels: the crop
        // shrinks to one pixel rather than to nothing, because the graph cannot
        // resize a crop with no pixels in it.
        let crop = crop_rect(&rect(10.0, 20.0, 0.0, 0.0), 640, 480);
        assert_eq!((crop.width, crop.height), (1, 1), "one pixel, not zero");
        let values = preprocess(&frame_with(640, 480, |_, _| (1, 2, 3)), crop).data;
        let values = values.as_f32().expect("an f32 image tensor");
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}
