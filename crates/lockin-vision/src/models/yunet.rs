//! YuNet face detection: the pipeline's stage 1.
//!
//! YuNet is anchor-free. Its graph emits one row per cell of three feature
//! grids — strides 8, 16 and 32 over the input — and each row holds a score,
//! four box parameters and five facial landmarks, all as offsets from that
//! cell's anchor. There is no separate anchor table to load; the grid *is* the
//! anchor set. On a 320x240 input that is 40*30 + 20*15 + 10*8 = 1580 rows.
//!
//! Preprocessing is this model's own regime and is deliberately not shared with
//! the other models: BGR passed through, values scaled by `1/255`, the frame
//! letterboxed to 320x240 with the aspect ratio preserved — so no face is
//! stretched — and the padding filled with zeros, which is the only value the
//! graph saw during training as "nothing here".
//!
//! Decode is the exact inverse of that letterbox, which is why
//! [`LetterboxGeometry`] carries the scale and offset rather than recomputing
//! them from a frame: the two must agree bit for bit or every box drifts by a
//! fraction of a pixel per frame.

use super::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};
use crate::analyze::nms::{Detection, Rect, nms};
use crate::frame::Frame;

/// Input width the exported model is fixed at, in pixels.
pub const TENSOR_WIDTH: usize = 320;

/// Input height the exported model is fixed at, in pixels.
pub const TENSOR_HEIGHT: usize = 240;

/// The three feature-grid strides, coarsest last.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Values per output row: a score, four box parameters, then five `(x, y)`
/// landmark pairs — `1 + 4 + 2 * LANDMARKS`.
///
/// The model file this decodes is the single fused tensor, not the twelve
/// separate `cls_*`/`obj_*`/`bbox_*`/`kps_*` outputs that the training export
/// also produces: one row per anchor, everything the anchor predicts in it.
const COLUMNS: usize = 1 + 4 + 2 * LANDMARKS;

/// Landmarks per face: right eye, left eye, nose tip, right mouth corner, left
/// mouth corner — YuNet's own order, and Face Mesh's stage-2 input.
const LANDMARKS: usize = 5;

/// The name of the graph's image input, as the exported model declares it.
const INPUT_NAME: &str = "input";

/// The name of the graph's fused output.
const OUTPUT_NAME: &str = "output1";

/// The output shape this decode reads: one row per anchor cell.
const EXPECTED_SHAPE: [i64; 3] = [1, 1580, COLUMNS as i64];

/// Raw scores at or above this are faces.
///
/// `0.6` is the value the model card and OpenCV's own demo use. Applying it to
/// the *sigmoid* of the raw score matters: a session that returns zeros for a
/// blank frame means "0.5, no face", not "1.0, certain face".
const SCORE_THRESHOLD: f32 = 0.6;

/// Boxes overlapping a kept box by more than this are the same face.
const NMS_IOU_THRESHOLD: f32 = 0.3;

/// The value `1/255` scales BGR bytes into the unit range with.
const UNIT_RANGE: f32 = 1.0 / 255.0;

/// The range a cell's raw parameters are clamped to before they are used.
///
/// Only a guard against a garbage tensor: this is what keeps `exp` out of the
/// range where it overflows to infinity and a NaN out of every downstream
/// rect. The bound is deliberately loose, because it is not this model's job to
/// decide where a face may be — the frame clamp in [`decode`] is, and a tight
/// number here would silently take that decision away from it.
const RAW_LIMIT: f32 = 30.0;

/// One detected face: where it is, how sure YuNet is, and its five landmarks in
/// frame pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBox {
    /// The face's box in frame coordinates.
    pub rect: Rect,
    /// Detector confidence in `0.0..=1.0`; higher is stronger.
    pub score: f32,
    /// Right eye, left eye, nose tip, right mouth corner, left mouth corner —
    /// each `(x, y)` in frame pixels, so stage 2 can be handed them directly.
    pub landmarks: [(f32, f32); LANDMARKS],
}

/// How a frame was mapped onto the tensor, and therefore how to map back.
///
/// Produced by [`preprocess`] and consumed by [`decode`] so the forward and
/// inverse transforms are provably the same arithmetic in two directions,
/// rather than two independent derivations that can disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxGeometry {
    /// Scale factor applied to the frame, so that `scale * width <= 320` and
    /// `scale * height <= 240` with the tighter axis exact. Above 1.0 for a
    /// frame smaller than the input, below it for a larger one.
    pub scale: f32,
    /// Left offset of the scaled frame inside the tensor, in pixels.
    pub left: usize,
    /// Top offset of the scaled frame inside the tensor, in pixels.
    pub top: usize,
    /// Frame width in pixels, as passed to [`preprocess`].
    pub frame_width: u32,
    /// Frame height in pixels, as passed to [`preprocess`].
    pub frame_height: u32,
}

impl LetterboxGeometry {
    /// The frame's height in the tensor, in tensor pixels.
    ///
    /// Only the height is needed: the scale is chosen so the tighter axis lands
    /// exactly on its bound, and a scale below 1.0 is only ever chosen because
    /// one axis needs it, so the other axis is the one that fills the tensor.
    /// The width is therefore always `TENSOR_WIDTH`.
    fn content_height(self) -> usize {
        let scaled = self.frame_height as f32 * self.scale;
        // Rounded, not truncated: the scaled size is the nearest whole number
        // of tensor pixels, and truncating would drop a row that rounding up of
        // a different frame kept.
        (scaled.round() as usize).min(TENSOR_HEIGHT)
    }

    /// The tensor pixel `(x, y)` a frame pixel `(x, y)` is sampled at.
    ///
    /// The exact inverse of [`to_frame`](Self::to_frame). It does **not** carry
    /// the `- 0.5` half-pixel convention that `to_frame` does: `to_frame`
    /// converts a tensor coordinate to a frame *position*, and a position minus
    /// half is the pixel centre, whereas this converts a frame pixel index to a
    /// tensor coordinate directly — sampling a pixel's centre means sampling
    /// its own coordinate. Getting that wrong puts every sample half a pixel
    /// out, which the round-trip assertion in the preprocessing test catches.
    fn to_tensor(self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.scale + self.left as f32,
            y * self.scale + self.top as f32,
        )
    }

    /// Maps a tensor-space `(x, y)` back to frame pixels.
    ///
    /// The padding is subtracted before the scale is divided out, so the result
    /// is where the pixel would have been in the original frame. A coordinate
    /// in the padding region maps outside the frame, which is expected: callers
    /// clamp.
    fn to_frame(self, x: f32, y: f32) -> (f32, f32) {
        // Guarded rather than trusted: `preprocess` only produces a zero scale
        // for a frame that is empty along an axis, which has no pixels to
        // decode anyway, but a division by zero here would put a NaN into a
        // rect and from there into every later stage.
        let inverse_scale = if self.scale > 0.0 {
            1.0 / self.scale
        } else {
            return (0.0, 0.0);
        };
        (
            (x - self.left as f32) * inverse_scale,
            (y - self.top as f32) * inverse_scale,
        )
    }
}

/// Builds the letterboxed NCHW tensor YuNet expects from `frame`.
///
/// Returns the input tensor and the geometry that inverts it. Pure: same frame
/// in, same tensor out, with no clock or randomness anywhere.
///
/// BGR is passed through in the order the frame already holds it, values are
/// scaled by `1/255`, and the padding is zero. Sampling is bilinear rather than
/// nearest-neighbour because a face at the far end of a 640-pixel row is under
/// two tensor pixels wide, and nearest-neighbour turns it into a different face
/// from frame to frame as the subject moves a few centimetres.
pub fn preprocess(frame: &Frame) -> (InputTensor, LetterboxGeometry) {
    let geometry = letterbox_geometry(frame.width, frame.height);

    // The content is centred and the frame fills the tensor's width, so the
    // rows to write are the content band; the zero fill left the padding around
    // it already zero.
    let content = geometry.top..geometry.top + geometry.content_height();

    // The whole tensor at once, rather than three channel buffers to copy in
    // afterwards: at 5 Hz this runs 200 times a minute and the copy would be
    // the largest cost in the function.
    let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
    let mut data = vec![0.0f32; plane * 3];

    for y in content {
        for x in 0..TENSOR_WIDTH {
            // The same map `decode` inverts, rather than a second derivation of
            // it — a `preprocess` and a `decode` that disagree by half a pixel
            // put every box half a pixel out, which is invisible in review and
            // visible on screen.
            let (source_x, source_y) = geometry.to_tensor(x as f32, y as f32);
            let (b, g, r) = sample_bilinear(frame, source_x, source_y);
            let index = y * TENSOR_WIDTH + x;
            data[index] = b * UNIT_RANGE;
            data[plane + index] = g * UNIT_RANGE;
            data[2 * plane + index] = r * UNIT_RANGE;
        }
    }

    let tensor = InputTensor {
        name: INPUT_NAME.to_owned(),
        shape: vec![1, 3, TENSOR_HEIGHT as i64, TENSOR_WIDTH as i64],
        data: TensorData::F32(data),
    };
    (tensor, geometry)
}

/// The scale and padding a `width` x `height` frame is letterboxed with.
///
/// `scale` is a **multiplier**: `scale * width <= 320` and `scale * height <=
/// 240` always hold, with the tighter axis landing exactly on its bound and the
/// other one centred. A multiplier is `1 / max(width/320, height/240)` — taking
/// the smaller of `320/width` and `240/height` instead, which is the same
/// numbers in the wrong direction, scales a small frame *up* until it falls off
/// the tensor, and the round-trip assertion in the preprocessing test is what
/// catches that.
fn letterbox_geometry(width: u32, height: u32) -> LetterboxGeometry {
    let (width, height) = (width as f32, height as f32);
    let scale = if width > 0.0 && height > 0.0 {
        let shrink = (width / TENSOR_WIDTH as f32).max(height / TENSOR_HEIGHT as f32);
        1.0 / shrink
    } else {
        // A frame with no pixels has no aspect ratio to preserve; scaling it by
        // zero leaves an all-zero tensor, which is the honest answer.
        0.0
    };
    // The scaled frame cannot exceed the tensor, so both subtractions are
    // non-negative and the casts saturate rather than wrapping.
    let left = (TENSOR_WIDTH as f32 - width * scale) as usize / 2;
    let top = (TENSOR_HEIGHT as f32 - height * scale) as usize / 2;
    LetterboxGeometry {
        scale,
        left,
        top,
        frame_width: width as u32,
        frame_height: height as u32,
    }
}

/// Bilinear BGR sample at a floating-point frame coordinate.
///
/// Coordinates outside the frame clamp to the edge pixel, which is what makes
/// the half-pixel offsets at the borders resolve to real pixels instead of
/// reading past the buffer.
fn sample_bilinear(frame: &Frame, x: f32, y: f32) -> (f32, f32, f32) {
    let (width, height) = (frame.width as usize, frame.height as usize);

    // Clamped to `dimension - 1`, not to `dimension`: this is a pixel *index*,
    // and `width` is one past the last one.
    let x0 = (x.floor().max(0.0) as usize).min(width - 1);
    let y0 = (y.floor().max(0.0) as usize).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);

    let bgra = &frame.bgra;
    let index = |x: usize, y: usize| (y * width + x) * 4;
    let (top_left, top_right) = (index(x0, y0), index(x1, y0));
    let (bottom_left, bottom_right) = (index(x0, y1), index(x1, y1));

    let channel = |offset: usize| {
        let top = f32::from(bgra[top_left + offset]) * (1.0 - fx)
            + f32::from(bgra[top_right + offset]) * fx;
        let bottom = f32::from(bgra[bottom_left + offset]) * (1.0 - fx)
            + f32::from(bgra[bottom_right + offset]) * fx;
        top * (1.0 - fy) + bottom * fy
    };

    // BGRA in memory is blue, green, red, alpha; the tensor wants BGR.
    (channel(0), channel(1), channel(2))
}

/// Decodes a YuNet output tensor into faces, in frame coordinates.
///
/// `geometry` must be the one [`preprocess`] produced for the frame the tensor
/// came from, or every box is off by the letterbox.
///
/// The output's score is a raw logit and its box parameters are raw offsets
/// from the anchor, so both are activated here: `sigmoid` for the score,
/// `exp` for the width and height. Boxes are mapped back out of the letterbox —
/// the four parameters are pixel distances and so scale without the padding
/// offset, while the centre and the landmarks are positions and take both — and
/// clamped to the frame, because a box that runs off the edge would hand stage 2
/// a crop outside the image.
pub fn decode(
    output: &OutputTensor,
    geometry: &LetterboxGeometry,
) -> Result<Vec<FaceBox>, ModelError> {
    check_output(output)?;
    let data = output
        .data
        .as_f32()
        .ok_or(ModelError::MissingOutput { name: OUTPUT_NAME })?;

    let (frame_width, frame_height) = (geometry.frame_width as f32, geometry.frame_height as f32);
    let mut candidates = Vec::new();

    // Rows arrive grid by grid, coarsest last, and within a grid row by row.
    // `first_row` is where the grid being walked starts in the tensor, which is
    // what turns a cell's position in its own grid into a row of the tensor.
    let mut first_row = 0usize;
    for stride in STRIDES {
        let stride = stride as f32;
        // Ceiling division: a stride does not have to divide the input exactly.
        // The stride-32 grid over 240 rows is 8 rows, not 7 — the last one
        // covers the 16 rows left over — and truncating here leaves ten anchors
        // unread, which is a strip 32 pixels tall across the bottom of every
        // frame where nobody is ever detected. The assertion after the loop is
        // what catches it.
        let columns = TENSOR_WIDTH.div_ceil(stride as usize);
        let rows = TENSOR_HEIGHT.div_ceil(stride as usize);

        for grid_y in 0..rows {
            // Walked as a flat offset rather than `row * COLUMNS` per cell: the
            // multiplication per anchor is the inner loop's whole cost, and at
            // 1580 anchors a frame this runs at 5 Hz.
            let mut offset = (first_row + grid_y * columns) * COLUMNS;
            for grid_x in 0..columns {
                let cell = &data[offset..offset + COLUMNS];
                let score = sigmoid(cell[0].clamp(-RAW_LIMIT, RAW_LIMIT));
                if score >= SCORE_THRESHOLD {
                    candidates.push(decode_cell(
                        cell,
                        score,
                        grid_x,
                        grid_y,
                        stride,
                        (frame_width, frame_height),
                        geometry,
                    ));
                }
                offset += COLUMNS;
            }
        }
        first_row += rows * columns;
    }
    // Every row of the tensor was walked exactly once. A stride set that does
    // not tile the input — the thing this guards — would leave cells unread and
    // silently drop the faces in them.
    debug_assert_eq!(first_row, EXPECTED_SHAPE[1] as usize);

    // Every box is a face, so every detection is class 0 and suppression
    // compares like with like. Encoding a candidate's index in `class_id`
    // instead would make every detection its own class and suppress *nothing* —
    // which is the one mistake this decode cannot afford, since a face would
    // then be counted once per overlapping anchor.
    let mut detections: Vec<Detection> = candidates
        .iter()
        .map(|face| Detection {
            rect: face.rect,
            score: face.score,
            class_id: 0,
        })
        .collect();
    // The score gate was applied per cell, so suppression must not apply one
    // again: a 0.0 threshold keeps everything that reached here.
    nms(&mut detections, NMS_IOU_THRESHOLD, 0.0);

    // `nms` reorders what it keeps, so the faces are recovered by score rather
    // than by position. Both loops run over the handful of boxes that cleared
    // the score gate, not over the 1580 anchors.
    let mut survived = vec![false; candidates.len()];
    for detection in &detections {
        let index = candidates
            .iter()
            .position(|face| face.score == detection.score && face.rect == detection.rect)
            .expect("nms only ever returns detections it was given");
        survived[index] = true;
    }
    let mut faces: Vec<FaceBox> = candidates
        .into_iter()
        .zip(survived)
        .filter_map(|(face, survived)| survived.then_some(face))
        .collect();
    // Strongest first, matching `nms`'s own output order — recomputed rather
    // than inferred, because a stable sort is what keeps it well defined.
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(faces)
}

/// Validates the output's name, shape and element type.
fn check_output(output: &OutputTensor) -> Result<(), ModelError> {
    if output.name != OUTPUT_NAME {
        return Err(ModelError::UnexpectedOutput {
            expected: OUTPUT_NAME,
            got: output.name.clone(),
        });
    }
    // Both the row count and the column count are checked: a graph exported
    // with a different stride set would give the right number of columns and
    // the wrong number of rows, and reading it would silently mis-index every
    // anchor after the first grid.
    if output.shape != EXPECTED_SHAPE {
        return Err(ModelError::UnexpectedShape {
            name: OUTPUT_NAME,
            expected: EXPECTED_SHAPE.to_vec(),
            got: output.shape.clone(),
        });
    }
    Ok(())
}

/// Decodes one anchor cell into a face.
fn decode_cell(
    cell: &[f32],
    score: f32,
    grid_x: usize,
    grid_y: usize,
    stride: f32,
    frame: (f32, f32),
    geometry: &LetterboxGeometry,
) -> FaceBox {
    // The anchor sits at the cell's top-left corner, and the raw offsets are
    // multiples of the stride from there — so the centre is `(grid + delta)`
    // scaled, while the size is `exp(delta)` scaled, since a size cannot be
    // negative.
    let center_x = (grid_x as f32 + cell[1].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
    let center_y = (grid_y as f32 + cell[2].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
    let tensor_width = cell[3].clamp(-RAW_LIMIT, RAW_LIMIT).exp() * stride;
    let tensor_height = cell[4].clamp(-RAW_LIMIT, RAW_LIMIT).exp() * stride;

    let half_width = tensor_width / 2.0;
    let half_height = tensor_height / 2.0;

    // Clamped in tensor space first so a garbage tensor cannot produce a rect
    // wider than the image it came from, then mapped to the frame, where the
    // padding is removed and the scale divided out. Sizes take the scale only:
    // the padding moves where a face is, never how big it is.
    let tensor_right = TENSOR_WIDTH as f32;
    let tensor_bottom = TENSOR_HEIGHT as f32;
    let left = (center_x - half_width).clamp(0.0, tensor_right);
    let top = (center_y - half_height).clamp(0.0, tensor_bottom);
    let right = (center_x + half_width).clamp(0.0, tensor_right);
    let bottom = (center_y + half_height).clamp(0.0, tensor_bottom);

    let (frame_left, frame_top) = geometry.to_frame(left, top);
    let (frame_right, frame_bottom) = geometry.to_frame(right, bottom);
    // The tensor clamp above bounds the box's *origin*, and that is enough for
    // `x` and `y`: the content fills the tensor's width exactly and its height
    // up to `top + content_height`, so a tensor-space origin is never above or
    // left of the content, and the inverse map therefore puts it in the frame.
    // Clamping those two again here guards against a state the geometry cannot
    // produce — established by deleting each guard and watching the suite still
    // pass, and by mutating each one and watching the mutation survive. That is
    // why there is no clamp on `y`.
    //
    // The bottom is different and does need its clamp: a box can end inside the
    // tensor and still project past the frame's last row, because the letterbox
    // scales the frame's height by more than the tensor holds. A 4x2 frame
    // against a 320x240 input is scaled by 80, so a tensor box reaching its
    // bottom edge maps to frame y = 2.31 in a 2-pixel-tall frame — and the rect
    // would claim a height running off the image.
    let (frame_width, frame_height) = frame;
    let x = frame_left;
    let y = frame_top;
    let rect = Rect {
        x,
        y,
        w: frame_right - x,
        h: frame_bottom.clamp(0.0, frame_height) - y,
    };

    let mut landmarks = [(0.0f32, 0.0f32); LANDMARKS];
    for (index, landmark) in landmarks.iter_mut().enumerate() {
        let raw_x = (grid_x as f32 + cell[5 + 2 * index].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
        let raw_y = (grid_y as f32 + cell[6 + 2 * index].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
        // A landmark is a bare point with no box around it, and unlike the box
        // its origin *can* land in the padding: a cell at the tensor's edge
        // carries one further out still, past the content and outside the
        // frame. The box's clamp cannot help it, so this is the only thing
        // between a landmark and a coordinate that would have Face Mesh
        // cropping outside the image.
        let (x, y) = geometry.to_frame(raw_x, raw_y);
        *landmark = (x.clamp(0.0, frame_width), y.clamp(0.0, frame_height));
    }

    FaceBox {
        rect,
        score,
        landmarks,
    }
}

/// The logistic function, saturating at the extremes.
///
/// A raw score well past `+30` is certain and one well past `-30` is noise, so
/// clamping there costs nothing and keeps `exp` out of the range where it
/// overflows to infinity — a garbage tensor must not put a NaN into a score.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// YuNet face detection over a session.
pub struct YuNet {
    session: Box<dyn InferenceSession>,
}

impl YuNet {
    /// Wraps a session. The session is the model; this type is the arithmetic
    /// around it.
    pub fn new(session: Box<dyn InferenceSession>) -> YuNet {
        YuNet { session }
    }

    /// The wrapped session's name, for logs and for the provider autotune.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Detects faces in `frame`, strongest first.
    ///
    /// An empty vector means "no faces", not "failed" — a blank frame is a
    /// normal frame, and the engine's short-circuit depends on it being cheap
    /// and unremarkable.
    pub fn detect(&self, frame: &Frame) -> Result<Vec<FaceBox>, ModelError> {
        let (input, geometry) = preprocess(frame);
        let mut outputs = self.session.run(&[input])?;
        // Taken rather than borrowed so the tensors are not held for the whole
        // decode; there is exactly one by contract.
        let output = outputs
            .drain(..)
            .next()
            .ok_or(ModelError::MissingOutput { name: OUTPUT_NAME })?;
        decode(&output, &geometry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FakeSession, FakeSessionHandle};
    use std::time::Instant;

    /// Row count of a 320x240 output: 40*30 cells at stride 8, 20*15 at 16 and
    /// 10*8 at 32.
    const ROWS: usize = 1580;

    /// An output tensor of zeroes, the shape a session returns for a blank
    /// frame.
    fn zero_output() -> OutputTensor {
        OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: EXPECTED_SHAPE.to_vec(),
            data: TensorData::F32(vec![0.0; ROWS * COLUMNS]),
        }
    }

    /// A session that always predicts nothing, plus the handle that records
    /// what it was asked to run.
    fn blank_session() -> (Box<dyn InferenceSession>, FakeSessionHandle) {
        let session = FakeSession::single("yunet", zero_output());
        let handle = session.handle();
        (Box::new(session), handle)
    }

    /// A frame of `width` x `height` with every pixel `(b, g, r)`.
    fn solid_frame(width: u32, height: u32, bgr: [u8; 3]) -> Frame {
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            bgra.extend_from_slice(&[bgr[0], bgr[1], bgr[2], 0xff]);
        }
        Frame::new(width, height, bgra, Instant::now()).expect("the buffer matches the dimensions")
    }

    /// A frame of `width` x `height` whose pixel `(x, y)` is `[x, y, 0, 0xff]`
    /// — blue carries the column, green the row. Channels that differ per axis
    /// are what make a transposed or swapped tensor visible.
    fn coordinate_frame(width: u32, height: u32) -> Frame {
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                bgra.extend_from_slice(&[x as u8, y as u8, 0, 0xff]);
            }
        }
        Frame::new(width, height, bgra, Instant::now()).expect("the buffer matches the dimensions")
    }

    /// The value a raw score of `raw` decodes to. Used only to build the
    /// failure messages, so a wrong threshold prints both sides of the
    /// comparison rather than just the raw number.
    fn decoded_score(raw: f32) -> f32 {
        1.0 / (1.0 + (-raw).exp())
    }

    /// The tensor row an anchor cell's values begin at.
    ///
    /// Recomputes the decode's own layout — the three grids laid end to end,
    /// rows before columns — rather than hardcoding 1580/3 or a stride-8
    /// offset. A test that hardcoded the layout would stop testing the layout.
    fn row_of(grid_x: usize, grid_y: usize, stride: usize) -> usize {
        let mut first_row = 0;
        for earlier in STRIDES {
            if earlier == stride {
                break;
            }
            first_row += TENSOR_WIDTH.div_ceil(earlier) * TENSOR_HEIGHT.div_ceil(earlier);
        }
        first_row + grid_y * TENSOR_WIDTH.div_ceil(stride) + grid_x
    }

    /// Writes one anchor cell's raw values into `output`.
    fn set_cell(
        output: &mut OutputTensor,
        grid_x: usize,
        grid_y: usize,
        stride: usize,
        raw: [f32; COLUMNS],
    ) {
        let anchor = row_of(grid_x, grid_y, stride);
        let row = &mut output.data.as_f32_mut().expect("f32 output")[anchor * COLUMNS..][..COLUMNS];
        row.copy_from_slice(&raw);
    }

    /// The raw score that decodes to `probability`, so a test can say "0.9"
    /// rather than carry a logit it did not derive.
    fn raw_score(probability: f32) -> f32 {
        (probability / (1.0 - probability)).ln()
    }

    /// The raw width or height that decodes to `size` tensor pixels at `stride`.
    fn raw_size(size: f32, stride: f32) -> f32 {
        (size / stride).ln()
    }

    /// A cell's 15 raw values: a score, `dx`, `dy`, `dw`, `dh`, then five
    /// landmark `(x, y)` pairs. Building them here rather than repeating a
    /// fifteen-element literal per case is what keeps a test's intent visible
    /// instead of buried in zeros.
    fn raw_cell(
        score: f32,
        dx: f32,
        dy: f32,
        dw: f32,
        dh: f32,
        landmarks: [(f32, f32); LANDMARKS],
    ) -> [f32; COLUMNS] {
        let mut cell = [0.0f32; COLUMNS];
        cell[0] = score;
        cell[1] = dx;
        cell[2] = dy;
        cell[3] = dw;
        cell[4] = dh;
        for (index, (x, y)) in landmarks.iter().enumerate() {
            cell[5 + 2 * index] = *x;
            cell[6 + 2 * index] = *y;
        }
        cell
    }

    #[test]
    fn preprocessing_scales_to_unit_range_and_preserves_aspect() {
        // A 4x2 frame: exactly twice as wide as it is tall, in a tensor that is
        // 4:3. Preserving the aspect means 320x160 of content with 40 blank
        // rows above and below — a stretch would have filled all 240.
        let frame = coordinate_frame(4, 2);
        let (tensor, geometry) = preprocess(&frame);

        assert_eq!(tensor.name, "input");
        assert_eq!(
            tensor.shape,
            vec![1, 3, 240, 320],
            "NCHW at the exported model's fixed input size"
        );
        let data = tensor.data.as_f32().expect("an f32 tensor");
        assert_eq!(data.len(), 3 * TENSOR_WIDTH * TENSOR_HEIGHT);

        // The geometry the decode inverts must be the one preprocessing used:
        // this frame scales up by 80, which fills the tensor's width exactly
        // and leaves 40 rows above and below.
        assert_eq!(geometry.scale, 80.0f32);
        assert_eq!(geometry.left, 0, "the width fills the tensor exactly");
        assert_eq!(geometry.top, 40, "160 rows of content leave 40 either side");
        assert_eq!((geometry.frame_width, geometry.frame_height), (4, 2));

        let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
        let at = |channel: usize, x: usize, y: usize| data[channel * plane + y * TENSOR_WIDTH + x];

        // The padding is zero, in every plane. These rows are the only ones
        // outside the content, and a non-zero pad value is exactly what would
        // make the graph see a false edge at the letterbox boundary.
        for row in [0, 39, 200, 239] {
            for x in [0, 1, 160, 318, 319] {
                assert_eq!(at(0, x, row), 0.0, "blue padding at ({x}, {row})");
                assert_eq!(at(1, x, row), 0.0, "green padding at ({x}, {row})");
                assert_eq!(at(2, x, row), 0.0, "red padding at ({x}, {row})");
            }
        }

        // Scaled by 1/255, from the source pixel the sample lands in.
        //
        // Tensor row 160 is the frame's *first* row's centre, so it samples
        // frame (3, 1) — this frame's blue carries its column and its red is
        // unused, so the pair (blue 3, green 1) is reachable only from frame
        // column 3, row 1. That is the co-ordinates talking, not the channel
        // order: the planes are checked separately against a solid frame below.
        //
        // Compared with a tolerance rather than exactly: `1/255` is not
        // representable in binary, so `3.0 * (1/255)` and `3.0 / 255` differ in
        // the last bit. The tolerance is a thousandth of a grey level.
        let channel = |got: f32, want: u8, what: &str| {
            assert!(
                (got - f32::from(want) / 255.0).abs() < 1e-6,
                "{what}: want {want}/255, got {got}"
            );
        };
        channel(at(0, 319, 160), 3, "frame (3, 1) blue is its column");
        channel(at(1, 319, 160), 1, "frame (3, 1) green is its row");
        channel(at(2, 319, 160), 0, "frame (3, 1) red is unused");
        // Tensor row 40 is the frame's row 0's centre: `to_tensor` puts frame
        // row 0 at tensor row 40, so its green is 0. A content band starting one
        // row late would leave this zero instead, and one starting a row early
        // would put content in the padding.
        channel(at(0, 319, 40), 3, "the content starts at row 40");
        assert_eq!(at(0, 319, 39), 0.0, "row 39 is padding");
        // The last content row is 199, so 200 is padding again. This is the
        // assertion a content band of the wrong height fails.
        channel(at(1, 319, 199), 1, "the content's last row is 199");
        assert_eq!(at(0, 319, 200), 0.0, "and row 200 is padding");

        // A solid frame's planes are in the documented order: this asserts the
        // *channel* identity that the coordinate frame above cannot, because
        // only a frame whose channels differ can tell blue from red. Tensor row
        // 80 is the centre of frame row 1.
        let (solid, _) = preprocess(&solid_frame(4, 2, [10, 20, 30]));
        let solid = solid.data.as_f32().expect("an f32 tensor");
        let row = 80 * TENSOR_WIDTH;
        channel(solid[row], 10, "plane 0 is blue");
        channel(solid[plane + row], 20, "plane 1 is green");
        channel(solid[2 * plane + row], 30, "plane 2 is red");

        // The geometry round-trips: `to_frame` is the exact inverse of the
        // `to_tensor` preprocessing samples with, so if the two ever disagree
        // every box in the frame is displaced by the difference. Both the
        // frame's corners and its centre are checked, which is what catches a
        // half-pixel error as well as a whole one.
        for (frame_point, tensor_wanted) in [
            ((0.0, 0.0), (0.0, 40.0)),
            ((3.0, 1.0), (240.0, 120.0)),
            ((1.5, 0.5), (120.0, 80.0)),
        ] {
            let (x, y) = geometry.to_tensor(frame_point.0, frame_point.1);
            assert!(
                (x - tensor_wanted.0).abs() < 1e-3 && (y - tensor_wanted.1).abs() < 1e-3,
                "frame {frame_point:?} is tensor {tensor_wanted:?}, got ({x}, {y})"
            );
            let (back_x, back_y) = geometry.to_frame(x, y);
            assert!(
                (back_x - frame_point.0).abs() < 1e-3 && (back_y - frame_point.1).abs() < 1e-3,
                "tensor ({x}, {y}) must map back to {frame_point:?}, got ({back_x}, {back_y})"
            );
        }
    }

    #[test]
    fn preprocessing_is_deterministic() {
        let frame = coordinate_frame(17, 11);
        let (first, first_geometry) = preprocess(&frame);
        let (second, second_geometry) = preprocess(&frame);
        assert_eq!(first, second, "the same frame must build the same tensor");
        assert_eq!(first_geometry, second_geometry);
    }

    #[test]
    fn decode_maps_anchor_grid_to_frame_coordinates() {
        // A 4x2 frame letterboxes with `scale = 80` and `top = 40`, so the
        // tensor-to-frame map is `(x, y) -> (x / 80, (y - 40) / 80)`.
        //
        // Cell (10, 10) of the stride-8 grid anchors at tensor (80, 80), which
        // is frame (1, 0.5). With `dx = 7.5` the box's centre lands at
        // `(10 + 7.5) * 8 = 140` across and with `dy = 15.0` at
        // `(10 + 15) * 8 = 200` down, so the centre is frame (1.75, 2.0) — half
        // a pixel below the frame's bottom edge, which is why the vertical
        // clamp bites here and the horizontal one does not.
        let mut output = zero_output();
        // `raw_score(0.9)` is the logit of 0.9, so the score is exact rather
        // than a literal that happens to round to it.
        set_cell(
            &mut output,
            10,
            10,
            8,
            raw_cell(
                raw_score(0.9),
                7.5,                 // dx: centre x = (10 + 7.5) * 8 = 140
                15.0,                // dy: centre y = (10 + 15) * 8 = 200
                raw_size(40.0, 8.0), // dw: width = 40 tensor pixels
                raw_size(50.0, 8.0), // dh: height = 50 tensor pixels
                [
                    (2.5, 7.5),  // right eye: 100, 140 tensor -> frame (1.25, 1.25)
                    (5.0, 7.5),  // left eye: 120, 140 -> frame (1.5, 1.25)
                    (7.5, 7.5),  // nose tip: 140, 140 -> frame (1.75, 1.25)
                    (10.0, 7.5), // right mouth: 160, 140 -> frame (2.0, 1.25)
                    (12.5, 7.5), // left mouth: 180, 140 -> frame (2.25, 1.25)
                ],
            ),
        );
        assert_eq!(
            [
                row_of(0, 0, 8),
                row_of(0, 0, 16),
                row_of(0, 0, 32),
                row_of(9, 7, 32)
            ],
            [0, 1200, 1500, ROWS - 1],
            "the grids tile the tensor: 1200 stride-8 cells, 300 stride-16, \
             then 80 stride-32 whose last is the tensor's last row"
        );

        let geometry = letterbox_geometry(4, 2);
        assert_eq!(
            (geometry.scale, geometry.top),
            (80.0, 40),
            "the case below is written against this letterbox"
        );
        // Four cells were written: this one, the coarser grids' below, and the
        // three that take no part in the suppression case yet. They decode to
        // boxes of their own, so the count is asserted at the end of the test
        // rather than here; what matters here is the box this cell produced.
        let faces = decode(&output, &geometry).expect("a well-formed output decodes");
        assert_eq!(faces.len(), 1, "one strong cell is one face");
        let face = faces[0];
        assert!(
            (face.score - 0.9).abs() < 1e-6,
            "the raw score is a logit, not a probability: {}",
            face.score
        );
        // Frame x: 140 / 80 = 1.75, and the box is 40 / 80 = 0.5 wide, so the
        // left edge is 1.5. Frame y: (200 - 40) / 80 = 2.0, and the box is
        // 50 / 80 = 0.625 tall, so the top edge is 2.0 - 0.3125 = 1.6875 — 0.3125
        // of it past the frame's 2-pixel bottom edge, which clamps the *top* to
        // 1.6875 and leaves the height at the trimmed 0.3125.
        let rect = face.rect;
        assert!(
            (rect.x - 1.5).abs() < 1e-3,
            "left = 1.75 - 0.25: {}",
            rect.x
        );
        assert!(
            (rect.y - 1.6875).abs() < 1e-3,
            "top = 2.0 - 0.3125: {}",
            rect.y
        );
        assert!(
            (rect.w - 0.5).abs() < 1e-3,
            "the width is inside the frame and untrimmed: {}",
            rect.w
        );
        assert!(
            (rect.h - 0.3125).abs() < 1e-3,
            "the height is trimmed at the frame edge: {}",
            rect.h
        );

        // The landmarks come through the same inverse transform, from the same
        // anchor, and in YuNet's own order. Every raw green delta in this cell
        // is 7.5, so every landmark is one row below the box's vertical centre.
        let expected = [
            (1.25, 1.25),
            (1.5, 1.25),
            (1.75, 1.25),
            (2.0, 1.25),
            (2.25, 1.25),
        ];
        for (index, (want, got)) in expected.iter().zip(face.landmarks).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-3 && (got.1 - want.1).abs() < 1e-3,
                "landmark {index}: want {want:?}, got {got:?}"
            );
        }

        // Every rect is inside the frame — here a 4x2 window, which the
        // `dh`-sized box above overflows. A detector that fired below the image
        // must not hand stage 2 a crop that is not there.
        for face in &faces {
            let rect = face.rect;
            assert!(rect.x >= 0.0 && rect.y >= 0.0, "{rect:?} starts inside");
            assert!(
                rect.x + rect.w <= 4.0 && rect.y + rect.h <= 2.0,
                "{rect:?} ends inside the 4x2 frame"
            );
            assert!(rect.w >= 0.0 && rect.h >= 0.0, "{rect:?} has no flip");
        }

        // A frame large enough to hold the whole box is *not* clamped: the
        // clamp must trim to the image, not shrink every box towards its centre.
        let roomy = letterbox_geometry(640, 480);
        let roomy_faces = decode(&output, &roomy).expect("decodes");
        let rect = roomy_faces[0].rect;
        // 640x480 scales by 0.5 with no padding at all, so tensor (140, 200) is
        // frame (280, 400) and the 40x50 tensor box is 80x100 frame pixels.
        assert!((rect.x - 240.0).abs() < 1e-3, "{}", rect.x);
        assert!((rect.y - 350.0).abs() < 1e-3, "{}", rect.y);
        assert!((rect.w - 80.0).abs() < 1e-3 && (rect.h - 100.0).abs() < 1e-3);

        // Suppression: two cells side by side decode to two boxes on one face,
        // and a third far away is a second face. Without the `nms` call there
        // would be three faces here, which downstream would read as three
        // people.
        //
        // Decoded against the 640x480 letterbox, not the 4x2 one above: the
        // 4x2 frame's clamp trims every box to the same two rows, which erases
        // the very overlap being tested. Both boxes here fit inside the frame
        // untrimmed, so the IoU is the arithmetic the decode actually produced.
        //
        // The neighbours are *not* decoded from one identical box on purpose:
        // two cells of different grids cannot produce bit-identical tensor
        // coordinates, and a test that tried would be asserting the arithmetic
        // of the grids rather than whether suppression ran at all.
        let mut crowded = output.clone();
        // Cell (11, 10) of the stride-8 grid: the anchor shifts 8 tensor pixels
        // right, the raw `dx` shifts 2.5 cells back left, and the box is the
        // same 40x50. The second box therefore starts 8 tensor pixels right of
        // the first and overlaps 32 of its 40 columns.
        set_cell(
            &mut crowded,
            11,
            10,
            8,
            raw_cell(
                raw_score(0.9),
                5.0,
                15.0,
                raw_size(40.0, 8.0),
                raw_size(50.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        // Cell (25, 15) decodes at tensor (260, 240) — 120 tensor pixels right
        // of and 40 below the first cell — with the same 40x50 box, so it
        // reaches neither of the others on either axis. It is kept away from
        // the tensor's edges deliberately: a box running past them would be
        // trimmed, and a trimmed box proves less about suppression than a whole
        // one. A `nms` that kept only the strongest candidate would fail here.
        set_cell(
            &mut crowded,
            25,
            15,
            8,
            raw_cell(
                raw_score(0.9),
                7.5,
                10.0,
                raw_size(40.0, 8.0),
                raw_size(50.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        let faces = decode(&crowded, &roomy).expect("decodes");
        // Which two of the *three* stride-8 cells survived is the part that
        // matters, and the scores cannot say — so the boxes do. The first and
        // second cells decoded to one 80x100 frame box and the third to a box
        // 240 frame pixels right of it, sharing no pixels with either.
        let mut stride_8_boxes: Vec<(i32, i32, i32, i32)> = faces
            .iter()
            .filter(|face| (face.score - 0.9).abs() < 1e-6)
            .map(|face| {
                let rect = face.rect;
                (
                    rect.x.round() as i32,
                    rect.y.round() as i32,
                    rect.w.round() as i32,
                    rect.h.round() as i32,
                )
            })
            .collect();
        stride_8_boxes.sort_unstable();
        // The first cell: tensor (120, 175)..(160, 225) at scale 0.5 and no
        // padding, so frame (240, 350)..(320, 450). The third: the same size
        // box centred on tensor (260, 240), so frame (480, 300)..(560, 400).
        assert_eq!(
            stride_8_boxes,
            vec![(240, 350, 80, 100), (480, 350, 80, 100)],
            "the second cell's box was suppressed and both faces survived"
        );

        // A weak cell is not a face however well formed it is: the score gate
        // is on the sigmoid, so a raw 0.0 is 0.5 and below the threshold.
        let mut weak = zero_output();
        set_cell(
            &mut weak,
            10,
            10,
            8,
            raw_cell(
                0.0,
                7.5,
                7.5,
                raw_size(40.0, 8.0),
                raw_size(50.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        assert!(
            decode(&weak, &geometry).expect("decodes").is_empty(),
            "a raw score of 0.0 is a probability of {}, below the {} threshold",
            decoded_score(0.0),
            SCORE_THRESHOLD
        );

        // The gate's boundary, both sides of it. `raw_score` inverts the
        // sigmoid exactly, so the cell below decodes to a hair under the
        // threshold and the one on it to exactly the threshold — which is the
        // case that separates `>=` from `>`. A gate that dropped a detection
        // scoring exactly the threshold would be off by one comparison, and
        // nothing else in the suite would notice.
        for (probability, expected) in [
            (SCORE_THRESHOLD - 1e-4, 0usize),
            (SCORE_THRESHOLD, 1),
            (SCORE_THRESHOLD + 1e-4, 1),
        ] {
            let mut cell = zero_output();
            set_cell(
                &mut cell,
                10,
                10,
                8,
                raw_cell(
                    raw_score(probability),
                    0.0,
                    0.0,
                    raw_size(40.0, 8.0),
                    raw_size(50.0, 8.0),
                    [(0.0, 0.0); LANDMARKS],
                ),
            );
            let faces = decode(&cell, &geometry).expect("decodes");
            assert_eq!(
                faces.len(),
                expected,
                "a cell decoding to {probability} against a {} threshold",
                SCORE_THRESHOLD
            );
            if expected == 1 {
                assert!(
                    (faces[0].score - probability).abs() < 1e-4,
                    "and the score that survived is the one that was set: {}",
                    faces[0].score
                );
            }
        }
    }

    #[test]
    fn decode_reads_all_three_grids() {
        // The tensor holds three feature grids laid end to end — strides 8, 16
        // and 32 — and the decode walks them as one flat run of anchors. A
        // decode that lost track of where the coarser grids start, or that
        // truncated the partial last row of the stride-32 grid (240 rows is
        // 7.5 cells of 32), reads a whole class of anchors at the wrong place
        // and silently loses every face they would have found.
        //
        // Each cell below is placed at its grid's own anchor and decodes to a
        // frame box that only the correct mapping produces, in the 4x2
        // letterbox where `scale = 80` and `top = 40`.
        let mut output = zero_output();

        // Cell (5, 5) at stride 16 anchors at tensor (80, 80). Deltas of 2.5
        // cells reach `(5 + 2.5) * 16 = 120` on both axes, so the centre is
        // frame (1.5, 1.0), and an 8x10 tensor box is 0.1x0.125 frame pixels.
        set_cell(
            &mut output,
            5,
            5,
            16,
            raw_cell(
                raw_score(0.8),
                2.5,
                2.5,
                raw_size(8.0, 16.0),
                raw_size(10.0, 16.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        // Cell (2, 2) at stride 32 anchors at tensor (64, 64). Deltas of 1.0
        // cell reach `(2 + 1) * 32 = 96` on both axes, so the centre is frame
        // (1.2, 0.7), with a 16x16 tensor box.
        set_cell(
            &mut output,
            2,
            2,
            32,
            raw_cell(
                raw_score(0.7),
                1.0,
                1.0,
                raw_size(16.0, 32.0),
                raw_size(16.0, 32.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        // Cell (9, 7) is the last cell of the stride-32 grid — the one that
        // only exists because the grid row count is a ceiling, not a
        // truncation. It anchors at tensor (288, 224); deltas of -1.0 cell
        // reach `(9 - 1) * 32 = 256` across and `(7 - 1) * 32 = 192` down, so
        // the centre is frame (3.2, 1.9) — the bottom-right corner, which is
        // where the last cell of a grid that overhangs the input belongs.
        set_cell(
            &mut output,
            9,
            7,
            32,
            raw_cell(
                raw_score(0.65),
                -1.0,
                -1.0,
                raw_size(16.0, 32.0),
                raw_size(16.0, 32.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&output, &geometry).expect("decodes");
        assert_eq!(
            faces.len(),
            3,
            "one cell per grid, each decoded once: {faces:?}"
        );
        assert_eq!(
            faces.iter().map(|face| face.score).collect::<Vec<_>>(),
            vec![0.8, 0.7, 0.65],
            "strongest first, across all three grids"
        );

        let boxes: Vec<(f32, f32, f32, f32)> = faces
            .iter()
            .map(|face| {
                let rect = face.rect;
                (rect.x, rect.y, rect.w, rect.h)
            })
            .collect();
        // The stride-16 box: centre (1.5, 1.0), half of 0.1 x 0.125 either side.
        assert!(
            (boxes[0].0 - 1.45).abs() < 1e-4
                && (boxes[0].1 - 0.9375).abs() < 1e-4
                && (boxes[0].2 - 0.1).abs() < 1e-4
                && (boxes[0].3 - 0.125).abs() < 1e-4,
            "the stride-16 cell: {:?}",
            boxes[0]
        );
        // The stride-32 box: centre (1.2, 0.7), half of 0.2 either side.
        assert!(
            (boxes[1].0 - 1.1).abs() < 1e-4
                && (boxes[1].1 - 0.6).abs() < 1e-4
                && (boxes[1].2 - 0.2).abs() < 1e-4
                && (boxes[1].3 - 0.2).abs() < 1e-4,
            "the stride-32 cell: {:?}",
            boxes[1]
        );
        // The overhanging row's box: centre (3.2, 1.9), 0.2 either side — so
        // its bottom 0.1 is past the frame's 2-pixel edge and is trimmed.
        assert!(
            (boxes[2].0 - 3.1).abs() < 1e-4
                && (boxes[2].1 - 1.8).abs() < 1e-4
                && (boxes[2].2 - 0.2).abs() < 1e-4
                && (boxes[2].3 - 0.2).abs() < 1e-4,
            "the last row's cell: {:?}",
            boxes[2]
        );

        // The three boxes are in different places, so a decode that mapped the
        // coarse grids onto stride-8 anchors — which would put two of them on
        // top of each other — cannot pass the assertions above.
        assert!(
            boxes[0].0 != boxes[1].0 || boxes[0].1 != boxes[1].1,
            "the grids' cells must not collapse onto one another: {boxes:?}"
        );
    }

    #[test]
    fn a_box_reaching_past_the_frame_edge_is_trimmed_to_it() {
        // A 640x480 frame scales by 0.5 with no padding, so the tensor's own
        // edges map onto the frame's, and a box the decode trims at the tensor
        // edge lands exactly on the frame edge.
        //
        // Cell (30, 20) of the stride-8 grid anchors at tensor (240, 160).
        // Deltas of 5.0 cells put the centre at `(30 + 5) * 8 = 280` across and
        // `(20 + 5) * 8 = 200` down, and an 80x160 tensor box is 160x320 frame
        // pixels — so the box wants to span frame (480, 240) to (640, 560),
        // and half its height is below the frame. The trim is what keeps the
        // bottom at 480.
        let mut output = zero_output();
        set_cell(
            &mut output,
            30,
            20,
            8,
            raw_cell(
                raw_score(0.9),
                5.0,
                5.0,
                raw_size(80.0, 8.0),
                raw_size(160.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );
        // Cell (2, 2): the same box near the *top-left* corner instead. Centre
        // `(2 + 4) * 8 = 48` on both axes, so the tensor box is
        // (-32, -32)..(48, 80) — the top-left of it above and left of the
        // tensor, and its width reaching only halfway across. The frame box is
        // (16, 0)..(176, 480).
        set_cell(
            &mut output,
            2,
            2,
            8,
            raw_cell(
                raw_score(0.8),
                4.0,
                4.0,
                raw_size(80.0, 8.0),
                raw_size(160.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );

        let geometry = letterbox_geometry(640, 480);
        let faces = decode(&output, &geometry).expect("decodes");
        assert_eq!(faces.len(), 2, "two cells, two faces: {faces:?}");

        // The far cell: trimmed at the frame's bottom and right edges.
        let rect = faces[0].rect;
        assert!(
            (rect.x - 480.0).abs() < 1e-3 && (rect.y - 240.0).abs() < 1e-3,
            "the box starts inside the frame, where it was decoded: {rect:?}"
        );
        assert!(
            (rect.w - 160.0).abs() < 1e-3 && (rect.h - 240.0).abs() < 1e-3,
            "and it stops at the frame edge rather than running past it: {rect:?}"
        );

        // The near cell: its top is trimmed at the frame's top edge. A clamp
        // that only had an upper bound would report a negative `y` here.
        let corner = faces[1].rect;
        assert!(
            (corner.x - 16.0).abs() < 1e-3 && corner.y == 0.0,
            "the box's top is trimmed to the frame's top edge: {corner:?}"
        );
        assert!(
            (corner.w - 160.0).abs() < 1e-3 && (corner.h - 256.0).abs() < 1e-3,
            "leaving the part of it that is on the frame: {corner:?}"
        );

        for face in &faces {
            let rect = face.rect;
            assert!(
                rect.x >= 0.0
                    && rect.y >= 0.0
                    && rect.x + rect.w <= 640.0
                    && rect.y + rect.h <= 480.0,
                "nothing may extend past the frame: {rect:?}"
            );
        }
    }

    #[test]
    fn a_box_whose_origin_lands_in_the_padding_is_trimmed_at_the_top() {
        // Padding exists above the content, so a box can decode to an origin
        // *above* the frame. Cell (2, 2) of the stride-8 grid anchors at tensor
        // (16, 16); with `dy = -1.0` the centre is `(2 + -1) * 8 = 8` down and
        // a 240-tall tensor box puts the box's top at 8 - 120 = -112. The
        // tensor clamp floors that at 0, which the inverse map sends to frame
        // y = -240 for a 640x480 frame — outside the image, and negative.
        //
        // A rect with a negative origin would be handed to Face Mesh as a crop
        // starting above the frame, and would fail any "inside the frame" check
        // downstream. The clamp is what stops it, and this is the only input in
        // the suite that reaches it.
        let mut output = zero_output();
        set_cell(
            &mut output,
            2,
            2,
            8,
            raw_cell(
                raw_score(0.9),
                5.0,
                -1.0,
                raw_size(80.0, 8.0),
                raw_size(240.0, 8.0),
                [(0.0, 0.0); LANDMARKS],
            ),
        );

        let geometry = letterbox_geometry(640, 480);
        let faces = decode(&output, &geometry).expect("decodes");
        assert_eq!(faces.len(), 1);
        let rect = faces[0].rect;
        assert!(
            rect.y == 0.0,
            "the box's top is floored at the frame's first row: {rect:?}"
        );
        assert!(
            (rect.x - 32.0).abs() < 1e-3 && (rect.w - 160.0).abs() < 1e-3,
            "its horizontal extent is untouched by the vertical trim: {rect:?}"
        );
        assert!(
            (rect.h - 256.0).abs() < 1e-3,
            "and its height is the part of it that is on the frame: {rect:?}"
        );
    }

    #[test]
    fn landmarks_outside_the_frame_come_back_inside_it() {
        // A landmark's offset is in the cell's own units, and a cell near the
        // tensor's edge can carry one into the padding. Here cell (10, 25) of
        // the stride-8 grid anchors at tensor (80, 200), so with `dx = dy = 20`
        // the first landmark is at `(10 + 20) * 8 = 240` tensor rows — the
        // very bottom of the tensor, 40 rows into the 4x2 frame's lower
        // padding.
        //
        // The frame maps that to y = 2.5 in a frame 2 pixels tall. The box is
        // clamped in tensor space, but a landmark is a bare point with no box
        // to clamp: the frame clamp is the only thing standing between it and
        // a coordinate outside the image. Face Mesh is handed these directly,
        // so a landmark at y = 2.5 would be a crop request above the frame.
        let mut output = zero_output();
        set_cell(
            &mut output,
            10,
            25,
            8,
            raw_cell(
                raw_score(0.9),
                7.5,
                0.0,
                raw_size(40.0, 8.0),
                raw_size(40.0, 8.0),
                [
                    (20.0, 20.0),
                    (20.0, -20.0),
                    (-20.0, 20.0),
                    (-20.0, -20.0),
                    (0.0, 0.0),
                ],
            ),
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&output, &geometry).expect("decodes");
        assert_eq!(faces.len(), 1, "one cell: {faces:?}");
        let landmarks = faces[0].landmarks;

        // The fifth landmark sits on the cell's own anchor, tensor (80, 200) ->
        // frame (1.0, 2.0), which is inside the frame — so a clamp applied to
        // *every* landmark rather than only the ones that need it would move
        // this one and show up here.
        assert!(
            (landmarks[4].0 - 1.0).abs() < 1e-3 && (landmarks[4].1 - 2.0).abs() < 1e-3,
            "the centre landmark is untouched: {:?}",
            landmarks[4]
        );
        // The first is 40 tensor rows below the frame's last row, so its y is
        // clamped to the frame's bottom edge; the second is 40 rows above the
        // top, so its y is clamped to `0`. Their x is where the cell put them.
        assert!(
            landmarks[0] == (3.0, 2.0),
            "the below-frame landmark is pulled onto the bottom edge: {:?}",
            landmarks[0]
        );
        assert!(
            landmarks[1] == (3.0, 0.0),
            "and the above-frame one onto the top edge: {:?}",
            landmarks[1]
        );
        // The third is 40 tensor columns left of the frame's first column, so
        // its x is clamped to `0` while its y is clamped as the first's.
        assert!(
            landmarks[2] == (0.0, 2.0),
            "the left-of-frame landmark is pulled onto the left edge: {:?}",
            landmarks[2]
        );

        for (index, (x, y)) in landmarks.iter().enumerate() {
            assert!(
                *x >= 0.0 && *y >= 0.0 && *x <= 4.0 && *y <= 2.0,
                "landmark {index} must be inside the 4x2 frame, got ({x}, {y})"
            );
        }
    }

    #[test]
    fn decode_rejects_an_output_it_cannot_read() {
        // A wrong-shaped tensor is a wrong model file or a wrong output
        // selected, and reading it anyway would index past the end of the data.
        // Either way it is an error, not a panic.
        let geometry = letterbox_geometry(640, 480);

        let mut wrong_shape = zero_output();
        wrong_shape.shape = vec![1, 395, COLUMNS as i64];
        assert!(matches!(
            decode(&wrong_shape, &geometry),
            Err(ModelError::UnexpectedShape {
                name: "output1",
                ..
            })
        ));

        let mut wrong_name = zero_output();
        wrong_name.name = "output0".to_owned();
        assert!(matches!(
            decode(&wrong_name, &geometry),
            Err(ModelError::UnexpectedOutput {
                expected: "output1",
                ..
            })
        ));

        let mut wrong_type = zero_output();
        wrong_type.data = TensorData::I32(vec![0; ROWS * COLUMNS]);
        assert!(matches!(
            decode(&wrong_type, &geometry),
            Err(ModelError::MissingOutput { name: "output1" })
        ));
    }

    #[test]
    fn detect_returns_empty_for_a_blank_frame_when_session_says_so() {
        let (session, handle) = blank_session();
        let yunet = YuNet::new(session);

        let faces = yunet
            .detect(&solid_frame(640, 480, [0x80, 0x80, 0x80]))
            .expect("a zero-returning session is not an error");

        assert!(
            faces.is_empty(),
            "all-zero output is a probability of {}, not a face: {faces:?}",
            decoded_score(0.0)
        );
        assert_eq!(
            yunet.name(),
            "yunet",
            "the wrapper reports the session's name"
        );

        // And the session was actually asked to run — a `detect` that skipped
        // inference would pass the assertion above for the wrong reason.
        assert_eq!(
            handle.call_count(),
            1,
            "detect must run the graph exactly once"
        );
    }

    #[test]
    fn detect_hands_the_session_the_letterboxed_frame() {
        // The tensor the session receives is the contract every later model
        // reads, so it is asserted where it is produced rather than assumed
        // from `preprocess`'s own test.
        let (session, handle) = blank_session();
        let yunet = YuNet::new(session);
        yunet.detect(&solid_frame(640, 480, [0; 3])).unwrap();

        let inputs = handle.last_inputs().expect("one run call");
        assert_eq!(inputs.len(), 1, "YuNet's graph takes one input");
        assert_eq!(inputs[0].name, "input");
        assert_eq!(inputs[0].shape, vec![1, 3, 240, 320]);
        assert_eq!(
            inputs[0].data.as_f32().expect("f32").len(),
            3 * TENSOR_WIDTH * TENSOR_HEIGHT,
            "the shape and the data must agree"
        );

        // A session that returns nothing at all is an error, not an empty
        // result: "no faces" and "the model did not answer" are different
        // facts and the engine must be able to tell them apart.
        let empty = YuNet::new(Box::new(FakeSession::new("yunet", Vec::new())));
        assert!(matches!(
            empty.detect(&solid_frame(640, 480, [0; 3])),
            Err(ModelError::MissingOutput { name: "output1" })
        ));
    }
}
