//! YuNet face detection: the pipeline's stage 1.
//!
//! YuNet is anchor-free. Its graph emits, per stride, one row per cell of a
//! feature grid — a class score, an objectness score, four box parameters and
//! five facial landmarks, all as offsets from that cell's anchor. There is no
//! separate anchor table to load; the grid *is* the anchor set.
//!
//! The shipping model is `face_detection_yunet_2023mar.onnx` (232,589 bytes,
//! sha256 `8f2383e4…`), whose graph was read directly to write this decode:
//!
//! * one input, `input`, `[1, 3, 640, 640]` f32, opset 11;
//! * twelve outputs — `cls_8/16/32` `[1, N, 1]`, `obj_8/16/32` `[1, N, 1]`,
//!   `bbox_8/16/32` `[1, N, 4]`, `kps_8/16/32` `[1, N, 10]`;
//! * `N` is 6400 / 1600 / 400 for the 80x80, 40x40 and 20x20 grids — 8400
//!   anchors in total.
//!
//! Two consequences shape everything below. The grids are derived from the
//! input size and the strides rather than written down, so a different export
//! cannot silently mis-index them. And **`cls_*` and `obj_*` each already pass
//! through a `Sigmoid` inside the graph** while `bbox_*` and `kps_*` do not —
//! so the confidence is `sqrt(cls * obj)` of two probabilities, *not* a sigmoid
//! applied here, and the box sizes do need `exp`. See [`decode`].
//!
//! Preprocessing is this model's own regime and is deliberately not shared:
//! BGR passed through, values scaled by `1/255`, the frame letterboxed with its
//! aspect ratio preserved, and the padding filled with zeros.

use super::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};
use crate::analyze::nms::{Detection, Rect, nms};
use crate::frame::Frame;

/// Input width the shipping model is fixed at, in pixels.
pub const TENSOR_WIDTH: usize = 640;

/// Input height the shipping model is fixed at, in pixels.
pub const TENSOR_HEIGHT: usize = 640;

/// The three feature-grid strides, tightest first.
const STRIDES: [usize; 3] = [8, 16, 32];

/// The class and objectness heads carry one value per anchor.
const CLASS_COLUMNS: usize = 1;

/// The box head carries `dx`, `dy`, `dw`, `dh`.
const BOX_COLUMNS: usize = 4;

/// Landmarks per face: right eye, left eye, nose tip, right mouth corner, left
/// mouth corner — YuNet's own order, and Face Mesh's stage-2 input.
const LANDMARKS: usize = 5;

/// The landmark head carries five `(x, y)` pairs.
const LANDMARK_COLUMNS: usize = 2 * LANDMARKS;

/// The name of the graph's image input, as the model declares it.
const INPUT_NAME: &str = "input";

/// The twelve head names, in the order the graph declares them.
const HEAD_NAMES: [&str; OUTPUT_COUNT] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

/// Outputs per stride: class, objectness, box, landmarks.
const HEADS: usize = 4;

/// How many outputs the graph declares: [`HEADS`] for each of [`STRIDES`].
const OUTPUT_COUNT: usize = HEADS * STRIDES.len();

/// Raw scores at or above this are faces.
///
/// `0.6` is the value the model card and OpenCV's own demo use. Note what it is
/// applied to: `sqrt(cls * obj)`, both of which the graph has already put
/// through a sigmoid. A session that returns zeros for a blank frame therefore
/// scores `0.0` — not `0.5`, and not `1.0` — and yields no faces.
const SCORE_THRESHOLD: f32 = 0.6;

/// Boxes overlapping a kept box by more than this are the same face.
const NMS_IOU_THRESHOLD: f32 = 0.3;

/// The value `1/255` scales BGR bytes into the unit range with.
const UNIT_RANGE: f32 = 1.0 / 255.0;

/// The range a cell's raw centre offsets are clamped to, in cells.
///
/// Only a guard against a garbage tensor: a centre offset is multiplied by the
/// stride and the tensor clamp then trims the box, so a loose bound here costs
/// nothing. What it buys is that `RAW_LIMIT * stride` cannot approach the range
/// where the arithmetic loses precision.
const RAW_LIMIT: f32 = 30.0;

/// The grid columns one stride has over an input `width`.
///
/// A stride does not have to divide the input. Truncating instead of taking the
/// ceiling drops the last partial cell, and with it every anchor covering the
/// right-hand strip of the frame — a strip where nobody is then ever detected.
///
/// Takes the dimension as an argument, and is `const`, so the decode's own use
/// of it is the same expression its tests exercise: at 640 every stride divides
/// exactly, and a test that could only run at that size could not tell the two
/// apart.
const fn anchor_columns(width: usize, stride: usize) -> usize {
    width.div_ceil(stride)
}

/// The grid rows one stride has over an input `height`. See [`anchor_columns`].
const fn anchor_rows(height: usize, stride: usize) -> usize {
    height.div_ceil(stride)
}

/// The anchor count of one stride's grid.
const fn anchor_count(width: usize, height: usize, stride: usize) -> usize {
    anchor_columns(width, stride) * anchor_rows(height, stride)
}

/// The range a cell's raw size parameters are clamped to before `exp`.
///
/// This one is load-bearing rather than a formality, because `exp` on a raw
/// float is not bounded by a constant: `exp` overflows to infinity past ~88, so
/// an unclamped raw size from a garbage tensor would put an infinity — and then
/// a NaN, via `inf - inf` — into every downstream rect. Clamping to 30 caps the
/// size at `exp(30) * stride`, which is about 3e10 tensor pixels: far beyond
/// any real face, and far below the overflow.
const SIZE_LIMIT: f32 = 30.0;

/// The largest box the tensor clamp can ever produce, in tensor pixels:
/// `exp(SIZE_LIMIT) * max(stride)`, saturating.
///
/// The box is clamped to the tensor right after it is built, so this is the
/// bound on every coordinate `to_frame` is ever handed. It is what keeps the
/// inverse map from overflowing to infinity and cancelling to a NaN in the
/// subtraction that builds the rect.
fn max_tensor_size() -> f32 {
    SIZE_LIMIT.exp() * STRIDES[STRIDES.len() - 1] as f32
}

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
/// Produced by [`preprocess`] and consumed by [`decode`], so the forward and
/// inverse transforms are one piece of arithmetic read in two directions rather
/// than two derivations that can disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxGeometry {
    /// Scale factor applied to the frame, so that `scale * width <= 640` and
    /// `scale * height <= 640` with the tighter axis exact. Above 1.0 for a
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
    /// The frame's width in the tensor, in tensor pixels.
    fn content_width(self) -> usize {
        ((self.frame_width as f32 * self.scale).round() as usize).min(TENSOR_WIDTH)
    }

    /// The frame's height in the tensor, in tensor pixels.
    fn content_height(self) -> usize {
        ((self.frame_height as f32 * self.scale).round() as usize).min(TENSOR_HEIGHT)
    }

    /// The frame coordinate a tensor position maps to.
    ///
    /// This is a **position** map, not a pixel-index map: a tensor coordinate is
    /// a continuous position in the letterboxed image, and this says where that
    /// position sits in the original frame. [`preprocess`] samples with it and
    /// [`decode`] inverts boxes with it, so the two directions agree by
    /// construction rather than by two separate derivations of the same algebra.
    ///
    /// A coordinate in the padding maps outside the frame, which is expected:
    /// callers clamp.
    fn to_frame(self, x: f32, y: f32) -> (f32, f32) {
        let inverse_scale = if self.scale > 0.0 {
            1.0 / self.scale
        } else {
            // A frame with no pixels has no scale, and no content either: every
            // tensor coordinate is padding. Zero keeps a NaN out of the rects.
            return (0.0, 0.0);
        };
        (
            (x - self.left as f32) * inverse_scale,
            (y - self.top as f32) * inverse_scale,
        )
    }

    /// The exact algebraic inverse of [`to_frame`](Self::to_frame).
    ///
    /// Only a test uses this, and that is the point: it is the *wrong* direction
    /// for sampling — handing it a tensor coordinate multiplies by the scale and
    /// magnifies a corner of the frame instead of covering it — so it exists to
    /// be asserted against rather than called. The round trip
    /// `to_frame(to_tensor(p)) == p` is what proves `to_frame` is the inverse of
    /// a well-defined forward map, which is the property `preprocess` and
    /// `decode` both rely on when they use `to_frame` in opposite roles.
    #[cfg(test)]
    fn to_tensor(self, x: f32, y: f32) -> (f32, f32) {
        let scale = if self.scale > 0.0 { self.scale } else { 1.0 };
        (x * scale + self.left as f32, y * scale + self.top as f32)
    }
}

/// Builds the letterboxed NCHW tensor YuNet expects from `frame`.
///
/// Returns the input tensor and the geometry that inverts it. Pure: same frame
/// in, same tensor out, with no clock or randomness anywhere.
///
/// BGR is passed through in the order the frame already holds it, values are
/// scaled by `1/255`, and the padding stays zero. Sampling is bilinear rather
/// than nearest-neighbour because at a 2:1 downscale nearest-neighbour turns a
/// face into a different pattern of pixels as the subject moves a few
/// centimetres, and the detector then reports a different box each frame.
pub fn preprocess(frame: &Frame) -> (InputTensor, LetterboxGeometry) {
    let geometry = letterbox_geometry(frame.width, frame.height);

    // The content is centred on *both* axes, so both bands carry their offset.
    // Writing `0..content_width` horizontally instead of `left..left +
    // content_width` is a bug that hides itself: `to_frame` subtracts `left`, so
    // every column before `left` samples the frame's first column again — the
    // left padding fills with a smeared copy of the frame's edge instead of
    // zeros — and the columns from `left + content_width` to the tensor's edge
    // are never written at all, so the frame's right-hand strip is never seen by
    // the model. Detection coordinates stay self-consistent because `decode`
    // inverts the same map, which is what makes it invisible content loss
    // rather than a visible shift. It only bites when the frame is portrait
    // (`left > 0`), which is why a portrait fixture is the test for it.
    let columns = geometry.left..geometry.left + geometry.content_width();
    let rows = geometry.top..geometry.top + geometry.content_height();

    // The whole tensor at once, rather than three channel buffers to copy in
    // afterwards: at 5 Hz this runs 200 times a minute and the copy would be
    // the largest cost in the function.
    let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
    let mut data = vec![0.0f32; plane * 3];

    for y in rows.clone() {
        for x in columns.clone() {
            // `to_frame`, not the frame-to-tensor map: `x` and `y` here are
            // *tensor* coordinates, and the question is which frame position
            // that tensor position came from. Multiplying by the scale where it
            // must divide would sample a magnified corner crop and clamp
            // everything else to that crop's edge, so no face outside the crop
            // could ever be detected, and every box the decode produced would
            // describe pixels the model never saw.
            let (source_x, source_y) = geometry.to_frame(x as f32, y as f32);
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
/// `scale` is a **multiplier**: `scale * width <= 640` and `scale * height <=
/// 640` always hold, with the tighter axis landing exactly on its bound and the
/// other one centred. A multiplier is `1 / max(width/640, height/640)` — taking
/// the smaller of `640/width` and `640/height` instead, which is the same
/// numbers in the wrong direction, scales a small frame *up* until it falls off
/// the tensor, and the round-trip assertion in the preprocessing test catches
/// exactly that.
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

/// Decodes the graph's twelve output tensors into faces, in frame coordinates.
///
/// `outputs` are matched by name, so their order does not matter. `geometry`
/// must be the one [`preprocess`] produced for the frame the tensors came from,
/// or every box is off by the letterbox.
///
/// `cls_*` and `obj_*` arrive **already sigmoid-activated** — the graph ends
/// each in a `Sigmoid` node — so the confidence is `sqrt(cls * obj)`, the
/// combination OpenCV's own post-processing uses. Neither head is activated
/// again here: a second sigmoid would turn a confident 0.9 into 0.71 and
/// quietly destroy the confidence the whole engine is built on. `bbox_*` and
/// `kps_*` are *not* activated in the graph, so their box sizes do need `exp`.
pub fn decode(
    outputs: &[OutputTensor],
    geometry: &LetterboxGeometry,
) -> Result<Vec<FaceBox>, ModelError> {
    let heads = Heads::read(outputs)?;

    let (frame_width, frame_height) = (geometry.frame_width as f32, geometry.frame_height as f32);
    let mut candidates = Vec::new();

    for (index, stride) in STRIDES.iter().enumerate() {
        let stride = *stride;
        // Both dimensions, through the one function every grid is derived from.
        // The alternative — `TENSOR_WIDTH / stride` — reads the same at the
        // shipping size and silently drops the last partial row at any other.
        let columns = anchor_columns(TENSOR_WIDTH, stride);
        let rows = anchor_rows(TENSOR_HEIGHT, stride);
        let scores = heads.classes[index];
        let objects = heads.objects[index];
        let boxes = heads.boxes[index];
        let landmarks = heads.landmarks[index];

        for grid_y in 0..rows {
            for grid_x in 0..columns {
                let anchor = grid_y * columns + grid_x;
                // Both heads are probabilities already. Their geometric mean is
                // the confidence, and the square root is why a zero on *either*
                // head means no face rather than a half-confident one.
                let class = scores[anchor].clamp(0.0, 1.0);
                let object = objects[anchor].clamp(0.0, 1.0);
                let score = (class * object).sqrt();
                if score >= SCORE_THRESHOLD {
                    candidates.push(decode_cell(
                        &boxes[anchor * BOX_COLUMNS..],
                        &landmarks[anchor * LANDMARK_COLUMNS..],
                        score,
                        grid_x,
                        grid_y,
                        stride as f32,
                        (frame_width, frame_height),
                        geometry,
                    ));
                }
            }
        }
    }

    Ok(finish(candidates))
}

/// Turns decoded candidates into the returned face list, suppressing duplicates.
fn finish(candidates: Vec<FaceBox>) -> Vec<FaceBox> {
    // Every box is a face, so every detection is class 0 and suppression
    // compares like with like. Encoding a candidate's index in `class_id`
    // instead would make every detection its own class and suppress *nothing* —
    // the one mistake this decode cannot afford, since a face would then be
    // counted once per overlapping anchor.
    let mut detections: Vec<Detection> = candidates
        .iter()
        .map(|face| Detection {
            rect: face.rect,
            score: face.score,
            class_id: 0,
        })
        .collect();
    nms(&mut detections, NMS_IOU_THRESHOLD, 0.0);

    // `nms` reorders what it keeps — strongest first, which is the order
    // `YuNet::detect` documents — so the survivors are walked *in `nms`'s
    // order* and each one is resolved back to the candidate it came from.
    //
    // The tempting alternative is to mark a `survived` flag per candidate and
    // then filter `candidates` in place. That is wrong, and wrong invisibly:
    // candidates are in *grid* order (stride 8 row by row, then 16, then 32),
    // not score order, so filtering them returns whatever the scan order
    // happened to be. A stride-8 cell scoring 0.7 that the scan reached first
    // would come back ahead of a later one scoring 0.9, and every caller that
    // takes `faces[0]` as the most confident face would take the wrong one.
    let mut kept = Vec::with_capacity(detections.len());
    for detection in &detections {
        let position = candidates
            .iter()
            .position(|face| face.score == detection.score && face.rect == detection.rect)
            .expect("nms only ever returns detections it was given");
        kept.push(candidates[position]);
    }
    kept
}

/// The twelve output tensors, validated, split per stride.
///
/// Holds borrowed slices so the decode never copies a head of 6400 rows.
struct Heads<'a> {
    classes: [&'a [f32]; STRIDES.len()],
    objects: [&'a [f32]; STRIDES.len()],
    boxes: [&'a [f32]; STRIDES.len()],
    landmarks: [&'a [f32]; STRIDES.len()],
}

impl<'a> Heads<'a> {
    /// Looks all twelve outputs up by name and checks each one's shape and
    /// length before anything indexes it.
    fn read(outputs: &'a [OutputTensor]) -> Result<Heads<'a>, ModelError> {
        if outputs.len() != OUTPUT_COUNT {
            return Err(ModelError::UnexpectedOutputCount {
                expected: OUTPUT_COUNT,
                got: outputs.len(),
            });
        }

        let mut heads = Heads {
            classes: [&[][..]; STRIDES.len()],
            objects: [&[][..]; STRIDES.len()],
            boxes: [&[][..]; STRIDES.len()],
            landmarks: [&[][..]; STRIDES.len()],
        };

        for (index, stride) in STRIDES.iter().enumerate() {
            let anchors = anchor_count(TENSOR_WIDTH, TENSOR_HEIGHT, *stride);
            // `cls_8` and `obj_8` differ only by their name, so the name is the
            // only thing that tells the two one-column heads apart.
            heads.classes[index] = head(outputs, "cls", *stride, anchors, CLASS_COLUMNS)?;
            heads.objects[index] = head(outputs, "obj", *stride, anchors, CLASS_COLUMNS)?;
            heads.boxes[index] = head(outputs, "bbox", *stride, anchors, BOX_COLUMNS)?;
            heads.landmarks[index] = head(outputs, "kps", *stride, anchors, LANDMARK_COLUMNS)?;
        }

        Ok(heads)
    }
}

/// One head's values, after checking its name, shape and length.
///
/// The length check is not redundant with the shape check: [`OutputTensor`]'s
/// fields are public, so a session — or a test — can hand back a declared shape
/// that its buffer does not fill. Slicing by anchor after that would read past
/// the end of the slice, which panics, and this function promises a `Result`.
fn head<'a>(
    outputs: &'a [OutputTensor],
    prefix: &str,
    stride: usize,
    anchors: usize,
    columns: usize,
) -> Result<&'a [f32], ModelError> {
    let name = head_name(prefix, stride);
    let output = outputs
        .iter()
        .find(|output| output.name == name)
        .ok_or_else(|| ModelError::UnexpectedOutput {
            expected: name,
            got: outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        })?;

    let expected = [1, anchors as i64, columns as i64];
    if output.shape != expected {
        return Err(ModelError::UnexpectedShape {
            name,
            expected: expected.to_vec(),
            got: output.shape.clone(),
        });
    }
    let values = output
        .data
        .as_f32()
        .ok_or(ModelError::MissingOutput { name })?;
    if values.len() != anchors * columns {
        return Err(ModelError::UnexpectedDataLength {
            name,
            shape: output.shape.clone(),
            got: values.len(),
        });
    }
    Ok(values)
}

/// The `'static` name of one head of one stride.
///
/// The twelve names are a fixed set, so this is a lookup rather than a leak:
/// only the names in [`HEAD_NAMES`] can be asked for, and the fallback keeps
/// the function total for a prefix that is not one of them.
fn head_name(prefix: &str, stride: usize) -> &'static str {
    match (prefix, stride) {
        ("cls", 8) => "cls_8",
        ("cls", 16) => "cls_16",
        ("cls", _) => "cls_32",
        ("obj", 8) => "obj_8",
        ("obj", 16) => "obj_16",
        ("obj", _) => "obj_32",
        ("bbox", 8) => "bbox_8",
        ("bbox", 16) => "bbox_16",
        ("bbox", _) => "bbox_32",
        ("kps", 8) => "kps_8",
        ("kps", 16) => "kps_16",
        ("kps", _) => "kps_32",
        _ => HEAD_NAMES[0],
    }
}

/// Decodes one anchor cell into a face.
///
/// Eight parameters is a lot for one function, but they are eight *unrelated*
/// pieces of scalars the caller already has in registers as it walks the grid;
/// bundling them into a struct would add an allocation-free but noise-only type
/// and a construction per anchor. The alternative that would genuinely shorten
/// the list — passing the raw head buffers and letting this function find its
/// own cell — would spread the anchor arithmetic across two functions.
#[allow(clippy::too_many_arguments)]
fn decode_cell(
    box_cell: &[f32],
    landmark_cell: &[f32],
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
    // negative. Unlike the scores, these are *not* activated in the graph.
    let center_x = (grid_x as f32 + box_cell[0].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
    let center_y = (grid_y as f32 + box_cell[1].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
    let tensor_width = box_cell[2].clamp(-SIZE_LIMIT, SIZE_LIMIT).exp() * stride;
    let tensor_height = box_cell[3].clamp(-SIZE_LIMIT, SIZE_LIMIT).exp() * stride;

    let half_width = tensor_width / 2.0;
    let half_height = tensor_height / 2.0;

    // Clamped in tensor space first, so a garbage tensor cannot produce a rect
    // wider than the image it came from, then mapped to the frame, where the
    // padding is removed and the scale divided out. Sizes take the scale only:
    // the padding moves where a face is, never how big it is.
    let tensor_right = TENSOR_WIDTH as f32;
    let tensor_bottom = TENSOR_HEIGHT as f32;
    let left = (center_x - half_width).clamp(0.0, tensor_right);
    let top = (center_y - half_height).clamp(0.0, tensor_bottom);
    let right = (center_x + half_width).clamp(0.0, tensor_right);
    let bottom = (center_y + half_height).clamp(0.0, tensor_bottom);

    // `left` is at most the tensor's width, but `center_x` is not: a raw delta
    // at the clamp's edge puts it at `(grid + 30) * stride`, which for the
    // coarsest grid is over 1000 tensor pixels. Subtracting the half-size can
    // leave a large value, and `to_frame` multiplies it by the inverse scale —
    // bounded, because the scale derives from a frame dimension, but only up to
    // the ratio between a frame and the tensor. Clamping the *inputs* to the
    // map here is what keeps every coordinate it returns inside a range the
    // subtraction below can subtract without cancelling to a NaN.
    let reach = max_tensor_size();
    let (frame_left, frame_top) = geometry.to_frame(left, top);
    let (frame_right, frame_bottom) = geometry.to_frame(right, bottom);
    let frame_left = frame_left.clamp(-reach, reach);
    let frame_top = frame_top.clamp(-reach, reach);
    let frame_right = frame_right.clamp(-reach, reach);
    let frame_bottom = frame_bottom.clamp(-reach, reach);

    // Clamped again in frame space, and this is *not* redundant with the clamp
    // above — on either axis. That one bounds a box to the whole tensor, but the
    // letterbox pads whichever axis is not the tighter one, and the padding is
    // inside the tensor. So a box whose origin falls in the padding has a
    // tensor-space edge that is perfectly legal (0, or the tensor's own width)
    // and a frame-space edge that is outside the frame: a negative origin from
    // the top or left padding, and an end past the frame's last row or column
    // from the bottom or right. That happens whenever the frame is not the
    // tensor's aspect ratio in that axis — a portrait 480x640 frame has `left =
    // 80`, so the left padding is a real region that a box can land in, not a
    // hypothetical one.
    //
    // Without these clamps `Rect` would carry a negative origin and a negative
    // height, and nothing downstream checks that: `Rect` has no non-negativity
    // invariant and `Rect::iou` reads a negative-origin box as having positive
    // area, so Face Mesh would be handed a crop outside the image.
    //
    // The sizes are *derived* from the clamped endpoints rather than clamped in
    // their own right. `to_frame` preserves the left-to-right order of its
    // inputs and both endpoints land in `0..frame_width`, so the right edge is
    // never left of the left one and a `max(0.0)` there would be a second guard
    // for something the arithmetic has already established — and a guard that
    // cannot fire cannot be tested, which is why there is one `debug_assert!`
    // stating the whole contract instead of two silent floors.
    let (frame_width, frame_height) = frame;
    let x = frame_left.clamp(0.0, frame_width);
    let y = frame_top.clamp(0.0, frame_height);
    let rect = Rect {
        x,
        y,
        w: frame_right.clamp(0.0, frame_width) - x,
        h: frame_bottom.clamp(0.0, frame_height) - y,
    };
    debug_assert!(
        rect.w >= 0.0 && rect.h >= 0.0 && rect.x >= 0.0 && rect.y >= 0.0,
        "the clamps and a non-negative span are this function's contract: {rect:?}"
    );

    let mut landmarks = [(0.0f32, 0.0f32); LANDMARKS];
    for (index, landmark) in landmarks.iter_mut().enumerate() {
        let raw_x =
            (grid_x as f32 + landmark_cell[2 * index].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
        let raw_y =
            (grid_y as f32 + landmark_cell[2 * index + 1].clamp(-RAW_LIMIT, RAW_LIMIT)) * stride;
        // A landmark is a bare point with no box around it, so it can land in
        // the padding even when its box cannot. The clamp above cannot help it,
        // so this is the only thing between a landmark and a coordinate that
        // would have Face Mesh cropping outside the image.
        let (x, y) = geometry.to_frame(raw_x, raw_y);
        *landmark = (x.clamp(0.0, frame_width), y.clamp(0.0, frame_height));
    }

    FaceBox {
        rect,
        score,
        landmarks,
    }
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
        let outputs = self.session.run(&[input])?;
        decode(&outputs, &geometry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FakeSession, FakeSessionHandle};
    use std::time::Instant;

    /// Anchor count of one stride's grid over the shipping input size:
    /// 80x80, 40x40 and 20x20 cells.
    fn anchors(stride: usize) -> usize {
        anchor_count(TENSOR_WIDTH, TENSOR_HEIGHT, stride)
    }

    /// The grid columns one stride has over an arbitrary input width.
    ///
    /// Calls the *production* function rather than repeating its expression: a
    /// copy would keep passing if the decode's own arithmetic were changed,
    /// which is the bug this pair of tests exists to catch.
    fn columns_for(width: usize, stride: usize) -> usize {
        anchor_columns(width, stride)
    }

    /// The grid rows one stride has over an arbitrary input height.
    ///
    /// Calls the *production* function for the same reason as [`columns_for`].
    fn rows_for(height: usize, stride: usize) -> usize {
        anchor_rows(height, stride)
    }

    /// Total anchors across the three grids: 6400 + 1600 + 400.
    const ANCHORS: usize = 8400;

    /// The values one head of one stride carries.
    fn columns(head: &str) -> usize {
        match head {
            "cls" | "obj" => CLASS_COLUMNS,
            "bbox" => BOX_COLUMNS,
            _ => LANDMARK_COLUMNS,
        }
    }

    /// An output tensor of zeroes with the shape a head declares.
    fn zero_head(name: &str, anchors: usize, columns: usize) -> OutputTensor {
        OutputTensor {
            name: name.to_owned(),
            shape: vec![1, anchors as i64, columns as i64],
            data: TensorData::F32(vec![0.0; anchors * columns]),
        }
    }

    /// The twelve outputs, all zeroes: what a session returns for a blank frame.
    fn zero_outputs() -> Vec<OutputTensor> {
        HEAD_NAMES
            .iter()
            .map(|name| {
                let (head, stride) = name
                    .split_once('_')
                    .expect("head names are '<head>_<stride>'");
                let stride = stride.parse::<usize>().expect("head names end in a stride");
                zero_head(name, anchors(stride), columns(head))
            })
            .collect()
    }

    /// Writes one anchor cell's values into the twelve outputs, so a test reads
    /// as the thing the graph predicts rather than as twelve indices.
    #[allow(clippy::too_many_arguments)]
    fn set_cell(
        outputs: &mut [OutputTensor],
        grid_x: usize,
        grid_y: usize,
        stride: usize,
        class: f32,
        objectness: f32,
        // `box` is a reserved word, hence the name.
        deltas: [f32; BOX_COLUMNS],
        landmarks: [(f32, f32); LANDMARKS],
    ) {
        let anchor = grid_y * anchor_columns(TENSOR_WIDTH, stride) + grid_x;
        for output in outputs.iter_mut() {
            let (head, head_stride) = output
                .name
                .split_once('_')
                .expect("head names are '<head>_<stride>'");
            if head_stride.parse::<usize>() != Ok(stride) {
                continue;
            }
            let values = output.data.as_f32_mut().expect("f32 heads");
            match head {
                "cls" => values[anchor] = class,
                "obj" => values[anchor] = objectness,
                "bbox" => {
                    let row = &mut values[anchor * BOX_COLUMNS..][..BOX_COLUMNS];
                    row.copy_from_slice(&deltas);
                }
                "kps" => {
                    for (index, (x, y)) in landmarks.iter().enumerate() {
                        values[anchor * LANDMARK_COLUMNS + 2 * index] = *x;
                        values[anchor * LANDMARK_COLUMNS + 2 * index + 1] = *y;
                    }
                }
                other => panic!("unknown head {other}"),
            }
        }
    }

    /// The raw size that decodes to `size` tensor pixels at `stride`.
    fn raw_size(size: f32, stride: f32) -> f32 {
        (size / stride).ln()
    }

    /// The class head value that yields `score` when objectness is 1.0.
    ///
    /// The confidence is `sqrt(cls * obj)`, so a test that wants a box scoring
    /// 0.9 sets the class head to `0.9 * 0.9` — writing 0.9 there instead would
    /// score `sqrt(0.9) = 0.949`, which is how this helper came to exist.
    fn class_for(score: f32) -> f32 {
        score * score
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
    /// — blue carries the column, green the row, both wrapping at 255.
    ///
    /// Channels that differ per axis are what make a transposed, swapped or
    /// mis-scaled tensor visible, but the wrap is a trap: a test that reads
    /// `blue == 600` from a 1280-wide frame is reading `600 % 256`. Keep the
    /// coordinates a test checks below 256, or use two frames.
    fn coordinate_frame(width: u32, height: u32) -> Frame {
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                bgra.extend_from_slice(&[x as u8, y as u8, 0, 0xff]);
            }
        }
        Frame::new(width, height, bgra, Instant::now()).expect("the buffer matches the dimensions")
    }

    /// A session that always predicts nothing, plus the handle that records
    /// what it was asked to run.
    fn blank_session() -> (Box<dyn InferenceSession>, FakeSessionHandle) {
        let session = FakeSession::new("yunet", zero_outputs());
        let handle = session.handle();
        (Box::new(session), handle)
    }

    #[test]
    fn preprocessing_samples_the_frame_through_the_inverse_map() {
        // The direction of the sampling map is the one thing a value-blind test
        // cannot see, because the two maps are the same arithmetic read either
        // way round and both fill the tensor. So this pins the direction on a
        // frame that scales *up*, with pixel values chosen to stay under 256 so
        // the coordinate encoding does not wrap.
        //
        // A 400x400 frame against the 640x640 tensor scales by 1.6 and fills it
        // exactly, with no padding. Tensor (16, 200) is the position frame
        // (16 / 1.6, 200 / 1.6) = (10, 125) — so the tensor must read blue 10
        // and green 125.
        //
        // Sampling with the frame-to-tensor map instead reads
        // `16 * 1.6 = 25` and `200 * 1.6 = 320` — blue 25, green past the
        // frame's height. Different numbers, so the direction cannot be
        // mistaken for the right answer.
        let frame = coordinate_frame(400, 400);
        let (tensor, geometry) = preprocess(&frame);
        assert_eq!(geometry.scale, 1.6f32, "400 sits inside a 640 tensor");
        assert_eq!(
            (geometry.left, geometry.top),
            (0, 0),
            "the tighter axis fills the tensor, so nothing is padded"
        );
        assert_eq!(geometry.content_width(), 640);
        assert_eq!(geometry.content_height(), 640);

        let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
        let data = tensor.data.as_f32().expect("f32");
        let at = |channel: usize, x: usize, y: usize| data[channel * plane + y * TENSOR_WIDTH + x];
        let channel = |got: f32, want: u16, what: &str| {
            assert!(
                (got * 255.0 - want as f32).abs() < 1e-3,
                "{what}: want {want}/255, got {}",
                got * 255.0
            );
        };
        channel(at(0, 16, 200), 10, "tensor (16, 200) samples frame x = 10");
        channel(at(1, 16, 200), 125, "and frame y = 125");

        // The map divides by the scale, so a step in tensor space is a
        // *fraction* of a frame pixel — and the sampler resolves that fraction
        // by interpolating, which is why these are compared with a tolerance
        // rather than exactly. Two tensor columns are 1.25 frame columns, so
        // the blue between frame columns 10 and 11 reads as 10.25.
        let interpolated = |got: f32, want: f32, what: &str| {
            assert!(
                (got * 255.0 - want).abs() < 0.01,
                "{what}: want {want}/255, got {}",
                got * 255.0
            );
        };
        interpolated(
            at(0, 18, 200),
            11.25,
            "two tensor columns are 1.25 frame columns",
        );
        // Two tensor rows are 1.25 frame rows for the same reason, so this is
        // the half of the map that a scale error would move most.
        interpolated(
            at(1, 16, 202),
            126.25,
            "two tensor rows are 1.25 frame rows",
        );

        // A frame the same size as the tensor neither scales nor pads, so its
        // tensor is its own pixels — the identity case, which a map that was
        // wrong in either direction would still pass. Included so the
        // non-identity cases above cannot be explained away as a scale of 1.
        let exact = coordinate_frame(640, 640);
        let (tensor, geometry) = preprocess(&exact);
        assert_eq!(geometry.scale, 1.0f32);
        assert_eq!((geometry.left, geometry.top), (0, 0));
        let data = tensor.data.as_f32().expect("f32");
        let exact_at =
            |channel: usize, x: usize, y: usize| data[channel * plane + y * TENSOR_WIDTH + x];
        channel(
            exact_at(0, 10, 100),
            10,
            "tensor (10, 100) is frame (10, 100)",
        );
        // One tensor row down is one frame row, so the green advances by one.
        // Read through the plane offset: indexing `data` directly would read
        // the blue channel and assert the wrong pixel.
        channel(
            exact_at(1, 10, 101),
            101,
            "and one tensor row down is one frame row down",
        );
    }

    #[test]
    fn preprocessing_scales_to_unit_range_and_preserves_aspect() {
        // A 4x2 frame: exactly twice as wide as it is tall, in a square tensor.
        // Preserving the aspect means 640x320 of content with 160 blank rows
        // above and below — a stretch would have filled all 640.
        let frame = coordinate_frame(4, 2);
        let (tensor, geometry) = preprocess(&frame);

        assert_eq!(tensor.name, "input");
        assert_eq!(
            tensor.shape,
            vec![1, 3, 640, 640],
            "NCHW at the shipping model's fixed input size"
        );
        let data = tensor.data.as_f32().expect("an f32 tensor");
        assert_eq!(data.len(), 3 * TENSOR_WIDTH * TENSOR_HEIGHT);

        // The geometry the decode inverts must be the one preprocessing used:
        // this frame scales up by 320, which fills the tensor's width exactly
        // and leaves 160 rows above and below.
        assert_eq!(geometry.scale, 160.0f32);
        assert_eq!(geometry.left, 0, "the width fills the tensor exactly");
        assert_eq!(
            geometry.top, 160,
            "320 rows of content leave 160 either side"
        );
        assert_eq!((geometry.frame_width, geometry.frame_height), (4, 2));

        let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
        let at = |channel: usize, x: usize, y: usize| data[channel * plane + y * TENSOR_WIDTH + x];

        // The padding is zero, in every plane. These rows are the only ones
        // outside the content, and a non-zero pad value is what would make the
        // graph see a false edge at the letterbox boundary.
        for row in [0, 159, 480, 639] {
            for x in [0, 1, 320, 638, 639] {
                assert_eq!(at(0, x, row), 0.0, "blue padding at ({x}, {row})");
                assert_eq!(at(1, x, row), 0.0, "green padding at ({x}, {row})");
                assert_eq!(at(2, x, row), 0.0, "red padding at ({x}, {row})");
            }
        }

        // Scaled by 1/255, from the source pixel the sample lands in. Tensor
        // row 160 is the content's first row and maps to frame row 0; tensor
        // column 639 maps to frame x = 3.99, whose blue is 3. Those two pin the
        // scale and the channel order together.
        let channel = |got: f32, want: u8, what: &str| {
            assert!(
                (got - f32::from(want) / 255.0).abs() < 1e-6,
                "{what}: want {want}/255, got {got}"
            );
        };
        channel(at(0, 639, 160), 3, "the content starts at tensor row 160");
        channel(at(1, 639, 160), 0, "and that row is frame row 0");
        assert_eq!(at(0, 639, 159), 0.0, "row 159 is padding");
        // The last content row is 479, so 480 is padding again. This is the
        // assertion a content band of the wrong height fails.
        channel(at(1, 639, 479), 1, "the content's last row is 479");
        assert_eq!(at(0, 639, 480), 0.0, "and row 480 is padding");

        // A solid frame's planes are in the documented order. Only a frame
        // whose channels differ can tell blue from red, so this is the channel
        // identity the coordinate frame above cannot assert.
        let (solid, _) = preprocess(&solid_frame(4, 2, [10, 20, 30]));
        let solid = solid.data.as_f32().expect("an f32 tensor");
        let row = 160 * TENSOR_WIDTH;
        channel(solid[row], 10, "plane 0 is blue");
        channel(solid[plane + row], 20, "plane 1 is green");
        channel(solid[2 * plane + row], 30, "plane 2 is red");

        // `to_frame` and `to_tensor` are exact algebraic inverses, and that is
        // worth asserting even though only `to_frame` is used by the crate:
        // `preprocess` samples with it and `decode` inverts boxes with it, so if
        // it were not the inverse of a well-defined forward map the two would
        // disagree about where the frame *is*, and every box would be displaced
        // by however much they differed.
        //
        // The composition asserted is `to_frame` after `to_tensor`, which is the
        // pair that composes: `to_tensor` maps a frame position into the tensor
        // and `to_frame` brings it back. (Separately, sampling needs the frame
        // pixel *centre* a tensor pixel covers, which is half a pixel off
        // `to_frame` — that is the resampler's own convention and is pinned by
        // the pixel-value assertions above, not here.)
        for point in [(0.0, 0.0), (3.0, 1.0), (1.5, 0.5), (2.75, 1.25)] {
            let (tensor_x, tensor_y) = geometry.to_tensor(point.0, point.1);
            let (back_x, back_y) = geometry.to_frame(tensor_x, tensor_y);
            assert!(
                (back_x - point.0).abs() < 1e-3 && (back_y - point.1).abs() < 1e-3,
                "frame {point:?} -> tensor ({tensor_x}, {tensor_y}) -> ({back_x}, {back_y})"
            );
        }

        // The same round trip through a *padded* geometry, where the offsets are
        // non-zero on both axes — the composition the square case above cannot
        // distinguish from an unpadded one.
        let portrait = letterbox_geometry(480, 640);
        for point in [(0.0, 0.0), (479.0, 639.0), (120.0, 300.0)] {
            let (tensor_x, tensor_y) = portrait.to_tensor(point.0, point.1);
            let (back_x, back_y) = portrait.to_frame(tensor_x, tensor_y);
            assert!(
                (back_x - point.0).abs() < 1e-3 && (back_y - point.1).abs() < 1e-3,
                "frame {point:?} -> tensor ({tensor_x}, {tensor_y}) -> ({back_x}, {back_y})"
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
        // A 4x2 frame letterboxes with `scale = 160` and `top = 160`, so the
        // tensor-to-frame map is `x / 160` and `(y - 160) / 160`.
        //
        // Cell (10, 10) of the stride-8 grid anchors at tensor (80, 80). With
        // `dx = 7.5` the box's centre lands at `(10 + 7.5) * 8 = 140` across
        // and with `dy = 15.0` at 200 down, so the centre is frame
        // (0.875, 0.25) and a 40x50 tensor box is 0.25x0.3125 frame pixels.
        let mut outputs = zero_outputs();
        set_cell(
            &mut outputs,
            10,
            10,
            8,
            class_for(0.9),
            1.0,
            [7.5, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [
                (2.5, 7.5), // right eye: 100, 140 tensor -> frame (0.625, 0.25)
                (5.0, 7.5),
                (7.5, 7.5),
                (10.0, 7.5),
                (12.5, 7.5),
            ],
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&outputs, &geometry).expect("a well-formed output decodes");
        assert_eq!(faces.len(), 1, "one strong cell is one face");
        let face = faces[0];
        assert!(
            (face.score - 0.9).abs() < 1e-6,
            "the confidence is sqrt(cls * obj) of two activated heads: {}",
            face.score
        );
        let rect = face.rect;
        assert!(
            (rect.x - 0.75).abs() < 1e-3 && (rect.w - 0.25).abs() < 1e-3,
            "left = 0.875 - 0.125, width = 40/160: {rect:?}"
        );
        assert!(
            (rect.y - 0.09375).abs() < 1e-3 && (rect.h - 0.3125).abs() < 1e-3,
            "top = 0.25 - 0.15625, height = 50/160: {rect:?}"
        );

        // The landmarks come through the same inverse transform, from the same
        // anchor, and in YuNet's own order. Their tensor row is 140, which is
        // *above* the content band (which starts at 160), so frame y comes out
        // at -0.125 before the frame clamp floors it at 0. That is not a
        // contrivance: a landmark's offset is in cells, and at this scale one
        // cell is 8 tensor rows, so a small vertical delta puts a landmark in
        // the padding even though its box is well inside the frame.
        let expected = [
            (0.625, 0.0),
            (0.75, 0.0),
            (0.875, 0.0),
            (1.0, 0.0),
            (1.125, 0.0),
        ];
        for (index, (want, got)) in expected.iter().zip(face.landmarks).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-3 && (got.1 - want.1).abs() < 1e-3,
                "landmark {index}: want {want:?}, got {got:?}"
            );
        }

        // Every rect is inside the frame — here a 4x2 window, which a box
        // reaching into the padding would overflow.
        for face in &faces {
            let rect = face.rect;
            assert!(rect.x >= 0.0 && rect.y >= 0.0, "{rect:?} starts inside");
            assert!(
                rect.x + rect.w <= 4.0 && rect.y + rect.h <= 2.0,
                "{rect:?} ends inside the 4x2 frame"
            );
            assert!(rect.w >= 0.0 && rect.h >= 0.0, "{rect:?} has no flip");
        }

        // A frame that does not need scaling is *not* trimmed: the clamp must
        // cut to the image, not shrink every box towards its centre. 640x480 is
        // narrower than the square tensor, so it scales by 1.0 and is padded 80
        // rows top and bottom — the tensor box's own coordinates, shifted by
        // that padding and otherwise untouched.
        let roomy = letterbox_geometry(640, 480);
        let roomy_faces = decode(&outputs, &roomy).expect("decodes");
        let rect = roomy_faces[0].rect;
        assert!(
            (rect.x - 120.0).abs() < 1e-3
                && (rect.y - 95.0).abs() < 1e-3
                && (rect.w - 40.0).abs() < 1e-3
                && (rect.h - 50.0).abs() < 1e-3,
            "the untrimmed box, shifted by the vertical padding: {rect:?}"
        );
    }

    #[test]
    fn decode_reads_all_three_grids() {
        // The twelve outputs hold three feature grids, and the decode walks each
        // with its own stride. A decode that lost track of which grid it was
        // reading — or that conflated `cls_N` with `obj_N`, the two one-column
        // heads — reads a whole class of anchors at the wrong place.
        let mut outputs = zero_outputs();

        // Cell (5, 5) at stride 16 anchors at tensor (80, 80). Deltas of 2.5
        // cells reach `(5 + 2.5) * 16 = 120`, so the centre is tensor
        // (120, 120), frame (0.75, -0.25) before clamping; an 8x10 tensor box
        // is 0.05x0.0625 frame pixels.
        set_cell(
            &mut outputs,
            5,
            5,
            16,
            class_for(0.8),
            1.0,
            [2.5, 2.5, raw_size(8.0, 16.0), raw_size(10.0, 16.0)],
            [(0.0, 0.0); LANDMARKS],
        );
        // Cell (2, 2) at stride 32 anchors at tensor (64, 64). Deltas of 4.0
        // cells reach `(2 + 4) * 32 = 192` on *both* axes, so the centre is
        // tensor (192, 192), frame (1.2, 0.2), with a 32x32 tensor box — 0.2x0.2
        // frame pixels.
        set_cell(
            &mut outputs,
            2,
            2,
            32,
            class_for(0.7),
            1.0,
            [4.0, 4.0, raw_size(32.0, 32.0), raw_size(32.0, 32.0)],
            [(0.0, 0.0); LANDMARKS],
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&outputs, &geometry).expect("decodes");
        assert_eq!(
            faces.len(),
            2,
            "one cell per coarse grid, each decoded once: {faces:?}"
        );
        assert_eq!(
            faces.iter().map(|face| face.score).collect::<Vec<_>>(),
            vec![0.8, 0.7],
            "strongest first, across grids"
        );

        // The stride-16 cell's box is entirely in the top padding — tensor rows
        // 115..125 against a content band starting at 160 — so it trims away to
        // nothing, which is the honest answer: a box above the frame covers no
        // pixels of it. The stride-32 cell straddles the frame's first row, so
        // its top is clamped to 0 and its lower half survives.
        let want = [(0.725, 0.0, 0.05, 0.0), (1.1, 0.1, 0.2, 0.2)];
        for (face, want) in faces.iter().zip(want) {
            let rect = face.rect;
            assert!(
                (rect.x - want.0).abs() < 1e-3
                    && (rect.y - want.1).abs() < 1e-3
                    && (rect.w - want.2).abs() < 1e-3
                    && (rect.h - want.3).abs() < 1e-3,
                "want {want:?}, got {rect:?}"
            );
        }

        // The anchor grids tile the tensor, 8400 anchors in total. A derivation
        // that disagreed with the model's would mis-index every head at once.
        //
        // Worth saying plainly: at 640x640 every stride divides the input
        // exactly, so `div_ceil` and truncating division agree here and *this
        // assertion cannot tell them apart*. It is written with `div_ceil`
        // because the alternative silently drops the last partial row for any
        // input size that does not divide — a 240-row input at stride 32 is 7
        // rows truncated and 8 with a ceiling — and the test below is what would
        // catch it at a size where the two differ.
        assert_eq!(anchors(8) + anchors(16) + anchors(32), ANCHORS);
        assert_eq!([anchors(8), anchors(16), anchors(32)], [6400, 1600, 400]);
        assert_eq!(
            rows_for(240, 32),
            8,
            "a height of 240 is 7 whole cells of 32 plus 16 rows, so the grid \
             has 8 rows — truncating here loses the anchors covering the \
             bottom strip of the frame"
        );
        assert_eq!(rows_for(240, 8), 30);
        assert_eq!(
            columns_for(240, 32),
            8,
            "a width of 240 is 7 whole cells of 32 plus 16 columns, so the grid \
             has 8 columns — the same ceiling on the other axis, and the one a \
             truncating `anchor_columns` would get wrong"
        );
        assert_eq!(columns_for(240, 8), 30);
        assert_eq!(
            columns_for(TENSOR_WIDTH, 32),
            20,
            "and at the shipping size the two agree, which is why the \
             assertion above is the one that does the work"
        );
        assert_eq!(
            rows_for(TENSOR_HEIGHT, 32),
            20,
            "and at the shipping size the two agree, which is why the \
             assertion above is the one that does the work"
        );
    }

    #[test]
    fn a_box_whose_origin_lands_in_the_padding_is_trimmed_at_the_top() {
        // Padding exists above the content, so a box can decode to an origin
        // *above* the frame. Cell (2, 1) of the stride-8 grid anchors at tensor
        // (16, 8); with `dx = 8` and `dy = -11` the centre is `(2 + 8) * 8 = 80`
        // across and `(1 - 11) * 8 = -80` down, and a 40x320 tensor box spans
        // y = -240..80. Clamped in tensor space the top becomes 0, and the
        // bottom is 80 — both inside the tensor, and both *above* the content
        // band, which does not start until tensor row 160.
        //
        // The inverse map sends tensor row 0 to frame `(0 - 160) / 160 = -1`:
        // outside a 2-pixel-tall frame, and negative. This is the case the
        // frame-space clamp exists for, and the reason the tensor-space clamp
        // alone is not enough — it bounds a box to the whole *tensor*, and the
        // padding is inside the tensor.
        let mut outputs = zero_outputs();
        set_cell(
            &mut outputs,
            2,
            1,
            8,
            class_for(0.9),
            1.0,
            [8.0, -11.0, raw_size(40.0, 8.0), raw_size(320.0, 8.0)],
            [(0.0, -11.0); LANDMARKS],
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&outputs, &geometry).expect("decodes");
        assert_eq!(faces.len(), 1);
        let rect = faces[0].rect;
        assert!(
            rect.y == 0.0 && rect.h >= 0.0,
            "a box whose origin is in the padding is trimmed to the frame's \
             first row, not left with a negative origin or height: {rect:?}"
        );
        // Tensor x spans 60..100, so frame x is 0.375..0.625: a 0.25-wide box
        // with its left edge untouched by the vertical trim.
        assert!(
            (rect.x - 0.375).abs() < 1e-3 && (rect.w - 0.25).abs() < 1e-3,
            "its horizontal extent is untouched by the vertical trim: {rect:?}"
        );
        assert!(
            rect.y + rect.h <= 2.0,
            "and it does not run past the frame either: {rect:?}"
        );

        // The landmarks were written with `dy = -11`, whose tensor row is -80 —
        // above the tensor's first row, so the sampler's own clamp puts them at
        // row 0 and the frame clamp then floors them at frame 0. Without the
        // frame clamp they would come back at -1.
        assert_eq!(
            faces[0].landmarks[0].1, 0.0,
            "the padding landmark is clamped to the frame's top edge, not left \
             at the negative coordinate the raw map gives"
        );
        for (index, (x, y)) in faces[0].landmarks.iter().enumerate() {
            assert!(
                *x >= 0.0 && *y >= 0.0 && *x <= 4.0 && *y <= 2.0,
                "landmark {index} must be inside the 4x2 frame, got ({x}, {y})"
            );
        }
    }

    #[test]
    fn a_box_reaching_past_the_frame_edge_is_trimmed_to_it() {
        // Cell (30, 20) of the stride-8 grid anchors at tensor (240, 160).
        // Deltas of 5.0 cells put the centre at `(30 + 5) * 8 = 280` across and
        // `(20 + 5) * 8 = 200` down; its 80x160 tensor box runs past the
        // tensor's right and bottom edges and is trimmed there, landing at
        // frame (240, 40)..(320, 200) in a 640x480 frame — which is inside the
        // frame, because at scale 1.0 the tensor clamp and the frame clamp
        // agree.
        let mut outputs = zero_outputs();
        set_cell(
            &mut outputs,
            30,
            20,
            8,
            class_for(0.9),
            1.0,
            [5.0, 5.0, raw_size(80.0, 8.0), raw_size(160.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );

        let geometry = letterbox_geometry(640, 480);
        let faces = decode(&outputs, &geometry).expect("decodes");
        assert_eq!(faces.len(), 1);
        let rect = faces[0].rect;
        assert!(
            (rect.x - 240.0).abs() < 1e-3 && (rect.y - 40.0).abs() < 1e-3,
            "the box starts inside the frame, where it was decoded: {rect:?}"
        );
        assert!(
            (rect.w - 80.0).abs() < 1e-3 && (rect.h - 160.0).abs() < 1e-3,
            "and it keeps the size the tensor box scaled to: {rect:?}"
        );
        assert!(
            rect.x >= 0.0 && rect.y >= 0.0 && rect.x + rect.w <= 640.0 && rect.y + rect.h <= 480.0,
            "nothing may extend past the frame: {rect:?}"
        );
    }

    #[test]
    fn suppression_collapses_overlapping_anchors_but_keeps_distant_ones() {
        // Two neighbouring cells decode to two boxes on one face, and a third
        // far away is a second face. Without the `nms` call there would be three
        // faces here, which downstream would read as three people.
        //
        // Decoded against the 4x2 letterbox, where `scale = 160`.
        let mut outputs = zero_outputs();
        // Cell (10, 10): centre tensor (140, 200), box 40x50 -> frame
        // (0.75, 0.09375, 0.25, 0.3125).
        set_cell(
            &mut outputs,
            10,
            10,
            8,
            class_for(0.9),
            1.0,
            [7.5, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );
        // Cell (11, 10): the anchor moves 8 tensor pixels right but `dx = 5.0`
        // instead of 7.5 pulls the centre 20 tensor pixels *left*, so this box
        // starts 12 tensor pixels left of the first — frame x 0.675 against
        // 0.75 — with the same 40x50 size. They share 28 of 40 columns and all
        // 50 rows, so the intersection is 28*50 and the union is 2*2000 less it:
        // an IoU of 0.538, comfortably over the 0.3 threshold.
        set_cell(
            &mut outputs,
            11,
            10,
            8,
            class_for(0.9),
            1.0,
            [5.0, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );
        // Cell (20, 15): centre tensor (200, 160), box 40x50 -> frame
        // (1.125, 0, 0.25, 0.3125) before the clamp, which trims the part above
        // the frame's first row. It shares no pixels with either of the others
        // — a `nms` that kept only the strongest would fail here.
        set_cell(
            &mut outputs,
            20,
            15,
            8,
            class_for(0.9),
            1.0,
            [5.0, 5.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );

        let geometry = letterbox_geometry(4, 2);
        let faces = decode(&outputs, &geometry).expect("decodes");
        assert_eq!(
            faces.len(),
            2,
            "two overlapping boxes are one face and the distant one is another: {faces:?}"
        );
        // Which two survived is the part that matters, and the scores cannot
        // say — all three are 0.9 — so the boxes do.
        let mut boxes: Vec<(i32, i32, i32, i32)> = faces
            .iter()
            .map(|face| {
                let rect = face.rect;
                (
                    (rect.x * 1000.0).round() as i32,
                    (rect.y * 1000.0).round() as i32,
                    (rect.w * 1000.0).round() as i32,
                    (rect.h * 1000.0).round() as i32,
                )
            })
            .collect();
        boxes.sort_unstable();
        assert_eq!(
            boxes,
            vec![(750, 94, 250, 313), (1125, 0, 250, 156)],
            "the second cell's box was suppressed and both faces survived"
        );
    }

    #[test]
    fn faces_come_back_strongest_first_not_in_scan_order() {
        // `YuNet::detect` documents "strongest first" and `nms` returns that
        // order, so a caller taking `faces[0]` gets the most confident face.
        //
        // The trap this pins is a scan-order bug that every other test is blind
        // to, because in all of them grid order and score order happen to
        // coincide. The candidates are collected in *grid* order — stride 8 row
        // by row, then 16, then 32 — so here the cell the scan reaches first
        // scores 0.7 and a later one scores 0.9. Returning them in the order
        // they were collected would put 0.7 first.
        let mut outputs = zero_outputs();
        // Cell (1, 1) is the third anchor the stride-8 scan visits, and it is
        // the *weaker* of the two.
        set_cell(
            &mut outputs,
            1,
            1,
            8,
            class_for(0.7),
            1.0,
            [7.5, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );
        // Cell (10, 10) is visited hundreds of anchors later and is stronger.
        set_cell(
            &mut outputs,
            10,
            10,
            8,
            class_for(0.9),
            1.0,
            [7.5, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
            [(0.0, 0.0); LANDMARKS],
        );

        let geometry = letterbox_geometry(640, 480);
        let faces = decode(&outputs, &geometry).expect("decodes");
        assert_eq!(faces.len(), 2, "two cells, two faces: {faces:?}");
        assert_eq!(
            faces.iter().map(|face| face.score).collect::<Vec<_>>(),
            vec![0.9, 0.7],
            "the later-scanned cell scored higher and must still come first"
        );
        assert!(
            faces[0].rect.x > faces[1].rect.x,
            "and it is genuinely the other face, not the same one twice: {:?}",
            faces.iter().map(|face| face.rect).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_portrait_frame_fills_the_tensor_without_smearing_its_edge() {
        // Every other fixture is landscape or square, and that is why the bug
        // this test exists for survived: when `width < height` the letterbox
        // pads *horizontally*, so `left > 0`, and the content band has to carry
        // that offset on both axes.
        //
        // Writing the columns as `0..content_width` instead fills the left
        // padding with a smeared copy of the frame's first column and leaves the
        // right-hand strip of the frame unsampled — invisible content loss,
        // because `decode` inverts the same map and the coordinates stay
        // self-consistent.
        //
        // A 480x640 frame against the square tensor scales by 1.0 (the height is
        // already 640) and is padded 80 columns either side.
        let frame = coordinate_frame(480, 640);
        let (tensor, geometry) = preprocess(&frame);
        assert_eq!(geometry.scale, 1.0f32);
        assert_eq!(
            (geometry.left, geometry.top),
            (80, 0),
            "the *narrow* axis is the one that gets padded"
        );
        assert_eq!(geometry.content_width(), 480);
        assert_eq!(geometry.content_height(), 640);

        let plane = TENSOR_WIDTH * TENSOR_HEIGHT;
        let data = tensor.data.as_f32().expect("f32");
        let at = |channel: usize, x: usize, y: usize| data[channel * plane + y * TENSOR_WIDTH + x];
        let channel = |got: f32, want: u16, what: &str| {
            assert!(
                (got * 255.0 - want as f32).abs() < 1e-3,
                "{what}: want {want}/255, got {}",
                got * 255.0
            );
        };

        // The left padding is genuinely zero, not a copy of frame column 0.
        // This is the assertion the un-offset loop fails: it samples `to_frame(0,
        // y) = -80`, which clamps to frame column 0, so every one of these
        // columns would hold a replica of the frame's first column.
        for x in [0, 1, 40, 79] {
            assert_eq!(at(0, x, 0), 0.0, "left padding column {x} is zero");
            assert_eq!(at(1, x, 0), 0.0, "left padding column {x} is zero");
        }
        // The content starts exactly where the padding ends, and the frame's
        // own first column lands there.
        channel(at(0, 80, 0), 0, "tensor column 80 is frame column 0");
        channel(at(1, 80, 0), 0, "and frame row 0");
        // Inside the first 256 frame columns the fixture does not wrap, so these
        // read their own column number.
        channel(at(0, 200, 0), 120, "tensor column 200 is frame column 120");
        channel(
            at(0, 335, 0),
            255,
            "and tensor column 335 is frame column 255",
        );

        // The frame's *right-hand* strip reaches the tensor. This is the
        // assertion the un-offset loop fails: it stops at tensor column 479 —
        // frame column 399 — so columns 480 onward are never written and read 0.
        // Past frame column 255 the fixture's blue byte wraps (479 % 256 = 223),
        // so the numbers below are the wrapped ones; what matters is that they
        // are the *right* wrapped numbers rather than zero.
        channel(at(0, 480, 0), 144, "tensor column 480 is frame column 400");
        channel(
            at(0, 559, 0),
            223,
            "and the last content column is frame x = 479, wrapped",
        );
        // Past the content is padding again, and zero.
        assert_eq!(at(0, 560, 0), 0.0, "the right padding is zero");
        assert_eq!(at(0, 639, 0), 0.0, "out to the tensor's last column");

        // The frame's last row comes through too, so the vertical band is not
        // the thing compensating for a horizontal mistake. These read their rows
        // modulo 256, the fixture being byte-valued: 639 % 256 = 127.
        channel(at(1, 80, 200), 200, "frame row 200 reads as itself");
        channel(at(1, 80, 383), 127, "frame row 383 reads as itself");
        channel(
            at(1, 80, 639),
            127,
            "and the last content row is 639, wrapped",
        );
    }

    #[test]
    fn the_score_is_the_geometric_mean_of_two_activated_heads() {
        // `cls_*` and `obj_*` each end in a Sigmoid inside the graph, so the
        // confidence is `sqrt(cls * obj)` — the combination OpenCV's own
        // post-processing uses. This pins all three parts of that: that neither
        // head is activated a second time, that the two are combined
        // multiplicatively rather than added, and that the gate's boundary
        // admits a score exactly on it.
        let geometry = letterbox_geometry(640, 480);
        let cell = |class: f32, objectness: f32| {
            let mut outputs = zero_outputs();
            set_cell(
                &mut outputs,
                10,
                10,
                8,
                class,
                objectness,
                [7.5, 15.0, raw_size(40.0, 8.0), raw_size(50.0, 8.0)],
                [(0.0, 0.0); LANDMARKS],
            );
            decode(&outputs, &geometry).expect("decodes")
        };

        // A confident pair. A second sigmoid would report 0.71 here instead of
        // 0.9, and a sum would report 1.8 and then clamp.
        let faces = cell(0.9, 0.9);
        assert_eq!(faces.len(), 1);
        assert!(
            (faces[0].score - 0.9).abs() < 1e-6,
            "sqrt(0.9 * 0.9) is 0.9, not a re-activated or summed value: {}",
            faces[0].score
        );

        // Mismatched heads: 0.81 and 0.4444 score sqrt(0.36) = 0.6, exactly the
        // threshold, so this pins the geometric mean *and* pins the gate as
        // inclusive. A sum would score 1.25 and a second sigmoid 0.777, so a
        // wrong combination changes whether this box survives at all.
        let faces = cell(0.81, 0.36 / 0.81);
        assert_eq!(
            faces.len(),
            1,
            "sqrt(0.81 * 0.4444) is exactly {}, which the gate admits",
            SCORE_THRESHOLD
        );
        assert!(
            (faces[0].score - SCORE_THRESHOLD).abs() < 1e-6,
            "got {}",
            faces[0].score
        );

        // A head at zero is no face whatever the other says: sqrt(0.9 * 0) = 0.
        assert!(
            cell(0.9, 0.0).is_empty(),
            "one silent head means no face, not half a face"
        );
        assert!(cell(0.0, 0.9).is_empty());

        // Just under, and just over.
        assert!(
            cell(0.81, 0.43).is_empty(),
            "sqrt(0.81 * 0.43) = 0.59, below the gate"
        );
        assert_eq!(cell(0.81, 0.45).len(), 1, "and 0.604 is above it");
    }

    #[test]
    fn decode_rejects_an_output_it_cannot_read() {
        // Every one of these is a real way a session can be wrong: a different
        // graph, a mis-selected output, an integer tensor, a truncated copy.
        // None of them may panic — this function returns a `Result`.
        let geometry = letterbox_geometry(640, 480);

        // A short buffer under a correct shape is the case a shape check alone
        // cannot see, and the one that would read past the end of the data.
        let mut truncated = zero_outputs();
        truncated[6].data = TensorData::F32(vec![0.0; 10]);
        assert!(matches!(
            decode(&truncated, &geometry),
            Err(ModelError::UnexpectedDataLength {
                name: "bbox_8",
                got: 10,
                ..
            })
        ));

        // The same, for a head the decode only reads when a cell clears the
        // gate — so a short buffer is caught by validation rather than by
        // whether any anchor happened to fire.
        let mut short_landmarks = zero_outputs();
        short_landmarks[9].data = TensorData::F32(vec![0.0; anchors(8) * LANDMARK_COLUMNS - 1]);
        assert!(matches!(
            decode(&short_landmarks, &geometry),
            Err(ModelError::UnexpectedDataLength { name: "kps_8", .. })
        ));

        // The right *column* count with the wrong row count: a graph exported
        // over a different input size, or a stride set this decode does not
        // share. 1580 is deliberately a plausible number — it is what a 320x240
        // input produced in an earlier revision of this file — so this is the
        // case where a stale constant would otherwise slip through unnoticed.
        let mut wrong_shape = zero_outputs();
        wrong_shape[0].shape = vec![1, 1580, CLASS_COLUMNS as i64];
        assert!(matches!(
            decode(&wrong_shape, &geometry),
            Err(ModelError::UnexpectedShape { name: "cls_8", .. })
        ));

        let mut renamed = zero_outputs();
        renamed[0].name = "classes_8".to_owned();
        assert!(matches!(
            decode(&renamed, &geometry),
            Err(ModelError::UnexpectedOutput {
                expected: "cls_8",
                ..
            })
        ));

        let mut wrong_type = zero_outputs();
        wrong_type[9].data = TensorData::I32(vec![0; anchors(8) * LANDMARK_COLUMNS]);
        assert!(matches!(
            decode(&wrong_type, &geometry),
            Err(ModelError::MissingOutput { name: "kps_8" })
        ));

        // A session that returns the wrong number of tensors at all.
        let mut too_few = zero_outputs();
        too_few.truncate(OUTPUT_COUNT - 1);
        assert!(matches!(
            decode(&too_few, &geometry),
            Err(ModelError::UnexpectedOutputCount {
                expected: 12,
                got: 11
            })
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
            "both heads at zero is a confidence of 0.0, not a face: {faces:?}"
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
        assert_eq!(inputs[0].shape, vec![1, 3, 640, 640]);
        assert_eq!(
            inputs[0].data.as_f32().expect("f32").len(),
            3 * TENSOR_WIDTH * TENSOR_HEIGHT,
            "the shape and the data must agree"
        );

        // A session that returns nothing at all is an error, not an empty
        // result: "no faces" and "the model did not answer" are different facts
        // and the engine must be able to tell them apart.
        let empty = YuNet::new(Box::new(FakeSession::new("yunet", Vec::new())));
        assert!(matches!(
            empty.detect(&solid_frame(640, 480, [0; 3])),
            Err(ModelError::UnexpectedOutputCount {
                expected: 12,
                got: 0
            })
        ));
    }
}
