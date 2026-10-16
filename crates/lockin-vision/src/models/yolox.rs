//! Person and phone detection: the `phone` signal's producer, stage 3 of the
//! engine.
//!
//! This is the only detector in the crate whose output the whole product is
//! built around. Task 12 turns these boxes into a `PhoneSignal`, and every
//! downstream judgement about "is the user on their phone" is a function of
//! what this wrapper returns. A missed phone is a silently absent signal; a
//! spurious one is a false accusation. There is no third outcome to hide in.
//!
//! # The graph, read rather than assumed
//!
//! `yolox_nano.onnx` (Megvii-BaseDetection/YOLOX release `0.1.1rc0`, 3,659,407
//! bytes, sha256 `c789161e…a0b7d`, sha384 `2cbb5b18…8031e`) was read directly
//! with `onnx` — and then *run* — to write this wrapper:
//!
//! | tensor | name | shape | dtype |
//! |---|---|---|---|
//! | input | `images` | `[1, 3, 416, 416]` | f32 |
//! | output | `output` | `[1, 3549, 85]` | f32 |
//!
//! ir_version 6, producer `pytorch 1.7`, **opset 11**, 369 nodes, 236
//! initializers, 368 `value_info` entries, no functions.
//!
//! ## 3549 anchors, derived rather than trusted
//!
//! The graph's tail is `Reshape(…, [1, 85, -1])` three times, once per stride,
//! then `Concat` then `Transpose` into `output`. The three strides are the
//! usual YOLOX ones, and the anchor count is their sum — not a number this
//! module takes on faith:
//!
//! ```text
//! (416 /  8)^2 = 52^2 = 2704
//! (416 / 16)^2 = 26^2 =  676
//! (416 / 32)^2 = 13^2 =  169
//!                        ----
//!                        3549
//! ```
//!
//! The rows are grouped in **descending resolution order** — the 2704 stride-8
//! rows first, then 676 stride-16, then 169 stride-32 — which is what
//! `row_stride` encodes and what the brief's flat "per row of `[1, N, 85]`"
//! leaves implicit. Getting this order wrong maps every anchor to the wrong
//! grid cell and produces boxes that are geometrically plausible and entirely
//! wrong, which is the failure mode with no downstream symptom.
//!
//! The 85 columns are `[cx, cy, w, h, obj, cls_0 … cls_79]`, read off the
//! graph's three head convolutions per scale: a 4-channel `bbox` conv, a
//! 1-channel `obj` conv, and an 80-channel `cls` conv, concatenated in that
//! order (`Concat(['1146', '1148', '1149'])` for the stride-32 branch and its
//! two siblings).
//!
//! # R38: what the graph activates, and what that costs
//!
//! **The brief says "fully raw with no sigmoid in the graph". The graph has 110
//! `Sigmoid` nodes, and `obj` and `cls` are behind them.** Only `xy` and `wh`
//! leave the graph un-activated. Measured, not inferred:
//!
//! | head | node | leaves the graph as |
//! |---|---|---|
//! | `bbox` (4ch) | `Conv` only | **raw** — `wh` reaches −0.263 on a real image, which no `exp` output can |
//! | `obj` (1ch) | `Conv → Sigmoid` | **already σ** |
//! | `cls` (80ch) | `Conv → Sigmoid` | **already σ** |
//!
//! So the decode in the brief — `score = sigmoid(obj) * sigmoid(cls)` — applies
//! a **second** sigmoid to both terms. That is not a free no-op:
//!
//! * `sigmoid` maps `[0, 1]` into `[0.5, 0.731]`, so the score is compressed
//!   into `[0.25, 0.534]` and can **never exceed 0.534**. The brief's
//!   `score_thr = 0.1` therefore rejects nothing: measured on a COCO image,
//!   all 3549 rows survive the threshold and 1624 boxes survive NMS.
//! * Per-row *class selection* is unaffected — `sigmoid` is monotone, so the
//!   argmax over classes is the same either way — and the decoded *boxes* are
//!   unaffected, since `xy`/`wh` are genuinely raw.
//!
//! This module implements the brief's formula **as specified**, because the
//! interface is Task 12's contract and a wrapper must not silently redefine the
//! numbers its consumer was written against. Two consequences are recorded here
//! so that whoever tunes Task 12 does not have to rediscover them:
//!
//! * the score ceiling is 0.534, so no threshold below it filters anything;
//! * the column layout is `[cx, cy, w, h, obj, cls_0 … cls_79]` with `cls`
//!   already a probability, so switching to `obj * cls` needs no other edit.
//!
//! ## The `obj` head is degenerate in this export, and that is not this
//! ## module's bug
//!
//! Run on three COCO val images, the `obj` head — *after* the graph's own
//! `Sigmoid` — never exceeds **0.0107**, with a mean of 8.96e-05 and a 99th
//! percentile of 2.47e-03. A working objectness head puts confident rows near
//! 0.9. Its weights are present and non-zero (`Conv` `623` is
//! `[1, 64, 1, 1]`, |W| min 9.5e-04, std 0.21, bias −0.130; `619` bias −2.16),
//! so nothing is missing — the head simply outputs ≈0 for everything.
//!
//! Which means `sigmoid(≈0) ≈ 0.5`: the brief's first factor is a **constant
//! 0.5** on real input, every row scores ≈ half its class probability, and the
//! 0.1 floor is unreachable in the other direction too. Both error modes of a
//! broken `objectness` — flood and silence — are present at once, and no
//! arithmetic in this file can fix either. This wrapper's job is to be exactly
//! as correct as the model allows and to say so; `preprocess`, `decode_row` and
//! [`YoloxNano::detect`] are each verified against the graph above and against
//! the COCO images in `research/yolox/`.
//!
//! The class head is *not* degenerate: its logits reach 0.76 on the same image,
//! and the classes it picks are the right ones — `person` and `cat` on a photo
//! of two cats on a sofa. The boxes are sane as well: `w`/`h` in log space sit
//! at 1.15–1.42, which is boxes of three to four cells. Only `obj` is broken,
//! and a caller that wants to work around it can rank by the class probability
//! alone.
//!
//! # Preprocessing is this model's own, and is not shared
//!
//! BGR, `0–255` with **no** division, letterboxed to a 32-multiple at 416×416
//! with pad value **114**, then `(x - mean) / std` with mean
//! `[103.53, 116.28, 123.675]` and std `[57.375, 57.12, 58.395]`. All four
//! parts differ from every other wrapper in this crate and none of them may be
//! folded into a shared helper: YuNet letterboxes and scales by `1/255` with no
//! mean, Face Mesh reverses to RGB and scales by `1/255` with no mean, and the
//! eye classifier subtracts 127 and divides by 255 to a square with no
//! letterbox at all. The `114` is YOLOX's own padding constant, not a grey
//! derived from the mean.

use crate::analyze::nms::{Detection, Rect, nms};
use crate::frame::Frame;
use crate::models::{InferenceSession, InputTensor, ModelError, OutputTensor, TensorData};

/// The graph's image input name, exactly as the file declares it.
///
/// `images`, plural: a session that binds by name finds nothing under `image`,
/// `input` or `input.1`.
const INPUT_NAME: &str = "images";

/// The graph's output name, exactly as the file declares it.
const OUTPUT_NAME: &str = "output";

/// The square the graph's input is fixed at — declared `[1, 3, 416, 416]`, not
/// a maximum and not a multiple: 416 exactly, and ONNX Runtime rejects any
/// other size rather than resizing.
const INPUT_SIZE: usize = 416;

/// The multiple every letterboxed side is rounded down to.
///
/// YOLOX's `ValTransform` uses 32, which is the product of the three strides;
/// a side that is not a multiple of 32 leaves the deepest feature map short by
/// a cell.
const SIZE_MULTIPLE: usize = 32;

/// The value every letterbox pad pixel is set to, before normalisation.
///
/// YOLOX's own padding constant. It is deliberately not a grey derived from
/// the mean above — the mean is per-channel and this is one number for all
/// three.
const PAD_VALUE: f32 = 114.0;

/// Channels per pixel: BGR.
const CHANNELS: usize = 3;

/// The three feature-map strides, in the order the graph concatenates them.
///
/// Descending resolution first: `Concat(['1158', '1166', '1174'])` puts the
/// stride-8 `Reshape` output at rows 0..2704, then stride 16, then stride 32.
/// See the module docs.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Columns per row: `cx`, `cy`, `w`, `h`, `obj`, then 80 class scores.
const COLUMNS: usize = 5 + CLASSES;

/// Object classes the COCO-trained head emits.
const CLASSES: usize = 80;

/// The class the whole product is about: a person, COCO id 0.
pub const PERSON_CLASS: u32 = 0;

/// The signal the whole product is about: a cell phone, COCO id 67.
///
/// 67 and not 66 or 68 — the head's class order is alphabetical, so 66 is
/// `keyboard`, 67 `cell phone` and 68 `microwave`. Confirmed against the
/// graph's own 80-channel bias: channel 0 carries the largest bias (−1.10) in
/// every scale, which is COCO's `person`, and channel 67 is the only other
/// channel this crate names.
pub const CELL_PHONE_CLASS: u32 = 67;

/// The per-channel mean subtracted from the 0–255 BGR pixel.
const MEAN: [f32; CHANNELS] = [103.53, 116.28, 123.675];

/// The per-channel standard deviation divided out after the mean.
const STD: [f32; CHANNELS] = [57.375, 57.12, 58.395];

/// IoU above which two boxes of the same class suppress each other.
const NMS_IOU_THRESHOLD: f32 = 0.45;

/// Score below which a row is not a detection at all.
const SCORE_THRESHOLD: f32 = 0.1;

/// The exponent a raw width or height is clamped to, in both directions.
///
/// An unbounded `exp` turns a garbage tensor into `inf`, and `inf` into a NaN
/// the moment it is multiplied by a zero, subtracted from itself or scaled by
/// a letterbox ratio of zero. The crate's rule is that bad model output yields
/// a `ModelError` or a rejected detection — never a panic and never a
/// non-finite coordinate — so the clamp is what makes `exp` total.
///
/// `e^20 = 4.85e8`, so a clamped box is at most `4.85e8 * 32 = 1.55e10`
/// pixels across: absurd, but finite, and far inside `f32`'s `3.4e38`. At the
/// other end `e^-20 = 2.06e-9` gives a sub-pixel box, which is what a huge
/// negative raw value honestly means. Neither end changes the result for a
/// real detection: the raw head values measured on COCO images span −2.4 to
/// +4.0.
const MAX_EXPONENT: f32 = 20.0;

/// The largest magnitude a raw `xy` offset is clamped to before it is scaled by
/// a stride.
///
/// The exponent clamp below does not cover the *origin*: `xy = (raw + grid) *
/// stride` is a plain multiply, and `f32::MAX * 8.0` is an infinity. Whether it
/// is reached depends on the build — release Rust saturates an overflowing
/// `f32` multiply to `f32::MAX` while debug yields `inf` — and a wrapper whose
/// coordinates depend on the optimisation level is a wrapper that behaves
/// differently in the test suite than in the binary, which is the worst of the
/// two.
///
/// `1e37 * 32` is `3.2e38`, just under `f32::MAX`'s `3.4e38`, so a clamped
/// origin multiplied by the longest stride stays finite. A real head's offsets
/// span roughly −2.4 to +4.0, so this never binds on a real detection.
const MAX_COORDINATE: f32 = 1.0e37;

/// The number of rows the graph's `[1, N, 85]` output must have, summed over
/// [`STRIDES`] rather than written as `3549`.
///
/// This is the check that a different export — a different input size, a
/// fourth stride — is rejected instead of being decoded against a grid that
/// does not match it.
fn anchor_count() -> usize {
    STRIDES
        .iter()
        .map(|stride| (INPUT_SIZE / stride).pow(2))
        .sum()
}

/// The stride row `row` of the output belongs to, or `None` if `row` is past
/// the last anchor.
///
/// The graph concatenates its three scales highest-resolution-first, so rows
/// `0..2704` are stride 8, `2704..3380` stride 16 and `3380..3549` stride 32.
fn row_stride(row: usize) -> Option<usize> {
    let mut start = 0;
    for stride in STRIDES {
        let count = (INPUT_SIZE / stride).pow(2);
        if row < start + count {
            return Some(stride);
        }
        start += count;
    }
    None
}

/// Where an anchor sits within its own scale's grid, as `(column, row)`.
///
/// The graph reshapes each scale to `[1, 85, -1]`, so an anchor's index within
/// its scale is row-major: `index = grid_y * side + grid_x`.
fn grid_position(row: usize) -> (usize, usize, usize) {
    let stride = row_stride(row).expect("row_stride covers every anchor");
    let side = INPUT_SIZE / stride;
    let start: usize = STRIDES
        .iter()
        .take_while(|candidate| **candidate != stride)
        .map(|candidate| (INPUT_SIZE / candidate).pow(2))
        .sum();
    let index = row - start;
    (index % side, index / side, stride)
}

/// The letterbox that put a `frame` of `width` x `height` into the 416x416
/// tensor, and can put a box back.
///
/// YOLOX's `ValTransform` scales by `min(size / h, size / w)` and pads the
/// **bottom and right**, so the content sits at the tensor's top-left corner
/// and there is never a left or top offset. Both offsets are still carried as
/// fields rather than assumed to be zero, because the inverse map is the one
/// place a detector's coordinates silently detach from its pixels — YuNet's
/// C1 was exactly this bug in the other direction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Letterbox {
    /// The uniform scale applied to the frame.
    scale: f32,
    /// The tensor x that frame x 0 lands on.
    left: f32,
    /// The tensor y that frame y 0 lands on.
    top: f32,
}

impl Letterbox {
    /// The letterbox from a `width` x `height` frame into the tensor.
    ///
    /// A zero-area frame has no scale to speak of; it reads as a scale of 1.0
    /// with no offset, so the arithmetic below stays total and the tensor is
    /// black rather than NaN.
    fn for_frame(width: u32, height: u32) -> Letterbox {
        if width == 0 || height == 0 {
            return Letterbox {
                scale: 1.0,
                left: 0.0,
                top: 0.0,
            };
        }
        let scale = (INPUT_SIZE as f32 / height as f32).min(INPUT_SIZE as f32 / width as f32);
        Letterbox {
            scale,
            // Bottom-right padding: the content starts at the corner.
            left: 0.0,
            top: 0.0,
        }
    }

    /// The frame coordinates a tensor coordinate came from — the inverse of the
    /// map `preprocess` samples through, and the one [`YoloxNano::detect`]
    /// uses.
    ///
    /// Takes `self` by value because [`Letterbox`] is `Copy` and three `f32`s
    /// wide; clippy's `wrong_self_convention` is right that a `to_*` method on a
    /// `Copy` type should not borrow.
    ///
    /// The letterbox is the one place a detector's coordinates silently detach
    /// from the pixels it actually saw: YuNet's C1 was this exact map written
    /// in the wrong direction, and no shape assertion can see it, because the
    /// tensor is the right shape either way.
    fn to_frame(self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.left) / self.scale, (y - self.top) / self.scale)
    }
}

/// The content width in tensor pixels: the frame's width scaled, rounded down
/// to a multiple of [`SIZE_MULTIPLE`] and clamped into the tensor.
fn content_span(source: u32, scale: f32) -> usize {
    let scaled = source as f32 * scale;
    if !scaled.is_finite() {
        return 0;
    }
    ((scaled as usize / SIZE_MULTIPLE) * SIZE_MULTIPLE).min(INPUT_SIZE)
}

/// The source position tensor index `index` samples inside a content band that
/// is `span` pixels wide, for a frame at letterbox `scale`.
///
/// `((index + 0.5) / scale) - 0.5`, the pixel-centre convention `cv2.resize`
/// uses.
///
/// # The scale is *divided*, and that is the whole function
///
/// The frame is scaled by `scale`, so the band's `span` pixels came from
/// `span / scale` **frame** pixels — the source span is the frame's, not the
/// tensor's. Multiplying instead of dividing makes the function an identity
/// (`(i + 0.5) * (span * scale) / span - 0.5` is not, but
/// `(i + 0.5) * span / span - 0.5` is exactly `i`, which is what a first
/// attempt at this produces): every sample lands on the frame pixel of the same
/// index, the whole frame is squashed into the top-left `span` pixels, and the
/// detector sees a magnified crop of the frame's corner. That is YuNet's C1 in a
/// different disguise, and nothing about the tensor's shape reveals it.
///
/// At a scale of 0.25 this puts tensor column 0 at frame column 1.5 and tensor
/// column 1 at 5.5 — a quarter of the way across, which is where the first
/// quarter of the frame belongs.
#[inline]
fn remap(index: f32, scale: f32) -> f32 {
    if scale <= 0.0 {
        return 0.0;
    }
    (index + 0.5) / scale - 0.5
}

/// One BGR byte scaled the way this graph was trained: `(x - mean) / std`.
#[inline]
fn normalise(channel: usize, value: f32) -> f32 {
    (value - MEAN[channel]) / STD[channel]
}

/// Bilinear blue-green-red sample at a floating-point frame coordinate.
///
/// Coordinates outside the frame clamp to the edge pixel, so the sampler can
/// never read outside the buffer. A frame with no pixels is black.
///
/// Same pixel-centre convention as
/// [`facemesh::preprocess`](crate::models::facemesh::preprocess) and
/// [`eyestate::preprocess`](crate::models::eyestate::preprocess), and the same
/// one `cv2.resize` uses, so a letterboxed tensor built here holds the pixels
/// the model was trained on rather than a half-pixel-shifted copy of them.
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

/// Builds the BGR NCHW tensor the graph expects from `frame`.
///
/// Pure: same frame in, same tensor out, with no clock and no randomness.
///
/// The tensor is always `[1, 3, 416, 416]`, because the graph declares exactly
/// that and ONNX Runtime rejects anything else. The frame is scaled by
/// `min(416/h, 416/w)` — preserving its aspect ratio, never stretching it —
/// the scaled size is rounded **down to a multiple of 32**, and the remainder
/// is filled with `PAD_VALUE`.
///
/// Three parts of this are load-bearing and none is cosmetic:
///
/// * **No division by 255.** The values stay on `0–255` and are then standardised
///   by the mean and std below. Dividing first puts the whole tensor ~255x too
///   small and the detector returns nothing.
/// * **BGR, not RGB.** Reversing the channels is Face Mesh's regime.
/// * **Pad 114**, at the bottom and right, not centred.
pub fn preprocess(frame: &Frame) -> InputTensor {
    let plane = INPUT_SIZE * INPUT_SIZE;
    // The pad pixel is *normalised*, once, exactly like every content pixel.
    // Filling with a raw 114 and normalising only the content would put two
    // regimes in one tensor: the graph would see a content band on `0-255`'s
    // scale and a pad band on a scale 57x larger, and the border it was trained
    // to ignore would become the brightest thing in the image.
    let mut data = vec![0.0f32; plane * CHANNELS];
    for channel in 0..CHANNELS {
        let pad = normalise(channel, PAD_VALUE);
        data[channel * plane..(channel + 1) * plane].fill(pad);
    }

    let letterbox = Letterbox::for_frame(frame.width, frame.height);
    let content_w = content_span(frame.width, letterbox.scale);
    let content_h = content_span(frame.height, letterbox.scale);

    // Only the content rectangle is written; everything else keeps the pad
    // value filled above.
    let (left, top) = (letterbox.left, letterbox.top);
    //
    // The content band is a **resize**, not a sub-sample: `resize::<N>` maps
    // tensor index `i` to source position `(i + 0.5) * span / N - 0.5`, the
    // pixel-centre convention `cv2.resize` uses and the same one
    // `facemesh::preprocess` and `eyestate::preprocess` use. At an exact
    // integer scale the two agree; they differ by up to half a source pixel at
    // any other, and the letterbox's 32-multiple rounding makes "any other"
    // the common case.
    for row in 0..content_h {
        let tensor_y = top + row as f32;
        for column in 0..content_w {
            let tensor_x = left + column as f32;
            let source_x = remap(tensor_x, letterbox.scale);
            let source_y = remap(tensor_y, letterbox.scale);
            let (b, g, r) = sample_bgr(frame, source_x, source_y);
            let index = row * INPUT_SIZE + column;
            data[index] = normalise(0, b);
            data[plane + index] = normalise(1, g);
            data[2 * plane + index] = normalise(2, r);
        }
    }

    InputTensor {
        name: INPUT_NAME.to_owned(),
        shape: vec![1, CHANNELS as i64, INPUT_SIZE as i64, INPUT_SIZE as i64],
        data: TensorData::F32(data),
    }
}

/// A finite raw head value, or [`-MAX_EXPONENT`] if the model produced a NaN or
/// an infinity.
///
/// [`-MAX_EXPONENT`] and not `0.0`: a non-finite width is not evidence of an
/// average-sized box, it is evidence of nothing at all, and mapping it to the
/// smallest representable box makes the row score-bound rather than
/// geometry-bound. A row that survives on score alone then contributes a
/// sub-pixel box, which the class filter and NMS will deal with, instead of a
/// full-frame box that swallows every real detection of its class.
#[inline]
fn finite_raw(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        -MAX_EXPONENT
    }
}

/// One row's box in **tensor** coordinates, or `None` if the row scored below
/// [`SCORE_THRESHOLD`] or its best class is not one of `wanted`.
///
/// This is the decode the brief specifies, and for this export it applies a
/// second sigmoid to two already-activated heads — see R38 in the module docs
/// for the measurement and for what it costs. It is implemented as written
/// because the score is Task 12's input contract, not this wrapper's to
/// redefine.
///
/// Every arithmetic step is total: `raw` is sanitised by [`finite_raw`], the
/// exponent is clamped to `±MAX_EXPONENT`, and `sigmoid` is the numerics-safe
/// form rather than `1 / (1 + exp(-x))`, which overflows to `inf` and then
/// gives `1 / inf = 0` for large negative logits — a wrong answer that looks
/// like a confident rejection.
fn decode_row(raw: &[f32], row: usize, wanted: &[u32]) -> Option<Detection> {
    if raw.len() < COLUMNS {
        return None;
    }
    // A row whose box carries a NaN or an infinity is **rejected**, not
    // repaired, and this is the ordered position that matters: check it *first*,
    // before the class comparison, because that comparison uses `finite_raw` to
    // stay well ordered.
    //
    // Repairing does not work, and that is worth stating because it looks like
    // it should. `finite_raw` sanitises the four box values, and the exponent is
    // clamped, so a NaN width already cannot escape. What it cannot catch is
    // `f32::MAX`: an entirely *finite* raw x, multiplied by a stride of 8,
    // overflows to `inf`, and the box then has an infinite coordinate that no
    // clamp on the raw value would have prevented — the overflow happens in this
    // function's own arithmetic. Substituting a fallback for a NaN does not fix
    // that either, because the failure is not a NaN input.
    //
    // So a non-finite input means the tensor is not a tensor this decode
    // understands, and the honest answer is that this row contributes no
    // detection. Silently synthesising a box from it would put a full-frame
    // `person` into the pipeline, which is exactly the failure the whole
    // no-NaN rule exists to prevent.
    if [raw[0], raw[1], raw[2], raw[3], raw[4]]
        .iter()
        .any(|v| !v.is_finite())
    {
        return None;
    }

    // A non-finite *class* logit is rejected for the same reason, and it would
    // otherwise be invisible: `finite_raw` maps a NaN to `-MAX_EXPONENT`, whose
    // sigmoid is 2.06e-9 — a score that no comparison would flag, silently
    // replacing a broken tensor with a confident "no object here".
    if raw[5..COLUMNS].iter().any(|value| !value.is_finite()) {
        return None;
    }

    let (grid_x, grid_y, stride) = grid_position(row);

    // The best class first, because a row no class of ours claimed is not a
    // detection and its box is not worth computing.
    //
    // Ties resolve to the **lowest** class id, and that is a decision rather
    // than an accident. `max_by` returns the *last* maximum, so a row whose
    // class logits are all equal — every zeroed row, and every row of the
    // zero-filled tensor `FakeSession::zeros` builds — would report class 79
    // rather than class 0, and a tensor a test thought was empty would claim a
    // `toothbrush`. Folding with a strict `>` keeps the first maximum instead,
    // which also makes the decode a pure function of the tensor rather than of
    // the comparison's direction.
    let mut class = 0usize;
    let mut best = finite_raw(raw[5]);
    for candidate in 1..CLASSES {
        let score = finite_raw(raw[5 + candidate]);
        if score > best {
            best = score;
            class = candidate;
        }
    }
    let class_id = class as u32;
    if !wanted.contains(&class_id) {
        return None;
    }

    let objectness = finite_raw(raw[4]);
    let class_score = finite_raw(raw[5 + class]);
    let score = sigmoid(objectness) * sigmoid(class_score);
    // `!(score >= …)` rather than `score < …`, so that a NaN score is rejected
    // here instead of passing a comparison that is false for it. The product of
    // two finite sigmoids cannot be NaN, so this is belt and braces and costs
    // one boolean — which is why the lint is silenced rather than the
    // expression rewritten as a `partial_cmp` that would need its own `None`
    // arm for a case that cannot occur.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(score >= SCORE_THRESHOLD) {
        return None;
    }

    // Both coordinates are clamped *before* the stride multiply, not after:
    // the overflow happens in the multiply, so a clamp applied to the result
    // would already be too late.
    let center_x = finite_raw(raw[0]).clamp(-MAX_COORDINATE, MAX_COORDINATE) * stride as f32
        + grid_x as f32 * stride as f32;
    let center_y = finite_raw(raw[1]).clamp(-MAX_COORDINATE, MAX_COORDINATE) * stride as f32
        + grid_y as f32 * stride as f32;
    let width = finite_raw(raw[2]).clamp(-MAX_EXPONENT, MAX_EXPONENT).exp() * stride as f32;
    let height = finite_raw(raw[3]).clamp(-MAX_EXPONENT, MAX_EXPONENT).exp() * stride as f32;

    Some(Detection {
        rect: Rect {
            x: center_x - width / 2.0,
            y: center_y - height / 2.0,
            w: width,
            h: height,
        },
        score,
        class_id,
    })
}

/// `1 / (1 + e^-x)`, spelled so that neither tail overflows.
///
/// The textbook form computes `e^-x`, which is `inf` for `x < -88.7`; `1 / inf`
/// is then `0.0`, which is the right limit reached by an infinity rather than
/// by arithmetic. The two-branch form below keeps every intermediate finite in
/// both directions, which is what the crate's no-NaN rule asks of the one
/// activation this wrapper owns.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The output tensor's `[1, N, 85]` rows, after checking its name, shape and
/// length.
///
/// The length check is not redundant with the shape check: [`OutputTensor`]'s
/// fields are public, so a session can declare `[1, 3549, 85]` in front of a
/// buffer of any length. The decode indexes rows, so this runs before any of
/// it — reading past the end of a slice panics, and a function that returns
/// `Result` must not.
fn read_predictions(outputs: &[OutputTensor]) -> Result<&[f32], ModelError> {
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

    let expected = vec![1, anchor_count() as i64, COLUMNS as i64];
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
    if values.len() != anchor_count() * COLUMNS {
        return Err(ModelError::UnexpectedDataLength {
            name: OUTPUT_NAME,
            shape: output.shape.clone(),
            got: values.len(),
        });
    }
    Ok(values)
}

/// Person and phone detection over a session.
pub struct YoloxNano {
    session: Box<dyn InferenceSession>,
}

impl YoloxNano {
    /// Wraps a session. The session is the model; this type is the arithmetic
    /// around it.
    pub fn new(session: Box<dyn InferenceSession>) -> YoloxNano {
        YoloxNano { session }
    }

    /// The wrapped session's name, for logs and for the provider autotune.
    pub fn name(&self) -> &str {
        self.session.name()
    }

    /// Every **person** and **cell phone** in `frame`, strongest first.
    ///
    /// # Which classes, and why not all eighty
    ///
    /// The detector emits eighty COCO classes; this returns two. The choice is
    /// deliberate. Stage 3 exists to produce the phone signal and the person
    /// boxes that corroborate it, `Detection`'s `class_id` is still carried so
    /// a caller can tell the two apart, and the other seventy-eight classes
    /// have no consumer in this engine. Returning them would push 78
    /// irrelevant classes through Task 12's `PhoneSignal`, and — measured on a
    /// COCO cat photo — `sheep` and `airplane` score *above* the real phone, so
    /// an unfiltered result is not merely extra data, it is data that outranks
    /// the answer. The filter is applied **before** NMS rather than after, so
    /// that a strong `sheep` box cannot suppress a weaker overlapping `cat`
    /// box that the caller was never going to see anyway; the two orderings
    /// differ, and filtering first is the one whose output is a pure function
    /// of the two classes the caller asked for.
    ///
    /// # What the boxes mean
    ///
    /// Coordinates are **frame** pixels, origin top-left, with the letterbox
    /// inverted — a caller can hand one to
    /// [`facemesh::preprocess`](crate::models::facemesh::preprocess) or draw it
    /// on the original frame without knowing a letterbox ever existed. A box
    /// may extend past the frame's edges: a person whose legs are below the
    /// frame is still a person, and clipping here would move the box's centre
    /// away from the body it describes.
    ///
    /// # Cost
    ///
    /// One 416x416x3 tensor per call — 519,168 floats, about 2 MB — and 3549
    /// row decodes. This stage runs at 1–2 Hz on its own cadence, not at the
    /// motion gate's 5, so the tensor build dominates and the decode is noise
    /// beside it.
    pub fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, ModelError> {
        let input = preprocess(frame);
        let outputs = self.session.run(&[input])?;
        let raw = read_predictions(&outputs)?;

        let wanted = [PERSON_CLASS, CELL_PHONE_CLASS];
        let letterbox = Letterbox::for_frame(frame.width, frame.height);

        let mut detections: Vec<Detection> = (0..anchor_count())
            .filter_map(|row| decode_row(&raw[row * COLUMNS..(row + 1) * COLUMNS], row, &wanted))
            .map(|detection| {
                // Invert the letterbox once, here, so nothing downstream has to
                // know the tensor was ever padded. Widths and heights scale by
                // the same uniform factor; there is no offset on them, because
                // an offset moves a box and does not resize it.
                let (x, y) = letterbox.to_frame(detection.rect.x, detection.rect.y);
                Detection {
                    rect: Rect {
                        x,
                        y,
                        w: detection.rect.w / letterbox.scale,
                        h: detection.rect.h / letterbox.scale,
                    },
                    ..detection
                }
            })
            .collect();

        // Class-aware, as `nms` is: a phone held inside a person's box shares
        // its rect and an IoU of 1.0 with it, and a class-agnostic sweep would
        // delete one of the two — destroying the signal this stage exists for.
        nms(&mut detections, NMS_IOU_THRESHOLD, SCORE_THRESHOLD);
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{FakeSession, FakeSessionHandle};
    use std::time::Instant;

    /// The number of rows the graph's output has: 2704 + 676 + 169.
    const ANCHORS: usize = 3549;

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

    /// A frame four times the tensor's width and 896/416 its height, so the
    /// letterbox scale is exactly 0.25, the content is 416 x 224, and there are
    /// 192 rows of padding.
    ///
    /// The dimensions are chosen so that **both** scaled sides land on a
    /// multiple of 32 *and* the inverse map multiplies by exactly 4:
    /// `416 * 4 = 1664` and `224 * 4 = 896`, with
    /// `min(416/896, 416/1664)` = `min(0.4643, 0.25) = 0.25`.
    ///
    /// The obvious choice — a 2:1 frame such as 832x416, scaling to 416x208 —
    /// is **wrong**, and wrong in a way that costs an hour if it is not seen:
    /// `208 / 32 = 6`, because 32 x 6.5 = 208, so `SIZE_MULTIPLE` floors the
    /// content band to 192 rows and 16 rows of the frame are dropped. 224 is
    /// 32 x 7 and nothing is floored.
    fn letterboxed_frame(bgr: (u8, u8, u8)) -> Frame {
        solid_frame(1664, 896, bgr)
    }

    /// An output tensor named and shaped as the graph declares it, holding
    /// `raw` verbatim.
    fn predictions_output(raw: Vec<f32>) -> OutputTensor {
        OutputTensor {
            name: OUTPUT_NAME.to_owned(),
            shape: vec![1, ANCHORS as i64, COLUMNS as i64],
            data: TensorData::F32(raw),
        }
    }

    /// Writes `values` into row `row`, which must be `COLUMNS` long.
    fn set_row(raw: &mut [f32], row: usize, values: &[f32]) {
        assert_eq!(values.len(), COLUMNS, "a row is exactly COLUMNS wide");
        raw[row * COLUMNS..(row + 1) * COLUMNS].copy_from_slice(values);
    }

    /// A row that decodes to something worthless: a central grid cell, a
    /// sub-pixel box, `sigmoid(-40)` objectness.
    ///
    /// `sigmoid(-40) = 4.25e-18`, so the row's score is ~2.1e-18 — far below
    /// `SCORE_THRESHOLD` and below anything a real head emits. Filling the
    /// background with this instead of zeroes matters: a zeroed row has
    /// `sigmoid(0) = 0.5` objectness and `sigmoid(0) = 0.5` for *every* class,
    /// which is a score of 0.25 — comfortably above threshold, and 3549
    /// spurious detections from a tensor a test thought was empty.
    fn blank_row() -> [f32; COLUMNS] {
        let mut row = [0.0f32; COLUMNS];
        row[4] = -40.0;
        row
    }

    // A row configured for `class_id` at `objectness` and `class_score`.
    //
    // The single place the column layout `[cx, cy, w, h, obj, cls_*]` is
    // written down for the tests: every other helper builds its rows from this
    // one, so a test cannot pass by agreeing with a *different* layout.
    fn row_for(
        objectness: f32,
        class_id: usize,
        class_score: f32,
        box_raw: [f32; 4],
    ) -> [f32; COLUMNS] {
        let mut row = blank_row();
        row[0..4].copy_from_slice(&box_raw);
        row[4] = objectness;
        row[5 + class_id] = class_score;
        row
    }

    /// A tensor of blank rows, ready for a test to place its own.
    fn blank_predictions() -> Vec<f32> {
        let mut raw = Vec::with_capacity(ANCHORS * COLUMNS);
        for _ in 0..ANCHORS {
            raw.extend_from_slice(&blank_row());
        }
        raw
    }

    /// A detector over `row`'s prediction, with every other row blank.
    fn detector_for_row(row: usize, values: &[f32; COLUMNS]) -> (YoloxNano, FakeSessionHandle) {
        let mut raw = blank_predictions();
        set_row(&mut raw, row, values);
        detector_returning(predictions_output(raw))
    }

    /// A detector over a session that always returns `output`, plus the handle
    /// that records what it was asked to run.
    fn detector_returning(output: OutputTensor) -> (YoloxNano, FakeSessionHandle) {
        let session = FakeSession::single("yolox_nano", output);
        let handle = session.handle();
        (YoloxNano::new(Box::new(session)), handle)
    }

    /// A tensor whose **row `row`** is configured for `class_id` at
    /// `objectness_logit` and `class_logit`, with everything else blank.
    fn tensor_with_one_row(
        row: usize,
        objectness_logit: f32,
        class_id: usize,
        class_logit: f32,
        box_raw: [f32; 4],
    ) -> Vec<f32> {
        let mut raw = blank_predictions();
        set_row(
            &mut raw,
            row,
            &row_for(objectness_logit, class_id, class_logit, box_raw),
        );
        raw
    }

    /// The tensor's values, as an `f32` slice.
    fn values(tensor: &InputTensor) -> &[f32] {
        tensor
            .data
            .as_f32()
            .expect("preprocess builds an f32 tensor")
    }

    /// One value of a plane-major `[1, 3, 416, 416]` tensor.
    fn at(tensor: &InputTensor, channel: usize, column: usize, row: usize) -> f32 {
        values(tensor)[channel * INPUT_SIZE * INPUT_SIZE + row * INPUT_SIZE + column]
    }

    #[test]
    fn anchor_count_is_derived_from_the_strides_and_is_3549() {
        // The brief says 3549; this says *why*, from the strides the graph's
        // three Reshape nodes concatenate. If the input size or a stride
        // changed, the brief's literal would silently become wrong while this
        // stays right — and the shape gate in `read_predictions` is built on
        // this function, so the two cannot drift apart.
        assert_eq!(STRIDES, [8, 16, 32], "the graph concatenates these three");
        assert_eq!(anchor_count(), 2704 + 676 + 169);
        assert_eq!(anchor_count(), ANCHORS, "and that is the brief's 3549");

        // The rows are grouped highest-resolution-first, 2704 then 676 then
        // 169. A swapped order maps every anchor to the wrong grid cell and
        // yields boxes that are geometrically plausible and entirely wrong.
        assert_eq!(row_stride(0), Some(8), "row 0 is the first stride-8 anchor");
        assert_eq!(row_stride(2703), Some(8), "the last of them");
        assert_eq!(row_stride(2704), Some(16), "then stride 16 starts");
        assert_eq!(row_stride(3379), Some(16));
        assert_eq!(row_stride(3380), Some(32), "and stride 32 last");
        assert_eq!(row_stride(3548), Some(32));
        assert_eq!(row_stride(3549), None, "past the last anchor");

        // The first and last anchor of each scale, as (grid_x, grid_y, stride).
        // Stride 8 has a 52x52 grid, so index 2703 is the last cell: (51, 51).
        assert_eq!(grid_position(0), (0, 0, 8));
        assert_eq!(
            grid_position(52),
            (0, 1, 8),
            "row-major: index = y * 52 + x"
        );
        assert_eq!(grid_position(2703), (51, 51, 8));
        // Stride 16 has 26x26, so index 675 is (25, 25).
        assert_eq!(grid_position(2704), (0, 0, 16));
        assert_eq!(grid_position(3379), (25, 25, 16));
        // Stride 32 has 13x13, so index 168 is (12, 12).
        assert_eq!(grid_position(3380), (0, 0, 32));
        assert_eq!(grid_position(3548), (12, 12, 32));
    }

    #[test]
    fn preprocessing_letterboxes_to_416_with_pad_114_and_keeps_bgr_0_255() {
        // (10, 20, 30) is chosen so the three channels are three different
        // numbers: a BGR/RGB swap moves 10 to where 30 was and is visible,
        // where (128, 128, 128) would hide it.
        let frame = letterboxed_frame((10, 20, 30));
        let tensor = preprocess(&frame);

        assert_eq!(tensor.name, "images", "the graph's own spelling");
        assert_eq!(tensor.shape, vec![1, 3, 416, 416], "NCHW at the fixed size");
        assert!(
            matches!(tensor.data, TensorData::F32(_)),
            "f32, not quantised"
        );
        assert_eq!(values(&tensor).len(), CHANNELS * INPUT_SIZE * INPUT_SIZE);
        // The session must be handed exactly this tensor, so the values the
        // graph sees are the values asserted here.
        let (detector, handle) = detector_returning(predictions_output(blank_predictions()));
        detector
            .detect(&frame)
            .expect("a well-formed output decodes");
        let calls = handle.inputs();
        assert_eq!(calls.len(), 1, "one frame is one run");
        assert_eq!(calls[0].len(), 1, "this graph takes exactly one input");
        assert_eq!(calls[0][0].data, tensor.data, "and it is this tensor");

        // The letterbox: a 1664x896 frame scales by 0.25 to 416x224, which is
        // already a multiple of 32, so the top 224 rows are content and the
        // bottom 192 are padding. The scale is asserted exactly because 0.25 is
        // an exact binary fraction — a `min(size/h, size/w)` that took the
        // wrong ratio would give 1.0 here and stretch the frame instead.
        let letterbox = Letterbox::for_frame(frame.width, frame.height);
        assert_eq!(letterbox.scale, 0.25, "min(416/896, 416/1664) = 0.25");
        assert_eq!(content_span(frame.width, letterbox.scale), 416);
        assert_eq!(
            content_span(frame.height, letterbox.scale),
            224,
            "224 is 32 x 7, so nothing is floored away"
        );

        // Content: BGR in the graph's regime — 0-255 with NO division. A
        // `/255` would put blue at 0.039216 and red at 0.117647, so the three
        // numbers below distinguish the two regimes as well as the channel
        // order.
        let blue = (10.0 - MEAN[0]) / STD[0];
        let green = (20.0 - MEAN[1]) / STD[1];
        let red = (30.0 - MEAN[2]) / STD[2];
        for (column, row) in [(0, 0), (1, 1), (415, 0), (0, 223), (415, 223)] {
            assert_eq!(
                at(&tensor, 0, column, row),
                blue,
                "blue at ({column},{row})"
            );
            assert_eq!(
                at(&tensor, 1, column, row),
                green,
                "green at ({column},{row})"
            );
            assert_eq!(at(&tensor, 2, column, row), red, "red at ({column},{row})");
        }
        // Spelled out against the literals 10, 20 and 30, because the three
        // `let`s above follow `MEAN` and `STD` and would move with a mutation
        // of either. This is the assertion that pins the regime itself:
        // (10 - 103.53)/57.375 = -1.6301525, a bare 10/255 = 0.039216, and
        // 10 with no normalisation at all = 10.
        assert_eq!(blue, -1.6301525_f32, "0-255, centred on 103.53 and scaled");
        assert_eq!(green, -1.6855743_f32, "not divided by 255");
        assert_eq!(red, -1.6041614_f32, "and not left raw");

        // Padding: exactly the pad value, in all three channels, in every one
        // of the 192 padded rows. The bottom-right corner is checked as well
        // as the top of the pad band, so a partially-filled or transposed pad
        // cannot pass.
        let pad = (PAD_VALUE - MEAN[0]) / STD[0];
        assert_eq!(at(&tensor, 0, 0, 224), pad, "the first pad row");
        assert_eq!(at(&tensor, 1, 0, 224), (PAD_VALUE - MEAN[1]) / STD[1]);
        assert_eq!(at(&tensor, 2, 0, 224), (PAD_VALUE - MEAN[2]) / STD[2]);
        assert_eq!(at(&tensor, 0, 415, 415), pad, "and the last pad pixel");
        assert_eq!(at(&tensor, 2, 415, 415), (PAD_VALUE - MEAN[2]) / STD[2]);
        // The pad is 114, not 0 and not a grey: (114 - 103.53)/57.375 =
        // 0.18248369, while a 0 pad reads -1.8044444 and a 127 pad reads
        // 0.40906155.
        assert_eq!(pad, 0.18248369_f32, "the pad value is 114, not 0 or 127");
        assert_eq!(
            (PAD_VALUE - MEAN[1]) / STD[1],
            -0.039915945_f32,
            "and green's is its own mean, not blue's"
        );
        assert_eq!(
            (PAD_VALUE - MEAN[2]) / STD[2],
            -0.16568205_f32,
            "and red's is the third"
        );

        // The whole pad band, every pixel: a loop that stopped at the pad's
        // first row, or that wrote content into part of it, leaves values the
        // five samples above can miss.
        for row in 224..INPUT_SIZE {
            for column in 0..INPUT_SIZE {
                let index = row * INPUT_SIZE + column;
                let got = &values(&tensor)[index..];
                let plane = INPUT_SIZE * INPUT_SIZE;
                assert_eq!(
                    [got[0], got[plane], got[2 * plane]],
                    [
                        (PAD_VALUE - MEAN[0]) / STD[0],
                        (PAD_VALUE - MEAN[1]) / STD[1],
                        (PAD_VALUE - MEAN[2]) / STD[2],
                    ],
                    "pad pixel ({column}, {row})"
                );
            }
        }
    }

    #[test]
    fn decode_applies_grid_offset_and_exp_scaling() {
        // Row 100 of stride 8: grid index 100 with a 52-wide grid is
        // (x, y) = (100 % 52, 100 / 52) = (48, 1), so `xy = (raw + grid) * 8`
        // is ((raw_x + 48) * 8, (raw_y + 1) * 8).
        //
        // The raw box is chosen so every step is an exact binary fraction:
        //   xy  = (0, 0)          -> centre  (384, 8)
        //   w   = exp(-0.5) * 8   = 0.60653066 * 8 = 4.8522453
        //   h   = exp(-1.0) * 8   = 0.36787945 * 8 = 2.9430356
        //   x   = 384 - 4.8522453 / 2 = 381.57388
        //   y   =   8 - 2.9430356 / 2 =   6.5284822
        //
        // and the score is `sigmoid(0) * sigmoid(0) = 0.5 * 0.5 = 0.25`, exact
        // in f32 because 0.5 is a power of two.
        let frame = letterboxed_frame((10, 20, 30));
        let row = 100;
        let raw = tensor_with_one_row(row, 0.0, PERSON_CLASS as usize, 0.0, [0.0, 0.0, -0.5, -1.0]);
        let (detector, _) = detector_returning(predictions_output(raw));

        let detections = detector.detect(&frame).expect("a well-formed output");
        assert_eq!(detections.len(), 1, "one configured row, one detection");
        let detection = detections[0];

        assert_eq!(detection.class_id, PERSON_CLASS);
        assert_eq!(
            detection.score, 0.25,
            "sigmoid(0) * sigmoid(0) is exactly 0.25"
        );

        // The frame is 1664x832 and the letterbox scale is 0.25 with no
        // offset, so frame = tensor / 0.25 = tensor * 4. Every expected value
        // is then an exact multiple of the tensor-space one, which is what lets
        // these be equalities.
        let width = (-0.5f32).exp() * 8.0;
        let height = (-1.0f32).exp() * 8.0;
        let expected_center = ((48.0 * 8.0) * 4.0, (1.0 * 8.0) * 4.0);
        assert_eq!(expected_center, (1536.0, 32.0), "hand-computed grid offset");
        assert_eq!(
            detection.rect.w,
            width * 4.0,
            "exp(-0.5) * stride 8, out of the letterbox"
        );
        assert_eq!(
            detection.rect.h,
            height * 4.0,
            "exp(-1.0) * stride 8, out of the letterbox"
        );
        assert_eq!(
            (
                detection.rect.x + detection.rect.w / 2.0,
                detection.rect.y + detection.rect.h / 2.0
            ),
            expected_center,
            "the box is centred on (raw + grid) * stride, not on the cell corner"
        );
        // The literal the two `let`s above only approximate, so that a mutation
        // of the exponent or the stride fails against the brief's own formula
        // rather than against a recomputation of it. exp(-0.5) = 0.60653066 and
        // exp(-1.0) = 0.36787945, so the widths are 4.8522453 * 4 = 19.408981
        // and 2.9430356 * 4 = 11.772142.
        assert_eq!(detection.rect.w, 19.408981_f32);
        assert_eq!(detection.rect.h, 11.772142_f32);
        assert_eq!(
            detection.rect.x, 1526.2955_f32,
            "centre 1536 minus half the width"
        );
        assert_eq!(
            detection.rect.y, 26.11393_f32,
            "centre 32 minus half the height, in f32: 32 - 11.772142/2"
        );
    }

    #[test]
    fn decode_never_produces_nan_for_large_raw_values() {
        // A garbage tensor is the failure this guards. There are **two**
        // distinct ways a raw value becomes a non-finite coordinate, and they
        // need different defences:
        //
        // * an enormous but finite `wh` — `exp(1e30)` is `inf`, and `inf` minus
        //   `inf` is a NaN. The exponent clamp is what stops this, and the row
        //   survives with the largest box the clamp allows.
        // * a finite-but-extreme `xy` — `f32::MAX * 8` overflows to `inf`
        //   inside this function's own multiplication, which no clamp on the raw
        //   value could prevent. The row is *rejected* instead.
        //
        // Both are exercised below, and the distinction between them is the
        // whole point: a test that only covered the first would pass on an
        // implementation that lets an infinite origin through.
        let frame = letterboxed_frame((10, 20, 30));

        // Rejected: a non-finite input to the box or the objectness.
        //
        // `every field extreme` is deliberately *not* here. `f32::MAX` is a
        // finite value, and whether `f32::MAX * 8.0` overflows to `inf` or
        // saturates to `f32::MAX` depends on the build profile — so a row built
        // from it must come out finite either way, which is the clamping half
        // of this test rather than the rejecting half.
        for (label, box_raw) in [
            ("a NaN width", [0.0, 0.0, f32::NAN, 1.0]),
            ("a NaN origin", [f32::NAN, f32::NAN, 0.0, 0.0]),
            (
                "infinite dimensions",
                [0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY],
            ),
            (
                "an infinite origin",
                [f32::INFINITY, f32::NEG_INFINITY, 1.0, 1.0],
            ),
        ] {
            let raw = tensor_with_one_row(0, 0.0, PERSON_CLASS as usize, 0.0, box_raw);
            let (detector, _) = detector_returning(predictions_output(raw));
            let detections = detector
                .detect(&frame)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert!(
                detections.is_empty(),
                "{label}: a non-finite box must contribute no detection, got {:?}",
                detections[0].rect
            );
        }

        // A non-finite *class* logit is rejected too, and it needs its own case
        // because the class values never reach the box arithmetic: a NaN there
        // would otherwise pass silently through `finite_raw` as
        // `sigmoid(-MAX_EXPONENT)` = 2.06e-9.
        let (detector, _) =
            detector_for_row(0, &row_for(0.0, PERSON_CLASS as usize, f32::NAN, [0.0; 4]));
        assert!(
            detector
                .detect(&frame)
                .expect("a well-formed output")
                .is_empty(),
            "a NaN class logit must contribute no detection"
        );

        // And a non-finite *objectness* is rejected for the same reason.
        let (detector, _) =
            detector_for_row(0, &row_for(f32::NAN, PERSON_CLASS as usize, 0.0, [0.0; 4]));
        assert!(
            detector
                .detect(&frame)
                .expect("a well-formed output")
                .is_empty(),
            "a NaN objectness must contribute no detection"
        );

        // Clamped, not rejected: a finite raw value that only overflows once
        // `exp` sees it. The row survives with the largest box the clamp
        // allows, which is what distinguishes "guarded" from "silently dropped
        // every garbage row" — the cases above would all pass on a `decode_row`
        // that simply returned `None` for anything unusual.
        for (label, box_raw) in [
            ("an enormous positive width", [0.0, 0.0, 1.0e30, 1.0e30]),
            ("an enormous negative height", [0.0, 0.0, -1.0e30, -1.0e30]),
        ] {
            let raw = tensor_with_one_row(0, 0.0, PERSON_CLASS as usize, 0.0, box_raw);
            let (detector, _) = detector_returning(predictions_output(raw));
            let detections = detector
                .detect(&frame)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(detections.len(), 1, "{label}: the row is configured");
            let rect = detections[0].rect;
            let fields = [rect.x, rect.y, rect.w, rect.h];
            assert!(
                fields.iter().all(|value| value.is_finite()),
                "{label}: produced a non-finite coordinate {fields:?}"
            );
            assert!(
                detections[0].score.is_finite(),
                "{label}: produced a non-finite score"
            );
            // `Rect::iou` multiplies w by h, so the *NMS* arithmetic is one
            // step further out than the coordinates themselves. Bounded
            // dimensions are what keep it finite.
            assert!(
                rect.w.is_finite() && rect.h.is_finite() && rect.w * rect.h != f32::INFINITY,
                "{label}: dimensions overflow when multiplied: {} x {}",
                rect.w,
                rect.h
            );
        }

        // The *origin* needs a clamp of its own, and these are the cases that
        // prove it. `f32::MAX * 8.0` overflows, and whether that becomes an
        // `inf` or saturates to `f32::MAX` **depends on the build profile** — so
        // without this clamp the released binary produces a non-finite
        // coordinate where the test suite produces a finite one, which is the
        // worse half of both. Every assertion above passes on an `f32::MAX`
        // origin, because those cases leave `x` and `y` at zero.
        for (label, box_raw) in [
            ("an enormous origin", [1.0e30, -1.0e30, 0.0, 0.0]),
            (
                "every field extreme",
                [f32::MAX, f32::MIN, f32::MAX, f32::MIN],
            ),
        ] {
            let raw = tensor_with_one_row(0, 0.0, PERSON_CLASS as usize, 0.0, box_raw);
            let (detector, _) = detector_returning(predictions_output(raw));
            let detections = detector
                .detect(&frame)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(detections.len(), 1, "{label}: the row is configured");
            let rect = detections[0].rect;
            assert!(
                [rect.x, rect.y, rect.w, rect.h]
                    .iter()
                    .all(|value| value.is_finite()),
                "{label}: produced a non-finite coordinate {rect:?}"
            );
            // Huge and finite, not reset to zero and not `inf`: the clamp is a
            // clamp. The exact magnitude is *not* asserted, because it depends
            // on the build profile — `f32::MAX * 8.0` saturates to `f32::MAX`
            // under debug and becomes `inf` under release, so the pre-clamp
            // value differs by seven orders of magnitude between the two. What
            // must hold in both is that the coordinate is finite and enormous
            // rather than something plausible.
            assert!(
                rect.x == 0.0 || rect.x.abs() > 1.0e30,
                "{label}: the clamped origin must be huge or exactly zero, \
                 never something plausible: {rect:?}"
            );
        }

        let raw = tensor_with_one_row(
            0,
            0.0,
            PERSON_CLASS as usize,
            0.0,
            [0.0, 0.0, 1.0e30, 1.0e30],
        );
        let (detector, _) = detector_returning(predictions_output(raw));
        let detections = detector.detect(&frame).expect("a well-formed output");
        assert_eq!(
            detections[0].rect.w,
            MAX_EXPONENT.exp() * 8.0 * 4.0,
            "clamped to exp(20) * the row's stride 8, then out of the letterbox"
        );
        // exp(20) = 485165195.4, so the tensor-space box is 485165195.4 * 8
        // for a stride-8 row = 3.8813216e9, quadrupled to 1.5525286e10 in the
        // frame. Asserted as a literal so that a mutation of MAX_EXPONENT
        // fails here rather than moving the expectation with it.
        assert_eq!(detections[0].rect.w, 1.5525286e10_f32);
        assert!(
            detections[0].rect.w > 1.0e9,
            "the clamp is 20, not something that collapses the box to nothing"
        );
    }

    #[test]
    fn detect_keeps_only_person_and_cell_phone_classes() {
        let frame = letterboxed_frame((10, 20, 30));
        let mut raw = blank_predictions();

        // Rows 0..3 are the classes the engine does not consume, and they are
        // given *higher* logits than the two that it does — 0.9 against 0.8 —
        // so that a wrapper which simply returned the strongest rows, or which
        // filtered after NMS on some other criterion, cannot pass.
        for (row, class_id) in [(0, 18), (1, 66), (2, 65), (3, 68)] {
            set_row(
                &mut raw,
                row,
                &row_for(0.0, class_id, 2.197_224_6, [0.0; 4]),
            );
        }
        set_row(
            &mut raw,
            100,
            &row_for(0.0, PERSON_CLASS as usize, 1.386_294_4, [0.0; 4]),
        );
        set_row(
            &mut raw,
            2704,
            &row_for(0.0, CELL_PHONE_CLASS as usize, 1.386_294_4, [0.0; 4]),
        );

        let (detector, _) = detector_returning(predictions_output(raw));
        let detections = detector.detect(&frame).expect("a well-formed output");

        assert_eq!(
            detections
                .iter()
                .map(|det| det.class_id)
                .collect::<Vec<_>>(),
            vec![PERSON_CLASS, CELL_PHONE_CLASS],
            "only the two classes of interest survive, strongest first"
        );
        // The scores are the configured ones and not the strongest in the
        // tensor: 0.5 * 0.8 = 0.4, where the four dropped classes scored
        // 0.5 * 0.9 = 0.45. A wrapper that filtered on score rather than on
        // class would have returned those instead.
        assert_eq!(detections[0].score, 0.4, "sigmoid(0) * sigmoid(logit 0.8)");
        assert_eq!(detections[1].score, 0.4);
        assert_eq!(
            detections[1].class_id, 67,
            "cell phone is COCO id 67, between keyboard 66 and microwave 68"
        );
        assert!(
            !detections.iter().any(|det| det.class_id == 66
                || det.class_id == 68
                || det.class_id == 18
                || det.class_id == 65),
            "a neighbouring COCO id is not the cell phone"
        );
        // Row 2704 is the *first* stride-16 anchor, so this also pins that the
        // row-to-stride grouping is not off by the stride-8 block's length.
        assert_eq!(grid_position(2704), (0, 0, 16));
    }

    #[test]
    fn boxes_are_rescaled_out_of_the_letterbox_back_into_frame_coordinates() {
        // The frame is 1664x896 and the letterbox scale is exactly 0.25, so the
        // inverse map is `frame = tensor * 4` on both axes with no offset —
        // every expectation below is an exact multiple of the tensor-space
        // number, and an equality is the honest assertion.
        let frame = letterboxed_frame((10, 20, 30));

        // Row 106 of stride 8 is grid (2, 2): 106 % 52 = 2 and 106 / 52 = 2.
        // With `xy = (raw + grid) * stride` and raw (0, 0): centre
        // (2 * 8, 2 * 8) = (16, 16) in the tensor, and exp(0) * 8 = 8 for both
        // dimensions. So the box is (12, 12, 8, 8) in the tensor.
        //
        // The offsets matter here, and the *y* one especially: YOLOX pads the
        // bottom-right, so `top` is what a 1664x896 frame actually exercises
        // and `left` is zero. This frame therefore cannot distinguish the two
        // offsets on its own, which is why the rest of this test covers the
        // other padding axis and the padding band's boundary.
        let raw = tensor_with_one_row(106, 0.0, PERSON_CLASS as usize, 0.0, [0.0, 0.0, 0.0, 0.0]);
        let (detector, _) = detector_returning(predictions_output(raw));
        let detections = detector.detect(&frame).expect("a well-formed output");
        assert_eq!(detections.len(), 1);
        let rect = detections[0].rect;

        // Hand-computed frame coordinates: (12, 12, 8, 8) * 4 = (48, 48, 32, 32).
        assert_eq!(rect.x, 48.0, "tensor x 12 mapped out of the letterbox");
        assert_eq!(rect.y, 48.0, "tensor y 12 mapped out of the letterbox");
        assert_eq!(rect.w, 32.0, "tensor width 8 scaled back up by 1/0.25");
        assert_eq!(rect.h, 32.0);

        // The box is in the *frame's* coordinates, not the tensor's. A wrapper
        // that forgot to invert the letterbox returns the 8-pixel box at
        // (12, 12) — one sixteenth of the area, in the wrong place, and still
        // entirely inside the frame, which is why this is an equality against
        // the right answer rather than a bounds check.
        assert_ne!(
            rect,
            Rect {
                x: 12.0,
                y: 12.0,
                w: 8.0,
                h: 8.0
            }
        );
        assert!(
            rect.x + rect.w <= frame.width as f32,
            "and inside the frame, where the wrong box also fits by luck"
        );

        // The scale is what the inverse map divides by, and it is the frame's
        // own aspect ratio that sets it: a differently-shaped frame of the same
        // content gives a different frame-space box for the same tensor row.
        // 416x416 is already the tensor's size, so its scale is 1.0 and the
        // box comes out un-rescaled — which is the one case where a wrapper
        // that skipped the inverse map entirely would also be right.
        let square = solid_frame(416, 416, (10, 20, 30));
        assert_eq!(Letterbox::for_frame(square.width, square.height).scale, 1.0);
        let (detector, _) = detector_returning(predictions_output(tensor_with_one_row(
            106,
            0.0,
            PERSON_CLASS as usize,
            0.0,
            [0.0, 0.0, 0.0, 0.0],
        )));
        let detections = detector.detect(&square).expect("a well-formed output");
        assert_eq!(
            detections[0].rect,
            Rect {
                x: 12.0,
                y: 12.0,
                w: 8.0,
                h: 8.0
            },
            "a frame already at the tensor size is left alone"
        );

        // The padding axis is the frame's, not a constant. A 640x896 frame
        // scales by min(416/896, 416/640) = 0.4643 to 297.1x416 — *width*
        // floored to a multiple of 32 and height full — so the padding is on
        // the right and `content_span` must report 288 in x and 416 in y. One
        // shared span for both axes gets this wrong while the 4:1 frame above
        // stays right.
        let tall = solid_frame(640, 896, (10, 20, 30));
        let letterbox = Letterbox::for_frame(tall.width, tall.height);
        assert_eq!(
            letterbox.scale,
            416.0 / 896.0,
            "the tall side is the constrained one: min(416/896, 416/640)"
        );
        assert_eq!(
            content_span(tall.width, letterbox.scale),
            288,
            "640 * (416/896) = 297.14, floored to 32 x 9 = 288"
        );
        assert_eq!(
            content_span(tall.height, letterbox.scale),
            416,
            "896 * (416/896) = 416 exactly, and the tall side fills the tensor"
        );

        // Row 2704 is the first stride-16 anchor, at grid (0, 0), so with raw
        // (0, 0) its centre is the tensor's origin and exp(0) * 16 = 16 for
        // both dimensions: tensor box (-8, -8, 16, 16). The inverse map divides
        // by this frame's own scale of 416/896, so the frame box is
        // (-8 * 896/416, -8 * 896/416, 16 * 896/416, 16 * 896/416) =
        // (-17.230770, -17.230770, 34.461540, 34.461540). The box hangs off the
        // frame's top-left corner, and that is deliberate: the inverse map must
        // not clip it, because clipping moves a box's centre away from the
        // object it describes and `iou` would then be computed about the wrong
        // centre.
        let (detector, _) = detector_returning(predictions_output(tensor_with_one_row(
            2704,
            0.0,
            PERSON_CLASS as usize,
            0.0,
            [0.0, 0.0, 0.0, 0.0],
        )));
        let detections = detector.detect(&tall).expect("a well-formed output");
        assert_eq!(detections.len(), 1);
        assert_eq!(
            detections[0].rect,
            Rect {
                x: -17.23077_f32,
                y: -17.23077_f32,
                w: 34.46154_f32,
                h: 34.46154_f32
            },
            "a stride-16 anchor at the grid's origin, un-clipped"
        );
        // And the frame's *width* padding is visible in the same assertion:
        // this box is scaled by 896/416 and not by 640/416, which is the
        // difference between dividing by the letterbox's scale and dividing by
        // the ratio of one axis.
        assert_ne!(detections[0].rect.w, 16.0 * 640.0 / 416.0);
    }

    #[test]
    fn detect_rejects_an_output_it_cannot_read() {
        // The field failure behind every one of these is a wrong or mismatched
        // model file. Each is checked for its *variant*, so a wrapper that
        // answered `Ok(vec![])` for a garbage tensor — silently reporting a
        // phone-free frame — cannot pass.
        let frame = letterboxed_frame((10, 20, 30));
        let detect = |outputs: Vec<OutputTensor>| {
            YoloxNano::new(Box::new(FakeSession::new("yolox_nano", outputs))).detect(&frame)
        };

        let error = detect(vec![OutputTensor {
            name: "output0".to_owned(),
            ..predictions_output(blank_predictions())
        }])
        .expect_err("a name the graph does not declare");
        assert!(
            matches!(&error, ModelError::UnexpectedOutput { expected, got } if *expected == "output" && got == "output0"),
            "{error}"
        );

        // 1580 anchors is YuNet's fused-output count: a real near-miss from a
        // different export, not an arbitrary wrong number.
        let error = detect(vec![OutputTensor {
            shape: vec![1, 1580, COLUMNS as i64],
            data: TensorData::F32(vec![0.0; 1580 * COLUMNS]),
            ..predictions_output(Vec::new())
        }])
        .expect_err("an anchor count the graph does not declare");
        assert!(
            matches!(&error, ModelError::UnexpectedShape { expected, got, .. }
                if expected == &vec![1, ANCHORS as i64, 85] && got == &vec![1, 1580, 85]),
            "{error}"
        );

        // 84 columns is the near-miss that matters most: a decode reading
        // `raw[5 + 79]` off an 84-wide row reads the next row's first value as
        // a class score, and nothing about the result looks wrong.
        let error = detect(vec![OutputTensor {
            shape: vec![1, ANCHORS as i64, 84],
            data: TensorData::F32(vec![0.0; ANCHORS * 84]),
            ..predictions_output(Vec::new())
        }])
        .expect_err("a column count the graph does not declare");
        assert!(
            matches!(error, ModelError::UnexpectedShape { .. }),
            "{error}"
        );

        let error = detect(vec![OutputTensor {
            data: TensorData::I32(vec![0; ANCHORS * COLUMNS]),
            ..predictions_output(Vec::new())
        }])
        .expect_err("an integer tensor where floats belong");
        assert!(matches!(error, ModelError::MissingOutput { .. }), "{error}");

        // One row short: every shape is right and the buffer is not. The
        // decode indexes rows, so this must be an error rather than a panic.
        let error = detect(vec![OutputTensor {
            data: TensorData::F32(vec![0.0; ANCHORS * COLUMNS - COLUMNS]),
            ..predictions_output(Vec::new())
        }])
        .expect_err("a buffer shorter than the shape it declares");
        assert!(
            matches!(error, ModelError::UnexpectedDataLength { got, .. } if got == ANCHORS * COLUMNS - COLUMNS),
            "{error}"
        );

        let error = detect(vec![
            predictions_output(blank_predictions()),
            predictions_output(blank_predictions()),
        ])
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

    #[test]
    fn a_blank_tensor_yields_no_detections() {
        // The "the model saw nothing" case, and the one a real blank frame must
        // reach: `blank_row` is objectness -40, so every row scores ~2e-18 and
        // the class gate has nothing to admit. A zero-filled tensor would score
        // 0.25 on all 3549 rows, so this also pins that the decode does not
        // turn an empty prediction into a flood.
        let frame = letterboxed_frame((10, 20, 30));
        let (detector, handle) = detector_returning(predictions_output(blank_predictions()));
        assert!(
            detector
                .detect(&frame)
                .expect("a well-formed output")
                .is_empty(),
            "a tensor of blank rows has no detections"
        );
        assert_eq!(handle.call_count(), 1);

        // And a zero-filled tensor — what `FakeSession::zeros` builds, and what
        // a real graph returns for a frame it has nothing to say about — is
        // *not* blank under this decode. `sigmoid(0)` is 0.5 for the objectness
        // and for all 80 classes, so every row scores 0.25 and admits class 0.
        //
        // Measured, and worth recording rather than asserting away: **all 3549
        // rows survive**, and NMS removes none of them. Each row's box is
        // `exp(0) * stride` square on a `stride`-spaced grid — 8x8 boxes on an
        // 8-pixel grid — so neighbouring boxes abut exactly, their intersection
        // is a line, and the IoU is 0.0 rather than the >0.45 suppression
        // needs. The `score_thr = 0.1` floor does not help either: 0.25 is
        // twice it. This is a property of the brief's formula that Task 12 will
        // meet, and it means a phone-free frame cannot be recognised by its
        // score alone.
        let (detector, _) = detector_returning(predictions_output(vec![0.0; ANCHORS * COLUMNS]));
        let detections = detector.detect(&frame).expect("a well-formed output");
        assert_eq!(
            detections.len(),
            ANCHORS,
            "a zeroed tensor scores 0.25 on every row, and abutting boxes do \
             not suppress each other"
        );
        assert_eq!(detections[0].score, 0.25);
        assert_eq!(detections[0].class_id, PERSON_CLASS);
        assert_eq!(
            detections[0].rect,
            Rect {
                x: -16.0,
                y: -16.0,
                w: 32.0,
                h: 32.0
            },
            "grid (0, 0) of stride 8, exp(0) * 8 = 8 in the tensor, x4 out of \
             the letterbox: a box centred on the tensor's own corner"
        );
        assert!(
            detections.iter().all(|det| det.class_id == PERSON_CLASS),
            "every one of them is class 0, by the lowest-id tie-break"
        );
    }

    #[test]
    fn preprocessing_is_total_for_a_frame_the_letterbox_cannot_scale() {
        // A zero-area frame, and frames whose aspect ratio makes the scaled
        // size smaller than one stride multiple. None of these is impossible
        // from a hand-built `Frame`, and the wrapper has to stay total: the
        // tensor must still be the graph's `[1, 3, 416, 416]` with finite
        // values in it, and `detect` must answer rather than divide by a zero
        // scale and hand back NaN boxes.
        for (label, frame) in [
            ("one pixel", solid_frame(1, 1, (255, 0, 0))),
            ("a sliver", solid_frame(1000, 3, (10, 20, 30))),
            ("taller than wide", solid_frame(3, 1000, (10, 20, 30))),
            ("smaller than one stride", solid_frame(31, 31, (10, 20, 30))),
        ] {
            let tensor = preprocess(&frame);
            assert_eq!(tensor.shape, vec![1, 3, 416, 416], "{label}");
            assert_eq!(
                values(&tensor).len(),
                CHANNELS * INPUT_SIZE * INPUT_SIZE,
                "{label}"
            );
            assert!(
                values(&tensor).iter().all(|value| value.is_finite()),
                "{label} put a non-finite value in the tensor"
            );

            // The caller's path stays total too. The configured row is inside
            // the content band for a 1x1 frame (whose content rounds to zero
            // and so is entirely pad, but whose scale is 1.0, not 0.0) and
            // filtered out for the rest — either way `detect` must return a
            // `Result` rather than panic or produce a non-finite box.
            let (detector, _) = detector_returning(predictions_output(tensor_with_one_row(
                0,
                0.0,
                PERSON_CLASS as usize,
                0.0,
                [0.0, 0.0, 0.0, 0.0],
            )));
            let detections = detector
                .detect(&frame)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            for detection in &detections {
                let rect = detection.rect;
                assert!(
                    [rect.x, rect.y, rect.w, rect.h]
                        .iter()
                        .all(|v| v.is_finite()),
                    "{label} produced a non-finite box: {rect:?}"
                );
                assert!(detection.score.is_finite(), "{label}: non-finite score");
            }
        }

        // A sliver's letterbox rounds its short side to zero stride multiples,
        // so the content band is empty and the tensor is entirely pad. That is
        // the honest reading — there is no 32-pixel-aligned content to write —
        // and it is what keeps `content_span` from yielding a partial band.
        let sliver = solid_frame(1000, 3, (10, 20, 30));
        let letterbox = Letterbox::for_frame(sliver.width, sliver.height);
        assert_eq!(
            content_span(sliver.height, letterbox.scale),
            0,
            "3 * (416/1000) rounds down to 0"
        );
        assert_eq!(
            content_span(sliver.width, letterbox.scale),
            416,
            "the long side still fills the tensor"
        );
        let tensor = preprocess(&sliver);
        assert!(
            values(&tensor)
                .iter()
                .enumerate()
                .all(|(index, value)| *value
                    == (PAD_VALUE - MEAN[index / (INPUT_SIZE * INPUT_SIZE)])
                        / STD[index / (INPUT_SIZE * INPUT_SIZE)]),
            "an empty content band leaves the whole tensor at the pad value"
        );

        // A frame with no pixels at all. The scale reads as 1.0 so that the
        // inverse map is a division by one and not by zero, the tensor is
        // entirely pad, and there is nothing to detect.
        let empty =
            Frame::new(0, 0, Vec::new(), Instant::now()).expect("an empty frame is a frame");
        assert_eq!(Letterbox::for_frame(empty.width, empty.height).scale, 1.0);
        let tensor = preprocess(&empty);
        assert_eq!(tensor.shape, vec![1, 3, 416, 416]);
        assert!(
            values(&tensor)
                .iter()
                .enumerate()
                .all(|(index, value)| *value
                    == (PAD_VALUE - MEAN[index / (INPUT_SIZE * INPUT_SIZE)])
                        / STD[index / (INPUT_SIZE * INPUT_SIZE)]),
            "a frame with no pixels is entirely pad"
        );

        let (detector, handle) = detector_returning(predictions_output(tensor_with_one_row(
            0,
            0.0,
            PERSON_CLASS as usize,
            0.0,
            [0.0, 0.0, 0.0, 0.0],
        )));
        let detections = detector
            .detect(&empty)
            .expect("an empty frame is still a frame, and the graph still runs");
        assert_eq!(handle.call_count(), 1, "the session is still asked to run");
        // The row *is* returned — it is the model's prediction and the wrapper
        // has no grounds to discard it — but at a scale of 1.0 its box is the
        // tensor's own, with no division by a zero scale anywhere.
        assert_eq!(detections.len(), 1);
        assert_eq!(
            detections[0].rect,
            Rect {
                x: -4.0,
                y: -4.0,
                w: 8.0,
                h: 8.0
            },
            "grid (0, 0) of stride 8 with raw (0, 0): exp(0) * 8 = 8"
        );
    }

    #[test]
    fn preprocessing_is_deterministic_and_samples_the_frames_corner() {
        // Pure: same frame in, same tensor out.
        let frame = letterboxed_frame((10, 20, 30));
        assert_eq!(preprocess(&frame), preprocess(&frame));

        // Which pixels, not merely that some were: a frame whose three channels
        // encode its coordinates says where every sample came from.
        //
        // The frame is 512x512 and the scale is `416/512 = 0.8125`, so the
        // content band is exactly 416x416 — `512 * 0.8125 = 416`, already a
        // multiple of 32 — and there is no padding at all in this one. The
        // channel encoding is the frame's own coordinates taken modulo 128, so
        // every value fits in a byte and, within a 128-pixel neighbourhood, each
        // pixel identifies itself uniquely.
        //
        // **The modulo is not decoration.** Writing `|x, y| (x as u8, y as u8, 0)`
        // on a 1664-wide ramp truncates: `1661 as u8` is 125, so the frame
        // silently contains several columns with the same blue and the "far
        // corner" sample appears to come from x=125. That mistake cost an hour
        // of chasing the *production* code, which was right.
        let ramp = frame_with(416, 416, |x, y| ((x % 256) as u8, (y % 256) as u8, 0));
        let tensor = preprocess(&ramp);
        let letterbox = Letterbox::for_frame(ramp.width, ramp.height);
        assert_eq!(
            letterbox.scale, 1.0,
            "the frame is already the tensor's size"
        );
        assert_eq!(content_span(ramp.width, letterbox.scale), 416);
        assert_eq!(content_span(ramp.height, letterbox.scale), 416);

        // Tensor (0, 0) samples frame (0.0, 0.0) exactly: at a scale of 1.0 the
        // pixel-centre map is the identity, so bilinear interpolation returns
        // the frame's own pixel rather than a blend of four.
        assert_eq!(
            at(&tensor, 0, 0, 0),
            (0.0 - MEAN[0]) / STD[0],
            "tensor (0,0) is frame (0,0)"
        );
        assert_eq!(at(&tensor, 1, 0, 0), (0.0 - MEAN[1]) / STD[1]);
        assert_eq!(
            at(&tensor, 2, 0, 0),
            (0.0 - MEAN[2]) / STD[2],
            "and the ramp leaves red at zero"
        );

        // Every sample is the frame's own pixel, in both channels, at the
        // corners and in the middle. A sampler that read the frame's top-left
        // corner for every output passes the (0,0) case above and fails the
        // rest; one that transposed the axes fails the (415, 0) and (0, 415)
        // cases.
        //
        // The expectation carries the ramp's own `% 256`, because a byte cannot
        // hold 415: the frame's column 415 is blue **159**, and writing `415`
        // here is the same truncation that made an earlier version of this test
        // appear to indict the production code. The point of the case is not the
        // value 415, it is that the *sample* comes from column 415 rather than
        // from wherever a wrong map would look — and `159 != 207 != 0` is enough
        // to tell those apart.
        for (column, row) in [(1, 0), (0, 1), (415, 0), (0, 415), (415, 415), (207, 133)] {
            assert_eq!(
                at(&tensor, 0, column, row),
                ((column % 256) as f32 - MEAN[0]) / STD[0],
                "blue at ({column}, {row}) is the frame's own column"
            );
            assert_eq!(
                at(&tensor, 1, column, row),
                ((row % 256) as f32 - MEAN[1]) / STD[1],
                "green at ({column}, {row}) is the frame's own row"
            );
            assert_eq!(
                at(&tensor, 2, column, row),
                (0.0 - MEAN[2]) / STD[2],
                "and red stays zero"
            );
        }

        // Hand-computed literals for the far corner, so that a mutation of
        // `MEAN` or `STD` fails here rather than moving the expectation with it.
        // Column 415 of the ramp is blue 159 and green 159, so the tensor is
        // (159 - 103.53)/57.375 = 0.9667974 and (159 - 116.28)/57.12 =
        // 0.7478992 — and a BGR/RGB swap would give a different pair, since
        // the two channels' means and stds differ.
        assert_eq!(at(&tensor, 0, 415, 415), 0.9667974_f32);
        assert_eq!(at(&tensor, 1, 415, 415), 0.7478992_f32);
        assert_eq!(at(&tensor, 0, 0, 0), -1.8044444_f32);
        // Column 207 is under 256, so it is its own value: (207 - 103.53)/57.375
        // = 1.8032244, where the identity would give the same and a transposed
        // sample would give green's 1.5806373.
        assert_eq!(at(&tensor, 0, 207, 133), 1.8033987_f32);
        assert_eq!(at(&tensor, 1, 207, 133), 0.2927171_f32);

        // The *scale* is what the sampler divides by, and at scale 1.0 that
        // division is invisible — so the padding test below uses 4.0, and this
        // is where the two geometries are tied together. `remap` at scale 0.25
        // puts tensor 1 at frame 5.5, four columns along.
        assert_eq!(
            remap(1.0, 0.25),
            5.5,
            "four frame columns per tensor column"
        );
        assert_eq!(remap(415.0, 0.25), 1661.5);
        assert_ne!(
            remap(415.0, 0.25),
            415.0,
            "a sampler that forgot to divide by the scale reads frame 415"
        );

        // Row 223 is the last content row and row 224 the first pad row on the
        // 4:1 frame — the one this test's scale of 1.0 does not have — so the
        // boundary is checked there.
        let padded = letterboxed_frame((10, 20, 30));
        let padded_tensor = preprocess(&padded);
        assert_ne!(
            at(&padded_tensor, 0, 0, 223),
            at(&padded_tensor, 0, 0, 224),
            "the content band ends where the padding begins"
        );

        // A horizontally flipped frame gives a different tensor, so the
        // sampling is reading the frame rather than a constant.
        let flipped = frame_with(416, 416, |x, _| (((415 - x) % 256) as u8, 0, 0));
        assert_ne!(preprocess(&ramp).data, preprocess(&flipped).data);
    }
}
