//! Camera intrinsics: the pinhole matrix that turns a point in front of the
//! camera into a pixel.
//!
//! Cached once per capture resolution and handed to the head-pose solve, which
//! needs `fx`/`fy` to tell a rotation of the head from a translation of it.
//! Wrong intrinsics do not make the pose *noisy*, they make it *biased*: a
//! monocular solve with a focal length that is off by a fifth reports pitch and
//! yaw that are off by a similar proportion, in the same direction, in every
//! frame. Calibration then measures that constant error as the wearer's neutral
//! and writes it into the profile, where nothing downstream can see it any
//! more. This module exists to keep that failure visible: [`CameraIntrinsics::from_frame`]
//! says out loud that its estimate is a guess, and
//! [`CameraIntrinsics::from_calibration`] is the only way to replace the guess
//! with a measurement.

use crate::frame::Frame;

/// The four free parameters of the pinhole camera matrix
///
/// ```text
///     [ fx   0  cx ]
/// K = [  0  fy  cy ]
///     [  0   0   1 ]
/// ```
///
/// in pixels, with the origin at the frame's top-left corner and `+x` right,
/// `+y` down — the same convention as [`Frame`] and the rest of [`crate::analyze`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along the horizontal axis.
    pub fx: f32,
    /// Focal length in pixels along the vertical axis. Equal to `fx` when the
    /// pixels are square, which is the assumption everywhere in this crate.
    pub fy: f32,
    /// Principal point x: the horizontal pixel where the optical axis meets the
    /// sensor.
    pub cx: f32,
    /// Principal point y: the vertical pixel where the optical axis meets the
    /// sensor.
    pub cy: f32,
}

impl CameraIntrinsics {
    /// The intrinsics to use for a frame when nothing has been calibrated:
    /// **a guess, not a measurement.**
    ///
    /// The focal length is guessed as the frame's wider dimension in pixels
    /// (`max(width, height)`, the usual "typical webcam" rule of thumb) and the
    /// principal point is put at the exact centre of the frame. Both are
    /// assumptions about hardware nobody measured: a real lens has a focal
    /// length that is neither exactly the frame's long edge nor necessarily the
    /// same in x and y, and its optical axis meets the sensor a few pixels away
    /// from the centre.
    ///
    /// This estimate is the source of the systematic bias the plan records as
    /// risk R15. A monocular `solvePnP` cannot separate "the head is turned"
    /// from "the focal length is wrong": the error is absorbed into a pitch and
    /// a yaw that are wrong by the same amount in every frame, and the
    /// calibration walk then learns that consistent error as the wearer's
    /// neutral — leaving a profile that treats a slightly turned head as
    /// looking straight at the screen, with no signal left to notice it by.
    ///
    /// So treat this as a first-run fallback, never as the intended value. The
    /// `probe` binary prints it precisely so a real measurement can be
    /// substituted, and [`CameraIntrinsics::from_calibration`] is the path that
    /// does the substituting.
    ///
    /// A frame with no area has no wider dimension and no centre, so the
    /// estimate collapses to zero rather than inventing a focal length for an
    /// image that does not exist.
    pub fn from_frame(frame: &Frame) -> CameraIntrinsics {
        // The frame's dimensions are the only input, so the estimate is a pure
        // function of them — no camera, no clock, no state. The `u32` to `f32`
        // conversion is lossless for any frame under 2^24 pixels a side and
        // rounds above that; halving is exact in binary floating point, so the
        // centre is not an approximation of half the width, it *is* half the
        // width.
        let width = frame.width as f32;
        let height = frame.height as f32;
        let focal = width.max(height);
        CameraIntrinsics {
            fx: focal,
            fy: focal,
            cx: width / 2.0,
            cy: height / 2.0,
        }
    }

    /// The intrinsics a calibration measured, stored verbatim.
    ///
    /// Prefer this constructor: it takes the four numbers off a real
    /// calibration instead of guessing them from the frame's shape, which is
    /// what keeps the pose solve clear of R15's bias. It is a plain value
    /// constructor — no frame is consulted and nothing is scaled or adjusted —
    /// so intrinsics measured at one resolution may only be handed to a frame
    /// at another if the caller has already scaled them; both focal lengths and
    /// the principal point scale with resolution, and nothing here does that
    /// for you.
    pub fn from_calibration(fx: f32, fy: f32, cx: f32, cy: f32) -> CameraIntrinsics {
        CameraIntrinsics { fx, fy, cx, cy }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// A frame of the given size, black. Only its dimensions matter here.
    fn frame(width: u32, height: u32) -> Frame {
        Frame::new(
            width,
            height,
            vec![0u8; (width * height * 4) as usize],
            Instant::now(),
        )
        .unwrap()
    }

    #[test]
    fn principal_point_is_the_frame_centre_not_a_swapped_axis() {
        // Half of 640 is 320 and half of 480 is 240, both exact in f32.
        let k = CameraIntrinsics::from_frame(&frame(640, 480));
        assert_eq!((k.cx, k.cy), (320.0, 240.0));

        // The transposed frame, where the two centres are the swapped pair.
        // A `cx`/`cy` mix-up that happened to survive the landscape case — one
        // that reads the width for both axes, say — cannot survive this one,
        // and neither can a pair of hard-coded constants.
        let k = CameraIntrinsics::from_frame(&frame(480, 640));
        assert_eq!((k.cx, k.cy), (240.0, 320.0));
    }

    #[test]
    fn default_focal_length_is_the_wider_dimension() {
        // Wider than tall: the estimate is the width. This is what kills an
        // estimate taken from the *shorter* side.
        let k = CameraIntrinsics::from_frame(&frame(640, 480));
        assert_eq!((k.fx, k.fy), (640.0, 640.0));

        // Taller than wide: the estimate is now the height, which is what kills
        // an estimate taken from the width unconditionally — the same wrong
        // number in both orientations would otherwise look right in one.
        let k = CameraIntrinsics::from_frame(&frame(480, 640));
        assert_eq!((k.fx, k.fy), (640.0, 640.0));
    }

    #[test]
    fn from_calibration_overrides_the_estimate() {
        let guess = CameraIntrinsics::from_frame(&frame(640, 480));

        // Four distinct values, none of them equal to anything the estimate
        // holds (640, 640, 320, 240), so no assert below can pass by handing
        // back the guess — and each one pins which argument landed in which
        // field. All four are exact in f32.
        let measured = CameraIntrinsics::from_calibration(1234.5, 1236.25, 300.0, 260.0);
        assert_ne!(measured, guess, "a calibration replaces the estimate");
        assert_eq!(measured.fx, 1234.5);
        assert_eq!(measured.fy, 1236.25);
        assert_eq!(measured.cx, 300.0);
        assert_eq!(measured.cy, 260.0);
    }

    #[test]
    fn a_zero_area_frame_estimates_no_focal_length() {
        // Not a camera anyone owns, but a `Frame` the constructor accepts, and
        // the honest answer for it: no pixels, so no wider dimension and no
        // centre. Pinned so that the estimate stays plain arithmetic over the
        // dimensions rather than growing a floor that would hide the case.
        let k = CameraIntrinsics::from_frame(&frame(0, 0));
        assert_eq!((k.fx, k.fy, k.cx, k.cy), (0.0, 0.0, 0.0, 0.0));
    }
}
