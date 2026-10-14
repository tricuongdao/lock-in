//! The live camera: nokhwa over Media Foundation on Windows.
//!
//! This is the only file in the crate that touches a device. It reads frames
//! and hands them to the pipeline as [`Frame`]s; it writes nothing anywhere.
//! There is deliberately no recording, no buffering to disk and no network
//! path in this module — the fixture writer is Task 21's, and keeping the two
//! apart is what makes "no frame data leaves the process" auditable by reading
//! one file.
//!
//! A camera that is absent, or that Zoom or Teams is already holding, is a
//! normal condition rather than a crash: every failure here is a
//! [`CaptureError`], and [`CameraSource::open`] has no path that panics.

use std::time::Instant;

use nokhwa::Camera;
use nokhwa::nokhwa_initialize;
use nokhwa::pixel_format::RgbAFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

use super::{CaptureError, FrameSource};
use crate::frame::Frame;

/// Bytes per pixel a camera frame is decoded to.
///
/// Four, not three. [`RgbAFormat`] decodes to RGBA8, which is deliberate: a
/// [`Frame`] holds BGRA8, so a 4-byte decode needs only a channel swap, whereas
/// a 3-byte RGB decode would have to be widened a pixel at a time — and until
/// it was, it would mismatch the `width * height * 4` a `Frame` demands and
/// every `Frame::new` would fail, so the source would never yield a frame at
/// all.
const BYTES_PER_PIXEL: usize = 4;

/// A live webcam, pulled for one frame at a time.
///
/// The stream is opened on the first pull rather than in [`open`](Self::open),
/// so opening a camera is cheap and a camera that is busy resolves to an error
/// on the tick that needs the frame instead of blocking startup.
pub struct CameraSource {
    index: u32,
    camera: Camera,
}

impl CameraSource {
    /// Opens the camera at `index`, the platform's device number — `0` is
    /// usually the built-in webcam.
    ///
    /// The format is requested as RGBA, which is what lets a frame reach the
    /// pipeline without this crate carrying a YUYV, NV12 and JPEG decoder of
    /// its own: Media Foundation hands the device's native format to nokhwa,
    /// and nokhwa converts it.
    ///
    /// Returns `Err` — never panics — when there is no such device, when
    /// another application already holds it, or when the backend cannot start.
    pub fn open(index: u32) -> Result<CameraSource, CaptureError> {
        // Media Foundation has to be started before any device is opened, and
        // this starts it for the process — it is not scoped to the call. It is
        // idempotent, so opening a second camera is fine, and it is also
        // asynchronous and reports failure through its callback rather than
        // here, which is why its result is ignored: a camera that never becomes
        // available fails at `Camera::new` below, where the error can be
        // returned to the caller who asked for a frame.
        nokhwa_initialize(|_granted| {});

        let format =
            RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let camera = Camera::new(CameraIndex::Index(index), format)
            .map_err(|source| camera_error(index, source))?;
        Ok(CameraSource { index, camera })
    }

    /// The device number this source was opened at.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl FrameSource for CameraSource {
    /// The next frame from the camera.
    ///
    /// Never returns `Ok(None)`: a camera does not run out of frames, and
    /// reporting exhaustion would tell a caller's loop to stop. A device that
    /// has been unplugged or taken reports `Err` for that tick instead, and the
    /// caller decides whether to reopen.
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        let buffer = self
            .camera
            .frame()
            .map_err(|source| camera_error(self.index, source))?;
        let image = buffer
            .decode_image::<RgbAFormat>()
            .map_err(|source| camera_error(self.index, source))?;
        let (width, height) = image.dimensions();

        // Timestamped at read time, like the replay source: the frame's time is
        // when the pipeline got it, not a device-provided one, so the motion
        // gate's intervals are the tick rate however the driver behaves.
        let frame = frame_from_rgba(width, height, image.into_raw(), Instant::now())?;
        Ok(Some(frame))
    }
}

/// Wraps a decoded RGBA8 buffer as the [`Frame`] the pipeline reads.
///
/// Split out from [`FrameSource::next_frame`] so it can be tested without a
/// device: the camera path is otherwise unverifiable on a machine with no
/// webcam, and a byte-width mistake there is invisible until a real camera is
/// plugged in. It is pure, so a test can hand it a buffer it built.
///
/// Returns `FrameError::BadBufferLength` for a buffer that is not exactly
/// `width * height * 4` bytes — which is the failure a 3-byte RGB decode would
/// have produced on every single frame.
fn frame_from_rgba(
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    captured_at: Instant,
) -> Result<Frame, crate::frame::FrameError> {
    let mut bgra = rgba;
    // Four bytes per pixel, so every chunk is a whole pixel. `as_chunks_mut`
    // rather than `chunks_exact_mut`: the latter leaves a partial-chunk path
    // that cannot be reached for a length this function has already accepted,
    // and the difference is the type of the chunk rather than a test.
    for pixel in bgra.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
        // RGBA8 in, BGRA8 out: the red and blue channels trade places and the
        // alpha the camera supplied is kept where it is. Swapping in place is
        // why the intermediate is the caller's own vec rather than a copy.
        pixel.swap(0, 2);
    }
    Frame::new(width, height, bgra, captured_at)
}

/// Wraps a backend failure with the device number it happened to.
///
/// Nokhwa's own error is kept as the source rather than flattened into a
/// string: it is what distinguishes "no such device" from "device busy", and
/// that distinction is the difference between telling a user to plug a webcam
/// in and telling them to close Zoom.
fn camera_error(index: u32, source: nokhwa::NokhwaError) -> CaptureError {
    CaptureError::Camera {
        index,
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameError;

    #[test]
    fn rgba_from_the_camera_becomes_bgra_in_the_frame() {
        // Two pixels, chosen so every channel is distinct — a swap that moved
        // the wrong pair, or dropped one, cannot pass. Pixel 0 is
        // RGBA (1, 2, 3, 4); pixel 1 is (250, 251, 252, 253).
        let rgba = vec![1u8, 2, 3, 4, 250, 251, 252, 253];
        let frame =
            frame_from_rgba(2, 1, rgba, Instant::now()).expect("two pixels, four bytes each");

        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(
            frame.bgra,
            vec![
                3, 2, 1, 4, // RGBA (1, 2, 3, 4) is BGRA (3, 2, 1, 4)
                252, 251, 250, 253, // and RGBA (250, .., 252, 253) is (252, .., 250, 253)
            ],
            "red and blue trade places and alpha stays put, per pixel"
        );

        // The buffer is exactly `width * height * 4` — two pixels of four bytes
        // — which is the whole point: a 3-byte-per-pixel decode would make
        // `Frame::new` reject every frame, so the length is asserted as well as
        // the contents.
        assert_eq!(frame.bgra.len(), 8);
    }

    #[test]
    fn a_byte_width_mismatch_is_an_error_not_a_frame() {
        // The exact shape of the bug this function exists to prevent: six
        // bytes for two pixels, which is 3 bytes per pixel — what `RgbFormat`
        // hands back. It must be reported, not swapped into a scrambled frame.
        let rgb_shaped = vec![0u8; 6];
        assert!(matches!(
            frame_from_rgba(2, 1, rgb_shaped, Instant::now()),
            Err(FrameError::BadBufferLength {
                expected: 8,
                got: 6
            })
        ));

        // A buffer with a trailing partial pixel is the other way this can go
        // wrong, and the one that would make an in-place swap panic if it
        // assumed whole pixels.
        let ragged = vec![0u8; 11];
        assert!(matches!(
            frame_from_rgba(2, 1, ragged, Instant::now()),
            Err(FrameError::BadBufferLength {
                expected: 8,
                got: 11
            })
        ));

        // And a correct buffer is accepted, so the two cases above are not
        // passing because everything fails.
        assert!(frame_from_rgba(2, 1, vec![0u8; 8], Instant::now()).is_ok());
    }

    #[test]
    fn camera_open_failure_is_not_fatal() {
        // 9999 is not a device any backend enumerates. The contract the engine
        // depends on is that this is an `Err` — a missing camera, or one Zoom
        // is holding, must never take the process down or panic.
        let opened = CameraSource::open(9999);
        assert!(
            opened.is_err(),
            "there is no camera at index 9999, so open must fail"
        );

        // ...and it fails as a camera error carrying the index that was asked
        // for, so a caller can say *which* device was unavailable.
        match opened {
            Err(CaptureError::Camera { index, .. }) => {
                assert_eq!(index, 9999, "the error names the device asked for");
            }
            Err(other) => panic!("expected a camera error, got {other:?}"),
            Ok(source) => panic!(
                "opening camera 9999 succeeded, which means this machine has \
                 one and the test cannot assert absence: {}",
                source.index()
            ),
        }

        // Opening twice must fail the same way rather than panicking or
        // latching something that makes the second attempt behave differently:
        // the pipeline reopens a camera on a tick, so a failed open is a
        // routine event, not a one-off. This is deliberately not a timing
        // assertion — it checks the *outcome* repeats, which is the part the
        // engine depends on.
        assert!(
            CameraSource::open(9999).is_err(),
            "a second failed open must fail the same way, not succeed or panic"
        );
    }
}
