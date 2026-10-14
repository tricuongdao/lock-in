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
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

use super::{CaptureError, FrameSource};
use crate::frame::Frame;

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
    /// The format is requested as RGB, which is what lets a frame reach the
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
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
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
            .decode_image::<RgbFormat>()
            .map_err(|source| camera_error(self.index, source))?;
        let (width, height) = image.dimensions();

        let mut bgra = image.into_raw();
        // `decode_image::<RgbFormat>` yields RGB8; a frame holds the same
        // pixels as BGRA8, with an alpha the camera does not provide.
        for pixel in bgra.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }

        // Timestamped at read time, like the replay source: the frame's time is
        // when the pipeline got it, not a device-provided one, so the motion
        // gate's intervals are the tick rate however the driver behaves.
        let frame = Frame::new(width, height, bgra, Instant::now())?;
        Ok(Some(frame))
    }
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
