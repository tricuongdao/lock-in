//! Frame sources: the seam that keeps camera access out of the pipeline.
//!
//! Every stage downstream of a [`FrameSource`] pulls frames through this trait,
//! so a live camera, a generated pattern and a recorded fixture are
//! interchangeable — which is what lets the whole pipeline be tested with no
//! camera present.

mod camera;
mod replay;
mod synthetic;

pub use camera::CameraSource;
pub use replay::ReplaySource;
pub use synthetic::{SyntheticPattern, SyntheticSource};

use std::path::PathBuf;

use crate::frame::{Frame, FrameError};

/// Why a [`FrameSource`] could not produce a frame.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// The fixture directory holds no image a replay source can read.
    #[error("fixture directory holds no png or jpg image")]
    EmptyFixture,

    /// The fixture directory could not be read at all.
    #[error("cannot read the fixture directory")]
    Io(#[from] std::io::Error),

    /// A fixture image could not be decoded.
    #[error("cannot decode fixture image {}", path.display())]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    /// A camera could not be opened, or a frame could not be read from one.
    ///
    /// A missing camera, a camera another application already holds, and a
    /// device that fails mid-stream are all the same thing to a caller looking
    /// for a frame: this tick has none. The backend's own error is kept as the
    /// source rather than flattened into a string.
    #[error("camera {index}: {source}")]
    Camera {
        index: u32,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A decoded image does not match the [`Frame`] layout.
    #[error("decoded image does not match the frame layout")]
    Frame(#[from] FrameError),
}

/// A pull source of [`Frame`]s.
///
/// `Ok(Some(frame))` while frames remain, `Ok(None)` once the source is
/// exhausted — permanently, so a caller's loop terminates — and `Err` when a
/// frame could not be read.
pub trait FrameSource {
    /// Returns the next frame, or `None` once this source is exhausted.
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError>;
}
