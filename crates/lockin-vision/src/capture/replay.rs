//! Replaying a fixture directory: one decoded image per pull.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{CaptureError, FrameSource};
use crate::frame::Frame;

/// A [`FrameSource`] over a directory of numbered images.
///
/// [`ReplaySource::open`] lists the images and sorts them lexicographically, so
/// `0001.png` is the first frame and `0002.png` the second; each pull decodes the
/// next file, which keeps a long fixture to one frame in memory.
#[derive(Debug)]
pub struct ReplaySource {
    paths: Vec<PathBuf>,
    next: usize,
}

impl ReplaySource {
    /// Opens a fixture directory, which must hold at least one `*.png` or `*.jpg`.
    ///
    /// Images are listed, not read: one that cannot be decoded is reported by
    /// [`FrameSource::next_frame`] when it is reached, and the source stays where
    /// it was, so a retry reports the same image rather than skipping past it.
    pub fn open(dir: &Path) -> Result<ReplaySource, CaptureError> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if is_fixture_image(&path) {
                paths.push(path);
            }
        }
        if paths.is_empty() {
            return Err(CaptureError::EmptyFixture);
        }
        paths.sort();
        Ok(ReplaySource { paths, next: 0 })
    }
}

impl FrameSource for ReplaySource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        let Some(path) = self.paths.get(self.next) else {
            return Ok(None);
        };
        let image = image::open(path).map_err(|source| CaptureError::Decode {
            path: path.clone(),
            source,
        })?;
        let rgba = image.into_rgba8();
        let (width, height) = rgba.dimensions();
        let mut bgra = rgba.into_raw();
        // `image` decodes to RGBA8; a frame holds the same pixels as BGRA8.
        for pixel in bgra.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        // Wall clock at read time: replay timing is the caller's tick rate, not a
        // timestamp baked into the fixture.
        let frame = Frame::new(width, height, bgra, Instant::now())?;
        self.next += 1;
        Ok(Some(frame))
    }
}

/// Whether `path` names a fixture image: a `png` or `jpg` extension, so a README
/// or a subdirectory in the fixture directory is not a frame.
fn is_fixture_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("png" | "jpg")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// Writes a solid-colour image, the format being taken from `name`.
    fn write_fixture(dir: &Path, name: &str, width: u32, height: u32, rgb: [u8; 3]) {
        RgbImage::from_pixel(width, height, Rgb(rgb))
            .save(dir.join(name))
            .expect("the fixture image must encode");
    }

    #[test]
    fn replay_source_reads_numbered_images_in_order() {
        let dir = tempfile::tempdir().unwrap();
        // Different sizes, so each frame's dimensions come from its own image.
        write_fixture(dir.path(), "0001.png", 2, 3, [0xff, 0x00, 0x00]); // red
        write_fixture(dir.path(), "0002.jpg", 3, 2, [0x00, 0x00, 0xff]); // blue
        // Only png and jpg are fixtures; anything else in the directory is not.
        fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();

        let mut source = ReplaySource::open(dir.path()).expect("two fixtures, so not empty");

        // `captured_at` is the wall clock at read time, not a fixture timestamp.
        let before = Instant::now();
        let red = source.next_frame().unwrap().expect("0001.png");
        let after = Instant::now();
        assert!(
            before <= red.captured_at && red.captured_at <= after,
            "captured_at must be the read time, not a baked-in or virtual stamp"
        );

        assert_eq!((red.width, red.height), (2, 3));
        assert_eq!(
            red.bgra,
            [0x00, 0x00, 0xff, 0xff].repeat(6),
            "0001.png is red"
        );

        // JPEG is lossy, so only the dominant channel is asserted here.
        let blue = source.next_frame().unwrap().expect("0002.jpg");
        assert_eq!((blue.width, blue.height), (3, 2));
        assert!(
            blue.bgra
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[2] < 0x40 && pixel[0] > 0xc0),
            "0002.jpg is blue: {:?}",
            blue.bgra
        );

        // Red came first, so a source that ignored the sort, or reversed it,
        // fails the two content assertions above.
        assert!(
            source.next_frame().unwrap().is_none(),
            "two fixtures, so two frames"
        );
    }

    #[test]
    fn replay_source_errors_on_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            ReplaySource::open(dir.path()),
            Err(CaptureError::EmptyFixture)
        ));

        // A directory holding no image is empty of fixtures even though it has
        // entries: the extension filter decides, not the entry count.
        fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();
        assert!(matches!(
            ReplaySource::open(dir.path()),
            Err(CaptureError::EmptyFixture)
        ));

        // A directory that cannot be read is a different mistake, and stays
        // distinguishable from a fixture directory with no images in it.
        assert!(matches!(
            ReplaySource::open(&dir.path().join("absent")),
            Err(CaptureError::Io(_))
        ));
    }
}
