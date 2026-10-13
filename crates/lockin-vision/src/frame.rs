//! The captured-frame type every later stage consumes.

use std::time::Instant;

/// Output size of [`Frame::luma_64x48`]. The motion gate compares successive
/// frames at this resolution, where sub-pixel accuracy is irrelevant.
const LUMA_WIDTH: usize = 64;
const LUMA_HEIGHT: usize = 48;
const LUMA_LEN: usize = LUMA_WIDTH * LUMA_HEIGHT;

/// Why a [`Frame`] could not be built.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// The buffer does not hold exactly one BGRA8 pixel per `width * height`.
    #[error("BGRA buffer holds {got} bytes, expected {expected}")]
    BadBufferLength { expected: usize, got: usize },
}

/// One captured frame, held as BGRA8 in memory.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    pub captured_at: Instant,
}

impl Frame {
    /// Wraps a BGRA8 buffer, which must hold exactly `width * height * 4` bytes.
    pub fn new(
        width: u32,
        height: u32,
        bgra: Vec<u8>,
        captured_at: Instant,
    ) -> Result<Frame, FrameError> {
        // Saturating rather than wrapping: no buffer can ever be this long, so a
        // saturated total always mismatches and the frame is rejected.
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if bgra.len() != expected {
            return Err(FrameError::BadBufferLength {
                expected,
                got: bgra.len(),
            });
        }
        Ok(Frame {
            width,
            height,
            bgra,
            captured_at,
        })
    }

    /// Nearest-neighbour downsample to `64x48` Rec. 601 luma, for the motion gate.
    ///
    /// Destination `(x, y)` samples source `(x * width / 64, y * height / 48)` —
    /// the top-left pixel of its block, integer division, never an average.
    /// Weights are integer `(77*r + 150*g + 29*b) >> 8`: the motion gate only
    /// needs "did this pixel change", and a box filter is measurably slower.
    pub fn luma_64x48(&self) -> [u8; LUMA_LEN] {
        let mut out = [0u8; LUMA_LEN];
        // A zero-area frame has no pixels to sample; stay black rather than
        // index past the end of the buffer.
        if self.width == 0 || self.height == 0 {
            return out;
        }
        let (width, height) = (self.width as usize, self.height as usize);
        for y in 0..LUMA_HEIGHT {
            let source_y = y * height / LUMA_HEIGHT;
            for x in 0..LUMA_WIDTH {
                let source_x = x * width / LUMA_WIDTH;
                let offset = (source_y * width + source_x) * 4;
                let (b, g, r) = (
                    u32::from(self.bgra[offset]),
                    u32::from(self.bgra[offset + 1]),
                    u32::from(self.bgra[offset + 2]),
                );
                out[y * LUMA_WIDTH + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_buffer_of_wrong_length() {
        let e = Frame::new(2, 2, vec![0u8; 15], Instant::now());
        assert!(matches!(
            e,
            Err(FrameError::BadBufferLength {
                expected: 16,
                got: 15
            })
        ));
    }

    #[test]
    fn luma_is_deterministic_and_sized() {
        let f = Frame::new(64, 48, vec![7u8; 64 * 48 * 4], Instant::now()).unwrap();
        assert_eq!(f.luma_64x48(), f.luma_64x48());
        assert_eq!(f.luma_64x48().len(), 3072);
    }
}
