//! A deterministic frame source: no camera, no fixture files.

use std::time::{Duration, Instant};

use super::{CaptureError, FrameSource};
use crate::frame::Frame;

/// Synthetic frames are generated at the reference webcam size, so they stand in
/// for live ones everywhere downstream.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

/// The pixel pattern a [`SyntheticSource`] paints.
#[derive(Debug, Clone, Copy)]
pub enum SyntheticPattern {
    /// Every pixel set to one grey level, fully opaque.
    Solid(u8),
    /// Squares of `cell` pixels alternating black and white, the square over the
    /// origin being black. `cell` is clamped to at least one pixel.
    Checkerboard { cell: u32 },
}

/// Hands out a fixed number of generated frames, then stops.
///
/// Generated frames are not paced: `fps` timestamps them on the source's own
/// clock, one frame period apart, so a test that measures cadence reads the same
/// numbers however fast it pulls.
#[derive(Debug)]
pub struct SyntheticSource {
    pattern: SyntheticPattern,
    /// Frames not handed out yet.
    remaining: usize,
    /// Capture time of the next frame, advanced one period per pull.
    captured_at: Instant,
    /// Gap between frames, from `fps` frames per second.
    frame_period: Duration,
}

impl SyntheticSource {
    /// Yields `count` frames, then `None` for every later call.
    pub fn new(pattern: SyntheticPattern, count: usize, fps: u32) -> SyntheticSource {
        // Zero frames per second has no meaning, so it counts as one.
        let fps = u64::from(fps.max(1));
        SyntheticSource {
            pattern,
            remaining: count,
            captured_at: Instant::now(),
            frame_period: Duration::from_nanos(1_000_000_000 / fps),
        }
    }

    /// One frame of the pattern, as BGRA8.
    fn paint(&self) -> Vec<u8> {
        let mut bgra = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let level = match self.pattern {
                    SyntheticPattern::Solid(level) => level,
                    SyntheticPattern::Checkerboard { cell } => {
                        let cell = cell.max(1);
                        if (x / cell + y / cell) % 2 == 0 {
                            0x00
                        } else {
                            0xff
                        }
                    }
                };
                bgra.extend_from_slice(&[level, level, level, 0xff]);
            }
        }
        bgra
    }
}

impl FrameSource for SyntheticSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let captured_at = self.captured_at;
        self.captured_at += self.frame_period;
        Ok(Some(Frame::new(WIDTH, HEIGHT, self.paint(), captured_at)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_source_yields_exact_count_then_none() {
        let mut source = SyntheticSource::new(SyntheticPattern::Solid(0x20), 3, 30);

        let first = source.next_frame().unwrap().expect("frame 1 of 3");
        let second = source.next_frame().unwrap().expect("frame 2 of 3");
        let third = source.next_frame().unwrap().expect("frame 3 of 3");

        // Frames stand in for webcam frames, so they are the reference size.
        assert_eq!((first.width, first.height), (640, 480));

        // Solid means solid: every pixel carries the grey level behind an opaque
        // alpha, so a source handing out blank frames could not pass.
        for frame in [&first, &second, &third] {
            assert!(
                frame
                    .bgra
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|pixel| pixel == &[0x20, 0x20, 0x20, 0xff])
            );
        }

        // `fps` places frames on the source's own clock instead of pacing the
        // caller, so the gap is exactly one frame period however fast we pull.
        let period = Duration::from_nanos(1_000_000_000 / 30);
        assert_eq!(second.captured_at - first.captured_at, period);
        assert_eq!(third.captured_at - second.captured_at, period);

        assert!(
            source.next_frame().unwrap().is_none(),
            "count 3 must stop after 3 frames"
        );
        assert!(
            source.next_frame().unwrap().is_none(),
            "exhaustion must be permanent"
        );
    }

    #[test]
    fn synthetic_source_is_deterministic() {
        const CELL: u32 = 8;
        let pattern = SyntheticPattern::Checkerboard { cell: CELL };
        let mut first = SyntheticSource::new(pattern, 2, 30);
        let mut second = SyntheticSource::new(pattern, 2, 30);

        let painted: Vec<Frame> = (0..2)
            .map(|_| first.next_frame().unwrap().unwrap())
            .collect();
        let repeated: Vec<Frame> = (0..2)
            .map(|_| second.next_frame().unwrap().unwrap())
            .collect();
        for (index, (a, b)) in painted.iter().zip(&repeated).enumerate() {
            // Compared without `assert_eq!`: a frame is 1.2 MB, and a failing
            // equality check would print both buffers in full.
            assert!(
                a.bgra == b.bgra,
                "frame {index} differs between two sources of one pattern"
            );
        }

        // ...and it is a checkerboard, not a uniform frame: squares of `CELL`
        // pixels alternate on both axes, the one over the origin being black.
        let pixel =
            |frame: &Frame, x: u32, y: u32| frame.bgra[((y * frame.width + x) * 4) as usize];
        let painted = &painted[0];
        assert_eq!(pixel(painted, 0, 0), 0x00);
        assert_eq!(
            pixel(painted, CELL - 1, CELL - 1),
            0x00,
            "uniform inside the first square"
        );
        assert_eq!(
            pixel(painted, CELL, 0),
            0xff,
            "the square to its right is white"
        );
        assert_eq!(
            pixel(painted, 0, CELL),
            0xff,
            "and so is the square below it"
        );
        assert_eq!(
            pixel(painted, CELL, CELL),
            0x00,
            "and black again two squares on"
        );

        // `cell: 0` can subdivide nothing, so it is clamped to one pixel rather
        // than dividing by zero.
        let mut degenerate =
            SyntheticSource::new(SyntheticPattern::Checkerboard { cell: 0 }, 1, 30);
        let one_pixel_cells = degenerate.next_frame().unwrap().unwrap();
        assert_eq!(pixel(&one_pixel_cells, 0, 0), 0x00);
        assert_eq!(pixel(&one_pixel_cells, 1, 0), 0xff);
    }
}
