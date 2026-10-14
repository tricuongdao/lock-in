//! Stage 0: the motion gate.
//!
//! The cheapest stage and the only one that runs on every tick: it compares the
//! incoming 64x48 luma frame against the previous one and reports how much the
//! scene moved, so the pipeline can skip all inference while nothing happens.
//! Sitting still and reading then costs one pass over 3072 bytes.
//!
//! Pure computation over an in-memory array — no I/O, no clock, no randomness,
//! no allocation. The only state it keeps is the previous frame and the motion
//! it measured from it, so the same sequence of luma inputs always yields the
//! same motion values.

use crate::frame::LUMA_LEN;

/// Decides whether a frame is worth running inference on.
///
/// Holds only the previous frame: carry one gate across ticks and it compares
/// each frame with the one before it.
pub struct MotionGate {
    /// Motion at or below this many grey levels counts as a static scene.
    threshold: f32,
    /// The luma the next [`MotionGate::update`] compares against, or `None`
    /// until the first frame arrives.
    previous: Option<[u8; LUMA_LEN]>,
    /// Motion reported by the most recent [`MotionGate::update`]. Starts at
    /// `f32::MAX` so a gate that has never seen a frame never claims the scene
    /// is static.
    last_motion: f32,
}

impl MotionGate {
    /// Builds a gate that calls the scene static at or below `threshold` grey
    /// levels of mean absolute difference.
    pub fn new(threshold: f32) -> MotionGate {
        MotionGate {
            threshold,
            previous: None,
            last_motion: f32::MAX,
        }
    }

    /// Compares `luma` against the previous frame and returns the mean absolute
    /// difference over all 3072 bytes, in grey levels (`0.0..=255.0`).
    ///
    /// The first call has no previous frame and returns `f32::MAX`: a cold
    /// start must never be mistaken for a static scene, or the first tick of
    /// every session would skip inference. Allocation-free and branch-light —
    /// this runs on every tick.
    pub fn update(&mut self, luma: &[u8; LUMA_LEN]) -> f32 {
        let motion = match &self.previous {
            Some(previous) => {
                // 3072 byte differences are at most 783_360 in total, so the sum
                // cannot overflow a u32 and dividing it by 3072 keeps the mean
                // inside the 0..=255 range the differences came from.
                let total: u32 = previous
                    .iter()
                    .zip(luma.iter())
                    .map(|(previous, current)| u32::from(previous.abs_diff(*current)))
                    .sum();
                total as f32 / LUMA_LEN as f32
            }
            // No previous frame: a cold start, which must never look static.
            None => f32::MAX,
        };
        self.previous = Some(*luma);
        self.last_motion = motion;
        motion
    }

    /// Whether the most recent [`MotionGate::update`] found the scene static.
    ///
    /// The threshold is inclusive, so frames identical to the last one are
    /// static even at a threshold of zero.
    pub fn is_static(&self) -> bool {
        self.last_motion <= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A luma array whose bytes differ from each other, so a comparison has to
    /// hold at every one of the 3072 positions rather than wherever two solid
    /// frames happen to agree.
    fn patterned() -> [u8; LUMA_LEN] {
        std::array::from_fn(|i| (i % 251) as u8)
    }

    #[test]
    fn identical_frames_report_zero_motion() {
        let first = patterned();
        let mut gate = MotionGate::new(2.0);
        assert_eq!(
            gate.update(&first),
            f32::MAX,
            "the first tick is a cold start"
        );

        // A separate array with the same contents, compared against the copy the
        // gate stored on the first call — the second call is the frame-to-frame
        // comparison, not the cold start asserted above.
        let second = patterned();
        assert_eq!(gate.update(&second), 0.0);
        assert!(gate.is_static());
    }

    #[test]
    fn a_changed_frame_reports_motion_above_threshold() {
        let threshold = 2.0;
        let mut gate = MotionGate::new(threshold);
        assert_eq!(gate.update(&[0u8; LUMA_LEN]), f32::MAX);

        // Every byte jumps the full 255 grey levels, so the mean is exactly 255.
        let motion = gate.update(&[255u8; LUMA_LEN]);
        assert_eq!(motion, 255.0);
        assert!(motion > threshold);
        assert!(!gate.is_static());
    }

    #[test]
    fn first_update_is_large_so_the_first_tick_never_skips() {
        let mut gate = MotionGate::new(2.0);

        // Before any frame has arrived there is nothing to compare, and a gate
        // that answered "static" here would skip inference on the first tick.
        assert!(!gate.is_static());

        assert_eq!(gate.update(&[90u8; LUMA_LEN]), f32::MAX);
        assert!(!gate.is_static());
    }

    #[test]
    fn sub_threshold_change_is_static_but_a_larger_one_is_not() {
        let mut gate = MotionGate::new(2.0);
        assert_eq!(gate.update(&[10u8; LUMA_LEN]), f32::MAX);

        // Every pixel shifts by one grey level: a mean absolute difference of
        // 1.0 is sensor noise, not motion, and must not wake the pipeline.
        assert_eq!(gate.update(&[11u8; LUMA_LEN]), 1.0);
        assert!(gate.is_static());

        // Every pixel shifts by three, which is above the threshold.
        assert_eq!(gate.update(&[14u8; LUMA_LEN]), 3.0);
        assert!(!gate.is_static());
    }

    #[test]
    fn motion_is_the_mean_over_all_pixels_not_just_the_changed_ones() {
        let mut gate = MotionGate::new(2.0);
        assert_eq!(gate.update(&[0u8; LUMA_LEN]), f32::MAX);

        // Half the frame jumps to white: 1536 * 255 / 3072 = 127.5 grey levels.
        // Summing the differences, or averaging over only the pixels that moved,
        // would report 391680 or 255 instead.
        let mut half_bright = [0u8; LUMA_LEN];
        half_bright[..LUMA_LEN / 2].fill(255);
        assert_eq!(gate.update(&half_bright), 127.5);
        assert!(!gate.is_static());
    }
}
