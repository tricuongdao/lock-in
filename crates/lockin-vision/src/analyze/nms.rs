//! Non-maximum suppression: turning a detector's cluster of overlapping boxes
//! into one box per object.
//!
//! Both detectors fire several times around a single object, so a raw decode
//! ends with near-duplicate boxes that would otherwise each count as a separate
//! person or phone. This stage sorts them strongest first and drops any box
//! that overlaps an already-kept box of its own class.
//!
//! Suppression here is **class-aware**, and that is the whole reason
//! [`Detection`] carries a class at all. A phone in someone's hand sits inside
//! that person's box; class-agnostic suppression would delete one of the two
//! and silently destroy the phone signal the rest of the app is built on.

/// An axis-aligned box in frame coordinates: pixels as `f32`, origin top-left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width, extending right from `x`.
    pub w: f32,
    /// Height, extending down from `y`.
    pub h: f32,
}

impl Rect {
    /// Intersection over union with `other`: `1.0` for identical boxes, `0.0`
    /// for boxes that share no area, and the overlapping fraction of the
    /// combined area in between.
    pub fn iou(&self, other: &Rect) -> f32 {
        // Gap between the boxes along each axis, then clamped: two boxes that
        // do not reach each other overlap in nothing, not in a negative area.
        let overlap_w = (self.x + self.w).min(other.x + other.w) - self.x.max(other.x);
        let overlap_h = (self.y + self.h).min(other.y + other.h) - self.y.max(other.y);
        let intersection = overlap_w.max(0.0) * overlap_h.max(0.0);
        let union = self.w * self.h + other.w * other.h - intersection;

        if union <= 0.0 {
            // Two boxes with no area at all have no union to divide by. They
            // share no pixels, which is what 0.0 says.
            return 0.0;
        }
        intersection / union
    }

    /// The box's centre point, `(x + w / 2.0, y + h / 2.0)`.
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }
}

/// One detected object: where it is, how sure the detector is, and which class
/// it was detected as.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// The box in frame coordinates.
    pub rect: Rect,
    /// Detector confidence; higher is stronger.
    pub score: f32,
    /// The class this box was detected as, in the detector's own numbering —
    /// COCO ids for YOLOX, where `0` is a person and `67` a cell phone.
    /// Suppression never crosses classes.
    pub class_id: u32,
}

/// Drops detections scoring below `score_threshold` and suppresses those that
/// overlap a stronger detection of the same class, in place.
///
/// A detection scoring exactly `score_threshold` survives — the brief drops
/// what falls *below* it — and a box is suppressed when its IoU with an
/// already-kept box is strictly *above* `iou_threshold`, so boxes meeting at
/// exactly the threshold are both kept.
///
/// Survivors are left in the vector strongest first, so a caller can take the
/// best detection without sorting again. Only boxes of the *same* class
/// suppress each other — see the module docs for why.
pub fn nms(dets: &mut Vec<Detection>, iou_threshold: f32, score_threshold: f32) {
    // Weak detections are noise whether or not anything overlaps them, so drop
    // them first and everything after this sees only candidates worth keeping.
    dets.retain(|det| det.score >= score_threshold);

    // Strongest first, by a total order so the result never depends on how the
    // scores compare. The sort is stable, so detections the detector scored
    // equally keep the order it produced them in.
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Sweep strongest first: a detection survives unless a box already kept for
    // its own class covers it. Tens of boxes, so the quadratic scan is cheaper
    // than anything cleverer would be to read.
    let mut kept: Vec<Detection> = Vec::with_capacity(dets.len());
    for candidate in dets.iter() {
        let suppressed = kept.iter().any(|survivor| {
            survivor.class_id == candidate.class_id
                && survivor.rect.iou(&candidate.rect) > iou_threshold
        });
        if !suppressed {
            kept.push(*candidate);
        }
    }

    *dets = kept;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two classes this crate actually cares about, in COCO numbering: a
    /// person and the phone they are holding.
    const PERSON: u32 = 0;
    const CELL_PHONE: u32 = 67;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect { x, y, w, h }
    }

    fn detection(rect: Rect, score: f32, class_id: u32) -> Detection {
        Detection {
            rect,
            score,
            class_id,
        }
    }

    /// The surviving scores, in the order `nms` left them.
    fn scores(dets: &[Detection]) -> Vec<f32> {
        dets.iter().map(|det| det.score).collect()
    }

    #[test]
    fn identical_rects_have_iou_one() {
        let a = rect(10.0, 20.0, 40.0, 60.0);
        let b = rect(10.0, 20.0, 40.0, 60.0);
        assert_eq!(a.iou(&b), 1.0, "a box covers itself completely");
        assert_eq!(b.iou(&a), 1.0, "and overlap is symmetric");

        // Identity is the easy case, in which the corners barely matter. A pair
        // offset from each other in *both* axes pins the arithmetic that
        // decides them, to a hand-computed value: the boxes share x in 0..=4
        // and y in 0..=2, so the intersection is 4 * 2 = 8, the union is
        // 16 + 32 - 8 = 40, and the IoU is 8 / 40.
        let small = rect(0.0, 0.0, 4.0, 4.0);
        let offset = rect(-4.0, -2.0, 8.0, 4.0);
        assert_eq!(small.iou(&offset), 0.2);
        assert_eq!(offset.iou(&small), 0.2, "and overlap is symmetric");
    }

    #[test]
    fn disjoint_rects_have_iou_zero() {
        // Side by side and touching: the shared edge is a line, and a line has
        // no area.
        let left = rect(0.0, 0.0, 10.0, 10.0);
        let right = rect(10.0, 0.0, 10.0, 10.0);
        assert_eq!(left.iou(&right), 0.0);
        assert_eq!(right.iou(&left), 0.0);

        // Separated in both axes. A small box beyond the corner is deliberate:
        // an overlap computed without clamping to zero would multiply the two
        // negative gaps into a positive area and claim these overlap.
        let beyond_the_corner = rect(20.0, 20.0, 5.0, 5.0);
        assert_eq!(left.iou(&beyond_the_corner), 0.0);

        // The same box reflected past the top-left corner.
        let behind_the_origin = rect(-20.0, -20.0, 5.0, 5.0);
        assert_eq!(left.iou(&behind_the_origin), 0.0);
    }

    #[test]
    fn nms_keeps_the_higher_scoring_of_two_overlapping_boxes() {
        // Two boxes on the same face, the weaker one listed first so that only
        // a real sort can put the survivor at the front.
        let face = rect(100.0, 50.0, 40.0, 40.0);
        let mut dets = vec![detection(face, 0.72, PERSON), detection(face, 0.91, PERSON)];
        nms(&mut dets, 0.5, 0.1);
        assert_eq!(scores(&dets), vec![0.91], "the weaker duplicate is dropped");

        // Overlapping is not on its own enough: a pair whose IoU is 0.5 at a
        // threshold of 0.5 is two objects the detector placed close together,
        // not one object detected twice.
        let mut dets = vec![
            detection(rect(0.0, 0.0, 4.0, 4.0), 0.9, PERSON),
            detection(rect(-4.0, 0.0, 8.0, 4.0), 0.8, PERSON),
        ];
        nms(&mut dets, 0.5, 0.1);
        assert_eq!(scores(&dets), vec![0.9, 0.8], "IoU 0.5 is at the threshold");
    }

    #[test]
    fn nms_keeps_boxes_belonging_to_different_classes() {
        // A phone seen in someone's hand sits inside that person's box, so the
        // two rects are identical and the IoU is 1.0 — the maximum there is. If
        // suppression ignored `class_id` one of the two would be deleted here,
        // and the phone signal the app depends on would be gone.
        let shared = rect(10.0, 10.0, 100.0, 100.0);
        let mut dets = vec![
            detection(shared, 0.90, PERSON),
            detection(shared, 0.80, CELL_PHONE),
        ];
        nms(&mut dets, 0.3, 0.1);

        assert_eq!(scores(&dets), vec![0.90, 0.80], "neither box suppresses");
        assert_eq!(
            dets.iter().map(|det| det.class_id).collect::<Vec<_>>(),
            vec![PERSON, CELL_PHONE],
            "and both classes survive"
        );
    }

    #[test]
    fn nms_drops_detections_below_score_threshold() {
        let mut dets = vec![
            detection(rect(0.0, 0.0, 10.0, 10.0), 0.29, PERSON),
            // Exactly on the threshold: the brief drops detections *below* it,
            // so this one is kept.
            detection(rect(50.0, 0.0, 10.0, 10.0), 0.30, PERSON),
            detection(rect(100.0, 0.0, 10.0, 10.0), 0.95, PERSON),
        ];
        nms(&mut dets, 0.5, 0.30);

        assert_eq!(
            scores(&dets),
            vec![0.95, 0.30],
            "the weak box is gone, the two strong ones are left"
        );
    }
}
