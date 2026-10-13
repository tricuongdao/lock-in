//! Semantics of `Frame::luma_64x48`: the Rec. 601 integer weights and the
//! nearest-neighbour sampling rule the motion gate's threshold depends on.
//!
//! Expected values are hand-computed from `(77*r + 150*g + 29*b) >> 8` and
//! written as literals, so the tests fail if the formula or the sampling rule
//! changes rather than merely drifting with it.

use lockin_vision::frame::Frame;
use std::time::Instant;

/// Builds a `width x height` BGRA8 frame. `px` returns RGB, which is stored in
/// BGRA order so the test data reads in the order the weights are named.
fn frame_from(width: u32, height: u32, px: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
    let mut bgra = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let [r, g, b] = px(x, y);
            bgra.extend_from_slice(&[b, g, r, 255]);
        }
    }
    Frame::new(width, height, bgra, Instant::now()).unwrap()
}

#[test]
fn luma_uses_rec601_integer_weights() {
    // One solid frame per channel coefficient.
    let red = frame_from(64, 48, |_, _| [255, 0, 0]).luma_64x48()[0];
    let green = frame_from(64, 48, |_, _| [0, 255, 0]).luma_64x48()[0];
    let blue = frame_from(64, 48, |_, _| [0, 0, 255]).luma_64x48()[0];
    let white = frame_from(64, 48, |_, _| [255, 255, 255]).luma_64x48()[0];

    assert_eq!(red, 76); // (77 * 255) >> 8 = 19635 >> 8
    assert_eq!(green, 149); // (150 * 255) >> 8 = 38250 >> 8
    assert_eq!(blue, 28); // (29 * 255) >> 8 = 7395 >> 8
    // The weights sum to 256, so white must land on 255 exactly, not 254.
    assert_eq!(white, 255); // (256 * 255) >> 8 = 65280 >> 8
}

#[test]
fn downsampling_is_nearest_neighbour_not_area_average() {
    // 128x96 -> 64x48 is exactly 2:1. One white 2x2 block at source (1,1)..(2,2)
    // straddles the four destination pixels that meet at destination (1,1).
    // Nearest neighbour samples source (2x, 2y), so destination (1,1) takes
    // source (2,2) whole -> 255. An area average would spread the block over
    // those four destination pixels -> 63 each (255 / 4).
    let single_block = frame_from(128, 96, |x, y| {
        let bright = (1..=2).contains(&x) && (1..=2).contains(&y);
        if bright { [255, 255, 255] } else { [0, 0, 0] }
    });
    let luma = single_block.luma_64x48();
    let sampled = 64 + 1; // destination (1, 1), indexed y * 64 + x

    assert_eq!(luma[sampled], 255, "sampled pixel, not a blend");
    assert_eq!(luma[0], 0);
    assert_eq!(luma[1], 0);
    assert_eq!(luma[64], 0);
    assert_eq!(luma[66], 0);
    assert_eq!(luma[2 * 64 + 1], 0);
    assert_eq!(
        luma.iter().filter(|&&value| value == 255).count(),
        1,
        "a box filter would smear the block across four destination pixels"
    );
}

#[test]
fn luma_is_deterministic_and_total_for_odd_sizes() {
    // Determinism matters for a frame whose samples are not all identical.
    let patterned = frame_from(640, 480, |x, y| {
        [(x % 251) as u8, (y % 241) as u8, ((x * y) % 239) as u8]
    });
    assert_eq!(patterned.luma_64x48(), patterned.luma_64x48());

    // Sizes that are not multiples of 64x48, smaller than it, and empty: each
    // call returning at all is the assertion, and each must fill all 64x48.
    let odd = frame_from(97, 53, |x, y| [x as u8, y as u8, 3]).luma_64x48();
    let tiny = frame_from(3, 2, |x, y| [x as u8, y as u8, 7]).luma_64x48();
    let zero_area = Frame::new(0, 0, vec![], Instant::now()).unwrap();
    let empty = zero_area.luma_64x48();
    assert_eq!(odd.len(), 3072);
    assert_eq!(tiny.len(), 3072);
    assert_eq!(empty, [0u8; 3072]);
}
