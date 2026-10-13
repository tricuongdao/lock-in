//! LOCK-IN vision engine — laptop-webcam frames in, focus-state events out.
//!
//! Headless and in-memory: this crate never writes frame data to disk and never
//! sends it over a network.

pub mod frame;
