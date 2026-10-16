//! LOCK-IN vision engine — laptop-webcam frames in, focus-state events out.
//!
//! Headless and in-memory: this crate never writes frame data to disk and never
//! sends it over a network.

pub mod analyze;
pub mod capture;
pub mod frame;
pub mod fuse;
pub mod models;
pub mod motion;
