//! Pure geometry and post-processing: everything done to detector output once
//! the tensors are gone.
//!
//! No Win32, no ONNX, no filesystem, no clock — every function in this module
//! tree is arithmetic over plain data, which is what makes the geometry-heavy
//! half of the engine testable without a camera or a model file.
//!
//! The modules below are the stages of that work, declared here as they land.

pub mod eyes;
pub mod intrinsics;
pub mod nms;
pub mod phone;
pub mod pose;
pub mod primary;
