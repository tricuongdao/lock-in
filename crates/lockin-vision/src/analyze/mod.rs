//! Pure geometry and post-processing: everything done to detector output once
//! the tensors are gone.
//!
//! No Win32, no ONNX, no filesystem, no clock — every function in this module
//! tree is arithmetic over plain data, which is what makes the geometry-heavy
//! half of the engine testable without a camera or a model file.
//!
//! Modules are declared here as they land: [`nms`] now, with intrinsics, head
//! pose, eye geometry, phone association and primary-face selection to follow.

pub mod nms;
