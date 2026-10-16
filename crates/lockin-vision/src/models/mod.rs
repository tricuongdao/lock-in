//! The seam between a model wrapper and whatever runs the graph.
//!
//! Every model here — YuNet now, Face Mesh, the eye classifier, YOLOX and the
//! glasses detector later — is a wrapper around a session: it builds input
//! tensors from a [`Frame`](crate::frame::Frame), hands them to
//! [`InferenceSession::run`], and turns the returned tensors into boxes and
//! landmarks. The wrappers hold `Box<dyn InferenceSession>` and nothing else.
//!
//! That is the whole point of this module. **No ONNX Runtime type appears in
//! any signature in this crate.** A wrapper built on this trait is testable
//! with a fake session that returns a tensor the test chose, so preprocessing
//! and decode — the two places detector bugs actually live — are pinned by
//! unit tests that need no model file, no GPU and no camera. Swapping the fake
//! for the real `ort` session in Task 19 changes one `Box<dyn ...>` and no
//! wrapper code at all.
//!
//! Modules are declared here as they land: [`yunet`], [`facemesh`] and
//! [`eyestate`] now, with the rest to follow.

pub mod eyestate;
pub mod facemesh;
pub mod glasses;
pub mod yolox;
pub mod yunet;

use std::sync::{Arc, Mutex};

/// The data one tensor holds.
///
/// Only the two element types the models in this crate need: float tensors for
/// images and predictions, and `i32` for Face Mesh's crop-rectangle inputs
/// (Task 6). `#[non_exhaustive]` keeps a model that needs a third from being a
/// breaking change to every `match` written against it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorData {
    /// 32-bit floats, in the row-major order the tensor's shape describes.
    F32(Vec<f32>),
    /// 32-bit signed integers, same layout.
    I32(Vec<i32>),
}

impl TensorData {
    /// The floats, or `None` if this tensor holds integers.
    ///
    /// A wrapper that reads the wrong tensor type has a bug, not a recoverable
    /// condition; this turns it into one place that decides how to report it.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(values) => Some(values),
            TensorData::I32(_) => None,
        }
    }

    /// The integers, or `None` if this tensor holds floats.
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            TensorData::I32(values) => Some(values),
            TensorData::F32(_) => None,
        }
    }

    /// The floats mutably, or `None` if this tensor holds integers.
    ///
    /// Exists for writing a tensor rather than reading one: a test that feeds a
    /// wrapper a known prediction has to place it at a known anchor, and
    /// rebuilding the whole tensor to change six values would be worse.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            TensorData::F32(values) => Some(values),
            TensorData::I32(_) => None,
        }
    }

    /// The integers mutably, or `None` if this tensor holds floats.
    pub fn as_i32_mut(&mut self) -> Option<&mut [i32]> {
        match self {
            TensorData::I32(values) => Some(values),
            TensorData::F32(_) => None,
        }
    }
}

/// One named input tensor handed to a session.
#[derive(Debug, Clone, PartialEq)]
pub struct InputTensor {
    /// The graph input's name, e.g. `input` for YuNet.
    pub name: String,
    /// The tensor's dimensions, outermost first — `[1, 3, 640, 640]` (NCHW)
    /// for a YuNet frame. Signed, because that is how ONNX Runtime reports
    /// them and a symbolic dimension is a `-1`.
    pub shape: Vec<i64>,
    /// The values, whose length is the product of `shape`.
    pub data: TensorData,
}

/// One named output tensor a session produced.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputTensor {
    /// The graph output's name, as the model file declares it.
    pub name: String,
    /// The tensor's dimensions, outermost first.
    pub shape: Vec<i64>,
    /// The values, whose length is the product of `shape`.
    pub data: TensorData,
}

/// Why a model wrapper could not produce a result.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The session refused to run, or the graph failed part-way through. The
    /// backend's own text is kept rather than flattened away: when this fires
    /// in the field it is usually a missing or mismatched model file, and the
    /// backend's message is what says which.
    #[error("inference session failed: {0}")]
    Session(String),

    /// The session returned no output at all, or returned one that does not
    /// hold event values of the expected type.
    #[error("session returned no usable output '{name}'")]
    MissingOutput {
        /// The output the wrapper needed.
        name: &'static str,
    },

    /// The output is there, but a name in it is not the one this wrapper looks
    /// for.
    #[error("session returned output '{got}', expected '{expected}'")]
    UnexpectedOutput {
        /// The output the wrapper needed.
        expected: &'static str,
        /// The output name the session actually returned.
        got: String,
    },

    /// The output's shape is not the one this model's decode expects, so
    /// reading it would index past the end of the data or misread the layout.
    #[error("model output '{name}' has shape {got:?}, expected {expected:?}")]
    UnexpectedShape {
        /// The output whose shape is wrong.
        name: &'static str,
        /// The shape this model's decode requires.
        expected: Vec<i64>,
        /// The shape the session returned.
        got: Vec<i64>,
    },

    /// The output's values do not fill the shape it declares.
    ///
    /// [`OutputTensor`]'s fields are public, so a session can hand back a shape
    /// and a buffer that disagree — a truncated copy, a graph that emitted a
    /// different batch, or a fake in a test. The decode indexes rows, so this
    /// is checked before any of it: reading past the end of a slice panics, and
    /// a function that returns `Result` must not.
    #[error(
        "model output '{name}' declares {} values but holds {got}",
        .shape.iter().product::<i64>()
    )]
    UnexpectedDataLength {
        /// The output whose data is short.
        name: &'static str,
        /// The shape it declared.
        shape: Vec<i64>,
        /// How many values it actually holds.
        got: usize,
    },

    /// The session returned a different number of outputs than the model has.
    #[error("session returned {got} output(s), expected {expected}")]
    UnexpectedOutputCount {
        /// How many outputs this model's graph declares.
        expected: usize,
        /// How many the session returned.
        got: usize,
    },
}

/// Runs one model graph.
///
/// Implemented by the real ONNX Runtime session (Task 19) and, in tests, by a
/// fake that returns whatever tensors the test hands it. `run` takes `&self`
/// so one session can be shared or called re-entrantly; anything a session
/// needs to mutate is its own business behind interior mutability.
///
/// The trait deliberately names no runtime type — see the module docs.
pub trait InferenceSession {
    /// Runs the graph on `inputs` and returns its outputs.
    ///
    /// Takes a slice of tensors rather than one because Face Mesh's graph takes
    /// five inputs: the image plus four crop-rectangle scalars.
    fn run(&self, inputs: &[InputTensor]) -> Result<Vec<OutputTensor>, ModelError>;

    /// The model filename or id, for logs and for the provider autotune's
    /// per-model bookkeeping.
    fn name(&self) -> &str;
}

/// A session that returns outputs a test chose, so a wrapper can be tested
/// with no model file.
///
/// Deliberately public and reusable: Tasks 6, 10, 11 and 13 each need a fake
/// session, and one shared fake that records what it was asked to run (rather
/// than four private ones that each assert differently) is what makes
/// "was the session handed the right tensor?" a question any wrapper's test
/// can answer.
///
/// Lives outside `#[cfg(test)]` because integration tests and the CLI's
/// harness build against the same crate. It costs one `Vec` and one `Mutex`
/// per fake, and is never constructed by the pipeline itself.
///
/// Hand a wrapper [`FakeSession::handle`] alongside the boxed session: the
/// wrapper owns its `Box<dyn InferenceSession>` and the trait it sees through
/// has no recording methods, so the handle is how a test reads back what the
/// wrapper asked for.
#[derive(Debug)]
pub struct FakeSession {
    name: String,
    outputs: Vec<OutputTensor>,
    /// Every input slice `run` has been called with, in call order.
    ///
    /// Behind an `Arc` so that [`FakeSession::handle`] can keep reading the log
    /// after the session itself has been boxed and handed to a wrapper.
    calls: Arc<Mutex<Vec<Vec<InputTensor>>>>,
}

impl FakeSession {
    /// A session named `name` that returns `outputs`, cloned, from every call.
    pub fn new(name: impl Into<String>, outputs: Vec<OutputTensor>) -> FakeSession {
        FakeSession {
            name: name.into(),
            outputs,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A session that returns one tensor and nothing else.
    pub fn single(name: impl Into<String>, output: OutputTensor) -> FakeSession {
        FakeSession::new(name, vec![output])
    }

    /// A session that returns `count` zeroed `f32` tensors, for the very common
    /// "the model saw nothing here" case.
    ///
    /// Each output is named `output0`, `output1`, …, and shaped `shape`; the
    /// zero fill is sized from the shape, so the two always agree. The name a
    /// wrapper looks for is the wrapper's business — a name this constructor
    /// invented cannot be the one a graph declares.
    pub fn zeros(name: impl Into<String>, count: usize, shape: &[i64]) -> FakeSession {
        let values = shape.iter().product::<i64>().max(0) as usize;
        let outputs = (0..count)
            .map(|index| OutputTensor {
                name: format!("output{index}"),
                shape: shape.to_vec(),
                data: TensorData::F32(vec![0.0; values]),
            })
            .collect();
        FakeSession::new(name, outputs)
    }

    /// A clonable, shared handle to this session's recording.
    ///
    /// The handle outlives the boxed session, so a test can keep asking what
    /// the wrapper sent after the wrapper has taken ownership. For YuNet — whose
    /// graph takes one `[1, 3, 640, 640]` image and returns twelve heads — that
    /// reads:
    ///
    /// ```ignore
    /// let session = FakeSession::zeros("yunet", 12, &[1, 6400, 1]);
    /// let handle = session.handle();
    /// let yunet = YuNet::new(Box::new(session));
    /// yunet.detect(&frame)?;
    /// assert_eq!(handle.last_inputs().unwrap()[0].shape, vec![1, 3, 640, 640]);
    /// ```
    ///
    /// A wrapper that wants a *specific* prediction builds its tensors with
    /// [`FakeSession::new`] instead: `zeros` names its outputs `output0`,
    /// `output1`, … and a graph's own names (`cls_8`, `bbox_16`, …) are what a
    /// wrapper looks for. `yunet.rs`'s test module has the builder to copy.
    pub fn handle(&self) -> FakeSessionHandle {
        FakeSessionHandle {
            calls: Arc::clone(&self.calls),
        }
    }
}

impl InferenceSession for FakeSession {
    fn run(&self, inputs: &[InputTensor]) -> Result<Vec<OutputTensor>, ModelError> {
        self.calls
            .lock()
            .expect("the call log is never poisoned")
            .push(inputs.to_vec());
        Ok(self.outputs.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A shared view of a [`FakeSession`]'s recording, held after the session
/// itself has been moved into whatever is being tested.
#[derive(Debug, Clone)]
pub struct FakeSessionHandle {
    calls: Arc<Mutex<Vec<Vec<InputTensor>>>>,
}

impl FakeSessionHandle {
    /// How many times the session has been run.
    pub fn call_count(&self) -> usize {
        self.calls
            .lock()
            .expect("the call log is never poisoned")
            .len()
    }

    /// The inputs of the most recent run, or `None` if there was none.
    ///
    /// Cloned rather than borrowed: the log sits behind a mutex, and handing
    /// out a reference into it would hold the lock for the caller's lifetime.
    /// Tests call this a handful of times on a tensor of a few hundred
    /// thousand floats, where the clone is free next to building the frame.
    pub fn last_inputs(&self) -> Option<Vec<InputTensor>> {
        self.calls
            .lock()
            .expect("the call log is never poisoned")
            .last()
            .cloned()
    }

    /// The inputs of every run, in call order.
    pub fn inputs(&self) -> Vec<Vec<InputTensor>> {
        self.calls
            .lock()
            .expect("the call log is never poisoned")
            .clone()
    }
}

impl From<OutputTensor> for InputTensor {
    fn from(output: OutputTensor) -> InputTensor {
        InputTensor {
            name: output.name,
            shape: output.shape,
            data: output.data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_session_returns_its_outputs_and_records_its_inputs() {
        let output = OutputTensor {
            name: "output1".to_owned(),
            shape: vec![1, 2, 6],
            data: TensorData::F32(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
        };
        let session = FakeSession::single("yunet", output.clone());
        let handle = session.handle();
        assert_eq!(
            session.name(),
            "yunet",
            "the fake reports the name it was given"
        );
        assert_eq!(handle.call_count(), 0, "nothing has run yet");
        assert!(
            handle.last_inputs().is_none(),
            "so there are no inputs to read"
        );

        let input = InputTensor {
            name: "input".to_owned(),
            shape: vec![1, 3, 640, 640],
            data: TensorData::F32(vec![0.5, 0.25]),
        };
        let returned = session.run(std::slice::from_ref(&input)).unwrap();

        assert_eq!(returned.len(), 1);
        assert_eq!(returned[0].name, "output1");
        assert_eq!(returned[0].shape, vec![1, 2, 6]);
        assert_eq!(
            returned[0].data.as_f32().expect("an f32 tensor"),
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );

        // The recording half is what makes "was the session handed the tensor
        // this preprocessing claims to build?" answerable. Both calls are logged,
        // and the handle sees them after the session has been moved away.
        session.run(std::slice::from_ref(&input)).unwrap();
        assert_eq!(handle.call_count(), 2);
        let calls = handle.inputs();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            handle.last_inputs().expect("two calls"),
            vec![input.clone()],
            "the fake must hand back exactly the tensors it was given"
        );

        // A second call returns as many outputs as the first: the fake clones
        // its script rather than draining it, so call counts are observable.
        assert_eq!(session.run(&[]).unwrap(), returned);
    }

    #[test]
    fn zeros_builds_the_agreed_number_of_blank_outputs() {
        let session = FakeSession::zeros("yunet", 2, &[1, 4, 15]);
        let outputs = session.run(&[]).unwrap();
        assert_eq!(outputs.len(), 2, "one output per requested");
        assert_eq!(outputs[0].name, "output0");
        assert_eq!(outputs[1].name, "output1");
        assert_eq!(outputs[0].shape, vec![1, 4, 15]);
        assert_eq!(
            outputs[0].data.as_f32().unwrap(),
            vec![0.0; 60],
            "the fill is sized from the shape, so the two cannot disagree"
        );
    }

    #[test]
    fn tensor_data_accessors_reject_the_other_element_type() {
        let floats = TensorData::F32(vec![1.5]);
        assert_eq!(floats.as_f32(), Some([1.5].as_slice()));
        assert_eq!(floats.as_i32(), None, "1.5 is not an integer tensor");

        let integers = TensorData::I32(vec![7]);
        assert_eq!(integers.as_i32(), Some([7].as_slice()));
        assert_eq!(integers.as_f32(), None);

        // The mutable accessors a test uses to place a known prediction must
        // agree with the immutable ones about which variant this is.
        let mut integers = TensorData::I32(vec![1, 2]);
        assert_eq!(integers.as_f32_mut(), None);
        integers.as_i32_mut().expect("an i32 tensor")[1] = 9;
        assert_eq!(integers.as_i32(), Some([1, 9].as_slice()));

        let mut floats = TensorData::F32(vec![1.0]);
        assert_eq!(floats.as_i32_mut(), None);
        floats.as_f32_mut().expect("an f32 tensor")[0] = 4.0;
        assert_eq!(floats.as_f32(), Some([4.0].as_slice()));
    }

    #[test]
    fn a_model_error_names_the_output_it_could_not_read() {
        // The messages are what an operator sees when a model file is wrong, so
        // each variant has to say which output and what was expected of it.
        let missing = ModelError::MissingOutput { name: "output1" }.to_string();
        assert!(missing.contains("output1"), "{missing}");

        let unexpected = ModelError::UnexpectedOutput {
            expected: "output1",
            got: "output0".to_owned(),
        }
        .to_string();
        assert!(
            unexpected.contains("output1") && unexpected.contains("output0"),
            "both names must appear: {unexpected}"
        );

        // The shapes here are a real head's: `bbox_32` is `[1, 400, 4]` for the
        // shipping model, so `[1, 395, 4]` is the kind of near-miss a wrong
        // export produces. The assertion is only that both numbers reach the
        // message, which is what an operator sees when a model file is wrong.
        let shape = ModelError::UnexpectedShape {
            name: "bbox_32",
            expected: vec![1, 400, 4],
            got: vec![1, 395, 4],
        }
        .to_string();
        assert!(
            shape.contains("400") && shape.contains("395"),
            "both shapes must appear: {shape}"
        );

        // And the data-length variant names the count it actually found, not
        // just the shape it was promised — those two are the whole diagnosis
        // when a session hands back a truncated tensor.
        let short = ModelError::UnexpectedDataLength {
            name: "cls_8",
            shape: vec![1, 6400, 1],
            got: 10,
        }
        .to_string();
        assert!(
            short.contains("cls_8") && short.contains("6400") && short.contains("10"),
            "the head, the declared size and the actual size must all appear: {short}"
        );
    }
}
