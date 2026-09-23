use std::fmt;
use std::path::Path;

use ort::session::Session;
use ort::value::{Outlet, TensorElementType, TensorRef, ValueType};

// TODO: rl_core の Policy trait が定義されたら、バッチ推論をその trait に接続する。

/// ONNX モデルの読み込みまたは推論に失敗したときのエラー。
#[derive(Debug)]
pub enum InferError {
    /// ONNX Runtime が返したエラー。
    Ort(ort::Error),
    /// モデルがこの crate の入出力契約に適合しない。
    InvalidModel(String),
    /// 推論の入力が契約に適合しない。
    InvalidInput(String),
}

impl fmt::Display for InferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ort(error) => write!(formatter, "ONNX Runtime error: {error}"),
            Self::InvalidModel(message) => write!(formatter, "invalid ONNX model: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid inference input: {message}"),
        }
    }
}

impl std::error::Error for InferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ort(error) => Some(error),
            Self::InvalidModel(_) | Self::InvalidInput(_) => None,
        }
    }
}

impl From<ort::Error> for InferError {
    fn from(error: ort::Error) -> Self {
        Self::Ort(error)
    }
}

/// ONNX policy-value model のバッチ推論器。
pub struct OnnxModel {
    session: Session,
    obs_dim: usize,
    n_actions: usize,
}

impl fmt::Debug for OnnxModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OnnxModel")
            .field("obs_dim", &self.obs_dim)
            .field("n_actions", &self.n_actions)
            .finish_non_exhaustive()
    }
}

impl OnnxModel {
    /// モデルを読み込み、固定された ONNX 入出力契約を検査する。
    pub fn load(path: &Path, obs_dim: usize, n_actions: usize) -> Result<Self, InferError> {
        if obs_dim == 0 {
            return Err(Self::invalid_model("obs_dim must be positive"));
        }
        if n_actions == 0 {
            return Err(Self::invalid_model("n_actions must be positive"));
        }

        let mut builder = Session::builder()?;
        let session = builder.commit_from_file(path)?;

        let input_names = session
            .inputs()
            .iter()
            .map(Outlet::name)
            .collect::<Vec<_>>();
        Self::validate_outlet_count(&input_names, 2, "input")?;
        let output_names = session
            .outputs()
            .iter()
            .map(Outlet::name)
            .collect::<Vec<_>>();
        Self::validate_outlet_count(&output_names, 2, "output")?;

        let obs = Self::find_outlet(session.inputs(), "obs", "input")?;
        Self::validate_contract(obs, "input `obs`", TensorElementType::Float32, &[obs_dim])?;

        let mask = Self::find_outlet(session.inputs(), "mask", "input")?;
        Self::validate_contract(mask, "input `mask`", TensorElementType::Bool, &[n_actions])?;

        let logits = Self::find_outlet(session.outputs(), "logits", "output")?;
        Self::validate_contract(
            logits,
            "output `logits`",
            TensorElementType::Float32,
            &[n_actions],
        )?;

        let value = Self::find_outlet(session.outputs(), "value", "output")?;
        Self::validate_contract(value, "output `value`", TensorElementType::Float32, &[])?;

        Ok(Self {
            session,
            obs_dim,
            n_actions,
        })
    }

    /// 観測と合法手マスクをバッチ推論する。
    ///
    /// `obs.len()` は `batch * obs_dim`、`mask.len()` は `batch * n_actions` である必要がある。
    pub fn infer(&mut self, obs: &[f32], mask: &[bool]) -> Result<Output, InferError> {
        if !obs.len().is_multiple_of(self.obs_dim) {
            return Err(Self::invalid_input(format!(
                "obs length {} is not a multiple of obs_dim {}",
                obs.len(),
                self.obs_dim
            )));
        }
        if !mask.len().is_multiple_of(self.n_actions) {
            return Err(Self::invalid_input(format!(
                "mask length {} is not a multiple of n_actions {}",
                mask.len(),
                self.n_actions
            )));
        }

        let obs_batch = obs.len() / self.obs_dim;
        let mask_batch = mask.len() / self.n_actions;
        if obs_batch != mask_batch {
            return Err(Self::invalid_input(format!(
                "obs batch {obs_batch} does not match mask batch {mask_batch}"
            )));
        }
        if obs_batch == 0 {
            return Ok(Output {
                logits: Vec::new(),
                values: Vec::new(),
            });
        }

        for (row_index, row) in mask.chunks_exact(self.n_actions).enumerate() {
            if !row.iter().copied().any(|legal| legal) {
                return Err(Self::invalid_input(format!(
                    "mask row {row_index} has no legal action"
                )));
            }
        }

        let obs_tensor = TensorRef::from_array_view(([obs_batch, self.obs_dim], obs))?;
        let mask_tensor = TensorRef::from_array_view(([obs_batch, self.n_actions], mask))?;
        let outputs = self.session.run(ort::inputs! {
            "obs" => obs_tensor,
            "mask" => mask_tensor,
        })?;

        let logits = Self::extract_output(&outputs, "logits", &[obs_batch, self.n_actions])?;
        let values = Self::extract_output(&outputs, "value", &[obs_batch])?;

        Ok(Output { logits, values })
    }

    fn find_outlet<'a>(
        outlets: &'a [Outlet],
        name: &str,
        kind: &str,
    ) -> Result<&'a Outlet, InferError> {
        outlets
            .iter()
            .find(|outlet| outlet.name() == name)
            .ok_or_else(|| {
                let available = outlets
                    .iter()
                    .map(Outlet::name)
                    .collect::<Vec<_>>()
                    .join(", ");
                Self::invalid_model(format!(
                    "missing {kind} named `{name}`; available names: [{available}]"
                ))
            })
    }

    fn validate_outlet_count(
        names: &[&str],
        expected: usize,
        kind: &str,
    ) -> Result<(), InferError> {
        if names.len() != expected {
            return Err(Self::invalid_model(format!(
                "expected {expected} {kind}s, got {}; actual names: [{}]",
                names.len(),
                names.join(", ")
            )));
        }
        Ok(())
    }

    fn validate_contract(
        outlet: &Outlet,
        label: &str,
        expected_type: TensorElementType,
        expected_tail: &[usize],
    ) -> Result<(), InferError> {
        let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
            return Err(Self::invalid_model(format!(
                "{label} must be a tensor, got {}",
                outlet.dtype()
            )));
        };

        if *ty != expected_type {
            return Err(Self::invalid_model(format!(
                "{label} must have element type {expected_type}, got {ty}"
            )));
        }

        let expected_rank = expected_tail.len() + 1;
        if shape.len() != expected_rank {
            return Err(Self::invalid_model(format!(
                "{label} must have rank {expected_rank} with a dynamic batch dimension, got shape {shape}"
            )));
        }
        if shape[0] >= 0 {
            return Err(Self::invalid_model(format!(
                "{label} must have a dynamic batch dimension, got shape {shape}"
            )));
        }
        for (index, expected) in expected_tail.iter().enumerate() {
            let expected = i64::try_from(*expected).map_err(|_| {
                Self::invalid_model(format!(
                    "{label} dimension {} is too large for ONNX",
                    index + 1
                ))
            })?;
            let actual = shape[index + 1];
            if actual != expected {
                return Err(Self::invalid_model(format!(
                    "{label} dimension {} must be {}, got shape {shape}",
                    index + 1,
                    expected,
                )));
            }
        }
        Ok(())
    }

    fn extract_output(
        outputs: &ort::session::SessionOutputs<'_>,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, InferError> {
        let output = outputs
            .get(name)
            .ok_or_else(|| Self::invalid_model(format!("inference output `{name}` is missing")))?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        if shape.len() != expected_shape.len() {
            return Err(Self::invalid_model(format!(
                "inference output `{name}` has shape {shape}, expected rank {}",
                expected_shape.len()
            )));
        }
        let mut expected_len = 1usize;
        for (index, expected) in expected_shape.iter().enumerate() {
            let expected_dimension = i64::try_from(*expected).map_err(|_| {
                Self::invalid_model(format!(
                    "inference output `{name}` dimension {index} is too large"
                ))
            })?;
            if shape[index] != expected_dimension {
                return Err(Self::invalid_model(format!(
                    "inference output `{name}` has shape {shape}, expected {expected_shape:?}"
                )));
            }
            expected_len = expected_len.checked_mul(*expected).ok_or_else(|| {
                Self::invalid_model(format!(
                    "inference output `{name}` contains too many elements"
                ))
            })?;
        }
        if data.len() != expected_len {
            return Err(Self::invalid_model(format!(
                "inference output `{name}` contains {} elements, expected {expected_len}",
                data.len()
            )));
        }
        Ok(data.to_vec())
    }

    fn invalid_model(message: impl Into<String>) -> InferError {
        InferError::InvalidModel(message.into())
    }

    fn invalid_input(message: impl Into<String>) -> InferError {
        InferError::InvalidInput(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{InferError, OnnxModel};

    #[test]
    fn validate_outlet_count_accepts_expected_and_rejects_mismatch() {
        assert!(OnnxModel::validate_outlet_count(&["obs", "mask"], 2, "input").is_ok());

        let error = OnnxModel::validate_outlet_count(&["obs", "mask", "temperature"], 2, "input")
            .expect_err("extra input must fail validation");
        assert!(
            matches!(error, InferError::InvalidModel(message) if message.contains("temperature"))
        );
    }
}

/// バッチ推論の出力。
pub struct Output {
    /// 行動ごとの logits。要素数は `batch * n_actions` である。
    pub logits: Vec<f32>,
    /// 状態価値。要素数は `batch` である。
    pub values: Vec<f32>,
}
