use std::path::{Path, PathBuf};

use rl_infer::{InferError, OnnxModel};
use serde_json::Value;

const ATOL: f32 = 1e-4;

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn flatten_f32(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("{key} rows must be arrays"))
                .iter()
                .map(|item| {
                    item.as_f64()
                        .unwrap_or_else(|| panic!("{key} must contain numbers"))
                        as f32
                })
        })
        .collect()
}

fn read_usize(value: &Value, key: &str) -> usize {
    value[key]
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .unwrap_or_else(|| panic!("{key} must contain a non-negative integer"))
}

fn read_expected() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(fixture_path("expected.json"))
            .expect("expected fixture must be readable"),
    )
    .expect("expected fixture must be valid JSON")
}

fn fixture_dimensions(value: &Value) -> (usize, usize) {
    (read_usize(value, "obs_dim"), read_usize(value, "n_actions"))
}

fn read_f32(value: &Value, key: &str) -> Vec<f32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
        .iter()
        .map(|item| {
            item.as_f64()
                .unwrap_or_else(|| panic!("{key} must contain numbers")) as f32
        })
        .collect()
}

fn flatten_bool(value: &Value, key: &str) -> Vec<bool> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap_or_else(|| panic!("{key} rows must be arrays"))
                .iter()
                .map(|item| {
                    item.as_bool()
                        .unwrap_or_else(|| panic!("{key} must contain booleans"))
                })
        })
        .collect()
}

#[test]
fn infer_matches_torch_fixture() {
    let expected = read_expected();
    let (obs_dim, n_actions) = fixture_dimensions(&expected);
    let obs = flatten_f32(&expected, "obs");
    let mask = flatten_bool(&expected, "mask");
    let expected_logits = flatten_f32(&expected, "logits");
    let expected_values = read_f32(&expected, "values");

    let mut model = OnnxModel::load(&fixture_path("tiny.onnx"), obs_dim, n_actions)
        .expect("fixture model must load");
    let output = model
        .infer(&obs, &mask)
        .expect("fixture inference must succeed");

    assert_eq!(output.logits.len(), expected_logits.len());
    assert_eq!(output.values.len(), expected_values.len());
    for (actual, expected) in output.logits.iter().zip(expected_logits) {
        assert!(
            (actual - expected).abs() <= ATOL,
            "logit mismatch: {actual} != {expected}"
        );
    }
    for (actual, expected) in output.values.iter().zip(expected_values) {
        assert!(
            (actual - expected).abs() <= ATOL,
            "value mismatch: {actual} != {expected}"
        );
    }
}

#[test]
fn load_rejects_wrong_observation_dimension() {
    let (obs_dim, n_actions) = fixture_dimensions(&read_expected());
    let error = OnnxModel::load(&fixture_path("tiny.onnx"), obs_dim - 1, n_actions)
        .expect_err("wrong observation dimension must fail");
    assert!(error.to_string().contains("obs"));
    assert!(error.to_string().contains("dimension"));
}

#[test]
fn infer_rejects_mismatched_lengths() {
    let (obs_dim, n_actions) = fixture_dimensions(&read_expected());
    let mut model = OnnxModel::load(&fixture_path("tiny.onnx"), obs_dim, n_actions)
        .expect("fixture model must load");

    assert!(
        model
            .infer(&vec![0.0_f32; obs_dim - 1], &vec![true; n_actions])
            .is_err()
    );
    assert!(
        model
            .infer(&vec![0.0_f32; obs_dim], &vec![true; n_actions - 1])
            .is_err()
    );
    assert!(
        model
            .infer(&vec![0.0_f32; obs_dim], &vec![true; n_actions * 2])
            .is_err()
    );
}

#[test]
fn infer_rejects_mask_row_without_legal_action() {
    let (obs_dim, n_actions) = fixture_dimensions(&read_expected());
    let mut model = OnnxModel::load(&fixture_path("tiny.onnx"), obs_dim, n_actions)
        .expect("fixture model must load");
    let obs = vec![0.0_f32; obs_dim * 2];
    let mask = vec![true; n_actions]
        .into_iter()
        .chain(std::iter::repeat_n(false, n_actions))
        .collect::<Vec<_>>();

    let error = match model.infer(&obs, &mask) {
        Ok(_) => panic!("an all-false mask row must fail"),
        Err(error) => error,
    };
    assert!(matches!(error, InferError::InvalidInput(message) if message.contains("row 1")));

    let output = model
        .infer(&vec![0.0_f32; obs_dim], &vec![true; n_actions])
        .expect("a valid mask must still work");
    assert_eq!(output.values.len(), 1);
}

#[test]
fn infer_returns_empty_output_for_empty_batch() {
    let (obs_dim, n_actions) = fixture_dimensions(&read_expected());
    let mut model = OnnxModel::load(&fixture_path("tiny.onnx"), obs_dim, n_actions)
        .expect("fixture model must load");
    let output = model.infer(&[], &[]).expect("empty batch must succeed");

    assert!(output.logits.is_empty());
    assert!(output.values.is_empty());
}
