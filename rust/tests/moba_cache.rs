use ndarray::{Array2, Array4};
use smaul_native::moba_attention::MobaAttention;

fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

#[test]
fn cache_selects_top_history_chunks_and_keeps_current_chunk() {
    let mut att = MobaAttention::new(1, 1, 2, 1);
    att.receptance.weight[[0, 0]] = 1.0;
    att.key.weight[[0, 0]] = 1.0;
    att.value.weight[[0, 0]] = 10.0;
    att.output.weight[[0, 0]] = 1.0;

    let cache_k = Array4::from_shape_vec((1, 4, 1, 1), vec![1.0, 1.0, -1.0, -1.0]).unwrap();
    let cache_v = Array4::from_shape_vec((1, 4, 1, 1), vec![10.0, 10.0, -10.0, -10.0]).unwrap();
    let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    let (actual, _, _) = att.cache_forward(&x, &cache_k, &cache_v);

    let keys = [1.0, 1.0, 1.0];
    let values = [10.0, 10.0, 10.0];
    let weights = softmax(&keys);
    let expected: f32 = weights.iter().zip(values).map(|(w, v)| w * v).sum();

    assert!((actual[[0, 0]] - expected).abs() < 1e-6, "expected {expected}, got {}", actual[[0, 0]]);
}

#[test]
fn cache_uses_all_history_before_moba_selection_kicks_in() {
    let mut att = MobaAttention::new(1, 1, 2, 4);
    att.receptance.weight[[0, 0]] = 1.0;
    att.key.weight[[0, 0]] = 1.0;
    att.value.weight[[0, 0]] = 10.0;
    att.output.weight[[0, 0]] = 1.0;

    let cache_k = Array4::from_shape_vec((1, 2, 1, 1), vec![-1.0, 1.0]).unwrap();
    let cache_v = Array4::from_shape_vec((1, 2, 1, 1), vec![-10.0, 10.0]).unwrap();
    let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

    let (actual, _, _) = att.cache_forward(&x, &cache_k, &cache_v);

    let keys = [-1.0, 1.0, 1.0];
    let values = [-10.0, 10.0, 10.0];
    let weights = softmax(&keys);
    let expected: f32 = weights.iter().zip(values).map(|(w, v)| w * v).sum();

    assert!((actual[[0, 0]] - expected).abs() < 1e-6, "expected {expected}, got {}", actual[[0, 0]]);
}
