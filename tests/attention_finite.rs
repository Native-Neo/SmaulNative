use ndarray::Array2;
use smaul_native::attention::MultiHeadAttention;

#[test]
fn causal_attention_never_emits_nan() {
    let attention = MultiHeadAttention::new(16, 4, 17);
    let x = Array2::from_shape_fn((7, 16), |(r, c)| (r as f32 - c as f32) * 0.2);
    let y = attention.forward(&x, true);
    assert!(y.iter().all(|v| v.is_finite()));
}
