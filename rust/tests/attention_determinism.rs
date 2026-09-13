use ndarray::Array2;
use smaul_native::attention::MultiHeadAttention;

#[test]
fn attention_is_deterministic_for_same_seed() {
    let a = MultiHeadAttention::new(8, 2, 99);
    let b = MultiHeadAttention::new(8, 2, 99);
    let x = Array2::from_shape_fn((3, 8), |(r, c)| (r * 8 + c) as f32 * 0.01);
    assert_eq!(a.forward(&x, true), b.forward(&x, true));
}
