use ndarray::Array2;
use smaul_native::attention::MultiHeadAttention;

#[test]
fn attention_preserves_sequence_and_model_dimensions() {
    let attention = MultiHeadAttention::new(12, 3, 5);
    let y = attention.forward(&Array2::ones((5, 12)), true);
    assert_eq!(y.dim(), (5, 12));
}
