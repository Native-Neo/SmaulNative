use ndarray::Array2;
use smaul_native::ffn::FeedForward;

#[test]
fn feed_forward_zero_input_is_finite() {
    let ffn = FeedForward::new(8, 16, 3);
    let y = ffn.forward(&Array2::zeros((3, 8)));
    assert_eq!(y.dim(), (3, 8));
    assert!(y.iter().all(|v| v.is_finite()));
}
