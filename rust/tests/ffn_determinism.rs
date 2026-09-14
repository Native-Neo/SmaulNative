use ndarray::Array2;
use smaul_native::ffn::FeedForward;

#[test]
fn feed_forward_initialization_is_seeded() {
    let a = FeedForward::new(8, 16, 77);
    let b = FeedForward::new(8, 16, 77);
    let x = Array2::<f32>::ones((2, 8));
    assert_eq!(a.forward(&x), b.forward(&x));
}
