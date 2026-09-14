use ndarray::array;
use smaul_native::linear::Linear;

#[test]
fn linear_backward_accumulates_bias_gradient() {
    let mut layer = Linear::from_weights(array![[1.0, 2.0]], array![0.0]);
    let _ = layer.backward(&array![[3.0, 4.0]], &array![[5.0]]);
    assert_eq!(layer.grad_weight, array![[15.0, 20.0]]);
    assert_eq!(layer.grad_bias, array![5.0]);
}
