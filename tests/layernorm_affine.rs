use ndarray::array;
use smaul_native::layer_norm::LayerNorm;

#[test]
fn layer_norm_affine_parameters_are_applied() {
    let mut norm = LayerNorm::new(2, 1e-5);
    norm.weight.fill(2.0);
    norm.bias.fill(3.0);
    let y = norm.forward(&array![[1.0, 3.0]]);
    assert!((y[[0, 0]] - 1.0).abs() < 1e-4);
    assert!((y[[0, 1]] - 5.0).abs() < 1e-4);
}
