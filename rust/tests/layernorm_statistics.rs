use ndarray::array;
use smaul_native::layer_norm::LayerNorm;

#[test]
fn layer_norm_centers_nonconstant_rows() {
    let norm = LayerNorm::new(4, 1e-5);
    let y = norm.forward(&array![[1.0, 2.0, 3.0, 4.0]]);
    let mean = y.sum() / 4.0;
    assert!(mean.abs() < 1e-5);
}
