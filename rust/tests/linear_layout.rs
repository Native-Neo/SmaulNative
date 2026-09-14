use ndarray::array;
use smaul_native::linear::Linear;

#[test]
fn linear_matrix_layout_is_stable() {
    let layer = Linear::from_weight(array![[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(layer.forward(&array![[2.0, 3.0]]), array![[8.0, 18.0]]);
}
