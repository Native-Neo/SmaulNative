use ndarray::array;
use smaul_native::group_norm::GroupNorm;

#[test]
fn group_norm_affine_shift_is_visible() {
    let mut norm = GroupNorm::new(4, 2, 1e-5);
    norm.bias.fill(2.0);
    let y = norm.forward(&array![[1.0, 3.0, 5.0, 7.0]]);
    assert!(y.iter().all(|v| v.is_finite()));
    assert!((y[[0, 0]] + y[[0, 1]] - 4.0).abs() < 1e-4);
}
