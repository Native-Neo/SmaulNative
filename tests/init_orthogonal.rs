use smaul_native::init::orthogonal;

#[test]
fn orthogonal_initialization_has_target_column_norms() {
    let x = orthogonal(10, 3, 0.2, 41);
    for c in 0..3 {
        let norm = (0..10).map(|r| x[[r, c]] * x[[r, c]]).sum::<f32>().sqrt();
        assert!((norm - 0.2).abs() < 1e-5);
    }
}
