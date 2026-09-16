use smaul_native::linear::Linear;

#[test]
fn linear_parameter_count_tracks_bias() {
    assert_eq!(Linear::new(5, 7).parameter_count(), 35 + 7);
    assert_eq!(Linear::new_no_bias(5, 7).parameter_count(), 35);
}
