use smaul_native::group_norm::GroupNorm;

#[test]
fn group_norm_parameter_count_matches_affine_vectors() {
    let norm = GroupNorm::new(12, 3, 1e-5);
    assert_eq!(norm.parameter_count(), 24);
}
