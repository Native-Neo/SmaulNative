use smaul_native::init::uniform;

#[test]
fn uniform_initialization_respects_bounds() {
    let x = uniform(16, 8, -0.25, 0.5, 31);
    assert!(x.iter().all(|v| *v >= -0.25 && *v <= 0.5));
}
