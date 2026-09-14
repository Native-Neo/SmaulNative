use ndarray::{Array2, Array4};
use smaul_native::wkv::run;

#[test]
fn wkv_returns_sequence_by_channel_output() {
    let state = Array4::zeros((1, 2, 4, 4));
    let inputs: Vec<Array2<f32>> = (0..6).map(|_| Array2::ones((5, 8))).collect();
    let (_, output) = run(state, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], 2, 4);
    assert_eq!(output.dim(), (5, 8));
}
