use ndarray::{Array2, Array4};
use smaul_native::wkv::run;

#[test]
fn wkv_zero_inputs_keep_state_and_output_zero() {
    let state = Array4::zeros((1, 2, 4, 4));
    let inputs: Vec<Array2<f32>> = (0..6).map(|_| Array2::zeros((3, 8))).collect();
    let (_, output) = run(state, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], 2, 4);
    assert!(output.iter().all(|v| *v == 0.0));
}
