use ndarray::{Array1, Array2, Array4};
use crate::group_norm_grad;
use crate::layer_norm_backward;
use crate::rwkv_block_tape::RwkvBlockTape;
use crate::rwkv_cmix_backward;
use crate::wkv_backward;

pub struct RwkvBlockBackward {
    pub grad_input: Array2<f32>,
    pub grad_time_state: Array4<f32>,
    pub grad_time_prev: Array1<f32>,
    pub grad_cmix_prev: Array1<f32>,
    pub grad_ln0_weight: Option<Array1<f32>>,
    pub grad_ln0_bias: Option<Array1<f32>>,
    pub grad_ln1_weight: Array1<f32>,
    pub grad_ln1_bias: Array1<f32>,
    pub grad_ln2_weight: Array1<f32>,
    pub grad_ln2_bias: Array1<f32>,
    pub grad_cmix_x_k: Array1<f32>,
    pub grad_cmix_key: Array2<f32>,
    pub grad_cmix_value: Array2<f32>,
    pub grad_time_w: Array2<f32>,
    pub grad_time_k: Array2<f32>,
    pub grad_time_v: Array2<f32>,
    pub grad_time_kk: Array2<f32>,
    pub grad_time_a: Array2<f32>,
    pub grad_time_r: Array2<f32>,
}

pub fn backward(
    tape: &RwkvBlockTape,
    grad_output: &Array2<f32>,
    time_state_initial: &Array4<f32>,
    time_prev_initial: &Array1<f32>,
    cmix_prev_initial: &Array1<f32>,
    time_k: &Array2<f32>,
    time_v: &Array2<f32>,
    time_kk: &Array2<f32>,
    time_a: &Array2<f32>,
    time_r: &Array2<f32>,
    time_w: &Array2<f32>,
    time_group_norm_input: &Array2<f32>,
    time_group_norm_weight: &Array1<f32>,
    time_group_norm_groups: usize,
    time_group_norm_eps: f32,
    cmix_x_k: &Array1<f32>,
    cmix_key_weight: &Array2<f32>,
    cmix_value_weight: &Array2<f32>,
    ln0_weight: Option<&Array1<f32>>,
    ln0_eps: f32,
    ln1_weight: &Array1<f32>,
    ln1_eps: f32,
    ln2_weight: &Array1<f32>,
    ln2_eps: f32,
) -> RwkvBlockBackward {
    assert_eq!(tape.output.dim(), grad_output.dim());
    let mut grad_residual = grad_output.clone();
    let grad_cmix_output = grad_output.clone();

    let cmix = rwkv_cmix_backward::backward(
        &tape.ln2_output,
        Some(cmix_prev_initial),
        &grad_cmix_output,
        cmix_x_k,
        cmix_key_weight,
        cmix_value_weight,
    );
    let (grad_ln2_input, grad_ln2_weight, grad_ln2_bias) = layer_norm_backward::backward(
        &tape.residual,
        &cmix.grad_input,
        ln2_weight,
        ln2_eps,
    );
    grad_residual += &grad_ln2_input;

    let grad_time_output = grad_residual.clone();
    let wkv = wkv_backward::backward(
        time_state_initial,
        time_w,
        time_k,
        time_v,
        time_kk,
        time_a,
        time_r,
        &grad_time_output,
        time_group_norm_groups,
        time_k.ncols() / time_group_norm_groups,
    );
    let (grad_time_gn, grad_gn_weight, grad_gn_bias) = group_norm_grad::backward(
        time_group_norm_input,
        &wkv.r,
        time_group_norm_weight,
        time_group_norm_groups,
        time_group_norm_eps,
    );
    let (grad_ln1_input, grad_ln1_weight, grad_ln1_bias) = layer_norm_backward::backward(
        &tape.input,
        &grad_time_gn,
        ln1_weight,
        ln1_eps,
    );
    grad_residual += &grad_ln1_input;

    let (grad_ln0_input, grad_ln0_weight, grad_ln0_bias) = match ln0_weight {
        Some(weight) => {
            let (dx, dw, db) = layer_norm_backward::backward(&tape.input, &grad_residual, weight, ln0_eps);
            (dx, Some(dw), Some(db))
        }
        None => (grad_residual.clone(), None, None),
    };

    RwkvBlockBackward {
        grad_input: grad_ln0_input,
        grad_time_state: wkv.state,
        grad_time_prev: Array1::zeros(time_prev_initial.raw_dim()),
        grad_cmix_prev: cmix.grad_prev,
        grad_ln0_weight,
        grad_ln0_bias,
        grad_ln1_weight,
        grad_ln1_bias,
        grad_ln2_weight,
        grad_ln2_bias,
        grad_cmix_x_k: cmix.grad_x_k,
        grad_cmix_key: cmix.grad_key,
        grad_cmix_value: cmix.grad_value,
        grad_time_w: wkv.w,
        grad_time_k: wkv.k,
        grad_time_v: wkv.v,
        grad_time_kk: wkv.kk,
        grad_time_a: wkv.a,
        grad_time_r: wkv.r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_block_tape::RwkvBlockTape;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn backward_produces_finite_core_gradients() {
        let tape = RwkvBlockTape::new(Array2::ones((2, 4)), Array2::ones((2, 4)));
        let state = Array4::zeros((1, 1, 4, 4));
        let vector = Array1::ones(4);
        let matrix = Array2::ones((4, 4));
        let hidden = Array2::ones((2, 4));
        let result = backward(
            &tape, &hidden, &state, &vector, &vector,
            &hidden, &hidden, &hidden, &hidden, &hidden, &hidden,
            &hidden, &vector, 1, 1e-5,
            &vector, &matrix, &matrix,
            None, 1e-5, &vector, 1e-5, &vector, 1e-5,
        );
        assert!(result.grad_input.iter().all(|v| v.is_finite()));
        assert!(result.grad_time_w.iter().all(|v| v.is_finite()));
        assert!(result.grad_cmix_key.iter().all(|v| v.is_finite()));
    }
}
