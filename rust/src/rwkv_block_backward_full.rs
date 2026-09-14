use ndarray::{Array1, Array2, Array4};
use crate::layer_norm_backward;
use crate::rwkv_block::{RwkvBlock, RwkvBlockState};
use crate::rwkv_block_full_tape::RwkvBlockFullTape;
use crate::rwkv_cmix_backward;
use crate::rwkv_time_mix_backward_full;

pub struct RwkvBlockBackward {
    pub grad_input: Array2<f32>,
    pub grad_time_prev: Array1<f32>,
    pub grad_cmix_prev: Array1<f32>,
    pub grad_time_state: Array4<f32>,
    pub grad_v_first: Array2<f32>,
    pub grad_ln0_weight: Option<Array1<f32>>,
    pub grad_ln0_bias: Option<Array1<f32>>,
    pub grad_ln1_weight: Array1<f32>,
    pub grad_ln1_bias: Array1<f32>,
    pub grad_ln2_weight: Array1<f32>,
    pub grad_ln2_bias: Array1<f32>,
    pub time: rwkv_time_mix_backward_full::RwkvTimeMixBackward,
    pub cmix: rwkv_cmix_backward::CmixBackward,
}

pub fn backward(
    block: &RwkvBlock,
    tape: &RwkvBlockFullTape,
    grad_output: &Array2<f32>,
    grad_next_state: Option<&RwkvBlockState>,
) -> RwkvBlockBackward {
    assert_eq!(grad_output.dim(), tape.output.dim());
    let channels = block.channels;

    let grad_cmix_output = grad_output.clone();
    let grad_residual = grad_output.clone();

    let cmix = rwkv_cmix_backward::backward(
        &tape.cmix_input,
        &tape.cmix_prev,
        &block.cmix.x_k,
        &block.cmix.key,
        &tape.cmix_pre,
        &tape.cmix_hidden,
        &grad_cmix_output,
    );

    let (grad_ln2_input, grad_ln2_weight, grad_ln2_bias) = layer_norm_backward::backward(
        &tape.ln2_input,
        &cmix.grad_x,
        &block.ln2.weight,
        block.ln2.eps,
    );

    let mut grad_residual_total = grad_residual + &grad_ln2_input;

    let time = rwkv_time_mix_backward_full::backward(
        &block.time_mix,
        &tape.time,
        &grad_residual_total,
    );

    let (grad_ln1_input, grad_ln1_weight, grad_ln1_bias) = layer_norm_backward::backward(
        &tape.ln1_input,
        &time.grad_input,
        &block.ln1.weight,
        block.ln1.eps,
    );
    grad_residual_total += &grad_ln1_input;

    let mut grad_ln0_weight = None;
    let mut grad_ln0_bias = None;
    let grad_input = if let (Some(ln0), Some(ln0_input), Some(ln0_output)) = (
        block.ln0.as_ref(),
        tape.ln0_input.as_ref(),
        tape.ln0_output.as_ref(),
    ) {
        let (dx, dw, db) = layer_norm_backward::backward(
            ln0_input,
            &grad_residual_total,
            &ln0.weight,
            ln0.eps,
        );
        grad_ln0_weight = Some(dw);
        grad_ln0_bias = Some(db);
        dx
    } else {
        grad_residual_total
    };

    let mut grad_time_state = time.grad_state.clone();
    let mut grad_time_prev = time.grad_prev.clone();
    let mut grad_cmix_prev = cmix.grad_prev.clone();
    let mut grad_v_first = time.grad_v_first.clone();

    if let Some(next) = grad_next_state {
        assert_eq!(next.time_state.dim(), grad_time_state.dim());
        assert_eq!(next.time_prev.len(), grad_time_prev.len());
        assert_eq!(next.cmix_prev.len(), grad_cmix_prev.len());
        grad_time_state += &next.time_state;
        grad_time_prev += &next.time_prev;
        grad_cmix_prev += &next.cmix_prev;
        if let Some(next_first) = &next.v_first {
            assert_eq!(next_first.dim(), grad_v_first.dim());
            grad_v_first += next_first;
        }
    }

    assert_eq!(grad_input.ncols(), channels);
    RwkvBlockBackward {
        grad_input,
        grad_time_prev,
        grad_cmix_prev,
        grad_time_state,
        grad_v_first,
        grad_ln0_weight,
        grad_ln0_bias,
        grad_ln1_weight,
        grad_ln1_bias,
        grad_ln2_weight,
        grad_ln2_bias,
        time,
        cmix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_block_full_tape::RwkvBlockFullTape;

    #[test]
    fn backward_output_has_expected_gradient_shapes() {
        let block = RwkvBlock::new(16, 2, 0, 4);
        let input = Array2::<f32>::zeros((3, 16));
        let state = block.forward(&input, None, None).1;
        let time = block.time_mix.forward_with_tape(&input, None, None, None).4;
        let tape = RwkvBlockFullTape::new(input.clone(), time, state);
        let grad = Array2::<f32>::ones((3, 16));
        let result = backward(&block, &tape, &grad, None);
        assert_eq!(result.grad_input.dim(), input.dim());
        assert_eq!(result.grad_time_prev.len(), 16);
        assert_eq!(result.grad_time_state.dim(), (1, 2, 8, 8));
        assert!(result.grad_input.iter().all(|v| v.is_finite()));
    }
}
