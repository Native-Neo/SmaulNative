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
    grad_next_time_state: Option<&Array4<f32>>,
    grad_next_time_prev: Option<&Array1<f32>>,
    grad_next_cmix_prev: Option<&Array1<f32>>,
    grad_next_v_first: Option<&Array2<f32>>,
) -> RwkvBlockBackward {
    assert_eq!(grad_output.dim(), tape.output.dim());

    let grad_cmix_output = grad_output.clone();
    let grad_skip = grad_output.clone();
    let initial_cmix_prev = tape.initial_state.as_ref().map(|s| &s.cmix_prev);

    let mut cmix = rwkv_cmix_backward::backward(
        &tape.cmix_input,
        initial_cmix_prev,
        &grad_cmix_output,
        &block.cmix.x_k,
        &block.cmix.key,
        &block.cmix.value,
    );
    if let Some(g) = grad_next_cmix_prev {
        assert_eq!(g.len(), cmix.grad_prev.len());
        cmix.grad_prev += g;
    }

    let (grad_ln2_input, grad_ln2_weight, grad_ln2_bias) = layer_norm_backward::backward(
        &tape.ln2_input,
        &cmix.grad_input,
        &block.ln2.weight,
        block.ln2.eps,
    );

    let grad_residual = grad_skip + &grad_ln2_input;

    let mut time = rwkv_time_mix_backward_full::backward(
        &block.time_mix,
        &tape.time,
        &grad_residual,
    );

    if let Some(g) = grad_next_time_state {
        assert_eq!(g.dim(), time.grad_state.dim());
        time.grad_state += g;
    }
    if let Some(g) = grad_next_time_prev {
        assert_eq!(g.len(), time.grad_prev.len());
        time.grad_prev += g;
    }
    if let Some(g) = grad_next_v_first {
        assert_eq!(g.dim(), time.grad_v_first.dim());
        time.grad_v_first += g;
    }

    let (grad_ln1_input, grad_ln1_weight, grad_ln1_bias) = layer_norm_backward::backward(
        &tape.ln1_input,
        &time.grad_input,
        &block.ln1.weight,
        block.ln1.eps,
    );

    let grad_into_input = grad_residual + &grad_ln1_input;
    let mut grad_ln0_weight = None;
    let mut grad_ln0_bias = None;
    let grad_input = if let (Some(ln0), Some(ln0_input)) = (block.ln0.as_ref(), tape.ln0_input.as_ref()) {
        let (dx, dw, db) = layer_norm_backward::backward(
            ln0_input,
            &grad_into_input,
            &ln0.weight,
            ln0.eps,
        );
        grad_ln0_weight = Some(dw);
        grad_ln0_bias = Some(db);
        dx
    } else {
        grad_into_input
    };

    RwkvBlockBackward {
        grad_input,
        grad_time_prev: time.grad_prev.clone(),
        grad_cmix_prev: cmix.grad_prev.clone(),
        grad_time_state: time.grad_state.clone(),
        grad_v_first: time.grad_v_first.clone(),
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
    use ndarray::Array2;

    fn loss(block: &RwkvBlock, x: &Array2<f32>) -> f32 {
        block.forward(x, None, None).0.sum()
    }

    #[test]
    fn input_gradient_matches_finite_difference() {
        let block=RwkvBlock::new(16,2,0,4);
        let x=Array2::from_shape_fn((2,16),|(r,c)|0.02*(r as f32)+0.003*(c as f32)+0.01);
        let (_,_,tape)=block.forward_with_full_tape(&x,None,None);
        let analytic=backward(&block,&tape,&Array2::ones((2,16)),None,None,None,None).grad_input;
        let eps=1e-3_f32;
        for &(row,col) in &[(0usize,0usize),(0,7),(1,3),(1,15)] {
            let mut plus=x.clone();
            let mut minus=x.clone();
            plus[[row,col]]+=eps;
            minus[[row,col]]-=eps;
            let numeric=(loss(&block,&plus)-loss(&block,&minus))/(2.0*eps);
            let a=analytic[[row,col]];
            let tolerance=3e-2_f32.max(3e-2*a.abs());
            assert!((numeric-a).abs()<=tolerance,"gradient mismatch at ({row},{col}): analytic={a}, numeric={numeric}");
        }
    }
}
