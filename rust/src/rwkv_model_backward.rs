use ndarray::{Array1, Array2};
use crate::model_backward::{BackwardBlockKind, ModelBackwardTape};
use crate::model_head_backward::HeadBackward;
use crate::rwkv_block::RwkvBlockState;
use crate::rwkv_block_backward_full::{self, RwkvBlockBackward};
use crate::rwkv_model::RwkvModel;

pub struct RwkvModelBackward {
    pub grad_input: Array2<f32>, pub grad_embedding: Array2<f32>,
    pub grad_ln_out_weight: Array1<f32>, pub grad_ln_out_bias: Array1<f32>, pub grad_head: Array2<f32>,
    pub head: HeadBackward, pub blocks: Vec<(usize, RwkvBlockBackward)>,
    pub grad_initial_states: Vec<Option<RwkvBlockState>>, pub grad_v_first: Option<Array2<f32>>,
}

pub fn backward(model: &RwkvModel, token_ids: &[usize], tape: &ModelBackwardTape, logits_grad: &Array2<f32>) -> RwkvModelBackward {
    backward_with_state_grads(model, token_ids, tape, logits_grad, None)
}

pub fn backward_with_state_grads(model: &RwkvModel, token_ids: &[usize], tape: &ModelBackwardTape, logits_grad: &Array2<f32>, grad_next_states: Option<&[Option<RwkvBlockState>]>) -> RwkvModelBackward {
    assert!(!token_ids.is_empty());
    assert_eq!(tape.block_order.len(), model.config.n_layer);
    assert_eq!(tape.inputs.len(), model.config.n_layer);
    assert_eq!(tape.outputs.len(), model.config.n_layer);
    let normalized = tape.normalized.as_ref().expect("missing normalized output");
    let ln_input = tape.ln_input.as_ref().expect("missing layer norm input");
    let logits = tape.logits.as_ref().expect("missing logits");
    assert_eq!(logits_grad.dim(), logits.dim());
    assert!(tape.block_order.iter().all(|kind| matches!(kind, BackwardBlockKind::Rwkv(_))), "MOBA backward is not integrated yet");
    if let Some(grads) = grad_next_states { assert_eq!(grads.len(), model.rwkv_blocks.len()); }
    let head = HeadBackward::backward(token_ids, normalized, logits_grad, &model.head.weight, ln_input, &model.ln_out.weight, model.ln_out.eps, model.config.vocab_size);
    let mut grad = head.grad_input.clone(); let mut grad_v_first = None; let mut blocks = Vec::with_capacity(model.config.n_layer);
    let mut grad_initial_states = (0..model.rwkv_blocks.len()).map(|_| None).collect::<Vec<_>>();
    for slot in (0..model.config.n_layer).rev() {
        let BackwardBlockKind::Rwkv(index) = tape.block_order[slot] else { unreachable!() };
        let block_tape = tape.rwkv_tapes[slot].as_ref().expect("missing RWKV block tape");
        let next = grad_next_states.and_then(|all| all[index].as_ref());
        let block_grad = rwkv_block_backward_full::backward(&model.rwkv_blocks[index], block_tape, &grad,
            next.map(|s| &s.time_state), next.map(|s| &s.time_prev), next.map(|s| &s.cmix_prev), grad_v_first.as_ref());
        grad = block_grad.grad_input.clone();
        grad_initial_states[index] = Some(RwkvBlockState { time_state: block_grad.grad_time_state.clone(), time_prev: block_grad.grad_time_prev.clone(), cmix_prev: block_grad.grad_cmix_prev.clone(), v_first: Some(block_grad.grad_v_first.clone()) });
        grad_v_first = Some(match grad_v_first { Some(mut total) => { total += &block_grad.grad_v_first; total }, None => block_grad.grad_v_first.clone() });
        blocks.push((index, block_grad));
    }
    blocks.reverse();
    RwkvModelBackward { grad_input: grad, grad_embedding: head.grad_embedding.clone(), grad_ln_out_weight: head.grad_ln_weight.clone(), grad_ln_out_bias: head.grad_ln_bias.clone(), grad_head: head.grad_weight.clone(), head, blocks, grad_initial_states, grad_v_first }
}

#[cfg(test)]
mod tests {
    use super::*; use crate::loss::{cross_entropy, cross_entropy_backward}; use crate::rwkv_model::{RwkvModel, RwkvModelConfig};
    #[test]
    fn reverse_rwkv_model_backward_produces_gradients() {
        let model=RwkvModel::new(RwkvModelConfig::new(32,16,2,4),123); let tokens=[1usize,2,3]; let (logits,tape)=model.forward_with_tape(&tokens); let result=backward(&model,&tokens,&tape,&cross_entropy_backward(&logits,&tokens));
        assert_eq!(result.blocks.len(),2); assert_eq!(result.grad_initial_states.len(),2); assert!(result.grad_input.iter().all(|v|v.is_finite()));
    }
    #[test]
    fn state_gradient_api_accepts_next_chunk_gradients() {
        let model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),123); let tokens=[1usize,2,3]; let (logits,tape)=model.forward_with_tape(&tokens); let mut next=(0..2).map(|_|None).collect::<Vec<_>>(); next[0]=Some(RwkvBlockState { time_state: ndarray::Array4::zeros((1,2,4,4)), time_prev: Array1::zeros(8), cmix_prev: Array1::zeros(8), v_first: None }); let result=backward_with_state_grads(&model,&tokens,&tape,&cross_entropy_backward(&logits,&tokens),Some(&next)); assert!(result.grad_input.iter().all(|v|v.is_finite()));
    }
    #[allow(dead_code)] fn _loss(model:&RwkvModel,tokens:&[usize])->f32 { cross_entropy(&model.forward(tokens,None).0,&Array1::from_vec(tokens.to_vec())).0 }
}
