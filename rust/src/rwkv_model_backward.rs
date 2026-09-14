use ndarray::{Array1, Array2};
use crate::model_backward::{BackwardBlockKind, ModelBackwardTape};
use crate::model_head_backward::HeadBackward;
use crate::rwkv_block_backward_full::{self, RwkvBlockBackward};
use crate::rwkv_model::RwkvModel;

pub struct RwkvModelBackward {
    pub grad_input: Array2<f32>,
    pub grad_embedding: Array2<f32>,
    pub grad_ln_out_weight: Array1<f32>,
    pub grad_ln_out_bias: Array1<f32>,
    pub grad_head: Array2<f32>,
    pub head: HeadBackward,
    pub blocks: Vec<(usize, RwkvBlockBackward)>,
}

pub fn backward(model: &RwkvModel, token_ids: &[usize], tape: &ModelBackwardTape, logits_grad: &Array2<f32>) -> RwkvModelBackward {
    assert!(!token_ids.is_empty());
    assert_eq!(tape.block_order.len(), model.config.n_layer);
    assert_eq!(tape.inputs.len(), model.config.n_layer);
    assert_eq!(tape.outputs.len(), model.config.n_layer);
    assert!(tape.ln_input.is_some());
    assert!(tape.normalized.is_some());
    assert!(tape.logits.is_some());
    assert_eq!(logits_grad.dim(), tape.logits.as_ref().unwrap().dim());
    assert!(tape.block_order.iter().all(|kind| matches!(kind, BackwardBlockKind::Rwkv(_))), "MOBA backward is not integrated yet");

    let normalized = tape.normalized.as_ref().unwrap();
    let ln_input = tape.ln_input.as_ref().unwrap();
    let head = HeadBackward::backward(token_ids, normalized, logits_grad, &model.head.weight, ln_input, &model.ln_out.weight, model.ln_out.eps, model.config.vocab_size);

    let mut grad = head.grad_input.clone();
    let mut grad_v_first = None;
    let mut blocks = Vec::with_capacity(model.config.n_layer);
    for slot in (0..model.config.n_layer).rev() {
        let BackwardBlockKind::Rwkv(index) = tape.block_order[slot] else { unreachable!() };
        assert_eq!(index, slot);
        let block_tape = tape.rwkv_tapes[slot].as_ref().expect("missing RWKV block tape");
        let block_grad = rwkv_block_backward_full::backward(&model.rwkv_blocks[index], block_tape, &grad, None, None, None, grad_v_first.as_ref());
        grad = block_grad.grad_input.clone();
        grad_v_first = Some(block_grad.grad_v_first.clone());
        blocks.push((index, block_grad));
    }
    blocks.reverse();

    RwkvModelBackward {
        grad_input: grad,
        grad_embedding: head.grad_embedding.clone(),
        grad_ln_out_weight: head.grad_ln_weight.clone(),
        grad_ln_out_bias: head.grad_ln_bias.clone(),
        grad_head: head.grad_weight.clone(),
        head,
        blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::cross_entropy_backward;
    use crate::rwkv_model::RwkvModelConfig;

    #[test]
    fn reverse_rwkv_model_backward_produces_gradients() {
        let model = RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4), 123);
        let tokens = [1usize, 2, 3];
        let (logits, tape) = model.forward_with_tape(&tokens);
        let grad = cross_entropy_backward(&logits, &tokens);
        let result = backward(&model, &tokens, &tape, &grad);
        assert_eq!(result.grad_input.dim(), (3, 16));
        assert_eq!(result.grad_embedding.dim(), (32, 16));
        assert_eq!(result.grad_head.dim(), (32, 16));
        assert_eq!(result.blocks.len(), 2);
        assert!(result.grad_input.iter().all(|v| v.is_finite()));
        assert!(result.grad_embedding.iter().all(|v| v.is_finite()));
    }
}
