use ndarray::{Array1, Array2};

use crate::linear_loss_backward::logits_cross_entropy_backward;
use crate::model_backward::ModelBackwardTape;
use crate::model_head_backward::{backward as head_backward, HeadBackward};
use crate::rwkv_model::RwkvModel;

pub struct ModelTrainStep {
    pub loss: f32,
    pub logits_gradient: Array2<f32>,
    pub head: HeadBackward,
    pub tape: ModelBackwardTape,
}

impl ModelTrainStep {
    pub fn run(model: &RwkvModel, token_ids: &[usize], targets: &Array1<usize>) -> Self {
        assert_eq!(token_ids.len(), targets.len());
        let (logits, tape) = model.forward_with_tape(token_ids);
        let (loss, logits_gradient) = logits_cross_entropy_backward(&logits, targets);
        let ln_input = tape.ln_input.as_ref().expect("forward tape missing layer norm input");
        let normalized = tape.normalized.as_ref().expect("forward tape missing normalized output");
        let head = head_backward(
            token_ids,
            normalized,
            &logits_gradient,
            &model.head.weight,
            ln_input,
            &model.ln_out.weight,
            model.ln_out.eps,
            model.config.vocab_size,
        );
        Self { loss, logits_gradient, head, tape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;

    #[test]
    fn computes_loss_and_head_gradients() {
        let config = RwkvModelConfig::new(16, 8, 2, 4);
        let model = RwkvModel::new(config, 1234);
        let tokens = [1usize, 2, 3];
        let targets = Array1::from_vec(vec![2, 3, 4]);
        let step = ModelTrainStep::run(&model, &tokens, &targets);
        assert!(step.loss.is_finite());
        assert_eq!(step.logits_gradient.dim(), (3, 16));
        assert_eq!(step.head.grad_weight.dim(), (16, 8));
        assert_eq!(step.head.grad_embedding.dim(), (16, 8));
        assert_eq!(step.tape.len(), 2);
    }
}
