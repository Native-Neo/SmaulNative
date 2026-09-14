use ndarray::{Array1, Array2};

use crate::linear_loss_backward::logits_cross_entropy_backward;
use crate::model_backward::ModelBackwardTape;
use crate::model_gradients::ModelGradients;
use crate::rwkv_model::RwkvModel;
use crate::rwkv_model_backward::{self, RwkvModelBackward};

pub struct ModelTrainStep {
    pub loss: f32,
    pub logits_gradient: Array2<f32>,
    pub backward: RwkvModelBackward,
    pub gradients: ModelGradients,
    pub tape: ModelBackwardTape,
}

impl ModelTrainStep {
    pub fn run(model: &RwkvModel, token_ids: &[usize], targets: &Array1<usize>) -> Self {
        assert_eq!(token_ids.len(), targets.len());
        let (logits, tape) = model.forward_with_tape(token_ids);
        let (loss, logits_gradient) = logits_cross_entropy_backward(&logits, targets);
        let backward = rwkv_model_backward::backward(model, token_ids, &tape, &logits_gradient);
        let gradients = ModelGradients::from_backward(&backward);
        Self { loss, logits_gradient, backward, gradients, tape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;

    #[test]
    fn computes_full_model_gradients() {
        let model = RwkvModel::new(RwkvModelConfig::new(16, 8, 2, 4), 1234);
        let tokens = [1usize, 2, 3]; let targets = Array1::from_vec(vec![2, 3, 4]);
        let step = ModelTrainStep::run(&model, &tokens, &targets);
        assert!(step.loss.is_finite()); assert_eq!(step.logits_gradient.dim(), (3, 16));
        assert_eq!(step.gradients.layers.len(), 2); assert!(step.gradients.layers.iter().all(|g| g.parameter_count() > 0));
        assert!(step.gradients.embedding.iter().all(|v| v.is_finite())); assert!(step.gradients.head.iter().all(|v| v.is_finite()));
    }
}
