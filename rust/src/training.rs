use ndarray::{Array1, Array2};

use crate::gradient::clip_by_global_norm;
use crate::loss::cross_entropy;
use crate::optimizer::{Lion, Optimizer};
use crate::rwkv_model::RwkvModel;

pub struct TrainStep {
    pub optimizer: Lion,
    pub max_grad_norm: Option<f32>,
}

impl TrainStep {
    pub fn new(parameter_count: usize, lr: f32) -> Self {
        Self { optimizer: Lion::new(parameter_count, lr, 0.9, 0.99, 0.0), max_grad_norm: Some(1.0) }
    }

    pub fn loss_and_gradient(logits: &Array2<f32>, targets: &Array1<usize>) -> (f32, Array2<f32>) {
        cross_entropy(logits, targets)
    }

    pub fn clip_logits_gradient(&self, grad: &mut Array2<f32>) -> f32 {
        let mut refs = vec![grad];
        match self.max_grad_norm {
            Some(max_norm) => clip_by_global_norm(&mut refs, max_norm),
            None => clip_by_global_norm(&mut refs, f32::MAX),
        }
    }

    pub fn token_count(model: &RwkvModel) -> usize { model.config.vocab_size }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn training_step_produces_loss_gradient() {
        let logits = Array2::from_shape_vec((2, 4), vec![1., 2., 3., 4., 4., 3., 2., 1.]).unwrap();
        let targets = Array1::from_vec(vec![3, 0]);
        let (loss, grad) = TrainStep::loss_and_gradient(&logits, &targets);
        assert!(loss.is_finite());
        assert_eq!(grad.dim(), logits.dim());
    }
}
