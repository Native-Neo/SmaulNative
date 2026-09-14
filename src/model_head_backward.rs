use ndarray::{Array1, Array2};

use crate::embedding_backward;
use crate::layer_norm_backward;

pub struct HeadBackward {
    pub grad_input: Array2<f32>,
    pub grad_weight: Array2<f32>,
    pub grad_ln_weight: Array1<f32>,
    pub grad_ln_bias: Array1<f32>,
    pub grad_embedding: Array2<f32>,
}

pub fn backward(
    token_ids: &[usize],
    normalized_input: &Array2<f32>,
    logits_grad: &Array2<f32>,
    head_weight: &Array2<f32>,
    ln_input: &Array2<f32>,
    ln_weight: &Array1<f32>,
    ln_eps: f32,
    vocab_size: usize,
) -> HeadBackward {
    assert_eq!(normalized_input.nrows(), logits_grad.nrows());
    assert_eq!(normalized_input.ncols(), head_weight.ncols());
    assert_eq!(logits_grad.ncols(), head_weight.nrows());
    assert_eq!(ln_input.dim(), normalized_input.dim());
    assert_eq!(ln_weight.len(), normalized_input.ncols());

    let grad_weight = logits_grad.t().dot(normalized_input);
    let grad_normalized = logits_grad.dot(head_weight);
    let (grad_input, grad_ln_weight, grad_ln_bias) =
        layer_norm_backward::backward(ln_input, &grad_normalized, ln_weight, ln_eps);
    let grad_embedding = embedding_backward::backward(token_ids, &grad_input, vocab_size);

    HeadBackward { grad_input, grad_weight, grad_ln_weight, grad_ln_bias, grad_embedding }
}

pub fn accumulate_parameter_gradient(target: &mut Array2<f32>, source: &Array2<f32>) {
    assert_eq!(target.dim(), source.dim());
    *target += source;
}

pub fn accumulate_vector_gradient(target: &mut Array1<f32>, source: &Array1<f32>) {
    assert_eq!(target.len(), source.len());
    *target += source;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn computes_head_and_embedding_gradients() {
        let tokens = [1usize, 2, 1];
        let normalized = array![[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]];
        let logits_grad = array![[1.0, 0.0, -1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]];
        let head = array![[0.2, 0.3], [0.4, -0.1], [0.5, 0.6]];
        let gamma = Array1::ones(2);
        let grads = backward(&tokens, &normalized, &logits_grad, &head, &normalized, &gamma, 1e-5, 4);
        assert_eq!(grads.grad_weight.dim(), (3, 2));
        assert_eq!(grads.grad_input.dim(), (3, 2));
        assert_eq!(grads.grad_embedding.dim(), (4, 2));
        assert!(grads.grad_embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn accumulation_adds_in_place() {
        let mut matrix = Array2::zeros((2, 2));
        let source = Array2::ones((2, 2));
        accumulate_parameter_gradient(&mut matrix, &source);
        accumulate_parameter_gradient(&mut matrix, &source);
        assert_eq!(matrix, Array2::from_elem((2, 2), 2.0));

        let mut vector = Array1::zeros(2);
        let source = Array1::ones(2);
        accumulate_vector_gradient(&mut vector, &source);
        assert_eq!(vector, Array1::from_elem(2, 1.0));
    }
}
