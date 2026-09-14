use ndarray::{Array1, Array2};

use crate::rwkv_cmix_backward;
use crate::rwkv_time_mix_backward;

pub struct SequenceBackward {
    pub grad_input: Array2<f32>,
    pub grad_prev: Array1<f32>,
}

pub fn chain_mix(
    input: &Array2<f32>,
    prev: &Array1<f32>,
    factor: &Array1<f32>,
    grad_output: &Array2<f32>,
) -> SequenceBackward {
    assert_eq!(input.dim(), grad_output.dim());
    let result = rwkv_time_mix_backward::mix_backward(input, prev, factor, grad_output);
    SequenceBackward {
        grad_input: result.grad_input,
        grad_prev: result.grad_prev,
    }
}

pub fn chain_cmix(
    input: &Array2<f32>,
    prev: &Array1<f32>,
    grad_output: &Array2<f32>,
) -> SequenceBackward {
    assert_eq!(input.dim(), grad_output.dim());
    let result = rwkv_cmix_backward::backward(input, prev, grad_output);
    SequenceBackward {
        grad_input: result.grad_input,
        grad_prev: result.grad_prev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn sequence_chain_preserves_shape() {
        let x = Array2::ones((3, 4));
        let p = Array1::zeros(4);
        let f = Array1::from_elem(4, 0.25);
        let g = Array2::ones((3, 4));
        let mixed = chain_mix(&x, &p, &f, &g);
        assert_eq!(mixed.grad_input.dim(), x.dim());
        assert_eq!(mixed.grad_prev.len(), p.len());
    }
}
