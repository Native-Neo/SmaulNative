use ndarray::{Array1, Array2};

pub fn backward(grad_output: &Array2<f32>, token_ids: &[usize], vocab_size: usize) -> Array2<f32> {
    assert_eq!(grad_output.nrows(), token_ids.len());
    let hidden = grad_output.ncols();
    let mut grad = Array2::<f32>::zeros((vocab_size, hidden));
    for (row, &token) in token_ids.iter().enumerate() {
        assert!(token < vocab_size);
        let mut dst = grad.row_mut(token);
        dst += &grad_output.row(row);
    }
    grad
}

pub fn backward_single(grad_output: &Array1<f32>, token: usize, vocab_size: usize) -> Array2<f32> {
    assert!(token < vocab_size);
    let mut grad = Array2::<f32>::zeros((vocab_size, grad_output.len()));
    grad.row_mut(token).assign(grad_output);
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn repeated_tokens_accumulate() {
        let grad = backward(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &[2, 1, 2], 4);
        assert_eq!(grad.row(2), array![6.0, 8.0]);
        assert_eq!(grad.row(1), array![3.0, 4.0]);
    }
}
