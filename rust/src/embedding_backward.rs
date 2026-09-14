use ndarray::Array2;

/// Accumulates embedding gradients from token ids into a [vocab, dim] matrix.
pub fn backward(token_ids: &[usize], grad_output: &Array2<f32>, vocab_size: usize) -> Array2<f32> {
    assert_eq!(token_ids.len(), grad_output.nrows());
    let dim = grad_output.ncols();
    let mut grad = Array2::<f32>::zeros((vocab_size, dim));
    for (row, &token) in token_ids.iter().enumerate() {
        assert!(token < vocab_size);
        for col in 0..dim {
            grad[[token, col]] += grad_output[[row, col]];
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn repeated_tokens_accumulate() {
        let grad = backward(&[1, 2, 1], &array![[1., 2.], [3., 4.], [5., 6.]], 4);
        assert_eq!(grad[[1, 0]], 6.0);
        assert_eq!(grad[[1, 1]], 8.0);
        assert_eq!(grad[[2, 0]], 3.0);
        assert_eq!(grad[[2, 1]], 4.0);
    }
}
