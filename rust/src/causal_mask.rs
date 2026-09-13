use ndarray::Array2;

/// Applies an additive causal mask to attention scores in place.
pub fn apply(scores: &mut Array2<f32>) {
    for row in 0..scores.nrows() {
        for col in (row + 1)..scores.ncols() {
            scores[[row, col]] = f32::NEG_INFINITY;
        }
    }
}

/// Clears gradients for entries that are masked by causal attention.
pub fn backward_mask(grad_scores: &mut Array2<f32>) {
    for row in 0..grad_scores.nrows() {
        for col in (row + 1)..grad_scores.ncols() {
            grad_scores[[row, col]] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn masks_future_tokens() {
        let mut x = Array2::zeros((3, 3));
        apply(&mut x);
        assert!(x[[0, 1]].is_infinite() && x[[0, 1]].is_sign_negative());
        assert_eq!(x[[2, 0]], 0.0);
    }

    #[test]
    fn backward_clears_future_gradients() {
        let mut x = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
        backward_mask(&mut x);
        assert_eq!(x[[0, 1]], 0.0);
        assert_eq!(x[[1, 2]], 0.0);
        assert_eq!(x[[2, 0]], 7.0);
    }
}
