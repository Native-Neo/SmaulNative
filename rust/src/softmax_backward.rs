use ndarray::Array2;

/// Backpropagates through row-wise softmax using the full Jacobian-vector product.
pub fn backward(probabilities: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(probabilities.dim(), grad_output.dim());
    let mut grad = Array2::<f32>::zeros(probabilities.raw_dim());
    for row in 0..probabilities.nrows() {
        let mut dot = 0.0;
        for col in 0..probabilities.ncols() {
            dot += probabilities[[row, col]] * grad_output[[row, col]];
        }
        for col in 0..probabilities.ncols() {
            grad[[row, col]] = probabilities[[row, col]] * (grad_output[[row, col]] - dot);
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradient_sums_to_zero() {
        let p = array![[0.2, 0.3, 0.5]];
        let g = array![[1.0, 2.0, 3.0]];
        let result = backward(&p, &g);
        assert!(result.sum().abs() < 1e-6);
    }
}
