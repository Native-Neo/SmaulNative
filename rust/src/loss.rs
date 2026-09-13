use ndarray::{Array1, Array2};

pub fn cross_entropy(logits: &Array2<f32>, targets: &Array1<usize>) -> (f32, Array2<f32>) {
    assert_eq!(logits.nrows(), targets.len());
    assert!(logits.ncols() > 0);
    let mut loss = 0.0;
    let mut grad = Array2::zeros(logits.dim());
    for row in 0..logits.nrows() {
        let mut max_logit = f32::NEG_INFINITY;
        for col in 0..logits.ncols() { max_logit = max_logit.max(logits[[row, col]]); }
        let mut denom = 0.0;
        for col in 0..logits.ncols() { denom += (logits[[row, col]] - max_logit).exp(); }
        let target = targets[row];
        assert!(target < logits.ncols());
        let target_prob = (logits[[row, target]] - max_logit).exp() / denom;
        loss -= target_prob.ln();
        for col in 0..logits.ncols() {
            grad[[row, col]] = (logits[[row, col]] - max_logit).exp() / denom;
        }
        grad[[row, target]] -= 1.0;
    }
    let n = logits.nrows() as f32;
    (loss / n, grad / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_is_finite() {
        let logits = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 3., 2., 1.]).unwrap();
        let targets = Array1::from_vec(vec![2, 0]);
        let (loss, grad) = cross_entropy(&logits, &targets);
        assert!(loss.is_finite());
        assert_eq!(grad.dim(), logits.dim());
    }
}
