use ndarray::{Array1, Array2, Axis};

pub fn loss_and_gradient(logits: &Array2<f32>, targets: &[usize]) -> (f32, Array2<f32>) {
    assert_eq!(logits.nrows(), targets.len());
    assert!(logits.ncols() > 0);
    let mut grad = Array2::<f32>::zeros(logits.raw_dim());
    let mut loss = 0.0;
    for (i, row) in logits.axis_iter(Axis(0)).enumerate() {
        assert!(targets[i] < row.len());
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
        let log_sum = max + sum.ln();
        loss += log_sum - row[targets[i]];
        for j in 0..row.len() {
            grad[[i, j]] = (row[j] - max).exp() / sum;
        }
        grad[[i, targets[i]]] -= 1.0;
    }
    let n = logits.nrows() as f32;
    grad /= n;
    (loss / n, grad)
}

pub fn per_token_loss(logits: &Array1<f32>, target: usize) -> f32 {
    assert!(target < logits.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln() - logits[target]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradient_rows_sum_to_zero() {
        let (_, grad) = loss_and_gradient(&array![[1.0, 2.0, 3.0]], &[2]);
        assert!(grad.row(0).sum().abs() < 1e-6);
    }
}
