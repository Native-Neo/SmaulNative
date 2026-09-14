use ndarray::{Array1, Array2};

pub fn logits_cross_entropy_backward(logits: &Array2<f32>, targets: &Array1<usize>) -> (f32, Array2<f32>) {
    assert_eq!(logits.nrows(), targets.len());
    let mut grad = Array2::zeros(logits.dim());
    let mut loss = 0.0;
    for i in 0..logits.nrows() {
        let row = logits.row(i);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (*v - max).exp()).sum();
        let logsum = max + sum.ln();
        let target = targets[i];
        assert!(target < logits.ncols());
        loss += logsum - row[target];
        for j in 0..logits.ncols() { grad[[i, j]] = (row[j] - max).exp() / sum; }
        grad[[i, target]] -= 1.0;
    }
    let n = logits.nrows() as f32;
    grad.mapv_inplace(|v| v / n);
    (loss / n, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_rows_sum_to_zero() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0,2.0,3.0,3.0,2.0,1.0]).unwrap();
        let targets = Array1::from_vec(vec![2, 0]);
        let (_, grad) = logits_cross_entropy_backward(&logits, &targets);
        for row in grad.rows() { assert!(row.sum().abs() < 1e-6); }
    }
}
