use ndarray::{Array1, Array2};

pub fn mean_negative_log_likelihood(logits: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    assert_eq!(logits.nrows(), targets.len());
    if targets.is_empty() { return 0.0; }
    let mut total = 0.0f64;
    for (row, &target) in logits.outer_iter().zip(targets.iter()) {
        assert!(target < row.len());
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&x| ((x - max) as f64).exp()).sum();
        let log_sum = sum.ln() as f32;
        total += (-(row[target] - max - log_sum)) as f64;
    }
    (total / targets.len() as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_for_certain_single_class() {
        let logits = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap();
        let targets = Array1::from_vec(vec![0]);
        assert!(mean_negative_log_likelihood(&logits, &targets).abs() < 1e-6);
    }
}
