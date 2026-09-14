use ndarray::{Array1, Array2};

pub fn value(logits: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    assert_eq!(logits.nrows(), targets.len());
    if targets.is_empty() { return 0.0; }
    let mut total = 0.0f64;
    for (row, &target) in logits.outer_iter().zip(targets.iter()) {
        assert!(target < row.len());
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&x| ((x - max) as f64).exp()).sum();
        total += -(row[target] - max) as f64 + sum.ln();
    }
    (total / targets.len() as f64) as f32
}
