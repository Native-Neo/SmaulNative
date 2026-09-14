use ndarray::{Array1, Array2};

pub fn token_accuracy(logits: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    assert_eq!(logits.nrows(), targets.len());
    if targets.is_empty() { return 0.0; }
    let mut correct = 0usize;
    for (row, &target) in logits.outer_iter().zip(targets.iter()) {
        assert!(target < row.len(), "target outside vocabulary");
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        correct += usize::from(best == target);
    }
    correct as f32 / targets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_argmax_accuracy() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 4.0, 2.0, 9.0, 1.0, 3.0]).unwrap();
        let targets = Array1::from_vec(vec![1, 0]);
        assert_eq!(token_accuracy(&logits, &targets), 1.0);
    }
}
