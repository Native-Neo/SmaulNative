use ndarray::{Array1, Array2};
use crate::linear_loss_backward::logits_cross_entropy_backward;

pub fn shifted_targets(tokens: &[usize]) -> (Vec<usize>, Vec<usize>) {
    assert!(tokens.len() >= 2);
    (tokens[..tokens.len()-1].to_vec(), tokens[1..].to_vec())
}

pub fn sequence_loss(logits: &Array2<f32>, tokens: &[usize]) -> (f32, Array2<f32>) {
    assert_eq!(logits.nrows() + 1, tokens.len());
    let targets = Array1::from_vec(tokens[1..].to_vec());
    logits_cross_entropy_backward(logits, &targets)
}

pub fn masked_sequence_loss(logits: &Array2<f32>, targets: &[i32], ignore_index: i32) -> (f32, Array2<f32>) {
    assert_eq!(logits.nrows(), targets.len());
    let active: Vec<usize> = targets.iter().enumerate().filter_map(|(i, &target)| {
        if target == ignore_index { None } else { Some(i) }
    }).collect();
    if active.is_empty() { return (0.0, Array2::zeros(logits.dim())); }
    let mut grad = Array2::zeros(logits.dim());
    let mut loss = 0.0;
    for &i in &active {
        let row = logits.row(i);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (*v - max).exp()).sum();
        let target = usize::try_from(targets[i]).expect("masked target must be non-negative");
        assert!(target < logits.ncols());
        loss += max + sum.ln() - row[target];
        for j in 0..logits.ncols() { grad[[i, j]] = (row[j] - max).exp() / sum; }
        grad[[i, target]] -= 1.0;
    }
    let n = active.len() as f32;
    grad.mapv_inplace(|v| v / n);
    (loss / n, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_next_token_pairs() {
        let (input, target) = shifted_targets(&[4, 5, 6]);
        assert_eq!(input, vec![4, 5]);
        assert_eq!(target, vec![5, 6]);
    }

    #[test]
    fn masked_loss_ignores_user_tokens() {
        let logits = Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
        let targets = vec![-100, 1, 2];
        let (loss, grad) = masked_sequence_loss(&logits, &targets, -100);
        assert!(loss > 0.0);
        assert!(grad.row(0).iter().all(|&v| v == 0.0));
        assert!(grad.row(1)[1] < 0.0);
    }

    #[test]
    fn empty_mask_returns_zero() {
        let logits = Array2::<f32>::zeros((2, 4));
        let (loss, grad) = masked_sequence_loss(&logits, &[-100, -100], -100);
        assert_eq!(loss, 0.0);
        assert!(grad.iter().all(|&v| v == 0.0));
    }
}
