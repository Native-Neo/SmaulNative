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
        for col in 0..logits.ncols() { grad[[row, col]] = (logits[[row, col]] - max_logit).exp() / denom; }
        grad[[row, target]] -= 1.0;
    }
    let n = logits.nrows() as f32;
    (loss / n, grad / n)
}

pub fn cross_entropy_backward(logits: &Array2<f32>, targets: &[usize]) -> Array2<f32> {
    assert_eq!(logits.nrows(), targets.len());
    cross_entropy(logits, &Array1::from_vec(targets.to_vec())).1
}

#[cfg(test)]
mod tests_loss {
    use super::*;

    #[test]
    fn cross_entropy_is_finite() {
        let logits = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 3., 2., 1.]).unwrap();
        let targets = Array1::from_vec(vec![2, 0]);
        let (loss, grad) = cross_entropy(&logits, &targets);
        assert!(loss.is_finite());
        assert_eq!(grad.dim(), logits.dim());
    }

    #[test]
    fn backward_matches_forward_gradient() {
        let logits = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 3., 2., 1.]).unwrap();
        let targets = [2, 0];
        assert_eq!(cross_entropy_backward(&logits, &targets), cross_entropy(&logits, &Array1::from_vec(targets.to_vec())).1);
    }
}


// ===== sequence_loss =====

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
mod tests_sequence_loss {
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


// ===== linear_loss_backward =====

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
mod tests_linear_loss_backward {
    use super::*;

    #[test]
    fn gradient_rows_sum_to_zero() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0,2.0,3.0,3.0,2.0,1.0]).unwrap();
        let targets = Array1::from_vec(vec![2, 0]);
        let (_, grad) = logits_cross_entropy_backward(&logits, &targets);
        for row in grad.rows() { assert!(row.sum().abs() < 1e-6); }
    }
}
