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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_next_token_pairs() {
        let (input, target) = shifted_targets(&[4, 5, 6]);
        assert_eq!(input, vec![4, 5]);
        assert_eq!(target, vec![5, 6]);
    }
}
