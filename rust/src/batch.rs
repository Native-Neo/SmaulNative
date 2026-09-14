use ndarray::Array1;

pub fn validate_token_batch(tokens: &[usize], vocab_size: usize) {
    assert!(vocab_size > 0);
    assert!(tokens.iter().all(|token| *token < vocab_size));
}

pub fn shift_targets(tokens: &[usize]) -> (Array1<usize>, Array1<usize>) {
    assert!(tokens.len() >= 2);
    let input = Array1::from_vec(tokens[..tokens.len() - 1].to_vec());
    let target = Array1::from_vec(tokens[1..].to_vec());
    (input, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_shift_matches_language_modeling_layout() {
        let (input, target) = shift_targets(&[4, 7, 9, 2]);
        assert_eq!(input.to_vec(), vec![4, 7, 9]);
        assert_eq!(target.to_vec(), vec![7, 9, 2]);
    }

    #[test]
    fn token_validation_accepts_valid_ids() {
        validate_token_batch(&[0, 2, 4], 5);
    }
}
