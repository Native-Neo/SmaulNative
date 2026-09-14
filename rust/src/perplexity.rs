pub fn perplexity(mean_negative_log_likelihood: f32) -> f32 {
    assert!(mean_negative_log_likelihood.is_finite());
    mean_negative_log_likelihood.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponentiates_mean_nll() {
        let value = perplexity(0.0);
        assert_eq!(value, 1.0);
    }
}
