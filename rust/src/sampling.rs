use ndarray::Array1;

pub fn argmax(logits: &Array1<f32>) -> usize {
    assert!(!logits.is_empty());
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] { best = i; }
    }
    best
}

pub fn temperature(logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
    assert!(temperature.is_finite() && temperature > 0.0);
    logits.mapv(|v| v / temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_selects_largest_logit() {
        let logits = Array1::from_vec(vec![0.2, 4.0, 1.0]);
        assert_eq!(argmax(&logits), 1);
    }
}
