use ndarray::Array2;

pub fn gradients_are_finite(gradients: &[&Array2<f32>]) -> bool {
    gradients.iter().all(|gradient| gradient.iter().all(|value| value.is_finite()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn detects_invalid_gradients() {
        let good = array![[1.0, 2.0]];
        let bad = array![[f32::NAN, 2.0]];
        assert!(gradients_are_finite(&[&good]));
        assert!(!gradients_are_finite(&[&good, &bad]));
    }
}
