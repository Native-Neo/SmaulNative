pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

pub fn require_finite(values: &[f32]) {
    assert!(all_finite(values), "non-finite value encountered");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_non_finite_values() {
        assert!(all_finite(&[0.0, 1.0, -2.0]));
        assert!(!all_finite(&[0.0, f32::NAN]));
    }
}
