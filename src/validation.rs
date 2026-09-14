pub fn require_nonempty<T>(values: &[T], name: &str) {
    assert!(!values.is_empty(), "{name} must not be empty");
}

pub fn require_positive(value: usize, name: &str) {
    assert!(value > 0, "{name} must be positive");
}
