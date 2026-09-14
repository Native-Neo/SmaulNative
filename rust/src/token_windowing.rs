pub fn bounded_range(length: usize, start: usize, count: usize) -> std::ops::Range<usize> {
    assert!(start <= length);
    let end = start.saturating_add(count).min(length);
    start..end
}
