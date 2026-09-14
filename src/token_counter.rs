#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenCounter {
    total: u64,
}

impl TokenCounter {
    pub fn new() -> Self { Self::default() }

    pub fn add(&mut self, tokens: usize) {
        self.total = self.total.saturating_add(tokens as u64);
    }

    pub fn total(&self) -> u64 { self.total }

    pub fn reset(&mut self) { self.total = 0; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_tokens_without_losing_progress() {
        let mut counter = TokenCounter::new();
        counter.add(10);
        counter.add(7);
        assert_eq!(counter.total(), 17);
        counter.reset();
        assert_eq!(counter.total(), 0);
    }
}
