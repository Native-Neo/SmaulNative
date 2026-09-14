#[derive(Clone, Debug)]
pub struct TokenBudget {
    limit: Option<u64>,
    consumed: u64,
}

impl TokenBudget {
    pub fn unlimited() -> Self { Self { limit: None, consumed: 0 } }
    pub fn new(limit: u64) -> Self { Self { limit: Some(limit), consumed: 0 } }
    pub fn consumed(&self) -> u64 { self.consumed }
    pub fn remaining(&self) -> Option<u64> { self.limit.map(|limit| limit.saturating_sub(self.consumed)) }
    pub fn exhausted(&self) -> bool { self.remaining() == Some(0) }
    pub fn consume(&mut self, tokens: u64) -> bool {
        let next = self.consumed.saturating_add(tokens);
        if let Some(limit) = self.limit { if next > limit { return false; } }
        self.consumed = next;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enforces_token_limit() {
        let mut budget = TokenBudget::new(100);
        assert!(budget.consume(60));
        assert_eq!(budget.remaining(), Some(40));
        assert!(!budget.consume(41));
        assert_eq!(budget.consumed(), 60);
    }
}
