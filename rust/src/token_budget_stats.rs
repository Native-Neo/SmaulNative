#[derive(Clone, Debug, Default)]
pub struct TokenBudgetStats {
    pub consumed: u64,
    pub rejected: u64,
}

impl TokenBudgetStats {
    pub fn accepted(&mut self, tokens: u64) { self.consumed = self.consumed.saturating_add(tokens); }
    pub fn rejected(&mut self) { self.rejected = self.rejected.saturating_add(1); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tracks_budget_events() {
        let mut stats = TokenBudgetStats::default();
        stats.accepted(12);
        stats.rejected();
        assert_eq!(stats.consumed, 12);
        assert_eq!(stats.rejected, 1);
    }
}
