#[derive(Clone, Debug, Default)]
pub struct TrainingCounter {
    pub steps: usize,
    pub tokens: u64,
}

impl TrainingCounter {
    pub fn record(&mut self, tokens: usize) {
        self.steps = self.steps.saturating_add(1);
        self.tokens = self.tokens.saturating_add(tokens as u64);
    }
}
