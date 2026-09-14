#[derive(Clone, Debug, Default)]
pub struct EvalState {
    pub batches: usize,
    pub tokens: u64,
    pub loss_sum: f64,
}

impl EvalState {
    pub fn record(&mut self, tokens: usize, loss: f32) {
        assert!(loss.is_finite());
        self.batches = self.batches.saturating_add(1);
        self.tokens = self.tokens.saturating_add(tokens as u64);
        self.loss_sum += loss as f64;
    }

    pub fn mean_loss(&self) -> f32 {
        if self.batches == 0 { 0.0 } else { (self.loss_sum / self.batches as f64) as f32 }
    }
}
