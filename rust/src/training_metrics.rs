#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub loss_sum: f64,
    pub tokens: u64,
    pub optimizer_steps: u64,
}

impl TrainingMetrics {
    pub fn record(&mut self, loss: f32, tokens: usize) {
        self.loss_sum += loss as f64 * tokens as f64;
        self.tokens = self.tokens.saturating_add(tokens as u64);
    }

    pub fn record_optimizer_step(&mut self) { self.optimizer_steps += 1; }

    pub fn mean_loss(&self) -> f32 {
        if self.tokens == 0 { 0.0 } else { (self.loss_sum / self.tokens as f64) as f32 }
    }

    pub fn reset(&mut self) { *self = Self::default(); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weighted_loss_is_correct() {
        let mut m = TrainingMetrics::default();
        m.record(2.0, 2);
        m.record(1.0, 1);
        assert!((m.mean_loss() - 1.6666666).abs() < 1e-5);
    }
}
