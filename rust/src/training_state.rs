#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainingState {
    pub step: usize,
    pub epoch: usize,
    pub tokens_seen: u64,
    pub micro_steps: usize,
    pub optimizer_steps: usize,
}

impl TrainingState {
    pub fn new() -> Self { Self::default() }

    pub fn record_micro_step(&mut self, tokens: usize) {
        self.micro_steps += 1;
        self.tokens_seen = self.tokens_seen.saturating_add(tokens as u64);
    }

    pub fn record_optimizer_step(&mut self) {
        self.optimizer_steps += 1;
        self.step += 1;
    }

    pub fn record_epoch(&mut self) { self.epoch += 1; }

    pub fn reset_accumulation(&mut self) { self.micro_steps = 0; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_training_progress() {
        let mut state = TrainingState::new();
        state.record_micro_step(16);
        state.record_micro_step(8);
        state.record_optimizer_step();
        state.record_epoch();
        assert_eq!(state.tokens_seen, 24);
        assert_eq!(state.micro_steps, 2);
        assert_eq!(state.optimizer_steps, 1);
        assert_eq!(state.step, 1);
        assert_eq!(state.epoch, 1);
        state.reset_accumulation();
        assert_eq!(state.micro_steps, 0);
    }
}
