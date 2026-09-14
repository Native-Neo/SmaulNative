#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EpochState {
    pub epoch: usize,
    pub batches_seen: usize,
    pub tokens_seen: u64,
}

impl EpochState {
    pub fn record_batch(&mut self, tokens: usize) {
        self.batches_seen = self.batches_seen.saturating_add(1);
        self.tokens_seen = self.tokens_seen.saturating_add(tokens as u64);
    }

    pub fn next_epoch(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
        self.batches_seen = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resets_batch_count_between_epochs() {
        let mut state = EpochState::default();
        state.record_batch(32);
        state.record_batch(16);
        state.next_epoch();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.batches_seen, 0);
        assert_eq!(state.tokens_seen, 48);
    }
}
