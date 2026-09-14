#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointState {
    pub step: usize,
    pub epoch: usize,
    pub tokens_seen: u64,
    pub rng_seed: u64,
}

impl CheckpointState {
    pub fn new(rng_seed: u64) -> Self { Self { rng_seed, ..Self::default() } }

    pub fn advance(&mut self, tokens: usize) {
        self.step = self.step.saturating_add(1);
        self.tokens_seen = self.tokens_seen.saturating_add(tokens as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_resume_position() {
        let mut state = CheckpointState::new(123);
        state.advance(64);
        assert_eq!(state.step, 1);
        assert_eq!(state.tokens_seen, 64);
        assert_eq!(state.rng_seed, 123);
    }
}
