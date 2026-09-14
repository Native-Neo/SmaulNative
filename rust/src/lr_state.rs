#[derive(Clone, Debug)]
pub struct LearningRateState {
    pub base: f32,
    pub step: usize,
}

impl LearningRateState {
    pub fn new(base: f32) -> Self {
        assert!(base.is_finite() && base > 0.0);
        Self { base, step: 0 }
    }

    pub fn current(&self) -> f32 { self.base }
    pub fn advance(&mut self) { self.step = self.step.saturating_add(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_scheduler_step() {
        let mut state = LearningRateState::new(1e-4);
        state.advance();
        assert_eq!(state.step, 1);
        assert_eq!(state.current(), 1e-4);
    }
}
