#[derive(Clone, Debug, Default)]
pub struct TrainingProgress {
    pub step: usize,
    pub tokens: u64,
    pub loss: f32,
}

impl TrainingProgress {
    pub fn update(&mut self, step: usize, tokens: u64, loss: f32) {
        assert!(loss.is_finite());
        self.step = step;
        self.tokens = tokens;
        self.loss = loss;
    }
}
