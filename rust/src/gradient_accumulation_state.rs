#[derive(Clone, Debug, Default)]
pub struct GradientAccumulationState {
    pub micro_steps: usize,
}

impl GradientAccumulationState {
    pub fn record(&mut self) { self.micro_steps = self.micro_steps.saturating_add(1); }
    pub fn ready(&self, accumulation: usize) -> bool {
        accumulation > 0 && self.micro_steps >= accumulation
    }
    pub fn reset(&mut self) { self.micro_steps = 0; }
}
