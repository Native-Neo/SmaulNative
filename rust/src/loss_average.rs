#[derive(Clone, Debug, Default)]
pub struct LossAverage {
    sum: f64,
    count: u64,
}

impl LossAverage {
    pub fn add(&mut self, loss: f32) {
        assert!(loss.is_finite());
        self.sum += loss as f64;
        self.count = self.count.saturating_add(1);
    }

    pub fn value(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.sum / self.count as f64) as f32 }
    }
}
