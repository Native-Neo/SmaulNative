#[derive(Clone, Debug, Default)]
pub struct MetricWindow {
    total: f64,
    count: u64,
}

impl MetricWindow {
    pub fn add(&mut self, value: f32) {
        assert!(value.is_finite());
        self.total += value as f64;
        self.count = self.count.saturating_add(1);
    }
    pub fn mean(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.total / self.count as f64) as f32 }
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn reset(&mut self) { self.total = 0.0; self.count = 0; }
}
