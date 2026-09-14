#[derive(Clone, Copy, Debug, Default)]
pub struct Throughput {
    pub tokens: u64,
    pub seconds: f64,
}

impl Throughput {
    pub fn tokens_per_second(&self) -> f64 {
        if self.seconds <= 0.0 { 0.0 } else { self.tokens as f64 / self.seconds }
    }
}
