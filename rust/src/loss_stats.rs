#[derive(Clone, Debug, Default)]
pub struct LossStats {
    pub sum: f64,
    pub count: u64,
}

impl LossStats {
    pub fn add(&mut self, loss: f32) {
        assert!(loss.is_finite());
        self.sum += loss as f64;
        self.count = self.count.saturating_add(1);
    }

    pub fn mean(&self) -> f32 {
        if self.count == 0 { 0.0 } else { (self.sum / self.count as f64) as f32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_losses() {
        let mut stats = LossStats::default();
        stats.add(2.0);
        stats.add(4.0);
        assert_eq!(stats.mean(), 3.0);
    }
}
