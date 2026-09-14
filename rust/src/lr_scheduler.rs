pub trait LrScheduler {
    fn learning_rate(&self, step: usize) -> f32;
}

pub struct WarmupCosine {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub min_lr: f32,
}

impl WarmupCosine {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, min_lr: f32) -> Self {
        assert!(base_lr >= 0.0);
        assert!(total_steps > 0);
        assert!(warmup_steps <= total_steps);
        assert!(min_lr >= 0.0 && min_lr <= base_lr);
        Self { base_lr, warmup_steps, total_steps, min_lr }
    }
}

impl LrScheduler for WarmupCosine {
    fn learning_rate(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        if step >= self.total_steps { return self.min_lr; }
        let span = (self.total_steps - self.warmup_steps).max(1) as f32;
        let progress = (step.saturating_sub(self.warmup_steps) as f32 / span).clamp(0.0, 1.0);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warms_then_decays() {
        let s = WarmupCosine::new(1.0, 2, 10, 0.1);
        assert!(s.learning_rate(0) < s.learning_rate(1));
        assert!(s.learning_rate(2) > s.learning_rate(9));
        assert_eq!(s.learning_rate(10), 0.1);
    }
}
