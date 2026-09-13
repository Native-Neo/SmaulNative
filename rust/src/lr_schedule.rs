pub trait LearningRateSchedule {
    fn value(&self, step: usize) -> f32;
}

pub struct CosineDecay {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub total_steps: usize,
}

impl LearningRateSchedule for CosineDecay {
    fn value(&self, step: usize) -> f32 {
        if self.total_steps == 0 { return self.min_lr; }
        let progress = (step.min(self.total_steps) as f32) / self.total_steps as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }
}

pub struct WarmupCosine {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LearningRateSchedule for WarmupCosine {
    fn value(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.initial_lr * (step as f32 + 1.0) / self.warmup_steps as f32;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = step.saturating_sub(self.warmup_steps).min(decay_steps) as f32 / decay_steps as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.initial_lr - self.min_lr) * cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_reaches_minimum() {
        let s = CosineDecay { initial_lr: 1.0, min_lr: 0.1, total_steps: 100 };
        assert!((s.value(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn warmup_increases_learning_rate() {
        let s = WarmupCosine { initial_lr: 1.0, min_lr: 0.1, warmup_steps: 10, total_steps: 100 };
        assert!(s.value(1) < s.value(9));
    }
}
