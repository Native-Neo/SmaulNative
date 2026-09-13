#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub grad_clip: Option<f32>,
    pub accumulation_steps: usize,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { learning_rate: 1e-4, weight_decay: 0.0, grad_clip: Some(1.0), accumulation_steps: 1, log_interval: 10 }
    }
}

impl TrainConfig {
    pub fn validate(&self) {
        assert!(self.learning_rate.is_finite() && self.learning_rate > 0.0);
        assert!(self.weight_decay.is_finite() && self.weight_decay >= 0.0);
        assert!(self.accumulation_steps > 0);
        assert!(self.log_interval > 0);
        if let Some(v) = self.grad_clip { assert!(v.is_finite() && v > 0.0); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_config_is_valid() { TrainConfig::default().validate(); }
}
