#[derive(Clone, Debug)]
pub struct ParameterGroup {
    pub name: String,
    pub parameter_count: usize,
    pub learning_rate_scale: f32,
    pub weight_decay: f32,
}

impl ParameterGroup {
    pub fn new(name: impl Into<String>, parameter_count: usize, learning_rate_scale: f32, weight_decay: f32) -> Self {
        assert!(learning_rate_scale.is_finite() && learning_rate_scale > 0.0);
        assert!(weight_decay.is_finite() && weight_decay >= 0.0);
        Self { name: name.into(), parameter_count, learning_rate_scale, weight_decay }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stores_optimizer_group_settings() {
        let group = ParameterGroup::new("rwkv", 128, 0.5, 0.01);
        assert_eq!(group.parameter_count, 128);
        assert_eq!(group.learning_rate_scale, 0.5);
    }
}
