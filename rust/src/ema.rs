#[derive(Clone, Debug)]
pub struct ExponentialAverage {
    decay: f32,
    value: Option<f32>,
}

impl ExponentialAverage {
    pub fn new(decay: f32) -> Self {
        assert!(decay.is_finite() && (0.0..1.0).contains(&decay));
        Self { decay, value: None }
    }

    pub fn update(&mut self, value: f32) -> f32 {
        assert!(value.is_finite());
        let next = match self.value { Some(old) => self.decay * old + (1.0 - self.decay) * value, None => value };
        self.value = Some(next);
        next
    }

    pub fn value(&self) -> Option<f32> { self.value }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initializes_from_first_value() {
        let mut avg = ExponentialAverage::new(0.9);
        assert_eq!(avg.update(3.0), 3.0);
        assert!(avg.update(5.0) > 3.0);
    }
}
