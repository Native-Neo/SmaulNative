#[derive(Clone, Debug)]
pub struct LogPoint {
    pub step: usize,
    pub tokens: u64,
    pub loss: f32,
    pub learning_rate: f32,
}

impl LogPoint {
    pub fn new(step: usize, tokens: u64, loss: f32, learning_rate: f32) -> Self {
        Self { step, tokens, loss, learning_rate }
    }

    pub fn line(&self) -> String {
        format!("step={} tokens={} loss={:.6} lr={:.8}", self.step, self.tokens, self.loss, self.learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_training_progress() {
        let point = LogPoint::new(3, 128, 2.5, 1e-4);
        assert!(point.line().contains("step=3"));
        assert!(point.line().contains("tokens=128"));
    }
}
