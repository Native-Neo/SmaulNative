pub trait Optimizer {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]);
    fn zero_grad(&mut self);
}

pub struct Lion {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    momentum: Vec<f32>,
}

impl Lion {
    pub fn new(parameter_count: usize, lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            weight_decay,
            momentum: vec![0.0; parameter_count],
        }
    }
}

impl Optimizer for Lion {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]) {
        assert_eq!(parameter.len(), gradient.len());
        assert_eq!(parameter.len(), self.momentum.len());

        for index in 0..parameter.len() {
            let update = self.beta1 * self.momentum[index] + (1.0 - self.beta1) * gradient[index];
            let direction = if update >= 0.0 { 1.0 } else { -1.0 };
            parameter[index] *= 1.0 - self.lr * self.weight_decay;
            parameter[index] -= self.lr * direction;
            self.momentum[index] = self.beta2 * self.momentum[index]
                + (1.0 - self.beta2) * gradient[index];
        }
    }

    fn zero_grad(&mut self) {
        self.momentum.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{Lion, Optimizer};

    #[test]
    fn lion_updates_parameters() {
        let mut optimizer = Lion::new(2, 0.01, 0.9, 0.99, 0.0);
        let mut parameter = vec![1.0, -1.0];
        let gradient = vec![1.0, -1.0];

        optimizer.step(&mut parameter, &gradient);

        assert!(parameter[0] < 1.0);
        assert!(parameter[1] > -1.0);
    }
}
