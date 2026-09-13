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
        Self { lr, beta1, beta2, weight_decay, momentum: vec![0.0; parameter_count] }
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
            self.momentum[index] = self.beta2 * self.momentum[index] + (1.0 - self.beta2) * gradient[index];
        }
    }

    fn zero_grad(&mut self) { self.momentum.fill(0.0); }
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step_count: usize,
    first: Vec<f32>,
    second: Vec<f32>,
}

impl AdamW {
    pub fn new(parameter_count: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self { lr, beta1, beta2, eps, weight_decay, step_count: 0, first: vec![0.0; parameter_count], second: vec![0.0; parameter_count] }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]) {
        assert_eq!(parameter.len(), gradient.len());
        assert_eq!(parameter.len(), self.first.len());
        self.step_count += 1;
        let b1t = 1.0 - self.beta1.powi(self.step_count as i32);
        let b2t = 1.0 - self.beta2.powi(self.step_count as i32);
        for i in 0..parameter.len() {
            self.first[i] = self.beta1 * self.first[i] + (1.0 - self.beta1) * gradient[i];
            self.second[i] = self.beta2 * self.second[i] + (1.0 - self.beta2) * gradient[i] * gradient[i];
            let m = self.first[i] / b1t;
            let v = self.second[i] / b2t;
            parameter[i] *= 1.0 - self.lr * self.weight_decay;
            parameter[i] -= self.lr * m / (v.sqrt() + self.eps);
        }
    }

    fn zero_grad(&mut self) {
        self.first.fill(0.0);
        self.second.fill(0.0);
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{AdamW, Lion, Optimizer};

    #[test]
    fn lion_updates_parameters() {
        let mut optimizer = Lion::new(2, 0.01, 0.9, 0.99, 0.0);
        let mut parameter = vec![1.0, -1.0];
        optimizer.step(&mut parameter, &[1.0, -1.0]);
        assert!(parameter[0] < 1.0 && parameter[1] > -1.0);
    }

    #[test]
    fn adamw_updates_parameters() {
        let mut optimizer = AdamW::new(1, 0.01, 0.9, 0.999, 1e-8, 0.0);
        let mut parameter = vec![1.0];
        optimizer.step(&mut parameter, &[1.0]);
        assert!(parameter[0] < 1.0);
    }
}
