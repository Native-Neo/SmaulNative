use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct ModelGradientAccumulator {
    pub embedding: Array2<f32>,
    pub head: Array2<f32>,
    pub ln_out_weight: Array1<f32>,
    pub ln_out_bias: Array1<f32>,
}

impl ModelGradientAccumulator {
    pub fn zeros(vocab_size: usize, channels: usize) -> Self {
        assert!(vocab_size > 0);
        assert!(channels > 0);
        Self {
            embedding: Array2::zeros((vocab_size, channels)),
            head: Array2::zeros((vocab_size, channels)),
            ln_out_weight: Array1::zeros(channels),
            ln_out_bias: Array1::zeros(channels),
        }
    }

    pub fn clear(&mut self) {
        self.embedding.fill(0.0);
        self.head.fill(0.0);
        self.ln_out_weight.fill(0.0);
        self.ln_out_bias.fill(0.0);
    }

    pub fn add_embedding(&mut self, grad: &Array2<f32>) {
        assert_eq!(grad.dim(), self.embedding.dim());
        self.embedding += grad;
    }

    pub fn add_head(&mut self, grad: &Array2<f32>) {
        assert_eq!(grad.dim(), self.head.dim());
        self.head += grad;
    }

    pub fn add_ln_out(&mut self, weight: &Array1<f32>, bias: &Array1<f32>) {
        assert_eq!(weight.len(), self.ln_out_weight.len());
        assert_eq!(bias.len(), self.ln_out_bias.len());
        self.ln_out_weight += weight;
        self.ln_out_bias += bias;
    }

    pub fn scale(&mut self, factor: f32) {
        assert!(factor.is_finite());
        self.embedding *= factor;
        self.head *= factor;
        self.ln_out_weight *= factor;
        self.ln_out_bias *= factor;
    }

    pub fn is_finite(&self) -> bool {
        self.embedding.iter().all(|v| v.is_finite())
            && self.head.iter().all(|v| v.is_finite())
            && self.ln_out_weight.iter().all(|v| v.is_finite())
            && self.ln_out_bias.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_adds_and_clears() {
        let mut grads = ModelGradientAccumulator::zeros(4, 3);
        grads.head.fill(1.0);
        grads.ln_out_bias.fill(2.0);
        grads.scale(0.5);
        assert_eq!(grads.head[[0, 0]], 0.5);
        assert_eq!(grads.ln_out_bias[0], 1.0);
        assert!(grads.is_finite());
        grads.clear();
        assert_eq!(grads.head[[0, 0]], 0.0);
        assert_eq!(grads.ln_out_bias[0], 0.0);
    }
}
