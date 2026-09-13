use ndarray::{Array1, Array2};

pub struct ModelGradients {
    pub embedding: Array2<f32>,
    pub ln_out_weight: Array1<f32>,
    pub ln_out_bias: Array1<f32>,
    pub head: Array2<f32>,
}

impl ModelGradients {
    pub fn zeros(vocab_size: usize, channels: usize) -> Self {
        Self {
            embedding: Array2::zeros((vocab_size, channels)),
            ln_out_weight: Array1::zeros(channels),
            ln_out_bias: Array1::zeros(channels),
            head: Array2::zeros((vocab_size, channels)),
        }
    }

    pub fn zero(&mut self) {
        self.embedding.fill(0.0);
        self.ln_out_weight.fill(0.0);
        self.ln_out_bias.fill(0.0);
        self.head.fill(0.0);
    }

    pub fn parameter_count(&self) -> usize {
        self.embedding.len() + self.ln_out_weight.len() + self.ln_out_bias.len() + self.head.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_model_gradient_storage() {
        let g = ModelGradients::zeros(32, 16);
        assert_eq!(g.embedding.dim(), (32, 16));
        assert_eq!(g.head.dim(), (32, 16));
        assert_eq!(g.parameter_count(), 32 * 16 * 2 + 32);
    }
}
