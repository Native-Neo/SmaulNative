use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct GradientCollection {
    pub embedding: Array2<f32>,
    pub head: Array2<f32>,
    pub ln_weight: Array1<f32>,
    pub ln_bias: Array1<f32>,
}

impl GradientCollection {
    pub fn zeros(vocab_size: usize, channels: usize) -> Self {
        assert!(vocab_size > 0 && channels > 0);
        Self {
            embedding: Array2::zeros((vocab_size, channels)),
            head: Array2::zeros((vocab_size, channels)),
            ln_weight: Array1::zeros(channels),
            ln_bias: Array1::zeros(channels),
        }
    }

    pub fn add(&mut self, other: &Self) {
        assert_eq!(self.embedding.dim(), other.embedding.dim());
        assert_eq!(self.head.dim(), other.head.dim());
        assert_eq!(self.ln_weight.len(), other.ln_weight.len());
        assert_eq!(self.ln_bias.len(), other.ln_bias.len());
        self.embedding += &other.embedding;
        self.head += &other.head;
        self.ln_weight += &other.ln_weight;
        self.ln_bias += &other.ln_bias;
    }

    pub fn scale(&mut self, factor: f32) {
        assert!(factor.is_finite());
        self.embedding *= factor;
        self.head *= factor;
        self.ln_weight *= factor;
        self.ln_bias *= factor;
    }

    pub fn clear(&mut self) {
        self.embedding.fill(0.0);
        self.head.fill(0.0);
        self.ln_weight.fill(0.0);
        self.ln_bias.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulation_is_elementwise() {
        let mut a = GradientCollection::zeros(2, 3);
        let mut b = GradientCollection::zeros(2, 3);
        a.head.fill(1.0);
        b.head.fill(2.0);
        a.add(&b);
        assert_eq!(a.head[[1, 2]], 3.0);
    }
}
