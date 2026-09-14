use ndarray::{Array1, Array2};

pub struct Embedding {
    pub weight: Array2<f32>,
    pub grad: Array2<f32>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize, seed: u64) -> Self {
        let scale = (1.0f32 / embedding_dim as f32).sqrt();
        let mut state = seed | 1;
        let mut weight = Array2::<f32>::zeros((vocab_size, embedding_dim));

        for value in weight.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state as f64 / u64::MAX as f64) as f32;
            *value = (unit * 2.0 - 1.0) * scale;
        }

        let grad = Array2::<f32>::zeros((vocab_size, embedding_dim));
        Self { weight, grad }
    }

    pub fn from_weights(weight: Array2<f32>) -> Self {
        let grad = Array2::<f32>::zeros(weight.raw_dim());
        Self { weight, grad }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        let embedding_dim = self.weight.ncols();
        let mut output = Array2::<f32>::zeros((token_ids.len(), embedding_dim));

        for (row, &token_id) in token_ids.iter().enumerate() {
            assert!(token_id < self.weight.nrows(), "token id out of range");
            output.row_mut(row).assign(&self.weight.row(token_id));
        }

        output
    }

    pub fn backward(&mut self, token_ids: &[usize], grad_output: &Array2<f32>) {
        assert_eq!(grad_output.nrows(), token_ids.len());
        assert_eq!(grad_output.ncols(), self.weight.ncols());

        for (row, &token_id) in token_ids.iter().enumerate() {
            assert!(token_id < self.weight.nrows(), "token id out of range");
            let gradient = grad_output.row(row);
            let mut target = self.grad.row_mut(token_id);
            target += &gradient;
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }

    pub fn flatten_weight(&self) -> Array1<f32> {
        Array1::from_iter(self.weight.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::Embedding;
    use ndarray::Array2;

    #[test]
    fn forward_selects_rows() {
        let weight = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let embedding = Embedding::from_weights(weight);
        let output = embedding.forward(&[2, 0]);

        assert_eq!(output[[0, 0]], 5.0);
        assert_eq!(output[[0, 1]], 6.0);
        assert_eq!(output[[1, 0]], 1.0);
        assert_eq!(output[[1, 1]], 2.0);
    }

    #[test]
    fn backward_accumulates_repeated_tokens() {
        let weight = Array2::zeros((2, 2));
        let mut embedding = Embedding::from_weights(weight);
        let grad = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        embedding.backward(&[1, 0, 1], &grad);

        assert_eq!(embedding.grad[[0, 0]], 3.0);
        assert_eq!(embedding.grad[[0, 1]], 4.0);
        assert_eq!(embedding.grad[[1, 0]], 6.0);
        assert_eq!(embedding.grad[[1, 1]], 8.0);
    }
}
