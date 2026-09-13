use ndarray::{Array1, Array2, Axis};

pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
    pub grad_weight: Array2<f32>,
    pub grad_bias: Array1<f32>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0);
        assert!(out_features > 0);

        let scale = (2.0 / in_features as f32).sqrt();
        let mut weight = Array2::<f32>::zeros((out_features, in_features));
        for o in 0..out_features {
            for i in 0..in_features {
                let value = (((o * in_features + i) * 1103515245 + 12345) % 100000) as f32;
                weight[[o, i]] = (value / 50000.0 - 1.0) * scale;
            }
        }

        Self {
            weight,
            bias: Array1::zeros(out_features),
            grad_weight: Array2::zeros((out_features, in_features)),
            grad_bias: Array1::zeros(out_features),
        }
    }

    pub fn from_weights(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(weight.nrows(), bias.len());
        Self {
            grad_weight: Array2::zeros(weight.raw_dim()),
            grad_bias: Array1::zeros(bias.raw_dim()),
            weight,
            bias,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.weight.ncols());
        let mut output = input.dot(&self.weight.t());
        output += &self.bias;
        output
    }

    pub fn backward(&mut self, input: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.weight.ncols());
        assert_eq!(grad_output.nrows(), input.nrows());
        assert_eq!(grad_output.ncols(), self.weight.nrows());

        self.grad_weight += &grad_output.t().dot(input);
        self.grad_bias += &grad_output.sum_axis(Axis(0));
        grad_output.dot(&self.weight)
    }

    pub fn zero_grad(&mut self) {
        self.grad_weight.fill(0.0);
        self.grad_bias.fill(0.0);
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_matches_matrix_multiplication() {
        let layer = Linear::from_weights(
            array![[1.0, 2.0], [3.0, 4.0]],
            array![0.5, -0.5],
        );
        let input = array![[2.0, 3.0]];
        let output = layer.forward(&input);
        assert_eq!(output, array![[8.5, 16.5]]);
    }

    #[test]
    fn backward_accumulates_gradients() {
        let mut layer = Linear::from_weights(
            array![[1.0, 2.0], [3.0, 4.0]],
            array![0.0, 0.0],
        );
        let input = array![[2.0, 3.0]];
        let grad = array![[5.0, 7.0]];
        let grad_input = layer.backward(&input, &grad);

        assert_eq!(layer.grad_weight, array![[10.0, 15.0], [14.0, 21.0]]);
        assert_eq!(layer.grad_bias, array![5.0, 7.0]);
        assert_eq!(grad_input, array![[26.0, 38.0]]);
    }

    #[test]
    fn zero_grad_clears_all_gradients() {
        let mut layer = Linear::new(2, 3);
        layer.grad_weight.fill(1.0);
        layer.grad_bias.fill(1.0);
        layer.zero_grad();
        assert!(layer.grad_weight.iter().all(|value| *value == 0.0));
        assert!(layer.grad_bias.iter().all(|value| *value == 0.0));
    }
}
