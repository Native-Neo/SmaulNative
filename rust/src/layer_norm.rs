use ndarray::{Array1, Array2, Axis};

pub struct LayerNorm {
    pub normalized_dim: usize,
    pub eps: f32,
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
}

impl LayerNorm {
    pub fn new(normalized_dim: usize, eps: f32) -> Self {
        Self {
            normalized_dim,
            eps,
            weight: Array1::ones(normalized_dim),
            bias: Array1::zeros(normalized_dim),
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.normalized_dim);
        let mut output = input.clone();

        for mut row in output.axis_iter_mut(Axis(0)) {
            let mean = row.sum() / self.normalized_dim as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let delta = *value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / self.normalized_dim as f32;
            let inv_std = (variance + self.eps).sqrt().recip();

            for index in 0..self.normalized_dim {
                row[index] = (row[index] - mean) * inv_std * self.weight[index] + self.bias[index];
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::LayerNorm;
    use ndarray::Array2;

    #[test]
    fn output_shape_is_preserved() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        )
        .unwrap();
        assert_eq!(norm.forward(&input).dim(), input.dim());
    }

    #[test]
    fn constant_input_is_finite() {
        let norm = LayerNorm::new(4, 1e-5);
        let input = Array2::<f32>::ones((2, 4));
        let output = norm.forward(&input);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
