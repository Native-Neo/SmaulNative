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
mod tests_layer_norm {
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


// ===== layer_norm_backward =====

pub fn backward(
    input: &Array2<f32>,
    grad_output: &Array2<f32>,
    gamma: &Array1<f32>,
    eps: f32,
) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    assert_eq!(input.dim(), grad_output.dim());
    assert_eq!(input.ncols(), gamma.len());

    let rows = input.nrows();
    let cols = input.ncols();
    let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
    let mut grad_gamma = Array1::<f32>::zeros(cols);
    let mut grad_beta = Array1::<f32>::zeros(cols);

    for row in 0..rows {
        let x = input.row(row);
        let dy = grad_output.row(row);
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.iter().map(|v| {
            let d = *v - mean;
            d * d
        }).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        let mut normalized = Array1::<f32>::zeros(cols);
        for i in 0..cols {
            normalized[i] = (x[i] - mean) * inv_std;
            grad_gamma[i] += dy[i] * normalized[i];
            grad_beta[i] += dy[i];
        }

        let sum_dy_gamma: f32 = (0..cols).map(|i| dy[i] * gamma[i]).sum();
        let sum_dy_gamma_xhat: f32 = (0..cols)
            .map(|i| dy[i] * gamma[i] * normalized[i])
            .sum();

        for i in 0..cols {
            let dyg = dy[i] * gamma[i];
            grad_input[[row, i]] = inv_std
                * (dyg - sum_dy_gamma / cols as f32
                    - normalized[i] * sum_dy_gamma_xhat / cols as f32);
        }
    }

    let _ = Axis(0);
    (grad_input, grad_gamma, grad_beta)
}

#[cfg(test)]
mod tests_layer_norm_backward {
    use super::*;
    use ndarray::array;

    #[test]
    fn backward_shapes_match_layer_norm() {
        let x = array![[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]];
        let dy = Array2::ones((2, 3));
        let gamma = Array1::ones(3);
        let (dx, dg, db) = backward(&x, &dy, &gamma, 1e-5);
        assert_eq!(dx.dim(), x.dim());
        assert_eq!(dg.len(), 3);
        assert_eq!(db.len(), 3);
    }

    #[test]
    fn constant_gradient_has_zero_input_sum_per_row() {
        let x = array![[1.0, 2.0, 4.0]];
        let dy = array![[1.0, 1.0, 1.0]];
        let gamma = Array1::ones(3);
        let (dx, _, _) = backward(&x, &dy, &gamma, 1e-5);
        assert!(dx.sum().abs() < 1e-5);
    }
}
