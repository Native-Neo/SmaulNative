use ndarray::{Array1, Array2};

pub struct GroupNorm {
    pub channels: usize,
    pub groups: usize,
    pub eps: f32,
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
}

impl GroupNorm {
    pub fn new(channels: usize, groups: usize, eps: f32) -> Self {
        assert!(groups > 0);
        assert_eq!(channels % groups, 0);
        Self {
            channels,
            groups,
            eps,
            weight: Array1::ones(channels),
            bias: Array1::zeros(channels),
        }
    }

    pub fn from_weights(
        channels: usize,
        groups: usize,
        eps: f32,
        weight: Array1<f32>,
        bias: Array1<f32>,
    ) -> Self {
        assert_eq!(weight.len(), channels);
        assert_eq!(bias.len(), channels);
        assert!(groups > 0);
        assert_eq!(channels % groups, 0);
        Self { channels, groups, eps, weight, bias }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels);
        let group_size = self.channels / self.groups;
        let mut out = Array2::zeros(x.raw_dim());

        for t in 0..x.nrows() {
            for group in 0..self.groups {
                let start = group * group_size;
                let end = start + group_size;
                let mut mean = 0.0;
                for c in start..end {
                    mean += x[[t, c]];
                }
                mean /= group_size as f32;

                let mut variance = 0.0;
                for c in start..end {
                    let d = x[[t, c]] - mean;
                    variance += d * d;
                }
                variance /= group_size as f32;
                let inv_std = (variance + self.eps).sqrt().recip();

                for c in start..end {
                    out[[t, c]] = (x[[t, c]] - mean) * inv_std * self.weight[c] + self.bias[c];
                }
            }
        }
        out
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests_group_norm {
    use super::*;
    use ndarray::array;

    #[test]
    fn normalizes_each_group() {
        let norm = GroupNorm::new(4, 2, 1e-5);
        let x = array![[1.0, 3.0, 10.0, 14.0]];
        let y = norm.forward(&x);
        assert!((y[[0, 0]] + y[[0, 1]]).abs() < 1e-6);
        assert!((y[[0, 2]] + y[[0, 3]]).abs() < 1e-6);
    }
}


// ===== group_norm_backward =====

pub struct GroupNormBackward {
    pub grad_input: Array2<f32>,
    pub grad_weight: Array1<f32>,
    pub grad_bias: Array1<f32>,
}

pub fn backward(
    input: &Array2<f32>,
    grad_output: &Array2<f32>,
    weight: &Array1<f32>,
    groups: usize,
    eps: f32,
) -> GroupNormBackward {
    assert_eq!(input.dim(), grad_output.dim());
    let (rows, channels) = input.dim();
    assert_eq!(weight.len(), channels);
    assert!(groups > 0 && channels % groups == 0);
    let size = channels / groups;
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_weight = Array1::zeros(channels);
    let mut grad_bias = Array1::zeros(channels);

    for t in 0..rows {
        for g in 0..groups {
            let start = g * size;
            let mut mean = 0.0;
            for c in start..start + size { mean += input[[t, c]]; }
            mean /= size as f32;
            let mut var = 0.0;
            for c in start..start + size { let d = input[[t, c]] - mean; var += d * d; }
            var /= size as f32;
            let inv = (var + eps).sqrt().recip();
            let mut sum_dy = 0.0;
            let mut sum_dy_xhat = 0.0;
            for c in start..start + size {
                let xhat = (input[[t, c]] - mean) * inv;
                let dy = grad_output[[t, c]];
                grad_weight[c] += dy * xhat;
                grad_bias[c] += dy;
                let dyg = dy * weight[c];
                sum_dy += dyg;
                sum_dy_xhat += dyg * xhat;
            }
            for c in start..start + size {
                let xhat = (input[[t, c]] - mean) * inv;
                let dyg = grad_output[[t, c]] * weight[c];
                grad_input[[t, c]] = inv * (dyg - sum_dy / size as f32 - xhat * sum_dy_xhat / size as f32);
            }
        }
    }
    GroupNormBackward { grad_input, grad_weight, grad_bias }
}

#[cfg(test)]
mod tests_group_norm_backward {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn backward_shapes_and_finiteness() {
        let x = Array2::ones((3, 8));
        let g = backward(&x, &x, &Array1::ones(8), 2, 1e-5);
        assert_eq!(g.grad_input.dim(), (3, 8));
        assert_eq!(g.grad_weight.len(), 8);
        assert!(g.grad_input.iter().all(|v| v.is_finite()));
    }
}
