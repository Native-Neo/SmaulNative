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
mod tests {
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
