use ndarray::{Array2, Axis};

pub struct FeedForward {
    pub d_model: usize,
    pub d_ff: usize,
    pub up_weight: Array2<f32>,
    pub up_bias: Array2<f32>,
    pub down_weight: Array2<f32>,
    pub down_bias: Array2<f32>,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            d_model,
            d_ff,
            up_weight: random_matrix(d_model, d_ff, seed ^ 0x101),
            up_bias: Array2::zeros((1, d_ff)),
            down_weight: random_matrix(d_ff, d_model, seed ^ 0x202),
            down_bias: Array2::zeros((1, d_model)),
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.d_model);

        let mut hidden = input.dot(&self.up_weight);
        for mut row in hidden.axis_iter_mut(Axis(0)) {
            row += &self.up_bias.row(0);
            for value in row.iter_mut() {
                *value = gelu(*value);
            }
        }

        let mut output = hidden.dot(&self.down_weight);
        for mut row in output.axis_iter_mut(Axis(0)) {
            row += &self.down_bias.row(0);
        }
        output
    }
}

fn gelu(x: f32) -> f32 {
    let coefficient = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (coefficient * (x + 0.044715 * x.powi(3))).tanh())
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let scale = (1.0f32 / rows as f32).sqrt();
    let mut state = seed | 1;
    let mut matrix = Array2::<f32>::zeros((rows, cols));

    for value in matrix.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state as f64 / u64::MAX as f64) as f32;
        *value = (unit * 2.0 - 1.0) * scale;
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::FeedForward;
    use ndarray::Array2;

    #[test]
    fn output_shape_is_preserved() {
        let ffn = FeedForward::new(8, 32, 123);
        let input = Array2::<f32>::ones((4, 8));
        let output = ffn.forward(&input);
        assert_eq!(output.dim(), (4, 8));
    }

    #[test]
    fn output_is_finite() {
        let ffn = FeedForward::new(8, 32, 123);
        let input = Array2::<f32>::ones((4, 8));
        let output = ffn.forward(&input);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
