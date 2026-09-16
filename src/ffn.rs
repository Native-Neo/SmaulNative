use crate::gelu;
use ndarray::{Array1, Array2, Axis};

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
mod tests_ffn {
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


// ===== ffn_backward =====


pub struct FfnGrads {
    pub grad_input: Array2<f32>,
    pub grad_up_weight: Array2<f32>,
    pub grad_up_bias: Array1<f32>,
    pub grad_down_weight: Array2<f32>,
    pub grad_down_bias: Array1<f32>,
}

pub fn backward(
    input: &Array2<f32>,
    grad_output: &Array2<f32>,
    up_weight: &Array2<f32>,
    up_bias: &Array1<f32>,
    down_weight: &Array2<f32>,
) -> FfnGrads {
    assert_eq!(input.ncols(), up_weight.nrows());
    assert_eq!(up_bias.len(), up_weight.ncols());
    assert_eq!(down_weight.nrows(), up_weight.ncols());
    assert_eq!(grad_output.ncols(), down_weight.ncols());

    let mut pre_activation = input.dot(up_weight);
    for mut row in pre_activation.axis_iter_mut(Axis(0)) {
        row += up_bias;
    }
    let hidden = gelu::forward(&pre_activation);

    let grad_down_weight = hidden.t().dot(grad_output);
    let grad_down_bias = grad_output.sum_axis(Axis(0));
    let grad_hidden = grad_output.dot(&down_weight.t());
    let grad_pre_activation = gelu::backward(&pre_activation, &grad_hidden);
    let grad_up_weight = input.t().dot(&grad_pre_activation);
    let grad_up_bias = grad_pre_activation.sum_axis(Axis(0));
    let grad_input = grad_pre_activation.dot(&up_weight.t());

    FfnGrads {
        grad_input,
        grad_up_weight,
        grad_up_bias,
        grad_down_weight,
        grad_down_bias,
    }
}

#[cfg(test)]
mod tests_ffn_backward {
    use super::*;
    use ndarray::array;

    #[test]
    fn backward_shapes_match_forward() {
        let input = array![[1.0, 2.0], [0.5, -1.0]];
        let up = array![[0.2, 0.3, -0.1], [0.4, -0.2, 0.5]];
        let up_bias = array![0.1, -0.1, 0.2];
        let down = array![[0.3, 0.2], [-0.4, 0.1], [0.5, -0.3]];
        let grad_output = Array2::ones((2, 2));
        let g = backward(&input, &grad_output, &up, &up_bias, &down);
        assert_eq!(g.grad_input.dim(), input.dim());
        assert_eq!(g.grad_up_weight.dim(), up.dim());
        assert_eq!(g.grad_down_weight.dim(), down.dim());
        assert_eq!(g.grad_up_bias.len(), 3);
        assert_eq!(g.grad_down_bias.len(), 2);
    }

    #[test]
    fn backward_is_finite() {
        let input = Array2::from_elem((3, 4), 0.25);
        let up = Array2::from_elem((4, 8), 0.1);
        let bias = Array1::zeros(8);
        let down = Array2::from_elem((8, 4), 0.1);
        let output_grad = Array2::ones((3, 4));
        let g = backward(&input, &output_grad, &up, &bias, &down);
        assert!(g.grad_input.iter().all(|v| v.is_finite()));
        assert!(g.grad_up_weight.iter().all(|v| v.is_finite()));
        assert!(g.grad_down_weight.iter().all(|v| v.is_finite()));
    }
}
