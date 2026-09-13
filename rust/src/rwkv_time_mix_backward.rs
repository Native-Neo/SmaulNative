use ndarray::{Array1, Array2};

pub struct TimeMixLinearBackward {
    pub grad_input: Array2<f32>,
    pub grad_weight: Array2<f32>,
    pub grad_bias: Array1<f32>,
}

pub fn linear_backward(
    input: &Array2<f32>,
    weight: &Array2<f32>,
    grad_output: &Array2<f32>,
) -> TimeMixLinearBackward {
    assert_eq!(input.ncols(), weight.ncols());
    assert_eq!(grad_output.ncols(), weight.nrows());
    TimeMixLinearBackward {
        grad_input: grad_output.dot(weight),
        grad_weight: grad_output.t().dot(input),
        grad_bias: grad_output.sum_axis(ndarray::Axis(0)),
    }
}

pub fn sigmoid_backward(output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad_output.dim());
    output * &(1.0 - output) * grad_output
}

pub fn tanh_backward(output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad_output.dim());
    (1.0 - output.mapv(|v| v * v)) * grad_output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sigmoid_gradient_is_zero_at_saturation() {
        let output = array![[1.0]];
        let grad = sigmoid_backward(&output, &array![[3.0]]);
        assert_eq!(grad[[0, 0]], 0.0);
    }
}
