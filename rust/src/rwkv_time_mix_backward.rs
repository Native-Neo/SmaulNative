use ndarray::{Array1, Array2};

pub struct TimeMixLinearBackward { pub grad_input: Array2<f32>, pub grad_weight: Array2<f32>, pub grad_bias: Array1<f32> }
pub struct MixBackward { pub grad_input: Array2<f32>, pub grad_prev: Array1<f32>, pub grad_factor: Array1<f32> }

pub fn linear_backward(input: &Array2<f32>, weight: &Array2<f32>, grad_output: &Array2<f32>) -> TimeMixLinearBackward {
    assert_eq!(input.ncols(), weight.ncols());
    assert_eq!(grad_output.dim(), (input.nrows(), weight.nrows()));
    TimeMixLinearBackward { grad_input: grad_output.dot(weight), grad_weight: grad_output.t().dot(input), grad_bias: grad_output.sum_axis(ndarray::Axis(0)) }
}

pub fn mix_backward(input: &Array2<f32>, prev: &Array1<f32>, factor: &Array1<f32>, grad_output: &Array2<f32>) -> MixBackward {
    assert_eq!(input.ncols(), factor.len());
    assert_eq!(input.dim(), grad_output.dim());
    assert_eq!(prev.len(), input.ncols());
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_prev = Array1::zeros(prev.raw_dim());
    let mut grad_factor = Array1::zeros(factor.raw_dim());
    for t in 0..input.nrows() { for c in 0..input.ncols() {
        let p = if t == 0 { prev[c] } else { input[[t - 1, c]] };
        let g = grad_output[[t, c]];
        grad_input[[t, c]] += g * (1.0 - factor[c]);
        grad_factor[c] += g * (p - input[[t, c]]);
        if t == 0 { grad_prev[c] += g * factor[c]; } else { grad_input[[t - 1, c]] += g * factor[c]; }
    }}
    MixBackward { grad_input, grad_prev, grad_factor }
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
    use ndarray::{Array1, Array2};
    #[test]
    fn mix_backward_propagates_previous_token() {
        let x = Array2::ones((2, 3)); let p = Array1::zeros(3); let f = Array1::from_elem(3, 0.5);
        let g = mix_backward(&x, &p, &f, &Array2::ones((2, 3)));
        assert_eq!(g.grad_input.dim(), (2, 3));
        assert_eq!(g.grad_prev.len(), 3);
        assert_eq!(g.grad_factor.len(), 3);
    }
}
