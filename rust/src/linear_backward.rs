use ndarray::Array2;

/// Backpropagates y = x W^T + b for the Linear layout used by SmaulNative.
pub fn backward(
    input: &Array2<f32>,
    weight: &Array2<f32>,
    grad_output: &Array2<f32>,
    has_bias: bool,
) -> (Array2<f32>, Array2<f32>, Option<Array2<f32>>) {
    assert_eq!(input.ncols(), weight.ncols());
    assert_eq!(grad_output.ncols(), weight.nrows());
    assert_eq!(grad_output.nrows(), input.nrows());

    let grad_input = grad_output.dot(weight);
    let grad_weight = grad_output.t().dot(input);
    let grad_bias = if has_bias {
        Some(grad_output.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)))
    } else {
        None
    };
    (grad_input, grad_weight, grad_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradients_have_linear_shapes() {
        let x = array![[1., 2., 3.], [4., 5., 6.]];
        let w = array![[1., 0., 1.], [0., 1., 1.]];
        let g = array![[1., 2.], [3., 4.]];
        let (gx, gw, gb) = backward(&x, &w, &g, true);
        assert_eq!(gx.dim(), x.dim());
        assert_eq!(gw.dim(), w.dim());
        assert_eq!(gb.unwrap().dim(), (1, 2));
    }
}
