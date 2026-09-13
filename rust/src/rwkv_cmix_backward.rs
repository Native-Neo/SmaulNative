use ndarray::{Array2, Axis};

pub struct CmixBackward {
    pub grad_input: Array2<f32>,
    pub grad_key: Array2<f32>,
    pub grad_value: Array2<f32>,
}

pub fn backward(
    input: &Array2<f32>,
    hidden: &Array2<f32>,
    grad_output: &Array2<f32>,
    key_weight: &Array2<f32>,
    value_weight: &Array2<f32>,
) -> CmixBackward {
    assert_eq!(input.ncols(), key_weight.ncols());
    assert_eq!(hidden.ncols(), key_weight.nrows());
    assert_eq!(hidden.ncols(), value_weight.nrows());
    assert_eq!(grad_output.ncols(), value_weight.ncols());
    let grad_hidden = grad_output.dot(&value_weight.t());
    let grad_value = grad_output.t().dot(hidden);
    let grad_pre = hidden.mapv(|v| if v > 0.0 { 2.0 * v } else { 0.0 }) * &grad_hidden;
    let grad_key = grad_pre.t().dot(input);
    let grad_input = grad_pre.dot(key_weight);
    CmixBackward { grad_input, grad_key, grad_value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn backward_shapes_match_cmix() {
        let input = array![[1.0, -2.0]];
        let hidden = array![[0.5, 0.0, 1.0]];
        let grad = backward(
            &input,
            &hidden,
            &array![[1.0, 2.0]],
            &array![[1.0, 2.0], [0.5, 1.0], [-1.0, 1.0]],
            &array![[1.0, 0.0], [0.0, 1.0]],
        );
        assert_eq!(grad.grad_input.dim(), input.dim());
        assert_eq!(grad.grad_key.dim(), (3, 2));
        assert_eq!(grad.grad_value.dim(), (2, 2));
    }
}
