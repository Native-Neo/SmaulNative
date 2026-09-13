use ndarray::Array2;

pub struct BlockBackward {
    pub grad_input: Array2<f32>,
    pub grad_residual: Array2<f32>,
}

pub fn residual_backward(grad_output: &Array2<f32>) -> BlockBackward {
    BlockBackward {
        grad_input: grad_output.clone(),
        grad_residual: grad_output.clone(),
    }
}

pub fn chain_residual(
    grad_branch: &Array2<f32>,
    grad_skip: &Array2<f32>,
) -> Array2<f32> {
    assert_eq!(grad_branch.dim(), grad_skip.dim());
    grad_branch + grad_skip
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn residual_gradient_reaches_both_paths() {
        let grad = array![[1.0, 2.0]];
        let result = residual_backward(&grad);
        assert_eq!(result.grad_input, grad);
        assert_eq!(result.grad_residual, grad);
    }

    #[test]
    fn residual_paths_accumulate() {
        assert_eq!(chain_residual(&array![[1.0]], &array![[2.0]]), array![[3.0]]);
    }
}
