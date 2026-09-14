use ndarray::Array2;

/// Backpropagates through y = x + residual.
pub fn backward(grad_output: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    (grad_output.clone(), grad_output.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn both_paths_receive_the_same_gradient() {
        let g = array![[1.0, -2.0], [3.0, 4.0]];
        let (dx, dr) = backward(&g);
        assert_eq!(dx, g);
        assert_eq!(dr, g);
    }
}
