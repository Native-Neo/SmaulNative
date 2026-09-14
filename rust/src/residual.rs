use ndarray::Array2;

pub fn add_forward(x: &Array2<f32>, residual: &Array2<f32>) -> Array2<f32> {
    assert_eq!(x.dim(), residual.dim());
    x + residual
}

pub fn add_backward(grad_output: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    (grad_output.clone(), grad_output.clone())
}

pub fn scaled_add_backward(grad_output: &Array2<f32>, scale: f32) -> (Array2<f32>, Array2<f32>) {
    (grad_output.clone(), grad_output * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn residual_gradients_are_identity() {
        let g = array![[1.0, -2.0]];
        let (a,b) = add_backward(&g);
        assert_eq!(a,g); assert_eq!(b,Array2::from_shape_vec((1,2),vec![1.0,-2.0]).unwrap());
    }
}
