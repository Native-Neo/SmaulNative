use ndarray::Array2;

pub fn relu(x: &Array2<f32>) -> Array2<f32> { x.mapv(|v| v.max(0.0)) }

pub fn relu_backward(x: &Array2<f32>, grad: &Array2<f32>) -> Array2<f32> {
    assert_eq!(x.dim(), grad.dim());
    Array2::from_shape_fn(x.dim(), |idx| if x[idx] > 0.0 { grad[idx] } else { 0.0 })
}

pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn sigmoid_backward(output: &Array2<f32>, grad: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad.dim());
    output * &(1.0 - output) * grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_zeroes_negative_values() {
        let x = Array2::from_shape_vec((1, 3), vec![-1., 0., 2.]).unwrap();
        assert_eq!(relu(&x), Array2::from_shape_vec((1, 3), vec![0., 0., 2.]).unwrap());
    }

    #[test]
    fn sigmoid_stays_in_unit_interval() {
        let x = sigmoid(&Array2::from_shape_vec((1, 3), vec![-20., 0., 20.]).unwrap());
        assert!(x.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }
}
