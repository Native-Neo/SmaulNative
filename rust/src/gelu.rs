use ndarray::Array2;

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

pub fn forward(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + 0.044715 * v.powi(3))).tanh()))
}

pub fn backward(x: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(x.dim(), grad_output.dim());
    let mut grad = Array2::<f32>::zeros(x.raw_dim());
    for ((dst, &v), &g) in grad.iter_mut().zip(x.iter()).zip(grad_output.iter()) {
        let inner = SQRT_2_OVER_PI * (v + 0.044715 * v.powi(3));
        let t = inner.tanh();
        let sech2 = 1.0 - t * t;
        let derivative = 0.5 * (1.0 + t)
            + 0.5 * v * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * v * v);
        *dst = g * derivative;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_is_zero_at_zero() {
        assert_eq!(forward(&array![[0.0]])[[0, 0]], 0.0);
    }

    #[test]
    fn backward_matches_finite_difference() {
        let x = array![[0.7]];
        let analytic = backward(&x, &array![[1.0]])[[0, 0]];
        let h = 1e-3;
        let plus = forward(&array![[0.7 + h]])[[0, 0]];
        let minus = forward(&array![[0.7 - h]])[[0, 0]];
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-4);
    }
}
