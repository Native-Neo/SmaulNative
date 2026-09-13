use ndarray::{Array1, Array2};

pub struct CmixBackward {
    pub grad_input: Array2<f32>,
    pub grad_prev: Array1<f32>,
    pub grad_x_k: Array1<f32>,
    pub grad_key: Array2<f32>,
    pub grad_value: Array2<f32>,
}

pub fn backward(input: &Array2<f32>, prev: Option<&Array1<f32>>, grad_output: &Array2<f32>, x_k: &Array1<f32>, key_weight: &Array2<f32>, value_weight: &Array2<f32>) -> CmixBackward {
    let (steps, channels) = input.dim();
    assert!(steps > 0);
    assert_eq!(x_k.len(), channels);
    assert_eq!(key_weight.nrows(), channels);
    assert_eq!(value_weight.ncols(), channels);
    assert_eq!(grad_output.dim(), (steps, channels));

    let mut mixed = input.clone();
    for t in 0..steps { for c in 0..channels {
        let p = if t == 0 { prev.map_or(0.0, |v| v[c]) } else { input[[t - 1, c]] };
        mixed[[t, c]] = input[[t, c]] + (p - input[[t, c]]) * x_k[c];
    }}
    let pre = mixed.dot(key_weight);
    let hidden = pre.mapv(|v| v.max(0.0).powi(2));
    let grad_hidden = grad_output.dot(&value_weight.t());
    let mut grad_pre = Array2::zeros(pre.raw_dim());
    for ((dst, &p), &g) in grad_pre.iter_mut().zip(pre.iter()).zip(grad_hidden.iter()) {
        *dst = if p > 0.0 { 2.0 * p * g } else { 0.0 };
    }
    let grad_key = grad_pre.t().dot(&mixed);
    let grad_value = grad_output.t().dot(&hidden);
    let grad_mixed = grad_pre.dot(&key_weight.t());
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_prev = Array1::zeros(channels);
    let mut grad_x_k = Array1::zeros(channels);
    for t in 0..steps { for c in 0..channels {
        let p = if t == 0 { prev.map_or(0.0, |v| v[c]) } else { input[[t - 1, c]] };
        let gm = grad_mixed[[t, c]];
        grad_input[[t, c]] += gm * (1.0 - x_k[c]);
        grad_x_k[c] += gm * (p - input[[t, c]]);
        if t == 0 { if prev.is_some() { grad_prev[c] += gm * x_k[c]; } }
        else { grad_input[[t - 1, c]] += gm * x_k[c]; }
    }}
    CmixBackward { grad_input, grad_prev, grad_x_k, grad_key, grad_value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    #[test]
    fn backward_shapes() {
        let x = Array2::ones((3, 4)); let prev = Array1::zeros(4);
        let key = Array2::ones((4, 16)); let value = Array2::ones((16, 4));
        let g = backward(&x, Some(&prev), &Array2::ones((3, 4)), &Array1::ones(4), &key, &value);
        assert_eq!(g.grad_input.dim(), (3, 4));
        assert_eq!(g.grad_key.dim(), (4, 16));
        assert_eq!(g.grad_value.dim(), (16, 4));
    }
}
