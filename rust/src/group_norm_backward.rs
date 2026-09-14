use ndarray::{Array1, Array2};

pub struct GroupNormBackward {
    pub grad_input: Array2<f32>,
    pub grad_weight: Array1<f32>,
    pub grad_bias: Array1<f32>,
}

pub fn backward(
    input: &Array2<f32>,
    grad_output: &Array2<f32>,
    weight: &Array1<f32>,
    groups: usize,
    eps: f32,
) -> GroupNormBackward {
    assert_eq!(input.dim(), grad_output.dim());
    let (rows, channels) = input.dim();
    assert_eq!(weight.len(), channels);
    assert!(groups > 0 && channels % groups == 0);
    let size = channels / groups;
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_weight = Array1::zeros(channels);
    let mut grad_bias = Array1::zeros(channels);

    for t in 0..rows {
        for g in 0..groups {
            let start = g * size;
            let mut mean = 0.0;
            for c in start..start + size { mean += input[[t, c]]; }
            mean /= size as f32;
            let mut var = 0.0;
            for c in start..start + size { let d = input[[t, c]] - mean; var += d * d; }
            var /= size as f32;
            let inv = (var + eps).sqrt().recip();
            let mut sum_dy = 0.0;
            let mut sum_dy_xhat = 0.0;
            for c in start..start + size {
                let xhat = (input[[t, c]] - mean) * inv;
                let dy = grad_output[[t, c]];
                grad_weight[c] += dy * xhat;
                grad_bias[c] += dy;
                let dyg = dy * weight[c];
                sum_dy += dyg;
                sum_dy_xhat += dyg * xhat;
            }
            for c in start..start + size {
                let xhat = (input[[t, c]] - mean) * inv;
                let dyg = grad_output[[t, c]] * weight[c];
                grad_input[[t, c]] = inv * (dyg - sum_dy / size as f32 - xhat * sum_dy_xhat / size as f32);
            }
        }
    }
    GroupNormBackward { grad_input, grad_weight, grad_bias }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn backward_shapes_and_finiteness() {
        let x = Array2::ones((3, 8));
        let g = backward(&x, &x, &Array1::ones(8), 2, 1e-5);
        assert_eq!(g.grad_input.dim(), (3, 8));
        assert_eq!(g.grad_weight.len(), 8);
        assert!(g.grad_input.iter().all(|v| v.is_finite()));
    }
}
