use ndarray::{Array2, Axis};

pub struct AttentionBackward {
    pub grad_q: Array2<f32>,
    pub grad_k: Array2<f32>,
    pub grad_v: Array2<f32>,
}

pub fn backward(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    grad_output: &Array2<f32>,
    scale: f32,
) -> AttentionBackward {
    assert_eq!(q.ncols(), k.ncols());
    assert_eq!(k.nrows(), v.nrows());
    assert_eq!(q.nrows(), grad_output.nrows());
    assert_eq!(v.ncols(), grad_output.ncols());

    let scores = q.dot(&k.t()) * scale;
    let mut probs = Array2::<f32>::zeros(scores.raw_dim());
    for (mut dst, src) in probs.axis_iter_mut(Axis(0)).zip(scores.axis_iter(Axis(0))) {
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = (s - max).exp();
            sum += *d;
        }
        for d in dst.iter_mut() { *d /= sum; }
    }

    let grad_v = probs.t().dot(grad_output);
    let grad_probs = grad_output.dot(&v.t());
    let mut grad_scores = Array2::<f32>::zeros(probs.raw_dim());
    for i in 0..probs.nrows() {
        let dot: f32 = probs.row(i).iter().zip(grad_probs.row(i).iter()).map(|(p,g)| p*g).sum();
        for j in 0..probs.ncols() {
            grad_scores[[i,j]] = probs[[i,j]] * (grad_probs[[i,j]] - dot);
        }
    }
    let grad_q = grad_scores.dot(k) * scale;
    let grad_k = grad_scores.t().dot(q) * scale;
    AttentionBackward { grad_q, grad_k, grad_v }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradients_have_expected_shapes() {
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let k = q.clone();
        let v = array![[1.0, 2.0], [3.0, 4.0]];
        let g = backward(&q, &k, &v, &v, 0.5);
        assert_eq!(g.grad_q.dim(), q.dim());
        assert_eq!(g.grad_k.dim(), k.dim());
        assert_eq!(g.grad_v.dim(), v.dim());
    }
}
