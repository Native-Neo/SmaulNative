use ndarray::Array2;

pub struct MobaBackward {
    pub grad_q: Array2<f32>,
    pub grad_k: Array2<f32>,
    pub grad_v: Array2<f32>,
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in &mut out { *v /= sum; }
    out
}

pub fn backward(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, grad_output: &Array2<f32>, chunk_size: usize, top_k: usize, scale: f32) -> MobaBackward {
    let (tokens, dim) = q.dim();
    assert_eq!(k.nrows(), tokens);
    assert_eq!(k.ncols(), dim);
    assert_eq!(v.nrows(), tokens);
    assert_eq!(grad_output.nrows(), tokens);
    assert_eq!(grad_output.ncols(), v.ncols());
    assert!(chunk_size > 0);

    let chunks = (tokens + chunk_size - 1) / chunk_size;
    let mut means = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(tokens);
        let mut mean = vec![0.0; dim];
        for t in start..end { for d in 0..dim { mean[d] += k[[t, d]]; } }
        let inv = 1.0 / (end - start) as f32;
        for x in &mut mean { *x *= inv; }
        means.push(mean);
    }

    let mut gq = Array2::zeros(q.raw_dim());
    let mut gk = Array2::zeros(k.raw_dim());
    let mut gv = Array2::zeros(v.raw_dim());

    for t in 0..tokens {
        let current = t / chunk_size;
        let cur_start = current * chunk_size;
        let mut selected = Vec::new();
        if current > 0 {
            let mut ranked: Vec<(f32, usize)> = (0..current).map(|c| {
                let score: f32 = (0..dim).map(|d| q[[t, d]] * means[c][d]).sum();
                (score, c)
            }).collect();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, c) in ranked.iter().take(top_k.min(ranked.len())) { selected.push((c * chunk_size, ((c + 1) * chunk_size).min(tokens))); }
        }
        let current_end = (t + 1).min((current + 1) * chunk_size).min(tokens);
        selected.push((cur_start, current_end));

        let mut indices = Vec::new();
        for (start, end) in selected { for j in start..end { if j <= t || start != cur_start { indices.push(j); } } }
        indices.sort_unstable();
        indices.dedup();

        let mut scores = Vec::with_capacity(indices.len());
        for &j in &indices { let mut s = 0.0; for d in 0..dim { s += q[[t, d]] * k[[j, d]]; } scores.push(s * scale); }
        let probs = softmax(&scores);
        let mut grad_probs = vec![0.0; indices.len()];
        for (p, &j) in probs.iter().zip(&indices) { for d in 0..v.ncols() { grad_probs[indices.iter().position(|&x| x == j).unwrap()] += grad_output[[t, d]] * v[[j, d]]; } let _ = p; }
        let dot: f32 = probs.iter().zip(&grad_probs).map(|(p, g)| p * g).sum();
        let mut grad_scores = vec![0.0; indices.len()];
        for i in 0..indices.len() { grad_scores[i] = probs[i] * (grad_probs[i] - dot); }

        for i in 0..indices.len() {
            let j = indices[i];
            for d in 0..dim {
                gq[[t, d]] += grad_scores[i] * k[[j, d]] * scale;
                gk[[j, d]] += grad_scores[i] * q[[t, d]] * scale;
                gv[[j, d]] += probs[i] * grad_output[[t, d]];
            }
        }
    }
    MobaBackward { grad_q: gq, grad_k: gk, grad_v: gv }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    #[test]
    fn routed_backward_has_finite_gradients() {
        let q = Array2::ones((4, 3));
        let k = Array2::ones((4, 3));
        let v = Array2::ones((4, 3));
        let g = backward(&q, &k, &v, &v, 2, 1, 0.5);
        assert_eq!(g.grad_q.dim(), (4, 3));
        assert!(g.grad_q.iter().all(|x| x.is_finite()));
    }
}
