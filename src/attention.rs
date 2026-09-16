use ndarray::{Array2, Array3, Axis, s};

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub q_weight: Array2<f32>,
    pub k_weight: Array2<f32>,
    pub v_weight: Array2<f32>,
    pub out_weight: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        assert!(n_heads > 0, "number of heads must be greater than zero");
        assert_eq!(d_model % n_heads, 0, "d_model must divide evenly across heads");

        let head_dim = d_model / n_heads;
        Self {
            d_model,
            n_heads,
            head_dim,
            q_weight: random_matrix(d_model, d_model, seed ^ 0x11),
            k_weight: random_matrix(d_model, d_model, seed ^ 0x22),
            v_weight: random_matrix(d_model, d_model, seed ^ 0x33),
            out_weight: random_matrix(d_model, d_model, seed ^ 0x44),
        }
    }

    pub fn forward(&self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        assert_eq!(input.ncols(), self.d_model);
        let seq_len = input.nrows();

        let q = input.dot(&self.q_weight);
        let k = input.dot(&self.k_weight);
        let v = input.dot(&self.v_weight);
        let mut heads = Array3::<f32>::zeros((self.n_heads, seq_len, self.head_dim));

        for head in 0..self.n_heads {
            let start = head * self.head_dim;
            let end = start + self.head_dim;
            let q_head = q.slice(ndarray::s![.., start..end]);
            let k_head = k.slice(ndarray::s![.., start..end]);
            let v_head = v.slice(ndarray::s![.., start..end]);
            let scale = (self.head_dim as f32).sqrt();
            let mut scores = q_head.dot(&k_head.t()) / scale;

            if causal {
                for row in 0..seq_len {
                    for col in (row + 1)..seq_len {
                        scores[[row, col]] = f32::NEG_INFINITY;
                    }
                }
            }

            softmax_rows(&mut scores);
            let output = scores.dot(&v_head);
            heads.index_axis_mut(Axis(0), head).assign(&output);
        }

        let mut merged = Array2::<f32>::zeros((seq_len, self.d_model));
        for head in 0..self.n_heads {
            let start = head * self.head_dim;
            let end = start + self.head_dim;
            merged
                .slice_mut(ndarray::s![.., start..end])
                .assign(&heads.index_axis(Axis(0), head));
        }

        merged.dot(&self.out_weight)
    }
}

fn softmax_rows(values: &mut Array2<f32>) {
    for mut row in values.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum != 0.0 {
            for value in row.iter_mut() {
                *value /= sum;
            }
        }
    }
}

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let scale = (1.0f32 / cols as f32).sqrt();
    let mut state = seed | 1;
    let mut matrix = Array2::<f32>::zeros((rows, cols));

    for value in matrix.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state as f64 / u64::MAX as f64) as f32;
        *value = (unit * 2.0 - 1.0) * scale;
    }

    matrix
}

#[cfg(test)]
mod tests_attention {
    use super::MultiHeadAttention;
    use ndarray::Array2;

    #[test]
    fn output_shape_is_preserved() {
        let attention = MultiHeadAttention::new(8, 2, 123);
        let input = Array2::<f32>::ones((4, 8));
        let output = attention.forward(&input, true);
        assert_eq!(output.dim(), (4, 8));
    }

    #[test]
    fn causal_attention_is_finite() {
        let attention = MultiHeadAttention::new(8, 2, 123);
        let input = Array2::<f32>::ones((4, 8));
        let output = attention.forward(&input, true);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}


// ===== attention_backward =====

pub struct AttentionGrads {
    pub grad_input: Array2<f32>,
    pub grad_q_weight: Array2<f32>,
    pub grad_k_weight: Array2<f32>,
    pub grad_v_weight: Array2<f32>,
    pub grad_out_weight: Array2<f32>,
}

pub fn backward(
    input: &Array2<f32>,
    grad_output: &Array2<f32>,
    q_weight: &Array2<f32>,
    k_weight: &Array2<f32>,
    v_weight: &Array2<f32>,
    out_weight: &Array2<f32>,
    n_heads: usize,
    causal: bool,
) -> AttentionGrads {
    let (seq_len, d_model) = input.dim();
    assert_eq!(grad_output.dim(), (seq_len, d_model));
    assert_eq!(q_weight.dim(), (d_model, d_model));
    assert_eq!(k_weight.dim(), (d_model, d_model));
    assert_eq!(v_weight.dim(), (d_model, d_model));
    assert_eq!(out_weight.dim(), (d_model, d_model));
    assert!(n_heads > 0 && d_model % n_heads == 0);

    let head_dim = d_model / n_heads;
    let scale = (head_dim as f32).sqrt();
    let q = input.dot(q_weight);
    let k = input.dot(k_weight);
    let v = input.dot(v_weight);
    let mut merged = Array2::<f32>::zeros((seq_len, d_model));
    let mut probabilities = Vec::with_capacity(n_heads);

    for head in 0..n_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let qh = q.slice(s![.., start..end]);
        let kh = k.slice(s![.., start..end]);
        let vh = v.slice(s![.., start..end]);
        let mut scores = qh.dot(&kh.t()) / scale;
        if causal {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
            }
        }
        softmax_rows(&mut scores);
        merged.slice_mut(s![.., start..end]).assign(&scores.dot(&vh));
        probabilities.push(scores);
    }

    let grad_out_weight = merged.t().dot(grad_output);
    let grad_merged = grad_output.dot(&out_weight.t());
    let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
    let mut grad_q_weight = Array2::<f32>::zeros(q_weight.raw_dim());
    let mut grad_k_weight = Array2::<f32>::zeros(k_weight.raw_dim());
    let mut grad_v_weight = Array2::<f32>::zeros(v_weight.raw_dim());

    for head in 0..n_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let qh = q.slice(s![.., start..end]);
        let kh = k.slice(s![.., start..end]);
        let vh = v.slice(s![.., start..end]);
        let dh = grad_merged.slice(s![.., start..end]);
        let a = &probabilities[head];

        let grad_v = a.t().dot(&dh);
        let grad_a = dh.dot(&vh.t());
        let mut grad_scores = Array2::<f32>::zeros((seq_len, seq_len));
        for i in 0..seq_len {
            let mut weighted = 0.0;
            for j in 0..seq_len {
                weighted += grad_a[[i, j]] * a[[i, j]];
            }
            for j in 0..seq_len {
                grad_scores[[i, j]] = a[[i, j]] * (grad_a[[i, j]] - weighted) / scale;
            }
        }

        let grad_q = grad_scores.dot(&kh);
        let grad_k = grad_scores.t().dot(&qh);
        let mut grad_q_full = Array2::<f32>::zeros((seq_len, d_model));
        let mut grad_k_full = Array2::<f32>::zeros((seq_len, d_model));
        let mut grad_v_full = Array2::<f32>::zeros((seq_len, d_model));
        grad_q_full.slice_mut(s![.., start..end]).assign(&grad_q);
        grad_k_full.slice_mut(s![.., start..end]).assign(&grad_k);
        grad_v_full.slice_mut(s![.., start..end]).assign(&grad_v);

        grad_input += &grad_q_full.dot(&q_weight.t());
        grad_input += &grad_k_full.dot(&k_weight.t());
        grad_input += &grad_v_full.dot(&v_weight.t());
        grad_q_weight += &input.t().dot(&grad_q_full);
        grad_k_weight += &input.t().dot(&grad_k_full);
        grad_v_weight += &input.t().dot(&grad_v_full);
    }

    AttentionGrads { grad_input, grad_q_weight, grad_k_weight, grad_v_weight, grad_out_weight }
}


#[cfg(test)]
mod tests_attention_backward {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn backward_returns_finite_gradients() {
        let input = Array2::from_elem((3, 4), 0.2);
        let weights = Array2::from_diag(&ndarray::Array1::ones(4));
        let dy = Array2::ones((3, 4));
        let g = backward(&input, &dy, &weights, &weights, &weights, &weights, 2, true);
        assert!(g.grad_input.iter().all(|v| v.is_finite()));
        assert!(g.grad_q_weight.iter().all(|v| v.is_finite()));
    }
}
