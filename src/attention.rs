use ndarray::{Array2, Array3, Axis};

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
mod tests {
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
