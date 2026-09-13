use crate::linear::Linear;
use ndarray::{Array2, Array4};

pub struct MobaAttention {
    pub channels: usize,
    pub heads: usize,
    pub head_size: usize,
    pub chunk_size: usize,
    pub top_k: usize,
    pub receptance: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
}

impl MobaAttention {
    pub fn new(channels: usize, head_size: usize, chunk_size: usize, top_k: usize) -> Self {
        assert!(channels > 0 && head_size > 0);
        assert_eq!(channels % head_size, 0);
        assert!(chunk_size > 0);
        Self {
            channels,
            heads: channels / head_size,
            head_size,
            chunk_size,
            top_k,
            receptance: Linear::new(channels, channels),
            key: Linear::new(channels, channels),
            value: Linear::new(channels, channels),
            output: Linear::new(channels, channels),
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn softmax(scores: &mut [f32]) {
        if scores.is_empty() { return; }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in scores.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in scores.iter_mut() { *x /= sum; }
        }
    }

    fn attend(
        q: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        scale: f32,
        causal_len: usize,
    ) -> Vec<f32> {
        let mut scores = Vec::with_capacity(keys.len());
        for key in keys.iter() {
            scores.push(Self::dot(q, key) * scale);
        }
        Self::softmax(&mut scores);
        let mut out = vec![0.0; q.len()];
        for (weight, value) in scores.iter().zip(values.iter()).take(causal_len) {
            for d in 0..out.len() { out[d] += weight * value[d]; }
        }
        out
    }

    fn project_heads(&self, x: &Array2<f32>, projection: &Linear) -> Vec<Vec<Vec<f32>>> {
        let projected = projection.forward(x);
        let chunks = (x.nrows() + self.chunk_size - 1) / self.chunk_size;
        let mut result = vec![vec![vec![0.0; self.head_size]; x.nrows()]; self.heads];
        let _ = chunks;
        for h in 0..self.heads {
            for t in 0..x.nrows() {
                for d in 0..self.head_size {
                    result[h][t][d] = projected[[t, h * self.head_size + d]];
                }
            }
        }
        result
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels);
        let t = x.nrows();
        let q = self.project_heads(x, &self.receptance);
        let k = self.project_heads(x, &self.key);
        let v = self.project_heads(x, &self.value);
        let chunks = (t + self.chunk_size - 1) / self.chunk_size;
        let scale = (self.head_size as f32).sqrt().recip();
        let mut out = Array2::<f32>::zeros((t, self.channels));

        for h in 0..self.heads {
            let mut means = vec![vec![0.0; self.head_size]; chunks];
            for c in 0..chunks {
                let lo = c * self.chunk_size;
                let hi = usize::min(lo + self.chunk_size, t);
                for row in lo..hi {
                    for d in 0..self.head_size { means[c][d] += k[h][row][d]; }
                }
                let n = (hi - lo) as f32;
                for d in 0..self.head_size { means[c][d] /= n; }
            }

            for row in 0..t {
                let chunk = row / self.chunk_size;
                let lo = chunk * self.chunk_size;
                let hi = usize::min(lo + self.chunk_size, t);
                let mut selected = Vec::new();
                for c in 0..chunk {
                    selected.push((Self::dot(&q[h][row], &means[c]), c));
                }
                selected.sort_by(|a, b| b.0.total_cmp(&a.0));
                selected.truncate(usize::min(self.top_k, selected.len()));

                let mut keys = Vec::new();
                let mut values = Vec::new();
                for &(_, c) in &selected {
                    let clo = c * self.chunk_size;
                    let chi = usize::min(clo + self.chunk_size, t);
                    for p in clo..chi {
                        keys.push(k[h][p].clone());
                        values.push(v[h][p].clone());
                    }
                }
                for p in lo..=row {
                    keys.push(k[h][p].clone());
                    values.push(v[h][p].clone());
                }

                let y = Self::attend(&q[h][row], &keys, &values, scale, keys.len());
                for d in 0..self.head_size { out[[row, h * self.head_size + d]] = y[d]; }
                let _ = hi;
            }
        }
        self.output.forward(&out)
    }

    pub fn cache_forward(
        &self,
        x: &Array2<f32>,
        cache_k: &Array4<f32>,
        cache_v: &Array4<f32>,
    ) -> (Array2<f32>, Array4<f32>, Array4<f32>) {
        assert_eq!(x.nrows(), 1);
        assert_eq!(cache_k.shape()[0], self.heads);
        let q = self.receptance.forward(x);
        let k = self.key.forward(x);
        let v = self.value.forward(x);
        let past = cache_k.shape()[1];
        let mut all_k = Array4::<f32>::zeros((self.heads, past + 1, 1, self.head_size));
        let mut all_v = Array4::<f32>::zeros((self.heads, past + 1, 1, self.head_size));
        for h in 0..self.heads {
            for p in 0..past {
                for d in 0..self.head_size {
                    all_k[[h, p, 0, d]] = cache_k[[h, p, 0, d]];
                    all_v[[h, p, 0, d]] = cache_v[[h, p, 0, d]];
                }
            }
            for d in 0..self.head_size {
                all_k[[h, past, 0, d]] = k[[0, h * self.head_size + d]];
                all_v[[h, past, 0, d]] = v[[0, h * self.head_size + d]];
            }
        }
        let mut result = Array2::<f32>::zeros((1, self.channels));
        let scale = (self.head_size as f32).sqrt().recip();
        for h in 0..self.heads {
            let qrow: Vec<f32> = (0..self.head_size).map(|d| q[[0, h * self.head_size + d]]).collect();
            let keys: Vec<Vec<f32>> = (0..past + 1).map(|p| (0..self.head_size).map(|d| all_k[[h, p, 0, d]]).collect()).collect();
            let values: Vec<Vec<f32>> = (0..past + 1).map(|p| (0..self.head_size).map(|d| all_v[[h, p, 0, d]]).collect()).collect();
            let y = Self::attend(&qrow, &keys, &values, scale, keys.len());
            for d in 0..self.head_size { result[[0, h * self.head_size + d]] = y[d]; }
        }
        (self.output.forward(&result), all_k, all_v)
    }

    pub fn parameter_count(&self) -> usize {
        self.receptance.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::MobaAttention;
    use ndarray::{Array2, Array4};

    #[test]
    fn forward_preserves_shape() {
        let att = MobaAttention::new(16, 4, 2, 1);
        assert_eq!(att.forward(&Array2::<f32>::zeros((5, 16))).dim(), (5, 16));
    }

    #[test]
    fn cache_appends_one_token() {
        let att = MobaAttention::new(16, 4, 2, 1);
        let (_, k, v) = att.cache_forward(
            &Array2::<f32>::zeros((1, 16)),
            &Array4::<f32>::zeros((4, 3, 1, 4)),
            &Array4::<f32>::zeros((4, 3, 1, 4)),
        );
        assert_eq!(k.shape(), &[4, 4, 1, 4]);
        assert_eq!(v.shape(), &[4, 4, 1, 4]);
    }
}
