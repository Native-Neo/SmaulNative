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
        assert!(channels > 0);
        assert!(head_size > 0);
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
        for value in scores.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        if sum > 0.0 {
            for value in scores.iter_mut() { *value /= sum; }
        }
    }

    fn attention_row(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], scale: f32, causal: usize) -> Vec<f32> {
        let mut scores = Vec::with_capacity(causal + 1);
        for key in keys.iter().take(causal + 1) {
            scores.push(Self::dot(q, key) * scale);
        }
        Self::softmax(&mut scores);
        let mut out = vec![0.0; q.len()];
        for (weight, value) in scores.iter().zip(values.iter()) {
            for d in 0..out.len() { out[d] += *weight * value[d]; }
        }
        out
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels);
        let t = x.nrows();
        let q = self.receptance.forward(x);
        let k = self.key.forward(x);
        let v = self.value.forward(x);
        let mut out = Array2::<f32>::zeros((t, self.channels));
        let scale = (self.head_size as f32).sqrt().recip();
        let chunks = (t + self.chunk_size - 1) / self.chunk_size;

        for head in 0..self.heads {
            let base = head * self.head_size;
            let mut chunk_keys = Vec::with_capacity(chunks);
            for chunk in 0..chunks {
                let start = chunk * self.chunk_size;
                let end = usize::min(start + self.chunk_size, t);
                let mut mean = vec![0.0; self.head_size];
                for row in start..end {
                    for d in 0..self.head_size { mean[d] += k[[row, base + d]]; }
                }
                let len = (end - start) as f32;
                if len > 0.0 { for d in 0..self.head_size { mean[d] /= len; } }
                chunk_keys.push(mean);
            }

            for row in 0..t {
                let current_chunk = row / self.chunk_size;
                let start = current_chunk * self.chunk_size;
                let end = usize::min(start + self.chunk_size, t);
                let mut selected = Vec::new();
                for chunk in 0..current_chunk {
                    let score = Self::dot(
                        &(0..self.head_size).map(|d| q[[row, base + d]]).collect::<Vec<_>>(),
                        &chunk_keys[chunk],
                    );
                    selected.push((score, chunk));
                }
                selected.sort_by(|a, b| b.0.total_cmp(&a.0));
                let keep = usize::min(self.top_k, selected.len());
                selected.truncate(keep);

                let mut keys = Vec::new();
                let mut values = Vec::new();
                for &(_, chunk) in &selected {
                    let lo = chunk * self.chunk_size;
                    let hi = usize::min(lo + self.chunk_size, t);
                    for r in lo..hi {
                        keys.push((0..self.head_size).map(|d| k[[r, base + d]]).collect());
                        values.push((0..self.head_size).map(|d| v[[r, base + d]]).collect());
                    }
                }
                for r in start..end {
                    keys.push((0..self.head_size).map(|d| k[[r, base + d]]).collect());
                    values.push((0..self.head_size).map(|d| v[[r, base + d]]).collect());
                }
                let qrow: Vec<f32> = (0..self.head_size).map(|d| q[[row, base + d]]).collect();
                let local_index = keys.len() - (end - start) + (row - start);
                let y = Self::attention_row(&qrow, &keys, &values, scale, local_index);
                for d in 0..self.head_size { out[[row, base + d]] = y[d]; }
            }
        }
        self.output.forward(&out)
    }

    pub fn parameter_count(&self) -> usize {
        self.receptance.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output.parameter_count()
    }

    pub fn cache_forward(&self, x: &Array2<f32>, cache_k: &Array4<f32>, cache_v: &Array4<f32>) -> (Array2<f32>, Array4<f32>, Array4<f32>) {
        assert_eq!(x.nrows(), 1);
        assert_eq!(cache_k.shape()[0], self.heads);
        assert_eq!(cache_v.shape()[0], self.heads);
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
            let y = Self::attention_row(&qrow, &keys, &values, scale, past);
            for d in 0..self.head_size { result[[0, h * self.head_size + d]] = y[d]; }
        }
        (self.output.forward(&result), all_k, all_v)
    }
}

#[cfg(test)]
mod tests {
    use super::MobaAttention;
    use ndarray::{Array2, Array4};

    #[test]
    fn forward_preserves_shape() {
        let att = MobaAttention::new(16, 4, 2, 1);
        let x = Array2::<f32>::zeros((5, 16));
        assert_eq!(att.forward(&x).dim(), (5, 16));
    }

    #[test]
    fn cache_appends_one_token() {
        let att = MobaAttention::new(16, 4, 2, 1);
        let x = Array2::<f32>::zeros((1, 16));
        let k = Array4::<f32>::zeros((4, 3, 1, 4));
        let v = Array4::<f32>::zeros((4, 3, 1, 4));
        let (_, nk, nv) = att.cache_forward(&x, &k, &v);
        assert_eq!(nk.shape(), &[4, 4, 1, 4]);
        assert_eq!(nv.shape(), &[4, 4, 1, 4]);
    }
}
