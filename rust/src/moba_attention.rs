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
        Self { channels, heads: channels / head_size, head_size, chunk_size, top_k, receptance: Linear::new_no_bias(channels, channels), key: Linear::new_no_bias(channels, channels), value: Linear::new_no_bias(channels, channels), output: Linear::new_no_bias(channels, channels) }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
    fn softmax(scores: &mut [f32]) { if scores.is_empty() { return; } let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max); let mut sum = 0.0; for x in scores.iter_mut() { *x = (*x - max).exp(); sum += *x; } if sum > 0.0 { for x in scores.iter_mut() { *x /= sum; } } }
    fn attention(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], scale: f32) -> Vec<f32> { let mut scores: Vec<f32> = keys.iter().map(|k| Self::dot(q, k) * scale).collect(); Self::softmax(&mut scores); let mut out = vec![0.0; q.len()]; for (weight, value) in scores.iter().zip(values.iter()) { for d in 0..out.len() { out[d] += weight * value[d]; } } out }

    fn project_heads(&self, x: &Array2<f32>, projection: &Linear) -> Vec<Vec<Vec<f32>>> { let projected = projection.forward(x); let mut result = vec![vec![vec![0.0; self.head_size]; x.nrows()]; self.heads]; for h in 0..self.heads { for t in 0..x.nrows() { for d in 0..self.head_size { result[h][t][d] = projected[[t, h * self.head_size + d]]; } } } result }
    fn full_causal(&self, q: &[Vec<Vec<f32>>], k: &[Vec<Vec<f32>>], v: &[Vec<Vec<f32>>], t: usize) -> Array2<f32> { let scale = (self.head_size as f32).sqrt().recip(); let mut out = Array2::<f32>::zeros((t, self.channels)); for h in 0..self.heads { for row in 0..t { let y = Self::attention(&q[h][row], &k[h][..=row], &v[h][..=row], scale); for d in 0..self.head_size { out[[row, h * self.head_size + d]] = y[d]; } } } out }

    pub fn forward_core(&self, x: &Array2<f32>) -> Array2<f32> {
        assert_eq!(x.ncols(), self.channels); let t = x.nrows(); assert!(t > 0);
        let q = self.project_heads(x, &self.receptance); let k = self.project_heads(x, &self.key); let v = self.project_heads(x, &self.value);
        let n_chunks = (t + self.chunk_size - 1) / self.chunk_size;
        if self.top_k == 0 || n_chunks <= self.top_k + 1 { return self.full_causal(&q, &k, &v, t); }
        let scale = (self.head_size as f32).sqrt().recip(); let mut out = Array2::<f32>::zeros((t, self.channels));
        for h in 0..self.heads {
            let mut means = vec![vec![0.0; self.head_size]; n_chunks];
            for c in 0..n_chunks { let lo = c * self.chunk_size; let hi = usize::min(lo + self.chunk_size, t); for row in lo..hi { for d in 0..self.head_size { means[c][d] += k[h][row][d]; } } let n = (hi - lo) as f32; for d in 0..self.head_size { means[c][d] /= n; } }
            for c in 0..n_chunks {
                let lo = c * self.chunk_size; let hi = usize::min(lo + self.chunk_size, t);
                if c == 0 { for row in lo..hi { let y = Self::attention(&q[h][row], &k[h][lo..=row], &v[h][lo..=row], scale); for d in 0..self.head_size { out[[row, h * self.head_size + d]] = y[d]; } } continue; }
                let qlen = hi - lo; let mut qmean = vec![0.0; self.head_size]; for row in lo..hi { for d in 0..self.head_size { qmean[d] += q[h][row][d]; } } for d in 0..self.head_size { qmean[d] /= qlen as f32; }
                let mut selected: Vec<(f32, usize)> = (0..c).map(|old| (Self::dot(&qmean, &means[old]), old)).collect(); selected.sort_by(|a, b| b.0.total_cmp(&a.0)); selected.truncate(usize::min(self.top_k, selected.len()));
                let mut historical_k = Vec::new(); let mut historical_v = Vec::new();
                for &(_, old) in &selected { let old_lo = old * self.chunk_size; let old_hi = usize::min(old_lo + self.chunk_size, t); historical_k.extend_from_slice(&k[h][old_lo..old_hi]); historical_v.extend_from_slice(&v[h][old_lo..old_hi]); }
                for row in lo..hi { let local_end = row - lo + 1; let mut keys = historical_k.clone(); let mut values = historical_v.clone(); keys.extend_from_slice(&k[h][lo..lo + local_end]); values.extend_from_slice(&v[h][lo..lo + local_end]); let y = Self::attention(&q[h][row], &keys, &values, scale); for d in 0..self.head_size { out[[row, h * self.head_size + d]] = y[d]; } }
            }
        }
        out
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> { self.output.forward(&self.forward_core(x)) }

    pub fn cache_forward(&self, x: &Array2<f32>, cache_k: &Array4<f32>, cache_v: &Array4<f32>) -> (Array2<f32>, Array4<f32>, Array4<f32>) {
        assert_eq!(x.nrows(), 1); let q = self.receptance.forward(x); let k = self.key.forward(x); let v = self.value.forward(x); let past = cache_k.shape()[1]; assert_eq!(cache_k.shape(), &[self.heads, past, 1, self.head_size]); assert_eq!(cache_v.shape(), &[self.heads, past, 1, self.head_size]);
        let total = past + 1; let mut all_k = Array4::<f32>::zeros((self.heads, total, 1, self.head_size)); let mut all_v = Array4::<f32>::zeros((self.heads, total, 1, self.head_size));
        for h in 0..self.heads { for p in 0..past { for d in 0..self.head_size { all_k[[h,p,0,d]]=cache_k[[h,p,0,d]]; all_v[[h,p,0,d]]=cache_v[[h,p,0,d]]; } } for d in 0..self.head_size { all_k[[h,past,0,d]]=k[[0,h*self.head_size+d]]; all_v[[h,past,0,d]]=v[[0,h*self.head_size+d]]; } }
        let cur_start=(past/self.chunk_size)*self.chunk_size; let n_prev_chunks=cur_start/self.chunk_size; let npick=usize::min(self.top_k,n_prev_chunks); let use_moba=self.top_k>0&&n_prev_chunks>self.top_k; let scale=(self.head_size as f32).sqrt().recip(); let mut result=Array2::<f32>::zeros((1,self.channels));
        for h in 0..self.heads { let qrow:Vec<f32>=(0..self.head_size).map(|d|q[[0,h*self.head_size+d]]).collect(); let mut keys=Vec::<Vec<f32>>::new(); let mut values=Vec::<Vec<f32>>::new();
            if use_moba { let mut ranked=Vec::<(f32,usize)>::with_capacity(n_prev_chunks); for chunk in 0..n_prev_chunks { let lo=chunk*self.chunk_size; let hi=lo+self.chunk_size; let mut mean=vec![0.0;self.head_size]; for p in lo..hi { for d in 0..self.head_size { mean[d]+=all_k[[h,p,0,d]]; } } for d in 0..self.head_size { mean[d]/=self.chunk_size as f32; } ranked.push((Self::dot(&qrow,&mean),chunk)); } ranked.sort_by(|a,b|b.0.total_cmp(&a.0)); ranked.truncate(npick); for &(_,chunk) in &ranked { let lo=chunk*self.chunk_size; let hi=lo+self.chunk_size; for p in lo..hi { keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect()); values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect()); } } }
            else { for p in 0..cur_start { keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect()); values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect()); } }
            for p in cur_start..total { keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect()); values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect()); }
            let y=Self::attention(&qrow,&keys,&values,scale); for d in 0..self.head_size { result[[0,h*self.head_size+d]]=y[d]; }
        }
        (self.output.forward(&result),all_k,all_v)
    }

    pub fn parameter_count(&self) -> usize { self.receptance.parameter_count()+self.key.parameter_count()+self.value.parameter_count()+self.output.parameter_count() }
}

#[cfg(test)]
mod tests {
    use super::MobaAttention; use ndarray::{Array2,Array4};
    #[test] fn forward_preserves_shape(){let att=MobaAttention::new(16,4,2,1);assert_eq!(att.forward(&Array2::<f32>::zeros((5,16))).dim(),(5,16));}
    #[test] fn short_sequence_uses_causal_fallback(){let att=MobaAttention::new(16,4,4,4);assert_eq!(att.forward(&Array2::<f32>::zeros((3,16))).dim(),(3,16));}
    #[test] fn core_is_before_output_projection(){let att=MobaAttention::new(16,4,2,1);let x=Array2::from_shape_fn((5,16),|(r,c)|0.01*(r+c)as f32);let core=att.forward_core(&x);let projected=att.forward(&x);assert_eq!(core.dim(),projected.dim());assert!(core.iter().zip(projected.iter()).any(|(a,b)|(a-b).abs()>1e-7));}
    #[test] fn cache_appends_one_token(){let att=MobaAttention::new(16,4,2,1);let(_,k,v)=att.cache_forward(&Array2::<f32>::zeros((1,16)),&Array4::<f32>::zeros((4,3,1,4)),&Array4::<f32>::zeros((4,3,1,4)));assert_eq!(k.shape(),&[4,4,1,4]);assert_eq!(v.shape(),&[4,4,1,4]);}
}
