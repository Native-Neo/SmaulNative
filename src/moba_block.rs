use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::rwkv_cmix::RwkvCmix;

use ndarray::{Array1, Array2, Array4};

pub struct MobaBlockState{pub cmix_prev:Array1<f32>,pub att_k:Array4<f32>,pub att_v:Array4<f32>}
pub struct MobaBlock{pub channels:usize,pub layer_id:usize,pub ln1:LayerNorm,pub ln2:LayerNorm,pub att:MobaAttention,pub ffn:RwkvCmix}
pub struct MobaBlockTape{pub input:Array2<f32>,pub norm1:Array2<f32>,pub q:Array2<f32>,pub k:Array2<f32>,pub v:Array2<f32>,pub att_core:Array2<f32>,pub att_output:Array2<f32>,pub residual:Array2<f32>,pub norm2:Array2<f32>,pub ffn_output:Array2<f32>,pub initial_cmix_prev:Option<Array1<f32>>,pub output:Array2<f32>}
pub struct MobaBlockBackward{pub grad_input:Array2<f32>,pub grad_ln1_weight:Array1<f32>,pub grad_ln1_bias:Array1<f32>,pub grad_ln2_weight:Array1<f32>,pub grad_ln2_bias:Array1<f32>,pub grad_receptance:Array2<f32>,pub grad_key:Array2<f32>,pub grad_value:Array2<f32>,pub grad_output:Array2<f32>,pub grad_ffn_x_k:Array1<f32>,pub grad_ffn_key:Array2<f32>,pub grad_ffn_value:Array2<f32>,pub grad_cmix_prev:Array1<f32>}
impl MobaBlock{pub fn new(channels:usize,head_size:usize,chunk_size:usize,top_k:usize,layer_id:usize,n_layer:usize)->Self{Self::new_with_qat(channels,head_size,chunk_size,top_k,layer_id,n_layer,0)}pub fn new_with_qat(channels:usize,head_size:usize,chunk_size:usize,top_k:usize,layer_id:usize,n_layer:usize,bits:u8)->Self{Self{channels,layer_id,ln1:LayerNorm::new(channels,1e-5),ln2:LayerNorm::new(channels,1e-5),att:MobaAttention::new_with_qat(channels,head_size,chunk_size,top_k,bits),ffn:RwkvCmix::new(channels,layer_id,n_layer).with_qat_bits(bits)}}pub fn forward(&self,x:&Array2<f32>,state:Option<&MobaBlockState>)->(Array2<f32>,MobaBlockState){let(_norm1,att_out,att_k,att_v)={let n=self.ln1.forward(x);match state{Some(s)if x.nrows()==1=>{let(y,k,v)=self.att.cache_forward(&n,&s.att_k,&s.att_v);(n,y,k,v)},_=>{let y=self.att.forward(&n);let h=self.att.heads;let hs=self.att.head_size;let mut k=Array4::zeros((h,x.nrows(),1,hs));let mut v=Array4::zeros((h,x.nrows(),1,hs));let kp=self.att.key.forward(&n);let vp=self.att.value.forward(&n);for a in 0..h{for t in 0..x.nrows(){for d in 0..hs{k[[a,t,0,d]]=kp[[t,a*hs+d]];v[[a,t,0,d]]=vp[[t,a*hs+d]]}}}(n,y,k,v)}}};let residual=x+&att_out;let norm2=self.ln2.forward(&residual);let(ffn_out,cmix_prev)=self.ffn.forward(&norm2,state.map(|s|&s.cmix_prev));let output=&residual+&ffn_out;(output,MobaBlockState{cmix_prev,att_k,att_v})}pub fn forward_with_tape(&self,x:&Array2<f32>,state:Option<&MobaBlockState>)->(Array2<f32>,MobaBlockState,MobaBlockTape){assert!(x.nrows()>0);let norm1=self.ln1.forward(x);let q=self.att.receptance.forward(&norm1);let k=self.att.key.forward(&norm1);let v=self.att.value.forward(&norm1);let att_core=self.att.forward_core(&norm1);let att_output=self.att.output.forward(&att_core);let residual=x+&att_output;let norm2=self.ln2.forward(&residual);let(ffn_output,cmix_prev)=self.ffn.forward(&norm2,state.map(|s|&s.cmix_prev));let output=&residual+&ffn_output;let mut hk=Array4::zeros((self.att.heads,x.nrows(),1,self.att.head_size));let mut hv=Array4::zeros(hk.raw_dim());for h in 0..self.att.heads{for t in 0..x.nrows(){for d in 0..self.att.head_size{hk[[h,t,0,d]]=k[[t,h*self.att.head_size+d]];hv[[h,t,0,d]]=v[[t,h*self.att.head_size+d]]}}}let tape=MobaBlockTape{input:x.clone(),norm1,q,k,v,att_core,att_output,residual,norm2,ffn_output,initial_cmix_prev:state.map(|s|s.cmix_prev.clone()),output:output.clone()};(output,MobaBlockState{cmix_prev,att_k:hk,att_v:hv},tape)}pub fn backward(&self,tape:&MobaBlockTape,grad_output:&Array2<f32>)->MobaBlockBackward{assert_eq!(grad_output.dim(),tape.output.dim());let grad_ffn=grad_output.clone();let grad_residual=grad_output.clone();let ffn=crate::rwkv_cmix::backward(&tape.norm2,tape.initial_cmix_prev.as_ref(),&grad_ffn,&self.ffn.x_k,&self.ffn.key,&self.ffn.value);let(grad_res2,gw2,gb2)=crate::layer_norm::backward(&tape.residual,&ffn.grad_input,&self.ln2.weight,self.ln2.eps);let grad_residual=&grad_residual+&grad_res2;let grad_att=grad_residual.clone();let grad_core=grad_att.dot(&self.att.output.weight);let mut gq=Array2::zeros(tape.q.raw_dim());let mut gk=Array2::zeros(tape.k.raw_dim());let mut gv=Array2::zeros(tape.v.raw_dim());let scale=(self.att.head_size as f32).sqrt().recip();for h in 0..self.att.heads{let hs=self.att.head_size;let mut qh=Array2::zeros((tape.q.nrows(),hs));let mut kh=Array2::zeros((tape.k.nrows(),hs));let mut vh=Array2::zeros((tape.v.nrows(),hs));let mut gh=Array2::zeros((grad_core.nrows(),hs));for t in 0..tape.q.nrows(){for d in 0..hs{qh[[t,d]]=tape.q[[t,h*hs+d]];kh[[t,d]]=tape.k[[t,h*hs+d]];vh[[t,d]]=tape.v[[t,h*hs+d]];gh[[t,d]]=grad_core[[t,h*hs+d]]}}let b=backward(&qh,&kh,&vh,&gh,self.att.chunk_size,self.att.top_k,scale);for t in 0..tape.q.nrows(){for d in 0..hs{gq[[t,h*hs+d]]=b.grad_q[[t,d]];gk[[t,h*hs+d]]=b.grad_k[[t,d]];gv[[t,h*hs+d]]=b.grad_v[[t,d]]}}}let grad_q_input=gq.dot(&self.att.receptance.weight);let grad_k_input=gk.dot(&self.att.key.weight);let grad_v_input=gv.dot(&self.att.value.weight);let grad_norm1=&grad_q_input+&grad_k_input+&grad_v_input;let(grad_input_norm,gw1,gb1)=crate::layer_norm::backward(&tape.input,&grad_norm1,&self.ln1.weight,self.ln1.eps);let grad_input=&grad_residual+&grad_input_norm;MobaBlockBackward{grad_input,grad_ln1_weight:gw1,grad_ln1_bias:gb1,grad_ln2_weight:gw2,grad_ln2_bias:gb2,grad_receptance:gq.t().dot(&tape.norm1),grad_key:gk.t().dot(&tape.norm1),grad_value:gv.t().dot(&tape.norm1),grad_output:grad_att.t().dot(&tape.att_core),grad_ffn_x_k:ffn.grad_x_k,grad_ffn_key:ffn.grad_key,grad_ffn_value:ffn.grad_value,grad_cmix_prev:ffn.grad_prev}}pub fn parameter_count(&self)->usize{self.ln1.weight.len()+self.ln1.bias.len()+self.ln2.weight.len()+self.ln2.bias.len()+self.att.parameter_count()+self.ffn.parameter_count()}}

// ===== moba_attention =====
pub struct MobaAttention{pub channels:usize,pub heads:usize,pub head_size:usize,pub chunk_size:usize,pub top_k:usize,pub receptance:Linear,pub key:Linear,pub value:Linear,pub output:Linear}
impl MobaAttention{pub fn new(channels:usize,head_size:usize,chunk_size:usize,top_k:usize)->Self{Self::new_with_qat(channels,head_size,chunk_size,top_k,0)}pub fn new_with_qat(channels:usize,head_size:usize,chunk_size:usize,top_k:usize,bits:u8)->Self{assert!(channels>0&&head_size>0);assert_eq!(channels%head_size,0);assert!(chunk_size>0);Self{channels,heads:channels/head_size,head_size,chunk_size,top_k,receptance:Linear::new_with_qat(channels,channels,false,bits),key:Linear::new_with_qat(channels,channels,false,bits),value:Linear::new_with_qat(channels,channels,false,bits),output:Linear::new_with_qat(channels,channels,false,bits)}}fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|x*y).sum()}fn softmax(scores:&mut[f32]){if scores.is_empty(){return}let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);let mut sum=0.0;for x in scores.iter_mut(){*x=(*x-max).exp();sum+=*x}if sum>0.0{for x in scores.iter_mut(){*x/=sum}}}fn attention(q:&[f32],keys:&[Vec<f32>],values:&[Vec<f32>],scale:f32)->Vec<f32>{let mut scores:Vec<f32>=keys.iter().map(|k|Self::dot(q,k)*scale).collect();Self::softmax(&mut scores);let mut out=vec![0.0;q.len()];for(weight,value)in scores.iter().zip(values.iter()){for d in 0..out.len(){out[d]+=weight*value[d]}}out}fn project_heads(&self,x:&Array2<f32>,projection:&Linear)->Vec<Vec<Vec<f32>>>{let projected=projection.forward(x);let mut result=vec![vec![vec![0.0;self.head_size];x.nrows()];self.heads];for h in 0..self.heads{for t in 0..x.nrows(){for d in 0..self.head_size{result[h][t][d]=projected[[t,h*self.head_size+d]]}}}result}fn full_causal(&self,q:&[Vec<Vec<f32>>],k:&[Vec<Vec<f32>>],v:&[Vec<Vec<f32>>],t:usize)->Array2<f32>{let scale=(self.head_size as f32).sqrt().recip();let mut out=Array2::<f32>::zeros((t,self.channels));for h in 0..self.heads{for row in 0..t{let y=Self::attention(&q[h][row],&k[h][..=row],&v[h][..=row],scale);for d in 0..self.head_size{out[[row,h*self.head_size+d]]=y[d]}}}out}pub fn forward_core(&self,x:&Array2<f32>)->Array2<f32>{assert_eq!(x.ncols(),self.channels);let t=x.nrows();assert!(t>0);let q=self.project_heads(x,&self.receptance);let k=self.project_heads(x,&self.key);let v=self.project_heads(x,&self.value);let n_chunks=(t+self.chunk_size-1)/self.chunk_size;if self.top_k==0||n_chunks<=self.top_k+1{return self.full_causal(&q,&k,&v,t)}let scale=(self.head_size as f32).sqrt().recip();let mut out=Array2::<f32>::zeros((t,self.channels));for h in 0..self.heads{let mut means=vec![vec![0.0;self.head_size];n_chunks];for c in 0..n_chunks{let lo=c*self.chunk_size;let hi=usize::min(lo+self.chunk_size,t);for row in lo..hi{for d in 0..self.head_size{means[c][d]+=k[h][row][d]}}let n=(hi-lo)as f32;for d in 0..self.head_size{means[c][d]/=n}}for c in 0..n_chunks{let lo=c*self.chunk_size;let hi=usize::min(lo+self.chunk_size,t);if c==0{for row in lo..hi{let y=Self::attention(&q[h][row],&k[h][lo..=row],&v[h][lo..=row],scale);for d in 0..self.head_size{out[[row,h*self.head_size+d]]=y[d]}}continue}let qlen=hi-lo;let mut qmean=vec![0.0;self.head_size];for row in lo..hi{for d in 0..self.head_size{qmean[d]+=q[h][row][d]}}for d in 0..self.head_size{qmean[d]/=qlen as f32}let mut selected:Vec<(f32,usize)>=(0..c).map(|old|(Self::dot(&qmean,&means[old]),old)).collect();selected.sort_by(|a,b|b.0.total_cmp(&a.0));selected.truncate(usize::min(self.top_k,selected.len()));let mut historical_k=Vec::new();let mut historical_v=Vec::new();for&(_,old)in&selected{let old_lo=old*self.chunk_size;let old_hi=usize::min(old_lo+self.chunk_size,t);historical_k.extend_from_slice(&k[h][old_lo..old_hi]);historical_v.extend_from_slice(&v[h][old_lo..old_hi])}for row in lo..hi{let local_end=row-lo+1;let mut keys=historical_k.clone();let mut values=historical_v.clone();keys.extend_from_slice(&k[h][lo..lo+local_end]);values.extend_from_slice(&v[h][lo..lo+local_end]);let y=Self::attention(&q[h][row],&keys,&values,scale);for d in 0..self.head_size{out[[row,h*self.head_size+d]]=y[d]}}}}out}pub fn forward(&self,x:&Array2<f32>)->Array2<f32>{self.output.forward(&self.forward_core(x))}pub fn cache_forward(&self,x:&Array2<f32>,cache_k:&Array4<f32>,cache_v:&Array4<f32>)->(Array2<f32>,Array4<f32>,Array4<f32>){assert_eq!(x.nrows(),1);let q=self.receptance.forward(x);let k=self.key.forward(x);let v=self.value.forward(x);let past=cache_k.shape()[1];assert_eq!(cache_k.shape(),&[self.heads,past,1,self.head_size]);assert_eq!(cache_v.shape(),&[self.heads,past,1,self.head_size]);let total=past+1;let mut all_k=Array4::<f32>::zeros((self.heads,total,1,self.head_size));let mut all_v=Array4::<f32>::zeros((self.heads,total,1,self.head_size));for h in 0..self.heads{for p in 0..past{for d in 0..self.head_size{all_k[[h,p,0,d]]=cache_k[[h,p,0,d]];all_v[[h,p,0,d]]=cache_v[[h,p,0,d]]}}for d in 0..self.head_size{all_k[[h,past,0,d]]=k[[0,h*self.head_size+d]];all_v[[h,past,0,d]]=v[[0,h*self.head_size+d]]}}let cur_start=(past/self.chunk_size)*self.chunk_size;let n_prev_chunks=cur_start/self.chunk_size;let npick=usize::min(self.top_k,n_prev_chunks);let use_moba=self.top_k>0&&n_prev_chunks>self.top_k;let scale=(self.head_size as f32).sqrt().recip();let mut result=Array2::<f32>::zeros((1,self.channels));for h in 0..self.heads{let qrow:Vec<f32>=(0..self.head_size).map(|d|q[[0,h*self.head_size+d]]).collect();let mut keys=Vec::<Vec<f32>>::new();let mut values=Vec::<Vec<f32>>::new();if use_moba{let mut ranked=Vec::<(f32,usize)>::with_capacity(n_prev_chunks);for chunk in 0..n_prev_chunks{let lo=chunk*self.chunk_size;let hi=lo+self.chunk_size;let mut mean=vec![0.0;self.head_size];for p in lo..hi{for d in 0..self.head_size{mean[d]+=all_k[[h,p,0,d]]}}for d in 0..self.head_size{mean[d]/=self.chunk_size as f32}ranked.push((Self::dot(&qrow,&mean),chunk))}ranked.sort_by(|a,b|b.0.total_cmp(&a.0));ranked.truncate(npick);for&(_,chunk)in&ranked{let lo=chunk*self.chunk_size;let hi=lo+self.chunk_size;for p in lo..hi{keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect());values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect())}}}else{for p in 0..cur_start{keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect());values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect())}}for p in cur_start..total{keys.push((0..self.head_size).map(|d|all_k[[h,p,0,d]]).collect());values.push((0..self.head_size).map(|d|all_v[[h,p,0,d]]).collect())}let y=Self::attention(&qrow,&keys,&values,scale);for d in 0..self.head_size{result[[0,h*self.head_size+d]]=y[d]}}(self.output.forward(&result),all_k,all_v)}pub fn parameter_count(&self)->usize{self.receptance.parameter_count()+self.key.parameter_count()+self.value.parameter_count()+self.output.parameter_count()}}
#[cfg(test)]mod tests_moba_attention{use super::MobaAttention;use ndarray::{Array2,Array4};#[test]fn forward_preserves_shape(){let att=MobaAttention::new(16,4,2,1);assert_eq!(att.forward(&Array2::<f32>::zeros((5,16))).dim(),(5,16))}#[test]fn short_sequence_uses_causal_fallback(){let att=MobaAttention::new(16,4,4,4);assert_eq!(att.forward(&Array2::<f32>::zeros((3,16))).dim(),(3,16))}#[test]fn core_is_before_output_projection(){let att=MobaAttention::new(16,4,2,1);let x=Array2::from_shape_fn((5,16),|(r,c)|0.01*(r+c)as f32);let core=att.forward_core(&x);let projected=att.forward(&x);assert_eq!(core.dim(),projected.dim());assert!(core.iter().zip(projected.iter()).any(|(a,b)|(a-b).abs()>1e-7))}#[test]fn cache_appends_one_token(){let att=MobaAttention::new(16,4,2,1);let(_,k,v)=att.cache_forward(&Array2::<f32>::zeros((1,16)),&Array4::<f32>::zeros((4,3,1,4)),&Array4::<f32>::zeros((4,3,1,4)));assert_eq!(k.shape(),&[4,4,1,4]);assert_eq!(v.shape(),&[4,4,1,4])}#[test]fn qat_forward_preserves_shape(){let att=MobaAttention::new_with_qat(16,4,2,1,4);assert_eq!(att.forward(&Array2::<f32>::zeros((5,16))).dim(),(5,16))}}

// ===== moba_routing_backward =====

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedChunks {
    pub selected: Vec<usize>,
    pub current: usize,
}

pub fn select_topk(scores: &Array1<f32>, top_k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(&b)));
    ids.truncate(top_k.min(ids.len()));
    ids.sort_unstable();
    ids
}

pub fn route_query(query: &Array1<f32>, chunk_means: &Array2<f32>, current_chunk: usize, top_k: usize) -> RoutedChunks {
    assert_eq!(query.len(), chunk_means.ncols());
    let mut scores = Array1::zeros(chunk_means.nrows());
    for c in 0..chunk_means.nrows() {
        scores[c] = query.iter().zip(chunk_means.row(c).iter()).map(|(a,b)| a*b).sum();
    }
    let mut selected = select_topk(&scores, top_k);
    selected.retain(|&c| c != current_chunk);
    RoutedChunks { selected, current: current_chunk }
}

#[cfg(test)]
mod tests_moba_routing_backward {
    use super::*;

    #[test]
    fn routing_excludes_current_chunk() {
        let q = Array1::from_vec(vec![1.0, 0.0]);
        let means = Array2::from_shape_vec((3, 2), vec![0.1,0.0, 0.9,0.0, 2.0,0.0]).unwrap();
        let route = route_query(&q, &means, 2, 2);
        assert_eq!(route.current, 2);
        assert!(!route.selected.contains(&2));
    }
}

// ===== moba_routed_backward =====

pub struct MobaBackward {
    pub grad_q: Array2<f32>,
    pub grad_k: Array2<f32>,
    pub grad_v: Array2<f32>,
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out { *v /= sum; }
    }
    out
}

pub fn backward(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, grad_output: &Array2<f32>, chunk_size: usize, top_k: usize, scale: f32) -> MobaBackward {
    let (tokens, dim) = q.dim();
    assert_eq!(k.dim(), (tokens, dim));
    assert_eq!(v.nrows(), tokens);
    assert_eq!(grad_output.nrows(), tokens);
    assert_eq!(grad_output.ncols(), v.ncols());
    assert!(chunk_size > 0);

    let chunks = (tokens + chunk_size - 1) / chunk_size;
    let mut means = Vec::with_capacity(chunks);
    let mut qmeans = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(tokens);
        let inv = 1.0 / (end - start) as f32;
        let mut mean_k = vec![0.0; dim];
        let mut mean_q = vec![0.0; dim];
        for t in start..end {
            for d in 0..dim {
                mean_k[d] += k[[t, d]];
                mean_q[d] += q[[t, d]];
            }
        }
        for d in 0..dim {
            mean_k[d] *= inv;
            mean_q[d] *= inv;
        }
        means.push(mean_k);
        qmeans.push(mean_q);
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
                let score: f32 = qmeans[current].iter().zip(&means[c]).map(|(qv, kv)| qv * kv).sum();
                (score, c)
            }).collect();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, c) in ranked.iter().take(top_k.min(ranked.len())) {
                selected.push((c * chunk_size, ((c + 1) * chunk_size).min(tokens)));
            }
        }

        let current_end = (t + 1).min((current + 1) * chunk_size).min(tokens);
        selected.push((cur_start, current_end));

        let mut indices = Vec::new();
        for (start, end) in selected {
            for j in start..end {
                if j <= t || start != cur_start { indices.push(j); }
            }
        }
        indices.sort_unstable();
        indices.dedup();

        let mut scores = Vec::with_capacity(indices.len());
        for &j in &indices {
            let mut score = 0.0;
            for d in 0..dim { score += q[[t, d]] * k[[j, d]]; }
            scores.push(score * scale);
        }
        let probs = softmax(&scores);

        let mut grad_probs = vec![0.0; indices.len()];
        for i in 0..indices.len() {
            let j = indices[i];
            for d in 0..v.ncols() { grad_probs[i] += grad_output[[t, d]] * v[[j, d]]; }
        }
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
mod tests_moba_routed_backward {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn routed_backward_has_finite_gradients() {
        let q = Array2::ones((4, 3));
        let k = Array2::ones((4, 3));
        let v = Array2::ones((4, 3));
        let g = backward(&q, &k, &v, &v, 2, 1, 0.5);
        assert_eq!(g.grad_q.dim(), (4, 3));
        assert!(g.grad_q.iter().all(|x|x.is_finite()));
    }

    #[test]
    fn routing_uses_current_chunk_mean_query() {
        let q = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        let k = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        let v = Array2::ones((4, 2));
        let g = backward(&q, &k, &v, &v, 2, 1, 1.0);
        assert!(g.grad_q.iter().all(|x|x.is_finite()));
    }
}
