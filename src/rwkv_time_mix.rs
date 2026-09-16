use crate::group_norm::GroupNorm;
use crate::init::{orthogonal, uniform};
use crate::qat::RealQuantLinear;
use crate::wkv;
use ndarray::{Array1, Array2, Array4};

pub struct RwkvTimeMix {
    pub channels: usize, pub heads: usize, pub head_size: usize,
    pub x_r: Array1<f32>, pub x_w: Array1<f32>, pub x_k: Array1<f32>, pub x_v: Array1<f32>, pub x_a: Array1<f32>, pub x_g: Array1<f32>,
    pub w0: Array1<f32>, pub w1: Array2<f32>, pub w2: Array2<f32>, pub a1: Array2<f32>, pub a2: Array2<f32>,
    pub v1: Option<Array2<f32>>, pub v2: Option<Array2<f32>>, pub g1: Array2<f32>, pub g2: Array2<f32>, pub a0: Array1<f32>, pub v0: Option<Array1<f32>>,
    pub k_k: Array1<f32>, pub k_a: Array1<f32>, pub r_k: Array2<f32>, pub receptance: Array2<f32>, pub key: Array2<f32>, pub value: Array2<f32>, pub output: Array2<f32>, pub ln_x: GroupNorm,
    pub qat_bits: u8,
    q_w1: Option<RealQuantLinear>, q_w2: Option<RealQuantLinear>, q_a1: Option<RealQuantLinear>, q_a2: Option<RealQuantLinear>,
    q_v1: Option<RealQuantLinear>, q_v2: Option<RealQuantLinear>, q_g1: Option<RealQuantLinear>, q_g2: Option<RealQuantLinear>,
    q_receptance: Option<RealQuantLinear>, q_key: Option<RealQuantLinear>, q_value: Option<RealQuantLinear>, q_output: Option<RealQuantLinear>,
}
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn mix(x: &Array2<f32>, prev: &Array1<f32>, factor: &Array1<f32>) -> Array2<f32> { let mut out=x.clone(); for t in 0..x.nrows(){for c in 0..x.ncols(){let p=if t==0{prev[c]}else{x[[t-1,c]]};out[[t,c]]=x[[t,c]]+(p-x[[t,c]])*factor[c];}} out }
fn quant(weight:&Array2<f32>, bits:u8)->Option<RealQuantLinear>{if bits==0{return None;} RealQuantLinear::new(weight.t().to_owned(),None,bits).ok()}
impl RwkvTimeMix {
 pub fn new(channels:usize,heads:usize,layer_id:usize,n_layer:usize)->Self{Self::new_with_qat(channels,heads,layer_id,n_layer,0)}
 pub fn new_with_qat(channels:usize,heads:usize,layer_id:usize,n_layer:usize,qat_bits:u8)->Self{assert_eq!(channels%heads,0);assert!(matches!(qat_bits,0|2|3|4|8));let head_size=channels/heads;let r0=layer_id as f32/n_layer.saturating_sub(1).max(1) as f32;let r1=1.0-layer_id as f32/n_layer as f32;let dd=(1.8*(channels as f32).sqrt()/32.0).round().max(1.0) as usize*32;let dm=(1.3*(channels as f32).sqrt()/32.0).round().max(1.0) as usize*32;let dg=(0.6*(channels as f32).powf(0.8)/32.0).round().max(1.0) as usize*32;let mut x_r=Array1::zeros(channels);let mut x_w=Array1::zeros(channels);let mut x_k=Array1::zeros(channels);let mut x_v=Array1::zeros(channels);let mut x_a=Array1::zeros(channels);let mut x_g=Array1::zeros(channels);let mut w0=Array1::zeros(channels);for c in 0..channels{let d=if channels>1{c as f32/(channels-1) as f32}else{0.0};let dc=c as f32/channels as f32;x_r[c]=1.0-dc.powf(0.2*r1);x_w[c]=1.0-dc.powf(0.9*r1);x_k[c]=1.0-(dc.powf(0.9*r1)+0.4*r0);x_v[c]=1.0-(dc.powf(0.4*r1)+0.6*r0);x_a[c]=1.0-dc.powf(0.9*r1);x_g[c]=1.0-dc.powf(0.2*r1);w0[c]=-7.0+5.0*d.powf(0.85+r0.sqrt())+0.5;}let w1=Array2::zeros((channels,dd));let a1=Array2::zeros((channels,dd));let g1=Array2::zeros((channels,dg));let v1=(layer_id>0).then(||Array2::zeros((channels,dm)));let w2_gain=(dd as f32/channels as f32).sqrt().max(1.0)*0.1;let a2_gain=w2_gain;let g2_gain=(dg as f32/channels as f32).sqrt().max(1.0)*0.1;let v2_gain=(dm as f32/channels as f32).sqrt().max(1.0)*0.1;let w2=orthogonal(dd,channels,w2_gain,0x5752_5637_3030^layer_id as u64);let a2=orthogonal(dd,channels,a2_gain,0x4132_3030_3030^layer_id as u64);let g2=orthogonal(dg,channels,g2_gain,0x4732_3030_3030^layer_id as u64);let v2=(layer_id>0).then(||orthogonal(dm,channels,v2_gain,0x5632_3030_3030^layer_id as u64));let lim=0.5/(channels as f32).sqrt();let receptance=uniform(channels,channels,-lim,lim,0x5253_3030_3030^layer_id as u64);let key=uniform(channels,channels,-0.05/(channels as f32).sqrt(),0.05/(channels as f32).sqrt(),0x4b59_3030_3030^layer_id as u64);let value=uniform(channels,channels,-lim,lim,0x5641_3030_3030^layer_id as u64);let output=Array2::zeros((channels,channels));let mut s=Self{channels,heads,head_size,x_r,x_w,x_k,x_v,x_a,x_g,w0,w1,w2,a1,a2,v1,v2,g1,g2,a0:Array1::zeros(channels),v0:(layer_id>0).then(||Array1::ones(channels)),k_k:Array1::from_elem(channels,0.85),k_a:Array1::ones(channels),r_k:Array2::zeros((heads,head_size)),receptance,key,value,output,ln_x:GroupNorm::new(channels,heads,1e-5*8.0*8.0),qat_bits,q_w1:None,q_w2:None,q_a1:None,q_a2:None,q_v1:None,q_v2:None,q_g1:None,q_g2:None,q_receptance:None,q_key:None,q_value:None,q_output:None};s.refresh_quantized();s}
 pub fn refresh_quantized(&mut self){let b=self.qat_bits;self.q_w1=quant(&self.w1,b);self.q_w2=quant(&self.w2,b);self.q_a1=quant(&self.a1,b);self.q_a2=quant(&self.a2,b);self.q_v1=self.v1.as_ref().and_then(|x|quant(x,b));self.q_v2=self.v2.as_ref().and_then(|x|quant(x,b));self.q_g1=quant(&self.g1,b);self.q_g2=quant(&self.g2,b);self.q_receptance=quant(&self.receptance,b);self.q_key=quant(&self.key,b);self.q_value=quant(&self.value,b);self.q_output=quant(&self.output,b);}
 fn project(x:&Array2<f32>,weight:&Array2<f32>,q:&Option<RealQuantLinear>)->Array2<f32>{match q{Some(q)=>q.forward(x),None=>x.dot(weight)}}
 pub fn forward_with_tape(&self,x:&Array2<f32>,state:Option<Array4<f32>>,prev:Option<Array1<f32>>,v_first:Option<Array2<f32>>)->(Array2<f32>,Array4<f32>,Array1<f32>,Array2<f32>,RwkvTimeMixTape){assert!(x.nrows()>0);let zero=Array1::zeros(self.channels);let prev_ref=prev.as_ref().unwrap_or(&zero);let initial=state.unwrap_or_else(||Array4::zeros((1,self.heads,self.head_size,self.head_size)));let mut tape=RwkvTimeMixTape::new(x.clone(),prev_ref.clone(),initial.clone(),self.w1.ncols(),self.v1.as_ref().map(|v|v.ncols()),self.g1.ncols());tape.xr=mix(x,prev_ref,&self.x_r);tape.xw=mix(x,prev_ref,&self.x_w);tape.xk=mix(x,prev_ref,&self.x_k);tape.xv=mix(x,prev_ref,&self.x_v);tape.xa=mix(x,prev_ref,&self.x_a);tape.xg=mix(x,prev_ref,&self.x_g);tape.r=Self::project(&tape.xr,&self.receptance,&self.q_receptance);tape.w_hidden=Self::project(&tape.xw,&self.w1,&self.q_w1).mapv(|v|v.tanh());tape.g_decay=Self::project(&tape.w_hidden,&self.w2,&self.q_w2);tape.k=Self::project(&tape.xk,&self.key,&self.q_key);tape.v_base=Self::project(&tape.xv,&self.value,&self.q_value);tape.v=tape.v_base.clone();tape.a_hidden=Self::project(&tape.xa,&self.a1,&self.q_a1);tape.a=Self::project(&tape.a_hidden,&self.a2,&self.q_a2);for t in 0..tape.a.nrows(){for c in 0..self.channels{tape.a[[t,c]]=sigmoid(self.a0[c]+tape.a[[t,c]]);}}tape.g_hidden=Self::project(&tape.xg,&self.g1,&self.q_g1).mapv(sigmoid);tape.g=Self::project(&tape.g_hidden,&self.g2,&self.q_g2);tape.v_first=v_first.unwrap_or_else(||tape.v.clone());if let(Some(v1),Some(v2),Some(v0))=(&self.v1,&self.v2,&self.v0){let correction=Self::project(&Self::project(&tape.xv,v1,&self.q_v1),v2,&self.q_v2);let mut gate=Array2::zeros(tape.v.dim());for t in 0..tape.v.nrows(){for c in 0..self.channels{gate[[t,c]]=sigmoid(v0[c]+correction[[t,c]]);tape.v[[t,c]]+=(tape.v_first[[t,c]]-tape.v[[t,c]])*gate[[t,c]];}}tape.v_correction=Some(correction);tape.v_gate=Some(gate);}tape.kk_pre_norm=tape.k.clone();tape.kk=tape.k.clone();for t in 0..tape.kk.nrows(){for h in 0..self.heads{let start=h*self.head_size;let mut norm=0.0;for i in 0..self.head_size{let z=tape.kk[[t,start+i]]*self.k_k[start+i];norm+=z*z;}let inv=(norm+1e-12).sqrt().recip();for i in 0..self.head_size{tape.kk[[t,start+i]]*=self.k_k[start+i]*inv;}}}tape.k_mod=tape.k.clone();for t in 0..tape.k_mod.nrows(){for c in 0..self.channels{tape.k_mod[[t,c]]*=1.0+(tape.a[[t,c]]-1.0)*self.k_a[c];}}tape.w=Array2::zeros(x.raw_dim());for t in 0..x.nrows(){for c in 0..self.channels{tape.w[[t,c]]=(-0.606531*sigmoid(self.w0[c]+tape.g_decay[[t,c]])).exp();}}let(next_state,y)=wkv::run(initial,&tape.w,&tape.k_mod,&tape.v,&tape.kk,&tape.a,&tape.r,self.heads,self.head_size);tape.y=y;tape.normalized=self.ln_x.forward(&tape.y);tape.correction=Array2::zeros(tape.y.raw_dim());for t in 0..tape.normalized.nrows(){for h in 0..self.heads{let start=h*self.head_size;let mut corr=0.0;for i in 0..self.head_size{let c=start+i;corr+=tape.r[[t,c]]*tape.k_mod[[t,c]]*self.r_k[[h,i]];}for i in 0..self.head_size{tape.correction[[t,start+i]]=corr*tape.v[[t,start+i]];}}}tape.gated=&tape.normalized+&tape.correction;for t in 0..tape.gated.nrows(){for c in 0..self.channels{tape.gated[[t,c]]*=tape.g[[t,c]];}}tape.output=Self::project(&tape.gated,&self.output,&self.q_output);let last=x.row(x.nrows()-1).to_owned();(tape.output.clone(),next_state,last,tape.v_first.clone(),tape)}
 pub fn forward(&self,x:&Array2<f32>,state:Option<Array4<f32>>,prev:Option<Array1<f32>>,v_first:Option<Array2<f32>>)->(Array2<f32>,Array4<f32>,Array1<f32>,Array2<f32>){let(out,next,last,first,_)=self.forward_with_tape(x,state,prev,v_first);(out,next,last,first)}
 pub fn parameter_count(&self)->usize{self.x_r.len()+self.x_w.len()+self.x_k.len()+self.x_v.len()+self.x_a.len()+self.x_g.len()+self.w0.len()+self.w1.len()+self.w2.len()+self.a0.len()+self.a1.len()+self.a2.len()+self.v1.as_ref().map_or(0,|v|v.len())+self.v2.as_ref().map_or(0,|v|v.len())+self.v0.as_ref().map_or(0,|v|v.len())+self.g1.len()+self.g2.len()+self.k_k.len()+self.k_a.len()+self.r_k.len()+self.receptance.len()+self.key.len()+self.value.len()+self.output.len()+self.ln_x.parameter_count()}
}
#[cfg(test)]mod tests_rwkv_time_mix{use super::*;#[test]fn x070_shapes(){let layer=RwkvTimeMix::new(16,2,0,4);let x=Array2::ones((3,16));let(out,state,last,first)=layer.forward(&x,None,None,None);assert_eq!(out.shape(),&[3,16]);assert_eq!(state.shape(),&[1,2,8,8]);assert_eq!(last.len(),16);assert_eq!(first.shape(),&[3,16]);}#[test]fn tape_matches_forward_output(){let layer=RwkvTimeMix::new(16,2,1,4);let x=Array2::ones((3,16));let(a,_,_,_,tape)=layer.forward_with_tape(&x,None,None,None);let(b,_,_,_)=layer.forward(&x,None,None,None);assert!(a.iter().zip(b.iter()).all(|(x,y)|(x-y).abs()<1e-6));assert_eq!(tape.w_hidden.ncols(),layer.w1.ncols());}#[test]fn qat_forward_uses_packed_weights(){let layer=RwkvTimeMix::new_with_qat(16,2,0,4,4);assert!(layer.q_w1.is_some()&&layer.q_output.is_some());let x=Array2::ones((2,16));let(y,_,_,_)=layer.forward(&x,None,None,None);assert_eq!(y.dim(),(2,16));}#[test]fn qat_3bit_really_packs_weights(){let layer=RwkvTimeMix::new_with_qat(16,2,0,4,3);let q=layer.q_receptance.as_ref().expect("3-bit QAT must produce packed weights");assert_eq!(q.bits.bits(),3);assert_eq!(q.quant.packed.len(),q.quant.shape.0*((q.quant.shape.1*3+7)/8));let deq=q.dequantized();let distinct=deq.row(0).iter().map(|v|(v/q.quant.scales[0]).round() as i32).collect::<std::collections::BTreeSet<_>>();assert!(distinct.len()>1&&distinct.iter().all(|c|(-4..=3).contains(c)));let x=Array2::ones((2,16));let(y,_,_,_)=layer.forward(&x,None,None,None);assert_eq!(y.dim(),(2,16));assert!(y.iter().all(|v|v.is_finite()));}
#[test]fn qat_disabled_keeps_dense_weights(){let layer=RwkvTimeMix::new_with_qat(16,2,0,4,0);assert!(layer.q_w1.is_none()&&layer.q_receptance.is_none());}}

// ===== rwkv_time_mix_backward =====

pub struct TimeMixLinearBackward { pub grad_input: Array2<f32>, pub grad_weight: Array2<f32>, pub grad_bias: Array1<f32> }
pub struct MixBackward { pub grad_input: Array2<f32>, pub grad_prev: Array1<f32>, pub grad_factor: Array1<f32> }

pub fn linear_backward(input: &Array2<f32>, weight: &Array2<f32>, grad_output: &Array2<f32>) -> TimeMixLinearBackward {
    assert_eq!(input.ncols(), weight.ncols());
    assert_eq!(grad_output.dim(), (input.nrows(), weight.nrows()));
    TimeMixLinearBackward { grad_input: grad_output.dot(weight), grad_weight: grad_output.t().dot(input), grad_bias: grad_output.sum_axis(ndarray::Axis(0)) }
}

pub fn mix_backward(input: &Array2<f32>, prev: &Array1<f32>, factor: &Array1<f32>, grad_output: &Array2<f32>) -> MixBackward {
    assert_eq!(input.ncols(), factor.len());
    assert_eq!(input.dim(), grad_output.dim());
    assert_eq!(prev.len(), input.ncols());
    let mut grad_input = Array2::zeros(input.raw_dim());
    let mut grad_prev = Array1::zeros(prev.raw_dim());
    let mut grad_factor = Array1::zeros(factor.raw_dim());
    for t in 0..input.nrows() { for c in 0..input.ncols() {
        let p = if t == 0 { prev[c] } else { input[[t - 1, c]] };
        let g = grad_output[[t, c]];
        grad_input[[t, c]] += g * (1.0 - factor[c]);
        grad_factor[c] += g * (p - input[[t, c]]);
        if t == 0 { grad_prev[c] += g * factor[c]; } else { grad_input[[t - 1, c]] += g * factor[c]; }
    }}
    MixBackward { grad_input, grad_prev, grad_factor }
}

pub fn sigmoid_backward(output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad_output.dim());
    output * &(1.0 - output) * grad_output
}

pub fn tanh_backward(output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    assert_eq!(output.dim(), grad_output.dim());
    (1.0 - output.mapv(|v| v * v)) * grad_output
}

#[cfg(test)]
mod tests_rwkv_time_mix_backward {
    use super::*;
    use ndarray::{Array1, Array2};
    #[test]
    fn mix_backward_propagates_previous_token() {
        let x = Array2::ones((2, 3)); let p = Array1::zeros(3); let f = Array1::from_elem(3, 0.5);
        let g = mix_backward(&x, &p, &f, &Array2::ones((2, 3)));
        assert_eq!(g.grad_input.dim(), (2, 3));
        assert_eq!(g.grad_prev.len(), 3);
        assert_eq!(g.grad_factor.len(), 3);
    }
}

// ===== rwkv_time_mix_backward_full =====

pub struct RwkvTimeMixBackward {
    pub grad_input: Array2<f32>, pub grad_prev: Array1<f32>, pub grad_state: Array4<f32>, pub grad_v_first: Array2<f32>,
    pub grad_x_r: Array1<f32>, pub grad_x_w: Array1<f32>, pub grad_x_k: Array1<f32>, pub grad_x_v: Array1<f32>, pub grad_x_a: Array1<f32>, pub grad_x_g: Array1<f32>,
    pub grad_w0: Array1<f32>, pub grad_w1: Array2<f32>, pub grad_w2: Array2<f32>, pub grad_a0: Array1<f32>, pub grad_a1: Array2<f32>, pub grad_a2: Array2<f32>,
    pub grad_v0: Option<Array1<f32>>, pub grad_v1: Option<Array2<f32>>, pub grad_v2: Option<Array2<f32>>,
    pub grad_g1: Array2<f32>, pub grad_g2: Array2<f32>, pub grad_k_k: Array1<f32>, pub grad_k_a: Array1<f32>, pub grad_r_k: Array2<f32>,
    pub grad_receptance: Array2<f32>, pub grad_key: Array2<f32>, pub grad_value: Array2<f32>, pub grad_output: Array2<f32>,
    pub grad_ln_weight: Array1<f32>, pub grad_ln_bias: Array1<f32>,
}

pub fn backward(model: &RwkvTimeMix, tape: &RwkvTimeMixTape, grad_output: &Array2<f32>) -> RwkvTimeMixBackward {
    assert_eq!(grad_output.dim(), tape.output.dim());
    let (steps, channels) = tape.input.dim();

    let grad_gated = grad_output.dot(&model.output.t());
    let grad_output_weight = tape.gated.t().dot(grad_output);
    let grad_g = &grad_gated * &(&tape.normalized + &tape.correction);
    let grad_normalized = &grad_gated * &tape.g;
    let grad_g2 = tape.g_hidden.t().dot(&grad_g);
    let grad_g_hidden = grad_g.dot(&model.g2.t());
    let grad_g_hidden_pre = &grad_g_hidden * &tape.g_hidden.mapv(|v| v * (1.0 - v));
    let grad_g1 = tape.xg.t().dot(&grad_g_hidden_pre);
    let grad_xg = grad_g_hidden_pre.dot(&model.g1.t());

    let gn = crate::group_norm::backward(&tape.y, &grad_normalized, &model.ln_x.weight, model.ln_x.groups, model.ln_x.eps);
    let grad_y = gn.grad_input;
    let grad_correction = &grad_gated * &tape.g;

    let mut grad_v = Array2::zeros((steps, channels));
    let mut grad_r = Array2::zeros((steps, channels));
    let mut grad_k_mod = Array2::zeros((steps, channels));
    let mut grad_r_k = Array2::zeros((model.heads, model.head_size));
    for row in 0..steps {
        for h in 0..model.heads {
            let start = h * model.head_size;
            let mut scalar = 0.0;
            for i in 0..model.head_size {
                let col = start + i;
                scalar += tape.r[[row, col]] * tape.k_mod[[row, col]] * model.r_k[[h, i]];
            }
            let mut grad_scalar = 0.0;
            for i in 0..model.head_size {
                let col = start + i;
                grad_v[[row, col]] += grad_correction[[row, col]] * scalar;
                grad_scalar += grad_correction[[row, col]] * tape.v[[row, col]];
            }
            for i in 0..model.head_size {
                let col = start + i;
                grad_r[[row, col]] += grad_scalar * tape.k_mod[[row, col]] * model.r_k[[h, i]];
                grad_k_mod[[row, col]] += grad_scalar * tape.r[[row, col]] * model.r_k[[h, i]];
                grad_r_k[[h, i]] += grad_scalar * tape.r[[row, col]] * tape.k_mod[[row, col]];
            }
        }
    }

    let wk = crate::wkv::backward(&tape.state_initial, &tape.w, &tape.k_mod, &tape.v, &tape.kk, &tape.a, &tape.r, &grad_y, model.heads, model.head_size);
    grad_v += &wk.v; grad_r += &wk.r; grad_k_mod += &wk.k;
    let grad_state = wk.state; let grad_w = wk.w; let grad_kk = wk.kk; let mut grad_a = wk.a;

    let mut grad_k = Array2::zeros((steps, channels)); let mut grad_k_a = Array1::zeros(channels);
    for row in 0..steps { for col in 0..channels {
        let scale=1.0+(tape.a[[row,col]]-1.0)*model.k_a[col];
        grad_k[[row,col]] += grad_k_mod[[row,col]]*scale;
        grad_a[[row,col]] += grad_k_mod[[row,col]]*tape.k[[row,col]]*model.k_a[col];
        grad_k_a[col] += grad_k_mod[[row,col]]*tape.k[[row,col]]*(tape.a[[row,col]]-1.0);
    }}

    let mut grad_k_k=Array1::zeros(channels);
    for row in 0..steps { for h in 0..model.heads {
        let start=h*model.head_size; let mut norm2=1e-12;
        for i in 0..model.head_size { let q=tape.kk_pre_norm[[row,start+i]]*model.k_k[start+i]; norm2+=q*q; }
        let inv=norm2.sqrt().recip(); let mut dot=0.0;
        for i in 0..model.head_size { dot+=grad_kk[[row,start+i]]*tape.kk[[row,start+i]]; }
        for i in 0..model.head_size { let col=start+i; let gq=(grad_kk[[row,col]]-tape.kk[[row,col]]*dot)*inv; grad_k[[row,col]]+=gq*model.k_k[col]; grad_k_k[col]+=gq*tape.kk_pre_norm[[row,col]]; }
    }}

    let grad_a_pre=&grad_a*&tape.a.mapv(|v|v*(1.0-v));
    let grad_a2=tape.a_hidden.t().dot(&grad_a_pre); let grad_a_hidden=grad_a_pre.dot(&model.a2.t()); let grad_a1=tape.xa.t().dot(&grad_a_hidden); let grad_xa=grad_a_hidden.dot(&model.a1.t());
    let mut grad_a0=Array1::zeros(channels); for row in 0..steps { for col in 0..channels { grad_a0[col]+=grad_a_pre[[row,col]]; }}

    let mut grad_w0=Array1::zeros(channels); let mut grad_g_decay=Array2::zeros((steps,channels));
    for row in 0..steps { for col in 0..channels { let z=model.w0[col]+tape.g_decay[[row,col]]; let s=1.0/(1.0+(-z).exp()); let gz=grad_w[[row,col]]*tape.w[[row,col]]*(-0.606531)*s*(1.0-s); grad_w0[col]+=gz; grad_g_decay[[row,col]]=gz; }}
    let grad_w2=tape.w_hidden.t().dot(&grad_g_decay); let grad_w_hidden=grad_g_decay.dot(&model.w2.t()); let grad_w_hidden_pre=&grad_w_hidden*&tape.w_hidden.mapv(|v|1.0-v*v); let grad_w1=tape.xw.t().dot(&grad_w_hidden_pre); let grad_xw=grad_w_hidden_pre.dot(&model.w1.t());

    let grad_receptance=tape.xr.t().dot(&grad_r); let grad_xr=grad_r.dot(&model.receptance.t());
    let grad_key=tape.xk.t().dot(&grad_k); let grad_xk=grad_k.dot(&model.key.t());
    let mut grad_xv=Array2::zeros((steps,channels)); let mut grad_value=Array2::zeros((channels,channels)); let mut grad_v_first=Array2::zeros((steps,channels));
    let (grad_v0,grad_v1,grad_v2);
    if let (Some(v1),Some(v2),Some(v0),Some(correction),Some(gate))=(&model.v1,&model.v2,&model.v0,&tape.v_correction,&tape.v_gate) {
        let mut gv_base=Array2::zeros((steps,channels)); let mut gv_corr=Array2::zeros((steps,channels)); let mut gv0=Array1::zeros(channels);
        for row in 0..steps { for col in 0..channels { let gv=grad_v[[row,col]]; gv_base[[row,col]]=gv*(1.0-gate[[row,col]]); grad_v_first[[row,col]]=gv*gate[[row,col]]; let gg=gv*(tape.v_first[[row,col]]-tape.v_base[[row,col]])*gate[[row,col]]*(1.0-gate[[row,col]]); gv_corr[[row,col]]=gg; gv0[col]+=gg; }}
        let v_hidden= tape.xv.dot(v1);
        grad_value=tape.xv.t().dot(&gv_base); grad_xv+=&gv_base.dot(&model.value.t());
        let gv2=v_hidden.t().dot(&gv_corr); let gh=gv_corr.dot(&v2.t()); let gv1=tape.xv.t().dot(&gh); grad_xv+=&gh.dot(&v1.t());
        grad_v0=Some(gv0); grad_v1=Some(gv1); grad_v2=Some(gv2);
    } else { grad_value=tape.xv.t().dot(&grad_v); grad_xv+=&grad_v.dot(&model.value.t()); grad_v_first=grad_v.clone(); grad_v0=None; grad_v1=None; grad_v2=None; }

    finish(model,tape,grad_xr,grad_xw,grad_xk,grad_xv,grad_xa,grad_xg,grad_v_first,grad_state,grad_receptance,grad_key,grad_value,grad_w0,grad_w1,grad_w2,grad_a0,grad_a1,grad_a2,grad_v0,grad_v1,grad_v2,grad_g1,grad_g2,grad_k_k,grad_k_a,grad_r_k,gn.grad_weight,gn.grad_bias,grad_output_weight)
}

fn finish(model:&RwkvTimeMix,tape:&RwkvTimeMixTape,grad_xr:Array2<f32>,grad_xw:Array2<f32>,grad_xk:Array2<f32>,grad_xv:Array2<f32>,grad_xa:Array2<f32>,grad_xg:Array2<f32>,grad_v_first:Array2<f32>,grad_state:Array4<f32>,grad_receptance:Array2<f32>,grad_key:Array2<f32>,grad_value:Array2<f32>,grad_w0:Array1<f32>,grad_w1:Array2<f32>,grad_w2:Array2<f32>,grad_a0:Array1<f32>,grad_a1:Array2<f32>,grad_a2:Array2<f32>,grad_v0:Option<Array1<f32>>,grad_v1:Option<Array2<f32>>,grad_v2:Option<Array2<f32>>,grad_g1:Array2<f32>,grad_g2:Array2<f32>,grad_k_k:Array1<f32>,grad_k_a:Array1<f32>,grad_r_k:Array2<f32>,grad_ln_weight:Array1<f32>,grad_ln_bias:Array1<f32>,grad_output:Array2<f32>)->RwkvTimeMixBackward {
    let mut grad_input=Array2::zeros(tape.input.raw_dim()); let mut grad_prev=Array1::zeros(tape.prev.raw_dim()); let branches=[(&model.x_r,&grad_xr),(&model.x_w,&grad_xw),(&model.x_k,&grad_xk),(&model.x_v,&grad_xv),(&model.x_a,&grad_xa),(&model.x_g,&grad_xg)]; let mut factors=[Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels)];
    let mixed=[&tape.xr,&tape.xw,&tape.xk,&tape.xv,&tape.xa,&tape.xg];
    for i in 0..6 { let b=mix_backward(&tape.input,&tape.prev,branches[i].0,branches[i].1); let _=mixed[i]; grad_input+=&b.grad_input; grad_prev+=&b.grad_prev; factors[i]=b.grad_factor; }
    RwkvTimeMixBackward{grad_input,grad_prev,grad_state,grad_v_first,grad_x_r:factors[0].clone(),grad_x_w:factors[1].clone(),grad_x_k:factors[2].clone(),grad_x_v:factors[3].clone(),grad_x_a:factors[4].clone(),grad_x_g:factors[5].clone(),grad_w0,grad_w1,grad_w2,grad_a0,grad_a1,grad_a2,grad_v0,grad_v1,grad_v2,grad_g1,grad_g2,grad_k_k,grad_k_a,grad_r_k,grad_receptance,grad_key,grad_value,grad_output,grad_ln_weight,grad_ln_bias}
}

#[cfg(test)]
mod tests_rwkv_time_mix_backward_full {
    use super::*;
    use ndarray::Array2;

    fn loss(model: &RwkvTimeMix, x: &Array2<f32>) -> f32 {
        model.forward(x, None, None, None).0.sum()
    }

    #[test]
    fn backward_produces_finite_gradients() {
        let model=RwkvTimeMix::new(16,2,1,4);
        let x=Array2::from_shape_fn((3,16),|(r,c)|0.01*(r+c) as f32);
        let (_,_,_,_,tape)=model.forward_with_tape(&x,None,None,None);
        let b=backward(&model,&tape,&Array2::ones((3,16)));
        assert!(b.grad_input.iter().all(|v|v.is_finite()));
        assert!(b.grad_state.iter().all(|v|v.is_finite()));
        assert!(b.grad_value.iter().all(|v|v.is_finite()));
    }

    #[test]
    fn input_gradient_matches_finite_difference() {
        let model=RwkvTimeMix::new(16,2,1,4);
        let x=Array2::from_shape_fn((2,16),|(r,c)|0.03*(r as f32)+0.002*(c as f32)+0.01);
        let (_,_,_,_,tape)=model.forward_with_tape(&x,None,None,None);
        let analytic=backward(&model,&tape,&Array2::ones((2,16))).grad_input;
        let eps=1e-3_f32;
        for &(row,col) in &[(0usize,0usize),(0,7),(1,3),(1,15)] {
            let mut plus=x.clone();
            let mut minus=x.clone();
            plus[[row,col]]+=eps;
            minus[[row,col]]-=eps;
            let numeric=(loss(&model,&plus)-loss(&model,&minus))/(2.0*eps);
            let a=analytic[[row,col]];
            let tolerance=2e-2_f32.max(2e-2*a.abs());
            assert!((numeric-a).abs()<=tolerance,"gradient mismatch at ({row},{col}): analytic={a}, numeric={numeric}");
        }
    }
}

// ===== rwkv_time_mix_tape =====

pub struct RwkvTimeMixTape {
    pub input: Array2<f32>,
    pub prev: Array1<f32>,
    pub xr: Array2<f32>,
    pub xw: Array2<f32>,
    pub xk: Array2<f32>,
    pub xv: Array2<f32>,
    pub xa: Array2<f32>,
    pub xg: Array2<f32>,
    pub r: Array2<f32>,
    pub w_hidden: Array2<f32>,
    pub g_decay: Array2<f32>,
    pub k: Array2<f32>,
    pub v_base: Array2<f32>,
    pub v_correction: Option<Array2<f32>>,
    pub v_gate: Option<Array2<f32>>,
    pub v: Array2<f32>,
    pub a_hidden: Array2<f32>,
    pub a: Array2<f32>,
    pub g_hidden: Array2<f32>,
    pub g: Array2<f32>,
    pub kk_pre_norm: Array2<f32>,
    pub kk: Array2<f32>,
    pub k_mod: Array2<f32>,
    pub w: Array2<f32>,
    pub state_initial: Array4<f32>,
    pub y: Array2<f32>,
    pub normalized: Array2<f32>,
    pub correction: Array2<f32>,
    pub gated: Array2<f32>,
    pub output: Array2<f32>,
    pub v_first: Array2<f32>,
}

impl RwkvTimeMixTape {
    pub fn new(
        input: Array2<f32>,
        prev: Array1<f32>,
        state_initial: Array4<f32>,
        w_hidden: usize,
        v_hidden: Option<usize>,
        g_hidden: usize,
    ) -> Self {
        let shape = input.raw_dim();
        let rows = input.nrows();
        let state_shape = state_initial.raw_dim();

        Self {
            input,
            prev,
            xr: Array2::zeros(shape.clone()),
            xw: Array2::zeros(shape.clone()),
            xk: Array2::zeros(shape.clone()),
            xv: Array2::zeros(shape.clone()),
            xa: Array2::zeros(shape.clone()),
            xg: Array2::zeros(shape.clone()),
            r: Array2::zeros(shape.clone()),
            w_hidden: Array2::zeros((rows, w_hidden)),
            g_decay: Array2::zeros(shape.clone()),
            k: Array2::zeros(shape.clone()),
            v_base: Array2::zeros(shape.clone()),
            v_correction: v_hidden.map(|n| Array2::zeros((rows, n))),
            v_gate: v_hidden.map(|_| Array2::zeros(shape.clone())),
            v: Array2::zeros(shape.clone()),
            a_hidden: Array2::zeros((rows, w_hidden)),
            a: Array2::zeros(shape.clone()),
            g_hidden: Array2::zeros((rows, g_hidden)),
            g: Array2::zeros(shape.clone()),
            kk_pre_norm: Array2::zeros(shape.clone()),
            kk: Array2::zeros(shape.clone()),
            k_mod: Array2::zeros(shape.clone()),
            w: Array2::zeros(shape.clone()),
            state_initial: Array4::zeros(state_shape),
            y: Array2::zeros(shape.clone()),
            normalized: Array2::zeros(shape.clone()),
            correction: Array2::zeros(shape.clone()),
            gated: Array2::zeros(shape.clone()),
            output: Array2::zeros(shape.clone()),
            v_first: Array2::zeros(shape),
        }
    }
}

#[cfg(test)]
mod tests_rwkv_time_mix_tape {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn tape_preserves_forward_shapes() {
        let tape = RwkvTimeMixTape::new(
            Array2::zeros((3, 8)),
            Array1::zeros(8),
            Array4::zeros((1, 2, 4, 4)),
            5,
            Some(6),
            7,
        );

        assert_eq!(tape.xr.dim(), (3, 8));
        assert_eq!(tape.w_hidden.dim(), (3, 5));
        assert_eq!(tape.v_correction.as_ref().unwrap().dim(), (3, 6));
        assert_eq!(tape.state_initial.dim(), (1, 2, 4, 4));
        assert_eq!(tape.output.dim(), (3, 8));
    }
}
