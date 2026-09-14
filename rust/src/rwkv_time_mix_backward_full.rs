use ndarray::{Array1, Array2, Array4};
use crate::group_norm_backward;
use crate::rwkv_time_mix::RwkvTimeMix;
use crate::rwkv_time_mix_backward::mix_backward;
use crate::rwkv_time_mix_tape::RwkvTimeMixTape;
use crate::wkv_backward;

pub struct RwkvTimeMixBackward {
    pub grad_input: Array2<f32>,
    pub grad_prev: Array1<f32>,
    pub grad_state: Array4<f32>,
    pub grad_v_first: Array2<f32>,
    pub grad_x_r: Array1<f32>,
    pub grad_x_w: Array1<f32>,
    pub grad_x_k: Array1<f32>,
    pub grad_x_v: Array1<f32>,
    pub grad_x_a: Array1<f32>,
    pub grad_x_g: Array1<f32>,
    pub grad_w0: Array1<f32>,
    pub grad_w1: Array2<f32>,
    pub grad_w2: Array2<f32>,
    pub grad_a0: Array1<f32>,
    pub grad_a1: Array2<f32>,
    pub grad_a2: Array2<f32>,
    pub grad_v0: Option<Array1<f32>>,
    pub grad_v1: Option<Array2<f32>>,
    pub grad_v2: Option<Array2<f32>>,
    pub grad_g1: Array2<f32>,
    pub grad_g2: Array2<f32>,
    pub grad_k_k: Array1<f32>,
    pub grad_k_a: Array1<f32>,
    pub grad_r_k: Array2<f32>,
    pub grad_receptance: Array2<f32>,
    pub grad_key: Array2<f32>,
    pub grad_value: Array2<f32>,
    pub grad_output: Array2<f32>,
    pub grad_ln_weight: Array1<f32>,
    pub grad_ln_bias: Array1<f32>,
}

pub fn backward(model: &RwkvTimeMix, tape: &RwkvTimeMixTape, grad_output: &Array2<f32>) -> RwkvTimeMixBackward {
    assert_eq!(grad_output.dim(), tape.output.dim());
    let t = tape.input.nrows();
    let c = model.channels;

    let mut grad_gated = grad_output.dot(&model.output.t());
    let grad_output_weight = tape.gated.t().dot(grad_output);
    let grad_g = &grad_gated * &(&tape.normalized + &tape.correction);
    let mut grad_normalized = &grad_gated * &tape.g;

    let grad_g2 = tape.g_hidden.t().dot(&grad_g);
    let grad_g_hidden = grad_g.dot(&model.g2.t());
    let grad_g_hidden_pre = &grad_g_hidden * &tape.g_hidden.mapv(|v| v * (1.0 - v));
    let grad_g1 = tape.xg.t().dot(&grad_g_hidden_pre);
    let mut grad_xg = grad_g_hidden_pre.dot(&model.g1.t());

    let gn = group_norm_backward::backward(&tape.y, &grad_normalized, &model.ln_x.weight, model.ln_x.groups, model.ln_x.eps);
    let mut grad_y = gn.grad_input;

    let mut grad_correction = &grad_gated * &tape.g;
    let mut grad_v = Array2::zeros((t, c));
    let mut grad_r = Array2::zeros((t, c));
    let mut grad_k_mod = Array2::zeros((t, c));
    let mut grad_r_k = Array2::zeros((model.heads, model.head_size));

    for row in 0..t {
        for h in 0..model.heads {
            let start = h * model.head_size;
            let mut grad_scalar = 0.0;
            for i in 0..model.head_size {
                let ci = start + i;
                grad_v[[row, ci]] += grad_correction[[row, ci]] * tape.correction[[row, ci]] / tape.v[[row, ci]].max(1e-30);
                grad_scalar += grad_correction[[row, ci]] * tape.v[[row, ci]];
            }
            for i in 0..model.head_size {
                let ci = start + i;
                grad_r[[row, ci]] += grad_scalar * tape.k_mod[[row, ci]] * model.r_k[[h, i]];
                grad_k_mod[[row, ci]] += grad_scalar * tape.r[[row, ci]] * model.r_k[[h, i]];
                grad_r_k[[h, i]] += grad_scalar * tape.r[[row, ci]] * tape.k_mod[[row, ci]];
            }
        }
    }
    for row in 0..t { for col in 0..c {
        let mut scalar_grad = 0.0;
        let h = col / model.head_size;
        let start = h * model.head_size;
        for i in 0..model.head_size { scalar_grad += grad_correction[[row, start+i]] * tape.v[[row, start+i]]; }
        grad_v[[row,col]] += 0.0;
        let _ = scalar_grad;
    }}

    let wk = wkv_backward::backward(&tape.state_initial, &tape.w, &tape.k_mod, &tape.v, &tape.kk, &tape.a, &tape.r, &grad_y, model.heads, model.head_size);
    let grad_state = wk.state;
    grad_v += &wk.v;
    grad_r += &wk.r;
    grad_k_mod += &wk.k;
    let mut grad_kk = wk.kk;
    let mut grad_a = wk.a;
    let grad_w = wk.w;

    let mut grad_k = Array2::zeros((t,c));
    for row in 0..t {
        for col in 0..c {
            let scale = 1.0 + (tape.a[[row,col]] - 1.0) * model.k_a[col];
            grad_k[[row,col]] += grad_k_mod[[row,col]] * scale;
            grad_a[[row,col]] += grad_k_mod[[row,col]] * tape.k[[row,col]] * model.k_a[col];
        }
    }
    let mut grad_k_a = Array1::zeros(c);
    for row in 0..t { for col in 0..c { grad_k_a[col] += grad_k_mod[[row,col]] * tape.k[[row,col]] * (tape.a[[row,col]] - 1.0); }}

    let mut grad_k_k = Array1::zeros(c);
    for row in 0..t {
        for h in 0..model.heads {
            let start = h * model.head_size;
            let mut norm2 = 1e-12;
            for i in 0..model.head_size { let q=tape.kk_pre_norm[[row,start+i]]*model.k_k[start+i]; norm2 += q*q; }
            let inv = norm2.sqrt().recip();
            let mut dot = 0.0;
            for i in 0..model.head_size { dot += grad_kk[[row,start+i]] * tape.kk[[row,start+i]]; }
            for i in 0..model.head_size {
                let ci=start+i; let q=tape.kk_pre_norm[[row,ci]]*model.k_k[ci];
                let gq=(grad_kk[[row,ci]] - tape.kk[[row,ci]]*dot)*inv;
                grad_k[[row,ci]] += gq*model.k_k[ci]; grad_k_k[ci] += gq*tape.kk_pre_norm[[row,ci]]; let _=q;
            }
        }
    }

    let grad_a_pre = &grad_a * &tape.a.mapv(|v| v * (1.0-v));
    let grad_a2 = tape.a_hidden.t().dot(&grad_a_pre);
    let grad_a_hidden = grad_a_pre.dot(&model.a2.t());
    let grad_a1 = tape.xa.t().dot(&grad_a_hidden);
    let grad_xa = grad_a_hidden.dot(&model.a1.t());
    let mut grad_a0 = Array1::zeros(c);
    for row in 0..t { for col in 0..c { grad_a0[col] += grad_a_pre[[row,col]]; }}

    let mut grad_w0 = Array1::zeros(c);
    let mut grad_g_decay = Array2::zeros((t,c));
    for row in 0..t { for col in 0..c {
        let z=model.w0[col]+tape.g_decay[[row,col]]; let s=1.0/(1.0+(-z).exp());
        let gz=grad_w[[row,col]]*tape.w[[row,col]]*(-0.606531)*s*(1.0-s);
        grad_w0[col]+=gz; grad_g_decay[[row,col]]=gz;
    }}
    let grad_w2 = tape.w_hidden.t().dot(&grad_g_decay);
    let grad_w_hidden = grad_g_decay.dot(&model.w2.t());
    let grad_w_hidden_pre = &grad_w_hidden * &tape.w_hidden.mapv(|v| 1.0-v*v);
    let grad_w1 = tape.xw.t().dot(&grad_w_hidden_pre);
    let grad_xw = grad_w_hidden_pre.dot(&model.w1.t());

    let grad_receptance = tape.xr.t().dot(&grad_r);
    let grad_xr = grad_r.dot(&model.receptance.t());
    let grad_key = tape.xk.t().dot(&grad_k);
    let grad_xk = grad_k.dot(&model.key.t());

    let mut grad_v_first = Array2::zeros((t,c));
    let mut grad_xv = Array2::zeros((t,c));
    let mut grad_value = Array2::zeros((c,c));
    if let (Some(v1), Some(v2), Some(v0), Some(correction), Some(gate)) = (&model.v1, &model.v2, &model.v0, &tape.v_correction, &tape.v_gate) {
        let mut gv_base = Array2::zeros((t,c));
        let mut gv_corr = Array2::zeros((t,c));
        let mut gv0 = Array1::zeros(c);
        for row in 0..t { for col in 0..c {
            let gv=grad_v[[row,col]]; gv_base[[row,col]] += gv*(1.0-gate[[row,col]]); grad_v_first[[row,col]] += gv*gate[[row,col]];
            let gg=gv*(tape.v_first[[row,col]]-tape.v_base[[row,col]])*gate[[row,col]]*(1.0-gate[[row,col]]); gv_corr[[row,col]]+=gg; gv0[col]+=gg;
        }}
        grad_value = tape.xv.t().dot(&gv_base);
        grad_xv += &gv_base.dot(&model.value.t());
        let grad_v2 = correction.t().dot(&gv_corr);
        let grad_corr_hidden = gv_corr.dot(&v2.t());
        let grad_v1 = tape.xv.t().dot(&grad_corr_hidden);
        grad_xv += &grad_corr_hidden.dot(&v1.t());
        grad_v_first += &Array2::zeros((t,c));
        let mut grad_v0_out=gv0;
        return finish(model,tape,grad_output,grad_xr,grad_xw,grad_xk,grad_xv,grad_xa,grad_xg,grad_v_first,grad_state,grad_receptance,grad_key,grad_value,grad_w0,grad_w1,grad_w2,grad_a0,grad_a1,grad_a2,Some(grad_v0_out),Some(grad_v1),Some(grad_v2),grad_g1,grad_g2,grad_k_k,grad_k_a,grad_r_k,gn.grad_weight,gn.grad_bias,grad_output_weight);
    }
    grad_value = tape.xv.t().dot(&grad_v);
    grad_xv += &grad_v.dot(&model.value.t());
    grad_v_first += &grad_v;
    finish(model,tape,grad_output,grad_xr,grad_xw,grad_xk,grad_xv,grad_xa,grad_xg,grad_v_first,grad_state,grad_receptance,grad_key,grad_value,grad_w0,grad_w1,grad_w2,grad_a0,grad_a1,grad_a2,None,None,None,grad_g1,grad_g2,grad_k_k,grad_k_a,grad_r_k,gn.grad_weight,gn.grad_bias,grad_output_weight)
}

fn finish(
    model:&RwkvTimeMix,tape:&RwkvTimeMixTape,grad_output:&Array2<f32>,grad_xr:Array2<f32>,grad_xw:Array2<f32>,grad_xk:Array2<f32>,grad_xv:Array2<f32>,grad_xa:Array2<f32>,grad_xg:Array2<f32>,grad_v_first:Array2<f32>,grad_state:Array4<f32>,grad_receptance:Array2<f32>,grad_key:Array2<f32>,grad_value:Array2<f32>,grad_w0:Array1<f32>,grad_w1:Array2<f32>,grad_w2:Array2<f32>,grad_a0:Array1<f32>,grad_a1:Array2<f32>,grad_a2:Array2<f32>,grad_v0:Option<Array1<f32>>,grad_v1:Option<Array2<f32>>,grad_v2:Option<Array2<f32>>,grad_g1:Array2<f32>,grad_g2:Array2<f32>,grad_k_k:Array1<f32>,grad_k_a:Array1<f32>,grad_r_k:Array2<f32>,grad_ln_weight:Array1<f32>,grad_ln_bias:Array1<f32>,grad_output_weight:Array2<f32>) -> RwkvTimeMixBackward {
    let mut grad_input=Array2::zeros(tape.input.raw_dim()); let mut grad_prev=Array1::zeros(tape.prev.raw_dim());
    let mixes=[(&tape.xr,&model.x_r,&grad_xr),(&tape.xw,&model.x_w,&grad_xw),(&tape.xk,&model.x_k,&grad_xk),(&tape.xv,&model.x_v,&grad_xv),(&tape.xa,&model.x_a,&grad_xa),(&tape.xg,&model.x_g,&grad_xg)];
    let mut gx=[Array2::zeros(tape.input.raw_dim()),Array2::zeros(tape.input.raw_dim()),Array2::zeros(tape.input.raw_dim()),Array2::zeros(tape.input.raw_dim()),Array2::zeros(tape.input.raw_dim()),Array2::zeros(tape.input.raw_dim())];
    let mut factors=[Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels),Array1::zeros(model.channels)];
    for i in 0..6 { let m=mix_backward(&tape.input,&tape.prev,mixes[i].1,mixes[i].2); gx[i]=m.grad_input; factors[i]=m.grad_factor; grad_prev+=&m.grad_prev; grad_input+=&m.grad_input; }
    let _=grad_output;
    RwkvTimeMixBackward { grad_input,grad_prev,grad_state,grad_v_first,grad_x_r:factors[0].clone(),grad_x_w:factors[1].clone(),grad_x_k:factors[2].clone(),grad_x_v:factors[3].clone(),grad_x_a:factors[4].clone(),grad_x_g:factors[5].clone(),grad_w0,grad_w1,grad_w2,grad_a0,grad_a1,grad_a2,grad_v0,grad_v1,grad_v2,grad_g1,grad_g2,grad_k_k,grad_k_a,grad_r_k,grad_receptance,grad_key,grad_value,grad_output:grad_output_weight,grad_ln_weight,grad_ln_bias }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    #[test]
    fn backward_produces_finite_gradients() {
        let model=RwkvTimeMix::new(16,2,1,4); let x=Array2::from_shape_fn((3,16),|(r,c)|0.01*(r+c) as f32);
        let (_,_,_,_,tape)=model.forward_with_tape(&x,None,None,None); let g=Array2::ones((3,16)); let b=backward(&model,&tape,&g);
        assert!(b.grad_input.iter().all(|v|v.is_finite())); assert!(b.grad_state.iter().all(|v|v.is_finite())); assert!(b.grad_value.iter().all(|v|v.is_finite())); assert!(b.grad_output.iter().all(|v|v.is_finite()));
    }
}
