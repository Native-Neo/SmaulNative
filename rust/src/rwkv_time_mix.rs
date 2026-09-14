use crate::group_norm::GroupNorm;
use crate::init::{orthogonal, uniform};
use crate::rwkv_time_mix_tape::RwkvTimeMixTape;
use crate::wkv;
use ndarray::{Array1, Array2, Array4};

pub struct RwkvTimeMix {
    pub channels: usize,
    pub heads: usize,
    pub head_size: usize,
    pub x_r: Array1<f32>, pub x_w: Array1<f32>, pub x_k: Array1<f32>,
    pub x_v: Array1<f32>, pub x_a: Array1<f32>, pub x_g: Array1<f32>,
    pub w0: Array1<f32>, pub w1: Array2<f32>, pub w2: Array2<f32>,
    pub a1: Array2<f32>, pub a2: Array2<f32>,
    pub v1: Option<Array2<f32>>, pub v2: Option<Array2<f32>>,
    pub g1: Array2<f32>, pub g2: Array2<f32>, pub a0: Array1<f32>,
    pub v0: Option<Array1<f32>>, pub k_k: Array1<f32>, pub k_a: Array1<f32>,
    pub r_k: Array2<f32>, pub receptance: Array2<f32>, pub key: Array2<f32>,
    pub value: Array2<f32>, pub output: Array2<f32>, pub ln_x: GroupNorm,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn mix(x: &Array2<f32>, prev: &Array1<f32>, factor: &Array1<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for t in 0..x.nrows() { for c in 0..x.ncols() {
        let p = if t == 0 { prev[c] } else { x[[t - 1, c]] };
        out[[t, c]] = x[[t, c]] + (p - x[[t, c]]) * factor[c];
    }}
    out
}

impl RwkvTimeMix {
    pub fn new(channels: usize, heads: usize, layer_id: usize, n_layer: usize) -> Self {
        assert_eq!(channels % heads, 0);
        let head_size = channels / heads;
        let r0 = layer_id as f32 / n_layer.saturating_sub(1).max(1) as f32;
        let r1 = 1.0 - layer_id as f32 / n_layer as f32;
        let dd = (1.8 * (channels as f32).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dm = (1.3 * (channels as f32).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dg = (0.6 * (channels as f32).powf(0.8) / 32.0).round().max(1.0) as usize * 32;
        let mut x_r = Array1::zeros(channels); let mut x_w = Array1::zeros(channels);
        let mut x_k = Array1::zeros(channels); let mut x_v = Array1::zeros(channels);
        let mut x_a = Array1::zeros(channels); let mut x_g = Array1::zeros(channels); let mut w0 = Array1::zeros(channels);
        for c in 0..channels {
            let d = if channels > 1 { c as f32 / (channels - 1) as f32 } else { 0.0 };
            let dc = c as f32 / channels as f32;
            x_r[c] = 1.0 - dc.powf(0.2 * r1); x_w[c] = 1.0 - dc.powf(0.9 * r1);
            x_k[c] = 1.0 - (dc.powf(0.9 * r1) + 0.4 * r0); x_v[c] = 1.0 - (dc.powf(0.4 * r1) + 0.6 * r0);
            x_a[c] = 1.0 - dc.powf(0.9 * r1); x_g[c] = 1.0 - dc.powf(0.2 * r1);
            w0[c] = -7.0 + 5.0 * d.powf(0.85 + r0.sqrt()) + 0.5;
        }
        let w1 = Array2::zeros((channels, dd));
        let a1 = Array2::zeros((channels, dd));
        let g1 = Array2::zeros((channels, dg));
        let v1 = (layer_id > 0).then(|| Array2::zeros((channels, dm)));
        let w2_gain = (dd as f32 / channels as f32).sqrt().max(1.0) * 0.1;
        let a2_gain = w2_gain;
        let g2_gain = (dg as f32 / channels as f32).sqrt().max(1.0) * 0.1;
        let v2_gain = (dm as f32 / channels as f32).sqrt().max(1.0) * 0.1;
        let w2 = orthogonal(dd, channels, w2_gain, 0x5752_5637_3030 ^ layer_id as u64);
        let a2 = orthogonal(dd, channels, a2_gain, 0x4132_3030_3030 ^ layer_id as u64);
        let g2 = orthogonal(dg, channels, g2_gain, 0x4732_3030_3030 ^ layer_id as u64);
        let v2 = (layer_id > 0).then(|| orthogonal(dm, channels, v2_gain, 0x5632_3030_3030 ^ layer_id as u64));
        let lim = 0.5 / (channels as f32).sqrt();
        let receptance = uniform(channels, channels, -lim, lim, 0x5253_3030_3030 ^ layer_id as u64);
        let key = uniform(channels, channels, -0.05 / (channels as f32).sqrt(), 0.05 / (channels as f32).sqrt(), 0x4b59_3030_3030 ^ layer_id as u64);
        let value = uniform(channels, channels, -lim, lim, 0x5641_3030_3030 ^ layer_id as u64);
        Self { channels, heads, head_size, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a1, a2, v1, v2, g1, g2,
            a0: Array1::zeros(channels), v0: (layer_id > 0).then(|| Array1::ones(channels)),
            k_k: Array1::from_elem(channels, 0.85), k_a: Array1::ones(channels), r_k: Array2::zeros((heads, head_size)),
            receptance, key, value, output: Array2::zeros((channels, channels)),
            ln_x: GroupNorm::new(channels, heads, 1e-5 * 8.0 * 8.0) }
    }

    fn project(x: &Array2<f32>, weight: &Array2<f32>) -> Array2<f32> { x.dot(weight) }

    pub fn forward_with_tape(
        &self,
        x: &Array2<f32>,
        state: Option<Array4<f32>>,
        prev: Option<Array1<f32>>,
        v_first: Option<Array2<f32>>,
    ) -> (Array2<f32>, Array4<f32>, Array1<f32>, Array2<f32>, RwkvTimeMixTape) {
        assert!(x.nrows() > 0);
        let zero = Array1::zeros(self.channels);
        let prev_ref = prev.as_ref().unwrap_or(&zero);
        let initial = state.unwrap_or_else(|| Array4::zeros((1, self.heads, self.head_size, self.head_size)));
        let w_hidden = self.w1.ncols();
        let v_hidden = self.v1.as_ref().map(|v| v.ncols());
        let g_hidden = self.g1.ncols();
        let mut tape = RwkvTimeMixTape::new(x.clone(), prev_ref.clone(), initial.clone(), w_hidden, v_hidden, g_hidden);

        tape.xr = mix(x, prev_ref, &self.x_r);
        tape.xw = mix(x, prev_ref, &self.x_w);
        tape.xk = mix(x, prev_ref, &self.x_k);
        tape.xv = mix(x, prev_ref, &self.x_v);
        tape.xa = mix(x, prev_ref, &self.x_a);
        tape.xg = mix(x, prev_ref, &self.x_g);
        tape.r = Self::project(&tape.xr, &self.receptance);
        tape.w_hidden = Self::project(&tape.xw, &self.w1).mapv(|v| v.tanh());
        tape.g_decay = Self::project(&tape.w_hidden, &self.w2);
        tape.k = Self::project(&tape.xk, &self.key);
        tape.v_base = Self::project(&tape.xv, &self.value);
        tape.v = tape.v_base.clone();
        tape.a_hidden = Self::project(&tape.xa, &self.a1);
        tape.a = Self::project(&tape.a_hidden, &self.a2).mapv(sigmoid);
        tape.g_hidden = Self::project(&tape.xg, &self.g1).mapv(sigmoid);
        tape.g = Self::project(&tape.g_hidden, &self.g2);
        tape.v_first = v_first.unwrap_or_else(|| tape.v.clone());

        if let (Some(v1), Some(v2), Some(v0)) = (&self.v1, &self.v2, &self.v0) {
            let correction = Self::project(&Self::project(&tape.xv, v1), v2);
            let mut gate = Array2::zeros(tape.v.dim());
            for t in 0..tape.v.nrows() { for c in 0..self.channels { gate[[t, c]] = sigmoid(v0[c] + correction[[t, c]]); tape.v[[t, c]] += (tape.v_first[[t, c]] - tape.v[[t, c]]) * gate[[t, c]]; }}
            tape.v_correction = Some(correction);
            tape.v_gate = Some(gate);
        }

        tape.kk_pre_norm = tape.k.clone();
        tape.kk = tape.k.clone();
        for t in 0..tape.kk.nrows() { for h in 0..self.heads {
            let start = h * self.head_size; let mut norm = 0.0;
            for i in 0..self.head_size { let z = tape.kk[[t, start+i]] * self.k_k[start+i]; norm += z*z; }
            let inv = (norm + 1e-12).sqrt().recip();
            for i in 0..self.head_size { tape.kk[[t,start+i]] *= self.k_k[start+i] * inv; }
        }}
        tape.k_mod = tape.k.clone();
        for t in 0..tape.k_mod.nrows() { for c in 0..self.channels { tape.k_mod[[t,c]] *= 1.0 + (tape.a[[t,c]]-1.0)*self.k_a[c]; }}
        tape.w = Array2::zeros(x.raw_dim());
        for t in 0..x.nrows() { for c in 0..self.channels { tape.w[[t,c]] = (-0.606531*sigmoid(self.w0[c]+tape.g_decay[[t,c]])).exp(); }}
        let (next_state, y) = wkv::run(initial, &tape.w, &tape.k_mod, &tape.v, &tape.kk, &tape.a, &tape.r, self.heads, self.head_size);
        tape.y = y;
        tape.normalized = self.ln_x.forward(&tape.y);
        tape.correction = Array2::zeros(tape.y.raw_dim());
        for t in 0..tape.normalized.nrows() { for h in 0..self.heads {
            let start=h*self.head_size; let mut corr=0.0;
            for i in 0..self.head_size { let c=start+i; corr += tape.r[[t,c]]*tape.k_mod[[t,c]]*self.r_k[[h,i]]; }
            for i in 0..self.head_size { tape.correction[[t,start+i]] = corr*tape.v[[t,start+i]]; }
        }}
        tape.gated = &tape.normalized + &tape.correction;
        for t in 0..tape.gated.nrows() { for c in 0..self.channels { tape.gated[[t,c]] *= tape.g[[t,c]]; }}
        tape.output = tape.gated.dot(&self.output);
        let last = x.row(x.nrows()-1).to_owned();
        (tape.output.clone(), next_state, last, tape.v_first.clone(), tape)
    }

    pub fn forward(&self, x: &Array2<f32>, state: Option<Array4<f32>>, prev: Option<Array1<f32>>, v_first: Option<Array2<f32>>) -> (Array2<f32>, Array4<f32>, Array1<f32>, Array2<f32>) {
        let (out, next_state, last, first, _) = self.forward_with_tape(x, state, prev, v_first);
        (out, next_state, last, first)
    }

    pub fn parameter_count(&self) -> usize {
        self.x_r.len()+self.x_w.len()+self.x_k.len()+self.x_v.len()+self.x_a.len()+self.x_g.len()+self.w0.len()+self.w1.len()+self.w2.len()+self.a0.len()+self.a1.len()+self.a2.len()+self.v1.as_ref().map_or(0,|v|v.len())+self.v2.as_ref().map_or(0,|v|v.len())+self.v0.as_ref().map_or(0,|v|v.len())+self.g1.len()+self.g2.len()+self.k_k.len()+self.k_a.len()+self.r_k.len()+self.receptance.len()+self.key.len()+self.value.len()+self.output.len()+self.ln_x.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn x070_shapes() {
        let layer=RwkvTimeMix::new(16,2,0,4); let x=Array2::ones((3,16)); let (out,state,last,first)=layer.forward(&x,None,None,None);
        assert_eq!(out.shape(),&[3,16]); assert_eq!(state.shape(),&[1,2,8,8]); assert_eq!(last.len(),16); assert_eq!(first.shape(),&[3,16]);
    }

    #[test]
    fn tape_matches_forward_output() {
        let layer=RwkvTimeMix::new(16,2,1,4); let x=Array2::ones((3,16));
        let (a,_,_,_,tape)=layer.forward_with_tape(&x,None,None,None);
        let (b,_,_,_)=layer.forward(&x,None,None,None);
        assert_eq!(a.dim(), b.dim());
        assert!(a.iter().zip(b.iter()).all(|(x,y)| (x-y).abs() < 1e-6));
        assert_eq!(tape.output.dim(), (3,16));
        assert_eq!(tape.w_hidden.ncols(), layer.w1.ncols());
    }
}
