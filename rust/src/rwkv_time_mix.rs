use crate::wkv;
use ndarray::{Array1, Array2, Array4};

pub struct RwkvTimeMix {
    pub channels: usize,
    pub heads: usize,
    pub head_size: usize,
    pub x_r: Array1<f32>,
    pub x_w: Array1<f32>,
    pub x_k: Array1<f32>,
    pub x_v: Array1<f32>,
    pub x_a: Array1<f32>,
    pub x_g: Array1<f32>,
    pub w0: Array1<f32>,
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub a1: Array2<f32>,
    pub a2: Array2<f32>,
    pub v1: Option<Array2<f32>>,
    pub v2: Option<Array2<f32>>,
    pub g1: Array2<f32>,
    pub g2: Array2<f32>,
    pub a0: Array1<f32>,
    pub v0: Option<Array1<f32>>,
    pub k_k: Array1<f32>,
    pub k_a: Array1<f32>,
    pub r_k: Array2<f32>,
    pub receptance: Array2<f32>,
    pub key: Array2<f32>,
    pub value: Array2<f32>,
    pub output: Array2<f32>,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn mix(x: &Array2<f32>, prev: &Array1<f32>, factor: &Array1<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for t in 0..x.nrows() {
        for c in 0..x.ncols() {
            let p = if t == 0 { prev[c] } else { x[[t - 1, c]] };
            out[[t, c]] = x[[t, c]] + (p - x[[t, c]]) * factor[c];
        }
    }
    out
}

fn pseudo_random(i: usize, scale: f32) -> f32 {
    ((i as f32 * 12.9898).sin() * 43758.547).fract() * 2.0 * scale - scale
}

impl RwkvTimeMix {
    pub fn new(channels: usize, heads: usize, layer_id: usize, n_layer: usize) -> Self {
        assert_eq!(channels % heads, 0);
        let head_size = channels / heads;
        let r0 = layer_id as f32 / (n_layer.saturating_sub(1).max(1) as f32);
        let r1 = 1.0 - layer_id as f32 / n_layer as f32;
        let dd = (1.8 * (channels as f32).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dm = (1.3 * (channels as f32).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dg = (0.6 * (channels as f32).powf(0.8) / 32.0).round().max(1.0) as usize * 32;

        let mut x_r = Array1::zeros(channels);
        let mut x_w = Array1::zeros(channels);
        let mut x_k = Array1::zeros(channels);
        let mut x_v = Array1::zeros(channels);
        let mut x_a = Array1::zeros(channels);
        let mut x_g = Array1::zeros(channels);
        let mut w0 = Array1::zeros(channels);
        for c in 0..channels {
            let d = c as f32 / channels as f32;
            x_r[c] = 1.0 - d.powf(0.2 * r1);
            x_w[c] = 1.0 - d.powf(0.9 * r1);
            x_k[c] = 1.0 - (d.powf(0.9 * r1) + 0.4 * r0);
            x_v[c] = 1.0 - (d.powf(0.4 * r1) + 0.6 * r0);
            x_a[c] = 1.0 - d.powf(0.9 * r1);
            x_g[c] = 1.0 - d.powf(0.2 * r1);
            w0[c] = -7.0 + 5.0 * d.powf(0.85 + r0.sqrt()) + 0.5;
        }

        let mut w1 = Array2::zeros((channels, dd));
        let mut w2 = Array2::zeros((dd, channels));
        let mut a1 = Array2::zeros((channels, dd));
        let mut a2 = Array2::zeros((dd, channels));
        let mut g1 = Array2::zeros((channels, dg));
        let mut g2 = Array2::zeros((dg, channels));
        for i in 0..channels * dd { w1[[i / dd, i % dd]] = 0.0; a1[[i / dd, i % dd]] = 0.0; }
        for i in 0..dd * channels { w2[[i / channels, i % channels]] = pseudo_random(i + 11, 0.1); a2[[i / channels, i % channels]] = pseudo_random(i + 31, 0.1); }
        for i in 0..dg * channels { g2[[i / channels, i % channels]] = pseudo_random(i + 71, 0.1); }
        let mut v1 = Array2::zeros((channels, dm));
        let mut v2 = Array2::zeros((dm, channels));
        for i in 0..dm * channels { v2[[i / channels, i % channels]] = pseudo_random(i + 101, 0.1); }

        Self {
            channels, heads, head_size, x_r, x_w, x_k, x_v, x_a, x_g, w0,
            w1, w2, a1, a2, v1: (layer_id > 0).then_some(v1), v2: (layer_id > 0).then_some(v2),
            g1, g2, a0: Array1::zeros(channels), v0: (layer_id > 0).then_some(Array1::ones(channels)),
            k_k: Array1::from_elem(channels, 0.85), k_a: Array1::ones(channels),
            r_k: Array2::zeros((heads, head_size)),
            receptance: Array2::eye(channels), key: Array2::eye(channels),
            value: Array2::eye(channels), output: Array2::zeros((channels, channels)),
        }
    }

    fn project(x: &Array2<f32>, weight: &Array2<f32>) -> Array2<f32> { x.dot(weight) }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        state: Option<Array4<f32>>,
        prev: Option<Array1<f32>>,
    ) -> (Array2<f32>, Array4<f32>, Array1<f32>) {
        let zero = Array1::zeros(self.channels);
        let prev_ref = prev.as_ref().unwrap_or(&zero);
        let xr = mix(x, prev_ref, &self.x_r);
        let xw = mix(x, prev_ref, &self.x_w);
        let xk = mix(x, prev_ref, &self.x_k);
        let xv = mix(x, prev_ref, &self.x_v);
        let xa = mix(x, prev_ref, &self.x_a);
        let xg = mix(x, prev_ref, &self.x_g);

        let r = project(&xr, &self.receptance);
        let g_hidden = project(&xw, &self.w1).mapv(|v| v.tanh());
        let g_ = project(&g_hidden, &self.w2);
        let k = project(&xk, &self.key);
        let mut v = project(&xv, &self.value);
        let a_hidden = project(&xa, &self.a1);
        let a = project(&a_hidden, &self.a2).mapv(sigmoid);
        let g_hidden = project(&xg, &self.g1);
        let g = project(&g_hidden, &self.g2).mapv(sigmoid);

        let v_first = v.clone();
        if self.v1.is_some() {
            let v1 = self.v1.as_ref().unwrap();
            let v2 = self.v2.as_ref().unwrap();
            let h = project(&xv, v1);
            let correction = project(&h, v2);
            for t in 0..v.nrows() { for c in 0..self.channels { v[[t,c]] += (v_first[[t,c]] - v[[t,c]]) * sigmoid(self.v0.as_ref().unwrap()[c] + correction[[t,c]]); } }
        }

        let mut kk = k.clone();
        for t in 0..kk.nrows() {
            for h in 0..self.heads {
                let start = h * self.head_size;
                let mut norm = 0.0;
                for i in 0..self.head_size { let z = kk[[t,start+i]] * self.k_k[start+i]; norm += z*z; }
                let inv = (norm + 1e-12).sqrt().recip();
                for i in 0..self.head_size { kk[[t,start+i]] = kk[[t,start+i]] * self.k_k[start+i] * inv; }
            }
        }
        let mut k_mod = k.clone();
        for t in 0..k_mod.nrows() { for c in 0..self.channels { k_mod[[t,c]] *= 1.0 + (a[[t,c]] - 1.0) * self.k_a[c]; } }
        let mut w = Array2::zeros(x.raw_dim());
        for t in 0..x.nrows() { for c in 0..self.channels { w[[t,c]] = (-0.606531 * sigmoid(self.w0[c] + g_[[t,c]])).exp(); } }

        let initial = state.unwrap_or_else(|| Array4::zeros((1, self.heads, self.head_size, self.head_size)));
        let (next_state, y) = wkv::run(initial, &w, &k_mod, &v, &kk, &a, &r, self.heads, self.head_size);
        let mut out = y.dot(&self.output);
        for t in 0..out.nrows() {
            for c in 0..self.channels { out[[t,c]] *= sigmoid(g[[t,c]]); }
        }
        (out, next_state, x.row(x.nrows()-1).to_owned())
    }

    pub fn parameter_count(&self) -> usize {
        self.w1.len() + self.w2.len() + self.a1.len() + self.a2.len()
            + self.g1.len() + self.g2.len() + self.receptance.len() + self.key.len()
            + self.value.len() + self.output.len() + self.r_k.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn x070_shapes() {
        let layer = RwkvTimeMix::new(16, 2, 0, 4);
        let x = ndarray::Array2::ones((3, 16));
        let (out, state, last) = layer.forward(&x, None, None);
        assert_eq!(out.shape(), &[3, 16]);
        assert_eq!(state.shape(), &[1, 2, 8, 8]);
        assert_eq!(last.len(), 16);
    }
}
