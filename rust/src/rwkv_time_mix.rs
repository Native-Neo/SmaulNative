use ndarray::{Array1, Array2, Array4};
use crate::wkv;

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
    pub k_k: Array1<f32>,
    pub k_a: Array1<f32>,
    pub r_k: Array2<f32>,
    pub receptance: Array2<f32>,
    pub key: Array2<f32>,
    pub value: Array2<f32>,
    pub output: Array2<f32>,
}

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

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

impl RwkvTimeMix {
    pub fn new(channels: usize, heads: usize, head_size: usize) -> Self {
        assert_eq!(channels, heads * head_size);
        Self {
            channels, heads, head_size,
            x_r: Array1::zeros(channels), x_w: Array1::zeros(channels),
            x_k: Array1::zeros(channels), x_v: Array1::zeros(channels),
            x_a: Array1::zeros(channels), x_g: Array1::zeros(channels),
            w0: Array1::zeros(channels), k_k: Array1::ones(channels),
            k_a: Array1::ones(channels), r_k: Array2::zeros((heads, head_size)),
            receptance: Array2::eye(channels), key: Array2::eye(channels),
            value: Array2::eye(channels), output: Array2::zeros((channels, channels)),
        }
    }

    pub fn forward(&self, x: &Array2<f32>, prev: &Array1<f32>, state: Array4<f32>)
        -> (Array2<f32>, Array4<f32>, Array1<f32>) {
        assert_eq!(x.ncols(), self.channels);
        assert_eq!(prev.len(), self.channels);
        let xr = mix(x, prev, &self.x_r);
        let xw = mix(x, prev, &self.x_w);
        let xk = mix(x, prev, &self.x_k);
        let xv = mix(x, prev, &self.x_v);
        let xa = mix(x, prev, &self.x_a);
        let xg = mix(x, prev, &self.x_g);
        let r = xr.dot(&self.receptance);
        let mut w = xw.dot(&self.key);
        let mut k = xk.dot(&self.key);
        let v = xv.dot(&self.value);
        let a_raw = xa.dot(&self.key);
        let g_raw = xg.dot(&self.key);
        let a = a_raw.mapv(sigmoid);
        for t in 0..x.nrows() {
            for c in 0..self.channels {
                w[[t, c]] = (-0.606531 * sigmoid(self.w0[c] + g_raw[[t, c]])).exp();
                k[[t, c]] *= 1.0 + (a[[t, c]] - 1.0) * self.k_a[c];
            }
        }
        let (next_state, y) = wkv::run(state, &w, &k, &v, &k, &a, &r, self.heads, self.head_size);
        let output = y.dot(&self.output);
        let last = x.row(x.nrows() - 1).to_owned();
        (output, next_state, last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_shapes_are_valid() {
        let layer = RwkvTimeMix::new(4, 2, 2);
        let x = ndarray::array![[1.0, 2.0, 3.0, 4.0]];
        let prev = Array1::zeros(4);
        let state = Array4::zeros((1, 2, 2, 2));
        let (out, next, last) = layer.forward(&x, &prev, state);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(next.shape(), &[1, 2, 2, 2]);
        assert_eq!(last, x.row(0));
    }
}
