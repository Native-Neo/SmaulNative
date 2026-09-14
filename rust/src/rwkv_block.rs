use crate::layer_norm::LayerNorm;
use crate::rwkv_cmix::RwkvCmix;
use crate::rwkv_time_mix::RwkvTimeMix;
use crate::rwkv_block_full_tape::RwkvBlockFullTape;
use ndarray::{Array1, Array2, Array4};

#[derive(Clone)]
pub struct RwkvBlockState {
    pub time_state: Array4<f32>,
    pub time_prev: Array1<f32>,
    pub cmix_prev: Array1<f32>,
    pub v_first: Option<Array2<f32>>,
}

pub struct RwkvBlock {
    pub channels: usize,
    pub layer_id: usize,
    pub n_layer: usize,
    pub ln0: Option<LayerNorm>,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub time_mix: RwkvTimeMix,
    pub cmix: RwkvCmix,
}

impl RwkvBlock {
    pub fn new(channels: usize, layer_id: usize, n_layer: usize, seed: u64) -> Self {
        Self { channels, layer_id, n_layer, ln0: if layer_id == 0 { Some(LayerNorm::new(channels, 1e-5)) } else { None }, ln1: LayerNorm::new(channels, 1e-5), ln2: LayerNorm::new(channels, 1e-5), time_mix: RwkvTimeMix::new(channels, layer_id, n_layer, seed), cmix: RwkvCmix::new(channels, layer_id, n_layer) }
    }

    pub fn zero_state(&self) -> RwkvBlockState {
        RwkvBlockState { time_state: Array4::zeros((1, self.channels / 64, 64, 64)), time_prev: Array1::zeros(self.channels), cmix_prev: Array1::zeros(self.channels), v_first: None }
    }

    pub fn forward(&self, input: &Array2<f32>, state: Option<&RwkvBlockState>) -> (Array2<f32>, RwkvBlockState) {
        let normalized = self.ln0.as_ref().map_or_else(|| input.clone(), |ln| ln.forward(input));
        let ln1 = self.ln1.forward(&normalized);
        let (att, time_state, time_prev, v_first) = self.time_mix.forward(&ln1, state.map(|s| &s.time_state), state.map(|s| &s.time_prev), state.and_then(|s| s.v_first.as_ref()));
        let residual = &normalized + &att;
        let ln2 = self.ln2.forward(&residual);
        let (ffn, cmix_prev) = self.cmix.forward(&ln2, state.map(|s| &s.cmix_prev));
        let output = &residual + &ffn;
        (output, RwkvBlockState { time_state, time_prev, cmix_prev, v_first })
    }

    pub fn forward_with_tape(&self, input: &Array2<f32>, state: Option<&RwkvBlockState>) -> (Array2<f32>, RwkvBlockState, RwkvBlockFullTape) {
        let normalized = self.ln0.as_ref().map_or_else(|| input.clone(), |ln| ln.forward(input));
        let ln1 = self.ln1.forward(&normalized);
        let (att, time_state, time_prev, v_first, time_tape) = self.time_mix.forward_with_tape(&ln1, state.map(|s| &s.time_state), state.map(|s| &s.time_prev), state.and_then(|s| s.v_first.as_ref()));
        let residual = &normalized + &att;
        let ln2 = self.ln2.forward(&residual);
        let (ffn, cmix_prev, cmix_tape) = self.cmix.forward_with_tape(&ln2, state.map(|s| &s.cmix_prev));
        let output = &residual + &ffn;
        let next_state = RwkvBlockState { time_state, time_prev, cmix_prev, v_first };
        let mut tape = RwkvBlockFullTape::new(input.clone(), time_tape, next_state.clone());
        tape.cmix_tape = cmix_tape;
        (output, next_state, tape)
    }
}
