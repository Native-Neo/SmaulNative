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
    pub heads: usize,
    pub layer_id: usize,
    pub ln0: Option<LayerNorm>,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub time_mix: RwkvTimeMix,
    pub cmix: RwkvCmix,
}

impl RwkvBlock {
    pub fn new(channels: usize, heads: usize, layer_id: usize, n_layer: usize) -> Self {
        assert!(channels > 0);
        assert!(heads > 0);
        assert_eq!(channels % heads, 0);
        Self {
            channels,
            heads,
            layer_id,
            ln0: if layer_id == 0 { Some(LayerNorm::new(channels, 1e-5)) } else { None },
            ln1: LayerNorm::new(channels, 1e-5),
            ln2: LayerNorm::new(channels, 1e-5),
            time_mix: RwkvTimeMix::new(channels, heads, layer_id, n_layer),
            cmix: RwkvCmix::new(channels, layer_id, n_layer),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        state: Option<&RwkvBlockState>,
        v_first: Option<&Array2<f32>>,
    ) -> (Array2<f32>, RwkvBlockState) {
        assert_eq!(x.ncols(), self.channels);
        let input = match &self.ln0 { Some(norm) => norm.forward(x), None => x.clone() };
        let time_input = self.ln1.forward(&input);
        let time_state = state.map(|s| s.time_state.clone());
        let time_prev = state.map(|s| s.time_prev.clone());
        let inherited_v_first = v_first.cloned().or_else(|| state.and_then(|s| s.v_first.clone()));
        let (time_output, next_time_state, next_time_prev, next_v_first) = self.time_mix.forward(&time_input, time_state, time_prev, inherited_v_first);
        let residual = &input + &time_output;
        let cmix_input = self.ln2.forward(&residual);
        let cmix_prev = state.map(|s| &s.cmix_prev);
        let (cmix_output, next_cmix_prev) = self.cmix.forward(&cmix_input, cmix_prev);
        let output = &residual + &cmix_output;
        (output, RwkvBlockState { time_state: next_time_state, time_prev: next_time_prev, cmix_prev: next_cmix_prev, v_first: Some(next_v_first) })
    }

    pub fn forward_with_full_tape(&self, x: &Array2<f32>, state: Option<&RwkvBlockState>, v_first: Option<&Array2<f32>>) -> (Array2<f32>, RwkvBlockState, RwkvBlockFullTape) {
        assert_eq!(x.ncols(), self.channels);
        let input = x.clone();
        let (ln0_input, ln0_output) = match &self.ln0 { Some(norm) => (Some(input.clone()), Some(norm.forward(&input))), None => (None, None) };
        let block_input = ln0_output.clone().unwrap_or_else(|| input.clone());
        let ln1_input = block_input.clone();
        let ln1_output = self.ln1.forward(&ln1_input);
        let time_state = state.map(|s| s.time_state.clone());
        let time_prev = state.map(|s| s.time_prev.clone());
        let inherited_v_first = v_first.cloned().or_else(|| state.and_then(|s| s.v_first.clone()));
        let (time_output, next_time_state, next_time_prev, next_v_first, time_tape) = self.time_mix.forward_with_tape(&ln1_output, time_state, time_prev, inherited_v_first);
        let residual = &block_input + &time_output;
        let ln2_input = residual.clone();
        let ln2_output = self.ln2.forward(&ln2_input);
        let cmix_prev = state.map(|s| s.cmix_prev.clone()).unwrap_or_else(|| Array1::zeros(self.channels));
        let mut cmix_mixed = ln2_output.clone();
        for t in 0..cmix_mixed.nrows() { for c in 0..self.channels { let previous = if t == 0 { cmix_prev[c] } else { ln2_output[[t - 1, c]] }; cmix_mixed[[t, c]] = ln2_output[[t, c]] + (previous - ln2_output[[t, c]]) * self.cmix.x_k[c]; } }
        let cmix_pre = cmix_mixed.dot(&self.cmix.key);
        let cmix_hidden = cmix_pre.mapv(|v| v.max(0.0).powi(2));
        let cmix_output = cmix_hidden.dot(&self.cmix.value);
        let output = &residual + &cmix_output;
        let next_state = RwkvBlockState { time_state: next_time_state, time_prev: next_time_prev, cmix_prev: ln2_output.row(ln2_output.nrows() - 1).to_owned(), v_first: Some(next_v_first) };
        let mut tape = RwkvBlockFullTape::new(input, time_tape, next_state.clone());
        tape.initial_state = state.map(|s| RwkvBlockState { time_state: s.time_state.clone(), time_prev: s.time_prev.clone(), cmix_prev: s.cmix_prev.clone(), v_first: s.v_first.clone() });
        tape.ln0_input = ln0_input; tape.ln0_output = ln0_output; tape.ln1_input = ln1_input; tape.ln1_output = ln1_output; tape.residual = residual; tape.ln2_input = ln2_input; tape.ln2_output = ln2_output; tape.cmix_input = tape.ln2_output.clone(); tape.cmix_prev = cmix_prev; tape.cmix_mixed = cmix_mixed; tape.cmix_pre = cmix_pre; tape.cmix_hidden = cmix_hidden; tape.cmix_output = cmix_output; tape.output = output.clone();
        (output, next_state, tape)
    }

    pub fn parameter_count(&self) -> usize { let norm_params = self.ln0.as_ref().map_or(0, |n| n.weight.len() + n.bias.len()) + self.ln1.weight.len() + self.ln1.bias.len() + self.ln2.weight.len() + self.ln2.bias.len(); norm_params + self.time_mix.parameter_count() + self.cmix.parameter_count() }
}

#[cfg(test)]
mod tests {
    use super::RwkvBlock;
    use ndarray::Array2;
    #[test] fn block_runs_and_carries_state() { let block = RwkvBlock::new(16, 2, 0, 4); let x = Array2::<f32>::zeros((3, 16)); let (y, state) = block.forward(&x, None, None); assert_eq!(y.shape(), &[3, 16]); assert_eq!(state.time_prev.len(), 16); assert_eq!(state.cmix_prev.len(), 16); assert_eq!(state.time_state.shape(), &[1, 2, 8, 8]); assert_eq!(state.v_first.as_ref().unwrap().shape(), &[3, 16]); }
    #[test] fn full_tape_matches_forward_shape() { let block = RwkvBlock::new(16, 2, 0, 4); let x = Array2::<f32>::ones((3, 16)); let (a, _, tape) = block.forward_with_full_tape(&x, None, None); let (b, _) = block.forward(&x, None, None); assert_eq!(a.dim(), b.dim()); assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)); assert_eq!(tape.cmix_pre.dim(), (3, 64)); }
    #[test] fn later_layer_accepts_v_first() { let first = RwkvBlock::new(16, 2, 0, 4); let later = RwkvBlock::new(16, 2, 1, 4); let x = Array2::<f32>::zeros((3, 16)); let (_, first_state) = first.forward(&x, None, None); let (y, later_state) = later.forward(&x, None, first_state.v_first.as_ref()); assert_eq!(y.shape(), &[3, 16]); assert!(later_state.v_first.is_some()); }
    #[test] fn only_first_layer_has_ln0() { let first = RwkvBlock::new(16, 2, 0, 4); let later = RwkvBlock::new(16, 2, 1, 4); assert!(first.ln0.is_some()); assert!(later.ln0.is_none()); }
}
