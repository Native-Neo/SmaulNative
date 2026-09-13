use crate::layer_norm::LayerNorm;
use crate::rwkv_cmix::RwkvCmix;
use crate::rwkv_time_mix::RwkvTimeMix;
use ndarray::{Array1, Array2, Array4};

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
        let (time_output, next_time_state, next_time_prev, next_v_first) = self.time_mix.forward(
            &time_input,
            time_state,
            time_prev,
            inherited_v_first,
        );
        let residual = &input + &time_output;
        let cmix_input = self.ln2.forward(&residual);
        let cmix_prev = state.map(|s| &s.cmix_prev);
        let (cmix_output, next_cmix_prev) = self.cmix.forward(&cmix_input, cmix_prev);
        let output = &residual + &cmix_output;
        (
            output,
            RwkvBlockState {
                time_state: next_time_state,
                time_prev: next_time_prev,
                cmix_prev: next_cmix_prev,
                v_first: Some(next_v_first),
            },
        )
    }

    pub fn parameter_count(&self) -> usize {
        let norm_params = self.ln0.as_ref().map_or(0, |n| n.weight.len() + n.bias.len())
            + self.ln1.weight.len() + self.ln1.bias.len()
            + self.ln2.weight.len() + self.ln2.bias.len();
        norm_params + self.time_mix.parameter_count() + self.cmix.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::RwkvBlock;
    use ndarray::Array2;

    #[test]
    fn block_runs_and_carries_state() {
        let block = RwkvBlock::new(16, 2, 0, 4);
        let x = Array2::<f32>::zeros((3, 16));
        let (y, state) = block.forward(&x, None, None);
        assert_eq!(y.shape(), &[3, 16]);
        assert_eq!(state.time_prev.len(), 16);
        assert_eq!(state.cmix_prev.len(), 16);
        assert_eq!(state.time_state.shape(), &[1, 2, 8, 8]);
        assert_eq!(state.v_first.as_ref().unwrap().shape(), &[3, 16]);
    }

    #[test]
    fn later_layer_accepts_v_first() {
        let first = RwkvBlock::new(16, 2, 0, 4);
        let later = RwkvBlock::new(16, 2, 1, 4);
        let x = Array2::<f32>::zeros((3, 16));
        let (_, first_state) = first.forward(&x, None, None);
        let (y, later_state) = later.forward(&x, None, first_state.v_first.as_ref());
        assert_eq!(y.shape(), &[3, 16]);
        assert!(later_state.v_first.is_some());
    }

    #[test]
    fn only_first_layer_has_ln0() {
        let first = RwkvBlock::new(16, 2, 0, 4);
        let later = RwkvBlock::new(16, 2, 1, 4);
        assert!(first.ln0.is_some());
        assert!(later.ln0.is_none());
    }
}
