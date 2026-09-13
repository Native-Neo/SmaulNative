use crate::layer_norm::LayerNorm;
use crate::rwkv_cmix::RwkvCmix;
use crate::rwkv_time_mix::RwkvTimeMix;
use ndarray::{Array1, Array2, Array4};

pub struct RwkvBlockState {
    pub time_state: Array4<f32>,
    pub time_prev: Array1<f32>,
    pub cmix_prev: Array1<f32>,
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
            time_mix: RwkvTimeMix::new(channels, heads, channels / heads),
            cmix: RwkvCmix::new(channels, layer_id, n_layer),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        state: Option<&RwkvBlockState>,
    ) -> (Array2<f32>, RwkvBlockState) {
        assert_eq!(x.ncols(), self.channels);

        let input = match &self.ln0 {
            Some(norm) => norm.forward(x),
            None => x.clone(),
        };

        let time_input = self.ln1.forward(&input);
        let zero_prev = Array1::<f32>::zeros(self.channels);
        let zero_state = Array4::<f32>::zeros((1, self.heads, self.channels / self.heads, self.channels / self.heads));
        let time_prev = state.map(|s| &s.time_prev).unwrap_or(&zero_prev);
        let time_state = state.map(|s| s.time_state.clone()).unwrap_or(zero_state);

        let (time_output, next_time_state, next_time_prev) =
            self.time_mix.forward(&time_input, time_prev, time_state);

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
            },
        )
    }

    pub fn parameter_count(&self) -> usize {
        let norm_params = self.ln0.as_ref().map_or(0, |n| n.weight.len() + n.bias.len())
            + self.ln1.weight.len() + self.ln1.bias.len()
            + self.ln2.weight.len() + self.ln2.bias.len();
        norm_params + self.time_mix.receptance.len()
            + self.time_mix.key.len()
            + self.time_mix.value.len()
            + self.time_mix.output.len()
            + self.time_mix.x_r.len()
            + self.time_mix.x_w.len()
            + self.time_mix.x_k.len()
            + self.time_mix.x_v.len()
            + self.time_mix.x_a.len()
            + self.time_mix.x_g.len()
            + self.time_mix.w0.len()
            + self.time_mix.k_k.len()
            + self.time_mix.k_a.len()
            + self.time_mix.r_k.len()
            + self.cmix.parameter_count()
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
        let (y, state) = block.forward(&x, None);
        assert_eq!(y.shape(), &[3, 16]);
        assert_eq!(state.time_prev.len(), 16);
        assert_eq!(state.cmix_prev.len(), 16);
        assert_eq!(state.time_state.shape(), &[1, 2, 8, 8]);
    }

    #[test]
    fn only_first_layer_has_ln0() {
        let first = RwkvBlock::new(16, 2, 0, 4);
        let later = RwkvBlock::new(16, 2, 1, 4);
        assert!(first.ln0.is_some());
        assert!(later.ln0.is_none());
    }
}
