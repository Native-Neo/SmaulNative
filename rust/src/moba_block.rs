use crate::layer_norm::LayerNorm;
use crate::moba_attention::MobaAttention;
use crate::rwkv_cmix::RwkvCmix;
use ndarray::{Array1, Array2, Array4};

pub struct MobaBlockState {
    pub cmix_prev: Array1<f32>,
    pub att_k: Array4<f32>,
    pub att_v: Array4<f32>,
}

pub struct MobaBlock {
    pub channels: usize,
    pub layer_id: usize,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub att: MobaAttention,
    pub ffn: RwkvCmix,
}

impl MobaBlock {
    pub fn new(
        channels: usize,
        head_size: usize,
        chunk_size: usize,
        top_k: usize,
        layer_id: usize,
        n_layer: usize,
    ) -> Self {
        Self {
            channels,
            layer_id,
            ln1: LayerNorm::new(channels, 1e-5),
            ln2: LayerNorm::new(channels, 1e-5),
            att: MobaAttention::new(channels, head_size, chunk_size, top_k),
            ffn: RwkvCmix::new(channels, layer_id, n_layer),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        state: Option<&MobaBlockState>,
    ) -> (Array2<f32>, MobaBlockState) {
        assert!(!x.is_empty());

        let norm1 = self.ln1.forward(x);
        let (att_out, att_k, att_v) = match state {
            Some(s) if x.nrows() == 1 => self.att.cache_forward(&norm1, &s.att_k, &s.att_v),
            _ => {
                let y = self.att.forward(&norm1);
                let heads = self.att.heads;
                let head_size = self.att.head_size;
                let mut k = Array4::<f32>::zeros((heads, x.nrows(), 1, head_size));
                let mut v = Array4::<f32>::zeros((heads, x.nrows(), 1, head_size));
                let k_proj = self.att.key.forward(&norm1);
                let v_proj = self.att.value.forward(&norm1);
                for h in 0..heads {
                    for t in 0..x.nrows() {
                        for d in 0..head_size {
                            k[[h, t, 0, d]] = k_proj[[t, h * head_size + d]];
                            v[[h, t, 0, d]] = v_proj[[t, h * head_size + d]];
                        }
                    }
                }
                (y, k, v)
            }
        };

        let residual = x + &att_out;
        let norm2 = self.ln2.forward(&residual);
        let (ffn_out, cmix_prev) = self.ffn.forward(
            &norm2,
            state.map(|s| &s.cmix_prev),
        );

        let output = residual + &ffn_out;
        (
            output,
            MobaBlockState {
                cmix_prev,
                att_k,
                att_v,
            },
        )
    }

    pub fn parameter_count(&self) -> usize {
        self.ln1.weight.len()
            + self.ln1.bias.len()
            + self.ln2.weight.len()
            + self.ln2.bias.len()
            + self.att.parameter_count()
            + self.ffn.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::MobaBlock;
    use ndarray::Array2;

    #[test]
    fn forward_preserves_shape_and_creates_state() {
        let block = MobaBlock::new(16, 4, 2, 1, 2, 4);
        let x = Array2::<f32>::zeros((4, 16));
        let (y, state) = block.forward(&x, None);
        assert_eq!(y.dim(), (4, 16));
        assert_eq!(state.cmix_prev.len(), 16);
        assert_eq!(state.att_k.shape(), &[4, 4, 1, 4]);
    }

    #[test]
    fn decode_uses_cached_attention() {
        let block = MobaBlock::new(16, 4, 2, 1, 2, 4);
        let x = Array2::<f32>::zeros((4, 16));
        let (_, state) = block.forward(&x, None);
        let next = Array2::<f32>::zeros((1, 16));
        let (y, next_state) = block.forward(&next, Some(&state));
        assert_eq!(y.dim(), (1, 16));
        assert_eq!(next_state.att_k.shape(), &[4, 5, 1, 4]);
    }
}
