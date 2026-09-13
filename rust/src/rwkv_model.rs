use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::rwkv_block::{RwkvBlock, RwkvBlockState};
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct RwkvModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub head_size: usize,
    pub head_size_divisor: usize,
}

impl RwkvModelConfig {
    pub fn new(vocab_size: usize, n_embd: usize, n_layer: usize, head_size: usize) -> Self {
        assert!(n_layer > 0);
        assert!(head_size > 0);
        assert_eq!(n_embd % head_size, 0);
        Self {
            vocab_size,
            n_embd,
            n_layer,
            head_size,
            head_size_divisor: 8,
        }
    }
}

pub struct RwkvModelState {
    pub blocks: Vec<RwkvBlockState>,
    pub v_first: Option<Array2<f32>>,
}

pub struct RwkvModel {
    pub config: RwkvModelConfig,
    pub embedding: Embedding,
    pub blocks: Vec<RwkvBlock>,
    pub ln_out: LayerNorm,
    pub head: Linear,
}

impl RwkvModel {
    pub fn new(config: RwkvModelConfig, seed: u64) -> Self {
        let embedding = Embedding::new(config.vocab_size, config.n_embd, seed ^ 0x454d_4245_4444_494e);
        let blocks = (0..config.n_layer)
            .map(|layer_id| RwkvBlock::new(config.n_embd, config.n_layer, config.head_size, layer_id, seed ^ (layer_id as u64 + 1)))
            .collect();
        let ln_out = LayerNorm::new(config.n_embd, 1e-5);
        let head = Linear::new(config.n_embd, config.vocab_size, seed ^ 0x4845_4144);
        Self {
            config,
            embedding,
            blocks,
            ln_out,
            head,
        }
    }

    pub fn forward(
        &self,
        token_ids: &[usize],
        state: Option<&RwkvModelState>,
    ) -> (Array2<f32>, RwkvModelState) {
        assert!(!token_ids.is_empty());
        let mut x = self.embedding.forward(token_ids);
        let mut next_blocks = Vec::with_capacity(self.blocks.len());
        let mut v_first = state.and_then(|s| s.v_first.clone());

        for (index, block) in self.blocks.iter().enumerate() {
            let block_state = state.and_then(|s| s.blocks.get(index));
            let (next_x, next_state) = block.forward(&x, block_state, v_first.as_ref());
            v_first = next_state.v_first.clone();
            x = next_x;
            next_blocks.push(next_state);
        }

        x = self.ln_out.forward(&x);
        let logits = self.head.forward(&x);

        (
            logits,
            RwkvModelState {
                blocks: next_blocks,
                v_first,
            },
        )
    }

    pub fn parameter_count(&self) -> usize {
        let mut total = self.embedding.parameter_count();
        total += self.blocks.iter().map(RwkvBlock::parameter_count).sum::<usize>();
        total += self.ln_out.weight.len() + self.ln_out.bias.len();
        total += self.head.parameter_count();
        total
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.head.zero_grad();
    }

    pub fn argmax(logits: &Array2<f32>) -> Array1<usize> {
        let mut result = Array1::zeros(logits.nrows());
        for row in 0..logits.nrows() {
            let mut best_index = 0;
            let mut best_value = f32::NEG_INFINITY;
            for col in 0..logits.ncols() {
                let value = logits[[row, col]];
                if value > best_value {
                    best_value = value;
                    best_index = col;
                }
            }
            result[row] = best_index;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_produces_vocab_logits() {
        let config = RwkvModelConfig::new(32, 16, 2, 4);
        let model = RwkvModel::new(config, 1234);
        let (logits, state) = model.forward(&[1, 2, 3], None);
        assert_eq!(logits.dim(), (3, 32));
        assert_eq!(state.blocks.len(), 2);
        assert_eq!(state.v_first.as_ref().unwrap().dim(), (3, 16));
    }

    #[test]
    fn argmax_selects_each_row() {
        let logits = Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 7.0, 2.0, 3.0, 9.0, 2.0, 8.0, 1.0],
        )
        .unwrap();
        assert_eq!(RwkvModel::argmax(&logits).to_vec(), vec![1, 0]);
    }
}
