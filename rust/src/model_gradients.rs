use ndarray::{Array1, Array2};
use crate::rwkv_model_backward::RwkvModelBackward;

pub struct LayerGradients {
    pub ln0_weight: Option<Array1<f32>>,
    pub ln0_bias: Option<Array1<f32>>,
    pub ln1_weight: Array1<f32>,
    pub ln1_bias: Array1<f32>,
    pub ln2_weight: Array1<f32>,
    pub ln2_bias: Array1<f32>,
    pub time_x_r: Array1<f32>,
    pub time_x_w: Array1<f32>,
    pub time_x_k: Array1<f32>,
    pub time_x_v: Array1<f32>,
    pub time_x_a: Array1<f32>,
    pub time_x_g: Array1<f32>,
    pub time_w0: Array1<f32>,
    pub time_w1: Array2<f32>,
    pub time_w2: Array2<f32>,
    pub time_a0: Array1<f32>,
    pub time_a1: Array2<f32>,
    pub time_a2: Array2<f32>,
    pub time_v0: Option<Array1<f32>>,
    pub time_v1: Option<Array2<f32>>,
    pub time_v2: Option<Array2<f32>>,
    pub time_g1: Array2<f32>,
    pub time_g2: Array2<f32>,
    pub time_k_k: Array1<f32>,
    pub time_k_a: Array1<f32>,
    pub time_r_k: Array2<f32>,
    pub time_receptance: Array2<f32>,
    pub time_key: Array2<f32>,
    pub time_value: Array2<f32>,
    pub time_output: Array2<f32>,
    pub time_ln_weight: Array1<f32>,
    pub time_ln_bias: Array1<f32>,
    pub cmix_x_k: Array1<f32>,
    pub cmix_key: Array2<f32>,
    pub cmix_value: Array2<f32>,
}

impl LayerGradients {
    pub fn from_backward(block: &crate::rwkv_block_backward_full::RwkvBlockBackward) -> Self {
        let t = &block.time;
        Self {
            ln0_weight: block.grad_ln0_weight.clone(),
            ln0_bias: block.grad_ln0_bias.clone(),
            ln1_weight: block.grad_ln1_weight.clone(), ln1_bias: block.grad_ln1_bias.clone(),
            ln2_weight: block.grad_ln2_weight.clone(), ln2_bias: block.grad_ln2_bias.clone(),
            time_x_r: t.grad_x_r.clone(), time_x_w: t.grad_x_w.clone(), time_x_k: t.grad_x_k.clone(),
            time_x_v: t.grad_x_v.clone(), time_x_a: t.grad_x_a.clone(), time_x_g: t.grad_x_g.clone(),
            time_w0: t.grad_w0.clone(), time_w1: t.grad_w1.clone(), time_w2: t.grad_w2.clone(),
            time_a0: t.grad_a0.clone(), time_a1: t.grad_a1.clone(), time_a2: t.grad_a2.clone(),
            time_v0: t.grad_v0.clone(), time_v1: t.grad_v1.clone(), time_v2: t.grad_v2.clone(),
            time_g1: t.grad_g1.clone(), time_g2: t.grad_g2.clone(), time_k_k: t.grad_k_k.clone(),
            time_k_a: t.grad_k_a.clone(), time_r_k: t.grad_r_k.clone(),
            time_receptance: t.grad_receptance.clone(), time_key: t.grad_key.clone(), time_value: t.grad_value.clone(),
            time_output: t.grad_output.clone(), time_ln_weight: t.grad_ln_weight.clone(), time_ln_bias: t.grad_ln_bias.clone(),
            cmix_x_k: Array1::zeros(block.cmix.grad_x_k.len()), cmix_key: block.cmix.grad_key.clone(), cmix_value: block.cmix.grad_value.clone(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.ln0_weight.as_ref().map_or(0, |x| x.len()) + self.ln0_bias.as_ref().map_or(0, |x| x.len()) +
        self.ln1_weight.len() + self.ln1_bias.len() + self.ln2_weight.len() + self.ln2_bias.len() +
        self.time_x_r.len() + self.time_x_w.len() + self.time_x_k.len() + self.time_x_v.len() + self.time_x_a.len() + self.time_x_g.len() +
        self.time_w0.len() + self.time_w1.len() + self.time_w2.len() + self.time_a0.len() + self.time_a1.len() + self.time_a2.len() +
        self.time_v0.as_ref().map_or(0, |x| x.len()) + self.time_v1.as_ref().map_or(0, |x| x.len()) + self.time_v2.as_ref().map_or(0, |x| x.len()) +
        self.time_g1.len() + self.time_g2.len() + self.time_k_k.len() + self.time_k_a.len() + self.time_r_k.len() +
        self.time_receptance.len() + self.time_key.len() + self.time_value.len() + self.time_output.len() +
        self.time_ln_weight.len() + self.time_ln_bias.len() + self.cmix_x_k.len() + self.cmix_key.len() + self.cmix_value.len()
    }
}

pub struct ModelGradients {
    pub embedding: Array2<f32>,
    pub ln_out_weight: Array1<f32>,
    pub ln_out_bias: Array1<f32>,
    pub head: Array2<f32>,
    pub layers: Vec<LayerGradients>,
}

impl ModelGradients {
    pub fn zeros(vocab_size: usize, channels: usize, n_layer: usize) -> Self {
        Self { embedding: Array2::zeros((vocab_size, channels)), ln_out_weight: Array1::zeros(channels), ln_out_bias: Array1::zeros(channels), head: Array2::zeros((vocab_size, channels)), layers: Vec::with_capacity(n_layer) }
    }

    pub fn from_backward(backward: &RwkvModelBackward) -> Self {
        Self {
            embedding: backward.grad_embedding.clone(), ln_out_weight: backward.grad_ln_out_weight.clone(), ln_out_bias: backward.grad_ln_out_bias.clone(),
            head: backward.grad_head.clone(), layers: backward.blocks.iter().map(|(_, b)| LayerGradients::from_backward(b)).collect(),
        }
    }

    pub fn zero(&mut self) {
        self.embedding.fill(0.0); self.ln_out_weight.fill(0.0); self.ln_out_bias.fill(0.0); self.head.fill(0.0);
        self.layers.clear();
    }

    pub fn parameter_count(&self) -> usize {
        self.embedding.len() + self.ln_out_weight.len() + self.ln_out_bias.len() + self.head.len() + self.layers.iter().map(LayerGradients::parameter_count).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::cross_entropy_backward;
    use crate::rwkv_model::{RwkvModel, RwkvModelConfig};

    #[test]
    fn collects_every_rwkv_layer_gradient() {
        let model = RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4), 123);
        let tokens = [1usize, 2, 3];
        let (logits, tape) = model.forward_with_tape(&tokens);
        let backward = crate::rwkv_model_backward::backward(&model, &tokens, &tape, &cross_entropy_backward(&logits, &tokens));
        let gradients = ModelGradients::from_backward(&backward);
        assert_eq!(gradients.layers.len(), 2);
        assert!(gradients.layers.iter().all(|g| g.parameter_count() > 0));
        assert!(gradients.embedding.iter().all(|v| v.is_finite()));
    }
}
