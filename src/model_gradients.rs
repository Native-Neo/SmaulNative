use ndarray::{Array1, Array2};
use crate::rwkv_model_backward::RwkvModelBackward;

#[derive(Clone)]
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
    pub moe_router: Option<Array2<f32>>,
    pub moe_x_k: Vec<Array1<f32>>,
    pub moe_key: Vec<Array2<f32>>,
    pub moe_value: Vec<Array2<f32>>,
}

impl LayerGradients {
    pub fn from_backward(block: &crate::rwkv_block_backward_full::RwkvBlockBackward) -> Self {
        let t = &block.time;
        let (cmix_x_k, cmix_key, cmix_value, moe_router, moe_x_k, moe_key, moe_value) =
            match (&block.cmix, &block.moe) {
                (Some(c), None) => (
                    c.grad_x_k.clone(), c.grad_key.clone(), c.grad_value.clone(),
                    None, Vec::new(), Vec::new(), Vec::new(),
                ),
                (None, Some(m)) => (
                    Array1::zeros(0),
                    Array2::zeros((0, 0)),
                    Array2::zeros((0, 0)),
                    Some(m.grad_router.clone()),
                    m.grad_x_k.clone(), m.grad_key.clone(), m.grad_value.clone(),
                ),
                _ => panic!("invalid RWKV CMix backward state"),
            };
        Self {
            ln0_weight: block.grad_ln0_weight.clone(), ln0_bias: block.grad_ln0_bias.clone(),
            ln1_weight: block.grad_ln1_weight.clone(), ln1_bias: block.grad_ln1_bias.clone(),
            ln2_weight: block.grad_ln2_weight.clone(), ln2_bias: block.grad_ln2_bias.clone(),
            time_x_r: t.grad_x_r.clone(), time_x_w: t.grad_x_w.clone(), time_x_k: t.grad_x_k.clone(),
            time_x_v: t.grad_x_v.clone(), time_x_a: t.grad_x_a.clone(), time_x_g: t.grad_x_g.clone(),
            time_w0: t.grad_w0.clone(), time_w1: t.grad_w1.clone(), time_w2: t.grad_w2.clone(),
            time_a0: t.grad_a0.clone(), time_a1: t.grad_a1.clone(), time_a2: t.grad_a2.clone(),
            time_v0: t.grad_v0.clone(), time_v1: t.grad_v1.clone(), time_v2: t.grad_v2.clone(),
            time_g1: t.grad_g1.clone(), time_g2: t.grad_g2.clone(), time_k_k: t.grad_k_k.clone(),
            time_k_a: t.grad_k_a.clone(), time_r_k: t.grad_r_k.clone(), time_receptance: t.grad_receptance.clone(),
            time_key: t.grad_key.clone(), time_value: t.grad_value.clone(), time_output: t.grad_output.clone(),
            time_ln_weight: t.grad_ln_weight.clone(), time_ln_bias: t.grad_ln_bias.clone(),
            cmix_x_k, cmix_key, cmix_value, moe_router, moe_x_k, moe_key, moe_value,
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.ln0_weight.as_ref().map_or(0, |x| x.len()) + self.ln0_bias.as_ref().map_or(0, |x| x.len())
            + self.ln1_weight.len() + self.ln1_bias.len() + self.ln2_weight.len() + self.ln2_bias.len()
            + self.time_x_r.len() + self.time_x_w.len() + self.time_x_k.len() + self.time_x_v.len()
            + self.time_x_a.len() + self.time_x_g.len() + self.time_w0.len() + self.time_w1.len() + self.time_w2.len()
            + self.time_a0.len() + self.time_a1.len() + self.time_a2.len()
            + self.time_v0.as_ref().map_or(0, |x| x.len()) + self.time_v1.as_ref().map_or(0, |x| x.len())
            + self.time_v2.as_ref().map_or(0, |x| x.len()) + self.time_g1.len() + self.time_g2.len()
            + self.time_k_k.len() + self.time_k_a.len() + self.time_r_k.len() + self.time_receptance.len()
            + self.time_key.len() + self.time_value.len() + self.time_output.len()
            + self.time_ln_weight.len() + self.time_ln_bias.len() + self.cmix_x_k.len() + self.cmix_key.len() + self.cmix_value.len()
            + self.moe_router.as_ref().map_or(0, |x| x.len())
            + self.moe_x_k.iter().map(Array1::len).sum::<usize>()
            + self.moe_key.iter().map(Array2::len).sum::<usize>()
            + self.moe_value.iter().map(Array2::len).sum::<usize>()
    }

    pub fn add_in_place(&mut self, o: &Self) {
        macro_rules! add { ($x:expr, $y:expr) => { $x.zip_mut_with($y, |a, b| *a += *b) }; }
        add!(&mut self.ln1_weight, &o.ln1_weight); add!(&mut self.ln1_bias, &o.ln1_bias);
        add!(&mut self.ln2_weight, &o.ln2_weight); add!(&mut self.ln2_bias, &o.ln2_bias);
        add!(&mut self.time_x_r, &o.time_x_r); add!(&mut self.time_x_w, &o.time_x_w); add!(&mut self.time_x_k, &o.time_x_k);
        add!(&mut self.time_x_v, &o.time_x_v); add!(&mut self.time_x_a, &o.time_x_a); add!(&mut self.time_x_g, &o.time_x_g);
        add!(&mut self.time_w0, &o.time_w0); add!(&mut self.time_w1, &o.time_w1); add!(&mut self.time_w2, &o.time_w2);
        add!(&mut self.time_a0, &o.time_a0); add!(&mut self.time_a1, &o.time_a1); add!(&mut self.time_a2, &o.time_a2);
        add!(&mut self.time_g1, &o.time_g1); add!(&mut self.time_g2, &o.time_g2); add!(&mut self.time_k_k, &o.time_k_k);
        add!(&mut self.time_k_a, &o.time_k_a); add!(&mut self.time_r_k, &o.time_r_k); add!(&mut self.time_receptance, &o.time_receptance);
        add!(&mut self.time_key, &o.time_key); add!(&mut self.time_value, &o.time_value); add!(&mut self.time_output, &o.time_output);
        add!(&mut self.time_ln_weight, &o.time_ln_weight); add!(&mut self.time_ln_bias, &o.time_ln_bias);
        add!(&mut self.cmix_x_k, &o.cmix_x_k); add!(&mut self.cmix_key, &o.cmix_key); add!(&mut self.cmix_value, &o.cmix_value);
        if let (Some(a), Some(b)) = (&mut self.ln0_weight, &o.ln0_weight) { add!(a, b); }
        if let (Some(a), Some(b)) = (&mut self.ln0_bias, &o.ln0_bias) { add!(a, b); }
        if let (Some(a), Some(b)) = (&mut self.time_v0, &o.time_v0) { add!(a, b); }
        if let (Some(a), Some(b)) = (&mut self.time_v1, &o.time_v1) { add!(a, b); }
        if let (Some(a), Some(b)) = (&mut self.time_v2, &o.time_v2) { add!(a, b); }
        if let (Some(a), Some(b)) = (&mut self.moe_router, &o.moe_router) { add!(a, b); }
        for (a, b) in self.moe_x_k.iter_mut().zip(&o.moe_x_k) { add!(a, b); }
        for (a, b) in self.moe_key.iter_mut().zip(&o.moe_key) { add!(a, b); }
        for (a, b) in self.moe_value.iter_mut().zip(&o.moe_value) { add!(a, b); }
    }
}

#[derive(Clone)]
pub struct MobaLayerGradients {
    pub ln1_weight: Array1<f32>, pub ln1_bias: Array1<f32>, pub ln2_weight: Array1<f32>, pub ln2_bias: Array1<f32>,
    pub receptance: Array2<f32>, pub key: Array2<f32>, pub value: Array2<f32>, pub output: Array2<f32>,
    pub cmix_x_k: Array1<f32>, pub cmix_key: Array2<f32>, pub cmix_value: Array2<f32>,
}
impl MobaLayerGradients {
    pub fn from_backward(block: &crate::moba_block::MobaBlockBackward) -> Self { Self {
        ln1_weight: block.grad_ln1_weight.clone(), ln1_bias: block.grad_ln1_bias.clone(),
        ln2_weight: block.grad_ln2_weight.clone(), ln2_bias: block.grad_ln2_bias.clone(),
        receptance: block.grad_receptance.clone(), key: block.grad_key.clone(), value: block.grad_value.clone(),
        output: block.grad_output.clone(), cmix_x_k: block.grad_ffn_x_k.clone(), cmix_key: block.grad_ffn_key.clone(),
        cmix_value: block.grad_ffn_value.clone(),
    }}
    pub fn parameter_count(&self) -> usize { self.ln1_weight.len()+self.ln1_bias.len()+self.ln2_weight.len()+self.ln2_bias.len()+self.receptance.len()+self.key.len()+self.value.len()+self.output.len()+self.cmix_x_k.len()+self.cmix_key.len()+self.cmix_value.len() }
    pub fn add_in_place(&mut self, o: &Self) { macro_rules! add { ($x:expr,$y:expr)=>{$x.zip_mut_with($y,|a,b|*a+=*b)} } add!(&mut self.ln1_weight,&o.ln1_weight); add!(&mut self.ln1_bias,&o.ln1_bias); add!(&mut self.ln2_weight,&o.ln2_weight); add!(&mut self.ln2_bias,&o.ln2_bias); add!(&mut self.receptance,&o.receptance); add!(&mut self.key,&o.key); add!(&mut self.value,&o.value); add!(&mut self.output,&o.output); add!(&mut self.cmix_x_k,&o.cmix_x_k); add!(&mut self.cmix_key,&o.cmix_key); add!(&mut self.cmix_value,&o.cmix_value); }
}

#[derive(Clone)]
pub struct ModelGradients { pub embedding: Array2<f32>, pub ln_out_weight: Array1<f32>, pub ln_out_bias: Array1<f32>, pub head: Array2<f32>, pub layers: Vec<LayerGradients>, pub moba_layers: Vec<MobaLayerGradients> }
impl ModelGradients {
    pub fn zeros(vocab_size: usize, channels: usize, n_layer: usize) -> Self { Self { embedding:Array2::zeros((vocab_size,channels)), ln_out_weight:Array1::zeros(channels), ln_out_bias:Array1::zeros(channels), head:Array2::zeros((vocab_size,channels)), layers:Vec::with_capacity(n_layer), moba_layers:Vec::new() } }
    pub fn from_backward(backward:&RwkvModelBackward)->Self { Self { embedding:backward.grad_embedding.clone(), ln_out_weight:backward.grad_ln_out_weight.clone(), ln_out_bias:backward.grad_ln_out_bias.clone(), head:backward.grad_head.clone(), layers:backward.blocks.iter().map(|(_,b)|LayerGradients::from_backward(b)).collect(), moba_layers:backward.moba_blocks.iter().map(|(_,b)|MobaLayerGradients::from_backward(b)).collect() } }
    pub fn zero(&mut self) { self.embedding.fill(0.0); self.ln_out_weight.fill(0.0); self.ln_out_bias.fill(0.0); self.head.fill(0.0); self.layers.clear(); self.moba_layers.clear(); }
    pub fn parameter_count(&self)->usize { self.embedding.len()+self.ln_out_weight.len()+self.ln_out_bias.len()+self.head.len()+self.layers.iter().map(LayerGradients::parameter_count).sum::<usize>()+self.moba_layers.iter().map(MobaLayerGradients::parameter_count).sum::<usize>() }
    pub fn add_in_place(&mut self,o:&Self) { assert_eq!(self.layers.len(),o.layers.len()); assert_eq!(self.moba_layers.len(),o.moba_layers.len()); self.embedding.zip_mut_with(&o.embedding,|a,b|*a+=*b); self.ln_out_weight.zip_mut_with(&o.ln_out_weight,|a,b|*a+=*b); self.ln_out_bias.zip_mut_with(&o.ln_out_bias,|a,b|*a+=*b); self.head.zip_mut_with(&o.head,|a,b|*a+=*b); for(a,b)in self.layers.iter_mut().zip(&o.layers){a.add_in_place(b)} for(a,b)in self.moba_layers.iter_mut().zip(&o.moba_layers){a.add_in_place(b)} }
}
