use crate::loss::logits_cross_entropy_backward;
use crate::moba_block::{MobaBlockBackward, MobaBlockTape};
use crate::rwkv_block::{RwkvBlockBackward, RwkvBlockFullTape, RwkvBlockState};
use crate::rwkv_model::{RwkvModel, RwkvModelState};

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackwardBlockKind { Rwkv(usize), Moba(usize) }

pub struct ModelBackwardTape {
    pub block_order: Vec<BackwardBlockKind>,
    pub inputs: Vec<Array2<f32>>,
    pub outputs: Vec<Array2<f32>>,
    pub rwkv_tapes: Vec<Option<RwkvBlockFullTape>>,
    pub moba_tapes: Vec<Option<MobaBlockTape>>,
    pub ln_input: Option<Array2<f32>>,
    pub normalized: Option<Array2<f32>>,
    pub logits: Option<Array2<f32>>,
}

impl ModelBackwardTape {
    pub fn new(block_order: Vec<BackwardBlockKind>) -> Self {
        let len = block_order.len();
        Self {
            rwkv_tapes: (0..len).map(|_| None).collect(),
            moba_tapes: (0..len).map(|_| None).collect(),
            block_order,
            inputs: Vec::new(),
            outputs: Vec::new(),
            ln_input: None,
            normalized: None,
            logits: None,
        }
    }

    pub fn record_block(&mut self, input: Array2<f32>, output: Array2<f32>) {
        assert_eq!(input.dim(), output.dim());
        self.inputs.push(input);
        self.outputs.push(output);
        assert!(self.inputs.len() <= self.block_order.len());
    }

    pub fn record_rwkv_tape(&mut self, tape: RwkvBlockFullTape) {
        let index = self.inputs.len();
        assert!(index > 0 && index <= self.block_order.len());
        let slot = index - 1;
        assert!(matches!(self.block_order[slot], BackwardBlockKind::Rwkv(_)));
        assert!(self.rwkv_tapes[slot].is_none());
        self.rwkv_tapes[slot] = Some(tape);
    }

    pub fn record_moba_tape(&mut self, tape: MobaBlockTape) {
        let index = self.inputs.len();
        assert!(index > 0 && index <= self.block_order.len());
        let slot = index - 1;
        assert!(matches!(self.block_order[slot], BackwardBlockKind::Moba(_)));
        assert!(self.moba_tapes[slot].is_none());
        self.moba_tapes[slot] = Some(tape);
    }

    pub fn record_head(&mut self, ln_input: Array2<f32>, normalized: Array2<f32>, logits: Array2<f32>) {
        assert_eq!(ln_input.dim(), normalized.dim());
        assert_eq!(normalized.nrows(), logits.nrows());
        self.ln_input = Some(ln_input);
        self.normalized = Some(normalized);
        self.logits = Some(logits);
    }

    pub fn reverse_blocks(&self) -> impl DoubleEndedIterator<Item = (BackwardBlockKind, &Array2<f32>, &Array2<f32>)> {
        assert_eq!(self.block_order.len(), self.inputs.len());
        assert_eq!(self.inputs.len(), self.outputs.len());
        self.block_order.iter().copied().zip(self.inputs.iter()).zip(self.outputs.iter()).map(|((kind, input), output)| (kind, input, output)).rev()
    }

    pub fn len(&self) -> usize { self.inputs.len() }
    pub fn is_empty(&self) -> bool { self.inputs.is_empty() }

    pub fn clear(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
        for tape in &mut self.rwkv_tapes { *tape = None; }
        for tape in &mut self.moba_tapes { *tape = None; }
        self.ln_input = None;
        self.normalized = None;
        self.logits = None;
    }
}

#[cfg(test)]
mod tests_model_backward {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn reverse_order_matches_model_execution_order() {
        let order = vec![BackwardBlockKind::Rwkv(0), BackwardBlockKind::Moba(0), BackwardBlockKind::Rwkv(1)];
        let mut tape = ModelBackwardTape::new(order);
        tape.record_block(Array2::zeros((1, 2)), Array2::ones((1, 2)));
        tape.record_block(Array2::ones((1, 2)), Array2::from_elem((1, 2), 2.0));
        tape.record_block(Array2::from_elem((1, 2), 2.0), Array2::from_elem((1, 2), 3.0));
        let kinds: Vec<_> = tape.reverse_blocks().map(|x| x.0).collect();
        assert_eq!(kinds, vec![BackwardBlockKind::Rwkv(1), BackwardBlockKind::Moba(0), BackwardBlockKind::Rwkv(0)]);
    }

    #[test]
    fn records_layer_norm_input_for_backward() {
        let mut tape = ModelBackwardTape::new(Vec::new());
        let input = Array2::ones((2, 3));
        let normalized = Array2::from_elem((2, 3), 2.0);
        let logits = Array2::zeros((2, 4));
        tape.record_head(input.clone(), normalized.clone(), logits.clone());
        assert_eq!(tape.ln_input.as_ref().unwrap(), &input);
        assert_eq!(tape.normalized.as_ref().unwrap(), &normalized);
        assert_eq!(tape.logits.as_ref().unwrap(), &logits);
    }
}

// ===== model_head_backward =====

pub struct HeadBackward {
    pub grad_input: Array2<f32>,
    pub grad_weight: Array2<f32>,
    pub grad_ln_weight: Array1<f32>,
    pub grad_ln_bias: Array1<f32>,
    pub grad_embedding: Array2<f32>,
}

pub fn head_backward(
    token_ids: &[usize],
    normalized_input: &Array2<f32>,
    logits_grad: &Array2<f32>,
    head_weight: &Array2<f32>,
    ln_input: &Array2<f32>,
    ln_weight: &Array1<f32>,
    ln_eps: f32,
    vocab_size: usize,
) -> HeadBackward {
    assert_eq!(normalized_input.nrows(), logits_grad.nrows());
    assert_eq!(normalized_input.ncols(), head_weight.ncols());
    assert_eq!(logits_grad.ncols(), head_weight.nrows());
    assert_eq!(ln_input.dim(), normalized_input.dim());
    assert_eq!(ln_weight.len(), normalized_input.ncols());

    let grad_weight = logits_grad.t().dot(normalized_input);
    let grad_normalized = logits_grad.dot(head_weight);
    let (grad_input, grad_ln_weight, grad_ln_bias) =
        crate::layer_norm::backward(ln_input, &grad_normalized, ln_weight, ln_eps);
    let grad_embedding = crate::embedding::backward(token_ids, &grad_input, vocab_size);

    HeadBackward { grad_input, grad_weight, grad_ln_weight, grad_ln_bias, grad_embedding }
}

pub fn accumulate_parameter_gradient(target: &mut Array2<f32>, source: &Array2<f32>) {
    assert_eq!(target.dim(), source.dim());
    *target += source;
}

pub fn accumulate_vector_gradient(target: &mut Array1<f32>, source: &Array1<f32>) {
    assert_eq!(target.len(), source.len());
    *target += source;
}

#[cfg(test)]
mod tests_model_head_backward {
    use super::*;
    use ndarray::array;

    #[test]
    fn computes_head_and_embedding_gradients() {
        let tokens = [1usize, 2, 1];
        let normalized = array![[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]];
        let logits_grad = array![[1.0, 0.0, -1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]];
        let head = array![[0.2, 0.3], [0.4, -0.1], [0.5, 0.6]];
        let gamma = Array1::ones(2);
        let grads = head_backward(&tokens, &normalized, &logits_grad, &head, &normalized, &gamma, 1e-5, 4);
        assert_eq!(grads.grad_weight.dim(), (3, 2));
        assert_eq!(grads.grad_input.dim(), (3, 2));
        assert_eq!(grads.grad_embedding.dim(), (4, 2));
        assert!(grads.grad_embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn accumulation_adds_in_place() {
        let mut matrix = Array2::zeros((2, 2));
        let source = Array2::ones((2, 2));
        accumulate_parameter_gradient(&mut matrix, &source);
        accumulate_parameter_gradient(&mut matrix, &source);
        assert_eq!(matrix, Array2::from_elem((2, 2), 2.0));

        let mut vector = Array1::zeros(2);
        let source = Array1::ones(2);
        accumulate_vector_gradient(&mut vector, &source);
        assert_eq!(vector, Array1::from_elem(2, 1.0));
    }
}

// ===== model_gradients =====

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
    pub fn from_backward(block: &crate::rwkv_block::RwkvBlockBackward) -> Self {
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

// ===== model_train_step =====

pub struct ModelTrainStep { pub loss:f32,pub logits_gradient:Array2<f32>,pub backward:RwkvModelBackward,pub gradients:ModelGradients,pub tape:ModelBackwardTape,pub next_state:RwkvModelState }
impl ModelTrainStep {
 pub fn run(model:&RwkvModel,token_ids:&[usize],targets:&Array1<usize>)->Self{Self::run_with_state(model,token_ids,targets,None)}
 pub fn run_with_state(model:&RwkvModel,token_ids:&[usize],targets:&Array1<usize>,state:Option<&RwkvModelState>)->Self{assert_eq!(token_ids.len(),targets.len());let(logits,tape,next_state)=model.forward_with_tape_and_state(token_ids,state);let(loss,logits_gradient)=logits_cross_entropy_backward(&logits,targets);Self::from_logits_gradient(model,token_ids,logits,tape,next_state,loss,logits_gradient)}
 /// Same as `run`, but scales the loss gradient. Backpropagation is linear in the
 /// output gradient, so this scales every parameter gradient by the same factor --
 /// used to average over an accumulation window without materialising the sum.
 pub fn run_scaled(model:&RwkvModel,token_ids:&[usize],targets:&Array1<usize>,scale:f32)->Self{assert_eq!(token_ids.len(),targets.len());let(logits,tape,next_state)=model.forward_with_tape_and_state(token_ids,None);let(loss,mut logits_gradient)=logits_cross_entropy_backward(&logits,targets);if scale!=1.0{logits_gradient.mapv_inplace(|v|v*scale)}Self::from_logits_gradient(model,token_ids,logits,tape,next_state,loss,logits_gradient)}
 pub fn from_logits_gradient(model:&RwkvModel,token_ids:&[usize],logits:Array2<f32>,tape:ModelBackwardTape,next_state:RwkvModelState,loss:f32,logits_gradient:Array2<f32>)->Self{assert_eq!(logits.dim(),logits_gradient.dim());let backward=backward(model,token_ids,&tape,&logits_gradient);let gradients=ModelGradients::from_backward(&backward);Self{loss,logits_gradient,backward,gradients,tape,next_state}}
 pub fn run_with_logits_gradient(model:&RwkvModel,token_ids:&[usize],logits_gradient:Array2<f32>)->Self{assert_eq!(token_ids.len(),logits_gradient.nrows());let(logits,tape,next_state)=model.forward_with_tape_and_state(token_ids,None);Self::from_logits_gradient(model,token_ids,logits,tape,next_state,0.0,logits_gradient)}
}
#[cfg(test)]mod tests_model_train_step{use super::*;use crate::rwkv_model::RwkvModelConfig;#[test]fn computes_full_model_gradients(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),1234);let tokens=[1usize,2,3];let targets=Array1::from_vec(vec![2,3,4]);let step=ModelTrainStep::run(&model,&tokens,&targets);assert!(step.loss.is_finite());assert_eq!(step.logits_gradient.dim(),(3,16));assert_eq!(step.gradients.layers.len(),2);assert!(step.gradients.layers.iter().all(|g|g.parameter_count()>0));assert!(step.gradients.embedding.iter().all(|v|v.is_finite()));assert!(step.gradients.head.iter().all(|v|v.is_finite()));}
#[test]fn scaling_the_loss_gradient_scales_every_parameter_gradient(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),99);let tokens=[1usize,2,3];let targets=Array1::from_vec(vec![2,3,4]);let full=ModelTrainStep::run(&model,&tokens,&targets);let half=ModelTrainStep::run_scaled(&model,&tokens,&targets,0.5);let err=full.gradients.head.iter().zip(half.gradients.head.iter()).map(|(a,b)|(a*0.5-b).abs()).fold(0.0f32,f32::max);assert!(err<1e-6,"head gradient did not scale linearly: {err}");let err=full.gradients.layers[0].time_key.iter().zip(half.gradients.layers[0].time_key.iter()).map(|(a,b)|(a*0.5-b).abs()).fold(0.0f32,f32::max);assert!(err<1e-6,"time_key gradient did not scale linearly: {err}");}}

// ===== rwkv_model_backward =====

pub struct RwkvModelBackward { pub grad_input: Array2<f32>, pub grad_embedding: Array2<f32>, pub grad_ln_out_weight: Array1<f32>, pub grad_ln_out_bias: Array1<f32>, pub grad_head: Array2<f32>, pub head: HeadBackward, pub blocks: Vec<(usize,RwkvBlockBackward)>, pub moba_blocks: Vec<(usize,MobaBlockBackward)>, pub grad_initial_states: Vec<Option<RwkvBlockState>>, pub grad_v_first: Option<Array2<f32>> }
pub fn backward(model:&RwkvModel,token_ids:&[usize],tape:&ModelBackwardTape,logits_grad:&Array2<f32>)->RwkvModelBackward{backward_with_state_grads(model,token_ids,tape,logits_grad,None)}
pub fn backward_with_state_grads(model:&RwkvModel,token_ids:&[usize],tape:&ModelBackwardTape,logits_grad:&Array2<f32>,grad_next_states:Option<&[Option<RwkvBlockState>]>)->RwkvModelBackward{
    assert!(!token_ids.is_empty());assert_eq!(tape.block_order.len(),model.config.n_layer);assert_eq!(tape.inputs.len(),model.config.n_layer);assert_eq!(tape.outputs.len(),model.config.n_layer);let normalized=tape.normalized.as_ref().expect("missing normalized output");let ln_input=tape.ln_input.as_ref().expect("missing layer norm input");let logits=tape.logits.as_ref().expect("missing logits");assert_eq!(logits_grad.dim(),logits.dim());if let Some(grads)=grad_next_states{assert_eq!(grads.len(),model.rwkv_blocks.len());}
    let head=head_backward(token_ids,normalized,logits_grad,&model.head.weight,ln_input,&model.ln_out.weight,model.ln_out.eps,model.config.vocab_size);let mut grad=head.grad_input.clone();let mut grad_v_first=None;let mut blocks=Vec::with_capacity(model.rwkv_blocks.len());let mut moba_blocks=Vec::with_capacity(model.moba_blocks.len());let mut grad_initial_states=(0..model.rwkv_blocks.len()).map(|_|None).collect::<Vec<_>>();
    for slot in (0..model.config.n_layer).rev(){match tape.block_order[slot]{BackwardBlockKind::Rwkv(index)=>{let block_tape=tape.rwkv_tapes[slot].as_ref().expect("missing RWKV block tape");let next=grad_next_states.and_then(|all|all[index].as_ref());let block_grad=crate::rwkv_block::backward(&model.rwkv_blocks[index],block_tape,&grad,next.map(|s|&s.time_state),next.map(|s|&s.time_prev),next.map(|s|&s.cmix_prev),grad_v_first.as_ref());grad=block_grad.grad_input.clone();grad_initial_states[index]=Some(RwkvBlockState{time_state:block_grad.grad_time_state.clone(),time_prev:block_grad.grad_time_prev.clone(),cmix_prev:block_grad.grad_cmix_prev.clone(),v_first:Some(block_grad.grad_v_first.clone())});grad_v_first=Some(match grad_v_first{Some(mut total)=>{total+=&block_grad.grad_v_first;total},None=>block_grad.grad_v_first.clone()});blocks.push((index,block_grad));}BackwardBlockKind::Moba(index)=>{let block_tape=tape.moba_tapes[slot].as_ref().expect("missing MOBA block tape");let block_grad=model.moba_blocks[index].backward(block_tape,&grad);grad=block_grad.grad_input.clone();moba_blocks.push((index,block_grad));}}}blocks.reverse();moba_blocks.reverse();RwkvModelBackward{grad_input:grad,grad_embedding:head.grad_embedding.clone(),grad_ln_out_weight:head.grad_ln_weight.clone(),grad_ln_out_bias:head.grad_ln_bias.clone(),grad_head:head.grad_weight.clone(),head,blocks,moba_blocks,grad_initial_states,grad_v_first}
}
#[cfg(test)]mod tests_rwkv_model_backward{use super::*;use crate::loss::cross_entropy_backward;use crate::rwkv_model::RwkvModelConfig;#[test]fn reverse_rwkv_model_backward_produces_gradients(){let model=RwkvModel::new(RwkvModelConfig::new(32,16,2,4),123);let tokens=[1usize,2,3];let(logits,tape)=model.forward_with_tape(&tokens);let result=backward(&model,&tokens,&tape,&cross_entropy_backward(&logits,&tokens));assert_eq!(result.blocks.len(),2);assert_eq!(result.grad_initial_states.len(),2);assert!(result.grad_input.iter().all(|v|v.is_finite()));}#[test]fn reverse_mixed_model_backward_produces_moba_gradients(){let model=RwkvModel::new(RwkvModelConfig::new(32,16,4,4).with_moba(1,2,1),123);let tokens=[1usize,2,3,4,5,6];let(logits,tape)=model.forward_with_tape(&tokens);let result=backward(&model,&tokens,&tape,&cross_entropy_backward(&logits,&tokens));assert_eq!(result.moba_blocks.len(),1);assert!(result.moba_blocks[0].1.grad_input.iter().all(|v|v.is_finite()));}}
