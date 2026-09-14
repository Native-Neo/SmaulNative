use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::moba_block::{MobaBlock, MobaBlockState};
use crate::model_backward::{BackwardBlockKind, ModelBackwardTape};
use crate::rwkv_block::{RwkvBlock, RwkvBlockState};
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct RwkvModelConfig { pub vocab_size: usize, pub n_embd: usize, pub n_layer: usize, pub head_size: usize, pub head_size_divisor: usize, pub n_moba_layer: usize, pub moba_chunk_size: usize, pub moba_topk: usize }
impl RwkvModelConfig { pub fn new(vocab_size:usize,n_embd:usize,n_layer:usize,head_size:usize)->Self{Self{vocab_size,n_embd,n_layer,head_size,head_size_divisor:8,n_moba_layer:0,moba_chunk_size:0,moba_topk:0}} pub fn with_moba(mut self,n_moba_layer:usize,chunk_size:usize,topk:usize)->Self{self.n_moba_layer=n_moba_layer;self.moba_chunk_size=chunk_size;self.moba_topk=topk;self} }

pub struct RwkvModel { pub config: RwkvModelConfig, pub embedding: Embedding, pub rwkv_blocks: Vec<RwkvBlock>, pub moba_blocks: Vec<MobaBlock>, pub ln_out: LayerNorm, pub head: Linear }

#[derive(Clone)]
pub struct RwkvModelState { pub rwkv_blocks: Vec<RwkvBlockState>, pub moba_blocks: Vec<MobaBlockState> }

impl RwkvModel {
    pub fn new(config:RwkvModelConfig,seed:u64)->Self { assert!(config.vocab_size>0); assert!(config.n_embd>0); assert!(config.n_layer>0); assert!(config.n_embd%config.head_size==0); let heads=config.n_embd/config.head_size; let mut rwkv_blocks=Vec::with_capacity(config.n_layer-config.n_moba_layer); let mut moba_blocks=Vec::with_capacity(config.n_moba_layer); for i in 0..config.n_layer { if i>=config.n_layer-config.n_moba_layer { moba_blocks.push(MobaBlock::new(config.n_embd,config.moba_chunk_size,config.moba_topk,seed+i as u64)); } else { rwkv_blocks.push(RwkvBlock::new(config.n_embd,heads,i,config.n_layer)); } } Self{embedding:Embedding::new(config.vocab_size,config.n_embd,seed),rwkv_blocks,moba_blocks,ln_out:LayerNorm::new(config.n_embd,1e-5),head:Linear::new(config.vocab_size,config.n_embd,seed+1),config} }
    pub fn parameter_count(&self)->usize { self.embedding.weight.len()+self.ln_out.weight.len()+self.ln_out.bias.len()+self.head.weight.len()+self.rwkv_blocks.iter().map(RwkvBlock::parameter_count).sum::<usize>() }
    pub fn forward(&self,tokens:&[usize],state:Option<&RwkvModelState>)->(Array2<f32>,RwkvModelState){let x=self.embedding.forward(tokens);let mut current=x;let mut rstates=Vec::with_capacity(self.rwkv_blocks.len());let mut mstates=Vec::with_capacity(self.moba_blocks.len());let mut ri=0;let mut mi=0;for slot in 0..self.config.n_layer{if slot>=self.config.n_layer-self.config.n_moba_layer{let(s,ns)=self.moba_blocks[mi].forward(&current,state.and_then(|s|s.moba_blocks.get(mi)));current=s;mstates.push(ns);mi+=1;}else{let(s,ns)=self.rwkv_blocks[ri].forward(&current,state.and_then(|s|s.rwkv_blocks.get(ri)),None);current=s;rstates.push(ns);ri+=1;}}let normalized=self.ln_out.forward(&current);let logits=self.head.forward(&normalized);(logits,RwkvModelState{rwkv_blocks:rstates,moba_blocks:mstates})}
    pub fn forward_with_tape(&self,tokens:&[usize])->(Array2<f32>,ModelBackwardTape){let x=self.embedding.forward(tokens);let mut current=x.clone();let mut tape=ModelBackwardTape::new(self.config.n_layer);let mut ri=0;let mut mi=0;for slot in 0..self.config.n_layer{if slot>=self.config.n_layer-self.config.n_moba_layer{let(y,block_tape)=self.moba_blocks[mi].forward_with_tape(&current);tape.moba_tapes[slot]=Some(block_tape);tape.block_order[slot]=BackwardBlockKind::Moba(mi);current=y;mi+=1;}else{let(y,_,block_tape)=self.rwkv_blocks[ri].forward_with_full_tape(&current,None,None);tape.rwkv_tapes[slot]=Some(block_tape);tape.block_order[slot]=BackwardBlockKind::Rwkv(ri);current=y;ri+=1;}}tape.ln_input=current.clone();let normalized=self.ln_out.forward(&current);tape.normalized=Some(normalized.clone());let logits=self.head.forward(&normalized);tape.logits=Some(logits.clone());(logits,tape)}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn max_abs_diff(a:&Array2<f32>,b:&Array2<f32>)->f32 { a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max) }
    fn max_abs_diff_1(a:&Array1<f32>,b:&Array1<f32>)->f32 { a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max) }
    fn max_abs_diff_4(a:&ndarray::Array4<f32>,b:&ndarray::Array4<f32>)->f32 { a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max) }
    fn state_max_abs_diff(a:&RwkvModelState,b:&RwkvModelState)->f32 { let mut diff:f32=0.0; for (x,y) in a.rwkv_blocks.iter().zip(b.rwkv_blocks.iter()) { diff=diff.max(max_abs_diff_4(&x.time_state,&y.time_state)); diff=diff.max(max_abs_diff_1(&x.time_prev,&y.time_prev)); diff=diff.max(max_abs_diff_1(&x.cmix_prev,&y.cmix_prev)); } diff }
    #[test] fn model_runs(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),1);let(logits,state)=model.forward(&[1,2,3],None);assert_eq!(logits.dim(),(3,16));assert_eq!(state.rwkv_blocks.len(),2);}
    #[test] fn helper_is_finite(){let a=Array2::zeros((1,1));assert_eq!(max_abs_diff(&a,&a),0.0);}
    #[test] fn state_diff_is_zero_for_equal_states(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),1);let(_,state)=model.forward(&[1,2,3],None);assert_eq!(state_max_abs_diff(&state,&state),0.0);}
}
