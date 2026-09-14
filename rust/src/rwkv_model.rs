use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::moba_block::{MobaBlock, MobaBlockState};
use crate::model_backward::{BackwardBlockKind, ModelBackwardTape};
use crate::rwkv_block::{RwkvBlock, RwkvBlockState};
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct RwkvModelConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub head_size: usize,
    pub head_size_divisor: usize,
    pub n_moba_layer: usize,
    pub moba_chunk_size: usize,
    pub moba_topk: usize,
}
impl RwkvModelConfig {
    pub fn new(vocab_size: usize, n_embd: usize, n_layer: usize, head_size: usize) -> Self { assert!(vocab_size>0); assert!(n_layer>0); assert!(head_size>0); assert_eq!(n_embd%head_size,0); Self{vocab_size,n_embd,n_layer,head_size,head_size_divisor:8,n_moba_layer:0,moba_chunk_size:512,moba_topk:4} }
    pub fn n_head(&self)->usize{self.n_embd/self.head_size}
    pub fn with_moba(mut self,n_moba_layer:usize,chunk_size:usize,topk:usize)->Self{assert!(n_moba_layer<self.n_layer);assert!(chunk_size>0);self.n_moba_layer=n_moba_layer;self.moba_chunk_size=chunk_size;self.moba_topk=topk;self}
}
#[derive(Clone, Copy)] enum BlockKind{Rwkv(usize),Moba(usize)}
pub struct RwkvModelState{pub rwkv_blocks:Vec<RwkvBlockState>,pub moba_blocks:Vec<MobaBlockState>,pub v_first:Option<Array2<f32>>}
pub struct RwkvModel{pub config:RwkvModelConfig,pub embedding:Embedding,pub rwkv_blocks:Vec<RwkvBlock>,pub moba_blocks:Vec<MobaBlock>,order:Vec<BlockKind>,pub ln_out:LayerNorm,pub head:Linear}
impl RwkvModel{
 pub fn new(config:RwkvModelConfig,seed:u64)->Self{let embedding=Embedding::new(config.vocab_size,config.n_embd,seed^0x454d_4245_4444_494e);let n_rwkv=config.n_layer-config.n_moba_layer;let rwkv_blocks=(0..n_rwkv).map(|layer_id|RwkvBlock::new(config.n_embd,config.n_head(),layer_id,config.n_layer)).collect();let moba_blocks=(0..config.n_moba_layer).map(|i|MobaBlock::new(config.n_embd,config.head_size,config.moba_chunk_size,config.moba_topk,n_rwkv+i,config.n_layer)).collect();let mut order=Vec::with_capacity(config.n_layer);let interval=if config.n_moba_layer==0{n_rwkv}else{(n_rwkv/config.n_moba_layer).max(1)};let mut ri=0;for m in 0..config.n_moba_layer{let take=if m+1<config.n_moba_layer{interval}else{n_rwkv-ri};for k in 0..take{order.push(BlockKind::Rwkv(ri+k));}ri+=take;order.push(BlockKind::Moba(m));}while ri<n_rwkv{order.push(BlockKind::Rwkv(ri));ri+=1;}let ln_out=LayerNorm::new(config.n_embd,1e-5);let head=Linear::new_no_bias(config.n_embd,config.vocab_size);Self{config,embedding,rwkv_blocks,moba_blocks,order,ln_out,head}}
 pub fn load_safetensors(&mut self,path:impl AsRef<std::path::Path>)->Result<(),String>{crate::model_loader::load_model_safetensors(self,path)}
 pub fn forward(&self,token_ids:&[usize],state:Option<&RwkvModelState>)->(Array2<f32>,RwkvModelState){let(logits,next_state,_)=self.forward_internal(token_ids,state,None);(logits,next_state)}
 pub fn forward_with_tape(&self,token_ids:&[usize])->(Array2<f32>,ModelBackwardTape){self.forward_with_state_and_tape(token_ids,None)}
 pub fn forward_with_state_and_tape(&self,token_ids:&[usize],state:Option<&RwkvModelState>)->(Array2<f32>,ModelBackwardTape){let order=self.order.iter().map(|kind|match *kind{BlockKind::Rwkv(index)=>BackwardBlockKind::Rwkv(index),BlockKind::Moba(index)=>BackwardBlockKind::Moba(index)}).collect();let mut tape=ModelBackwardTape::new(order);let(logits,_,_)=self.forward_internal(token_ids,state,Some(&mut tape));(logits,tape)}
 fn forward_internal(&self,token_ids:&[usize],state:Option<&RwkvModelState>,mut tape:Option<&mut ModelBackwardTape>)->(Array2<f32>,RwkvModelState,Array2<f32>){assert!(!token_ids.is_empty());let mut x=self.embedding.forward(token_ids);let mut next_rwkv=Vec::with_capacity(self.rwkv_blocks.len());let mut next_moba=Vec::with_capacity(self.moba_blocks.len());let mut v_first=state.and_then(|s|s.v_first.clone());for kind in &self.order{let input=x.clone();match *kind{BlockKind::Rwkv(index)=>{let block_state=state.and_then(|s|s.rwkv_blocks.get(index));if let Some(ref mut model_tape)=tape{let(next_x,next_state,block_tape)=self.rwkv_blocks[index].forward_with_full_tape(&x,block_state,v_first.as_ref());v_first=next_state.v_first.clone();x=next_x;next_rwkv.push((index,next_state));model_tape.record_block(input,x.clone());model_tape.record_rwkv_tape(block_tape);}else{let(next_x,next_state)=self.rwkv_blocks[index].forward(&x,block_state,v_first.as_ref());v_first=next_state.v_first.clone();x=next_x;next_rwkv.push((index,next_state));}}BlockKind::Moba(index)=>{let block_state=state.and_then(|s|s.moba_blocks.get(index));if let Some(ref mut model_tape)=tape{let(next_x,next_state,block_tape)=self.moba_blocks[index].forward_with_tape(&x,block_state);x=next_x;next_moba.push((index,next_state));model_tape.record_block(input,x.clone());model_tape.record_moba_tape(block_tape);}else{let(next_x,next_state)=self.moba_blocks[index].forward(&x,block_state);x=next_x;next_moba.push((index,next_state));}}}}
 let mut rwkv_states:Vec<Option<RwkvBlockState>>=(0..self.rwkv_blocks.len()).map(|_|None).collect();for(index,block_state)in next_rwkv{rwkv_states[index]=Some(block_state);}let rwkv_states=rwkv_states.into_iter().map(Option::unwrap).collect();let mut moba_states:Vec<Option<MobaBlockState>>=(0..self.moba_blocks.len()).map(|_|None).collect();for(index,block_state)in next_moba{moba_states[index]=Some(block_state);}let moba_states=moba_states.into_iter().map(Option::unwrap).collect();let ln_input=x.clone();x=self.ln_out.forward(&x);let normalized=x.clone();let logits=self.head.forward(&x);if let Some(ref mut tape)=tape{tape.record_head(ln_input,normalized,logits.clone());}(logits,RwkvModelState{rwkv_blocks:rwkv_states,moba_blocks:moba_states,v_first},x)}
 pub fn parameter_count(&self)->usize{self.embedding.parameter_count()+self.rwkv_blocks.iter().map(RwkvBlock::parameter_count).sum::<usize>()+self.moba_blocks.iter().map(MobaBlock::parameter_count).sum::<usize>()+self.ln_out.weight.len()+self.ln_out.bias.len()+self.head.parameter_count()}
 pub fn argmax(logits:&Array2<f32>)->Array1<usize>{let mut result=Array1::zeros(logits.nrows());for row in 0..logits.nrows(){let mut best_index=0;let mut best_value=f32::NEG_INFINITY;for col in 0..logits.ncols(){let value=logits[[row,col]];if value>best_value{best_value=value;best_index=col;}}result[row]=best_index;}result}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a:&Array2<f32>,b:&Array2<f32>)->f32 {
        a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0,f32::max)
    }

    #[test]
    fn stateful_tape_matches_stateful_forward() {
        let model=RwkvModel::new(RwkvModelConfig::new(32,16,2,4),123);
        let prefix=[1usize,2,3];
        let suffix=[4usize,5,6];
        let (_,state)=model.forward(&prefix,None);
        let (plain,_)=model.forward(&suffix,Some(&state));
        let (taped,tape)=model.forward_with_state_and_tape(&suffix,Some(&state));
        assert!(max_abs_diff(&plain,&taped)<1e-6);
        assert_eq!(tape.rwkv_tapes.len(),model.config.n_layer);
        for tape in tape.rwkv_tapes.iter().flatten() {
            assert!(tape.initial_state.is_some());
        }
    }
}
