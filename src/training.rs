use crate::dataset::{MultiFileDatasetStream, MultiFileTextStream, ParquetTextStream, TextStream};
use crate::loss::cross_entropy;
use crate::model_backward::{LayerGradients, MobaLayerGradients, ModelGradients, ModelTrainStep};
use crate::rwkv_model::RwkvModel;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum OptimizerKind{Lion,AdamW}
pub struct TrainStep{pub optimizer:Lion,pub adamw:Option<AdamW>,pub max_grad_norm:Option<f32>}
impl TrainStep{
 pub fn new(model:&RwkvModel,lr:f32)->Self{Self{optimizer:Lion::new(model.parameter_count(),lr,0.9,0.99,0.0),adamw:None,max_grad_norm:Some(1.0)}}
 pub fn new_with_optimizer(model:&RwkvModel,lr:f32,kind:OptimizerKind)->Self{match kind{OptimizerKind::Lion=>Self::new(model,lr),OptimizerKind::AdamW=>Self{optimizer:Lion::new(model.parameter_count(),lr,0.9,0.99,0.0),adamw:Some(AdamW::new(model.parameter_count(),lr,0.9,0.999,1e-8,0.01)),max_grad_norm:Some(1.0)}}}
 pub fn optimizer_kind(&self)->OptimizerKind{if self.adamw.is_some(){OptimizerKind::AdamW}else{OptimizerKind::Lion}}
 pub fn set_lr(&mut self,lr:f32){self.optimizer.lr=lr;if let Some(x)=&mut self.adamw{x.lr=lr}}
 pub fn loss_and_gradient(logits:&Array2<f32>,targets:&Array1<usize>)->(f32,Array2<f32>){cross_entropy(logits,targets)}
 pub fn clip_logits_gradient(&self,grad:&mut Array2<f32>)->f32{let mut refs=vec![grad];match self.max_grad_norm{Some(max_norm)=>clip_by_global_norm(&mut refs,max_norm),None=>clip_by_global_norm(&mut refs,f32::MAX)}}
 pub fn token_count(model:&RwkvModel)->usize{model.config.vocab_size}
 pub fn step(&mut self,model:&mut RwkvModel,gradients:&ModelGradients)->f32{assert_eq!(gradients.layers.len(),model.rwkv_blocks.len());assert_eq!(gradients.moba_layers.len(),model.moba_blocks.len());assert_eq!(gradients.parameter_count(),model.parameter_count());let mut parameters=Vec::with_capacity(model.parameter_count());let mut grads=Vec::with_capacity(model.parameter_count());collect_model(model,gradients,&mut parameters,&mut grads);let norm=flat_norm(&grads);let scale=self.max_grad_norm.map(|max|if norm>max{max/norm}else{1.0}).unwrap_or(1.0);if scale!=1.0{for g in &mut grads{*g*=scale;}}if let Some(x)=&mut self.adamw{x.step(&mut parameters,&grads)}else{self.optimizer.step(&mut parameters,&grads)}write_model(model,&parameters);norm}
 pub fn step_sgd(&self,model:&mut RwkvModel,gradients:&ModelGradients,lr:f32)->f32{assert_eq!(gradients.parameter_count(),model.parameter_count());let mut parameters=Vec::with_capacity(model.parameter_count());let mut grads=Vec::with_capacity(model.parameter_count());collect_model(model,gradients,&mut parameters,&mut grads);let norm=flat_norm(&grads);let scale=self.max_grad_norm.map(|max|if norm>max{max/norm}else{1.0}).unwrap_or(1.0);for(p,g)in parameters.iter_mut().zip(grads){*p-=lr*g*scale;}write_model(model,&parameters);norm}
 pub fn optimizer_state(&self)->Vec<f32>{if let Some(x)=&self.adamw{let(a,b,_)=x.state();let mut out=Vec::with_capacity(a.len()+b.len());out.extend_from_slice(a);out.extend_from_slice(b);out}else{self.optimizer.state().to_vec()}}
 pub fn optimizer_state_bytes(&self)->Vec<u8>{if let Some(x)=&self.adamw{let(a,b,step)=x.state();let mut out=Vec::with_capacity(25+(a.len()+b.len())*4);out.push(1);out.extend_from_slice(&(step as u64).to_le_bytes());out.extend_from_slice(&(a.len()as u64).to_le_bytes());for v in a{out.extend_from_slice(&v.to_le_bytes())}for v in b{out.extend_from_slice(&v.to_le_bytes())}out}else{let s=self.optimizer.state();let mut out=Vec::with_capacity(9+s.len()*4);out.push(0);out.extend_from_slice(&(s.len()as u64).to_le_bytes());for v in s{out.extend_from_slice(&v.to_le_bytes())}out}}
 pub fn load_optimizer_state(&mut self,state:&[f32])->Result<(),String>{if let Some(x)=&mut self.adamw{let n=x.parameter_count();if state.len()!=n*2{return Err(format!("AdamW state has {} values, expected {}",state.len(),n*2))}x.load_state(&state[..n],&state[n..],0)}else{self.optimizer.load_state(state)}}
 pub fn load_optimizer_state_bytes(&mut self,state:&[u8])->Result<(),String>{fn u64v(s:&[u8],o:&mut usize)->Result<u64,String>{if s.len()<*o+8{return Err("optimizer state truncated".into())}let mut a=[0;8];a.copy_from_slice(&s[*o..*o+8]);*o+=8;Ok(u64::from_le_bytes(a))}fn f32v(s:&[u8],o:&mut usize)->Result<f32,String>{if s.len()<*o+4{return Err("optimizer state truncated".into())}let mut a=[0;4];a.copy_from_slice(&s[*o..*o+4]);*o+=4;Ok(f32::from_le_bytes(a))}if state.is_empty(){return Err("optimizer state is empty".into())}let mut o=1;match(state[0],self.adamw.as_mut()){(0,None)=>{let n=u64v(state,&mut o)?as usize;if n!=self.optimizer.parameter_count(){return Err("Lion optimizer state size mismatch".into())}let mut v=Vec::with_capacity(n);for _ in 0..n{v.push(f32v(state,&mut o)?)}if o!=state.len(){return Err("Lion optimizer state has trailing data".into())}self.optimizer.load_state(&v)},(1,Some(x))=>{let step=u64v(state,&mut o)?;let n=u64v(state,&mut o)?as usize;if n!=x.parameter_count(){return Err("AdamW optimizer state size mismatch".into())}let mut a=Vec::with_capacity(n);let mut b=Vec::with_capacity(n);for _ in 0..n{a.push(f32v(state,&mut o)?)}for _ in 0..n{b.push(f32v(state,&mut o)?)}if o!=state.len(){return Err("AdamW optimizer state has trailing data".into())}x.load_state(&a,&b,step as usize)},(0,Some(_))=>Err("optimizer state is Lion but AdamW was selected".into()),(1,None)=>Err("optimizer state is AdamW but Lion was selected".into()),_=>Err("unknown optimizer state format".into())}}
}
fn flat_norm(values:&[f32])->f32{values.iter().map(|v|v*v).sum::<f32>().sqrt()}fn push(a:&mut Vec<f32>,g:&mut Vec<f32>,p:&[f32],d:&[f32]){assert_eq!(p.len(),d.len());a.extend_from_slice(p);g.extend_from_slice(d);}
fn collect_layer(layer:&crate::rwkv_block::RwkvBlock,grad:&LayerGradients,p:&mut Vec<f32>,g:&mut Vec<f32>){if let(Some(n),Some(w),Some(b))=(&layer.ln0,&grad.ln0_weight,&grad.ln0_bias){push(p,g,n.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),w.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,n.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),b.as_slice().expect("parameter tensor must be in standard contiguous layout"));}push(p,g,layer.ln1.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln1_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln1.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln1_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln2.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln2_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln2.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln2_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));let t=&layer.time_mix;push(p,g,t.x_r.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_r.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.x_w.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_w.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.x_v.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_v.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.x_a.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_a.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.x_g.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_x_g.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.w0.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_w0.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.w1.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_w1.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.w2.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_w2.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.a0.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_a0.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.a1.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_a1.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.a2.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_a2.as_slice().expect("parameter tensor must be in standard contiguous layout"));if let(Some(x),Some(d))=(&t.v0,&grad.time_v0){push(p,g,x.as_slice().expect("parameter tensor must be in standard contiguous layout"),d.as_slice().expect("parameter tensor must be in standard contiguous layout"));}if let(Some(x),Some(d))=(&t.v1,&grad.time_v1){push(p,g,x.as_slice().expect("parameter tensor must be in standard contiguous layout"),d.as_slice().expect("parameter tensor must be in standard contiguous layout"));}if let(Some(x),Some(d))=(&t.v2,&grad.time_v2){push(p,g,x.as_slice().expect("parameter tensor must be in standard contiguous layout"),d.as_slice().expect("parameter tensor must be in standard contiguous layout"));}push(p,g,t.g1.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_g1.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.g2.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_g2.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.k_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_k_k.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.k_a.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_k_a.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.r_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_r_k.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.receptance.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_receptance.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.key.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_key.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.value.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_value.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.output.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_output.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.ln_x.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_ln_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,t.ln_x.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.time_ln_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));if layer.moe.is_none(){push(p,g,layer.cmix.x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.cmix.key.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_key.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.cmix.value.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_value.as_slice().expect("parameter tensor must be in standard contiguous layout"));}if let Some(moe)=&layer.moe{push(p,g,moe.router.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.moe_router.as_ref().unwrap().as_slice().expect("parameter tensor must be in standard contiguous layout"));for i in 0..moe.experts.len(){push(p,g,moe.experts[i].x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.moe_x_k[i].as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,moe.experts[i].key.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.moe_key[i].as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,moe.experts[i].value.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.moe_value[i].as_slice().expect("parameter tensor must be in standard contiguous layout"));}}}
fn collect_moba(layer:&crate::moba_block::MobaBlock,grad:&MobaLayerGradients,p:&mut Vec<f32>,g:&mut Vec<f32>){push(p,g,layer.ln1.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln1_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln1.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln1_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln2.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln2_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ln2.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.ln2_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.att.receptance.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.receptance.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.att.key.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.key.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.att.value.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.value.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.att.output.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.output.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ffn.x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_x_k.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ffn.key.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_key.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,layer.ffn.value.as_slice().expect("parameter tensor must be in standard contiguous layout"),grad.cmix_value.as_slice().expect("parameter tensor must be in standard contiguous layout"));}
fn collect_model(model:&RwkvModel,gradients:&ModelGradients,p:&mut Vec<f32>,g:&mut Vec<f32>){push(p,g,model.embedding.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),gradients.embedding.as_slice().expect("parameter tensor must be in standard contiguous layout"));for(layer,grad)in model.rwkv_blocks.iter().zip(&gradients.layers){collect_layer(layer,grad,p,g);}for(layer,grad)in model.moba_blocks.iter().zip(&gradients.moba_layers){collect_moba(layer,grad,p,g);}push(p,g,model.ln_out.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),gradients.ln_out_weight.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,model.ln_out.bias.as_slice().expect("parameter tensor must be in standard contiguous layout"),gradients.ln_out_bias.as_slice().expect("parameter tensor must be in standard contiguous layout"));push(p,g,model.head.weight.as_slice().expect("parameter tensor must be in standard contiguous layout"),gradients.head.as_slice().expect("parameter tensor must be in standard contiguous layout"));}
fn write_slice(dst:&mut[f32],src:&[f32],offset:&mut usize){let end=*offset+dst.len();dst.copy_from_slice(&src[*offset..end]);*offset=end;}
fn write_layer(layer:&mut crate::rwkv_block::RwkvBlock,src:&[f32],o:&mut usize){if let Some(n)=&mut layer.ln0{write_slice(n.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(n.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}write_slice(layer.ln1.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln1.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln2.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln2.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);let t=&mut layer.time_mix;write_slice(t.x_r.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.x_w.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.x_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.x_v.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.x_a.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.x_g.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.w0.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.w1.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.w2.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.a0.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.a1.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.a2.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);if let Some(x)=&mut t.v0{write_slice(x.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}if let Some(x)=&mut t.v1{write_slice(x.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}if let Some(x)=&mut t.v2{write_slice(x.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}write_slice(t.g1.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.g2.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.k_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.k_a.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.r_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.receptance.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.key.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.value.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.output.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.ln_x.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(t.ln_x.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);if layer.moe.is_none(){write_slice(layer.cmix.x_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.cmix.key.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.cmix.value.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}if let Some(moe)=&mut layer.moe{write_slice(moe.router.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);for expert in &mut moe.experts{write_slice(expert.x_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(expert.key.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(expert.value.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}}}
fn write_moba(layer:&mut crate::moba_block::MobaBlock,src:&[f32],o:&mut usize){write_slice(layer.ln1.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln1.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln2.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ln2.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.att.receptance.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.att.key.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.att.value.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.att.output.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ffn.x_k.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ffn.key.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);write_slice(layer.ffn.value.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,o);}
fn write_model(model:&mut RwkvModel,src:&[f32]){let mut o=0;write_slice(model.embedding.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,&mut o);for layer in &mut model.rwkv_blocks{write_layer(layer,src,&mut o);}for layer in &mut model.moba_blocks{write_moba(layer,src,&mut o);}write_slice(model.ln_out.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,&mut o);write_slice(model.ln_out.bias.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,&mut o);write_slice(model.head.weight.as_slice_mut().expect("parameter tensor must be in standard contiguous layout"),src,&mut o);assert_eq!(o,src.len());}
#[cfg(test)]mod tests_training{use super::*;use crate::model_backward::ModelTrainStep;use crate::rwkv_model::RwkvModelConfig;#[test]fn optimizer_updates_full_rwkv_model(){let mut model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4),1234);let tokens=[1usize,2,3];let targets=Array1::from_vec(vec![2,3,4]);let step=ModelTrainStep::run(&model,&tokens,&targets);let before=model.head.weight[[0,0]];let mut train=TrainStep::new(&model,1e-3);train.step(&mut model,&step.gradients);assert_ne!(before,model.head.weight[[0,0]]);}#[test]fn optimizer_updates_moba_parameters(){let mut model=RwkvModel::new(RwkvModelConfig::new(16,8,3,4).with_moba(1,2,1),1234);let tokens=[1usize,2,3,4,5,6];let targets=Array1::from_vec(vec![2,3,4,5,6,7]);let step=ModelTrainStep::run(&model,&tokens,&targets);let before=model.moba_blocks[0].att.output.weight[[0,0]];let mut train=TrainStep::new(&model,1e-3);train.step(&mut model,&step.gradients);assert_ne!(before,model.moba_blocks[0].att.output.weight[[0,0]]);}#[test]fn optimizer_updates_moe_parameters(){let mut model=RwkvModel::new(RwkvModelConfig::new(16,8,2,4).with_moe(true,3,2),1234);let tokens=[1usize,2,3];let targets=Array1::from_vec(vec![2,3,4]);let step=ModelTrainStep::run(&model,&tokens,&targets);let before=model.rwkv_blocks[0].moe.as_ref().unwrap().router.weight[[0,0]];let mut train=TrainStep::new(&model,1e-3);train.step(&mut model,&step.gradients);assert_ne!(before,model.rwkv_blocks[0].moe.as_ref().unwrap().router.weight[[0,0]]);}#[test]fn optimizer_state_is_exposed(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,1,4),1);let mut train=TrainStep::new(&model,1e-4);assert_eq!(train.optimizer_state().len(),model.parameter_count());assert!(train.load_optimizer_state(&vec![0.0;1]).is_err());}#[test]fn adamw_updates_parameters_in_training(){let mut model=RwkvModel::new(RwkvModelConfig::new(16,8,1,4),7);let tokens=[1usize,2,3];let targets=Array1::from_vec(vec![2,3,4]);let step=ModelTrainStep::run(&model,&tokens,&targets);let before=model.head.weight[[0,0]];let mut train=TrainStep::new_with_optimizer(&model,1e-3,OptimizerKind::AdamW);assert_eq!(train.optimizer_kind(),OptimizerKind::AdamW);train.step(&mut model,&step.gradients);assert_ne!(before,model.head.weight[[0,0]]);assert!(train.optimizer_state_bytes().len()>1);}#[test]fn optimizer_state_bytes_round_trip(){let model=RwkvModel::new(RwkvModelConfig::new(16,8,1,4),9);let mut train=TrainStep::new_with_optimizer(&model,1e-3,OptimizerKind::AdamW);let bytes=train.optimizer_state_bytes();let mut restored=TrainStep::new_with_optimizer(&model,1e-3,OptimizerKind::AdamW);restored.load_optimizer_state_bytes(&bytes).unwrap();assert_eq!(restored.optimizer_state_bytes(),bytes);}}


// ===== optimizer =====
pub trait Optimizer {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]);
    fn reset_state(&mut self);
}

pub struct Lion {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    momentum: Vec<f32>,
}

impl Lion {
    pub fn new(parameter_count: usize, lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self { Self { lr, beta1, beta2, weight_decay, momentum: vec![0.0; parameter_count] } }
    pub fn parameter_count(&self) -> usize { self.momentum.len() }
    pub fn state(&self) -> &[f32] { &self.momentum }
    pub fn load_state(&mut self, state: &[f32]) -> Result<(), String> { if state.len() != self.momentum.len() { return Err(format!("Lion state has {} values, expected {}", state.len(), self.momentum.len())); } self.momentum.copy_from_slice(state); Ok(()) }
}
impl Optimizer for Lion {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]) {
        assert_eq!(parameter.len(), gradient.len()); assert_eq!(parameter.len(), self.momentum.len());
        for index in 0..parameter.len() { let update=self.beta1*self.momentum[index]+(1.0-self.beta1)*gradient[index]; let direction=if update>=0.0{1.0}else{-1.0}; parameter[index]*=1.0-self.lr*self.weight_decay; parameter[index]-=self.lr*direction; self.momentum[index]=self.beta2*self.momentum[index]+(1.0-self.beta2)*gradient[index]; }
    }
    fn reset_state(&mut self) { self.momentum.fill(0.0); }
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step_count: usize,
    first: Vec<f32>,
    second: Vec<f32>,
}
impl AdamW {
    pub fn new(parameter_count: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self { Self { lr, beta1, beta2, eps, weight_decay, step_count: 0, first: vec![0.0; parameter_count], second: vec![0.0; parameter_count] } }
    pub fn parameter_count(&self) -> usize { self.first.len() }
    pub fn state(&self) -> (&[f32], &[f32], usize) { (&self.first, &self.second, self.step_count) }
    pub fn load_state(&mut self, first: &[f32], second: &[f32], step_count: usize) -> Result<(), String> { if first.len()!=self.first.len()||second.len()!=self.second.len(){return Err("AdamW state size mismatch".into())} self.first.copy_from_slice(first); self.second.copy_from_slice(second); self.step_count=step_count; Ok(()) }
}
impl Optimizer for AdamW {
    fn step(&mut self, parameter: &mut [f32], gradient: &[f32]) {
        assert_eq!(parameter.len(), gradient.len()); assert_eq!(parameter.len(), self.first.len()); self.step_count+=1;
        let b1t=1.0-self.beta1.powi(self.step_count as i32); let b2t=1.0-self.beta2.powi(self.step_count as i32);
        for i in 0..parameter.len() { self.first[i]=self.beta1*self.first[i]+(1.0-self.beta1)*gradient[i]; self.second[i]=self.beta2*self.second[i]+(1.0-self.beta2)*gradient[i]*gradient[i]; let m=self.first[i]/b1t; let v=self.second[i]/b2t; parameter[i]*=1.0-self.lr*self.weight_decay; parameter[i]-=self.lr*m/(v.sqrt()+self.eps); }
    }
    fn reset_state(&mut self) { self.first.fill(0.0); self.second.fill(0.0); self.step_count=0; }
}

#[cfg(test)]
mod tests_optimizer {
    use super::{AdamW,Lion,Optimizer};
    #[test]fn lion_updates_parameters(){let mut optimizer=Lion::new(2,0.01,0.9,0.99,0.0);let mut parameter=vec![1.0,-1.0];optimizer.step(&mut parameter,&[1.0,-1.0]);assert!(parameter[0]<1.0&&parameter[1]>-1.0)}
    #[test]fn lion_state_round_trip(){let mut first=Lion::new(2,0.01,0.9,0.99,0.0);let mut parameter=vec![1.0,-1.0];first.step(&mut parameter,&[2.0,-3.0]);let state=first.state().to_vec();let mut second=Lion::new(2,0.01,0.9,0.99,0.0);second.load_state(&state).unwrap();assert_eq!(second.state(),state.as_slice())}
    #[test]fn lion_rejects_wrong_state_size(){let mut optimizer=Lion::new(2,0.01,0.9,0.99,0.0);assert!(optimizer.load_state(&[0.0]).is_err())}
    #[test]fn lion_reset_does_not_mean_zero_grad(){let mut optimizer=Lion::new(1,0.01,0.9,0.99,0.0);let mut parameter=vec![1.0];optimizer.step(&mut parameter,&[1.0]);assert_ne!(optimizer.state()[0],0.0);optimizer.reset_state();assert_eq!(optimizer.state()[0],0.0)}
    #[test]fn adamw_updates_parameters(){let mut optimizer=AdamW::new(1,0.01,0.9,0.999,1e-8,0.0);let mut parameter=vec![1.0];optimizer.step(&mut parameter,&[1.0]);assert!(parameter[0]<1.0);assert_eq!(optimizer.state().2,1)}
}


// ===== gradient =====

pub fn global_norm(grads: &[&Array2<f32>]) -> f32 {
    grads.iter().flat_map(|g| g.iter()).map(|v| v * v).sum::<f32>().sqrt()
}

pub fn clip_by_global_norm(grads: &mut [&mut Array2<f32>], max_norm: f32) -> f32 {
    assert!(max_norm > 0.0);
    let norm = global_norm(&grads.iter().map(|g| &**g).collect::<Vec<_>>());
    if norm > max_norm {
        let scale = max_norm / norm;
        for grad in grads.iter_mut() { **grad *= scale; }
    }
    norm
}

#[cfg(test)]
mod tests_gradient {
    use super::*;

    #[test]
    fn clipping_limits_norm() {
        let mut a = Array2::from_elem((2, 2), 3.0);
        let mut grads: Vec<&mut Array2<f32>> = vec![&mut a];
        let before = clip_by_global_norm(&mut grads, 1.0);
        assert!(before > 1.0);
        assert!((global_norm(&[&a]) - 1.0).abs() < 1e-5);
    }
}


// ===== lr_scheduler =====
pub trait LrScheduler {
    fn learning_rate(&self, step: usize) -> f32;
}

pub struct WarmupCosine {
    pub base_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub min_lr: f32,
}

impl WarmupCosine {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, min_lr: f32) -> Self {
        assert!(base_lr >= 0.0);
        assert!(total_steps > 0);
        assert!(warmup_steps <= total_steps);
        assert!(min_lr >= 0.0 && min_lr <= base_lr);
        Self { base_lr, warmup_steps, total_steps, min_lr }
    }
}

impl LrScheduler for WarmupCosine {
    fn learning_rate(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        if step >= self.total_steps { return self.min_lr; }
        let span = (self.total_steps - self.warmup_steps).max(1) as f32;
        let progress = (step.saturating_sub(self.warmup_steps) as f32 / span).clamp(0.0, 1.0);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

#[cfg(test)]
mod tests_lr_scheduler {
    use super::*;

    #[test]
    fn warms_then_decays() {
        let s = WarmupCosine::new(1.0, 2, 10, 0.1);
        assert!(s.learning_rate(0) < s.learning_rate(1));
        assert!(s.learning_rate(2) > s.learning_rate(9));
        assert_eq!(s.learning_rate(10), 0.1);
    }
}


// ===== train_loop =====

pub struct TrainingConfig { pub max_steps: usize, pub log_every: usize, pub save_every: usize, pub max_grad_norm: Option<f32>, pub checkpoint_dir: Option<PathBuf> }
impl Default for TrainingConfig { fn default()->Self{Self{max_steps:0,log_every:1,save_every:0,max_grad_norm:Some(1.0),checkpoint_dir:None}} }
pub struct TrainingRunner { pub optimizer: TrainStep, pub state: TrainingState }
impl TrainingRunner {
 pub fn new(model:&RwkvModel,learning_rate:f32)->Self{Self{optimizer:TrainStep::new(model,learning_rate),state:TrainingState::new()}}
 pub fn resume_from_checkpoint(&mut self,path:impl AsRef<Path>)->Result<(),String>{let bundle=CheckpointBundle::load(path)?;if let Some(state)=bundle.optimizer_state.get("lion.momentum"){self.optimizer.load_optimizer_state(state)?;}else{return Err("checkpoint has no lion.momentum optimizer state".into());}self.state=bundle.training;Ok(())}
 pub fn resume_checkpoint(&mut self,model:&mut RwkvModel,path:impl AsRef<Path>)->Result<(),String>{
  let path=path.as_ref();
  let name=path.file_stem().and_then(|x|x.to_str()).ok_or("checkpoint path has no valid filename")?;
  let step=name.strip_prefix("training-").ok_or("checkpoint filename must be training-<step>.json")?;
  if step.is_empty()||!step.chars().all(|c|c.is_ascii_digit()){return Err("checkpoint filename must be training-<step>.json".into());}
  let model_path=path.with_file_name(format!("model-{step}.safetensors"));
  let bundle=CheckpointBundle::load(path)?;
  let optimizer_state=bundle.optimizer_state.get("lion.momentum").ok_or("checkpoint has no lion.momentum optimizer state")?;
  crate::checkpoint::load_model_safetensors(model,&model_path)?;
  self.optimizer.load_optimizer_state(optimizer_state)?;
  self.state=bundle.training;
  Ok(())
 }
 pub fn run_stream(&mut self,model:&mut RwkvModel,stream:&mut TextStream,config:&TrainingConfig)->Result<TrainingState,String>{self.run_batches(model,config,||stream.next_batch())}
 pub fn run_multi_stream(&mut self,model:&mut RwkvModel,stream:&mut MultiFileTextStream,config:&TrainingConfig)->Result<TrainingState,String>{self.run_batches(model,config,||stream.next_batch())}
 pub fn run_dataset_stream(&mut self,model:&mut RwkvModel,stream:&mut MultiFileDatasetStream,config:&TrainingConfig)->Result<TrainingState,String>{self.run_batches(model,config,||stream.next_batch())}
 pub fn run_parquet_stream(&mut self,model:&mut RwkvModel,stream:&mut ParquetTextStream,config:&TrainingConfig)->Result<TrainingState,String>{self.run_batches(model,config,||stream.next_batch())}
 fn run_batches<F>(&mut self,model:&mut RwkvModel,config:&TrainingConfig,mut next:F)->Result<TrainingState,String> where F:FnMut()->Result<Option<crate::dataset::TokenBatch>,String>{
  if config.log_every==0{return Err("log_every must be greater than zero".into());}
  if config.save_every>0&&config.checkpoint_dir.is_none(){return Err("checkpoint_dir is required when save_every is enabled".into());}
  while config.max_steps==0||self.state.step<config.max_steps{
   let Some(batch)=next()? else{break}; self.state.record_micro_step(batch.input.len()); let targets=Array1::from_vec(batch.target);
   let result=ModelTrainStep::run(model,&batch.input,&targets); self.optimizer.max_grad_norm=config.max_grad_norm; self.optimizer.step(model,&result.gradients); self.state.record_optimizer_step();
   if self.state.step%config.log_every==0{println!("step {} | loss {:.6} | tokens {}",self.state.step,result.loss,self.state.tokens_seen);}
   if config.save_every>0&&self.state.step%config.save_every==0{self.save_checkpoint(model,config.checkpoint_dir.as_ref().unwrap())?;}
   self.state.reset_accumulation();
  } Ok(self.state.clone())
 }
 fn save_checkpoint(&self,model:&RwkvModel,directory:&Path)->Result<(),String>{
  std::fs::create_dir_all(directory).map_err(|e|format!("failed to create checkpoint directory: {e}"))?; let step=self.state.step;
  crate::checkpoint::save_model_safetensors(model,directory.join(format!("model-{step}.safetensors")))?;
  let mut bundle=CheckpointBundle::new(self.state.clone()); bundle.optimizer_state.insert("lion.momentum".into(),self.optimizer.optimizer_state());
  bundle.save(directory.join(format!("training-{step}.json")))?; Ok(())
 }
 pub fn save_state(&self,path:impl AsRef<Path>)->Result<(),String>{let mut bundle=CheckpointBundle::new(self.state.clone());bundle.optimizer_state.insert("lion.momentum".into(),self.optimizer.optimizer_state());bundle.save(path)}
}

#[cfg(test)]
mod tests_train_loop{
 use super::*; use crate::rwkv_model::RwkvModelConfig; use crate::tokenizer::Tokenizer; use std::fs;
 fn tokenizer()->Tokenizer{Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into(),"b".into()," ".into()])}
 #[test]fn trains_native_stream(){let p=std::env::temp_dir().join("smaul-train.txt");fs::write(&p,"a b a b a b a b").unwrap();let mut s=TextStream::open(&p,tokenizer(),3).unwrap();let mut m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let state=r.run_stream(&mut m,&mut s,&TrainingConfig{max_steps:1,..Default::default()}).unwrap();assert_eq!(state.step,1);assert!(state.tokens_seen>=3);let _=fs::remove_file(p);}
 #[test]fn trains_multiple_files(){let d=std::env::temp_dir().join("smaul-train-multi");fs::create_dir_all(&d).unwrap();fs::write(d.join("a.txt"),"a b a b a b").unwrap();fs::write(d.join("b.txt"),"b a b a b a").unwrap();let mut s=MultiFileTextStream::open_discovered(&d,tokenizer(),3).unwrap();let mut m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let state=r.run_multi_stream(&mut m,&mut s,&TrainingConfig{max_steps:2,..Default::default()}).unwrap();assert_eq!(state.step,2);let _=fs::remove_dir_all(d);}
 #[test]fn parquet_missing_file_is_reported(){let p=std::env::temp_dir().join("smaul-not-parquet.parquet");fs::write(&p,b"not parquet").unwrap();assert!(ParquetTextStream::open(&p,tokenizer(),3).is_err());let _=fs::remove_file(p);}
 #[test]fn checkpoint_contains_lion_state(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("smaul-training-state.json");r.save_state(&p).unwrap();let c=CheckpointBundle::load(&p).unwrap();assert_eq!(c.optimizer_state.get("lion.momentum").unwrap().len(),m.parameter_count());let _=fs::remove_file(p);}
 #[test]fn resume_restores_training_state(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("smaul-training-resume.json");r.save_state(&p).unwrap();let mut resumed=TrainingRunner::new(&m,1e-4);resumed.resume_from_checkpoint(&p).unwrap();assert_eq!(resumed.state,r.state);assert_eq!(resumed.optimizer.optimizer_state(),r.optimizer.optimizer_state());let _=fs::remove_file(p);}
 #[test]fn resume_checkpoint_rejects_wrong_filename(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("resume.json");r.save_state(&p).unwrap();let mut loaded=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);assert!(r.resume_checkpoint(&mut loaded,&p).is_err());assert_eq!(r.state.step,0);let _=fs::remove_file(p);}
}

// ===== training_state =====

#[derive(Clone,Debug,Default,PartialEq,Eq)]
pub struct TrainingState{pub step:usize,pub epoch:usize,pub tokens_seen:u64,pub micro_steps:usize,pub optimizer_steps:usize}
impl TrainingState{pub fn new()->Self{Self::default()}pub fn record_micro_step(&mut self,tokens:usize){self.micro_steps+=1;self.tokens_seen=self.tokens_seen.saturating_add(tokens as u64)}pub fn record_optimizer_step(&mut self){self.optimizer_steps+=1;self.step+=1}pub fn record_epoch(&mut self){self.epoch+=1}pub fn reset_accumulation(&mut self){self.micro_steps=0}}

#[derive(Clone,Debug,PartialEq)]
pub struct CheckpointTensor{pub shape:Vec<usize>,pub data:Vec<f32>}
#[derive(Clone,Debug,PartialEq)]
pub struct CheckpointBundle{pub training:TrainingState,pub tensors:BTreeMap<String,CheckpointTensor>,pub optimizer_state:BTreeMap<String,Vec<f32>>,pub tokenizer_metadata:Value,pub rng_state:Option<String>}
impl CheckpointBundle{
 pub fn new(training:TrainingState)->Self{Self{training,tensors:BTreeMap::new(),optimizer_state:BTreeMap::new(),tokenizer_metadata:json!({}),rng_state:None}}
 pub fn insert_tensor(&mut self,name:impl Into<String>,shape:Vec<usize>,data:Vec<f32>){let expected:usize=shape.iter().product();assert_eq!(expected,data.len(),"checkpoint tensor shape does not match data length");self.tensors.insert(name.into(),CheckpointTensor{shape,data});}
 pub fn save(&self,path:impl AsRef<Path>)->Result<(),String>{let tensors=self.tensors.iter().map(|(name,t)| (name.clone(),json!({"shape":t.shape,"data":t.data}))).collect::<serde_json::Map<_,_>>();let optimizer=self.optimizer_state.iter().map(|(k,v)|(k.clone(),json!(v))).collect::<serde_json::Map<_,_>>();let root=json!({"format":"smaul-rust-checkpoint-v1","training":{"step":self.training.step,"epoch":self.training.epoch,"tokens_seen":self.training.tokens_seen,"micro_steps":self.training.micro_steps,"optimizer_steps":self.training.optimizer_steps},"tensors":tensors,"optimizer_state":optimizer,"tokenizer":self.tokenizer_metadata,"rng_state":self.rng_state});let text=serde_json::to_string(&root).map_err(|e|e.to_string())?;fs::write(path,text).map_err(|e|e.to_string())}
 pub fn load(path:impl AsRef<Path>)->Result<Self,String>{let text=fs::read_to_string(path).map_err(|e|e.to_string())?;let root:Value=serde_json::from_str(&text).map_err(|e|e.to_string())?;if root.get("format").and_then(Value::as_str)!=Some("smaul-rust-checkpoint-v1"){return Err("unsupported checkpoint format".into())}let tr=root.get("training").ok_or("checkpoint has no training state")?;let training=TrainingState{step:tr.get("step").and_then(Value::as_u64).ok_or("invalid step")? as usize,epoch:tr.get("epoch").and_then(Value::as_u64).ok_or("invalid epoch")? as usize,tokens_seen:tr.get("tokens_seen").and_then(Value::as_u64).ok_or("invalid tokens_seen")?,micro_steps:tr.get("micro_steps").and_then(Value::as_u64).ok_or("invalid micro_steps")? as usize,optimizer_steps:tr.get("optimizer_steps").and_then(Value::as_u64).ok_or("invalid optimizer_steps")? as usize};let mut tensors=BTreeMap::new();if let Some(obj)=root.get("tensors").and_then(Value::as_object){for(name,value)in obj{let shape=value.get("shape").and_then(Value::as_array).ok_or("invalid tensor shape")?.iter().map(|x|x.as_u64().ok_or("invalid tensor dimension").map(|v|v as usize)).collect::<Result<Vec<_>,_>>()?;let data=value.get("data").and_then(Value::as_array).ok_or("invalid tensor data")?.iter().map(|x|x.as_f64().ok_or("invalid tensor value").map(|v|v as f32)).collect::<Result<Vec<_>,_>>()?;let expected:usize=shape.iter().product();if expected!=data.len(){return Err(format!("tensor '{name}' has invalid element count"));}tensors.insert(name.clone(),CheckpointTensor{shape,data});}}let mut optimizer_state=BTreeMap::new();if let Some(obj)=root.get("optimizer_state").and_then(Value::as_object){for(name,value)in obj{let data=value.as_array().ok_or("invalid optimizer state")?.iter().map(|x|x.as_f64().ok_or("invalid optimizer value").map(|v|v as f32)).collect::<Result<Vec<_>,_>>()?;optimizer_state.insert(name.clone(),data);}}Ok(Self{training,tensors,optimizer_state,tokenizer_metadata:root.get("tokenizer").cloned().unwrap_or_else(||json!({})),rng_state:root.get("rng_state").and_then(Value::as_str).map(str::to_owned)})}
}

#[cfg(test)]
mod tests_training_state{use super::*;#[test]fn tracks_training_progress(){let mut s=TrainingState::new();s.record_micro_step(16);s.record_micro_step(8);s.record_optimizer_step();s.record_epoch();assert_eq!(s.tokens_seen,24);assert_eq!(s.step,1);assert_eq!(s.epoch,1);s.reset_accumulation();assert_eq!(s.micro_steps,0)}#[test]fn checkpoint_round_trip(){let p=std::env::temp_dir().join("smaul-rust-checkpoint.json");let mut c=CheckpointBundle::new(TrainingState{step:7,epoch:2,tokens_seen:1234,micro_steps:3,optimizer_steps:7});c.insert_tensor("embedding",vec![2,3],vec![1.,2.,3.,4.,5.,6.]);c.optimizer_state.insert("lion_momentum".into(),vec![0.1,0.2]);c.tokenizer_metadata=json!({"vocab_size":64});c.rng_state=Some("seed:123".into());c.save(&p).unwrap();let d=CheckpointBundle::load(&p).unwrap();assert_eq!(c,d);let _=fs::remove_file(p);}}
