use crate::dataset::{MultiFileTextStream, ParquetTextStream, TextStream};
use crate::dataset_multi::MultiFileDatasetStream;
use crate::model_train_step::ModelTrainStep;
use crate::rwkv_model::RwkvModel;
use crate::training::TrainStep;
use crate::training_state::{CheckpointBundle, TrainingState};
use ndarray::Array1;
use std::path::{Path, PathBuf};

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
  crate::model_loader::load_model_safetensors(model,&model_path)?;
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
  crate::model_saver::save_model_safetensors(model,directory.join(format!("model-{step}.safetensors")))?;
  let mut bundle=CheckpointBundle::new(self.state.clone()); bundle.optimizer_state.insert("lion.momentum".into(),self.optimizer.optimizer_state());
  bundle.save(directory.join(format!("training-{step}.json")))?; Ok(())
 }
 pub fn save_state(&self,path:impl AsRef<Path>)->Result<(),String>{let mut bundle=CheckpointBundle::new(self.state.clone());bundle.optimizer_state.insert("lion.momentum".into(),self.optimizer.optimizer_state());bundle.save(path)}
}

#[cfg(test)]
mod tests{
 use super::*; use crate::rwkv_model::RwkvModelConfig; use crate::tokenizer::Tokenizer; use std::fs;
 fn tokenizer()->Tokenizer{Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into(),"b".into()," ".into()])}
 #[test]fn trains_native_stream(){let p=std::env::temp_dir().join("smaul-train.txt");fs::write(&p,"a b a b a b a b").unwrap();let mut s=TextStream::open(&p,tokenizer(),3).unwrap();let mut m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let state=r.run_stream(&mut m,&mut s,&TrainingConfig{max_steps:1,..Default::default()}).unwrap();assert_eq!(state.step,1);assert!(state.tokens_seen>=3);let _=fs::remove_file(p);}
 #[test]fn trains_multiple_files(){let d=std::env::temp_dir().join("smaul-train-multi");fs::create_dir_all(&d).unwrap();fs::write(d.join("a.txt"),"a b a b a b").unwrap();fs::write(d.join("b.txt"),"b a b a b a").unwrap();let mut s=MultiFileTextStream::open_discovered(&d,tokenizer(),3).unwrap();let mut m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let state=r.run_multi_stream(&mut m,&mut s,&TrainingConfig{max_steps:2,..Default::default()}).unwrap();assert_eq!(state.step,2);let _=fs::remove_dir_all(d);}
 #[test]fn parquet_missing_file_is_reported(){let p=std::env::temp_dir().join("smaul-not-parquet.parquet");fs::write(&p,b"not parquet").unwrap();assert!(ParquetTextStream::open(&p,tokenizer(),3).is_err());let _=fs::remove_file(p);}
 #[test]fn checkpoint_contains_lion_state(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("smaul-training-state.json");r.save_state(&p).unwrap();let c=CheckpointBundle::load(&p).unwrap();assert_eq!(c.optimizer_state.get("lion.momentum").unwrap().len(),m.parameter_count());let _=fs::remove_file(p);}
 #[test]fn resume_restores_training_state(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("smaul-training-resume.json");r.save_state(&p).unwrap();let mut resumed=TrainingRunner::new(&m,1e-4);resumed.resume_from_checkpoint(&p).unwrap();assert_eq!(resumed.state,r.state);assert_eq!(resumed.optimizer.optimizer_state(),r.optimizer.optimizer_state());let _=fs::remove_file(p);}
 #[test]fn resume_checkpoint_rejects_wrong_filename(){let m=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);let mut r=TrainingRunner::new(&m,1e-4);let p=std::env::temp_dir().join("resume.json");r.save_state(&p).unwrap();let mut loaded=RwkvModel::new(RwkvModelConfig::new(9,8,1,4),42);assert!(r.resume_checkpoint(&mut loaded,&p).is_err());assert_eq!(r.state.step,0);let _=fs::remove_file(p);}
}