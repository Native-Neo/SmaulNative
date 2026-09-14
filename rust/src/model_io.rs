use crate::config::RwkvXConfig;
use crate::model_loader::load_model_safetensors;
use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use std::path::{Path,PathBuf};

pub struct PretrainedModel{pub config:RwkvXConfig,pub model:RwkvModel,pub tokenizer:Tokenizer}
impl PretrainedModel{
 pub fn load(directory:impl AsRef<Path>)->Result<Self,String>{let directory=directory.as_ref();let config=RwkvXConfig::load(directory.join("config.json"))?;let tokenizer=Tokenizer::from_json_file(directory.join("tokenizer.json"))?;if tokenizer.vocab_size()!=config.vocab_size{return Err(format!("tokenizer vocabulary is {}, config expects {}",tokenizer.vocab_size(),config.vocab_size));}let mut model=RwkvModel::new(config.to_model_config(),0);load_model_safetensors(&mut model,directory.join("model.safetensors"))?;Ok(Self{config,model,tokenizer})}
 pub fn save_model(&self,directory:impl AsRef<Path>)->Result<(),String>{let directory=directory.as_ref();std::fs::create_dir_all(directory).map_err(|e|e.to_string())?;self.config.save(directory.join("config.json"))?;crate::model_saver::save_model_safetensors(&self.model,directory.join("model.safetensors"))?;self.tokenizer.save_json(directory.join("tokenizer.json"))?;Ok(())}
 pub fn save(&self,directory:impl AsRef<Path>)->Result<(),String>{self.save_model(directory)}
 pub fn paths(directory:impl AsRef<Path>)->(PathBuf,PathBuf,PathBuf){let directory=directory.as_ref();(directory.join("config.json"),directory.join("model.safetensors"),directory.join("tokenizer.json"))}
}
#[cfg(test)]mod tests{use super::*;#[test]fn standard_paths_are_stable(){let(config,model,tokenizer)=PretrainedModel::paths("model");assert_eq!(config,PathBuf::from("model/config.json"));assert_eq!(model,PathBuf::from("model/model.safetensors"));assert_eq!(tokenizer,PathBuf::from("model/tokenizer.json"));}}
