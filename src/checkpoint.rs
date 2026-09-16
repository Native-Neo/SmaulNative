use crate::config::RwkvXConfig;
use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2, IxDyn};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

// ===== safetensors =====

pub struct SafetensorsLoader { data: Vec<u8> }
impl SafetensorsLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> { let path = path.as_ref(); let data = fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?; SafeTensors::deserialize(&data).map_err(|e| format!("invalid safetensors file: {e}"))?; Ok(Self { data }) }
    fn tensors(&self) -> Result<SafeTensors<'_>, String> { SafeTensors::deserialize(&self.data).map_err(|e| format!("invalid safetensors file: {e}")) }
    fn tensor(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>, String> { self.tensors()?.tensor(name).map_err(|e| format!("missing tensor '{name}': {e}")) }
    pub fn names(&self) -> Result<Vec<String>, String> { Ok(self.tensors()?.names().iter().map(|name| (*name).to_owned()).collect()) }
    pub fn metadata(&self, key: &str) -> Result<String, String> { let (_, metadata) = SafeTensors::read_metadata(&self.data).map_err(|e| format!("invalid safetensors metadata: {e}"))?; metadata.metadata().as_ref().and_then(|m| m.get(key)).cloned().ok_or_else(|| format!("missing safetensors metadata '{key}'")) }
    pub fn shape(&self, name: &str) -> Result<Vec<usize>, String> { Ok(self.tensor(name)?.shape().to_vec()) }
    pub fn f32(&self, name: &str) -> Result<ArrayD<f32>, String> { let tensor = self.tensor(name)?; if tensor.dtype() != Dtype::F32 { return Err(format!("tensor '{name}' has dtype {:?}, expected F32", tensor.dtype())); } let bytes = tensor.data(); if bytes.len() % 4 != 0 { return Err(format!("tensor '{name}' has invalid F32 byte length {}", bytes.len())); } let mut values = Vec::with_capacity(bytes.len() / 4); for chunk in bytes.chunks_exact(4) { values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])); } ArrayD::from_shape_vec(IxDyn(tensor.shape()), values).map_err(|e| format!("invalid shape for tensor '{name}': {e}")) }
    pub fn f32_1d(&self, name: &str) -> Result<Array1<f32>, String> { self.f32(name)?.into_dimensionality::<Ix1>().map_err(|e| format!("tensor '{name}' is not rank 1: {e}")) }
    pub fn f32_2d(&self, name: &str) -> Result<Array2<f32>, String> { self.f32(name)?.into_dimensionality::<Ix2>().map_err(|e| format!("tensor '{name}' is not rank 2: {e}")) }
    pub fn f32_2d_transposed(&self, name: &str) -> Result<Array2<f32>, String> { // reversed_axes alone leaves column-major strides; parameters must stay row-major
        Ok(self.f32_2d(name)?.reversed_axes().as_standard_layout().into_owned()) }
}

#[cfg(test)]
mod tests_safetensors {
    use super::*; use safetensors::tensor::{serialize, Dtype, TensorView}; use std::collections::HashMap;
    #[test] fn reads_f32_matrix_and_transpose() { let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; let mut raw = Vec::with_capacity(values.len() * 4); for value in values { raw.extend_from_slice(&value.to_le_bytes()); } let view = TensorView::new(Dtype::F32, vec![2, 3], &raw).unwrap(); let mut tensors = HashMap::new(); tensors.insert("weight", view); let bytes = serialize(tensors, &None).unwrap(); let loader = SafetensorsLoader { data: bytes }; let matrix = loader.f32_2d("weight").unwrap(); assert_eq!(matrix.shape(), &[2, 3]); assert_eq!(matrix[[1, 2]], 6.0); let transposed = loader.f32_2d_transposed("weight").unwrap(); assert_eq!(transposed.shape(), &[3, 2]); assert_eq!(transposed[[2, 1]], 6.0); assert!(transposed.as_slice().is_some(), "transposed tensor must be row-major contiguous"); }
}


// ===== checkpoint_version =====
pub const CHECKPOINT_FORMAT_VERSION:u32=2;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct CheckpointVersion(pub u32);
impl CheckpointVersion{pub fn current()->Self{Self(CHECKPOINT_FORMAT_VERSION)}pub fn is_supported(self)->bool{self.0==CHECKPOINT_FORMAT_VERSION}}
#[cfg(test)]
mod tests_checkpoint_version{use super::*;#[test]fn current_is_supported(){assert!(CheckpointVersion::current().is_supported());assert_eq!(CheckpointVersion::current().0,2);}}


// ===== model_io =====

pub struct PretrainedModel{pub config:RwkvXConfig,pub model:RwkvModel,pub tokenizer:Tokenizer}
impl PretrainedModel{
 pub fn load(directory:impl AsRef<Path>)->Result<Self,String>{let directory=directory.as_ref();let config=RwkvXConfig::load(directory.join("config.json"))?;let tokenizer=Tokenizer::from_json_file(directory.join("tokenizer.json"))?;if tokenizer.vocab_size()!=config.vocab_size{return Err(format!("tokenizer vocabulary is {}, config expects {}",tokenizer.vocab_size(),config.vocab_size));}let mut model=RwkvModel::new(config.to_model_config(),0);load_model_safetensors(&mut model,directory.join("model.safetensors"))?;Ok(Self{config,model,tokenizer})}
 pub fn save_model(&self,directory:impl AsRef<Path>)->Result<(),String>{let directory=directory.as_ref();std::fs::create_dir_all(directory).map_err(|e|e.to_string())?;self.config.save(directory.join("config.json"))?;save_model_safetensors(&self.model,directory.join("model.safetensors"))?;self.tokenizer.save_json(directory.join("tokenizer.json"))?;Ok(())}
 pub fn save(&self,directory:impl AsRef<Path>)->Result<(),String>{self.save_model(directory)}
 pub fn paths(directory:impl AsRef<Path>)->(PathBuf,PathBuf,PathBuf){let directory=directory.as_ref();(directory.join("config.json"),directory.join("model.safetensors"),directory.join("tokenizer.json"))}
}
#[cfg(test)]mod tests_model_io{use super::*;#[test]fn standard_paths_are_stable(){let(config,model,tokenizer)=PretrainedModel::paths("model");assert_eq!(config,PathBuf::from("model/config.json"));assert_eq!(model,PathBuf::from("model/model.safetensors"));assert_eq!(tokenizer,PathBuf::from("model/tokenizer.json"));}}


// ===== model_loader =====
fn a1(l:&SafetensorsLoader,n:&str)->Result<Array1<f32>,String>{l.f32_1d(n)}
fn a2(l:&SafetensorsLoader,n:&str)->Result<Array2<f32>,String>{l.f32_2d(n)}
fn direct_linear(l:&SafetensorsLoader,n:&str)->Result<Array2<f32>,String>{l.f32_2d_transposed(n)}
fn module_linear(l:&SafetensorsLoader,n:&str)->Result<Array2<f32>,String>{l.f32_2d(n)}
fn expect_shape(a:&[usize],e:&[usize],n:&str)->Result<(),String>{if a!=e{Err(format!("tensor '{n}' has shape {a:?}, expected {e:?}"))}else{Ok(())}}
fn validate_schema(l:&SafetensorsLoader,model:&RwkvModel)->Result<(),String>{let version=l.metadata("smaul_checkpoint_version")?.parse::<u32>().map_err(|_|"invalid smaul_checkpoint_version metadata".to_owned())?;if !CheckpointVersion(version).is_supported(){return Err(format!("unsupported checkpoint format version {version}"));}let layout=l.metadata("smaul_tensor_layout")?;if layout!=TENSOR_LAYOUT_VERSION{return Err(format!("unsupported tensor layout '{layout}', expected '{TENSOR_LAYOUT_VERSION}'"));}let expected=expected_tensor_names(model);let actual=l.names()?.into_iter().collect::<std::collections::BTreeSet<_>>();if actual!=expected{let missing=expected.difference(&actual).cloned().collect::<Vec<_>>();let extra=actual.difference(&expected).cloned().collect::<Vec<_>>();return Err(format!("checkpoint tensor schema mismatch; missing={missing:?}, extra={extra:?}"));}Ok(())}
fn set_norm(l:&SafetensorsLoader,p:&str,w:&mut Array1<f32>,b:&mut Array1<f32>)->Result<(),String>{let wn=format!("{p}.weight");let bn=format!("{p}.bias");let nw=a1(l,&wn)?;let nb=a1(l,&bn)?;expect_shape(nw.shape(),w.shape(),&wn)?;expect_shape(nb.shape(),b.shape(),&bn)?;*w=nw;*b=nb;Ok(())}
fn set_linear(l:&SafetensorsLoader,n:&str,t:&mut Array2<f32>,tr:bool)->Result<(),String>{let x=if tr{direct_linear(l,n)?}else{module_linear(l,n)?};expect_shape(x.shape(),t.shape(),n)?;*t=x;Ok(())}
fn load_rwkv_block(l:&SafetensorsLoader,m:&mut RwkvModel,i:usize)->Result<(),String>{let p=format!("rwkv_blocks.{i}");let b=&mut m.rwkv_blocks[i];if let Some(n)=b.ln0.as_mut(){set_norm(l,&format!("{p}.ln0"),&mut n.weight,&mut n.bias)?;}set_norm(l,&format!("{p}.ln1"),&mut b.ln1.weight,&mut b.ln1.bias)?;set_norm(l,&format!("{p}.ln2"),&mut b.ln2.weight,&mut b.ln2.bias)?;let t=&mut b.time_mix;for(n,x)in[("x_r",&mut t.x_r),("x_w",&mut t.x_w),("x_k",&mut t.x_k),("x_v",&mut t.x_v),("x_a",&mut t.x_a),("x_g",&mut t.x_g),("w0",&mut t.w0),("a0",&mut t.a0),("k_k",&mut t.k_k),("k_a",&mut t.k_a)]{let z=a1(l,&format!("{p}.att.{n}"))?;expect_shape(z.shape(),x.shape(),&format!("{p}.att.{n}"))?;*x=z;}for(n,x)in[("w1",&mut t.w1),("w2",&mut t.w2),("a1",&mut t.a1),("a2",&mut t.a2),("g1",&mut t.g1),("g2",&mut t.g2),("r_k",&mut t.r_k)]{let z=a2(l,&format!("{p}.att.{n}"))?;expect_shape(z.shape(),x.shape(),&format!("{p}.att.{n}"))?;*x=z;}for(n,x)in[("v1",&mut t.v1),("v2",&mut t.v2)]{if let Some(x)=x.as_mut(){let z=a2(l,&format!("{p}.att.{n}"))?;expect_shape(z.shape(),x.shape(),&format!("{p}.att.{n}"))?;*x=z;}}if let Some(x)=t.v0.as_mut(){let z=a1(l,&format!("{p}.att.v0"))?;expect_shape(z.shape(),x.shape(),&format!("{p}.att.v0"))?;*x=z;}set_linear(l,&format!("{p}.att.receptance.weight"),&mut t.receptance,true)?;set_linear(l,&format!("{p}.att.key.weight"),&mut t.key,true)?;set_linear(l,&format!("{p}.att.value.weight"),&mut t.value,true)?;set_linear(l,&format!("{p}.att.output.weight"),&mut t.output,true)?;set_norm(l,&format!("{p}.att.ln_x"),&mut t.ln_x.weight,&mut t.ln_x.bias)?;match &mut b.moe{Some(moe)=>{let rw=a2(l,&format!("{p}.ffn.router.weight"))?;expect_shape(rw.shape(),moe.router.weight.shape(),&format!("{p}.ffn.router.weight"))?;moe.router.weight=rw;for(i,e)in moe.experts.iter_mut().enumerate(){let x=a1(l,&format!("{p}.ffn.experts.{i}.x_k"))?;expect_shape(x.shape(),e.x_k.shape(),&format!("{p}.ffn.experts.{i}.x_k"))?;e.x_k=x;set_linear(l,&format!("{p}.ffn.experts.{i}.key.weight"),&mut e.key,true)?;set_linear(l,&format!("{p}.ffn.experts.{i}.value.weight"),&mut e.value,true)?;}}None=>{let x=a1(l,&format!("{p}.ffn.x_k"))?;expect_shape(x.shape(),b.cmix.x_k.shape(),&format!("{p}.ffn.x_k"))?;b.cmix.x_k=x;set_linear(l,&format!("{p}.ffn.key.weight"),&mut b.cmix.key,true)?;set_linear(l,&format!("{p}.ffn.value.weight"),&mut b.cmix.value,true)?;}}b.time_mix.refresh_quantized();Ok(())}
fn load_moba_block(l:&SafetensorsLoader,m:&mut RwkvModel,i:usize)->Result<(),String>{let p=format!("moba_blocks.{i}");let b=&mut m.moba_blocks[i];set_norm(l,&format!("{p}.ln1"),&mut b.ln1.weight,&mut b.ln1.bias)?;set_norm(l,&format!("{p}.ln2"),&mut b.ln2.weight,&mut b.ln2.bias)?;for(n,x)in[("receptance.weight",&mut b.att.receptance.weight),("key.weight",&mut b.att.key.weight),("value.weight",&mut b.att.value.weight),("output.weight",&mut b.att.output.weight)]{set_linear(l,&format!("{p}.att.{n}"),x,false)?;}let x=a1(l,&format!("{p}.ffn.x_k"))?;expect_shape(x.shape(),b.ffn.x_k.shape(),&format!("{p}.ffn.x_k"))?;b.ffn.x_k=x;set_linear(l,&format!("{p}.ffn.key.weight"),&mut b.ffn.key,true)?;set_linear(l,&format!("{p}.ffn.value.weight"),&mut b.ffn.value,true)?;Ok(())}
pub fn load_model_safetensors(model:&mut RwkvModel,path:impl AsRef<std::path::Path>)->Result<(),String>{let l=SafetensorsLoader::open(path)?;validate_schema(&l,model)?;let e=l.f32_2d("emb.weight")?;expect_shape(e.shape(),model.embedding.weight.shape(),"emb.weight")?;model.embedding.weight=e;set_norm(&l,"ln_out",&mut model.ln_out.weight,&mut model.ln_out.bias)?;set_linear(&l,"head.weight",&mut model.head.weight,false)?;for i in 0..model.rwkv_blocks.len(){load_rwkv_block(&l,model,i)?;}for i in 0..model.moba_blocks.len(){load_moba_block(&l,model,i)?;}Ok(())}
#[cfg(test)]mod tests_model_loader{use super::*;use save_model_safetensors;use crate::rwkv_model::RwkvModelConfig;use std::path::Path;#[test]fn loader_rejects_missing_checkpoint(){let mut model=RwkvModel::new(RwkvModelConfig::new(32,16,2,4),1);assert!(load_model_safetensors(&mut model,Path::new("/definitely/missing/model.safetensors")).is_err());}#[test]fn shape_guard_rejects_wrong_dimensions(){assert!(expect_shape(&[4,5],&[4,6],"test").is_err());assert!(expect_shape(&[4,5],&[4,5],"test").is_ok());}#[test]fn saved_checkpoint_round_trips_schema(){let mut model=RwkvModel::new(RwkvModelConfig::new(16,8,1,4),1);let path=std::env::temp_dir().join("smaul-native-loader-schema.safetensors");save_model_safetensors(&model,&path).unwrap();assert!(load_model_safetensors(&mut model,&path).is_ok());let _=std::fs::remove_file(path);}}


// ===== model_saver =====

pub const TENSOR_LAYOUT_VERSION: &str = "smaul-native-row-major-v1";

fn bytes_1d(a: &Array1<f32>) -> Vec<u8> { a.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn bytes_2d(a: &Array2<f32>) -> Vec<u8> { let a = a.as_standard_layout(); a.iter().flat_map(|v| v.to_le_bytes()).collect() }
struct OwnedTensor { name: String, shape: Vec<usize>, bytes: Vec<u8> }
impl OwnedTensor {
    fn one(name: impl Into<String>, a: &Array1<f32>) -> Self { Self { name: name.into(), shape: vec![a.len()], bytes: bytes_1d(a) } }
    fn two(name: impl Into<String>, a: &Array2<f32>) -> Self { Self { name: name.into(), shape: vec![a.nrows(), a.ncols()], bytes: bytes_2d(a) } }
    fn two_transposed(name: impl Into<String>, a: &Array2<f32>) -> Self { let t = a.t().to_owned(); Self::two(name, &t) }
}
fn add_norm(v: &mut Vec<OwnedTensor>, p: &str, w: &Array1<f32>, b: &Array1<f32>) { v.push(OwnedTensor::one(format!("{p}.weight"), w)); v.push(OwnedTensor::one(format!("{p}.bias"), b)); }
fn add_linear(v: &mut Vec<OwnedTensor>, n: &str, w: &Array2<f32>, tr: bool) { v.push(if tr { OwnedTensor::two_transposed(n, w) } else { OwnedTensor::two(n, w) }); }
fn add_rwkv_block(v: &mut Vec<OwnedTensor>, model: &RwkvModel, index: usize) {
    let b = &model.rwkv_blocks[index]; let p = format!("rwkv_blocks.{index}");
    if let Some(n) = &b.ln0 { add_norm(v, &format!("{p}.ln0"), &n.weight, &n.bias); }
    add_norm(v, &format!("{p}.ln1"), &b.ln1.weight, &b.ln1.bias); add_norm(v, &format!("{p}.ln2"), &b.ln2.weight, &b.ln2.bias);
    let t = &b.time_mix;
    for (n, x) in [("x_r", &t.x_r), ("x_w", &t.x_w), ("x_k", &t.x_k), ("x_v", &t.x_v), ("x_a", &t.x_a), ("x_g", &t.x_g), ("w0", &t.w0), ("a0", &t.a0), ("k_k", &t.k_k), ("k_a", &t.k_a)] { v.push(OwnedTensor::one(format!("{p}.att.{n}"), x)); }
    for (n, x) in [("w1", &t.w1), ("w2", &t.w2), ("a1", &t.a1), ("a2", &t.a2), ("g1", &t.g1), ("g2", &t.g2), ("r_k", &t.r_k)] { v.push(OwnedTensor::two(format!("{p}.att.{n}"), x)); }
    if let Some(x) = &t.v1 { v.push(OwnedTensor::two(format!("{p}.att.v1"), x)); } if let Some(x) = &t.v2 { v.push(OwnedTensor::two(format!("{p}.att.v2"), x)); } if let Some(x) = &t.v0 { v.push(OwnedTensor::one(format!("{p}.att.v0"), x)); }
    add_linear(v, &format!("{p}.att.receptance.weight"), &t.receptance, true); add_linear(v, &format!("{p}.att.key.weight"), &t.key, true); add_linear(v, &format!("{p}.att.value.weight"), &t.value, true); add_linear(v, &format!("{p}.att.output.weight"), &t.output, true); add_norm(v, &format!("{p}.att.ln_x"), &t.ln_x.weight, &t.ln_x.bias);
    match &b.moe {
        Some(m) => { v.push(OwnedTensor::two(format!("{p}.ffn.router.weight"), &m.router.weight)); for (i, e) in m.experts.iter().enumerate() { v.push(OwnedTensor::one(format!("{p}.ffn.experts.{i}.x_k"), &e.x_k)); add_linear(v, &format!("{p}.ffn.experts.{i}.key.weight"), &e.key, true); add_linear(v, &format!("{p}.ffn.experts.{i}.value.weight"), &e.value, true); } }
        None => { v.push(OwnedTensor::one(format!("{p}.ffn.x_k"), &b.cmix.x_k)); add_linear(v, &format!("{p}.ffn.key.weight"), &b.cmix.key, true); add_linear(v, &format!("{p}.ffn.value.weight"), &b.cmix.value, true); }
    }
}
fn add_moba_block(v: &mut Vec<OwnedTensor>, model: &RwkvModel, index: usize) {
    let b = &model.moba_blocks[index]; let p = format!("moba_blocks.{index}"); add_norm(v, &format!("{p}.ln1"), &b.ln1.weight, &b.ln1.bias); add_norm(v, &format!("{p}.ln2"), &b.ln2.weight, &b.ln2.bias);
    add_linear(v, &format!("{p}.att.receptance.weight"), &b.att.receptance.weight, false); add_linear(v, &format!("{p}.att.key.weight"), &b.att.key.weight, false); add_linear(v, &format!("{p}.att.value.weight"), &b.att.value.weight, false); add_linear(v, &format!("{p}.att.output.weight"), &b.att.output.weight, false);
    v.push(OwnedTensor::one(format!("{p}.ffn.x_k"), &b.ffn.x_k)); add_linear(v, &format!("{p}.ffn.key.weight"), &b.ffn.key, true); add_linear(v, &format!("{p}.ffn.value.weight"), &b.ffn.value, true);
}
pub fn expected_tensor_names(model: &RwkvModel) -> BTreeSet<String> { let mut owned = Vec::new(); owned.push(OwnedTensor::two("emb.weight", &model.embedding.weight)); add_norm(&mut owned, "ln_out", &model.ln_out.weight, &model.ln_out.bias); add_linear(&mut owned, "head.weight", &model.head.weight, false); for i in 0..model.rwkv_blocks.len() { add_rwkv_block(&mut owned, model, i); } for i in 0..model.moba_blocks.len() { add_moba_block(&mut owned, model, i); } owned.into_iter().map(|t| t.name).collect() }
pub fn save_model_safetensors(model: &RwkvModel, path: impl AsRef<Path>) -> Result<(), String> {
    let mut owned = Vec::new(); owned.push(OwnedTensor::two("emb.weight", &model.embedding.weight)); add_norm(&mut owned, "ln_out", &model.ln_out.weight, &model.ln_out.bias); add_linear(&mut owned, "head.weight", &model.head.weight, false); for i in 0..model.rwkv_blocks.len() { add_rwkv_block(&mut owned, model, i); } for i in 0..model.moba_blocks.len() { add_moba_block(&mut owned, model, i); }
    let mut tensors = HashMap::with_capacity(owned.len()); for t in &owned { let view = TensorView::new(Dtype::F32, t.shape.clone(), &t.bytes).map_err(|e| format!("failed to build tensor '{}': {e}", t.name))?; tensors.insert(t.name.as_str(), view); }
    let mut metadata = HashMap::new(); metadata.insert("smaul_checkpoint_version".to_owned(), CheckpointVersion::current().0.to_string()); metadata.insert("smaul_tensor_layout".to_owned(), TENSOR_LAYOUT_VERSION.to_owned());
    serialize_to_file(tensors, &Some(metadata), path.as_ref()).map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}

#[cfg(test)]
mod tests_model_saver { use super::*; use crate::rwkv_model::{RwkvModel, RwkvModelConfig}; use SafetensorsLoader;
    #[test] fn saved_model_contains_loader_tensors() { let model = RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4), 7); let path = std::env::temp_dir().join("smaul-native-model.safetensors"); save_model_safetensors(&model, &path).unwrap(); let loader = SafetensorsLoader::open(&path).unwrap(); let names = loader.names().unwrap(); assert!(names.contains(&"emb.weight".to_owned())); assert!(names.contains(&"ln_out.weight".to_owned())); assert!(names.contains(&"head.weight".to_owned())); assert!(names.contains(&"rwkv_blocks.0.att.key.weight".to_owned())); assert_eq!(loader.metadata("smaul_checkpoint_version").unwrap(), "2"); assert_eq!(loader.metadata("smaul_tensor_layout").unwrap(), TENSOR_LAYOUT_VERSION); let _ = std::fs::remove_file(path); }
    #[test] fn saved_moe_contains_router_and_experts() { let model = RwkvModel::new(RwkvModelConfig::new(16, 8, 1, 4).with_moe(true, 3, 2), 7); let path = std::env::temp_dir().join("smaul-native-moe.safetensors"); save_model_safetensors(&model, &path).unwrap(); let loader = SafetensorsLoader::open(&path).unwrap(); let names = loader.names().unwrap(); assert!(names.contains(&"rwkv_blocks.0.ffn.router.weight".to_owned())); assert!(names.contains(&"rwkv_blocks.0.ffn.experts.2.key.weight".to_owned())); let _ = std::fs::remove_file(path); }
}
