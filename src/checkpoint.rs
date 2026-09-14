use std::collections::BTreeMap;
use crate::checkpoint_version::CheckpointVersion;

pub struct TensorCheckpoint {
    version: CheckpointVersion,
    tensors: BTreeMap<String, Vec<f32>>,
}

impl TensorCheckpoint {
    pub fn new() -> Self { Self { version: CheckpointVersion::current(), tensors: BTreeMap::new() } }
    pub fn version(&self) -> CheckpointVersion { self.version }
    pub fn validate_version(&self) -> Result<(), String> { if self.version.is_supported() { Ok(()) } else { Err(format!("unsupported checkpoint format version {}", self.version.0)) } }
    pub fn insert(&mut self, name: impl Into<String>, values: Vec<f32>) { self.tensors.insert(name.into(), values); }
    pub fn get(&self, name: &str) -> Option<&[f32]> { self.tensors.get(name).map(Vec::as_slice) }
    pub fn len(&self) -> usize { self.tensors.len() }
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }
}
impl Default for TensorCheckpoint { fn default() -> Self { Self::new() } }

#[cfg(test)]mod tests{use super::*;#[test]fn checkpoint_stores_named_tensors(){let mut checkpoint=TensorCheckpoint::new();checkpoint.insert("embedding.weight",vec![1.0,2.0]);assert_eq!(checkpoint.len(),1);assert_eq!(checkpoint.get("embedding.weight"),Some(&[1.0,2.0][..]));assert!(!checkpoint.is_empty());assert!(checkpoint.validate_version().is_ok())}}
