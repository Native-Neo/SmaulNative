use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2, IxDyn};
use safetensors::tensor::{Dtype, SafeTensors};
use std::fs;
use std::path::Path;

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
mod tests {
    use super::*; use safetensors::tensor::{serialize, Dtype, TensorView}; use std::collections::HashMap;
    #[test] fn reads_f32_matrix_and_transpose() { let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; let mut raw = Vec::with_capacity(values.len() * 4); for value in values { raw.extend_from_slice(&value.to_le_bytes()); } let view = TensorView::new(Dtype::F32, vec![2, 3], &raw).unwrap(); let mut tensors = HashMap::new(); tensors.insert("weight", view); let bytes = serialize(tensors, &None).unwrap(); let loader = SafetensorsLoader { data: bytes }; let matrix = loader.f32_2d("weight").unwrap(); assert_eq!(matrix.shape(), &[2, 3]); assert_eq!(matrix[[1, 2]], 6.0); let transposed = loader.f32_2d_transposed("weight").unwrap(); assert_eq!(transposed.shape(), &[3, 2]); assert_eq!(transposed[[2, 1]], 6.0); assert!(transposed.as_slice().is_some(), "transposed tensor must be row-major contiguous"); }
}
