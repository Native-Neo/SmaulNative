use crate::rwkv_model::RwkvModel;
use crate::safetensors::SafetensorsLoader;
use ndarray::{Array1, Array2};

fn a1(loader: &SafetensorsLoader, name: &str) -> Result<Array1<f32>, String> { loader.f32_1d(name) }
fn a2(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d(name) }
fn direct_linear(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d_transposed(name) }
fn module_linear(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d(name) }

fn set_norm(loader: &SafetensorsLoader, prefix: &str, weight: &mut Array1<f32>, bias: &mut Array1<f32>) -> Result<(), String> {
    *weight = a1(loader, &format!("{prefix}.weight"))?;
    *bias = a1(loader, &format!("{prefix}.bias"))?;
    Ok(())
}

fn load_rwkv_block(loader: &SafetensorsLoader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let block = model.rwkv_blocks.get_mut(i).ok_or_else(|| format!("missing Rust RWKV block {i}"))?;
    let p = format!("rwkv_blocks.{i}");
    if let Some(norm) = block.ln0.as_mut() { set_norm(loader, &format!("{p}.ln0"), &mut norm.weight, &mut norm.bias)?; }
    set_norm(loader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(loader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    let t = &mut block.time_mix;
    t.x_r = a1(loader, &format!("{p}.att.x_r"))?;
    t.x_w = a1(loader, &format!("{p}.att.x_w"))?;
    t.x_k = a1(loader, &format!("{p}.att.x_k"))?;
    t.x_v = a1(loader, &format!("{p}.att.x_v"))?;
    t.x_a = a1(loader, &format!("{p}.att.x_a"))?;
    t.x_g = a1(loader, &format!("{p}.att.x_g"))?;
    t.w0 = a1(loader, &format!("{p}.att.w0"))?;
    t.w1 = a2(loader, &format!("{p}.att.w1"))?;
    t.w2 = a2(loader, &format!("{p}.att.w2"))?;
    t.a1 = a2(loader, &format!("{p}.att.a1"))?;
    t.a2 = a2(loader, &format!("{p}.att.a2"))?;
    if let Some(v1) = t.v1.as_mut() { *v1 = a2(loader, &format!("{p}.att.v1"))?; }
    if let Some(v2) = t.v2.as_mut() { *v2 = a2(loader, &format!("{p}.att.v2"))?; }
    t.g1 = a2(loader, &format!("{p}.att.g1"))?;
    t.g2 = a2(loader, &format!("{p}.att.g2"))?;
    t.a0 = a1(loader, &format!("{p}.att.a0"))?;
    if let Some(v0) = t.v0.as_mut() { *v0 = a1(loader, &format!("{p}.att.v0"))?; }
    t.k_k = a1(loader, &format!("{p}.att.k_k"))?;
    t.k_a = a1(loader, &format!("{p}.att.k_a"))?;
    t.r_k = a2(loader, &format!("{p}.att.r_k"))?;
    t.receptance = direct_linear(loader, &format!("{p}.att.receptance.weight"))?;
    t.key = direct_linear(loader, &format!("{p}.att.key.weight"))?;
    t.value = direct_linear(loader, &format!("{p}.att.value.weight"))?;
    t.output = direct_linear(loader, &format!("{p}.att.output.weight"))?;
    set_norm(loader, &format!("{p}.att.ln_x"), &mut t.ln_x.weight, &mut t.ln_x.bias)?;
    let c = &mut block.cmix;
    c.x_k = a1(loader, &format!("{p}.ffn.x_k"))?;
    c.key = direct_linear(loader, &format!("{p}.ffn.key.weight"))?;
    c.value = direct_linear(loader, &format!("{p}.ffn.value.weight"))?;
    Ok(())
}

fn load_moba_block(loader: &SafetensorsLoader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let block = model.moba_blocks.get_mut(i).ok_or_else(|| format!("missing Rust MOBA block {i}"))?;
    let p = format!("moba_blocks.{i}");
    set_norm(loader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(loader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    block.att.receptance.weight = module_linear(loader, &format!("{p}.att.receptance.weight"))?;
    block.att.key.weight = module_linear(loader, &format!("{p}.att.key.weight"))?;
    block.att.value.weight = module_linear(loader, &format!("{p}.att.value.weight"))?;
    block.att.output.weight = module_linear(loader, &format!("{p}.att.output.weight"))?;
    block.ffn.x_k = a1(loader, &format!("{p}.ffn.x_k"))?;
    block.ffn.key = direct_linear(loader, &format!("{p}.ffn.key.weight"))?;
    block.ffn.value = direct_linear(loader, &format!("{p}.ffn.value.weight"))?;
    Ok(())
}

pub fn load_model_safetensors(model: &mut RwkvModel, path: impl AsRef<std::path::Path>) -> Result<(), String> {
    let loader = SafetensorsLoader::open(path)?;
    model.embedding.weight = loader.f32_2d("emb.weight")?;
    set_norm(&loader, "ln_out", &mut model.ln_out.weight, &mut model.ln_out.bias)?;
    model.head.weight = module_linear(&loader, "head.weight")?;
    for i in 0..model.rwkv_blocks.len() { load_rwkv_block(&loader, model, i)?; }
    for i in 0..model.moba_blocks.len() { load_moba_block(&loader, model, i)?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;
    use std::path::Path;

    #[test]
    fn loader_rejects_missing_checkpoint() {
        let config = RwkvModelConfig::new(32, 16, 2, 4);
        let mut model = RwkvModel::new(config, 1);
        assert!(load_model_safetensors(&mut model, Path::new("/definitely/missing/model.safetensors")).is_err());
    }
}
