use ndarray::{Array1, Array2};

use crate::rwkv_model::RwkvModel;
use crate::safetensors::SafetensorsLoader;

fn a1(loader: &SafetensorsLoader, name: &str) -> Result<Array1<f32>, String> { loader.f32_1d(name) }
fn a2(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d(name) }
fn direct_linear(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d_transposed(name) }
fn module_linear(loader: &SafetensorsLoader, name: &str) -> Result<Array2<f32>, String> { loader.f32_2d(name) }

fn expect_shape(actual: &[usize], expected: &[usize], name: &str) -> Result<(), String> {
    if actual != expected { return Err(format!("tensor '{name}' has shape {actual:?}, expected {expected:?}")); }
    Ok(())
}

fn set_norm(loader: &SafetensorsLoader, prefix: &str, weight: &mut Array1<f32>, bias: &mut Array1<f32>) -> Result<(), String> {
    let w = format!("{prefix}.weight");
    let b = format!("{prefix}.bias");
    let next_weight = a1(loader, &w)?;
    let next_bias = a1(loader, &b)?;
    expect_shape(next_weight.shape(), weight.shape(), &w)?;
    expect_shape(next_bias.shape(), bias.shape(), &b)?;
    *weight = next_weight;
    *bias = next_bias;
    Ok(())
}

fn set_linear(loader: &SafetensorsLoader, name: &str, target: &mut Array2<f32>, transposed: bool) -> Result<(), String> {
    let next = if transposed { direct_linear(loader, name)? } else { module_linear(loader, name)? };
    expect_shape(next.shape(), target.shape(), name)?;
    *target = next;
    Ok(())
}

fn load_rwkv_block(loader: &SafetensorsLoader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("rwkv_blocks.{i}");
    let block = &mut model.rwkv_blocks[i];
    if let Some(norm) = block.ln0.as_mut() { set_norm(loader, &format!("{p}.ln0"), &mut norm.weight, &mut norm.bias)?; }
    set_norm(loader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(loader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    let t = &mut block.time_mix;
    for (name, target) in [
        ("x_r", &mut t.x_r), ("x_w", &mut t.x_w), ("x_k", &mut t.x_k),
        ("x_v", &mut t.x_v), ("x_a", &mut t.x_a), ("x_g", &mut t.x_g),
        ("w0", &mut t.w0), ("a0", &mut t.a0), ("k_k", &mut t.k_k), ("k_a", &mut t.k_a),
    ] { let loaded = a1(loader, &format!("{p}.att.{name}"))?; expect_shape(loaded.shape(), target.shape(), &format!("{p}.att.{name}"))?; *target = loaded; }
    for (name, target) in [("w1", &mut t.w1), ("w2", &mut t.w2), ("a1", &mut t.a1), ("a2", &mut t.a2), ("g1", &mut t.g1), ("g2", &mut t.g2), ("r_k", &mut t.r_k)] {
        let loaded = a2(loader, &format!("{p}.att.{name}"))?;
        expect_shape(loaded.shape(), target.shape(), &format!("{p}.att.{name}"))?;
        *target = loaded;
    }
    for (name, target) in [("v1", &mut t.v1), ("v2", &mut t.v2)] {
        if let Some(target) = target.as_mut() { let loaded = a2(loader, &format!("{p}.att.{name}"))?; expect_shape(loaded.shape(), target.shape(), &format!("{p}.att.{name}"))?; *target = loaded; }
    }
    if let Some(target) = t.v0.as_mut() { let loaded = a1(loader, &format!("{p}.att.v0"))?; expect_shape(loaded.shape(), target.shape(), &format!("{p}.att.v0"))?; *target = loaded; }
    set_linear(loader, &format!("{p}.att.receptance.weight"), &mut t.receptance, true)?;
    set_linear(loader, &format!("{p}.att.key.weight"), &mut t.key, true)?;
    set_linear(loader, &format!("{p}.att.value.weight"), &mut t.value, true)?;
    set_linear(loader, &format!("{p}.att.output.weight"), &mut t.output, true)?;
    set_norm(loader, &format!("{p}.att.ln_x"), &mut t.ln_x.weight, &mut t.ln_x.bias)?;
    let c = &mut block.cmix;
    let x_k = a1(loader, &format!("{p}.ffn.x_k"))?;
    expect_shape(x_k.shape(), c.x_k.shape(), &format!("{p}.ffn.x_k"))?;
    c.x_k = x_k;
    set_linear(loader, &format!("{p}.ffn.key.weight"), &mut c.key, true)?;
    set_linear(loader, &format!("{p}.ffn.value.weight"), &mut c.value, true)?;
    Ok(())
}

fn load_moba_block(loader: &SafetensorsLoader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("moba_blocks.{i}");
    let block = &mut model.moba_blocks[i];
    set_norm(loader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(loader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    for (name, target) in [
        ("receptance.weight", &mut block.att.receptance.weight),
        ("key.weight", &mut block.att.key.weight),
        ("value.weight", &mut block.att.value.weight),
        ("output.weight", &mut block.att.output.weight),
    ] { set_linear(loader, &format!("{p}.att.{name}"), target, false)?; }
    let x_k = a1(loader, &format!("{p}.ffn.x_k"))?;
    expect_shape(x_k.shape(), block.ffn.x_k.shape(), &format!("{p}.ffn.x_k"))?;
    block.ffn.x_k = x_k;
    set_linear(loader, &format!("{p}.ffn.key.weight"), &mut block.ffn.key, true)?;
    set_linear(loader, &format!("{p}.ffn.value.weight"), &mut block.ffn.value, true)?;
    Ok(())
}

pub fn load_model_safetensors(model: &mut RwkvModel, path: impl AsRef<std::path::Path>) -> Result<(), String> {
    let loader = SafetensorsLoader::open(path)?;
    let embedding = loader.f32_2d("emb.weight")?;
    expect_shape(embedding.shape(), model.embedding.weight.shape(), "emb.weight")?;
    model.embedding.weight = embedding;
    set_norm(&loader, "ln_out", &mut model.ln_out.weight, &mut model.ln_out.bias)?;
    set_linear(&loader, "head.weight", &mut model.head.weight, false)?;
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

    #[test]
    fn shape_guard_rejects_wrong_dimensions() {
        assert!(expect_shape(&[4, 5], &[4, 6], "test").is_err());
        assert!(expect_shape(&[4, 5], &[4, 5], "test").is_ok());
    }
}