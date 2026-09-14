use crate::gguf::{GgufReader, GgufValue};
use crate::rwkv_model::{RwkvModel, RwkvModelConfig};
use crate::tokenizer::Tokenizer;
use ndarray::{Array1, Array2};

fn metadata_usize(reader: &GgufReader, key: &str) -> Result<usize, String> {
    match reader.metadata.get(key) {
        Some(GgufValue::U32(v)) => Ok(*v as usize),
        Some(GgufValue::U64(v)) => usize::try_from(*v).map_err(|_| format!("GGUF metadata '{key}' is too large")),
        Some(_) => Err(format!("GGUF metadata '{key}' is not an integer")),
        None => Err(format!("missing GGUF metadata '{key}'")),
    }
}

fn tensor1(reader: &mut GgufReader, name: &str, expected: usize) -> Result<Array1<f32>, String> {
    let info = reader.tensor(name).cloned().ok_or_else(|| format!("GGUF tensor '{name}' not found"))?;
    if info.shape != [expected as u64] { return Err(format!("tensor '{name}' has shape {:?}, expected [{expected}]", info.shape)); }
    Array1::from_shape_vec(expected, reader.tensor_f32(name)?).map_err(|e| format!("invalid tensor '{name}': {e}"))
}

fn tensor2(reader: &mut GgufReader, name: &str, rows: usize, cols: usize) -> Result<Array2<f32>, String> {
    let info = reader.tensor(name).cloned().ok_or_else(|| format!("GGUF tensor '{name}' not found"))?;
    if info.shape != [rows as u64, cols as u64] { return Err(format!("tensor '{name}' has shape {:?}, expected [{rows}, {cols}]", info.shape)); }
    Array2::from_shape_vec((rows, cols), reader.tensor_f32(name)?).map_err(|e| format!("invalid tensor '{name}': {e}"))
}

fn tensor2_transposed(reader: &mut GgufReader, name: &str, rows: usize, cols: usize) -> Result<Array2<f32>, String> {
    Ok(tensor2(reader, name, cols, rows)?.reversed_axes().to_owned())
}

fn set_norm(reader: &mut GgufReader, prefix: &str, weight: &mut Array1<f32>, bias: &mut Array1<f32>) -> Result<(), String> {
    *weight = tensor1(reader, &format!("{prefix}.weight"), weight.len())?;
    *bias = tensor1(reader, &format!("{prefix}.bias"), bias.len())?;
    Ok(())
}

fn set_linear(reader: &mut GgufReader, name: &str, target: &mut Array2<f32>, transposed: bool) -> Result<(), String> {
    let shape = target.raw_dim();
    *target = if transposed { tensor2_transposed(reader, name, shape[0], shape[1])? } else { tensor2(reader, name, shape[0], shape[1])? };
    Ok(())
}

fn load_rwkv_block(reader: &mut GgufReader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("rwkv_blocks.{i}");
    let block = &mut model.rwkv_blocks[i];
    if let Some(norm) = block.ln0.as_mut() { set_norm(reader, &format!("{p}.ln0"), &mut norm.weight, &mut norm.bias)?; }
    set_norm(reader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(reader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    let t = &mut block.time_mix;
    for (name, target) in [
        ("x_r", &mut t.x_r), ("x_w", &mut t.x_w), ("x_k", &mut t.x_k),
        ("x_v", &mut t.x_v), ("x_a", &mut t.x_a), ("x_g", &mut t.x_g),
        ("w0", &mut t.w0), ("a0", &mut t.a0), ("k_k", &mut t.k_k), ("k_a", &mut t.k_a),
    ] {
        let len = target.len();
        *target = tensor1(reader, &format!("{p}.att.{name}"), len)?;
    }
    for (name, target) in [
        ("w1", &mut t.w1), ("w2", &mut t.w2), ("a1", &mut t.a1),
        ("a2", &mut t.a2), ("g1", &mut t.g1), ("g2", &mut t.g2), ("r_k", &mut t.r_k),
    ] {
        let shape = target.raw_dim();
        *target = tensor2(reader, &format!("{p}.att.{name}"), shape[0], shape[1])?;
    }
    for (name, target) in [("v1", &mut t.v1), ("v2", &mut t.v2)] {
        if let Some(target) = target.as_mut() {
            let shape = target.raw_dim();
            *target = tensor2(reader, &format!("{p}.att.{name}"), shape[0], shape[1])?;
        }
    }
    if let Some(target) = t.v0.as_mut() {
        let len = target.len();
        *target = tensor1(reader, &format!("{p}.att.v0"), len)?;
    }
    set_linear(reader, &format!("{p}.att.receptance.weight"), &mut t.receptance, true)?;
    set_linear(reader, &format!("{p}.att.key.weight"), &mut t.key, true)?;
    set_linear(reader, &format!("{p}.att.value.weight"), &mut t.value, true)?;
    set_linear(reader, &format!("{p}.att.output.weight"), &mut t.output, true)?;
    set_norm(reader, &format!("{p}.att.ln_x"), &mut t.ln_x.weight, &mut t.ln_x.bias)?;
    let c = &mut block.cmix;
    let len = c.x_k.len();
    c.x_k = tensor1(reader, &format!("{p}.ffn.x_k"), len)?;
    set_linear(reader, &format!("{p}.ffn.key.weight"), &mut c.key, true)?;
    set_linear(reader, &format!("{p}.ffn.value.weight"), &mut c.value, true)?;
    Ok(())
}

fn load_moba_block(reader: &mut GgufReader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("moba_blocks.{i}");
    let block = &mut model.moba_blocks[i];
    set_norm(reader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(reader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    for (name, target) in [
        ("receptance.weight", &mut block.att.receptance.weight),
        ("key.weight", &mut block.att.key.weight),
        ("value.weight", &mut block.att.value.weight),
        ("output.weight", &mut block.att.output.weight),
    ] { set_linear(reader, &format!("{p}.att.{name}"), target, false)?; }
    let len = block.ffn.x_k.len();
    block.ffn.x_k = tensor1(reader, &format!("{p}.ffn.x_k"), len)?;
    set_linear(reader, &format!("{p}.ffn.key.weight"), &mut block.ffn.key, true)?;
    set_linear(reader, &format!("{p}.ffn.value.weight"), &mut block.ffn.value, true)?;
    Ok(())
}

pub fn load_gguf_model(path: impl AsRef<std::path::Path>) -> Result<RwkvModel, String> {
    let mut reader = GgufReader::open(path)?;
    let vocab_size = metadata_usize(&reader, "vocab_size")?;
    let n_embd = metadata_usize(&reader, "embedding_length")?;
    let n_layer = metadata_usize(&reader, "block_count")?;
    let head_size = metadata_usize(&reader, "rwkv_x.head_size")?;
    let n_moba_layer = metadata_usize(&reader, "rwkv_x.n_moba_layer")?;
    let chunk_size = metadata_usize(&reader, "rwkv_x.moba_chunk_size")?;
    let topk = metadata_usize(&reader, "rwkv_x.moba_topk")?;
    let mut config = RwkvModelConfig::new(vocab_size, n_embd, n_layer, head_size);
    config.head_size_divisor = metadata_usize(&reader, "rwkv_x.head_size_divisor")?;
    config = config.with_moba(n_moba_layer, chunk_size, topk);
    let mut model = RwkvModel::new(config, 0);
    model.embedding.weight = tensor2(&mut reader, "emb.weight", vocab_size, n_embd)?;
    set_norm(&mut reader, "ln_out", &mut model.ln_out.weight, &mut model.ln_out.bias)?;
    set_linear(&mut reader, "head.weight", &mut model.head.weight, false)?;
    for i in 0..model.rwkv_blocks.len() { load_rwkv_block(&mut reader, &mut model, i)?; }
    for i in 0..model.moba_blocks.len() { load_moba_block(&mut reader, &mut model, i)?; }
    Ok(model)
}

pub fn load_gguf_tokenizer(path: impl AsRef<std::path::Path>) -> Result<Tokenizer, String> {
    let reader = GgufReader::open(path)?;
    let tokens = match reader.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(values)) => values.iter().map(|v| match v { GgufValue::String(s) => Ok(s.clone()), _ => Err("GGUF tokenizer token list contains a non-string value".into()) }).collect::<Result<Vec<_>, String>>()?,
        Some(_) => return Err("GGUF tokenizer token list is not an array".into()),
        None => return Err("missing GGUF metadata 'tokenizer.ggml.tokens'".into()),
    };
    if tokens.is_empty() { return Err("GGUF tokenizer vocabulary is empty".into()); }
    Ok(Tokenizer::from_vocab(tokens))
}

pub fn load_gguf(path: impl AsRef<std::path::Path>) -> Result<(RwkvModel, Tokenizer), String> {
    let path = path.as_ref();
    let model = load_gguf_model(path)?;
    let tokenizer = load_gguf_tokenizer(path)?;
    Ok((model, tokenizer))
}
