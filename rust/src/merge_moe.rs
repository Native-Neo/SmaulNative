use crate::config::RwkvXConfig;
use crate::moe::{MoeCmix, MoeRouter};
use crate::model_io::PretrainedModel;
use crate::rwkv_cmix::RwkvCmix;
use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::{Path, PathBuf};

fn compatible(a: &RwkvXConfig, b: &RwkvXConfig, path: &Path) -> Result<(), String> {
    for (name, x, y) in [("n_embd", a.n_embd, b.n_embd), ("n_layer", a.n_layer, b.n_layer), ("n_moba_layer", a.n_moba_layer, b.n_moba_layer), ("head_size", a.head_size, b.head_size), ("moba_chunk_size", a.moba_chunk_size, b.moba_chunk_size), ("moba_topk", a.moba_topk, b.moba_topk), ("qat_bits", a.effective_qat_bits(), b.effective_qat_bits())] {
        if x != y { return Err(format!("{}: {name}={y} does not match base {x}", path.display())); }
    }
    Ok(())
}

fn branch_experts(model: &RwkvModel, block: usize) -> Vec<RwkvCmix> {
    let block = &model.rwkv_blocks[block];
    block.moe.as_ref().map(|m| m.experts.clone()).unwrap_or_else(|| vec![block.cmix.clone()])
}

fn resize_vocab(weight: &Array2<f32>, target: usize, seed: u64) -> Array2<f32> {
    if target <= weight.nrows() { return weight.clone(); }
    let mut out = Array2::zeros((target, weight.ncols()));
    for r in 0..weight.nrows() { out.row_mut(r).assign(&weight.row(r)); }
    let mean = weight.iter().sum::<f32>() / weight.len().max(1) as f32;
    let variance = weight.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / weight.len().max(1) as f32;
    let std = variance.sqrt().max(0.02);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for r in weight.nrows()..target {
        for c in 0..weight.ncols() {
            let u1 = rng.random::<f32>().max(1e-7);
            let u2 = rng.random::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            out[[r, c]] = mean + std * z;
        }
    }
    out
}

fn merged_vocab(base: &Tokenizer, branches: &[Tokenizer]) -> Vec<String> {
    let mut out = (0..base.vocab_size()).filter_map(|i| base.id_to_token(i).map(str::to_owned)).collect::<Vec<_>>();
    let mut seen = out.iter().cloned().collect::<std::collections::HashSet<_>>();
    for tokenizer in branches {
        for i in 0..tokenizer.vocab_size() {
            if let Some(token) = tokenizer.id_to_token(i) {
                if seen.insert(token.to_owned()) { out.push(token.to_owned()); }
            }
        }
    }
    out
}

pub fn merge(base_dir: impl AsRef<Path>, branch_dirs: &[PathBuf], out_dir: impl AsRef<Path>, top_k: usize) -> Result<(), String> {
    if branch_dirs.is_empty() { return Err("at least one branch is required".into()); }
    if top_k == 0 { return Err("top_k must be at least 1".into()); }
    let base_dir = base_dir.as_ref();
    let base = PretrainedModel::load(base_dir)?;
    let mut branches = Vec::with_capacity(branch_dirs.len());
    let mut tokenizers = vec![base.tokenizer.clone()];
    for path in branch_dirs {
        let branch = PretrainedModel::load(path)?;
        compatible(&base.config, &branch.config, path)?;
        tokenizers.push(branch.tokenizer.clone());
        branches.push(branch);
    }
    let counts = branches.iter().map(|m| if m.config.is_moe { m.config.num_experts } else { 1 }).collect::<Vec<_>>();
    let num_experts: usize = counts.iter().sum();
    if top_k > num_experts { return Err(format!("top_k ({top_k}) cannot exceed merged expert count ({num_experts})")); }
    let vocab = merged_vocab(&base.tokenizer, &tokenizers[1..]);
    let mut config = base.config.clone();
    config.vocab_size = vocab.len();
    config.is_moe = true;
    config.num_experts = num_experts;
    config.num_experts_per_tok = top_k;
    let mut merged = RwkvModel::new(config.to_model_config(), 0x4d4f_455f_4d455247);
    merged.embedding.weight = resize_vocab(&base.model.embedding.weight, vocab.len(), 0x454d_4245);
    merged.head.weight = resize_vocab(&base.model.head.weight, vocab.len(), 0x48454144);
    merged.ln_out = base.model.ln_out.clone();
    for i in 0..merged.rwkv_blocks.len() {
        merged.rwkv_blocks[i].ln0 = base.model.rwkv_blocks[i].ln0.clone();
        merged.rwkv_blocks[i].ln1 = base.model.rwkv_blocks[i].ln1.clone();
        merged.rwkv_blocks[i].ln2 = base.model.rwkv_blocks[i].ln2.clone();
        merged.rwkv_blocks[i].time_mix = base.model.rwkv_blocks[i].time_mix.clone();
        let mut experts = Vec::with_capacity(num_experts);
        for model in &branches { experts.extend(branch_experts(&model.model, i)); }
        let router = MoeRouter::new(config.n_embd, num_experts, top_k, 0x524f55544552 ^ i as u64);
        merged.rwkv_blocks[i].moe = Some(MoeCmix::from_experts(experts, router)?);
    }
    merged.moba_blocks = base.model.moba_blocks.iter().map(Clone::clone).collect();
    let tokenizer = Tokenizer::from_vocab(vocab);
    let output = out_dir.as_ref();
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let result = PretrainedModel { config, model: merged, tokenizer };
    result.save(output)?;
    let meta = serde_json::json!({"engine":"smaul-native merge-moe","base_model":base_dir,"branches":branch_dirs,"branch_expert_counts":counts,"num_experts":num_experts,"top_k":top_k,"tokenizer":{"base_vocab_size":base.tokenizer.vocab_size(),"merged_vocab_size":result.tokenizer.vocab_size()},"note":"Channel-Mix experts are concatenated and routers are freshly initialized."});
    fs::write(output.join("merge_config.json"), serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    println!("[DONE] merged model -> {} ({} experts, vocab_size={})", output.display(), num_experts, result.tokenizer.vocab_size());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vocab_merge_preserves_base_ids() {
        let a = Tokenizer::from_vocab(vec!["a".into(), "b".into()]);
        let b = Tokenizer::from_vocab(vec!["b".into(), "c".into()]);
        assert_eq!(merged_vocab(&a, &[b]), vec!["a", "b", "c"]);
    }
}
