use crate::rwkv_model::RwkvModelConfig;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct RwkvXConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub head_size: usize,
    pub n_moba_layer: usize,
    pub moba_chunk_size: usize,
    pub moba_topk: usize,
    pub dropout: f32,
    pub head_size_divisor: usize,
    pub ctx_len_hint: usize,
    pub wkv_chunk_size: usize,
    pub checkpoint_ffn: bool,
    pub is_moe: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl Default for RwkvXConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65530,
            n_embd: 832,
            n_layer: 17,
            head_size: 64,
            n_moba_layer: 5,
            moba_chunk_size: 512,
            moba_topk: 4,
            dropout: 0.0,
            head_size_divisor: 8,
            ctx_len_hint: 2048,
            wkv_chunk_size: 64,
            checkpoint_ffn: true,
            is_moe: false,
            num_experts: 1,
            num_experts_per_tok: 1,
        }
    }
}

impl RwkvXConfig {
    pub fn to_model_config(&self) -> RwkvModelConfig {
        let mut config = RwkvModelConfig::new(
            self.vocab_size,
            self.n_embd,
            self.n_layer,
            self.head_size,
        );
        config.head_size_divisor = self.head_size_divisor;
        config.with_moba(
            self.n_moba_layer,
            self.moba_chunk_size,
            self.moba_topk,
        )
    }

    pub fn from_model_config(config: &RwkvModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            n_embd: config.n_embd,
            n_layer: config.n_layer,
            head_size: config.head_size,
            n_moba_layer: config.n_moba_layer,
            moba_chunk_size: config.moba_chunk_size,
            moba_topk: config.moba_topk,
            head_size_divisor: config.head_size_divisor,
            ..Self::default()
        }
    }

    pub fn approx_param_count(&self) -> usize {
        let c = self.n_embd;
        let v = self.vocab_size;
        let l = self.n_layer;
        let dd = (1.8 * (c as f64).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dm = (1.3 * (c as f64).sqrt() / 32.0).round().max(1.0) as usize * 32;
        let dg = (0.6 * (c as f64).powf(0.8) / 32.0).round().max(1.0) as usize * 32;
        let tmix = 4 * c * c + c * (4 * dd + 2 * dm + 2 * dg);
        let cmix = 8 * c * c;
        2 * v * c + (l - self.n_moba_layer) * (tmix + cmix)
            + self.n_moba_layer * (4 * c * c + cmix)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let value = json!({
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "head_size": self.head_size,
            "n_moba_layer": self.n_moba_layer,
            "moba_chunk_size": self.moba_chunk_size,
            "moba_topk": self.moba_topk,
            "dropout": self.dropout,
            "head_size_divisor": self.head_size_divisor,
            "ctx_len_hint": self.ctx_len_hint,
            "wkv_chunk_size": self.wkv_chunk_size,
            "checkpoint_ffn": self.checkpoint_ffn,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        });
        let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| e.to_string())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let root: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        Ok(Self {
            vocab_size: usize_field(&root, "vocab_size", Self::default().vocab_size)?,
            n_embd: usize_field(&root, "n_embd", Self::default().n_embd)?,
            n_layer: usize_field(&root, "n_layer", Self::default().n_layer)?,
            head_size: usize_field(&root, "head_size", Self::default().head_size)?,
            n_moba_layer: usize_field(&root, "n_moba_layer", Self::default().n_moba_layer)?,
            moba_chunk_size: usize_field(&root, "moba_chunk_size", Self::default().moba_chunk_size)?,
            moba_topk: usize_field(&root, "moba_topk", Self::default().moba_topk)?,
            dropout: float_field(&root, "dropout", 0.0)?,
            head_size_divisor: usize_field(&root, "head_size_divisor", 8)?,
            ctx_len_hint: usize_field(&root, "ctx_len_hint", 2048)?,
            wkv_chunk_size: usize_field(&root, "wkv_chunk_size", 64)?,
            checkpoint_ffn: bool_field(&root, "checkpoint_ffn", true)?,
            is_moe: bool_field(&root, "is_moe", false)?,
            num_experts: usize_field(&root, "num_experts", 1)?,
            num_experts_per_tok: usize_field(&root, "num_experts_per_tok", 1)?,
        })
    }
}

fn usize_field(root: &Value, name: &str, default: usize) -> Result<usize, String> {
    match root.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("config field '{name}' must be an integer")),
    }
}

fn float_field(root: &Value, name: &str, default: f32) -> Result<f32, String> {
    match root.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| format!("config field '{name}' must be a number")),
    }
}

fn bool_field(root: &Value, name: &str, default: bool) -> Result<bool, String> {
    match root.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("config field '{name}' must be boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let path = std::env::temp_dir().join("smaul-rwkv-config.json");
        let config = RwkvXConfig::default();
        config.save(&path).unwrap();
        let loaded = RwkvXConfig::load(&path).unwrap();
        assert_eq!(loaded.vocab_size, config.vocab_size);
        assert_eq!(loaded.n_embd, config.n_embd);
        assert_eq!(loaded.n_layer, config.n_layer);
        assert_eq!(loaded.n_moba_layer, config.n_moba_layer);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn model_config_preserves_architecture() {
        let config = RwkvXConfig::default();
        let model = config.to_model_config();
        assert_eq!(model.vocab_size, config.vocab_size);
        assert_eq!(model.n_embd, config.n_embd);
        assert_eq!(model.n_layer, config.n_layer);
        assert_eq!(model.n_moba_layer, config.n_moba_layer);
    }
}
