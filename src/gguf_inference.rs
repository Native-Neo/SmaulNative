use crate::gguf_model_loader::load_gguf;
use crate::inference::sample;
use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use rand::{SeedableRng, rngs::StdRng};

pub struct GgufInference {
    pub model: RwkvModel,
    pub tokenizer: Tokenizer,
}

impl GgufInference {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let (model, tokenizer) = load_gguf(path)?;
        Ok(Self { model, tokenizer })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Result<String, String> {
        if !temperature.is_finite() || temperature < 0.0 { return Err("temperature must be non-negative".into()); }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) { return Err("top_p must be in (0, 1]".into()); }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 { return Err("repetition_penalty must be positive".into()); }

        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(self.tokenizer.bos_id()); }
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut logits, mut state) = self.model.forward(&tokens, None);
        let mut recent = tokens.iter().copied().rev().take(128).collect::<Vec<_>>();
        recent.reverse();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let row = logits.row(logits.nrows() - 1);
            let token = sample(row.as_slice().ok_or("model logits are not contiguous")?, temperature, top_k, top_p, repetition_penalty, &recent, &mut rng);
            if token == self.tokenizer.eos_id() { break; }
            generated.push(token);
            recent.push(token);
            if recent.len() > 128 { recent.remove(0); }
            (logits, state) = self.model.forward(&[token], Some(&state));
        }
        Ok(self.tokenizer.decode(&generated))
    }

    pub fn chat_prompt(&self, messages: &[(&str, &str)], system: Option<&str>) -> String {
        let mut out = String::new();
        if let Some(system) = system {
            out.push_str("System:\n");
            out.push_str(system);
            out.push_str("\n\n");
        }
        for &(role, content) in messages {
            let role = if role.is_empty() { "user" } else { role };
            let mut chars = role.chars();
            let label = chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default();
            out.push_str(&label);
            out.push_str(":\n");
            out.push_str(content);
            out.push_str("\n\n");
        }
        out.push_str("Assistant:\n");
        out
    }

    pub fn chat_generate(
        &self,
        messages: &[(&str, &str)],
        system: Option<&str>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Result<String, String> {
        let prompt = self.chat_prompt(messages, system);
        self.generate(&prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_is_stable() {
        let tokenizer = Tokenizer::from_vocab(vec![
            "<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(),
            "<cap>".into(), "<upper>".into(), "a".into(),
        ]);
        let model = RwkvModel::new(crate::rwkv_model::RwkvModelConfig::new(7, 8, 1, 4), 1);
        let engine = GgufInference { model, tokenizer };
        assert_eq!(engine.chat_prompt(&[("user", "hello")], Some("sys")), "System:\nsys\n\nUser:\nhello\n\nAssistant:\n");
    }
}
