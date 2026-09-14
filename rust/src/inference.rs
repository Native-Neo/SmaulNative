use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use rand::{Rng, SeedableRng};

pub fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, recent: &[usize], rng: &mut impl Rng) -> usize {
    assert!(temperature >= 0.0 && top_p > 0.0 && top_p <= 1.0 && repetition_penalty > 0.0);
    let mut scores = logits.to_vec();
    if repetition_penalty != 1.0 {
        for &id in recent { if id < scores.len() { scores[id] = if scores[id] > 0.0 { scores[id] / repetition_penalty } else { scores[id] * repetition_penalty }; } }
    }
    if temperature == 0.0 { return argmax(&scores); }
    for x in &mut scores { *x /= temperature; }
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < ids.len() { ids.truncate(top_k); }
    let max = ids.iter().map(|&i| scores[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = ids.into_iter().map(|i| (i, (scores[i] - max).exp())).collect();
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
    if total == 0.0 || !total.is_finite() { return argmax(&scores); }
    for (_, p) in &mut probs { *p /= total; }
    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 1;
        for i in 0..probs.len() { cumulative += probs[i].1; if cumulative > top_p { keep = i + 1; break; } }
        probs.truncate(keep.max(1));
        let total: f32 = probs.iter().map(|(_, p)| *p).sum();
        for (_, p) in &mut probs { *p /= total; }
    }
    let mut r = rng.random::<f32>();
    for (id, p) in probs { if r <= p { return id; } r -= p; }
    argmax(logits)
}

fn argmax(values: &[f32]) -> usize { values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0) }

pub struct Inference<'a> { pub model: &'a RwkvModel, pub tokenizer: &'a Tokenizer, pub eos_id: usize }

impl<'a> Inference<'a> {
    fn validate(max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32) -> Result<(), String> {
        if !temperature.is_finite() || temperature < 0.0 { return Err("temperature must be non-negative".into()); }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) { return Err("top_p must be in (0, 1]".into()); }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 { return Err("repetition_penalty must be positive".into()); }
        let _ = (max_new_tokens, top_k);
        Ok(())
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: u64) -> String {
        self.stream(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, &[], seed).collect()
    }

    pub fn stream<'b>(&'b self, prompt: &str, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, stop: &'b [&'b str], seed: u64) -> impl Iterator<Item = String> + 'b {
        Self::validate(max_new_tokens, temperature, top_k, top_p, repetition_penalty).expect("invalid generation arguments");
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(self.tokenizer.bos_id()); }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (mut logits, mut state) = self.model.forward(&tokens, None);
        let mut recent = tokens.iter().copied().rev().take(128).collect::<Vec<_>>();
        recent.reverse();
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut chunks = Vec::new();
        for _ in 0..max_new_tokens {
            let token = sample(logits.row(logits.nrows() - 1).as_slice().unwrap(), temperature, top_k, top_p, repetition_penalty, &recent, &mut rng);
            if token == self.eos_id { break; }
            generated.push(token);
            recent.push(token);
            if recent.len() > 128 { recent.remove(0); }
            let current = self.tokenizer.decode(&generated);
            let mut end = current.len();
            for &marker in stop {
                if !marker.is_empty() { if let Some(pos) = current[emitted.len()..].find(marker) { end = end.min(emitted.len() + pos); } }
            }
            if end > emitted.len() { chunks.push(current[emitted.len()..end].to_owned()); }
            emitted = current[..end].to_owned();
            if end < current.len() { break; }
            (logits, state) = self.model.forward(&[token], Some(&state));
        }
        chunks.into_iter()
    }

    pub fn chat_prompt(&self, messages: &[(&str, &str)], system: Option<&str>) -> String {
        let mut out = String::new();
        if let Some(system) = system { out.push_str("System:\n"); out.push_str(system); out.push_str("\n\n"); }
        for &(role, content) in messages { let role = if role.is_empty() { "user" } else { role }; let mut chars = role.chars(); let label = chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default(); out.push_str(&label); out.push_str(":\n"); out.push_str(content); out.push_str("\n\n"); }
        out.push_str("Assistant:\n");
        out
    }

    pub fn chat_generate(&self, messages: &[(&str, &str)], system: Option<&str>, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: u64) -> String {
        self.generate(&self.chat_prompt(messages, system), max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;
    #[test] fn greedy_sampling_selects_maximum() { let mut rng=rand::rngs::StdRng::seed_from_u64(1); assert_eq!(sample(&[1.0,4.0,2.0],0.0,0,1.0,1.0,&[],&mut rng),1); }
    #[test] fn inference_can_generate() { let tokenizer=Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into()]); let model=RwkvModel::new(RwkvModelConfig::new(7,8,1,4),1); let engine=Inference{model:&model,tokenizer:&tokenizer,eos_id:tokenizer.eos_id()}; let _=engine.generate("a",1,0.0,0,1.0,1.0,1); }
    #[test] fn chat_prompt_matches_format() { let tokenizer=Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into()]); let model=RwkvModel::new(RwkvModelConfig::new(7,8,1,4),1); let engine=Inference{model:&model,tokenizer:&tokenizer,eos_id:3}; assert_eq!(engine.chat_prompt(&[("user","hello")],Some("sys")),"System:\nsys\n\nUser:\nhello\n\nAssistant:\n"); }
}
