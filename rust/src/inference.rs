use crate::rwkv_model::{RwkvModel, RwkvModelState};
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
    ids.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    if top_k > 0 && top_k < ids.len() { ids.truncate(top_k); }
    let max = ids.iter().map(|&i| scores[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = ids.into_iter().map(|i| (i, (scores[i] - max).exp())).collect();
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
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

fn argmax(values: &[f32]) -> usize { values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0) }

pub struct Inference<'a> { pub model: &'a RwkvModel, pub tokenizer: &'a Tokenizer, pub eos_id: usize }

impl<'a> Inference<'a> {
    pub fn generate(&self, prompt: &str, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: u64) -> String {
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(self.tokenizer.bos_id()); }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let (mut logits, mut state) = self.model.forward(&tokens, None);
        let mut recent = tokens.iter().copied().rev().take(128).collect::<Vec<_>>();
        recent.reverse();
        let mut generated = Vec::new();
        for _ in 0..max_new_tokens {
            let token = sample(logits.row(logits.nrows() - 1).as_slice().unwrap(), temperature, top_k, top_p, repetition_penalty, &recent, &mut rng);
            if token == self.eos_id { break; }
            generated.push(token);
            recent.push(token);
            if recent.len() > 128 { recent.remove(0); }
            (logits, state) = self.model.forward(&[token], Some(&state));
        }
        self.tokenizer.decode(&generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;

    #[test]
    fn greedy_sampling_selects_maximum() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert_eq!(sample(&[1.0, 4.0, 2.0], 0.0, 0, 1.0, 1.0, &[], &mut rng), 1);
    }

    #[test]
    fn inference_can_generate() {
        let vocab = vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "a".into()];
        let tokenizer = Tokenizer::from_vocab(vocab);
        let model = RwkvModel::new(RwkvModelConfig::new(7, 8, 1, 4), 1);
        let engine = Inference { model: &model, tokenizer: &tokenizer, eos_id: tokenizer.eos_id() };
        let _ = engine.generate("a", 1, 0.0, 0, 1.0, 1.0, 1);
    }
}
