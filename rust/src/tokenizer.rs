use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const PAD: &str = "<pad>";
pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const CAP: &str = "<cap>";
pub const UPPER: &str = "<upper>";

#[derive(Clone, Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    inverse: Vec<String>,
    unk_id: usize,
    pad_id: usize,
    bos_id: usize,
    eos_id: usize,
    cap_id: usize,
    upper_id: usize,
}

impl Tokenizer {
    pub fn from_vocab(vocab: Vec<String>) -> Self {
        assert!(!vocab.is_empty());
        let mut map = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() { map.insert(token.clone(), id); }
        let id = |name: &str| *map.get(name).unwrap_or(&0);
        Self { unk_id: id(UNK), pad_id: id(PAD), bos_id: id(BOS), eos_id: id(EOS), cap_id: id(CAP), upper_id: id(UPPER), vocab: map, inverse: vocab }
    }

    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut vocab = Vec::new();
        for line in text.lines() {
            let token = line.trim_end_matches('\r');
            if !token.is_empty() { vocab.push(token.to_owned()); }
        }
        if vocab.is_empty() { return Err("vocabulary is empty".into()); }
        Ok(Self::from_vocab(vocab))
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let object = root.get("vocab").and_then(serde_json::Value::as_object).ok_or("tokenizer JSON has no vocab object")?;
        let mut indexed = Vec::with_capacity(object.len());
        for (token, id) in object {
            let id = id.as_u64().ok_or("vocabulary id is not an integer")? as usize;
            if id >= object.len() { return Err("vocabulary ids are not dense".into()); }
            indexed.push((id, token.clone()));
        }
        indexed.sort_by_key(|(id, _)| *id);
        if indexed.iter().enumerate().any(|(expected, (actual, _))| expected != *actual) { return Err("vocabulary ids are not contiguous".into()); }
        Ok(Self::from_vocab(indexed.into_iter().map(|(_, token)| token).collect()))
    }

    pub fn vocab_size(&self) -> usize { self.inverse.len() }
    pub fn token_to_id(&self, token: &str) -> Option<usize> { self.vocab.get(token).copied() }
    pub fn id_to_token(&self, id: usize) -> Option<&str> { self.inverse.get(id).map(String::as_str) }
    pub fn unk_id(&self) -> usize { self.unk_id }
    pub fn pad_id(&self) -> usize { self.pad_id }
    pub fn bos_id(&self) -> usize { self.bos_id }
    pub fn eos_id(&self) -> usize { self.eos_id }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for token in lexical_tokens(text) {
            if token.chars().all(char::is_whitespace) {
                for ch in token.chars() { out.push(self.vocab.get(&ch.to_string()).copied().unwrap_or(self.unk_id)); }
                continue;
            }
            let canonical = token.to_lowercase();
            if let Some(&id) = self.vocab.get(&canonical) {
                if is_capitalized(&token) { out.push(self.cap_id); }
                else if token.chars().all(|c| !c.is_lowercase() && c.is_alphabetic()) { out.push(self.upper_id); }
                out.push(id);
                continue;
            }
            for unit in devanagari_units(&token) {
                if let Some(&id) = self.vocab.get(&unit) { out.push(id); }
                else { for ch in unit.chars() { out.push(self.vocab.get(&ch.to_string()).copied().unwrap_or(self.unk_id)); } }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        let mut case_mode = None;
        for &id in ids {
            let Some(token) = self.id_to_token(id) else { out.push_str(UNK); continue };
            match token {
                CAP => { case_mode = Some(true); continue; }
                UPPER => { case_mode = Some(false); continue; }
                PAD | BOS | EOS => continue,
                _ => {}
            }
            match case_mode.take() {
                Some(true) => { let mut chars = token.chars(); if let Some(first) = chars.next() { out.extend(first.to_uppercase()); out.extend(chars); } }
                Some(false) => out.push_str(&token.to_uppercase()),
                None => out.push_str(token),
            }
        }
        out
    }
}

fn is_capitalized(s: &str) -> bool {
    let mut chars = s.chars();
    match (chars.next(), chars.as_str()) {
        (Some(first), rest) => first.is_uppercase() && rest.chars().any(char::is_lowercase),
        None => false,
    }
}

fn lexical_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let flush = |out: &mut Vec<String>, current: &mut String| { if !current.is_empty() { out.push(std::mem::take(current)); } };
    for ch in text.chars() {
        if ch.is_whitespace() { flush(&mut out, &mut current); current.push(ch); continue; }
        if ch.is_alphanumeric() || ('\u{0900}'..='\u{097F}').contains(&ch) || ch == '\'' { current.push(ch); }
        else { flush(&mut out, &mut current); out.push(ch.to_string()); }
    }
    flush(&mut out, &mut current);
    out
}

fn is_devanagari_mark(c: char) -> bool {
    matches!(c, '\u{0900}'..='\u{0903}' | '\u{093A}'..='\u{093C}' | '\u{093E}'..='\u{094F}' | '\u{0951}'..='\u{0957}' | '\u{0962}'..='\u{0963}')
}

fn devanagari_units(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !('\u{0900}'..='\u{097F}').contains(&c) { out.push(c.to_string()); i += 1; continue; }
        let mut unit = String::new(); unit.push(c); i += 1;
        while i < chars.len() {
            let c = chars[i];
            if is_devanagari_mark(c) || c == '\u{200C}' || c == '\u{200D}' { unit.push(c); i += 1; continue; }
            if c == '\u{094D}' { unit.push(c); i += 1; if i < chars.len() && ('\u{0900}'..='\u{097F}').contains(&chars[i]) { unit.push(chars[i]); i += 1; } continue; }
            break;
        }
        out.push(unit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer { Tokenizer::from_vocab(vec![PAD.into(), UNK.into(), BOS.into(), EOS.into(), CAP.into(), UPPER.into(), "hello".into(), "world".into(), " ".into(), "!".into(), "नमस्ते".into()]) }

    #[test]
    fn encodes_known_words_and_case_markers() { assert_eq!(tokenizer().encode("Hello world!"), vec![4, 6, 8, 7, 9]); }

    #[test]
    fn decodes_case_markers() { assert_eq!(tokenizer().decode(&[4, 6, 8, 7]), "Hello world"); }

    #[test]
    fn unknown_text_is_represented() { assert!(!tokenizer().encode("xyz").is_empty()); }
}
