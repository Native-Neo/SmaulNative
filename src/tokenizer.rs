use serde_json::Value;
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
    pub fn from_vocab(vocab: Vec<String>) -> Self { Self::try_from_vocab(vocab).expect("invalid tokenizer vocabulary") }

    pub fn try_from_vocab(vocab: Vec<String>) -> Result<Self, String> {
        if vocab.is_empty() { return Err("tokenizer vocabulary is empty".into()); }
        let mut map = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            if map.insert(token.clone(), id).is_some() { return Err(format!("duplicate tokenizer vocabulary token: {token:?}")); }
        }
        for special in [PAD, UNK, BOS, EOS, CAP, UPPER] {
            if !map.contains_key(special) { return Err(format!("tokenizer is missing required token '{special}'")); }
        }
        let id = |token: &str| map[token];
        Ok(Self { unk_id:id(UNK), pad_id:id(PAD), bos_id:id(BOS), eos_id:id(EOS), cap_id:id(CAP), upper_id:id(UPPER), vocab:map, inverse:vocab })
    }

    pub fn character_vocab(size: usize) -> Result<Self, String> {
        if size < 6 { return Err("character tokenizer requires at least six vocabulary entries".into()); }
        let mut vocab = vec![PAD.into(), UNK.into(), BOS.into(), EOS.into(), CAP.into(), UPPER.into()];
        let mut add_range = |start: u32, end: u32| {
            for code in start..=end {
                if vocab.len() >= size { return; }
                if let Some(ch) = char::from_u32(code) {
                    let token = ch.to_string();
                    if !vocab.contains(&token) { vocab.push(token); }
                }
            }
        };
        add_range(0x20, 0x7e);
        add_range(0xa0, 0xff);
        add_range(0x900, 0x97f);
        add_range(0x370, 0x3ff);
        add_range(0x400, 0x4ff);
        add_range(0x1000, 0x109f);
        add_range(0x3040, 0x30ff);
        add_range(0x4e00, 0x9fff);
        add_range(0x1f300, 0x1faff);
        add_range(0, 0x10ffff);
        if vocab.len() != size { return Err(format!("could not construct character vocabulary of size {size}")); }
        Self::try_from_vocab(vocab)
    }

    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let vocab = text.lines().map(|x| x.trim_end_matches('\r').to_owned()).filter(|x| !x.is_empty()).collect();
        Ok(Self::from_vocab(vocab))
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let root: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if let Some(version) = root.get("version").and_then(Value::as_u64) { if version < 5 { return Err(format!("unsupported SmaulTokenizer version {version}")); } }
        let object = root.get("vocab").and_then(Value::as_object).ok_or("tokenizer JSON has no vocab object")?;
        let mut indexed = Vec::with_capacity(object.len());
        for (token, id) in object { indexed.push((id.as_u64().ok_or("vocabulary id is not an integer")? as usize, token.clone())); }
        indexed.sort_by_key(|x| x.0);
        for (expected, (actual, _)) in indexed.iter().enumerate() { if expected != *actual { return Err("vocabulary ids are not contiguous".into()); } }
        let tokenizer = Self::try_from_vocab(indexed.into_iter().map(|(_, token)| token).collect())?;
        if let Some(id) = root.get("unk_id").and_then(Value::as_u64) { if id as usize != tokenizer.unk_id { return Err("unk_id does not match vocab".into()); } }
        Ok(tokenizer)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let mut vocab = serde_json::Map::new();
        for (id, token) in self.inverse.iter().enumerate() { vocab.insert(token.clone(), Value::from(id)); }
        let root = serde_json::json!({"version": 5,"vocab": vocab,"special_tokens": [PAD, UNK, BOS, EOS],"case_tokens": [CAP, UPPER],"unk_id": self.unk_id,"stats": {"vocab_size": self.vocab_size()}});
        fs::write(path, serde_json::to_string(&root).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
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
            if token.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c)) {
                for unit in devanagari_units(&token) {
                    if let Some(&id) = self.vocab.get(&unit) { out.push(id); }
                    else { for ch in unit.chars() { out.push(self.vocab.get(&ch.to_string()).copied().unwrap_or(self.unk_id)); } }
                }
            } else { for ch in token.chars() { out.push(self.vocab.get(&ch.to_string()).copied().unwrap_or(self.unk_id)); } }
        }
        out
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        let mut capitalize = false;
        let mut uppercase = false;
        for &id in ids {
            let token = match self.id_to_token(id) { Some(token) => token, None => { out.push_str(UNK); continue; } };
            if token == CAP { capitalize = true; uppercase = false; continue; }
            if token == UPPER { uppercase = true; capitalize = false; continue; }
            if token == PAD || token == BOS || token == EOS { continue; }
            if capitalize { let mut chars = token.chars(); if let Some(first) = chars.next() { out.extend(first.to_uppercase()); out.extend(chars); } }
            else if uppercase { out.push_str(&token.to_uppercase()); }
            else { out.push_str(token); }
            capitalize = false;
            uppercase = false;
        }
        out
    }
}

fn is_capitalized(s: &str) -> bool { let mut chars = s.chars(); match chars.next() { Some(first) => first.is_uppercase() && chars.any(char::is_lowercase), None => false } }
fn lexical_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new(); let mut current = String::new();
    let flush = |out: &mut Vec<String>, current: &mut String| { if !current.is_empty() { out.push(std::mem::take(current)); } };
    let multi = ["==", "!=", "<=", ">=", "=>", "->", "::", "//", "**", "&&", "||"];
    let chars: Vec<char> = text.chars().collect(); let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() { flush(&mut out, &mut current); let mut s = String::new(); s.push(ch); i += 1; while i < chars.len() && chars[i].is_whitespace() { s.push(chars[i]); i += 1; } out.push(s); continue; }
        if ch.is_alphanumeric() || ('\u{0900}'..='\u{097F}').contains(&ch) || ch == '\'' { current.push(ch); i += 1; continue; }
        flush(&mut out, &mut current);
        if i + 1 < chars.len() { let pair = format!("{}{}", ch, chars[i + 1]); if multi.contains(&pair.as_str()) { out.push(pair); i += 2; continue; } }
        out.push(ch.to_string()); i += 1;
    }
    flush(&mut out, &mut current); out
}
fn is_devanagari_mark(c: char) -> bool { matches!(c,'\u{0900}'..='\u{0903}'|'\u{093A}'..='\u{093C}'|'\u{093E}'..='\u{094F}'|'\u{0951}'..='\u{0957}'|'\u{0962}'..='\u{0963}') }
fn devanagari_units(text: &str) -> Vec<String> {
    let mut out = Vec::new(); let chars: Vec<char> = text.chars().collect(); let mut i = 0;
    while i < chars.len() { let c = chars[i]; if !('\u{0900}'..='\u{097F}').contains(&c) { out.push(c.to_string()); i += 1; continue; } let mut unit = String::new(); unit.push(c); i += 1; while i < chars.len() { let c = chars[i]; if is_devanagari_mark(c) || c == '\u{200C}' || c == '\u{200D}' { unit.push(c); i += 1; continue; } if c == '\u{094D}' { unit.push(c); i += 1; if i < chars.len() && ('\u{0900}'..='\u{097F}').contains(&chars[i]) { unit.push(chars[i]); i += 1; } continue; } break; } out.push(unit); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Tokenizer { Tokenizer::from_vocab(vec![PAD.into(), UNK.into(), BOS.into(), EOS.into(), CAP.into(), UPPER.into(), "hello".into(), "world".into(), " ".into(), "!".into(), "नमस्ते".into()]) }
    #[test] fn rejects_duplicate_vocab_tokens() { let result=std::panic::catch_unwind(||Tokenizer::from_vocab(vec![PAD.into(), PAD.into()])); assert!(result.is_err()); }
    #[test] fn encodes_known_words_and_case_markers() { assert_eq!(tokenizer().encode("Hello world!"), vec![4, 6, 8, 7, 9]); }
    #[test] fn decodes_case_markers() { assert_eq!(tokenizer().decode(&[4, 6, 8, 7]), "Hello world"); }
    #[test] fn json_round_trip() { let tokenizer=tokenizer(); let path=std::env::temp_dir().join("smaul-tokenizer.json"); tokenizer.save_json(&path).unwrap(); let loaded=Tokenizer::from_json_file(&path).unwrap(); assert_eq!(loaded.vocab_size(),tokenizer.vocab_size()); assert_eq!(loaded.encode("Hello world!"),tokenizer.encode("Hello world!")); let _=std::fs::remove_file(path); }
    #[test] fn character_vocab_has_requested_size_and_core_scripts() { let tokenizer=Tokenizer::character_vocab(1024).unwrap(); assert_eq!(tokenizer.vocab_size(),1024); assert!(tokenizer.token_to_id("a").is_some()); assert!(tokenizer.token_to_id("न").is_some()); }
}