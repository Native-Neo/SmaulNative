use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::tokenizer::{Tokenizer, BOS, CAP, EOS, PAD, UNK, UPPER};

const TEXT_KEYS: &[&str] = &["text", "content", "document", "body", "code", "prompt", "completion"];

pub struct WordTokenizerTrainer {
    counts: HashMap<String, usize>,
    max_records: Option<usize>,
    records: usize,
}

impl WordTokenizerTrainer {
    pub fn new(max_records: Option<usize>) -> Self {
        Self { counts: HashMap::new(), max_records, records: 0 }
    }

    pub fn train_from_path(path: impl AsRef<Path>, vocab_size: usize, max_records: Option<usize>) -> Result<Tokenizer, String> {
        if vocab_size < 8 { return Err("word tokenizer requires a vocabulary size of at least 8".into()); }
        let mut trainer = Self::new(max_records);
        trainer.visit(path.as_ref())?;
        if trainer.records == 0 || trainer.counts.is_empty() { return Err("dataset contains no usable text records".into()); }
        trainer.finish(vocab_size)
    }

    fn visit(&mut self, path: &Path) -> Result<(), String> {
        if self.max_records.is_some_and(|limit| self.records >= limit) { return Ok(()); }
        if path.is_dir() {
            let mut entries = fs::read_dir(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
            entries.sort_by_key(|e| e.path());
            for entry in entries { self.visit(&entry.path())?; if self.max_records.is_some_and(|limit| self.records >= limit) { break; } }
            return Ok(());
        }
        if !path.is_file() { return Ok(()); }
        match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
            "parquet" => self.read_parquet(path),
            "json" => self.read_json(path),
            "jsonl" => self.read_jsonl(path),
            "csv" => self.read_csv(path),
            _ => self.read_text(path),
        }
    }

    fn add_record(&mut self, text: &str) {
        if text.trim().is_empty() || self.max_records.is_some_and(|limit| self.records >= limit) { return; }
        for token in lexical_tokens(text) { *self.counts.entry(token).or_default() += 1; }
        self.records += 1;
    }

    fn read_text(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        for line in BufReader::new(file).lines() { self.add_record(&line.map_err(|e| e.to_string())?); if self.max_records.is_some_and(|limit| self.records >= limit) { break; } }
        Ok(())
    }

    fn read_jsonl(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| e.to_string())?;
            if let Ok(value) = serde_json::from_str::<Value>(&line) { if let Some(text) = json_value_text(&value) { self.add_record(&text); } }
            if self.max_records.is_some_and(|limit| self.records >= limit) { break; }
        }
        Ok(())
    }

    fn read_json(&mut self, path: &Path) -> Result<(), String> {
        let mut text = String::new();
        File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?.read_to_string(&mut text).map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_str(&text).map_err(|e| format!("invalid JSON dataset {}: {e}", path.display()))?;
        match value { Value::Array(values) => for value in values { if let Some(text) = json_value_text(&value) { self.add_record(&text); } if self.max_records.is_some_and(|limit| self.records >= limit) { break; } }, value => if let Some(text) = json_value_text(&value) { self.add_record(&text); } }
        Ok(())
    }

    fn read_csv(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let mut lines = BufReader::new(file).lines();
        let Some(header) = lines.next() else { return Ok(()); };
        let headers = parse_csv_record(&header.map_err(|e| e.to_string())?)?;
        let text_index = headers.iter().position(|h| matches!(h.to_ascii_lowercase().as_str(), "text" | "content" | "document" | "body" | "code" | "prompt"));
        for line in lines {
            let fields = parse_csv_record(&line.map_err(|e| e.to_string())?)?;
            let text = text_index.and_then(|i| fields.get(i).cloned()).or_else(|| fields.iter().filter(|x| !looks_numeric(x)).max_by_key(|x| x.len()).cloned());
            if let Some(text) = text { self.add_record(&text); }
            if self.max_records.is_some_and(|limit| self.records >= limit) { break; }
        }
        Ok(())
    }

    fn read_parquet(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let reader = SerializedFileReader::new(file).map_err(|e| format!("failed to read parquet {}: {e}", path.display()))?;
        for group_index in 0..reader.num_row_groups() {
            let group = reader.get_row_group(group_index).map_err(|e| e.to_string())?;
            let rows = group.get_row_iter(None).map_err(|e| e.to_string())?;
            for row in rows { let row = row.map_err(|e| e.to_string())?; if let Some(text) = row_text(&row) { self.add_record(&text); } if self.max_records.is_some_and(|limit| self.records >= limit) { break; } }
            if self.max_records.is_some_and(|limit| self.records >= limit) { break; }
        }
        Ok(())
    }

    fn finish(self, vocab_size: usize) -> Result<Tokenizer, String> {
        let mut ranked: Vec<(String, usize)> = self.counts.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut vocab = vec![PAD.into(), UNK.into(), BOS.into(), EOS.into(), CAP.into(), UPPER.into()];
        for token in ranked.into_iter().filter(|(token, _)| token != PAD && token != UNK && token != BOS && token != EOS && token != CAP && token != UPPER) {
            if vocab.len() >= vocab_size { break; }
            if !vocab.contains(&token.0) { vocab.push(token.0); }
        }
        while vocab.len() < vocab_size { vocab.push(format!("<unused:{}>", vocab.len())); }
        Tokenizer::try_from_vocab(vocab)
    }
}

fn lexical_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let flush = |out: &mut Vec<String>, current: &mut String| { if !current.is_empty() { out.push(std::mem::take(current)); } };
    let multi = ["==", "!=", "<=", ">=", "=>", "->", "::", "//", "**", "&&", "||"];
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() { flush(&mut out, &mut current); out.push(" ".into()); i += 1; while i < chars.len() && chars[i].is_whitespace() { i += 1; } continue; }
        if ch.is_alphanumeric() || ('\u{0900}'..='\u{097F}').contains(&ch) || ch == '\'' { current.push(ch); i += 1; continue; }
        flush(&mut out, &mut current);
        if i + 1 < chars.len() { let pair = format!("{}{}", ch, chars[i + 1]); if multi.contains(&pair.as_str()) { out.push(pair); i += 2; continue; } }
        out.push(ch.to_string()); i += 1;
    }
    flush(&mut out, &mut current);
    out
}

fn json_value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(object) => {
            let lower: HashMap<String, &Value> = object.iter().map(|(k, v)| (k.to_ascii_lowercase(), v)).collect();
            if let (Some(prompt), Some(completion)) = (lower.get("prompt").and_then(|v| v.as_str()), lower.get("completion").and_then(|v| v.as_str())) { return Some(format!("{prompt}\n{completion}")); }
            for key in TEXT_KEYS { if let Some(text) = lower.get(*key).and_then(|v| v.as_str()).filter(|x| !x.trim().is_empty()) { return Some(text.to_owned()); } }
            object.values().filter_map(|v| v.as_str()).filter(|x| !x.trim().is_empty()).max_by_key(|x| x.len()).map(str::to_owned)
        }
        Value::Array(values) => { let texts = values.iter().filter_map(json_value_text).collect::<Vec<_>>(); (!texts.is_empty()).then(|| texts.join("\n")) }
        _ => None,
    }
}

fn row_text(row: &Row) -> Option<String> {
    for key in TEXT_KEYS { if let Some(value) = row.get_column_iter().find_map(|(name, value)| if name.eq_ignore_ascii_case(key) { Some(value) } else { None }) { if let Some(text) = field_to_text(value) { return Some(text); } } }
    row.get_column_iter().filter_map(|(_, value)| field_to_text(value)).max_by_key(|x| x.len())
}

fn field_to_text(field: &Field) -> Option<String> {
    match field { Field::Str(value) => Some(value.clone()), Field::Bytes(value) => String::from_utf8(value.data().to_vec()).ok(), Field::ListInternal(values) => { let parts = values.elements().iter().filter_map(field_to_text).collect::<Vec<_>>(); (!parts.is_empty()).then(|| parts.join(" ")) }, _ => None }
}

fn parse_csv_record(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new(); let mut current = String::new(); let mut chars = line.chars().peekable(); let mut quoted = false;
    while let Some(ch) = chars.next() { if quoted { if ch == '"' { if chars.peek() == Some(&'"') { current.push('"'); chars.next(); } else { quoted = false; } } else { current.push(ch); } } else if ch == '"' { quoted = true; } else if ch == ',' { fields.push(std::mem::take(&mut current)); } else { current.push(ch); } }
    if quoted { return Err("unterminated CSV quote".into()); }
    fields.push(current); Ok(fields)
}

fn looks_numeric(text: &str) -> bool { let text = text.trim(); !text.is_empty() && text.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | ':' | '/')) }
