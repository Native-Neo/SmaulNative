use crate::tokenizer::Tokenizer;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const IGNORE_INDEX: i32 = -100;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SftRecord {
    pub messages: Vec<SftMessage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SftMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SftExample {
    pub input: Vec<usize>,
    pub labels: Vec<i32>,
}

pub struct SftDataset {
    records: Vec<SftRecord>,
    tokenizer: Tokenizer,
    ctx_len: usize,
    index: usize,
}

impl SftDataset {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if ctx_len == 0 { return Err("ctx_len must be greater than zero".into()); }
        let file = File::open(path.as_ref()).map_err(|e| format!("failed to open {}: {e}", path.as_ref().display()))?;
        let mut records = Vec::new();
        for (line_no, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("failed to read JSONL line {}: {e}", line_no + 1))?;
            if line.trim().is_empty() { continue; }
            records.push(parse_record(&line).map_err(|e| format!("line {}: {e}", line_no + 1))?);
        }
        Ok(Self { records, tokenizer, ctx_len, index: 0 })
    }

    pub fn from_records(records: Vec<SftRecord>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if ctx_len == 0 { return Err("ctx_len must be greater than zero".into()); }
        Ok(Self { records, tokenizer, ctx_len, index: 0 })
    }

    pub fn next_example(&mut self) -> Result<Option<SftExample>, String> {
        while self.index < self.records.len() {
            let record = &self.records[self.index];
            self.index += 1;
            if let Some(example) = preprocess_record(record, &self.tokenizer, self.ctx_len)? {
                return Ok(Some(example));
            }
        }
        Ok(None)
    }

    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    pub fn reset(&mut self) { self.index = 0; }
}

pub fn discover_sft_records(path: impl AsRef<Path>) -> Result<Vec<SftRecord>, String> {
    let file = File::open(path.as_ref()).map_err(|e| format!("failed to open {}: {e}", path.as_ref().display()))?;
    let mut records = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("failed to read JSONL line {}: {e}", line_no + 1))?;
        if line.trim().is_empty() { continue; }
        records.push(parse_record(&line).map_err(|e| format!("line {}: {e}", line_no + 1))?);
    }
    Ok(records)
}

fn parse_record(line: &str) -> Result<SftRecord, String> {
    let root: Value = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let messages = root.get("messages")
        .and_then(Value::as_array)
        .or_else(|| root.get("conversations").and_then(Value::as_array))
        .ok_or("record has no messages/conversations array")?;
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message.get("role")
            .or_else(|| message.get("from"))
            .and_then(Value::as_str)
            .ok_or("message has no string role/from")?;
        let content = message.get("content")
            .or_else(|| message.get("value"))
            .or_else(|| message.get("text"))
            .and_then(Value::as_str)
            .ok_or("message has no string content/value/text")?;
        out.push(SftMessage { role: normalize_role(role), content: content.to_owned() });
    }
    if out.is_empty() { return Err("record has no messages".into()); }
    Ok(SftRecord { messages: out })
}

fn normalize_role(role: &str) -> String {
    match role.to_ascii_lowercase().as_str() {
        "human" | "user" => "user".into(),
        "gpt" | "assistant" | "bot" => "assistant".into(),
        "system" => "system".into(),
        other => other.to_owned(),
    }
}

fn preprocess_record(record: &SftRecord, tokenizer: &Tokenizer, ctx_len: usize) -> Result<Option<SftExample>, String> {
    let mut input = Vec::new();
    let mut labels = Vec::new();
    for message in &record.messages {
        let role_tokens = tokenizer.encode(&format!("{}: ", message.role));
        let content_tokens = tokenizer.encode(&message.content);
        if input.len() + role_tokens.len() + content_tokens.len() + 1 > ctx_len {
            break;
        }
        input.extend_from_slice(&role_tokens);
        labels.extend(std::iter::repeat_n(IGNORE_INDEX, role_tokens.len()));
        let assistant = message.role == "assistant";
        input.extend_from_slice(&content_tokens);
        if assistant {
            labels.extend(content_tokens.iter().map(|&id| id as i32));
        } else {
            labels.extend(std::iter::repeat_n(IGNORE_INDEX, content_tokens.len()));
        }
        input.push(tokenizer.eos_id());
        labels.push(if assistant { tokenizer.eos_id() as i32 } else { IGNORE_INDEX });
    }
    if input.is_empty() || !labels.iter().any(|&label| label != IGNORE_INDEX) { return Ok(None); }
    Ok(Some(SftExample { input, labels }))
}

pub fn discover_sft_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.as_ref()).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().and_then(|x| x.to_str()) == Some("jsonl") { files.push(path); }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_vocab(vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "user".into(), "assistant".into(), ":".into(), " ".into(), "hello".into(), "world".into()])
    }

    #[test]
    fn parses_chat_roles() {
        let record = parse_record(r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}"#).unwrap();
        assert_eq!(record.messages[1].role, "assistant");
    }

    #[test]
    fn labels_only_assistant_content() {
        let record = SftRecord { messages: vec![
            SftMessage { role: "user".into(), content: "hello".into() },
            SftMessage { role: "assistant".into(), content: "world".into() },
        ]};
        let example = preprocess_record(&record, &tokenizer(), 64).unwrap().unwrap();
        assert!(example.labels.iter().any(|&x| x == tokenizer().token_to_id("world").unwrap() as i32));
        assert!(example.labels.iter().any(|&x| x == IGNORE_INDEX));
    }

    #[test]
    fn skips_records_without_assistant_targets() {
        let record = SftRecord { messages: vec![SftMessage { role: "user".into(), content: "hello".into() }] };
        assert!(preprocess_record(&record, &tokenizer(), 64).unwrap().is_none());
    }
}
