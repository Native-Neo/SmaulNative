use crate::tokenizer::Tokenizer;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const IGNORE_INDEX: i32 = -100;
const DEFAULT_STOP_TOKEN: &str = "\n\n";

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
    match role.trim().to_ascii_lowercase().as_str() {
        "human" | "user" => "User".into(),
        "gpt" | "assistant" | "bot" => "Assistant".into(),
        _ => "Other".into(),
    }
}

fn preprocess_record(record: &SftRecord, tokenizer: &Tokenizer, ctx_len: usize) -> Result<Option<SftExample>, String> {
    let mut input = Vec::new();
    let mut labels = Vec::new();

    for message in &record.messages {
        let role_prefix = format!("{}: ", message.role);
        let formatted = format!("{}{}{}", role_prefix, message.content, DEFAULT_STOP_TOKEN);
        let ids = tokenizer.encode(&formatted);
        let prefix_len = tokenizer.encode(&role_prefix).len();

        if input.len() + ids.len() > ctx_len { break; }

        input.extend_from_slice(&ids);
        if message.role == "Assistant" {
            labels.extend(ids.iter().enumerate().map(|(i, &id)| {
                if i < prefix_len { IGNORE_INDEX } else { id as i32 }
            }));
        } else {
            labels.extend(std::iter::repeat_n(IGNORE_INDEX, ids.len()));
        }
    }

    if input.is_empty() || !labels.iter().any(|&label| label != IGNORE_INDEX) {
        return Ok(None);
    }

    let pad_id = tokenizer.token_to_id("<pad>")
        .ok_or("tokenizer is missing <pad> token")?;
    let pad_len = ctx_len - input.len();
    if pad_len > 0 {
        input.extend(std::iter::repeat_n(pad_id, pad_len));
        labels.extend(std::iter::repeat_n(IGNORE_INDEX, pad_len));
    }

    Ok(Some(SftExample { input, labels }))
}

pub fn discover_sft_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.as_ref()).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("jsonl")).unwrap_or(false) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_vocab(vec![
            "<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(),
            "<cap>".into(), "<upper>".into(), "User".into(), "Assistant".into(),
            ":".into(), " ".into(), "hello".into(), "world".into(), "\n".into(),
        ])
    }

    #[test]
    fn parses_chat_roles() {
        let record = parse_record(r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}"#).unwrap();
        assert_eq!(record.messages[0].role, "User");
        assert_eq!(record.messages[1].role, "Assistant");
    }

    #[test]
    fn labels_only_assistant_content() {
        let tok = tokenizer();
        let record = SftRecord { messages: vec![
            SftMessage { role: "User".into(), content: "hello".into() },
            SftMessage { role: "Assistant".into(), content: "world".into() },
        ]};
        let example = preprocess_record(&record, &tok, 64).unwrap().unwrap();
        let world = tok.token_to_id("world").unwrap() as i32;
        assert!(example.labels.iter().any(|&x| x == world));
        assert!(example.labels.iter().any(|&x| x == IGNORE_INDEX));
    }

    #[test]
    fn masks_assistant_prefix() {
        let tok = tokenizer();
        let record = SftRecord { messages: vec![SftMessage { role: "Assistant".into(), content: "world".into() }] };
        let example = preprocess_record(&record, &tok, 64).unwrap().unwrap();
        let prefix_len = tok.encode("Assistant: ").len();
        assert!(example.labels[..prefix_len].iter().all(|&x| x == IGNORE_INDEX));
        assert_eq!(example.labels[prefix_len], tok.token_to_id("world").unwrap() as i32);
    }

    #[test]
    fn pads_to_context_length() {
        let tok = tokenizer();
        let record = SftRecord { messages: vec![SftMessage { role: "Assistant".into(), content: "world".into() }] };
        let example = preprocess_record(&record, &tok, 64).unwrap().unwrap();
        assert_eq!(example.input.len(), 64);
        assert_eq!(example.labels.len(), 64);
        assert_eq!(*example.labels.last().unwrap(), IGNORE_INDEX);
    }

    #[test]
    fn skips_records_without_assistant_targets() {
        let record = SftRecord { messages: vec![SftMessage { role: "User".into(), content: "hello".into() }] };
        assert!(preprocess_record(&record, &tokenizer(), 64).unwrap().is_none());
    }
}
