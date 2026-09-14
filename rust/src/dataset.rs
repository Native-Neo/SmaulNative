use crate::tokenizer::Tokenizer;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetPosition { pub file: PathBuf, pub record: usize }

#[derive(Clone, Debug)]
pub struct TokenBatch { pub input: Vec<usize>, pub target: Vec<usize>, pub position: DatasetPosition }

pub struct TextStream {
    reader: BufReader<File>,
    tokenizer: Tokenizer,
    ctx_len: usize,
    record: usize,
    buffer: Vec<usize>,
    path: PathBuf,
    jsonl_text_field: Option<String>,
    eof: bool,
}

impl TextStream {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> { Self::open_jsonl(path, tokenizer, ctx_len, None) }

    pub fn open_jsonl(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize, text_field: Option<impl Into<String>>) -> Result<Self, String> {
        if ctx_len == 0 { return Err("ctx_len must be greater than zero".into()); }
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        Ok(Self { reader: BufReader::new(file), tokenizer, ctx_len, record: 0, buffer: Vec::new(), path, jsonl_text_field: text_field.map(Into::into), eof: false })
    }

    fn read_text(&self, line: &str) -> Result<String, String> {
        if self.jsonl_text_field.is_none() { return Ok(line.trim_end_matches(['\n', '\r']).to_owned()); }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("invalid JSONL record {}: {e}", self.record))?;
        let field = self.jsonl_text_field.as_ref().unwrap();
        value.get(field).and_then(serde_json::Value::as_str).map(str::to_owned).ok_or_else(|| format!("JSONL record {} has no string field '{field}'", self.record))
    }

    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> {
        loop {
            if self.buffer.len() >= self.ctx_len + 1 {
                let input = self.buffer[..self.ctx_len].to_vec();
                let target = self.buffer[1..self.ctx_len + 1].to_vec();
                self.buffer.drain(..self.ctx_len);
                return Ok(Some(TokenBatch { input, target, position: DatasetPosition { file: self.path.clone(), record: self.record } }));
            }
            if self.eof {
                if self.buffer.len() < 2 { return Ok(None); }
                self.buffer.resize(self.ctx_len + 1, self.tokenizer.eos_id());
                continue;
            }
            let mut line = String::new();
            if self.reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 { self.eof = true; continue; }
            let text = self.read_text(&line)?;
            self.buffer.extend(self.tokenizer.encode(&text));
            self.buffer.push(self.tokenizer.eos_id());
            self.record += 1;
        }
    }

    pub fn position(&self) -> DatasetPosition { DatasetPosition { file: self.path.clone(), record: self.record } }
    pub fn buffered_tokens(&self) -> &[usize] { &self.buffer }
}

pub fn discover_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.as_ref()).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && matches!(path.extension().and_then(|x| x.to_str()), Some("txt" | "text" | "jsonl")) { files.push(path); }
    }
    files.sort();
    Ok(files)
}

pub fn load_texts(path: impl AsRef<Path>) -> Result<Vec<String>, String> {
    let file = File::open(path.as_ref()).map_err(|e| e.to_string())?;
    BufReader::new(file).lines().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn tokenizer() -> Tokenizer { Tokenizer::from_vocab(vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "a".into(), "b".into(), " ".into()]) }
    #[test] fn streams_fixed_length_training_pairs() { let path=std::env::temp_dir().join("smaul-dataset.txt"); fs::write(&path,"a b a b a b").unwrap(); let mut stream=TextStream::open(&path,tokenizer(),3).unwrap(); let batch=stream.next_batch().unwrap().unwrap(); assert_eq!(batch.input.len(),3); assert_eq!(batch.target.len(),3); assert_eq!(batch.position.record,1); let _=fs::remove_file(path); }
    #[test] fn stops_at_eof() { let path=std::env::temp_dir().join("smaul-eof.txt"); fs::write(&path,"a b").unwrap(); let mut stream=TextStream::open(&path,tokenizer(),2).unwrap(); assert!(stream.next_batch().unwrap().is_some()); assert!(stream.next_batch().unwrap().is_none()); let _=fs::remove_file(path); }
    #[test] fn reads_jsonl_text_field() { let path=std::env::temp_dir().join("smaul-dataset.jsonl"); fs::write(&path,"{\"text\":\"a b a\"}\n{\"text\":\"b a\"}\n").unwrap(); let mut stream=TextStream::open_jsonl(&path,tokenizer(),2,Some("text")).unwrap(); assert!(stream.next_batch().unwrap().is_some()); let _=fs::remove_file(path); }
}
