use crate::tokenizer::Tokenizer;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetPosition { pub file: PathBuf, pub record: usize }
#[derive(Clone, Debug)]
pub struct TokenBatch { pub input: Vec<usize>, pub target: Vec<usize>, pub position: DatasetPosition }

pub struct TextStream {
    reader: BufReader<File>, tokenizer: Tokenizer, ctx_len: usize, record: usize,
    buffer: Vec<usize>, path: PathBuf, jsonl_text_field: Option<String>, eof: bool,
}
impl TextStream {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> { Self::open_jsonl(path, tokenizer, ctx_len, None::<String>) }
    pub fn open_jsonl(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize, text_field: Option<impl Into<String>>) -> Result<Self, String> {
        if ctx_len == 0 { return Err("ctx_len must be greater than zero".into()); }
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        Ok(Self { reader: BufReader::new(file), tokenizer, ctx_len, record: 0, buffer: Vec::new(), path, jsonl_text_field: text_field.map(Into::into), eof: false })
    }
    fn read_text(&self, line: &str) -> Result<String, String> {
        let field = match &self.jsonl_text_field {
            Some(field) => field,
            None => return Ok(line.trim_end_matches(['\n', '\r']).to_owned()),
        };
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("invalid JSONL record {}: {e}", self.record))?;
        value.get(field).and_then(serde_json::Value::as_str).map(str::to_owned).ok_or_else(|| format!("JSONL record {} has no string field '{field}'", self.record))
    }
    fn read_jsonl_record(&self, line: &str) -> Option<String> {
        let value: serde_json::Value = serde_json::from_str(line).ok()?;
        json_value_text(&value)
    }
    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> {
        loop {
            if self.buffer.len() >= self.ctx_len + 1 {
                let input = self.buffer[..self.ctx_len].to_vec();
                let target = self.buffer[1..self.ctx_len + 1].to_vec();
                self.buffer.drain(..self.ctx_len);
                return Ok(Some(TokenBatch { input, target, position: DatasetPosition { file: self.path.clone(), record: self.record } }));
            }
            if self.eof { return Ok(None); }
            let mut line = String::new();
            if self.reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 { self.eof = true; continue; }
            let text = if self.jsonl_text_field.is_some() {
                self.read_text(&line)?
            } else if self.path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("jsonl")) {
                self.read_jsonl_record(&line).unwrap_or_default()
            } else {
                self.read_text(&line)?
            };
            if !text.is_empty() {
                self.buffer.extend(self.tokenizer.encode(&text));
                self.buffer.push(self.tokenizer.eos_id());
            }
            self.record += 1;
        }
    }
    pub fn position(&self) -> DatasetPosition { DatasetPosition { file: self.path.clone(), record: self.record } }
    pub fn buffered_tokens(&self) -> &[usize] { &self.buffer }
}

fn json_value_text(value: &serde_json::Value) -> Option<String> {
    const TEXT_KEYS: &[&str] = &["text", "content", "document", "body", "code", "prompt", "completion"];
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Object(object) => {
            let lower: std::collections::HashMap<String, &serde_json::Value> = object.iter().map(|(k, v)| (k.to_ascii_lowercase(), v)).collect();
            if let (Some(prompt), Some(completion)) = (lower.get("prompt").and_then(|v| v.as_str()), lower.get("completion").and_then(|v| v.as_str())) {
                return Some(format!("{prompt}\n{completion}"));
            }
            for key in TEXT_KEYS {
                if let Some(text) = lower.get(*key).and_then(|v| v.as_str()).filter(|text| !text.trim().is_empty()) {
                    return Some(text.to_owned());
                }
            }
            object.values()
                .filter_map(|v| v.as_str())
                .filter(|text| !text.trim().is_empty() && !looks_numeric(text))
                .max_by_key(|text| text.len())
                .map(str::to_owned)
        }
        serde_json::Value::Array(values) => {
            let texts = values.iter().filter_map(json_value_text).collect::<Vec<_>>();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}

fn looks_numeric(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() { return true; }
    let mut core = text.to_owned();
    for ch in ['.', '-', ':', '/'] { core = core.replacen(ch, "", 1); }
    core.chars().all(|c| c.is_ascii_digit())
}

pub struct ParquetTextStream {
    tokenizer: Tokenizer, ctx_len: usize, path: PathBuf, reader: SerializedFileReader<File>,
    rows: VecDeque<Row>, row_group: usize, row_group_count: usize, record: usize,
    buffer: Vec<usize>, eof: bool, text_field: Option<String>,
}
impl ParquetTextStream {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> { Self::open_with_field(path, tokenizer, ctx_len, None::<String>) }
    pub fn open_with_field(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize, text_field: Option<impl Into<String>>) -> Result<Self, String> {
        if ctx_len == 0 { return Err("ctx_len must be greater than zero".into()); }
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let reader = SerializedFileReader::new(file).map_err(|e| format!("failed to read parquet {}: {e}", path.display()))?;
        let row_group_count = reader.num_row_groups();
        Ok(Self { tokenizer, ctx_len, path, reader, rows: VecDeque::new(), row_group: 0, row_group_count, record: 0, buffer: Vec::new(), eof: false, text_field: text_field.map(Into::into) })
    }
    fn load_next_row_group(&mut self) -> Result<bool, String> {
        if self.row_group >= self.row_group_count { return Ok(false); }
        let group = self.reader.get_row_group(self.row_group).map_err(|e| format!("failed to read parquet row group {} in {}: {e}", self.row_group, self.path.display()))?;
        let rows = group.get_row_iter(None).map_err(|e| format!("failed to decode parquet row group {} in {}: {e}", self.row_group, self.path.display()))?.collect::<Result<Vec<_>, _>>().map_err(|e| format!("failed to decode parquet row group {} in {}: {e}", self.row_group, self.path.display()))?;
        self.row_group += 1;
        self.rows.extend(rows);
        Ok(true)
    }
    fn field_text(&self, row: &Row) -> Option<String> {
        if let Some(field) = &self.text_field { return row.get_column_iter().find_map(|(name, value)| if name == field { field_to_text(value) } else { None }); }
        for name in ["text", "content", "document", "body", "code", "prompt", "completion"] {
            if let Some(value) = row.get_column_iter().find_map(|(column, value)| if column == name { Some(value) } else { None }) {
                if let Some(text) = field_to_text(value) { return Some(text); }
            }
        }
        row.get_column_iter().filter_map(|(_, value)| field_to_text(value)).max_by_key(|text| text.len())
    }
    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> {
        loop {
            if self.buffer.len() >= self.ctx_len + 1 {
                let input = self.buffer[..self.ctx_len].to_vec();
                let target = self.buffer[1..self.ctx_len + 1].to_vec();
                self.buffer.drain(..self.ctx_len);
                return Ok(Some(TokenBatch { input, target, position: DatasetPosition { file: self.path.clone(), record: self.record } }));
            }
            if self.eof { return Ok(None); }
            if self.rows.is_empty() && !self.load_next_row_group()? { self.eof = true; continue; }
            if let Some(row) = self.rows.pop_front() {
                let text = self.field_text(&row).ok_or_else(|| format!("Parquet record {} in {} has no usable text field", self.record, self.path.display()))?;
                self.buffer.extend(self.tokenizer.encode(&text));
                self.buffer.push(self.tokenizer.eos_id());
                self.record += 1;
            }
        }
    }
    pub fn position(&self) -> DatasetPosition { DatasetPosition { file: self.path.clone(), record: self.record } }
}
fn field_to_text(field: &Field) -> Option<String> {
    match field {
        Field::Str(value) => Some(value.clone()),
        Field::Bytes(value) => String::from_utf8(value.data().to_vec()).ok(),
        Field::ListInternal(values) => { let parts = values.elements().iter().filter_map(field_to_text).collect::<Vec<_>>(); if parts.is_empty() { None } else { Some(parts.join(" ")) } }
        _ => None,
    }
}
pub enum DatasetStream { Text(TextStream), Parquet(ParquetTextStream) }
impl DatasetStream { pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> { match self { Self::Text(stream) => stream.next_batch(), Self::Parquet(stream) => stream.next_batch() } } }
pub struct MultiFileTextStream { streams: Vec<TextStream>, current: usize }
impl MultiFileTextStream {
    pub fn open(paths: &[PathBuf], tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> { if paths.is_empty() { return Err("dataset contains no files".into()); } let streams = paths.iter().map(|path| TextStream::open(path, tokenizer.clone(), ctx_len)).collect::<Result<Vec<_>, _>>()?; Ok(Self { streams, current: 0 }) }
    pub fn open_discovered(dir: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> { Self::open(&discover_files(dir)?, tokenizer, ctx_len) }
    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> { while !self.streams.is_empty() { if self.current >= self.streams.len() { self.current = 0; } match self.streams[self.current].next_batch()? { Some(batch) => { self.current = (self.current + 1) % self.streams.len(); return Ok(Some(batch)); } None => { self.streams.remove(self.current); if self.current >= self.streams.len() && !self.streams.is_empty() { self.current = 0; } } } } Ok(None) }
    pub fn file_count(&self) -> usize { self.streams.len() }
}

pub fn discover_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>, String> {
    const PLAIN: &[&str] = &["txt", "text", "py", "cpp", "c", "h", "hpp", "cc", "cxx", "rs", "js", "ts", "tsx", "jsx", "java", "go", "cs", "php", "rb", "swift", "kt", "kts", "scala", "sh", "bash", "zsh", "html", "css", "scss", "sql", "md", "rst", "yaml", "yml", "toml", "xml"];
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir.as_ref()).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let ext = path.extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase);
        if path.is_file() && matches!(ext.as_deref(), Some("jsonl") | Some("parquet")) || path.is_file() && ext.as_deref().is_some_and(|x| PLAIN.contains(&x)) { files.push(path); }
    }
    files.sort();
    Ok(files)
}
pub fn load_texts(path: impl AsRef<Path>) -> Result<Vec<String>, String> { let file = File::open(path.as_ref()).map_err(|e| e.to_string())?; BufReader::new(file).lines().collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string()) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn tokenizer() -> Tokenizer { Tokenizer::from_vocab(vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "a".into(), "b".into(), " ".into()]) }
    #[test] fn streams_fixed_length_training_pairs() { let path=std::env::temp_dir().join("smaul-dataset.txt"); fs::write(&path,"a b a b a b").unwrap(); let mut stream=TextStream::open(&path,tokenizer(),3).unwrap(); let batch=stream.next_batch().unwrap().unwrap(); assert_eq!(batch.input.len(),3); assert_eq!(batch.target.len(),3); assert_eq!(batch.position.record,1); let _=fs::remove_file(path); }
    #[test] fn stops_at_eof() { let path=std::env::temp_dir().join("smaul-eof.txt"); fs::write(&path,"a b").unwrap(); let mut stream=TextStream::open(&path,tokenizer(),2).unwrap(); assert!(stream.next_batch().unwrap().is_some()); assert!(stream.next_batch().unwrap().is_none()); let _=fs::remove_file(path); }
    #[test] fn drops_incomplete_tail() { let path=std::env::temp_dir().join("smaul-tail.txt"); fs::write(&path,"a").unwrap(); let mut stream=TextStream::open(&path,tokenizer(),2).unwrap(); assert!(stream.next_batch().unwrap().is_none()); let _=fs::remove_file(path); }
    #[test] fn reads_jsonl_text_field() { let path=std::env::temp_dir().join("smaul-dataset.jsonl"); fs::write(&path,"{\"text\":\"a b a\"}\n{\"text\":\"b a\"}\n").unwrap(); let mut stream=TextStream::open_jsonl(&path,tokenizer(),2,Some("text")).unwrap(); assert!(stream.next_batch().unwrap().is_some()); let _=fs::remove_file(path); }
    #[test] fn reads_jsonl_default_text_field() { let path=std::env::temp_dir().join("smaul-default.jsonl"); fs::write(&path,"{\"content\":\"a b a\"}\n").unwrap(); let mut stream=TextStream::open_jsonl(&path,tokenizer(),2,None::<String>).unwrap(); let batch=stream.next_batch().unwrap(); assert!(batch.is_some()); let _=fs::remove_file(path); }
    #[test] fn reads_jsonl_prompt_and_completion() { let path=std::env::temp_dir().join("smaul-prompt.jsonl"); fs::write(&path,"{\"prompt\":\"a\",\"completion\":\"b a\"}\n").unwrap(); let mut stream=TextStream::open_jsonl(&path,tokenizer(),2,None::<String>).unwrap(); assert!(stream.next_batch().unwrap().is_some()); let _=fs::remove_file(path); }
    #[test] fn discovers_parquet_files() { let dir=std::env::temp_dir().join("smaul-discover"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("data.parquet"),b"not parquet").unwrap(); assert_eq!(discover_files(&dir).unwrap().len(),1); let _=fs::remove_dir_all(dir); }
    #[test] fn discovers_source_files() { let dir=std::env::temp_dir().join("smaul-source-discover"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("model.rs"),"fn main() {}").unwrap(); fs::write(dir.join("ignore.bin"),b"x").unwrap(); assert_eq!(discover_files(&dir).unwrap(), vec![dir.join("model.rs")]); let _=fs::remove_dir_all(dir); }
    #[test] fn streams_multiple_files() { let dir=std::env::temp_dir().join("smaul-multi"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("a.txt"),"a b a b").unwrap(); fs::write(dir.join("b.txt"),"b a b a").unwrap(); let mut stream=MultiFileTextStream::open_discovered(&dir,tokenizer(),2).unwrap(); assert!(stream.next_batch().unwrap().is_some()); assert!(stream.next_batch().unwrap().is_some()); let _=fs::remove_dir_all(dir); }
}