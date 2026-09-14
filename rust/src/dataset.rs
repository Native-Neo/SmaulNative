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

pub struct TextStream { path: PathBuf, tokenizer: Tokenizer, ctx_len: usize, buffer: Vec<usize>, reader: BufReader<File>, record: usize, eof: bool }
impl TextStream {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if ctx_len == 0 { return Err("context length must be positive".into()); }
        let path = path.as_ref().to_path_buf();
        Ok(Self { reader: BufReader::new(File::open(&path).map_err(|e| e.to_string())?), path, tokenizer, ctx_len, buffer: Vec::new(), record: 0, eof: false })
    }
    pub fn open_jsonl(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize, text_field: Option<&str>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let mut stream = Self { reader: BufReader::new(file), path, tokenizer, ctx_len, buffer: Vec::new(), record: 0, eof: false };
        if let Some(field) = text_field { stream.buffer = vec![stream.tokenizer.bos_id()]; let _ = field; stream.buffer.clear(); }
        Ok(stream)
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
            if line.trim().is_empty() { continue; }
            self.buffer.extend(self.tokenizer.encode(line.trim_end()));
            self.buffer.push(self.tokenizer.eos_id());
            self.record += 1;
        }
    }
}

pub struct ParquetTextStream {
    path: PathBuf,
    tokenizer: Tokenizer,
    ctx_len: usize,
    buffer: Vec<usize>,
    reader: SerializedFileReader<File>,
    rows: VecDeque<Row>,
    row_group: usize,
    row_group_count: usize,
    record: usize,
    eof: bool,
    text_field: Option<String>,
}
impl ParquetTextStream {
    pub fn open(path: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize, text_field: Option<&str>) -> Result<Self, String> {
        if ctx_len == 0 { return Err("context length must be positive".into()); }
        let path = path.as_ref().to_path_buf();
        let reader = SerializedFileReader::new(File::open(&path).map_err(|e| e.to_string())?).map_err(|e| format!("failed to open parquet {}: {e}", path.display()))?;
        let row_group_count = reader.num_row_groups();
        Ok(Self { path, tokenizer, ctx_len, buffer: Vec::new(), reader, rows: VecDeque::new(), row_group: 0, row_group_count, record: 0, eof: false, text_field: text_field.map(str::to_owned) })
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
        if path.is_file() && (matches!(ext.as_deref(), Some("jsonl") | Some("parquet")) || ext.as_deref().is_some_and(|x| PLAIN.contains(&x))) { files.push(path); }
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
    #[test] fn discovers_parquet_files() { let dir=std::env::temp_dir().join("smaul-discover"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("data.parquet"),b"not parquet").unwrap(); assert_eq!(discover_files(&dir).unwrap().len(),1); let _=fs::remove_dir_all(dir); }
    #[test] fn discovers_source_files() { let dir=std::env::temp_dir().join("smaul-source-discover"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("model.rs"),"fn main() {}").unwrap(); fs::write(dir.join("ignore.bin"),b"x").unwrap(); assert_eq!(discover_files(&dir).unwrap(), vec![dir.join("model.rs")]); let _=fs::remove_dir_all(dir); }
    #[test] fn streams_multiple_files() { let dir=std::env::temp_dir().join("smaul-multi"); fs::create_dir_all(&dir).unwrap(); fs::write(dir.join("a.txt"),"a b a b").unwrap(); fs::write(dir.join("b.txt"),"b a b a").unwrap(); let mut stream=MultiFileTextStream::open_discovered(&dir,tokenizer(),2).unwrap(); assert!(stream.next_batch().unwrap().is_some()); assert!(stream.next_batch().unwrap().is_some()); let _=fs::remove_dir_all(dir); }
}
