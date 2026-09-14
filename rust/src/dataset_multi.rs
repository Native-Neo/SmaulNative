use crate::dataset::{DatasetStream, ParquetTextStream, TextStream, TokenBatch};
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

const PLAIN_SUFFIXES: &[&str] = &["txt", "text", "py", "cpp", "c", "h", "hpp", "cc", "cxx", "rs", "js", "ts", "tsx", "jsx", "java", "go", "cs", "php", "rb", "swift", "kt", "kts", "scala", "sh", "bash", "zsh", "html", "css", "scss", "sql", "md", "rst", "yaml", "yml", "toml", "xml"];

pub struct MultiFileDatasetStream {
    streams: Vec<DatasetStream>,
    paths: Vec<PathBuf>,
    current: usize,
}

impl MultiFileDatasetStream {
    pub fn open(paths: &[PathBuf], tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if paths.is_empty() { return Err("dataset contains no files".into()); }
        let mut streams = Vec::with_capacity(paths.len());
        let mut accepted_paths = Vec::with_capacity(paths.len());
        for path in paths {
            let extension = path.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase();
            let stream = match extension.as_str() {
                "parquet" => DatasetStream::Parquet(ParquetTextStream::open(path, tokenizer.clone(), ctx_len)?),
                "jsonl" => DatasetStream::Text(TextStream::open(path, tokenizer.clone(), ctx_len)?),
                ext if PLAIN_SUFFIXES.contains(&ext) => DatasetStream::Text(TextStream::open(path, tokenizer.clone(), ctx_len)?),
                _ => continue,
            };
            streams.push(stream);
            accepted_paths.push(path.clone());
        }
        if streams.is_empty() { return Err("dataset contains no supported files".into()); }
        Ok(Self { streams, paths: accepted_paths, current: 0 })
    }

    pub fn open_discovered(dir: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(dir.as_ref()).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let extension = path.extension().and_then(|x| x.to_str()).map(|x| x.to_ascii_lowercase());
            let supported = extension.as_deref() == Some("jsonl") || extension.as_deref() == Some("parquet") || extension.as_deref().is_some_and(|x| PLAIN_SUFFIXES.contains(&x));
            if path.is_file() && supported { paths.push(path); }
        }
        paths.sort();
        Self::open(&paths, tokenizer, ctx_len)
    }

    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> {
        while !self.streams.is_empty() {
            if self.current >= self.streams.len() { self.current = 0; }
            match self.streams[self.current].next_batch()? {
                Some(batch) => { self.current = (self.current + 1) % self.streams.len(); return Ok(Some(batch)); }
                None => { self.streams.remove(self.current); self.paths.remove(self.current); if self.current >= self.streams.len() && !self.streams.is_empty() { self.current = 0; } }
            }
        }
        Ok(None)
    }

    pub fn file_count(&self) -> usize { self.streams.len() }
    pub fn paths(&self) -> &[PathBuf] { &self.paths }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tokenizer() -> Tokenizer { Tokenizer::from_vocab(vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "a".into(), "b".into(), " ".into()]) }

    #[test]
    fn opens_mixed_text_extensions() {
        let dir = std::env::temp_dir().join("smaul-mixed-dataset");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "a b a b").unwrap();
        fs::write(dir.join("b.jsonl"), "a b a b\n").unwrap();
        fs::write(dir.join("c.rs"), "a b a b").unwrap();
        let stream = MultiFileDatasetStream::open_discovered(&dir, tokenizer(), 2).unwrap();
        assert_eq!(stream.file_count(), 3);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_unsupported_files() {
        let dir = std::env::temp_dir().join("smaul-unsupported-dataset");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("data.bin"), b"x").unwrap();
        assert!(MultiFileDatasetStream::open_discovered(&dir, tokenizer(), 2).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
