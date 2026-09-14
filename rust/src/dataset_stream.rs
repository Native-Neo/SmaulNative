use crate::dataset::{discover_files, DatasetStream, ParquetTextStream, TextStream, TokenBatch};
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

pub struct MultiFileDatasetStream {
    streams: Vec<DatasetStream>,
    current: usize,
}

impl MultiFileDatasetStream {
    pub fn open(paths: &[PathBuf], tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("dataset contains no files".into());
        }
        let mut streams = Vec::with_capacity(paths.len());
        for path in paths {
            let stream = match path.extension().and_then(|x| x.to_str()) {
                Some("parquet") => DatasetStream::Parquet(ParquetTextStream::open(path, tokenizer.clone(), ctx_len)?),
                Some("txt") | Some("text") | Some("jsonl") => DatasetStream::Text(TextStream::open(path, tokenizer.clone(), ctx_len)?),
                _ => continue,
            };
            streams.push(stream);
        }
        if streams.is_empty() {
            return Err("dataset contains no supported text streams".into());
        }
        Ok(Self { streams, current: 0 })
    }

    pub fn open_discovered(dir: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        let paths = discover_files(dir)?;
        Self::open(&paths, tokenizer, ctx_len)
    }

    pub fn next_batch(&mut self) -> Result<Option<TokenBatch>, String> {
        while !self.streams.is_empty() {
            if self.current >= self.streams.len() {
                self.current = 0;
            }
            match self.streams[self.current].next_batch()? {
                Some(batch) => {
                    self.current = (self.current + 1) % self.streams.len();
                    return Ok(Some(batch));
                }
                None => {
                    self.streams.remove(self.current);
                    if self.current >= self.streams.len() && !self.streams.is_empty() {
                        self.current = 0;
                    }
                }
            }
        }
        Ok(None)
    }

    pub fn file_count(&self) -> usize {
        self.streams.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_vocab(vec![
            "<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(),
            "<cap>".into(), "<upper>".into(), "a".into(), "b".into(), " ".into(),
        ])
    }

    #[test]
    fn mixes_text_streams() {
        let dir = std::env::temp_dir().join("smaul-mixed-stream");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "a b a b a b").unwrap();
        fs::write(dir.join("b.jsonl"), "{\"text\":\"b a b a b a\"}\n").unwrap();
        let mut stream = MultiFileDatasetStream::open_discovered(&dir, tokenizer(), 3).unwrap();
        assert_eq!(stream.file_count(), 2);
        assert!(stream.next_batch().unwrap().is_some());
        assert!(stream.next_batch().unwrap().is_some());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_empty_paths() {
        assert!(MultiFileDatasetStream::open(&[], tokenizer(), 3).is_err());
    }
}
