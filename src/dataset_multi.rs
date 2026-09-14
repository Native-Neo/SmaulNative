use crate::dataset::{DatasetStream, TokenBatch};
use crate::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

pub struct MultiFileDatasetStream {
    streams: Vec<DatasetStream>,
    paths: Vec<PathBuf>,
    current: usize,
}
impl MultiFileDatasetStream {
    pub fn open(paths: &[PathBuf], tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        if paths.is_empty() { return Err("dataset contains no files".into()); }
        let mut streams=Vec::with_capacity(paths.len());
        let mut accepted_paths=Vec::with_capacity(paths.len());
        for path in paths { if !matches!(path.extension().and_then(|x|x.to_str()).unwrap_or("").to_ascii_lowercase().as_str(),"txt"|"text"|"py"|"cpp"|"c"|"h"|"hpp"|"cc"|"cxx"|"rs"|"js"|"ts"|"tsx"|"jsx"|"java"|"go"|"cs"|"php"|"rb"|"swift"|"kt"|"kts"|"scala"|"sh"|"bash"|"zsh"|"html"|"css"|"scss"|"sql"|"md"|"rst"|"yaml"|"yml"|"toml"|"xml"|"jsonl"|"json"|"csv"|"parquet"){continue;} streams.push(DatasetStream::open(path,tokenizer.clone(),ctx_len)?);accepted_paths.push(path.clone());}
        if streams.is_empty(){return Err("dataset contains no supported files".into());}
        Ok(Self{streams,paths:accepted_paths,current:0})
    }
    pub fn open_discovered(dir: impl AsRef<Path>, tokenizer: Tokenizer, ctx_len: usize) -> Result<Self, String> {
        fn visit(path:&Path,paths:&mut Vec<PathBuf>)->Result<(),String>{for entry in std::fs::read_dir(path).map_err(|e|format!("failed to read {}: {e}",path.display()))?{let path=entry.map_err(|e|e.to_string())?.path();if path.is_dir(){visit(&path,paths)?;continue;}if !path.is_file(){continue;}let ext=path.extension().and_then(|x|x.to_str()).map(|x|x.to_ascii_lowercase());if matches!(ext.as_deref(),Some("json")|Some("jsonl")|Some("csv")|Some("parquet")|Some("txt")|Some("text")|Some("py")|Some("cpp")|Some("c")|Some("h")|Some("hpp")|Some("cc")|Some("cxx")|Some("rs")|Some("js")|Some("ts")|Some("tsx")|Some("jsx")|Some("java")|Some("go")|Some("cs")|Some("php")|Some("rb")|Some("swift")|Some("kt")|Some("kts")|Some("scala")|Some("sh")|Some("bash")|Some("zsh")|Some("html")|Some("css")|Some("scss")|Some("sql")|Some("md")|Some("rst")|Some("yaml")|Some("yml")|Some("toml")|Some("xml")){paths.push(path);}}Ok(())}
        let mut paths=Vec::new();visit(dir.as_ref(),&mut paths)?;paths.sort();Self::open(&paths,tokenizer,ctx_len)
    }
    pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{while !self.streams.is_empty(){if self.current>=self.streams.len(){self.current=0;}match self.streams[self.current].next_batch()?{Some(batch)=>{self.current=(self.current+1)%self.streams.len();return Ok(Some(batch));}None=>{self.streams.remove(self.current);self.paths.remove(self.current);if self.current>=self.streams.len()&&!self.streams.is_empty(){self.current=0;}}}}Ok(None)}
    pub fn file_count(&self)->usize{self.streams.len()}
    pub fn paths(&self)->&[PathBuf]{&self.paths}
}

#[cfg(test)]
mod tests { use super::*; use std::fs; fn tokenizer()->Tokenizer{Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into(),"b".into()," ".into()])} #[test]fn opens_mixed_formats(){let dir=std::env::temp_dir().join("smaul-mixed-dataset");fs::create_dir_all(&dir).unwrap();fs::write(dir.join("a.txt"),"a b a b").unwrap();fs::write(dir.join("b.jsonl"),"a b a b\n").unwrap();fs::write(dir.join("c.json"),"[{\"text\":\"a b a b\"}]").unwrap();fs::write(dir.join("d.csv"),"text\na b a b\n").unwrap();let stream=MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).unwrap();assert_eq!(stream.file_count(),4);let _=fs::remove_dir_all(dir);} #[test]fn discovers_nested_files(){let dir=std::env::temp_dir().join("smaul-mixed-nested-dataset");let nested=dir.join("nested");fs::create_dir_all(&nested).unwrap();fs::write(nested.join("a.txt"),"a b a b").unwrap();fs::write(nested.join("b.rs"),"a b a b").unwrap();fs::write(nested.join("ignore.bin"),b"x").unwrap();let stream=MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).unwrap();assert_eq!(stream.file_count(),2);assert_eq!(stream.paths(),&[nested.join("a.txt"),nested.join("b.rs")]);let _=fs::remove_dir_all(dir);} #[test]fn rejects_unsupported_files(){let dir=std::env::temp_dir().join("smaul-unsupported-dataset");fs::create_dir_all(&dir).unwrap();fs::write(dir.join("data.bin"),b"x").unwrap();assert!(MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).is_err());let _=fs::remove_dir_all(dir);} }
