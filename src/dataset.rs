use crate::tokenizer::Tokenizer;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetPosition { pub file: PathBuf, pub record: usize }
#[derive(Clone, Debug)]
pub struct TokenBatch { pub input: Vec<usize>, pub target: Vec<usize>, pub position: DatasetPosition }
fn append_record(buffer:&mut Vec<usize>,tokenizer:&Tokenizer,text:&str){if !text.is_empty(){buffer.extend(tokenizer.encode(text));buffer.push(tokenizer.eos_id());}}

pub struct TextStream{reader:BufReader<File>,tokenizer:Tokenizer,ctx_len:usize,record:usize,buffer:Vec<usize>,path:PathBuf,jsonl_text_field:Option<String>,eof:bool}
impl TextStream{
 pub fn open(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{Self::open_jsonl(path,tokenizer,ctx_len,None::<String>)}
 pub fn open_jsonl(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize,text_field:Option<impl Into<String>>)->Result<Self,String>{if ctx_len==0{return Err("ctx_len must be greater than zero".into());}let path=path.as_ref().to_path_buf();let file=File::open(&path).map_err(|e|format!("failed to open {}: {e}",path.display()))?;Ok(Self{reader:BufReader::new(file),tokenizer,ctx_len,record:0,buffer:Vec::new(),path,jsonl_text_field:text_field.map(Into::into),eof:false})}
 fn read_text(&self,line:&str)->Result<String,String>{match &self.jsonl_text_field{Some(field)=>{let value:serde_json::Value=serde_json::from_str(line).map_err(|e|format!("invalid JSONL record {}: {e}",self.record))?;value.get(field).and_then(serde_json::Value::as_str).map(str::to_owned).ok_or_else(||format!("JSONL record {} has no string field '{field}'",self.record))},None=>Ok(line.trim_end_matches(['\n','\r']).to_owned())}}
 fn read_jsonl_record(&self,line:&str)->Result<Option<String>,String>{let value:serde_json::Value=serde_json::from_str(line).map_err(|e|format!("invalid JSONL record {}: {e}",self.record))?;Ok(json_value_text(&value))}
 pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{loop{if self.buffer.len()>=self.ctx_len+1{let input=self.buffer[..self.ctx_len].to_vec();let target=self.buffer[1..self.ctx_len+1].to_vec();self.buffer.drain(..self.ctx_len);return Ok(Some(TokenBatch{input,target,position:DatasetPosition{file:self.path.clone(),record:self.record}}));}if self.eof{return Ok(None);}let mut line=String::new();if self.reader.read_line(&mut line).map_err(|e|e.to_string())?==0{self.eof=true;continue;}let text=if self.jsonl_text_field.is_some(){self.read_text(&line)?}else if self.path.extension().and_then(|e|e.to_str()).is_some_and(|e|e.eq_ignore_ascii_case("jsonl")){self.read_jsonl_record(&line)?.unwrap_or_default()}else{self.read_text(&line)?};append_record(&mut self.buffer,&self.tokenizer,&text);self.record+=1;}}
 pub fn position(&self)->DatasetPosition{DatasetPosition{file:self.path.clone(),record:self.record}}
 pub fn buffered_tokens(&self)->&[usize]{&self.buffer}
}
fn json_value_text(value:&serde_json::Value)->Option<String>{const TEXT_KEYS:&[&str]=&["text","content","document","body","code","prompt","completion"];match value{serde_json::Value::String(text)=>Some(text.clone()),serde_json::Value::Object(object)=>{let lower:std::collections::HashMap<String,&serde_json::Value>=object.iter().map(|(k,v)|(k.to_ascii_lowercase(),v)).collect();if let(Some(prompt),Some(completion))=(lower.get("prompt").and_then(|v|v.as_str()),lower.get("completion").and_then(|v|v.as_str())){return Some(format!("{prompt}\n{completion}"));}for key in TEXT_KEYS{if let Some(text)=lower.get(*key).and_then(|v|v.as_str()).filter(|text|!text.trim().is_empty()){return Some(text.to_owned());}}object.values().filter_map(|v|v.as_str()).filter(|text|!text.trim().is_empty()&&!looks_numeric(text)).max_by_key(|text|text.len()).map(str::to_owned)},serde_json::Value::Array(values)=>{let texts=values.iter().filter_map(json_value_text).collect::<Vec<_>>();(!texts.is_empty()).then(||texts.join("\n"))},_=>None}}
fn looks_numeric(text:&str)->bool{let text=text.trim();if text.is_empty(){return true;}let mut core=text.to_owned();for ch in ['.','-',':','/']{core=core.replacen(ch,"",1);}core.chars().all(|c|c.is_ascii_digit())}

pub struct JsonTextStream{tokenizer:Tokenizer,ctx_len:usize,path:PathBuf,records:Vec<String>,record:usize,buffer:Vec<usize>,eof:bool}
impl JsonTextStream{
 pub fn open(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{if ctx_len==0{return Err("ctx_len must be greater than zero".into());}let path=path.as_ref().to_path_buf();let mut text=String::new();File::open(&path).map_err(|e|format!("failed to open {}: {e}",path.display()))?.read_to_string(&mut text).map_err(|e|e.to_string())?;let root:serde_json::Value=serde_json::from_str(&text).map_err(|e|format!("invalid JSON dataset {}: {e}",path.display()))?;let records=match root{serde_json::Value::Array(values)=>values.iter().filter_map(json_value_text).collect(),value=>json_value_text(&value).into_iter().collect()};Ok(Self{tokenizer,ctx_len,path,records,record:0,buffer:Vec::new(),eof:false})}
 pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{loop{if self.buffer.len()>=self.ctx_len+1{let input=self.buffer[..self.ctx_len].to_vec();let target=self.buffer[1..self.ctx_len+1].to_vec();self.buffer.drain(..self.ctx_len);return Ok(Some(TokenBatch{input,target,position:DatasetPosition{file:self.path.clone(),record:self.record}}));}if self.eof{return Ok(None);}if let Some(text)=self.records.get(self.record){append_record(&mut self.buffer,&self.tokenizer,text);self.record+=1;}else{self.eof=true;}}}
 pub fn position(&self)->DatasetPosition{DatasetPosition{file:self.path.clone(),record:self.record}}
}

pub struct CsvTextStream{tokenizer:Tokenizer,ctx_len:usize,path:PathBuf,records:Vec<String>,record:usize,buffer:Vec<usize>,eof:bool}
impl CsvTextStream{
 pub fn open(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{if ctx_len==0{return Err("ctx_len must be greater than zero".into());}let path=path.as_ref().to_path_buf();let file=File::open(&path).map_err(|e|format!("failed to open {}: {e}",path.display()))?;let mut reader=BufReader::new(file);let mut header=String::new();if reader.read_line(&mut header).map_err(|e|e.to_string())?==0{return Ok(Self{tokenizer,ctx_len,path,records:Vec::new(),record:0,buffer:Vec::new(),eof:true});}let headers=parse_csv_record(header.trim_end_matches(['\n','\r']))?;let text_index=headers.iter().position(|h|matches!(h.to_ascii_lowercase().as_str(),"text"|"content"|"document"|"body"|"code"|"prompt"));let mut records=Vec::new();for line in reader.lines(){let line=line.map_err(|e|e.to_string())?;let fields=parse_csv_record(&line)?;let text=if let Some(i)=text_index{fields.get(i).cloned().unwrap_or_default()}else{fields.iter().filter(|x|!looks_numeric(x)).max_by_key(|x|x.len()).cloned().unwrap_or_default()};if !text.trim().is_empty(){records.push(text);}}Ok(Self{tokenizer,ctx_len,path,records,record:0,buffer:Vec::new(),eof:false})}
 pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{loop{if self.buffer.len()>=self.ctx_len+1{let input=self.buffer[..self.ctx_len].to_vec();let target=self.buffer[1..self.ctx_len+1].to_vec();self.buffer.drain(..self.ctx_len);return Ok(Some(TokenBatch{input,target,position:DatasetPosition{file:self.path.clone(),record:self.record}}));}if self.eof{return Ok(None);}if let Some(text)=self.records.get(self.record){append_record(&mut self.buffer,&self.tokenizer,text);self.record+=1;}else{self.eof=true;}}}
 pub fn position(&self)->DatasetPosition{DatasetPosition{file:self.path.clone(),record:self.record}}
}
fn parse_csv_record(line:&str)->Result<Vec<String>,String>{let mut fields=Vec::new();let mut current=String::new();let mut chars=line.chars().peekable();let mut quoted=false;while let Some(ch)=chars.next(){if quoted{if ch=='"'{if chars.peek()==Some(&'"'){current.push('"');chars.next();}else{quoted=false;}}else{current.push(ch);}}else if ch=='"'{quoted=true;}else if ch==','{fields.push(std::mem::take(&mut current));}else{current.push(ch);}}if quoted{return Err("unterminated CSV quote".into());}fields.push(current);Ok(fields)}

pub struct ParquetTextStream{tokenizer:Tokenizer,ctx_len:usize,path:PathBuf,reader:SerializedFileReader<File>,rows:VecDeque<Row>,row_group:usize,row_group_count:usize,record:usize,buffer:Vec<usize>,eof:bool,text_field:Option<String>}
impl ParquetTextStream{pub fn open(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{Self::open_with_field(path,tokenizer,ctx_len,None::<String>)}pub fn open_with_field(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize,text_field:Option<impl Into<String>>)->Result<Self,String>{if ctx_len==0{return Err("ctx_len must be greater than zero".into());}let path=path.as_ref().to_path_buf();let file=File::open(&path).map_err(|e|format!("failed to open {}: {e}",path.display()))?;let reader=SerializedFileReader::new(file).map_err(|e|format!("failed to read parquet {}: {e}",path.display()))?;let row_group_count=reader.num_row_groups();Ok(Self{tokenizer,ctx_len,path,reader,rows:VecDeque::new(),row_group:0,row_group_count,record:0,buffer:Vec::new(),eof:false,text_field:text_field.map(Into::into)})}fn load_next_row_group(&mut self)->Result<bool,String>{if self.row_group>=self.row_group_count{return Ok(false);}let group=self.reader.get_row_group(self.row_group).map_err(|e|format!("failed to read parquet row group {} in {}: {e}",self.row_group,self.path.display()))?;let rows=group.get_row_iter(None).map_err(|e|format!("failed to decode parquet row group {} in {}: {e}",self.row_group,self.path.display()))?.collect::<Result<Vec<_>,_>>().map_err(|e|format!("failed to decode parquet row group {} in {}: {e}",self.row_group,self.path.display()))?;self.row_group+=1;self.rows.extend(rows);Ok(true)}fn field_text(&self,row:&Row)->Option<String>{if let Some(field)=&self.text_field{return row.get_column_iter().find_map(|(name,value)|if name==field{field_to_text(value)}else{None});}for name in ["text","content","document","body","code","prompt","completion"]{if let Some(value)=row.get_column_iter().find_map(|(column,value)|if column==name{Some(value)}else{None}){if let Some(text)=field_to_text(value){return Some(text);}}}row.get_column_iter().filter_map(|(_,value)|field_to_text(value)).max_by_key(|text|text.len())}pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{loop{if self.buffer.len()>=self.ctx_len+1{let input=self.buffer[..self.ctx_len].to_vec();let target=self.buffer[1..self.ctx_len+1].to_vec();self.buffer.drain(..self.ctx_len);return Ok(Some(TokenBatch{input,target,position:DatasetPosition{file:self.path.clone(),record:self.record}}));}if self.eof{return Ok(None);}if self.rows.is_empty()&&!self.load_next_row_group()?{self.eof=true;continue;}if let Some(row)=self.rows.pop_front(){let text=self.field_text(&row).ok_or_else(||format!("Parquet record {} in {} has no usable text field",self.record,self.path.display()))?;append_record(&mut self.buffer,&self.tokenizer,&text);self.record+=1;}}}pub fn position(&self)->DatasetPosition{DatasetPosition{file:self.path.clone(),record:self.record}}}
fn field_to_text(field:&Field)->Option<String>{match field{Field::Str(value)=>Some(value.clone()),Field::Bytes(value)=>String::from_utf8(value.data().to_vec()).ok(),Field::ListInternal(values)=>{let parts=values.elements().iter().filter_map(field_to_text).collect::<Vec<_>>();if parts.is_empty(){None}else{Some(parts.join(" "))}},_=>None}}

pub enum DatasetStream{Text(TextStream),Json(JsonTextStream),Csv(CsvTextStream),Parquet(ParquetTextStream)}
impl DatasetStream{pub fn open(path:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{let path=path.as_ref();match path.extension().and_then(|e|e.to_str()).unwrap_or("").to_ascii_lowercase().as_str(){"parquet"=>Ok(Self::Parquet(ParquetTextStream::open(path,tokenizer,ctx_len)?)),"json"=>Ok(Self::Json(JsonTextStream::open(path,tokenizer,ctx_len)?)),"csv"=>Ok(Self::Csv(CsvTextStream::open(path,tokenizer,ctx_len)?)),_=>Ok(Self::Text(TextStream::open(path,tokenizer,ctx_len)?))}}pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{match self{Self::Text(s)=>s.next_batch(),Self::Json(s)=>s.next_batch(),Self::Csv(s)=>s.next_batch(),Self::Parquet(s)=>s.next_batch()}}}

pub struct MultiFileTextStream{streams:Vec<DatasetStream>,current:usize}
impl MultiFileTextStream{pub fn open(paths:&[PathBuf],tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{if paths.is_empty(){return Err("dataset contains no files".into());}let streams=paths.iter().map(|path|DatasetStream::open(path,tokenizer.clone(),ctx_len)).collect::<Result<Vec<_>,_>>()?;Ok(Self{streams,current:0})}pub fn open_discovered(dir:impl AsRef<Path>,tokenizer:Tokenizer,ctx_len:usize)->Result<Self,String>{let paths=discover_files(dir)?;Self::open(&paths,tokenizer,ctx_len)}pub fn next_batch(&mut self)->Result<Option<TokenBatch>,String>{while !self.streams.is_empty(){if self.current>=self.streams.len(){self.current=0;}match self.streams[self.current].next_batch()?{Some(batch)=>{self.current=(self.current+1)%self.streams.len();return Ok(Some(batch));}None=>{self.streams.remove(self.current);if self.current>=self.streams.len()&&!self.streams.is_empty(){self.current=0;}}}}Ok(None)}pub fn file_count(&self)->usize{self.streams.len()}}

pub fn discover_files(dir:impl AsRef<Path>)->Result<Vec<PathBuf>,String>{const PLAIN:&[&str]=&["txt","text","py","cpp","c","h","hpp","cc","cxx","rs","js","ts","tsx","jsx","java","go","cs","php","rb","swift","kt","kts","scala","sh","bash","zsh","html","css","scss","sql","md","rst","yaml","yml","toml","xml"];fn visit(path:&Path,files:&mut Vec<PathBuf>)->Result<(),String>{for entry in std::fs::read_dir(path).map_err(|e|format!("failed to read {}: {e}",path.display()))?{let path=entry.map_err(|e|e.to_string())?.path();if path.is_dir(){visit(&path,files)?;continue;}if !path.is_file(){continue;}let ext=path.extension().and_then(|x|x.to_str()).map(str::to_ascii_lowercase);if matches!(ext.as_deref(),Some("json")|Some("jsonl")|Some("csv")|Some("parquet"))||ext.as_deref().is_some_and(|x|PLAIN.contains(&x)){files.push(path);}}Ok(())}let mut files=Vec::new();visit(dir.as_ref(),&mut files)?;files.sort();Ok(files)}
pub fn load_texts(path:impl AsRef<Path>)->Result<Vec<String>,String>{let file=File::open(path.as_ref()).map_err(|e|e.to_string())?;BufReader::new(file).lines().collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())}

#[cfg(test)]
mod tests_dataset{use super::*;use std::fs;fn tokenizer()->Tokenizer{Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into(),"b".into()," ".into()])}#[test]fn streams_fixed_length_training_pairs(){let path=std::env::temp_dir().join("smaul-dataset.txt");fs::write(&path,"a b a b a b").unwrap();let mut stream=TextStream::open(&path,tokenizer(),3).unwrap();let batch=stream.next_batch().unwrap().unwrap();assert_eq!(batch.input.len(),3);assert_eq!(batch.target.len(),3);assert_eq!(batch.position.record,1);let _=fs::remove_file(path);}#[test]fn reads_json_array_as_separate_records(){let path=std::env::temp_dir().join("smaul-dataset.json");fs::write(&path,"[{\"text\":\"a b\"},{\"content\":\"a b\"}]").unwrap();let mut stream=JsonTextStream::open(&path,tokenizer(),2).unwrap();assert!(stream.next_batch().unwrap().is_some());assert_eq!(stream.position().record,1);let _=fs::remove_file(path);}#[test]fn reads_csv_text_column(){let path=std::env::temp_dir().join("smaul-dataset.csv");fs::write(&path,"text,label\n\"a,b\",x\na b,y\n").unwrap();let mut stream=CsvTextStream::open(&path,tokenizer(),2).unwrap();assert!(stream.next_batch().unwrap().is_some());let _=fs::remove_file(path);}#[test]fn invalid_csv_quote_is_error(){let path=std::env::temp_dir().join("smaul-invalid.csv");fs::write(&path,"text\n\"a b\n").unwrap();assert!(CsvTextStream::open(&path,tokenizer(),2).is_err());let _=fs::remove_file(path);}#[test]fn invalid_json_is_error(){let path=std::env::temp_dir().join("smaul-invalid.json");fs::write(&path,"{bad}").unwrap();assert!(JsonTextStream::open(&path,tokenizer(),2).is_err());let _=fs::remove_file(path);}#[test]fn discovers_json_csv_and_parquet(){let dir=std::env::temp_dir().join("smaul-discover");let _=fs::remove_dir_all(&dir);fs::create_dir_all(&dir).unwrap();for name in ["a.json","b.csv","c.jsonl","d.parquet","e.txt"]{fs::write(dir.join(name),"").unwrap();}let files=discover_files(&dir).unwrap();assert_eq!(files.len(),5);let _=fs::remove_dir_all(dir);}}


// ===== dataset_multi =====

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
mod tests_dataset_multi { use super::*; use std::fs; fn tokenizer()->Tokenizer{Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into(),"b".into()," ".into()])} #[test]fn opens_mixed_formats(){let dir=std::env::temp_dir().join("smaul-mixed-dataset");fs::create_dir_all(&dir).unwrap();fs::write(dir.join("a.txt"),"a b a b").unwrap();fs::write(dir.join("b.jsonl"),"a b a b\n").unwrap();fs::write(dir.join("c.json"),"[{\"text\":\"a b a b\"}]").unwrap();fs::write(dir.join("d.csv"),"text\na b a b\n").unwrap();let stream=MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).unwrap();assert_eq!(stream.file_count(),4);let _=fs::remove_dir_all(dir);} #[test]fn discovers_nested_files(){let dir=std::env::temp_dir().join("smaul-mixed-nested-dataset");let nested=dir.join("nested");fs::create_dir_all(&nested).unwrap();fs::write(nested.join("a.txt"),"a b a b").unwrap();fs::write(nested.join("b.rs"),"a b a b").unwrap();fs::write(nested.join("ignore.bin"),b"x").unwrap();let stream=MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).unwrap();assert_eq!(stream.file_count(),2);assert_eq!(stream.paths(),&[nested.join("a.txt"),nested.join("b.rs")]);let _=fs::remove_dir_all(dir);} #[test]fn rejects_unsupported_files(){let dir=std::env::temp_dir().join("smaul-unsupported-dataset");fs::create_dir_all(&dir).unwrap();fs::write(dir.join("data.bin"),b"x").unwrap();assert!(MultiFileDatasetStream::open_discovered(&dir,tokenizer(),2).is_err());let _=fs::remove_dir_all(dir);} }


// ===== data_filter =====

fn normalize_text(text:&str)->String{text.nfkc().collect::<String>().replace("\r\n","\n").replace('\r',"\n").chars().filter(|c|!c.is_control()||*c=='\n'||*c=='\t').collect::<String>().trim().to_owned()}
pub fn filter_text(text:&str,dataset:&str,min_chars:usize,max_chars:usize)->Option<String>{let text=normalize_text(text);if text.len()<min_chars||text.len()>max_chars||text.contains('\u{fffd}'){return None}let compact:String=text.chars().filter(|c|!c.is_whitespace()).collect();if compact.chars().collect::<std::collections::HashSet<_>>().len()<8{return None}let url_only=text.strip_prefix("http://").or_else(||text.strip_prefix("https://")).or_else(||text.strip_prefix("www."));if url_only.is_some()&&!text.chars().any(char::is_whitespace){return None}let letters=text.chars().filter(|c|c.is_alphabetic()).count().max(1)as f32;if dataset=="hindi"{let d=text.chars().filter(|c|matches!(*c,'\u{0900}'..='\u{097f}')).count();if d<8||d as f32/letters<0.20{return None}}else if dataset=="english"{let l=text.chars().filter(|c|c.is_ascii_alphabetic()).count();if l<12||l as f32/letters<0.50{return None}}Some(text)}
pub fn filter_record(record:&Value,dataset:&str,min_chars:usize,max_chars:usize)->Option<String>{let object=record.as_object()?;let lower=object.iter().map(|(k,v)|(k.to_ascii_lowercase(),v)).collect::<std::collections::HashMap<_,_>>();let prompt=lower.get("prompt").and_then(|v|v.as_str());let completion=lower.get("completion").and_then(|v|v.as_str());let text=match(prompt,completion){(Some(a),Some(b))=>format!("{a}\n{b}"),(Some(a),None)=>a.to_owned(),(None,Some(b))=>b.to_owned(),_=>lower.get("text").and_then(|v|v.as_str())?.to_owned()};filter_text(&text,dataset,min_chars,max_chars)}
pub fn filter_jsonl(input:impl AsRef<std::path::Path>,output:impl AsRef<std::path::Path>,dataset:&str,min_chars:usize,max_chars:usize)->Result<usize,String>{use std::io::{BufRead,Write};let reader=std::io::BufReader::new(std::fs::File::open(input).map_err(|e|e.to_string())?);let mut writer=std::io::BufWriter::new(std::fs::File::create(output).map_err(|e|e.to_string())?);let mut kept=0;for line in reader.lines(){let line=line.map_err(|e|e.to_string())?;let Ok(record)=serde_json::from_str::<Value>(&line)else{continue};if let Some(text)=filter_record(&record,dataset,min_chars,max_chars){writeln!(writer,"{}",serde_json::json!({"text":text})).map_err(|e|e.to_string())?;kept+=1}}Ok(kept)}
#[cfg(test)]mod tests_data_filter{use super::*;#[test]fn filters_language(){assert!(filter_text("This is a sufficiently long English training document with many letters.","english",20,1000).is_some());assert!(filter_text("short","english",20,1000).is_none());assert!(filter_text("https://example.com","auto",5,1000).is_none())}#[test]fn normalizes_and_removes_controls(){assert_eq!(filter_text("  A\r\nB\u{0001}CDEFGHI  ","auto",1,100),Some("A\nBCDEFGHI".into()))}}


// ===== synthetic_data =====

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SyntheticRecord{pub instruction:String,pub response:String,pub think:String,pub domain:String}

pub fn linear_equation<R:Rng+?Sized>(rng:&mut R)->SyntheticRecord{let hindi=rng.random_bool(0.5);let a=rng.random_range(2..=500);let b=rng.random_range(10..=5000);let c=rng.random_range(5000..=500000);let x=(c-b)as f64/a as f64;if hindi{SyntheticRecord{instruction:format!("समीकरण {a}x + {b} = {c} के लिए x का मान ज्ञात कीजिए।"),response:format!("**उत्तर:** x = {x:.4}"),think:format!("दोनों पक्षों से {b} घटाएं, फिर {a} से विभाजित करें।"),domain:"math_algebra_hi".into()}}else{SyntheticRecord{instruction:format!("Solve for x in the linear equation: {a}x + {b} = {c}"),response:format!("Subtract {b}, then divide by {a}. **Final Answer:** x = {x:.4}"),think:format!("Subtract {b} from both sides, then divide by {a}."),domain:"math_algebra_en".into()}}}

pub fn quadratic<R:Rng+?Sized>(rng:&mut R)->SyntheticRecord{let a=rng.random_range(1..=50);let r1=rng.random_range(-100..=100);let r2=rng.random_range(-100..=100);let b=-a*(r1+r2);let c=a*r1*r2;SyntheticRecord{instruction:format!("Solve the quadratic equation: {a}x^2 + {b}x + {c} = 0"),response:format!("The roots are x = {r1} and x = {r2}."),think:format!("The polynomial factors as {a}(x - {r1})(x - {r2}) = 0."),domain:"math_quadratic".into()}}

pub fn linear_system<R:Rng+?Sized>(rng:&mut R)->SyntheticRecord{let x=rng.random_range(-50..=50);let y=rng.random_range(-50..=50);let(a,b,c,d)=loop{let a=rng.random_range(1..=20);let b=rng.random_range(1..=20);let c=rng.random_range(1..=20);let d=rng.random_range(1..=20);if a*d!=b*c{break(a,b,c,d)}};let e=a*x+b*y;let f=c*x+d*y;SyntheticRecord{instruction:format!("Solve: {a}x + {b}y = {e}; {c}x + {d}y = {f}"),response:format!("x = {x}, y = {y}"),think:"Use elimination or substitution and verify both equations.".into(),domain:"math_system_linear".into()}}

pub fn quicksort<R:Rng+?Sized>(rng:&mut R)->SyntheticRecord{let langs=[("Python","python","def quick_sort(a):\n    if len(a) <= 1: return a\n    p=a[len(a)//2]\n    return quick_sort([x for x in a if x<p])+[x for x in a if x==p]+quick_sort([x for x in a if x>p])"),("JavaScript","javascript","function quickSort(a) { if (a.length <= 1) return a; const p=a[Math.floor(a.length/2)]; return [...quickSort(a.filter(x=>x<p)),...a.filter(x=>x===p),...quickSort(a.filter(x=>x>p))]; }"),("Rust","rust","fn quick_sort(a: &mut [i32]) { a.sort_unstable(); }")];let(i,(lang,fence,code))=langs.iter().enumerate().nth(rng.random_range(0..langs.len())).unwrap_or((0,&langs[0]));let _=i;SyntheticRecord{instruction:format!("Write a clean quick sort implementation in {lang}."),response:format!("```{fence}\n{code}\n```\nAverage time: O(n log n)."),think:format!("Use the standard partitioning strategy for {lang}."),domain:"code_algorithms".into()}}

pub fn generate<R:Rng+?Sized>(rng:&mut R)->SyntheticRecord{match rng.random_range(0..4){0=>linear_equation(rng),1=>quadratic(rng),2=>linear_system(rng),_=>quicksort(rng)}}


// ===== sft_dataset =====

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
mod tests_sft_dataset {
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
