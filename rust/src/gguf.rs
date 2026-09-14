use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    String(String), Array(Vec<GgufValue>), U64(u64), I64(i64), F64(f64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

pub struct GgufReader {
    path: PathBuf,
    file: File,
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    data_start: u64,
}

struct Reader<'a> { file: &'a mut File }
impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> { let mut b=vec![0u8;n]; self.file.read_exact(&mut b).map_err(|e|e.to_string())?; Ok(b) }
    fn u8(&mut self)->Result<u8,String>{Ok(self.bytes(1)?[0])}
    fn i8(&mut self)->Result<i8,String>{Ok(self.u8()? as i8)}
    fn u16(&mut self)->Result<u16,String>{let b=self.bytes(2)?;Ok(u16::from_le_bytes([b[0],b[1]]))}
    fn i16(&mut self)->Result<i16,String>{Ok(self.u16()? as i16)}
    fn u32(&mut self)->Result<u32,String>{let b=self.bytes(4)?;Ok(u32::from_le_bytes(b.try_into().unwrap()))}
    fn i32(&mut self)->Result<i32,String>{Ok(self.u32()? as i32)}
    fn u64(&mut self)->Result<u64,String>{let b=self.bytes(8)?;Ok(u64::from_le_bytes(b.try_into().unwrap()))}
    fn i64(&mut self)->Result<i64,String>{Ok(self.u64()? as i64)}
    fn f32(&mut self)->Result<f32,String>{Ok(f32::from_bits(self.u32()?))}
    fn f64(&mut self)->Result<f64,String>{Ok(f64::from_bits(self.u64()?))}
    fn string(&mut self)->Result<String,String>{let n=self.u64()?;let n=usize::try_from(n).map_err(|_|"GGUF string is too large")?;String::from_utf8(self.bytes(n)?).map_err(|e|format!("invalid UTF-8 in GGUF string: {e}"))}
    fn value(&mut self, ty:u32)->Result<GgufValue,String>{match ty{0=>Ok(GgufValue::U8(self.u8()?)),1=>Ok(GgufValue::I8(self.i8()?)),2=>Ok(GgufValue::U16(self.u16()?)),3=>Ok(GgufValue::I16(self.i16()?)),4=>Ok(GgufValue::U32(self.u32()?)),5=>Ok(GgufValue::I32(self.i32()?)),6=>Ok(GgufValue::F32(self.f32()?)),7=>Ok(GgufValue::Bool(self.u8()?!=0)),8=>Ok(GgufValue::String(self.string()?)),9=>{let elem_ty=self.u32()?;let n=usize::try_from(self.u64()?).map_err(|_|"GGUF array is too large")?;let mut out=Vec::with_capacity(n);for _ in 0..n{out.push(self.value(elem_ty)?);}Ok(GgufValue::Array(out))},10=>Ok(GgufValue::U64(self.u64()?)),11=>Ok(GgufValue::I64(self.i64()?)),12=>Ok(GgufValue::F64(self.f64()?)),_=>Err(format!("unsupported GGUF metadata type {ty}"))}}
}

fn align(value:u64,alignment:u64)->Result<u64,String>{if alignment==0||!alignment.is_power_of_two(){return Err(format!("invalid GGUF alignment {alignment}"));}value.checked_add(alignment-1).map(|x|x/alignment*alignment).ok_or("GGUF offset overflow".into())}
fn element_count(shape:&[u64])->Result<u64,String>{shape.iter().try_fold(1u64,|a,&b|a.checked_mul(b)).ok_or_else(||"tensor element count overflow".into())}
fn block_size(dtype:u32)->Option<(u64,u64)>{match dtype{2=>Some((32,18)),3=>Some((32,20)),6=>Some((32,22)),7=>Some((32,24)),8=>Some((32,34)),_=>None}}

impl GgufReader {
    pub fn open(path: impl AsRef<Path>)->Result<Self,String>{
        let path=path.as_ref().to_path_buf(); let mut file=File::open(&path).map_err(|e|e.to_string())?; let mut magic=[0u8;4];file.read_exact(&mut magic).map_err(|e|e.to_string())?;if &magic!=MAGIC{return Err("not a GGUF file".into());}
        let mut r=Reader{file:&mut file}; let version=r.u32()?;if !(2..=3).contains(&version){return Err(format!("unsupported GGUF version {version}"));}let tensor_count=r.u64()?;let metadata_count=r.u64()?;
        let mut metadata=BTreeMap::new();for _ in 0..metadata_count{let key=r.string()?;let ty=r.u32()?;metadata.insert(key,r.value(ty)?);}
        let tensor_count_usize=usize::try_from(tensor_count).map_err(|_|"too many GGUF tensors")?;let mut tensors=Vec::with_capacity(tensor_count_usize);for _ in 0..tensor_count_usize{let name=r.string()?;let dims=usize::try_from(r.u32()?).map_err(|_|"too many tensor dimensions")?;let mut shape=Vec::with_capacity(dims);for _ in 0..dims{shape.push(r.u64()?);}let dtype=r.u32()?;let offset=r.u64()?;tensors.push(GgufTensorInfo{name,shape,dtype,offset});}
        let pos=r.file.stream_position().map_err(|e|e.to_string())?;let alignment=match metadata.get("general.alignment"){Some(GgufValue::U32(x))=>u64::from(*x),Some(GgufValue::U64(x))=>*x,None=>DEFAULT_ALIGNMENT,_=>return Err("general.alignment must be an integer".into())};let data_start=align(pos,alignment)?;
        Ok(Self{path,file,version,metadata,tensors,data_start})
    }
    pub fn path(&self)->&Path{&self.path}
    pub fn tensor(&self,name:&str)->Option<&GgufTensorInfo>{self.tensors.iter().find(|t|t.name==name)}
    fn tensor_size(info:&GgufTensorInfo)->Result<usize,String>{
        let elements=element_count(&info.shape)?;
        let bytes=match info.dtype {
            0=>elements.checked_mul(4), 1|30=>elements.checked_mul(2),
            24=>Some(elements), 25=>elements.checked_mul(2), 26=>elements.checked_mul(4),
            27=>elements.checked_mul(8), 28=>elements.checked_mul(8),
            2|3|6|7|8=>{let(block,bytes)=block_size(info.dtype).unwrap();if elements%block!=0{return Err(format!("tensor '{}' has {} elements, not divisible by quant block size {}",info.name,elements,block));}Some(elements/block*bytes)},
            _=>None,
        }.ok_or_else(||format!("unsupported GGML type {} ({})",info.dtype,gguf_dtype_name(info.dtype)))?;
        usize::try_from(bytes).map_err(|_|"tensor is too large for memory".into())
    }
    pub fn tensor_bytes(&mut self,name:&str)->Result<Vec<u8>,String>{let info=self.tensor(name).cloned().ok_or_else(||format!("GGUF tensor '{name}' not found"))?;let size=Self::tensor_size(&info)?;self.file.seek(SeekFrom::Start(self.data_start.checked_add(info.offset).ok_or("tensor offset overflow")?)).map_err(|e|e.to_string())?;let mut bytes=vec![0u8;size];self.file.read_exact(&mut bytes).map_err(|e|e.to_string())?;Ok(bytes)}
    pub fn tensor_f32(&mut self,name:&str)->Result<Vec<f32>,String>{let info=self.tensor(name).cloned().ok_or_else(||format!("GGUF tensor '{name}' not found"))?;let bytes=self.tensor_bytes(name)?;match info.dtype{0=>Ok(bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()),1=>Ok(bytes.chunks_exact(2).map(|b|f16_to_f32(u16::from_le_bytes([b[0],b[1]]))).collect()),30=>Ok(bytes.chunks_exact(2).map(|b|f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16)).collect()),24=>Ok(bytes.into_iter().map(|x|x as i8 as f32).collect()),25=>Ok(bytes.chunks_exact(2).map(|b|i16::from_le_bytes([b[0],b[1]]) as f32).collect()),26=>Ok(bytes.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap()) as f32).collect()),27=>Ok(bytes.chunks_exact(8).map(|b|i64::from_le_bytes(b.try_into().unwrap()) as f32).collect()),28=>Ok(bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap()) as f32).collect()),2=>Ok(decode_q4_0(&bytes)),3=>Ok(decode_q4_1(&bytes)),6=>Ok(decode_q5_0(&bytes)),7=>Ok(decode_q5_1(&bytes)),8=>Ok(decode_q8_0(&bytes)),_=>Err(format!("tensor '{}' uses unsupported GGML type {} ({})",info.name,info.dtype,gguf_dtype_name(info.dtype)))}}
}

fn read_f16(b:&[u8])->f32{f16_to_f32(u16::from_le_bytes([b[0],b[1]]))}
fn decode_q4_0(bytes:&[u8])->Vec<f32>{let mut out=Vec::with_capacity(bytes.len()/18*32);for block in bytes.chunks_exact(18){let d=read_f16(&block[..2]);for i in 0..32{let q=if i&1==0{block[2+i/2]&0x0f}else{block[2+i/2]>>4};out.push(d*(q as f32-8.0));}}out}
fn decode_q4_1(bytes:&[u8])->Vec<f32>{let mut out=Vec::with_capacity(bytes.len()/20*32);for block in bytes.chunks_exact(20){let d=read_f16(&block[..2]);let m=read_f16(&block[2..4]);for i in 0..32{let q=if i&1==0{block[4+i/2]&0x0f}else{block[4+i/2]>>4};out.push(d*q as f32+m);}}out}
fn decode_q5_0(bytes:&[u8])->Vec<f32>{let mut out=Vec::with_capacity(bytes.len()/22*32);for block in bytes.chunks_exact(22){let d=read_f16(&block[..2]);let qh=u32::from_le_bytes(block[2..6].try_into().unwrap());for i in 0..32{let low=if i&1==0{block[6+i/2]&0x0f}else{block[6+i/2]>>4};let high=((qh>>i)&1) as u8;out.push(d*((low|(high<<4)) as f32-16.0));}}out}
fn decode_q5_1(bytes:&[u8])->Vec<f32>{let mut out=Vec::with_capacity(bytes.len()/24*32);for block in bytes.chunks_exact(24){let d=read_f16(&block[..2]);let m=read_f16(&block[2..4]);let qh=u32::from_le_bytes(block[4..8].try_into().unwrap());for i in 0..32{let low=if i&1==0{block[8+i/2]&0x0f}else{block[8+i/2]>>4};let high=((qh>>i)&1) as u8;out.push(d*(low|(high<<4)) as f32+m);}}out}
fn decode_q8_0(bytes:&[u8])->Vec<f32>{let mut out=Vec::with_capacity(bytes.len()/34*32);for block in bytes.chunks_exact(34){let d=read_f16(&block[..2]);for i in 0..32{out.push(d*(block[2+i] as i8 as f32));}}out}

fn f16_to_f32(bits:u16)->f32{let sign=((bits as u32)&0x8000)<<16;let exp=((bits>>10)&0x1f) as u32;let frac=(bits&0x03ff) as u32;if exp==0{if frac==0{return f32::from_bits(sign);}let mut f=frac;let mut e=-14i32;while f&0x0400==0{f<<=1;e-=1;}f&=0x03ff;return f32::from_bits(sign|(((e+127)as u32)<<23)|(f<<13));}if exp==0x1f{return f32::from_bits(sign|0x7f800000|(frac<<13));}f32::from_bits(sign|((exp+112)<<23)|(frac<<13))}

pub fn gguf_dtype_name(dtype:u32)->&'static str{match dtype{0=>"F32",1=>"F16",2=>"Q4_0",3=>"Q4_1",6=>"Q5_0",7=>"Q5_1",8=>"Q8_0",9=>"Q8_1",10=>"Q2_K",11=>"Q3_K",12=>"Q4_K",13=>"Q5_K",14=>"Q6_K",15=>"Q8_K",16=>"IQ2_XXS",17=>"IQ2_XS",18=>"IQ3_XXS",19=>"IQ1_S",20=>"IQ4_NL",21=>"IQ3_S",22=>"IQ2_S",23=>"IQ4_XS",24=>"I8",25=>"I16",26=>"I32",27=>"I64",28=>"F64",29=>"IQ1_M",30=>"BF16",_=>"UNKNOWN"}}

#[cfg(test)]
mod tests{use super::*;#[test]fn half_conversion_covers_basic_values(){assert_eq!(f16_to_f32(0),0.0);assert_eq!(f16_to_f32(0x3c00),1.0);assert_eq!(f16_to_f32(0xc000),-2.0);assert!(f16_to_f32(0x7c00).is_infinite());}#[test]fn alignment_is_power_of_two(){assert_eq!(align(33,32).unwrap(),64);assert!(align(33,24).is_err());}#[test]fn dtype_names_cover_common_types(){assert_eq!(gguf_dtype_name(0),"F32");assert_eq!(gguf_dtype_name(14),"Q6_K");}#[test]fn q4_0_decodes_zero_block(){let mut b=vec![0u8;18];b[0]=0x00;b[1]=0x3c;let out=decode_q4_0(&b);assert_eq!(out.len(),32);assert!(out.iter().all(|x|*x==-8.0));}#[test]fn q8_0_decodes_zero_block(){let mut b=vec![0u8;34];b[0]=0x00;b[1]=0x3c;let out=decode_q8_0(&b);assert!(out.iter().all(|x|*x==0.0));}}
