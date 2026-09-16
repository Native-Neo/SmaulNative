use crate::rwkv_model::{RwkvModel, RwkvModelConfig};
use crate::tokenizer::Tokenizer;
use ndarray::{Array1, Array2};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC:&[u8;4]=b"GGUF";const DEFAULT_ALIGNMENT:u64=32;const QK_K:u64=256;const QK4_NL:u64=32;
#[derive(Clone,Debug,PartialEq)]pub enum GgufValue{U8(u8),I8(i8),U16(u16),I16(i16),U32(u32),I32(i32),F32(f32),Bool(bool),String(String),Array(Vec<GgufValue>),U64(u64),I64(i64),F64(f64)}
#[derive(Clone,Debug,PartialEq,Eq)]pub struct GgufTensorInfo{pub name:String,pub shape:Vec<u64>,pub dtype:u32,pub offset:u64}
pub struct GgufReader{path:PathBuf,file:File,pub version:u32,pub metadata:BTreeMap<String,GgufValue>,pub tensors:Vec<GgufTensorInfo>,data_start:u64}
struct Reader<'a>{file:&'a mut File}
impl<'a>Reader<'a>{fn bytes(&mut self,n:usize)->Result<Vec<u8>,String>{let mut b=vec![0;n];self.file.read_exact(&mut b).map_err(|e|e.to_string())?;Ok(b)}fn u8(&mut self)->Result<u8,String>{Ok(self.bytes(1)?[0])}fn i8(&mut self)->Result<i8,String>{Ok(self.u8()?as i8)}fn u16(&mut self)->Result<u16,String>{let b=self.bytes(2)?;Ok(u16::from_le_bytes([b[0],b[1]]))}fn i16(&mut self)->Result<i16,String>{Ok(self.u16()?as i16)}fn u32(&mut self)->Result<u32,String>{let b=self.bytes(4)?;Ok(u32::from_le_bytes(b.try_into().unwrap()))}fn i32(&mut self)->Result<i32,String>{Ok(self.u32()?as i32)}fn u64(&mut self)->Result<u64,String>{let b=self.bytes(8)?;Ok(u64::from_le_bytes(b.try_into().unwrap()))}fn i64(&mut self)->Result<i64,String>{Ok(self.u64()?as i64)}fn f32(&mut self)->Result<f32,String>{Ok(f32::from_bits(self.u32()?))}fn f64(&mut self)->Result<f64,String>{Ok(f64::from_bits(self.u64()?))}fn string(&mut self)->Result<String,String>{let n=usize::try_from(self.u64()?).map_err(|_|"GGUF string is too large")?;String::from_utf8(self.bytes(n)?).map_err(|e|format!("invalid UTF-8 in GGUF string: {e}"))}fn value(&mut self,ty:u32)->Result<GgufValue,String>{match ty{0=>Ok(GgufValue::U8(self.u8()?)),1=>Ok(GgufValue::I8(self.i8()?)),2=>Ok(GgufValue::U16(self.u16()?)),3=>Ok(GgufValue::I16(self.i16()?)),4=>Ok(GgufValue::U32(self.u32()?)),5=>Ok(GgufValue::I32(self.i32()?)),6=>Ok(GgufValue::F32(self.f32()?)),7=>Ok(GgufValue::Bool(self.u8()?!=0)),8=>Ok(GgufValue::String(self.string()?)),9=>{let et=self.u32()?;let n=usize::try_from(self.u64()?).map_err(|_|"GGUF array is too large")?;let mut v=Vec::with_capacity(n);for _ in 0..n{v.push(self.value(et)?)}Ok(GgufValue::Array(v))},10=>Ok(GgufValue::U64(self.u64()?)),11=>Ok(GgufValue::I64(self.i64()?)),12=>Ok(GgufValue::F64(self.f64()?)),_=>Err(format!("unsupported GGUF metadata type {ty}"))}}}
fn align(v:u64,a:u64)->Result<u64,String>{if a==0||!a.is_power_of_two(){return Err(format!("invalid GGUF alignment {a}"))}v.checked_add(a-1).map(|x|x/a*a).ok_or_else(||"GGUF offset overflow".into())}fn element_count(s:&[u64])->Result<u64,String>{s.iter().try_fold(1u64,|a,&b|a.checked_mul(b)).ok_or_else(||"tensor element count overflow".into())}fn block_size(t:u32)->Option<(u64,u64)>{match t{2=>Some((32,18)),3=>Some((32,20)),6=>Some((32,22)),7=>Some((32,24)),8=>Some((32,34)),9=>Some((32,36)),10=>Some((QK_K,84)),11=>Some((QK_K,110)),12=>Some((QK_K,144)),13=>Some((QK_K,176)),14=>Some((QK_K,210)),20=>Some((QK4_NL,18)),_=>None}}
impl GgufReader{pub fn open(path:impl AsRef<Path>)->Result<Self,String>{let path=path.as_ref().to_path_buf();let mut file=File::open(&path).map_err(|e|e.to_string())?;let mut magic=[0;4];file.read_exact(&mut magic).map_err(|e|e.to_string())?;if &magic!=MAGIC{return Err("not a GGUF file".into())}let mut r=Reader{file:&mut file};let version=r.u32()?;if !(2..=3).contains(&version){return Err(format!("unsupported GGUF version {version}"))}let tensor_count=r.u64()?;let metadata_count=r.u64()?;let mut metadata=BTreeMap::new();for _ in 0..metadata_count{let key=r.string()?;let ty=r.u32()?;metadata.insert(key,r.value(ty)?);}let tc=usize::try_from(tensor_count).map_err(|_|"too many GGUF tensors")?;let mut tensors=Vec::with_capacity(tc);for _ in 0..tc{let name=r.string()?;let dims=usize::try_from(r.u32()?).map_err(|_|"too many tensor dimensions")?;let mut shape=Vec::with_capacity(dims);for _ in 0..dims{shape.push(r.u64()?)}let dtype=r.u32()?;let offset=r.u64()?;tensors.push(GgufTensorInfo{name,shape,dtype,offset})}let pos=r.file.stream_position().map_err(|e|e.to_string())?;let alignment=match metadata.get("general.alignment"){Some(GgufValue::U32(x))=>u64::from(*x),Some(GgufValue::U64(x))=>*x,None=>DEFAULT_ALIGNMENT,_=>return Err("general.alignment must be an integer".into())};let data_start=align(pos,alignment)?;Ok(Self{path,file,version,metadata,tensors,data_start})}pub fn path(&self)->&Path{&self.path}pub fn tensor(&self,name:&str)->Option<&GgufTensorInfo>{self.tensors.iter().find(|t|t.name==name)}fn tensor_size(info:&GgufTensorInfo)->Result<usize,String>{let n=element_count(&info.shape)?;let b=match info.dtype{0=>n.checked_mul(4),1|30=>n.checked_mul(2),24=>Some(n),25=>n.checked_mul(2),26=>n.checked_mul(4),27|28=>n.checked_mul(8),2|3|6|7|8|9|10|11|12|13|14|20=>{let(bl,s)=block_size(info.dtype).unwrap();if n%bl!=0{return Err(format!("tensor '{}' has {} elements, not divisible by quant block size {}",info.name,n,bl))}n.checked_div(bl).and_then(|x|x.checked_mul(s))},_=>None}.ok_or_else(||format!("unsupported GGML type {} ({})",info.dtype,gguf_dtype_name(info.dtype)))?;usize::try_from(b).map_err(|_|"tensor is too large for memory".into())}pub fn tensor_bytes(&mut self,name:&str)->Result<Vec<u8>,String>{let info=self.tensor(name).cloned().ok_or_else(||format!("GGUF tensor '{name}' not found"))?;let size=Self::tensor_size(&info)?;let start=self.data_start.checked_add(info.offset).ok_or("tensor offset overflow")?;let end=start.checked_add(u64::try_from(size).map_err(|_|"tensor size overflow")?).ok_or("tensor end overflow")?;if end>self.file.metadata().map_err(|e|e.to_string())?.len(){return Err(format!("tensor '{}' extends past end of GGUF file",info.name))}self.file.seek(SeekFrom::Start(start)).map_err(|e|e.to_string())?;let mut bytes=vec![0;size];self.file.read_exact(&mut bytes).map_err(|e|e.to_string())?;Ok(bytes)}pub fn tensor_f32(&mut self,name:&str)->Result<Vec<f32>,String>{let info=self.tensor(name).cloned().ok_or_else(||format!("GGUF tensor '{name}' not found"))?;let bytes=self.tensor_bytes(name)?;match info.dtype{0=>Ok(bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()),1=>Ok(bytes.chunks_exact(2).map(|b|f16_to_f32(u16::from_le_bytes([b[0],b[1]]))).collect()),30=>Ok(bytes.chunks_exact(2).map(|b|f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16)).collect()),24=>Ok(bytes.into_iter().map(|x|x as i8 as f32).collect()),25=>Ok(bytes.chunks_exact(2).map(|b|i16::from_le_bytes([b[0],b[1]])as f32).collect()),26=>Ok(bytes.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())as f32).collect()),27=>Ok(bytes.chunks_exact(8).map(|b|i64::from_le_bytes(b.try_into().unwrap())as f32).collect()),28=>Ok(bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())as f32).collect()),2=>Ok(decode_q4_0(&bytes)),3=>Ok(decode_q4_1(&bytes)),6=>Ok(decode_q5_0(&bytes)),7=>Ok(decode_q5_1(&bytes)),8=>Ok(decode_q8_0(&bytes)),9=>Ok(decode_q8_1(&bytes)),10=>Ok(decode_q2_k(&bytes)),11=>Ok(decode_q3_k(&bytes)),12=>Ok(decode_q4_k(&bytes)),13=>Ok(decode_q5_k(&bytes)),14=>Ok(decode_q6_k(&bytes)),20=>Ok(decode_iq4_nl(&bytes)),_=>Err(format!("tensor '{}' uses unsupported GGML type {} ({})",info.name,info.dtype,gguf_dtype_name(info.dtype)))}}}
fn read_f16(b:&[u8])->f32{f16_to_f32(u16::from_le_bytes([b[0],b[1]]))}fn read_f32(b:&[u8])->f32{f32::from_le_bytes(b[..4].try_into().unwrap())}
fn decode_q4_0(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/18*32);for x in b.chunks_exact(18){let d=read_f16(&x[..2]);for i in 0..32{let q=if i&1==0{x[2+i/2]&15}else{x[2+i/2]>>4};o.push(d*(q as f32-8.0))}}o}fn decode_q4_1(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/20*32);for x in b.chunks_exact(20){let d=read_f16(&x[..2]);let m=read_f16(&x[2..4]);for i in 0..32{let q=if i&1==0{x[4+i/2]&15}else{x[4+i/2]>>4};o.push(d*q as f32+m)}}o}fn decode_q5_0(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/22*32);for x in b.chunks_exact(22){let d=read_f16(&x[..2]);let qh=u32::from_le_bytes(x[2..6].try_into().unwrap());for i in 0..32{let q=if i&1==0{x[6+i/2]&15}else{x[6+i/2]>>4};o.push(d*((q|(((qh>>i)&1)as u8)<<4)as f32-16.0))}}o}fn decode_q5_1(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/24*32);for x in b.chunks_exact(24){let d=read_f16(&x[..2]);let m=read_f16(&x[2..4]);let qh=u32::from_le_bytes(x[4..8].try_into().unwrap());for i in 0..32{let q=if i&1==0{x[8+i/2]&15}else{x[8+i/2]>>4};o.push(d*(q|(((qh>>i)&1)as u8)<<4)as f32+m)}}o}fn decode_q8_0(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/34*32);for x in b.chunks_exact(34){let d=read_f16(&x[..2]);for i in 0..32{o.push(d*x[2+i]as i8 as f32)}}o}fn decode_q8_1(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/36*32);for x in b.chunks_exact(36){let d=read_f16(&x[..2]);for i in 0..32{o.push(d*x[4+i]as i8 as f32)}}o}
fn get_scale_min_k4(j:usize,q:&[u8])->(u8,u8){if j<4{(q[j]&63,q[j+4]&63)}else{((q[j+4]&15)|((q[j-4]>>6)<<4),(q[j+4]>>4)|((q[j]>>6)<<4))}}
fn decode_q2_k(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/84*256);for x in b.chunks_exact(84){let d=read_f16(&x[80..82]);let min=read_f16(&x[82..84]);let scales=&x[..16];let q=&x[16..80];let mut is=0;for half in 0..2{let mut shift=0;for _ in 0..4{let sc=scales[is];is+=1;let dl=d*(sc&15)as f32;let ml=min*(sc>>4)as f32;for l in 0..16{o.push(dl*((q[half*32+l]>>shift)&3)as f32-ml)}let sc=scales[is];is+=1;let dl=d*(sc&15)as f32;let ml=min*(sc>>4)as f32;for l in 0..16{o.push(dl*((q[half*32+16+l]>>shift)&3)as f32-ml)}shift+=2}}}o}
fn decode_q3_k(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/110*256);for x in b.chunks_exact(110){let mut a=[0u32;4];a[0]=u32::from_le_bytes(x[96..100].try_into().unwrap());a[1]=u32::from_le_bytes(x[100..104].try_into().unwrap());a[2]=u32::from_le_bytes(x[104..108].try_into().unwrap());let tmp=a[2];a[2]=((a[0]>>4)&0x0f0f0f0f)|(((tmp>>4)&0x03030303)<<4);a[3]=((a[1]>>4)&0x0f0f0f0f)|(((tmp>>6)&0x03030303)<<4);a[0]=(a[0]&0x0f0f0f0f)|(((tmp)&0x03030303)<<4);a[1]=(a[1]&0x0f0f0f0f)|(((tmp>>2)&0x03030303)<<4);let sc=a.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();let d=read_f16(&x[108..110]);let q=&x[..64];let hm=&x[64..96];let mut is=0;let mut m=1u8;for half in 0..2{let mut shift=0;for _ in 0..4{let dl=d*(sc[is]as i8 as f32-32.0);is+=1;for l in 0..16{o.push(dl*(((q[half*32+l]>>shift)&3)as i8 as f32-if hm[l]&m!=0{0.0}else{4.0}))}let dl=d*(sc[is]as i8 as f32-32.0);is+=1;for l in 0..16{o.push(dl*(((q[half*32+16+l]>>shift)&3)as i8 as f32-if hm[l+16]&m!=0{0.0}else{4.0}))}shift+=2;m<<=1}}}o}
fn decode_q4_k(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/144*256);for x in b.chunks_exact(144){let d=read_f16(&x[..2]);let min=read_f16(&x[2..4]);let sc=&x[4..16];let q=&x[16..144];for span in 0..4{let(s1,m1)=get_scale_min_k4(span*2,sc);let(s2,m2)=get_scale_min_k4(span*2+1,sc);let q=&q[span*32..span*32+32];for &v in q{o.push(d*s1 as f32*(v&15)as f32-min*m1 as f32)}for &v in q{o.push(d*s2 as f32*(v>>4)as f32-min*m2 as f32)}}}o}fn decode_q5_k(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/176*256);for x in b.chunks_exact(176){let d=read_f16(&x[..2]);let min=read_f16(&x[2..4]);let sc=&x[4..16];let qh=&x[16..48];let q=&x[48..176];for span in 0..4{let(s1,m1)=get_scale_min_k4(span*2,sc);let(s2,m2)=get_scale_min_k4(span*2+1,sc);let u1=1u8<<(2*span);let u2=2u8<<(2*span);let q=&q[span*32..span*32+32];for l in 0..32{o.push(d*s1 as f32*((q[l]&15)+if qh[l]&u1!=0{16}else{0})as f32-min*m1 as f32)}for l in 0..32{o.push(d*s2 as f32*((q[l]>>4)+if qh[l]&u2!=0{16}else{0})as f32-min*m2 as f32)}}}o}fn decode_q6_k(b:&[u8])->Vec<f32>{let mut o=Vec::with_capacity(b.len()/210*256);for x in b.chunks_exact(210){let d=read_f16(&x[208..210]);for half in 0..2{let ql=&x[half*64..half*64+64];let qh=&x[128+half*32..128+half*32+32];let sc=&x[192+half*8..192+half*8+8];for l in 0..32{let is=l/16;let q1=((ql[l]&15)as i32|(((qh[l]&3)as i32)<<4))-32;let q2=((ql[l+32]&15)as i32|((((qh[l]>>2)&3)as i32)<<4))-32;let q3=((ql[l]>>4)as i32|((((qh[l]>>4)&3)as i32)<<4))-32;let q4=((ql[l+32]>>4)as i32|((((qh[l]>>6)&3)as i32)<<4))-32;o.push(d*sc[is]as i8 as f32*q1 as f32);o.push(d*sc[is+2]as i8 as f32*q2 as f32);o.push(d*sc[is+4]as i8 as f32*q3 as f32);o.push(d*sc[is+6]as i8 as f32*q4 as f32)}}}o}fn decode_iq4_nl(b:&[u8])->Vec<f32>{const VALUES:[f32;16]=[-127.0,-104.0,-83.0,-65.0,-49.0,-35.0,-23.0,-13.0,-5.0,5.0,13.0,23.0,35.0,49.0,65.0,83.0];let mut o=Vec::with_capacity(b.len()/18*32);for x in b.chunks_exact(18){let d=read_f16(&x[..2]);for i in 0..32{let q=if i&1==0{x[2+i/2]&15}else{x[2+i/2]>>4};o.push(d*VALUES[q as usize])}}o}
fn f16_to_f32(bits:u16)->f32{let sign=((bits as u32)&0x8000)<<16;let exp=((bits>>10)&0x1f)as u32;let frac=(bits&0x03ff)as u32;if exp==0{if frac==0{return f32::from_bits(sign)}let mut f=frac;let mut e=-14i32;while f&0x0400==0{f<<=1;e-=1}f&=0x03ff;return f32::from_bits(sign|(((e+127)as u32)<<23)|(f<<13))}if exp==0x1f{return f32::from_bits(sign|0x7f800000|(frac<<13))}f32::from_bits(sign|((exp+112)<<23)|(frac<<13))}
pub fn gguf_dtype_name(t:u32)->&'static str{match t{0=>"F32",1=>"F16",2=>"Q4_0",3=>"Q4_1",6=>"Q5_0",7=>"Q5_1",8=>"Q8_0",9=>"Q8_1",10=>"Q2_K",11=>"Q3_K",12=>"Q4_K",13=>"Q5_K",14=>"Q6_K",15=>"Q8_K",16=>"IQ2_XXS",17=>"IQ2_XS",18=>"IQ3_XXS",19=>"IQ1_S",20=>"IQ4_NL",21=>"IQ3_S",22=>"IQ2_S",23=>"IQ4_XS",24=>"I8",25=>"I16",26=>"I32",27=>"I64",28=>"F64",29=>"IQ1_M",30=>"BF16",_=>"UNKNOWN"}}
#[cfg(test)]mod tests_gguf{use super::*;#[test]fn half(){assert_eq!(f16_to_f32(0x3c00),1.0);assert_eq!(f16_to_f32(0xc000),-2.0);assert!(f16_to_f32(0x7c00).is_infinite())}#[test]fn alignment(){assert_eq!(align(33,32).unwrap(),64);assert!(align(33,24).is_err())}#[test]fn k_sizes(){assert_eq!(block_size(10),Some((256,84)));assert_eq!(block_size(11),Some((256,110)));assert_eq!(block_size(12),Some((256,144)));assert_eq!(block_size(13),Some((256,176)));assert_eq!(block_size(14),Some((256,210)))}#[test]fn zero_k_blocks(){assert!(decode_q2_k(&vec![0;84]).iter().all(|x|*x==0.0));assert!(decode_q3_k(&vec![0;110]).iter().all(|x|*x==0.0));assert!(decode_q4_k(&vec![0;144]).iter().all(|x|*x==0.0));assert!(decode_q5_k(&vec![0;176]).iter().all(|x|*x==0.0));assert!(decode_q6_k(&vec![0;210]).iter().all(|x|*x==0.0))}#[test]fn q8_1_ignores_auxiliary_sum(){let mut block=vec![0u8;36];block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());block[2..4].copy_from_slice(&0x7b00u16.to_le_bytes());block[4]=1;block[5]=u8::MAX;let values=decode_q8_1(&block);assert_eq!(values[0],1.0);assert_eq!(values[1],-1.0);assert!(values[2..].iter().all(|x|*x==0.0))}#[test]fn iq4_nl_zero_block(){assert!(decode_iq4_nl(&vec![0;18]).iter().all(|x|*x==0.0))}#[test]fn iq4_nl_decodes_codebook(){let mut block=vec![0u8;18];block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());for i in 0..16{block[2+i/2]|=if i&1==0{i as u8}else{(i as u8)<<4}}let values=decode_iq4_nl(&block);assert_eq!(values.len(),32);assert_eq!(values[0],-127.0);assert_eq!(values[1],-104.0);assert_eq!(values[15],83.0)}}


// ===== gguf_writer =====

pub struct Tensor<'a>{pub name:&'a str,pub shape:&'a [u64],pub dtype:u32,pub data:&'a [u8]}
pub struct Writer{metadata:Vec<u8>,metadata_count:u64,tensors:Vec<Vec<u8>>,infos:Vec<(String,Vec<u64>,u32)>}
impl Writer{
 pub fn new()->Self{Self{metadata:Vec::new(),metadata_count:0,tensors:Vec::new(),infos:Vec::new()}}
 fn u32(v:&mut Vec<u8>,x:u32){v.extend(x.to_le_bytes())}
 fn u64(v:&mut Vec<u8>,x:u64){v.extend(x.to_le_bytes())}
 fn s(v:&mut Vec<u8>,x:&str){Self::u64(v,x.len() as u64);v.extend(x.as_bytes())}
 pub fn add_meta_str(&mut self,k:&str,v:&str){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,8);Self::s(&mut self.metadata,v);self.metadata_count+=1}
 pub fn add_meta_u32(&mut self,k:&str,v:u32){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,4);Self::u32(&mut self.metadata,v);self.metadata_count+=1}
 pub fn add_meta_bool(&mut self,k:&str,v:bool){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,7);self.metadata.push(v as u8);self.metadata_count+=1}
 pub fn add_tokens(&mut self,k:&str,tokens:&[String]){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,9);Self::u32(&mut self.metadata,8);Self::u64(&mut self.metadata,tokens.len() as u64);for t in tokens{Self::s(&mut self.metadata,t)}self.metadata_count+=1}
 pub fn tensor(&mut self,t:Tensor<'_>){self.tensors.push(t.data.to_vec());self.infos.push((t.name.into(),t.shape.into(),t.dtype));}
 pub fn finish(self,path:&str)->Result<(),String>{let mut head=Vec::new();head.extend(b"GGUF");Self::u32(&mut head,3);Self::u64(&mut head,self.tensors.len() as u64);Self::u64(&mut head,self.metadata_count);head.extend(self.metadata);let mut offs=0u64;for(i,(n,s,d))in self.infos.iter().enumerate(){offs=(offs+31)&!31;Self::s(&mut head,n);Self::u32(&mut head,s.len() as u32);for x in s{Self::u64(&mut head,*x)}Self::u32(&mut head,*d);Self::u64(&mut head,offs);offs+=self.tensors[i].len() as u64;}
 let data_start=(head.len()+31)&!31;let mut f=File::create(path).map_err(|e|e.to_string())?;f.write_all(&head).map_err(|e|e.to_string())?;if data_start>head.len(){f.write_all(&vec![0;data_start-head.len()]).map_err(|e|e.to_string())?}let mut pos=data_start as u64;for d in self.tensors{let a=(pos+31)&!31;if a>pos{f.write_all(&vec![0;(a-pos)as usize]).map_err(|e|e.to_string())?}f.write_all(&d).map_err(|e|e.to_string())?;pos=a+d.len() as u64;}Ok(())}
}


// ===== gguf_convert =====

fn f32_to_f16(x:f32)->u16{let b=x.to_bits();let s=((b>>16)&0x8000)as u16;let e=((b>>23)&255)as i32-127+15;let m=b&0x7fffff;if e<=0{if e< -10{return s}return s|(((m|0x800000)>>(1-e)+0x1000)>>13)as u16}if e>=31{return s|0x7c00}s|((e as u16)<<10)|((m+0x1000)>>13)as u16}
fn f32_bytes_to_f16(b:&[u8])->Vec<u8>{let mut o=Vec::with_capacity(b.len()/2);for c in b.chunks_exact(4){o.extend(f32_to_f16(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes())}o}
fn tokens(path:&Path)->Result<Vec<String>,String>{let v:Value=serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;let x=v.get("vocab").or_else(||v.get("model").and_then(|m|m.get("vocab"))).ok_or("tokenizer.json has no vocabulary")?.as_object().ok_or("invalid tokenizer vocabulary")?;let max=x.values().filter_map(|v|v.as_u64()).max().ok_or("empty tokenizer")? as usize;let mut out=vec![String::new();max+1];for(k,id)in x{out[id.as_u64().ok_or("invalid tokenizer id")? as usize]=k.clone()}if out.iter().any(|s|s.is_empty()){return Err("tokenizer vocabulary contains gaps or empty tokens".into())}Ok(out)}
fn cfg_u32(w:&mut Writer,c:&Value,key:&str,out:&str){if let Some(v)=c.get(key).and_then(|v|v.as_u64()){w.add_meta_u32(out,v as u32)}}

pub fn convert(input_dir:impl AsRef<Path>,output:impl AsRef<Path>,dtype:&str)->Result<(),String>{if dtype!="f16"&&dtype!="f32"{return Err("dtype must be f16 or f32".into())}let dir=input_dir.as_ref();let cfg:Value=serde_json::from_slice(&fs::read(dir.join("config.json")).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;let toks=tokens(&dir.join("tokenizer.json"))?;if cfg.get("vocab_size").and_then(|v|v.as_u64()).unwrap_or(0)as usize!=toks.len(){return Err("tokenizer vocabulary does not match config vocab_size".into())}let bytes=fs::read(dir.join("model.safetensors")).map_err(|e|e.to_string())?;let st=SafeTensors::deserialize(&bytes).map_err(|e|e.to_string())?;let mut w=Writer::new();w.add_meta_str("general.name","SmaulNative RWKV-X");w.add_meta_str("general.description","RWKV-X checkpoint exported from SmaulNative");w.add_meta_u32("general.alignment",32);w.add_meta_str("general.architecture","rwkv_x");cfg_u32(&mut w,&cfg,"vocab_size","rwkv_x.vocab_size");cfg_u32(&mut w,&cfg,"ctx_len_hint","rwkv_x.context_length");cfg_u32(&mut w,&cfg,"n_embd","rwkv_x.embedding_length");cfg_u32(&mut w,&cfg,"n_layer","rwkv_x.block_count");if let Some(n)=cfg.get("n_embd").and_then(|v|v.as_u64()){if let Some(h)=cfg.get("head_size").and_then(|v|v.as_u64()){w.add_meta_u32("rwkv_x.head_count",(n/h)as u32);w.add_meta_u32("rwkv_x.head_count_kv",(n/h)as u32)}}cfg_u32(&mut w,&cfg,"head_size","rwkv_x.head_size");cfg_u32(&mut w,&cfg,"n_moba_layer","rwkv_x.n_moba_layer");cfg_u32(&mut w,&cfg,"moba_chunk_size","rwkv_x.moba_chunk_size");cfg_u32(&mut w,&cfg,"moba_topk","rwkv_x.moba_topk");cfg_u32(&mut w,&cfg,"wkv_chunk_size","rwkv_x.wkv_chunk_size");cfg_u32(&mut w,&cfg,"head_size_divisor","rwkv_x.head_size_divisor");w.add_meta_bool("rwkv_x.is_moe",cfg.get("is_moe").and_then(|v|v.as_bool()).unwrap_or(false));cfg_u32(&mut w,&cfg,"num_experts","rwkv_x.num_experts");cfg_u32(&mut w,&cfg,"num_experts_per_tok","rwkv_x.num_experts_per_tok");w.add_meta_str("tokenizer.ggml.model","rwkv");w.add_tokens("tokenizer.ggml.tokens",&toks);
for name in st.names(){let t=st.tensor(name).map_err(|e|e.to_string())?;let (data,ty)=match t.dtype(){Dtype::F32=>{if dtype=="f16"{(f32_bytes_to_f16(t.data()),1)}else{(t.data().to_vec(),0)}},Dtype::F16=>(t.data().to_vec(),1),Dtype::I8=>(t.data().to_vec(),24),Dtype::I16=>(t.data().to_vec(),25),Dtype::I32=>(t.data().to_vec(),26),Dtype::I64=>(t.data().to_vec(),27),Dtype::U8=>(t.data().to_vec(),24),Dtype::BOOL=>(t.data().to_vec(),24),other=>return Err(format!("unsupported Safetensors dtype for {name}: {other:?}"))};let shape=t.shape().iter().map(|&x|x as u64).collect::<Vec<_>>();w.tensor(Tensor{name,shape:&shape,dtype:ty,data:&data});}w.finish(output.as_ref().to_str().ok_or("invalid output path")?)?;Ok(())}

pub fn convert_cli(args:&[String])->Result<(),String>{if args.len()<3{return Err("usage: smaul-convert-gguf <input_dir> <output.gguf> [--dtype f16|f32]".into())}let dtype=if args.len()>=5&&args[3]=="--dtype"{&args[4]}else{"f16"};convert(&args[1],&args[2],dtype)}


// ===== gguf_model_loader =====

fn metadata_usize(reader: &GgufReader, key: &str) -> Result<usize, String> {
    match reader.metadata.get(key) {
        Some(GgufValue::U32(v)) => Ok(*v as usize),
        Some(GgufValue::U64(v)) => usize::try_from(*v).map_err(|_| format!("GGUF metadata '{key}' is too large")),
        Some(_) => Err(format!("GGUF metadata '{key}' is not an integer")),
        None => Err(format!("missing GGUF metadata '{key}'")),
    }
}

fn tensor1(reader: &mut GgufReader, name: &str, expected: usize) -> Result<Array1<f32>, String> {
    let info = reader.tensor(name).cloned().ok_or_else(|| format!("GGUF tensor '{name}' not found"))?;
    if info.shape != [expected as u64] { return Err(format!("tensor '{name}' has shape {:?}, expected [{expected}]", info.shape)); }
    Array1::from_shape_vec(expected, reader.tensor_f32(name)?).map_err(|e| format!("invalid tensor '{name}': {e}"))
}

fn tensor2(reader: &mut GgufReader, name: &str, rows: usize, cols: usize) -> Result<Array2<f32>, String> {
    let info = reader.tensor(name).cloned().ok_or_else(|| format!("GGUF tensor '{name}' not found"))?;
    if info.shape != [rows as u64, cols as u64] { return Err(format!("tensor '{name}' has shape {:?}, expected [{rows}, {cols}]", info.shape)); }
    Array2::from_shape_vec((rows, cols), reader.tensor_f32(name)?).map_err(|e| format!("invalid tensor '{name}': {e}"))
}

fn tensor2_transposed(reader: &mut GgufReader, name: &str, rows: usize, cols: usize) -> Result<Array2<f32>, String> {
    Ok(tensor2(reader, name, cols, rows)?.reversed_axes().to_owned())
}

fn set_norm(reader: &mut GgufReader, prefix: &str, weight: &mut Array1<f32>, bias: &mut Array1<f32>) -> Result<(), String> {
    *weight = tensor1(reader, &format!("{prefix}.weight"), weight.len())?;
    *bias = tensor1(reader, &format!("{prefix}.bias"), bias.len())?;
    Ok(())
}

fn set_linear(reader: &mut GgufReader, name: &str, target: &mut Array2<f32>, transposed: bool) -> Result<(), String> {
    let shape = target.raw_dim();
    *target = if transposed { tensor2_transposed(reader, name, shape[0], shape[1])? } else { tensor2(reader, name, shape[0], shape[1])? };
    Ok(())
}

fn load_rwkv_block(reader: &mut GgufReader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("rwkv_blocks.{i}");
    let block = &mut model.rwkv_blocks[i];
    if let Some(norm) = block.ln0.as_mut() { set_norm(reader, &format!("{p}.ln0"), &mut norm.weight, &mut norm.bias)?; }
    set_norm(reader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(reader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    let t = &mut block.time_mix;
    for (name, target) in [
        ("x_r", &mut t.x_r), ("x_w", &mut t.x_w), ("x_k", &mut t.x_k),
        ("x_v", &mut t.x_v), ("x_a", &mut t.x_a), ("x_g", &mut t.x_g),
        ("w0", &mut t.w0), ("a0", &mut t.a0), ("k_k", &mut t.k_k), ("k_a", &mut t.k_a),
    ] {
        let len = target.len();
        *target = tensor1(reader, &format!("{p}.att.{name}"), len)?;
    }
    for (name, target) in [
        ("w1", &mut t.w1), ("w2", &mut t.w2), ("a1", &mut t.a1),
        ("a2", &mut t.a2), ("g1", &mut t.g1), ("g2", &mut t.g2), ("r_k", &mut t.r_k),
    ] {
        let shape = target.raw_dim();
        *target = tensor2(reader, &format!("{p}.att.{name}"), shape[0], shape[1])?;
    }
    for (name, target) in [("v1", &mut t.v1), ("v2", &mut t.v2)] {
        if let Some(target) = target.as_mut() {
            let shape = target.raw_dim();
            *target = tensor2(reader, &format!("{p}.att.{name}"), shape[0], shape[1])?;
        }
    }
    if let Some(target) = t.v0.as_mut() {
        let len = target.len();
        *target = tensor1(reader, &format!("{p}.att.v0"), len)?;
    }
    set_linear(reader, &format!("{p}.att.receptance.weight"), &mut t.receptance, true)?;
    set_linear(reader, &format!("{p}.att.key.weight"), &mut t.key, true)?;
    set_linear(reader, &format!("{p}.att.value.weight"), &mut t.value, true)?;
    set_linear(reader, &format!("{p}.att.output.weight"), &mut t.output, true)?;
    set_norm(reader, &format!("{p}.att.ln_x"), &mut t.ln_x.weight, &mut t.ln_x.bias)?;
    let c = &mut block.cmix;
    let len = c.x_k.len();
    c.x_k = tensor1(reader, &format!("{p}.ffn.x_k"), len)?;
    set_linear(reader, &format!("{p}.ffn.key.weight"), &mut c.key, true)?;
    set_linear(reader, &format!("{p}.ffn.value.weight"), &mut c.value, true)?;
    Ok(())
}

fn load_moba_block(reader: &mut GgufReader, model: &mut RwkvModel, i: usize) -> Result<(), String> {
    let p = format!("moba_blocks.{i}");
    let block = &mut model.moba_blocks[i];
    set_norm(reader, &format!("{p}.ln1"), &mut block.ln1.weight, &mut block.ln1.bias)?;
    set_norm(reader, &format!("{p}.ln2"), &mut block.ln2.weight, &mut block.ln2.bias)?;
    for (name, target) in [
        ("receptance.weight", &mut block.att.receptance.weight),
        ("key.weight", &mut block.att.key.weight),
        ("value.weight", &mut block.att.value.weight),
        ("output.weight", &mut block.att.output.weight),
    ] { set_linear(reader, &format!("{p}.att.{name}"), target, false)?; }
    let len = block.ffn.x_k.len();
    block.ffn.x_k = tensor1(reader, &format!("{p}.ffn.x_k"), len)?;
    set_linear(reader, &format!("{p}.ffn.key.weight"), &mut block.ffn.key, true)?;
    set_linear(reader, &format!("{p}.ffn.value.weight"), &mut block.ffn.value, true)?;
    Ok(())
}

pub fn load_gguf_model(path: impl AsRef<std::path::Path>) -> Result<RwkvModel, String> {
    let mut reader = GgufReader::open(path)?;
    let vocab_size = metadata_usize(&reader, "vocab_size")?;
    let n_embd = metadata_usize(&reader, "embedding_length")?;
    let n_layer = metadata_usize(&reader, "block_count")?;
    let head_size = metadata_usize(&reader, "rwkv_x.head_size")?;
    let n_moba_layer = metadata_usize(&reader, "rwkv_x.n_moba_layer")?;
    let chunk_size = metadata_usize(&reader, "rwkv_x.moba_chunk_size")?;
    let topk = metadata_usize(&reader, "rwkv_x.moba_topk")?;
    let mut config = RwkvModelConfig::new(vocab_size, n_embd, n_layer, head_size);
    config.head_size_divisor = metadata_usize(&reader, "rwkv_x.head_size_divisor")?;
    config = config.with_moba(n_moba_layer, chunk_size, topk);
    let mut model = RwkvModel::new(config, 0);
    model.embedding.weight = tensor2(&mut reader, "emb.weight", vocab_size, n_embd)?;
    set_norm(&mut reader, "ln_out", &mut model.ln_out.weight, &mut model.ln_out.bias)?;
    set_linear(&mut reader, "head.weight", &mut model.head.weight, false)?;
    for i in 0..model.rwkv_blocks.len() { load_rwkv_block(&mut reader, &mut model, i)?; }
    for i in 0..model.moba_blocks.len() { load_moba_block(&mut reader, &mut model, i)?; }
    Ok(model)
}

pub fn load_gguf_tokenizer(path: impl AsRef<std::path::Path>) -> Result<Tokenizer, String> {
    let reader = GgufReader::open(path)?;
    let tokens = match reader.metadata.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(values)) => values.iter().map(|v| match v { GgufValue::String(s) => Ok(s.clone()), _ => Err("GGUF tokenizer token list contains a non-string value".into()) }).collect::<Result<Vec<_>, String>>()?,
        Some(_) => return Err("GGUF tokenizer token list is not an array".into()),
        None => return Err("missing GGUF metadata 'tokenizer.ggml.tokens'".into()),
    };
    if tokens.is_empty() { return Err("GGUF tokenizer vocabulary is empty".into()); }
    Ok(Tokenizer::from_vocab(tokens))
}

pub fn load_gguf(path: impl AsRef<std::path::Path>) -> Result<(RwkvModel, Tokenizer), String> {
    let path = path.as_ref();
    let model = load_gguf_model(path)?;
    let tokenizer = load_gguf_tokenizer(path)?;
    Ok((model, tokenizer))
}
