use std::io::{self,Write};
use std::fs::File;

pub struct Tensor<'a>{pub name:&'a str,pub shape:&'a [u64],pub dtype:u32,pub data:&'a [u8]}
pub struct Writer{header:Vec<u8>,tensors:Vec<Vec<u8>>,infos:Vec<(String,Vec<u64>,u32,u64)>}
impl Writer{
 pub fn new()->Self{Self{header:Vec::new(),tensors:Vec::new(),infos:Vec::new()}}
 fn u32(v:&mut Vec<u8>,x:u32){v.extend(x.to_le_bytes())} fn u64(v:&mut Vec<u8>,x:u64){v.extend(x.to_le_bytes())}
 fn s(v:&mut Vec<u8>,x:&str){Self::u64(v,x.len() as u64);v.extend(x.as_bytes())}
 pub fn add_meta_str(&mut self,k:&str,v:&str){Self::s(&mut self.header,k);Self::u32(&mut self.header,8);Self::s(&mut self.header,v)}
 pub fn add_meta_u32(&mut self,k:&str,v:u32){Self::s(&mut self.header,k);Self::u32(&mut self.header,4);Self::u32(&mut self.header,v)}
 pub fn add_meta_bool(&mut self,k:&str,v:bool){Self::s(&mut self.header,k);Self::u32(&mut self.header,7);self.header.push(v as u8)}
 pub fn add_tokens(&mut self,k:&str,tokens:&[String]){Self::s(&mut self.header,k);Self::u32(&mut self.header,9);Self::u32(&mut self.header,8);Self::u64(&mut self.header,tokens.len() as u64);for t in tokens{Self::s(&mut self.header,t)}}
 pub fn tensor(&mut self,t:Tensor<'_>){self.tensors.push(t.data.to_vec());self.infos.push((t.name.into(),t.shape.into(),t.dtype,0))}
 pub fn finish(mut self,path:&str)->Result<(),String>{let count=self.tensors.len();let base=4+4+8+8+self.header.len();let mut head=Vec::with_capacity(base+4096);head.extend(b"GGUF");Self::u32(&mut head,3);Self::u64(&mut head,count as u64);Self::u64(&mut head,0);head.extend(self.header);let meta_pos=16;let _=meta_pos;let mut offs=0u64;for(i,(n,s,d,_))in self.infos.iter_mut().enumerate(){offs=(offs+31)&!31;*self.infos.get_mut(i).unwrap()=(n.clone(),s.clone(),*d,offs);offs+=self.tensors[i].len() as u64;}
 let meta_count=0u64;head[16..24].copy_from_slice(&(meta_count.to_le_bytes()));let mut file=File::create(path).map_err(|e|e.to_string())?;file.write_all(&head).map_err(|e|e.to_string())?;let data_start=((head.len()+31)/32)*32;file.write_all(&vec![0;data_start-head.len()]).map_err(|e|e.to_string())?;let mut pos=data_start as u64;for data in self.tensors{let a=(pos+31)&!31;if a>pos{file.write_all(&vec![0;(a-pos)as usize]).map_err(|e|e.to_string())?}file.write_all(&data).map_err(|e|e.to_string())?;pos=a+data.len() as u64;}Ok(())}
}
